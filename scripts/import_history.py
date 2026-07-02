"""
Import channel history and extract knowledge day by day.
Days are chunked into groups of 50 messages.
Unanswered questions carry forward between chunks and days.

Extracted knowledge is stored as:
- A narrative summary (situation, context, decision, reasoning)
- Keywords (picked from master list or newly coined, merged back)
- Raw conversation lines
- No more qa.md — all goes to DB
"""

import asyncio
import json
import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import discord

from langchain_core.messages import HumanMessage

from shared.config import DISCORD_BOT_TOKEN, SUPPORT_CHANNEL_ID, POSTGRES_URL, ADMIN_USER_IDS
from shared.llm import get_llm
from support_bot.rules import db_save_entry, db_date_has_entries
from shared.embeddings import embed_text


CHUNK_SIZE = 50

EXTRACTION_PROMPT = """You are analyzing a support channel conversation log for a solar sales company (SES).

Users are sales agents. Admins are managers/supervisors.
The log is chronological. Messages show role (ADMIN/USER), display name, and timestamp.
@ADMIN and @USER tags replace Discord mention IDs — they refer to the mentioned person's role, not the speaker.

Log format notes:
- Messages marked [replying to HH:MM ROLE (name)] are Discord replies — they are placed directly after the message they reply to.
- Messages with no reply marker are standalone — reason from context (proximity, topic, participants) whether they are part of the same exchange or directed at a different user/topic.
- Each exchange may span multiple messages across a thread or a natural back-and-forth. Group them accordingly.

Your job:
1. Identify exchanges where a user raised a situation or question and an admin resolved it with USEFUL information.
   - Useful = something that helps answer similar situations in the future (policy, decision criteria, procedure, exception, assignment).
   - SKIP if the admin only said things like "fixed", "done", "ok", "handled", "sure" with no explanation.
   - SKIP if an image was involved and the admin gave no context about what they saw or why they decided what they did.
   - SKIP if a question was never answered by an admin.
2. For each useful exchange, produce:
   - topic: short label (5 words max), e.g. "Credit score DQ threshold Illinois"
   - summary: 3-5 sentences covering:
       * What was the situation (include specifics: state, score range, program, role, etc.)
       * What context or details mattered to the decision
       * What the admin decided
       * Why (if stated or clearly implied)

   - conversation: the relevant raw lines only (trim irrelevant filler)
3. Flag exchanges where a user asked something and no admin answered — carry those forward as unanswered.
4. Return ONLY valid JSON. No explanation, no markdown fences.

Return format:
{
  "pairs": [
    {
      "topic": "Short topic label",
      "summary": "Narrative summary of the full exchange.",

      "conversation": "Relevant raw lines"
    }
  ],
  "unanswered": [
    "Raw lines of questions with no admin response"
  ]
}

If nothing useful: {"pairs": [], "unanswered": []}

Conversation log:
LOG_PLACEHOLDER
"""


def strip_mentions(content: str, admin_ids: set[int], name_lookup: dict[int, str] | None = None) -> str:
    """
    Replace <@user_id> with @ADMIN or @Name based on whether the ID is a known admin.
    Uses name_lookup to resolve actual display names for non-admin mentions.
    Falls back to @USER only if name is unknown.
    """
    def replace(match):
        uid = int(match.group(1))
        if uid in admin_ids:
            return "@ADMIN"
        if name_lookup and uid in name_lookup:
            return f"@{name_lookup[uid]}"
        return "@USER"
    return re.sub(r"<@!?(\d+)>", replace, content)


def build_threaded_order(messages: list[dict], prev_day_ids: set[int]) -> list[dict]:
    """
    Reorder messages so replies appear directly after the message they reply to.
    Uses DFS on a parent->children tree.

    Messages replying to a previous day's message become roots (floaters dropped
    since we no longer carry cross-day context — the prev_day_ids set tells us
    which prior-day messages were replied to so we can inject them as roots).
    """
    day_ids = {m["id"] for m in messages}
    children: dict[int, list[dict]] = {m["id"]: [] for m in messages}
    roots: list[dict] = []

    for m in messages:
        ref = m.get("reply_to")
        if ref and ref in day_ids:
            # Reply within same day — attach to parent
            children[ref].append(m)
        else:
            # No reply, or reply is to a previous day — treat as root
            roots.append(m)

    # DFS flatten with visual tree print
    result = []
    tree_lines = []

    def dfs(node: dict, depth: int = 0):
        result.append(node)
        prefix = "  " * depth + ("↳ " if depth > 0 else "• ")
        tree_lines.append(f"{prefix}#{node['idx']} [{node['timestamp']}] {node['role']} ({node['author']}): {node['content'][:60]}")
        for child in children.get(node["id"], []):
            dfs(child, depth + 1)

    for root in roots:
        dfs(root)

    reordered = any(result[i]["idx"] != result[i-1]["idx"] + 1 for i in range(1, len(result)))
    has_replies = any(children[m["id"]] for m in messages if children.get(m["id"]))
    if has_replies:
        moved = sum(1 for i, m in enumerate(result) if m["idx"] != i)
        print(f"  Thread tree: ({moved} message(s) reordered)")
        for line in tree_lines:
            print(f"    {line}")
    else:
        print("  No replies detected — chronological order kept.")

    return result


def format_log(messages: list[dict], all_msgs: dict[int, dict] | None = None) -> str:
    lines = []
    for m in messages:
        ref = m.get("reply_to")
        reply_hint = ""
        if ref and all_msgs and ref in all_msgs:
            p = all_msgs[ref]
            reply_hint = f" [replying to {p['timestamp']} {p['role']} ({p['author']})]"
        lines.append(
            f"[{m['timestamp']}]{reply_hint} {m['role']} ({m['author']}): {m['content']}"
        )
    return "\n".join(lines)


async def fetch_messages_by_day(
    channel: discord.TextChannel,
    admin_ids: list[int],
    days_back: int,
) -> tuple[dict[date, list[dict]], dict[int, dict]]:
    days: dict[date, list[dict]] = defaultdict(list)
    admin_id_set = set(admin_ids)
    all_msgs: dict[int, dict] = {}
    print("Fetching messages...")
    count = 0

    # Build a running name lookup: user_id -> display_name from all mentions seen
    name_lookup: dict[int, str] = {}

    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    async for msg in channel.history(limit=None, oldest_first=True, after=cutoff):
        # Index mentioned users' display names as we see them
        for member in msg.mentions:
            if member.id not in name_lookup:
                name_lookup[member.id] = member.display_name
        # Also index the message author
        if msg.author.id not in name_lookup:
            name_lookup[msg.author.id] = msg.author.display_name

        has_image = any(
            a.content_type and a.content_type.startswith("image/")
            for a in msg.attachments
        )
        content_text = strip_mentions(msg.content.strip(), admin_id_set, name_lookup)
        if has_image:
            content_text = (content_text + " [Image Attached]").strip()
        if not content_text:
            continue

        day = msg.created_at.astimezone(timezone.utc).date()
        role = "ADMIN" if msg.author.id in admin_id_set else ("BOT" if msg.author.bot else "USER")
        timestamp = msg.created_at.strftime("%H:%M")
        reply_to = msg.reference.message_id if msg.reference else None

        entry = {
            "id":        msg.id,
            "idx":       count,  # original chronological index
            "timestamp": timestamp,
            "author":    msg.author.display_name,
            "role":      role,
            "content":   content_text,
            "reply_to":  reply_to,
        }
        all_msgs[msg.id] = entry
        days[day].append(entry)
        count += 1

    print(f"Fetched {count} messages across {len(days)} day(s).")
    return dict(sorted(days.items())), all_msgs


async def extract_from_log(
    log: str,
    source_date: date,
    chunk_label: str = "",
) -> list[str]:
    llm = get_llm(temperature=0.1)

    prompt = EXTRACTION_PROMPT.replace("LOG_PLACEHOLDER", log)

    try:
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = resp.content.strip()
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # LLM sometimes emits unescaped backslashes — fix and retry
            import re as _re
            cleaned = _re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
            data = json.loads(cleaned)
    except Exception as e:
        print(f"  [{chunk_label}] LLM/parse error: {e}")
        return []

    pairs = data.get("pairs", [])
    unanswered = data.get("unanswered", [])

    if pairs:
        print(f"  [{chunk_label}] {len(pairs)} exchange(s) extracted:")
        for pair in pairs:
            topic        = pair.get("topic", "General")
            summary      = pair.get("summary", "")
            conversation = pair.get("conversation", "")

            if summary:
                try:
                    embedding = await embed_text(summary)
                except Exception as e:
                    print(f"      [embedding failed: {e}]")
                    embedding = None

                await asyncio.to_thread(
                    db_save_entry,
                    source_date,
                    topic,
                    conversation,
                    summary,
                    embedding,
                )

                print(f"    + [{topic}]")
                print(f"      summary:  {summary[:100]}")
    else:
        print(f"  [{chunk_label}] No useful exchanges found.")

    if unanswered:
        print(f"  [{chunk_label}] {len(unanswered)} unanswered — LLM will discard, no carry forward.")

    return []


async def run_import(
    channel: discord.TextChannel,
    admin_ids: list[int],
    status_cb=None,
    days_back: int = 30,
):
    days, all_msgs = await fetch_messages_by_day(channel, admin_ids, days_back)
    total_days = 0

    # Track which msg IDs from previous days get replied to on subsequent days
    # so we can inject them as roots on those days
    prev_day_ids: set[int] = set()

    all_days = list(days.items())
    print(f"\nProcessing {len(all_days)} day(s)...\n{'='*50}")

    for day, messages in all_days:
        day_label = day.strftime("%B %d, %Y")

        print(f"\n── {day_label} ({len(messages)} messages) ──")

        if db_date_has_entries(day):
            print(f"  [SKIP] {day_label} already processed.")
            if status_cb:
                await status_cb(f"Skipping {day_label} (already processed).")
            # Still update prev_day_ids so cross-day reply detection works
            prev_day_ids = {m["id"] for m in messages}
            continue

        if status_cb:
            await status_cb(f"Processing {day_label} ({len(messages)} messages)...")

        # Reorder messages by thread tree
        threaded = build_threaded_order(messages, prev_day_ids)
        chunks = [threaded[i:i + CHUNK_SIZE] for i in range(0, len(threaded), CHUNK_SIZE)]
        print(f"  Threaded into {len(chunks)} chunk(s).")

        for ci, chunk in enumerate(chunks):
            chunk_label = f"{day_label} {ci+1}/{len(chunks)}"
            log_lines = format_log(chunk, all_msgs)
            print(f"  --- LLM INPUT [{chunk_label}] ---")
            for line in log_lines.splitlines():
                print(f"  {line}")
            print(f"  --- END LLM INPUT ---")
            await extract_from_log(log_lines, day, chunk_label=chunk_label)

        prev_day_ids = {m["id"] for m in messages}
        total_days += 1

    print(f"\n{'='*50}")
    print(f"Done. {total_days} day(s) processed.")


# ── Standalone entry ───────────────────────────────────────────────────────────

async def _standalone():
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"Logged in as {client.user}")
        channel = client.get_channel(SUPPORT_CHANNEL_ID)
        if not channel:
            print(f"Channel {SUPPORT_CHANNEL_ID} not found")
            await client.close()
            return
        await run_import(channel, ADMIN_USER_IDS)
        await client.close()

    await client.start(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    asyncio.run(_standalone())