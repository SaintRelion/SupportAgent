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
from typing import Optional

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

Your job:
1. Identify exchanges where a user raised a situation or question and an admin resolved it with USEFUL information.
   - Useful = something that helps answer similar situations in the future (policy, decision criteria, procedure, exception, assignment).
   - SKIP if the admin only said things like "fixed", "done", "ok", "handled", "sure" with no explanation.
   - SKIP if an image was involved and the admin gave no context about what they saw or why they decided what they did.
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


def strip_mentions(content: str, admin_ids: set[int]) -> str:
    """
    Replace <@user_id> with @ADMIN or @USER based on whether the ID is a known admin.
    Falls back to @USER for any unresolved mention.
    """
    def replace(match):
        uid = int(match.group(1))
        return "@ADMIN" if uid in admin_ids else "@USER"
    return re.sub(r"<@!?(\d+)>", replace, content)


def format_log(messages: list[dict]) -> str:
    return "\n".join(
        f"[{m['timestamp']}] {m['role']} ({m['author']}): {m['content']}"
        for m in messages
    )


async def fetch_messages_by_day(
    channel: discord.TextChannel,
    admin_ids: list[int],
    days_back: int,
) -> dict[date, list[dict]]:
    days: dict[date, list[dict]] = defaultdict(list)
    admin_id_set = set(admin_ids)
    print("Fetching messages...")
    count = 0

    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    async for msg in channel.history(limit=None, oldest_first=True, after=cutoff):
        has_image = any(
            a.content_type and a.content_type.startswith("image/")
            for a in msg.attachments
        )
        content_text = strip_mentions(msg.content.strip(), admin_id_set)
        if has_image:
            content_text = (content_text + " [Image Attached]").strip()
        if not content_text:
            continue

        day = msg.created_at.astimezone(timezone.utc).date()
        role = "ADMIN" if msg.author.id in admin_id_set else ("BOT" if msg.author.bot else "USER")

        days[day].append({
            "timestamp": msg.created_at.strftime("%H:%M"),
            "author":    msg.author.display_name,
            "role":      role,
            "content":   content_text,
        })
        count += 1

    print(f"Fetched {count} messages across {len(days)} day(s).")
    return dict(sorted(days.items()))


async def extract_from_log(
    log: str,
    source_date: date,
    chunk_label: str = "",
) -> list[str]:
    llm = get_llm(model_override="deepseek/deepseek-v4-flash", temperature=0.1)

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
                    embedding = await embed_text(f"{topic}\n{summary}")
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
        print(f"  [{chunk_label}] {len(unanswered)} unanswered — carrying forward.")

    return unanswered


async def run_import(
    channel: discord.TextChannel,
    admin_ids: list[int],
    status_cb=None,
    days_back: int = 30,
):
    days = await fetch_messages_by_day(channel, admin_ids, days_back)
    carry = []
    total_days = 0

    all_days = list(days.items())
    print(f"\nProcessing {len(all_days)} day(s)...\n{'='*50}")

    for day, messages in all_days:
        day_label = day.strftime("%B %d, %Y")
        chunks = [messages[i:i + CHUNK_SIZE] for i in range(0, len(messages), CHUNK_SIZE)]

        print(f"\n── {day_label} ({len(messages)} messages, {len(chunks)} chunk(s)) ──")

        if db_date_has_entries(day):
            print(f"  [SKIP] {day_label} already processed.")
            if status_cb:
                await status_cb(f"Skipping {day_label} (already processed).")
            continue

        if status_cb:
            await status_cb(f"Processing {day_label} ({len(messages)} messages)...")

        for ci, chunk in enumerate(chunks):
            chunk_label = f"{day_label} {ci+1}/{len(chunks)}"
            log_lines   = format_log(chunk)

            if carry:
                carried   = "=== Carried (unanswered from previous chunk) ===\n" + "\n".join(carry)
                log_lines = carried + "\n\n" + log_lines

            carry = await extract_from_log(log_lines, day, chunk_label=chunk_label)

        total_days += 1

    print(f"\n{'='*50}")
    print(f"Done. {total_days} day(s) processed.")
    if carry:
        print(f"Note: {len(carry)} item(s) still unanswered at end of history.")


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