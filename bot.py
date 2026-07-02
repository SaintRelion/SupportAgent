"""
Discord bot entry point.
"""

import asyncio
import logging
import re
import selectors
import uuid

# ── Logging ────────────────────────────────────────────────────────────────────

class ColorFormatter(logging.Formatter):
    TAGS = [
        "[LLM]", "[NODE]", "[GRAPH]", "[TOOL]", "[RUN]",
        "[SUPPORT]", "[ESCALATION]", "[ANALYZER]", "[SCHEDULER]",
        "[INIT]", "[SHUTDOWN]", "[CMD]", "[MSG]", "[VERIFY]",
    ]

    _last_tag: str = ""

    def format(self, record: logging.LogRecord) -> str:
        time_str  = self.formatTime(record, "%H:%M:%S")
        level_str = f"[{record.levelname[0]}]"
        name_str  = record.name
        msg       = record.getMessage()

        current_tag = ""
        body        = msg
        for tag in self.TAGS:
            if msg.startswith(tag):
                current_tag = tag
                body        = msg[len(tag):].lstrip()
                break

        if not current_tag:
            ColorFormatter._last_tag = ""
            return f"{time_str} {level_str} {name_str}: {msg}"

        same_group = current_tag == ColorFormatter._last_tag
        ColorFormatter._last_tag = current_tag

        if same_group:
            indent = " " * (len(current_tag) + 1)
            return f"{indent}-- {body}"
        else:
            return f"\n{time_str} {level_str} {name_str}  {current_tag} {body}"


handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter())
logging.root.setLevel(logging.DEBUG)
logging.root.handlers = [handler]

logging.getLogger("discord").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

log = logging.getLogger("bot")

import discord
from discord.ext import commands

from support_bot.admin_review import start_history_review, start_rules_review

from shared.config import (
    DISCORD_BOT_TOKEN,
    SUPPORT_CHANNEL_ID,
    ANALYZER_INPUT_CHANNEL_ID,
    ANALYZER_OUTPUT_CHANNEL_ID,
    ADMIN_USER_IDS,
    KNOWLEDGE_LOG_CHANNEL_ID,
)
from shared.llm import get_llm
from support_bot.agent import run_agent, close_checkpointer, check_should_clear, clear_thread_history
from support_bot.rules import append_rule, load_rules

from analyzer.pipeline import analyze_recording

# ── Bot setup ──────────────────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.reactions = True

bot = commands.Bot(command_prefix="!", intents=intents)

CLOUDFRONT_RE = re.compile(
    r"https?://[a-z0-9]+\.cloudfront\.net/[^\s>\"']+",
    re.IGNORECASE,
)


# ── Guard ─────────────────────────────────────────────────────────────────────
 
def _is_admin(interaction: discord.Interaction) -> bool:
    return interaction.user.id in ADMIN_USER_IDS

# ── Message queue system ───────────────────────────────────────────────────────

MAX_CONCURRENT   = 5
_semaphore       = asyncio.Semaphore(MAX_CONCURRENT)
QUEUE_WARN_AFTER = 30    # seconds before "hang tight" message
QUEUE_DROP_AFTER = 120   # seconds before drop with apology

# user_id → [discord.Message, ...]
_user_queues:     dict[int, list] = {}
_user_processing: dict[int, bool] = {}


async def enqueue_user_message(message: discord.Message):
    """Add message to user queue and start processing if not already running."""
    user_id = message.author.id

    if user_id not in _user_queues:
        _user_queues[user_id] = []
    _user_queues[user_id].append(message)

    if _user_processing.get(user_id):
        log.info("[SUPPORT] User %s queued message (%d pending)", user_id, len(_user_queues[user_id]))
        return

    _user_processing[user_id] = True
    asyncio.create_task(_process_user_queue(user_id))


async def _process_user_queue(user_id: int):
    """Drain a user queue, batching messages that arrive while waiting for a slot."""
    try:
        while _user_queues.get(user_id):
            batch       = _user_queues.pop(user_id, [])
            last_msg    = batch[-1]
            warn_task   = asyncio.create_task(_warn_if_slow(last_msg))
            acquired    = False
            try:
                try:
                    await asyncio.wait_for(asyncio.shield(_semaphore.acquire()), timeout=QUEUE_DROP_AFTER)
                    acquired = True
                except asyncio.TimeoutError:
                    warn_task.cancel()
                    await last_msg.reply(
                        "We are quite busy right now and couldn't process your message in time. "
                        "Please try again in a moment."
                    )
                    return
                warn_task.cancel()
                await _handle_user_message_locked(last_msg, all_messages=batch)
            finally:
                if acquired:
                    _semaphore.release()
    finally:
        _user_processing[user_id] = False
        if _user_queues.get(user_id):
            _user_processing[user_id] = True
            asyncio.create_task(_process_user_queue(user_id))


async def _warn_if_slow(message: discord.Message):
    await asyncio.sleep(QUEUE_WARN_AFTER)
    try:
        await message.reply("We are a bit busy right now — hang tight, your message is up next.")
    except Exception:
        pass




# ── Events ─────────────────────────────────────────────────────────────────────

@bot.event
async def on_ready():
    log.info("[INIT] Bot online: %s (id=%s)", bot.user, bot.user.id)
    await bot.tree.sync()
    log.info("[INIT] Slash commands synced globally")


@bot.event
async def on_close():
    log.info("Bot shutting down...")
    await close_checkpointer()


@bot.event
async def on_error(event: str, *args, **kwargs):
    import traceback
    log.error("[BOT] Exception in %s:\n%s", event, traceback.format_exc())


@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    # Ignore messages outside our channels
    if message.channel.id not in (SUPPORT_CHANNEL_ID, ANALYZER_INPUT_CHANNEL_ID):
        return

    log.debug("[MSG] #%s | %s: %s", message.channel.name, message.author.name, message.content[:100])

    try:
        # ── Analyzer channel ───────────────────────────────────────────────────
        if message.channel.id == ANALYZER_INPUT_CHANNEL_ID:
            urls = CLOUDFRONT_RE.findall(message.content)
            if urls:
                log.info("[ANALYZER] CloudFront URL detected: %s", urls[0])
                await handle_recording(message, urls[0])
            await bot.process_commands(message)
            return

        # ── Support channel ────────────────────────────────────────────────────
        if message.channel.id == SUPPORT_CHANNEL_ID:
            is_admin = message.author.id in ADMIN_USER_IDS

            if not is_admin:
                await enqueue_user_message(message)

        await bot.process_commands(message)

    except Exception:
        import traceback
        log.error("[MSG] Exception while handling message from %s in #%s:\n%s",
                  message.author.name, message.channel.name, traceback.format_exc())


# ── Support handlers ───────────────────────────────────────────────────────────

async def _handle_user_message_locked(message: discord.Message, all_messages: list = None):
    user_id = message.author.id

    # Strip bot mention from content
    # Only respond if bot or @support is mentioned
    bot_mentioned   = bot.user in message.mentions
    support_mention = "@support" in message.content.lower()
    admin_tagged    = any(f"<@{uid}>" in message.content for uid in ADMIN_USER_IDS)

    if not (bot_mentioned or support_mention or admin_tagged):
        log.debug("[SUPPORT] No mention detected — ignoring message from %s", user_id)
        return

    # Image uploaded — answer the question + flag for verification
    has_image = bool(message.attachments and any(
        a.content_type and a.content_type.startswith("image/")
        for a in message.attachments
    ))

    thread_id = f"discord-{message.channel.id}-{user_id}"
    log.info("[SUPPORT] Routing user %s to agent | thread_id: %s | image=%s | admin_tagged=%s",
             user_id, thread_id, has_image, admin_tagged)

    # Collect all queued message contents for batched agent call
    def _clean(m: discord.Message) -> str:
        text = m.content
        if bot.user:
            text = text.replace(f'<@{bot.user.id}>', '').replace(f'<@!{bot.user.id}>', '')
        text = text.replace('@support', '').strip()
        has_img = any(a.content_type and a.content_type.startswith("image/") for a in m.attachments)
        if not text and not has_img:
            return "==empty, user may have messaged previously=="
        if has_img:
            text = (text or "") + " [Image attached]"
        return text
    messages_to_send = [_clean(m) for m in (all_messages or [message])]
    log.info("[SUPPORT] Sending %d message(s) to agent", len(messages_to_send))

    async with message.channel.typing():
        # Check BEFORE answering: pull recent checkpoint state + new message, ask LLM
        # if this looks concluded. If so, clear checkpoint now so the agent answers
        # fresh instead of dragging stale context into the new topic.
        if await check_should_clear(thread_id, messages_to_send[-1]):
            try:
                clear_thread_history(thread_id)
                log.info("[SUPPORT] User %s concluded previous exchange — cleared thread_id=%s before answering", user_id, thread_id)
            except Exception:
                import traceback
                log.error("[SUPPORT] Failed to clear checkpoints for thread_id=%s:\n%s", thread_id, traceback.format_exc())

        result = await run_agent(messages_to_send, thread_id)

    reply                 = result["reply"]
    escalate              = result["escalate"]
    log.info("[SUPPORT] Agent done | escalate=%s | reply_len=%d", escalate, len(reply))

    admin_mentions = " ".join(f"<@{uid}>" for uid in ADMIN_USER_IDS)

    # Escalation
    if escalate or not reply or has_image:
        if has_image:
            frame = "I'm not able to process images directly, so I'm passing this to the team for you."
        elif admin_tagged:
            frame = "Let me pass this to someone who can give you a better answer."
        else:
            frame = "I don't have enough information on this one. Passing it to the team."
        await message.reply(f"{frame} {admin_mentions}")
        log.info("[SUPPORT] Pinged admin for user %s", user_id)
        return

    await message.reply(reply)
    log.info("[SUPPORT] Replied to user %s", user_id)

    # Post knowledge usage to log channel
    matched_ids = result.get("matched_entry_ids", [])
    matched_scores = result.get("matched_entry_scores", {})
    top3_ids = result.get("top3_entry_ids", [])
    top3_scores = result.get("top3_entry_scores", {})

    log_channel = message.guild.get_channel(KNOWLEDGE_LOG_CHANNEL_ID)
    if log_channel and (matched_ids or top3_ids):
        from support_bot.admin_review import KnowledgeLogView
        if matched_ids:
            display_ids = matched_ids[:10]
            ids_str = "  ".join(f"`#{i}` {matched_scores.get(i,0)*100:.0f}%" for i in display_ids)
            label = "**Knowledge used:**"
        else:
            display_ids = top3_ids
            ids_str = "  ".join(f"`#{i}` {top3_scores.get(i,0)*100:.0f}%" for i in display_ids)
            label = "**No match — top 3:**"
        view = KnowledgeLogView(display_ids)
        await log_channel.send(
            f"**User:** {message.author.mention}\n"
            f"**Q:** {message.content[:300]}\n"
            f"{label} {ids_str}",
            view=view,
        )





# ── Analyzer handler ───────────────────────────────────────────────────────────

async def handle_recording(message: discord.Message, url: str):
    output_channel = bot.get_channel(ANALYZER_OUTPUT_CHANNEL_ID)
    if not output_channel:
        log.warning("[ANALYZER] Output channel not found: %s", ANALYZER_OUTPUT_CHANNEL_ID)
        return

    filename = url.split("/")[-1].split("?")[0] or "recording"
    log.info("[ANALYZER] Starting pipeline for: %s", filename)

    status = await message.reply("Downloading and analyzing recording, this may take a few minutes.")

    result = await analyze_recording(url)

    if result["error"]:
        log.error("[ANALYZER] Pipeline failed: %s", result["error"])
        await status.edit(content=f"Analysis failed after retries: {result['error']}")
        return

    log.info("[ANALYZER] Pipeline complete | transcript_len=%d | feedback_len=%d",
             len(result["transcript"]), len(result["feedback"]))

    await status.edit(content="Analysis complete. Feedback posted.")

    embed = discord.Embed(
        title=f"Call Feedback — {message.author.display_name} — {filename}",
        description=result["feedback"][:4000],
        color=discord.Color.blurple(),
    )
    await output_channel.send(embed=embed)

    if len(result["feedback"]) > 4000:
        remainder = result["feedback"][4000:]
        await output_channel.send(f"(continued)\n{remainder[:2000]}")


# ── Commands ───────────────────────────────────────────────────────────────────
@bot.tree.command(name="import_history", description="Import channel history and extract Q&A. Admin only.")
@discord.app_commands.checks.has_permissions(administrator=True)
async def import_history_cmd(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    try:
        from scripts.import_history import run_import

        channel = bot.get_channel(1460998685033758971)
        if not channel:
            await interaction.followup.send("Support channel not found.", ephemeral=True)
            return

        updates = []
        async def status_cb(msg: str):
            updates.append(msg)
            log.info("[CMD] import_history: %s", msg)

        await run_import(channel, ADMIN_USER_IDS, status_cb=status_cb, days_back=60)
        try:
            await interaction.followup.send(f"Import complete. Processed {len(updates)} days.", ephemeral=True)
        except Exception:
            await channel.send(f"✅ Import complete. Processed {len(updates)} days.")

    except Exception:
        import traceback
        tb = traceback.format_exc()
        log.error("[CMD] import_history failed:\n%s", tb)
        try:
            await interaction.followup.send(f"Import failed:\n```{tb[-1500:]}```", ephemeral=True)
        except Exception:
            await channel.send(f"❌ Import failed:\n```{tb[-1500:]}```")


@bot.tree.command(name="nuke_chat", description="Delete all messages in this channel. Admin only.")
@discord.app_commands.checks.has_permissions(administrator=True)
async def nuke_chat(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    # Bulk delete messages under 14 days (fast), then fall back for older ones
    deleted = await interaction.channel.purge(limit=None)
    await interaction.followup.send(f"Deleted {len(deleted)} messages.", ephemeral=True)

@bot.tree.command(
    name="review_knowledge",
    description="[Admin] Review and approve/edit/delete extracted knowledge entries.",
)
async def review_knowledge(interaction: discord.Interaction):
    if not _is_admin(interaction):
        await interaction.response.send_message(
            "You don't have permission to use this command.", ephemeral=True
        )
        return
    await start_history_review(interaction)
 
@bot.tree.command(
    name="rules",
    description="[Admin] View, edit, add, or delete rules sections.",
)
async def rules_command(interaction: discord.Interaction):
    if not _is_admin(interaction):
        await interaction.response.send_message(
            "You don't have permission to use this command.", ephemeral=True
        )
        return
    await start_rules_review(interaction)

@bot.event
async def on_ready_sync_commands():
    await bot.tree.sync()


# ── Entry ──────────────────────────────────────────────────────────────────────

async def _run_with_retry():
    """
    Retries bot.start() on connection failures (no internet, DNS issues, etc.)
    so a temporary outage doesn't kill the process. Uses increasing backoff,
    capped at 60s between attempts.
    """
    backoff = 5
    max_backoff = 60
    while True:
        try:
            await bot.start(DISCORD_BOT_TOKEN)
            break  # bot.start() only returns on clean logout — exit loop
        except (discord.errors.ConnectionClosed, discord.errors.GatewayNotFound,
                OSError, discord.errors.HTTPException) as e:
            log.error("[STARTUP] Connection failed: %s — retrying in %ss", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)
        except Exception as e:
            log.error("[STARTUP] Unexpected error: %s — retrying in %ss", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)


if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    loop = asyncio.SelectorEventLoop(selectors.SelectSelector())
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_run_with_retry())