"""
Rule extraction:
  - Daily job: scans all escalated threads from last 24h
  - Immediate: extract from a single thread right after admin resolves it
"""

import json
import re
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage, SystemMessage
from psycopg_pool import ConnectionPool

from shared.config import POSTGRES_URL
from shared.llm import get_llm
from support_bot.rules import load_rules, overwrite_rules


EXTRACTION_PROMPT = """You are a knowledge curator for a company support bot.

Below are recent support conversations. Each conversation shows:
- USER: the question asked
- ADMIN: the answer given by a human admin (these are ground truth)

Your job:
1. Extract anything the admin said that could help answer a similar question in the future.
   This includes: policies, facts, clarifications, partial answers, or even "we don't have X yet".
   Even negative answers are useful rules (e.g. "We do not currently offer X.").
2. Convert them into concise, factual statements the bot can use to answer future questions.
3. Avoid duplicating rules that already exist.
4. Return ONLY a JSON array of new rule strings, no explanation.
5. If there is truly nothing useful to extract, return an empty array: []

Example output:
["Refunds are processed within 5 business days.", "We do not currently offer payment plans.", "Free trial lasts 14 days, no credit card required."]

Existing rules (do not duplicate):
{existing_rules}

Conversations:
{conversations}
"""


def _fetch_thread_messages(conn, thread_id: str) -> list[dict]:
    """Fetch latest checkpoint messages for a single thread."""
    row = conn.execute(
        """
        SELECT checkpoint -> 'channel_values' -> 'messages' AS messages
        FROM checkpoints
        WHERE thread_id = %s AND checkpoint_ns = ''
        ORDER BY checkpoint_id DESC
        LIMIT 1
        """,
        (thread_id,),
    ).fetchone()

    if not row or not row[0]:
        return []

    raw = row[0]
    if isinstance(raw, str):
        raw = json.loads(raw)
    return raw if isinstance(raw, list) else []


def fetch_recent_escalated_threads(hours: int = 24) -> list[dict]:
    """Pull all escalated threads from last N hours."""
    threads = []
    with ConnectionPool(POSTGRES_URL) as pool:
        with pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT thread_id
                FROM checkpoints
                WHERE checkpoint_ns = ''
                  AND checkpoint::jsonb -> 'channel_values' -> 'escalate' = 'true'::jsonb
                """,
            ).fetchall()

            for (thread_id,) in rows:
                messages = _fetch_thread_messages(conn, thread_id)
                if messages:
                    threads.append({"thread_id": thread_id, "messages": messages})

    return threads


def fetch_single_thread(thread_id: str) -> list[dict]:
    """Fetch a single thread by thread_id."""
    with ConnectionPool(POSTGRES_URL) as pool:
        with pool.connection() as conn:
            messages = _fetch_thread_messages(conn, thread_id)
            if messages:
                return [{"thread_id": thread_id, "messages": messages}]
    return []


def format_conversations(threads: list[dict]) -> str:
    """Format thread messages into readable conversation blocks."""
    blocks = []
    for t in threads:
        lines = [f"--- Thread {t['thread_id']} ---"]
        for msg in t["messages"]:
            role = msg.get("type", msg.get("role", "unknown"))
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            if role in ("human", "user"):
                lines.append(f"USER: {content}")
            elif role in ("ai", "assistant"):
                if content and not content.startswith("ESCALATE"):
                    lines.append(f"BOT: {content}")
            elif role == "admin":
                lines.append(f"ADMIN: {content}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _run_extraction(threads: list[dict]) -> list[str]:
    """Core extraction logic — shared by daily job and immediate trigger."""
    if not threads:
        return []

    conversations = format_conversations(threads)
    existing_rules = load_rules()
    llm = get_llm(temperature=0.1, model_override="deepseek/deepseek-v4-flash")

    response = llm.invoke([
        SystemMessage(content="You extract knowledge from support conversations."),
        HumanMessage(content=EXTRACTION_PROMPT.format(
            existing_rules=existing_rules,
            conversations=conversations,
        )),
    ])

    raw = response.content.strip()
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        new_rules: list[str] = json.loads(raw)
    except json.JSONDecodeError:
        print(f"Failed to parse LLM response as JSON:\n{raw}")
        return []

    if not new_rules:
        return []

    # Generate a short topic label using last 5 messages + admin answer as context
    try:
        # Collect last 5 messages across all threads for context
        context_lines = []
        for t in threads:
            msgs = t.get("messages", [])[-5:]
            for msg in msgs:
                role = msg.get("type", msg.get("role", ""))
                content_text = msg.get("content", "")
                if isinstance(content_text, list):
                    content_text = " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content_text)
                if role in ("human", "user"):
                    context_lines.append(f"USER: {content_text}")
                elif role in ("ai", "assistant") and content_text and not content_text.startswith("ESCALATE"):
                    context_lines.append(f"BOT: {content_text}")
                elif role == "admin":
                    context_lines.append(f"ADMIN: {content_text}")
        context = "\n".join(context_lines) if context_lines else conversations

        topic_response = llm.invoke([
            HumanMessage(content=(
                f"Given this support conversation and the admin's clarification, "
                f"summarize the topic in 3-6 words. Focus on what the user needed help with. "
                f"Return only the topic label, no punctuation or explanation.\n\n{context}"
            ))
        ])
        topic = topic_response.content.strip().strip(".")
    except Exception:
        topic = "General"

    current = load_rules()
    additions = "\n".join(f"- {r}" for r in new_rules)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    overwrite_rules(
        current.rstrip() + f"\n\n## {topic} ({timestamp})\n{additions}\n"
    )

    return new_rules



def extract_and_append_rules() -> list[str]:
    """Daily job — extract from all escalated threads in last 24h."""
    print(f"[{datetime.utcnow().isoformat()}] Starting daily rule extraction...")
    threads = fetch_recent_escalated_threads(hours=24)
    new_rules = _run_extraction(threads)
    print(f"Added {len(new_rules)} new rules.")
    return new_rules


def extract_from_thread(thread_id: str) -> list[str]:
    """
    Immediate extraction — called right after an escalation is resolved.
    Extracts rules from a single thread only.
    """
    print(f"[{datetime.utcnow().isoformat()}] Immediate extraction for thread: {thread_id}")
    threads = fetch_single_thread(thread_id)
    new_rules = _run_extraction(threads)
    print(f"Immediate extraction added {len(new_rules)} new rules from thread {thread_id}.")
    return new_rules


if __name__ == "__main__":
    rules = extract_and_append_rules()
    for r in rules:
        print(f"  + {r}")


def _parse_conversation(conversation: str) -> list[dict]:
    """Convert 'USER: ...' / 'ADMIN: ...' string into message dicts."""
    messages = []
    for line in conversation.strip().splitlines():
        if line.startswith("USER: "):
            messages.append({"type": "human", "content": line[6:]})
        elif line.startswith("ADMIN: "):
            messages.append({"type": "admin", "content": line[7:]})
        elif line.startswith("BOT: "):
            messages.append({"type": "ai", "content": line[5:]})
    return messages


def extract_from_conversation(conversation: str, thread_id: str = "manual", db_id: int = None) -> list[str]:
    """
    Immediate extraction from a raw conversation string.
    Used after escalation resolution — passes the in-memory log directly,
    bypassing the checkpoint store (which doesn't have admin messages).
    """
    print(f"[{datetime.utcnow().isoformat()}] Extracting from conversation (thread: {thread_id})")
    threads = [{"thread_id": thread_id, "messages": _parse_conversation(conversation)}]
    new_rules = _run_extraction(threads)
    print(f"Extracted {len(new_rules)} new rules.")
    return new_rules