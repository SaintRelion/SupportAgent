"""
Load and update rules.md.
Parse sections by ### headers and ** subsections.
Load and query extracted knowledge from DB.
"""
import os
import re
import psycopg
from shared.config import RULES_FILE, POSTGRES_URL


# ── Rules (rules.md) ──────────────────────────────────────────────────────────

def load_rules() -> str:
    try:
        with open(RULES_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def overwrite_rules(content: str):
    with open(RULES_FILE, "w", encoding="utf-8") as f:
        f.write(content)


def append_rule(rule: str):
    with open(RULES_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n- {rule}\n")


# ── Section parsing ───────────────────────────────────────────────────────────

def parse_sections(content: str) -> list[dict]:
    """
    Split rules.md into sections by ### headers.
    Each section dict: { title, body, subsections, raw }
      - title: text after ###
      - body: full text of this section (excluding the ### line itself)
      - subsections: list of { title, body } for each **Title** block found
      - raw: the complete original block including the ### line
    """
    pattern = re.compile(r"^### (.+)$", re.MULTILINE)
    matches = list(pattern.finditer(content))
    sections = []

    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[start:end].strip()
        raw = content[match.start():end]

        # Parse ** subsections ** within the body
        sub_pattern = re.compile(r"\*\*(.+?)\*\*[:\s]*([\s\S]*?)(?=\n\*\*|\Z)")
        subsections = []
        for sm in sub_pattern.finditer(body):
            subsections.append({
                "title": sm.group(1).strip(),
                "body": sm.group(2).strip(),
            })

        sections.append({
            "title": title,
            "body": body,
            "subsections": subsections,
            "raw": raw,
        })

    return sections


def replace_section(title: str, new_body: str) -> bool:
    """
    Replace the body of a ### section by title.
    Returns True if found and replaced, False if not found.
    """
    content = load_rules()
    sections = parse_sections(content)

    for section in sections:
        if section["title"].lower() == title.lower():
            old_raw = section["raw"]
            new_raw = f"### {section['title']}\n{new_body.strip()}\n"
            overwrite_rules(content.replace(old_raw, new_raw, 1))
            return True
    return False


def replace_subsection(section_title: str, subsection_title: str, new_body: str) -> bool:
    """
    Replace a **Subsection** body within a named ### section.
    Returns True if found and replaced.
    """
    content = load_rules()
    sections = parse_sections(content)

    for section in sections:
        if section["title"].lower() != section_title.lower():
            continue
        for sub in section["subsections"]:
            if sub["title"].lower() != subsection_title.lower():
                continue
            old_block = f"**{sub['title']}**"
            # Find the full subsection in raw and replace its body
            sub_pattern = re.compile(
                rf"(\*\*{re.escape(sub['title'])}\*\*[:\s]*)([\s\S]*?)(?=\n\*\*|\Z)"
            )
            new_raw = sub_pattern.sub(
                lambda m: m.group(1) + "\n" + new_body.strip() + "\n",
                section["raw"],
                count=1,
            )
            overwrite_rules(content.replace(section["raw"], new_raw, 1))
            return True
    return False


def add_section(title: str, body: str):
    """Append a new ### section to rules.md."""
    with open(RULES_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n### {title}\n{body.strip()}\n")


def delete_section(title: str) -> bool:
    """Remove a ### section entirely. Returns True if found."""
    content = load_rules()
    sections = parse_sections(content)
    for section in sections:
        if section["title"].lower() == title.lower():
            overwrite_rules(content.replace(section["raw"], "", 1).strip() + "\n")
            return True
    return False





# ── Extracted history (DB) ────────────────────────────────────────────────────




def db_date_has_entries(source_date) -> bool:
    """Returns True if any entries already exist for this date."""
    with psycopg.connect(POSTGRES_URL) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM extracted_history WHERE source_date = %s AND view_count >= 0",
            (source_date,),
        ).fetchone()[0]
    return count > 0


def db_save_entry(source_date, topic, conversation, summary, embedding=None):
    with psycopg.connect(POSTGRES_URL) as conn:
        conn.execute(
            """
            INSERT INTO extracted_history
                (source_date, topic, conversation, summary, view_count, embedding, created_at)
            VALUES (%s, %s, %s, %s, 0, %s, NOW())
            """,
            (source_date, topic, conversation, summary, embedding),
        )
        conn.commit()


def db_fetch_all() -> list[dict]:
    """Fetch all non-deleted entries ordered by view_count DESC."""
    with psycopg.connect(POSTGRES_URL) as conn:
        rows = conn.execute(
            """
            SELECT id, source_date, topic, summary, conversation, view_count
            FROM extracted_history
            WHERE view_count >= 0
            ORDER BY view_count DESC, source_date ASC, id ASC
            """,
        ).fetchall()
    return [
        {
            "id": r[0], "source_date": r[1], "topic": r[2],
            "summary": r[3], "conversation": r[4], "view_count": r[5],
        }
        for r in rows
    ]


def db_fetch_all_with_embeddings() -> list[dict]:
    """Fetch all non-deleted entries that have embeddings, for cosine similarity ranking."""
    with psycopg.connect(POSTGRES_URL) as conn:
        rows = conn.execute(
            """
            SELECT id, topic, summary, conversation, view_count, embedding
            FROM extracted_history
            WHERE view_count >= 0 AND embedding IS NOT NULL
            ORDER BY source_date DESC, id DESC
            """,
        ).fetchall()
    return [
        {
            "id": r[0], "topic": r[1], "summary": r[2],
            "conversation": r[3], "view_count": r[4], "embedding": r[5],
        }
        for r in rows
    ]


def db_save_embedding(entry_id: int, embedding: list[float]):
    """Update embedding for an existing entry (used for backfill)."""
    with psycopg.connect(POSTGRES_URL) as conn:
        conn.execute(
            "UPDATE extracted_history SET embedding = %s WHERE id = %s",
            (embedding, entry_id),
        )
        conn.commit()


def db_increment_view(entry_id: int):
    """Increment view_count by 1."""
    with psycopg.connect(POSTGRES_URL) as conn:
        conn.execute(
            "UPDATE extracted_history SET view_count = view_count + 1 WHERE id = %s",
            (entry_id,),
        )
        conn.commit()


def db_update_entry(entry_id: int, topic: str = None, summary: str = None):
    """Update topic and/or summary. view_count unchanged."""
    fields = []
    values = []
    if topic is not None:
        fields.append("topic = %s")
        values.append(topic)
    if summary is not None:
        fields.append("summary = %s")
        values.append(summary)
    if not fields:
        return
    values.append(entry_id)
    with psycopg.connect(POSTGRES_URL) as conn:
        conn.execute(
            f"UPDATE extracted_history SET {', '.join(fields)} WHERE id = %s",
            values,
        )
        conn.commit()


def db_fetch_entry(entry_id: int) -> dict | None:
    """Fetch a single entry by ID."""
    with psycopg.connect(POSTGRES_URL) as conn:
        row = conn.execute(
            """
            SELECT id, source_date, topic, summary, conversation, view_count
            FROM extracted_history WHERE id = %s
            """,
            (entry_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "id": row[0], "source_date": row[1], "topic": row[2],
        "summary": row[3], "conversation": row[4], "view_count": row[5],
    }


def db_delete_entry(entry_id: int):
    """Soft delete — sets view_count to -1, row stays for audit."""
    with psycopg.connect(POSTGRES_URL) as conn:
        conn.execute(
            "UPDATE extracted_history SET view_count = -1 WHERE id = %s",
            (entry_id,),
        )
        conn.commit()


# ── Checkpointer cleanup (LangGraph Postgres tables) ──────────────────────────
#
# AsyncPostgresSaver persists graph state in three tables, all keyed by
# thread_id: checkpoints, checkpoint_writes, checkpoint_blobs.
# This deletes rows for ONE thread_id only — never a blanket wipe.

def db_clear_thread_checkpoints(thread_id: str) -> dict:
    """
    Delete all checkpoint rows for a single thread_id from
    checkpoints, checkpoint_writes, checkpoint_blobs.
    Returns a dict of rows deleted per table, for logging.
    Safe no-op if thread_id has no rows.
    """
    deleted = {}
    with psycopg.connect(POSTGRES_URL) as conn:
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            cur = conn.execute(
                f"DELETE FROM {table} WHERE thread_id = %s",
                (thread_id,),
            )
            deleted[table] = cur.rowcount
        conn.commit()
    return deleted


# ── Combined load for agent ───────────────────────────────────────────────────

def load_rules_section() -> str:
    rules = load_rules().strip()
    if not rules:
        return ""
    return (
        "## OFFICIAL COMPANY RULES (rules.md — source of truth, admin-maintained)\n"
        f"{rules}"
    )


def format_entries(entries: list[dict]) -> str:
    if not entries:
        return ""
    parts = [f"### {e['topic']}\n{e['summary']}" for e in entries]
    return (
        "## LEARNED FROM PAST CONVERSATIONS (extracted from prior admin answers — "
        "may reflect one-off exceptions or case-by-case calls an admin made, "
        "not necessarily standing policy)\n\n" + "\n\n".join(parts)
    )