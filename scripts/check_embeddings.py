"""
Diagnostic: for each extracted_history row, find its top 5 most similar
neighbors (excluding itself) using cosine similarity.

Run with: python scripts/check_embeddings.py

Useful for verifying embedding quality and spotting potential duplicates.
"""
import asyncio
import psycopg
from shared.config import POSTGRES_URL
from shared.embeddings import cosine_similarity


def fetch_all():
    with psycopg.connect(POSTGRES_URL) as conn:
        rows = conn.execute(
            """
            SELECT id, topic, summary, embedding
            FROM extracted_history
            WHERE view_count >= 0 AND embedding IS NOT NULL
            ORDER BY id ASC
            """
        ).fetchall()
    return [
        {"id": r[0], "topic": r[1], "summary": r[2], "embedding": r[3]}
        for r in rows
    ]


def check_embeddings():
    entries = fetch_all()
    if not entries:
        print("No entries with embeddings found.")
        return

    print(f"Checking {len(entries)} entries...\n{'='*60}")

    for entry in entries:
        scores = []
        for other in entries:
            if other["id"] == entry["id"]:
                continue
            score = cosine_similarity(entry["embedding"], other["embedding"])
            scores.append((score, other))

        scores.sort(key=lambda x: x[0], reverse=True)
        top5 = scores[:5]

        print(f"\n#{entry['id']} — {entry['topic']}")
        print(f"  Summary: {entry['summary'][:80]}...")
        print(f"  Top 5 similar:")
        for score, other in top5:
            marker = "  ⚠️ " if score >= 0.95 else "    "
            print(f"{marker}#{other['id']} {score*100:.1f}%  {other['topic']}")

    print(f"\n{'='*60}")
    print("⚠️  = 95%+ similarity (potential duplicate)")


if __name__ == "__main__":
    check_embeddings()