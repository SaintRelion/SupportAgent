"""
Embedding generation via OpenRouter (openai/text-embedding-3-large, 3072 dims).
"""
import logging
import numpy as np
import httpx

from shared.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

log = logging.getLogger("embeddings")

EMBEDDING_MODEL = "openai/text-embedding-3-large"
EMBEDDING_DIMS = 3072


async def embed_text(text: str) -> list[float]:
    """Generate embedding for a single text string."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OPENROUTER_BASE_URL}/embeddings",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": EMBEDDING_MODEL, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]


def embed_text_sync(text: str) -> list[float]:
    """Sync version for use in non-async contexts."""
    import httpx as _httpx
    with _httpx.Client() as client:
        resp = client.post(
            f"{OPENROUTER_BASE_URL}/embeddings",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": EMBEDDING_MODEL, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))


def rank_entries_by_similarity(
    query_vector: list[float],
    entries: list[dict],
    top_k: int = 5,
    min_score: float = 0.75,
) -> list[dict]:
    """
    Score each entry against query_vector, return top_k above min_score.
    Entries must have an 'embedding' key with a list[float].
    """
    scored = []
    for entry in entries:
        emb = entry.get("embedding")
        if not emb:
            continue
        score = cosine_similarity(query_vector, emb)
        scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    import logging
    _log = logging.getLogger("embeddings")
    top_preview = [(f"{s:.1%}", e["topic"]) for s, e in scored[:3]]
    _log.info("[EMBED] top scores: %s", top_preview)

    matched = [(score, e) for score, e in scored[:top_k] if score >= min_score]

    # Always include top 3 for visibility, even if below threshold
    # Mark them so caller knows they didn't pass the threshold
    if not matched:
        return [(score, e, False) for score, e in scored[:3] if scored]

    return [(score, e, True) for score, e in matched]