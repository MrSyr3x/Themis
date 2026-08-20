"""Gemini embedding generation for document chunks."""

from __future__ import annotations

import httpx

from app.config import settings

GEMINI_EMBED_MODEL = "text-embedding-004"
EMBED_BATCH_SIZE = 100


def embed_texts(texts: list[str], *, batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    if not texts:
        return []

    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        resp = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_EMBED_MODEL}:batchEmbedContents?key={settings.gemini_api_key}",
            json={
                "requests": [
                    {"model": f"models/{GEMINI_EMBED_MODEL}", "content": {"parts": [{"text": t}]}}
                    for t in batch
                ]
            },
            timeout=60,
        )
        resp.raise_for_status()
        for item in resp.json()["embeddings"]:
            vectors.append(item["values"])

    return vectors
