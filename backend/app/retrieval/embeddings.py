"""Gemini query embedding for live retrieval."""

from __future__ import annotations

import httpx

from app.config import settings

GEMINI_EMBED_MODEL = "text-embedding-004"
GEMINI_EMBED_DIMENSIONS = 768


def embed_query(text: str) -> list[float]:
    response = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_EMBED_MODEL}:embedContent?key={settings.gemini_api_key}",
        json={
            "model": f"models/{GEMINI_EMBED_MODEL}",
            "content": {"parts": [{"text": text}]},
        },
        timeout=30,
    )
    response.raise_for_status()
    embedding = response.json()["embedding"]["values"]
    return embedding
