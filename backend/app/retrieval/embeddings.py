"""Ollama query embedding for live retrieval."""

from __future__ import annotations

import httpx

from app.config import settings


def embed_query(text: str) -> list[float]:
    response = httpx.post(
        f"{settings.ollama_base_url}/api/embeddings",
        json={"model": settings.ollama_embedding_model, "prompt": text},
        timeout=30,
    )
    response.raise_for_status()
    embedding = response.json()["embedding"]
    expected_dims = settings.ollama_embedding_dimensions
    if len(embedding) != expected_dims:
        raise ValueError(
            f"Expected embedding dimension {expected_dims}, got {len(embedding)}"
        )
    return embedding
