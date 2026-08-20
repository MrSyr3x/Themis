"""Ollama embedding generation for document chunks."""

from __future__ import annotations

import httpx

from app.config import settings

EMBED_BATCH_SIZE = 50


def embed_texts(texts: list[str], *, batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    if not texts:
        return []

    expected_dims = settings.ollama_embedding_dimensions
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        # Ollama doesn't support batch embedding, send one at a time
        for text in batch:
            resp = httpx.post(
                f"{settings.ollama_base_url}/api/embeddings",
                json={"model": settings.ollama_embedding_model, "prompt": text},
                timeout=60,
            )
            resp.raise_for_status()
            embedding = resp.json()["embedding"]
            if len(embedding) != expected_dims:
                raise ValueError(
                    f"Expected embedding dimension {expected_dims}, got {len(embedding)}"
                )
            vectors.append(embedding)

    return vectors
