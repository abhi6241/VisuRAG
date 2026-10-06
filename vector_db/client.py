"""Qdrant client factory (local embedded mode by default, server optional)."""

from __future__ import annotations

from qdrant_client import QdrantClient

from .config import VectorDBConfig


def create_qdrant_client(config: VectorDBConfig) -> QdrantClient:
    """Create a Qdrant client per *config*.

    - ``memory`` → ``QdrantClient(":memory:")`` (no disk I/O, for tests).
    - ``path`` → ``QdrantClient(path=...)`` (persistent local storage,
      no Docker/server needed — the default).
    - ``url`` → ``QdrantClient(url=..., api_key=...)`` (external server,
      e.g. Qdrant running in Docker).
    """
    if config.mode == "memory":
        return QdrantClient(":memory:")
    if config.mode == "path":
        config.path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(config.path))
    return QdrantClient(url=config.url, api_key=config.api_key)
