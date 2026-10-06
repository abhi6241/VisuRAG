"""Embedding providers: deterministic offline default + lazy real models.

- :class:`HashEmbeddingProvider` — dependency-free, deterministic
  (SHA256-seeded Gaussian, L2-normalized). Used for tests and offline
  indexing; lets the full ingest → upsert → search path run with zero
  model downloads.
- :class:`FastEmbedTextProvider` — lazy ``fastembed.TextEmbedding``
  (ONNX, e.g. ``BAAI/bge-small-en-v1.5`` → 384-d). Downloaded on first
  use only.
- :class:`SentenceTransformerProvider` — lazy ``sentence_transformers``
  (optional dependency; imported only when instantiated).
- :class:`FastEmbedImageProvider` — lazy ``fastembed.ImageEmbedding``
  (ONNX CLIP, e.g. ``Qdrant/clip-ViT-B-32-vision`` → 512-d) over patch
  image files. Downloaded on first use only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    dim: int

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return ``(len(texts), dim)`` float32 array, rows L2-normalized."""
        ...

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        """Return ``(len(paths), dim)`` float32 array, rows L2-normalized."""
        ...


def _normalize(rows: np.ndarray) -> np.ndarray:
    rows = rows.astype(np.float32, copy=False)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return rows / norms


class HashEmbeddingProvider:
    """Deterministic pseudo-embeddings from SHA256 (offline / tests).

    Each input string seeds a Gaussian draw; outputs are normalized so
    cosine similarity behaves like real encoders. Different *labels*
    (e.g. ``"text"`` vs ``"image"`` prefixes used by callers) give
    different vectors for the same raw content.
    """

    def __init__(self, dim: int = 384) -> None:
        if dim <= 0:
            raise ValueError("dim must be > 0")
        self.dim = dim

    def _one(self, key: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim, dtype=np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._one(f"text::{t}") for t in texts]))

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        return _normalize(np.stack([self._one(f"image::{p}") for p in paths]))


class FastEmbedTextProvider:
    """Real text embeddings via FastEmbed (lazy, downloads on first use)."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model)
        # fastembed exposes _dim; fall back to probing if unavailable.
        try:
            self.dim: int = int(self._model._dim)  # type: ignore[attr-defined]
        except Exception:
            self.dim = len(next(iter(self._model.embed(["probe"]))))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.asarray(list(self._model.embed(texts)), dtype=np.float32))

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        raise NotImplementedError("text-only provider; use FastEmbedImageProvider")


class SentenceTransformerProvider:
    """Text embeddings via sentence-transformers (lazy optional import)."""

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed; "
                "run `pip install sentence-transformers` or use "
                "FastEmbedTextProvider instead."
            ) from e
        self._model = SentenceTransformer(model)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return _normalize(
            np.asarray(
                self._model.encode(texts, normalize_embeddings=True), dtype=np.float32
            )
        )

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        raise NotImplementedError("text-only provider")


class FastEmbedImageProvider:
    """Visual embeddings via FastEmbed ONNX CLIP (lazy, downloads on first use)."""

    def __init__(self, model: str = "Qdrant/clip-ViT-B-32-vision") -> None:
        from fastembed import ImageEmbedding

        self._model = ImageEmbedding(model_name=model)
        try:
            self.dim: int = int(self._model._dim)  # type: ignore[attr-defined]
        except Exception:
            self.dim = len(next(iter(self._model.embed([str(Path("."))]))))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError("image-only provider")

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        return _normalize(
            np.asarray(
                list(self._model.embed([str(p) for p in paths])), dtype=np.float32
            )
        )
