"""Sparse retrieval layer: in-memory BM25 over indexed text chunks.

Uses ``rank-bm25`` (``BM25Okapi``) with a dependency-free regex tokenizer
(lowercased alphanumeric tokens) — no NLTK downloads, deterministic, and a
good fit for exact keyword matches like part numbers (``NE555``), pin names
(``VCC``), and spec values (``5V``) that dense embeddings may paraphrase away.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization shared by index and query."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """BM25 over (key → text) pairs. Rebuilt whenever chunks are re-indexed."""

    def __init__(self) -> None:
        self._keys: list[str] = []
        self._texts: list[str] = []
        self._tokenized: list[list[str]] = []
        self._bm25 = None

    def __len__(self) -> int:
        return len(self._keys)

    def build(self, keys: list[str], texts: list[str]) -> None:
        """(Re)build the index from parallel key/text lists."""
        if len(keys) != len(texts):
            raise ValueError("keys and texts length mismatch")
        from rank_bm25 import BM25Okapi

        self._keys = list(keys)
        self._texts = list(texts)
        self._tokenized = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokenized) if self._keys else None

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Return ``[(key, bm25_score)]`` sorted by descending score."""
        if self._bm25 is None or not query.strip():
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            ((k, float(s)) for k, s in zip(self._keys, scores) if s > 0),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return ranked[: max(top_k, 0)]
