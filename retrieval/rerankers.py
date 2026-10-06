"""Rerankers: cross-encoder (production) + heuristic (offline/tests).

- :class:`HeuristicReranker` — dependency-free token-overlap scorer with an
  exact-phrase bonus. Deterministic, zero downloads; used for tests and as
  a safe fallback when model weights are unavailable.
- :class:`CrossEncoderReranker` — lazy ``sentence_transformers.CrossEncoder``
  (default ``cross-encoder/ms-marco-MiniLM-L-6-v2``; swap in
  ``BAAI/bge-reranker-base`` for BGE quality). The transformer + weights
  load only on first :meth:`rerank` call. Joint query–document encoding
  beats bi-encoder cosine for the final top-k cut before the LLM.
"""

from __future__ import annotations

from typing import Protocol

from .bm25_index import tokenize


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[dict]) -> list[tuple[int, float]]:
        """Score candidates (each with a ``text`` field).

        Returns ``[(candidate_index, score)]`` sorted by descending score.
        """
        ...


class HeuristicReranker:
    """Offline reranker: token overlap + exact-phrase bonus."""

    def __init__(self, phrase_bonus: float = 2.0) -> None:
        self.phrase_bonus = phrase_bonus

    def score(self, query: str, text: str) -> float:
        q_tokens, d_tokens = set(tokenize(query)), set(tokenize(text or ""))
        overlap = len(q_tokens & d_tokens)
        bonus = self.phrase_bonus if query.lower().strip() in (text or "").lower() else 0.0
        # Small length-normalization so long blocks don't always win.
        return overlap / (1.0 + 0.05 * len(d_tokens)) + bonus

    def rerank(self, query: str, candidates: list[dict]) -> list[tuple[int, float]]:
        scored = [
            (i, self.score(query, c.get("text") or "")) for i, c in enumerate(candidates)
        ]
        return sorted(scored, key=lambda iv: iv[1], reverse=True)


class CrossEncoderReranker:
    """Production reranker over sentence-transformers CrossEncoders.

    Example models: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (fast default),
    ``BAAI/bge-reranker-base`` / ``BAAI/bge-reranker-v2-m3`` (BGE quality).
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.model_name = model
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is not installed; run "
                    "`pip install sentence-transformers` (inside venv) or use "
                    "HeuristicReranker instead."
                ) from e
            kwargs = {"device": self.device} if self.device else {}
            self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model

    def rerank(self, query: str, candidates: list[dict]) -> list[tuple[int, float]]:
        if not candidates:
            return []
        model = self._load()
        pairs = [(query, c.get("text") or "") for c in candidates]
        scores = model.predict(pairs, batch_size=self.batch_size, convert_to_numpy=True)
        ranked = sorted(
            ((i, float(s)) for i, s in enumerate(scores)),
            key=lambda iv: iv[1],
            reverse=True,
        )
        return ranked
