"""Shared result models for the retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FusedHit:
    """One candidate after RRF fusion, before reranking."""

    key: str  # namespaced id: "t::<chunk_id>" (text) or "v::<patch_id>" (visual)
    modality: str  # "text" | "image"
    fused_score: float
    ranks: dict = field(default_factory=dict)  # e.g. {"dense": 0, "sparse": 2}
    payload: dict = field(default_factory=dict)
    text: str | None = None  # chunk text for text hits (reranker input)


@dataclass(frozen=True)
class RerankedHit:
    """Final context block after reranking — what is sent to the LLM."""

    key: str
    modality: str
    fused_score: float
    rerank_score: float
    payload: dict = field(default_factory=dict)
    text: str | None = None
    # Sibling visual patches on the same page (visual citation evidence).
    visual_evidence: tuple = field(default_factory=tuple)
