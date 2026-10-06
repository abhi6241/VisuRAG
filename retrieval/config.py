"""Retrieval configuration (hybrid fan-out + fusion + rerank budgets)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalConfig:
    """Budgets for each retrieval stage (query → candidates → context)."""

    # Fan-out widths: how many candidates each first-stage retriever returns.
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_visual: int = 20  # only used when a visual query encoder is set

    # Reciprocal Rank Fusion smoothing (standard RRF constant).
    rrf_k: int = 60

    # How many fused candidates enter the reranker.
    candidate_pool_size: int = 20

    # Final context size sent to the LLM.
    final_top_k: int = 5

    # Reranking switch. Heuristic (offline) by default; set a
    # CrossEncoderReranker on the engine for production quality.
    use_reranker: bool = True

    def __post_init__(self) -> None:
        for name in (
            "top_k_dense",
            "top_k_sparse",
            "top_k_visual",
            "candidate_pool_size",
            "final_top_k",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.rrf_k < 0:
            raise ValueError("rrf_k must be >= 0")
