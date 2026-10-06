"""Reciprocal Rank Fusion (RRF) for dense + sparse (+ visual) rankings.

Standard RRF (Cormack et al., SIGIR 2009), the rank-based fusion used by
most hybrid search stacks because it needs no score normalization across
heterogeneous retrievers (cosine vs BM25 vs CLIP are not comparable)::

    score(d) = Σ_rankings  weight / (k + rank(d))      # rank is 0-based here

Keys are namespaced by the caller (``t::<chunk_id>``, ``v::<patch_id>``)
so text and visual candidates never collide in the fused space.
"""

from __future__ import annotations


def rrf_fuse(
    rankings: list[list[str]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> dict[str, float]:
    """Fuse ordered key-lists into ``{key: fused_score}`` (descending order).

    Each ranking contributes ``weight / (k + rank)`` per key present in it;
    keys absent from a ranking simply get no contribution from it.
    """
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights length must match rankings length")
    weights = list(weights) if weights is not None else [1.0] * len(rankings)
    fused: dict[str, float] = {}
    for keys, w in zip(rankings, weights):
        for rank, key in enumerate(keys):
            fused[key] = fused.get(key, 0.0) + w / (k + rank)
    return fused


def ranks_of(rankings: list[list[str]], names: list[str]) -> dict[str, dict[str, int]]:
    """Map each key to ``{retriever_name: rank}`` for explainability."""
    out: dict[str, dict[str, int]] = {}
    for keys, name in zip(rankings, names):
        for rank, key in enumerate(keys):
            out.setdefault(key, {})[name] = rank
    return out
