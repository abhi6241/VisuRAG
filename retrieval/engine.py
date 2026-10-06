"""Hybrid retrieval engine: dense + BM25 → RRF → rerank → visual evidence.

Read path for one text query::

    dense (Qdrant cosine over visurag_text) ─┐
                                             ├─→ RRF fuse ─→ rerank ─→ top-k + sibling patches
    sparse (in-memory BM25 over chunk text) ─┘         ↑
                  optional 3rd arm: visual dense ──────┘ (only when a
                  text→visual query encoder, e.g. CLIP text tower, is set)

Key design notes:

- Score spaces (cosine vs BM25) are never compared directly — RRF fuses
  *ranks*, so heterogeneous retrievers combine without normalization.
- Candidate keys are namespaced (``t::<chunk_id>`` / ``v::<patch_id>``).
- Visual evidence for a text hit = sibling patches on the same
  (``document_id``, ``page_num``) via a filtered scroll — this is what the
  frontend renders as visual citations, no cross-modal query encoder needed.
- Real cross-modal text→patch search is supported as an opt-in third RRF
  arm via ``visual_text_encoder`` (e.g. a CLIP text tower whose space
  matches the visual collection).
"""

from __future__ import annotations

from collections.abc import Sequence

from .bm25_index import BM25Index
from .config import RetrievalConfig
from .fusion import ranks_of, rrf_fuse
from .models import FusedHit, RerankedHit
from .rerankers import HeuristicReranker

TEXT_PREFIX = "t::"
VISUAL_PREFIX = "v::"


class VisuRAGRetriever:
    """Orchestrates hybrid search over a :class:`VisuRAGStore`.

    The retriever owns the BM25 side-index (built from indexed chunks via
    :meth:`index_chunks`) and delegates dense search to the store.
    """

    def __init__(
        self,
        store,  # VisuRAGStore (duck-typed to avoid a hard import cycle)
        text_encoder,
        *,
        config: RetrievalConfig | None = None,
        reranker=None,
        visual_text_encoder=None,
        image_encoder=None,
    ) -> None:
        self.store = store
        self.text_encoder = text_encoder
        self.config = config or RetrievalConfig()
        self.reranker = reranker if reranker is not None else HeuristicReranker()
        self.visual_text_encoder = visual_text_encoder
        # Image-query encoder (query-by-example). Defaults to the text
        # encoder when it also embeds images (e.g. HashEmbeddingProvider).
        if image_encoder is None and hasattr(text_encoder, "embed_images"):
            image_encoder = text_encoder
        self.image_encoder = image_encoder
        self._bm25 = BM25Index()
        self._chunks: dict[str, object] = {}

    # -- indexing ----------------------------------------------------
    def index_chunks(self, chunks: Sequence) -> int:
        """Load text chunks into the BM25 side-index + lookup table."""
        self._chunks = {c.chunk_id: c for c in chunks}
        self._bm25.build(
            keys=[f"{TEXT_PREFIX}{c.chunk_id}" for c in chunks],
            texts=[c.text for c in chunks],
        )
        return len(self._chunks)

    # -- queries -----------------------------------------------------
    def search(
        self,
        query: str,
        *,
        document_id: str | None = None,
        limit: int | None = None,
    ) -> list[RerankedHit]:
        """Hybrid text-query search → reranked context blocks."""
        cfg = self.config
        limit = limit or cfg.final_top_k
        if not query.strip():
            return []

        rankings: list[list[str]] = []
        names: list[str] = []
        payloads: dict[str, dict] = {}
        texts: dict[str, str] = {}

        # Arm 1 — dense semantic search (Qdrant cosine, doc-scoped).
        query_vector = self.text_encoder.embed_texts([query])[0]
        dense_hits = self.store.search_text(
            query_vector, limit=cfg.top_k_dense, document_id=document_id
        )
        dense_keys = []
        for h in dense_hits:
            key = f"{TEXT_PREFIX}{h.payload['chunk_id']}"
            dense_keys.append(key)
            payloads[key] = dict(h.payload)
            texts[key] = h.payload.get("text")
        rankings.append(dense_keys)
        names.append("dense")

        # Arm 2 — sparse exact-keyword search (BM25, doc-scoped).
        sparse_keys = []
        for key, _score in self._bm25.search(query, top_k=cfg.top_k_sparse):
            chunk = self._chunks.get(key[len(TEXT_PREFIX):])
            if chunk is None:
                continue
            if document_id is not None and chunk.document_id != document_id:
                continue
            sparse_keys.append(key)
            if key not in payloads:
                payloads[key] = chunk.payload()
                texts[key] = chunk.text
        rankings.append(sparse_keys)
        names.append("sparse")

        # Arm 3 (opt-in) — cross-modal dense search over visual patches.
        if self.visual_text_encoder is not None:
            visual_vector = self.visual_text_encoder.embed_texts([query])[0]
            visual_keys = []
            for h in self.store.search_visual(
                visual_vector, limit=cfg.top_k_visual, document_id=document_id
            ):
                key = f"{VISUAL_PREFIX}{h.payload['patch_id']}"
                visual_keys.append(key)
                payloads[key] = dict(h.payload)
            rankings.append(visual_keys)
            names.append("visual")

        # Fuse ranks (RRF) → candidate pool.
        fused = rrf_fuse(rankings, k=cfg.rrf_k)
        if not fused:
            return []
        rank_map = ranks_of(rankings, names)
        pool = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        pool = pool[: cfg.candidate_pool_size]
        fused_hits = [
            FusedHit(
                key=key,
                modality="image" if key.startswith(VISUAL_PREFIX) else "text",
                fused_score=score,
                ranks=rank_map.get(key, {}),
                payload=payloads.get(key, {}),
                text=texts.get(key),
            )
            for key, score in pool
        ]

        # Rerank pool → final top-k (reranker sees text hits; visual hits
        # keep fused order after text hits when they lack text).
        text_idx = [i for i, h in enumerate(fused_hits) if h.text]
        rerank_scores: dict[int, float] = {}
        if cfg.use_reranker and self.reranker is not None and text_idx:
            cands = [{"text": fused_hits[i].text} for i in text_idx]
            for pos, (cand_pos, score) in enumerate(
                self.reranker.rerank(query, cands)
            ):
                rerank_scores[text_idx[cand_pos]] = score
            ordered = sorted(text_idx, key=lambda i: rerank_scores[i], reverse=True)
            visual_idx = [i for i, h in enumerate(fused_hits) if not h.text]
            final_order = ordered + visual_idx
        else:
            final_order = list(range(len(fused_hits)))

        final = [fused_hits[i] for i in final_order[:limit]]

        # Attach sibling-patch visual evidence for the pages hit.
        evidence = self._visual_evidence_for(final, document_id)
        return [
            RerankedHit(
                key=h.key,
                modality=h.modality,
                fused_score=h.fused_score,
                rerank_score=rerank_scores.get(
                    fused_hits.index(h), h.fused_score
                ),
                payload=h.payload,
                text=h.text,
                visual_evidence=tuple(
                    evidence.get((h.payload.get("document_id"), h.payload.get("page_num")), ())
                ),
            )
            for h in final
        ]

    def search_by_image(
        self, image_path, *, document_id: str | None = None, limit: int = 5
    ):
        """Visual query-by-example over the patch collection."""
        if self.image_encoder is None or not hasattr(self.image_encoder, "embed_images"):
            raise AttributeError(
                "no image encoder configured; pass image_encoder= to "
                "VisuRAGRetriever (e.g. FastEmbedImageProvider or "
                "HashEmbeddingProvider)."
            )
        from pathlib import Path

        query_vector = self.image_encoder.embed_images([Path(image_path)])[0]
        return self.store.search_visual(
            query_vector, limit=limit, document_id=document_id
        )

    # -- helpers -----------------------------------------------------
    def _visual_evidence_for(
        self, hits: list[FusedHit], document_id: str | None
    ) -> dict[tuple, list[dict]]:
        """Sibling patches per (document_id, page_num) hit by the query."""
        pages_by_doc: dict[str, set[int]] = {}
        for h in hits:
            doc = h.payload.get("document_id") or document_id
            page = h.payload.get("page_num")
            if doc is not None and page is not None:
                pages_by_doc.setdefault(doc, set()).add(page)
        evidence: dict[tuple, list[dict]] = {}
        for doc, pages in pages_by_doc.items():
            try:
                records = self.store.get_visual_for_pages(doc, sorted(pages))
            except AttributeError:
                continue  # store without scroll support — skip evidence
            for r in records:
                payload = dict(r.payload)
                evidence.setdefault((doc, payload.get("page_num")), []).append(payload)
        return evidence
