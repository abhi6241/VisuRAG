"""CLI demo: python -m retrieval --pdf <pdf> --query "pin table VCC" [--dpi 150].

Self-contained: ingest (cached) → index into an in-memory Qdrant store with
offline hash embeddings → hybrid search with heuristic rerank → print hits
with fused/rerank scores and visual-evidence counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ingestion import IngestionCache, IngestionConfig, ingest_pdf
from vector_db import HashEmbeddingProvider, VectorDBConfig, VisuRAGStore, index_ingestion_result

from .engine import VisuRAGRetriever
from .rerankers import HeuristicReranker


def main() -> None:
    ap = argparse.ArgumentParser(description="VisuRAG hybrid search demo")
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    cfg = IngestionConfig(dpi=args.dpi, cache_dir=Path("data/cache"))
    result = ingest_pdf(args.pdf, config=cfg, cache=IngestionCache(cfg.cache_dir))
    store = VisuRAGStore(VectorDBConfig(mode="memory", text_dim=384, visual_dim=512))
    text_enc = HashEmbeddingProvider(dim=384)
    index_ingestion_result(
        result, store,
        text_encoder=text_enc, image_encoder=HashEmbeddingProvider(dim=512),
    )

    from vector_db import build_text_chunks

    retriever = VisuRAGRetriever(store, text_enc, reranker=HeuristicReranker())
    retriever.index_chunks(build_text_chunks(result))
    hits = retriever.search(args.query, document_id=result.document_id, limit=args.top)

    print(f"query: {args.query!r} (doc={result.document_id})")
    for i, h in enumerate(hits):
        snippet = (h.text or "")[:120].replace("\n", " ")
        print(f"[{i}] {h.key} modality={h.modality} "
              f"fused={h.fused_score:.4f} rerank={h.rerank_score:.4f} "
              f"page={h.payload.get('page_num')} evidence_patches={len(h.visual_evidence)}")
        print(f"     {snippet}")


if __name__ == "__main__":
    main()
