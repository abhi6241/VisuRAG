"""VisuRAG retrieval: hybrid search (dense + BM25 → RRF → rerank).

Public API::

    from retrieval import VisuRAGRetriever, RetrievalConfig, HeuristicReranker

    retriever = VisuRAGRetriever(store, text_encoder)
    retriever.index_chunks(chunks)          # BM25 side-index over chunk texts
    hits = retriever.search("pin table VCC", document_id=doc_id)
    # hits[i]: RerankedHit(key, modality, fused_score, rerank_score,
    #                      payload, text, visual_evidence)
"""

from .bm25_index import BM25Index, tokenize
from .config import RetrievalConfig
from .engine import TEXT_PREFIX, VISUAL_PREFIX, VisuRAGRetriever
from .fusion import ranks_of, rrf_fuse
from .models import FusedHit, RerankedHit
from .rerankers import CrossEncoderReranker, HeuristicReranker, Reranker

__all__ = [
    "BM25Index",
    "CrossEncoderReranker",
    "FusedHit",
    "HeuristicReranker",
    "RerankedHit",
    "Reranker",
    "RetrievalConfig",
    "TEXT_PREFIX",
    "VISUAL_PREFIX",
    "VisuRAGRetriever",
    "ranks_of",
    "rrf_fuse",
    "tokenize",
]
