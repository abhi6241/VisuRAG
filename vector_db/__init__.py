"""VisuRAG vector_db: Qdrant client and collection configuration.

Public API::

    from vector_db import (
        VectorDBConfig, VisuRAGStore,
        HashEmbeddingProvider, FastEmbedTextProvider, FastEmbedImageProvider,
        index_ingestion_result,
    )

    store = VisuRAGStore(VectorDBConfig(mode="path"))  # data/qdrant, no Docker needed
    summary = index_ingestion_result(result, store,
                                     text_encoder=HashEmbeddingProvider(dim=384),
                                     image_encoder=HashEmbeddingProvider(dim=512))
"""

from .client import create_qdrant_client
from .config import VectorDBConfig
from .embeddings import (
    EmbeddingProvider,
    FastEmbedImageProvider,
    FastEmbedTextProvider,
    HashEmbeddingProvider,
    SentenceTransformerProvider,
)
from .indexing import (
    IndexSummary,
    build_text_chunks,
    build_visual_patches,
    chunk_text,
    index_ingestion_result,
)
from .models import TextChunk, VisualPatch, deterministic_point_id
from .schema import document_filter, ensure_collections
from .store import VisuRAGStore

__all__ = [
    "EmbeddingProvider",
    "FastEmbedImageProvider",
    "FastEmbedTextProvider",
    "HashEmbeddingProvider",
    "IndexSummary",
    "SentenceTransformerProvider",
    "TextChunk",
    "VectorDBConfig",
    "VisuRAGStore",
    "VisualPatch",
    "build_text_chunks",
    "build_visual_patches",
    "chunk_text",
    "create_qdrant_client",
    "deterministic_point_id",
    "document_filter",
    "ensure_collections",
    "index_ingestion_result",
]
