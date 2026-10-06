"""High-level Qdrant store: batched upserts + filtered search per modality."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue, PointStruct

from .client import create_qdrant_client
from .config import VectorDBConfig
from .models import TextChunk, VisualPatch
from .schema import document_filter, ensure_collections


def _batched(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


class VisuRAGStore:
    """Owns all Qdrant interaction for VisuRAG (write + read primitives).

    Retrieval-level fusion (hybrid, rerank) lives in ``retrieval/``;
    this class only stores points with rich payloads and runs filtered
    nearest-neighbor search per collection.
    """

    def __init__(
        self,
        config: VectorDBConfig | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        self.config = config or VectorDBConfig()
        self.client = client or create_qdrant_client(self.config)
        ensure_collections(self.client, self.config)

    # -- writes ------------------------------------------------------
    def upsert_text_chunks(
        self, chunks: Sequence[TextChunk], vectors: np.ndarray
    ) -> int:
        """Batch-insert text chunks + embeddings. Returns points written."""
        if len(chunks) == 0:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if vectors.shape[1] != self.config.text_dim:
            raise ValueError(
                f"text vector dim {vectors.shape[1]} != collection dim "
                f"{self.config.text_dim}"
            )
        written = 0
        idx = 0
        for batch in _batched(list(chunks), self.config.batch_size):
            vecs = vectors[idx : idx + len(batch)]
            points = [
                PointStruct(
                    id=c.point_id(), vector=v.tolist(), payload=c.payload()
                )
                for c, v in zip(batch, vecs)
            ]
            self.client.upsert(
                collection_name=self.config.text_collection, points=points
            )
            written += len(points)
            idx += len(batch)
        return written

    def upsert_visual_patches(
        self, patches: Sequence[VisualPatch], vectors: np.ndarray
    ) -> int:
        """Batch-insert visual patches + embeddings. Returns points written."""
        if len(patches) == 0:
            return 0
        if len(patches) != len(vectors):
            raise ValueError("patches and vectors length mismatch")
        if vectors.shape[1] != self.config.visual_dim:
            raise ValueError(
                f"visual vector dim {vectors.shape[1]} != collection dim "
                f"{self.config.visual_dim}"
            )
        written = 0
        idx = 0
        for batch in _batched(list(patches), self.config.batch_size):
            vecs = vectors[idx : idx + len(batch)]
            points = [
                PointStruct(
                    id=p.point_id(), vector=v.tolist(), payload=p.payload()
                )
                for p, v in zip(batch, vecs)
            ]
            self.client.upsert(
                collection_name=self.config.visual_collection, points=points
            )
            written += len(points)
            idx += len(batch)
        return written

    # -- reads -------------------------------------------------------
    def search_text(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        document_id: str | None = None,
    ):
        """Nearest-neighbor search over text chunks, optionally doc-scoped."""
        return self.client.query_points(
            collection_name=self.config.text_collection,
            query=list(query_vector),
            limit=limit,
            query_filter=document_filter(document_id) if document_id else None,
        ).points

    def search_visual(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        document_id: str | None = None,
    ):
        """Nearest-neighbor search over visual patches, optionally doc-scoped."""
        return self.client.query_points(
            collection_name=self.config.visual_collection,
            query=list(query_vector),
            limit=limit,
            query_filter=document_filter(document_id) if document_id else None,
        ).points

    # -- visual evidence (sibling patches for citation) ----------------
    def get_visual_for_pages(
        self,
        document_id: str,
        page_nums: Sequence[int],
        *,
        limit: int = 100,
    ) -> list:
        """Scroll visual patches on the given pages of one document.

        Used by retrieval to attach schematic/figure evidence to text hits.
        Single scroll page (``limit``) is enough for per-document page sets.
        """
        from qdrant_client.http.models import MatchAny

        pages = sorted(set(page_nums))
        must = [
            FieldCondition(key="document_id", match=MatchValue(value=document_id)),
        ]
        if pages:
            must.append(
                FieldCondition(key="page_num", match=MatchAny(any=pages))
            )
        records, _ = self.client.scroll(
            collection_name=self.config.visual_collection,
            scroll_filter=Filter(must=must),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return list(records)

    # -- maintenance -------------------------------------------------
    def count(self) -> dict[str, int]:
        return {
            "text": self.client.count(
                collection_name=self.config.text_collection, exact=True
            ).count,
            "visual": self.client.count(
                collection_name=self.config.visual_collection, exact=True
            ).count,
        }

    def delete_document(self, document_id: str) -> None:
        """Remove every point (both modalities) belonging to *document_id*."""
        for collection in (self.config.text_collection, self.config.visual_collection):
            self.client.delete(
                collection_name=collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id", match=MatchValue(value=document_id)
                        )
                    ]
                ),
            )
