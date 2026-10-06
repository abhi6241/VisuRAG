"""Indexing utils: IngestionResult → text chunks + visual points → Qdrant.

Typical write path::

    result = ingest_pdf("opamp.pdf", config=IngestionConfig(dpi=300))
    store = VisuRAGStore(VectorDBConfig(path=Path("data/qdrant")))
    summary = index_ingestion_result(
        result, store,
        text_encoder=HashEmbeddingProvider(dim=384),
        image_encoder=HashEmbeddingProvider(dim=512),
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import EmbeddingProvider
from .models import TextChunk, VisualPatch
from .store import VisuRAGStore

try:
    from ingestion.models import IngestionResult
except ImportError:  # pragma: no cover - standalone use without ingestion pkg
    IngestionResult = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class IndexSummary:
    document_id: str
    source: str
    text_points: int
    visual_points: int


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 100) -> list[str]:
    """Split long text blocks into overlapping char windows.

    Short blocks (the common datasheet case) pass through untouched as a
    single chunk; only oversized blocks are windowed so table rows and
    pin lists are never needlessly fragmented.
    """
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text] if text else []
    if overlap >= max_chars:
        raise ValueError("overlap must be < max_chars")
    chunks, start, step = [], 0, max_chars - overlap
    while start < len(text):
        chunks.append(text[start : start + max_chars])
        if start + max_chars >= len(text):
            break
        start += step
    return chunks


def build_text_chunks(
    result: IngestionResult, *, max_chars: int = 1000, overlap: int = 100
) -> list[TextChunk]:
    """One chunk per text block (windowed if oversized); one fallback chunk
    per page when a page has no extractable text (scanned figure pages)."""
    source = Path(result.source_path).name
    chunks: list[TextChunk] = []
    for page in result.pages:
        if not page.text_blocks:
            chunk_id = f"{result.document_id}:p{page.page_num:03d}:full"
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    document_id=result.document_id,
                    source=source,
                    page_num=page.page_num,
                    text=f"[page {page.page_num} figure/schematic page, no extractable text]",
                    block_no=None,
                    bbox_pdf=None,
                )
            )
            continue
        for block in page.text_blocks:
            for i, window in enumerate(chunk_text(block.text, max_chars, overlap)):
                chunk_id = f"{result.document_id}:p{page.page_num:03d}:b{block.block_no:02d}"
                if i:
                    chunk_id += f":w{i}"
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        document_id=result.document_id,
                        source=source,
                        page_num=page.page_num,
                        text=window,
                        block_no=block.block_no,
                        bbox_pdf=block.bbox,
                    )
                )
    return chunks


def build_visual_patches(result: IngestionResult) -> list[VisualPatch]:
    source = Path(result.source_path).name
    return [
        VisualPatch(
            patch_id=p.patch_id,
            document_id=p.document_id,
            source=source,
            page_num=p.page_num,
            bbox_px=p.bbox,
            width=p.width,
            height=p.height,
            image_path=p.image_path,
        )
        for p in result.patches
    ]


def index_ingestion_result(
    result: IngestionResult,
    store: VisuRAGStore,
    *,
    text_encoder: EmbeddingProvider,
    image_encoder: EmbeddingProvider,
    max_chars: int = 1000,
    overlap: int = 100,
    batch_size: int | None = None,
) -> IndexSummary:
    """Encode + batch-upsert one ingested document. Idempotent: re-indexing
    the same document overwrites its points (deterministic UUIDs)."""
    if text_encoder.dim != store.config.text_dim:
        raise ValueError(
            f"text encoder dim {text_encoder.dim} != store text_dim "
            f"{store.config.text_dim}"
        )
    if image_encoder.dim != store.config.visual_dim:
        raise ValueError(
            f"image encoder dim {image_encoder.dim} != store visual_dim "
            f"{store.config.visual_dim}"
        )

    text_chunks = build_text_chunks(result, max_chars=max_chars, overlap=overlap)
    visual_patches = build_visual_patches(result)

    text_vectors = (
        text_encoder.embed_texts([c.text for c in text_chunks])
        if text_chunks
        else np.zeros((0, store.config.text_dim), dtype=np.float32)
    )
    image_vectors = (
        image_encoder.embed_images([p.image_path for p in visual_patches])
        if visual_patches
        else np.zeros((0, store.config.visual_dim), dtype=np.float32)
    )

    if batch_size is not None:
        # Honor caller batch size via a replaced (frozen) config copy.
        import dataclasses

        store.config = dataclasses.replace(store.config, batch_size=batch_size)

    n_text = store.upsert_text_chunks(text_chunks, text_vectors)
    n_visual = store.upsert_visual_patches(visual_patches, image_vectors)
    return IndexSummary(
        document_id=result.document_id,
        source=Path(result.source_path).name,
        text_points=n_text,
        visual_points=n_visual,
    )
