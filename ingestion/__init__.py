"""VisuRAG ingestion: PDF parsing & image patch extraction.

Public API::

    from ingestion import IngestionConfig, IngestionCache, ingest_pdf
    result = ingest_pdf("datasheet.pdf")
"""

from .cache import IngestionCache, compute_document_id, sha256_file
from .config import IngestionConfig
from .models import ImagePatch, IngestionResult, RenderedPage, TextBlock
from .patch_extractor import compute_tile_origins, extract_patches
from .pdf_renderer import render_pdf_to_images, render_with_pdf2image_fallback
from .pipeline import ingest_pdf

__all__ = [
    "IngestionCache",
    "IngestionConfig",
    "IngestionResult",
    "ImagePatch",
    "RenderedPage",
    "TextBlock",
    "compute_document_id",
    "compute_tile_origins",
    "extract_patches",
    "ingest_pdf",
    "render_pdf_to_images",
    "render_with_pdf2image_fallback",
    "sha256_file",
]
