"""Shared data models for the ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TextBlock:
    """One text block with its PDF bbox (in PDF points, top-left origin)."""

    text: str
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    block_no: int


@dataclass(frozen=True)
class RenderedPage:
    """A single PDF page rasterized to a high-res image."""

    document_id: str
    page_num: int  # 0-based
    width: int  # pixels
    height: int  # pixels
    dpi: int
    image_path: Path
    text_blocks: tuple[TextBlock, ...] = ()


@dataclass(frozen=True)
class ImagePatch:
    """A fixed-size crop of a rendered page for visual embedding."""

    document_id: str
    page_num: int
    patch_id: str  # e.g. "p000_r02_c03"
    bbox: tuple[int, int, int, int]  # pixel coords (x0, y0, x1, y1)
    width: int
    height: int
    image_path: Path


@dataclass(frozen=True)
class IngestionResult:
    """Outcome of ingesting one PDF document."""

    document_id: str
    source_path: Path
    dpi: int
    pages: tuple[RenderedPage, ...] = ()
    patches: tuple[ImagePatch, ...] = ()
    cache_hit: bool = False

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    @property
    def num_patches(self) -> int:
        return len(self.patches)

    def patches_for_page(self, page_num: int) -> tuple[ImagePatch, ...]:
        return tuple(p for p in self.patches if p.page_num == page_num)


@dataclass(frozen=True)
class CachedManifest:
    """JSON-serializable manifest stored next to cached page images."""

    document_id: str
    source_name: str
    source_sha256: str
    dpi: int
    patch_size: tuple[int, int]
    overlap: int
    pages: tuple[dict, ...] = field(default_factory=tuple)
    patches: tuple[dict, ...] = field(default_factory=tuple)
