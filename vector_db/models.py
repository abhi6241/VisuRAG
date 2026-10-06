"""Point models + payload builders shared by the store and indexing utils."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path


def deterministic_point_id(*parts: str) -> str:
    """Stable UUID for a point so re-indexing overwrites instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "|".join(parts)))


@dataclass(frozen=True)
class TextChunk:
    """One retrievable text unit with layout grounding."""

    chunk_id: str  # e.g. "<document_id>:p003:b07" or char-window suffix
    document_id: str
    source: str  # PDF filename, e.g. "opamp_datasheet.pdf"
    page_num: int  # 0-based
    text: str
    block_no: int | None = None
    bbox_pdf: tuple[float, float, float, float] | None = None  # PDF points

    def payload(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source": self.source,
            "page_num": self.page_num,
            "modality": "text",
            "text": self.text,
            "block_no": self.block_no,
            "bbox_pdf": list(self.bbox_pdf) if self.bbox_pdf else None,
        }

    def point_id(self) -> str:
        return deterministic_point_id(self.document_id, self.chunk_id)


@dataclass(frozen=True)
class VisualPatch:
    """One retrievable image patch with visual-citation metadata."""

    patch_id: str  # e.g. "p003_r02_c01"
    document_id: str
    source: str  # PDF filename
    page_num: int  # 0-based
    bbox_px: tuple[int, int, int, int]  # (x0, y0, x1, y1) in rendered pixels
    width: int
    height: int
    image_path: Path

    def payload(self) -> dict:
        return {
            "patch_id": self.patch_id,
            "document_id": self.document_id,
            "source": self.source,
            "page_num": self.page_num,
            "modality": "image",
            "bbox_px": list(self.bbox_px),
            "width": self.width,
            "height": self.height,
            "image_path": str(self.image_path),
        }

    def point_id(self) -> str:
        return deterministic_point_id(self.document_id, self.patch_id)
