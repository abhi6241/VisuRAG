"""VisuRAG ingestion configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class IngestionConfig:
    """Tunable parameters for PDF → page images → patches."""

    # Render resolution. 300 DPI preserves small schematic text / pin labels.
    dpi: int = 300
    # Sliding-window patch size in pixels (w, h) at rendered resolution.
    patch_size: tuple[int, int] = (512, 512)
    # Overlap between adjacent patches in pixels. Prevents cutting
    # symbols / table borders at tile edges.
    overlap: int = 64
    # On-disk image format for cached pages/patches.
    image_format: str = "PNG"
    # Root cache dir. Pages → <cache_dir>/pages/, patches → <cache_dir>/patches/.
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))
    # If set, only process the first N pages (useful for smoke tests).
    max_pages: int | None = None
    # Whether to also extract text blocks with bboxes for layout grounding.
    extract_text_blocks: bool = True

    def __post_init__(self) -> None:
        if self.dpi <= 0:
            raise ValueError("dpi must be > 0")
        pw, ph = self.patch_size
        if pw <= 0 or ph <= 0:
            raise ValueError("patch_size dimensions must be > 0")
        if not 0 <= self.overlap < min(pw, ph):
            raise ValueError("overlap must satisfy 0 <= overlap < min(patch_size)")
        # Normalize cache_dir to Path (frozen dataclass → object.__setattr__).
        if not isinstance(self.cache_dir, Path):
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))
