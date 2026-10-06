"""Vector DB configuration (Qdrant connection + dual-collection schema)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class VectorDBConfig:
    """Connection + schema settings for the VisuRAG Qdrant layer.

    Modes:
      - ``mode="memory"``: ephemeral in-process Qdrant (tests, CI).
      - ``mode="path"``: persistent local file storage at ``path``
        (default ``data/qdrant`` — no Docker required).
      - ``mode="url"``: connect to an external Qdrant server
        (e.g. Docker: ``http://localhost:6333``) via ``url`` + ``api_key``.
    """

    mode: str = "path"
    path: Path = field(default_factory=lambda: Path("data/qdrant"))
    url: str | None = None
    api_key: str | None = None

    # Dual-index collection names (one per modality, shared payload keys).
    text_collection: str = "visurag_text"
    visual_collection: str = "visurag_visual"

    # Vector dimensions. Defaults match common lightweight models:
    # text 384 (MiniLM/bge-small class), visual 512 (CLIP ViT-B/32 class).
    # Must match the encoder output used at index/search time.
    text_dim: int = 384
    visual_dim: int = 512

    # Cosine suits normalized embedding similarity for both modalities.
    distance: str = "Cosine"

    # Batch size for upserts (Qdrant handles large batches, but smaller
    # batches keep memory flat when indexing whole datasheet corpora).
    batch_size: int = 128

    def __post_init__(self) -> None:
        if self.mode not in ("memory", "path", "url"):
            raise ValueError(f"mode must be memory|path|url, got {self.mode!r}")
        if self.mode == "url" and not self.url:
            raise ValueError("mode='url' requires url (e.g. http://localhost:6333)")
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError("vector dims must be > 0")
        if not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))
