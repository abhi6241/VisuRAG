"""Local disk cache for rendered pages, patches, and manifests.

Layout (under ``config.cache_dir``)::

    data/cache/
        pages/<document_id>/p000.png ...      # high-res page images
        patches/<document_id>/p000_r00_c00.png ...
        manifests/<document_id>.json          # render + patch metadata

Cache key = SHA256(file bytes)[:16]. Re-render is skipped when the
manifest exists *and* matches (sha256, dpi, patch_size, overlap).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_document_id(pdf_path: Path) -> str:
    """Stable 16-hex-char ID derived from file contents."""
    return sha256_file(pdf_path)[:16]


class IngestionCache:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.pages_dir = self.root / "pages"
        self.patches_dir = self.root / "patches"
        self.manifests_dir = self.root / "manifests"
        for d in (self.pages_dir, self.patches_dir, self.manifests_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- paths ---------------------------------------------------------
    def page_path(self, document_id: str, page_num: int, ext: str = "png") -> Path:
        return self.pages_dir / document_id / f"p{page_num:03d}.{ext}"

    def patch_path(self, document_id: str, patch_id: str, ext: str = "png") -> Path:
        return self.patches_dir / document_id / f"{patch_id}.{ext}"

    def manifest_path(self, document_id: str) -> Path:
        return self.manifests_dir / f"{document_id}.json"

    # -- images --------------------------------------------------------
    def save_image(self, image: Image.Image, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return path

    def load_image(self, path: Path) -> Image.Image:
        return Image.open(path)

    def has_page(self, document_id: str, page_num: int) -> bool:
        return self.page_path(document_id, page_num).exists()

    # -- manifests -----------------------------------------------------
    def save_manifest(self, document_id: str, payload: dict) -> Path:
        path = self.manifest_path(document_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        return path

    def load_manifest(self, document_id: str) -> dict | None:
        path = self.manifest_path(document_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def manifest_matches(
        self,
        manifest: dict,
        *,
        source_sha256: str,
        dpi: int,
        patch_size: tuple[int, int],
        overlap: int,
    ) -> bool:
        return (
            manifest.get("source_sha256") == source_sha256
            and manifest.get("dpi") == dpi
            and tuple(manifest.get("patch_size", ())) == tuple(patch_size)
            and manifest.get("overlap") == overlap
        )
