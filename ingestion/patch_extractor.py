"""Split rendered pages into overlapping visual patches.

Sliding-window tiling with overlap keeps table borders, pin labels, and
schematic symbols from being split at tile edges without context.
Edge tiles are clamped to the page bounds (never padded), so every
patch corresponds to real pixels and its bbox is directly citable.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .cache import IngestionCache
from .models import ImagePatch


def compute_tile_origins(length: int, tile: int, overlap: int) -> list[int]:
    """1-D tile origins covering ``[0, length)`` with ``tile``/``overlap``."""
    if tile >= length:
        return [0]
    step = tile - overlap
    origins = list(range(0, length - tile + 1, step))
    if origins[-1] != length - tile:
        origins.append(length - tile)
    return origins


def extract_patches(
    page_image: Image.Image,
    *,
    document_id: str,
    page_num: int,
    patch_size: tuple[int, int] = (512, 512),
    overlap: int = 64,
    cache: IngestionCache | None = None,
) -> tuple[ImagePatch, ...]:
    patch_w, patch_h = patch_size
    img_w, img_h = page_image.size
    xs = compute_tile_origins(img_w, patch_w, overlap)
    ys = compute_tile_origins(img_h, patch_h, overlap)

    patches: list[ImagePatch] = []
    for r, y0 in enumerate(ys):
        for c, x0 in enumerate(xs):
            x1 = min(x0 + patch_w, img_w)
            y1 = min(y0 + patch_h, img_h)
            # Clamp small-page case: if the page is smaller than the tile,
            # the crop is the full page (single patch).
            crop = page_image.crop((x0, y0, x1, y1))
            patch_id = f"p{page_num:03d}_r{r:02d}_c{c:02d}"
            if cache is not None:
                out_path = cache.patch_path(document_id, patch_id)
                if not out_path.exists():
                    cache.save_image(crop, out_path)
            else:
                out_path = Path(f"{patch_id}.png")
            patches.append(
                ImagePatch(
                    document_id=document_id,
                    page_num=page_num,
                    patch_id=patch_id,
                    bbox=(x0, y0, x1, y1),
                    width=x1 - x0,
                    height=y1 - y0,
                    image_path=out_path,
                )
            )
    return tuple(patches)
