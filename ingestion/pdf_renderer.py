"""Render PDF pages to high-resolution images with PyMuPDF.

Primary renderer is PyMuPDF (``pymupdf``) — no system-level poppler
dependency, and it preserves vector text / schematic lines when
rasterizing at 300 DPI. ``pdf2image`` is kept as an optional fallback
documented here but not required for the default path.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf  # PyMuPDF (import name is `pymupdf`; legacy alias `fitz`)
from PIL import Image

from .cache import IngestionCache, compute_document_id, sha256_file
from .config import IngestionConfig
from .models import RenderedPage, TextBlock


def _pixmap_to_pil(pix: pymupdf.Pixmap) -> Image.Image:
    # Schematics are typically RGB; normalize CMYK/gray to RGB.
    if pix.n - pix.alpha > 3:  # CMYK
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    mode = "RGBA" if pix.alpha else "RGB"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    return img.convert("RGB") if mode == "RGBA" else img


def extract_text_blocks(doc: pymupdf.Document, page_num: int) -> tuple[TextBlock, ...]:
    """Extract text blocks with bboxes (PDF points) for layout grounding."""
    page = doc[page_num]
    raw = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    blocks: list[TextBlock] = []
    for x0, y0, x1, y1, text, block_no, block_type in raw:
        if block_type != 0:  # 0 = text, 1 = image
            continue
        text = (text or "").strip()
        if not text:
            continue
        blocks.append(
            TextBlock(text=text, bbox=(x0, y0, x1, y1), block_no=int(block_no))
        )
    return tuple(blocks)


def render_pdf_to_images(
    pdf_path: Path | str,
    config: IngestionConfig,
    cache: IngestionCache | None = None,
) -> tuple[str, tuple[RenderedPage, ...]]:
    """Rasterize every (or first-N) page of *pdf_path* at ``config.dpi``.

    Returns ``(document_id, pages)``. Rendered PNGs are written through
    *cache* when provided so repeat runs are instant.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document_id = compute_document_id(pdf_path)
    _ = sha256_file(pdf_path)  # computed for manifest use by the pipeline
    zoom = config.dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)

    pages: list[RenderedPage] = []
    with pymupdf.open(pdf_path) as doc:
        total = len(doc)
        if config.max_pages is not None:
            total = min(total, config.max_pages)
        for page_num in range(total):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=matrix)
            img = _pixmap_to_pil(pix)

            if cache is not None:
                out_path = cache.page_path(document_id, page_num)
                # Reuse cached file only if dimensions already match.
                if out_path.exists():
                    try:
                        with Image.open(out_path) as cached:
                            cached.load()
                            if cached.size == (img.width, img.height):
                                img = cached.copy()
                            else:
                                cache.save_image(img, out_path)
                    except OSError:
                        cache.save_image(img, out_path)
                else:
                    cache.save_image(img, out_path)
            else:
                out_path = Path(f"p{page_num:03d}.png")

            text_blocks: tuple[TextBlock, ...] = ()
            if config.extract_text_blocks:
                text_blocks = extract_text_blocks(doc, page_num)

            pages.append(
                RenderedPage(
                    document_id=document_id,
                    page_num=page_num,
                    width=img.width,
                    height=img.height,
                    dpi=config.dpi,
                    image_path=out_path,
                    text_blocks=text_blocks,
                )
            )
    return document_id, tuple(pages)


def render_with_pdf2image_fallback(
    pdf_path: Path | str, dpi: int = 300, first_page: int | None = None
) -> list[Image.Image]:
    """Optional fallback using ``pdf2image`` (requires system poppler).

    Not used by the default pipeline; provided for environments where
    PyMuPDF rendering needs cross-checking.
    """
    from pdf2image import convert_from_path

    kwargs: dict = {"dpi": dpi}
    if first_page is not None:
        kwargs.update({"first_page": first_page, "last_page": first_page})
    return convert_from_path(str(pdf_path), **kwargs)
