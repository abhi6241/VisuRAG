"""End-to-end ingestion: PDF → cached page images → cached patches."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .cache import IngestionCache, compute_document_id, sha256_file
from .config import IngestionConfig
from .models import ImagePatch, IngestionResult, RenderedPage, TextBlock
from .patch_extractor import extract_patches
from .pdf_renderer import render_pdf_to_images


def _manifest_from_result(
    result: IngestionResult, source_sha256: str, config: IngestionConfig
) -> dict:
    return {
        "document_id": result.document_id,
        "source_name": result.source_path.name,
        "source_sha256": source_sha256,
        "dpi": config.dpi,
        "patch_size": list(config.patch_size),
        "overlap": config.overlap,
        "pages": [
            {
                "page_num": p.page_num,
                "width": p.width,
                "height": p.height,
                "dpi": p.dpi,
                "image_path": str(p.image_path),
                "text_blocks": [
                    {"text": b.text, "bbox": list(b.bbox), "block_no": b.block_no}
                    for b in p.text_blocks
                ],
            }
            for p in result.pages
        ],
        "patches": [
            {
                "patch_id": p.patch_id,
                "page_num": p.page_num,
                "bbox": list(p.bbox),
                "width": p.width,
                "height": p.height,
                "image_path": str(p.image_path),
            }
            for p in result.patches
        ],
    }


def _result_from_manifest(
    manifest: dict, source_path: Path, cache_hit: bool
) -> IngestionResult:
    pages = tuple(
        RenderedPage(
            document_id=manifest["document_id"],
            page_num=p["page_num"],
            width=p["width"],
            height=p["height"],
            dpi=p["dpi"],
            image_path=Path(p["image_path"]),
            text_blocks=tuple(
                TextBlock(
                    text=b["text"], bbox=tuple(b["bbox"]), block_no=b["block_no"]
                )
                for b in p.get("text_blocks", [])
            ),
        )
        for p in manifest.get("pages", [])
    )
    patches = tuple(
        ImagePatch(
            document_id=manifest["document_id"],
            page_num=p["page_num"],
            patch_id=p["patch_id"],
            bbox=tuple(p["bbox"]),
            width=p["width"],
            height=p["height"],
            image_path=Path(p["image_path"]),
        )
        for p in manifest.get("patches", [])
    )
    return IngestionResult(
        document_id=manifest["document_id"],
        source_path=source_path,
        dpi=manifest["dpi"],
        pages=pages,
        patches=patches,
        cache_hit=cache_hit,
    )


def ingest_pdf(
    pdf_path: Path | str,
    config: IngestionConfig | None = None,
    cache: IngestionCache | None = None,
    use_cache: bool = True,
) -> IngestionResult:
    """Ingest one PDF into page images + patches, using the disk cache.

    - First run renders at ``config.dpi`` via PyMuPDF, tiles patches with
      ``(patch_size, overlap)``, writes PNGs + a JSON manifest.
    - Repeat runs with identical (file hash, dpi, patch_size, overlap)
      skip rendering/tiling and reload from the manifest (``cache_hit=True``).
    """
    config = config or IngestionConfig()
    cache = cache or IngestionCache(config.cache_dir)
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document_id = compute_document_id(pdf_path)
    source_sha256 = sha256_file(pdf_path)

    if use_cache:
        manifest = cache.load_manifest(document_id)
        if manifest is not None and cache.manifest_matches(
            manifest,
            source_sha256=source_sha256,
            dpi=config.dpi,
            patch_size=config.patch_size,
            overlap=config.overlap,
        ):
            # Verify page files still exist before declaring a hit.
            if all(Path(p["image_path"]).exists() for p in manifest.get("pages", [])):
                return _result_from_manifest(manifest, pdf_path, cache_hit=True)

    # Cache miss — full render + tile.
    _, pages = render_pdf_to_images(pdf_path, config, cache=cache)
    all_patches: list[ImagePatch] = []
    for page in pages:
        with Image.open(page.image_path) as img:
            img.load()
            patches = extract_patches(
                img,
                document_id=document_id,
                page_num=page.page_num,
                patch_size=config.patch_size,
                overlap=config.overlap,
                cache=cache,
            )
            all_patches.extend(patches)

    result = IngestionResult(
        document_id=document_id,
        source_path=pdf_path,
        dpi=config.dpi,
        pages=pages,
        patches=tuple(all_patches),
        cache_hit=False,
    )
    cache.save_manifest(document_id, _manifest_from_result(result, source_sha256, config))
    return result
