"""CLI: python -m ingestion <pdf> [--dpi 300] [--patch 512] [--overlap 64] [--no-cache]."""

from __future__ import annotations

import argparse
from pathlib import Path

from .cache import IngestionCache
from .config import IngestionConfig
from .pipeline import ingest_pdf


def main() -> None:
    ap = argparse.ArgumentParser(description="VisuRAG ingestion: PDF → pages → patches")
    ap.add_argument("pdf", type=Path, help="Input PDF datasheet")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--patch", type=int, default=512, help="Square patch size in px")
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    config = IngestionConfig(
        dpi=args.dpi,
        patch_size=(args.patch, args.patch),
        overlap=args.overlap,
        cache_dir=args.cache_dir,
        max_pages=args.max_pages,
    )
    result = ingest_pdf(
        args.pdf, config=config, cache=IngestionCache(config.cache_dir),
        use_cache=not args.no_cache,
    )
    print(f"document_id: {result.document_id}")
    print(f"source:      {result.source_path}")
    print(f"dpi:         {result.dpi}")
    print(f"pages:       {result.num_pages} (cache_hit={result.cache_hit})")
    print(f"patches:     {result.num_patches}")
    for page in result.pages:
        n = len(result.patches_for_page(page.page_num))
        print(f"  page {page.page_num:03d}: {page.width}x{page.height} px, "
              f"{len(page.text_blocks)} text blocks, {n} patches → {page.image_path}")


if __name__ == "__main__":
    main()
