# Changelog

All notable changes to VisuRAG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-10-06
### Added
- Hybrid retrieval engine in `retrieval/` (dense + sparse → RRF → rerank):
  - `bm25_index.py`: in-memory `BM25Okapi` side-index over chunk texts (dependency-free regex tokenizer) for exact keyword matches (part numbers, pin names); doc-scoped search.
  - `fusion.py`: `rrf_fuse()` (`score = Σ w/(k+rank)`, `k=60`) + `ranks_of()` for explainability; namespaced keys (`t::` / `v::`).
  - `rerankers.py`: `CrossEncoderReranker` (lazy `sentence_transformers.CrossEncoder`, default `ms-marco-MiniLM-L-6-v2`, BGE `bge-reranker-base` swap-in) + `HeuristicReranker` (token-overlap + phrase bonus, offline/tests).
  - `engine.py`: `VisuRAGRetriever` — 3-arm fan-out (dense Qdrant + BM25 + opt-in cross-modal visual) → RRF pool → rerank → top-k `RerankedHit` with sibling-patch `visual_evidence`; `search_by_image()` query-by-example; `config.py` (`RetrievalConfig`); `models.py` (`FusedHit`, `RerankedHit`); CLI via `python -m retrieval --pdf <pdf> --query "..."`.
  - `vector_db/store.py`: new `get_visual_for_pages()` filtered scroll backing visual-citation evidence.
- `requirements.txt` adds `rank-bm25==0.2.2`, `sentence-transformers==6.1.0`, `torch==2.14.1`, `transformers==5.18.0` (venv-only).
- Smoke-tested: RRF/BM25/heuristic units; E2E `pin table VCC GND` → `VCC` pin-table chunks top (fused 0.0325, rerank 2.5) with 12 evidence patches each; by-image self-match 1.0; live cross-encoder (+7.28 vs −11.33); CLI verified.

## [0.3.0] - 2026-10-06
### Added
- Vector storage & indexing layer in `vector_db/` (Qdrant local-first, no Docker required):
  - `config.py` (`VectorDBConfig`: `memory` / `path` (default `data/qdrant`) / `url` modes, dual-collection names, 384-d text / 512-d visual dims, Cosine, batch size).
  - `client.py` (`create_qdrant_client()` factory); `schema.py` (`ensure_collections()` for `visurag_text` + `visurag_visual` with keyword/integer payload indexes, `document_filter()`).
  - `embeddings.py`: `HashEmbeddingProvider` (deterministic SHA256-seeded, normalized, zero downloads — used for tests) plus lazy `FastEmbedTextProvider` (bge-small 384-d), `FastEmbedImageProvider` (CLIP ViT-B/32 512-d), optional `SentenceTransformerProvider`.
  - `models.py` (`TextChunk` / `VisualPatch` with `source` PDF name, `page_num`, `bbox_pdf` / `bbox_px`, `image_path` payloads; deterministic UUID5 point IDs for idempotent re-indexing).
  - `store.py` (`VisuRAGStore`: batched `upsert_text_chunks()` / `upsert_visual_patches()`, filtered `search_text()` / `search_visual()`, `count()`, `delete_document()`).
  - `indexing.py` (`index_ingestion_result()`: text-block chunking with 1000-char windowing + figure-page fallback → batch-encode → batch-upsert → `IndexSummary`).
- `requirements.txt` adds `qdrant-client==1.19.1`, `fastembed==0.8.1`, `numpy==2.5.3` (venv-only); `.gitignore` adds `data/qdrant/`.
- Smoke-tested end-to-end (memory mode): 2-page synthetic PDF → 6 text + 24 visual points; text/visual self-match score 1.0 with correct metadata payloads; re-index idempotent; `delete_document()` verified; path mode verified.

## [0.2.0] - 2026-10-06
### Added
- Multimodal ingestion pipeline in `ingestion/`:
  - `pdf_renderer.py`: PyMuPDF high-resolution page rasterization (`Matrix(dpi/72)`, default 300 DPI) with `Pixmap` → RGB PIL conversion; text-block extraction via `page.get_text("blocks")` for layout grounding; `render_with_pdf2image_fallback()` (poppler) for cross-checks.
  - `patch_extractor.py`: overlapping sliding-window tiler (default 512×512 px, 64 px overlap) with edge clamping (no padding); citable `p{page}_r{row}_c{col}` IDs and pixel bboxes.
  - `cache.py`: `IngestionCache` with `sha256(file)[:16]` document IDs; pages/patches/manifests under `data/cache/`; manifest match on `(sha256, dpi, patch_size, overlap)` for instant `cache_hit` reloads.
  - `pipeline.py`: `ingest_pdf()` orchestration (render → tile → manifest); `config.py` (`IngestionConfig`); `models.py` (`RenderedPage`, `ImagePatch`, `TextBlock`, `IngestionResult`); CLI via `python -m ingestion <pdf>`.
- `requirements.txt` pinned (`PyMuPDF==1.28.2`, `pdf2image==1.17.0`, `Pillow==12.3.0`, venv-only) and `.gitignore` (`venv/`, `data/cache/`, `__pycache__`, `.env`).
- Smoke-tested on 2-page synthetic vector PDF at 150 DPI: 1240×1755 px pages, 3 text blocks/page, 24 patches; second run `cache_hit=True`; CLI verified.

## [0.1.0] - 2026-10-05
### Added
- Initial project setup for VisuRAG (multimodal RAG for engineering datasheets and schematics).
- Created local Python virtual environment (`venv/`, Python 3.12.12) with no global pip installs.
- Created root documentation: `CONTEXT.md` (architecture, tech stack, file structure, state) and `CHANGELOG.md` (this file).
- Created clean project structure with package directories:
  - `ingestion/` (PDF parsing & image patch extraction)
  - `vector_db/` (Qdrant client and collection configuration)
  - `retrieval/` (search and reranking logic)
  - `api/` (FastAPI backend)
  - `frontend/` (Next.js client placeholder)
