# VisuRAG — CONTEXT.md

> Living architectural record for VisuRAG. Update this file as the project evolves.

## 1. Project Overview
- **Name:** VisuRAG
- **Description:** Multimodal RAG pipeline for engineering datasheets and schematics.
- **Goals:**
  - Ingest PDFs (text + tables + figures/schematics) and extract image patches.
  - Index text and visual embeddings in a vector DB for hybrid retrieval.
  - Provide search + reranking + grounded generation via API.
  - Provide Next.js frontend for upload, search, and visual citation.
- **Status:** Step 3 done (2026-10-06) — ingestion + Qdrant vector layer + hybrid retrieval (dense + BM25 → RRF → cross-encoder rerank) implemented and smoke-tested; api still skeleton.

## 2. Tech Stack Choices
| Layer | Choice | Rationale / Notes |
|-------|--------|-------------------|
| Python | 3.12 (via local `venv/`) | Pinned to system Python 3.12.12; all deps installed only in venv, never global pip. |
| Ingestion (PDF render) | `PyMuPDF==1.28.2` (primary) + `pdf2image==1.17.0` (optional fallback) | PyMuPDF rasterizes vector text/schematics at 300 DPI via `Matrix(dpi/72)`; no poppler needed. `pdf2image` kept only as `render_with_pdf2image_fallback()` for cross-checks (needs poppler). |
| Image / patches | `Pillow==12.3.0` | `Image.frombytes` conversion, sliding-window tiling, PNG cache I/O. |
| Vector DB | Qdrant (`qdrant-client==1.19.1`, local `path` mode default, no Docker needed) | Dual collections `visurag_text` (384-d) + `visurag_visual` (512-d), Cosine, keyword/integer payload indexes. Config + client + store all in `vector_db/`. |
| Embeddings | `fastembed==0.8.1` (ONNX text + CLIP-vision, lazy) + `HashEmbeddingProvider` (offline deterministic fallback) | `FastEmbedTextProvider` (bge-small 384-d), `FastEmbedImageProvider` (CLIP ViT-B/32 512-d), optional `SentenceTransformerProvider`. Hash provider needs no downloads — used for tests. |
| Retrieval (hybrid) | `rank-bm25==0.2.2` (sparse) + Qdrant dense + RRF (`k=60`) | Dense fan-out over `visurag_text` (+ opt-in CLIP-text arm over `visurag_visual`), BM25 side-index for exact keyword matches (part numbers, pin names), rank-based RRF fusion (no score normalization). Lives in `retrieval/`. |
| Reranker | `sentence-transformers==6.1.0` (`CrossEncoder`, default `ms-marco-MiniLM-L-6-v2`, BGE swap-in) + `HeuristicReranker` (offline fallback) | `torch==2.14.1`, `transformers==5.18.0`. Cross-encoder loads lazily on first rerank; heuristic uses token-overlap + phrase bonus for tests. |
| API | FastAPI + Uvicorn | Lives in `api/`. |
| Frontend | Next.js (TypeScript) | Lives in `frontend/`. Separate Node project, not Python. |
| Config / Env | `python-dotenv`, `pydantic-settings` (planned) | For Qdrant URL, API keys, model names. |
| Dev tooling | `pytest`, `ruff`, `black` (planned) | To be added to `requirements.txt`. |

## 3. File Structure
```
VisuRAG/
├── venv/                  # Local Python virtual environment (not committed)
├── CONTEXT.md             # This file — architecture + state
├── CHANGELOG.md           # Keep a Changelog history
├── requirements.txt       # Pinned Python deps (PyMuPDF, pdf2image, Pillow), venv-only
├── .gitignore             # venv/, __pycache__, .env, data/cache/, data/qdrant/, node_modules
├── README.md              # (planned) setup + usage
├── data/cache/            # (gitignored) ingestion disk cache
│   ├── pages/<doc_id>/p000.png ...
│   ├── patches/<doc_id>/p000_r00_c00.png ...
│   └── manifests/<doc_id>.json
├── ingestion/             # PDF parsing & image patch extraction (Step 1 DONE)
│   ├── __init__.py        # Public API re-exports
│   ├── __main__.py        # CLI: python -m ingestion <pdf>
│   ├── config.py          # IngestionConfig (dpi, patch_size, overlap, cache_dir)
│   ├── models.py          # RenderedPage, ImagePatch, TextBlock, IngestionResult
│   ├── cache.py           # IngestionCache + sha256 doc IDs + manifest match
│   ├── pdf_renderer.py    # PyMuPDF rasterizer + text-block extraction
│   ├── patch_extractor.py # Overlapping sliding-window tiler
│   └── pipeline.py        # ingest_pdf() orchestration with cache-hit path
├── data/qdrant/           # (gitignored) Qdrant persistent local storage (path mode)
├── vector_db/             # Qdrant client + dual-collection schema + store (Step 2 DONE)
│   ├── __init__.py        # Public API re-exports
│   ├── config.py          # VectorDBConfig (mode memory|path|url, dims, batch_size)
│   ├── client.py          # create_qdrant_client() factory
│   ├── schema.py          # ensure_collections() + document_filter()
│   ├── embeddings.py      # Hash / FastEmbed-text / SentenceTransformer / FastEmbed-image
│   ├── models.py          # TextChunk, VisualPatch, deterministic UUID point IDs
│   ├── store.py           # VisuRAGStore (batched upserts, filtered search, delete)
│   └── indexing.py        # index_ingestion_result() (IngestionResult → Qdrant)
├── retrieval/             # Hybrid search + reranking (Step 3 DONE)
│   ├── __init__.py        # Public API re-exports
│   ├── __main__.py        # CLI demo: python -m retrieval --pdf <pdf> --query "..."
│   ├── config.py          # RetrievalConfig (fan-out widths, rrf_k, pool, top-k)
│   ├── models.py          # FusedHit, RerankedHit (payload + visual_evidence)
│   ├── bm25_index.py      # BM25Okapi side-index over chunk texts (regex tokenizer)
│   ├── fusion.py          # rrf_fuse() (rank-based, no score normalization)
│   ├── rerankers.py       # HeuristicReranker + lazy CrossEncoderReranker
│   └── engine.py          # VisuRAGRetriever (fan-out → RRF → rerank → evidence)
├── api/                   # FastAPI backend
│   └── __init__.py
└── frontend/              # Next.js client (Node, not Python)
    └── README.md          # Placeholder until `create-next-app` is run
```

## 4. Architectural Decisions
1. **Local venv only:** All Python work uses `./venv/bin/python` and `./venv/bin/pip`. No global installs.
2. **Separation of concerns:** `ingestion/` is write-path (parse → chunk → embed → upsert); `retrieval/` is read-path (query → search → rerank); `vector_db/` owns all Qdrant interaction; `api/` is thin orchestration over those.
3. **Multimodal-first:** Store text chunks and image patches as separate points with shared `document_id` / `page_num` payload for joint retrieval and visual citation.
4. **Qdrant collections (finalized Step 2):** Two collections, one per modality (`visurag_text` 384-d, `visurag_visual` 512-d, Cosine) — avoids dim-mismatch hacks of a single named-vector collection and allows per-modality re-indexing. Shared payload keys (`document_id`, `source`, `page_num`, `modality`) enable joint filtered retrieval; fusion lives in `retrieval/`. Config centralized in `VectorDBConfig`, not scattered.
5. **Frontend decoupling:** `frontend/` is isolated Next.js app; communicates only via `api/` REST endpoints.

## 5. Ingestion Approach (Step 1)
- **Render:** `render_pdf_to_images()` opens the PDF with PyMuPDF, applies `Matrix(dpi/72)` (default 300 DPI), converts each `Pixmap` to RGB PIL via `Image.frombytes`, and extracts text blocks via `page.get_text("blocks")` (type 0 only) for layout grounding. Vector schematics stay crisp because rasterization happens after vector scaling.
- **Tile:** `extract_patches()` runs a sliding window (default 512×512 px, 64 px overlap) with `compute_tile_origins()` clamping edge tiles to page bounds — no padding, so every bbox is real pixels and citable. Patch IDs are `p{page:03d}_r{row:02d}_c{col:02d}`.
- **Cache:** `IngestionCache` keys by `sha256(file)[:16]` as `document_id`. Pages → `data/cache/pages/<doc_id>/p000.png`, patches → `data/cache/patches/<doc_id>/<patch_id>.png`, metadata → `data/cache/manifests/<doc_id>.json`. `ingest_pdf()` reloads from the manifest (`cache_hit=True`) when `(sha256, dpi, patch_size, overlap)` match and page files still exist; otherwise it re-renders.
- **Entry points:** `ingest_pdf(pdf, config, cache)` in `pipeline.py`; CLI via `python -m ingestion <pdf> [--dpi 300] [--patch 512] [--overlap 64]`.
- **Smoke test (2026-10-06):** 2-page synthetic vector PDF at 150 DPI → 1240×1755 px pages, 3 text blocks/page, 24 patches (12/page); run 2 returned `cache_hit=True`. CLI verified.

## 6. Vector DB Architecture (Step 2)
- **Client:** `create_qdrant_client()` supports three modes — `memory` (`:memory:`, tests/CI), `path` (default `data/qdrant`, persistent embedded Qdrant, **no Docker needed** — Docker unavailable in this env), `url` (external server, e.g. Docker `http://localhost:6333` + `api_key`). Verified memory + path modes.
- **Schema:** `ensure_collections()` creates `visurag_text` (384-d) + `visurag_visual` (512-d), Cosine distance, idempotent (existing collections untouched). Payload indexes: keyword (`document_id`, `source`, `modality`) + integer (`page_num`) for per-document / per-page / per-source filtered retrieval. (`Payload indexes are server-side only; local mode warns harmlessly.`)
- **Embeddings:** `HashEmbeddingProvider` (SHA256-seeded Gaussian, normalized — zero downloads, used for tests) vs lazy real providers: `FastEmbedTextProvider` (bge-small 384-d), `FastEmbedImageProvider` (CLIP ViT-B/32 512-d), optional `SentenceTransformerProvider`. Dim mismatch vs store config raises `ValueError` at index time.
- **Points:** `TextChunk` (one chunk per text block, char-windowed only if >1000 chars; fallback chunk for figure-only pages; payload: `chunk_id`, `source` PDF name, `page_num`, `text`, `block_no`, `bbox_pdf`) and `VisualPatch` (payload: `patch_id`, `source`, `page_num`, `bbox_px`, `width`/`height`, `image_path`). Point IDs are deterministic UUID5(`document_id`, `chunk_id`/`patch_id`) → re-indexing overwrites, never duplicates.
- **Store:** `VisuRAGStore` — `upsert_text_chunks()` / `upsert_visual_patches()` (batched, default 128), `search_text()` / `search_visual()` (top-k + optional `document_id` filter), `get_visual_for_pages()` (filtered scroll for sibling-patch evidence), `count()`, `delete_document()` (both collections).
- **Write path:** `index_ingestion_result(result, store, text_encoder, image_encoder)` — build chunks/patches → batch-encode → batch-upsert → `IndexSummary`. Lives in `vector_db/indexing.py`.
- **Smoke test (2026-10-06):** 2-page synthetic PDF (ingest cache-hit) → indexed 6 text + 24 visual points (memory mode); text self-match score 1.0 with correct `chunk_id`/`source`/`page_num`; visual self-match score 1.0 with correct `patch_id`/`bbox_px`/`image_path`; re-index idempotent; `delete_document()` → counts 0; path mode verified.

## 7. Retrieval Pipeline Design (Step 3)
- **Fan-out (3 arms):** (1) dense semantic search — query encoded by `text_encoder` → cosine over `visurag_text` (doc-scoped, `top_k_dense=20`); (2) sparse BM25 (`BM25Okapi`, regex tokenizer, in-memory side-index over chunk texts, doc-scoped) for exact keyword matches; (3) opt-in cross-modal arm over `visurag_visual` when a text→visual query encoder (e.g. CLIP text tower) is supplied. Candidate keys namespaced (`t::<chunk_id>` / `v::<patch_id>`).
- **Fusion:** `rrf_fuse()` — `score(d) = Σ w/(k+rank(d))`, `k=60`. Rank-based, so cosine and BM25 scores are never directly compared. Unit-verified (`b > a > c` on `[a,b],[b,c]`).
- **Rerank:** pool (`candidate_pool_size=20`) → `CrossEncoderReranker` (lazy `sentence_transformers.CrossEncoder`, default `ms-marco-MiniLM-L-6-v2`, BGE `bge-reranker-base` swap-in) or `HeuristicReranker` (token-overlap + phrase bonus, offline). Visual candidates without text keep fused order after text hits.
- **Visual evidence:** each final hit carries sibling patches from its `(document_id, page_num)` via `get_visual_for_pages()` scroll — what the frontend renders as visual citations. Plus `search_by_image()` (query-by-example over patches).
- **Entry points:** `VisuRAGRetriever(store, text_encoder, ...)` (`index_chunks()` → `search()` → `RerankedHit` list); CLI via `python -m retrieval --pdf <pdf> --query "..."`.
- **Smoke test (2026-10-06):** query `pin table VCC GND` over 6-chunk/24-patch index → top hits are the `VCC` pin-table chunks (fused 0.0325, rerank 2.5) with 12 evidence patches each on the correct page; by-image self-match score 1.0; live cross-encoder check (+7.28 relevant vs −11.33 irrelevant); CLI verified.

## 8. Current Implementation State
- [x] Local venv created (`python3 -m venv venv`, Python 3.12.12)
- [x] Root docs created (`CONTEXT.md`, `CHANGELOG.md`)
- [x] Package dirs created: `ingestion/`, `vector_db/`, `retrieval/`, `api/`, `frontend/`
- [x] `requirements.txt` pinned (PyMuPDF, pdf2image, Pillow, qdrant-client, fastembed, numpy, rank-bm25, sentence-transformers, torch, transformers), venv-only
- [x] `.gitignore` added (`venv/`, `data/cache/`, `data/qdrant/`, `__pycache__`, `.env`, Node artifacts)
- [x] Ingestion pipeline implemented + smoke-tested (render, text blocks, patches, cache, CLI)
- [x] Vector DB layer implemented + smoke-tested (Qdrant dual collections, batch upserts, filtered search, delete)
- [x] Hybrid retrieval implemented + smoke-tested (dense + BM25 → RRF → cross-encoder rerank + visual evidence, CLI)
- [ ] `README.md` — planned next
- [ ] No api logic implemented yet
- [ ] No Next.js app scaffolded yet (only `frontend/README.md` placeholder)

## 9. Next Steps
1. Add `README.md` (setup + usage).
2. Scaffold `api/main.py` with FastAPI health check (+ `/search` over the retriever).
3. Scaffold Next.js app in `frontend/`.

---
*Last updated: 2026-10-06 — Step 3 done (hybrid retrieval + rerank).*
