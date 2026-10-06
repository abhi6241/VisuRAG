# Changelog

All notable changes to VisuRAG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Groq hosted-LLM provider (`GroqVLMProvider`, exported from `api`):
  `VISURAG_VLM_PROVIDER=groq` + `VISURAG_GROQ_API_KEY` (or `GROQ_API_KEY`,
  free at `console.groq.com/keys`) sends the RAG prompt + schematic images
  to `https://api.groq.com/openai/v1/chat/completions` (Bearer auth) with a
  vision-capable model (default `meta-llama/llama-4-scout-17b-16e-instruct`,
  swap via `VISURAG_GROQ_MODEL`, base override via `VISURAG_GROQ_BASE_URL`).
  Missing key fails fast with a clear error. README gains a "Using Groq"
  section + config rows.
- Verified end-to-end (TestClient, memory Qdrant) against a stub Groq
  server: Bearer header, model id, text + 2 base64 `image_url` blocks sent;
  stubbed answer flows through `/query` with attribution intact.

## [0.7.0] - 2026-10-06
### Added
- Root `README.md`: overview, pipeline diagram, prerequisites, quickstart (venv + `uvicorn` + `npm run dev`), API curl examples (ingest/search/query/delete/files), `VISURAG_` config table, structure, verification, roadmap.
- Per-document delete: `DELETE /documents/{document_id}` (`DeleteResponse`, 404 on unknown id; removes both Qdrant collections + BM25 registry entries, restart-safe via existence scroll) + frontend Delete button on the scoped doc (confirm dialog, resets scope/selection) via `deleteDocument()` in `lib/api.ts`.
- Smoke-tested: ingest → 1 hit → delete (200) → re-delete/unknown (404) → 0 hits after; frontend `npm run build` clean.

## [0.6.0] - 2026-10-06
### Added
- Next.js frontend in `frontend/` (Step 5 — VisuRAG MVP complete):
  - `app/page.tsx`: chat UI (health badge, PDF upload bar → `POST /ingest` with Scope selector over ingested `document_id`s, message list with selectable answers, ask form → `POST /query` with `top_k=5`/`max_images=4`, loading/error states).
  - `components/EvidencePane.tsx`: visual source attribution split-pane (top-source card, schematic-patch grid, per-citation `[Sn]` cards with snippets + rerank scores + thumbnails) plus lightbox modal (Escape/backdrop/Close).
  - `lib/api.ts`: typed backend client (`Health`/`IngestedDoc`/`Citation`/`QueryAnswer` mirroring `api/schemas.py`); `evidenceUrl()` maps backend `data/...` patch paths to `GET /files/...`; base URL via `NEXT_PUBLIC_VISURAG_API_URL` (default `http://localhost:8000`).
  - Pinned `package.json` (next 15.5.27, react 19.3.0, tailwindcss 4.3.3) + committed `package-lock.json`; rewritten `frontend/README.md` (setup + usage + layout); `.env.example`.
- Smoke-tested: `npm install` + `npm run build` clean (4 static routes); live E2E (backend memory + echo, `npm start`) — `/ingest` → 2 text + 24 visual, `/query` → attributed answer, `/files/...` → 200 PNG, frontend `/` → 200.

## [0.5.0] - 2026-10-06
### Added
- FastAPI backend in `api/` (Step 4):
  - `config.py`: `APISettings` (`VISURAG_` env prefix — qdrant mode/path/url, encoder/reranker choices, `vlm_provider`, Ollama/OpenAI-compatible URLs + models, `ingest_dpi`, top-k budgets); offline-safe defaults (path Qdrant, hash embeddings, heuristic rerank, echo VLM).
  - `schemas.py`: `HealthResponse` / `IngestResponse` / `SearchRequest+Response` (`SearchHit` with `evidence_image_paths`) / `QueryRequest` (accepts both `query` and `prompt` keys) / `QueryResponse` (`answer`, `model`, `provider`, `source`, `page_num`, `image_patch_paths`, `citations[]`, `prompt_chars`, `images_sent`).
  - `service.py`: `VisuRAGService` + `get_service()` singleton — `ingest_upload()` (PDF → `data/uploads/` → `ingest_pdf()` → `index_ingestion_result()` → BM25 registry), `search()`, `query()` (retrieve → prompt → VLM → attributed response); `_ensure_bm25()` rehydrates the in-memory BM25 index from Qdrant scroll after restarts.
  - `main.py`: `GET /health`, `POST /ingest` (multipart PDF, `?dpi=`; 400 non-PDF/empty, 422 render/index failure), `POST /search` (retrieval only), `POST /query` (RAG + VLM; 400 empty, 502 generation failure, graceful 200 on no hits), `GET /files/...` (traversal-blocked PNG evidence serving); CORS open for `localhost:3000`; `__main__.py` runner (`python -m api`).
- VLM integration in `api/vlm.py`:
  - `build_rag_prompt()` (`[S1..Sn]` blocks with `source` + page + text, 6000-char budget, cite-`[Sn]` system instruction), `pick_evidence_images()` (first-N unique sibling patches), `encode_image_base64()` (1024px downscale).
  - `EchoVLMProvider` (offline extractive `[S1]`-grounded fallback, default), `OllamaVLMProvider` (OpenAI-compatible `/v1/chat/completions` first, native `/api/chat` fallback; default `llama3.2-vision`, `qwen2-vl` swap-in), `OpenAICompatibleVLMProvider` (vLLM/LM-Studio/hosted); `create_vlm_provider()` factory via `VISURAG_VLM_PROVIDER`.
- `requirements.txt` adds `fastapi==0.118.0`, `uvicorn==0.34.2`, `httpx==0.28.1`, `starlette==0.48.0`, `pydantic-settings==2.7.0`, `python-dotenv==1.2.4`, `python-multipart==0.0.20`, `requests==2.32.4` (venv-only); `.gitignore` adds `data/uploads/`.
- Smoke-tested: TestClient (memory Qdrant + echo VLM) on 2-page synthetic PDF → `/health` ok; `/ingest` → 2 text + 24 visual points; `/search` → 2 hits with evidence; `/query` (both keys) → attributed answer (`source=smoke.pdf`, int `page_num`, `image_patch_paths` + 2 citations, `images_sent=2`); empty → 400; `/files/...` → 200 PNG; no-match → graceful 200; live `uvicorn` boot in `path` mode verified.

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
