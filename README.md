# VisuRAG — Multimodal RAG for Engineering Datasheets & Schematics

Ask technical questions over PDF datasheets and get grounded answers with
**visual source attribution**: the exact source file, page number, and
schematic image patches behind every answer.

```
PDF ingest → page render + patch tiling → dual Qdrant collections
                                                      ↓
user question → dense + BM25 → RRF fusion → rerank → VLM → answer + citations
```

## How it works

1. **Ingestion** (`ingestion/`): PDFs are rasterized with PyMuPDF and tiled
   into overlapping image patches; text blocks are extracted with layout
   bboxes. Results are disk-cached by file hash.
2. **Vector layer** (`vector_db/`): text chunks and image patches are embedded
   and stored in two Qdrant collections (`visurag_text`, `visurag_visual`)
   with shared `document_id` / `page_num` payloads.
3. **Retrieval** (`retrieval/`): hybrid fan-out (Qdrant dense + BM25 sparse
   → RRF fusion → cross-encoder/heuristic rerank) plus sibling-patch visual
   evidence per hit page.
4. **Backend** (`api/`): FastAPI — `/health`, `/ingest`, `/search`,
   `/query` (RAG + vision-language model), `/files` (evidence PNGs).
5. **Frontend** (`frontend/`): Next.js chat UI with a source-attribution
   split-pane (cited pages, patch grid, lightbox).

**No API keys required.** Defaults are fully offline (deterministic hash
embeddings, heuristic rerank, extractive echo VLM). Real models
(FastEmbed, cross-encoder, Ollama vision) are environment opt-ins —
see [Configuration](#configuration).

## Prerequisites

- Python 3.12 (`python3 --version`)
- Node 20+ (`node --version`) — frontend only

## Quickstart

```bash
# 1. Backend env (from repo root)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Start the API
uvicorn api.main:app --port 8000
# → {"status":"ok", ...} at http://localhost:8000/health

# 3. Frontend (new terminal)
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

Upload a datasheet PDF in the UI, then ask e.g. *“Which pin is VCC?”*.
Click an answer to inspect its cited pages and schematic patches on the
right; click any patch for a full-size lightbox.

## API usage (without the frontend)

```bash
# Ingest a PDF (returns document_id for scoped search)
curl -X POST http://localhost:8000/ingest \
  -F "file=@opamp.pdf;type=application/pdf"

# Retrieval only (no generation)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"pin table VCC GND","document_id":"<id>"}'

# Full RAG (accepts "query" or "prompt")
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Which pin is VCC?","document_id":"<id>"}'

# Streaming RAG (SSE: retrieval -> token* -> done; errors as "error")
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"Which pin is VCC?","document_id":"<id>"}'

# Delete one document (both modalities + in-memory BM25 entries)
curl -X DELETE http://localhost:8000/documents/<document_id>

# Serve an evidence image (paths come from search/query responses)
curl http://localhost:8000/files/cache/patches/<document_id>/<patch>.png
```

## Configuration

Backend settings use the `VISURAG_` prefix (see `api/config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `VISURAG_QDRANT_MODE` | `path` | `memory` (tests) / `path` (`data/qdrant`, no Docker) / `url` (external server) |
| `VISURAG_QDRANT_URL` / `VISURAG_QDRANT_API_KEY` | — | External Qdrant server + key |
| `VISURAG_TEXT_ENCODER` | `hash` | `hash` (offline) / `fastembed` (bge-small, downloads once) / `sbert` |
| `VISURAG_IMAGE_ENCODER` | `hash` | `hash` (offline) / `fastembed` (CLIP ViT-B/32, downloads once) |
| `VISURAG_RERANKER` | `heuristic` | `heuristic` (offline) / `cross-encoder` (downloads once) |
| `VISURAG_VLM_PROVIDER` | `echo` | `echo` (offline extractive) / `ollama` / `openai-compatible` / `groq` |
| `VISURAG_OLLAMA_MODEL` | `llama3.2-vision` | e.g. `qwen2-vl`; needs Ollama running |
| `VISURAG_GROQ_API_KEY` | — | Groq key (free at `console.groq.com/keys`); or plain `GROQ_API_KEY` |
| `VISURAG_GROQ_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | e.g. `qwen/qwen3.8-27b` |
| `VISURAG_OPENAI_COMPATIBLE_BASE_URL` / `..._MODEL` / `..._API_KEY` | — | vLLM / LM Studio / hosted vision endpoint |
| `VISURAG_INGEST_DPI` | `150` | PDF render resolution for `/ingest` |
| `VISURAG_API_KEY` | — (open) | If set, all routes except `/health` require `X-API-Key` (`?api_key=` works for `/files` images) |
| `VISURAG_RATE_LIMIT_PER_MIN` | `120` | Per-IP sliding window (`0` disables); `/health` exempt, 429 + `Retry-After` |

Frontend: `NEXT_PUBLIC_VISURAG_API_URL` (default `http://localhost:8000`).
Optional `NEXT_PUBLIC_VISURAG_API_KEY` sends the key with every request —
only for trusted self-hosted deployments, since browser keys are visible.

## Using Groq (hosted LLM, no local server)

1. Get a free API key at `https://console.groq.com/keys` (never commit it —
   `.env` is gitignored).
2. Start the backend with it:
   ```bash
   source venv/bin/activate
   VISURAG_VLM_PROVIDER=groq VISURAG_GROQ_API_KEY=gsk_... \
     uvicorn api.main:app --port 8000
   ```
3. `/health` will report `"vlm_model": "groq:meta-llama/llama-4-..."`.
   Every `/query` answer is now generated by Groq from the retrieved
   `[Sn]` context blocks plus schematic patch images.

Swap models with `VISURAG_GROQ_MODEL` (e.g. `qwen/qwen3.8-27b`).
On any generation failure the API returns 502 with the upstream detail.

## Project structure

```
├── ingestion/    # PDF render → patches → disk cache
├── vector_db/    # Qdrant client, schema, embeddings, store, indexing
├── retrieval/    # BM25 + RRF fusion + rerankers + hybrid engine
├── api/          # FastAPI backend + VLM providers
├── frontend/     # Next.js chat + attribution UI (Node-only)
├── data/         # gitignored: cache/, qdrant/, uploads/
├── CONTEXT.md    # living architecture record
└── CHANGELOG.md  # release history
```

## Verification

```bash
source venv/bin/activate
python -m py_compile api/*.py ingestion/*.py vector_db/*.py retrieval/*.py
cd frontend && npm run build
```

See `CONTEXT.md` for per-step smoke-test records and `CHANGELOG.md`
for release history (current: `0.6.0`, MVP complete).

## Roadmap (post-MVP)

- Auth / rate-limits, streaming `/query`
- Real-model latency/quality notes (FastEmbed + cross-encoder + Ollama vision)
