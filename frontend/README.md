# VisuRAG Frontend

Next.js 15 (TypeScript, App Router) + Tailwind CSS v4 client for VisuRAG —
chat over engineering datasheets with visual source attribution.

## Prerequisites

- Node 20+ and the running FastAPI backend (`uvicorn api.main:app --port 8000`
  from the repo root, venv activated).

## Setup

```bash
cd frontend
cp .env.example .env.local   # optional; defaults to http://localhost:8000
npm install
npm run dev                  # http://localhost:3000
```

Production check: `npm run build && npm start`.

## What it does

- **Upload** a datasheet PDF → `POST /ingest` (multipart). Ingested docs
  appear in the **Scope** selector (all docs, or one `document_id`).
- **Ask** a technical question → `POST /query` (`{query, document_id?,
  top_k, max_images}`).
- **Attribution split-pane** (right): top source (`source` filename +
  `page_num` badges, VLM model), schematic patch grid (click → lightbox
  modal), and per-citation `[Sn]` cards with text snippets, rerank scores,
  and patch thumbnails. Patch paths (`data/...`) are mapped to backend
  `GET /files/...` URLs in `lib/api.ts` (`evidenceUrl()`).
- **Header health badge**: green with VLM model + index counts when
  `GET /health` succeeds, red with a start-backend hint otherwise.

## Layout

```
frontend/
├── package.json          # pinned: next 15.5.27, react 19.3.0, tailwindcss 4.3.3
├── .env.example          # NEXT_PUBLIC_VISURAG_API_URL
├── app/
│   ├── layout.tsx        # metadata + global shell
│   ├── page.tsx          # chat + upload + scope + split-pane wiring
│   └── globals.css       # tailwindcss v4 import
├── components/
│   └── EvidencePane.tsx  # attribution pane + image lightbox modal
└── lib/
    └── api.ts            # typed backend client (health/ingest/query, evidenceUrl)
```

This directory is Node-only (never Python). `node_modules/` and `.next/`
are gitignored at the repo root; `package-lock.json` is committed.
