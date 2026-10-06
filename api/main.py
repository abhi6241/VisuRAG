"""FastAPI backend for VisuRAG.

Endpoints:

- ``GET /health`` — liveness + Qdrant counts + active VLM provider/model.
- ``POST /ingest`` — multipart PDF upload → ingest → index (Qdrant +
  BM25). Returns ``document_id`` for scoped ``/search`` / ``/query``.
- ``POST /search`` — hybrid retrieval only (no generation). Returns ranked
  hits with per-hit visual-evidence patch paths.
- ``POST /query`` — full RAG: hybrid retrieval → multimodal prompt
  (text + schematic patches) → VLM → attributed answer JSON
  (``answer``, ``source``, ``page_num``, ``image_patch_paths``).
- ``DELETE /documents/{document_id}`` — remove one document from Qdrant
  (both modalities) + the BM25 side-index. Unknown id → 404.
- ``GET /files/...`` — serves cached page/patch PNGs for visual citation.

Run locally::

    source venv/bin/activate
    uvicorn api.main:app --reload --port 8000
    # or: python -m api
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import APISettings
from .schemas import (
    DeleteResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from .service import get_service

settings = APISettings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = get_service(settings)
    yield


app = FastAPI(title="VisuRAG", version="0.7.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def svc():
    return get_service()


# -- health ------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    service = svc()
    vlm_name = getattr(service.vlm, "name", "unknown")
    try:
        counts = service.store.count()
    except Exception:
        counts = {}
    return HealthResponse(
        status="ok",
        qdrant_mode=service.settings.qdrant_mode,
        vlm_provider=service.settings.vlm_provider,
        vlm_model=vlm_name,
        counts=counts,
    )


# -- ingest ------------------------------------------------------------
@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    dpi: int | None = Query(default=None, ge=72, le=600),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="upload must be a .pdf file")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="uploaded PDF is empty")
    service = svc()
    try:
        result, summary = service.ingest_upload(file.filename, data, dpi=dpi)
    except Exception as e:  # corrupt PDF, render failure, Qdrant error
        raise HTTPException(status_code=422, detail=f"ingest failed: {e}") from e
    return IngestResponse(
        document_id=result.document_id,
        source=Path(result.source_path).name,
        pages=len(result.pages),
        text_points=summary.text_points,
        visual_points=summary.visual_points,
        cache_hit=result.cache_hit,
    )


# -- search (retrieval only) -------------------------------------------
@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must be non-empty")
    hits = svc().search(req.query, document_id=req.document_id, limit=req.limit)
    return SearchResponse(
        query=req.query,
        document_id=req.document_id,
        hits=[
            SearchHit(
                key=h.key,
                modality=h.modality,
                text=h.text,
                source=h.payload.get("source"),
                page_num=h.payload.get("page_num"),
                fused_score=h.fused_score,
                rerank_score=h.rerank_score,
                evidence_image_paths=[
                    ev.get("image_path")
                    for ev in (h.visual_evidence or ())
                    if isinstance(ev, dict) and ev.get("image_path")
                ],
            )
            for h in hits
        ],
    )


# -- query (retrieval + VLM generation) --------------------------------
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        q = req.effective_query()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    service = svc()
    try:
        return service.query(
            q, document_id=req.document_id,
            top_k=req.top_k, max_images=req.max_images,
        )
    except Exception as e:  # VLM connection failure etc. — 502, not 500
        raise HTTPException(
            status_code=502, detail=f"generation failed: {e}"
        ) from e


# -- documents -------------------------------------------------------
@app.delete("/documents/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: str):
    if not svc().delete_document(document_id):
        raise HTTPException(status_code=404, detail="document not found")
    return DeleteResponse(document_id=document_id, deleted=True)


# -- evidence files (visual citation) ----------------------------------
@app.get("/files/{subpath:path}")
def evidence_file(subpath: str):
    """Serve a cached page/patch PNG. ``subpath`` must stay under ``data/``."""
    base = Path("data").resolve()
    target = (base / subpath).resolve()
    if base not in target.parents and target != base:
        raise HTTPException(status_code=403, detail="path escapes data/")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    if target.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        raise HTTPException(status_code=403, detail="only image files are served")
    return FileResponse(target)
