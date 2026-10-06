"""Pydantic request/response schemas for the VisuRAG API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# -- health ------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    qdrant_mode: str
    vlm_provider: str
    vlm_model: str
    counts: dict[str, int] = Field(default_factory=dict)


# -- ingest ------------------------------------------------------------
class IngestResponse(BaseModel):
    document_id: str
    source: str
    pages: int
    text_points: int
    visual_points: int
    cache_hit: bool


# -- documents ---------------------------------------------------------
class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool


# -- search ------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    document_id: str | None = None
    limit: int = Field(default=5, ge=1, le=20)


class SearchHit(BaseModel):
    key: str
    modality: str
    text: str | None = None
    source: str | None = None
    page_num: int | None = None
    fused_score: float
    rerank_score: float
    evidence_image_paths: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    document_id: str | None = None
    hits: list[SearchHit] = Field(default_factory=list)


# -- query (RAG + VLM) --------------------------------------------------
class QueryRequest(BaseModel):
    """User prompt for grounded generation.

    Accepts both ``query`` (canonical) and ``prompt`` (alias used by the
    task spec) so frontend clients can send either key.
    """

    query: str | None = Field(default=None, min_length=1)
    prompt: str | None = Field(default=None, min_length=1)
    document_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    max_images: int = Field(default=2, ge=0, le=5)

    def effective_query(self) -> str:
        q = (self.query or self.prompt or "").strip()
        if not q:
            raise ValueError("either 'query' or 'prompt' must be non-empty")
        return q


class SourceCitation(BaseModel):
    source: str | None = None  # PDF file name, e.g. "opamp.pdf"
    page_num: int | None = None
    chunk_id: str | None = None
    text_snippet: str | None = None
    image_patch_paths: list[str] = Field(default_factory=list)
    fused_score: float | None = None
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    model: str  # VLM model (or "echo-fallback")
    provider: str  # "echo" | "ollama" | "openai-compatible"
    source: str | None = None  # top-hit PDF file name
    page_num: int | None = None  # top-hit page
    image_patch_paths: list[str] = Field(default_factory=list)  # top-hit evidence
    citations: list[SourceCitation] = Field(default_factory=list)
    prompt_chars: int = 0  # grounded prompt length sent to the VLM
    images_sent: int = 0  # evidence images actually sent to the VLM
