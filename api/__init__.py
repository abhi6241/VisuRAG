"""VisuRAG api: FastAPI backend (Step 4).

Public API::

    from api import APISettings, VisuRAGService, app, get_service

    # Serve: uvicorn api.main:app --port 8000
    # Query: POST /query {"query": "...", "document_id": "..."}
"""

from .config import APISettings
from .main import app
from .schemas import (
    DeleteResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    SourceCitation,
)
from .service import VisuRAGService, get_service
from .vlm import (
    EchoVLMProvider,
    OllamaVLMProvider,
    OpenAICompatibleVLMProvider,
    build_rag_prompt,
    create_vlm_provider,
    encode_image_base64,
    pick_evidence_images,
)

__all__ = [
    "APISettings",
    "DeleteResponse",
    "EchoVLMProvider",
    "HealthResponse",
    "IngestResponse",
    "OllamaVLMProvider",
    "OpenAICompatibleVLMProvider",
    "QueryRequest",
    "QueryResponse",
    "SearchHit",
    "SearchRequest",
    "SearchResponse",
    "SourceCitation",
    "VisuRAGService",
    "app",
    "build_rag_prompt",
    "create_vlm_provider",
    "encode_image_base64",
    "get_service",
    "pick_evidence_images",
]
