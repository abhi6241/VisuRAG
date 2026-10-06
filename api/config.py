"""API configuration (env-driven, ``VISURAG_`` prefix).

All settings have offline-safe defaults so the backend boots with zero
model downloads and no Ollama server:

- Qdrant ``path`` mode (``data/qdrant``), hash embeddings, heuristic rerank.
- VLM ``echo`` provider (extractive fallback; see :mod:`api.vlm`).

Set ``VISURAG_VLM_PROVIDER=ollama`` + ``VISURAG_OLLAMA_MODEL`` (e.g.
``llama3.2-vision`` or ``qwen2-vl``) to get real vision generation.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    """Environment-driven backend settings."""

    model_config = SettingsConfigDict(env_prefix="VISURAG_", extra="ignore")

    # -- vector store -------------------------------------------------
    qdrant_mode: str = "path"  # memory | path | url
    qdrant_path: Path = Path("data/qdrant")
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None

    # -- encoders ("hash" = offline deterministic; "fastembed" = real ONNX) --
    text_encoder: str = "hash"
    text_model: str = "BAAI/bge-small-en-v1.5"
    image_encoder: str = "hash"
    image_model: str = "Qdrant/clip-ViT-B-32-vision"

    # -- reranker ("heuristic" offline; "cross-encoder" = real) --
    reranker: str = "heuristic"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # -- retrieval budgets (mirror RetrievalConfig defaults) --
    default_top_k: int = 5
    candidate_pool_size: int = 20

    # -- ingestion --
    ingest_dpi: int = 150
    cache_dir: Path = Path("data/cache")
    upload_dir: Path = Path("data/uploads")

    # -- VLM ("echo" offline fallback; "ollama" / "openai-compatible" live) --
    vlm_provider: str = "echo"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2-vision"
    openai_compatible_base_url: str | None = None
    openai_compatible_model: str | None = None
    openai_compatible_api_key: str | None = None
    # Groq hosted inference (OpenAI-compatible). Free key at
    # https://console.groq.com/keys — no local server needed.
    groq_api_key: str | None = None
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    vlm_timeout_s: float = 120.0
    max_evidence_images: int = 2
    max_context_chars: int = 6000
