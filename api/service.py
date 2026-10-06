"""Backend service: ingest → index → hybrid retrieve → VLM generate.

Thin orchestration over the existing pipelines — ``ingestion/`` (write),
``vector_db/`` (store), ``retrieval/`` (read), ``api/vlm.py`` (generate).
All heavy objects (Qdrant client, encoders, reranker, VLM) are built once
and reused across requests.
"""

from __future__ import annotations

from pathlib import Path

from ingestion import IngestionCache, IngestionConfig, ingest_pdf
from retrieval import HeuristicReranker, RetrievalConfig, VisuRAGRetriever
from vector_db import (
    HashEmbeddingProvider,
    VectorDBConfig,
    VisuRAGStore,
    build_text_chunks,
    index_ingestion_result,
)
from vector_db.models import TextChunk

from .config import APISettings
from .schemas import QueryResponse, SourceCitation
from .vlm import (
    build_rag_prompt,
    create_vlm_provider,
    encode_image_base64,
    pick_evidence_images,
)


def _make_text_encoder(settings: APISettings):
    if settings.text_encoder == "fastembed":
        from vector_db import FastEmbedTextProvider

        return FastEmbedTextProvider(model=settings.text_model)
    if settings.text_encoder in ("sbert", "sentence-transformer"):
        from vector_db import SentenceTransformerProvider

        return SentenceTransformerProvider(model=settings.text_model)
    if settings.text_encoder == "hash":
        return HashEmbeddingProvider(dim=384)
    raise ValueError(f"unknown text_encoder: {settings.text_encoder!r}")


def _make_image_encoder(settings: APISettings):
    if settings.image_encoder == "fastembed":
        from vector_db import FastEmbedImageProvider

        return FastEmbedImageProvider(model=settings.image_model)
    if settings.image_encoder == "hash":
        return HashEmbeddingProvider(dim=512)
    raise ValueError(f"unknown image_encoder: {settings.image_encoder!r}")


def _make_reranker(settings: APISettings):
    if settings.reranker == "cross-encoder":
        from retrieval import CrossEncoderReranker

        return CrossEncoderReranker(model_name=settings.cross_encoder_model)
    return HeuristicReranker()


class VisuRAGService:
    """Singleton backend state (store + retriever + VLM)."""

    def __init__(self, settings: APISettings | None = None) -> None:
        self.settings = settings or APISettings()
        s = self.settings
        s.cache_dir.mkdir(parents=True, exist_ok=True)
        s.upload_dir.mkdir(parents=True, exist_ok=True)

        self.store = VisuRAGStore(
            VectorDBConfig(
                mode=s.qdrant_mode,  # type: ignore[arg-type]
                path=s.qdrant_path,
                url=s.qdrant_url,
                api_key=s.qdrant_api_key,
            )
        )
        self.text_encoder = _make_text_encoder(s)
        self.image_encoder = _make_image_encoder(s)
        self.retriever = VisuRAGRetriever(
            self.store,
            self.text_encoder,
            config=RetrievalConfig(
                final_top_k=s.default_top_k,
                candidate_pool_size=s.candidate_pool_size,
            ),
            reranker=_make_reranker(s),
        )
        self.vlm = create_vlm_provider(s)
        # document_id -> chunks (backs the in-memory BM25 side-index).
        self._chunks_by_doc: dict[str, list] = {}

    # -- write path --------------------------------------------------
    def ingest_file(self, pdf_path: Path):
        """Ingest one PDF on disk → index → register BM25 chunks."""
        s = self.settings
        result = ingest_pdf(
            pdf_path,
            config=IngestionConfig(dpi=s.ingest_dpi, cache_dir=s.cache_dir),
            cache=IngestionCache(s.cache_dir),
        )
        summary = index_ingestion_result(
            result, self.store,
            text_encoder=self.text_encoder,
            image_encoder=self.image_encoder,
        )
        chunks = build_text_chunks(result)
        self._chunks_by_doc[result.document_id] = chunks
        # Rebuild the combined BM25 index over all known docs.
        all_chunks = [c for cs in self._chunks_by_doc.values() for c in cs]
        self.retriever.index_chunks(all_chunks)
        return result, summary

    def ingest_upload(self, filename: str, data: bytes, *, dpi: int | None = None):
        """Persist an uploaded PDF then run :meth:`ingest_file`."""
        dest = self.settings.upload_dir / Path(filename).name
        dest.write_bytes(data)
        if dpi is not None:
            orig = self.settings.ingest_dpi
            self.settings.ingest_dpi = dpi
            try:
                return self.ingest_file(dest)
            finally:
                self.settings.ingest_dpi = orig
        return self.ingest_file(dest)

    # -- read path ---------------------------------------------------
    def _ensure_bm25(self, document_id: str | None) -> None:
        """Rebuild BM25 from Qdrant payloads when the registry missed.

        Makes the API restart-safe: Qdrant ``path`` mode persists vectors,
        but the BM25 side-index is in-memory, so after a reboot the first
        query for a document rehydrates chunk texts via scroll.
        """
        if document_id is None or document_id in self._chunks_by_doc:
            return
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue

        records, _ = self.store.client.scroll(
            collection_name=self.store.config.text_collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id", match=MatchValue(value=document_id)
                    )
                ]
            ),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return
        chunks = [
            TextChunk(
                chunk_id=r.payload["chunk_id"],
                document_id=document_id,
                source=r.payload.get("source", ""),
                page_num=r.payload.get("page_num", 0),
                text=r.payload.get("text", ""),
                block_no=r.payload.get("block_no"),
                bbox_pdf=tuple(r.payload["bbox_pdf"])
                if r.payload.get("bbox_pdf") else None,
            )
            for r in records
        ]
        self._chunks_by_doc[document_id] = chunks
        all_chunks = [c for cs in self._chunks_by_doc.values() for c in cs]
        self.retriever.index_chunks(all_chunks)

    def search(self, query: str, *, document_id: str | None = None, limit: int = 5):
        self._ensure_bm25(document_id)
        return self.retriever.search(query, document_id=document_id, limit=limit)

    def delete_document(self, document_id: str) -> bool:
        """Remove a document from Qdrant + the BM25 registry.

        Returns True when the document existed (in either store),
        False when nothing was found.
        """
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue

        known = document_id in self._chunks_by_doc
        if not known:
            records, _ = self.store.client.scroll(
                collection_name=self.store.config.text_collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id", match=MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            known = bool(records)
        if not known:
            return False
        self.store.delete_document(document_id)
        self._chunks_by_doc.pop(document_id, None)
        all_chunks = [c for cs in self._chunks_by_doc.values() for c in cs]
        self.retriever.index_chunks(all_chunks)
        return True

    def query(
        self, prompt: str, *, document_id: str | None = None,
        top_k: int = 5, max_images: int = 2,
    ) -> QueryResponse:
        """Hybrid retrieve → multimodal prompt → VLM → attributed response."""
        hits, rag_prompt, images_b64 = self._prepare(
            prompt, document_id=document_id, top_k=top_k, max_images=max_images,
        )
        if not hits:
            return self._empty_response()
        answer = self.vlm.generate(rag_prompt, images_b64)
        return self._respond(hits, rag_prompt, images_b64, answer)

    def query_stream(
        self, prompt: str, *, document_id: str | None = None,
        top_k: int = 5, max_images: int = 2,
    ):
        """Yield ``(event, data)`` SSE pairs: ``retrieval`` → ``token``* →
        ``done``. ``done`` carries the full :class:`QueryResponse` dict."""
        hits, rag_prompt, images_b64 = self._prepare(
            prompt, document_id=document_id, top_k=top_k, max_images=max_images,
        )
        if not hits:
            empty = self._empty_response()
            yield "retrieval", {"citations": [], "images_sent": 0}
            yield "done", empty.model_dump()
            return
        citations = [c.model_dump() for c in self._citations(hits)]
        yield "retrieval", {
            "source": hits[0].payload.get("source"),
            "page_num": hits[0].payload.get("page_num"),
            "citations": citations,
            "images_sent": len(images_b64),
        }
        streamer = getattr(self.vlm, "generate_stream", None)
        if streamer is None:  # provider without streaming — one chunk
            streamer = lambda p, i: iter([self.vlm.generate(p, i)])  # noqa: E731
        pieces: list[str] = []
        for piece in streamer(rag_prompt, images_b64):
            pieces.append(piece)
            yield "token", {"text": piece}
        yield "done", self._respond(
            hits, rag_prompt, images_b64, "".join(pieces)
        ).model_dump()

    # -- shared RAG helpers ------------------------------------------
    def _prepare(self, prompt, *, document_id, top_k, max_images):
        """Retrieve + build prompt + encode evidence images."""
        s = self.settings
        hits = self.search(prompt, document_id=document_id, limit=top_k)
        if not hits:
            return hits, "", []
        rag_prompt = build_rag_prompt(prompt, hits, max_chars=s.max_context_chars)
        image_paths = pick_evidence_images(hits, max_images=max_images)
        images_b64: list[str] = []
        for p in image_paths:
            try:
                images_b64.append(encode_image_base64(p))
            except (OSError, ValueError):
                continue  # missing/corrupt patch file — text context still works
        return hits, rag_prompt, images_b64

    def _empty_response(self) -> QueryResponse:
        s = self.settings
        return QueryResponse(
            answer="No relevant context was found in the indexed documents, "
            "so I cannot answer. Try ingesting the datasheet first.",
            model=getattr(self.vlm, "name", "unknown"),
            provider=s.vlm_provider,
            citations=[],
        )

    def _citations(self, hits) -> list[SourceCitation]:
        return [
            SourceCitation(
                source=h.payload.get("source"),
                page_num=h.payload.get("page_num"),
                chunk_id=h.payload.get("chunk_id"),
                text_snippet=(h.text[:300] + "…" if h.text and len(h.text) > 300 else h.text),
                image_patch_paths=[
                    ev.get("image_path")
                    for ev in (h.visual_evidence or ())
                    if isinstance(ev, dict) and ev.get("image_path")
                ],
                fused_score=h.fused_score,
                rerank_score=h.rerank_score,
            )
            for h in hits
        ]

    def _respond(self, hits, rag_prompt, images_b64, answer) -> QueryResponse:
        s = self.settings
        citations = self._citations(hits)
        top = hits[0].payload
        top_evidence = citations[0].image_patch_paths if citations else []
        return QueryResponse(
            answer=answer,
            model=getattr(self.vlm, "name", "unknown"),
            provider=s.vlm_provider,
            source=top.get("source"),
            page_num=top.get("page_num"),
            image_patch_paths=top_evidence,
            citations=citations,
            prompt_chars=len(rag_prompt),
            images_sent=len(images_b64),
        )


_service: VisuRAGService | None = None


def get_service(settings: APISettings | None = None) -> VisuRAGService:
    """Process-wide singleton (rebuilt when explicit ``settings`` given)."""
    global _service
    if _service is None or settings is not None:
        _service = VisuRAGService(settings)
    return _service
