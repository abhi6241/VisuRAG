"""Collection schemas: dual indexes for text chunks + visual patches.

Design decision: **two collections, one per modality**, sharing the same
payload keys (``document_id``, ``source``, ``page_num``, ``modality``).
Rationale vs. a single collection with named vectors:

- Text and visual encoders emit different dims (384 vs 512); separate
  collections avoid padding/truncation hacks and let each index use its
  native space and HNSW tuning.
- Joint retrieval is a thin fan-out: query both collections with the same
  ``document_id``/``page_num`` filters, then fuse in ``retrieval/``.
- Per-modality re-indexing (e.g. swapping the CLIP model) never touches
  the other collection.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    VectorParams,
)

from .config import VectorDBConfig

_DISTANCE_BY_NAME = {
    "cosine": Distance.COSINE,
    "dot": Distance.DOT,
    "euclid": Distance.EUCLID,
    "manhattan": Distance.MANHATTAN,
}


def _distance(name: str) -> Distance:
    try:
        return _DISTANCE_BY_NAME[name.strip().lower()]
    except KeyError:
        raise ValueError(f"unknown distance {name!r}; use Cosine|Dot|Euclid|Manhattan")


def ensure_collections(client: QdrantClient, config: VectorDBConfig) -> tuple[str, str]:
    """Create both collections (if missing) with payload indexes.

    Returns ``(text_collection, visual_collection)``.
    Safe to call repeatedly — existing collections are left untouched.
    """
    specs = (
        (config.text_collection, config.text_dim),
        (config.visual_collection, config.visual_dim),
    )
    existing = {c.name for c in client.get_collections().collections}
    for name, dim in specs:
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=_distance(config.distance)),
            )
        # Keyword / integer indexes power the metadata filters used by
        # retrieval (per-document, per-page, per-source scoping).
        for field, schema in (
            ("document_id", PayloadSchemaType.KEYWORD),
            ("source", PayloadSchemaType.KEYWORD),
            ("modality", PayloadSchemaType.KEYWORD),
            ("page_num", PayloadSchemaType.INTEGER),
        ):
            try:
                client.create_payload_index(
                    collection_name=name, field_name=field, field_schema=schema
                )
            except Exception:
                # Index already exists (or local-mode race) — not fatal.
                pass
    return config.text_collection, config.visual_collection


def document_filter(document_id: str) -> Filter:
    """Exact-match filter scoping a search to one ingested document."""
    return Filter(
        must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
    )
