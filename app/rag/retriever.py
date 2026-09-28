"""原始文本 query 的 dense 召回入口。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from qdrant_client import models

from app.clients.dashscope import DashScopeClient
from app.config import Settings
from app.schemas.embedding import EmbeddingVector
from app.schemas.rag import SearchHit, SearchRequest, SearchResponse, SourceReference
from app.stores.qdrant import QdrantStore


class QueryEmbeddingProvider(Protocol):
    async def embed_query(
        self,
        text: str,
        *,
        instruct: str | None = None,
    ) -> EmbeddingVector: ...


class DenseVectorStore(Protocol):
    async def search_text_dense(
        self,
        query_vector: list[float],
        *,
        limit: int,
        course_ids: Sequence[str] = (),
        document_ids: Sequence[str] = (),
    ) -> list[models.ScoredPoint]: ...


class HybridRetriever:
    """当前执行 dense baseline；后续在同一入口增加混合召回与重排。"""

    def __init__(
        self,
        query_embedding_provider: QueryEmbeddingProvider,
        vector_store: DenseVectorStore,
    ) -> None:
        self.query_embedding_provider = query_embedding_provider
        self.vector_store = vector_store

    @classmethod
    def from_settings(cls, settings: Settings) -> HybridRetriever:
        return cls(DashScopeClient(settings), QdrantStore(settings))

    async def search(self, request: SearchRequest) -> SearchResponse:
        """保持原问题不改写，执行 text_dense Cosine Top K 召回。"""

        query_vector = await self.query_embedding_provider.embed_query(request.query)
        points = await self.vector_store.search_text_dense(
            query_vector.dense,
            limit=request.top_k,
            course_ids=request.scope.course_ids,
            document_ids=request.scope.document_ids,
        )
        return SearchResponse(
            query=request.query,
            hits=[_to_search_hit(point) for point in points],
        )


def _to_search_hit(point: models.ScoredPoint) -> SearchHit:
    payload = point.payload or {}
    point_id = str(point.id)
    chunk_id = _required_string(payload, "chunk_id", point_id)
    section_id = _required_string(payload, "section_id", point_id)
    representation_type = _required_string(payload, "representation_type", point_id)
    content = _required_string(payload, "content", point_id)
    source_path = _required_string(payload, "source_path", point_id)
    course_id = _required_string(payload, "course_id", point_id)
    document_id = _required_string(payload, "document_id", point_id)
    cell_ids = _string_list(payload.get("cell_ids"))
    section_path = _string_list(payload.get("section_path"))
    return SearchHit(
        chunk_id=chunk_id,
        section_id=section_id,
        representation_type=representation_type,
        score=float(point.score),
        content=content,
        source=SourceReference(
            course_id=course_id,
            document_id=document_id,
            file=source_path,
            section_path=section_path,
            cell_ids=cell_ids,
            cell_id=cell_ids[0] if cell_ids else None,
        ),
    )


def _required_string(payload: dict[str, object], key: str, point_id: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Qdrant point {point_id} 的 payload 缺少有效字段 {key!r}")
    return value


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]
