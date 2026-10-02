"""混合检索入口：文本 Dense+Sparse RRF 与按需视觉召回。"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

from qdrant_client import models

from app.clients.dashscope import DashScopeClient
from app.config import Settings
from app.rag.fusion import prefer_visual_pages, reciprocal_rank_fusion, visual_query_intent
from app.rag.query import construct_query
from app.schemas.embedding import EmbeddingVector, MultimodalEmbeddingBatch, SparseValue
from app.schemas.rag import SearchHit, SearchRequest, SearchResponse, SourceReference
from app.stores.qdrant import QdrantStore
from app.stores.sqlite import MetadataStore, VisibleVersion

logger = logging.getLogger(__name__)


class QueryEmbeddingProvider(Protocol):
    async def embed_query(
        self,
        text: str,
        *,
        instruct: str | None = None,
    ) -> EmbeddingVector: ...

    async def embed_visual_query(self, text: str) -> MultimodalEmbeddingBatch: ...


class HybridVectorStore(Protocol):
    async def search_text_dense(
        self,
        query_vector: list[float],
        *,
        limit: int,
        course_ids: Sequence[str] = (),
        document_ids: Sequence[str] = (),
        visible_versions: Sequence[tuple[str, int]] | None = None,
    ) -> list[models.ScoredPoint]: ...

    async def search_text_sparse(
        self,
        query_sparse: models.SparseVector,
        *,
        limit: int,
        course_ids: Sequence[str] = (),
        document_ids: Sequence[str] = (),
        visible_versions: Sequence[tuple[str, int]] | None = None,
    ) -> list[models.ScoredPoint]: ...

    async def search_visual_fusion(
        self,
        query_vector: list[float],
        *,
        limit: int,
        course_ids: Sequence[str] = (),
        document_ids: Sequence[str] = (),
        visible_versions: Sequence[tuple[str, int]] | None = None,
    ) -> list[models.ScoredPoint]: ...


class VisibilityStore(Protocol):
    def visible_versions(
        self,
        *,
        course_ids: list[str] | tuple[str, ...] = (),
        document_ids: list[str] | tuple[str, ...] = (),
    ) -> list[VisibleVersion]: ...


class HybridRetriever:
    """混合检索入口：文本 RRF 与视觉页优先路由。"""

    def __init__(
        self,
        query_embedding_provider: QueryEmbeddingProvider,
        vector_store: HybridVectorStore,
        metadata_store: VisibilityStore | None = None,
        visual_enabled: bool = False,
    ) -> None:
        self.query_embedding_provider = query_embedding_provider
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.visual_enabled = visual_enabled

    @classmethod
    def from_settings(cls, settings: Settings) -> HybridRetriever:
        metadata_store = MetadataStore(settings)
        metadata_store.initialize()
        dashscope_client = DashScopeClient(settings)
        return cls(
            query_embedding_provider=dashscope_client,
            vector_store=QdrantStore(settings),
            metadata_store=metadata_store,
            visual_enabled=settings.retrieval_enable_visual,
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        """文本问题用 Dense+Sparse；显式视觉问题优先返回视觉页。"""

        constructed = construct_query(request)
        visible_versions: list[tuple[str, int]] | None = None
        if self.metadata_store is not None:
            visible = self.metadata_store.visible_versions(
                course_ids=request.scope.course_ids,
                document_ids=request.scope.document_ids,
            )
            visible_versions = [(item.document_id, item.version) for item in visible]
            if not visible_versions:
                return SearchResponse(
                    query=request.query,
                    hits=[],
                    retrieval_query=constructed.text,
                    query_method=constructed.method,
                )

        # 1. Query embedding（dense + sparse）
        query_vector = await self.query_embedding_provider.embed_query(constructed.text)

        # 2. 并行召回：dense、sparse
        # 每路召回更多候选以便融合后有足够多样性
        recall_limit = min(request.top_k * 3, 50)

        dense_points = await self.vector_store.search_text_dense(
            query_vector.dense,
            limit=recall_limit,
            course_ids=request.scope.course_ids,
            document_ids=request.scope.document_ids,
            visible_versions=visible_versions,
        )

        sparse_points: list[models.ScoredPoint] = []
        if query_vector.sparse:
            sparse_vector = _to_sparse_vector(query_vector.sparse)
            sparse_points = await self.vector_store.search_text_sparse(
                sparse_vector,
                limit=recall_limit,
                course_ids=request.scope.course_ids,
                document_ids=request.scope.document_ids,
                visible_versions=visible_versions,
            )

        # 3. RRF 融合 dense 和 sparse
        text_fused = reciprocal_rank_fusion([dense_points, sparse_points], k=60)

        final_points = text_fused[: request.top_k]
        mode_used = "text"
        use_visual = request.retrieval_mode == "visual" or (
            request.retrieval_mode == "auto"
            and self.visual_enabled
            and visual_query_intent(constructed.text)
        )
        if use_visual:
            try:
                visual_query = await self.query_embedding_provider.embed_visual_query(
                    constructed.text
                )
                if len(visual_query.vectors) != 1:
                    raise RuntimeError("视觉查询向量数量不为 1")
                visual_points = await self.vector_store.search_visual_fusion(
                    visual_query.vectors[0].dense,
                    limit=recall_limit,
                    course_ids=request.scope.course_ids,
                    document_ids=request.scope.document_ids,
                    visible_versions=visible_versions,
                )
                if visual_points:
                    final_points = prefer_visual_pages(
                        visual_points, text_fused, top_k=request.top_k
                    )
                    mode_used = "visual"
            except Exception:
                if request.retrieval_mode == "visual":
                    raise
                logger.exception("自动视觉召回失败，回退文本结果")

        return SearchResponse(
            query=request.query,
            hits=[_to_search_hit(point) for point in final_points],
            mode_used=mode_used,
            retrieval_query=constructed.text,
            query_method=constructed.method,
        )


def _to_search_hit(point: models.ScoredPoint) -> SearchHit:
    payload = point.payload or {}
    point_id = str(point.id)
    representation_type = _required_string(payload, "representation_type", point_id)

    # document_outline 没有 chunk_id，使用 representation_id 作为标识
    if representation_type == "document_outline":
        chunk_id = _required_string(payload, "representation_id", point_id)
        section_id = _required_string(payload, "document_id", point_id)
    else:
        chunk_id = _required_string(payload, "chunk_id", point_id)
        section_id = _required_string(payload, "section_id", point_id)

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
        evidence_origin=payload.get("evidence_origin", "course_text"),
        origin_chunk_id=payload.get("origin_chunk_id"),
        source=SourceReference(
            course_id=course_id,
            document_id=document_id,
            version=payload.get("version") if isinstance(payload.get("version"), int) else None,
            file=source_path,
            section_path=section_path,
            cell_ids=cell_ids,
            cell_id=cell_ids[0] if cell_ids else None,
            page=payload.get("page") if isinstance(payload.get("page"), int) else None,
            image_path=payload.get("image_path")
            if isinstance(payload.get("image_path"), str)
            else None,
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


def _to_sparse_vector(sparse_values: list[SparseValue]) -> models.SparseVector:
    """将 SparseValue 列表转换为 Qdrant SparseVector。"""
    if not sparse_values:
        return models.SparseVector(indices=[], values=[])
    sorted_values = sorted(sparse_values, key=lambda v: v.index)
    return models.SparseVector(
        indices=[v.index for v in sorted_values],
        values=[v.value for v in sorted_values],
    )
