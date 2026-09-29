"""混合检索入口：支持 dense、sparse、RRF、outline、rerank 和查询理解。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from qdrant_client import models

from app.clients.dashscope import DashScopeClient
from app.clients.deepseek import DeepSeekClient
from app.config import Settings
from app.rag.fusion import reciprocal_rank_fusion
from app.schemas.embedding import EmbeddingVector, SparseValue
from app.schemas.rag import SearchHit, SearchRequest, SearchResponse, SourceReference
from app.stores.qdrant import QdrantStore
from app.stores.sqlite import MetadataStore, VisibleVersion


class QueryEmbeddingProvider(Protocol):
    async def embed_query(
        self,
        text: str,
        *,
        instruct: str | None = None,
    ) -> EmbeddingVector: ...


class RerankProvider(Protocol):
    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int | None = None,
    ) -> list[tuple[int, float]]: ...


class QueryUnderstandingProvider(Protocol):
    async def understand_query(
        self,
        query: str,
        *,
        enable_rewrite: bool = False,
    ) -> dict[str, object]: ...


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

    async def search_document_outline(
        self,
        query_vector: list[float],
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

    async def get_chunks_by_ids(
        self,
        chunk_ids: Sequence[str],
        *,
        document_id: str,
        version: int,
    ) -> list[models.ScoredPoint]: ...


class VisibilityStore(Protocol):
    def visible_versions(
        self,
        *,
        course_ids: list[str] | tuple[str, ...] = (),
        document_ids: list[str] | tuple[str, ...] = (),
    ) -> list[VisibleVersion]: ...


class HybridRetriever:
    """混合检索入口：支持 dense+sparse RRF、outline 融合、rerank 和查询理解。"""

    def __init__(
        self,
        query_embedding_provider: QueryEmbeddingProvider,
        vector_store: HybridVectorStore,
        metadata_store: VisibilityStore | None = None,
        rerank_provider: RerankProvider | None = None,
        query_understanding_provider: QueryUnderstandingProvider | None = None,
    ) -> None:
        self.query_embedding_provider = query_embedding_provider
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.rerank_provider = rerank_provider
        self.query_understanding_provider = query_understanding_provider

    @classmethod
    def from_settings(cls, settings: Settings) -> HybridRetriever:
        metadata_store = MetadataStore(settings)
        metadata_store.initialize()
        dashscope_client = DashScopeClient(settings)
        deepseek_client = DeepSeekClient(settings)
        return cls(
            query_embedding_provider=dashscope_client,
            vector_store=QdrantStore(settings),
            metadata_store=metadata_store,
            rerank_provider=dashscope_client,
            query_understanding_provider=deepseek_client,
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        """执行混合检索：dense+sparse RRF、outline 融合、rerank。"""

        visible_versions: list[tuple[str, int]] | None = None
        if self.metadata_store is not None:
            visible = self.metadata_store.visible_versions(
                course_ids=request.scope.course_ids,
                document_ids=request.scope.document_ids,
            )
            visible_versions = [(item.document_id, item.version) for item in visible]
            if not visible_versions:
                return SearchResponse(query=request.query, hits=[])

        # 1. Query embedding（dense + sparse）
        query_vector = await self.query_embedding_provider.embed_query(request.query)

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

        # 4. Outline 召回：完全分离 pipeline，仅在 text 召回不足时补充
        # 暂时禁用 outline 召回，避免影响精确匹配
        # outline_points = await self.vector_store.search_document_outline(...)
        # 未来可考虑：只在 Recall < 阈值时才启用 outline 作为兜底

        all_candidates = text_fused[:request.top_k]  # 直接使用 RRF 结果，不融合 outline

        # 5. Rerank（可选）
        if self.rerank_provider and len(all_candidates) > request.top_k:
            reranked = await self._rerank_candidates(
                request.query,
                all_candidates,
                top_n=request.top_k,
            )
            final_points = reranked
        else:
            # 不 rerank 时直接截断
            final_points = all_candidates[: request.top_k]

        return SearchResponse(
            query=request.query,
            hits=[_to_search_hit(point) for point in final_points],
        )

    async def _rerank_candidates(
        self,
        query: str,
        candidates: list[models.ScoredPoint],
        *,
        top_n: int,
    ) -> list[models.ScoredPoint]:
        """使用 rerank API 重新排序候选。"""
        if not candidates or not self.rerank_provider:
            return candidates

        documents = [
            str((point.payload or {}).get("content", ""))
            for point in candidates
        ]

        try:
            ranked_indices = await self.rerank_provider.rerank(
                query,
                documents,
                top_n=top_n,
            )
            # 按 rerank 得分重新排列
            reranked = [
                models.ScoredPoint(
                    id=candidates[idx].id,
                    score=score,
                    payload=candidates[idx].payload,
                    version=candidates[idx].version,
                    vector=candidates[idx].vector,
                )
                for idx, score in ranked_indices
            ]
            return reranked
        except Exception:
            # Rerank 失败时降级到原始排序
            return candidates[:top_n]


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


def _to_sparse_vector(sparse_values: list[SparseValue]) -> models.SparseVector:
    """将 SparseValue 列表转换为 Qdrant SparseVector。"""
    if not sparse_values:
        return models.SparseVector(indices=[], values=[])
    sorted_values = sorted(sparse_values, key=lambda v: v.index)
    return models.SparseVector(
        indices=[v.index for v in sorted_values],
        values=[v.value for v in sorted_values],
    )
