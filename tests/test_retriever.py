"""文本混合检索和视觉路由测试。"""

from collections.abc import Sequence

import pytest
from qdrant_client import models

from app.rag.retriever import HybridRetriever
from app.schemas.embedding import (
    EmbeddingVector,
    MultimodalEmbeddingBatch,
    MultimodalEmbeddingVector,
)
from app.schemas.rag import ConversationTurn, SearchRequest, SearchScope
from app.stores.sqlite import VisibleVersion


class FakeQueryEmbeddingProvider:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def embed_query(
        self,
        text: str,
        *,
        instruct: str | None = None,
    ) -> EmbeddingVector:
        assert instruct is None
        self.queries.append(text)
        return EmbeddingVector(index=0, dense=[0.8, 0.2, 0.0])


class FakeVisualQueryEmbeddingProvider(FakeQueryEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.visual_queries: list[str] = []

    async def embed_visual_query(self, text: str) -> MultimodalEmbeddingBatch:
        self.visual_queries.append(text)
        return MultimodalEmbeddingBatch(
            model="test-visual",
            vectors=[MultimodalEmbeddingVector(index=0, dense=[1.0, 0.0], input_type="text")],
        )


class FailingVisualQueryEmbeddingProvider(FakeQueryEmbeddingProvider):
    async def embed_visual_query(self, text: str) -> MultimodalEmbeddingBatch:
        raise RuntimeError("visual API unavailable")


class FakeDenseVectorStore:
    def __init__(self) -> None:
        self.calls: list[
            tuple[list[float], int, list[str], list[str], list[tuple[str, int]] | None]
        ] = []

    async def search_text_dense(
        self,
        query_vector: list[float],
        *,
        limit: int,
        course_ids: Sequence[str] = (),
        document_ids: Sequence[str] = (),
        visible_versions: Sequence[tuple[str, int]] | None = None,
    ) -> list[models.ScoredPoint]:
        self.calls.append(
            (
                query_vector,
                limit,
                list(course_ids),
                list(document_ids),
                list(visible_versions) if visible_versions is not None else None,
            )
        )
        return [
            models.ScoredPoint(
                id=1,
                version=1,
                score=0.91,
                payload={
                    "chunk_id": "chunk-1",
                    "section_id": "section-1",
                    "representation_type": "text_content",
                    "content": "梯度下降沿负梯度方向更新参数。",
                    "course_id": "CS0001",
                    "document_id": "doc-1",
                    "source_path": "course/lecture.ipynb",
                    "section_path": ["优化", "梯度下降"],
                    "cell_ids": ["cell-4", "cell-5"],
                },
            )
        ]

    async def search_text_sparse(
        self,
        query_sparse: models.SparseVector,
        *,
        limit: int,
        course_ids: Sequence[str] = (),
        document_ids: Sequence[str] = (),
        visible_versions: Sequence[tuple[str, int]] | None = None,
    ) -> list[models.ScoredPoint]:
        return []


class FakeVisualVectorStore(FakeDenseVectorStore):
    def __init__(self) -> None:
        super().__init__()
        self.visual_calls: list[tuple[int, list[tuple[str, int]] | None]] = []

    async def search_visual_fusion(
        self, query_vector, *, limit, course_ids=(), document_ids=(), visible_versions=None
    ):
        self.visual_calls.append((limit, list(visible_versions) if visible_versions else None))
        return [
            models.ScoredPoint(
                id=2,
                version=1,
                score=0.92,
                payload={
                    "chunk_id": "slide-8",
                    "section_id": "section-slide",
                    "representation_type": "visual_fusion",
                    "content": "Cameras",
                    "course_id": "CS5187",
                    "document_id": "slide-doc",
                    "source_path": "/course/cameras.pdf",
                    "page": 8,
                    "image_path": "/assets/slide-0008.png",
                },
            )
        ]


class FakeVisibilityStore:
    def __init__(self, versions: list[VisibleVersion]) -> None:
        self.versions = versions
        self.calls: list[tuple[list[str], list[str]]] = []

    def visible_versions(self, *, course_ids=(), document_ids=()):
        self.calls.append((list(course_ids), list(document_ids)))
        return self.versions


async def test_search_keeps_raw_query_and_returns_dense_hits() -> None:
    provider = FakeQueryEmbeddingProvider()
    store = FakeDenseVectorStore()
    retriever = HybridRetriever(provider, store)

    response = await retriever.search(SearchRequest(query="梯度下降是什么？", top_k=8))

    assert provider.queries == ["梯度下降是什么？"]
    # 混合检索使用 recall_limit = min(top_k * 3, 50) = 24
    assert store.calls == [([0.8, 0.2, 0.0], 24, [], [], None)]
    assert response.query == "梯度下降是什么？"
    assert len(response.hits) == 1
    # RRF 融合后得分变为 RRF 得分而非原始 cosine 得分
    assert response.hits[0].score > 0
    assert response.hits[0].chunk_id == "chunk-1"
    assert response.hits[0].source.file == "course/lecture.ipynb"
    assert response.hits[0].source.course_id == "CS0001"
    assert response.hits[0].source.document_id == "doc-1"
    assert response.hits[0].source.section_path == ["优化", "梯度下降"]
    assert response.hits[0].source.cell_ids == ["cell-4", "cell-5"]
    assert response.hits[0].source.cell_id == "cell-4"


async def test_search_passes_course_and_document_scope() -> None:
    store = FakeDenseVectorStore()
    retriever = HybridRetriever(FakeQueryEmbeddingProvider(), store)

    await retriever.search(
        SearchRequest(
            query="梯度下降是什么？",
            scope=SearchScope(course_ids=["CS0001", "CS0002"], document_ids=["doc-1"]),
            top_k=5,
        )
    )

    # 混合检索使用 recall_limit = min(top_k * 3, 50) = 15
    assert store.calls == [([0.8, 0.2, 0.0], 15, ["CS0001", "CS0002"], ["doc-1"], None)]


async def test_search_filters_to_metadata_visible_versions() -> None:
    provider = FakeQueryEmbeddingProvider()
    store = FakeDenseVectorStore()
    visibility = FakeVisibilityStore([VisibleVersion(document_id="doc-1", version=2)])
    retriever = HybridRetriever(provider, store, visibility)

    await retriever.search(
        SearchRequest(
            query="梯度下降是什么？",
            scope=SearchScope(course_ids=["CS0001"], document_ids=["doc-1"]),
            top_k=3,
        )
    )

    assert visibility.calls == [(["CS0001"], ["doc-1"])]
    # 混合检索使用 recall_limit = min(top_k * 3, 50) = 9
    assert store.calls == [([0.8, 0.2, 0.0], 9, ["CS0001"], ["doc-1"], [("doc-1", 2)])]


async def test_search_skips_embedding_when_scope_has_no_visible_version() -> None:
    provider = FakeQueryEmbeddingProvider()
    store = FakeDenseVectorStore()
    retriever = HybridRetriever(provider, store, FakeVisibilityStore([]))

    response = await retriever.search(SearchRequest(query="尚未发布的内容", top_k=3))

    assert response.hits == []
    assert response.retrieval_query == "尚未发布的内容"
    assert response.query_method == "raw"
    assert provider.queries == []
    assert store.calls == []


async def test_auto_visual_route_returns_image_only_page_and_source() -> None:
    provider = FakeVisualQueryEmbeddingProvider()
    store = FakeVisualVectorStore()
    visible = FakeVisibilityStore([VisibleVersion(document_id="slide-doc", version=1)])
    retriever = HybridRetriever(provider, store, visible, visual_enabled=True)

    response = await retriever.search(
        SearchRequest(
            query="哪张图展示相机中心和成像平面？",
            top_k=2,
            scope=SearchScope(document_ids=["slide-doc"]),
        )
    )

    assert response.mode_used == "visual"
    assert response.hits[0].source.page == 8
    assert response.hits[0].source.image_path == "/assets/slide-0008.png"
    assert store.visual_calls == [(6, [("slide-doc", 1)])]
    assert provider.visual_queries == ["哪张图展示相机中心和成像平面？"]


async def test_explicit_text_mode_overrides_visual_intent() -> None:
    provider = FakeVisualQueryEmbeddingProvider()
    store = FakeVisualVectorStore()
    retriever = HybridRetriever(provider, store, visual_enabled=True)

    response = await retriever.search(
        SearchRequest(
            query="照片里的相机",
            retrieval_mode="text",
            top_k=2,
        )
    )

    assert response.mode_used == "text"
    assert store.visual_calls == []
    assert provider.visual_queries == []


async def test_auto_visual_failure_returns_text_results() -> None:
    retriever = HybridRetriever(
        FailingVisualQueryEmbeddingProvider(), FakeDenseVectorStore(), visual_enabled=True
    )

    response = await retriever.search(SearchRequest(query="图中展示什么？", top_k=2))

    assert response.mode_used == "text"
    assert response.hits[0].chunk_id == "chunk-1"


async def test_explicit_visual_failure_is_reported() -> None:
    retriever = HybridRetriever(
        FailingVisualQueryEmbeddingProvider(), FakeDenseVectorStore(), visual_enabled=True
    )

    with pytest.raises(RuntimeError, match="visual API unavailable"):
        await retriever.search(
            SearchRequest(query="图中展示什么？", retrieval_mode="visual", top_k=2)
        )


async def test_contextual_followup_uses_constructed_query_and_preserves_scope() -> None:
    provider = FakeVisualQueryEmbeddingProvider()
    store = FakeVisualVectorStore()
    retriever = HybridRetriever(provider, store, visual_enabled=True)

    response = await retriever.search(
        SearchRequest(
            query="它在哪页？",
            query_construction="contextual",
            history=[ConversationTurn(role="user", content="相机成像示意图是什么？")],
            scope=SearchScope(document_ids=["slide-doc"]),
            top_k=2,
        )
    )

    expected = "相机成像示意图是什么？\n追问：它在哪页？"
    assert provider.queries == [expected]
    assert provider.visual_queries == [expected]
    assert store.calls[0][3] == ["slide-doc"]
    assert response.query == "它在哪页？"
    assert response.retrieval_query == expected
    assert response.query_method == "history_expanded"
    assert response.mode_used == "visual"
