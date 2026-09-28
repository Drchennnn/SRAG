"""原始文本 query 的 dense 召回测试。"""

from collections.abc import Sequence

from qdrant_client import models

from app.rag.retriever import HybridRetriever
from app.schemas.embedding import EmbeddingVector
from app.schemas.rag import SearchRequest, SearchScope


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


class FakeDenseVectorStore:
    def __init__(self) -> None:
        self.calls: list[tuple[list[float], int, list[str], list[str]]] = []

    async def search_text_dense(
        self,
        query_vector: list[float],
        *,
        limit: int,
        course_ids: Sequence[str] = (),
        document_ids: Sequence[str] = (),
    ) -> list[models.ScoredPoint]:
        self.calls.append((query_vector, limit, list(course_ids), list(document_ids)))
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


async def test_search_keeps_raw_query_and_returns_dense_hits() -> None:
    provider = FakeQueryEmbeddingProvider()
    store = FakeDenseVectorStore()
    retriever = HybridRetriever(provider, store)

    response = await retriever.search(SearchRequest(query="梯度下降是什么？", top_k=8))

    assert provider.queries == ["梯度下降是什么？"]
    assert store.calls == [([0.8, 0.2, 0.0], 8, [], [])]
    assert response.query == "梯度下降是什么？"
    assert len(response.hits) == 1
    assert response.hits[0].score == 0.91
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

    assert store.calls == [([0.8, 0.2, 0.0], 5, ["CS0001", "CS0002"], ["doc-1"])]
