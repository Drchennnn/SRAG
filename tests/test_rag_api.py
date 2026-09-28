"""RAG HTTP 接口测试。"""

import httpx

from app.api.rag import get_document_indexer, get_retriever
from app.main import create_app
from app.schemas.rag import (
    IndexStatus,
    JobState,
    SearchHit,
    SearchResponse,
    SourceReference,
)


class FakeIndexer:
    def __init__(self) -> None:
        self.requests = []

    async def index(self, request):
        self.requests.append(request)
        return IndexStatus(
            job_id="index-test",
            state=JobState.COMPLETED,
            indexed_files=1,
        )


class FakeRetriever:
    def __init__(self) -> None:
        self.requests = []

    async def search(self, request):
        self.requests.append(request)
        return SearchResponse(
            query=request.query,
            hits=[
                SearchHit(
                    chunk_id="chunk-1",
                    section_id="section-1",
                    representation_type="text_content",
                    score=0.9,
                    content="pandas 使用 read_csv 读取 CSV。",
                    source=SourceReference(
                        course_id="CS5489",
                        document_id="doc-1",
                        file="course/lecture.ipynb",
                        section_path=["pandas"],
                        cell_ids=["cell-1"],
                        cell_id="cell-1",
                    ),
                )
            ],
        )


async def test_index_endpoint_calls_document_indexer() -> None:
    app = create_app()
    indexer = FakeIndexer()
    app.dependency_overrides[get_document_indexer] = lambda: indexer

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/rag/index",
            json={
                "course_id": "CS5489",
                "paths": ["/tmp/lecture.ipynb"],
            },
        )

    assert response.status_code == 200
    assert response.json()["state"] == "completed"
    assert response.json()["indexed_files"] == 1
    assert len(indexer.requests) == 1
    assert indexer.requests[0].course_id == "CS5489"


async def test_search_endpoint_calls_retriever_with_scope() -> None:
    app = create_app()
    retriever = FakeRetriever()
    app.dependency_overrides[get_retriever] = lambda: retriever

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/rag/search",
            json={
                "query": "如何读取 CSV？",
                "scope": {"course_ids": ["CS5489"], "document_ids": ["doc-1"]},
                "top_k": 5,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "如何读取 CSV？"
    assert body["hits"][0]["source"]["course_id"] == "CS5489"
    assert body["hits"][0]["source"]["cell_ids"] == ["cell-1"]
    assert len(retriever.requests) == 1
    assert retriever.requests[0].top_k == 5
    assert retriever.requests[0].scope.document_ids == ["doc-1"]
