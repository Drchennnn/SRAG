"""RAG HTTP 接口测试。"""

from datetime import UTC, datetime

import httpx

from app.api.rag import (
    get_document_indexer,
    get_metadata_store,
    get_qdrant_store,
    get_retriever,
)
from app.main import create_app
from app.schemas.rag import (
    DocumentSummary,
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


class FakeMetadataStore:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.documents = {
            "doc-1": DocumentSummary(
                document_id="doc-1",
                course_id="CS5489",
                source_path="course/lecture.ipynb",
                version=2,
                indexed_at=datetime(2026, 9, 29, tzinfo=UTC),
            )
        }

    def get_job(self, job_id: str):
        if job_id != "index-test":
            return None
        return IndexStatus(
            job_id=job_id,
            state=JobState.COMPLETED,
            indexed_files=1,
        )

    def list_documents(self):
        return list(self.documents.values())

    def has_document(self, document_id: str):
        return document_id in self.documents

    def delete_document(self, document_id: str):
        if document_id not in self.documents:
            return False
        del self.documents[document_id]
        self.deleted.append(document_id)
        return True

    def asset_paths(self, document_id: str):
        assert document_id in self.documents
        return ["data/assets/doc-1/figure.png"]

    def delete_managed_asset_files(self, paths):
        assert paths == ["data/assets/doc-1/figure.png"]
        return 1


class FakeQdrantStore:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def delete_document_versions(self, document_id: str, *, keep_version=None):
        assert keep_version is None
        self.deleted.append(document_id)


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
                "query_construction": "contextual",
                "history": [{"role": "user", "content": "pandas 在哪里介绍？"}],
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
    assert retriever.requests[0].query_construction == "contextual"
    assert retriever.requests[0].history[0].content == "pandas 在哪里介绍？"


async def test_document_and_job_management_endpoints() -> None:
    app = create_app()
    metadata = FakeMetadataStore()
    vector_store = FakeQdrantStore()
    app.dependency_overrides[get_metadata_store] = lambda: metadata
    app.dependency_overrides[get_qdrant_store] = lambda: vector_store

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        job_response = await client.get("/api/rag/index/index-test")
        missing_job = await client.get("/api/rag/index/missing")
        documents_response = await client.get("/api/rag/documents")
        delete_response = await client.delete("/api/rag/documents/doc-1")
        missing_document = await client.delete("/api/rag/documents/doc-1")

    assert job_response.status_code == 200
    assert job_response.json()["state"] == "completed"
    assert missing_job.status_code == 404
    assert documents_response.status_code == 200
    assert documents_response.json()[0]["version"] == 2
    assert delete_response.status_code == 200
    assert delete_response.json() == {"document_id": "doc-1", "deleted": True}
    assert missing_document.status_code == 404
    assert vector_store.deleted == ["doc-1"]
    assert metadata.deleted == ["doc-1"]
