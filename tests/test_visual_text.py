"""视觉文字提取、缓存和派生向量点的边界测试。"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ApiException

from app.clients.dashscope import DashScopeClient
from app.config import Settings
from app.rag.vision import VisualTextReader
from app.schemas.embedding import EmbeddingBatch, EmbeddingVector, SparseValue
from app.schemas.rag import (
    DocumentContent,
    DocumentSummary,
    SearchHit,
    SearchScope,
    SourceReference,
)
from app.schemas.vision import VisualExtraction
from app.stores.qdrant import QdrantStore


def _document(*, version: int = 1) -> DocumentSummary:
    return DocumentSummary(
        document_id="doc-1",
        course_id="CS5187",
        source_path="/course/Lecture2-1.pdf",
        version=version,
        indexed_at=datetime.now(UTC),
    )


def _hit(image_path: Path, *, version: int = 1) -> SearchHit:
    return SearchHit(
        chunk_id=f"chunk-v{version}",
        section_id="section-gaussian",
        representation_type="text_content",
        score=0.8,
        content="The 1D Gaussian and its derivatives",
        source=SourceReference(
            course_id="CS5187",
            document_id="doc-1",
            file="/course/Lecture2-1.pdf",
            section_path=["Gaussian"],
            page=16,
            image_path=image_path.as_posix(),
        ),
    )


class _Catalog:
    def __init__(self, image_path: Path) -> None:
        self.image_path = image_path
        self.version = 1
        self.read_scopes: list[tuple[list[str], list[str]]] = []

    def list_documents(self) -> list[DocumentSummary]:
        return [_document(version=self.version)]

    def read_document(
        self,
        document_id: str,
        version: int,
        *,
        course_ids: list[str],
        document_ids: list[str],
    ) -> DocumentContent:
        self.read_scopes.append((course_ids, document_ids))
        if course_ids and "CS5187" not in course_ids:
            raise ValueError("文档不在请求范围内")
        if document_ids and document_id not in document_ids:
            raise ValueError("文档不在请求范围内")
        if version != self.version:
            raise ValueError("文档版本不可见")
        document = _document(version=version)
        hit = _hit(self.image_path, version=version)
        return DocumentContent(
            document=document,
            title="Lecture2-1",
            outline=["Gaussian"],
            hits=[hit],
            pages=[16],
            image_only_pages=[16],
        )


class _Client:
    vision_model = "fake-vision"
    embedding_model = "fake-text"
    embedding_dimension = 3

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.visual_paths: list[Path] = []
        self.embedded_texts: list[str] = []

    async def extract_visual_text(self, image_path: Path) -> VisualExtraction:
        self.visual_paths.append(image_path)
        if self.fail:
            raise ValueError("invalid visual response")
        return VisualExtraction(
            text="Gaussian formula",
            formulas=["G(x) = exp(-x^2 / (2 sigma^2))"],
            symbols=["sigma: standard deviation"],
        )

    async def embed_documents(self, texts: list[str]) -> EmbeddingBatch:
        self.embedded_texts.extend(texts)
        return EmbeddingBatch(
            model=self.embedding_model,
            vectors=[
                EmbeddingVector(
                    index=0,
                    dense=[1.0, 0.0, 0.0],
                    sparse=[SparseValue(index=2, value=0.5)],
                )
            ],
        )


class _VectorStore:
    def __init__(self) -> None:
        self.upserts: list[tuple[DocumentSummary, SearchHit, str]] = []

    async def upsert_visual_text(
        self,
        document: DocumentSummary,
        hit: SearchHit,
        vector: EmbeddingVector,
        *,
        cache_key: str,
        embedding_model: str,
    ) -> None:
        assert vector.dense == [1.0, 0.0, 0.0]
        assert embedding_model == "fake-text"
        self.upserts.append((document, hit, cache_key))


@pytest.mark.asyncio
async def test_reader_reuses_cache_without_repeating_model_calls(tmp_path: Path) -> None:
    registered = tmp_path / "registered.png"
    registered.write_bytes(b"registered-image")
    catalog = _Catalog(registered)
    client = _Client()
    vectors = _VectorStore()
    reader = VisualTextReader(catalog, client, vectors, tmp_path / "cache")
    scope = SearchScope(course_ids=["CS5187"], document_ids=["doc-1"])

    first = await reader.read(_hit(registered), scope)
    second = await reader.read(_hit(registered), scope)

    assert client.visual_paths == [registered]
    assert len(client.embedded_texts) == 1
    assert first.content == second.content
    assert first.evidence_origin == second.evidence_origin == "visual_extract"
    assert len(vectors.upserts) == 2
    assert vectors.upserts[0][2] == vectors.upserts[1][2]
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


@pytest.mark.asyncio
async def test_reader_uses_separate_cache_when_document_version_changes(tmp_path: Path) -> None:
    registered = tmp_path / "registered.png"
    registered.write_bytes(b"same-image")
    catalog = _Catalog(registered)
    client = _Client()
    reader = VisualTextReader(catalog, client, _VectorStore(), tmp_path / "cache")
    scope = SearchScope(course_ids=["CS5187"])

    await reader.read(_hit(registered, version=1), scope)
    catalog.version = 2
    await reader.read(_hit(registered, version=2), scope)

    assert client.visual_paths == [registered, registered]
    assert len(client.embedded_texts) == 2
    assert len(list((tmp_path / "cache").glob("*.json"))) == 2


@pytest.mark.asyncio
async def test_reader_resolves_registered_path_and_enforces_scope(tmp_path: Path) -> None:
    registered = tmp_path / "registered.png"
    registered.write_bytes(b"registered-image")
    unregistered = tmp_path / "unregistered.png"
    unregistered.write_bytes(b"must-not-be-sent")
    catalog = _Catalog(registered)
    client = _Client()
    reader = VisualTextReader(catalog, client, _VectorStore(), tmp_path / "cache")

    await reader.read(
        _hit(unregistered),
        SearchScope(course_ids=["CS5187"], document_ids=["doc-1"]),
    )
    with pytest.raises(ValueError, match="范围内"):
        await reader.read(_hit(registered), SearchScope(course_ids=["CS0000"]))

    assert client.visual_paths == [registered]
    assert catalog.read_scopes[0] == (["CS5187"], ["doc-1"])


@pytest.mark.asyncio
async def test_reader_failure_does_not_write_cache(tmp_path: Path) -> None:
    registered = tmp_path / "registered.png"
    registered.write_bytes(b"registered-image")
    client = _Client(fail=True)
    vectors = _VectorStore()
    reader = VisualTextReader(_Catalog(registered), client, vectors, tmp_path / "cache")

    with pytest.raises(ValueError, match="invalid visual response"):
        await reader.read(_hit(registered), SearchScope(course_ids=["CS5187"]))

    assert not (tmp_path / "cache").exists()
    assert client.embedded_texts == []
    assert vectors.upserts == []


@pytest.mark.asyncio
async def test_cached_document_uses_only_existing_cache_and_does_not_publish(
    tmp_path: Path,
) -> None:
    registered = tmp_path / "registered.png"
    registered.write_bytes(b"registered-image")
    catalog = _Catalog(registered)
    client = _Client()
    vectors = _VectorStore()
    reader = VisualTextReader(catalog, client, vectors, tmp_path / "cache")
    scope = SearchScope(course_ids=["CS5187"])
    source = catalog.read_document("doc-1", 1, course_ids=["CS5187"], document_ids=[])
    await reader.read(source.hits[0], scope)
    api_counts = (len(client.visual_paths), len(client.embedded_texts), len(vectors.upserts))

    cached = await reader.cached_document(source, scope)

    assert (
        len(client.visual_paths),
        len(client.embedded_texts),
        len(vectors.upserts),
    ) == api_counts
    assert cached.hits[0].evidence_origin == "visual_extract"
    assert cached.image_only_pages == []


@pytest.mark.asyncio
async def test_cached_document_does_not_call_models_when_cache_is_missing(
    tmp_path: Path,
) -> None:
    registered = tmp_path / "registered.png"
    registered.write_bytes(b"registered-image")
    catalog = _Catalog(registered)
    client = _Client()
    vectors = _VectorStore()
    reader = VisualTextReader(catalog, client, vectors, tmp_path / "empty-cache")
    source = catalog.read_document("doc-1", 1, course_ids=["CS5187"], document_ids=[])

    unchanged = await reader.cached_document(source, SearchScope(course_ids=["CS5187"]))

    assert client.visual_paths == []
    assert client.embedded_texts == []
    assert vectors.upserts == []
    assert unchanged.hits == source.hits
    assert unchanged.image_only_pages == [16]


@pytest.mark.asyncio
async def test_dashscope_extract_visual_text_uses_native_payload(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/services/aigc/multimodal-generation/generation")
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "output": {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": [
                                    {
                                        "text": json.dumps(
                                            {
                                                "text": "Gaussian",
                                                "formulas": ["G(x) = exp(-x^2)"],
                                                "symbols": ["x"],
                                                "descriptions": [],
                                                "uncertain": [],
                                            }
                                        )
                                    }
                                ]
                            },
                        }
                    ]
                }
            },
        )

    image = tmp_path / "slide.png"
    image.write_bytes(b"png-content")
    client = DashScopeClient(
        Settings(dashscope_api_key="test-key", dashscope_max_retries=0),
        transport=httpx.MockTransport(handler),
    )

    result = await client.extract_visual_text(image)

    content = captured["input"]["messages"][0]["content"]
    assert content[0]["image"].startswith("data:image/png;base64,")
    assert "禁止根据常识补公式" in content[1]["text"]
    assert captured["parameters"] == {
        "enable_thinking": False,
        "temperature": 0,
        "max_tokens": 4096,
    }
    assert result.formulas == ["G(x) = exp(-x^2)"]


@pytest.mark.asyncio
async def test_dashscope_rejects_truncated_visual_response(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "output": {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": [{"text": '{"text":"partial"'}]},
                        }
                    ]
                }
            },
        )

    image = tmp_path / "slide.png"
    image.write_bytes(b"png-content")
    client = DashScopeClient(
        Settings(dashscope_api_key="test-key", dashscope_max_retries=0),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ValueError, match="截断"):
        await client.extract_visual_text(image)


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")
async def test_qdrant_visual_text_upsert_is_idempotent_and_keeps_origin() -> None:
    client = AsyncQdrantClient(":memory:")
    settings = Settings(
        qdrant_collection="visual_text_points",
        dashscope_embedding_dimension=3,
        dashscope_visual_embedding_dimension=2,
    )
    store = QdrantStore(settings, client=client)
    document = _document()
    hit = _hit(Path("/registered/slide-0016.png")).model_copy(
        update={"evidence_origin": "visual_extract"}
    )
    vector = EmbeddingVector(
        index=0,
        dense=[1.0, 0.0, 0.0],
        sparse=[SparseValue(index=7, value=0.6)],
    )

    await store.upsert_visual_text(
        document, hit, vector, cache_key="updated-key", embedding_model="fake-text"
    )
    # 旧缓存键产生的派生点清除，原始文字点保留。
    await client.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            models.PointStruct(
                id=1,
                vector={"text_dense": vector.dense},
                payload={
                    "document_id": document.document_id,
                    "version": 1,
                    "origin_chunk_id": hit.chunk_id,
                    "evidence_origin": "visual_extract",
                },
            ),
            models.PointStruct(
                id=2,
                vector={"text_dense": vector.dense},
                payload={
                    "document_id": document.document_id,
                    "version": 1,
                    "origin_chunk_id": hit.chunk_id,
                    "evidence_origin": "course_text",
                },
            ),
        ],
    )
    await store.upsert_visual_text(
        document, hit, vector, cache_key="stable-key", embedding_model="fake-text"
    )
    records, _ = await client.scroll(
        collection_name=settings.qdrant_collection,
        limit=10,
        with_payload=True,
        with_vectors=True,
    )

    assert len(records) == 2 and 1 not in {record.id for record in records}
    assert any(
        record.id == 2 and record.payload["evidence_origin"] == "course_text" for record in records
    )
    records = [
        record for record in records if record.payload["evidence_origin"] == "visual_extract"
    ]
    assert records[0].payload["representation_id"] == f"visual-text:doc-1:{hit.chunk_id}"
    assert records[0].payload["extraction_cache_key"] == "stable-key"
    assert records[0].payload["origin_chunk_id"] == hit.chunk_id
    assert records[0].payload["evidence_origin"] == "visual_extract"
    assert records[0].payload["image_path"] == "/registered/slide-0016.png"
    assert set(records[0].vector) == {"text_dense", "text_sparse"}
    await client.close()


@pytest.mark.asyncio
async def test_enrich_reports_index_failure_and_keeps_original_evidence(tmp_path: Path) -> None:
    image = tmp_path / "slide.png"
    image.write_bytes(b"png-content")
    catalog, client = _Catalog(image), _Client()

    class FailingStore:
        async def upsert_visual_text(self, *args, **kwargs):
            raise ApiException("index unavailable")

    reader = VisualTextReader(catalog, client, FailingStore(), tmp_path / "cache")
    original = _hit(image)
    hits, warnings = await reader.enrich(
        "高斯公式是什么", [original], SearchScope(course_ids=["CS5187"])
    )
    assert hits == [original]
    assert len(warnings) == 1 and "ApiException" in warnings[0]
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1
