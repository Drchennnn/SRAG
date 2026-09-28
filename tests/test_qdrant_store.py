from __future__ import annotations

from datetime import UTC, datetime

import pytest
from qdrant_client import AsyncQdrantClient, models

from app.config import Settings
from app.schemas.document import CellGroupKind, NotebookChunk
from app.schemas.embedding import SparseValue
from app.schemas.indexing import (
    EmbeddedTextRepresentation,
    PreparedDocumentIndex,
    RepresentationType,
    TextRepresentation,
)
from app.stores.qdrant import (
    TEXT_DENSE_VECTOR,
    TEXT_SPARSE_VECTOR,
    VISUAL_DENSE_VECTOR,
    QdrantSchemaError,
    QdrantStore,
)


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")
async def test_ensure_collection_creates_named_vector_schema() -> None:
    client = AsyncQdrantClient(":memory:")
    settings = Settings(qdrant_collection="test_course_chunks")
    store = QdrantStore(settings, client=client)

    info = await store.ensure_collection()

    dense = info.config.params.vectors
    sparse = info.config.params.sparse_vectors
    assert isinstance(dense, dict)
    assert dense[TEXT_DENSE_VECTOR].size == settings.dashscope_embedding_dimension
    assert dense[TEXT_DENSE_VECTOR].distance == models.Distance.COSINE
    assert dense[VISUAL_DENSE_VECTOR].size == settings.dashscope_visual_embedding_dimension
    assert dense[VISUAL_DENSE_VECTOR].distance == models.Distance.COSINE
    assert sparse is not None
    assert TEXT_SPARSE_VECTOR in sparse
    await client.close()


@pytest.mark.asyncio
async def test_ensure_collection_rejects_incompatible_existing_schema() -> None:
    client = AsyncQdrantClient(":memory:")
    settings = Settings(qdrant_collection="wrong_course_chunks")
    await client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            TEXT_DENSE_VECTOR: models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
            ),
            VISUAL_DENSE_VECTOR: models.VectorParams(
                size=settings.dashscope_visual_embedding_dimension,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={TEXT_SPARSE_VECTOR: models.SparseVectorParams()},
    )
    store = QdrantStore(settings, client=client)

    with pytest.raises(QdrantSchemaError, match="维度应为"):
        await store.ensure_collection()

    await client.close()


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")
async def test_upsert_text_representations_is_traceable_and_idempotent() -> None:
    client = AsyncQdrantClient(":memory:")
    settings = Settings(
        qdrant_collection="upsert_course_chunks",
        dashscope_embedding_dimension=3,
        dashscope_visual_embedding_dimension=2,
    )
    store = QdrantStore(settings, client=client)
    chunk = NotebookChunk(
        chunk_id="chunk-1",
        section_id="section-1",
        document_id="doc-1",
        kind=CellGroupKind.EXECUTION,
        display_text="print(1)\n1",
        embedding_text="代码：print(1)\n运行输出：1",
        group_ids=["group-1"],
        block_ids=["block-1", "block-2"],
        cell_ids=["cell-1"],
        section_path=["Python", "输出"],
        asset_ids=[],
        token_count=8,
    )
    prepared = PreparedDocumentIndex(
        document_id="doc-1",
        source_path="course/lecture.ipynb",
        title="Lecture",
        groups=[],
        sections=[],
        chunks=[chunk],
        text_representations=[
            EmbeddedTextRepresentation(
                representation=TextRepresentation(
                    representation_id="representation-outline",
                    representation_type=RepresentationType.DOCUMENT_OUTLINE,
                    source_id="doc-1",
                    document_id="doc-1",
                    text="课程：Lecture",
                ),
                dense=[1.0, 0.0, 0.0],
                sparse=[SparseValue(index=2, value=0.2)],
            ),
            EmbeddedTextRepresentation(
                representation=TextRepresentation(
                    representation_id="representation-chunk",
                    representation_type=RepresentationType.TEXT_CONTENT,
                    source_id="chunk-1",
                    document_id="doc-1",
                    text=chunk.embedding_text,
                ),
                dense=[0.0, 1.0, 0.0],
                sparse=[SparseValue(index=4, value=0.8)],
            ),
        ],
        text_embedding_model="fake-text",
        text_embedding_dimension=3,
        visual_embedding_model="fake-visual",
        visual_embedding_dimension=2,
    )

    first_count = await store.upsert_text_representations(
        prepared,
        course_id="CS0001",
        release_at=datetime(2026, 9, 25, tzinfo=UTC),
    )
    second_count = await store.upsert_text_representations(
        prepared,
        course_id="CS0001",
        release_at=datetime(2026, 9, 25, tzinfo=UTC),
    )
    records, _ = await client.scroll(
        collection_name=settings.qdrant_collection,
        limit=10,
        with_payload=True,
        with_vectors=True,
    )

    assert first_count == second_count == 2
    assert len(records) == 2
    chunk_record = next(record for record in records if record.payload["chunk_id"] == "chunk-1")
    assert chunk_record.payload["course_id"] == "CS0001"
    assert chunk_record.payload["source_path"] == "course/lecture.ipynb"
    assert chunk_record.payload["cell_ids"] == ["cell-1"]
    assert chunk_record.payload["section_path"] == ["Python", "输出"]
    assert chunk_record.payload["release_at"] == "2026-09-25T00:00:00Z"
    assert set(chunk_record.vector) == {TEXT_DENSE_VECTOR, TEXT_SPARSE_VECTOR}

    hits = await store.search_text_dense(
        [1.0, 0.0, 0.0],
        limit=5,
        course_ids=["missing", "CS0001"],
        document_ids=["doc-1"],
    )
    assert len(hits) == 1
    assert hits[0].payload["representation_type"] == "text_content"
    assert hits[0].payload["chunk_id"] == "chunk-1"
    no_hits = await store.search_text_dense(
        [1.0, 0.0, 0.0],
        limit=5,
        course_ids=["CS0001"],
        document_ids=["missing"],
    )
    assert no_hits == []
    await client.close()
