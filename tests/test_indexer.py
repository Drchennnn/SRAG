"""Notebook 多表示索引准备测试。"""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.config import Settings
from app.rag.chunker import NotebookChunker
from app.rag.indexer import DocumentIndexer
from app.schemas.document import AssetRef, BlockKind, ContentBlock, ParsedDocument
from app.schemas.embedding import (
    EmbeddingBatch,
    EmbeddingVector,
    MultimodalEmbeddingBatch,
    MultimodalEmbeddingVector,
    SparseValue,
)
from app.schemas.indexing import PreparedDocumentIndex, RepresentationType
from app.schemas.rag import IndexRequest, JobState
from app.stores.sqlite import MetadataStore, VisibleVersion


class FakeEmbeddingProvider:
    embedding_model = "fake-text-embedding"
    embedding_dimension = 3
    visual_embedding_model = "fake-visual-embedding"
    visual_embedding_dimension = 2

    def __init__(self) -> None:
        self.document_calls = 0

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        self.document_calls += 1
        return EmbeddingBatch(
            model=self.embedding_model,
            vectors=[
                EmbeddingVector(
                    index=index,
                    dense=[float(index), 0.5, 1.0],
                    sparse=[SparseValue(index=100 + index, value=0.8, token="测试")],
                )
                for index, _ in enumerate(texts)
            ],
            total_tokens=sum(len(text) for text in texts),
            request_ids=["text-request"],
        )

    async def embed_visual_fusion(
        self,
        *,
        text: str,
        image_path: Path,
        media_type: str,
    ) -> MultimodalEmbeddingBatch:
        assert text
        assert image_path.is_file()
        assert media_type == "image/png"
        return MultimodalEmbeddingBatch(
            model=self.visual_embedding_model,
            vectors=[MultimodalEmbeddingVector(index=0, dense=[0.25, 0.75], input_type="fusion")],
            total_tokens=12,
            request_ids=["visual-request"],
        )


class FakeParser:
    def __init__(self, document: ParsedDocument) -> None:
        self.document = document
        self.paths: list[Path] = []

    def parse(self, path: Path, document_id: str | None = None) -> ParsedDocument:
        assert document_id is None
        self.paths.append(path)
        return self.document.model_copy(update={"source_path": path.as_posix()})


class FakeTextStore:
    def __init__(self) -> None:
        self.calls: list[tuple[PreparedDocumentIndex, str, int, datetime | None]] = []
        self.stale_calls: list[tuple[str, int]] = []
        self.delete_version_calls: list[tuple[str, int | None]] = []

    async def upsert_text_representations(
        self,
        prepared: PreparedDocumentIndex,
        *,
        course_id: str,
        version: int = 1,
        release_at: datetime | None = None,
    ) -> int:
        self.calls.append((prepared, course_id, version, release_at))
        return len(prepared.text_representations)

    async def delete_stale_text_points(
        self,
        prepared: PreparedDocumentIndex,
        *,
        version: int,
    ) -> int:
        self.stale_calls.append((prepared.document_id, version))
        return 0

    async def delete_document_versions(
        self,
        document_id: str,
        *,
        keep_version: int | None = None,
    ) -> None:
        self.delete_version_calls.append((document_id, keep_version))


@pytest.mark.asyncio
async def test_prepare_document_builds_text_and_visual_representations(tmp_path: Path) -> None:
    image_path = tmp_path / "curve.png"
    image_path.write_bytes(b"png")
    document = ParsedDocument(
        document_id="doc-index",
        source_path="course/index.ipynb",
        title="优化方法",
        blocks=[
            ContentBlock(
                block_id="cell-1",
                document_id="doc-index",
                kind=BlockKind.MARKDOWN,
                text="# 学习曲线\n训练损失随迭代轮数下降。",
                section_path=["学习曲线"],
                heading_level=1,
                cell_id="cell-1",
                assets=[
                    AssetRef(
                        asset_id="asset-curve",
                        media_type="image/png",
                        path=image_path.as_posix(),
                        caption="学习曲线",
                    )
                ],
            )
        ],
    )
    chunker = NotebookChunker(
        soft_min_tokens=10,
        target_tokens=40,
        max_tokens=80,
        overlap_tokens=5,
    )
    provider = FakeEmbeddingProvider()
    indexer = DocumentIndexer(chunker, provider, provider)

    prepared = await indexer.prepare_document(document)

    assert prepared.groups
    assert prepared.sections
    assert prepared.chunks
    assert prepared.source_path == "course/index.ipynb"
    assert prepared.title == "优化方法"
    assert prepared.text_embedding_model == "fake-text-embedding"
    assert prepared.visual_embedding_model == "fake-visual-embedding"
    assert prepared.request_ids == ["text-request", "visual-request"]
    assert prepared.text_representations[0].representation.representation_type is (
        RepresentationType.DOCUMENT_OUTLINE
    )
    assert len(prepared.text_representations) == len(prepared.chunks) + 1
    assert len(prepared.visual_representations) == 1
    assert prepared.visual_representations[0].representation.source_id == "asset-curve"
    assert prepared.visual_representations[0].dense == [0.25, 0.75]


@pytest.mark.asyncio
async def test_index_runs_notebook_to_text_store_flow(tmp_path: Path) -> None:
    path = tmp_path / "lecture.ipynb"
    document = ParsedDocument(
        document_id="doc-flow",
        source_path=path.as_posix(),
        title="线性回归",
        blocks=[
            ContentBlock(
                block_id="block-flow",
                document_id="doc-flow",
                kind=BlockKind.MARKDOWN,
                text="# 线性回归\n最小化平方误差。",
                section_path=["线性回归"],
                heading_level=1,
                cell_id="cell-flow",
            )
        ],
    )
    parser = FakeParser(document)
    store = FakeTextStore()
    provider = FakeEmbeddingProvider()
    indexer = DocumentIndexer(
        NotebookChunker(soft_min_tokens=10, target_tokens=40, max_tokens=80),
        provider,
        provider,
        parser=parser,
        text_store=store,
    )
    release_at = datetime(2026, 10, 1, tzinfo=UTC)

    status = await indexer.index(
        IndexRequest(course_id="CS0001", paths=[path], release_at=release_at)
    )

    assert status.state is JobState.COMPLETED
    assert status.indexed_files == 1
    assert status.failed_files == 0
    assert status.errors == {}
    assert status.job_id.startswith("index_")
    assert parser.paths == [path.resolve()]
    assert len(store.calls) == 1
    prepared, course_id, version, stored_release_at = store.calls[0]
    assert course_id == "CS0001"
    assert version == 1
    assert stored_release_at == release_at
    assert prepared.document_id == "doc-flow"
    assert prepared.visual_representations == []
    assert len(prepared.text_representations) == len(prepared.chunks) + 1


@pytest.mark.asyncio
async def test_index_uses_content_hash_and_switches_active_version(tmp_path: Path) -> None:
    path = tmp_path / "lecture.ipynb"
    path.write_text("version one", encoding="utf-8")
    document = ParsedDocument(
        document_id="doc-lifecycle",
        source_path=path.as_posix(),
        title="版本课程",
        blocks=[
            ContentBlock(
                block_id="block-lifecycle",
                document_id="doc-lifecycle",
                kind=BlockKind.MARKDOWN,
                text="# 版本课程\n第一版内容。",
                section_path=["版本课程"],
                heading_level=1,
                cell_id="cell-lifecycle",
            )
        ],
    )
    parser = FakeParser(document)
    vector_store = FakeTextStore()
    provider = FakeEmbeddingProvider()
    metadata_store = MetadataStore(Settings(sqlite_path=tmp_path / "metadata.sqlite3"))
    metadata_store.initialize()
    indexer = DocumentIndexer(
        NotebookChunker(soft_min_tokens=10, target_tokens=40, max_tokens=80),
        provider,
        provider,
        parser=parser,
        text_store=vector_store,
        metadata_store=metadata_store,
    )
    request = IndexRequest(course_id="CS0001", paths=[path])

    first = await indexer.index(request)
    second = await indexer.index(request)
    path.write_text("version two", encoding="utf-8")
    third = await indexer.index(request)

    assert first.state is JobState.COMPLETED
    assert first.indexed_files == 1
    assert second.state is JobState.COMPLETED
    assert second.indexed_files == 0
    assert second.skipped_files == 1
    assert third.indexed_files == 1
    assert provider.document_calls == 2
    assert [call[2] for call in vector_store.calls] == [1, 2]
    assert vector_store.stale_calls == [("doc-lifecycle", 1), ("doc-lifecycle", 2)]
    assert vector_store.delete_version_calls == [
        ("doc-lifecycle", 1),
        ("doc-lifecycle", 2),
    ]
    assert metadata_store.visible_versions() == [
        VisibleVersion(document_id="doc-lifecycle", version=2)
    ]
    assert metadata_store.get_job(second.job_id) == second
