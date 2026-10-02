"""从 SQLite 按当前可见版本完整读取文档结构。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.config import Settings
from app.rag.chunker import NotebookChunker
from app.schemas.document import (
    BlockKind,
    ContentBlock,
    ParsedDocument,
    Slide,
    SlideChunk,
    SlideChunkingResult,
    SlideDeck,
    SlideSection,
    SlideSourceFormat,
)
from app.schemas.indexing import PreparedDocumentIndex
from app.stores.sqlite import MetadataStore


def _prepared(document: ParsedDocument) -> PreparedDocumentIndex:
    structure = NotebookChunker(
        soft_min_tokens=8,
        target_tokens=16,
        max_tokens=30,
        overlap_tokens=0,
    ).chunk(document)
    return PreparedDocumentIndex(
        document_id=document.document_id,
        source_path=document.source_path,
        title=document.title,
        groups=structure.groups,
        sections=structure.sections,
        chunks=structure.chunks,
        text_representations=[],
        text_embedding_model="fake-text",
        text_embedding_dimension=3,
        visual_embedding_model="fake-visual",
        visual_embedding_dimension=2,
    )


def test_read_notebook_returns_all_chunks_in_cell_order(tmp_path: Path) -> None:
    source = tmp_path / "lesson.ipynb"
    source.write_text("{}", encoding="utf-8")
    document = ParsedDocument(
        document_id="notebook-1",
        source_path=source.as_posix(),
        title="Notebook Lesson",
        blocks=[
            ContentBlock(
                block_id="block-1",
                document_id="notebook-1",
                kind=BlockKind.MARKDOWN,
                text="# 第一节\nFIRST_UNIQUE_CONTENT",
                section_path=["第一节"],
                heading_level=1,
                cell_id="cell-1",
                cell_index=0,
            ),
            ContentBlock(
                block_id="block-2",
                document_id="notebook-1",
                kind=BlockKind.CODE,
                text="SECOND_UNIQUE_CONTENT = 2",
                section_path=["第二节"],
                cell_id="cell-2",
                cell_index=1,
            ),
            ContentBlock(
                block_id="block-3",
                document_id="notebook-1",
                kind=BlockKind.OUTPUT,
                text="THIRD_UNIQUE_CONTENT",
                section_path=["第二节"],
                cell_id="cell-2",
                cell_index=1,
                related_block_id="block-2",
            ),
        ],
    )
    store = MetadataStore(Settings(sqlite_path=tmp_path / "metadata.sqlite3"))
    store.initialize()
    plan = store.reserve_version(
        document,
        course_id="CS0001",
        content_hash="hash-1",
        release_at=datetime.now(UTC) - timedelta(minutes=1),
    )
    store.save_structure(document, _prepared(document), version=plan.version)
    store.activate_version(document.document_id, plan.version)

    result = store.read_document(
        document.document_id,
        plan.version,
        course_ids=["CS0001"],
        document_ids=[document.document_id],
    )

    combined = "\n".join(hit.content for hit in result.hits)
    assert "FIRST_UNIQUE_CONTENT" in combined
    assert "SECOND_UNIQUE_CONTENT" in combined
    assert "THIRD_UNIQUE_CONTENT" in combined
    assert combined.index("FIRST_UNIQUE_CONTENT") < combined.index("SECOND_UNIQUE_CONTENT")
    assert result.pages == []
    assert result.image_only_pages == []
    assert result.outline == ["第一节", "第二节"]


def test_read_long_notebook_cell_follows_chunk_links(tmp_path: Path) -> None:
    source = tmp_path / "long-cell.ipynb"
    source.write_text("{}", encoding="utf-8")
    document = ParsedDocument(
        document_id="notebook-long",
        source_path=source.as_posix(),
        title="Long Cell",
        blocks=[
            ContentBlock(
                block_id="block-long",
                document_id="notebook-long",
                kind=BlockKind.MARKDOWN,
                text="# Long\n" + "SENTENCE. " * 30,
                section_path=["Long"],
                heading_level=1,
                cell_id="cell-long",
                cell_index=0,
            )
        ],
    )
    prepared = _prepared(document)
    assert len(prepared.chunks) > 2
    # 哈希 ID 的字典序与链顺序不同，防止测试只因巧合通过。
    assert [chunk.chunk_id for chunk in prepared.chunks] != sorted(
        chunk.chunk_id for chunk in prepared.chunks
    )
    store = MetadataStore(Settings(sqlite_path=tmp_path / "metadata.sqlite3"))
    store.initialize()
    plan = store.reserve_version(
        document,
        course_id="CS0001",
        content_hash="hash-long",
        release_at=None,
    )
    store.save_structure(document, prepared, version=plan.version)
    store.activate_version(document.document_id, plan.version)

    result = store.read_document(
        document.document_id,
        plan.version,
        course_ids=["CS0001"],
        document_ids=[document.document_id],
    )

    assert [hit.chunk_id for hit in result.hits] == [chunk.chunk_id for chunk in prepared.chunks]


def test_read_slides_covers_every_page_and_marks_image_only_pages(tmp_path: Path) -> None:
    source = tmp_path / "Lecture1.pdf"
    source.write_bytes(b"pdf")
    page_1 = tmp_path / "slide-0001.png"
    page_2 = tmp_path / "slide-0002.png"
    page_1.write_bytes(b"one")
    page_2.write_bytes(b"two")
    deck = SlideDeck(
        document_id="slides-1",
        source_path=source.as_posix(),
        source_format=SlideSourceFormat.PDF,
        title="Slide Lesson",
        slides=[
            Slide(
                slide_id="slide-1",
                document_id="slides-1",
                slide_no=1,
                title="Gaussian",
                page_image_path="slides/slide-0001.png",
            ),
            Slide(
                slide_id="slide-2",
                document_id="slides-1",
                slide_no=2,
                page_image_path="slides/slide-0002.png",
            ),
        ],
    )
    chunks = SlideChunkingResult(
        document_id="slides-1",
        outline_text="Gaussian",
        sections=[
            SlideSection(
                section_id="section-1",
                document_id="slides-1",
                title="Gaussian",
                slide_ids=["slide-1", "slide-2"],
                chunk_ids=["chunk-1", "chunk-2"],
            )
        ],
        chunks=[
            SlideChunk(
                chunk_id="chunk-1",
                section_id="section-1",
                document_id="slides-1",
                slide_id="slide-1",
                slide_no=1,
                title="Gaussian",
                display_text="Gaussian formula",
                embedding_text="Gaussian formula",
                token_count=2,
            ),
            SlideChunk(
                chunk_id="chunk-2",
                section_id="section-1",
                document_id="slides-1",
                slide_id="slide-2",
                slide_no=2,
                display_text="",
                embedding_text="",
                token_count=0,
            ),
        ],
    )
    store = MetadataStore(Settings(sqlite_path=tmp_path / "metadata.sqlite3"))
    store.initialize()
    plan = store.reserve_version(
        deck,
        course_id="CS5187",
        content_hash="slide-hash",
        release_at=None,
        document_type="pdf",
    )
    store.save_slide_structure(
        deck,
        chunks,
        version=plan.version,
        asset_paths={
            "slides/slide-0001.png": page_1.as_posix(),
            "slides/slide-0002.png": page_2.as_posix(),
        },
    )
    store.activate_version(deck.document_id, plan.version)

    result = store.read_document(
        deck.document_id,
        plan.version,
        course_ids=["CS5187"],
        document_ids=[],
    )

    assert result.pages == [1, 2]
    assert [hit.source.page for hit in result.hits] == [1, 2]
    assert [hit.source.image_path for hit in result.hits] == [
        page_1.as_posix(),
        page_2.as_posix(),
    ]
    assert result.image_only_pages == [2]
    assert result.hits[1].content == "Gaussian"


def test_read_document_enforces_scope_release_and_active_version(tmp_path: Path) -> None:
    source = tmp_path / "future.ipynb"
    source.write_text("{}", encoding="utf-8")
    document = ParsedDocument(
        document_id="future-1",
        source_path=source.as_posix(),
        title="Future",
        blocks=[
            ContentBlock(
                block_id="block-1",
                document_id="future-1",
                kind=BlockKind.TEXT,
                text="future content",
                cell_id="cell-1",
                cell_index=0,
            )
        ],
    )
    store = MetadataStore(Settings(sqlite_path=tmp_path / "metadata.sqlite3"))
    store.initialize()
    future = store.reserve_version(
        document,
        course_id="CS0002",
        content_hash="future-1",
        release_at=datetime.now(UTC) + timedelta(days=1),
    )
    store.save_structure(document, _prepared(document), version=future.version)
    store.activate_version(document.document_id, future.version)

    with pytest.raises(ValueError, match="不可见"):
        store.read_document("future-1", 1, course_ids=["CS0002"], document_ids=[])

    released = store.reserve_version(
        document,
        course_id="CS0002",
        content_hash="released-2",
        release_at=datetime.now(UTC) - timedelta(minutes=1),
        force=True,
    )
    store.save_structure(document, _prepared(document), version=released.version)
    store.activate_version(document.document_id, released.version)

    with pytest.raises(ValueError, match="不可见"):
        store.read_document("future-1", 1, course_ids=["CS0002"], document_ids=[])
    with pytest.raises(ValueError, match="当前范围"):
        store.read_document("future-1", 2, course_ids=["OTHER"], document_ids=[])
    with pytest.raises(ValueError, match="当前范围"):
        store.read_document("future-1", 2, course_ids=[], document_ids=["other-doc"])
