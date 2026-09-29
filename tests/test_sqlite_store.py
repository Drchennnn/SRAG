"""SQLite 文档生命周期存储测试。"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.config import Settings
from app.rag.chunker import NotebookChunker
from app.schemas.document import AssetRef, BlockKind, ContentBlock, ParsedDocument
from app.schemas.indexing import PreparedDocumentIndex
from app.schemas.rag import IndexStatus, JobState
from app.stores.sqlite import MetadataStore


def _document(path: Path, document_id: str = "doc-1") -> ParsedDocument:
    asset_path = path.parent / "figure.png"
    asset_path.write_bytes(b"png")
    return ParsedDocument(
        document_id=document_id,
        source_path=path.resolve().as_posix(),
        title="概率课程",
        blocks=[
            ContentBlock(
                block_id=f"{document_id}-block",
                document_id=document_id,
                kind=BlockKind.MARKDOWN,
                text="# 贝叶斯公式\n条件概率与边缘概率。",
                section_path=["贝叶斯公式"],
                heading_level=1,
                cell_id=f"{document_id}-cell",
                cell_index=0,
                assets=[
                    AssetRef(
                        asset_id=f"{document_id}-asset",
                        media_type="image/png",
                        path=asset_path.as_posix(),
                        caption="概率图",
                    )
                ],
            )
        ],
    )


def _prepared(document: ParsedDocument) -> PreparedDocumentIndex:
    structure = NotebookChunker(
        soft_min_tokens=10,
        target_tokens=40,
        max_tokens=80,
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


def test_metadata_store_persists_structure_and_switches_versions(tmp_path: Path) -> None:
    settings = Settings(sqlite_path=tmp_path / "metadata.sqlite3")
    store = MetadataStore(settings)
    store.initialize()
    store.initialize()
    path = tmp_path / "lecture.ipynb"
    path.write_text("{}", encoding="utf-8")
    document = _document(path)
    prepared = _prepared(document)
    released = datetime.now(UTC) - timedelta(days=1)

    first = store.reserve_version(
        document,
        course_id="CS0001",
        content_hash="hash-1",
        release_at=released,
    )
    assert first.version == 1
    assert first.should_index is True

    store.save_structure(document, prepared, version=first.version)
    store.activate_version(document.document_id, first.version)

    visible = store.visible_versions(course_ids=["CS0001"])
    assert [(item.document_id, item.version) for item in visible] == [("doc-1", 1)]
    summaries = store.list_documents()
    assert len(summaries) == 1
    assert summaries[0].version == 1
    assert summaries[0].release_at == released

    unchanged = store.reserve_version(
        document,
        course_id="CS0001",
        content_hash="hash-1",
        release_at=released,
    )
    assert unchanged.should_index is False
    assert unchanged.version == 1

    failed = store.reserve_version(
        document,
        course_id="CS0001",
        content_hash="hash-2",
        release_at=released,
    )
    assert failed.version == 2
    store.fail_version(document.document_id, failed.version, "embedding error")
    assert store.visible_versions() == visible

    third = store.reserve_version(
        document,
        course_id="CS0001",
        content_hash="hash-3",
        release_at=released,
    )
    assert third.version == 3
    store.save_structure(document, prepared, version=third.version)
    store.activate_version(document.document_id, third.version)
    assert store.visible_versions() == [
        type(visible[0])(document_id="doc-1", version=3)
    ]

    with store._connect() as connection:
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("content_blocks", "assets", "cell_groups", "sections", "chunks")
        }
        states = connection.execute(
            "SELECT version, state FROM document_versions ORDER BY version"
        ).fetchall()
        migration = connection.execute(
            "SELECT version FROM schema_migrations"
        ).fetchone()[0]

    assert migration == 1
    assert counts["content_blocks"] == 2
    assert counts["assets"] == 2
    assert counts["cell_groups"] == len(prepared.groups) * 2
    assert counts["sections"] == len(prepared.sections) * 2
    assert counts["chunks"] == len(prepared.chunks) * 2
    assert [(row["version"], row["state"]) for row in states] == [
        (1, "superseded"),
        (2, "failed"),
        (3, "active"),
    ]


def test_metadata_store_hides_future_release_and_persists_jobs(tmp_path: Path) -> None:
    store = MetadataStore(Settings(sqlite_path=tmp_path / "metadata.sqlite3"))
    store.initialize()
    path = tmp_path / "future.ipynb"
    path.write_text("{}", encoding="utf-8")
    document = _document(path, document_id="doc-future")
    future = datetime.now(UTC) + timedelta(days=7)

    plan = store.reserve_version(
        document,
        course_id="CS0002",
        content_hash="future-hash",
        release_at=future,
    )
    store.save_structure(document, _prepared(document), version=plan.version)
    store.activate_version(document.document_id, plan.version)

    assert store.visible_versions() == []
    assert store.list_documents() == []

    status = IndexStatus(
        job_id="index-job",
        state=JobState.COMPLETED,
        indexed_files=1,
        skipped_files=2,
    )
    store.create_job(status.job_id)
    store.finish_job(status)
    assert store.get_job(status.job_id) == status
    assert store.get_job("missing") is None

    assert store.has_document(document.document_id) is True
    assert store.delete_document(document.document_id) is True
    assert store.has_document(document.document_id) is False
    assert store.delete_document(document.document_id) is False


def test_metadata_store_only_deletes_managed_asset_files(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    store = MetadataStore(
        Settings(sqlite_path=data_dir / "metadata.sqlite3", data_dir=data_dir)
    )
    managed = data_dir / "assets" / "doc-1" / "figure.png"
    managed.parent.mkdir(parents=True)
    managed.write_bytes(b"managed")
    external = tmp_path / "course" / "original.png"
    external.parent.mkdir(parents=True)
    external.write_bytes(b"source")

    deleted = store.delete_managed_asset_files(
        [managed.as_posix(), external.as_posix()]
    )

    assert deleted == 1
    assert managed.exists() is False
    assert external.read_bytes() == b"source"
