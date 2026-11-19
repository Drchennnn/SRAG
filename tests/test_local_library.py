"""End-to-end local publication and independently readable historical references."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.config import Settings
from app.rag.chunker import NotebookChunker
from app.rag.local_library import LocalLibrary
from app.rag.parsers.local_notebook import parse_local_notebook
from app.rag.parsers.notebook import NotebookParser
from app.rag.parsers.pptx import PPTXParser
from app.rag.slide_chunker import SlideChunker
from app.schemas.indexing import PreparedDocumentIndex
from app.schemas.local import SourceKey
from app.schemas.rag import SearchRequest, SearchScope
from app.stores import local_library as storage
from app.stores.local_library import PublicationConflict
from app.stores.sqlite import MetadataStore

SAMPLES = Path(__file__).resolve().parents[1] / "docs" / "A" / "样例"


@pytest.fixture
def library(tmp_path: Path) -> LocalLibrary:
    course = tmp_path / "course"
    course.mkdir()
    return LocalLibrary(
        Settings(
            _env_file=None,
            sqlite_path=tmp_path / "library.sqlite3",
            data_dir=tmp_path / "data",
            course_root=course,
            dashscope_api_key="",
            deepseek_api_key="",
            qdrant_url="http://127.0.0.1:1",
        )
    )


def _note(path: Path, token: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 4,
                "metadata": {},
                "cells": [{"cell_type": "markdown", "source": f"# {token}", "metadata": {}}],
            }
        )
    )
    return path


def _hits(library: LocalLibrary, query: str, course: str = "demo"):
    return library.search(SearchRequest(query=query, scope=SearchScope(course_ids=[course])))


def _read(library: LocalLibrary, document_id: str, version: int, source: str = "cell:0"):
    return library.store.read_source(
        SourceKey(document_id=document_id, version=version, source_id=source), course_ids=["demo"]
    )


def test_local_import_reuses_bytes_and_keeps_history_across_switch(library: LocalLibrary) -> None:
    path = _note(library.course_root / "lesson.ipynb", "beforeword")
    first_bytes = path.read_bytes()
    first = library.import_paths("demo", [path])[0]
    again = library.import_paths("demo", [path])[0]
    assert first.status == "published"
    assert again.status == "skipped"
    assert (again.document_id, again.version, again.publication_epoch) == (
        first.document_id,
        first.version,
        first.publication_epoch,
    )
    assert _hits(library, "beforeword")[0].version == first.version == 1
    assert _read(library, first.document_id, 1).original == first_bytes

    _note(path, "afterword")
    second = library.import_paths("demo", [path])[0]
    assert second.version == 2
    assert second.document_id == first.document_id
    assert second.publication_epoch > first.publication_epoch
    assert not _hits(library, "beforeword")
    assert _hits(library, "afterword")[0].version == 2
    assert _read(library, first.document_id, 1).original == first_bytes
    assert _read(library, first.document_id, 2).original == path.read_bytes()
    denied = library.store.read_source(
        SourceKey(document_id=first.document_id, version=1, source_id="cell:0"),
        course_ids=["another"],
    )
    assert denied.status == "not_accessible" and denied.source is None

    path.write_bytes(first_bytes)
    revived = library.import_paths("demo", [path])[0]
    assert revived.version == 1
    assert revived.publication_epoch > second.publication_epoch
    assert _hits(library, "beforeword")[0].version == 1
    assert not _hits(library, "afterword")
    assert _read(library, second.document_id, 2).status == "available"


def test_interrupted_transaction_exposes_neither_sources_nor_fts(
    library: LocalLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _note(library.course_root / "lesson.ipynb", "committedword")
    first = library.import_paths("demo", [path])[0]
    old_bytes = path.read_bytes()
    _note(path, "uncommittedword")
    write_fts = storage._save_fts

    def interrupt_after_fts(*args):
        write_fts(*args)
        assert not _hits(library, "uncommittedword")
        assert _hits(library, "committedword")[0].version == 1
        raise RuntimeError("interrupted after FTS insert")

    monkeypatch.setattr(storage, "_save_fts", interrupt_after_fts)
    failed = library.import_paths("demo", [path])[0]
    assert failed.status == "failed" and "interrupted" in failed.error
    assert not _hits(library, "uncommittedword")
    assert _read(library, first.document_id, 1).original == old_bytes
    assert _read(library, first.document_id, 2).status == "not_accessible"
    with sqlite3.connect(library.settings.sqlite_path) as db:
        assert db.execute("SELECT version FROM local_versions").fetchall() == [(1,)]
        assert db.execute("SELECT version FROM local_sources").fetchall() == [(1,)]
        assert db.execute("SELECT body FROM local_fts").fetchall() == [("# committedword",)]

    monkeypatch.setattr(storage, "_save_fts", write_fts)
    assert library.import_paths("demo", [path])[0].version == 2
    assert not _hits(library, "committedword")
    assert _hits(library, "uncommittedword")[0].version == 2
    assert _read(library, first.document_id, 1).original == old_bytes


def test_enhancement_cannot_publish_after_replacement_or_delete(
    library: LocalLibrary,
) -> None:
    path = _note(library.course_root / "lesson.ipynb", "nativeword")
    first = library.import_paths("demo", [path])[0]
    key = SourceKey(document_id=first.document_id, version=1, source_id="cell:0")
    original_ticket = dict(
        course_ids=["demo"],
        content_sha256=first.content_sha256,
        publication_epoch=first.publication_epoch,
    )
    assert library.store.write_enhancement(key, **original_ticket, text="extraone")
    assert _hits(library, "extraone")[0].origin == "enhancement"
    assert library.store.write_enhancement(key, **original_ticket, text="extratwo")
    assert not _hits(library, "extraone")
    assert _hits(library, "extratwo")[0].version == 1
    assert not library.store.write_enhancement(
        key, **(original_ticket | {"course_ids": ["wrong"]}), text="deniedword"
    )
    assert not library.store.write_enhancement(
        key, **(original_ticket | {"content_sha256": "0" * 64}), text="deniedword"
    )

    _note(path, "replacementword")
    second = library.import_paths("demo", [path])[0]
    assert not library.store.write_enhancement(key, **original_ticket, text="latereplacement")
    assert not _hits(library, "extratwo")
    assert _read(library, first.document_id, 1).status == "available"
    assert library.delete(first.document_id)

    second_key = key.model_copy(update={"version": 2})
    deleted_ticket = dict(
        course_ids=["demo"],
        content_sha256=second.content_sha256,
        publication_epoch=second.publication_epoch,
    )
    assert not library.store.write_enhancement(second_key, **deleted_ticket, text="latedelete")
    assert not _hits(library, "replacementword")
    assert _read(library, first.document_id, 1).status == "not_accessible"
    assert _read(library, first.document_id, 2).status == "not_accessible"

    reimport = library.import_paths("demo", [path])[0]
    assert reimport.version == 2 and reimport.publication_epoch > second.publication_epoch
    assert not library.store.write_enhancement(second_key, **deleted_ticket, text="latedelete")
    assert not _hits(library, "latedelete latereplacement deniedword")
    assert _hits(library, "replacementword")[0].version == 2
    assert _read(library, first.document_id, 2).original == path.read_bytes()


def test_release_at_keeps_old_version_then_promotes_once(
    library: LocalLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 1_800_000_000.0
    monkeypatch.setattr(storage, "_now", lambda: now)
    path = _note(library.course_root / "lesson.ipynb", "visibleword")
    first = library.import_paths("demo", [path])[0]
    _note(path, "scheduledword")
    rejected = library.import_paths(
        "demo", [path], release_at=datetime.fromtimestamp(now + 10, UTC)
    )[0]
    _note(path, "nextword")
    pending = library.import_paths(
        "demo", [path], release_at=datetime.fromtimestamp(now + 20, UTC)
    )[0]
    assert rejected.status == pending.status == "scheduled"
    assert pending.version == 3
    assert _hits(library, "visibleword")[0].version == 1
    assert not _hits(library, "scheduledword nextword")
    assert _read(library, first.document_id, 2).status == "not_accessible"
    assert _read(library, first.document_id, 3).status == "not_accessible"

    now += 11
    assert _hits(library, "visibleword")[0].version == 1
    now += 10
    ready = _hits(library, "nextword")[0]
    assert ready.version == 3
    assert ready.publication_epoch == pending.publication_epoch + 1
    assert not _hits(library, "visibleword scheduledword")
    assert _hits(library, "nextword")[0].publication_epoch == ready.publication_epoch
    assert _read(library, first.document_id, 1).status == "available"
    assert _read(library, first.document_id, 2).status == "not_accessible"
    assert _read(library, first.document_id, 3).status == "available"


def test_scheduling_current_content_replaces_an_existing_pending_version(
    library: LocalLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 1_800_000_000.0
    monkeypatch.setattr(storage, "_now", lambda: now)
    path = _note(library.course_root / "lesson.ipynb", "activeword")
    archive = path.read_bytes()
    first = library.import_paths("demo", [path])[0]
    _note(path, "discardedword")
    discarded = library.import_paths(
        "demo", [path], release_at=datetime.fromtimestamp(now + 10, UTC)
    )[0]
    path.write_bytes(archive)
    latest = library.import_paths("demo", [path], release_at=datetime.fromtimestamp(now + 20, UTC))[
        0
    ]
    assert latest.status == "scheduled" and latest.version == 1
    assert latest.publication_epoch > discarded.publication_epoch
    now += 11
    assert _hits(library, "activeword")[0].version == 1
    assert not _hits(library, "discardedword")
    assert _read(library, first.document_id, 2).status == "not_accessible"
    now += 10
    assert _hits(library, "activeword")[0].publication_epoch == latest.publication_epoch + 1
    assert _read(library, first.document_id, 1).original == archive
    assert _read(library, first.document_id, 2).status == "not_accessible"


def test_move_preserves_alias_and_competing_plans_cannot_overwrite(
    library: LocalLibrary,
) -> None:
    path = _note(library.course_root / "lesson.ipynb", "originalword")
    current = library.import_paths("demo", [path])[0]
    older = library.store.prepare("demo", path)
    moved = Path(library.course_root / "new.ipynb")
    shutil.copyfile(path, moved)
    move = library.import_paths("demo", [moved], document_id=current.document_id)[0]
    assert move.document_id == current.document_id
    assert move.version == 1 and move.publication_epoch > current.publication_epoch
    assert library.import_paths("demo", [path])[0].document_id == current.document_id
    assert _hits(library, "originalword")[0].source_path == str(moved)
    assert _read(library, current.document_id, 1).status == "available"
    other = library.import_paths("another", [path])[0]
    assert other.document_id != current.document_id

    _note(path, "preparedword")
    data = path.read_bytes()
    with pytest.raises(PublicationConflict):
        library.store.publish(
            older,
            parse_local_notebook(path),
            original=data,
            content_sha256=hashlib.sha256(data).hexdigest(),
            document_format="ipynb",
        )
    assert not _hits(library, "preparedword")
    assert _hits(library, "originalword")[0].version == 1
    assert _read(library, current.document_id, 2).status == "not_accessible"


def test_legacy_sqlite_backfill_and_current_visibility_cutback(library: LocalLibrary) -> None:
    source = _note(library.course_root / "legacy.ipynb", "legacyword")
    archive = source.read_bytes()
    legacy = MetadataStore(library.settings)
    parsed = NotebookParser(library.settings.data_dir / "assets").parse(source)
    chunks = NotebookChunker().chunk(parsed)
    prepared = PreparedDocumentIndex(
        document_id=parsed.document_id,
        source_path=parsed.source_path,
        title=parsed.title,
        groups=chunks.groups,
        sections=chunks.sections,
        chunks=chunks.chunks,
        text_representations=[],
        text_embedding_model="unused",
        text_embedding_dimension=1,
        visual_embedding_model="unused",
        visual_embedding_dimension=1,
    )
    version = legacy.reserve_version(
        parsed, course_id="demo", content_hash=hashlib.sha256(archive).hexdigest(), release_at=None
    ).version
    legacy.save_structure(parsed, prepared, version=version)
    legacy.activate_version(parsed.document_id, version)

    migrated = library.migrate_legacy(parsed.document_id)
    assert migrated.document_id == parsed.document_id
    keys = library.store.legacy_sources(
        parsed.document_id, version, f"chunk:{chunks.chunks[0].chunk_id}", course_ids=["demo"]
    )
    assert len(keys) == 1
    assert keys[0].source_id == "cell:0"
    assert _read(library, migrated.document_id, migrated.version).original == archive
    assert _hits(library, "legacyword")[0].version == migrated.version
    assert [item.version for item in legacy.visible_versions()] == [version]

    _note(source, "newlocalword")
    new = library.import_paths("demo", [source])[0]
    assert new.version > migrated.version
    assert not _hits(library, "legacyword")
    assert _hits(library, "newlocalword")[0].version == new.version
    assert not legacy.visible_versions()
    assert not legacy.list_documents()
    with pytest.raises(ValueError, match="不可见"):
        legacy.read_document(parsed.document_id, version, course_ids=["demo"], document_ids=[])
    assert _read(library, migrated.document_id, migrated.version).original == archive
    assert library.migrate_legacy(parsed.document_id).status == "backfilled"
    assert _hits(library, "newlocalword")[0].version == new.version

    source.write_bytes(archive)
    restored = library.import_paths("demo", [source])[0]
    assert restored.version == migrated.version
    assert [item.version for item in legacy.visible_versions()] == [version]
    assert legacy.read_document(
        parsed.document_id,
        version,
        course_ids=["demo"],
        document_ids=[],
    ).hits
    assert _read(library, migrated.document_id, new.version).status == "available"
    assert legacy.delete_document(parsed.document_id)
    assert not _hits(library, "legacyword")
    assert _read(library, migrated.document_id, migrated.version).status == "not_accessible"


def test_legacy_bundle_missing_original_backfills_same_page_then_hydrates(
    library: LocalLibrary,
    tmp_path: Path,
) -> None:
    source = library.course_root / "Signal-slides.pptx"
    shutil.copyfile(SAMPLES / source.name, source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    deck = PPTXParser(tmp_path / "old-prep", render_slides=False).parse(source)
    bundle = tmp_path / "old-bundle"
    bundle.mkdir()
    chunks = SlideChunker().chunk(deck)
    table_key = (
        f"element:{next(e.element_id for e in deck.slides[0].elements if e.kind == 'table')}"
    )
    (bundle / "deck.json").write_text(deck.model_dump_json())
    (bundle / "chunks.json").write_text(chunks.model_dump_json())
    source.unlink()

    result = library.import_legacy_bundle(
        course_id="demo",
        source_path=source,
        bundle_dir=bundle,
        content_sha256=digest,
        version=7,
    )
    assert result.status == "published"
    assert result.document_id == deck.document_id
    hit = _hits(library, "Peak")[0]
    assert hit.origin == "legacy_text"
    assert hit.source_id == "page:1"
    keys = library.store.legacy_sources(
        deck.document_id, 7, f"chunk:{chunks.chunks[0].chunk_id}", course_ids=["demo"]
    )
    assert [key.source_id for key in keys] == ["page:1"]
    assert [
        key.source_id
        for key in library.store.legacy_sources(deck.document_id, 7, table_key, course_ids=["demo"])
    ] == ["page:1"]
    absent = _read(library, result.document_id, result.version, "page:1")
    assert absent.status == "original_unavailable" and absent.source.locator["page"] == 1
    assert absent.original is None

    shutil.copyfile(SAMPLES / source.name, source)
    hydrated = library.import_paths("demo", [source])[0]
    assert hydrated.version == result.version
    table = _read(library, result.document_id, result.version, "page:1:shape:3")
    assert table.status == "available" and table.source.locator["shape_id_path"] == [3]
    assert (
        _read(library, result.document_id, result.version, "page:1").original == source.read_bytes()
    )
    assert {hit.origin for hit in _hits(library, "Peak")} == {"native_text"}
    assert (
        library.import_legacy_bundle(
            course_id="demo",
            source_path=source,
            bundle_dir=bundle,
            content_sha256=digest,
            version=7,
        ).status
        == "backfilled"
    )
    assert [
        key.source_id
        for key in library.store.legacy_sources(deck.document_id, 7, table_key, course_ids=["demo"])
    ] == ["page:1:shape:3"]
    source.unlink()
    library.import_legacy_bundle(
        course_id="demo",
        source_path=source,
        bundle_dir=bundle,
        content_sha256=digest,
        version=7,
    )
    assert [
        key.source_id
        for key in library.store.legacy_sources(deck.document_id, 7, table_key, course_ids=["demo"])
    ] == ["page:1:shape:3"]


def test_directory_import_keeps_scan_page_but_indexes_no_empty_text(
    library: LocalLibrary,
    tmp_path: Path,
) -> None:
    from PIL import Image

    Image.new("RGB", (30, 30), "white").save(library.course_root / "scan.pdf")
    (library.course_root / "bad.pdf").write_bytes(b"not a PDF")
    _note(library.course_root / "text.ipynb", "directoryword")
    (library.course_root / "ignore.txt").write_text("ignore me")
    _note(tmp_path / "outside.ipynb", "outsideword")
    (library.course_root / "link.ipynb").symlink_to(tmp_path / "outside.ipynb")
    results = {
        Path(item.source_path).name: item for item in library.import_paths("demo", [Path(".")])
    }
    assert set(results) == {"scan.pdf", "bad.pdf", "text.ipynb"}
    assert results["bad.pdf"].status == "failed"
    scan = results["scan.pdf"]
    assert scan.status == "published" and scan.source_count == 1 and scan.searchable_count == 0
    assert (
        library.store.current_sources(scan.document_id, course_ids=["demo"])[0].state == "needs_ocr"
    )
    assert _read(library, scan.document_id, 1, "page:1").status == "available"
    assert _hits(library, "directoryword")[0].document_id == results["text.ipynb"].document_id
    assert not _hits(library, "outsideword")
    assert library.import_paths("demo", [tmp_path / "outside.ipynb"])[0].status == "failed"
