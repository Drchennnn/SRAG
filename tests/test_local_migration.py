"""B04 legacy migration input validation and conservative mapping tests."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.config import Settings
from app.rag.local_migration import (
    LegacySnapshot,
    load_legacy_bundle,
    load_legacy_sqlite,
    map_legacy_sources,
)
from app.schemas.document import (
    NormalizedBoundingBox,
    Slide,
    SlideAsset,
    SlideChunk,
    SlideChunkingResult,
    SlideDeck,
    SlideElement,
    SlideElementKind,
    SlideSection,
    SlideSourceFormat,
)
from app.schemas.local import ParsedSource, SourceDraft
from app.stores.sqlite import MetadataStore

FULL_HASH = "a" * 64


def test_load_legacy_sqlite_reads_visible_active_snapshot_and_maps_notebook(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data", sqlite_path=tmp_path / "legacy.sqlite3")
    MetadataStore(settings).initialize()
    source_path = tmp_path / "missing-old.ipynb"
    with sqlite3.connect(settings.sqlite_path) as connection:
        connection.execute(
            """INSERT INTO documents(document_id, course_id, source_path, title, document_type,
               created_at, updated_at, deleted_at) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)""",
            ("old-doc", "CS", str(source_path), "Old lab", "notebook", "t", "t"),
        )
        connection.execute(
            """INSERT INTO document_versions(document_id, version, content_hash, release_at,
               state, created_at, indexed_at, error)
               VALUES ('old-doc', 2, ?, NULL, 'active', 't', 't', NULL)""",
            (FULL_HASH,),
        )
        blocks = [
            {
                "block_id": "intro:markdown:0",
                "document_id": "old-doc",
                "kind": "markdown",
                "text": "# Old lab",
                "cell_id": "intro",
                "cell_index": 0,
                "assets": [],
            },
            {
                "block_id": "plots:output:1",
                "document_id": "old-doc",
                "kind": "figure",
                "text": "",
                "cell_id": "plots",
                "cell_index": 1,
                "related_block_id": "plots:code",
                "assets": [],
            },
        ]
        for block in blocks:
            connection.execute(
                """INSERT INTO content_blocks(document_id, version, block_id, kind, cell_id,
                   cell_index, page, payload_json) VALUES ('old-doc', 2, ?, ?, ?, ?, NULL, ?)""",
                (
                    block["block_id"],
                    block["kind"],
                    block["cell_id"],
                    block["cell_index"],
                    json.dumps(block),
                ),
            )
        chunk = {
            "chunk_id": "chunk-old",
            "section_id": "section-old",
            "document_id": "old-doc",
            "kind": "figure",
            "display_text": "old plot",
            "embedding_text": "old plot",
            "group_ids": [],
            "block_ids": ["plots:output:1"],
            "cell_ids": ["plots"],
            "token_count": 2,
        }
        connection.execute(
            """INSERT INTO chunks(document_id, version, chunk_id, section_id, kind,
               previous_chunk_id, next_chunk_id, payload_json)
               VALUES ('old-doc', 2, 'chunk-old', 'section-old', 'figure', NULL, NULL, ?)""",
            (json.dumps(chunk),),
        )
        connection.execute(
            """INSERT INTO assets(document_id, version, asset_id, block_id, media_type, path,
               caption) VALUES ('old-doc', 2, 'asset-old', 'plots:output:1', 'image/png',
               '/must/not/be/opened.png', 'plot')"""
        )
        # A local mask must not prevent explicitly requested legacy re-migration.
        connection.execute("CREATE TABLE local_deleted(document_id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO local_deleted VALUES ('old-doc')")

    snapshot = load_legacy_sqlite(settings, "old-doc")
    native = ParsedSource(
        title="Old lab",
        sources=[
            SourceDraft(
                source_id="cell:0",
                kind="cell",
                ordinal=0,
                locator={"cell_index": 0, "cell_id": "intro"},
            ),
            SourceDraft(
                source_id="cell:1",
                kind="cell",
                ordinal=1,
                locator={"cell_index": 1, "cell_id": "plots"},
            ),
            SourceDraft(
                source_id="cell:1:output:1",
                kind="notebook_output",
                ordinal=2,
                parent_id="cell:1",
                locator={"cell_index": 1, "cell_id": "plots", "output_index": 1},
            ),
        ],
    )

    parsed, mapping = map_legacy_sources(snapshot, native)

    assert parsed is native
    assert snapshot.version == 2
    assert snapshot.document_format == "ipynb"
    assert snapshot.source_path == source_path
    assert snapshot.content_sha256 == FULL_HASH
    assert mapping["block:intro:markdown:0"] == ["cell:0"]
    assert mapping["block:plots:output:1"] == ["cell:1:output:1"]
    assert mapping["chunk:chunk-old"] == ["cell:1:output:1"]
    assert mapping["asset:asset-old"] == ["cell:1:output:1"]


def test_load_legacy_sqlite_rejects_incomplete_hash(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", sqlite_path=tmp_path / "legacy.sqlite3")
    MetadataStore(settings).initialize()
    with sqlite3.connect(settings.sqlite_path) as connection:
        connection.execute(
            """INSERT INTO documents VALUES
               ('doc', 'CS', '/old/lab.ipynb', 'Lab', 'notebook', 't', 't', NULL)"""
        )
        connection.execute(
            """INSERT INTO document_versions VALUES
               ('doc', 1, 'abc123', NULL, 'active', 't', 't', NULL)"""
        )

    with pytest.raises(ValueError, match="complete SHA-256"):
        load_legacy_sqlite(settings, "doc")


def test_bundle_allows_missing_original_validates_identity_and_maps_exact_object(
    tmp_path: Path,
) -> None:
    source = tmp_path / "missing-lecture.pptx"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    document_id = f"document_{FULL_HASH[:24]}"
    bbox = NormalizedBoundingBox(left=0.1, top=0.2, right=0.9, bottom=0.8)
    element = SlideElement(
        element_id="element-table",
        kind=SlideElementKind.TABLE,
        text="Metric | Value\nDelay | 2",
        bbox=bbox,
        asset_id="asset-table",
    )
    asset = SlideAsset(
        asset_id="asset-table",
        media_type="image/png",
        path="../../outside-and-missing.png",
        sha256="b" * 64,
        bbox=bbox,
    )
    slide = Slide(
        slide_id="slide-one",
        document_id=document_id,
        slide_no=1,
        title="Results",
        elements=[element],
        assets=[asset],
    )
    deck = SlideDeck(
        document_id=document_id,
        source_path=f"/old/{source.name}",
        source_format=SlideSourceFormat.PPTX,
        title="Lecture",
        slides=[slide],
    )
    chunk = SlideChunk(
        chunk_id="chunk-one",
        section_id="section-one",
        document_id=document_id,
        slide_id="slide-one",
        slide_no=1,
        display_text="Metric | Value\nDelay | 2",
        embedding_text="Metric Value Delay 2",
        element_ids=["element-table"],
        asset_ids=["asset-table"],
        token_count=4,
    )
    chunking = SlideChunkingResult(
        document_id=document_id,
        outline_text="Lecture",
        sections=[
            SlideSection(
                section_id="section-one",
                document_id=document_id,
                title="Results",
                slide_ids=["slide-one"],
                chunk_ids=["chunk-one"],
            )
        ],
        chunks=[chunk],
    )
    (bundle / "deck.json").write_text(deck.model_dump_json(), encoding="utf-8")
    (bundle / "chunks.json").write_text(chunking.model_dump_json(), encoding="utf-8")

    snapshot = load_legacy_bundle(
        course_id="CS5187",
        source_path=source,
        bundle_dir=bundle,
        content_sha256=FULL_HASH,
    )
    native = ParsedSource(
        title="Lecture",
        sources=[
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=0,
                locator={"format": "pptx", "page": 1},
            ),
            SourceDraft(
                source_id="page:1:shape:3",
                kind="page_object",
                ordinal=1,
                parent_id="page:1",
                text="Metric | Value\nDelay | 2",
                locator={
                    "format": "pptx",
                    "page": 1,
                    "object_type": "table",
                    "bbox": [0.1, 0.2, 0.9, 0.8],
                },
            ),
        ],
    )

    _, mapping = map_legacy_sources(snapshot, native)

    assert not source.exists()
    assert snapshot.document_format == "pptx"
    assert mapping["block:slide-one"] == ["page:1"]
    assert mapping["element:element-table"] == ["page:1:shape:3"]
    assert mapping["asset:asset-table"] == ["page:1:shape:3"]
    assert mapping["chunk:chunk-one"] == ["page:1:shape:3"]


def test_ambiguous_or_absent_native_uses_page_and_legacy_text_fallback() -> None:
    snapshot = LegacySnapshot(
        document_id="doc",
        version=1,
        course_id="CS",
        source_path=Path("missing.pdf"),
        content_sha256=FULL_HASH,
        document_format="pdf",
        title="Old PDF",
        release_at=None,
        _blocks=(
            {
                "slide_id": "slide-one",
                "slide_no": 1,
                "elements": [{"element_id": "old-text", "kind": "text", "text": "Legacy body"}],
            },
        ),
        _chunks=(
            {
                "chunk_id": "old-chunk",
                "slide_id": "slide-one",
                "slide_no": 1,
                "display_text": "Legacy body",
                "element_ids": ["old-text"],
            },
        ),
        _elements=(
            {
                "element_id": "old-text",
                "kind": "text",
                "text": "Legacy body",
                "_legacy_page": 1,
            },
        ),
    )

    parsed, mapping = map_legacy_sources(snapshot, None)

    assert [source.source_id for source in parsed.sources] == ["page:1"]
    assert parsed.sources[0].text == "Legacy body"
    assert parsed.sources[0].locator["text_origin"] == "legacy_text"
    assert mapping["element:old-text"] == ["page:1"]
    assert mapping["chunk:old-chunk"] == ["page:1"]


def test_bundle_rejects_broken_relations(tmp_path: Path) -> None:
    source = tmp_path / "lecture.pdf"
    bundle = tmp_path / "broken"
    bundle.mkdir()
    document_id = f"document_{FULL_HASH[:24]}"
    deck = SlideDeck(
        document_id=document_id,
        source_path=f"/old/{source.name}",
        source_format=SlideSourceFormat.PDF,
        title="Lecture",
        slides=[Slide(slide_id="slide-one", document_id=document_id, slide_no=1)],
    )
    chunking = SlideChunkingResult(
        document_id=document_id,
        outline_text="",
        sections=[
            SlideSection(
                section_id="section-one",
                document_id=document_id,
                title="Lecture",
                slide_ids=["slide-one"],
                chunk_ids=["missing-chunk"],
            )
        ],
        chunks=[],
    )
    (bundle / "deck.json").write_text(deck.model_dump_json(), encoding="utf-8")
    (bundle / "chunks.json").write_text(chunking.model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="missing chunk"):
        load_legacy_bundle(
            course_id="CS",
            source_path=source,
            bundle_dir=bundle,
            content_sha256=FULL_HASH,
        )


def test_load_legacy_sqlite_excludes_future_and_deleted_active_rows(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", sqlite_path=tmp_path / "legacy.sqlite3")
    MetadataStore(settings).initialize()
    future = datetime(2999, 1, 1, tzinfo=UTC).isoformat()
    with sqlite3.connect(settings.sqlite_path) as connection:
        for document_id, deleted_at, release_at in (
            ("future", None, future),
            ("deleted", "now", None),
        ):
            connection.execute(
                "INSERT INTO documents VALUES (?, 'CS', ?, 'Title', 'pdf', 't', 't', ?)",
                (document_id, f"/{document_id}.pdf", deleted_at),
            )
            connection.execute(
                "INSERT INTO document_versions VALUES (?, 1, ?, ?, 'active', 't', 't', NULL)",
                (document_id, FULL_HASH, release_at),
            )

    with pytest.raises(ValueError, match="no currently visible"):
        load_legacy_sqlite(settings, "future")
    with pytest.raises(ValueError, match="no currently visible"):
        load_legacy_sqlite(settings, "deleted")
