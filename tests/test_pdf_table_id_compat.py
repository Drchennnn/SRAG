"""Source-key and reading-order compatibility when a PDF gains a horizontal-rule table."""

from pathlib import Path

import pytest
from test_local_slides import _build_pdf, _pdf_line, _pdf_text

from app.config import Settings
from app.rag.local_library import LocalLibrary
from app.rag.parsers import local_slides
from app.rag.workspace_objects import object_choices, object_intent
from app.schemas.local import SourceKey
from app.stores.structure_snapshot import SnapshotRequest, StructureSnapshotStore


def test_mixed_tables_keep_legacy_key_and_visual_order_on_reimport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "mixed-table-styles.pdf"
    commands = [
        _pdf_text(80, 665, "Model"),
        _pdf_text(200, 665, "GPUs"),
        _pdf_text(80, 635, "13B"),
        _pdf_text(200, 635, "A100"),
        _pdf_text(80, 618, "66B"),
        _pdf_text(200, 618, "4xA100"),
        _pdf_text(82, 420, "Metric"),
        _pdf_text(212, 420, "Value"),
        _pdf_text(82, 370, "Delay"),
        _pdf_text(212, 370, "2"),
    ]
    commands.extend(_pdf_line(72, y, 320, y) for y in (680, 655, 610))
    commands.extend(_pdf_line(72, y, 348, y) for y in (450, 400, 350))
    commands.extend(_pdf_line(x, 350, x, 450) for x in (72, 200, 348))
    pdf.write_bytes(_build_pdf("\n".join(commands).encode("ascii")))

    settings = Settings(
        _env_file=None, sqlite_path=tmp_path / "library.db", data_dir=tmp_path, course_root=tmp_path
    )
    library = LocalLibrary(settings)
    with monkeypatch.context() as legacy:
        legacy.setattr(local_slides, "extract_horizontal_rule_tables", lambda *args: [])
        (old,) = library.import_paths("demo", [pdf])

    key = SourceKey(document_id=old.document_id, version=old.version, source_id="page:1:table:1")
    original = library.store.read_source(key, course_ids=["demo"])
    assert original.source is not None
    assert original.source.locator["rows"] == [["Metric", "Value"], ["Delay", "2"]]

    (updated,) = library.import_paths("demo", [pdf])
    (again,) = library.import_paths("demo", [pdf])
    assert updated.version == again.version == old.version
    assert old.publication_epoch + 1 == updated.publication_epoch == again.publication_epoch
    assert updated.source_count == again.source_count == 3
    historical = library.store.read_source(key, course_ids=["demo"])
    assert historical.source is not None
    assert historical.source.locator == original.source.locator
    assert historical.source.text == original.source.text
    assert historical.original == pdf.read_bytes()

    def table_positions(sources):
        return [
            (s.source_id, s.locator["rows"][0])
            for s in sorted(sources, key=lambda s: s.ordinal)
            if s.locator.get("object_type") == "table"
        ]

    expected = [("page:1:table:2", ["Model", "GPUs"]), ("page:1:table:1", ["Metric", "Value"])]
    snapshot = StructureSnapshotStore(settings).read(SnapshotRequest(course_ids=("demo",)))
    assert table_positions(snapshot.sources) == expected
    assert table_positions(local_slides.parse_local_slides(pdf).sources) == expected
    intent = object_intent("打开本页第一个表格")
    assert intent is not None
    (first,) = object_choices(intent, snapshot.sources)
    assert first.source_id == "page:1:table:2"
    assert first.title == "Model | GPUs"
