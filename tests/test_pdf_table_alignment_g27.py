"""G27 tests for borderless PDF table extraction, column alignment, and reimport."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from test_local_slides import _build_pdf, _pdf_line, _pdf_text

from app.config import Settings
from app.rag import local_library
from app.rag.parsers.local_slides import parse_local_slides
from app.schemas.local import SourceKey

ROOT = Path(__file__).parents[1]
PAGED_ATTENTION = ROOT / "docs/G/G20-fixtures/PagedAttention-2309.06180v1.pdf"
MIXTRAL = ROOT / "docs/G/G20-fixtures/Mixtral-of-Experts-2401.04088v1.pdf"
MAMBA = ROOT / "docs/G/G21-fixtures/Mamba-2312.00752v1.pdf"


def _booktabs_pdf(
    path: Path,
    *,
    lines: list[tuple[float, float, float]],
    texts: list[tuple[float, float, str]],
) -> None:
    commands = [_pdf_text(x, y, text) for x, y, text in texts]
    commands.extend(_pdf_line(x0, y, x1, y) for x0, x1, y in lines)
    path.write_bytes(_build_pdf("\n".join(commands).encode("ascii")))


def test_paged_attention_h02_table1_recovers_all_12_cells_and_gpu_column_alignment() -> None:
    parsed = parse_local_slides(PAGED_ATTENTION)
    p9_tables = [
        source
        for source in parsed.sources
        if source.locator.get("page") == 9 and source.locator.get("object_type") == "table"
    ]
    assert len(p9_tables) == 1
    table = p9_tables[0]
    assert table.source_id == "page:9:table:1"
    assert table.parent_id == "page:9"
    assert table.locator["columns"] == ["Model size", "13B", "66B", "175B"]
    assert table.locator["rows"] == [
        ["Model size", "13B", "66B", "175B"],
        ["GPUs", "A100", "4×A100", "8×A100-80GB"],
        ["Total GPU memory", "40 GB", "160 GB", "640 GB"],
        ["Parameter size", "26 GB", "132 GB", "346 GB"],
        ["Memory for KV cache", "12 GB", "21 GB", "264 GB"],
        ["Max. # KV cache slots", "15.7K", "9.7K", "60.1K"],
    ]
    assert table.locator["coordinate_frame"] == "visible_page_norm_v1"
    assert table.locator["position_basis"] == "axis_aligned_native_rule_grid_v1"
    x0, y0, x1, y1 = table.locator["bbox"]
    assert 0.1 < x0 < x1 < 0.5 and 0.08 < y0 < y1 < 0.25


def test_mixtral_page8_spanner_header_and_mamba_page11_borderless_table() -> None:
    mixtral_parsed = parse_local_slides(MIXTRAL)
    p8_tables = [
        source
        for source in mixtral_parsed.sources
        if source.locator.get("page") == 8 and source.locator.get("object_type") == "table"
    ]
    assert len(p8_tables) == 1
    p8_table = p8_tables[0]
    assert p8_table.locator["columns"] == [
        "",
        "First choice: Layer 0",
        "First choice: Layer 15",
        "First choice: Layer 31",
        "First or second choice: Layer 0",
        "First or second choice: Layer 15",
        "First or second choice: Layer 31",
    ]
    assert ["ArXiv", "14.0%", "27.9%", "22.7%", "46.5%", "62.3%", "52.9%"] in p8_table.locator[
        "rows"
    ]
    assert [
        "DM Mathematics",
        "14.1%",
        "28.4%",
        "19.7%",
        "44.9%",
        "67.0%",
        "44.5%",
    ] in p8_table.locator["rows"]

    mamba_parsed = parse_local_slides(MAMBA)
    p11_tables = [
        source
        for source in mamba_parsed.sources
        if source.locator.get("page") == 11 and source.locator.get("object_type") == "table"
    ]
    assert len(p11_tables) == 1
    assert p11_tables[0].locator["columns"] == ["Model", "Arch.", "Layer", "Acc."]
    assert ["Mamba", "Mamba", "S6", "99.8"] in p11_tables[0].locator["rows"]


def test_swapped_column_order_moves_values_with_headers(tmp_path: Path) -> None:
    pdf_a = tmp_path / "order-a.pdf"
    pdf_b = tmp_path / "order-b.pdf"
    rules = [(72, 360, 680), (72, 360, 655), (72, 360, 600)]
    _booktabs_pdf(
        pdf_a,
        lines=rules,
        texts=[
            (80, 665, "Config"),
            (170, 665, "13B"),
            (240, 665, "66B"),
            (310, 665, "175B"),
            (80, 635, "GPUs"),
            (170, 635, "A100"),
            (240, 635, "4xA100"),
            (310, 635, "8xA100"),
            (80, 615, "KV"),
            (170, 615, "12GB"),
            (240, 615, "21GB"),
            (310, 615, "264GB"),
        ],
    )
    _booktabs_pdf(
        pdf_b,
        lines=rules,
        texts=[
            (80, 665, "Config"),
            (170, 665, "175B"),
            (240, 665, "13B"),
            (310, 665, "66B"),
            (80, 635, "GPUs"),
            (170, 635, "8xA100"),
            (240, 635, "A100"),
            (310, 635, "4xA100"),
            (80, 615, "KV"),
            (170, 615, "264GB"),
            (240, 615, "12GB"),
            (310, 615, "21GB"),
        ],
    )
    rows_a = parse_local_slides(pdf_a).sources[1].locator["rows"]
    rows_b = parse_local_slides(pdf_b).sources[1].locator["rows"]
    map_a = dict(zip(rows_a[0], rows_a[1], strict=True))
    map_b = dict(zip(rows_b[0], rows_b[1], strict=True))
    assert map_a == {"Config": "GPUs", "13B": "A100", "66B": "4xA100", "175B": "8xA100"}
    assert map_b == map_a


def test_multiline_header_empty_cells_and_wrapped_row_do_not_shift_columns(tmp_path: Path) -> None:
    pdf = tmp_path / "complex-booktabs.pdf"
    _booktabs_pdf(
        pdf,
        lines=[(72, 360, 695), (72, 360, 655), (72, 360, 585)],
        texts=[
            # 2-line wrapped headers
            (80, 680, "Metric"),
            (180, 680, "Small"),
            (280, 680, "Large"),
            (80, 665, "Name"),
            (180, 665, "Model"),
            (280, 665, "Model"),
            # Row 1 with empty middle cell (Small Model omitted)
            (80, 640, "Latency"),
            (280, 640, "14.2ms"),
            # Row 2 with wrapped Col 0 label
            (80, 620, "Memory"),
            (180, 620, "40GB"),
            (280, 620, "80GB"),
            (80, 608, "Limit"),
        ],
    )
    parsed = parse_local_slides(pdf)
    tables = [s for s in parsed.sources if s.locator.get("object_type") == "table"]
    assert len(tables) == 1
    assert tables[0].locator["columns"] == ["Metric Name", "Small Model", "Large Model"]
    assert tables[0].locator["rows"][1] == ["Latency", "", "14.2ms"]
    assert tables[0].locator["rows"][2] == ["Memory Limit", "40GB", "80GB"]


def test_multicolumn_prose_and_chart_gridlines_are_not_registered_as_tables(tmp_path: Path) -> None:
    pdf = tmp_path / "prose-and-chart.pdf"
    # Two-column prose + 5 uniformly spaced horizontal chart gridlines crossing prose
    commands = [
        _pdf_text(60, 640, "Left column prose continues across"),
        _pdf_text(310, 640, "Right column prose discusses"),
        _pdf_text(60, 610, "many lines without table structure"),
        _pdf_text(310, 610, "performance in detail"),
        _pdf_text(60, 580, "and must remain page text only"),
        _pdf_text(310, 580, "with zero table objects"),
    ]
    commands.extend(_pdf_line(190, y, 480, y) for y in (650, 620, 590, 560, 530))
    pdf.write_bytes(_build_pdf("\n".join(commands).encode("ascii")))

    parsed = parse_local_slides(pdf)
    assert [source.source_id for source in parsed.sources] == ["page:1"]
    assert "Left column prose" in parsed.sources[0].text


def test_same_byte_reimport_appends_borderless_table_without_mutating_page_or_fts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "booktabs-reimport.pdf"
    _booktabs_pdf(
        pdf,
        lines=[(72, 320, 680), (72, 320, 655), (72, 320, 610)],
        texts=[
            (80, 665, "Model"),
            (200, 665, "GPUs"),
            (80, 635, "13B"),
            (200, 635, "A100"),
            (80, 618, "66B"),
            (200, 618, "4xA100"),
        ],
    )
    archived = pdf.read_bytes()
    sqlite_path = tmp_path / "library.sqlite3"
    library = local_library.LocalLibrary(
        Settings(
            _env_file=None,
            sqlite_path=sqlite_path,
            data_dir=tmp_path / "data",
            course_root=tmp_path,
        )
    )

    def parse_without_tables(path: Path):
        parsed = parse_local_slides(path)
        return parsed.model_copy(
            update={"sources": [s for s in parsed.sources if s.kind == "page"]}
        )

    with monkeypatch.context() as baseline:
        baseline.setattr(local_library, "parse_local_slides", parse_without_tables)
        (old,) = library.import_paths("demo", [pdf])

    def page_snapshot():
        with sqlite3.connect(sqlite_path) as db:
            return db.execute(
                "SELECT * FROM local_sources WHERE source_id = 'page:1'"
            ).fetchall()

    before = page_snapshot()
    (reimported,) = library.import_paths("demo", [pdf])
    (again,) = library.import_paths("demo", [pdf])
    assert page_snapshot() == before
    assert reimported.status == again.status == "skipped"
    assert reimported.version == again.version == old.version == 1
    assert reimported.publication_epoch == old.publication_epoch + 1
    assert old.source_count == 1 and reimported.source_count == again.source_count == 2
    key = SourceKey(document_id=old.document_id, version=1, source_id="page:1:table:1")
    stored = library.store.read_source(key, course_ids=["demo"])
    assert stored.status == "available" and stored.original == archived
    assert stored.source is not None
    assert ["66B", "4xA100"] in stored.source.locator["rows"]
