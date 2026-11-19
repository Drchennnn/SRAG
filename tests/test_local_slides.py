from __future__ import annotations

import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image
from pptx import Presentation
from pptx.util import Inches

from app.config import Settings
from app.rag import local_library
from app.rag.parsers.local_slides import parse_local_slides
from app.schemas.local import SourceKey

ROOT = Path(__file__).parents[1]
A03_PDF = ROOT / "docs/A/样例/Signal-reference.pdf"
A03_PPTX = ROOT / "docs/A/样例/Signal-slides.pptx"


def test_a03_pdf_preserves_page_order_native_text_and_does_not_render(monkeypatch) -> None:
    import pypdfium2

    original_hash = _sha256(A03_PDF)

    def forbid_render(*args, **kwargs):
        raise AssertionError("base PDF import must not render pages")

    monkeypatch.setattr(pypdfium2.PdfPage, "render", forbid_render)
    parsed = parse_local_slides(A03_PDF)

    assert [source.source_id for source in parsed.sources] == ["page:1", "page:2"]
    assert "future samples = (W - 1) / 2" in parsed.sources[0].text
    assert "(b) Median, W = 5" in parsed.sources[1].text
    assert all(source.locator["render_status"] == "not_attempted" for source in parsed.sources)
    assert _sha256(A03_PDF) == original_hash


def test_pdfium_parsing_is_safe_for_concurrent_worker_calls() -> None:
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(parse_local_slides, [A03_PDF] * 8))

    assert all(len(parsed.sources) == 2 for parsed in results)
    assert all("Median Filter Reference" in parsed.sources[0].text for parsed in results)


def test_pdf_detects_conservative_native_rule_table(tmp_path: Path) -> None:
    pdf = tmp_path / "native-table.pdf"
    commands = [
        _pdf_text(82, 620, "Metric"),
        _pdf_text(212, 620, "Value [ms]"),
        _pdf_text(82, 570, "Delay"),
        _pdf_text(212, 570, "2"),
    ]
    commands.extend(_pdf_line(72, y, 328, y) for y in (650, 600, 550))
    commands.extend(_pdf_line(x, 550, x, 650) for x in (72, 200, 328))
    pdf.write_bytes(_build_pdf("\n".join(commands).encode("ascii")))

    parsed = parse_local_slides(pdf)

    assert [source.source_id for source in parsed.sources] == [
        "page:1",
        "page:1:table:1",
    ]
    table = parsed.sources[1]
    assert table.parent_id == "page:1"
    assert table.locator["columns"] == ["Metric", "Value [ms]"]
    assert table.locator["rows"] == [["Metric", "Value [ms]"], ["Delay", "2"]]
    assert table.locator["coordinate_frame"] == "visible_page_norm_v1"
    assert table.locator["position_basis"] == "axis_aligned_native_rule_grid_v1"
    assert table.locator["bbox"] == pytest.approx(
        [72 / 612, (792 - 650) / 792, 328 / 612, (792 - 550) / 792], abs=0.006
    )


def test_pdf_numbers_multiple_tables_by_visible_position(tmp_path: Path) -> None:
    pdf = tmp_path / "two-tables.pdf"
    commands = []
    commands.extend(_pdf_grid(72, 550, "Top", "A"))
    commands.extend(_pdf_grid(300, 300, "Bottom", "B"))
    pdf.write_bytes(_build_pdf("\n".join(commands).encode("ascii")))

    parsed = parse_local_slides(pdf)
    tables = [source for source in parsed.sources if source.kind == "page_object"]

    assert [source.source_id for source in tables] == [
        "page:1:table:1",
        "page:1:table:2",
    ]
    assert tables[0].locator["rows"][0][0] == "Top"
    assert tables[1].locator["rows"][0][0] == "Bottom"
    assert tables[0].locator["bbox"][1] < tables[1].locator["bbox"][1]


def test_raster_only_pdf_is_needs_ocr_without_fts_text_or_render(
    tmp_path: Path, monkeypatch
) -> None:
    import pypdfium2

    scan = tmp_path / "scan.pdf"
    Image.new("RGB", (120, 80), "white").save(scan, format="PDF")

    def forbid_render(*args, **kwargs):
        raise AssertionError("scan detection must not render or OCR")

    monkeypatch.setattr(pypdfium2.PdfPage, "render", forbid_render)
    (source,) = parse_local_slides(scan).sources

    assert source.source_id == "page:1"
    assert source.state == "needs_ocr"
    assert source.text == ""
    assert source.locator["format"] == "pdf"
    assert source.locator["page"] == 1
    assert source.locator["native_text"] is False
    assert source.locator["rotation_degrees_clockwise"] == 0
    assert source.locator["render_status"] == "not_attempted"
    assert source.locator["ocr_status"] == "not_attempted"


def test_pdf_registers_rotated_cropped_images_in_visible_order(tmp_path: Path) -> None:
    pdf = tmp_path / "two-panels.pdf"
    # The raster painted first becomes the right panel after the clockwise page rotation.
    commands = [
        _pdf_text(60, 670, "Induction lab"),
        "q 160 0 0 145 220 420 cm /Panel Do Q",
        "q 160 0 0 145 220 100 cm /Panel Do Q",
    ]
    pdf.write_bytes(
        _build_pdf("\n".join(commands).encode(), crop="25 40 587 752", rotation=90, raster=True)
    )

    first = parse_local_slides(pdf)
    assert parse_local_slides(pdf) == first
    page, left, right = first.sources
    assert page.source_id == "page:1" and page.text == "Induction lab"
    assert [left.source_id, right.source_id] == ["page:1:image:1", "page:1:image:2"]
    assert [page.ordinal, left.ordinal, right.ordinal] == [0, 1, 2]
    assert left.locator["native_bbox"] == [220, 100, 380, 245]
    assert right.locator["native_bbox"] == [220, 420, 380, 565]
    for image in (left, right):
        assert image.parent_id == page.source_id and not image.text
        assert image.locator["position_basis"] == "pdfium_native_image_extents_v1"
        assert image.locator["native_to_visible_transform"]["matrix"] == pytest.approx(
            [0, 1 / 562, 1 / 712, 0, -40 / 712, -25 / 562]
        )
    assert left.locator["bbox"] == pytest.approx(
        [60 / 712, 195 / 562, 205 / 712, 355 / 562], abs=1e-6
    )
    assert right.locator["bbox"] == pytest.approx(
        [380 / 712, 195 / 562, 525 / 712, 355 / 562], abs=1e-6
    )


@pytest.mark.parametrize(
    "other",
    [
        "q 70 0 0 105 430 420 cm /Panel Do Q",  # Overlap
        "q 110 15 0 90 360 540 cm /Panel Do Q",  # Skew on the same visible half
        "q 100 0 0 90 560 540 cm /Panel Do Q",  # Clipped by the CropBox
        "q 415 565 55 70 re W n 100 0 0 90 390 550 cm /Panel Do Q",  # Path clip
    ],
)
def test_pdf_uses_page_fallback_for_competing_unreliable_rasters(
    tmp_path: Path, other: str
) -> None:
    pdf = tmp_path / "unreliable.pdf"
    commands = [
        _pdf_text(60, 670, "Induction lab"),
        "q 150 0 0 130 360 400 cm /Panel Do Q",
        other,
    ]
    pdf.write_bytes(_build_pdf("\n".join(commands).encode(), raster=True))

    (page,) = parse_local_slides(pdf).sources
    assert page.source_id == "page:1" and page.text == "Induction lab"


def test_legacy_pdf_reimport_appends_images_without_mutating_page_rows_or_fts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "lab.pdf"
    native = [
        _pdf_text(60, 670, "Induction lab"),
        "q 170 0 0 150 330 180 cm /Panel Do Q",
    ]
    pdf.write_bytes(_build_pdf("\n".join(native).encode(), raster=True))
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

    def parse_as_old_release(path: Path):
        parsed = parse_local_slides(path)
        return parsed.model_copy(update={"sources": parsed.sources[:1]})

    with monkeypatch.context() as baseline:
        baseline.setattr(local_library, "parse_local_slides", parse_as_old_release)
        (old,) = library.import_paths("demo", [pdf])

    def original_rows():
        with sqlite3.connect(sqlite_path) as database:
            return (
                database.execute(
                    "SELECT * FROM local_sources WHERE source_id = 'page:1'"
                ).fetchall(),
                database.execute("SELECT * FROM local_fts").fetchall(),
                database.execute("SELECT * FROM local_versions").fetchall(),
                database.execute("SELECT * FROM local_paths").fetchall(),
            )

    before = original_rows()
    (reimported,) = library.import_paths("demo", [pdf])
    (again,) = library.import_paths("demo", [pdf])
    assert original_rows() == before
    assert reimported.status == again.status == "skipped"
    assert reimported.document_id == again.document_id == old.document_id
    assert reimported.version == again.version == old.version == 1
    assert reimported.publication_epoch == again.publication_epoch == old.publication_epoch + 1
    assert old.source_count == 1 and reimported.source_count == again.source_count == 2
    assert old.searchable_count == reimported.searchable_count == again.searchable_count == 1
    for source_id in ("page:1", "page:1:image:1"):
        key = SourceKey(document_id=old.document_id, version=1, source_id=source_id)
        historical = library.store.read_source(key, course_ids=["demo"])
        assert historical.status == "available" and historical.original == archived
        assert library.store.read_source(key, course_ids=["other"]).status == "not_accessible"


def test_a03_pptx_keeps_native_table_shape_path_units_and_bbox() -> None:
    original_hash = _sha256(A03_PPTX)

    parsed = parse_local_slides(A03_PPTX)

    assert [source.source_id for source in parsed.sources] == [
        "page:1",
        "page:1:shape:2",
        "page:1:shape:3",
        "page:2",
        "page:2:shape:2",
    ]
    table = next(source for source in parsed.sources if source.source_id == "page:1:shape:3")
    assert table.locator["shape_id_path"] == [3]
    assert table.locator["bbox"] == [0.1, 0.3, 0.9, 0.9]
    assert table.locator["coordinate_frame"] == "visible_page_norm_v1"
    assert table.locator["native_coordinate_frame"] == "pptx_slide_emu_v1"
    assert table.locator["columns"] == [
        "Filter",
        "Window [samples]",
        "Peak delay [samples]",
    ]
    assert ["Median", "5", "2"] in table.locator["rows"]
    assert parsed.sources[0].locator["render_status"] == "not_attempted"
    assert _sha256(A03_PPTX) == original_hash


def test_pptx_grouped_shape_keeps_path_but_omits_unverified_bbox(tmp_path: Path) -> None:
    pptx = tmp_path / "grouped.pptx"
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    group = slide.shapes.add_group_shape()
    child = group.shapes.add_textbox(Inches(1), Inches(1), Inches(3), Inches(1))
    child.text = "Nested native text"
    presentation.save(pptx)

    parsed = parse_local_slides(pptx)
    nested = next(source for source in parsed.sources if source.kind == "page_object")

    assert nested.source_id == (f"page:1:shape:{group.shape_id}.{child.shape_id}")
    assert nested.locator["shape_id_path"] == [group.shape_id, child.shape_id]
    assert nested.locator["position_basis"] == "group_transform_unverified"
    assert "bbox" not in nested.locator
    assert nested.text == "Nested native text"


def test_picture_only_pptx_is_needs_ocr_without_rendering(tmp_path: Path) -> None:
    image = tmp_path / "scan.png"
    Image.new("RGB", (120, 80), "white").save(image)
    pptx = tmp_path / "scan.pptx"
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    slide.shapes.add_picture(str(image), 0, 0, presentation.slide_width, presentation.slide_height)
    presentation.save(pptx)

    parsed = parse_local_slides(pptx)

    assert parsed.sources[0].state == "needs_ocr"
    assert parsed.sources[0].text == ""
    assert parsed.sources[0].locator["render_status"] == "not_attempted"
    assert parsed.sources[1].locator["object_type"] == "picture"


def test_off_slide_pptx_shape_omits_degenerate_clamped_bbox(tmp_path: Path) -> None:
    pptx = tmp_path / "off-slide.pptx"
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    shape = slide.shapes.add_textbox(Inches(20), Inches(1), Inches(2), Inches(1))
    shape.text = "Off slide"
    presentation.save(pptx)

    source = parse_local_slides(pptx).sources[1]

    assert source.text == "Off slide"
    assert source.locator["position_basis"] == "outside_slide_or_degenerate_after_clamp"
    assert "bbox" not in source.locator


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pdf_text(x: float, y: float, value: str) -> str:
    return f"BT /F1 12 Tf {x:g} {y:g} Td ({value}) Tj ET"


def _pdf_line(x1: float, y1: float, x2: float, y2: float) -> str:
    return f"1 w {x1:g} {y1:g} m {x2:g} {y2:g} l S"


def _pdf_grid(left: float, bottom: float, header: str, value: str) -> list[str]:
    middle_x = left + 90
    right = left + 180
    middle_y = bottom + 40
    top = bottom + 80
    commands = [
        _pdf_text(left + 8, middle_y + 12, header),
        _pdf_text(middle_x + 8, middle_y + 12, "Value"),
        _pdf_text(left + 8, bottom + 12, value),
        _pdf_text(middle_x + 8, bottom + 12, "1"),
    ]
    commands.extend(_pdf_line(left, y, right, y) for y in (bottom, middle_y, top))
    commands.extend(_pdf_line(x, bottom, x, top) for x in (left, middle_x, right))
    return commands


def _build_pdf(
    stream: bytes, *, crop: str = "0 0 612 792", rotation: int = 0, raster: bool = False
) -> bytes:
    objects: list[bytes] = [b""]

    def add(value: bytes) -> int:
        objects.append(value)
        return len(objects) - 1

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    xobjects = ""
    if raster:
        pixels = b"\x20\x40\xc0" * 16
        image = add(
            b"<< /Type /XObject /Subtype /Image /Width 4 /Height 4 /ColorSpace /DeviceRGB "
            b"/BitsPerComponent 8 /Length %d >>\nstream\n" % len(pixels) + pixels + b"\nendstream"
        )
        xobjects = f"/XObject << /Panel {image} 0 R >>"
    pages = add(b"")
    content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    page = add(
        (
            f"<< /Type /Page /Parent {pages} 0 R /MediaBox [0 0 612 792] "
            f"/CropBox [{crop}] /Rotate {rotation} "
            f"/Resources << /Font << /F1 {font} 0 R >> {xobjects} >> /Contents {content} 0 R >>"
        ).encode("ascii")
    )
    objects[pages] = f"<< /Type /Pages /Count 1 /Kids [{page} 0 R] >>".encode()
    catalog = add(f"<< /Type /Catalog /Pages {pages} 0 R >>".encode())
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_id, body in enumerate(objects[1:], start=1):
        offsets.append(len(output))
        output.extend(f"{object_id} 0 obj\n".encode())
        output.extend(body)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        (
            f"trailer\n<< /Size {len(objects)} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF\n"
        ).encode()
    )
    return bytes(output)
