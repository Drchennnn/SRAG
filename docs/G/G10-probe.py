"""Generate synthetic public G10 fixtures and collect isolated G09 diagnostics."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import subprocess
import zlib
from pathlib import Path
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import nbformat
import pypdfium2 as pdfium
import uvicorn
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches

from app.rag.parsers.local_slides import parse_local_slides

BASELINE = "e921fa87f48a4e67c58010181a187c6e0e12b833"
COURSE = "drum-lab"
FIGURE_QUESTION = "Drum-Lab.pdf 的右侧子图，横轴的单位是什么？"
FIGURE_INFORMAL = "Drum-Lab.pdf 的右侧图，横轴的单位是什么？"
ROTATED_QUESTION = "Drum-Rotated.pdf 右侧子图的横轴单位是什么？"
TABLE_QUESTION = (
    "请核对 Drum-Readings.pptx 的原表。表格中 Frequency 这一列的单位是什么？Cedar 对应多少？"
)
TABLE_INFORMAL = "请核对 Drum-Readings.pptx 的原表，Frequency 这一列的单位是什么？Cedar 对应多少？"
COMPARE_QUESTION = "比较 Drum-Trend.ipynb 上一次看到的旧版和现在的当前版 Cell 1 保存输出。"
MISSING_QUESTION = "Drum-Scan.pdf 左上角的子图横轴单位是什么？"
AMBIGUOUS_QUESTION = "Drum-Lab.pdf 那张子图的横轴单位是什么？"
ONE_FIGURE_QUESTION = "Drum-Trend.ipynb 的输出图片横轴单位是什么？"
TABLE_ROWS = (
    ("Series", "Frequency [Hz]", "Current [A]"),
    ("Cedar", "13.75", "6.25"),
    ("Ash", "17.50", "8.50"),
)
FIGURES = {
    "Drum-Lab.pdf": {
        "page": 1,
        "media": (600, 400),
        "crop": (0, 0, 600, 400),
        "rotation": 0,
        "objects": (
            ("Phase [degrees]", (60, 103, 260, 263), "#125827"),
            ("Time [milliseconds]", (340, 103, 540, 263), "#843048"),
        ),
    },
    "Drum-Rotated.pdf": {
        "page": 1,
        "media": (640, 520),
        "crop": (35, 30, 600, 480),
        "rotation": 90,
        "objects": (
            ("Mass [grams]", (153, 67, 353, 217), "#294287"),
            ("Speed [rpm]", (153, 268, 353, 418), "#965412"),
        ),
    },
    "Drum-Scan.pdf": {
        "page": 1,
        "media": (600, 400),
        "crop": (0, 0, 600, 400),
        "rotation": 0,
        "objects": (("Overview [unscaled]", (0, 0, 600, 400), "#303030"),),
    },
}

spec = importlib.util.spec_from_file_location("g05_probe", Path(__file__).with_name("G05-probe.py"))
assert spec and spec.loader
g05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g05)


def panel(axis: str, color: str, rotation: int) -> Image.Image:
    image = Image.new("RGB", (640, 450), "#fffdf6")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=38)
    draw.rectangle((8, 8, 631, 441), outline=color, width=12)
    draw.line(((72, 62), (72, 338), (574, 338)), fill="#171717", width=6)
    draw.line(((73, 294), (220, 251), (342, 276), (555, 85)), fill=color, width=11)
    draw.text((320, 394), axis, fill="#171717", font=font, anchor="mm")
    return image.rotate(rotation, expand=True)


def pdf_stream(dictionary: str, data: bytes) -> bytes:
    return f"<< {dictionary} /Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream"


def write_pdf(path: Path, layout: dict) -> None:
    width, height = layout["media"]
    crop = " ".join(map(str, layout["crop"]))
    images = []
    commands = []
    if path.name != "Drum-Scan.pdf":
        commands.append("BT /F0 18 Tf 60 340 Td (Drum modulation lab) Tj ET")
    for index, (axis, (x0, y0, x1, y1), color) in enumerate(layout["objects"]):
        raster = panel(axis, color, layout["rotation"])
        images.append(
            pdf_stream(
                f"/Type /XObject /Subtype /Image /Width {raster.width} /Height {raster.height} "
                "/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode",
                zlib.compress(raster.tobytes()),
            )
        )
        commands.append(f"q {x1 - x0} 0 0 {y1 - y0} {x0} {y0} cm /Im{index} Do Q")
    references = " ".join(f"/Im{index} {index + 6} 0 R" for index in range(len(images)))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            f"/CropBox [{crop}] /Rotate {layout['rotation']} /Contents 4 0 R "
            f"/Resources << /XObject << {references} >> /Font << /F0 5 0 R >> >> >>"
        ).encode(),
        pdf_stream("", "\n".join(commands).encode()),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        *images,
    ]
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for index, content in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode() + content + b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    path.write_bytes(output)


def build(directory: Path) -> dict:
    fixtures = directory / "fixtures"
    fixtures.mkdir()
    for filename, layout in FIGURES.items():
        write_pdf(fixtures / filename, layout)
    for version, value in ((1, "11.25 kHz"), (2, "19.75 kHz")):
        destination = fixtures / f"v{version}"
        destination.mkdir()
        png = io.BytesIO()
        panel("Cycle [minutes]", "#704318", 0).save(png, "PNG")
        g05.notebook(
            destination / "Drum-Trend.ipynb",
            [
                nbformat.v4.new_code_cell(
                    "plot_resonance_drift()",
                    id="resonance-drift",
                    execution_count=1,
                    outputs=[
                        nbformat.v4.new_output(
                            "stream",
                            name="stdout",
                            text=f"Resonance drift measured at {value}.\n",
                        ),
                        nbformat.v4.new_output(
                            "display_data",
                            data={"image/png": base64.b64encode(png.getvalue()).decode()},
                        ),
                    ],
                )
            ],
        )
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    table = slide.shapes.add_table(3, 3, Inches(1), Inches(2), Inches(8), Inches(2))
    table.name = "Drum calibration readings"
    for row_index, row in enumerate(TABLE_ROWS):
        for col_index, value in enumerate(row):
            table.table.cell(row_index, col_index).text = value
    package = io.BytesIO()
    deck.save(package)
    with ZipFile(package) as original, ZipFile(fixtures / "Drum-Readings.pptx", "w") as stable:
        for member in sorted(original.namelist()):
            stable.writestr(ZipInfo(member), original.read(member), compress_type=ZIP_DEFLATED)
    manifest = {
        "figures": FIGURES,
        "questions": {
            "normal": FIGURE_QUESTION,
            "informal": FIGURE_INFORMAL,
            "rotated": ROTATED_QUESTION,
            "table": TABLE_QUESTION,
            "table_informal": TABLE_INFORMAL,
            "compare": COMPARE_QUESTION,
            "missing": MISSING_QUESTION,
            "ambiguous": AMBIGUOUS_QUESTION,
            "one_figure": ONE_FIGURE_QUESTION,
        },
        "table_cells": TABLE_ROWS,
        "fixture_sha256": {
            str(path.relative_to(fixtures)): g05.sha(path.read_bytes())
            for path in sorted(fixtures.rglob("*"))
            if path.is_file()
        },
        # Required by the reused offline G05 server guard. No provider call uses this hash.
        "image_sha256": {"1": {g05.RIGHT: g05.sha(png.getvalue())}},
    }
    g05.write(directory / "fixtures.json", manifest)
    return manifest


def visible_rect(
    layout: dict, bounds: tuple[int, int, int, int]
) -> tuple[float, float, float, float]:
    left, bottom, right, top = layout["crop"]
    x0, y0, x1, y1 = bounds
    if layout["rotation"] == 90:
        return (
            (y0 - bottom) / (top - bottom),
            (x0 - left) / (right - left),
            (y1 - bottom) / (top - bottom),
            (x1 - left) / (right - left),
        )
    return (
        (x0 - left) / (right - left),
        (top - y1) / (top - bottom),
        (x1 - left) / (right - left),
        (top - y0) / (top - bottom),
    )


def direct_diagnostics(directory: Path) -> dict:
    fixtures = directory / "fixtures"
    rendered = []
    for filename, layout in FIGURES.items():
        path = fixtures / filename
        parsed = parse_local_slides(path)
        with pdfium.PdfDocument(path) as doc:
            page = doc[0]
            crop_box = page.get_bbox()
            rotation = page.get_rotation()
            raw_images = [
                {"bounds": obj.get_bounds(), "matrix": vars(obj.get_matrix())}
                for obj in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE], max_depth=1)
            ]
            assert crop_box == layout["crop"] and rotation == layout["rotation"]
            assert [item["bounds"] for item in raw_images] == [
                item[1] for item in layout["objects"]
            ]
            started = perf_counter()
            bitmap = page.render(scale=2)
            render_ms = round((perf_counter() - started) * 1000, 3)
            try:
                pixels = bitmap.to_pil().copy()
            finally:
                bitmap.close()
                page.close()
        crops = []
        for index, (axis, bounds, _) in enumerate(layout["objects"], start=1):
            rectangle = visible_rect(layout, bounds)
            size = (pixels.width, pixels.height, pixels.width, pixels.height)
            bounds_px = tuple(
                round(value * limit) for value, limit in zip(rectangle, size, strict=True)
            )
            png = directory / f"{path.stem}-diagnostic-{index}.png"
            pixels.crop(bounds_px).save(png)
            crops.append(
                {
                    "native_bounds": bounds,
                    "visible_rect": rectangle,
                    "sha256": g05.sha(png.read_bytes()),
                    "path": png.name,
                    "fixture_truth": axis,
                }
            )
        rendered.append(
            {
                "file": filename,
                "page": 1,
                "rotation": rotation,
                "crop_box": crop_box,
                "pdfium_image_objects": raw_images,
                "pdfium_render_ms": render_ms,
                "registered": [
                    {"source_id": s.source_id, "kind": s.kind, "locator": s.locator}
                    for s in parsed.sources
                ],
                "crops": crops,
            }
        )
    table = parse_local_slides(fixtures / "Drum-Readings.pptx")
    return {
        "exact_object_and_crop": {
            "execution": "original PDFium render and exact fixture rectangle; outside product turn",
            "pdfs": rendered,
        },
        "verified_evidence_reference": {
            "execution": "manual source and parser checks; no measured model answer",
            "expected_figure": "milliseconds",
            "expected_rotated_figure": "rpm",
            "expected_table": "13.75 Hz",
            "expected_v1": "11.25 kHz",
            "expected_v2": "19.75 kHz",
            "table_native_sources": [
                {"source_id": s.source_id, "locator": s.locator, "text": s.text}
                for s in table.sources
                if s.kind == "page_object"
            ],
        },
    }


def verify(directory: Path, subject: str) -> None:
    http = [json.loads(line) for line in (directory / "http.jsonl").read_text().splitlines()]
    with (directory / "fixtures.json").open() as stream:
        manifest = json.load(stream)
    cases = {}
    for label in (
        "normal",
        "page",
        "informal",
        "rotated",
        "table",
        "table_page",
        "table_informal",
        "missing",
        "ambiguous",
        "one_figure",
        "old",
        "compare",
    ):
        browser = g05.load(directory, label)
        matches = [
            event
            for event in http
            if event["category"] == "turn"
            and isinstance(event["response"], dict)
            and event["response"].get("message", {}).get("message_id") == browser["message_id"]
        ]
        assert len(matches) == 1, (label, browser["message_id"])
        (entry,) = matches
        assert entry["status"] == 200 and entry["request"].get("anchor") is None
        response = entry["response"]
        cases[label] = {"request": entry["request"], "browser": browser, "server": response}
    for normal, focused in (("normal", "page"), ("table", "table_page")):
        assert cases[normal]["request"]["text"] == cases[focused]["request"]["text"]
        assert cases[normal]["browser"]["focus"] is None
        assert cases[focused]["browser"]["focus"].startswith("第 1 页")
    assert cases["compare"]["request"]["kind"] == "ask"
    citation_opens = [e for e in http if e["category"] == "citation"]
    assert {1, 2} <= {
        item["response"].get("source", {}).get("version")
        for item in citation_opens
        if item["status"] == 200
    }
    provider = directory / "provider.jsonl"
    attempts = provider.read_text().splitlines() if provider.exists() else []
    diagnostics = direct_diagnostics(directory)
    table = diagnostics["exact_object_and_crop"]
    table["table"] = {
        "execution": "actual native table rendition in a separate display-alternative turn",
        "same_question_isolation": (
            "the targeted turn supplies the exact object and native read already; "
            "it has no supported input for attaching a display rendition"
        ),
        "render_status": "unavailable",
    }
    table_block = next(
        (
            block
            for block in cases["table_informal"]["server"]["message"]["blocks"]
            if block["type"] == "source_image" and block["rendition_kind"] == "object_crop"
        ),
        None,
    )
    if table_block:
        path = table_block["media_url"].partition("?")[0]
        requests = [item for item in http if item["category"] == "media" and item["path"] == path]
        assert requests and all(item["status"] == 200 for item in requests)
        digest = requests[0]["response"]["sha256"]
        for path in (directory / "data" / "source_evidence_cache").glob("*.json"):
            cache = json.loads(path.read_text())
            if cache["sha256"] != digest:
                continue
            image_bytes = base64.b64decode(cache["data_base64"])
            assert g05.sha(image_bytes) == digest
            crop_path = directory / "Drum-Readings-diagnostic.png"
            crop_path.write_bytes(image_bytes)
            table["table"].update(
                {
                    "display": table_block,
                    "render_status": "available",
                    "sha256": digest,
                    "path": crop_path.name,
                    "renderer": cache["renderer"],
                    "renderer_version": cache["renderer_version"],
                }
            )
            break
        assert table["table"]["render_status"] == "available"
    summary = {
        "subject": subject,
        "measurement": "ordinary Chromium workspace, offline",
        "browser": g05.load(directory, "browser"),
        "fixture_sha256": manifest["fixture_sha256"],
        "provider_attempts_blocked": len(attempts),
        "media_fetches": [e for e in http if e["category"] == "media"],
        "provider_calls_actual": 0,
        "traces": cases,
        "citation_opens": citation_opens,
        **diagnostics,
    }
    g05.write(directory / "G10-基线实测.json", summary)
    print(directory / "G10-基线实测.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--port", type=int)
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--fixtures", action="store_true")
    parser.add_argument("--soffice")
    parser.add_argument("--subject", default=BASELINE)
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    if args.port:
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        assert actual == args.subject, (actual, args.subject)
        args.real_vision = False
        args.soffice = args.soffice or None
        g05.COURSES = [COURSE]
        g05.build = build
        uvicorn.run(
            g05.make_app(args.data_dir, args), host="127.0.0.1", port=args.port, log_level="warning"
        )
    elif args.verify:
        verify(args.data_dir, args.subject)
    else:
        build(args.data_dir)


if __name__ == "__main__":
    main()
