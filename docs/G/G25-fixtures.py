"""Generate clean, self-contained CC BY 4.0 public fixtures for G25 geometry & ambiguity tests.

Produces:
1. docs/G/G25-fixtures/SensorLab-Layouts-CCBY4.pdf
   - Page 1: Normal side-by-side figures (left & right)
   - Page 2: Right figure whose y-axis label crosses the page midline (x < 0.5)
   - Page 3: Left-right mirrored layout (left figure label crosses midline x > 0.5)
   - Page 4: Top-bottom stacked figures
   - Page 5: Single centered figure
2. docs/G/G25-fixtures/SensorLab-Overlap-CCBY4.pptx
   - Slide 1: Two clearly overlapping picture shapes
3. docs/G/G25-判据与标注.json
   - Ground-truth annotations & expected behaviors for all cases.
"""

from __future__ import annotations

import ctypes
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches

OUT_DIR = Path(__file__).resolve().parent / "G25-fixtures"
PDF_PATH = OUT_DIR / "SensorLab-Layouts-CCBY4.pdf"
PPTX_PATH = OUT_DIR / "SensorLab-Overlap-CCBY4.pptx"
SPEC_PATH = Path(__file__).resolve().parent / "G25-判据与标注.json"

PAGE_WIDTH_PT = 612.0
PAGE_HEIGHT_PT = 792.0


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _draw_chart(
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_ticks: tuple[str, str, str],
    y_ticks: tuple[str, str, str],
    points: list[tuple[float, float]],
    color: str,
    y_label_in_left_margin: bool = False,
    right_legend: str | None = None,
    bar_mode: bool = False,
) -> Image.Image:
    """Render a crisp 900x600 plot where labels/values appear only inside the bitmap."""
    width, height = 900, 600
    image = Image.new("RGB", (width, height), "#fcfbf7")
    draw = ImageDraw.Draw(image)
    font_title = ImageFont.load_default(size=30)
    font_axis = ImageFont.load_default(size=24)
    font_tick = ImageFont.load_default(size=20)

    plot_left = 230 if y_label_in_left_margin else 155
    plot_right = 720 if right_legend else 850
    plot_top = 95
    plot_bottom = 485

    draw.rectangle((6, 6, width - 7, height - 7), outline="#d0d4d8", width=2)
    draw.text(
        ((plot_left + plot_right) // 2, 42),
        title,
        fill="#101819",
        font=font_title,
        anchor="mm",
    )

    # Grid & axes
    for frac in (0.0, 0.5, 1.0):
        gx = int(plot_left + frac * (plot_right - plot_left))
        gy = int(plot_bottom - frac * (plot_bottom - plot_top))
        draw.line((gx, plot_top, gx, plot_bottom), fill="#e5e7eb", width=2)
        draw.line((plot_left, gy, plot_right, gy), fill="#e5e7eb", width=2)

    draw.line(
        (plot_left, plot_top, plot_left, plot_bottom, plot_right, plot_bottom),
        fill="#1f2933",
        width=5,
    )

    # Ticks
    for frac, label in zip((0.0, 0.5, 1.0), x_ticks, strict=True):
        tx = int(plot_left + frac * (plot_right - plot_left))
        draw.text((tx, plot_bottom + 24), label, fill="#1f2933", font=font_tick, anchor="mm")
    for frac, label in zip((0.0, 0.5, 1.0), y_ticks, strict=True):
        ty = int(plot_bottom - frac * (plot_bottom - plot_top))
        draw.text((plot_left - 14, ty), label, fill="#1f2933", font=font_tick, anchor="rm")

    # Axis labels
    draw.text(
        ((plot_left + plot_right) // 2, 555),
        x_label,
        fill="#101819",
        font=font_axis,
        anchor="mm",
    )
    # Draw y-label vertically-ish (stacked or horizontal in wide left margin)
    if y_label_in_left_margin:
        lines = y_label.split(" ")
        start_y = (plot_top + plot_bottom) // 2 - (len(lines) - 1) * 16
        for idx, word in enumerate(lines):
            draw.text((18, start_y + idx * 32), word, fill="#101819", font=font_axis, anchor="lm")
    else:
        y_img = Image.new("RGB", (380, 42), "#fcfbf7")
        y_draw = ImageDraw.Draw(y_img)
        y_draw.text((190, 21), y_label, fill="#101819", font=font_axis, anchor="mm")
        rotated = y_img.rotate(90, expand=True)
        image.paste(rotated, (18, (plot_top + plot_bottom) // 2 - rotated.height // 2))

    # Plot series
    if bar_mode:
        bar_w = max(24, (plot_right - plot_left) // (len(points) * 2))
        for px, py in points:
            cx = int(plot_left + px * (plot_right - plot_left))
            cy = int(plot_bottom - py * (plot_bottom - plot_top))
            draw.rectangle((cx - bar_w // 2, cy, cx + bar_w // 2, plot_bottom - 3), fill=color)
    else:
        coords = [
            (
                int(plot_left + px * (plot_right - plot_left)),
                int(plot_bottom - py * (plot_bottom - plot_top)),
            )
            for px, py in points
        ]
        draw.line(coords, fill=color, width=8)
        for cx, cy in coords:
            draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=color)

    if right_legend:
        draw.rectangle(
            (plot_right + 16, 130, width - 16, 220),
            outline="#1f2933",
            fill="#f3f4f6",
            width=2,
        )
        draw.line((plot_right + 28, 175, plot_right + 68, 175), fill=color, width=6)
        draw.text((plot_right + 78, 175), right_legend, fill="#101819", font=font_tick, anchor="lm")

    return image


def _add_pdf_text(
    doc: pdfium.PdfDocument,
    page: pdfium.PdfPage,
    text: str,
    x_pt: float,
    y_pt: float,
    size: float = 13.0,
) -> None:
    font = pdfium_raw.FPDFText_LoadStandardFont(doc.raw, b"Helvetica")
    obj = pdfium_raw.FPDFPageObj_CreateTextObj(doc.raw, font, size)
    encoded = (text + "\x00").encode("utf-16-le")
    buf = (ctypes.c_ushort * (len(encoded) // 2)).from_buffer_copy(encoded)
    pdfium_raw.FPDFText_SetText(obj, buf)
    pdfium_raw.FPDFPageObj_Transform(obj, 1, 0, 0, 1, x_pt, y_pt)
    pdfium_raw.FPDFPage_InsertObject(page.raw, obj)


def _add_pdf_image(
    doc: pdfium.PdfDocument,
    page: pdfium.PdfPage,
    pil_img: Image.Image,
    bbox: tuple[float, float, float, float],
) -> None:
    x0, y0, x1, y1 = bbox
    w_pt = (x1 - x0) * PAGE_WIDTH_PT
    h_pt = (y1 - y0) * PAGE_HEIGHT_PT
    left_pt = x0 * PAGE_WIDTH_PT
    bottom_pt = (1.0 - y1) * PAGE_HEIGHT_PT
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    img_obj = pdfium.PdfImage.new(doc)
    img_obj.load_jpeg(io.BytesIO(buf.getvalue()))
    img_obj.set_matrix(pdfium.PdfMatrix(w_pt, 0, 0, h_pt, left_pt, bottom_pt))
    page.insert_obj(img_obj)


def build_fixtures() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument.new()

    pages_spec = [
        {
            "page": 1,
            "layout": "normal_dual",
            "header": "SensorLab Benchmark (CC BY 4.0) - Page 1: Normal Side-by-Side Subplots",
            "caption": "Figure 1: Thermal drift vs voltage ripple under nominal load.",
            "objects": [
                {
                    "expected_source_id": "page:1:image:1",
                    "side": "left",
                    "bbox": (0.07, 0.28, 0.45, 0.64),
                    "chart": dict(
                        title="Thermal Drift A",
                        x_label="Elapsed Time [min]",
                        y_label="Core Temp [C]",
                        x_ticks=("0", "30", "60"),
                        y_ticks=("25", "35", "45"),
                        points=[(0.05, 0.15), (0.3, 0.35), (0.6, 0.62), (0.95, 0.88)],
                        color="#2563eb",
                    ),
                },
                {
                    "expected_source_id": "page:1:image:2",
                    "side": "right",
                    "bbox": (0.55, 0.28, 0.93, 0.64),
                    "chart": dict(
                        title="Voltage Ripple B",
                        x_label="Load Current [A]",
                        y_label="Ripple [mV]",
                        x_ticks=("0", "6", "12"),
                        y_ticks=("10", "50", "90"),
                        points=[(0.05, 0.18), (0.35, 0.32), (0.7, 0.68), (0.95, 0.85)],
                        color="#0d9488",
                    ),
                },
            ],
        },
        {
            "page": 2,
            "layout": "right_label_crossing",
            "header": "SensorLab Benchmark (CC BY 4.0) - Page 2: Wide Y-Label Crossing Midline",
            "caption": (
                "Figure 2: Queue wait bars (left) and throughput curve "
                "with left-margin label crossing x=0.5 (right)."
            ),
            "objects": [
                {
                    "expected_source_id": "page:2:image:1",
                    "side": "left",
                    "bbox": (0.07, 0.28, 0.40, 0.64),
                    "chart": dict(
                        title="Latency Histogram",
                        x_label="Queue Depth [req]",
                        y_label="Wait [ms]",
                        x_ticks=("4", "16", "32"),
                        y_ticks=("0", "10", "20"),
                        points=[(0.2, 0.25), (0.5, 0.55), (0.8, 0.82)],
                        color="#6366f1",
                        bar_mode=True,
                    ),
                },
                {
                    "expected_source_id": "page:2:image:2",
                    "side": "right",
                    "bbox": (0.43, 0.28, 0.88, 0.64),
                    "chart": dict(
                        title="Throughput Scaling",
                        x_label="Batch Window [us]",
                        y_label="Throughput [Gbps]",
                        x_ticks=("10", "50", "100"),
                        y_ticks=("1.2", "3.0", "4.8"),
                        points=[(0.05, 0.20), (0.35, 0.48), (0.65, 0.74), (0.95, 0.90)],
                        color="#d97706",
                        y_label_in_left_margin=True,
                    ),
                },
            ],
        },
        {
            "page": 3,
            "layout": "left_right_mirrored",
            "header": "SensorLab Benchmark (CC BY 4.0) - Page 3: Mirrored Layout Crossing Midline",
            "caption": (
                "Figure 3: Carrier gain with right legend crossing x=0.5 (left) "
                "and bias noise floor (right)."
            ),
            "objects": [
                {
                    "expected_source_id": "page:3:image:1",
                    "side": "left",
                    "bbox": (0.12, 0.28, 0.57, 0.64),
                    "chart": dict(
                        title="Mirror Signal Left",
                        x_label="Carrier Freq [MHz]",
                        y_label="Gain [dB]",
                        x_ticks=("100", "300", "500"),
                        y_ticks=("-20", "-10", "0"),
                        points=[(0.05, 0.88), (0.35, 0.62), (0.7, 0.35), (0.95, 0.14)],
                        color="#dc2626",
                        right_legend="Gain",
                    ),
                },
                {
                    "expected_source_id": "page:3:image:2",
                    "side": "right",
                    "bbox": (0.60, 0.28, 0.93, 0.64),
                    "chart": dict(
                        title="Mirror Noise Right",
                        x_label="Bias [V]",
                        y_label="Floor [uV]",
                        x_ticks=("0", "2.5", "5.0"),
                        y_ticks=("5", "20", "35"),
                        points=[(0.05, 0.25), (0.35, 0.30), (0.7, 0.42), (0.95, 0.60)],
                        color="#7c3aed",
                    ),
                },
            ],
        },
        {
            "page": 4,
            "layout": "top_bottom_stacked",
            "header": "SensorLab Benchmark (CC BY 4.0) - Page 4: Vertically Stacked Subplots",
            "caption": "Figure 4: Upper spectral absorbance trace and lower phase error trace.",
            "objects": [
                {
                    "expected_source_id": "page:4:image:1",
                    "side": "top",
                    "bbox": (0.20, 0.14, 0.80, 0.45),
                    "chart": dict(
                        title="Upper Spectral Trace",
                        x_label="Wavenumber [cm-1]",
                        y_label="Absorbance [AU]",
                        x_ticks=("400", "1000", "1600"),
                        y_ticks=("0.0", "0.5", "1.0"),
                        points=[
                            (0.05, 0.15),
                            (0.35, 0.45),
                            (0.50, 0.92),
                            (0.65, 0.42),
                            (0.95, 0.12),
                        ],
                        color="#0284c7",
                    ),
                },
                {
                    "expected_source_id": "page:4:image:2",
                    "side": "bottom",
                    "bbox": (0.20, 0.54, 0.80, 0.85),
                    "chart": dict(
                        title="Lower Phase Error",
                        x_label="Sample Index [k]",
                        y_label="Phase Error [deg]",
                        x_ticks=("0", "100", "200"),
                        y_ticks=("-15", "0", "+15"),
                        points=[
                            (0.05, 0.85),
                            (0.25, 0.22),
                            (0.50, 0.72),
                            (0.75, 0.38),
                            (0.95, 0.52),
                        ],
                        color="#b91c1c",
                    ),
                },
            ],
        },
        {
            "page": 5,
            "layout": "single_centered",
            "header": "SensorLab Benchmark (CC BY 4.0) - Page 5: Single Centered Calibration Plot",
            "caption": "Figure 5: Single centered calibration residual plot.",
            "objects": [
                {
                    "expected_source_id": "page:5:image:1",
                    "side": "center",
                    "bbox": (0.20, 0.28, 0.80, 0.68),
                    "chart": dict(
                        title="Centered Calibration Curve",
                        x_label="Step Count [N]",
                        y_label="Residual [ppm]",
                        x_ticks=("0", "250", "500"),
                        y_ticks=("2", "10", "18"),
                        points=[(0.05, 0.90), (0.3, 0.52), (0.65, 0.24), (0.95, 0.12)],
                        color="#15803d",
                    ),
                },
            ],
        },
    ]

    for page_spec in pages_spec:
        page = doc.new_page(PAGE_WIDTH_PT, PAGE_HEIGHT_PT)
        _add_pdf_text(doc, page, page_spec["header"], 42.0, 742.0, size=13.0)
        _add_pdf_text(doc, page, page_spec["caption"], 42.0, 718.0, size=10.5)
        for obj in page_spec["objects"]:
            chart_img = _draw_chart(**obj["chart"])
            _add_pdf_image(doc, page, chart_img, obj["bbox"])
        page.gen_content()

    doc.save(PDF_PATH)

    # Build PPTX with two clearly overlapping picture shapes on slide 1
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    if slide.shapes.title:
        slide.shapes.title.text = "SensorLab Overlapping Candidates (CC BY 4.0)"

    overlap_a = _draw_chart(
        title="Overlap Alpha",
        x_label="Channel [idx]",
        y_label="Power [dBm]",
        x_ticks=("1", "4", "8"),
        y_ticks=("-40", "-20", "0"),
        points=[(0.1, 0.3), (0.5, 0.7), (0.9, 0.5)],
        color="#1d4ed8",
    )
    overlap_b = _draw_chart(
        title="Overlap Beta",
        x_label="Offset [ns]",
        y_label="Jitter [ps]",
        x_ticks=("0", "50", "100"),
        y_ticks=("5", "25", "45"),
        points=[(0.1, 0.2), (0.5, 0.5), (0.9, 0.85)],
        color="#be123c",
    )
    buf_a, buf_b = io.BytesIO(), io.BytesIO()
    overlap_a.save(buf_a, format="PNG")
    overlap_b.save(buf_b, format="PNG")
    buf_a.seek(0)
    buf_b.seek(0)
    # Overlapping shapes (~49% area overlap): [0.15, 0.22, 0.65, 0.72] and [0.38, 0.25, 0.88, 0.75]
    slide.shapes.add_picture(
        buf_a, Inches(1.5), Inches(1.65), width=Inches(5.0), height=Inches(3.75)
    )
    slide.shapes.add_picture(
        buf_b, Inches(3.8), Inches(1.875), width=Inches(5.0), height=Inches(3.75)
    )
    prs.save(PPTX_PATH)

    metadata = {
        "license": "CC BY 4.0",
        "pdf": {
            "file": str(PDF_PATH.relative_to(SPEC_PATH.parent.parent.parent)),
            "sha256": _sha256_bytes(PDF_PATH.read_bytes()),
            "bytes": PDF_PATH.stat().st_size,
            "pages": [
                {
                    "page": p["page"],
                    "layout": p["layout"],
                    "objects": [
                        {
                            "expected_source_id": o["expected_source_id"],
                            "side": o["side"],
                            "bbox": list(o["bbox"]),
                            "title": o["chart"]["title"],
                            "x_label": o["chart"]["x_label"],
                            "y_label": o["chart"]["y_label"],
                            "x_ticks": list(o["chart"]["x_ticks"]),
                            "y_ticks": list(o["chart"]["y_ticks"]),
                        }
                        for o in p["objects"]
                    ],
                }
                for p in pages_spec
            ],
        },
        "pptx_overlap": {
            "file": str(PPTX_PATH.relative_to(SPEC_PATH.parent.parent.parent)),
            "sha256": _sha256_bytes(PPTX_PATH.read_bytes()),
            "bytes": PPTX_PATH.stat().st_size,
            "shapes": [
                {"side": "left_overlapping", "bbox": [0.15, 0.22, 0.65, 0.72]},
                {"side": "right_overlapping", "bbox": [0.38, 0.25, 0.88, 0.75]},
            ],
        },
    }
    SPEC_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


if __name__ == "__main__":
    info = build_fixtures()
    print(json.dumps(info, ensure_ascii=False, indent=2))
