#!/usr/bin/env python3
"""Generate deterministic G29 independent blind-gate PPTX and multi-version Notebook fixtures.

Strictly isolated inside docs/G/G29-gate/fixtures/. Does not modify any production code.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"
NB_DIR = FIXTURES / "notebook"


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
        if bold
        else [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    fill: tuple[int, int, int],
    width: int = 4,
    dash: int = 16,
    gap: int = 10,
) -> None:
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    dist = (dx * dx + dy * dy) ** 0.5
    if dist == 0:
        return
    ux, uy = dx / dist, dy / dist
    pos = 0.0
    while pos < dist:
        end = min(pos + dash, dist)
        draw.line(
            [(x0 + ux * pos, y0 + uy * pos), (x0 + ux * end, y0 + uy * end)],
            fill=fill,
            width=width,
        )
        pos += dash + gap


def _draw_line_chart(
    *,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    x_ticks: list[tuple[float, str]],
    y_ticks: list[tuple[float, str]],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    series: list[dict],
    threshold: dict | None = None,
    callout: str | None = None,
    legend_pos: str = "top_right",
    width: int = 1200,
    height: int = 720,
) -> bytes:
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    f_title = _load_font(30, bold=True)
    f_sub = _load_font(20, bold=False)
    f_axis = _load_font(24, bold=True)
    f_tick = _load_font(20, bold=False)
    f_leg = _load_font(22, bold=True)
    f_note = _load_font(22, bold=True)

    # Outer border
    draw.rectangle([2, 2, width - 3, height - 3], outline=(210, 215, 222), width=2)

    # Title & subtitle
    draw.text((70, 24), title, fill=(17, 24, 39), font=f_title)
    draw.text((70, 62), subtitle, fill=(75, 85, 99), font=f_sub)

    # Plot box
    left, top, right, bottom = 140, 115, width - 60, height - 115
    draw.rectangle([left, top, right, bottom], fill=(250, 251, 253), outline=(31, 41, 55), width=3)

    xmin, xmax = x_range
    ymin, ymax = y_range

    def px(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * (right - left)

    def py(y: float) -> float:
        return bottom - (y - ymin) / (ymax - ymin) * (bottom - top)

    # Grid and ticks
    for xv, label in x_ticks:
        x = px(xv)
        draw.line([(x, top), (x, bottom)], fill=(226, 232, 240), width=1)
        draw.line([(x, bottom), (x, bottom + 8)], fill=(31, 41, 55), width=2)
        bbox = draw.textbbox((0, 0), label, font=f_tick)
        tw = bbox[2] - bbox[0]
        draw.text((x - tw / 2, bottom + 14), label, fill=(31, 41, 55), font=f_tick)

    for yv, label in y_ticks:
        y = py(yv)
        draw.line([(left, y), (right, y)], fill=(226, 232, 240), width=1)
        draw.line([(left - 8, y), (left, y)], fill=(31, 41, 55), width=2)
        bbox = draw.textbbox((0, 0), label, font=f_tick)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((left - tw - 16, y - th / 2 - 2), label, fill=(31, 41, 55), font=f_tick)

    # Axis labels
    xb = draw.textbbox((0, 0), x_label, font=f_axis)
    draw.text(
        ((left + right) / 2 - (xb[2] - xb[0]) / 2, height - 56),
        x_label,
        fill=(17, 24, 39),
        font=f_axis,
    )
    # Rotated Y-axis label
    yb = draw.textbbox((0, 0), y_label, font=f_axis)
    ytxt = Image.new("RGBA", (yb[2] - yb[0] + 12, yb[3] - yb[1] + 12), (255, 255, 255, 0))
    ImageDraw.Draw(ytxt).text((4, 4), y_label, fill=(17, 24, 39), font=f_axis)
    ytxt_rot = ytxt.rotate(90, expand=True)
    img.paste(ytxt_rot, (24, int((top + bottom) / 2 - ytxt_rot.height / 2)), ytxt_rot)

    # Optional horizontal threshold line
    if threshold is not None:
        ty = py(threshold["y"])
        _draw_dashed_line(draw, (left, ty), (right, ty), fill=threshold["color"], width=3, dash=12, gap=8)
        draw.text(
            (left + 16, ty - 28),
            threshold["label"],
            fill=threshold["color"],
            font=f_sub,
        )

    # Data series
    for s in series:
        pts = [(px(x), py(y)) for x, y in s["points"]]
        color = s["color"]
        dashed = s.get("dashed", False)
        for i in range(len(pts) - 1):
            if dashed:
                _draw_dashed_line(draw, pts[i], pts[i + 1], fill=color, width=5)
            else:
                draw.line([pts[i], pts[i + 1]], fill=color, width=5)
        for x, y in pts:
            r = 7
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(255, 255, 255), width=2)
        if "peak_label" in s:
            px0, py0 = px(s["peak_point"][0]), py(s["peak_point"][1])
            dx, dy = s.get("peak_offset", (-240, -42))
            draw.rectangle(
                [px0 + dx - 8, py0 + dy - 6, px0 + dx + 310, py0 + dy + 30],
                fill=(255, 255, 255),
                outline=color,
                width=2,
            )
            draw.text((px0 + dx, py0 + dy), s["peak_label"], fill=color, font=f_note)

    # Legend box
    leg_w = 340
    leg_h = 20 + 36 * len(series)
    lx0 = right - leg_w - 20
    ly0 = (bottom - leg_h - 20) if legend_pos == "bottom_right" else (top + 18)
    draw.rectangle([lx0, ly0, lx0 + leg_w, ly0 + leg_h], fill=(255, 255, 255), outline=(148, 163, 184), width=2)
    for idx, s in enumerate(series):
        cy = ly0 + 26 + idx * 36
        if s.get("dashed"):
            _draw_dashed_line(draw, (lx0 + 16, cy), (lx0 + 66, cy), fill=s["color"], width=5, dash=10, gap=6)
        else:
            draw.line([(lx0 + 16, cy), (lx0 + 66, cy)], fill=s["color"], width=5)
        draw.text((lx0 + 78, cy - 13), s["name"], fill=(17, 24, 39), font=f_leg)

    if callout:
        cb = draw.textbbox((0, 0), callout, font=f_note)
        cw = cb[2] - cb[0]
        cx = left + 24
        cy = bottom - 48
        draw.rectangle([cx - 10, cy - 6, cx + cw + 14, cy + 32], fill=(254, 249, 195), outline=(202, 138, 4), width=2)
        draw.text((cx, cy), callout, fill=(113, 63, 18), font=f_note)

    return _to_png_bytes(img)


def _draw_dtm_diagram() -> bytes:
    """Block diagram for PPTX Slide 4 containing voltage levels exclusively inside the image."""
    w, h = 960, 620
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    f_title = _load_font(26, bold=True)
    f_box = _load_font(22, bold=True)
    f_sub = _load_font(19, bold=False)

    draw.rectangle([2, 2, w - 3, h - 3], outline=(203, 213, 225), width=2)
    draw.text((30, 18), "DTM Hysteresis & DVFS Voltage Rail Diagram", fill=(15, 23, 42), font=f_title)

    boxes = [
        (40, 85, 420, 195, (219, 234, 254), (37, 99, 235), "1. L2 Cache Controller", "64B line | 8-way set-assoc"),
        (520, 85, 920, 195, (220, 252, 231), (22, 163, 74), "2. Victim Buffer (8-way)", "Eviction FIFO/LRU bypass"),
        (40, 275, 420, 395, (254, 243, 199), (217, 119, 6), "3. On-Die Thermal Sensor", "Sample interval dt = 2 ms"),
        (520, 275, 920, 395, (254, 226, 226), (220, 38, 38), "4. DVFS Governor Rails", "P0: 1.15 V | P1: 0.98 V | P2: 0.85 V"),
    ]
    for x0, y0, x1, y1, fill, border, t1, t2 in boxes:
        draw.rounded_rectangle([x0, y0, x1, y1], radius=14, fill=fill, outline=border, width=3)
        draw.text((x0 + 18, y0 + 20), t1, fill=(15, 23, 42), font=f_box)
        draw.text((x0 + 18, y0 + 60), t2, fill=(51, 65, 85), font=f_sub)

    # Arrows
    # 1 -> 2
    draw.line([(420, 140), (516, 140)], fill=(30, 41, 59), width=4)
    draw.polygon([(520, 140), (504, 131), (504, 149)], fill=(30, 41, 59))
    # 2 -> 3 (diagonal down-left via elbow)
    draw.line([(720, 195), (720, 235), (230, 235), (230, 271)], fill=(30, 41, 59), width=4)
    draw.polygon([(230, 275), (221, 259), (239, 259)], fill=(30, 41, 59))
    # 3 -> 4
    draw.line([(420, 335), (516, 335)], fill=(30, 41, 59), width=4)
    draw.polygon([(520, 335), (504, 326), (504, 344)], fill=(30, 41, 59))

    # Bottom voltage summary strip (only inside image)
    draw.rounded_rectangle([40, 450, 920, 580], radius=12, fill=(248, 250, 252), outline=(100, 116, 139), width=2)
    draw.text((62, 468), "DVFS Operating Points (Diagram Specification):", fill=(15, 23, 42), font=f_box)
    draw.text(
        (62, 505),
        "P0 (Nominal): 3.60 GHz @ 1.15 V   |   P1 (Balanced): 3.20 GHz @ 0.98 V",
        fill=(30, 64, 175),
        font=f_sub,
    )
    draw.text(
        (62, 538),
        "P2 (Throttled): 2.80 GHz @ 0.85 V   |   Hard Guardband: Tj >= 100 C -> Halt",
        fill=(185, 28, 28),
        font=f_sub,
    )
    return _to_png_bytes(img)


def build_pptx() -> tuple[Path, dict[str, bytes]]:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # ---------- Slide 1: Cover ----------
    s1 = prs.slides.add_slide(blank)
    bg = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.8))
    bg.fill.solid()
    bg.fill.fore_color.rgb = RGBColor(15, 23, 42)
    bg.line.fill.background()

    tb = s1.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.7), Inches(1.2))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = "ECE-418: On-Chip Thermal & Cache Hierarchy"
    p.font.size = Pt(34)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    p2 = tf.add_paragraph()
    p2.text = "Week 06 Lecture: Thermal Throttling & L2 Victim Cache Design"
    p2.font.size = Pt(20)
    p2.font.color.rgb = RGBColor(191, 219, 254)

    body = s1.shapes.add_textbox(Inches(0.9), Inches(2.2), Inches(11.5), Inches(4.5))
    btf = body.text_frame
    btf.word_wrap = True
    lines = [
        ("Course Parameters & Baseline Envelope", True, 24),
        ("• Package Thermal Design Power (TDP): 65 W at ambient T_amb = 25 °C", False, 20),
        ("• Maximum Junction Temperature Ceiling (Tj_max): 95 °C (Emergency Halt: 100 °C)", False, 20),
        ("• L2 Victim Cache Geometry: 8-way fully associative bypass buffer, 64-byte line", False, 20),
        ("• Lecture Scope: Workload thermal table (Slide 2), Victim-cache sweep (Slide 3), DTM loop (Slide 4)", False, 20),
    ]
    for idx, (text, bold, sz) in enumerate(lines):
        par = btf.paragraphs[0] if idx == 0 else btf.add_paragraph()
        par.text = text
        par.font.bold = bold
        par.font.size = Pt(sz)
        par.space_after = Pt(14)

    # ---------- Slide 2: Native Multi-Column Table with Headers & Units ----------
    s2 = prs.slides.add_slide(blank)
    h2 = s2.shapes.add_textbox(Inches(0.7), Inches(0.35), Inches(11.9), Inches(0.8))
    p = h2.text_frame.paragraphs[0]
    p.text = "Slide 2: Workload L2 Cache & Thermal Telemetry (T_amb = 25 °C)"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = RGBColor(15, 23, 42)

    rows, cols = 5, 5
    tbl_shape = s2.shapes.add_table(rows, cols, Inches(0.7), Inches(1.4), Inches(11.9), Inches(4.3))
    table = tbl_shape.table
    headers = [
        "Workload",
        "L2 Miss Rate (%)",
        "Peak Tj (°C)",
        "Sustained Clock (GHz)",
        "Energy-Delay (mJ·s)",
    ]
    data = [
        ["MatrixMul-FP32", "1.8", "84.2", "3.60", "14.2"],
        ["GraphBFS-CSR", "11.4", "79.5", "3.45", "28.9"],
        ["SparseFFT-16K", "6.2", "91.8", "2.95", "21.6"],
        ["VideoHEVC-4K", "3.1", "88.0", "3.20", "17.4"],
    ]
    for c, htext in enumerate(headers):
        cell = table.cell(0, c)
        cell.text = htext
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(30, 58, 138)
        for par in cell.text_frame.paragraphs:
            par.font.bold = True
            par.font.size = Pt(17)
            par.font.color.rgb = RGBColor(255, 255, 255)
            par.alignment = PP_ALIGN.CENTER

    for r, row_vals in enumerate(data, start=1):
        for c, val in enumerate(row_vals):
            cell = table.cell(r, c)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(241, 245, 249) if r % 2 == 1 else RGBColor(255, 255, 255)
            for par in cell.text_frame.paragraphs:
                par.font.size = Pt(17)
                par.font.color.rgb = RGBColor(15, 23, 42)
                par.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER

    foot2 = s2.shapes.add_textbox(Inches(0.7), Inches(6.1), Inches(11.9), Inches(0.9))
    fp = foot2.text_frame.paragraphs[0]
    fp.text = "Measurement conditions: 64-byte L2 cache line, 8-way victim buffer disabled, ambient T_amb = 25 °C, steady-state 60 s run."
    fp.font.size = Pt(15)
    fp.font.color.rgb = RGBColor(71, 85, 105)

    # ---------- Slide 3: Embedded High-Contrast Experimental Chart PNG ----------
    s3 = prs.slides.add_slide(blank)
    h3 = s3.shapes.add_textbox(Inches(0.7), Inches(0.3), Inches(11.9), Inches(0.7))
    p = h3.text_frame.paragraphs[0]
    p.text = "Slide 3: Victim Cache Capacity Sweep (GraphBFS-CSR)"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = RGBColor(15, 23, 42)

    s3_chart = _draw_line_chart(
        title="L2 Miss Rate vs Victim Cache Entries (GraphBFS-CSR)",
        subtitle="Replacement Policy Comparison at T_amb = 25 C (64-byte line)",
        x_label="Victim Cache Entries (blocks)",
        y_label="L2 Miss Rate (%)",
        x_ticks=[(0, "0"), (4, "4"), (8, "8"), (16, "16"), (32, "32")],
        y_ticks=[(0, "0"), (4, "4"), (8, "8"), (12, "12"), (16, "16")],
        x_range=(0, 34),
        y_range=(0, 16),
        series=[
            {
                "name": "LRU Policy",
                "color": (37, 99, 235),
                "dashed": False,
                "points": [(0, 14.0), (4, 11.4), (8, 9.2), (16, 8.1), (32, 7.8)],
            },
            {
                "name": "FIFO Policy",
                "color": (234, 88, 12),
                "dashed": True,
                "points": [(0, 14.0), (4, 12.6), (8, 10.8), (16, 9.7), (32, 9.4)],
            },
        ],
        callout="Knee at 8 entries: LRU 9.2% vs FIFO 10.8% (1.6 pp gain)",
    )
    s3.shapes.add_picture(io.BytesIO(s3_chart), Inches(0.9), Inches(1.15), width=Inches(11.5))

    # ---------- Slide 4: Mixed Text + Embedded Mechanism Diagram ----------
    s4 = prs.slides.add_slide(blank)
    h4 = s4.shapes.add_textbox(Inches(0.7), Inches(0.3), Inches(11.9), Inches(0.7))
    p = h4.text_frame.paragraphs[0]
    p.text = "Slide 4: Dynamic Thermal Management (DTM) Hysteresis Loop"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = RGBColor(15, 23, 42)

    left_box = s4.shapes.add_textbox(Inches(0.7), Inches(1.25), Inches(5.3), Inches(5.7))
    ltf = left_box.text_frame
    ltf.word_wrap = True
    s4_lines = [
        ("Hysteresis Control Policy", True, 21),
        ("• Trip High (92 °C): Transition DVFS governor down to P2 throttled clock (2.80 GHz).", False, 17),
        ("• Release Low (85 °C): Restore DVFS governor to P0 nominal clock (3.60 GHz).", False, 17),
        ("• Deadband Width: 7 °C prevents rapid clock oscillation under bursty L2 traffic.", False, 17),
        ("• See the block diagram on the right for the exact voltage rails of P0, P1, and P2.", False, 17),
    ]
    for idx, (text, bold, sz) in enumerate(s4_lines):
        par = ltf.paragraphs[0] if idx == 0 else ltf.add_paragraph()
        par.text = text
        par.font.bold = bold
        par.font.size = Pt(sz)
        par.space_after = Pt(12)

    s4_diag = _draw_dtm_diagram()
    s4.shapes.add_picture(io.BytesIO(s4_diag), Inches(6.25), Inches(1.3), width=Inches(6.45))

    out_path = FIXTURES / "Week06-Thermal-Cache-Lab.pptx"
    prs.save(out_path)
    return out_path, {"slide3_chart": s3_chart, "slide4_diagram": s4_diag}


def _png_output(png_bytes: bytes) -> nbformat.NotebookNode:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return new_output(
        output_type="display_data",
        data={
            "image/png": b64,
            "text/plain": "<Figure size 1200x720 with 1 Axes>",
        },
        metadata={},
    )


def _stream_output(text: str) -> nbformat.NotebookNode:
    return new_output(output_type="stream", name="stdout", text=text)


def build_notebooks() -> tuple[dict[str, Path], dict[str, bytes]]:
    # v1 charts
    v1_temp = _draw_line_chart(
        title="Junction Temperature vs Time (Rev-A, v1)",
        subtitle="Workload: SparseFFT-16K | Ambient 25 C | 8-way Victim Cache",
        x_label="Elapsed Time (s)",
        y_label="Junction Temp (C)",
        x_ticks=[(0, "0"), (15, "15"), (30, "30"), (45, "45"), (60, "60")],
        y_ticks=[(50, "50"), (60, "60"), (70, "70"), (80, "80"), (90, "90"), (100, "100")],
        x_range=(0, 62),
        y_range=(50, 100),
        series=[
            {
                "name": "Rev-A Core Tj (v1)",
                "color": (220, 38, 38),
                "points": [(0, 56.0), (15, 74.5), (30, 86.2), (42, 91.8), (60, 89.4)],
                "peak_point": (42, 91.8),
                "peak_label": "Peak: 91.8 C at 42 s",
                "peak_offset": (-320, -48),
            }
        ],
        threshold={"y": 92.0, "color": (180, 83, 9), "label": "DTM Trip High = 92 C"},
        legend_pos="bottom_right",
    )
    v1_power = _draw_line_chart(
        title="Package Power vs Clock Frequency (Rev-A, v1)",
        subtitle="Steady-State Power Sweep | TDP Ceiling = 65 W",
        x_label="Clock Frequency (GHz)",
        y_label="Package Power (W)",
        x_ticks=[(2.0, "2.0"), (2.4, "2.4"), (2.8, "2.8"), (3.2, "3.2"), (3.6, "3.6")],
        y_ticks=[(20, "20"), (30, "30"), (40, "40"), (50, "50"), (60, "60"), (70, "70")],
        x_range=(1.9, 3.75),
        y_range=(20, 70),
        series=[
            {
                "name": "Rev-A Power (v1)",
                "color": (37, 99, 235),
                "points": [(2.0, 29.4), (2.4, 36.8), (2.8, 45.1), (3.2, 54.6), (3.6, 64.5)],
                "peak_point": (3.6, 64.5),
                "peak_label": "Max: 64.5 W at 3.6 GHz",
                "peak_offset": (-360, -45),
            }
        ],
        threshold={"y": 65.0, "color": (185, 28, 28), "label": "Package TDP = 65 W"},
        legend_pos="bottom_right",
    )

    # v2 charts (updated x-axis unit s -> min, lowered peak temp & power for Rev-B)
    v2_temp = _draw_line_chart(
        title="Junction Temperature vs Time (Rev-B, v2)",
        subtitle="Workload: SparseFFT-16K | Copper Heatspreader Upgrade | Ambient 25 C",
        x_label="Elapsed Time (min)",
        y_label="Junction Temp (C)",
        x_ticks=[(0, "0"), (2, "2"), (4, "4"), (6, "6"), (8, "8"), (10, "10")],
        y_ticks=[(50, "50"), (60, "60"), (70, "70"), (80, "80"), (90, "90"), (100, "100")],
        x_range=(0, 10.5),
        y_range=(50, 100),
        series=[
            {
                "name": "Rev-B Core Tj (v2)",
                "color": (22, 163, 74),
                "points": [(0, 54.0), (2, 68.5), (4, 77.2), (7, 83.4), (10, 82.1)],
                "peak_point": (7, 83.4),
                "peak_label": "Peak: 83.4 C at 7.0 min",
                "peak_offset": (-310, -52),
            }
        ],
        threshold={"y": 92.0, "color": (180, 83, 9), "label": "DTM Trip High = 92 C"},
        legend_pos="bottom_right",
    )
    v2_power = _draw_line_chart(
        title="Package Power vs Clock Frequency (Rev-B, v2)",
        subtitle="Undervolted Rail Sweep | TDP Ceiling = 65 W",
        x_label="Clock Frequency (GHz)",
        y_label="Package Power (W)",
        x_ticks=[(2.0, "2.0"), (2.4, "2.4"), (2.8, "2.8"), (3.2, "3.2"), (3.6, "3.6")],
        y_ticks=[(20, "20"), (30, "30"), (40, "40"), (50, "50"), (60, "60"), (70, "70")],
        x_range=(1.9, 3.75),
        y_range=(20, 70),
        series=[
            {
                "name": "Rev-B Power (v2)",
                "color": (124, 58, 237),
                "points": [(2.0, 24.1), (2.4, 30.2), (2.8, 37.5), (3.2, 45.0), (3.6, 53.2)],
                "peak_point": (3.6, 53.2),
                "peak_label": "Max: 53.2 W at 3.6 GHz",
                "peak_offset": (-360, -45),
            }
        ],
        threshold={"y": 65.0, "color": (185, 28, 28), "label": "Package TDP = 65 W"},
        legend_pos="bottom_right",
    )

    def make_nb(rev: str, ver_tag: str, outputs_c1: list, margin_val: str) -> nbformat.NotebookNode:
        c0 = new_markdown_cell(
            f"# ECE-418 Lab 06: Thermal Sweep & Victim Cache Profiling ({ver_tag})\n\n"
            f"- Silicon revision: `{rev}`\n"
            f"- Ambient temperature: `25.0 C`, Tj ceiling: `95.0 C`, TDP: `65 W`\n"
            f"- Cell `code-sweep` logs summary telemetry and plots junction temperature and package power curves."
        )
        c0["id"] = "md-intro"

        c1 = new_code_cell(
            source=(
                "# Run thermal & power sweep for SparseFFT-16K\n"
                f"revision = '{rev}'\n"
                "summary = run_thermal_sweep(revision, workload='SparseFFT-16K')\n"
                "print(summary)\n"
                "plot_junction_temperature(summary)\n"
                "plot_package_power(summary)\n"
            ),
            execution_count=1,
            outputs=outputs_c1,
        )
        c1["id"] = "code-sweep"

        c2 = new_code_cell(
            source=(
                "tj_max = 95.0\n"
                f"margin_c = {margin_val}\n"
                "print(f'thermal_margin_c = {margin_c:.1f} C')\n"
            ),
            execution_count=2,
            outputs=[_stream_output(f"thermal_margin_c = {margin_val} C\n")],
        )
        c2["id"] = "code-margin"

        nb = new_notebook(
            cells=[c0, c1, c2],
            metadata={
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.11"},
            },
        )
        nb.nbformat = 4
        nb.nbformat_minor = 5
        return nb

    v1_stdout = (
        "[v1 | Rev-A] Thermal sweep summary:\n"
        "peak_temp_c = 91.8 C at freq = 3.60 GHz\n"
        "victim_hit_rate = 34.2 % (8-way LRU)\n"
        "throttle_events = 4\n"
    )
    v2_stdout = (
        "[v2 | Rev-B] Thermal sweep summary:\n"
        "peak_temp_c = 83.4 C at freq = 3.60 GHz\n"
        "victim_hit_rate = 41.6 % (8-way LRU)\n"
        "throttle_events = 0\n"
    )
    v2_precheck = "[v2-reordered | Rev-B] Pre-check: thermal sensor calibration OK (offset = +0.0 C)\n"

    nb_v1 = make_nb(
        "Rev-A",
        "v1",
        [
            _stream_output(v1_stdout),
            _png_output(v1_temp),
            _png_output(v1_power),
        ],
        "3.2",
    )
    nb_v2 = make_nb(
        "Rev-B",
        "v2",
        [
            _stream_output(v2_stdout),
            _png_output(v2_temp),
            _png_output(v2_power),
        ],
        "11.6",
    )
    nb_v2_reordered = make_nb(
        "Rev-B",
        "v2_reordered",
        [
            _stream_output(v2_precheck),
            _stream_output(v2_stdout),
            _png_output(v2_power),
            _png_output(v2_temp),
        ],
        "11.6",
    )

    paths = {
        "v1": NB_DIR / "v1" / "Lab06-Thermal-Sweep.ipynb",
        "v2": NB_DIR / "v2" / "Lab06-Thermal-Sweep.ipynb",
        "v2_reordered": NB_DIR / "v2_reordered" / "Lab06-Thermal-Sweep.ipynb",
    }
    for key, nb in [("v1", nb_v1), ("v2", nb_v2), ("v2_reordered", nb_v2_reordered)]:
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        paths[key].write_text(nbformat.writes(nb) + "\n", encoding="utf-8")

    return paths, {
        "v1_temp": v1_temp,
        "v1_power": v1_power,
        "v2_temp": v2_temp,
        "v2_power": v2_power,
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    pptx_path, pptx_imgs = build_pptx()
    nb_paths, nb_imgs = build_notebooks()

    all_files = [
        FIXTURES / "Speculative-Decoding-2211.17192v2.pdf",
        FIXTURES / "YaRN-2309.00071v3.pdf",
        pptx_path,
        nb_paths["v1"],
        nb_paths["v2"],
        nb_paths["v2_reordered"],
    ]
    summary = {}
    for f in all_files:
        if f.exists():
            summary[str(f.relative_to(ROOT))] = {
                "bytes": f.stat().st_size,
                "sha256": sha256_file(f),
            }
    summary["embedded_png_sha256"] = {
        k: hashlib.sha256(v).hexdigest() for k, v in {**pptx_imgs, **nb_imgs}.items()
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
