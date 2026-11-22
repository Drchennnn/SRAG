"""Generate and verify the frozen public synthetic PDF fixture suite for codex/g-pdf-object-recovery.

Produces:
1. docs/G/G-pdf-object-recovery/fixtures/Vector-Subfigures-Guide.pdf (2 pages)
   - Page 1: Title, introductory text, and a 3-line native vector table (Table 1)
   - Page 2: Dual vector-drawn subfigures side-by-side ((a) Left Panel and (b) Right Panel)
     drawn purely with PDF vector paths (zero embedded raster PNGs), plus shared Figure 1 caption
2. docs/G/G-pdf-object-recovery/fixtures/Formula-And-Notes.pdf (2 pages)
   - Page 1: Background text + irrelevant formula (E_loss) + irrelevant caption (Figure 2)
   - Page 2: Standalone formula (tau_eff) + nearby irrelevant formula (alpha_max) clearly
     separated vertically + vector plot + standalone Figure 3 caption

Also verifies with pypdfium2 + PIL that every page renders cleanly and every frozen bbox
has non-empty crop variance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
from PIL import Image, ImageStat

BASE_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = BASE_DIR / "fixtures"
CRITERIA_PATH = BASE_DIR / "frozen_criteria.json"

PAGE_WIDTH_PT = 612.0
PAGE_HEIGHT_PT = 792.0


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _norm_to_pdf_rect(bbox: list[float]) -> tuple[float, float, float, float]:
    """Convert normalized top-left [x0, y0, x1, y1] to PDF bottom-left (x0, y0, x1, y1) pt."""
    x0_n, y0_n, x1_n, y1_n = bbox
    x0 = x0_n * PAGE_WIDTH_PT
    x1 = x1_n * PAGE_WIDTH_PT
    y1 = (1.0 - y0_n) * PAGE_HEIGHT_PT
    y0 = (1.0 - y1_n) * PAGE_HEIGHT_PT
    return (x0, y0, x1, y1)


class _PDFPageBuilder:
    """Minimal deterministic PDF 1.4 page content-stream builder using pure vector paths & Type1 text."""

    def __init__(self) -> None:
        self.ops: list[str] = []

    def set_stroke_rgb(self, r: float, g: float, b: float) -> None:
        self.ops.append(f"{r:.3f} {g:.3f} {b:.3f} RG")

    def set_fill_rgb(self, r: float, g: float, b: float) -> None:
        self.ops.append(f"{r:.3f} {g:.3f} {b:.3f} rg")

    def set_line_width(self, w: float) -> None:
        self.ops.append(f"{w:.2f} w")

    def set_dash(self, pattern: str = "[] 0") -> None:
        self.ops.append(f"{pattern} d")

    def rect(self, x: float, y: float, w: float, h: float, *, stroke: bool = True, fill: bool = False) -> None:
        op = "B" if (stroke and fill) else ("f" if fill else "S")
        self.ops.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re {op}")

    def line(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.ops.append(f"{x0:.2f} {y0:.2f} m {x1:.2f} {y1:.2f} l S")

    def polyline(self, pts: list[tuple[float, float]]) -> None:
        if not pts:
            return
        parts = [f"{pts[0][0]:.2f} {pts[0][1]:.2f} m"]
        for px, py in pts[1:]:
            parts.append(f"{px:.2f} {py:.2f} l")
        parts.append("S")
        self.ops.append(" ".join(parts))

    def marker_square(self, cx: float, cy: float, half: float = 2.5) -> None:
        self.rect(cx - half, cy - half, half * 2.0, half * 2.0, stroke=True, fill=True)

    def text(self, x: float, y: float, txt: str, *, font: str = "F1", size: float = 11.0) -> None:
        esc = _pdf_escape(txt)
        self.ops.append(f"BT /{font} {size:.2f} Tf 1 0 0 1 {x:.2f} {y:.2f} Tm ({esc}) Tj ET")

    def build_stream(self) -> bytes:
        return ("\n".join(self.ops) + "\n").encode("ascii")


def _write_pdf(path: Path, page_streams: list[bytes]) -> None:
    """Write a deterministic PDF 1.4 file with Helvetica (F1), Helvetica-Bold (F2), Courier (F3)."""
    objects: list[bytes] = []

    def add_obj(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    # 1: Catalog, 2: Pages (placeholders updated after page creation)
    add_obj(b"")  # obj 1
    add_obj(b"")  # obj 2

    # Standard Type1 fonts
    f1_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    f2_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    f3_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    font_res = f"<< /F1 {f1_id} 0 R /F2 {f2_id} 0 R /F3 {f3_id} 0 R >>".encode("ascii")
    page_ids: list[int] = []

    for stream_bytes in page_streams:
        content_id = add_obj(
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
            + stream_bytes
            + b"endstream"
        )
        page_dict = (
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 {PAGE_WIDTH_PT:.0f} {PAGE_HEIGHT_PT:.0f}] "
            f"/Resources << /Font {font_res.decode('ascii')} >> "
            f"/Contents {content_id} 0 R >>"
        ).encode("ascii")
        page_ids.append(add_obj(page_dict))

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for idx, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{idx} 0 obj\n".encode("ascii"))
        out.extend(body)
        out.extend(b"\nendobj\n")

    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    out.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("ascii")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))


def _draw_vector_plot(
    pb: _PDFPageBuilder,
    bbox_norm: list[float],
    *,
    panel_title: str,
    x_label: str,
    y_label: str,
    x_ticks: list[tuple[float, str]],
    y_ticks: list[tuple[float, str]],
    series_pts: list[tuple[float, float]],
    series_rgb: tuple[float, float, float],
    guide_x_frac: float | None = None,
) -> None:
    """Draw a pure vector plot inside the given normalized bbox."""
    x0, y0, x1, y1 = _norm_to_pdf_rect(bbox_norm)
    w = x1 - x0
    h = y1 - y0

    # Outer panel frame
    pb.set_stroke_rgb(0.75, 0.78, 0.82)
    pb.set_fill_rgb(0.985, 0.985, 0.975)
    pb.set_line_width(1.0)
    pb.rect(x0, y0, w, h, stroke=True, fill=True)

    # Panel title
    pb.set_fill_rgb(0.08, 0.10, 0.14)
    pb.text(x0 + 14, y1 - 22, panel_title, font="F2", size=10.5)

    # Plot area inside panel
    px0 = x0 + 52.0
    px1 = x1 - 16.0
    py0 = y0 + 42.0
    py1 = y1 - 38.0
    pw = px1 - px0
    ph = py1 - py0

    # Light grid lines
    pb.set_stroke_rgb(0.88, 0.90, 0.92)
    pb.set_line_width(0.6)
    for frac, _ in x_ticks:
        gx = px0 + frac * pw
        pb.line(gx, py0, gx, py1)
    for frac, _ in y_ticks:
        gy = py0 + frac * ph
        pb.line(px0, gy, px1, gy)

    # Optional vertical dashed guide
    if guide_x_frac is not None:
        pb.set_stroke_rgb(0.55, 0.58, 0.62)
        pb.set_line_width(0.8)
        pb.set_dash("[3 3] 0")
        gx = px0 + guide_x_frac * pw
        pb.line(gx, py0, gx, py1)
        pb.set_dash("[] 0")

    # Axes
    pb.set_stroke_rgb(0.15, 0.18, 0.22)
    pb.set_line_width(1.4)
    pb.line(px0, py0, px0, py1)
    pb.line(px0, py0, px1, py0)

    # Ticks & tick labels
    pb.set_fill_rgb(0.15, 0.18, 0.22)
    for frac, label in x_ticks:
        tx = px0 + frac * pw
        pb.line(tx, py0 - 3.5, tx, py0)
        pb.text(tx - 4.0, py0 - 14.0, label, font="F1", size=8.5)
    for frac, label in y_ticks:
        ty = py0 + frac * ph
        pb.line(px0 - 3.5, ty, px0, ty)
        pb.text(px0 - 26.0, ty - 3.0, label, font="F1", size=8.5)

    # Axis labels
    pb.set_fill_rgb(0.10, 0.12, 0.16)
    pb.text(px0 + pw * 0.34, y0 + 11.0, x_label, font="F2", size=9.0)
    pb.text(x0 + 8.0, py1 + 6.0, y_label, font="F2", size=8.5)

    # Vector waveform series
    sr, sg, sb = series_rgb
    pb.set_stroke_rgb(sr, sg, sb)
    pb.set_fill_rgb(sr, sg, sb)
    pb.set_line_width(2.0)
    mapped = [(px0 + fx * pw, py0 + fy * ph) for fx, fy in series_pts]
    pb.polyline(mapped)
    pb.set_line_width(0.8)
    for mx, my in mapped:
        pb.marker_square(mx, my, half=2.3)


# Frozen normalized bounding boxes [x0, y0, x1, y1] in top-left [0..1]
PDF1_TABLE1_BBOX = [0.10, 0.30, 0.90, 0.48]
PDF1_SUBFIG_A_BBOX = [0.06, 0.20, 0.48, 0.62]
PDF1_SUBFIG_B_BBOX = [0.52, 0.20, 0.94, 0.62]
PDF1_FIG1_CAPTION_BBOX = [0.08, 0.64, 0.92, 0.70]

PDF2_TAU_EFF_BBOX = [0.12, 0.16, 0.88, 0.28]
PDF2_ALPHA_MAX_BBOX = [0.12, 0.36, 0.88, 0.46]
PDF2_PLOT3_BBOX = [0.18, 0.52, 0.82, 0.76]
PDF2_FIG3_CAPTION_BBOX = [0.12, 0.78, 0.88, 0.83]


def build_vector_subfigures_guide(out_path: Path) -> None:
    # --- Page 1: Title, intro text, and 3-line native vector table (Table 1) ---
    p1 = _PDFPageBuilder()
    p1.set_fill_rgb(0.08, 0.10, 0.14)
    p1.text(60, 736, "Benchtop Waveform & Calibration Guide", font="F2", size=16.0)
    p1.text(60, 714, "Public Synthetic Fixture (CC BY 4.0) - Vector-Subfigures-Guide.pdf", font="F1", size=10.0)

    p1.set_stroke_rgb(0.20, 0.25, 0.32)
    p1.set_line_width(1.0)
    p1.line(60, 704, 552, 704)

    p1.set_fill_rgb(0.10, 0.12, 0.16)
    p1.text(60, 676, "Section 1: Overview", font="F2", size=12.5)
    p1.text(60, 654, "This public synthetic calibration guide documents two-channel benchtop acquisition.", font="F1", size=10.5)
    p1.text(60, 636, "Default sampling interval is fixed at 0.25 ms with a 50-ohm termination impedance.", font="F1", size=10.5)
    p1.text(60, 618, "All measurements are recorded at an ambient reference temperature of 23.0 C after", font="F1", size=10.5)
    p1.text(60, 602, "a 15-minute warm-up period.", font="F1", size=10.5)

    # Table 1 (3-line native vector table inside PDF1_TABLE1_BBOX)
    tx0, ty0, tx1, ty1 = _norm_to_pdf_rect(PDF1_TABLE1_BBOX)
    p1.set_fill_rgb(0.08, 0.10, 0.14)
    p1.text(tx0 + 4, ty1 - 16, "Table 1: Channel Calibration Parameters", font="F2", size=11.0)

    top_rule_y = ty1 - 26
    hdr_sep_y = ty1 - 50
    row1_y = ty1 - 74
    bot_rule_y = ty0 + 10
    mid_col_x = tx0 + 210

    p1.set_stroke_rgb(0.10, 0.12, 0.16)
    p1.set_line_width(1.4)
    p1.line(tx0, top_rule_y, tx1, top_rule_y)
    p1.set_line_width(0.9)
    p1.line(tx0, hdr_sep_y, tx1, hdr_sep_y)
    p1.set_line_width(1.4)
    p1.line(tx0, bot_rule_y, tx1, bot_rule_y)

    # Table text
    p1.text(tx0 + 10, top_rule_y - 16, "Channel", font="F2", size=10.5)
    p1.text(mid_col_x + 10, top_rule_y - 16, "Nominal Range", font="F2", size=10.5)

    p1.text(tx0 + 10, hdr_sep_y - 18, "CH-A (Step)", font="F1", size=10.5)
    p1.text(mid_col_x + 10, hdr_sep_y - 18, "0.0 V to 5.0 V", font="F1", size=10.5)

    p1.text(tx0 + 10, row1_y - 18, "CH-B (Decay)", font="F1", size=10.5)
    p1.text(mid_col_x + 10, row1_y - 18, "-12.0 dB to +6.0 dB", font="F1", size=10.5)

    # --- Page 2: Dual vector subfigures (a) & (b) + shared Figure 1 caption ---
    p2 = _PDFPageBuilder()
    p2.set_fill_rgb(0.08, 0.10, 0.14)
    p2.text(60, 736, "Section 2: Dual-Panel Benchtop Waveforms", font="F2", size=13.5)
    p2.text(60, 716, "Side-by-side pure vector subfigures sharing Figure 1 caption below.", font="F1", size=10.0)

    # Left subfigure (a): step-up peak at t = 4 ms, Amplitude [V]
    _draw_vector_plot(
        p2,
        PDF1_SUBFIG_A_BBOX,
        panel_title="(a) Left Panel: Step Response",
        x_label="Time [ms]",
        y_label="Amplitude [V]",
        x_ticks=[(0.0, "0"), (0.5, "4"), (1.0, "8")],
        y_ticks=[(0.0, "0"), (0.5, "2.5"), (1.0, "5.0")],
        series_pts=[
            (0.00, 0.10),
            (0.25, 0.10),
            (0.38, 0.60),
            (0.50, 1.00),
            (0.75, 0.86),
            (1.00, 0.84),
        ],
        series_rgb=(0.10, 0.35, 0.75),
        guide_x_frac=0.50,
    )

    # Right subfigure (b): damped decay / spike at t = 8 ms, Gain [dB]
    _draw_vector_plot(
        p2,
        PDF1_SUBFIG_B_BBOX,
        panel_title="(b) Right Panel: Damped Decay",
        x_label="Time [ms]",
        y_label="Gain [dB]",
        x_ticks=[(0.0, "0"), (0.333, "4"), (0.667, "8"), (1.0, "12")],
        y_ticks=[(0.0, "-12"), (0.5, "-3"), (1.0, "+6")],
        series_pts=[
            (0.00, 0.67),
            (0.17, 0.42),
            (0.333, 0.17),
            (0.55, 0.12),
            (0.667, 1.00),
            (0.83, 0.25),
            (1.00, 0.00),
        ],
        series_rgb=(0.78, 0.18, 0.18),
        guide_x_frac=0.667,
    )

    # Shared Figure 1 caption
    cx0, cy0, cx1, cy1 = _norm_to_pdf_rect(PDF1_FIG1_CAPTION_BBOX)
    p2.set_fill_rgb(0.10, 0.12, 0.16)
    p2.text(cx0 + 4, cy1 - 15, "Figure 1: Qualitative bench waveforms. Only a schematic illustration;", font="F2", size=10.5)
    p2.text(cx0 + 4, cy1 - 31, "does not support causal delay claims.", font="F2", size=10.5)

    _write_pdf(out_path, [p1.build_stream(), p2.build_stream()])


def build_formula_and_notes(out_path: Path) -> None:
    # --- Page 1: Background + irrelevant formula E_loss + irrelevant caption Figure 2 ---
    p1 = _PDFPageBuilder()
    p1.set_fill_rgb(0.08, 0.10, 0.14)
    p1.text(60, 736, "Signal Windowing & Thermal Notes", font="F2", size=16.0)
    p1.text(60, 714, "Public Synthetic Fixture (CC BY 4.0) - Formula-And-Notes.pdf", font="F1", size=10.0)
    p1.set_stroke_rgb(0.20, 0.25, 0.32)
    p1.set_line_width(1.0)
    p1.line(60, 704, 552, 704)

    p1.set_fill_rgb(0.10, 0.12, 0.16)
    p1.text(60, 676, "Section 1: Background", font="F2", size=12.5)
    p1.text(60, 654, "Benchtop thermal drift is monitored over a 30-minute stabilization window.", font="F1", size=10.5)
    p1.text(60, 636, "Mechanical spring losses are modeled separately from discrete-time filter delay.", font="F1", size=10.5)

    # Irrelevant nearby formula E_loss
    p1.set_stroke_rgb(0.65, 0.68, 0.72)
    p1.set_fill_rgb(0.96, 0.97, 0.98)
    p1.rect(74, 520, 464, 72, stroke=True, fill=True)
    p1.set_fill_rgb(0.08, 0.10, 0.14)
    p1.text(92, 566, "E_loss = 0.5 * k * x^2", font="F3", size=13.0)
    p1.text(92, 540, "where k is spring stiffness [N/m] and x is displacement [m].", font="F1", size=10.0)

    # Irrelevant nearby caption Figure 2
    p1.set_stroke_rgb(0.75, 0.78, 0.82)
    p1.set_fill_rgb(0.985, 0.985, 0.975)
    p1.rect(110, 340, 392, 110, stroke=True, fill=True)
    p1.set_stroke_rgb(0.20, 0.50, 0.35)
    p1.set_line_width(1.8)
    p1.polyline([(140, 365), (240, 380), (340, 395), (460, 415)])
    p1.set_fill_rgb(0.10, 0.12, 0.16)
    p1.text(110, 316, "Figure 2: Thermal drift baseline.", font="F2", size=10.5)

    # --- Page 2: Standalone tau_eff + separated irrelevant alpha_max + plot + Figure 3 caption ---
    p2 = _PDFPageBuilder()
    p2.set_fill_rgb(0.08, 0.10, 0.14)
    p2.text(60, 736, "Section 2: Discrete Window Delay & Envelope", font="F2", size=13.5)
    p2.text(60, 716, "Distinct formulas are separated vertically; Figure 3 shows the peak-hold envelope.", font="F1", size=10.0)

    # Standalone formula tau_eff
    fx0, fy0, fx1, fy1 = _norm_to_pdf_rect(PDF2_TAU_EFF_BBOX)
    p2.set_stroke_rgb(0.18, 0.35, 0.65)
    p2.set_fill_rgb(0.95, 0.97, 1.00)
    p2.set_line_width(1.2)
    p2.rect(fx0, fy0, fx1 - fx0, fy1 - fy0, stroke=True, fill=True)
    p2.set_fill_rgb(0.08, 0.10, 0.14)
    p2.text(fx0 + 18, fy1 - 24, "Effective Group Delay (Standalone Formula):", font="F2", size=10.5)
    p2.text(fx0 + 18, fy1 - 50, "tau_eff = (W - 1) / 2", font="F3", size=14.0)
    p2.text(fx0 + 18, fy1 - 74, "Condition: for odd W >= 3 only", font="F2", size=11.0)

    # Nearby irrelevant formula alpha_max (clearly separated vertically)
    ax0, ay0, ax1, ay1 = _norm_to_pdf_rect(PDF2_ALPHA_MAX_BBOX)
    p2.set_stroke_rgb(0.65, 0.68, 0.72)
    p2.set_fill_rgb(0.97, 0.97, 0.97)
    p2.set_line_width(1.0)
    p2.rect(ax0, ay0, ax1 - ax0, ay1 - ay0, stroke=True, fill=True)
    p2.set_fill_rgb(0.10, 0.12, 0.16)
    p2.text(ax0 + 18, ay1 - 22, "Smoothing Factor (Auxiliary Note):", font="F2", size=10.0)
    p2.text(ax0 + 18, ay1 - 44, "alpha_max = 1 / (N + 1)", font="F3", size=13.0)
    p2.text(ax0 + 18, ay1 - 64, "Applies to exponential moving average window length N.", font="F1", size=9.5)

    # Vector plot above Figure 3 caption
    _draw_vector_plot(
        p2,
        PDF2_PLOT3_BBOX,
        panel_title="Peak-Hold Envelope Trace",
        x_label="Sample Index [n]",
        y_label="Level [V]",
        x_ticks=[(0.0, "0"), (0.5, "50"), (1.0, "100")],
        y_ticks=[(0.0, "0.0"), (0.5, "1.5"), (1.0, "3.0")],
        series_pts=[
            (0.00, 0.15),
            (0.20, 0.85),
            (0.45, 0.85),
            (0.60, 0.60),
            (0.80, 0.60),
            (1.00, 0.35),
        ],
        series_rgb=(0.12, 0.48, 0.38),
        guide_x_frac=None,
    )

    # Standalone Figure 3 caption
    c3x0, c3y0, c3x1, c3y1 = _norm_to_pdf_rect(PDF2_FIG3_CAPTION_BBOX)
    p2.set_fill_rgb(0.10, 0.12, 0.16)
    p2.text(c3x0 + 4, c3y1 - 20, "Figure 3: Peak-hold envelope under dry bench conditions.", font="F2", size=10.5)

    _write_pdf(out_path, [p1.build_stream(), p2.build_stream()])


def verify_pdf_renders_and_crops(
    pdf_path: Path,
    targets: list[dict[str, Any]],
    dpi: int = 150,
) -> dict[str, Any]:
    """Render every page with pypdfium2 and verify non-empty crop variance in every frozen bbox."""
    scale = dpi / 72.0
    doc = pdfium.PdfDocument(str(pdf_path))
    page_count = len(doc)
    rendered_pages: dict[int, Image.Image] = {}
    for idx in range(page_count):
        page = doc[idx]
        bitmap = page.render(scale=scale)
        rendered_pages[idx + 1] = bitmap.to_pil()

    crop_checks: list[dict[str, Any]] = []
    for target in targets:
        pno = int(target["page"])
        bbox = target["bbox"]
        img = rendered_pages[pno]
        w_px, h_px = img.size
        crop_box = (
            max(0, int(round(bbox[0] * w_px))),
            max(0, int(round(bbox[1] * h_px))),
            min(w_px, int(round(bbox[2] * w_px))),
            min(h_px, int(round(bbox[3] * h_px))),
        )
        crop = img.crop(crop_box).convert("L")
        stat = ImageStat.Stat(crop)
        stddev = float(stat.stddev[0])
        if stddev <= 1.0:
            raise AssertionError(
                f"Empty/flat crop variance for {pdf_path.name} {target['object_id']}: stddev={stddev}"
            )
        crop_checks.append(
            {
                "object_id": target["object_id"],
                "page": pno,
                "crop_px": list(crop_box),
                "stddev": round(stddev, 3),
            }
        )

    return {
        "file": pdf_path.name,
        "pages": page_count,
        "sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "bytes": pdf_path.stat().st_size,
        "crop_checks": crop_checks,
    }


def main() -> None:
    pdf1_path = FIXTURES_DIR / "Vector-Subfigures-Guide.pdf"
    pdf2_path = FIXTURES_DIR / "Formula-And-Notes.pdf"

    build_vector_subfigures_guide(pdf1_path)
    build_formula_and_notes(pdf2_path)

    pdf1_report = verify_pdf_renders_and_crops(
        pdf1_path,
        [
            {"object_id": "pdf1_p1_table1", "page": 1, "bbox": PDF1_TABLE1_BBOX},
            {"object_id": "pdf1_p2_subfigure_a", "page": 2, "bbox": PDF1_SUBFIG_A_BBOX},
            {"object_id": "pdf1_p2_subfigure_b", "page": 2, "bbox": PDF1_SUBFIG_B_BBOX},
            {"object_id": "pdf1_p2_figure1_caption", "page": 2, "bbox": PDF1_FIG1_CAPTION_BBOX},
        ],
    )
    pdf2_report = verify_pdf_renders_and_crops(
        pdf2_path,
        [
            {"object_id": "pdf2_p2_formula_tau_eff", "page": 2, "bbox": PDF2_TAU_EFF_BBOX},
            {"object_id": "pdf2_p2_formula_alpha_max", "page": 2, "bbox": PDF2_ALPHA_MAX_BBOX},
            {"object_id": "pdf2_p2_plot3", "page": 2, "bbox": PDF2_PLOT3_BBOX},
            {"object_id": "pdf2_p2_figure3_caption", "page": 2, "bbox": PDF2_FIG3_CAPTION_BBOX},
        ],
    )

    print(json.dumps({"pdf1": pdf1_report, "pdf2": pdf2_report}, indent=2))


if __name__ == "__main__":
    main()
