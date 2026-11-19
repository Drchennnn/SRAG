"""Generate the original two-page A03 PDF fixture without third-party packages."""

from __future__ import annotations

from pathlib import Path

PAGE_WIDTH = 612
PAGE_HEIGHT = 792
OUTPUT = Path(__file__).parent / "样例" / "Signal-reference.pdf"


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _text(x: float, y: float, size: float, value: str, *, bold: bool = False) -> str:
    font = "F2" if bold else "F1"
    return f"BT /{font} {size:g} Tf {x:g} {y:g} Td ({_escape(value)}) Tj ET"


def _line(x1: float, y1: float, x2: float, y2: float, *, width: float = 1) -> str:
    return f"{width:g} w {x1:g} {y1:g} m {x2:g} {y2:g} l S"


def _polyline(points: list[tuple[float, float]], *, width: float = 2) -> str:
    start, *rest = points
    commands = [f"{width:g} w {start[0]:g} {start[1]:g} m"]
    commands.extend(f"{x:g} {y:g} l" for x, y in rest)
    commands.append("S")
    return " ".join(commands)


def _page_one() -> bytes:
    commands = [
        "0.11 0.18 0.23 rg",
        _text(54, 726, 25, "Median Filter Reference", bold=True),
        "0.76 0.30 0.08 rg",
        _text(54, 695, 11, "SIGNAL PROCESSING - SHAREABLE COURSE FIXTURE", bold=True),
        "0.11 0.18 0.23 rg",
        _text(54, 645, 17, "Definition", bold=True),
        _text(
            54,
            620,
            12,
            "A one-dimensional median filter replaces the center sample with the median",
        ),
        _text(
            54,
            602,
            12,
            "of an odd-length local window. It is robust to isolated impulse spikes.",
        ),
        "0.94 0.96 0.97 rg 50 470 512 100 re f",
        "0.11 0.18 0.23 rg",
        _text(72, 535, 15, "Centered-window lookahead", bold=True),
        _text(72, 503, 18, "future samples = (W - 1) / 2", bold=True),
        _text(54, 430, 17, "Conditions and limits", bold=True),
        _text(66, 402, 12, "1. W is a positive odd integer."),
        _text(66, 378, 12, "2. The formula is lookahead relative to the centered sample."),
        _text(
            66,
            354,
            12,
            "3. At the boundaries, padding or a restricted valid region is required.",
        ),
        _text(66, 330, 12, "4. This handout gives no numeric noise-reduction percentage."),
        _text(54, 270, 17, "Worked value", bold=True),
        _text(54, 240, 12, "For W = 5, an offline centered window requires 2 future samples."),
        _text(
            54,
            222,
            10,
            "This is not a universal group delay or peak-delay claim for a nonlinear filter.",
        ),
        "0.76 0.30 0.08 RG",
        _line(54, 210, 558, 210, width=1.5),
        "0.11 0.18 0.23 rg",
        _text(54, 180, 11, "See Figure 1 on page 2 for a qualitative impulse-spike illustration."),
        _text(54, 54, 9, "A03 fixture - page 1 of 2"),
    ]
    return "\n".join(commands).encode("ascii")


def _panel(
    left: float,
    bottom: float,
    title: str,
    points: list[tuple[float, float]],
    note: str,
) -> list[str]:
    width = 228
    height = 260
    plot_left = left + 30
    plot_bottom = bottom + 55
    return [
        "0.96 0.97 0.98 rg",
        f"{left:g} {bottom:g} {width:g} {height:g} re f",
        "0.11 0.18 0.23 rg",
        _text(left + 16, bottom + 228, 13, title, bold=True),
        "0.11 0.18 0.23 RG",
        _line(plot_left, plot_bottom, plot_left, plot_bottom + 135, width=1.2),
        _line(plot_left, plot_bottom, plot_left + 174, plot_bottom, width=1.2),
        _text(plot_left + 58, plot_bottom - 18, 10, "Time [ms]"),
        _text(left + 8, plot_bottom + 143, 9, "Amplitude [a.u.]"),
        "0.76 0.30 0.08 RG",
        _polyline([(plot_left + x, plot_bottom + y) for x, y in points], width=2.4),
        "0.11 0.18 0.23 rg",
        _text(left + 16, bottom + 7, 9, note),
    ]


def _page_two() -> bytes:
    raw = [(0, 45), (26, 50), (52, 43), (76, 128), (88, 42), (116, 47), (145, 50), (174, 45)]
    filtered = [(0, 45), (26, 49), (52, 45), (76, 47), (88, 46), (116, 47), (145, 49), (174, 45)]
    commands = [
        "0.11 0.18 0.23 rg",
        _text(54, 726, 23, "Impulse-spike illustration", bold=True),
        _text(
            54,
            698,
            11,
            "Qualitative schematic; axes and units are part of the source evidence.",
        ),
    ]
    commands.extend(_panel(54, 370, "(a) Raw signal", raw, "One isolated high spike is visible."))
    commands.extend(
        _panel(330, 370, "(b) Median, W = 5", filtered, "The isolated spike is suppressed.")
    )
    commands.extend(
        [
            "0.11 0.18 0.23 rg",
            _text(
                54,
                318,
                12,
                "Figure 1. Median filtering suppresses the isolated impulse spike "
                "in this schematic.",
            ),
            _text(
                54,
                288,
                11,
                "The figure supports the qualitative observation only. "
                "It does not report a measured",
            ),
            _text(
                54,
                270,
                11,
                "error reduction, frequency response, or causal explanation "
                "for another experiment.",
            ),
            _text(54, 54, 9, "A03 fixture - page 2 of 2"),
        ]
    )
    return "\n".join(commands).encode("ascii")


def _build_pdf(pages: list[bytes]) -> bytes:
    objects: list[bytes] = [b""]

    def add(value: bytes) -> int:
        objects.append(value)
        return len(objects) - 1

    font_regular = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font_bold = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    pages_id = add(b"")
    page_ids: list[int] = []
    for stream in pages:
        content_id = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        page_id = add(
            (
                f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                f"/Resources << /Font << /F1 {font_regular} 0 R /F2 {font_bold} 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        page_ids.append(page_id)
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[pages_id] = f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>".encode("ascii")
    catalog_id = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("ascii"))

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id, body in enumerate(objects[1:], start=1):
        offsets.append(len(output))
        output.extend(f"{object_id} 0 obj\n".encode("ascii"))
        output.extend(body)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects)}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    trailer = (
        f"trailer\n<< /Size {len(objects)} /Root {catalog_id} 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    )
    output.extend(trailer.encode("ascii"))
    return bytes(output)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(_build_pdf([_page_one(), _page_two()]))
    print(OUTPUT)


if __name__ == "__main__":
    main()
