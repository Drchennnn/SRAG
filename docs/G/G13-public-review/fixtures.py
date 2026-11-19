"""Author an unrelated, fictional course and its answer-independent probe inputs."""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import nbformat
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
MATERIALS = HERE / "materials"
PDF_NAME = "Azimuth-Workshop.pdf"
PPTX_NAME = "Beacon-Inspection.pptx"
NOTEBOOK_NAME = "BenchRetest.ipynb"


def make_plot(name: str, unit: str, color: str) -> Image.Image:
    image = Image.new("RGB", (640, 340), "white")
    draw = ImageDraw.Draw(image)
    bold = ImageFont.load_default(36)
    normal = ImageFont.load_default(25)
    draw.rounded_rectangle((3, 3, 637, 336), radius=14, outline=color, width=6)
    draw.text((28, 19), name, font=bold, fill=color)
    draw.line((93, 80, 93, 258, 598, 258), width=4, fill="black")
    draw.line([(x, y) for x, y in ((110, 221), (189, 192), (287, 202), (405, 155), (558, 125))], width=8, fill=color)
    for x, tick in ((148, "2"), (327, "6"), (506, "10")):
        draw.line((x, 254, x, 268), width=3, fill="black")
        draw.text((x - 8, 267), tick, font=normal, fill="black")
    draw.text((182, 297), f"AZIMUTH [{unit}]", font=bold, fill="black")
    return image


def pdf_stream(data: bytes, options: bytes = b"") -> bytes:
    return b"<< /Length %d %s >>\nstream\n" % (len(data), options) + data + b"\nendstream"


def pdf_image(image: Image.Image) -> bytes:
    options = b"/Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode" % image.size
    return pdf_stream(zlib.compress(image.tobytes()), options)


def workshop(path: Path) -> None:
    # Both are independent raster objects. Painting the right-hand object first deliberately
    # makes paint order differ from left-to-right placement.
    layout = b"""BT /F1 25 Tf 57 499 Td (Azimuth workshop: acceptance plots) Tj ET
BT /F1 16 Tf 59 465 Td (Position convention: compare the two measurement panels below.) Tj ET
q 323 0 0 171 438 191 cm /Right Do Q
q 323 0 0 171 48 191 cm /Left Do Q
BT /F1 16 Tf 58 125 Td (Both sensors report azimuth. Their panel scales are intentionally different.) Tj ET
"""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 815 540] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> /XObject << /Right 6 0 R /Left 7 0 R >> >> >>",
        pdf_stream(layout),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        pdf_image(make_plot("SOUTH BEACON", "mrad", "#135854")),
        pdf_image(make_plot("NORTH BEACON", "deg", "#613D83")),
    ]
    result = bytearray(b"%PDF-1.4\n%\xff\xff\xff\xff\n")
    offsets = [0]
    for index, item in enumerate(objects, 1):
        offsets.append(len(result))
        result.extend(b"%d 0 obj\n" % index + item + b"\nendobj\n")
    xref = len(result)
    result.extend(b"xref\n0 %d\n0000000000 65535 f \n" % len(offsets))
    for offset in offsets[1:]:
        result.extend(b"%010d 00000 n \n" % offset)
    result.extend(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(offsets), xref))
    path.write_bytes(result)


def inspection(path: Path) -> None:
    slides = Presentation()
    slides.slide_width = Inches(13.333)
    slides.slide_height = Inches(7.5)
    slide = slides.slides.add_slide(slides.slide_layouts[6])
    title = slide.shapes.add_textbox(Inches(0.7), Inches(0.46), Inches(11.8), Inches(0.7))
    title.text_frame.text = "Beacon inspection records"
    title.text_frame.paragraphs[0].font.size = Pt(24)
    slide.shapes.add_textbox(Inches(0.8), Inches(1.32), Inches(11.3), Inches(0.5)).text_frame.text = (
        "Installation bench, values verified at the start of each shift."
    )
    table = slide.shapes.add_table(3, 3, Inches(1.5), Inches(2.4), Inches(10.3), Inches(2.48)).table
    rows = [
        ["Beacon", "Lateral drift (mm)", "Settle time (ms)"],
        ["South beacon", "3.8", "72"],
        ["North beacon", "1.6", "54"],
    ]
    for row_index, row in enumerate(rows):
        for col_index, text in enumerate(row):
            cell = table.cell(row_index, col_index)
            cell.text = text
            cell.text_frame.paragraphs[0].font.size = Pt(21)
    slide.shapes.add_textbox(Inches(1.5), Inches(5.8), Inches(10.2), Inches(0.52)).text_frame.text = (
        "Record dimensions exactly as labelled. Calibrated from the installation bench."
    )
    slides.save(path)


def bench_note(path: Path, settling: int, comment: str) -> None:
    cell = nbformat.v4.new_code_cell(
        '# Bench retest settling clock\nprint("Saved instrument settling measurement")',
        id="settle-retest",
        outputs=[nbformat.v4.new_output("stream", name="stdout", text=f"Bench retest settling clock: {settling} ms. {comment}\n")],
        execution_count=1,
    )
    book = nbformat.v4.new_notebook(cells=[cell])
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(book, path)


def main() -> None:
    MATERIALS.mkdir(exist_ok=True)
    workshop(MATERIALS / PDF_NAME)
    inspection(MATERIALS / PPTX_NAME)
    bench_note(MATERIALS / "v1" / NOTEBOOK_NAME, 83, "Initial deployment.")
    bench_note(MATERIALS / "v2" / NOTEBOOK_NAME, 47, "After mounting adjustment.")
    print(json.dumps({"pdf": 1, "pptx": 1, "notebook_revisions": 2, "files": 4}))


if __name__ == "__main__":
    main()
