"""New synthetic public inputs for the independent G19 browser gate."""

import argparse
import base64
import hashlib
import io
import json
import zlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches

COURSE = "g19-independent-canopy"
GUIDE = "Canopy-Coupler-Field-Notes.pdf"
DECK = "Sparrow-Coupler-Release-Log.pptx"
NOTEBOOK = "Sparrow-Repeat-Run.ipynb"
QUESTIONS = {
    "first": (
        "综合这份英文课件，用中文解释现场初检、潮湿复核和正式放行三种流程怎么选，各自有哪些条件？"
    ),
    "summary": "用两句话总结。",
    "show_saved": "请展示第 1 个 Cell 的保存输出。",
    "second": "刚才第二张图，横轴的单位是什么？",
    "pdf_visual": "第 4 页右侧图的横轴单位是什么？",
    "pdf_v2": "这份课件里潮湿复核需要等多久，允许的湿度范围是多少？",
    "version": "和旧版相比呢？",
    "cross_file": (
        "Sparrow 潮湿复核，讲义规定的适用条件是什么？"
        "实验记录的表格里 Residual drift 是多少？"
        "结合保存输出说明是否完成了规定的重复次数。"
    ),
    "missing": (
        "Sparrow 接头在盐雾室连续运行 72 小时以后的 Residual drift 是多少？"
        "湿台复核的读数能直接当作这个值吗？"
    ),
    "composition_missing": (
        "Sparrow 接头在盐雾室连续运行 72 小时后的 Residual drift 是多少？"
        "请根据讲义适用条件、实验表格和保存输出，判断潮湿复核数值能否代用。"
    ),
}
PAGES = (
    (
        "Canopy coupler field course",
        "An English field workshop. Start at the welcome desk and collect a badge.",
        "These notes separate screening, wet bench review, and release.",
    ),
    (
        "What an interval means",
        "Hold time starts only after the coupler and chamber have equilibrated.",
        "Illustrations are separate from instrument specifications.",
    ),
    (
        "On-site screening / dry intake",
        "Choose an on-site screening when the receiving coupler is dry and intact.",
        "Hold for 6 minutes after equilibration, without energizing it.",
        "A screening stamp does not authorize release.",
    ),
    (
        "Coupler diagnostic plots",
        "Two diagnostic plots show different variables. Axis units appear inside the panels.",
        "A panel axis unit must be read from the diagram, not from the paragraph.",
    ),
    (
        "Demonstration with spare leads",
        "The tray demo uses an open spare lead at 12 minutes. It is not a release process.",
        "No diagnostics were signed during this demonstration.",
    ),
    (
        "Humid bench review",
        "Use this route when an installed coupler is damp but has no salt exposure.",
        "Condition: after equilibration, wait at least {duration} minutes at {humidity}% RH.",
        "Run three complete repetitions before the technical review.",
        "Sparrow dry-bench values cannot stand in for continuous salt fog.",
        "No 72-hour salt fog Residual drift is available in this course.",
    ),
    (
        "Build records and responsibilities",
        "Record the equipment ID and preserve the instrument trace.",
        "Do not treat a saved demonstration image as approval.",
    ),
    (
        "Formal release",
        "Use formal release only after humid-bench review first passes.",
        "Require a signed sheet and a supervisor-confirmed seal code. "
        "A repeat screen is insufficient.",
        "The formal release log never supplies endurance qualification.",
    ),
)
SPECS = {1: (27, "55-65"), 2: (34, "48-58")}
NOTEBOOK_UNITS = {
    1: ("Samples [points]", "Displacement [mm]"),
    2: ("Displacement [mm]", "Samples [points]"),
}
PDF_PANEL_UNITS = {
    1: ("Temperature [C]", "Rotation [deg]"),
    2: ("Temperature [C]", "Rotation [rad]"),
}


def panel(label, color):
    image = Image.new("RGB", (840, 480), "#faf6e6")
    draw = ImageDraw.Draw(image)
    draw.rectangle((14, 14, 825, 465), outline="#1e3037", width=5)
    for x in (185, 313, 441, 569, 697):
        draw.line((x, 80, x, 332), fill="#cbcebd", width=3)
    draw.line((92, 71, 92, 334, 749, 334), fill="#1e3037", width=7)
    draw.line((100, 283, 205, 253, 333, 195, 498, 225, 636, 115, 737, 106), fill=color, width=13)
    draw.text((420, 394), label, fill="#101819", font=ImageFont.load_default(size=57), anchor="mm")
    return image


def pdf_stream(data, header=""):
    return f"<< {header} /Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream"


def write_pdf(path, version):
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    page_ids = []
    duration, humidity = SPECS[version]
    for number, page in enumerate(PAGES, 1):
        commands = ["0.05 0.20 0.25 rg 0 540 792 72 re f", "1 1 1 rg"]
        refs = []
        for line, original in enumerate(page):
            value = (
                original.format(duration=duration, humidity=humidity)
                .replace("\\", "\\\\")
                .replace("(", "\\(")
                .replace(")", "\\)")
            )
            y, size = (567, 24) if line == 0 else (517 - 27 * (line - 1), 14)
            if line == 1:
                commands.append("0.09 0.15 0.16 rg")
            commands.append(f"BT /Title {size} Tf 35 {y} Td ({value}) Tj ET")
        commands.append(f"BT /Title 10 Tf 713 30 Td (PAGE {number}) Tj ET")
        if number == 4:
            for name, x, label, color in zip(
                ("Left", "Right"),
                (37, 414),
                PDF_PANEL_UNITS[version],
                ("#192f96", "#963f19"),
                strict=True,
            ):
                image = panel(label, color)
                refs.append(f"/{name} {len(objects) + 1} 0 R")
                header = (
                    f"/Type /XObject /Subtype /Image /Width {image.width} /Height {image.height} "
                    "/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode"
                )
                objects.append(pdf_stream(zlib.compress(image.tobytes()), header))
                commands.append(f"q 340 0 0 194 {x} 171 cm /{name} Do Q")
        stream = len(objects) + 1
        objects.append(pdf_stream("\n".join(commands).encode("ascii")))
        page_ids.append(f"{len(objects) + 1} 0 R")
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 792 612] /Resources "
                f"<< /Font << /Title 3 0 R >> /XObject << {' '.join(refs)} >> >> "
                f"/Contents {stream} 0 R >>"
            ).encode()
        )
    objects[1] = f"<< /Type /Pages /Count {len(page_ids)} /Kids [{' '.join(page_ids)}] >>".encode()
    contents = bytearray(b"%PDF-1.4\n")
    starts = []
    for number, obj in enumerate(objects, 1):
        starts.append(len(contents))
        contents.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    startxref = len(contents)
    contents.extend(f"xref\n0 {len(starts) + 1}\n0000000000 65535 f \n".encode())
    for start in starts:
        contents.extend(f"{start:010d} 00000 n \n".encode())
    contents.extend(
        f"trailer\n<< /Size {len(starts) + 1} /Root 1 0 R >>\n"
        f"startxref\n{startxref}\n%%EOF\n".encode()
    )
    path.write_bytes(contents)


def write_deck(path):
    deck = Presentation()
    cover = deck.slides.add_slide(deck.slide_layouts[5])
    cover.shapes.title.text = "Sparrow coupler runs"
    cover.shapes.add_textbox(
        Inches(1), Inches(2), Inches(8), Inches(1)
    ).text = "This deck has a signed wet-bench table; it does not include salt fog tests."
    slide = deck.slides.add_slide(deck.slide_layouts[5])
    slide.shapes.title.text = "Sparrow humid bench review"
    rows = (
        ("Coupler", "Residual drift [mV]", "Disposition"),
        ("Sparrow", "3.6", "Signed humid bench"),
        ("Kingfisher", "11.8", "Fixture comparison"),
        ("Sparrow 72 h salt fog", "Not measured", "No qualification"),
    )
    table = slide.shapes.add_table(
        len(rows), len(rows[0]), Inches(0.8), Inches(2), Inches(8.3), Inches(3)
    ).table
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            table.cell(i, j).text = value
    buf = io.BytesIO()
    deck.save(buf)
    with ZipFile(buf) as raw, ZipFile(path, "w") as normalized:
        for name in sorted(raw.namelist()):
            normalized.writestr(ZipInfo(name), raw.read(name), compress_type=ZIP_DEFLATED)


def write_notebook(path, version):
    outputs = [
        {
            "output_type": "stream",
            "name": "stdout",
            "text": [
                "Sparrow humid bench review\n",
                "Completed repetitions: 3 of 3.\n",
                "No 72-hour salt fog run recorded.\n",
            ],
        }
    ]
    for label, ink in zip(NOTEBOOK_UNITS[version], ("#527a0a", "#89255b"), strict=True):
        buf = io.BytesIO()
        panel(label, ink).save(buf, format="PNG")
        outputs.append(
            {
                "output_type": "display_data",
                "metadata": {},
                "data": {"image/png": base64.b64encode(buf.getvalue()).decode()},
            }
        )
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "id": "wet-run",
                        "metadata": {},
                        "execution_count": 1,
                        "source": [
                            "# Sparrow humid bench review\n",
                            "# Read saved diagnostic outputs and the repetition count.\n",
                        ],
                        "outputs": outputs,
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build(out):
    files = []
    for version in SPECS:
        folder = out / "fixtures" / f"v{version}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / GUIDE
        write_pdf(path, version)
        files.append(path)
        path = folder / NOTEBOOK
        write_notebook(path, version)
        files.append(path)
    path = out / "fixtures" / DECK
    write_deck(path)
    files.append(path)
    manifest = {
        "course": COURSE,
        "questions": QUESTIONS,
        "files": {
            str(path.relative_to(out)): {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for path in files
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    build(parser.parse_args().out)
