"""Deterministic public course materials for a paired retrieval evaluation."""

from __future__ import annotations

import argparse
import io
import zlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches

COURSE = "calibration-workshop-public"
GUIDE = "Calibrated-Guard-Band.pdf"
RESULTS = "Cold-Venue-Readings.pptx"
PLATE = "Q-Lamp-Calibration-Plate.pdf"

CASES = {
    "formula": {
        "question": "边界保护的安全带宽度公式是什么？有哪些适用条件？",
        "gold": [(GUIDE, "page:2"), (GUIDE, "page:4")],
        "required": [
            r"2\s*[×*·]?\s*j\s*\+\s*1",
            r"11",
            r"镜像|mirrored",
            r"j[^。；]{0,15}(?:≥|>=|\\geq?|至少|不小于|大于等于)\s*2",
        ],
        "unallowed": [r"任意长度|任意采样"],
    },
    "formula_alt": {
        "question": "Kestrel 规则下，边缘安全带怎么计算？短采样窗或残缺采样可以套用吗？",
        "gold": [(GUIDE, "page:2"), (GUIDE, "page:4")],
        "required": [r"2\s*[×*·]?\s*j\s*\+\s*1", r"11", r"残缺|不完整|incomplete"],
        "unallowed": [r"适用于所有|任意长度"],
    },
    "cross_file": {
        "question": "Azure 模块在低温实验的稳定时间是多少？请核对表的列名、单位及适用条件。",
        "gold": [(RESULTS, "table:azure"), (GUIDE, "page:6")],
        "required": [
            r"18\.4",
            r"ms|毫秒",
            r"4\s*(?:°\s*C|℃|摄氏|度|C\b)",
            r"40\s*%",
            r"Settling time|稳定时间",
        ],
        "unallowed": [r"稳定时间(?:是|为)\s*7\.3"],
    },
    "missing": {
        "question": "Azure 模块在露天湿度 90% RH 时的稳定时间是多少？能否用低温实验值？",
        "gold": [(GUIDE, "page:6")],
        "required": [r"无法|不足|没有|未给|未提供|未记录|不能|不可"],
        "unallowed": [
            r"90\s*%[^。！？]{0,20}(?:是|为|等于|预计)\s*(?:7\.3|18\.4)",
            r"(?<!不)可(?:以)?(?:直接)?外推(?!的)",
        ],
    },
    "caption": {
        "question": "Q-lamp 热保持后的衰减图中，右侧图横轴单位是什么？",
        "gold": [(PLATE, "page:1"), (PLATE, "page:1:image:2")],
        "caption_phrase": "Q-lamp drift after thermal hold",
        "visual_target": (PLATE, "page:1:image:2"),
        "required_vision": r"\bnm\b|纳米|nanomet(?:er|re)s?",
        "unallowed": [r"单位[是为]\s*(?:毫秒|ms)"],
    },
    "linked_ticket": {
        "question": "Tamarack 的放行阈值是多少？",
        "gold": [(GUIDE, "page:3"), (GUIDE, "page:9")],
        "required": [r"31", r"\bdB\b|分贝"],
        "unallowed": [r"放行阈值(?:是|为)\s*18\.4"],
    },
}

_GUIDE_PAGES = (
    (
        "Kestrel calibration workshop",
        "Fictitious lab records for practice with instrument boundaries and measurement scope.",
        "The workshop distinguishes a tested calibration protocol from field extrapolation.",
    ),
    (
        "Edge safety band calculation",
        "For the Kestrel boundary profile, the edge safety band is B = 2*j + 1 samples.",
        "The integer j is the extent of the mirrored band on one side.",
        "Check the KSB-17 qualification sheet before using the expression.",
    ),
    (
        "Instrument naming and availability",
        "Azure, Amber and Citrine are three lab module names and not interchangeable conditions.",
        "The run log and the signed qualification sheet describe different measurements.",
        "Tamarack bench release threshold is defined by linked ticket DXW-63.",
        "Ticket columns are ID, level, and unit.",
    ),
    (
        "KSB-17 qualification sheet",
        "For KSB-17, use j >= 2 and a complete window L >= 11 samples.",
        "Use mirrored padding. Zero padding and incomplete captures are outside the rule.",
    ),
    (
        "Azure warm bench demonstration",
        "The warm demonstration reads 7.3 ms under a different load and temperature.",
        "It is not a cold-venue qualification measurement or an outdoor humidity result.",
    ),
    (
        "Cold-venue validity for Azure",
        "Azure cold-venue qualification applies only at ambient <= 4 C and humidity <= 40% RH.",
        "Cold-Venue-Readings is the signed table. Its settling time is not transferable outdoors.",
        "There is no recorded outdoor test at 90% RH. Do not extrapolate the cold-venue row.",
    ),
    ("QA-22", "HOLD"),
    ("VAL-12", "ARCHIVE"),
    ("DXW-63", "DXW-63 | 31 | dB"),
)


def _raster(label: str, color: str) -> Image.Image:
    image = Image.new("RGB", (720, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 710, 470), outline=color, width=14)
    draw.line((86, 90, 86, 335, 635, 335), fill="black", width=7)
    draw.line((101, 130, 190, 223, 315, 283, 552, 301), fill=color, width=14)
    font = ImageFont.load_default(size=51)
    draw.text((360, 403), label, fill="black", font=font, anchor="mm")
    return image


def _stream(data: bytes, dictionary: str = "") -> bytes:
    return f"<< {dictionary} /Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream"


def _write_pdf(path: Path, pages: tuple[tuple[str, ...], ...], *, panels: bool = False) -> None:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    children = []
    for index, lines in enumerate(pages):
        commands = []
        for line, original in enumerate(lines):
            value = original.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            commands.append(f"BT /F0 12 Tf 32 {742 - line * 24} Td ({value}) Tj ET")
        references = []
        if panels and index == 0:
            for name, x, label, color in (
                ("Left", 24, "Delay [ms]", "#123a86"),
                ("Right", 319, "Wavelength [nm]", "#7d1457"),
            ):
                raster = _raster(label, color)
                image_id = len(objects) + 1
                objects.append(
                    _stream(
                        zlib.compress(raster.tobytes()),
                        f"/Type /XObject /Subtype /Image /Width {raster.width} "
                        f"/Height {raster.height} "
                        "/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode",
                    )
                )
                references.append(f"/{name} {image_id} 0 R")
                commands.append(f"q 250 0 0 210 {x} 385 cm /{name} Do Q")
        stream_id = len(objects) + 1
        objects.append(_stream("\n".join(commands).encode("ascii")))
        page_id = len(objects) + 1
        children.append(f"{page_id} 0 R")
        reference = " ".join(references)
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F0 3 0 R >> /XObject << {reference} >> >> "
                f"/Contents {stream_id} 0 R >>"
            ).encode()
        )
    objects[1] = f"<< /Type /Pages /Count {len(children)} /Kids [{' '.join(children)}] >>".encode()
    contents = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(contents))
        contents.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(contents)
    contents.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        contents.extend(f"{offset:010d} 00000 n \n".encode())
    contents.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n".encode())
    contents.extend(f"startxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(contents)


def _write_table(path: Path) -> None:
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[5])
    slide.shapes.title.text = "Cold-venue instrument qualification"
    table = slide.shapes.add_table(4, 3, Inches(1), Inches(2), Inches(8), Inches(2)).table
    rows = (
        ("Module", "Settling time [ms]", "Lab status"),
        ("Azure", "18.4", "Qualified"),
        ("Amber", "41.0", "Qualified"),
        ("Citrine", "not recorded", "Not tested"),
    )
    for row, cells in enumerate(rows):
        for column, cell in enumerate(cells):
            table.cell(row, column).text = cell
    slide.shapes.add_textbox(
        Inches(1), Inches(4.5), Inches(8), Inches(1)
    ).text = "Apply the cold-venue validity limits in the signed calibration guide."
    stream = io.BytesIO()
    deck.save(stream)
    with ZipFile(stream) as raw, ZipFile(path, "w") as output:
        for name in sorted(raw.namelist()):
            output.writestr(ZipInfo(name), raw.read(name), compress_type=ZIP_DEFLATED)


def make_fixture(directory: Path) -> tuple[Path, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    guide, result, plate = (directory / name for name in (GUIDE, RESULTS, PLATE))
    _write_pdf(guide, _GUIDE_PAGES)
    _write_table(result)
    _write_pdf(
        plate,
        (
            (
                "Calibration plate",
                "Figure 4. Q-lamp drift after thermal hold.",
                "The right panel is the wavelength plot. The axis unit is printed in the panel.",
                "The left panel is a separate delay measurement.",
            ),
            (
                "Q-lamp warmup log",
                "The previous warmup inspection is not the thermal-hold plate.",
                "The two quantities from the actual plate must remain distinct.",
            ),
        ),
        panels=True,
    )
    return guide, result, plate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    make_fixture(parser.parse_args().directory)
