#!/usr/bin/env python3
"""Build and freeze the shareable public demo course in examples/demo-course/.

Downloads two fresh CC BY 4.0 arXiv papers (2611.07820v1 and 2407.08608v2),
generates companion Lecture-01-Retrieval-and-Attention.pptx,
v1/Lab-01-Retrieval-Bench.ipynb, v2/Lab-01-Retrieval-Bench.ipynb,
Distractor-Notes.pdf, NOTICE.txt, MANIFEST.json, and eval_cases.json,
then verifies LocalLibrary parsing and LocalKgStore concept extraction.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import nbformat
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = Path(__file__).resolve().parent

PAPERS = [
    {
        "file_name": "Axis-LoRA-2611.07820v1.pdf",
        "arxiv_id": "2611.07820v1",
        "title": "Axis-LoRA: Probing and Adapting Embedding Models for Personal-Memory Retrieval",
        "license": "CC BY 4.0",
        "license_url": "http://creativecommons.org/licenses/by/4.0/",
        "abs_url": "https://arxiv.org/abs/2611.07820v1",
        "pdf_url": "https://arxiv.org/pdf/2611.07820v1.pdf",
    },
    {
        "file_name": "FlashAttention-3-2407.08608v2.pdf",
        "arxiv_id": "2407.08608v2",
        "title": "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision",
        "license": "CC BY 4.0",
        "license_url": "http://creativecommons.org/licenses/by/4.0/",
        "abs_url": "https://arxiv.org/abs/2407.08608v2",
        "pdf_url": "https://arxiv.org/pdf/2407.08608v2.pdf",
    },
]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download_papers(*, refresh: bool = False) -> None:
    for spec in PAPERS:
        target = DEMO_DIR / spec["file_name"]
        if target.exists() and not refresh:
            continue
        req = urllib.request.Request(
            spec["pdf_url"], headers={"User-Agent": "SRAG-K02-DemoCourse/1.0"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if not data.startswith(b"%PDF-"):
            raise RuntimeError(f"Downloaded file for {spec['arxiv_id']} is not a PDF")
        target.write_bytes(data)


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for candidate in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _render_curve_png(
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, tuple[int, int, int], list[tuple[int, int]]]],
    annotations: list[tuple[int, int, str, tuple[int, int, int]]],
    width: int = 920,
    height: int = 520,
) -> bytes:
    img = Image.new("RGB", (width, height), (250, 252, 251))
    draw = ImageDraw.Draw(img)
    f_title = _font(24)
    f_axis = _font(19)
    f_note = _font(18)
    # Border & plot box
    draw.rectangle((12, 12, width - 12, height - 12), outline=(210, 222, 218), width=2)
    left, top, right, bottom = 140, 82, width - 50, height - 88
    draw.rectangle((left, top, right, bottom), outline=(70, 88, 82), width=2)
    # Grid lines
    for frac in (0.25, 0.5, 0.75):
        gy = int(top + frac * (bottom - top))
        draw.line([(left, gy), (right, gy)], fill=(230, 236, 234), width=1)
    draw.text((34, 24), title, fill=(20, 34, 30), font=f_title)
    draw.text((left + 160, height - 48), x_label, fill=(35, 50, 45), font=f_axis)
    draw.text((18, (top + bottom) // 2 - 14), y_label, fill=(35, 50, 45), font=f_axis)
    # Curves & legend
    lx = left + 20
    for idx, (label, color, pts) in enumerate(series):
        draw.line(pts, fill=color, width=5)
        for px, py in pts:
            draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color)
        ly = top + 16 + idx * 30
        draw.rectangle((lx, ly + 4, lx + 24, ly + 18), fill=color)
        draw.text((lx + 34, ly), label, fill=(25, 38, 34), font=f_note)
    for ax, ay, text, color in annotations:
        draw.text((ax, ay), text, fill=color, font=f_note)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _render_simple_box_png(title: str, subtitle: str, bg: tuple[int, int, int], border: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (640, 380), bg)
    draw = ImageDraw.Draw(img)
    f_title = _font(22)
    f_body = _font(18)
    draw.rectangle((14, 14, 626, 366), outline=border, width=5)
    draw.rectangle((54, 104, 272, 224), fill=(255, 255, 255), outline=border, width=3)
    draw.rectangle((368, 104, 586, 224), fill=(255, 255, 255), outline=border, width=3)
    draw.line([(272, 164), (368, 164)], fill=border, width=4)
    draw.text((34, 34), title, fill=(18, 30, 26), font=f_title)
    draw.text((34, 274), subtitle, fill=(28, 44, 38), font=f_body)
    draw.text((88, 152), "Stage-A Gate", fill=border, font=f_body)
    draw.text((400, 152), "Stage-B Cache", fill=border, font=f_body)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _build_pptx(path: Path) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    def add_text_slide(title: str, lines: list[str]) -> None:
        slide = prs.slides.add_slide(blank)
        tb = slide.shapes.add_textbox(Inches(0.65), Inches(0.45), Inches(12.0), Inches(0.95))
        tf = tb.text_frame
        tf.word_wrap = True
        p0 = tf.paragraphs[0]
        p0.text = title
        p0.font.size = Pt(26)
        p0.font.bold = True

        body = slide.shapes.add_textbox(Inches(0.65), Inches(1.55), Inches(12.0), Inches(5.3))
        btf = body.text_frame
        btf.word_wrap = True
        for i, line in enumerate(lines):
            p = btf.paragraphs[0] if i == 0 else btf.add_paragraph()
            p.text = line
            p.font.size = Pt(18)
            p.space_after = Pt(10)

    # Slide 1: Cover / Agenda anti-example (R1 cooccurrence_only)
    add_text_slide(
        "Course Outline: Retrieval & Attention Systems",
        [
            "Agenda: 1. 轴向门控路由 2. 双阶异步流水 3. 单向边界守卫",
            "Module overview only; definitions and benchmark scripts appear on later slides.",
        ],
    )

    # Slide 2: Concept 1 (Axis-Gated Routing -> probe_axis_gate)
    add_text_slide(
        "1. 核心概念定义：轴向门控路由 (Axis-Gated Routing)",
        [
            "定义：轴向门控路由 (Axis-Gated Routing) 是对注意力头灵敏度谱驱动的嵌入低秩投影缩放机制，门控权重满足 0 < g_l <= 1.0，蒸馏正则系数 lambda_kd = 0.15；适用条件为编码器已完成轴向反事实探针校准，未校准时退回余弦相似度检索。",
            "Experiment-ID: AxisGate1",
            "实验验证请参见配套实验脚本中函数 `probe_axis_gate` 的实测输出。",
        ],
    )

    # Slide 3: Concept 2 (Two-Stage Asynchronous Pipeline -> bench_fa3_pingpong)
    add_text_slide(
        "2. 核心概念定义：双阶异步流水 (Two-Stage Asynchronous Pipeline)",
        [
            "定义：双阶异步流水 (Two-Stage Asynchronous Pipeline) 是指在双战团并发调度下将矩阵乘法 WGMMA 与指数归一化 Softmax 异步重叠执行，张量块维度设为 (128, 128)，FP8 路径启用非相干量化；适用条件为硬件具备异步张量内存加速器 TMA 与可重排共享内存。",
            "Experiment-ID: FA3Ping1",
            "对应基准测试请见配套实验中的函数 `bench_fa3_pingpong` 及其保存输出。",
        ],
    )

    # Slide 4: Concept 3 (No experiment code -> fallback_to_search)
    add_text_slide(
        "3. 核心概念定义：单向边界守卫 (Unidirectional Boundary Guard)",
        [
            "正式定义：单向边界守卫在缺少参考基线时限制上下文窗口外推倍率 <= 4x，防止长程注意权重在尾部位置发生数值发散。",
            "说明：本概念仅在讲义中给出理论界定，当前课程暂无配套实验代码演示。",
        ],
    )

    # Slide 5: Native 5x5 PPTX table with headers & units
    slide5 = prs.slides.add_slide(blank)
    tb5 = slide5.shapes.add_textbox(Inches(0.65), Inches(0.4), Inches(12.0), Inches(0.8))
    p5 = tb5.text_frame.paragraphs[0]
    p5.text = "表 1 · 调度配置实测指标对比（单位：GB/s、ms、MB、°C）"
    p5.font.size = Pt(24)
    p5.font.bold = True

    rows = [
        ["调度配置", "硬件带宽 (GB/s)", "端到端延迟 (ms)", "峰值显存 (MB)", "核心温度 (°C)"],
        ["Sync-Baseline", "1840 GB/s", "28.4 ms", "4120 MB", "76.5 °C"],
        ["FA3-Pingpong-FP16", "2680 GB/s", "14.2 ms", "3680 MB", "72.1 °C"],
        ["FA3-Pingpong-FP8", "3120 GB/s", "9.6 ms", "2940 MB", "68.4 °C"],
        ["Axis-LoRA-Gate", "2410 GB/s", "11.8 ms", "3190 MB", "70.2 °C"],
    ]
    tbl_shape = slide5.shapes.add_table(
        len(rows), len(rows[0]), Inches(0.65), Inches(1.45), Inches(12.0), Inches(4.6)
    )
    table = tbl_shape.table
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            para = cell.text_frame.paragraphs[0]
            para.font.size = Pt(16)
            if r_idx == 0:
                para.font.bold = True
                para.font.color.rgb = RGBColor(20, 40, 34)

    # Slide 6: Pure-diagram / ultra-low-text slide (<40 chars text) for C6 S0 vs S3 test
    slide6 = prs.slides.add_slide(blank)
    tb6 = slide6.shapes.add_textbox(Inches(0.6), Inches(0.25), Inches(3.2), Inches(0.45))
    tb6.text_frame.paragraphs[0].text = "Slide 6"
    tb6.text_frame.paragraphs[0].font.size = Pt(14)
    pure_png = _render_curve_png(
        title="Async Latency Scaling vs Context Length",
        x_label="Sequence Length (tokens: 2K / 4K / 8K / 16K)",
        y_label="Step Latency (ms)",
        series=[
            ("Sync-Baseline", (175, 58, 46), [(150, 340), (330, 290), (540, 200), (740, 100)]),
            ("FA3-Pingpong", (22, 118, 86), [(150, 355), (330, 330), (540, 285), (740, 232)]),
        ],
        annotations=[
            (315, 244, "4K: 14.2 ms", (175, 58, 46)),
            (315, 344, "4K: 6.8 ms", (22, 118, 86)),
            (635, 74, "16K: 48.6 ms", (175, 58, 46)),
            (635, 254, "16K: 19.4 ms", (22, 118, 86)),
        ],
    )
    slide6.shapes.add_picture(
        io.BytesIO(pure_png), Inches(0.8), Inches(0.85), width=Inches(11.7), height=Inches(6.2)
    )

    # Slide 7: Side-by-side two images (left vs right orientation selection)
    slide7 = prs.slides.add_slide(blank)
    tb7 = slide7.shapes.add_textbox(Inches(0.65), Inches(0.35), Inches(12.0), Inches(0.75))
    p7 = tb7.text_frame.paragraphs[0]
    p7.text = "Slide 7 · Dual-Chart Architecture Comparison (Left vs Right)"
    p7.font.size = Pt(22)
    p7.font.bold = True

    left_png = _render_simple_box_png(
        "Left: Static Router (Recall@5 = 71.2%)",
        "Fixed cosine score; latency = 18.5 ms",
        (244, 248, 252),
        (40, 96, 154),
    )
    right_png = _render_simple_box_png(
        "Right: Axis-Gated Router (Recall@5 = 86.9%)",
        "Per-axis weights; latency = 11.4 ms",
        (244, 251, 247),
        (25, 118, 84),
    )
    slide7.shapes.add_picture(
        io.BytesIO(left_png), Inches(0.65), Inches(1.4), width=Inches(5.8), height=Inches(4.2)
    )
    slide7.shapes.add_picture(
        io.BytesIO(right_png), Inches(6.85), Inches(1.4), width=Inches(5.8), height=Inches(4.2)
    )

    # Slide 8: Grouped unboxed diagram (shape_id_path len > 1 -> parent_page_fallback)
    slide8 = prs.slides.add_slide(blank)
    tb8 = slide8.shapes.add_textbox(Inches(0.65), Inches(0.35), Inches(12.0), Inches(0.75))
    p8 = tb8.text_frame.paragraphs[0]
    p8.text = "Slide 8 · Grouped Pipeline Schematic (Unboxed Fallback Demo)"
    p8.font.size = Pt(22)
    p8.font.bold = True

    grp = slide8.shapes.add_group_shape()
    grp_png = _render_simple_box_png(
        "Grouped Two-Stage Flow",
        "Step 1: Axis Probe -> Step 2: Async GEMM+Softmax",
        (250, 249, 244),
        (146, 94, 28),
    )
    grp.shapes.add_picture(
        io.BytesIO(grp_png), Inches(1.2), Inches(1.5), width=Inches(10.8), height=Inches(5.0)
    )
    cap_box = grp.shapes.add_textbox(Inches(1.2), Inches(6.55), Inches(10.8), Inches(0.5))
    cap_box.text_frame.paragraphs[0].text = "Grouped Figure: Unified dispatch and cache writeback"

    prs.save(path)


def _build_ipynb(path: Path, *, version: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if version == 1:
        axis_png = _render_curve_png(
            title="Axis-Gate Probe v1: Recall@10 vs Rank",
            x_label="LoRA Rank (r = 4 / 8 / 16)",
            y_label="Recall@10 (%)",
            series=[
                ("Axis-LoRA", (22, 118, 86), [(150, 300), (430, 195), (720, 145)]),
                ("Frozen Base", (130, 138, 136), [(150, 325), (430, 325), (720, 325)]),
            ],
            annotations=[
                (610, 118, "r=16: 86.4%", (22, 118, 86)),
                (610, 298, "Base: 71.2%", (100, 108, 106)),
            ],
        )
        fa3_out1_png = _render_curve_png(
            title="FA3 Pingpong Output 1: Throughput vs Batch",
            x_label="Batch Size (1 / 4 / 8 / 16)",
            y_label="TFLOP/s",
            series=[
                ("FP16 Pingpong", (36, 98, 158), [(150, 310), (350, 230), (540, 170), (740, 128)]),
            ],
            annotations=[(630, 102, "Peak: 662 TFLOP/s", (36, 98, 158))],
        )
        fa3_out2_png = _render_curve_png(
            title="FA3 Pingpong Output 2: Quantization Error vs SeqLen",
            x_label="Context Tokens (K)",
            y_label="RMSE (1e-3)",
            series=[
                ("FP8 Incoherent", (175, 58, 46), [(150, 295), (350, 260), (540, 218), (740, 180)]),
            ],
            annotations=[(605, 152, "Peak RMSE: 4.12e-3", (175, 58, 46))],
        )
        cells = [
            nbformat.v4.new_markdown_cell(
                "# Lab-01 Retrieval & Attention Benchmark (v1)\n"
                "Reproducible saved outputs for axis gating and asynchronous attention."
            ),
            nbformat.v4.new_code_cell(
                "# Run v1 axis probe without repeating lecture concept labels\n"
                "metrics = probe_axis_gate(lambda_kd=0.15, lora_rank=16)\n"
                "print(f\"probe_axis_gate Recall@10={metrics['recall']:.1f}%\")",
                outputs=[
                    nbformat.v4.new_output(
                        output_type="stream",
                        text=(
                            "Experiment-ID: AxisGate1; Split: test; Metric: recall; Unit: %; "
                            "Baseline: 71.2; Trial: 86.4; Samples: 1000.\n"
                            "probe_axis_gate Recall@10=86.4%\n"
                        ),
                    ),
                    nbformat.v4.new_output(
                        output_type="display_data",
                        data={
                            "image/png": base64.b64encode(axis_png).decode("ascii"),
                            "text/plain": ["<AxisProbeCurve v1>"],
                        },
                        metadata={},
                    ),
                ],
            ),
            nbformat.v4.new_code_cell(
                "# Benchmark v1 asynchronous schedule with dual curve outputs\n"
                "stats = bench_fa3_pingpong(block_shape=(128, 128), precision='fp8')\n"
                "print(f\"bench_fa3_pingpong fa3_throughput={stats['tflops']} TFLOP/s, fp8_rmse={stats['rmse']}e-3\")",
                outputs=[
                    nbformat.v4.new_output(
                        output_type="stream",
                        text=(
                            "Experiment-ID: FA3Ping1; Split: test; Metric: throughput; Unit: TFLOP/s; "
                            "Baseline: 410; Trial: 662; Samples: 512.\n"
                            "bench_fa3_pingpong fa3_throughput=662 TFLOP/s, fp8_rmse=4.12e-3\n"
                        ),
                    ),
                    nbformat.v4.new_output(
                        output_type="display_data",
                        data={
                            "image/png": base64.b64encode(fa3_out1_png).decode("ascii"),
                            "text/plain": ["<FA3ThroughputCurve v1>"],
                        },
                        metadata={},
                    ),
                    nbformat.v4.new_output(
                        output_type="display_data",
                        data={
                            "image/png": base64.b64encode(fa3_out2_png).decode("ascii"),
                            "text/plain": ["<FA3ErrorCurve v1>"],
                        },
                        metadata={},
                    ),
                ],
            ),
            nbformat.v4.new_code_cell(
                "# Do NOT treat this cell as 轴向门控路由 or 双阶异步流水;\n"
                "# it does NOT implement or demonstrate probe_axis_gate or bench_fa3_pingpong.\n"
                "baseline_score = 42.0\nprint(f'baseline_only={baseline_score}')",
                outputs=[nbformat.v4.new_output(output_type="stream", text="baseline_only=42.0\n")],
            ),
            nbformat.v4.new_code_cell(
                "# Planned evaluation for future release\n"
                "pending_run = bench_fa3_pingpong(block_shape=(64, 64), precision='fp4')",
                outputs=[],
            ),
        ]
    else:
        # v2: reordered cells, updated measured metrics, removed bench_fa3_pingpong edge
        axis_v2_png = _render_curve_png(
            title="Axis-Gate Probe v2: Recall@10 vs Rank",
            x_label="LoRA Rank (r = 4 / 8 / 16)",
            y_label="Recall@10 (%)",
            series=[
                ("Axis-LoRA v2", (28, 100, 152), [(150, 290), (430, 168), (720, 118)]),
            ],
            annotations=[(600, 92, "r=16 v2: 91.8%", (28, 100, 152))],
        )
        cells = [
            nbformat.v4.new_markdown_cell(
                "# Lab-01 Retrieval & Attention Benchmark (v2)\n"
                "Updated v2 run: cell reordered, axis metric updated to 91.8%, async demo moved."
            ),
            nbformat.v4.new_code_cell(
                "# Updated placeholder cell without async benchmark edge\n"
                "print('maintenance_check=ok')",
                outputs=[nbformat.v4.new_output(output_type="stream", text="maintenance_check=ok\n")],
            ),
            nbformat.v4.new_code_cell(
                "# Run v2 axis probe with recalibrated weights\n"
                "metrics = probe_axis_gate(lambda_kd=0.15, lora_rank=16)\n"
                "print(f\"probe_axis_gate Recall@10={metrics['recall']:.1f}%\")",
                outputs=[
                    nbformat.v4.new_output(
                        output_type="stream",
                        text=(
                            "Experiment-ID: AxisGate1; Split: test; Metric: recall; Unit: %; "
                            "Baseline: 71.2; Trial: 91.8; Samples: 1000.\n"
                            "probe_axis_gate Recall@10=91.8%\n"
                        ),
                    ),
                    nbformat.v4.new_output(
                        output_type="display_data",
                        data={
                            "image/png": base64.b64encode(axis_v2_png).decode("ascii"),
                            "text/plain": ["<AxisProbeCurve v2>"],
                        },
                        metadata={},
                    ),
                ],
            ),
        ]
    nb = nbformat.v4.new_notebook(cells=cells)
    for idx, cell in enumerate(nb.cells):
        if "probe_axis_gate(" in cell.source:
            cell.id = "lab01-axis-probe"
        else:
            cell.id = f"lab01-v{version}-c{idx}"
    nbformat.write(nb, path)


def _write_type1_pdf(path: Path, pages: list[tuple[str, list[str]]]) -> None:
    def esc(text: str) -> bytes:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode("ascii")

    kids = " ".join(f"{4 + idx * 2} 0 R" for idx in range(len(pages)))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for page_num, (heading, lines) in enumerate(pages, start=1):
        stream_id = 2 * page_num + 3
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 650 460] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {stream_id} 0 R >>"
            ).encode("ascii")
        )
        stream = b"BT /F1 18 Tf 42 405 Td (" + esc(heading) + b") Tj ET\n"
        for idx, line in enumerate(lines):
            stream += b"BT /F1 11 Tf 42 %d Td (" % (366 - idx * 22) + esc(line) + b") Tj ET\n"
        stream += b"BT /F1 10 Tf 42 32 Td (Page %d) Tj ET\n" % page_num
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream")

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(b"%d 0 obj\n" % idx + obj + b"\nendobj\n")
    xref_pos = len(out)
    out.extend(b"xref\n0 %d\n0000000000 65535 f \n" % len(offsets))
    for off in offsets[1:]:
        out.extend(b"%010d 00000 n \n" % off)
    out.extend(
        b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(offsets), xref_pos)
    )
    path.write_bytes(bytes(out))


def _write_notice_and_eval_cases() -> None:
    notice = """SRAG Shareable Demo Course Materials — License & Attribution Notice

1. Axis-LoRA-2611.07820v1.pdf
   Title: Axis-LoRA: Probing and Adapting Embedding Models for Personal-Memory Retrieval
   Source: https://arxiv.org/abs/2611.07820v1
   License: Creative Commons Attribution 4.0 International (CC BY 4.0)
   License URL: http://creativecommons.org/licenses/by/4.0/

2. FlashAttention-3-2407.08608v2.pdf
   Title: FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision
   Source: https://arxiv.org/abs/2407.08608v2
   License: Creative Commons Attribution 4.0 International (CC BY 4.0)
   License URL: http://creativecommons.org/licenses/by/4.0/

3. Companion synthetic materials (Lecture-01-Retrieval-and-Attention.pptx,
   v1/Lab-01-Retrieval-Bench.ipynb, v2/Lab-01-Retrieval-Bench.ipynb, Distractor-Notes.pdf)
   were authored for SRAG K02 product verification under CC BY 4.0 / repository license.
"""
    (DEMO_DIR / "NOTICE.txt").write_text(notice, encoding="utf-8")

    eval_cases = {
        "schema_version": 1,
        "suite_id": "k02-demo-course-public-v1",
        "course_id": "demo-course",
        "distractor_course_id": "demo-distractor",
        "thresholds": {
            "completion_rate_min": 1.0,
            "severe_error_max": 0,
            "all_2_rate_min": 0.9,
            "visual_selection_alignment_min": 1.0,
        },
        "cases": [
            {
                "case_id": "K-Pub-01",
                "category": "cross_file_definition_and_experiment",
                "turns": [
                    {
                        "turn_id": "K-Pub-01-T1",
                        "query": "请说明 轴向门控路由 的定义，给出配套实验的实测 recall 结果。",
                        "expected_sources": [
                            {"file_name": "Lecture-01-Retrieval-and-Attention.pptx", "page": 2},
                            {"file_name": "Lab-01-Retrieval-Bench.ipynb", "cell_index": 1},
                        ],
                        "must_contain_facts": ["0 < g_l <= 1.0", "0.15", "86.4%"],
                        "applicable_dimensions": ["fact", "citation", "coverage"],
                    }
                ],
            },
            {
                "case_id": "K-Pub-02",
                "category": "two_column_pdf_formula_and_figure",
                "turns": [
                    {
                        "turn_id": "K-Pub-02-T1",
                        "file_filter": "Axis-LoRA-2611.07820v1.pdf",
                        "query": "Axis-LoRA 第 6 页的公式是什么？第 3 页的图片展示了什么趋势？",
                        "expected_sources": [
                            {"file_name": "Axis-LoRA-2611.07820v1.pdf", "page": 6},
                            {"file_name": "Axis-LoRA-2611.07820v1.pdf", "page": 3},
                        ],
                        "must_contain_facts": ["InfoNCE"],
                        "applicable_dimensions": ["fact", "citation", "coverage"],
                    }
                ],
            },
            {
                "case_id": "K-Pub-03",
                "category": "pdf_and_pptx_tables_with_units",
                "turns": [
                    {
                        "turn_id": "K-Pub-03-T1",
                        "file_filter": "Lecture-01-Retrieval-and-Attention.pptx",
                        "query": "表 1 中 FA3-Pingpong-FP8 的硬件带宽、端到端延迟、峰值显存和核心温度分别是多少？",
                        "expected_sources": [
                            {"file_name": "Lecture-01-Retrieval-and-Attention.pptx", "page": 5},
                        ],
                        "must_contain_facts": ["3120", "9.6", "2940", "68.4"],
                        "applicable_dimensions": ["fact", "citation", "coverage"],
                    }
                ],
            },
            {
                "case_id": "K-Pub-04",
                "category": "notebook_multi_output_and_followup",
                "turns": [
                    {
                        "turn_id": "K-Pub-04-T1",
                        "focus": {"file_name": "Lab-01-Retrieval-Bench.ipynb", "cell_index": 2},
                        "expect_clarification": True,
                        "choose_label_contains": "输出 2",
                        "query": "请读取当前 Cell 输出曲线图的横轴单位和峰值数值。",
                        "expected_sources": [
                            {"file_name": "Lab-01-Retrieval-Bench.ipynb", "source_id": "cell:2:output:2"},
                        ],
                        "must_contain_facts": ["4.12", "RMSE"],
                        "applicable_dimensions": ["fact", "citation", "coverage"],
                    },
                    {
                        "turn_id": "K-Pub-04-T2",
                        "query": "刚才那张图的纵轴单位和曲线标签是什么？",
                        "expected_sources": [
                            {"file_name": "Lab-01-Retrieval-Bench.ipynb", "source_id": "cell:2:output:2"},
                        ],
                        "must_contain_facts": ["FP8 Incoherent"],
                        "applicable_dimensions": ["fact", "citation", "coverage", "consistency"],
                    },
                ],
            },
            {
                "case_id": "K-Pub-05",
                "category": "multi_image_direction_and_unboxed_fallback",
                "turns": [
                    {
                        "turn_id": "K-Pub-05-T1",
                        "focus": {"file_name": "Lecture-01-Retrieval-and-Attention.pptx", "page": 7},
                        "query": "第 7 页右侧那张图展示了什么指标和延迟？",
                        "expected_sources": [
                            {"file_name": "Lecture-01-Retrieval-and-Attention.pptx", "page": 7},
                        ],
                        "must_contain_facts": ["86.9%", "11.4"],
                        "applicable_dimensions": ["fact", "citation", "coverage"],
                    }
                ],
            },
            {
                "case_id": "K-Pub-06",
                "category": "pure_diagram_pptx_s0_vs_s3",
                "turns": [
                    {
                        "turn_id": "K-Pub-06-T1",
                        "strategy_compare": ["s0", "vector"],
                        "query": "异步流水延迟曲线图展示在哪一页幻灯片？请读出图中 4K 和 16K 序列长度对应的全部四个 Step Latency (ms) 实测数值。",
                        "expected_sources": [
                            {"file_name": "Lecture-01-Retrieval-and-Attention.pptx", "page": 6},
                        ],
                        "must_contain_facts": ["14.2", "6.8", "48.6", "19.4"],
                        "applicable_dimensions": ["fact", "citation", "coverage"],
                    }
                ],
            },
            {
                "case_id": "K-Pub-07",
                "category": "clarification_disambiguation",
                "turns": [
                    {
                        "turn_id": "K-Pub-07-T1",
                        "focus": {"file_name": "FlashAttention-3-2407.08608v2.pdf", "page": 6},
                        "expect_clarification": True,
                        "choose_index": 0,
                        "query": "第 6 页的图片展示了什么调度？",
                        "expected_sources": [
                            {"file_name": "FlashAttention-3-2407.08608v2.pdf", "page": 6},
                        ],
                        "must_contain_facts": ["pingpong"],
                        "applicable_dimensions": ["fact", "citation", "coverage"],
                    }
                ],
            },
            {
                "case_id": "K-Pub-08",
                "category": "insufficient_or_out_of_scope",
                "turns": [
                    {
                        "turn_id": "K-Pub-08-T1",
                        "file_filter": "Axis-LoRA-2611.07820v1.pdf",
                        "query": "请给出 Axis-LoRA 论文中测得的 1-bit FP1.58 量化在 H100 8-GPU 集群上的训练能耗焦耳数以及 Lab-01 的 fp8_rmse 数值。",
                        "expected_sources": [],
                        "must_not_fabricate": True,
                        "applicable_dimensions": ["fact", "coverage"],
                    }
                ],
            },
            {
                "case_id": "K-Pub-09",
                "category": "kg_dual_entry_navigation_and_fallback",
                "turns": [
                    {
                        "turn_id": "K-Pub-09-T1",
                        "query": "请说明 双阶异步流水 的定义，给出配套实验的实测结果。",
                        "expected_concepts": ["轴向门控路由", "双阶异步流水", "单向边界守卫"],
                        "kg_navigation_targets": [
                            {"file_name": "Lecture-01-Retrieval-and-Attention.pptx", "page": 3},
                            {"file_name": "Lab-01-Retrieval-Bench.ipynb", "cell_index": 2, "must_contain_facts": ["662"]},
                        ],
                        "must_contain_facts": ["128"],
                        "applicable_dimensions": ["fact", "citation", "coverage"],
                    }
                ],
            },
            {
                "case_id": "K-Pub-10",
                "category": "version_upgrade_and_historical_view",
                "turns": [
                    {
                        "turn_id": "K-Pub-10-T1",
                        "upload_v2_before_turn": True,
                        "query": "请对比最新版和旧版 Lab-01-Retrieval-Bench.ipynb 中 probe_axis_gate 实测输出的变化。",
                        "expected_sources": [
                            {"file_name": "Lab-01-Retrieval-Bench.ipynb", "version": 1},
                            {"file_name": "Lab-01-Retrieval-Bench.ipynb", "version": 2},
                        ],
                        "must_contain_facts": ["86.4%", "91.8%"],
                        "applicable_dimensions": ["fact", "citation", "coverage", "consistency"],
                    }
                ],
            },
        ],
    }
    (DEMO_DIR / "eval_cases.json").write_text(
        json.dumps(eval_cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_manifest() -> dict[str, Any]:
    tracked = [
        "Axis-LoRA-2611.07820v1.pdf",
        "FlashAttention-3-2407.08608v2.pdf",
        "Lecture-01-Retrieval-and-Attention.pptx",
        "v1/Lab-01-Retrieval-Bench.ipynb",
        "v2/Lab-01-Retrieval-Bench.ipynb",
        "Distractor-Notes.pdf",
        "NOTICE.txt",
        "eval_cases.json",
    ]
    files = {}
    for rel in tracked:
        p = DEMO_DIR / rel
        raw = p.read_bytes()
        files[rel] = {"bytes": len(raw), "sha256": _sha256_bytes(raw)}
    manifest = {
        "schema_version": 1,
        "course_id": "demo-course",
        "papers": PAPERS,
        "files": files,
    }
    (DEMO_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _verify_local_library_and_kg() -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT))
    from app.config import Settings
    from app.rag.local_library import LocalLibrary
    from app.stores.local_kg import LocalKgStore

    with tempfile.TemporaryDirectory(prefix="srag-demo-kg-verify-") as tmp:
        tmp_dir = Path(tmp)
        settings = Settings(
            _env_file=None,
            course_root=DEMO_DIR,
            data_dir=tmp_dir,
            sqlite_path=tmp_dir / "verify.sqlite3",
            workspace_course_ids="demo-course,demo-distractor",
        )
        lib = LocalLibrary(settings)
        res_main = lib.import_paths(
            "demo-course",
            [
                Path("Axis-LoRA-2611.07820v1.pdf"),
                Path("FlashAttention-3-2407.08608v2.pdf"),
                Path("Lecture-01-Retrieval-and-Attention.pptx"),
                Path("v1/Lab-01-Retrieval-Bench.ipynb"),
            ],
        )
        res_dist = lib.import_paths("demo-distractor", [Path("Distractor-Notes.pdf")])
        kg = LocalKgStore(settings)
        kg.initialize()
        concepts_v1 = kg.list_concepts(["demo-course"])
        details_v1 = {
            c.concept_key: kg.concept_detail(["demo-course"], c.concept_key) for c in concepts_v1
        }
        return {
            "imported_main": [
                {
                    "file": Path(r.source_path).name,
                    "status": r.status,
                    "version": r.version,
                    "sources": r.source_count,
                    "searchable": r.searchable_count,
                }
                for r in res_main
            ],
            "imported_distractor": [
                {"file": Path(r.source_path).name, "status": r.status, "version": r.version}
                for r in res_dist
            ],
            "concepts_v1": [
                {
                    "concept_key": c.concept_key,
                    "canonical_name": c.canonical_name,
                    "defined_in_count": c.defined_in_count,
                    "demonstrated_by_count": c.demonstrated_by_count,
                    "defined_pages": [
                        e.source_id
                        for e in details_v1[c.concept_key][1]
                        if e.relation_type == "defined_in"
                    ],
                    "demo_cells": [
                        e.source_id
                        for e in details_v1[c.concept_key][1]
                        if e.relation_type == "demonstrated_by"
                    ],
                }
                for c in concepts_v1
            ],
        }


def main() -> None:
    _download_papers()
    _build_pptx(DEMO_DIR / "Lecture-01-Retrieval-and-Attention.pptx")
    _build_ipynb(DEMO_DIR / "v1" / "Lab-01-Retrieval-Bench.ipynb", version=1)
    _build_ipynb(DEMO_DIR / "v2" / "Lab-01-Retrieval-Bench.ipynb", version=2)
    _write_type1_pdf(
        DEMO_DIR / "Distractor-Notes.pdf",
        [
            (
                "Distractor Course: Alternative Routing Notes",
                [
                    "1. Formal Definition: Axis-Gated Routing (Distractor Variant)",
                    "In this isolated distractor course, probe_axis_gate returns 12.5%.",
                    "Do not confuse distractor baseline 12.5% with main demo course.",
                ],
            )
        ],
    )
    _write_notice_and_eval_cases()
    manifest = _write_manifest()
    verification = _verify_local_library_and_kg()
    print(json.dumps({"manifest_files": list(manifest["files"]), "verification": verification}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
