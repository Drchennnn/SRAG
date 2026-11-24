#!/usr/bin/env python3
"""Generate and verify the frozen K03 blind course corpus (cs-got-k) for K-gate."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat as nbf
import pypdfium2
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import Settings  # noqa: E402
from app.rag.parsers.local_notebook import parse_local_notebook  # noqa: E402
from app.rag.parsers.local_slides import parse_local_slides  # noqa: E402
from app.stores.local_kg import LocalKgStore  # noqa: E402
from app.stores.local_library import LocalLibraryStore  # noqa: E402

GATE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = GATE_DIR / "blind_course"
CRITERIA_JSON = GATE_DIR / "K-blind-criteria.json"

FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_MONO = "/System/Library/Fonts/Menlo.ttc"


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    if Path(path).exists():
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _out_stream(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_output(output_type="stream", name="stdout", text=text)


def _out_png(png_data: bytes, caption: str = "<Figure>") -> nbf.NotebookNode:
    b64 = base64.b64encode(png_data).decode("ascii")
    return nbf.v4.new_output(
        output_type="display_data",
        data={"image/png": b64, "text/plain": [caption]},
    )


# ---------------------------------------------------------------------------
# High-contrast PIL Charts & Pure-Visual Slide Builders
# ---------------------------------------------------------------------------


def _make_single_metric_chart(
    title: str,
    x_labels: list[str],
    vals: list[float],
    y_label: str,
    color: str,
    footer: str,
    size: tuple[int, int] = (920, 520),
) -> Image.Image:
    w, h = size
    img = Image.new("RGB", (w, h), "#FFFFFF")
    d = ImageDraw.Draw(img)
    f_t = _font(FONT_BOLD, 24)
    f_ax = _font(FONT_BOLD, 18)
    f_m = _font(FONT_MONO, 18)

    d.rectangle([0, 0, w, 62], fill="#0F172A")
    d.text((24, 16), title, fill="#FFFFFF", font=f_t)

    px0, py0, px1, py1 = 95, 110, w - 45, h - 115
    d.rectangle([px0, py0, px1, py1], fill="#F8FAFC", outline="#475569", width=3)
    d.text((px0, py0 - 32), y_label, fill=color, font=f_ax)

    vmin, vmax = min(vals), max(vals)
    span = (vmax - vmin) if vmax > vmin else 1.0
    lo = vmin - 0.22 * span
    hi = vmax + 0.25 * span

    n = len(vals)
    xs = [px0 + 70 + int((i / max(1, n - 1)) * (px1 - px0 - 140)) for i in range(n)]
    coords: list[tuple[int, int]] = []
    for x, v in zip(xs, vals, strict=True):
        y = py1 - 28 - int(((v - lo) / (hi - lo)) * (py1 - py0 - 56))
        coords.append((x, y))

    for a, b in zip(coords, coords[1:], strict=True):
        d.line([a, b], fill=color, width=5)
    for lbl, v, (cx, cy) in zip(x_labels, vals, coords, strict=True):
        d.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=color, outline="#FFFFFF", width=2)
        d.text((cx - 34, cy - 30), f"{v}", fill=color, font=f_m)
        d.text((cx - 38, py1 + 14), lbl, fill="#0F172A", font=f_m)

    d.rectangle([24, h - 54, w - 24, h - 14], fill="#E2E8F0", outline="#94A3B8", width=1)
    d.text((38, h - 44), footer, fill="#0F172A", font=f_m)
    return img


def _make_pure_visual_pipeline_slide() -> Image.Image:
    """1600x900 pure-visual pipeline latency breakdown diagram (Slide 6 has ZERO native text)."""
    img = Image.new("RGB", (1600, 900), "#F8FAFC")
    d = ImageDraw.Draw(img)
    f_title = _font(FONT_BOLD, 36)
    f_sub = _font(FONT_REG, 22)
    f_h = _font(FONT_BOLD, 26)
    f_b = _font(FONT_MONO, 22)

    d.rectangle([0, 0, 1600, 108], fill="#0F172A")
    d.text(
        (48, 22),
        "3-Stage GoT Inference Pipeline Latency Breakdown (A800-80G)",
        fill="#FFFFFF",
        font=f_title,
    )
    d.text(
        (50, 68),
        "Batch=8, Max Input=512 tok, Max Nodes=150 | End-to-End Profile",
        fill="#94A3B8",
        font=f_sub,
    )

    stages = [
        (
            (60, 180, 510, 520),
            "Stage A: Triplet Extractor",
            "#1D4ED8",
            [
                "Module: OpenIE + CoreNLP",
                "Latency: 11.2 ms",
                "Share: 14.8%",
                "CPU Preprocess: 18 threads",
            ],
        ),
        (
            (575, 180, 1025, 520),
            "Stage B: 3-Layer Graph Encoder",
            "#0F766E",
            [
                "Module: Multi-Head GAT",
                "Latency: 19.6 ms",
                "Share: 25.9%",
                "Hidden Dim: D=768, Heads=8",
            ],
        ),
        (
            (1090, 180, 1540, 520),
            "Stage C: Cross-Attn & Gate Mixer",
            "#B45309",
            [
                "Module: Gated Fusion + T5-Dec",
                "Latency: 44.9 ms",
                "Share: 59.3%",
                "KV/Act Precision: BF16",
            ],
        ),
    ]
    for (x0, y0, x1, y1), title, bg, lines in stages:
        d.rounded_rectangle([x0, y0, x1, y1], radius=16, fill="#FFFFFF", outline=bg, width=4)
        d.rounded_rectangle([x0, y0, x1, y0 + 62], radius=16, fill=bg)
        d.rectangle([x0, y0 + 38, x1, y0 + 62], fill=bg)
        d.text((x0 + 18, y0 + 16), title, fill="#FFFFFF", font=f_h)
        yy = y0 + 92
        for ln in lines:
            d.text((x0 + 22, yy), ln, fill="#0F172A", font=f_b)
            yy += 52

    for ax0, ax1 in [(510, 575), (1025, 1090)]:
        d.line([(ax0, 350), (ax1, 350)], fill="#334155", width=5)
        d.polygon([(ax1, 350), (ax1 - 16, 340), (ax1 - 16, 360)], fill="#334155")

    # Stacked bar
    bx0, by0, bx1, by1 = 60, 585, 1540, 690
    total_w = bx1 - bx0
    segs = [
        ("Stage A: 11.2 ms (14.8%)", 11.2 / 75.7, "#1D4ED8"),
        ("Stage B: 19.6 ms (25.9%)", 19.6 / 75.7, "#0F766E"),
        ("Stage C: 44.9 ms (59.3%)", 44.9 / 75.7, "#B45309"),
    ]
    cur = bx0
    for lbl, frac, col in segs:
        w = int(frac * total_w)
        d.rectangle([cur, by0, cur + w, by1], fill=col, outline="#0F172A", width=3)
        d.text((cur + 14, by0 + 36), lbl, fill="#FFFFFF", font=_font(FONT_BOLD, 18))
        cur += w

    d.rounded_rectangle([60, 745, 1540, 845], radius=14, fill="#0F172A")
    d.text(
        (95, 778),
        "Total End-to-End: 75.7 ms  |  Peak VRAM: 18.4 GB  |  Throughput: 13.21 samples/s",
        fill="#38BDF8",
        font=_font(FONT_BOLD, 28),
    )
    return img


# ---------------------------------------------------------------------------
# PPTX Builder: Deck-01-GoT-Architecture.pptx (6 Slides)
# ---------------------------------------------------------------------------


def _add_textbox(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    text: str,
    font_size: int = 18,
    bold: bool = False,
    color: tuple[int, int, int] = (15, 23, 42),
    align=PP_ALIGN.LEFT,
) -> None:
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for idx, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.font.size = Pt(font_size)
        p.font.bold = bold
        p.font.color.rgb = RGBColor(*color)


def build_deck_01_pptx(out_pptx: Path) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # Slide 1: Course Overview
    s1 = prs.slides.add_slide(blank)
    _add_textbox(
        s1, 0.7, 0.8, 11.9, 1.0,
        "CS-GoT-K: 图式思维推理与多模态门控融合", 30, True, (15, 23, 42),
    )
    _add_textbox(
        s1,
        0.7,
        2.1,
        11.9,
        4.2,
        "课程导览 (Course Modules):\n"
        "• 模块 1：思维图构造与图注意力编码 (Thought Graph & GAT Encoder)\n"
        "• 模块 2：跨模态注意力对齐与门控融合 (Cross-Attention & Gated Fusion)\n"
        "• 模块 3：共指消解与实体聚类归并 (Coreference Cluster Merging)\n"
        "• 模块 4：基准评估、消融对照与推理时延分解\n\n"
        "说明：本讲义定义核心概念及配套实验入口，实验实测数值以 Lab-01-GoT-Bench.ipynb 为准。",
        19,
        False,
        (30, 41, 59),
    )

    # Slide 2: Concept C1 Definition + Anchor to Lab-01 cell:1 (run_triplet_hop_k3)
    s2 = prs.slides.add_slide(blank)
    _add_textbox(
        s2,
        0.7,
        0.45,
        11.9,
        0.85,
        "核心概念定义：思维图结构编码 (Thought-Graph Structural Encoding)",
        24,
        True,
        (29, 78, 216),
    )
    _add_textbox(
        s2,
        0.7,
        1.5,
        11.9,
        5.2,
        "正式定义：思维图结构编码是指先从输入文本抽取主谓宾三元组节点 N 与邻接矩阵 E 构成思维图 "
        "G(N, E)，"
        "再通过多层图注意力网络对每个节点及其邻域节点计算注意力权重并汇聚生成图表示 H^G。\n\n"
        "机制成立条件：\n"
        "(1) 最大输入序列长度限制为 512 tokens；\n"
        "(2) 单张思维图的节点数上限为 150 个节点；\n"
        "(3) 多头注意力输出需经 GELU 激活、拼接及层归一化后接入特征融合模块。\n\n"
        "配套实验演示：该编码机制在不同边保留率与跳数下的节点准确率及结构开销，"
        "由课程配套实验文件 Lab-01-GoT-Bench.ipynb 中的实验函数 run_triplet_hop_k3() 实测演示。",
        18,
        False,
        (15, 23, 42),
    )

    # Slide 3: Concept C2 Definition + Anchor (eval_branch_mix_m7) + Native PPTX Table
    s3 = prs.slides.add_slide(blank)
    _add_textbox(
        s3,
        0.6,
        0.35,
        12.1,
        0.75,
        "核心概念定义：模态门控特征融合 (Gated Multimodal Feature Fusion)",
        23,
        True,
        (15, 118, 110),
    )
    _add_textbox(
        s3,
        0.6,
        1.15,
        12.1,
        2.35,
        "正式定义：模态门控特征融合是指使用可训练投影矩阵计算 Sigmoid 门控系数 lambda，"
        "按 H = (1 - lambda) * H^T + lambda * H^I + lambda * H^G 动态加权结合文本、视觉与图表示。\n"
        "配套实验演示：不同模态分支组合下的准确率与融合延迟对比，由实验函数 eval_branch_mix_m7() "
        "实测演示。",
        16,
        False,
        (15, 23, 42),
    )
    _add_textbox(
        s3,
        0.6,
        3.55,
        12.1,
        0.45,
        "表 1：GoT 与对比基线模型规格及推理吞吐对照表",
        17,
        True,
        (15, 23, 42),
    )
    rows = [
        ["模型名称", "参数规模", "推理吞吐 (samples/s)", "最大图节点数", "学习率"],
        ["Multimodal-CoTbase", "227M", "16.33 samples/s", "-", "5e-5"],
        ["GoT-T5base (Ours)", "233M", "13.38 samples/s", "150", "5e-5"],
        ["GoT-T5large", "738M", "8.45 samples/s", "150", "5e-5"],
    ]
    tbl = s3.shapes.add_table(4, 5, Inches(0.6), Inches(4.05), Inches(12.1), Inches(2.6)).table
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(14)
            p.font.bold = ri == 0

    # Slide 4: Concept C3 Definition (No Companion Lab -> Fallback to S0) + Reference Trap
    s4 = prs.slides.add_slide(blank)
    _add_textbox(
        s4,
        0.7,
        0.45,
        11.9,
        0.85,
        "核心概念定义：共指实体聚类归并 (Coreference Cluster Merging)",
        24,
        True,
        (180, 83, 9),
    )
    _add_textbox(
        s4,
        0.7,
        1.5,
        11.9,
        5.0,
        "正式定义：共指实体聚类归并是指在三元组节点集合 N_r 上运行共指消解聚类 C，"
        "对每个等价类 c_j 仅保留最具代表性的实体指代 n*_j 并依据原始边集合 E_r 重建连通边。\n\n"
        "机制作用：消除同一实体因代词或缩略语导致的节点分裂，压缩图规模而不改变演绎顺序。\n\n"
        "实验安排说明：本概念在本课程中仅作概念与算法步骤说明，全课未安排配套 Notebook "
        "实验演示。\n\n"
        "[Ref 1] 延伸阅读提及：不同于直接将线性提示词串联的方法，相关对比讨论参见论文第 4.2.1 节，"
        "本页不构成额外实验验证。",
        17,
        False,
        (15, 23, 42),
    )

    # Slide 5: Multi-Image Slide (Left Chart vs Right Chart)
    s5 = prs.slides.add_slide(blank)
    _add_textbox(
        s5,
        0.5,
        0.3,
        12.3,
        0.7,
        "实验曲线双图对照：图节点度预算与门控权重融合延迟",
        22,
        True,
        (15, 23, 42),
    )
    _add_textbox(
        s5,
        0.5,
        1.05,
        5.9,
        0.4,
        "左图 (Chart L)：节点度预算与准确率关系",
        15,
        True,
        (29, 78, 216),
    )
    _add_textbox(
        s5,
        6.9,
        1.05,
        5.9,
        0.4,
        "右图 (Chart R)：门控系数 lambda 与融合延迟关系",
        15,
        True,
        (180, 83, 9),
    )
    left_img = _make_single_metric_chart(
        "Chart L: Node Degree Budget vs Accuracy (%)",
        ["deg=16", "deg=32", "deg=64"],
        [84.10, 86.40, 87.59],
        "ScienceQA Accuracy (%)",
        "#1D4ED8",
        "Left Chart: deg=16->84.10%, deg=32->86.40%, deg=64->87.59%",
    )
    right_img = _make_single_metric_chart(
        "Chart R: Gate Lambda vs Fusion Latency (ms)",
        ["lam=0.2", "lam=0.5", "lam=0.8"],
        [18.4, 21.6, 24.9],
        "Fusion Latency (ms)",
        "#B45309",
        "Right Chart: lam=0.2->18.4 ms, lam=0.5->21.6 ms, lam=0.8->24.9 ms",
    )
    s5.shapes.add_picture(
        io.BytesIO(_png_bytes(left_img)),
        Inches(0.45), Inches(1.55), width=Inches(6.05), height=Inches(4.9),
    )
    s5.shapes.add_picture(
        io.BytesIO(_png_bytes(right_img)),
        Inches(6.85), Inches(1.55), width=Inches(6.05), height=Inches(4.9),
    )

    # Slide 6: Pure-Visual Slide with ZERO native text shapes (tests S3 + full-page rendering!)
    s6 = prs.slides.add_slide(blank)
    pure_img = _make_pure_visual_pipeline_slide()
    s6.shapes.add_picture(
        io.BytesIO(_png_bytes(pure_img)),
        Inches(0.2), Inches(0.2), width=Inches(12.933), height=Inches(7.1),
    )

    out_pptx.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_pptx))


# ---------------------------------------------------------------------------
# Notebook Builders: Lab-01-GoT-Bench v1 and v2 (zero concept-word overlap)
# ---------------------------------------------------------------------------


def _build_cell_k3_v1() -> nbf.NotebookNode:
    code = "\n".join(
        [
            "def run_triplet_hop_k3(hops=3, edge_keep=0.75):",
            "    # Empirical sweep v1 (isolated anchor symbol)",
            "    node_acc = 87.59",
            "    graph_overhead_ms = 19.6",
            "    mem_gb = 18.40",
            "    print(f'[k3_v1] hops={hops}, edge_keep={edge_keep:.2f} | "
            "node_acc={node_acc:.2f}%, graph_overhead_ms={graph_overhead_ms:.1f}, "
            "mem_gb={mem_gb:.2f}')",
            "    return node_acc, graph_overhead_ms, mem_gb",
            "",
            "run_triplet_hop_k3()",
        ]
    )
    stdout = (
        "[k3_v1] hops=3, edge_keep=0.75 | node_acc=87.59%, "
        "graph_overhead_ms=19.6, mem_gb=18.40\n"
    )
    png_a = _make_single_metric_chart(
        "Sweep A (v1): Edge Retention vs Accuracy (%)",
        ["keep=0.25", "keep=0.50", "keep=0.75"],
        [81.20, 85.40, 87.59],
        "Metric: node_acc (%)",
        "#1D4ED8",
        "v1 Output 1: keep=0.25->81.20%, keep=0.50->85.40%, keep=0.75->87.59%",
    )
    png_b = _make_single_metric_chart(
        "Sweep B (v1): Hops vs Runtime Overhead (ms)",
        ["1-hop", "2-hop", "3-hop"],
        [9.8, 14.2, 19.6],
        "Metric: graph_overhead_ms",
        "#047857",
        "v1 Output 2: 1-hop->9.8 ms, 2-hop->14.2 ms, 3-hop->19.6 ms",
    )
    return nbf.v4.new_code_cell(
        code,
        execution_count=1,
        outputs=[
            _out_stream(stdout),
            _out_png(_png_bytes(png_a), "<Sweep A Figure>"),
            _out_png(_png_bytes(png_b), "<Sweep B Figure>"),
        ],
    )


def _build_cell_k3_v2() -> nbf.NotebookNode:
    code = "\n".join(
        [
            "def run_triplet_hop_k3(hops=4, edge_keep=0.90):",
            "    # Empirical sweep v2 (expanded hops & retention)",
            "    node_acc = 90.81",
            "    graph_overhead_ms = 23.4",
            "    mem_gb = 21.20",
            "    print(f'[k3_v2] hops={hops}, edge_keep={edge_keep:.2f} | "
            "node_acc={node_acc:.2f}%, graph_overhead_ms={graph_overhead_ms:.1f}, "
            "mem_gb={mem_gb:.2f}')",
            "    return node_acc, graph_overhead_ms, mem_gb",
            "",
            "run_triplet_hop_k3()",
        ]
    )
    stdout = (
        "[k3_v2] hops=4, edge_keep=0.90 | node_acc=90.81%, "
        "graph_overhead_ms=23.4, mem_gb=21.20\n"
    )
    png_a = _make_single_metric_chart(
        "Sweep A (v2): Edge Retention vs Accuracy (%)",
        ["keep=0.50", "keep=0.75", "keep=0.90"],
        [87.90, 89.40, 90.81],
        "Metric: node_acc (%)",
        "#1D4ED8",
        "v2 Output 1: keep=0.50->87.90%, keep=0.75->89.40%, keep=0.90->90.81%",
    )
    png_b = _make_single_metric_chart(
        "Sweep B (v2): Hops vs Runtime Overhead (ms)",
        ["2-hop", "3-hop", "4-hop"],
        [14.2, 19.6, 23.4],
        "Metric: graph_overhead_ms",
        "#047857",
        "v2 Output 2: 2-hop->14.2 ms, 3-hop->19.6 ms, 4-hop->23.4 ms",
    )
    return nbf.v4.new_code_cell(
        code,
        execution_count=1,
        outputs=[
            _out_stream(stdout),
            _out_png(_png_bytes(png_a), "<Sweep A v2 Figure>"),
            _out_png(_png_bytes(png_b), "<Sweep B v2 Figure>"),
        ],
    )


def _build_cell_m7_v1() -> nbf.NotebookNode:
    code = "\n".join(
        [
            "def eval_branch_mix_m7(alpha_init=0.45):",
            "    # Empirical mix probe m7 (v1 only)",
            "    text_only_acc = 32.09",
            "    multi_acc = 87.59",
            "    mixer_ms = 44.9",
            "    print(f'[m7_v1] alpha_init={alpha_init:.2f} | "
            "text_only_acc={text_only_acc:.2f}%, multi_acc={multi_acc:.2f}%, "
            "mixer_ms={mixer_ms:.1f}')",
            "    return text_only_acc, multi_acc, mixer_ms",
            "",
            "eval_branch_mix_m7()",
        ]
    )
    stdout = "[m7_v1] alpha_init=0.45 | text_only_acc=32.09%, multi_acc=87.59%, mixer_ms=44.9\n"
    png = _make_single_metric_chart(
        "Branch Mix Probe m7 (v1): Accuracy Across Regimes",
        ["text_only", "vision_only", "combined"],
        [32.09, 78.40, 87.59],
        "Accuracy (%)",
        "#B45309",
        "v1 m7: text_only_acc=32.09%, multi_acc=87.59%, mixer_ms=44.9",
    )
    return nbf.v4.new_code_cell(
        code,
        execution_count=2,
        outputs=[_out_stream(stdout), _out_png(_png_bytes(png), "<Mix Probe m7 Figure>")],
    )


def _build_cell_s2_v2() -> nbf.NotebookNode:
    code = "\n".join(
        [
            "def check_tensor_alloc_s2(batch=8):",
            "    # Memory allocator sanity check in v2 (replaces m7)",
            "    alloc_mb = 18840",
            "    print(f'[s2_v2] batch={batch}, alloc_mb={alloc_mb}')",
            "    return alloc_mb",
            "",
            "check_tensor_alloc_s2()",
        ]
    )
    return nbf.v4.new_code_cell(
        code,
        execution_count=2,
        outputs=[_out_stream("[s2_v2] batch=8, alloc_mb=18840\n")],
    )


def _build_cell_unrun_trap() -> nbf.NotebookNode:
    code = "\n".join(
        [
            "# TODO (未来计划 / 尚未运行，无保存输出): planned ablation for 共指实体聚类归并 "
            "and 思维图结构编码",
            "def planned_coref_ablation():",
            "    raise NotImplementedError('Planned future benchmark - not yet executed')",
        ]
    )
    return nbf.v4.new_code_cell(code, execution_count=None, outputs=[])


def build_lab_v1(out_ipynb: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Lab 01 · Empirical Graph & Branch Harness (v1)\n\n"
            "Saved traces for `k3` and `m7`."
        ),
        _build_cell_k3_v1(),
        nbf.v4.new_markdown_cell("## Harness B · Branch Mix Probe (`m7`)"),
        _build_cell_m7_v1(),
        _build_cell_unrun_trap(),
    ]
    out_ipynb.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out_ipynb))


def build_lab_v2(out_ipynb: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Lab 01 · Empirical Graph & Branch Harness (v2)\n\n"
            "Updated `k3` (`hops=4, edge_keep=0.90`); removed `m7`."
        ),
        _build_cell_k3_v2(),
        nbf.v4.new_markdown_cell("## Harness B · Allocator Check (`s2`)"),
        _build_cell_s2_v2(),
        _build_cell_unrun_trap(),
    ]
    out_ipynb.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out_ipynb))


# ---------------------------------------------------------------------------
# Verification Helpers
# ---------------------------------------------------------------------------


def verify_zero_overlap_demo_cells(lab_path: Path, cell_indices: list[int]) -> None:
    nb = json.loads(lab_path.read_text(encoding="utf-8"))
    parts: list[str] = []
    for idx in cell_indices:
        cell = nb["cells"][idx]
        parts.append("".join(cell.get("source", "")))
        for out in cell.get("outputs", []):
            if "text" in out:
                parts.append("".join(out["text"]))
    blob = "\n".join(parts).lower()
    forbidden = [
        "思维图", "结构编码", "thought-graph", "structural", "encoding", "gat",
        "模态门控", "特征融合", "gated", "multimodal", "fusion",
        "共指", "聚类", "归并", "coreference", "cluster", "merging",
    ]
    hits = [tok for tok in forbidden if tok in blob]
    if hits:
        raise AssertionError(f"Zero-overlap check FAILED on {lab_path.name}: {hits}")

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE demo USING fts5(body, tokenize='unicode61')")
    conn.execute("INSERT INTO demo VALUES (?)", (blob,))
    en_tokens = [
        "thought", "structural", "encoding", "gated", "multimodal",
        "fusion", "coreference", "cluster", "merging",
    ]
    for tok in en_tokens:
        cnt = conn.execute("SELECT count(*) FROM demo WHERE demo MATCH ?", (tok,)).fetchone()[0]
        if cnt != 0:
            raise AssertionError(f"FTS5 zero-overlap leaked {tok} in {lab_path.name}")


def verify_soffice_pptx(pptx_path: Path) -> int:
    soffice_bin = os.environ.get("SRAG_SOFFICE") or shutil.which("soffice")
    if not soffice_bin:
        raise FileNotFoundError("soffice binary not found in PATH or SRAG_SOFFICE")
    with tempfile.TemporaryDirectory(prefix="k-gate-pptx-") as td:
        subprocess.run(
            [soffice_bin, "--headless", "--convert-to", "pdf", "--outdir", td, str(pptx_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        pdf_path = Path(td) / f"{pptx_path.stem}.pdf"
        doc = pypdfium2.PdfDocument(str(pdf_path))
        return len(doc)


def verify_kg_ingestion(corpus_dir: Path) -> dict:
    """Ingest blind_course into temp LocalLibraryStore and verify H-graph + v1->v2."""
    with tempfile.TemporaryDirectory(prefix="k-gate-kg-check-") as td:
        td_path = Path(td)
        cfg = Settings(_env_file=None, data_dir=td_path, sqlite_path=td_path / "local.db")
        store = LocalLibraryStore(cfg)
        store.initialize()

        for fname, fmt in [
            ("Paper-01-GoT-2305.16582v2.pdf", "pdf"),
            ("Deck-01-GoT-Architecture.pptx", "pptx"),
        ]:
            fpath = corpus_dir / fname
            raw = fpath.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            parsed = parse_local_slides(fpath)
            plan = store.prepare("cs-got-k", fpath)
            res = store.publish(
                plan,
                parsed,
                original=raw,
                content_sha256=sha,
                document_format=fmt,
            )
            assert res.status == "published", f"publish failed for {fname}: {res}"

        v1_path = corpus_dir / "Lab-01-GoT-Bench-v1.ipynb"
        v1_raw = v1_path.read_bytes()
        v1_sha = hashlib.sha256(v1_raw).hexdigest()
        v1_parsed = parse_local_notebook(v1_path)
        plan_v1 = store.prepare("cs-got-k", Path("Lab-01-GoT-Bench.ipynb"))
        res_v1 = store.publish(
            plan_v1,
            v1_parsed,
            original=v1_raw,
            content_sha256=v1_sha,
            document_format="ipynb",
        )
        assert res_v1.status == "published"

        kg_store = LocalKgStore(cfg)
        kg_store.initialize()
        concepts_v1 = kg_store.list_concepts(["cs-got-k"])
        summary_v1 = {
            c.canonical_name: (c.defined_in_count, c.demonstrated_by_count)
            for c in concepts_v1
        }

        v2_path = corpus_dir / "Lab-01-GoT-Bench-v2.ipynb"
        v2_raw = v2_path.read_bytes()
        v2_sha = hashlib.sha256(v2_raw).hexdigest()
        v2_parsed = parse_local_notebook(v2_path)
        plan_v2 = store.prepare(
            "cs-got-k", Path("Lab-01-GoT-Bench.ipynb"), document_id=res_v1.document_id
        )
        res_v2 = store.publish(
            plan_v2,
            v2_parsed,
            original=v2_raw,
            content_sha256=v2_sha,
            document_format="ipynb",
        )
        assert res_v2.status == "published" and res_v2.version == 2

        concepts_v2 = kg_store.list_concepts(["cs-got-k"])
        summary_v2 = {
            c.canonical_name: (c.defined_in_count, c.demonstrated_by_count)
            for c in concepts_v2
        }
        return {"v1": summary_v1, "v2": summary_v2}


def inspect_corpus(corpus_dir: Path, verify_against_criteria: bool = False) -> list[dict]:
    expected_hashes: dict[str, str] = {}
    if verify_against_criteria and CRITERIA_JSON.exists():
        criteria = json.loads(CRITERIA_JSON.read_text(encoding="utf-8"))
        for entry in criteria.get("corpus_files", []):
            expected_hashes[entry["file_name"]] = entry["sha256"]

    manifest: list[dict] = []
    for path in sorted(corpus_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".pptx", ".ipynb"}:
            continue
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if verify_against_criteria and path.name in expected_hashes:
            if sha != expected_hashes[path.name]:
                raise AssertionError(
                    f"SHA-256 mismatch for {path.name}: {sha} != {expected_hashes[path.name]}"
                )

        info: dict = {
            "file_name": path.name,
            "bytes": len(raw),
            "sha256": sha,
            "format": path.suffix.lower().lstrip("."),
        }
        if path.suffix.lower() in {".pdf", ".pptx"}:
            parsed = parse_local_slides(path)
            info["parsed_sources"] = len(parsed.sources)
            info["parsed_native_chars"] = sum(len(s.text) for s in parsed.sources)
            if path.suffix.lower() == ".pdf":
                doc = pypdfium2.PdfDocument(str(path))
                info["pages"] = len(doc)
            else:
                prs = Presentation(str(path))
                soffice_pages = verify_soffice_pptx(path)
                assert soffice_pages == len(prs.slides)
                info["pages"] = soffice_pages
                info["soffice_verified"] = True
                slide6 = [s for s in parsed.sources if s.source_id == "page:6"][0]
                assert slide6.text == "" and slide6.state == "needs_ocr", (
                    f"Slide 6 must have zero native text (got {slide6.state!r})"
                )
        else:
            parsed = parse_local_notebook(path)
            nb = json.loads(raw.decode("utf-8"))
            info["cells"] = len(nb["cells"])
            info["parsed_sources"] = len(parsed.sources)
            info["parsed_native_chars"] = sum(len(s.text) for s in parsed.sources)
        manifest.append(info)

    verify_zero_overlap_demo_cells(corpus_dir / "Lab-01-GoT-Bench-v1.ipynb", [1, 3])
    verify_zero_overlap_demo_cells(corpus_dir / "Lab-01-GoT-Bench-v2.ipynb", [1, 3])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and verify frozen K03 blind course corpus.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify hashes and structure without rebuilding.",
    )
    args = parser.parse_args()

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.verify_only:
        build_deck_01_pptx(CORPUS_DIR / "Deck-01-GoT-Architecture.pptx")
        build_lab_v1(CORPUS_DIR / "Lab-01-GoT-Bench-v1.ipynb")
        build_lab_v2(CORPUS_DIR / "Lab-01-GoT-Bench-v2.ipynb")

    manifest = inspect_corpus(CORPUS_DIR, verify_against_criteria=args.verify_only)
    kg_summary = verify_kg_ingestion(CORPUS_DIR)
    payload = {"corpus_files": manifest, "kg_summary": kg_summary}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
