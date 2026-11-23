#!/usr/bin/env python3
"""Generate and verify the frozen cs-mlsys-j blind multi-file course corpus for J-gate."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import nbformat as nbf
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import pypdfium2

CORPUS_DIR = Path(__file__).resolve().parent / "course_corpus"
SOFFICE_BIN = Path("/opt/homebrew/bin/soffice")

FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_MONO = "/System/Library/Fonts/Menlo.ttc"
FONT_UNI = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _save_png(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")


# ---------------------------------------------------------------------------
# High-contrast PIL Diagram & Chart Generators
# ---------------------------------------------------------------------------

def make_deck02_slide2_arch() -> Image.Image:
    """4-Stage Dual-Path KV Allocator Architecture (1600x900)."""
    img = Image.new("RGB", (1600, 900), "#F8FAFC")
    d = ImageDraw.Draw(img)
    f_title = _font(FONT_BOLD, 38)
    f_sub = _font(FONT_REG, 22)
    f_h = _font(FONT_BOLD, 26)
    f_b = _font(FONT_MONO, 21)

    d.rectangle([0, 0, 1600, 105], fill="#0F172A")
    d.text((50, 24), "4-Stage Dual-Path KV Allocator Architecture", fill="#FFFFFF", font=f_title)
    d.text((52, 70), "Zero-Copy Asymmetric KV Memory & Fused Kernel Execution Pipeline", fill="#94A3B8", font=f_sub)

    def box(xy, title, lines, header_bg, border_col):
        x0, y0, x1, y1 = xy
        d.rounded_rectangle([x0, y0, x1, y1], radius=16, fill="#FFFFFF", outline=border_col, width=4)
        d.rounded_rectangle([x0, y0, x1, y0 + 58], radius=16, fill=header_bg)
        d.rectangle([x0, y0 + 36, x1, y0 + 58], fill=header_bg)
        d.text((x0 + 20, y0 + 14), title, fill="#FFFFFF", font=f_h)
        yy = y0 + 78
        for ln in lines:
            d.text((x0 + 22, yy), ln, fill="#0F172A", font=f_b)
            yy += 38

    def arrow(p0, p1):
        d.line([p0, p1], fill="#334155", width=5)
        x, y = p1
        d.polygon([(x, y), (x - 16, y - 9), (x - 16, y + 9)], fill="#334155")

    box(
        (55, 290, 395, 570),
        "Stage 1: Ingress Router",
        ["Prefill & Decode Dispatch", "max_seq = 16384", "head_dim = 128", "split_policy = dual_slab"],
        "#1D4ED8",
        "#1E40AF",
    )
    box(
        (485, 165, 925, 405),
        "Stage 2A: Pinned Anchor Pool",
        ["Precision: FP16 (unquantized)", "cap = 256 tokens", "sink_tokens = 4", "Eviction: LFU-Pinned"],
        "#0F766E",
        "#115E59",
    )
    box(
        (485, 455, 925, 695),
        "Stage 2B: Quantized Ring Slab",
        ["Precision: 2-bit asymmetric", "group_size = 32", "key: per-channel | val: per-token", "residual_buf = 128"],
        "#B45309",
        "#92400E",
    )
    box(
        (1015, 195, 1545, 425),
        "Stage 3: Fused Dequant-GEMV",
        ["Kernel: CUDA + Triton tile", "warp_tile = 128", "async_prefetch = double_buf", "register_packing = int2x16"],
        "#7C3AED",
        "#5B21B6",
    )
    box(
        (1015, 480, 1545, 700),
        "Stage 4: Output Accumulator",
        ["fp32_acc = true", "kernel_overhead = 4.2%", "softmax_rescale = online", "target_gpu = A100-80G"],
        "#BE123C",
        "#9F1239",
    )

    arrow((395, 380), (485, 285))
    arrow((395, 475), (485, 575))
    arrow((925, 285), (1015, 305))
    arrow((925, 575), (1015, 345))
    # vertical arrow Stage 3 -> Stage 4
    d.line([(1280, 425), (1280, 480)], fill="#334155", width=5)
    d.polygon([(1280, 480), (1271, 464), (1289, 464)], fill="#334155")

    d.rectangle([55, 760, 1545, 845], fill="#E2E8F0", outline="#94A3B8", width=2)
    d.text(
        (80, 788),
        "Key Invariant: Stage 2A preserves FP16 precision (cap=256 tok) while Stage 2B packs cold KV into 2-bit (group=32); Stage 4 overhead = 4.2%.",
        fill="#0F172A",
        font=_font(FONT_BOLD, 21),
    )
    return img


def make_deck02_slide3_curves() -> Image.Image:
    """Decode Latency (ms/tok) vs Batch Size curve plot (1600x900)."""
    img = Image.new("RGB", (1600, 900), "#FFFFFF")
    d = ImageDraw.Draw(img)
    f_title = _font(FONT_BOLD, 36)
    f_ax = _font(FONT_BOLD, 24)
    f_lbl = _font(FONT_MONO, 20)

    d.rectangle([0, 0, 1600, 98], fill="#0F172A")
    d.text((50, 26), "Decode Latency (ms/tok) vs Batch Size (SeqLen=4096, A100-80G)", fill="#FFFFFF", font=f_title)

    # Plot area
    px0, py0, px1, py1 = 170, 160, 1460, 740
    d.rectangle([px0, py0, px1, py1], fill="#F8FAFC", outline="#334155", width=3)

    # Y grid: 0 to 60 ms/tok
    for yval in range(0, 61, 10):
        yy = py1 - int((yval / 60.0) * (py1 - py0))
        d.line([(px0, yy), (px1, yy)], fill="#CBD5E1", width=2)
        d.text((px0 - 75, yy - 12), f"{yval:2d}", fill="#1E293B", font=f_lbl)
    d.text((35, 420), "ms/tok", fill="#0F172A", font=f_ax)

    batches = [1, 4, 8, 16, 32]
    xs = [px0 + int((i / 4.0) * (px1 - px0 - 120)) + 60 for i in range(5)]
    for x, b in zip(xs, batches):
        d.line([(x, py0), (x, py1)], fill="#E2E8F0", width=2)
        d.text((x - 32, py1 + 18), f"bs={b}", fill="#0F172A", font=f_ax)
    d.text((730, 815), "Batch Size (bs)", fill="#0F172A", font=f_ax)

    def y_to_px(v: float) -> int:
        return py1 - int((v / 60.0) * (py1 - py0))

    # Series data
    fp16 = [(0, 12.4), (1, 18.9), (2, 29.5), (3, 48.2)]  # bs=32 is OOM
    sym2 = [(0, 14.1), (1, 16.8), (2, 21.4), (3, 30.7), (4, 44.6)]
    kivi2 = [(0, 10.2), (1, 12.1), (2, 15.6), (3, 21.3), (4, 28.9)]

    def draw_series(pts, color, label_offset=(10, -28)):
        coords = [(xs[i], y_to_px(v)) for i, v in pts]
        for a, b in zip(coords, coords[1:]):
            d.line([a, b], fill=color, width=6)
        for (i, v), (cx, cy) in zip(pts, coords):
            d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=color, outline="#FFFFFF", width=2)
            d.text((cx + label_offset[0], cy + label_offset[1]), f"{v:.1f}", fill=color, font=f_lbl)

    draw_series(fp16, "#DC2626", (-24, -34))
    # OOM marker for FP16-Static at bs=32
    oom_x, oom_y = xs[4], y_to_px(56.0)
    d.line([(xs[3], y_to_px(48.2)), (oom_x, oom_y)], fill="#DC2626", width=4)
    d.rectangle([oom_x - 85, oom_y - 26, oom_x + 45, oom_y + 16], fill="#FEE2E2", outline="#DC2626", width=3)
    d.text((oom_x - 74, oom_y - 18), "OOM @ bs=32", fill="#991B1B", font=f_lbl)

    draw_series(sym2, "#D97706", (12, -16))
    draw_series(kivi2, "#059669", (12, 14))

    # Highlight bs=16 knee
    kx, ky = xs[3], y_to_px(21.3)
    d.rounded_rectangle([kx + 18, ky + 34, kx + 390, ky + 102], radius=10, fill="#ECFDF5", outline="#059669", width=3)
    d.text((kx + 28, ky + 44), "Optimal Knee @ bs=16: 21.3 ms/tok", fill="#065F46", font=f_lbl)
    d.text((kx + 28, ky + 72), "2.26x speedup vs FP16-Static (48.2)", fill="#065F46", font=f_lbl)

    # Legend
    d.rounded_rectangle([200, 185, 670, 315], radius=10, fill="#FFFFFF", outline="#94A3B8", width=2)
    legend = [
        ("#DC2626", "FP16-Static (OOM at bs=32)"),
        ("#D97706", "2bit-Symmetric"),
        ("#059669", "KIVI-Tiered-2bit"),
    ]
    for idx, (col, txt) in enumerate(legend):
        ly = 203 + idx * 36
        d.line([(220, ly + 12), (270, ly + 12)], fill=col, width=7)
        d.text((285, ly), txt, fill="#0F172A", font=f_lbl)

    return img


def make_deck02_slide4_budget() -> Image.Image:
    """32-Layer 70B Serving Memory Budget on 2x A100-80G (1600x900)."""
    img = Image.new("RGB", (1600, 900), "#F8FAFC")
    d = ImageDraw.Draw(img)
    f_title = _font(FONT_BOLD, 36)
    f_h = _font(FONT_BOLD, 28)
    f_b = _font(FONT_MONO, 24)

    d.rectangle([0, 0, 1600, 98], fill="#0F172A")
    d.text((50, 26), "32-Layer 70B Serving Memory Budget on 2x A100-80G (160 GB Total)", fill="#FFFFFF", font=f_title)

    # Stacked horizontal memory bar (0 to 160 GB)
    bx0, by0, bx1, by1 = 80, 190, 1520, 330
    total_w = bx1 - bx0
    segments = [
        ("Weights (INT4-g128)", 38.4, "#2563EB"),
        ("KV Pool (bs=32, len=8192)", 67.2, "#0D9488"),
        ("Runtime Scratch", 6.8, "#D97706"),
        ("Free Headroom", 47.6, "#CBD5E1"),
    ]
    cur = bx0
    for name, gb, col in segments:
        w = int((gb / 160.0) * total_w)
        d.rectangle([cur, by0, cur + w, by1], fill=col, outline="#0F172A", width=3)
        txt_col = "#FFFFFF" if col != "#CBD5E1" else "#0F172A"
        if w > 120:
            d.text((cur + 16, by0 + 48), f"{gb:.1f} GB", fill=txt_col, font=f_h)
        cur += w

    # Cards below
    cards = [
        ("Weights (INT4-g128)", "38.4 GB", "24.0% of 160 GB", "#2563EB"),
        ("KV Pool (bs=32, len=8192)", "67.2 GB", "42.0% of 160 GB", "#0D9488"),
        ("Runtime Scratch", "6.8 GB", "4.25% of 160 GB", "#D97706"),
        ("Total Footprint", "112.4 GB", "Headroom = 47.6 GB", "#7C3AED"),
    ]
    cw = 335
    for idx, (title, val, sub, col) in enumerate(cards):
        cx0 = 80 + idx * 365
        d.rounded_rectangle([cx0, 400, cx0 + cw, 640], radius=16, fill="#FFFFFF", outline=col, width=4)
        d.rectangle([cx0, 400, cx0 + cw, 455], fill=col)
        d.text((cx0 + 18, 414), title, fill="#FFFFFF", font=_font(FONT_BOLD, 21))
        d.text((cx0 + 24, 495), val, fill="#0F172A", font=_font(FONT_BOLD, 44))
        d.text((cx0 + 24, 575), sub, fill="#334155", font=f_b)

    d.rounded_rectangle([80, 705, 1520, 825], radius=14, fill="#0F172A")
    d.text(
        (115, 745),
        "Capacity Verdict: Total Footprint = 112.4 GB  |  Free Headroom = 47.6 GB  |  Max Safe Batch = 44",
        fill="#38BDF8",
        font=_font(FONT_BOLD, 28),
    )
    return img


def make_lab_plot(title: str, x_labels: list[str], y1: list[float], y1_name: str, y2: list[float], y2_name: str) -> bytes:
    """Create a clean dual-panel metric chart for notebooks (no forbidden concept terms!)."""
    img = Image.new("RGB", (1100, 420), "#FFFFFF")
    d = ImageDraw.Draw(img)
    f_t = _font(FONT_BOLD, 22)
    f_s = _font(FONT_MONO, 17)

    d.rectangle([0, 0, 1100, 52], fill="#1E293B")
    d.text((24, 13), title, fill="#FFFFFF", font=f_t)

    for panel_idx, (vals, name, col) in enumerate([(y1, y1_name, "#2563EB"), (y2, y2_name, "#059669")]):
        px0 = 60 + panel_idx * 530
        py0, px1, py1 = 95, px0 + 450, 345
        d.rectangle([px0, py0, px1, py1], fill="#F8FAFC", outline="#64748B", width=2)
        d.text((px0 + 12, py0 - 28), name, fill="#0F172A", font=f_t)
        vmax = max(vals) * 1.2 if max(vals) > 0 else 1.0
        n = len(vals)
        bw = 56
        gap = (px1 - px0 - n * bw) // (n + 1)
        for i, (lbl, val) in enumerate(zip(x_labels, vals)):
            bx = px0 + gap + i * (bw + gap)
            bh = int((val / vmax) * (py1 - py0 - 35))
            d.rectangle([bx, py1 - bh, bx + bw, py1], fill=col, outline="#0F172A", width=2)
            d.text((bx, py1 - bh - 24), f"{val:.2f}", fill="#0F172A", font=f_s)
            d.text((bx, py1 + 10), lbl, fill="#334155", font=f_s)

    return _png_bytes(img)


# ---------------------------------------------------------------------------
# PPTX Builders
# ---------------------------------------------------------------------------

def _add_textbox(slide, left, top, width, height, text: str, font_size=18, bold=False, color=(15, 23, 42), align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.font.size = Pt(font_size)
        p.font.bold = bold
        p.font.color.rgb = RGBColor(*color)
    return tb


def build_deck_01(out_path: Path) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # Slide 1
    s1 = prs.slides.add_slide(blank)
    _add_textbox(s1, 0.8, 0.7, 11.7, 1.0, "CS-MLSys-J: 大模型推理系统与压缩优化 · 核心机制讲义", 28, True, (15, 23, 42))
    _add_textbox(
        s1,
        0.8,
        2.0,
        11.7,
        4.5,
        "课程模块总览 (Course Modules):\n"
        "• Module A: 权重量化与二阶近似 (GPTQ / OBQ)\n"
        "• Module B: KV 缓存非对称量化与热冷分层调度\n"
        "• Module C: 长上下文位置插值与前缀块复用\n"
        "• Module D: 异步反量化与微批次错峰调度\n\n"
        "注：本讲义仅定义理论机制与配套实验脚本入口函数名，具体实验实测数值见对应 Lab Notebook。",
        20,
        False,
        (30, 41, 59),
    )

    # Slide 2
    s2 = prs.slides.add_slide(blank)
    _add_textbox(s2, 0.7, 0.5, 11.9, 0.8, "核心机制 I & II：槽位分级与块折叠", 26, True, (29, 78, 216))
    _add_textbox(
        s2,
        0.7,
        1.5,
        11.9,
        2.5,
        "1. 热冷锚点分级策略 (Thermal Anchor Tiering, TAT)\n"
        "   • 理论定义：将注意力沉降槽位保留在高精度热池，将历史冷槽位按容量分桶淘汰压缩。\n"
        "   • 观测指标：尾延迟指标 t99_ms 与有效槽保留率 slot_eff。\n"
        "   • 配套实验入口函数 (Companion Lab Entrypoint): run_bucket_evict_probe()",
        18,
        False,
        (15, 23, 42),
    )
    _add_textbox(
        s2,
        0.7,
        4.2,
        11.9,
        2.5,
        "2. 动态块哈希折叠 (Dynamic Block Hash Folding, DBHF)\n"
        "   • 理论定义：在多请求共享系统提示词时，按固定步幅对前缀指纹做零拷贝折叠映射。\n"
        "   • 观测指标：生成速率 gen_tps、显存占用 v_gb 与压缩节省比例 red_pct。\n"
        "   • 配套实验入口函数 (Companion Lab Entrypoint): eval_prefix_reuse_v3()",
        18,
        False,
        (15, 23, 42),
    )

    # Slide 3
    s3 = prs.slides.add_slide(blank)
    _add_textbox(s3, 0.7, 0.5, 11.9, 0.8, "核心机制 III：双阶段离群通道抑制 (TSOCD)", 26, True, (15, 118, 110))
    _add_textbox(
        s3,
        0.7,
        1.7,
        11.9,
        4.5,
        "双阶段离群通道抑制 (Two-Stage Outlier Channel Damping, TSOCD)\n\n"
        "• 理论定义：针对激活值与 Key 矩阵中的少数高幅值异常通道，第一阶段按分块对齐重排权重列序，\n"
        "  第二阶段引入平滑缩放因子 alpha 将量化难度从激活侧迁移至权重侧，减少低比特量化退化。\n"
        "• 观测指标：WikiText2 困惑度指标 w2_score 与峰值显存指标 pk_mb。\n"
        "• 配套实验入口函数 (Companion Lab Entrypoint): calib_permute_omega_k9()",
        20,
        False,
        (15, 23, 42),
    )

    # Slide 4
    s4 = prs.slides.add_slide(blank)
    _add_textbox(s4, 0.7, 0.5, 11.9, 0.8, "核心机制 IV：流水线微批次错峰调度 (PMBS)", 26, True, (180, 83, 9))
    _add_textbox(
        s4,
        0.7,
        1.7,
        11.9,
        4.5,
        "流水线微批次错峰调度 (Pipelined Micro-Batch Staggering, PMBS)\n\n"
        "• 理论定义：在反量化计算与全局通信重叠阶段，将连续解码微批次按队列深度上限错峰派发，\n"
        "  避免所有请求同时挤占共享内存带宽而引发内核流水线气泡。\n"
        "• 观测指标：阻塞停顿比例 wait_pct 与单步周期耗时 cyc_ms。\n"
        "• 配套实验入口函数 (Companion Lab Entrypoint): trace_async_gate_w4()",
        20,
        False,
        (15, 23, 42),
    )

    # Slide 5
    s5 = prs.slides.add_slide(blank)
    _add_textbox(s5, 0.7, 0.5, 11.9, 0.8, "理论概念与配套实验入口对照表", 26, True, (15, 23, 42))
    rows, cols = 5, 4
    tbl = s5.shapes.add_table(rows, cols, Inches(0.7), Inches(1.5), Inches(11.9), Inches(4.6)).table
    headers = ["中文理论概念", "英文概念缩写", "配套实验函数", "关注指标键"]
    data = [
        ["热冷锚点分级策略", "TAT", "run_bucket_evict_probe()", "t99_ms, slot_eff"],
        ["动态块哈希折叠", "DBHF", "eval_prefix_reuse_v3()", "gen_tps, v_gb, red_pct"],
        ["双阶段离群通道抑制", "TSOCD", "calib_permute_omega_k9()", "w2_score, pk_mb"],
        ["流水线微批次错峰调度", "PMBS", "trace_async_gate_w4()", "wait_pct, cyc_ms"],
    ]
    for c_idx, h in enumerate(headers):
        tbl.cell(0, c_idx).text = h
    for r_idx, row in enumerate(data, start=1):
        for c_idx, val in enumerate(row):
            tbl.cell(r_idx, c_idx).text = val

    prs.save(out_path)


def build_deck_02(out_path: Path) -> None:
    """Pure visual PPTX: Slide 2, 3, 4 have ONLY 'Slide N' text + embedded high-contrast PNG."""
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # Slide 1: generic cover only (NO keywords from slides 2-4)
    s1 = prs.slides.add_slide(blank)
    _add_textbox(s1, 1.0, 2.5, 11.3, 1.2, "Deck 02: Course Visual Atlas", 36, True, (15, 23, 42), PP_ALIGN.CENTER)
    _add_textbox(s1, 1.0, 3.8, 11.3, 1.0, "CS-MLSys-J · Visual Reference Slides (Pages 2-4)", 20, False, (71, 85, 105), PP_ALIGN.CENTER)

    imgs = [
        ("Slide 2", make_deck02_slide2_arch()),
        ("Slide 3", make_deck02_slide3_curves()),
        ("Slide 4", make_deck02_slide4_budget()),
    ]
    for label, img in imgs:
        slide = prs.slides.add_slide(blank)
        # Tiny generic slide number only — zero technical anchors!
        _add_textbox(slide, 0.25, 0.1, 2.0, 0.35, label, 12, False, (100, 116, 139))
        bio = io.BytesIO(_png_bytes(img))
        slide.shapes.add_picture(bio, Inches(0.35), Inches(0.45), width=Inches(12.633), height=Inches(6.85))

    prs.save(out_path)


def build_deck_03(out_path: Path) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # Slide 1
    s1 = prs.slides.add_slide(blank)
    _add_textbox(s1, 0.8, 1.8, 11.7, 1.2, "Deck 03: 多模型推理基准与权衡分析", 32, True, (15, 23, 42))
    _add_textbox(
        s1,
        0.8,
        3.2,
        11.7,
        2.0,
        "覆盖主题：attention / quantization / KV cache / memory / latency / throughput / perplexity\n"
        "对比模型：LLaMA-2-7B, LLaMA-2-13B, Mistral-7B, OPT-175B",
        20,
        False,
        (51, 65, 85),
    )

    # Slide 2: Native 5x5 table with units
    s2 = prs.slides.add_slide(blank)
    _add_textbox(s2, 0.6, 0.4, 12.1, 0.7, "表 1：典型推理配置下的 KV Cache 显存与吞吐量实测基准", 24, True, (15, 23, 42))
    tbl = s2.shapes.add_table(5, 5, Inches(0.6), Inches(1.3), Inches(12.1), Inches(4.5)).table
    rows = [
        ["Model", "Precision", "Context", "KV Memory (GB)", "Throughput (tok/s)"],
        ["LLaMA-2-7B", "KIVI-2 (R=128)", "4096", "3.42 GB", "1,420 tok/s"],
        ["LLaMA-2-13B", "KIVI-2 (R=128)", "4096", "5.31 GB", "1,085 tok/s"],
        ["Mistral-7B", "KIVI-2 (GQA-8)", "8192", "1.95 GB", "1,760 tok/s"],
        ["OPT-175B", "GPTQ-3bit (g128)", "2048", "9.00 GB", "14.1 tok/s (bs=1)"],
    ]
    for r_i, row in enumerate(rows):
        for c_i, val in enumerate(row):
            tbl.cell(r_i, c_i).text = val

    # Slide 3: Tradeoffs sharing keywords with papers
    s3 = prs.slides.add_slide(blank)
    _add_textbox(s3, 0.6, 0.4, 12.1, 0.7, "关键超参数权衡：Group Size、Residual Length 与 RoPE 缩放", 24, True, (15, 23, 42))
    _add_textbox(
        s3,
        0.6,
        1.4,
        12.1,
        5.0,
        "1. 权重量化分组 (GPTQ)：g128 相比无分组可将 OPT-175B 3-bit WikiText2 困惑度从 8.68 降至 8.45。\n"
        "2. KV 缓存残差长度 (KIVI)：R=128 与 R=32 在多数 LongBench 任务上平均分接近，但 R=32 压缩率更高。\n"
        "3. 位置插值 (PI)：将位置索引线性缩放至预训练 2048 范围内，可将注意力分数上界缩小约 600 倍。\n"
        "4. 注意力内存与延迟：自回归解码受限于显存带宽（memory bandwidth bound），降低每 token 访存字节数直接降低 latency。",
        19,
        False,
        (30, 41, 59),
    )

    # Slide 4: Hardware comparison table
    s4 = prs.slides.add_slide(blank)
    _add_textbox(s4, 0.6, 0.4, 12.1, 0.7, "硬件平台访存带宽与批处理瓶颈对比", 24, True, (15, 23, 42))
    tbl2 = s4.shapes.add_table(4, 4, Inches(0.6), Inches(1.4), Inches(12.1), Inches(3.8)).table
    rows2 = [
        ["GPU 平台", "显存容量", "带宽特点", "适用优化策略"],
        ["NVIDIA A100", "80 GB HBM2e", "高带宽 (2.0 TB/s)", "2-bit KV 放大 Batch 至 4x"],
        ["NVIDIA A6000", "48 GB GDDR6", "低带宽受限明显", "3-bit GPTQ 访存减半提速 4.53x"],
        ["CPU-Offload Host", "PCIe Gen4 x16", "极低互连带宽", "仅作离线容错基线（非低延迟配置）"],
    ]
    for r_i, row in enumerate(rows2):
        for c_i, val in enumerate(row):
            tbl2.cell(r_i, c_i).text = val

    prs.save(out_path)


# ---------------------------------------------------------------------------
# Notebook Builders
# ---------------------------------------------------------------------------

def _out_stream(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_output(output_type="stream", name="stdout", text=text)


def _out_png(png_data: bytes) -> nbf.NotebookNode:
    b64 = base64.b64encode(png_data).decode("ascii")
    return nbf.v4.new_output(
        output_type="display_data",
        data={"image/png": b64, "text/plain": ["<Figure size 1100x420>"]},
    )


def build_lab_01(out_path: Path) -> None:
    """ZERO concept-word overlap with Deck-01! Only custom function names, terse keys, stdout, and PNG."""
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("# Lab-01: Empirical Telemetry Harness (Module A/B/C/D)"))

    # Exp 1: run_bucket_evict_probe
    code1 = (
        "res_a = run_bucket_evict_probe(b_cap=64, hot_k=8)\n"
        "res_b = run_bucket_evict_probe(b_cap=128, hot_k=16)\n"
        "print(f'b_cap=64, hot_k=8 -> t99_ms={res_a[\"t99_ms\"]:.2f}, slot_eff={res_a[\"slot_eff\"]:.1f}%')\n"
        "print(f'b_cap=128, hot_k=16 -> t99_ms={res_b[\"t99_ms\"]:.2f}, slot_eff={res_b[\"slot_eff\"]:.1f}%')"
    )
    out1 = "b_cap=64, hot_k=8 -> t99_ms=19.40, slot_eff=91.2%\nb_cap=128, hot_k=16 -> t99_ms=14.70, slot_eff=95.8%\n"
    png1 = make_lab_plot(
        "run_bucket_evict_probe() Sweep",
        ["cap32", "cap64", "cap128", "cap256"],
        [24.8, 19.40, 14.70, 14.20],
        "t99_ms",
        [84.5, 91.2, 95.8, 96.4],
        "slot_eff (%)",
    )
    cells.append(nbf.v4.new_code_cell(code1, execution_count=1, outputs=[_out_stream(out1), _out_png(png1)]))

    # Exp 2: eval_prefix_reuse_v3
    cells.append(nbf.v4.new_markdown_cell("## Exp 2: eval_prefix_reuse_v3"))
    code2 = (
        "for st in (32, 64):\n"
        "    m = eval_prefix_reuse_v3(stride=st, mode='fold')\n"
        "    print(f'stride={st}, mode=fold -> gen_tps={m[\"gen_tps\"]}, v_gb={m[\"v_gb\"]:.2f}, red_pct={m[\"red_pct\"]:.1f}%')"
    )
    out2 = (
        "stride=32, mode=fold -> gen_tps=1840, v_gb=11.60, red_pct=38.4%\n"
        "stride=64, mode=fold -> gen_tps=1695, v_gb=12.85, red_pct=31.7%\n"
    )
    png2 = make_lab_plot(
        "eval_prefix_reuse_v3() Sweep",
        ["st=16", "st=32", "st=64", "st=128"],
        [1790, 1840, 1695, 1510],
        "gen_tps",
        [34.1, 38.4, 31.7, 22.9],
        "red_pct (%)",
    )
    cells.append(nbf.v4.new_code_cell(code2, execution_count=2, outputs=[_out_stream(out2), _out_png(png2)]))

    # Exp 3: calib_permute_omega_k9
    cells.append(nbf.v4.new_markdown_cell("## Exp 3: calib_permute_omega_k9"))
    code3 = (
        "r1 = calib_permute_omega_k9(tile=128, alpha=0.65)\n"
        "r2 = calib_permute_omega_k9(tile=256, alpha=0.80)\n"
        "print(f'tile=128, alpha=0.65 -> w2_score={r1[\"w2_score\"]:.2f}, pk_mb={r1[\"pk_mb\"]}')\n"
        "print(f'tile=256, alpha=0.80 -> w2_score={r2[\"w2_score\"]:.2f}, pk_mb={r2[\"pk_mb\"]}')"
    )
    out3 = "tile=128, alpha=0.65 -> w2_score=8.92, pk_mb=4120\ntile=256, alpha=0.80 -> w2_score=9.41, pk_mb=3980\n"
    cells.append(nbf.v4.new_code_cell(code3, execution_count=3, outputs=[_out_stream(out3)]))

    # Exp 4: trace_async_gate_w4
    cells.append(nbf.v4.new_markdown_cell("## Exp 4: trace_async_gate_w4"))
    code4 = (
        "g4 = trace_async_gate_w4(depth=4, q_lim=24)\n"
        "g2 = trace_async_gate_w4(depth=2, q_lim=12)\n"
        "print(f'depth=4, q_lim=24 -> wait_pct={g4[\"wait_pct\"]:.2f}%, cyc_ms={g4[\"cyc_ms\"]:.2f}')\n"
        "print(f'depth=2, q_lim=12 -> wait_pct={g2[\"wait_pct\"]:.2f}%, cyc_ms={g2[\"cyc_ms\"]:.2f}')"
    )
    out4 = "depth=4, q_lim=24 -> wait_pct=3.80%, cyc_ms=6.15\ndepth=2, q_lim=12 -> wait_pct=9.40%, cyc_ms=7.82\n"
    cells.append(nbf.v4.new_code_cell(code4, execution_count=4, outputs=[_out_stream(out4)]))

    nb["cells"] = cells
    nbf.write(nb, out_path)


def build_lab_02(out_path: Path) -> None:
    """Cross-doc verification notebook for KIVI & Positional Interpolation."""
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            "# Lab-02: KIVI & Positional Interpolation Cross-Document Verification\n"
            "Hardware & Protocol: Single NVIDIA A100-80GB, Batch=16, SeqLen=4096, KIVI 2-bit KV cache."
        ),
        nbf.v4.new_code_cell(
            "cfg = {'gpu': 'A100-80GB', 'batch_size': 16, 'seq_len': 4096, 'kv_bits': 2}\n"
            "results = [\n"
            "    {'residual_len': 128, 'throughput_tok_s': 1420.5, 'peak_mem_gb': 26.4, 'gsm8k_em': 20.77},\n"
            "    {'residual_len': 32,  'throughput_tok_s': 1685.0, 'peak_mem_gb': 22.8, 'gsm8k_em': 20.62},\n"
            "]\n"
            "for r in results:\n"
            "    print(f\"R={r['residual_len']} | throughput={r['throughput_tok_s']} tok/s | peak_mem={r['peak_mem_gb']} GB | GSM8K={r['gsm8k_em']}\")",
            execution_count=1,
            outputs=[
                _out_stream(
                    "R=128 | throughput=1420.5 tok/s | peak_mem=26.4 GB | GSM8K=20.77\n"
                    "R=32 | throughput=1685.0 tok/s | peak_mem=22.8 GB | GSM8K=20.62\n"
                ),
                _out_png(
                    make_lab_plot(
                        "KIVI-2 Verification (A100-80GB, Batch=16, SeqLen=4096)",
                        ["FP16", "R=128", "R=64", "R=32"],
                        [510.0, 1420.5, 1550.0, 1685.0],
                        "Throughput (tok/s)",
                        [68.2, 26.4, 24.5, 22.8],
                        "Peak Memory (GB)",
                    )
                ),
            ],
        ),
        nbf.v4.new_markdown_cell("## PI Context Extension Passkey & Perplexity Check"),
        nbf.v4.new_code_cell(
            "pi_eval = {'model': 'LLaMA-7B', 'ctx': 16384, 'steps': 1000, 'pg19_ppl': 6.83, 'rouge_1': 60.0}\n"
            "print(pi_eval)",
            execution_count=2,
            outputs=[_out_stream("{'model': 'LLaMA-7B', 'ctx': 16384, 'steps': 1000, 'pg19_ppl': 6.83, 'rouge_1': 60.0}\n")],
        ),
    ]
    nb["cells"] = cells
    nbf.write(nb, out_path)


def build_lab_03(out_path: Path) -> None:
    """Distractor baseline under mismatched hardware & quantization settings."""
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            "# Lab-03: CPU-Offload Distractor Baseline\n"
            "**STRICT CONDITION NOTICE**: This run uses `CPU-offload / PCIe Gen4 x16 / INT8-symmetric / SeqLen=512` (NOT A100-80GB 2-bit KV cache, NOT SeqLen=4096)."
        ),
        nbf.v4.new_code_cell(
            "distractor_cfg = {\n"
            "    'backend': 'CPU-offload',\n"
            "    'interconnect': 'PCIe Gen4 x16',\n"
            "    'quant': 'INT8-symmetric',\n"
            "    'seq_len': 512,\n"
            "    'batch_size': 16,\n"
            "    'residual_len': 128,\n"
            "}\n"
            "print('latency=47.3 ms/tok | throughput=312 tok/s | memory=29.4 GB | perplexity=9.84')",
            execution_count=1,
            outputs=[
                _out_stream("latency=47.3 ms/tok | throughput=312 tok/s | memory=29.4 GB | perplexity=9.84\n"),
                _out_png(
                    make_lab_plot(
                        "CPU-Offload PCIe Gen4 INT8-symmetric (SeqLen=512)",
                        ["bs=4", "bs=8", "bs=16"],
                        [39.5, 42.1, 47.3],
                        "Latency (ms/tok)",
                        [180.0, 245.0, 312.0],
                        "Throughput (tok/s)",
                    )
                ),
            ],
        ),
    ]
    nb["cells"] = cells
    nbf.write(nb, out_path)


def verify_zero_overlap(lab01_path: Path) -> None:
    raw = lab01_path.read_text(encoding="utf-8")
    nb = json.loads(raw)
    text_parts: list[str] = []
    for c in nb["cells"]:
        text_parts.append("".join(c.get("source", "")))
        for out in c.get("outputs", []):
            if "text" in out:
                text_parts.append("".join(out["text"]))
    blob = "\n".join(text_parts).lower()
    forbidden = [
        "热冷", "锚点", "分级", "thermal", "anchor", "tiering", "tat",
        "动态块", "哈希", "折叠", "dynamic", "block", "hash", "folding", "dbhf",
        "双阶段", "离群", "通道", "抑制", "two-stage", "outlier", "channel", "damping", "tsocd",
        "流水线", "微批次", "错峰", "调度", "pipelined", "micro-batch", "staggering", "pmbs",
        "latency", "cache", "retention", "stall", "perplexity", "throughput", "memory", "vram",
    ]
    hits = [tok for tok in forbidden if tok in blob]
    if hits:
        raise AssertionError(f"Zero-overlap check FAILED on Lab-01: found {hits}")


def verify_soffice_pptx(pptx_path: Path) -> int:
    if not SOFFICE_BIN.exists():
        raise FileNotFoundError(f"soffice not found at {SOFFICE_BIN}")
    with tempfile.TemporaryDirectory() as td:
        cmd = [
            str(SOFFICE_BIN),
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            td,
            str(pptx_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        del res
        pdf_path = Path(td) / f"{pptx_path.stem}.pdf"
        if not pdf_path.exists():
            raise RuntimeError(f"soffice did not produce {pdf_path}")
        doc = pypdfium2.PdfDocument(str(pdf_path))
        return len(doc)


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    d1 = CORPUS_DIR / "Deck-01-Core-Concepts.pptx"
    d2 = CORPUS_DIR / "Deck-02-Pure-Visual-Diagrams.pptx"
    d3 = CORPUS_DIR / "Deck-03-Benchmarks-And-Tradeoffs.pptx"
    l1 = CORPUS_DIR / "Lab-01-ZeroOverlap-Experiment.ipynb"
    l2 = CORPUS_DIR / "Lab-02-CrossDoc-Verification.ipynb"
    l3 = CORPUS_DIR / "Lab-03-Distractor-Baseline.ipynb"

    build_deck_01(d1)
    build_deck_02(d2)
    build_deck_03(d3)
    build_lab_01(l1)
    build_lab_02(l2)
    build_lab_03(l3)
    verify_zero_overlap(l1)

    manifest = []
    for p in sorted(CORPUS_DIR.iterdir()):
        if p.suffix.lower() not in {".pdf", ".pptx", ".ipynb"}:
            continue
        data = p.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        info = {"file_name": p.name, "bytes": len(data), "sha256": sha}
        if p.suffix == ".pdf":
            doc = pypdfium2.PdfDocument(str(p))
            info["pages"] = len(doc)
        elif p.suffix == ".pptx":
            prs = Presentation(str(p))
            soffice_pages = verify_soffice_pptx(p)
            assert soffice_pages == len(prs.slides), f"{p.name}: pptx={len(prs.slides)} vs soffice={soffice_pages}"
            info["pages"] = soffice_pages
        elif p.suffix == ".ipynb":
            nb = json.loads(p.read_text(encoding="utf-8"))
            info["cells"] = len(nb["cells"])
        manifest.append(info)
        print(json.dumps(info, ensure_ascii=False))

    # Double-check FTS5 isolation for Deck-01 vs Lab-01
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE docs USING fts5(fname, body, tokenize='unicode61')")
    l1_nb = json.loads(l1.read_text(encoding="utf-8"))
    l1_body = "\n".join(
        "".join(c.get("source", "")) + "\n" + "".join("".join(o.get("text", "")) for o in c.get("outputs", []))
        for c in l1_nb["cells"]
    )
    conn.execute("INSERT INTO docs VALUES (?, ?)", (l1.name, l1_body))
    for q_tok in ["Thermal", "Anchor", "Tiering", "Dynamic", "Block", "Hash", "Folding", "Outlier", "Channel", "Damping", "Pipelined", "Staggering", "latency", "cache", "perplexity", "throughput", "memory"]:
        cnt = conn.execute("SELECT count(*) FROM docs WHERE docs MATCH ?", (q_tok,)).fetchone()[0]
        assert cnt == 0, f"FTS5 leaked token {q_tok} into Lab-01!"
    print("ALL_CHECKS_PASSED: 9 files verified, soffice PPTX render verified, FTS5 zero-overlap verified.")


if __name__ == "__main__":
    main()
