"""Build public open synthetic J02 dev-eval PPTX / IPYNB / Distractor fixtures."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.util import Inches, Pt

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
MAIN_DIR = FIXTURE_DIR / "main_generated"
DISTRACTOR_DIR = FIXTURE_DIR / "distractor"


def _draw_panel(width: int, height: int, bg: tuple[int, int, int] = (250, 252, 255)) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (width, height), bg)
    return img, ImageDraw.Draw(img)


def _save_png(img: Image.Image, path: Path) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    path.write_bytes(data)
    return data


def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill=(20, 28, 45), scale: int = 2) -> None:
    # Render readable bitmap text without depending on external TTF fonts (Pillow default font + 2x crisp upsampling).
    tmp = Image.new("RGBA", (max(400, len(text) * 10 + 40), 28), (255, 255, 255, 0))
    td = ImageDraw.Draw(tmp)
    td.text((2, 2), text, fill=(*fill, 255))
    resized = tmp.resize((tmp.width * scale, tmp.height * scale), Image.Resampling.NEAREST)
    draw._image.paste(resized, xy, resized)


def _make_architecture_diagram(path: Path) -> None:
    img, draw = _draw_panel(1280, 720, (248, 250, 254))
    _draw_text(draw, (60, 28), "Sparse MoE Token Dispatch Architecture (Diagram)", (15, 23, 42), scale=3)
    # Input box
    draw.rectangle((420, 100, 860, 175), fill=(224, 242, 254), outline=(2, 132, 199), width=4)
    _draw_text(draw, (445, 122), "Input Hidden State: dim d = 4096", (12, 74, 110), scale=2)
    # Gate box
    draw.rectangle((380, 235, 900, 325), fill=(254, 243, 199), outline=(217, 119, 6), width=4)
    _draw_text(draw, (405, 255), "Router Linear Gate W_g -> Top-2 Softmax", (120, 53, 15), scale=2)
    draw.line((640, 175, 640, 235), fill=(30, 41, 59), width=4)
    # 8 Experts
    for i in range(8):
        x0 = 65 + i * 146
        x1 = x0 + 126
        active = i in {2, 5}
        fill = (254, 215, 170) if active else (241, 245, 249)
        outline = (234, 88, 12) if active else (148, 163, 184)
        draw.rectangle((x0, 410, x1, 490), fill=fill, outline=outline, width=4)
        _draw_text(draw, (x0 + 14, 434), f"Expert-{i}", (30, 41, 59), scale=2)
        if active:
            draw.line((640, 325, (x0 + x1) // 2, 410), fill=(234, 88, 12), width=5)
    # Metrics box
    draw.rectangle((140, 545, 1140, 670), fill=(236, 253, 245), outline=(5, 150, 105), width=4)
    _draw_text(draw, (170, 562), "Active Params per Token = 12.9 B   |   Capacity Factor = 1.25", (6, 78, 59), scale=2)
    _draw_text(draw, (170, 610), "Max Tokens Dropped / Step = 0.4%   |   Dispatch Ceiling = 1.85 ms", (6, 78, 59), scale=2)
    _save_png(img, path)


def _make_heatmap_diagram(path: Path) -> None:
    img, draw = _draw_panel(1280, 720, (252, 252, 254))
    _draw_text(draw, (60, 28), "Router Probability Distribution Heatmap (RouteHeat-Eval32)", (15, 23, 42), scale=3)
    layers = ["Layer 0", "Layer 8", "Layer 16", "Layer 24"]
    vals = [
        [12.5, 12.4, 12.6, 12.5, 12.5, 12.4, 12.6, 12.5],
        [11.0, 10.2, 16.4, 11.8, 15.2, 13.1, 11.5, 10.8],
        [9.4, 8.1, 14.2, 10.3, 24.6, 13.8, 10.7, 8.9],
        [13.8, 4.2, 15.1, 12.0, 16.9, 14.5, 12.2, 11.3],
    ]
    for r, layer in enumerate(layers):
        y0 = 135 + r * 95
        _draw_text(draw, (55, y0 + 20), layer, (30, 41, 59), scale=2)
        for c, v in enumerate(vals[r]):
            x0 = 220 + c * 120
            intensity = int(max(40, min(255, 255 - (v - 4.0) * 9.5)))
            fill = (255, intensity, max(60, intensity - 40)) if v >= 20 else (intensity, intensity + 20 if intensity <= 235 else 255, 245)
            if v <= 5.0:
                fill = (219, 234, 254)
            draw.rectangle((x0, y0, x0 + 108, y0 + 75), fill=fill, outline=(71, 85, 105), width=2)
            _draw_text(draw, (x0 + 14, y0 + 24), f"{v:.1f}%", (15, 23, 42), scale=2)
    for c in range(8):
        _draw_text(draw, (232 + c * 120, 95), f"E-{c}", (30, 41, 59), scale=2)
    draw.rectangle((100, 545, 1180, 675), fill=(248, 250, 252), outline=(51, 65, 85), width=3)
    _draw_text(draw, (130, 562), "Peak Share: Layer 16 Expert-4 = 24.6%   |   Uniform Baseline = 12.5%", (127, 29, 29), scale=2)
    _draw_text(draw, (130, 610), "Cold Minimum: Layer 24 Expert-1 = 4.2%", (30, 58, 138), scale=2)
    _save_png(img, path)


def _make_curve_diagram(path: Path) -> None:
    img, draw = _draw_panel(1280, 720, (255, 255, 255))
    _draw_text(draw, (60, 28), "KV Block Size vs Memory Overhead Curve (Optimal Knee Plot)", (15, 23, 42), scale=3)
    # Axes
    draw.line((180, 120, 180, 510), fill=(15, 23, 42), width=4)
    draw.line((180, 510, 1140, 510), fill=(15, 23, 42), width=4)
    _draw_text(draw, (500, 525), "KV Block Size (tokens)", (30, 41, 59), scale=2)
    _draw_text(draw, (20, 260), "Overhead %", (30, 41, 59), scale=2)
    pts = [(300, 200), (620, 460), (960, 380)]
    draw.line(pts, fill=(37, 99, 235), width=6)
    labels = [
        ("Block=4: 28.5%", pts[0], (220, 38, 38)),
        ("Block=16: 3.9% (Optimal Knee)", pts[1], (5, 150, 105)),
        ("Block=64: 11.4%", pts[2], (37, 99, 235)),
    ]
    for txt, (px, py), col in labels:
        draw.ellipse((px - 12, py - 12, px + 12, py + 12), fill=col, outline=(15, 23, 42), width=3)
        _draw_text(draw, (px - 80, py - 52), txt, col, scale=2)
    draw.rectangle((180, 585, 1120, 670), fill=(239, 246, 255), outline=(29, 78, 216), width=3)
    _draw_text(draw, (210, 610), "Optimal Point Peak Context = 128k tokens   (Block=16 tokens)", (30, 58, 138), scale=2)
    _save_png(img, path)


def _make_dual_subplot(left_path: Path, right_path: Path) -> None:
    # Left subplot: Perplexity vs Steps (decreasing)
    img1, draw1 = _draw_panel(640, 420, (250, 252, 255))
    draw1.rectangle((10, 10, 630, 410), outline=(100, 116, 139), width=3)
    _draw_text(draw1, (45, 28), "Left: Eval Perplexity vs Steps", (15, 23, 42), scale=2)
    draw1.line((80, 340, 560, 340), fill=(30, 41, 59), width=3)
    draw1.line((80, 90, 80, 340), fill=(30, 41, 59), width=3)
    draw1.line([(100, 120), (260, 195), (420, 260), (540, 295)], fill=(220, 38, 38), width=5)
    _draw_text(draw1, (140, 355), "Steps: 0k -> 40k (PPL 28.4 -> 15.2)", (127, 29, 29), scale=1)
    _save_png(img1, left_path)

    # Right subplot: Peak VRAM (GB) vs Sequence Length (increasing)
    img2, draw2 = _draw_panel(640, 420, (250, 252, 255))
    draw2.rectangle((10, 10, 630, 410), outline=(100, 116, 139), width=3)
    _draw_text(draw2, (45, 28), "Right: Peak VRAM (GB) vs Context", (15, 23, 42), scale=2)
    draw2.line((80, 340, 560, 340), fill=(30, 41, 59), width=3)
    draw2.line((80, 90, 80, 340), fill=(30, 41, 59), width=3)
    draw2.line([(100, 310), (260, 275), (420, 210), (540, 135)], fill=(5, 150, 105), width=5)
    _draw_text(draw2, (130, 355), "Context: 4k -> 64k (VRAM 4.2 -> 18.6 GB)", (6, 78, 59), scale=1)
    _save_png(img2, right_path)


def _add_text_slide(prs: Presentation, title: str, bullets: list[str]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title_box = slide.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(12.0), Inches(0.9))
    tf = title_box.text_frame
    tf.text = title
    tf.paragraphs[0].font.size = Pt(26)
    tf.paragraphs[0].font.bold = True

    body_box = slide.shapes.add_textbox(Inches(0.6), Inches(1.4), Inches(12.0), Inches(5.4))
    btf = body_box.text_frame
    btf.word_wrap = True
    for idx, line in enumerate(bullets):
        para = btf.paragraphs[0] if idx == 0 else btf.add_paragraph()
        para.text = line
        para.font.size = Pt(16)


def _add_pure_image_slide(prs: Presentation, img_path: Path, placeholder_text: str = "") -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    if placeholder_text:
        box = slide.shapes.add_textbox(Inches(0.4), Inches(0.15), Inches(3.0), Inches(0.4))
        box.text_frame.text = placeholder_text
    slide.shapes.add_picture(str(img_path), Inches(0.4), Inches(0.65), width=Inches(12.5))


def build_all() -> None:
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    DISTRACTOR_DIR.mkdir(parents=True, exist_ok=True)
    tmp_img = FIXTURE_DIR / "_assets"
    tmp_img.mkdir(parents=True, exist_ok=True)

    arch_png = tmp_img / "arch_diagram.png"
    heat_png = tmp_img / "heatmap_diagram.png"
    curve_png = tmp_img / "kv_curve_diagram.png"
    left_png = tmp_img / "left_ppl.png"
    right_png = tmp_img / "right_vram.png"
    _make_architecture_diagram(arch_png)
    _make_heatmap_diagram(heat_png)
    _make_curve_diagram(curve_png)
    _make_dual_subplot(left_png, right_png)

    # 1. Lecture03-Routing-Architecture.pptx (8 slides, 2 pure-image slides)
    prs3 = Presentation()
    prs3.slide_width = Inches(13.333)
    prs3.slide_height = Inches(7.5)
    _add_text_slide(
        prs3,
        "Lecture 03 · Sparse Mixture-of-Experts Routing Architecture",
        [
            "Course Unit: Distributed Large Language Model Systems",
            "Overview: Token routing topology, capacity limits, load balancing, and empirical audits.",
        ],
    )
    _add_text_slide(
        prs3,
        "Token Dispatch Rules & Auxiliary Load Balancing",
        [
            "Sparse MoE 分发规则：每 Token 默认分派 Top-2 专家（默认分派专家总数 = 8 个，Expert-0..7）。",
            "Router logits 采用辅助负载均衡系数 α = 0.01，确保跨设备容量分配均衡。",
            "真实算子审计与实测指标记录于 `dispatch_audit_v7`（日志标签 `MetricSet-8X`）。",
        ],
    )
    _add_pure_image_slide(prs3, arch_png, "")
    _add_pure_image_slide(prs3, heat_png, "")
    _add_text_slide(
        prs3,
        "Capacity Factor Trade-offs",
        [
            "Capacity factor scales expert buffer slots relative to average token share.",
            "Lower capacity increases token drops; higher capacity expands activation memory.",
        ],
    )
    _add_text_slide(
        prs3,
        "All-to-All Dispatch Communication Overlap",
        [
            "NCCL all-to-all scatter/gather kernels overlap across micro-batches in pipeline stage.",
        ],
    )
    _add_text_slide(
        prs3,
        "Auxiliary Loss Sensitivity Across Depths",
        [
            "Middle and upper transformer blocks exhibit stronger domain specialization.",
        ],
    )
    _add_text_slide(
        prs3,
        "Summary · Lecture 03 Routing Guidelines",
        [
            "Keep auxiliary alpha small to preserve gradient direction while avoiding expert collapse.",
        ],
    )
    prs3.save(MAIN_DIR / "Lecture03-Routing-Architecture.pptx")

    # 2. Lecture05-KV-Quant-LabGuide.pptx (8 slides, 1 minimal-text pure-image slide + 2-hop anchors)
    prs5 = Presentation()
    prs5.slide_width = Inches(13.333)
    prs5.slide_height = Inches(7.5)
    _add_text_slide(
        prs5,
        "Lecture 05 · KV Cache Fragmentation & Compression Lab Guide",
        [
            "Course Unit: Inference Memory Management & Quantization Protocols",
            "Covers buddy allocator blocks, adaptive compaction, and empirical lab probes.",
        ],
    )
    _add_text_slide(
        prs5,
        "KV 缓存伙伴分页分配政策（Buddy Slab Policy）",
        [
            "默认页块容量 block_size = 16 tokens；理论内部碎片控制上限 < 4%。",
            "实测回放日志与分配器轨迹由函数 `slab_probe_k12` 在数据集 `TraceLog-L128` 执行并保存。",
        ],
    )
    _add_text_slide(
        prs5,
        "FP8 KV Dequantization Kernel Notes",
        [
            "E4M3 per-head scaling factors preserve attention logit fidelity at long context windows.",
        ],
    )
    _add_pure_image_slide(prs5, curve_png, "Slide 4")
    _add_text_slide(
        prs5,
        "Preemption Ordering Under Burst Load",
        [
            "Least-remaining-tokens eviction protects near-completion decoding streams.",
        ],
    )
    _add_text_slide(
        prs5,
        "动态阈值页面压缩（Adaptive Slack Compaction）",
        [
            "定义与触发条件：当可用连续块池空闲比例低于 12%（free block ratio < 0.12）时触发压缩整理。",
            "该机制的实测回收比例与重映射延迟记录于 Lab06 实验遥测单元中。",
        ],
    )
    _add_text_slide(
        prs5,
        "Multi-Tenant Prefix Sharing Checklist",
        [
            "Copy-on-write page tables require hash verification before merging identical system prompts.",
        ],
    )
    _add_text_slide(
        prs5,
        "Summary · Lecture 05 Lab Safety Rules",
        [
            "Do not extrapolate unmeasured hardware counts across unseen cluster topologies.",
        ],
    )
    prs5.save(MAIN_DIR / "Lecture05-KV-Quant-LabGuide.pptx")

    # 3. Lecture07-Optimizer-Tradeoffs.pptx (6 slides: native table + side-by-side dual image + 2-hop concept)
    prs7 = Presentation()
    prs7.slide_width = Inches(13.333)
    prs7.slide_height = Inches(7.5)
    _add_text_slide(
        prs7,
        "Lecture 07 · Low-Rank Projection & Optimizer Trade-offs",
        [
            "Course Unit: Memory-Efficient Pretraining & Fine-Tuning",
        ],
    )
    # Slide 2: Native PPTX table
    s2 = prs7.slides.add_slide(prs7.slide_layouts[6])
    s2.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(11), Inches(0.7)).text_frame.text = (
        "Optimizer Memory & Perplexity Comparison Table (1B Model)"
    )
    table_shape = s2.shapes.add_table(4, 5, Inches(0.6), Inches(1.4), Inches(11.8), Inches(3.2))
    rows = [
        ["Method", "Rank (r)", "Update Freq (T)", "Memory (1B Model)", "Perplexity (C4)"],
        ["AdamW", "full", "1", "13.60 GB", "15.12"],
        ["Adafactor", "rank-1", "1", "13.15 GB", "15.89"],
        ["GaLore-Int8", "256", "200", "7.95 GB", "15.24"],
    ]
    for r_i, row in enumerate(rows):
        for c_i, val in enumerate(row):
            table_shape.table.cell(r_i, c_i).text = val

    # Slide 3: Side-by-side two images (tests Boundary #2 comparative directional disambiguation)
    s3 = prs7.slides.add_slide(prs7.slide_layouts[6])
    s3.shapes.add_textbox(Inches(0.6), Inches(0.3), Inches(12), Inches(0.7)).text_frame.text = (
        "Training Steps vs Context Scaling Curves"
    )
    s3.shapes.add_picture(str(left_png), Inches(0.5), Inches(1.3), width=Inches(5.9))
    s3.shapes.add_picture(str(right_png), Inches(6.9), Inches(1.3), width=Inches(5.9))

    # Slide 4: Implicit two-hop without literal snake_case anchor
    _add_text_slide(
        prs7,
        "异构双阶投影切换策略（Two-Stage Rank Transition）",
        [
            "适用阶段：主要面向长上下文预训练后期梯度主方向变化趋缓的训练阶段。",
            "通过高低秩交替切换平衡状态压缩率与收敛稳定性；详细消融实测记于 Lab06 正交切换实验单元。",
        ],
    )
    _add_text_slide(
        prs7,
        "Gradient Accumulation & Master Weights",
        [
            "Accumulate gradients in BF16 and update low-rank factors cleanly after boundary sync.",
        ],
    )
    _add_text_slide(
        prs7,
        "Summary · Optimizer Selection",
        [
            "Verify exact rank, scale alpha, and token batch size from tables before deploying.",
        ],
    )
    prs7.save(MAIN_DIR / "Lecture07-Optimizer-Tradeoffs.pptx")

    # 4. Lab04-Sparse-Routing-Bench.ipynb (zero shared conceptual keywords!)
    png_b64 = base64.b64encode(curve_png.read_bytes()).decode("ascii")
    nb4 = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "id": "lab04-md-0",
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Lab04 · Execution Telemetry Artifacts\n"],
            },
            {
                "id": "lab04-md-1",
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## MetricSet-8X Dispatch Kernel Run\n"],
            },
            {
                "id": "lab04-c-1",
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "source": [
                    "def dispatch_audit_v7(batch_tokens=4096):\n",
                    "    return execute_kernel(batch_tokens, tag='MetricSet-8X')\n",
                ],
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": [
                            "[MetricSet-8X] dispatch_audit_v7 -> ops_share=0.284 | skew_cv=0.062 | cycle_ms=11.40 | safe_seq=3072\n"
                        ],
                    }
                ],
            },
            {
                "id": "lab04-md-2",
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## TraceLog-L128 Replay Harness\n"],
            },
            {
                "id": "lab04-c-2",
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "source": [
                    "def slab_probe_k12(profile='TraceLog-L128'):\n",
                    "    return replay_corpus(profile)\n",
                ],
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": [
                            "[TraceLog-L128] slab_probe_k12 -> internal_slack=2.14% | tail_reserve_mb=48.5 | miss_traps=19 | sustained_rate=3890\n"
                        ],
                    }
                ],
            },
            {
                "id": "lab04-c-3",
                "cell_type": "code",
                "execution_count": 3,
                "metadata": {},
                "source": ["# Artifact summary render\n", "render_curve_snapshot()\n"],
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {"image/png": png_b64, "text/plain": ["<CurveFigure>"]},
                        "metadata": {},
                    }
                ],
            },
        ],
    }
    (MAIN_DIR / "Lab04-Sparse-Routing-Bench.ipynb").write_text(
        json.dumps(nb4, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 5. Lab06-KV-Precision-Ablation.ipynb (implicit cross-term cells, distinct markdown headings)
    nb6 = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "id": "lab06-md-0",
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Lab06 · Telemetry & Ablation Records\n"],
            },
            {
                "id": "lab06-md-1",
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Phase-II SVD RankSwitch (32k ctx)\n"],
            },
            {
                "id": "lab06-c-1",
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "source": [
                    "run_rank_switch(rank_pair=(512, 128), ctx=32768)\n",
                ],
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": [
                            "[Phase-II SVD RankSwitch] ctx=32768 | vram_peak=6.82 GB | eval_score=14.91 ppl | switch_cost=42.0 ms\n"
                        ],
                    }
                ],
            },
            {
                "id": "lab06-md-2",
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Compact-Ratio-64K Telemetry\n"],
            },
            {
                "id": "lab06-c-2",
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "source": [
                    "collect_asc_telemetry(window=65536)\n",
                ],
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": [
                            "profile=ASC-64K -> reclaim_pct=18.4% | remap_lat=0.87 ms | stall_cnt=0\n"
                        ],
                    }
                ],
            },
        ],
    }
    (MAIN_DIR / "Lab06-KV-Precision-Ablation.ipynb").write_text(
        json.dumps(nb6, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 6. Distractor course (j-course-distractor): same anchors/keywords, decoy bogus metrics!
    pdf_text_lines = [
        "Distractor Routing Notes (Cross-Course Isolation Decoy)",
        "Distractor-405B-Q2: measured throughput = 9999 tok/s, alpha = 0.99.",
        "LLaMA-3 70B 4-bit GaLore end-to-end throughput = 7777 tok/s.",
        "dispatch_audit_v7 fake trap: ops_share = 0.999, skew_cv = 0.91.",
    ]
    stream_lines = ["BT", "/F1 12 Tf", "50 740 Td", "18 TL"]
    for ln in pdf_text_lines:
        esc = ln.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_lines.append(f"({esc}) Tj T*")
    stream_lines.append("ET")
    stream_data = "\n".join(stream_lines).encode("latin-1")
    objs = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n",
        f"4 0 obj << /Length {len(stream_data)} >> stream\n".encode("latin-1") + stream_data + b"\nendstream endobj\n",
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
    ]
    buf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objs:
        offsets.append(len(buf))
        buf.extend(obj)
    xref_pos = len(buf)
    buf.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode("latin-1"))
    for off in offsets[1:]:
        buf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    buf.extend(f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin-1"))
    (DISTRACTOR_DIR / "Distractor-Routing-Notes.pdf").write_bytes(bytes(buf))

    d_nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "id": "dist-c-0",
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "source": ["def slab_probe_k12(): return 'trap'\n"],
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": ["[TraceLog-L128] internal_slack=99.9% | tail_reserve_mb=999.0 | profile=ASC-64K reclaim_pct=99.9%\n"],
                    }
                ],
            }
        ],
    }
    (DISTRACTOR_DIR / "Distractor-Lab.ipynb").write_text(
        json.dumps(d_nb, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    shutil.rmtree(tmp_img, ignore_errors=True)
    print("Built generated fixtures cleanly in", FIXTURE_DIR)


if __name__ == "__main__":
    build_all()
