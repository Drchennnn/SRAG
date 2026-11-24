#!/usr/bin/env python3
"""Generate and verify the frozen cs-ctrl-h blind knowledge-graph corpus for H-gate."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat as nbf
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches as PptxInches, Pt as PptxPt
import pypdfium2

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.rag.parsers.local_notebook import parse_local_notebook  # noqa: E402
from app.rag.parsers.local_slides import parse_local_slides  # noqa: E402

GATE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = GATE_DIR / "course_corpus"
CRITERIA_JSON = GATE_DIR / "H-blind-criteria.json"
SOFFICE_BIN = Path("/opt/homebrew/bin/soffice")
CHROME_BIN = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_MONO = "/System/Library/Fonts/Menlo.ttc"


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _out_stream(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_output("stream", name="stdout", text=text)


def _out_png(png_bytes: bytes) -> nbf.NotebookNode:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return nbf.v4.new_output("display_data", data={"image/png": b64, "text/plain": ["<Figure>"]})


def _make_dual_metric_chart(
    title: str,
    x_labels: list[str],
    y1: list[float],
    y1_name: str,
    y2: list[float],
    y2_name: str,
    footer_note: str,
) -> bytes:
    """Render a crisp high-contrast PNG chart using purely neutral metric labels."""
    img = Image.new("RGB", (1120, 440), "#FFFFFF")
    d = ImageDraw.Draw(img)
    f_t = _font(FONT_BOLD, 22)
    f_m = _font(FONT_MONO, 17)
    f_s = _font(FONT_BOLD, 16)

    d.rectangle([0, 0, 1120, 54], fill="#0F172A")
    d.text((24, 14), title, fill="#FFFFFF", font=f_t)

    for idx, (vals, name, col) in enumerate([(y1, y1_name, "#1D4ED8"), (y2, y2_name, "#047857")]):
        px0 = 65 + idx * 535
        py0, px1, py1 = 96, px0 + 460, 340
        d.rectangle([px0, py0, px1, py1], fill="#F8FAFC", outline="#475569", width=2)
        d.text((px0 + 10, py0 - 28), name, fill=col, font=f_s)

        vmin, vmax = min(vals), max(vals)
        span = (vmax - vmin) if vmax > vmin else 1.0
        lo = vmin - 0.15 * span
        hi = vmax + 0.20 * span

        n = len(vals)
        xs = [px0 + 55 + int((i / max(1, n - 1)) * (px1 - px0 - 110)) for i in range(n)]
        coords: list[tuple[int, int]] = []
        for x, v in zip(xs, vals):
            y = py1 - 24 - int(((v - lo) / (hi - lo)) * (py1 - py0 - 52))
            coords.append((x, y))

        for a, b in zip(coords, coords[1:]):
            d.line([a, b], fill=col, width=4)
        for lbl, v, (cx, cy) in zip(x_labels, vals, coords):
            d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=col, outline="#FFFFFF", width=2)
            d.text((cx - 26, cy - 26), f"{v}", fill=col, font=f_m)
            d.text((cx - 28, py1 + 10), lbl, fill="#0F172A", font=f_m)

    d.rectangle([24, 384, 1096, 424], fill="#E2E8F0", outline="#94A3B8", width=1)
    d.text((38, 395), footer_note, fill="#0F172A", font=f_m)
    return _png_bytes(img)


# ---------------------------------------------------------------------------
# PDF Builders via headless Chrome (100% clean CJK + table extraction in PDFium)
# ---------------------------------------------------------------------------

_PDF_STYLE = """
@page { size: A4; margin: 18mm; }
body {
  font-family: "PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", sans-serif;
  font-size: 10.8pt;
  line-height: 1.55;
  color: #0F172A;
}
.page { page-break-after: always; }
.page:last-child { page-break-after: auto; }
h1 { font-size: 16pt; color: #0F172A; border-bottom: 2px solid #1E293B; padding-bottom: 6px; margin-bottom: 12px; }
h2 { font-size: 13pt; color: #1E3A8A; margin-top: 14px; margin-bottom: 8px; }
p { margin: 7px 0; }
.label { font-weight: 700; color: #0F172A; }
.box { background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 6px; padding: 10px 12px; margin: 10px 0; }
table {
  border-collapse: separate;
  border-spacing: 0;
  width: 100%;
  margin: 12px 0;
  border-top: 1px solid #334155;
  border-left: 1px solid #334155;
  font-size: 10pt;
}
th, td {
  border-right: 1px solid #334155;
  border-bottom: 1px solid #334155;
  padding: 6px 8px;
  text-align: left;
}
th { background: #E2E8F0; font-weight: 700; }
code { font-family: "Menlo", monospace; font-size: 9.8pt; background: #F1F5F9; padding: 1px 4px; border-radius: 3px; }
"""


def _html_to_pdf(pages_html: list[str], out_pdf: Path) -> None:
    if not CHROME_BIN.exists():
        raise FileNotFoundError(f"Chrome binary not found at {CHROME_BIN}")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"<div class='page'>{p}</div>" for p in pages_html)
    doc_html = (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{_PDF_STYLE}</style></head>"
        f"<body>{body}</body></html>"
    )
    with tempfile.TemporaryDirectory(prefix="h-gate-pdf-") as td:
        hpath = Path(td) / f"{out_pdf.stem}.html"
        hpath.write_text(doc_html, encoding="utf-8")
        subprocess.run(
            [
                str(CHROME_BIN),
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={out_pdf}",
                str(hpath),
            ],
            check=True,
            capture_output=True,
            text=True,
        )


def build_lecture_01_pdf(out_pdf: Path) -> None:
    """Build 5-page Lecture-01-Core-Theory.pdf defining C1, C2, C3 + Trap T4."""
    p1 = """
    <h1>CS-Ctrl-H · Lecture 01: 离散状态反馈、协方差自适应与滑模抗扰理论</h1>
    <p><span class='label'>课程归属：</span>离散控制系统与状态估计 (CS-Ctrl-H 核心理论讲义)</p>
    <h2>1. 核心概念定义：极点配置滞后补偿 (Pole-Placement Lag Compensation)</h2>
    <div class='box'>
      <p><span class='label'>正式定义：</span>极点配置滞后补偿是指在离散线性定常状态空间模型
      <code>x_{k+1} = A x_k + B u_k, y_k = C x_k</code> 中，引入离散积分误差增广状态
      <code>z_{k+1} = z_k + (r_k - y_k)</code>，构造状态反馈律 <code>u_k = -K_x x_k + K_i z_k</code>，
      将闭环特征多项式 <code>det(zI - A_aug + B_aug K_aug)</code> 的全部特征根配置在 z 平面单位圆内满足
      <code>|z_i| &lt; 0.85</code> 的扇区，从而在保证期望瞬态阻尼比的同时消除阶跃与斜坡指令的稳态追踪误差。</p>
      <p><span class='label'>数学与机制成立条件：</span>
      (1) 能控性条件：增广系统矩阵对 <code>(A_aug, B_aug)</code> 必须完全能控，即能控性矩阵满秩 <code>rank(C_aug) = n + 1</code>；
      (2) 采样周期约束：离散采样周期须满足 <code>T_s &lt;= 0.05 s</code>，避免相位滞后侵蚀稳定裕度；
      (3) 极点扇区界：全部闭环特征根模长严格满足 <code>|z_i| &lt; 0.85</code>。</p>
      <p><span class='label'>实验演示对应关系：</span>本节所定义的极点配置滞后补偿机理、不同采样周期与积分增益下的阻尼比裕度及超调表现，
      由课程配套实验文件 <code>Lab-01-Experiments.ipynb</code> 中的实验函数 <code>run_loop_trim_p1()</code> 实测演示。</p>
    </div>
    """

    p2 = """
    <h2>1.2 极点配置滞后补偿 · 设计参数与采样裕度表</h2>
    <p>下表总结极点配置滞后补偿在典型工程参数档位下的配置约束；具体时域阶跃/斜坡指标请核对实验函数 <code>run_loop_trim_p1()</code> 的输出记录。</p>
    <table>
      <tr><th>配置档位</th><th>采样周期 T_s (s)</th><th>积分增益 k_i</th><th>最大极点模长 |z|_max</th><th>稳态斜坡误差目标</th></tr>
      <tr><td>基准配置 (Nominal)</td><td>0.020 s</td><td>1.45</td><td>&lt; 0.85</td><td>0.000</td></tr>
      <tr><td>紧致配置 (Tight)</td><td>0.010 s</td><td>1.95</td><td>&lt; 0.78</td><td>0.000</td></tr>
    </table>
    <p><span class='label'>边界提示：</span>若采样周期超出 <code>T_s &lt;= 0.05 s</code> 限制或 <code>(A_aug, B_aug)</code> 失去能控性，上述闭环配置条件不成立。</p>
    """

    p3 = """
    <h2>2. 核心概念定义：扩展卡尔曼协方差膨胀 (EKF Covariance Inflation)</h2>
    <div class='box'>
      <p><span class='label'>正式定义：</span>扩展卡尔曼协方差膨胀是指在非线性离散观测滤波中，当雅可比矩阵线性化引入不可忽略的高阶截断误差时，
      在时间更新后利用自适应膨胀因子 <code>lambda_k = max(1.0, tr(r_k r_k^T) / tr(H_k P_{k|k-1} H_k^T + R_k))</code>
      对先验预测协方差矩阵执行放大修正 <code>P_{k|k-1} &lt;- lambda_k P_{k|k-1}</code>（其中 <code>r_k = y_k - h(x_{k|k-1})</code> 为量测新息），
      以此防止协方差矩阵过度收缩导致的滤波器发散 (filter divergence)。</p>
      <p><span class='label'>数学与机制成立条件：</span>
      (1) 正定性条件：量测噪声协方差矩阵必须严格正定 (<code>R_k &gt; 0</code>)，先验协方差保持对称半正定；
      (2) 触发门控条件：新息归一化卡方统计量 <code>d_k^2 = r_k^T S_k^{-1} r_k</code> 须连续 3 步超过阈值 <code>chi2_0.95 = 7.81</code> 才激活膨胀。</p>
      <p><span class='label'>实验演示对应关系：</span>扩展卡尔曼协方差膨胀对非线性发散的抑制效果、开启前后 RMSE 对比及卡方拒收率统计，
      由实验文件 <code>Lab-01-Experiments.ipynb</code> 中的实验函数 <code>eval_res_scale_k7()</code> 定量演示。</p>
    </div>
    """

    p4 = """
    <h2>3. 核心概念定义：预测时域滑模抗扰 (Horizon Sliding-Mode Disturbance Rejection)</h2>
    <div class='box'>
      <p><span class='label'>正式定义：</span>预测时域滑模抗扰将有限时域滚动优化 (MPC) 与离散指数趋近律
      <code>s_{k+1} = (1 - q T_s) s_k - eps * T_s * sat(s_k / phi)</code> 相结合，
      通过在饱和函数 <code>sat(s_k / phi)</code> 中引入厚度为 <code>phi &gt; 0</code> 的边界层，将高频控制开关抖振转化为有界稳态带，同时抑制有界外部匹配扰动。</p>
      <p><span class='label'>数学与机制成立条件：</span>
      (1) 扰动有界条件：外部未知扰动满足无穷范数上界 <code>||d_k||_inf &lt;= d_max</code>；
      (2) 边界层抑制条件：边界层厚度必须满足不等式 <code>phi &gt;= 2 * eps * T_s</code>，否则无法消除离散步长内的穿越抖振。</p>
      <p><span class='label'>实验演示对应关系：</span>不同边界层宽度下开关抖振幅值、稳态误差带与开关频率的定量实测，
      由实验文件 <code>Lab-01-Experiments.ipynb</code> 中的实验函数 <code>probe_band_law_m4()</code> 演示。</p>
    </div>
    """

    p5 = """
    <h2>4. 附注：硬件量化舍入噪声与字长预算（通用词汇对照区）</h2>
    <p>在嵌入式数字控制器实现阶段，模数转换器 (ADC) 与数模转换器 (DAC) 的有限字长会引入均匀分布的量化舍入噪声，
    其方差为 <code>Delta^2 / 12</code>。工程上需在固定的采样周期 <code>T_s</code> 下评估量化协方差对稳态误差地板的贡献，
    并检验极点靠近单位圆边界时的系数截断灵敏度。本附注仅讨论通用硬件字长预算，不构成前述控制算法的定义或实验演示。</p>
    """

    _html_to_pdf([p1, p2, p3, p4, p5], out_pdf)


def build_no_edge_notes_pdf(out_pdf: Path) -> None:
    """Build 2-page NoEdge-Notes.pdf defining C6 with ZERO course experiments."""
    p1 = """
    <h1>CS-Ctrl-H · 专题讲义：H∞ 混合灵敏度环路成形 (无实验对照概念)</h1>
    <h2>1. 核心概念定义：H∞ 混合灵敏度环路成形 (H-infinity Mixed-Sensitivity Loop Shaping)</h2>
    <div class='box'>
      <p><span class='label'>正式定义：</span>H∞ 混合灵敏度环路成形是在频域内构造广义加权灵敏度传递矩阵
      <code>M(s) = [W_1(s) S(s); W_2(s) K(s) S(s); W_3(s) T(s)]</code>，
      其中灵敏度函数 <code>S = (I + GK)^{-1}</code> 刻画扰动抑制与指令追踪性能，补灵敏度函数 <code>T = GK(I + GK)^{-1}</code>
      刻画乘性模型不确定性鲁棒性，通过求解次优 H∞ 综合问题使闭环无穷范数满足 <code>||M(s)||_inf &lt; gamma</code>。</p>
      <p><span class='label'>数学与频域约束条件：</span>
      (1) 低频性能权函数 <code>W_1(s)</code> 须在低频段提供足够大增益以压低追踪误差；
      (2) 高频鲁棒权函数 <code>W_3(s)</code> 在高频段须具备至少 <code>20 dB/dec</code> 的滚降斜率以抑制未建模高频动态；
      (3) 内部稳定性：控制器 <code>K(s)</code> 必须内部镇定标称广义被控对象。</p>
      <p><span class='label'>实验安排说明：</span>本概念在本门课程 (<code>cs-ctrl-h</code>) 中仅作理论推导与频域权函数说明，全课未安排任何配套仿真代码或 Notebook 实验演示。</p>
    </div>
    """

    p2 = """
    <h2>2. 频域权函数设计折中参考表</h2>
    <table>
      <tr><th>权函数</th><th>目标频段约束 (rad/s)</th><th>幅值约束 (dB)</th><th>物理含义</th></tr>
      <tr><td>W_1(s)</td><td>omega &lt; 5.0 rad/s</td><td>|S(j*omega)| &lt;= -26 dB</td><td>低频稳态精度与低频扰动抑制</td></tr>
      <tr><td>W_2(s)</td><td>全频段</td><td>|K(j*omega)S(j*omega)| &lt;= 14 dB</td><td>限制控制能量与执行器带宽</td></tr>
      <tr><td>W_3(s)</td><td>omega &gt; 80.0 rad/s</td><td>滚降 &gt;= 20 dB/dec</td><td>抗高频未建模模态与传感器噪声</td></tr>
    </table>
    """

    _html_to_pdf([p1, p2], out_pdf)


# ---------------------------------------------------------------------------
# PPTX Builder (Slides-02-Method-Walkthrough.pptx)
# ---------------------------------------------------------------------------

def _add_slide_text(prs: Presentation, title: str, bullets: list[str]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    tbox = slide.shapes.add_textbox(PptxInches(0.6), PptxInches(0.4), PptxInches(12.1), PptxInches(0.95))
    tf = tbox.text_frame
    tf.word_wrap = True
    tf.text = title
    tf.paragraphs[0].font.size = PptxPt(24)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    bbox = slide.shapes.add_textbox(PptxInches(0.6), PptxInches(1.45), PptxInches(12.1), PptxInches(5.5))
    btf = bbox.text_frame
    btf.word_wrap = True
    for idx, line in enumerate(bullets):
        p = btf.paragraphs[0] if idx == 0 else btf.add_paragraph()
        p.text = line
        p.font.size = PptxPt(16)
        p.space_after = PptxPt(10)


def _add_slide_table(prs: Presentation, title: str, subtitle: str, headers: list[str], rows: list[list[str]]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    tbox = slide.shapes.add_textbox(PptxInches(0.6), PptxInches(0.35), PptxInches(12.1), PptxInches(0.8))
    tf = tbox.text_frame
    tf.text = title
    tf.paragraphs[0].font.size = PptxPt(24)
    tf.paragraphs[0].font.bold = True

    sbox = slide.shapes.add_textbox(PptxInches(0.6), PptxInches(1.15), PptxInches(12.1), PptxInches(0.55))
    stf = sbox.text_frame
    stf.word_wrap = True
    stf.text = subtitle
    stf.paragraphs[0].font.size = PptxPt(14)

    n_rows = len(rows) + 1
    n_cols = len(headers)
    table_shape = slide.shapes.add_table(
        n_rows, n_cols, PptxInches(0.6), PptxInches(1.85), PptxInches(12.1), PptxInches(4.3)
    )
    table = table_shape.table
    for ci, h in enumerate(headers):
        cell = table.cell(0, ci)
        cell.text = h
        cell.text_frame.paragraphs[0].font.bold = True
        cell.text_frame.paragraphs[0].font.size = PptxPt(14)
    for ri, r in enumerate(rows, start=1):
        for ci, v in enumerate(r):
            cell = table.cell(ri, ci)
            cell.text = v
            cell.text_frame.paragraphs[0].font.size = PptxPt(14)


def build_slides_02_pptx(out_pptx: Path) -> None:
    """Build 7-slide Slides-02-Method-Walkthrough.pptx with C4, C5, unit tables, and mention traps."""
    prs = Presentation()
    prs.slide_width = PptxInches(13.333)
    prs.slide_height = PptxInches(7.5)

    # Slide 1: Cover
    _add_slide_text(
        prs,
        "CS-Ctrl-H · Lecture 02: 数字陷波滤波与抗饱和积分控制",
        [
            "课程单元：数字滤波工程实现、执行器抗饱和策略与延伸阅读导引",
            "本课件包含两个新增核心概念定义、带单位设计/对比表格以及参考文献导读。",
        ],
    )

    # Slide 2: C4 Definition
    _add_slide_text(
        prs,
        "概念定义：双二阶自适应陷波滤波 (Biquad Adaptive Notch Filtering)",
        [
            "正式定义：采用二阶 IIR 陷波传递函数 H(z) = (1 - 2 cos(w0) z^-1 + z^-2) / (1 - 2 r cos(w0) z^-1 + r^2 z^-2) 精准衰减窄带谐振干扰。",
            "机制条件：(1) 极点半径满足 r = 1 - (pi * B_-3dB) / f_s 且 r in (0.85, 0.995)；(2) 品质因数 Q = f0 / B_-3dB；(3) 中心数字频率 w0 = 2 * pi * f0 / f_s。",
            "实验演示对应关系：不同品质因数与 -3dB 带宽下的实际幅频陷波衰减及通带纹波，由 Lab-01-Experiments.ipynb 中的 verify_iir_dip_b2() 实测演示。",
        ],
    )

    # Slide 3: C4 Parameter Table with Units
    _add_slide_table(
        prs,
        "双二阶自适应陷波滤波 · 带单位设计基准表",
        "依据极点半径公式 r = 1 - (pi * B_-3dB) / f_s 计算得到的两组标准工程档位：",
        ["配置档位", "采样频率 f_s (Hz)", "中心频率 f0 (Hz)", "-3dB 带宽 (Hz)", "极点半径 r", "理论品质因数 Q"],
        [
            ["Profile-A (对应 verify_iir_dip_b2)", "1000.0 Hz", "120.0 Hz", "15.0 Hz", "0.9529", "8.00"],
            ["Profile-B (低频窄带)", "1000.0 Hz", "60.0 Hz", "10.0 Hz", "0.9686", "6.00"],
        ],
    )

    # Slide 4: C5 Definition
    _add_slide_text(
        prs,
        "概念定义：条件积分抗饱和钳位 (Conditional Anti-Windup Clamping)",
        [
            "正式定义：当控制器输出触及物理执行器饱和限幅 |u_cmd| > u_max 且追踪误差与饱和输出同号 (e_k * u_sat > 0) 时，冻结积分器状态 z_{k+1} = z_k。",
            "恢复条件：仅当指令退出饱和区 (|u_cmd| <= u_max) 或误差与饱和输出反号 (e_k * u_sat <= 0) 时恢复积分累加，防止积分风卷引起大超调。",
            "说明：本概念的性能结论通过下一页带单位定量对比表给出，课程 Notebook 中未单独安排代码演示。",
        ],
    )

    # Slide 5: C5 Unit Comparison Table
    _add_slide_table(
        prs,
        "条件积分抗饱和钳位 · 策略定量对比表 (含物理单位)",
        "固定阶跃给定与执行器限幅条件下的三种抗饱和策略仿真对比：",
        ["抗饱和策略", "采样周期 (ms)", "饱和限幅 (V)", "峰值超调 (%)", "退饱和恢复时间 (ms)", "稳态误差 (deg)"],
        [
            ["无抗饱和 (No-AW)", "20.0 ms", "±12.0 V", "28.6 %", "640.0 ms", "0.05 deg"],
            ["反计算追踪 (Back-Calc K_aw=4.0)", "20.0 ms", "±12.0 V", "9.4 %", "265.0 ms", "0.02 deg"],
            ["条件积分钳位 (Conditional Clamp)", "20.0 ms", "±12.0 V", "4.8 %", "190.0 ms", "0.01 deg"],
        ],
    )

    # Slide 6: Trap T1 — Reference & Ordinary Mention Trap
    _add_slide_text(
        prs,
        "参考文献与延伸阅读（普通提及段落 · 非定义与非实验演示）",
        [
            "[Ref 1] Åström & Wittenmark (1997), Ch. 8：仅作背景文献提及 Smith 预估器 (Smith Predictor) 与零相位前馈预整形 (Zero-Phase Feedforward Shaping)，本课未定义亦无对应实验。",
            "[Ref 2] Bar-Shalom et al. (2001)：文献背景对比注记——扩展卡尔曼协方差膨胀 (EKF Covariance Inflation) 常与无迹卡尔曼滤波并列讨论，正式定义见 Lecture-01。",
            "[Ref 3] Khalil (2002) & Zhou (1996)：延伸阅读提及极点配置滞后补偿 (Pole-Placement Lag Compensation) 与 H∞ 混合灵敏度环路成形的频域/时域关联。",
            "提示：本页仅汇总课程参考书目与术语背景提及，不提供任何概念的正式定义或实验代码验证。",
        ],
    )

    # Slide 7: Trap T4 — Shared generic terminology without concept edge
    _add_slide_text(
        prs,
        "硬件接口备忘：采样时钟抖动与传感器标定常识",
        [
            "在多通道传感器同步采集时，需检查采样周期 T_s 的时钟漂移与量测协方差标定矩阵的对称性。",
            "若离散极点靠近单位圆，建议预留足够定点数位宽以降低稳态误差。",
            "（通用硬件工程备忘，不涉及具体控制律定义或实验验证。）",
        ],
    )

    out_pptx.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_pptx))


# ---------------------------------------------------------------------------
# Notebook Builders (v1, v2, and cs-other-h distractor)
# ---------------------------------------------------------------------------

def _build_demo_cells_common() -> dict[str, nbf.NotebookNode]:
    """Return reusable code cells with ZERO Chinese/English theoretical concept overlap."""
    c1_v1_code = "\n".join(
        [
            "def run_loop_trim_p1(ts_sec=0.02, ki_gain=1.45):",
            "    # Empirical step/ramp bench v1 (isolated identifier)",
            "    zeta_ratio = 0.72",
            "    overshoot_pct = 4.15",
            "    settling_ms = 185.0",
            "    ramp_steady_err = 0.000",
            "    print(f'[sweep_v1] ts={ts_sec:.3f} s, ki={ki_gain:.2f} | zeta_ratio={zeta_ratio:.2f}, overshoot_pct={overshoot_pct:.2f}%, settling_ms={settling_ms:.1f}, ramp_steady_err={ramp_steady_err:.3f}')",
            "    return zeta_ratio, overshoot_pct, settling_ms",
            "",
            "run_loop_trim_p1()",
        ]
    )
    c1_v1_out = "[sweep_v1] ts=0.020 s, ki=1.45 | zeta_ratio=0.72, overshoot_pct=4.15%, settling_ms=185.0, ramp_steady_err=0.000\n"
    c1_v1_png = _make_dual_metric_chart(
        "Empirical Envelope: run_loop_trim_p1 (v1)",
        ["ki=0.9", "ki=1.2", "ki=1.45"],
        [0.58, 0.65, 0.72],
        "Metric A: zeta_ratio",
        [11.2, 6.8, 4.15],
        "Metric B: overshoot_pct (%)",
        "v1 Run: ts=0.020 s, ki=1.45 => zeta_ratio=0.72, overshoot_pct=4.15%, settling_ms=185.0",
    )

    c1_v2_code = "\n".join(
        [
            "def run_loop_trim_p1(ts_sec=0.01, ki_gain=1.95):",
            "    # Empirical step/ramp bench v2 (tightened sampling & gain)",
            "    zeta_ratio = 0.84",
            "    overshoot_pct = 1.42",
            "    settling_ms = 132.0",
            "    ramp_steady_err = 0.000",
            "    print(f'[sweep_v2] ts={ts_sec:.3f} s, ki={ki_gain:.2f} | zeta_ratio={zeta_ratio:.2f}, overshoot_pct={overshoot_pct:.2f}%, settling_ms={settling_ms:.1f}, ramp_steady_err={ramp_steady_err:.3f}')",
            "    return zeta_ratio, overshoot_pct, settling_ms",
            "",
            "run_loop_trim_p1()",
        ]
    )
    c1_v2_out = "[sweep_v2] ts=0.010 s, ki=1.95 | zeta_ratio=0.84, overshoot_pct=1.42%, settling_ms=132.0, ramp_steady_err=0.000\n"
    c1_v2_png = _make_dual_metric_chart(
        "Empirical Envelope: run_loop_trim_p1 (v2)",
        ["ki=1.2", "ki=1.6", "ki=1.95"],
        [0.69, 0.77, 0.84],
        "Metric A: zeta_ratio",
        [5.9, 2.85, 1.42],
        "Metric B: overshoot_pct (%)",
        "v2 Run: ts=0.010 s, ki=1.95 => zeta_ratio=0.84, overshoot_pct=1.42%, settling_ms=132.0",
    )

    c2_v1_code = "\n".join(
        [
            "def eval_res_scale_k7(chi2_gate=7.81):",
            "    # Empirical residual trace bench k7 (v1 only)",
            "    rmse_unscaled = 0.418",
            "    rmse_scaled = 0.064",
            "    chi2_reject_rate = 2.30",
            "    print(f'[k7_v1] chi2_gate={chi2_gate:.2f} | rmse_unscaled={rmse_unscaled:.3f} rad, rmse_scaled={rmse_scaled:.3f} rad, chi2_reject_rate={chi2_reject_rate:.2f}%')",
            "    return rmse_unscaled, rmse_scaled, chi2_reject_rate",
            "",
            "eval_res_scale_k7()",
        ]
    )
    c2_v1_out = "[k7_v1] chi2_gate=7.81 | rmse_unscaled=0.418 rad, rmse_scaled=0.064 rad, chi2_reject_rate=2.30%\n"
    c2_v1_png = _make_dual_metric_chart(
        "Residual Trace: eval_res_scale_k7 (v1)",
        ["t=20", "t=60", "t=100"],
        [0.145, 0.289, 0.418],
        "Unscaled RMSE (rad)",
        [0.082, 0.071, 0.064],
        "Scaled RMSE (rad)",
        "v1 k7: chi2_gate=7.81 => rmse_unscaled=0.418 rad, rmse_scaled=0.064 rad, reject=2.30%",
    )

    c3_code = "\n".join(
        [
            "def probe_band_law_m4(phi_band=0.12, q_rate=12.0, eps=2.5):",
            "    # Empirical boundary strip trace m4",
            "    osc_amp = 0.018",
            "    steady_band = 0.024",
            "    switch_hz = 14.2",
            "    print(f'[m4_trace] phi={phi_band:.2f}, d_max=0.35 | osc_amp={osc_amp:.3f} Nm, steady_band={steady_band:.3f} rad, switch_hz={switch_hz:.1f} Hz')",
            "    return osc_amp, steady_band, switch_hz",
            "",
            "probe_band_law_m4()",
        ]
    )
    c3_out = "[m4_trace] phi=0.12, d_max=0.35 | osc_amp=0.018 Nm, steady_band=0.024 rad, switch_hz=14.2 Hz\n"
    c3_png = _make_dual_metric_chart(
        "Band Strip Profile: probe_band_law_m4",
        ["phi=0.02", "phi=0.06", "phi=0.12"],
        [0.095, 0.044, 0.018],
        "Osc Amp (Nm)",
        [48.5, 26.0, 14.2],
        "Switch Rate (Hz)",
        "m4 Result: phi=0.12 => osc_amp=0.018 Nm, steady_band=0.024 rad, switch_hz=14.2 Hz",
    )

    c4_code = "\n".join(
        [
            "def verify_iir_dip_b2(fs_hz=1000.0, f0_hz=120.0, bw_hz=15.0):",
            "    # Empirical spectrum dip bench b2",
            "    r_coeff = 0.9529",
            "    q_factor = 8.00",
            "    attenuation_db = -31.4",
            "    passband_ripple_db = 0.18",
            "    print(f'[b2_dip] f0={f0_hz:.1f} Hz, bw={bw_hz:.1f} Hz, r_coeff={r_coeff:.4f} | q_factor={q_factor:.2f}, dip_attenuation_db={attenuation_db:.1f} dB, passband_ripple_db={passband_ripple_db:.2f} dB')",
            "    return q_factor, attenuation_db, passband_ripple_db",
            "",
            "verify_iir_dip_b2()",
        ]
    )
    c4_out = "[b2_dip] f0=120.0 Hz, bw=15.0 Hz, r_coeff=0.9529 | q_factor=8.00, dip_attenuation_db=-31.4 dB, passband_ripple_db=0.18 dB\n"
    c4_png = _make_dual_metric_chart(
        "Spectrum Attenuation: verify_iir_dip_b2",
        ["80 Hz", "120 Hz", "160 Hz"],
        [-0.22, -31.4, -0.25],
        "Magnitude (dB)",
        [0.14, 0.18, 0.15],
        "Passband Ripple (dB)",
        "b2 Result: f0=120.0 Hz, r_coeff=0.9529 => q_factor=8.00, dip=-31.4 dB, ripple=0.18 dB",
    )

    trap_mention_code = "\n".join(
        [
            "# WARNING (仅术语提及，非概念实验): Do NOT treat this timer check as",
            "# 扩展卡尔曼协方差膨胀 (EKF Covariance Inflation), 双二阶自适应陷波滤波, or 极点配置滞后补偿.",
            "# It only benchmarks numpy.fft wall-clock overhead.",
            "def measure_fft_wallclock_overhead(n_pts=4096):",
            "    mean_us = 68.4",
            "    p99_us = 92.1",
            "    print(f'[wallclock] n_pts={n_pts}, mean_us={mean_us:.1f} us, p99_us={p99_us:.1f} us')",
            "    return mean_us, p99_us",
            "",
            "measure_fft_wallclock_overhead()",
        ]
    )
    trap_mention_out = "[wallclock] n_pts=4096, mean_us=68.4 us, p99_us=92.1 us\n"

    trap_unrun_code = "\n".join(
        [
            "# TODO (未来计划 / 尚未运行，无保存输出): planned ablation for 极点配置滞后补偿 and Smith 预估器",
            "def planned_lag_comp_ablation():",
            "    raise NotImplementedError('Planned future benchmark - not yet executed')",
        ]
    )

    return {
        "c1_v1": nbf.v4.new_code_cell(c1_v1_code, execution_count=1, outputs=[_out_stream(c1_v1_out), _out_png(c1_v1_png)]),
        "c1_v2": nbf.v4.new_code_cell(c1_v2_code, execution_count=1, outputs=[_out_stream(c1_v2_out), _out_png(c1_v2_png)]),
        "c2_v1": nbf.v4.new_code_cell(c2_v1_code, execution_count=2, outputs=[_out_stream(c2_v1_out), _out_png(c2_v1_png)]),
        "c3": nbf.v4.new_code_cell(c3_code, execution_count=3, outputs=[_out_stream(c3_out), _out_png(c3_png)]),
        "c4": nbf.v4.new_code_cell(c4_code, execution_count=4, outputs=[_out_stream(c4_out), _out_png(c4_png)]),
        "trap_mention": nbf.v4.new_code_cell(trap_mention_code, execution_count=5, outputs=[_out_stream(trap_mention_out)]),
        "trap_unrun": nbf.v4.new_code_cell(trap_unrun_code, execution_count=None, outputs=[]),
    }


def build_lab_v1(out_ipynb: Path) -> None:
    """Build v1 Lab-01-Experiments.ipynb (9 cells)."""
    pool = _build_demo_cells_common()
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell("# Lab 01 · Discrete Dynamics & Signal Bench (v1)\n\nBench scripts and saved telemetry traces (`p1`, `k7`, `m4`, `b2`)."),
        pool["c1_v1"],  # cell:1 -> C1 demo
        nbf.v4.new_markdown_cell("## Bench 2 · Residual Scaling Run (`k7`)"),
        pool["c2_v1"],  # cell:3 -> C2 demo (removed in v2)
        nbf.v4.new_markdown_cell("## Bench 3 & 4 · Band Strip (`m4`) & Spectrum Dip (`b2`)"),
        pool["c3"],     # cell:5 -> C3 demo (shifts to cell:3 in v2)
        pool["c4"],     # cell:6 -> C4 demo (shifts to cell:4 in v2)
        pool["trap_mention"],  # cell:7 -> Trap T2 mention-only cell
        pool["trap_unrun"],    # cell:8 -> Trap T3 unrun future plan cell
    ]
    out_ipynb.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out_ipynb))


def build_lab_v2(out_ipynb: Path) -> None:
    """Build v2 Lab-01-Experiments-v2.ipynb (7 cells: modifies C1, removes C2, shifts C3/C4)."""
    pool = _build_demo_cells_common()
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell("# Lab 01 · Discrete Dynamics & Signal Bench (v2)\n\nUpdated telemetry: `p1` tightened to `ts=0.010 s`; legacy `k7` bench removed."),
        pool["c1_v2"],  # cell:1 -> C1 updated v2 demo
        nbf.v4.new_markdown_cell("## Bench 2 & 3 · Band Strip (`m4`) & Spectrum Dip (`b2`)"),
        pool["c3"],     # cell:3 -> C3 demo (shifted from cell:5!)
        pool["c4"],     # cell:4 -> C4 demo (shifted from cell:6!)
        pool["trap_mention"],  # cell:5 -> Trap T2
        pool["trap_unrun"],    # cell:6 -> Trap T3
    ]
    out_ipynb.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out_ipynb))


def build_other_course_distractor(out_ipynb: Path) -> None:
    """Build OtherCourse-Distractor.ipynb for cs-other-h with same concept names & distinct metrics."""
    nb = nbf.v4.new_notebook()
    c1_other = "\n".join(
        [
            "# 另一门课程 cs-other-h 同名概念：极点配置滞后补偿 (Distractor)",
            "def run_loop_trim_p1():",
            "    print('[other_course_p1] zeta_ratio=0.51, overshoot_pct=16.80%, settling_ms=420.0')",
        ]
    )
    c2_other = "\n".join(
        [
            "# 另一门课程 cs-other-h 同名概念：扩展卡尔曼协方差膨胀 (Distractor)",
            "def eval_res_scale_k7():",
            "    print('[other_course_k7] rmse_unscaled=0.890 rad, rmse_scaled=0.295 rad, chi2_reject_rate=11.40%')",
        ]
    )
    c6_other = "\n".join(
        [
            "# 另一门课程 cs-other-h 独有实验：H∞ 混合灵敏度环路成形",
            "def run_hinf_other_x9():",
            "    print('[other_hinf_x9] gamma_achieved=1.18, phase_margin_deg=54.2, bw_rad_s=18.5')",
        ]
    )
    c6_png = _make_dual_metric_chart(
        "OtherCourse H-inf Loop Shaping: run_hinf_other_x9",
        ["w=5", "w=20", "w=80"],
        [-24.0, -11.5, -1.2],
        "Sensitivity S (dB)",
        [-0.3, -4.5, -22.8],
        "Comp Sensitivity T (dB)",
        "cs-other-h ONLY: gamma_achieved=1.18, phase_margin_deg=54.2, bw_rad_s=18.5",
    )
    nb["cells"] = [
        nbf.v4.new_markdown_cell("# CS-Other-H · Cross-Course Isolation Distractor"),
        nbf.v4.new_code_cell(
            c1_other,
            execution_count=1,
            outputs=[_out_stream("[other_course_p1] zeta_ratio=0.51, overshoot_pct=16.80%, settling_ms=420.0\n")],
        ),
        nbf.v4.new_code_cell(
            c2_other,
            execution_count=2,
            outputs=[_out_stream("[other_course_k7] rmse_unscaled=0.890 rad, rmse_scaled=0.295 rad, chi2_reject_rate=11.40%\n")],
        ),
        nbf.v4.new_code_cell(
            c6_other,
            execution_count=3,
            outputs=[
                _out_stream("[other_hinf_x9] gamma_achieved=1.18, phase_margin_deg=54.2, bw_rad_s=18.5\n"),
                _out_png(c6_png),
            ],
        ),
    ]
    out_ipynb.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out_ipynb))


# ---------------------------------------------------------------------------
# Verification Helpers
# ---------------------------------------------------------------------------

def verify_zero_overlap_demo_cells(lab_path: Path, demo_cell_indices: list[int]) -> None:
    """Verify real experiment cells in Lab-01 share ZERO concept-theory tokens with Lecture/Slides."""
    nb = json.loads(lab_path.read_text(encoding="utf-8"))
    parts: list[str] = []
    for idx in demo_cell_indices:
        cell = nb["cells"][idx]
        parts.append("".join(cell.get("source", "")))
        for out in cell.get("outputs", []):
            if "text" in out:
                parts.append("".join(out["text"]))
    blob = "\n".join(parts).lower()
    forbidden = [
        "极点", "配置", "滞后", "补偿", "pole", "placement", "lag", "compensation",
        "能控", "单位圆", "controllab",
        "卡尔曼", "协方差", "膨胀", "kalman", "ekf", "covariance", "inflation", "新息", "innovation", "卡方",
        "预测时域", "滑模", "抗扰", "horizon", "sliding", "disturbance", "chatter", "抖振", "边界层",
        "双二阶", "自适应", "陷波", "biquad", "adaptive", "notch",
    ]
    hits = [tok for tok in forbidden if tok in blob]
    if hits:
        raise AssertionError(f"Zero-overlap check FAILED on {lab_path.name} cells {demo_cell_indices}: {hits}")

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE demo USING fts5(body, tokenize='unicode61')")
    conn.execute("INSERT INTO demo VALUES (?)", (blob,))
    for tok in ["pole", "lag", "kalman", "covariance", "inflation", "sliding", "notch", "biquad", "adaptive"]:
        cnt = conn.execute("SELECT count(*) FROM demo WHERE demo MATCH ?", (tok,)).fetchone()[0]
        if cnt != 0:
            raise AssertionError(f"FTS5 zero-overlap leaked {tok} in {lab_path.name}")


def verify_soffice_pptx(pptx_path: Path) -> int:
    if not SOFFICE_BIN.exists():
        raise FileNotFoundError(f"soffice not found at {SOFFICE_BIN}")
    with tempfile.TemporaryDirectory(prefix="h-gate-pptx-") as td:
        subprocess.run(
            [str(SOFFICE_BIN), "--headless", "--convert-to", "pdf", "--outdir", td, str(pptx_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        pdf_path = Path(td) / f"{pptx_path.stem}.pdf"
        doc = pypdfium2.PdfDocument(str(pdf_path))
        return len(doc)


def inspect_corpus(corpus_dir: Path, verify_against_criteria: bool = False) -> list[dict]:
    """Parse every frozen file with SRAG's real parsers and collect byte/hash/structural metadata."""
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
                raise AssertionError(f"SHA-256 mismatch for {path.name}: {sha} != {expected_hashes[path.name]}")

        info: dict = {
            "file_name": path.name,
            "bytes": len(raw),
            "sha256": sha,
            "format": path.suffix.lower().lstrip("."),
        }
        if path.suffix.lower() in {".pdf", ".pptx"}:
            parsed = parse_local_slides(path)
            info["parsed_title"] = parsed.title
            info["parsed_sources"] = len(parsed.sources)
            info["parsed_native_chars"] = sum(len(s.text) for s in parsed.sources)
            info["source_ids_sample"] = [s.source_id for s in parsed.sources[:12]]
            if path.suffix.lower() == ".pdf":
                doc = pypdfium2.PdfDocument(str(path))
                info["pages"] = len(doc)
            else:
                prs = Presentation(str(path))
                soffice_pages = verify_soffice_pptx(path)
                assert soffice_pages == len(prs.slides), f"{path.name}: pptx={len(prs.slides)} vs soffice={soffice_pages}"
                info["pages"] = soffice_pages
                info["soffice_verified"] = True
                info["table_sources"] = [
                    s.source_id for s in parsed.sources if s.locator.get("object_type") == "table"
                ]
        else:
            parsed = parse_local_notebook(path)
            nb = json.loads(raw.decode("utf-8"))
            info["cells"] = len(nb["cells"])
            info["parsed_title"] = parsed.title
            info["parsed_sources"] = len(parsed.sources)
            info["parsed_native_chars"] = sum(len(s.text) for s in parsed.sources)
            info["source_ids"] = [s.source_id for s in parsed.sources]
        manifest.append(info)

    verify_zero_overlap_demo_cells(corpus_dir / "Lab-01-Experiments.ipynb", [1, 3, 5, 6])
    verify_zero_overlap_demo_cells(corpus_dir / "Lab-01-Experiments-v2.ipynb", [1, 3, 4])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or verify cs-ctrl-h frozen corpus.")
    parser.add_argument("--verify-only", action="store_true", help="Verify frozen corpus hashes and structure without rebuilding.")
    parser.add_argument("--rebuild-all", action="store_true", help="Rebuild all files including PDFs (PDF timestamps will change SHA-256).")
    args = parser.parse_args()

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    lec = CORPUS_DIR / "Lecture-01-Core-Theory.pdf"
    noedge = CORPUS_DIR / "NoEdge-Notes.pdf"
    slides = CORPUS_DIR / "Slides-02-Method-Walkthrough.pptx"
    lab_v1 = CORPUS_DIR / "Lab-01-Experiments.ipynb"
    lab_v2 = CORPUS_DIR / "Lab-01-Experiments-v2.ipynb"
    other = CORPUS_DIR / "OtherCourse-Distractor.ipynb"

    if not args.verify_only:
        if args.rebuild_all or not lec.exists():
            build_lecture_01_pdf(lec)
        if args.rebuild_all or not noedge.exists():
            build_no_edge_notes_pdf(noedge)
        build_slides_02_pptx(slides)
        build_lab_v1(lab_v1)
        build_lab_v2(lab_v2)
        build_other_course_distractor(other)

    manifest = inspect_corpus(CORPUS_DIR, verify_against_criteria=args.verify_only)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("ALL_H_CORPUS_CHECKS_PASSED")


if __name__ == "__main__":
    main()
