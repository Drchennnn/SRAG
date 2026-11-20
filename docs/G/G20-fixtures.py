"""Frozen G20 corpus builder and ground-truth evidence annotations (Step A).

Materials:
1. Mixtral-of-Experts-2401.04088v1.pdf (arXiv:2401.04088v1, CC BY 4.0)
2. Direct-Preference-Optimization-2305.18290v3.pdf (arXiv:2305.18290v3, CC BY 4.0)
3. Mixtral-Router-Verification.ipynb (v1 and v2, CC BY 4.0 controlled versioned Notebook)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

COURSE_ID = "g20-moe-dpo-course"
MIXTRAL_PDF = "Mixtral-of-Experts-2401.04088v1.pdf"
DPO_PDF = "Direct-Preference-Optimization-2305.18290v3.pdf"
NOTEBOOK = "Mixtral-Router-Verification.ipynb"

EXPECTED_PDF_SHA256 = {
    MIXTRAL_PDF: "f8bbf0e9d979b7a8ce7be65119266545a229a85b57e077d8bd048e458bb642da",
    DPO_PDF: "92cb3a2b71362acda98a789b03d88688fd33cf5fcf13f81d2b1de30ee7d3b67a",
}

NOTEBOOK_SPECS: dict[int, dict[str, Any]] = {
    1: {
        "evaluated_tokens": 32768,
        "layer15_first_choice": "27.9%",
        "layer15_first_or_second": "62.3%",
        "dispatch_latency_ms": "0.42",
        "capacity_factor": "1.25",
        "plots": (
            ("Expert Load Balance (v1)", "Expert ID [index]", "#2b5c8f"),
            ("Dispatch Latency Trace (v1)", "Batch Size [tokens]", "#8f3d2b"),
        ),
    },
    2: {
        "evaluated_tokens": 65536,
        "layer15_first_choice": "28.4%",
        "layer15_first_or_second": "63.1%",
        "dispatch_latency_ms": "0.31",
        "capacity_factor": "1.50",
        "plots": (
            ("Dispatch Latency Trace (v2)", "Sequence Length [k-tokens]", "#1f7a4c"),
            ("Expert Load Balance (v2)", "Router Layer [layer_idx]", "#6b2b8f"),
        ),
    },
}


def _render_plot(title: str, x_label: str, color: str) -> bytes:
    image = Image.new("RGB", (840, 480), "#fbfaf5")
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 16, 823, 463), outline="#1d2d35", width=4)
    draw.text(
        (420, 48),
        title,
        fill="#101819",
        font=ImageFont.load_default(size=30),
        anchor="mm",
    )
    for x in (180, 300, 420, 540, 660):
        draw.line((x, 96, x, 336), fill="#d5d8ce", width=2)
    draw.line((96, 86, 96, 338, 748, 338), fill="#1d2d35", width=6)
    draw.line(
        (106, 292, 214, 248, 336, 198, 486, 216, 624, 134, 734, 118),
        fill=color,
        width=11,
    )
    draw.text(
        (420, 404),
        x_label,
        fill="#101819",
        font=ImageFont.load_default(size=48),
        anchor="mm",
    )
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def build_notebook_bytes(version: int) -> tuple[bytes, list[dict[str, str]]]:
    spec = NOTEBOOK_SPECS[version]
    stdout_lines = [
        f"Experiment: Mixtral-8x7B-ArXiv-Routing (v{version})\n",
        "Router config: num_experts=8, top_k_experts=2, dispatch_kernel=Megablocks-EP\n",
        f"Evaluated tokens: {spec['evaluated_tokens']} (ArXiv validation subset)\n",
        f"Layer 15 first-choice consecutive repetition: {spec['layer15_first_choice']} (random baseline: 12.5%)\n",
        f"Layer 15 first-or-second consecutive repetition: {spec['layer15_first_or_second']} (random baseline: 46.4%)\n",
        f"Mean routing dispatch latency: {spec['dispatch_latency_ms']} ms / step (capacity_factor={spec['capacity_factor']})\n",
        "Note: 64-GPU all-to-all communication latency under 128k context was not measured in this run.\n",
    ]
    outputs: list[dict[str, Any]] = [
        {
            "output_type": "stream",
            "name": "stdout",
            "text": stdout_lines,
        }
    ]
    image_meta: list[dict[str, str]] = []
    for idx, (title, x_label, color) in enumerate(spec["plots"], start=1):
        png_bytes = _render_plot(title, x_label, color)
        image_meta.append(
            {
                "source_id": f"cell:0:output:{idx}",
                "visible_image_order": idx,
                "title": title,
                "x_axis_label": x_label,
                "png_sha256": hashlib.sha256(png_bytes).hexdigest(),
            }
        )
        outputs.append(
            {
                "output_type": "display_data",
                "metadata": {},
                "data": {
                    "image/png": base64.b64encode(png_bytes).decode("ascii"),
                },
            }
        )
    payload = {
        "cells": [
            {
                "cell_type": "code",
                "id": "router-eval-run",
                "metadata": {},
                "execution_count": 1,
                "source": [
                    "# Mixtral 8x7B SMoE Router & Megablocks-EP Verification Benchmark\n",
                    "# Evaluates ArXiv validation subset consecutive expert repetition and dispatch latency.\n",
                    f"run_router_verification(num_experts=8, top_k=2, capacity_factor={spec['capacity_factor']}, tokens={spec['evaluated_tokens']})\n",
                ],
                "outputs": outputs,
            }
        ],
        "metadata": {
            "license": "CC-BY-4.0",
            "course": COURSE_ID,
            "version": version,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    raw = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    return raw, image_meta


def build_annotations(file_records: dict[str, Any], image_records: dict[int, list[dict[str, str]]]) -> dict[str, Any]:
    return {
        "task_id": "G20",
        "step": "A",
        "product_baseline_commit": "d993748c01fabaff26586908de9bf72499534012",
        "architecture_handoff_commit": "ef9aa1cb57bb1e108d379518b29d717ba2a1d927",
        "authorization_commit": "78c0194b3085511fb98616784eed93a8baf1223e",
        "course_id": COURSE_ID,
        "default_runtime_config": {
            "WORKSPACE_COURSE_IDS": COURSE_ID,
            "WORKSPACE_ANSWER_ENABLED": True,
            "WORKSPACE_RETRIEVAL_STRATEGY": "fts",
            "SOURCE_VISION_ENABLED": True,
            "DEEPSEEK_CHAT_MODEL": "deepseek-flash",
            "DEEPSEEK_QUERY_MODEL": "deepseek-flash",
            "DASHSCOPE_VISION_MODEL": "qwen3-vl-plus",
            "JEV_MODEL": "Zev-1",
            "ANSWER_CONTEXT_TOKENS": 64000,
            "WORKSPACE_INPUT_TOKEN_LIMIT": 56000,
            "OUTPUT_RESERVE": 8192,
            "STRUCTURE_SEARCH_TOP_K": 12,
            "RETRIEVAL_PLAN_BUDGET": 4,
        },
        "a03_sealed_reference": {
            "policy": "reuse_existing_sealed_result_only",
            "summary_file": "docs/G/G19-A03-聚合.json",
            "candidate_sha": "703e5aac4856749e0091bc3d3ac78cd4f3c22f6b",
            "archive_digest": "3bc3cb170a8cdca43de4c02f9aca532b5e331e0b1ef674748b2f755cd6cce5cc",
            "full_question_pass": "4/10",
            "key_question_pass": "2/5",
            "evidence_coverage": "4/13",
            "visual_selected_and_read": "1/3",
            "critical_errors": 0,
        },
        "files": file_records,
        "notebook_images": image_records,
        "questions": [
            {
                "question_id": "Q01",
                "turn_index": 1,
                "category": "短文件综合与公式",
                "scope_action": "select_document:Mixtral-of-Experts-2401.04088v1.pdf",
                "referred_version": {"Mixtral-of-Experts-2401.04088v1.pdf": 1},
                "question": "综合这份课件，用中文解释稀疏混合专家层（SMoE）是怎么根据门控网络计算输出的？每个词元激活几个专家、总参数量和激活参数量分别是多少？在多卡专家并行（Expert Parallelism）时会带来什么挑战？",
                "required_evidence_units": ["E01", "E02", "E03"],
                "expected_conclusion": "说明每层含 8 个专家，路由器对每个词元选择 Top-2 (K=2) 个专家并按门控权重加权求和 sum_{i=0}^{n-1} G(x)_i * E_i(x)，其中 G(x)=Softmax(TopK(x*W_g))；总参数量（稀疏参数量）47B，推理时每个词元仅使用 13B 激活参数；多卡专家并行（EP）将词元路由到对应专家所在 GPU，会带来负载均衡（load balancing）挑战，需均匀分配负载以防单卡过载或计算瓶颈。",
            },
            {
                "question_id": "Q02",
                "turn_index": 2,
                "category": "图表或公式（PDF 三线表）",
                "scope_action": "select_document:Mixtral-of-Experts-2401.04088v1.pdf",
                "referred_version": {"Mixtral-of-Experts-2401.04088v1.pdf": 1},
                "question": "第 8 页表格里的 ArXiv 和 DM Mathematics 数据集，在第 0 层、第 15 层和第 31 层（First choice）连续两个词元选到同一个专家的重复比例分别是多少？随机分配的理论基准是多少？",
                "required_evidence_units": ["E04"],
                "expected_conclusion": "根据第 8 页 Table 5，First choice 连续词元专家重复比例：ArXiv 在 Layer 0/15/31 分别为 14.0%、27.9%、22.7%；DM Mathematics 在 Layer 0/15/31 分别为 14.1%、28.4%、19.7%。随机分配（random assignments）下 First choice 的理论期望重复比例为 1/8 = 12.5%（First and second choice 约为 46%）。",
            },
            {
                "question_id": "Q03",
                "turn_index": 3,
                "category": "图表或公式（PDF 双子图定位与读图）",
                "scope_action": "select_document:Mixtral-of-Experts-2401.04088v1.pdf",
                "referred_version": {"Mixtral-of-Experts-2401.04088v1.pdf": 1},
                "question": "第 5 页右侧图的横轴和纵轴分别表示什么？图中纵轴的刻度范围大约从多少到多少，曲线随上下文长度增加呈现什么趋势？",
                "required_evidence_units": ["E05"],
                "expected_conclusion": "第 5 页 Figure 4 右图（page:5:image:1）横轴为 Context length（刻度 0 到 30k），纵轴为 Perplexity（刻度从 2.0 到 3.8）；随上下文长度增加，Mixtral_8x7B 在 proof-pile 数据集上的困惑度（Perplexity）单调下降（从约 3.7–3.8 降至约 1.9–2.0）。",
            },
            {
                "question_id": "Q04",
                "turn_index": 4,
                "category": "图表或公式（PDF 图像榜单实读）",
                "scope_action": "select_document:Mixtral-of-Experts-2401.04088v1.pdf",
                "referred_version": {"Mixtral-of-Experts-2401.04088v1.pdf": 1},
                "question": "第 6 页的排行榜图里，Mixtral-8x7b-Instruct-v0.1 的 Arena Elo 评分、MT-bench 分数和开源许可证分别是什么？图中排名第一的模型及其 Arena Elo 评分是多少？",
                "required_evidence_units": ["E06"],
                "expected_conclusion": "第 6 页 Figure 6（page:6:image:1）榜单图中，Mixtral-8x7b-Instruct-v0.1 的 Arena Elo rating 为 1121，MT-bench (score) 为 8.3，License 为 Apache 2.0；图中排名第一的模型是 GPT-4-Turbo，其 Arena Elo rating 为 1243（MT-bench 为 9.32）。",
            },
            {
                "question_id": "Q05",
                "turn_index": 5,
                "category": "长文检索（27 页 PDF 附录后段）",
                "scope_action": "select_document:Direct-Preference-Optimization-2305.18290v3.pdf",
                "referred_version": {"Direct-Preference-Optimization-2305.18290v3.pdf": 1},
                "question": "在这份论文的附录实现细节里，默认使用的优化器、学习率、线性预热步数和批次大小分别是多少？默认的温度惩罚系数是多少，在做文本摘要实验时改成了多少？",
                "required_evidence_units": ["E07"],
                "expected_conclusion": "根据 DPO 论文第 20 页（Appendix B），默认使用 RMSprop 优化器、学习率 1e-6、在前 150 步从 0 线性预热到 1e-6、批次大小（batch size）为 64；默认温度控制参数 β = 0.1，而在 TL;DR 文本摘要（summarization）实验中 β 改为 0.5（其余参数保持不变）。",
            },
            {
                "question_id": "Q06",
                "turn_index": 6,
                "category": "跨文件检索与候选排序（两份 PDF）",
                "scope_action": "course_scope",
                "referred_version": {
                    "Mixtral-of-Experts-2401.04088v1.pdf": 1,
                    "Direct-Preference-Optimization-2305.18290v3.pdf": 1,
                },
                "question": "Mixtral 和 DPO 这两份资料里，Mixtral-8x7B-Instruct 微调时用了哪两种方法？而在 DPO 论文附录里 TL;DR 摘要的 β 参数和默认 β 分别是多少？",
                "required_evidence_units": ["E08", "E09"],
                "expected_conclusion": "根据 Mixtral 论文第 2 页（或第 1/5 页），Mixtral 8x7B – Instruct 微调使用了监督微调（SFT, supervised fine-tuning）和直接偏好优化（DPO, Direct Preference Optimization）两种方法；根据 DPO 论文第 20 页附录，默认 β = 0.1，而在 TL;DR 摘要（summarization）任务中 β = 0.5。",
            },
            {
                "question_id": "Q07",
                "turn_index": 7,
                "category": "跨文件定义与实验核对（PDF + Notebook）",
                "scope_action": "course_scope",
                "referred_version": {
                    "Mixtral-of-Experts-2401.04088v1.pdf": 1,
                    "Mixtral-Router-Verification.ipynb": 1,
                },
                "question": "在 Mixtral 课件和配套实验里，门控网络 G(x) 的计算公式是怎么定义的？实验在 ArXiv 验证集上测出的第 15 层连续词元首选专家重复率和平均路由分发延迟分别是多少？",
                "required_evidence_units": ["E10", "E11"],
                "expected_conclusion": "根据 Mixtral 课件第 2 页，门控网络定义为 G(x) := Softmax(TopK(x · W_g))，仅对线性层输出 logits 的前 K（K=2）项取 Softmax，其余位置置为 -∞；根据配套实验 Mixtral-Router-Verification.ipynb（v1）第 1 个 Cell 的保存输出（cell:0:output:0），在 ArXiv 验证集（32768 tokens）上测得第 15 层首选专家连续重复率为 27.9%（随机基准 12.5%），平均路由分发延迟为 0.42 ms / step（capacity_factor=1.25）。",
            },
            {
                "question_id": "Q08a",
                "turn_index": 8,
                "category": "跨轮跨版（Notebook v1 展示与读取）",
                "scope_action": "select_document:Mixtral-Router-Verification.ipynb",
                "referred_version": {"Mixtral-Router-Verification.ipynb": 1},
                "question": "请展示第 1 个 Cell 的保存输出，并说明实测的平均路由分发延迟、容量因子以及评估词元数分别是多少？",
                "required_evidence_units": ["E12"],
                "expected_conclusion": "在 Mixtral-Router-Verification.ipynb v1 第 1 个 Cell 保存输出（cell:0:output:0）中，实测平均路由分发延迟为 0.42 ms / step，容量因子（capacity_factor）为 1.25，评估词元数（Evaluated tokens）为 32768；同时消息展示该 Cell 的两张保存图片（output:1 与 output:2）。",
            },
            {
                "question_id": "Q08b",
                "turn_index": 9,
                "category": "跨轮跨版（按展示图序追问第 2 张图）",
                "scope_action": "keep_previous_scope",
                "referred_version": {"Mixtral-Router-Verification.ipynb": 1},
                "question": "刚才第二张图的横轴名称和方括号里的单位是什么？",
                "required_evidence_units": ["E13"],
                "expected_conclusion": "按上一轮消息实际展示顺序，第 2 张图为 Mixtral-Router-Verification.ipynb v1 的 cell:0:output:2（Dispatch Latency Trace (v1)），视觉实读横轴名称为 Batch Size，方括号内单位为 tokens（即 Batch Size [tokens]）。",
            },
            {
                "question_id": "Q09",
                "turn_index": 10,
                "category": "跨轮跨版（Notebook 更新至 v2 后自然问法跨版比较）",
                "scope_action": "update_notebook_to_v2_and_ask",
                "referred_version": {"Mixtral-Router-Verification.ipynb": "v1_vs_v2"},
                "question": "跟上一版相比，第 1 个 Cell 保存输出里的平均路由分发延迟、容量因子和评估词元数有什么变化？",
                "required_evidence_units": ["E14", "E15"],
                "expected_conclusion": "对比 Mixtral-Router-Verification.ipynb 旧版 v1 与当前版 v2 的 cell:0:output:0：平均路由分发延迟从 v1 的 0.42 ms / step 降至 v2 的 0.31 ms / step；容量因子（capacity_factor）从 v1 的 1.25 提高到 v2 的 1.50；评估词元数从 v1 的 32768 增加到 v2 的 65536（第 15 层首选重复率亦从 27.9% 变为 28.4%）。两侧引用需分别绑定 v1 与 v2。",
            },
            {
                "question_id": "Q10",
                "turn_index": 11,
                "category": "合理澄清（同页双图无方位/序号指代消歧）",
                "scope_action": "select_document:Mixtral-of-Experts-2401.04088v1.pdf",
                "referred_version": {"Mixtral-of-Experts-2401.04088v1.pdf": 1},
                "question": "第 5 页的插图里具体展示了什么结果？",
                "required_evidence_units": ["E16"],
                "expected_conclusion": "Mixtral-of-Experts-2401.04088v1.pdf 第 5 页包含两张独立插图（page:5:image:1 与 page:5:image:2），问题未指明左图/右图或图序；系统应识别出该页存在多个候选图像并提示澄清（object_ambiguous），不得只猜测其中一张冒充整页唯一图作答。",
            },
            {
                "question_id": "Q11",
                "turn_index": 12,
                "category": "缺证边界（明确未测项并拒绝代用）",
                "scope_action": "select_document:Mixtral-Router-Verification.ipynb",
                "referred_version": {"Mixtral-Router-Verification.ipynb": 2},
                "question": "在实验记录里，64 卡在 128k 上下文长度下的 All-to-All 通信延迟实测是多少毫秒？能把单步 0.31 ms 的路由分发延迟直接当作这个通信延迟吗？",
                "required_evidence_units": ["E17"],
                "expected_conclusion": "根据 Mixtral-Router-Verification.ipynb v2 第 1 个 Cell 的保存输出（cell:0:output:0），明确记录 64-GPU 在 128k 上下文下的 All-to-All 通信延迟在本次运行中未测量（not measured in this run）；不能把单步 0.31 ms 的路由分发延迟（Mean routing dispatch latency）当作 64 卡 128k 上下文的 All-to-All 通信延迟。",
            },
        ],
        "evidence_units": [
            {
                "unit_id": "E01",
                "question_id": "Q01",
                "turn_index": 1,
                "file_name": MIXTRAL_PDF,
                "version": 1,
                "parent_source_id": "page:1",
                "exact_source_id": None,
                "acceptable_equivalents": ["page:1", "page:2", "page:3"],
                "bbox": None,
                "verbatim_anchor": "each layer is composed of 8 feedforward blocks (i.e. experts). For every token, at each layer, a router network selects two experts... each token has access to 47B parameters, but only uses 13B active parameters during inference.",
                "required_facts": ["8 experts", "selects 2 experts (K=2)", "47B total/sparse parameters", "13B active parameters"],
            },
            {
                "unit_id": "E02",
                "question_id": "Q01",
                "turn_index": 1,
                "file_name": MIXTRAL_PDF,
                "version": 1,
                "parent_source_id": "page:2",
                "exact_source_id": None,
                "acceptable_equivalents": ["page:2"],
                "bbox": None,
                "verbatim_anchor": "sum_{i=0}^{n-1} G(x)_i * E_i(x) ... G(x) := Softmax(TopK(x · W_g))",
                "required_facts": ["weighted sum of expert outputs G(x)_i * E_i(x)", "G(x) = Softmax(TopK(x · W_g))"],
            },
            {
                "unit_id": "E03",
                "question_id": "Q01",
                "turn_index": 1,
                "file_name": MIXTRAL_PDF,
                "version": 1,
                "parent_source_id": "page:3",
                "exact_source_id": None,
                "acceptable_equivalents": ["page:3"],
                "bbox": None,
                "verbatim_anchor": "Expert Parallelism (EP) [28]... Note that EP introduces challenges in load balancing, as it is essential to distribute the workload evenly across the GPUs to prevent overloading individual GPUs or hitting computational bottlenecks.",
                "required_facts": ["Expert Parallelism (EP)", "load balancing challenge / prevent overloading individual GPUs"],
            },
            {
                "unit_id": "E04",
                "question_id": "Q02",
                "turn_index": 2,
                "file_name": MIXTRAL_PDF,
                "version": 1,
                "parent_source_id": "page:8",
                "exact_source_id": "page:8:table:1",
                "acceptable_equivalents": ["page:8 (page-level text support only; exact table object unregistered in baseline)"],
                "bbox": None,
                "verbatim_anchor": "ArXiv 14.0% 27.9% 22.7% ... DM Mathematics 14.1% 28.4% 19.7% ... Table 5: Percentage of expert assignment repetitions... random assignments is 1/8 = 12.5% for First choice",
                "required_facts": ["ArXiv: 14.0%, 27.9%, 22.7%", "DM Mathematics: 14.1%, 28.4%, 19.7%", "random baseline: 1/8 = 12.5%"],
            },
            {
                "unit_id": "E05",
                "question_id": "Q03",
                "turn_index": 3,
                "file_name": MIXTRAL_PDF,
                "version": 1,
                "parent_source_id": "page:5",
                "exact_source_id": "page:5:image:1",
                "acceptable_equivalents": ["page:5:image:1"],
                "bbox": [0.444992, 0.688088, 0.800889, 0.863947],
                "verbatim_anchor": "Visual plot labels inside page:5:image:1: x-axis 'Context length' (0..30k), y-axis 'Perplexity' (2.0..3.8), curve 'Mixtral_8x7B' decreasing monotonically.",
                "required_facts": ["x-axis: Context length", "y-axis: Perplexity", "y-axis range: 2.0 to 3.8", "monotonically decreasing trend"],
            },
            {
                "unit_id": "E06",
                "question_id": "Q04",
                "turn_index": 4,
                "file_name": MIXTRAL_PDF,
                "version": 1,
                "parent_source_id": "page:6",
                "exact_source_id": "page:6:image:1",
                "acceptable_equivalents": ["page:6:image:1"],
                "bbox": [0.208828, 0.561018, 0.791163, 0.809371],
                "verbatim_anchor": "Visual table inside page:6:image:1: Mixtral-8x7b-Instruct-v0.1 | 1121 | 8.3 | Apache 2.0; rank #1 GPT-4-Turbo | 1243 | 9.32 | Proprietary.",
                "required_facts": ["Mixtral Arena Elo: 1121", "Mixtral MT-bench: 8.3", "License: Apache 2.0", "Top model: GPT-4-Turbo (Elo 1243)"],
            },
            {
                "unit_id": "E07",
                "question_id": "Q05",
                "turn_index": 5,
                "file_name": DPO_PDF,
                "version": 1,
                "parent_source_id": "page:20",
                "exact_source_id": None,
                "acceptable_equivalents": ["page:20"],
                "bbox": None,
                "verbatim_anchor": "Unless noted otherwise, we use a β = 0.1, batch size of 64 and the RMSprop optimizer with a learning rate of 1e-6 by default. We linearly warmup the learning rate from 0 to 1e-6 over 150 steps. For TL;DR summarization, we use β = 0.5",
                "required_facts": ["RMSprop optimizer", "learning rate 1e-6", "warmup 150 steps", "batch size 64", "default beta = 0.1", "TL;DR summarization beta = 0.5"],
            },
            {
                "unit_id": "E08",
                "question_id": "Q06",
                "turn_index": 6,
                "file_name": MIXTRAL_PDF,
                "version": 1,
                "parent_source_id": "page:2",
                "exact_source_id": None,
                "acceptable_equivalents": ["page:2", "page:5"],
                "bbox": None,
                "verbatim_anchor": "We also present Mixtral 8x7B – Instruct, a chat model fine-tuned to follow instructions using supervised fine-tuning and Direct Preference Optimization [25].",
                "required_facts": ["supervised fine-tuning (SFT)", "Direct Preference Optimization (DPO)"],
            },
            {
                "unit_id": "E09",
                "question_id": "Q06",
                "turn_index": 6,
                "file_name": DPO_PDF,
                "version": 1,
                "parent_source_id": "page:20",
                "exact_source_id": None,
                "acceptable_equivalents": ["page:20"],
                "bbox": None,
                "verbatim_anchor": "Unless noted otherwise, we use a β = 0.1... For TL;DR summarization, we use β = 0.5, while rest of the parameters remain the same.",
                "required_facts": ["default beta = 0.1", "TL;DR summarization beta = 0.5"],
            },
            {
                "unit_id": "E10",
                "question_id": "Q07",
                "turn_index": 7,
                "file_name": MIXTRAL_PDF,
                "version": 1,
                "parent_source_id": "page:2",
                "exact_source_id": None,
                "acceptable_equivalents": ["page:2"],
                "bbox": None,
                "verbatim_anchor": "G(x) := Softmax(TopK(x · W_g)), where (TopK(l))_i := l_i if l_i is among the top-K coordinates of logits l in R^n and (TopK(l))_i := -infinity otherwise.",
                "required_facts": ["G(x) = Softmax(TopK(x · W_g))", "Top-K logits kept, others set to -infinity"],
            },
            {
                "unit_id": "E11",
                "question_id": "Q07",
                "turn_index": 7,
                "file_name": NOTEBOOK,
                "version": 1,
                "parent_source_id": "cell:0",
                "exact_source_id": "cell:0:output:0",
                "acceptable_equivalents": ["cell:0:output:0"],
                "bbox": None,
                "verbatim_anchor": "Layer 15 first-choice consecutive repetition: 27.9% (random baseline: 12.5%)\nMean routing dispatch latency: 0.42 ms / step (capacity_factor=1.25)",
                "required_facts": ["Layer 15 first-choice consecutive repetition: 27.9%", "Mean routing dispatch latency: 0.42 ms / step"],
            },
            {
                "unit_id": "E12",
                "question_id": "Q08a",
                "turn_index": 8,
                "file_name": NOTEBOOK,
                "version": 1,
                "parent_source_id": "cell:0",
                "exact_source_id": "cell:0:output:0",
                "acceptable_equivalents": ["cell:0:output:0 (plus displaying cell:0:output:1 and cell:0:output:2)"],
                "bbox": None,
                "verbatim_anchor": "Evaluated tokens: 32768 (ArXiv validation subset)\nMean routing dispatch latency: 0.42 ms / step (capacity_factor=1.25)",
                "required_facts": ["0.42 ms / step", "capacity_factor = 1.25", "32768 tokens", "2 images displayed in message"],
            },
            {
                "unit_id": "E13",
                "question_id": "Q08b",
                "turn_index": 9,
                "file_name": NOTEBOOK,
                "version": 1,
                "parent_source_id": "cell:0",
                "exact_source_id": "cell:0:output:2",
                "acceptable_equivalents": ["cell:0:output:2"],
                "bbox": None,
                "verbatim_anchor": "Visual plot x-axis label in cell:0:output:2 (v1 2nd displayed image): 'Batch Size [tokens]'",
                "required_facts": ["x-axis name: Batch Size", "unit in brackets: tokens"],
            },
            {
                "unit_id": "E14",
                "question_id": "Q09",
                "turn_index": 10,
                "file_name": NOTEBOOK,
                "version": 1,
                "parent_source_id": "cell:0",
                "exact_source_id": "cell:0:output:0",
                "acceptable_equivalents": ["cell:0:output:0 (v1 historical)"],
                "bbox": None,
                "verbatim_anchor": "v1: Evaluated tokens: 32768 ... Mean routing dispatch latency: 0.42 ms / step (capacity_factor=1.25)",
                "required_facts": ["v1 latency: 0.42 ms / step", "v1 capacity_factor: 1.25", "v1 evaluated tokens: 32768"],
            },
            {
                "unit_id": "E15",
                "question_id": "Q09",
                "turn_index": 10,
                "file_name": NOTEBOOK,
                "version": 2,
                "parent_source_id": "cell:0",
                "exact_source_id": "cell:0:output:0",
                "acceptable_equivalents": ["cell:0:output:0 (v2 current)"],
                "bbox": None,
                "verbatim_anchor": "v2: Evaluated tokens: 65536 ... Mean routing dispatch latency: 0.31 ms / step (capacity_factor=1.50)",
                "required_facts": ["v2 latency: 0.31 ms / step", "v2 capacity_factor: 1.50", "v2 evaluated tokens: 65536"],
            },
            {
                "unit_id": "E16",
                "question_id": "Q10",
                "turn_index": 11,
                "file_name": MIXTRAL_PDF,
                "version": 1,
                "parent_source_id": "page:5",
                "exact_source_id": "page:5:image:1|page:5:image:2",
                "acceptable_equivalents": ["clarification / object_ambiguous over page:5:image:1 and page:5:image:2"],
                "bbox": None,
                "verbatim_anchor": "Page 5 contains two distinct registered images: page:5:image:1 [0.444992, 0.688088, 0.800889, 0.863947] and page:5:image:2 [0.176471, 0.688093, 0.422344, 0.856755].",
                "required_facts": ["detects 2 candidate images on page 5", "requests clarification (object_ambiguous) without guessing one image"],
            },
            {
                "unit_id": "E17",
                "question_id": "Q11",
                "turn_index": 12,
                "file_name": NOTEBOOK,
                "version": 2,
                "parent_source_id": "cell:0",
                "exact_source_id": "cell:0:output:0",
                "acceptable_equivalents": ["cell:0:output:0 (v2)"],
                "bbox": None,
                "verbatim_anchor": "Mean routing dispatch latency: 0.31 ms / step (capacity_factor=1.50)\nNote: 64-GPU all-to-all communication latency under 128k context was not measured in this run.",
                "required_facts": ["64-GPU 128k context all-to-all communication latency was not measured", "cannot substitute 0.31 ms / step routing dispatch latency"],
            },
        ],
    }


def materialize(fixture_dir: Path, annotations_out: Path | None = None) -> dict[str, Any]:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    repo_fixture_dir = Path(__file__).resolve().parent / "G20-fixtures"

    file_records: dict[str, Any] = {}
    for pdf_name, expected_sha in EXPECTED_PDF_SHA256.items():
        src = repo_fixture_dir / pdf_name
        dst = fixture_dir / pdf_name
        if src.exists() and src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        if not dst.exists():
            raise FileNotFoundError(f"Missing frozen CC-BY 4.0 PDF: {dst}")
        raw = dst.read_bytes()
        actual_sha = hashlib.sha256(raw).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f"SHA-256 mismatch for {pdf_name}: {actual_sha} != {expected_sha}")
        file_records[pdf_name] = {
            "sha256": actual_sha,
            "bytes": len(raw),
            "license": "CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/)",
            "origin": (
                "https://arxiv.org/abs/2401.04088"
                if "2401.04088" in pdf_name
                else "https://arxiv.org/abs/2305.18290"
            ),
        }

    image_records: dict[int, list[dict[str, str]]] = {}
    for version in (1, 2):
        v_dir = fixture_dir / f"v{version}"
        v_dir.mkdir(parents=True, exist_ok=True)
        nb_bytes, img_meta = build_notebook_bytes(version)
        nb_path = v_dir / NOTEBOOK
        nb_path.write_bytes(nb_bytes)
        image_records[version] = img_meta
        file_records[f"v{version}/{NOTEBOOK}"] = {
            "sha256": hashlib.sha256(nb_bytes).hexdigest(),
            "bytes": len(nb_bytes),
            "license": "CC-BY-4.0",
            "origin": "G20 controlled versioned Notebook accompanying Mixtral 8x7B courseware",
        }

    annotations = build_annotations(file_records, image_records)
    if annotations_out is not None:
        annotations_out.write_text(
            json.dumps(annotations, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return annotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "G20-fixtures",
    )
    parser.add_argument(
        "--annotations-out",
        type=Path,
        default=Path(__file__).resolve().parent / "G20-A-判据与标注.json",
    )
    result = materialize(parser.parse_args().fixture_dir, parser.parse_args().annotations_out)
    print(json.dumps({"course_id": result["course_id"], "files": result["files"]}, indent=2))
