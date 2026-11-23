"""J02 real same-budget multi-file strategy comparison runner (`j-course-multifile`)."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.workspace import get_workspace_service, router
from app.config import Settings
from app.rag.local_library import LocalLibrary
from app.rag.workspace import WorkspaceService, make_workspace
from app.stores.structure_snapshot import SnapshotRequest, StructureSnapshotStore

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"
MAIN_COURSE = "j-course-multifile"
DISTRACTOR_COURSE = "j-course-distractor"
BASE_URL = "/api/workspace"
HOST = "http://127.0.0.1:8001"

MAIN_PDF_PATHS = (
    REPO_ROOT / "docs/G/G20-fixtures/Mixtral-of-Experts-2401.04088v1.pdf",
    REPO_ROOT / "docs/G/G20-fixtures/Direct-Preference-Optimization-2305.18290v3.pdf",
    REPO_ROOT / "docs/G/G20-fixtures/PagedAttention-2309.06180v1.pdf",
    REPO_ROOT / "docs/G/G22-fixtures/QLoRA-2305.14314v1.pdf",
    REPO_ROOT / "docs/G/G23-fixtures/GaLore-2403.03507v2.pdf",
)
MAIN_GEN_NAMES = (
    "Lecture03-Routing-Architecture.pptx",
    "Lecture05-KV-Quant-LabGuide.pptx",
    "Lecture07-Optimizer-Tradeoffs.pptx",
    "Lab04-Sparse-Routing-Bench.ipynb",
    "Lab06-KV-Precision-Ablation.ipynb",
)
DISTRACTOR_NAMES = (
    "Distractor-Routing-Notes.pdf",
    "Distractor-Lab.ipynb",
)


@dataclass(frozen=True)
class RequiredEvidence:
    file_name: str
    parent_source_id: str
    exact_source_id: str
    is_visual_only: bool = False


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    group: str
    question: str
    required_evidences: tuple[RequiredEvidence, ...]
    fact_patterns: tuple[tuple[str, str], ...]
    forbidden_patterns: tuple[str, ...] = ("9999", "7777")
    select_file: str | None = None
    focus_file_source: tuple[str, str] | None = None
    expected_clarification: bool = False
    expected_missing_refusal: bool = False
    expected_zero_retrieval: bool = False


CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        case_id="A1",
        group="A",
        question=(
            "DPO 论文附录 B 给出的实现超参数中，默认 β、batch size、优化器、最大学习率"
            "和线性 warmup 步数分别是多少？用于 TL;DR 摘要任务时 β 改为多少？"
        ),
        required_evidences=(
            RequiredEvidence(
                "Direct-Preference-Optimization-2305.18290v3.pdf", "page:20", "page:20"
            ),
        ),
        fact_patterns=(
            ("beta_default", r"0\.1"),
            ("batch_64", r"\b64\b"),
            ("rmsprop", r"RMSprop"),
            ("lr_1e6", r"1e-?6|10\^\{?-6\}?|0\.000001"),
            ("warmup_150", r"\b150\b"),
            ("beta_tldr", r"0\.5"),
        ),
    ),
    CaseSpec(
        case_id="A2",
        group="A",
        question=(
            "QLoRA 论文附录 B.2 实验细节中，13B 及以下与 33B/65B 模型的 LoRA dropout 比例"
            "分别设为多少？微调学习率调度策略、Adam beta2 与最大梯度范数各自如何设置？"
        ),
        required_evidences=(
            RequiredEvidence("QLoRA-2305.14314v1.pdf", "page:23", "page:23"),
        ),
        fact_patterns=(
            ("dropout_13b", r"0\.1"),
            ("dropout_33b", r"0\.05"),
            ("constant_lr", r"constant|恒定|固定"),
            ("beta2", r"0\.999"),
            ("grad_norm", r"0\.3"),
        ),
    ),
    CaseSpec(
        case_id="A3",
        group="A",
        question=(
            "在 GaLore 论文附录 E 的 C4 数据集、token batch size 为 256 的 LLaMA 1B "
            "非按层（✘）预训练对比中，8-bit GaLore 的内存开销和 Tokens/s 吞吐量是多少，"
            "相比 Adafactor 和 Adam8bit 的这两项指标各自高出还是低了？"
        ),
        required_evidences=(
            RequiredEvidence("GaLore-2403.03507v2.pdf", "page:23", "page:23"),
        ),
        fact_patterns=(
            ("galore_mem", r"7\.95"),
            ("galore_tok", r"1109\.38|1,?109\.38"),
            ("adam8_mem", r"9\.54"),
            ("adam8_tok", r"1569\.89|1,?569\.89"),
            ("adafactor_mem", r"13\.15"),
            ("adafactor_tok", r"581\.02"),
        ),
    ),
    CaseSpec(
        case_id="B1",
        group="B",
        question=(
            "在第三讲混合专家分发规则中，辅助负载均衡系数 α 与默认分派专家总数是多少？"
            "对应实验审计结果测得的有效算力占比、不均衡系数与单步壁钟耗时分别是多少？"
        ),
        required_evidences=(
            RequiredEvidence("Lecture03-Routing-Architecture.pptx", "page:2", "page:2"),
            RequiredEvidence("Lab04-Sparse-Routing-Bench.ipynb", "cell:2", "cell:2:output:0"),
        ),
        fact_patterns=(
            ("alpha_001", r"0\.01"),
            ("experts_8", r"\b8\b|八个"),
            ("ops_share", r"0\.284"),
            ("skew_cv", r"0\.062"),
            ("cycle_ms", r"11\.40|11\.4"),
        ),
    ),
    CaseSpec(
        case_id="B2",
        group="B",
        question=(
            "第五讲的 KV 缓存伙伴分页策略默认页块容量和理论碎片控制上限分别是多少？"
            "在配套实验轨迹回放里，实测内部松弛比例、尾部保留兆字节数与缺页捕获次数分别是多少？"
        ),
        required_evidences=(
            RequiredEvidence("Lecture05-KV-Quant-LabGuide.pptx", "page:2", "page:2"),
            RequiredEvidence("Lab04-Sparse-Routing-Bench.ipynb", "cell:4", "cell:4:output:0"),
        ),
        fact_patterns=(
            ("block_16", r"\b16\b"),
            ("frag_4pct", r"4\s*%"),
            ("internal_slack", r"2\.14\s*%"),
            ("tail_reserve", r"48\.5"),
            ("miss_traps", r"\b19\b"),
        ),
    ),
    CaseSpec(
        case_id="B3",
        group="B",
        question=(
            "第五讲提到的动态阈值页面压缩机制在空闲比例低于多少时触发？"
            "在配套实验遥测里，实际回收比例与重映射延迟分别是多少？"
        ),
        required_evidences=(
            RequiredEvidence("Lecture05-KV-Quant-LabGuide.pptx", "page:6", "page:6"),
            RequiredEvidence("Lab06-KV-Precision-Ablation.ipynb", "cell:4", "cell:4:output:0"),
        ),
        fact_patterns=(
            ("trigger_12", r"12\s*%|0\.12"),
            ("reclaim_pct", r"18\.4\s*%"),
            ("remap_lat", r"0\.87"),
        ),
    ),
    CaseSpec(
        case_id="B4",
        group="B",
        question=(
            "第七讲说明的异构双阶投影切换策略面向什么训练阶段？"
            "在配套消融实验中，该策略在 32k 上下文下的实测峰值显存（GB）与验证困惑度分别达到多少？"
        ),
        required_evidences=(
            RequiredEvidence("Lecture07-Optimizer-Tradeoffs.pptx", "page:4", "page:4"),
            RequiredEvidence("Lab06-KV-Precision-Ablation.ipynb", "cell:2", "cell:2:output:0"),
        ),
        fact_patterns=(
            ("stage_desc", r"长上下文|预训练后期|趋缓"),
            ("vram_682", r"6\.82"),
            ("ppl_1491", r"14\.91"),
        ),
    ),
    CaseSpec(
        case_id="C1",
        group="C",
        question=(
            "在稀疏专家分发架构图里，输入隐状态维度、单 Token 实际激活参数量、"
            "容量系数以及最大丢弃上限百分比各是多少？"
        ),
        required_evidences=(
            RequiredEvidence(
                "Lecture03-Routing-Architecture.pptx",
                "page:3",
                "page:3",
                is_visual_only=True,
            ),
        ),
        fact_patterns=(
            ("dim_4096", r"4096|4,?096"),
            ("active_129b", r"12\.9"),
            ("cap_125", r"1\.25"),
            ("drop_04", r"0\.4\s*%"),
        ),
    ),
    CaseSpec(
        case_id="C2",
        group="C",
        question=(
            "在路由概率分布热力图里，第 16 层占比最高的专家是哪一个、占比多少？"
            "第 24 层最冷门的专家是哪一个、占比多少？"
        ),
        required_evidences=(
            RequiredEvidence(
                "Lecture03-Routing-Architecture.pptx",
                "page:4",
                "page:4",
                is_visual_only=True,
            ),
        ),
        fact_patterns=(
            ("l16_expert4", r"(?:Expert|E)[-\s]?4|4\s*号"),
            ("l16_246", r"24\.6\s*%"),
            ("l24_expert1", r"(?:Expert|E)[-\s]?1|1\s*号"),
            ("l24_42", r"4\.2\s*%"),
        ),
    ),
    CaseSpec(
        case_id="C3",
        group="C",
        question=(
            "在 KV 块大小与内存额外开销的实验曲线图里，Block=4、16 和 64 分别对应多少开销百分比？"
            "图上标注的最优点峰值上下文长度是多少？"
        ),
        required_evidences=(
            RequiredEvidence(
                "Lecture05-KV-Quant-LabGuide.pptx",
                "page:4",
                "page:4",
                is_visual_only=True,
            ),
        ),
        fact_patterns=(
            ("b4_285", r"28\.5\s*%"),
            ("b16_39", r"3\.9\s*%"),
            ("b64_114", r"11\.4\s*%"),
            ("ctx_128k", r"128\s*[kK]"),
        ),
    ),
    CaseSpec(
        case_id="D1",
        group="D",
        question=(
            "PagedAttention-2309.06180v1.pdf 第 13 页比较了哪两种请求恢复机制？"
            "为什么在块大小很小时 swapping 开销远大于 recomputation？"
        ),
        required_evidences=(
            RequiredEvidence("PagedAttention-2309.06180v1.pdf", "page:13", "page:13"),
        ),
        fact_patterns=(
            ("swapping", r"swap|换入|换出"),
            ("recomputation", r"recomput|重算|重新计算"),
            ("pcie_small", r"CPU|PCIe|小.*传输|带宽"),
        ),
        expected_zero_retrieval=True,
    ),
    CaseSpec(
        case_id="D2",
        group="D",
        question="这份实验笔记本完整记录了哪两段基准函数的名称及其关键输出指标值？",
        required_evidences=(
            RequiredEvidence("Lab04-Sparse-Routing-Bench.ipynb", "cell:2", "cell:2:output:0"),
            RequiredEvidence("Lab04-Sparse-Routing-Bench.ipynb", "cell:4", "cell:4:output:0"),
        ),
        fact_patterns=(
            ("fn1", r"dispatch_audit_v7"),
            ("val1", r"0\.284"),
            ("fn2", r"slab_probe_k12"),
            ("val2", r"2\.14\s*%"),
        ),
        select_file="Lab04-Sparse-Routing-Bench.ipynb",
        expected_zero_retrieval=True,
    ),
    CaseSpec(
        case_id="D3",
        group="D",
        question=(
            "本课程已导入资料中是否包含 Distractor-405B-Q2 或 LLaMA-3 70B 使用 4-bit GaLore "
            "投影量化的实测吞吐量？如有请给出精确数值与出处，如无请明确说明未测/不存在。"
        ),
        required_evidences=(),
        fact_patterns=(
            ("refusal", r"没有|未找到|未包含|未提供|未测|不存在|无法确认"),
        ),
        forbidden_patterns=("9999", "7777", "999 tok", "777 tok"),
        expected_missing_refusal=True,
    ),
    CaseSpec(
        case_id="D4",
        group="D",
        question="与左侧子图相比，另一张图的坐标指标和变化趋势有什么不同？",
        required_evidences=(
            RequiredEvidence("Lecture07-Optimizer-Tradeoffs.pptx", "page:3", "page:3:shape:4"),
        ),
        fact_patterns=(),
        select_file="Lecture07-Optimizer-Tradeoffs.pptx",
        focus_file_source=("Lecture07-Optimizer-Tradeoffs.pptx", "page:3"),
        expected_clarification=True,
    ),
)

ARM_CONFIGS: dict[str, dict[str, Any]] = {
    "S0_prod_current": {
        "workspace_retrieval_strategy": "fts",
        "workspace_retrieval_pool_cap": 12,
        "workspace_retrieval_budget": 0,
        "workspace_vector_retrieval_enabled": False,
        "source_vision_enabled": False,
    },
    "S0_pool12": {
        "workspace_retrieval_strategy": "s0",
        "workspace_retrieval_pool_cap": 12,
        "workspace_retrieval_budget": 6,
        "workspace_vector_retrieval_enabled": False,
        "source_vision_enabled": False,
    },
    "S0_pool24": {
        "workspace_retrieval_strategy": "s0",
        "workspace_retrieval_pool_cap": 24,
        "workspace_retrieval_budget": 6,
        "workspace_vector_retrieval_enabled": False,
        "source_vision_enabled": False,
    },
    "S1_pool12": {
        "workspace_retrieval_strategy": "jev",
        "workspace_retrieval_pool_cap": 12,
        "workspace_retrieval_budget": 6,
        "workspace_vector_retrieval_enabled": False,
        "source_vision_enabled": False,
    },
    "S1_pool24": {
        "workspace_retrieval_strategy": "jev",
        "workspace_retrieval_pool_cap": 24,
        "workspace_retrieval_budget": 6,
        "workspace_vector_retrieval_enabled": False,
        "source_vision_enabled": False,
    },
    "S2a_pool12": {
        "workspace_retrieval_strategy": "s2a",
        "workspace_retrieval_pool_cap": 12,
        "workspace_retrieval_budget": 6,
        "workspace_vector_retrieval_enabled": False,
        "source_vision_enabled": False,
    },
    "S2b_pool12": {
        "workspace_retrieval_strategy": "s2b",
        "workspace_retrieval_pool_cap": 12,
        "workspace_retrieval_budget": 6,
        "workspace_vector_retrieval_enabled": False,
        "source_vision_enabled": False,
    },
    "S3_pool12": {
        "workspace_retrieval_strategy": "vector",
        "workspace_retrieval_pool_cap": 12,
        "workspace_retrieval_budget": 6,
        "workspace_vector_retrieval_enabled": True,
        "source_vision_enabled": True,
        "workspace_s3_max_visual_reads": 2,
    },
}


def _nearest_rank(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int((pct / 100.0) * len(ordered) + 0.999999) - 1))
    return round(ordered[idx], 2)


def _load_env_overrides(env_file: Path) -> dict[str, Any]:
    raw: dict[str, str] = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        raw[k.strip()] = v.strip().strip('"').strip("'")
    mapped: dict[str, Any] = {}
    key_map = {
        "DEEPSEEK_API_KEY": "deepseek_api_key",
        "DEEPSEEK_BASE_URL": "deepseek_base_url",
        "DEEPSEEK_QUERY_MODEL": "deepseek_query_model",
        "DEEPSEEK_TIMEOUT_SECONDS": "deepseek_timeout_seconds",
        "DEEPSEEK_MAX_RETRIES": "deepseek_max_retries",
        "JEV_API_KEY": "jev_api_key",
        "JEV_BASE_URL": "jev_base_url",
        "JEV_MODEL": "jev_model",
        "DASHSCOPE_API_KEY": "dashscope_api_key",
        "DASHSCOPE_BASE_URL": "dashscope_base_url",
        "DASHSCOPE_EMBEDDING_MODEL": "dashscope_embedding_model",
        "DASHSCOPE_EMBEDDING_DIMENSION": "dashscope_embedding_dimension",
        "DASHSCOPE_EMBEDDING_BATCH_SIZE": "dashscope_embedding_batch_size",
        "DASHSCOPE_VISUAL_EMBEDDING_MODEL": "dashscope_visual_embedding_model",
        "DASHSCOPE_VISUAL_EMBEDDING_DIMENSION": "dashscope_visual_embedding_dimension",
        "DASHSCOPE_VISION_MODEL": "dashscope_vision_model",
        "DASHSCOPE_TIMEOUT_SECONDS": "dashscope_timeout_seconds",
        "DASHSCOPE_MAX_RETRIES": "dashscope_max_retries",
    }
    for env_k, field in key_map.items():
        if env_k in raw and raw[env_k] != "":
            val: Any = raw[env_k]
            if field.endswith(("dimension", "batch_size", "retries")):
                val = int(val)
            elif field.endswith("seconds"):
                val = float(val)
            mapped[field] = val
    return mapped


def _make_settings(base_env: dict[str, Any], work_dir: Path, overrides: dict[str, Any]) -> Settings:
    params = {
        "_env_file": None,
        **base_env,
        "app_env": "test",
        "course_root": work_dir / "courses",
        "data_dir": work_dir / "data",
        "sqlite_path": work_dir / "j02_eval.sqlite3",
        "workspace_course_ids": f"{MAIN_COURSE},{DISTRACTOR_COURSE}",
        "workspace_answer_enabled": True,
        "workspace_input_token_limit": 32_768,
        "qdrant_url": "",
        "qdrant_api_key": "",
        "qdrant_collection": "j02_dev_eval_chunks",
        "qdrant_local_path": None,
        **overrides,
    }
    return Settings(**params)


def _prepare_corpus(work_dir: Path, base_env: dict[str, Any]) -> dict[str, Any]:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    main_dir = work_dir / "courses" / MAIN_COURSE
    dist_dir = work_dir / "courses" / DISTRACTOR_COURSE
    main_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)

    for pdf in MAIN_PDF_PATHS:
        shutil.copy2(pdf, main_dir / pdf.name)
    for name in MAIN_GEN_NAMES:
        shutil.copy2(FIXTURE_ROOT / "main_generated" / name, main_dir / name)
    for name in DISTRACTOR_NAMES:
        shutil.copy2(FIXTURE_ROOT / "distractor" / name, dist_dir / name)

    settings = _make_settings(base_env, work_dir, {})
    lib = LocalLibrary(settings)
    main_res = lib.import_paths(MAIN_COURSE, [main_dir])
    dist_res = lib.import_paths(DISTRACTOR_COURSE, [dist_dir])
    assert all(item.status == "published" for item in (*main_res, *dist_res)), (main_res, dist_res)

    snapshot = StructureSnapshotStore(settings).read(SnapshotRequest((MAIN_COURSE,)))
    doc_by_name = {doc.file_name: doc.document_id for doc in snapshot.documents}
    assert len(doc_by_name) == 10, doc_by_name

    # Save baseline sqlite snapshot for identical clean start per arm
    shutil.copy2(work_dir / "j02_eval.sqlite3", work_dir / "j02_corpus_clean.sqlite3")
    return {
        "main_files": len(main_res),
        "distractor_files": len(dist_res),
        "main_sources_total": len(snapshot.sources),
        "doc_by_name": doc_by_name,
    }


def _node_id_for(service: WorkspaceService, session_id: str, doc_id: str, source_id: str) -> str:
    catalog = service._catalog(session_id)
    for nid, node in catalog.nodes.items():
        if node.document.document_id == doc_id and node.source and node.source.source_id == source_id:
            return nid
    raise KeyError((doc_id, source_id))


def _evaluate_case(
    spec: CaseSpec,
    payload: dict[str, Any],
    doc_by_name: dict[str, str],
    client: TestClient,
    session_path: str,
) -> dict[str, Any]:
    turn = payload["turn"]
    msg = payload["message"]
    trace = turn.get("trace") or {}
    blocks = msg.get("blocks") or []
    sources = msg.get("sources") or {}
    answer_text = "\n".join(b.get("text", "") for b in blocks if b.get("type") in {"text", "notice"})

    # Check cited evidence HTTP accessibility
    cited_ids: list[str] = []
    for blk in blocks:
        for eid in blk.get("citations") or []:
            if eid not in cited_ids:
                cited_ids.append(eid)
    http_checks: dict[str, Any] = {}
    for eid in cited_ids:
        r_ev = client.get(f"{session_path}/messages/{msg['message_id']}/evidence/{eid}", headers={"origin": HOST})
        has_prev = r_ev.status_code == 200 and r_ev.json().get("has_preview")
        r_prev = (
            client.get(f"{session_path}/messages/{msg['message_id']}/evidence/{eid}/preview", headers={"origin": HOST})
            if has_prev
            else None
        )
        http_checks[eid] = {
            "evidence_status": r_ev.status_code,
            "preview_status": r_prev.status_code if r_prev is not None else None,
        }

    # Check severe errors first
    severe_reasons: list[str] = []
    for ev_id, ev in sources.items():
        if ev.get("revision", {}).get("course_id") != MAIN_COURSE:
            severe_reasons.append(f"cross_course_citation:{ev_id}")
    for pat in spec.forbidden_patterns:
        if re.search(pat, answer_text, re.IGNORECASE):
            severe_reasons.append(f"forbidden_leak:{pat}")
    for blk in blocks:
        if blk.get("type") == "text":
            for eid in blk.get("citations") or []:
                ev = sources.get(eid)
                if ev is None or ev.get("read_status") == "not_attempted":
                    severe_reasons.append(f"cited_unread_source:{eid}")
    # Check if cited evidence is broken on HTTP endpoint
    for eid, st in http_checks.items():
        if st["evidence_status"] != 200:
            severe_reasons.append(f"broken_evidence_http:{eid}:{st['evidence_status']}")

    retrieval = trace.get("retrieval")
    pool_keys = set()
    selected_keys = set()
    vis_cand_keys = set()
    vis_target_keys = set()
    if isinstance(retrieval, dict):
        for item in retrieval.get("candidate_pool") or []:
            s = item.get("source") or {}
            pool_keys.add((s.get("document_id"), s.get("source_id")))
        for item in retrieval.get("visual_candidates") or []:
            s = item.get("source") or {}
            vis_cand_keys.add((s.get("document_id"), s.get("source_id")))
        for s in retrieval.get("selected") or []:
            selected_keys.add((s.get("document_id"), s.get("source_id")))
        for item in retrieval.get("visual_read_targets") or []:
            s = item.get("source") or {}
            vis_target_keys.add((s.get("document_id"), s.get("source_id")))

    # Check special control behaviors
    if spec.expected_clarification:
        ok = (
            turn.get("status") == "clarification"
            and trace.get("clarification_reason") == "object_ambiguous"
            and len(turn.get("options") or []) >= 2
            and not severe_reasons
        )
        return {
            "case_id": spec.case_id,
            "group": spec.group,
            "passed": ok,
            "parent_funnel": {"L1": True, "L2": True, "L3": True, "L4": ok, "L5": ok},
            "exact_funnel": {"L1": True, "L2": True, "L3": True, "L4": ok, "L5": ok},
            "severe_reasons": severe_reasons,
            "zero_retrieval_ok": True,
            "fact_hits": {"clarification": ok},
            "http_checks": http_checks,
            "answer_preview": answer_text[:280],
        }

    # Evaluate fact checks
    fact_hits = {
        label: bool(re.search(pattern, answer_text, re.IGNORECASE | re.DOTALL))
        for label, pattern in spec.fact_patterns
    }
    all_facts = all(fact_hits.values())
    zero_ret_ok = (not spec.expected_zero_retrieval) or (retrieval is None)

    if spec.expected_missing_refusal:
        ok = all_facts and not severe_reasons and turn.get("status") == "completed"
        return {
            "case_id": spec.case_id,
            "group": spec.group,
            "passed": ok,
            "parent_funnel": {"L1": True, "L2": ok, "L3": ok, "L4": ok, "L5": ok},
            "exact_funnel": {"L1": True, "L2": ok, "L3": ok, "L4": ok, "L5": ok},
            "severe_reasons": severe_reasons,
            "zero_retrieval_ok": zero_ret_ok,
            "fact_hits": fact_hits,
            "http_checks": http_checks,
            "answer_preview": answer_text[:280],
        }

    # Funnel per required evidence (both parent & exact口径)
    parent_l2 = True
    parent_l3 = True
    parent_l4 = True
    parent_l5 = True
    exact_l2 = True
    exact_l3 = True
    exact_l4 = True
    exact_l5 = True
    req_details: list[dict[str, Any]] = []

    for req in spec.required_evidences:
        did = doc_by_name[req.file_name]
        ekey = (did, req.exact_source_id)

        def _match_source(keys: set[tuple[Any, Any]], target_id: str, prefix_parent: str) -> bool:
            if (did, target_id) in keys:
                return True
            return any(d == did and (sid == prefix_parent or str(sid).startswith(f"{prefix_parent}:")) for d, sid in keys)

        if spec.expected_zero_retrieval:
            p_l2 = e_l2 = p_l3 = e_l3 = True
        else:
            p_l2 = _match_source(pool_keys | vis_cand_keys, req.parent_source_id, req.parent_source_id)
            e_l2 = ekey in (pool_keys | vis_cand_keys) or (req.parent_source_id == req.exact_source_id and p_l2)
            p_l3 = _match_source(selected_keys | vis_target_keys, req.parent_source_id, req.parent_source_id)
            e_l3 = ekey in (selected_keys | vis_target_keys) or (req.parent_source_id == req.exact_source_id and p_l3)

        # L4 strictly requires actual read (pure visual MUST have real vision read, NOT not_attempted!)
        matching_eids_parent: list[str] = []
        matching_eids_exact: list[str] = []
        for eid, ev in sources.items():
            src = ev.get("revision", {}).get("source") or {}
            if src.get("document_id") != did:
                continue
            sid = src.get("source_id", "")
            reads = ev.get("reads") or []
            if req.is_visual_only:
                ok_read = any(
                    r.get("method") == "vision"
                    and r.get("status") in {"succeeded", "partial"}
                    and bool(r.get("text", "").strip())
                    for r in reads
                )
            else:
                ok_read = any(
                    r.get("method") in {"native_text", "native_table", "ocr"}
                    and r.get("status") in {"succeeded", "partial"}
                    and bool(r.get("text", "").strip())
                    for r in reads
                )
            if not ok_read:
                continue
            if sid == req.exact_source_id:
                matching_eids_exact.append(eid)
            if sid == req.parent_source_id or str(sid).startswith(f"{req.parent_source_id}:"):
                matching_eids_parent.append(eid)

        p_l4 = bool(matching_eids_parent)
        e_l4 = bool(matching_eids_exact)
        p_l5 = (
            p_l4
            and all_facts
            and any(eid in cited_ids and http_checks.get(eid, {}).get("evidence_status") == 200 for eid in matching_eids_parent)
        )
        e_l5 = (
            e_l4
            and all_facts
            and any(eid in cited_ids and http_checks.get(eid, {}).get("evidence_status") == 200 for eid in matching_eids_exact)
        )
        parent_l2 &= p_l2
        parent_l3 &= p_l3
        parent_l4 &= p_l4
        parent_l5 &= p_l5
        exact_l2 &= e_l2
        exact_l3 &= e_l3
        exact_l4 &= e_l4
        exact_l5 &= e_l5
        req_details.append(
            {
                "file_name": req.file_name,
                "parent": req.parent_source_id,
                "exact": req.exact_source_id,
                "parent_eids": matching_eids_parent,
                "exact_eids": matching_eids_exact,
                "p_l2": p_l2,
                "p_l3": p_l3,
                "p_l4": p_l4,
                "p_l5": p_l5,
                "e_l4": e_l4,
                "e_l5": e_l5,
            }
        )

    passed = (
        turn.get("status") == "completed"
        and exact_l5
        and all_facts
        and zero_ret_ok
        and not severe_reasons
    )
    return {
        "case_id": spec.case_id,
        "group": spec.group,
        "passed": passed,
        "parent_funnel": {
            "L1": True,
            "L2": parent_l2,
            "L3": parent_l3,
            "L4": parent_l4,
            "L5": parent_l5,
        },
        "exact_funnel": {
            "L1": True,
            "L2": exact_l2,
            "L3": exact_l3,
            "L4": exact_l4,
            "L5": exact_l5,
        },
        "severe_reasons": severe_reasons,
        "zero_retrieval_ok": zero_ret_ok,
        "fact_hits": fact_hits,
        "req_details": req_details,
        "http_checks": http_checks,
        "answer_preview": answer_text[:280],
    }


def _collect_provider_usage(trace: dict[str, Any]) -> dict[str, Any]:
    usage = {
        "deepseek_calls": 0,
        "deepseek_tokens": 0,
        "jev_calls": 0,
        "jev_tokens": 0,
        "dashscope_embed_calls": 0,
        "dashscope_vision_calls": 0,
        "dashscope_tokens": 0,
    }
    ret = trace.get("retrieval") or {}
    rw = ret.get("rewrite") or {}
    usage["deepseek_calls"] += rw.get("external_call_count", 0)
    usage["deepseek_tokens"] += rw.get("total_tokens", 0) or 0
    pl = ret.get("planner") or {}
    usage["deepseek_calls"] += pl.get("external_call_count", 0)
    usage["deepseek_tokens"] += pl.get("total_tokens", 0) or 0
    gen = trace.get("generation") or {}
    usage["deepseek_calls"] += gen.get("external_call_count", 0)
    usage["deepseek_tokens"] += gen.get("total_tokens", 0) or 0
    jv = ret.get("jev") or {}
    usage["jev_calls"] += jv.get("external_call_count", 0)
    usage["jev_tokens"] += jv.get("total_tokens", 0) or 0
    vec = ret.get("vector") or {}
    usage["dashscope_embed_calls"] += vec.get("external_call_count", 0)
    s3v = trace.get("s3_visual_reads") or {}
    for tgt in s3v.get("targets") or []:
        usage["dashscope_vision_calls"] += tgt.get("external_call_count", 0)
    return usage


def _run_arm(
    arm_name: str,
    arm_cfg: dict[str, Any],
    base_env: dict[str, Any],
    work_dir: Path,
    doc_by_name: dict[str, str],
    qdrant_dir: Path | None,
) -> dict[str, Any]:
    # Isolate per-arm sqlite copy from clean pre-indexed corpus so every arm shares exact SHA/version/epoch!
    arm_db = work_dir / f"db_{arm_name}.sqlite3"
    shutil.copy2(work_dir / "j02_corpus_clean.sqlite3", arm_db)

    overrides = dict(arm_cfg)
    if qdrant_dir is not None and arm_cfg.get("workspace_vector_retrieval_enabled"):
        overrides["qdrant_local_path"] = qdrant_dir
    settings = _make_settings(base_env, work_dir, {**overrides, "sqlite_path": arm_db})
    service = make_workspace(settings)

    app = FastAPI()
    app.include_router(router, prefix=BASE_URL)
    app.dependency_overrides[get_workspace_service] = lambda: service

    results: list[dict[str, Any]] = []
    latencies: list[float] = []
    agg_usage = {
        "deepseek_calls": 0,
        "deepseek_tokens": 0,
        "jev_calls": 0,
        "jev_tokens": 0,
        "dashscope_embed_calls": 0,
        "dashscope_vision_calls": 0,
        "dashscope_tokens": 0,
    }

    try:
        with TestClient(app, base_url=HOST) as client:
            for spec in CASES:
                req_body: dict[str, Any] = {"course_ids": [MAIN_COURSE]}
                if spec.select_file:
                    req_body["document_ids"] = [doc_by_name[spec.select_file]]
                s_resp = client.post(
                    f"{BASE_URL}/sessions",
                    json=req_body,
                    headers={"origin": HOST},
                )
                assert s_resp.status_code == 201, s_resp.text
                session_id = s_resp.json()["session_id"]
                session_path = f"{BASE_URL}/sessions/{session_id}"
                if spec.focus_file_source is not None:
                    fn, sid = spec.focus_file_source
                    nid = _node_id_for(service, session_id, doc_by_name[fn], sid)
                    f_resp = client.put(
                        f"{session_path}/focus",
                        json={"node_id": nid, "scope_revision": 1},
                        headers={"origin": HOST},
                    )
                    assert f_resp.status_code == 200, f_resp.text
                    scope_rev = f_resp.json()["scope_revision"]
                else:
                    scope_rev = 1

                t0 = perf_counter()
                t_resp = client.post(
                    f"{session_path}/turns",
                    json={
                        "client_turn_id": f"{arm_name}-{spec.case_id}",
                        "kind": "ask",
                        "text": spec.question,
                        "scope_revision": scope_rev,
                    },
                    headers={"origin": HOST},
                )
                elapsed_ms = round((perf_counter() - t0) * 1000.0, 2)
                assert t_resp.status_code == 200, t_resp.text
                payload = t_resp.json()
                trace = payload["turn"].get("trace") or {}
                ev = _evaluate_case(spec, payload, doc_by_name, client, session_path)
                usage = _collect_provider_usage(trace)
                for k, v in usage.items():
                    agg_usage[k] += v
                latencies.append(elapsed_ms)

                ret = trace.get("retrieval") or {}
                gen = trace.get("generation") or {}
                reading = trace.get("reading") or {}
                ev.update(
                    {
                        "elapsed_ms": elapsed_ms,
                        "turn_status": payload["turn"].get("status"),
                        "route": trace.get("route"),
                        "effective_strategy": ret.get("effective_strategy"),
                        "rewrite_status": (ret.get("rewrite") or {}).get("status"),
                        "jev_trace": ret.get("jev"),
                        "expansion_trace": (
                            {k: v for k, v in (ret.get("expansion") or {}).items() if k != "new_candidates"}
                            if ret.get("expansion")
                            else None
                        ),
                        "planner_trace": (
                            {k: v for k, v in (ret.get("planner") or {}).items() if k != "new_candidates"}
                            if ret.get("planner")
                            else None
                        ),
                        "vector_trace": ret.get("vector"),
                        "s3_visual_reads": trace.get("s3_visual_reads"),
                        "estimated_message_input": reading.get("estimated_message_input"),
                        "generation_status": gen.get("status"),
                        "selected_sources": [
                            {"doc": s.get("document_id"), "sid": s.get("source_id")}
                            for s in (ret.get("selected") or [])
                        ],
                        "usage": usage,
                    }
                )
                results.append(ev)
                print(
                    f"  [{arm_name}] {spec.case_id}: pass={ev['passed']} "
                    f"exactL5={ev['exact_funnel']['L5']} parentL5={ev['parent_funnel']['L5']} "
                    f"sev={len(ev['severe_reasons'])} eff={ev['effective_strategy']} "
                    f"ms={elapsed_ms}",
                    flush=True,
                )
    finally:
        service.close()

    total = len(results)
    grp = lambda g: [r for r in results if r["group"] == g]
    grp_b = grp("B")
    grp_c = grp("C")
    grp_d = grp("D")
    vis_l23 = sum(
        1 for r in grp_c if r["exact_funnel"]["L2"] and r["exact_funnel"]["L3"]
    )
    parent_passed = sum(
        1
        for r in results
        if r["parent_funnel"]["L5"]
        and all(r["fact_hits"].values())
        and r["zero_retrieval_ok"]
        and not r["severe_reasons"]
    )
    summary = {
        "arm": arm_name,
        "total_cases": total,
        "whole_case_passed": sum(1 for r in results if r["passed"]),
        "whole_case_pass_rate": round(sum(1 for r in results if r["passed"]) / total, 4),
        "parent_whole_case_passed": parent_passed,
        "parent_whole_case_pass_rate": round(parent_passed / total, 4),
        "group_a_passed": sum(1 for r in grp("A") if r["passed"]),
        "group_b_two_hop_passed": sum(1 for r in grp_b if r["passed"]),
        "group_b_parent_passed": sum(
            1
            for r in grp_b
            if r["parent_funnel"]["L5"] and all(r["fact_hits"].values()) and not r["severe_reasons"]
        ),
        "two_hop_cross_file_rate": round(sum(1 for r in grp_b if r["passed"]) / len(grp_b), 4),
        "group_c_visual_recall_at_budget": round(vis_l23 / len(grp_c), 4),
        "group_c_e2e_passed": sum(1 for r in grp_c if r["passed"]),
        "pure_image_e2e_rate": round(sum(1 for r in grp_c if r["passed"]) / len(grp_c), 4),
        "group_d_control_passed": sum(1 for r in grp_d if r["passed"]),
        "group_d_control_rate": round(sum(1 for r in grp_d if r["passed"]) / len(grp_d), 4),
        "severe_error_count": sum(len(r["severe_reasons"]) for r in results),
        "latency_p50_ms": _nearest_rank(latencies, 50),
        "latency_p95_ms": _nearest_rank(latencies, 95),
        "latency_max_ms": round(max(latencies), 2) if latencies else 0.0,
        "usage": agg_usage,
    }
    return {"summary": summary, "cases": results}


def _run_d4b_version_and_resilience_checks(
    base_env: dict[str, Any],
    work_dir: Path,
    qdrant_dir: Path,
) -> dict[str, Any]:
    """Verify v1->v2 version isolation (including Qdrant stale filter) & fallback probes."""
    ver_db = work_dir / "db_version_iso.sqlite3"
    shutil.copy2(work_dir / "j02_corpus_clean.sqlite3", ver_db)
    v2_nb_path = work_dir / "courses" / MAIN_COURSE / "Lab06-KV-Precision-Ablation.ipynb"
    # Republish v2 on ver_db (using fresh copy of Lab06 fixture, then restore clean v1 file on disk)
    shutil.copy2(FIXTURE_ROOT / "main_generated" / "Lab06-KV-Precision-Ablation.ipynb", v2_nb_path)
    nb6 = json.loads(v2_nb_path.read_text(encoding="utf-8"))
    nb6["cells"][4]["outputs"][0]["text"] = [
        "profile=ASC-64K -> reclaim_pct=29.5% | remap_lat=0.41 ms | stall_cnt=0\n"
    ]
    v2_nb_path.write_text(json.dumps(nb6, indent=2, ensure_ascii=False), encoding="utf-8")

    s_pub = _make_settings(base_env, work_dir, {"sqlite_path": ver_db})
    lib = LocalLibrary(s_pub)
    snap_before = StructureSnapshotStore(s_pub).read(SnapshotRequest((MAIN_COURSE,)))
    lab6_doc = next(d for d in snap_before.documents if d.file_name == "Lab06-KV-Precision-Ablation.ipynb")
    repub = lib.import_paths(MAIN_COURSE, [v2_nb_path], document_id=lab6_doc.document_id)
    assert repub[0].status == "published" and repub[0].version == 2, repub
    shutil.copy2(FIXTURE_ROOT / "main_generated" / "Lab06-KV-Precision-Ablation.ipynb", v2_nb_path)

    # Test S2b & S3 against updated v2 snapshot while stale v1 Qdrant points exist!
    checks: dict[str, Any] = {"v2_republish": {"document_id": lab6_doc.document_id, "version": repub[0].version}}
    for probe_name, strat, qpath in (
        ("S2b_v2", "s2b", None),
        ("S3_v2", "vector", qdrant_dir),
    ):
        st = _make_settings(
            base_env,
            work_dir,
            {
                "sqlite_path": ver_db,
                "workspace_retrieval_strategy": strat,
                "workspace_vector_retrieval_enabled": strat == "vector",
                "qdrant_local_path": qpath,
                "workspace_retrieval_budget": 6,
            },
        )
        svc = make_workspace(st)
        app = FastAPI()
        app.include_router(router, prefix=BASE_URL)
        app.dependency_overrides[get_workspace_service] = (lambda t: (lambda: t))(svc)
        try:
            with TestClient(app, base_url=HOST) as client:
                sr = client.post(
                    f"{BASE_URL}/sessions",
                    json={"course_ids": [MAIN_COURSE]},
                    headers={"origin": HOST},
                )
                sid = sr.json()["session_id"]
                tr = client.post(
                    f"{BASE_URL}/sessions/{sid}/turns",
                    json={
                        "client_turn_id": f"d4b-{probe_name}",
                        "kind": "ask",
                        "text": "第五讲提到的动态阈值页面压缩机制在空闲比例低于多少时触发？配套实验最新实测回收比例与重映射延迟是多少？",
                        "scope_revision": 1,
                    },
                    headers={"origin": HOST},
                )
                data = tr.json()
                txt = "\n".join(b.get("text", "") for b in data["message"]["blocks"] if b.get("type") == "text")
                srcs = data["message"]["sources"]
                lab6_versions = [
                    ev["revision"]["source"]["version"]
                    for ev in srcs.values()
                    if ev["file_name"] == "Lab06-KV-Precision-Ablation.ipynb"
                ]
                checks[probe_name] = {
                    "has_29_5": "29.5" in txt,
                    "leaked_18_4": "18.4" in txt,
                    "lab6_versions": lab6_versions,
                    "passed": ("29.5" in txt and "18.4" not in txt and all(v == 2 for v in lab6_versions)),
                }
        finally:
            svc.close()

    # Corrupted Qdrant fallback probe
    bad_qdrant = work_dir / "corrupted_qdrant"
    shutil.rmtree(bad_qdrant, ignore_errors=True)
    bad_qdrant.mkdir(parents=True, exist_ok=True)
    # Create wrong-dimension collection so QdrantSchemaError triggers zero-blocking fallback
    from qdrant_client import QdrantClient, models as qmodels
    qc = QdrantClient(path=str(bad_qdrant))
    qc.create_collection(
        collection_name="j02_dev_eval_chunks",
        vectors_config={"text_dense": qmodels.VectorParams(size=4, distance=qmodels.Distance.COSINE)},
    )
    qc.close()
    st_bad = _make_settings(
        base_env,
        work_dir,
        {
            "sqlite_path": ver_db,
            "workspace_retrieval_strategy": "vector",
            "workspace_vector_retrieval_enabled": True,
            "qdrant_local_path": bad_qdrant,
            "workspace_retrieval_budget": 6,
        },
    )
    svc_bad = make_workspace(st_bad)
    app_bad = FastAPI()
    app_bad.include_router(router, prefix=BASE_URL)
    app_bad.dependency_overrides[get_workspace_service] = lambda: svc_bad
    try:
        with TestClient(app_bad, base_url=HOST) as cbad:
            sid = cbad.post(f"{BASE_URL}/sessions", json={"course_ids": [MAIN_COURSE]}, headers={"origin": HOST}).json()["session_id"]
            tr = cbad.post(
                f"{BASE_URL}/sessions/{sid}/turns",
                json={"client_turn_id": "bad-q", "kind": "ask", "text": CASES[0].question, "scope_revision": 1},
                headers={"origin": HOST},
            )
            d = tr.json()
            vtrace = (d["turn"].get("trace") or {}).get("retrieval", {}).get("vector")
            checks["s3_corrupt_qdrant_fallback"] = {
                "turn_status": d["turn"]["status"],
                "vector_status": (vtrace or {}).get("status"),
                "passed": d["turn"]["status"] == "completed" and (vtrace or {}).get("status") == "unavailable",
            }
    finally:
        svc_bad.close()
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path("/Users/Link/code/SRAG/.env"))
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/srag-j02-paired-run"))
    parser.add_argument("--arms", nargs="*", default=list(ARM_CONFIGS))
    parser.add_argument("--reuse-corpus", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "docs/J/J02-dev-eval/J02-对照实测.json",
    )
    args = parser.parse_args()

    base_env = _load_env_overrides(args.env_file)
    clean_db = args.work_dir / "j02_corpus_clean.sqlite3"
    if args.reuse_corpus and clean_db.exists():
        st_snap = _make_settings(base_env, args.work_dir, {"sqlite_path": clean_db})
        snap_existing = StructureSnapshotStore(st_snap).read(SnapshotRequest((MAIN_COURSE,)))
        corpus_info = {
            "main_files": len(snap_existing.documents),
            "distractor_files": 2,
            "main_sources_total": len(snap_existing.sources),
            "doc_by_name": {d.file_name: d.document_id for d in snap_existing.documents},
        }
    else:
        corpus_info = _prepare_corpus(args.work_dir, base_env)
    doc_by_name = corpus_info["doc_by_name"]

    qdrant_dir = args.work_dir / "qdrant_s3"
    s3_sync_info: dict[str, Any] = {"status": "skipped"}
    if "S3_pool12" in args.arms:
        print("Building isolated S3 local Qdrant index...", flush=True)
        st_sync = _make_settings(
            base_env,
            args.work_dir,
            {
                "sqlite_path": args.work_dir / "j02_corpus_clean.sqlite3",
                "workspace_vector_retrieval_enabled": True,
                "qdrant_local_path": qdrant_dir,
            },
        )
        svc_sync = make_workspace(st_sync)
        try:
            snap = StructureSnapshotStore(st_sync).read(SnapshotRequest((MAIN_COURSE,)))
            assert svc_sync.vector_bridge is not None
            s3_sync_info = svc_sync.vector_bridge.ensure_synced(
                snap, svc_sync.library.store, svc_sync.evidence
            )
            print("S3 index sync completed:", s3_sync_info, flush=True)
        finally:
            svc_sync.close()

    arm_outputs: dict[str, Any] = {}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        print(f"\n=== Running Arm: {arm} ===", flush=True)
        arm_outputs[arm] = _run_arm(
            arm,
            ARM_CONFIGS[arm],
            base_env,
            args.work_dir,
            doc_by_name,
            qdrant_dir if "S3_pool12" in args.arms else None,
        )
        partial = {
            "corpus": corpus_info,
            "s3_offline_index_build": s3_sync_info,
            "arms": arm_outputs,
        }
        serialized_partial = json.dumps(partial, indent=2, ensure_ascii=False)
        args.output_json.write_text(serialized_partial, encoding="utf-8")

    resilience = _run_d4b_version_and_resilience_checks(base_env, args.work_dir, qdrant_dir) if "S3_pool12" in args.arms else {}

    report = {
        "corpus": corpus_info,
        "s3_offline_index_build": s3_sync_info,
        "arms": arm_outputs,
        "resilience_and_version_isolation": resilience,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    # Double guard against accidental key leakage
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    for kname in ("DEEPSEEK_API_KEY", "JEV_API_KEY", "DASHSCOPE_API_KEY"):
        secret = base_env.get(kname.lower())
        if secret and secret in serialized:
            raise RuntimeError(f"Secret leakage detected for {kname}!")
    args.output_json.write_text(serialized, encoding="utf-8")
    print("\nSaved redacted evaluation output to", args.output_json)


if __name__ == "__main__":
    main()
