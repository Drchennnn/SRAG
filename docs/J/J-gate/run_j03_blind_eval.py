#!/usr/bin/env python3
"""J03 independent blind verification runner for frozen cs-mlsys-j (9 files, 18 questions)."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

MAIN_COURSE = "cs-mlsys-j"
ISOLATED_COURSE = "other-course-isolated"
BASE_URL = "/api/workspace"
HOST = "http://127.0.0.1:8001"
GATE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = GATE_DIR / "course_corpus"
CRITERIA_PATH = GATE_DIR / "J-blind-criteria.json"


@dataclass(frozen=True)
class ReqEv:
    file_name: str
    parent_source_id: str
    exact_source_id: str
    is_visual_only: bool = False


@dataclass(frozen=True)
class BlindCase:
    case_id: str
    category: str
    question: str
    course_id: str
    required_evidences: tuple[ReqEv, ...]
    fact_patterns: tuple[tuple[str, str], ...]
    forbidden_patterns: tuple[str, ...] = ()
    select_file: str | None = None
    expected_zero_retrieval: bool = False
    expected_missing_refusal: bool = False
    expected_cross_course_isolated: bool = False


FACT_PATTERNS_BY_ID: dict[str, tuple[tuple[str, str], ...]] = {
    "J_Q01": (
        ("obq_6932", r"69\.32"),
        ("gptq_5385", r"53\.85"),
        ("rtn_309", r"309(?:\.0+)?"),
        ("gptq_1123", r"11\.23"),
        ("c4_not_zero_shot", r"C4|校准|calibration|zero-?shot|非完全"),
    ),
    "J_Q02": (
        ("gsm8k_2077", r"20\.77"),
        ("gsm8k_2062", r"20\.62"),
        ("lb_4469", r"44\.69"),
        ("lb_4475", r"44\.75"),
        ("passkey_7", r"7\s*位|7-?digit"),
        ("paul_graham", r"Paul\s+Graham"),
    ),
    "J_Q03": (
        ("rouge1", r"60(?:\.0)?"),
        ("rouge2", r"28(?:\.0)?"),
        ("rougel", r"29\.5"),
        ("temp_05", r"0\.5"),
        ("topp_095", r"0\.95"),
        ("dim_128", r"128"),
        ("base_10000", r"10,?000|10000"),
    ),
    "J_Q04": (
        ("gptq_71ms", r"71\s*ms"),
        ("fp16_230ms", r"230\s*ms"),
        ("fp16_5gpu", r"5\s*[张块个x×]?\s*GPU|5\s*张"),
        ("kivi_4x", r"4\s*[x×倍]"),
        ("kivi_235_347", r"2\.35.*3\.47|3\.47"),
    ),
    "J_Q05": (
        ("t99_1940", r"19\.40?"),
        ("eff_912", r"91\.2\s*%?"),
        ("t99_1470", r"14\.70?"),
        ("eff_958", r"95\.8\s*%?"),
    ),
    "J_Q06": (
        ("tps_1840", r"1,?840"),
        ("vgb_1160", r"11\.60?"),
        ("red_384", r"38\.4\s*%?"),
        ("tps_1695", r"1,?695"),
        ("vgb_1285", r"12\.85"),
        ("red_317", r"31\.7\s*%?"),
    ),
    "J_Q07": (
        ("w2_892", r"8\.92"),
        ("pk_4120", r"4,?120"),
        ("w2_941", r"9\.41"),
        ("pk_3980", r"3,?980"),
    ),
    "J_Q08": (
        ("wait_380", r"3\.80?\s*%?"),
        ("cyc_615", r"6\.15"),
        ("wait_940", r"9\.40?\s*%?"),
        ("cyc_782", r"7\.82"),
    ),
    "J_Q09": (
        ("s2a_fp16", r"FP16|16\s*位|半精度"),
        ("s2a_256", r"256"),
        ("s2b_2bit", r"2-?\s*bit|2\s*位"),
        ("s2b_g32", r"32"),
        ("s4_42", r"4\.2\s*%"),
    ),
    "J_Q10": (
        ("kivi_213", r"21\.3"),
        ("sym_307", r"30\.7"),
        ("fp16_482", r"48\.2"),
        ("oom_32", r"32"),
    ),
    "J_Q11": (
        ("w_384", r"38\.4"),
        ("kv_672", r"67\.2"),
        ("sc_68", r"6\.8"),
        ("tot_1124", r"112\.4"),
        ("bs_44", r"44"),
    ),
    "J_Q12": (
        ("speedup_226", r"2\.26"),
        ("max_seq_16384", r"16,?384|16384"),
        ("warp_128", r"128"),
    ),
    "J_Q13": (
        ("base_834", r"8\.34"),
        ("gptq4_837", r"8\.37"),
        ("gptq3g128_845", r"8\.45"),
    ),
    "J_Q14": (
        ("prec_kivi2", r"KIVI-2|GQA-8"),
        ("ctx_8192", r"8,?192|8192"),
        ("kv_195", r"1\.95"),
        ("tps_1760", r"1,?760"),
    ),
    "J_Q15": (
        ("r128_1420", r"1,?420\.5"),
        ("r128_264", r"26\.4"),
        ("r32_1685", r"1,?685(?:\.0)?"),
        ("r32_228", r"22\.8"),
    ),
    "J_Q16": (
        ("honest_refusal", r"没有|未|不存在|未测|未给出|不包含|未提供|无法"),
    ),
    "J_Q17": (
        ("reject_kivi", r"不是|并非|不属于|不匹配|不同|CPU-offload|INT8|512"),
        ("cpu_offload", r"CPU-?offload|PCIe|INT8"),
        ("seq_512", r"512"),
    ),
    "J_Q18": (
        ("isolated_refusal", r"没有|未找到|未包含|未提供|未出现|未给出|未提及|不存在|无法|隔离"),
    ),
}

FORBIDDEN_BY_ID: dict[str, tuple[str, ...]] = {
    "J_Q16": (
        r"1-bit\s+GPTQ[^。\n]{0,25}(?:为|是|=)\s*\d",
        r"H200[^。\n]{0,35}128K[^。\n]{0,25}(?:吞吐|throughput)[^。\n]{0,15}(?:为|是|=)\s*\d",
    ),
    "J_Q18": (r"run_bucket_evict_probe", r"19\.40", r"14\.70", r"91\.2", r"95\.8"),
}

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
    "S1_pool12": {
        "workspace_retrieval_strategy": "jev",
        "workspace_retrieval_pool_cap": 12,
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


def _build_cases() -> tuple[BlindCase, ...]:
    criteria = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
    cases: list[BlindCase] = []
    for q in criteria["questions"]:
        qid = q["question_id"]
        cat = q["category"]
        scope = q.get("scope") or {}
        course_id = scope.get("course_id", MAIN_COURSE)
        select_file = scope.get("file_name")
        is_vis = cat == "Cat_C"
        reqs = tuple(
            ReqEv(
                file_name=s["file_name"],
                parent_source_id=s["source_id"],
                exact_source_id=s["source_id"],
                is_visual_only=is_vis,
            )
            for s in q.get("expected_sources") or []
            if not (qid == "J_Q17" and s["source_id"] == "cell:0")
        )
        cases.append(
            BlindCase(
                case_id=qid,
                category=cat,
                question=q["question"],
                course_id=course_id,
                required_evidences=reqs,
                fact_patterns=FACT_PATTERNS_BY_ID[qid],
                forbidden_patterns=FORBIDDEN_BY_ID.get(qid, ()),
                select_file=select_file,
                expected_zero_retrieval=(qid in {"J_Q13", "J_Q15"}),
                expected_missing_refusal=(qid == "J_Q16"),
                expected_cross_course_isolated=(qid == "J_Q18"),
            )
        )
    return tuple(cases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-root", type=Path, default=Path("/tmp/srag-j03-gate-subject"))
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/srag-j03-gate-run"))
    parser.add_argument("--env-file", type=Path, default=Path("/Users/Link/code/SRAG/.env"))
    parser.add_argument("--arms", nargs="*", default=list(ARM_CONFIGS))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=GATE_DIR / "J03-盲验实测.json",
    )
    args = parser.parse_args()

    os.environ["SRAG_SOFFICE"] = "/opt/homebrew/bin/soffice"
    sys.path.insert(0, str(args.subject_root.resolve()))

    import app.rag.workspace as ws_mod
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import nbformat as nbf
    from app.api.workspace import get_workspace_service, router
    from app.config import Settings
    from app.rag.local_library import LocalLibrary
    from app.rag.workspace import WorkspaceService, make_workspace
    from app.stores.structure_snapshot import SnapshotRequest, StructureSnapshotStore

    loaded_ws = Path(ws_mod.__file__).resolve()
    subject_res = args.subject_root.resolve()
    assert str(loaded_ws).startswith(str(subject_res)), (
        f"Subject mismatch: loaded {loaded_ws}, expected under {subject_res}"
    )
    print(f"Verified archived subject loaded from: {loaded_ws}", flush=True)

    base_env = _load_env_overrides(args.env_file)
    work_dir = args.work_dir

    def make_settings(overrides: dict[str, Any]) -> Settings:
        params: dict[str, Any] = {
            "_env_file": None,
            **base_env,
            "app_env": "test",
            "course_root": work_dir / "courses",
            "data_dir": work_dir / "data",
            "sqlite_path": work_dir / "j03_eval.sqlite3",
            "workspace_course_ids": f"{MAIN_COURSE},{ISOLATED_COURSE}",
            "workspace_answer_enabled": True,
            "workspace_input_token_limit": 32_768,
            "qdrant_url": "",
            "qdrant_api_key": "",
            "qdrant_collection": "j03_gate_chunks",
            "qdrant_local_path": None,
            "srag_soffice": Path("/opt/homebrew/bin/soffice"),
            **overrides,
        }
        return Settings(**params)

    # Prepare clean corpus
    if work_dir.exists():
        shutil.rmtree(work_dir)
    main_dir = work_dir / "courses" / MAIN_COURSE
    iso_dir = work_dir / "courses" / ISOLATED_COURSE
    main_dir.mkdir(parents=True, exist_ok=True)
    iso_dir.mkdir(parents=True, exist_ok=True)

    for p in sorted(CORPUS_DIR.iterdir()):
        if p.suffix.lower() in {".pdf", ".pptx", ".ipynb"}:
            shutil.copy2(p, main_dir / p.name)

    # Create unrelated isolated course file for J_Q18
    iso_nb = nbf.v4.new_notebook()
    iso_nb["cells"] = [
        nbf.v4.new_markdown_cell("# Isolated Course: Operating Systems Scheduler Lab\nCovers CFS and EEVDF CPU scheduling only."),
        nbf.v4.new_code_cell(
            "print('cfs_min_granularity_ms=3.0 | target_latency_ms=24.0')",
            execution_count=1,
            outputs=[nbf.v4.new_output(output_type="stream", name="stdout", text="cfs_min_granularity_ms=3.0 | target_latency_ms=24.0\n")],
        ),
    ]
    nbf.write(iso_nb, iso_dir / "OS-Scheduler-Lab.ipynb")

    s_prep = make_settings({})
    lib = LocalLibrary(s_prep)
    res_main = lib.import_paths(MAIN_COURSE, [main_dir])
    res_iso = lib.import_paths(ISOLATED_COURSE, [iso_dir])
    assert all(r.status == "published" for r in (*res_main, *res_iso)), (res_main, res_iso)

    snap_main = StructureSnapshotStore(s_prep).read(SnapshotRequest((MAIN_COURSE,)))
    doc_by_name = {d.file_name: d.document_id for d in snap_main.documents}
    assert len(doc_by_name) == 9, doc_by_name
    shutil.copy2(work_dir / "j03_eval.sqlite3", work_dir / "j03_corpus_clean.sqlite3")
    corpus_info = {
        "main_course": MAIN_COURSE,
        "isolated_course": ISOLATED_COURSE,
        "main_files": len(res_main),
        "main_sources_total": len(snap_main.sources),
        "doc_by_name": doc_by_name,
    }
    print(f"Prepared clean corpus: 9 files, {len(snap_main.sources)} sources.", flush=True)

    # Build isolated local Qdrant index if S3 requested
    qdrant_dir = work_dir / "qdrant_s3"
    s3_sync_info: dict[str, Any] = {"status": "skipped"}
    if "S3_pool12" in args.arms:
        print("Building isolated S3 local Qdrant index...", flush=True)
        st_sync = make_settings(
            {
                "sqlite_path": work_dir / "j03_corpus_clean.sqlite3",
                "workspace_vector_retrieval_enabled": True,
                "qdrant_local_path": qdrant_dir,
            }
        )
        svc_sync = make_workspace(st_sync)
        try:
            snap_all = StructureSnapshotStore(st_sync).read(SnapshotRequest((MAIN_COURSE, ISOLATED_COURSE)))
            assert svc_sync.vector_bridge is not None
            s3_sync_info = svc_sync.vector_bridge.ensure_synced(
                snap_all, svc_sync.library.store, svc_sync.evidence
            )
            print("S3 index sync completed:", s3_sync_info, flush=True)
        finally:
            svc_sync.close()

    cases = _build_cases()

    def collect_usage(trace: dict[str, Any]) -> dict[str, int]:
        u = {
            "deepseek_calls": 0,
            "deepseek_tokens": 0,
            "jev_calls": 0,
            "jev_tokens": 0,
            "dashscope_embed_calls": 0,
            "dashscope_vision_calls": 0,
        }
        ret = trace.get("retrieval") or {}
        for key in ("rewrite", "planner"):
            sub = ret.get(key) or {}
            u["deepseek_calls"] += sub.get("external_call_count", 0)
            u["deepseek_tokens"] += sub.get("total_tokens", 0) or 0
        gen = trace.get("generation") or {}
        u["deepseek_calls"] += gen.get("external_call_count", 0)
        u["deepseek_tokens"] += gen.get("total_tokens", 0) or 0
        jv = ret.get("jev") or {}
        u["jev_calls"] += jv.get("external_call_count", 0)
        u["jev_tokens"] += jv.get("total_tokens", 0) or 0
        vec = ret.get("vector") or {}
        u["dashscope_embed_calls"] += vec.get("external_call_count", 0)
        s3v = trace.get("s3_visual_reads") or {}
        for tgt in s3v.get("targets") or []:
            u["dashscope_vision_calls"] += tgt.get("external_call_count", 0)
        return u

    def eval_case(spec: BlindCase, payload: dict[str, Any], client: TestClient, spath: str) -> dict[str, Any]:
        turn = payload["turn"]
        msg = payload["message"]
        trace = turn.get("trace") or {}
        blocks = msg.get("blocks") or []
        sources = msg.get("sources") or {}
        answer_text = "\n".join(b.get("text", "") for b in blocks if b.get("type") in {"text", "notice"})

        cited_ids: list[str] = []
        for blk in blocks:
            for eid in blk.get("citations") or []:
                if eid not in cited_ids:
                    cited_ids.append(eid)
        http_checks: dict[str, Any] = {}
        for eid in cited_ids:
            r_ev = client.get(f"{spath}/messages/{msg['message_id']}/evidence/{eid}", headers={"origin": HOST})
            has_prev = r_ev.status_code == 200 and r_ev.json().get("has_preview")
            r_prev = (
                client.get(f"{spath}/messages/{msg['message_id']}/evidence/{eid}/preview", headers={"origin": HOST})
                if has_prev
                else None
            )
            http_checks[eid] = {
                "evidence_status": r_ev.status_code,
                "preview_status": r_prev.status_code if r_prev is not None else None,
            }

        severe_reasons: list[str] = []
        for ev_id, ev in sources.items():
            if ev.get("revision", {}).get("course_id") != spec.course_id:
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
        for eid, st in http_checks.items():
            if st["evidence_status"] != 200:
                severe_reasons.append(f"broken_evidence_http:{eid}:{st['evidence_status']}")

        retrieval = trace.get("retrieval")
        pool_keys: set[tuple[Any, Any]] = set()
        selected_keys: set[tuple[Any, Any]] = set()
        vis_cand_keys: set[tuple[Any, Any]] = set()
        vis_target_keys: set[tuple[Any, Any]] = set()
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

        fact_hits = {
            lbl: bool(re.search(pat, answer_text, re.IGNORECASE | re.DOTALL))
            for lbl, pat in spec.fact_patterns
        }
        all_facts = all(fact_hits.values())
        zero_ret_ok = (not spec.expected_zero_retrieval) or (retrieval is None)

        if spec.expected_missing_refusal or spec.expected_cross_course_isolated:
            ok = all_facts and not severe_reasons and turn.get("status") == "completed"
            return {
                "case_id": spec.case_id,
                "category": spec.category,
                "passed": ok,
                "parent_funnel": {"L1": True, "L2": ok, "L3": ok, "L4": ok, "L5": ok},
                "exact_funnel": {"L1": True, "L2": ok, "L3": ok, "L4": ok, "L5": ok},
                "severe_reasons": severe_reasons,
                "zero_retrieval_ok": zero_ret_ok,
                "fact_hits": fact_hits,
                "http_checks": http_checks,
                "answer_preview": answer_text[:300],
            }

        parent_l2 = parent_l3 = parent_l4 = parent_l5 = True
        exact_l2 = exact_l3 = exact_l4 = exact_l5 = True
        req_details: list[dict[str, Any]] = []

        for req in spec.required_evidences:
            did = doc_by_name[req.file_name]
            ekey = (did, req.exact_source_id)

            def match_src(keys: set[tuple[Any, Any]], tid: str, p_sid: str) -> bool:
                if (did, tid) in keys:
                    return True
                return any(
                    d == did and (sid == p_sid or str(sid).startswith(f"{p_sid}:"))
                    for d, sid in keys
                )

            if spec.expected_zero_retrieval or spec.select_file:
                # For single-file short docs (Deck-03/Lab-02) or explicit page direct reads (J_Q13)
                if retrieval is None:
                    p_l2 = e_l2 = p_l3 = e_l3 = True
                else:
                    p_l2 = match_src(pool_keys | vis_cand_keys, req.parent_source_id, req.parent_source_id)
                    e_l2 = ekey in (pool_keys | vis_cand_keys) or p_l2
                    p_l3 = match_src(selected_keys | vis_target_keys, req.parent_source_id, req.parent_source_id)
                    e_l3 = ekey in (selected_keys | vis_target_keys) or p_l3
            else:
                p_l2 = match_src(pool_keys | vis_cand_keys, req.parent_source_id, req.parent_source_id)
                e_l2 = ekey in (pool_keys | vis_cand_keys) or (req.parent_source_id == req.exact_source_id and p_l2)
                p_l3 = match_src(selected_keys | vis_target_keys, req.parent_source_id, req.parent_source_id)
                e_l3 = ekey in (selected_keys | vis_target_keys) or (req.parent_source_id == req.exact_source_id and p_l3)

            m_parent: list[str] = []
            m_exact: list[str] = []
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
                if sid == req.exact_source_id or sid.startswith(f"{req.exact_source_id}:"):
                    m_exact.append(eid)
                if sid == req.parent_source_id or str(sid).startswith(f"{req.parent_source_id}:"):
                    m_parent.append(eid)

            p_l4 = bool(m_parent)
            e_l4 = bool(m_exact)
            p_l5 = p_l4 and all_facts and any(
                eid in cited_ids and http_checks.get(eid, {}).get("evidence_status") == 200
                for eid in m_parent
            )
            e_l5 = e_l4 and all_facts and any(
                eid in cited_ids and http_checks.get(eid, {}).get("evidence_status") == 200
                for eid in m_exact
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
                    "source_id": req.exact_source_id,
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
            "category": spec.category,
            "passed": passed,
            "parent_funnel": {"L1": True, "L2": parent_l2, "L3": parent_l3, "L4": parent_l4, "L5": parent_l5},
            "exact_funnel": {"L1": True, "L2": exact_l2, "L3": exact_l3, "L4": exact_l4, "L5": exact_l5},
            "severe_reasons": severe_reasons,
            "zero_retrieval_ok": zero_ret_ok,
            "fact_hits": fact_hits,
            "req_details": req_details,
            "http_checks": http_checks,
            "answer_preview": answer_text[:1200],
        }

    def run_arm(arm_name: str, cfg: dict[str, Any]) -> dict[str, Any]:
        arm_db = work_dir / f"db_{arm_name}.sqlite3"
        shutil.copy2(work_dir / "j03_corpus_clean.sqlite3", arm_db)
        overrides = dict(cfg)
        if cfg.get("workspace_vector_retrieval_enabled"):
            overrides["qdrant_local_path"] = qdrant_dir
        st = make_settings({**overrides, "sqlite_path": arm_db})
        svc = make_workspace(st)
        app = FastAPI()
        app.include_router(router, prefix=BASE_URL)
        app.dependency_overrides[get_workspace_service] = lambda: svc

        results: list[dict[str, Any]] = []
        latencies: list[float] = []
        agg_usage = {
            "deepseek_calls": 0,
            "deepseek_tokens": 0,
            "jev_calls": 0,
            "jev_tokens": 0,
            "dashscope_embed_calls": 0,
            "dashscope_vision_calls": 0,
        }
        try:
            with TestClient(app, base_url=HOST) as client:
                for spec in cases:
                    req_body: dict[str, Any] = {"course_ids": [spec.course_id]}
                    if spec.select_file:
                        req_body["document_ids"] = [doc_by_name[spec.select_file]]
                    s_resp = client.post(f"{BASE_URL}/sessions", json=req_body, headers={"origin": HOST})
                    assert s_resp.status_code == 201, s_resp.text
                    sid = s_resp.json()["session_id"]
                    spath = f"{BASE_URL}/sessions/{sid}"
                    t0 = perf_counter()
                    t_resp = client.post(
                        f"{spath}/turns",
                        json={
                            "client_turn_id": f"{arm_name}-{spec.case_id}",
                            "kind": "ask",
                            "text": spec.question,
                            "scope_revision": 1,
                        },
                        headers={"origin": HOST},
                    )
                    elapsed_ms = round((perf_counter() - t0) * 1000.0, 2)
                    assert t_resp.status_code == 200, t_resp.text
                    payload = t_resp.json()
                    trace = payload["turn"].get("trace") or {}
                    ev = eval_case(spec, payload, client, spath)
                    usage = collect_usage(trace)
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
                            "usage": usage,
                        }
                    )
                    results.append(ev)
                    print(
                        f"  [{arm_name}] {spec.case_id} ({spec.category}): pass={ev['passed']} "
                        f"pL5={ev['parent_funnel']['L5']} eL5={ev['exact_funnel']['L5']} "
                        f"sev={len(ev['severe_reasons'])} eff={ev['effective_strategy']} ms={elapsed_ms}",
                        flush=True,
                    )
        finally:
            svc.close()

        total = len(results)
        by_cat = lambda c: [r for r in results if r["category"] == c]
        cat_c = by_cat("Cat_C")
        vis_l23 = sum(1 for r in cat_c if r["exact_funnel"]["L2"] and r["exact_funnel"]["L3"])
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
            "cat_a_passed": sum(1 for r in by_cat("Cat_A") if r["passed"]),
            "cat_b_exact_passed": sum(1 for r in by_cat("Cat_B") if r["passed"]),
            "cat_b_parent_passed": sum(
                1 for r in by_cat("Cat_B")
                if r["parent_funnel"]["L5"] and all(r["fact_hits"].values()) and not r["severe_reasons"]
            ),
            "cat_c_visual_l23_rate": round(vis_l23 / len(cat_c), 4),
            "cat_c_e2e_passed": sum(1 for r in cat_c if r["passed"]),
            "cat_d_control_passed": sum(1 for r in by_cat("Cat_D") if r["passed"]),
            "cat_e_negative_passed": sum(1 for r in by_cat("Cat_E") if r["passed"]),
            "severe_error_count": sum(len(r["severe_reasons"]) for r in results),
            "latency_p50_ms": _nearest_rank(latencies, 50),
            "latency_p95_ms": _nearest_rank(latencies, 95),
            "latency_max_ms": round(max(latencies), 2) if latencies else 0.0,
            "usage": agg_usage,
        }
        return {"summary": summary, "cases": results}

    # Resilience & Version Isolation Probes
    def run_resilience_probes() -> dict[str, Any]:
        probes: dict[str, Any] = {}
        ver_db = work_dir / "db_resilience.sqlite3"
        shutil.copy2(work_dir / "j03_corpus_clean.sqlite3", ver_db)

        # 1. Corrupt Qdrant dir -> 0-blocking fallback to S0
        bad_qdir = work_dir / "bad_qdrant"
        bad_qdir.mkdir(parents=True, exist_ok=True)
        from qdrant_client import QdrantClient, models as qmodels
        qc = QdrantClient(path=str(bad_qdir))
        qc.create_collection(
            collection_name="j03_gate_chunks",
            vectors_config={"text_dense": qmodels.VectorParams(size=4, distance=qmodels.Distance.COSINE)},
        )
        qc.close()
        st_badq = make_settings(
            {
                "sqlite_path": ver_db,
                "workspace_retrieval_strategy": "vector",
                "workspace_vector_retrieval_enabled": True,
                "qdrant_local_path": bad_qdir,
                "workspace_retrieval_budget": 6,
            }
        )
        svc_bq = make_workspace(st_badq)
        app_bq = FastAPI()
        app_bq.include_router(router, prefix=BASE_URL)
        app_bq.dependency_overrides[get_workspace_service] = lambda: svc_bq
        try:
            with TestClient(app_bq, base_url=HOST) as cbq:
                sid = cbq.post(f"{BASE_URL}/sessions", json={"course_ids": [MAIN_COURSE]}, headers={"origin": HOST}).json()["session_id"]
                tr = cbq.post(
                    f"{BASE_URL}/sessions/{sid}/turns",
                    json={"client_turn_id": "probe-badq", "kind": "ask", "text": cases[0].question, "scope_revision": 1},
                    headers={"origin": HOST},
                ).json()
                vt = (tr["turn"].get("trace") or {}).get("retrieval", {}).get("vector") or {}
                probes["corrupt_qdrant_fallback"] = {
                    "turn_status": tr["turn"]["status"],
                    "vector_status": vt.get("status"),
                    "passed": tr["turn"]["status"] == "completed" and vt.get("status") == "unavailable",
                }
        finally:
            svc_bq.close()

        # 2. Invalid Jev credentials -> 0-blocking fallback to S0
        st_badj = make_settings(
            {
                "sqlite_path": ver_db,
                "workspace_retrieval_strategy": "jev",
                "workspace_retrieval_budget": 6,
                "jev_api_key": "invalid-gate-test-key",
            }
        )
        svc_bj = make_workspace(st_badj)
        app_bj = FastAPI()
        app_bj.include_router(router, prefix=BASE_URL)
        app_bj.dependency_overrides[get_workspace_service] = lambda: svc_bj
        try:
            with TestClient(app_bj, base_url=HOST) as cbj:
                sid = cbj.post(f"{BASE_URL}/sessions", json={"course_ids": [MAIN_COURSE]}, headers={"origin": HOST}).json()["session_id"]
                tr = cbj.post(
                    f"{BASE_URL}/sessions/{sid}/turns",
                    json={"client_turn_id": "probe-badj", "kind": "ask", "text": cases[0].question, "scope_revision": 1},
                    headers={"origin": HOST},
                ).json()
                jt = (tr["turn"].get("trace") or {}).get("retrieval", {}).get("jev") or {}
                probes["invalid_jev_fallback"] = {
                    "turn_status": tr["turn"]["status"],
                    "jev_status": jt.get("status"),
                    "passed": tr["turn"]["status"] == "completed" and jt.get("status") in {"provider_error", "unavailable"},
                }
        finally:
            svc_bj.close()

        # 3. Version update v1 -> v2 on Lab-01 while stale v1 vectors exist in qdrant_s3
        lab1_tmp = work_dir / "courses" / MAIN_COURSE / "Lab-01-ZeroOverlap-Experiment.ipynb"
        nb1 = json.loads(lab1_tmp.read_text(encoding="utf-8"))
        nb1["cells"][5]["outputs"][0]["text"] = [
            "tile=128, alpha=0.65 -> w2_score=7.77, pk_mb=3333\n"
            "tile=256, alpha=0.80 -> w2_score=8.11, pk_mb=3111\n"
        ]
        lab1_tmp.write_text(json.dumps(nb1, indent=2, ensure_ascii=False), encoding="utf-8")
        st_pub = make_settings({"sqlite_path": ver_db})
        repub = LocalLibrary(st_pub).import_paths(
            MAIN_COURSE, [lab1_tmp], document_id=doc_by_name["Lab-01-ZeroOverlap-Experiment.ipynb"]
        )
        assert repub[0].status == "published" and repub[0].version == 2
        for pname, strat, use_vec in (("S2a_v2", "s2a", False), ("S3_v2", "vector", True)):
            st_v = make_settings(
                {
                    "sqlite_path": ver_db,
                    "workspace_retrieval_strategy": strat,
                    "workspace_vector_retrieval_enabled": use_vec,
                    "qdrant_local_path": qdrant_dir if use_vec else None,
                    "workspace_retrieval_budget": 6,
                }
            )
            svc_v = make_workspace(st_v)
            app_v = FastAPI()
            app_v.include_router(router, prefix=BASE_URL)
            app_v.dependency_overrides[get_workspace_service] = (lambda s: (lambda: s))(svc_v)
            try:
                with TestClient(app_v, base_url=HOST) as cv:
                    sid = cv.post(f"{BASE_URL}/sessions", json={"course_ids": [MAIN_COURSE]}, headers={"origin": HOST}).json()["session_id"]
                    tr = cv.post(
                        f"{BASE_URL}/sessions/{sid}/turns",
                        json={"client_turn_id": f"probe-{pname}", "kind": "ask", "text": cases[6].question, "scope_revision": 1},
                        headers={"origin": HOST},
                    ).json()
                    txt = "\n".join(b.get("text", "") for b in tr["message"]["blocks"] if b.get("type") == "text")
                    vers = [
                        ev["revision"]["source"]["version"]
                        for ev in tr["message"]["sources"].values()
                        if ev["file_name"] == "Lab-01-ZeroOverlap-Experiment.ipynb"
                    ]
                    probes[pname] = {
                        "has_new_777": "7.77" in txt,
                        "leaked_old_892": "8.92" in txt,
                        "lab1_versions": vers,
                        "passed": ("7.77" in txt and "8.92" not in txt and bool(vers) and all(v == 2 for v in vers)),
                    }
            finally:
                svc_v.close()
        return probes

    arm_outputs: dict[str, Any] = {}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        print(f"\n=== Running J03 Blind Arm: {arm} ===", flush=True)
        arm_outputs[arm] = run_arm(arm, ARM_CONFIGS[arm])
        partial = {
            "subject_commit": "4bb1c41d41d89f835e19334bdf19b8dcde4d64b1",
            "corpus": corpus_info,
            "s3_offline_index_build": s3_sync_info,
            "arms": arm_outputs,
        }
        args.output_json.write_text(json.dumps(partial, indent=2, ensure_ascii=False), encoding="utf-8")

    resilience = run_resilience_probes()
    print("\nResilience & Version Isolation:", json.dumps(resilience, ensure_ascii=False), flush=True)

    report = {
        "subject_commit": "4bb1c41d41d89f835e19334bdf19b8dcde4d64b1",
        "corpus": corpus_info,
        "s3_offline_index_build": s3_sync_info,
        "arms": arm_outputs,
        "resilience_and_version_isolation": resilience,
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    for kname in ("deepseek_api_key", "jev_api_key", "dashscope_api_key"):
        sec = base_env.get(kname)
        if sec and sec in serialized:
            raise RuntimeError(f"Secret leakage detected for {kname}!")
    args.output_json.write_text(serialized, encoding="utf-8")
    print(f"\nSaved redacted J03 blind results to {args.output_json}")


if __name__ == "__main__":
    main()
