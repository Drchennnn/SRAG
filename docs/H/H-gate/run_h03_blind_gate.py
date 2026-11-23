#!/usr/bin/env python3
"""Independent H03 blind gate runner for cs-ctrl-h + cs-other-h on clean archive."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

GATE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = GATE_DIR / "course_corpus"
CRITERIA_PATH = GATE_DIR / "H-blind-criteria.json"
DEFAULT_ENV_PATH = Path("/Users/Link/code/SRAG/.env")
DEFAULT_ARCHIVE_DIR = Path("/tmp/srag-h03-archive")
DEFAULT_OUT_JSON = GATE_DIR / "H03-盲验实测.json"

MAIN_COURSE = "cs-ctrl-h"
OTHER_COURSE = "cs-other-h"


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in {"'", '"'}:
            v = v[1:-1]
        values[k] = v
    return values


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_health(base_url: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/api/health", timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except Exception:  # noqa: BLE001
            time.sleep(0.25)
    raise RuntimeError(f"Server at {base_url} failed healthcheck")


def _http_status(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=5.0) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8", errors="replace")


def _ab(session: str, *args: str) -> str:
    cmd = ["agent-browser", "--session", session, *args]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(
            f"agent-browser failed ({res.returncode}): {' '.join(cmd)}\n"
            f"stdout: {res.stdout}\nstderr: {res.stderr}"
        )
    return res.stdout.strip()


def _ab_eval(session: str, js_expr: str) -> Any:
    return json.loads(_ab(session, "eval", js_expr))


def _ab_click(session: str, selector: str) -> None:
    _ab_eval(
        session,
        f"(() => {{ const el = document.querySelector({json.dumps(selector)}); "
        "if (el) el.scrollIntoView({block: 'center', inline: 'nearest'}); return !!el; })()",
    )
    _ab(session, "click", selector)


def _wait_for_js(
    session: str, predicate_js: str, *, timeout: float = 40.0, desc: str = ""
) -> Any:
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            val = _ab_eval(session, predicate_js)
            if val:
                return val
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(0.25)
    raise AssertionError(f"Timed out waiting for: {desc or predicate_js} (last_err={last_err})")


# ---------------------------------------------------------------------------
# M1 & M4 Lifecycle: Direct Archive Verification (Offline, Zero LLM)
# ---------------------------------------------------------------------------

def run_m1_and_lifecycle(archive_dir: Path, work_dir: Path, criteria: dict[str, Any]) -> dict[str, Any]:
    from app.config import Settings
    from app.rag.workspace import make_workspace
    from app.rag.workspace_evidence import readable_source_text
    from app.schemas.workspace import CreateSession, TurnRequest

    assert str(Path(sys.modules["app"].__file__).resolve()).startswith(str(archive_dir.resolve())), (
        f"app loaded from unexpected path: {sys.modules['app'].__file__}"
    )

    m1_dir = work_dir / "m1_lifecycle"
    if m1_dir.exists():
        shutil.rmtree(m1_dir)
    m1_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        sqlite_path=m1_dir / "srag.sqlite3",
        data_dir=m1_dir / "data",
        course_root=m1_dir / "courses",
        workspace_course_ids=f"{MAIN_COURSE},{OTHER_COURSE}",
        workspace_answer_enabled=False,
        workspace_kg_retrieval_enabled=True,
        source_vision_enabled=False,
        srag_soffice=Path("/opt/homebrew/bin/soffice"),
    )
    ws = make_workspace(settings)
    lib = ws.library

    # Copy v1 files to staging inside course_root so LocalLibrary accepts them.
    stage = m1_dir / "courses" / "stage"
    stage.mkdir(parents=True, exist_ok=True)
    files_v1 = [
        "Lecture-01-Core-Theory.pdf",
        "Slides-02-Method-Walkthrough.pptx",
        "Lab-01-Experiments.ipynb",
        "NoEdge-Notes.pdf",
    ]
    staged_v1 = []
    for name in files_v1:
        dst = stage / name
        shutil.copy2(CORPUS_DIR / name, dst)
        staged_v1.append(dst)
    other_dst = stage / "OtherCourse-Distractor.ipynb"
    shutil.copy2(CORPUS_DIR / "OtherCourse-Distractor.ipynb", other_dst)

    res_main = lib.import_paths(MAIN_COURSE, staged_v1)
    res_other = lib.import_paths(OTHER_COURSE, [other_dst])
    assert all(r.status == "published" for r in [*res_main, *res_other])

    doc_ids = {Path(r.source_path).name: r.document_id for r in res_main}
    other_doc_id = res_other[0].document_id

    # Verify M1: accepted edges on v1, exact span text == excerpt, and 0 counterexample leakage.
    kg = ws.kg_store
    kg.ensure_course_kg_synced((MAIN_COURSE, OTHER_COURSE))
    concepts_v1, edges_v1 = kg.active_edges_for_courses((MAIN_COURSE,))
    concept_by_key = {c.concept_key: c for c in concepts_v1}
    sources_v1 = {
        (doc_id, s.source_id): s
        for doc_id in doc_ids.values()
        for s in lib.store.current_sources(doc_id, course_ids=(MAIN_COURSE,))
    }

    span_mismatches = []
    edge_records_v1 = []
    for e in edges_v1:
        src = sources_v1.get((e.document_id, e.source_id))
        assert src is not None, f"Edge points to non-current source: {e}"
        raw_text = readable_source_text(src)
        slice_text = raw_text[e.evidence_span_start : e.evidence_span_end]
        if slice_text != e.evidence_excerpt:
            span_mismatches.append((e.edge_id, slice_text, e.evidence_excerpt))
        fname = next(k for k, v in doc_ids.items() if v == e.document_id)
        cname = concept_by_key[e.concept_key].canonical_name
        edge_records_v1.append(
            {
                "relation": e.relation_type,
                "concept": cname,
                "concept_key": e.concept_key,
                "file_name": fname,
                "source_id": e.source_id,
                "version": e.version,
                "rule": e.evidence_rule,
                "excerpt_head": e.evidence_excerpt[:80],
            }
        )
    assert not span_mismatches, f"Span mismatches found: {span_mismatches}"

    # Check expected accepted edges v1 (6 defined_in + 4 demonstrated_by = 10).
    expected_v1 = criteria["expected_accepted_edges_v1"]
    missing_v1 = []
    for exp in expected_v1:
        hit = any(
            rec["relation"] == exp["relation"].lower()
            and exp["concept_name"] in rec["concept"]
            and rec["file_name"] == exp["file_name"]
            and rec["source_id"] == exp["source_id"]
            and rec["version"] == exp["version"]
            for rec in edge_records_v1
        )
        if not hit:
            missing_v1.append(exp["edge_id"])

    # Check REJ-01..REJ-09 on v1 snapshot (0 false acceptances).
    false_acceptances = []
    for rec in edge_records_v1:
        if rec["file_name"] == "Slides-02-Method-Walkthrough.pptx" and rec["source_id"] == "page:6":
            false_acceptances.append(("REJ-01..04", rec))
        if rec["file_name"] == "Slides-02-Method-Walkthrough.pptx" and rec["source_id"] == "page:7":
            false_acceptances.append(("REJ-08", rec))
        if rec["file_name"] == "Lecture-01-Core-Theory.pdf" and rec["source_id"] == "page:5":
            false_acceptances.append(("REJ-08", rec))
        if rec["file_name"] == "Lab-01-Experiments.ipynb" and rec["source_id"] in {"cell:7", "cell:8"}:
            false_acceptances.append(("REJ-05/06/07", rec))
        if rec["file_name"] == "OtherCourse-Distractor.ipynb" or rec["concept_key"].startswith("smith"):
            false_acceptances.append(("REJ-04/09", rec))
        if "H∞" in rec["concept"] and rec["relation"] == "demonstrated_by":
            false_acceptances.append(("REJ-03", rec))

    # Also verify no edge in MAIN_COURSE has document_id == other_doc_id.
    assert all(e.document_id != other_doc_id for e in edges_v1), "Cross-course edge leaked!"
    assert not missing_v1, f"M1 missing expected v1 edges: {missing_v1}; got={edge_records_v1}"
    assert not false_acceptances, f"M1 counterexample false acceptances: {false_acceptances}"
    assert len(edge_records_v1) == len(expected_v1), (
        f"M1 unexpected extra edges ({len(edge_records_v1)} != {len(expected_v1)}): {edge_records_v1}"
    )

    # Test M4 lifecycle:
    # (a) Scheduled v2 does NOT prematurely leak into active v1 edges.
    lab_v2_stage = stage / "Lab-01-Experiments-v2.ipynb"
    shutil.copy2(CORPUS_DIR / "Lab-01-Experiments-v2.ipynb", lab_v2_stage)
    future_at = datetime.now(UTC) + timedelta(minutes=30)
    sched_res = lib.import_paths(
        MAIN_COURSE, [lab_v2_stage], document_id=doc_ids["Lab-01-Experiments.ipynb"], release_at=future_at
    )[0]
    assert sched_res.status == "scheduled"
    _, edges_while_sched = kg.active_edges_for_courses((MAIN_COURSE,))
    assert all(
        e.version == 1 for e in edges_while_sched if e.document_id == doc_ids["Lab-01-Experiments.ipynb"]
    ), "Scheduled v2 leaked before release_at!"

    # (b) Immediate publish v2 -> check REJ-10 (C2 old edge gone), C1=v2 cell:1, C3=v2 cell:3, C4=v2 cell:4.
    pub_v2 = lib.import_paths(
        MAIN_COURSE, [lab_v2_stage], document_id=doc_ids["Lab-01-Experiments.ipynb"]
    )[0]
    assert pub_v2.status == "published" and pub_v2.version == 2
    concepts_v2, edges_v2 = kg.active_edges_for_courses((MAIN_COURSE,))
    c_v2_map = {c.concept_key: c for c in concepts_v2}
    v2_records = [
        {
            "relation": e.relation_type,
            "concept": c_v2_map[e.concept_key].canonical_name,
            "file_name": next(k for k, v in doc_ids.items() if v == e.document_id),
            "source_id": e.source_id,
            "version": e.version,
        }
        for e in edges_v2
    ]
    c2_v2_demos = [
        r for r in v2_records if "扩展卡尔曼" in r["concept"] and r["relation"] == "demonstrated_by"
    ]
    assert not c2_v2_demos, f"REJ-10 leaked stale C2 v1 demo edge: {c2_v2_demos}"
    c3_v2_demos = [
        r for r in v2_records if "预测时域滑模" in r["concept"] and r["relation"] == "demonstrated_by"
    ]
    c4_v2_demos = [
        r for r in v2_records if "双二阶" in r["concept"] and r["relation"] == "demonstrated_by"
    ]
    assert any(r["source_id"] == "cell:3" and r["version"] == 2 for r in c3_v2_demos), c3_v2_demos
    assert any(r["source_id"] == "cell:4" and r["version"] == 2 for r in c4_v2_demos), c4_v2_demos

    # (c) Test promote_due -> direct QA without opening /kg/concepts first!
    # Re-stage a fresh DB for promote_due direct-QA verification.
    pd_dir = m1_dir / "promote_due_qa"
    pd_dir.mkdir(parents=True, exist_ok=True)
    pd_settings = Settings(
        sqlite_path=pd_dir / "srag.sqlite3",
        data_dir=pd_dir / "data",
        course_root=pd_dir / "courses",
        workspace_course_ids=MAIN_COURSE,
        workspace_answer_enabled=False,
        workspace_kg_retrieval_enabled=True,
        srag_soffice=Path("/opt/homebrew/bin/soffice"),
    )
    pd_ws = make_workspace(pd_settings)
    pd_lib = pd_ws.library
    pd_stage = pd_dir / "courses" / "stage"
    pd_stage.mkdir(parents=True, exist_ok=True)
    pd_staged = []
    for p in staged_v1:
        dst = pd_stage / p.name
        shutil.copy2(p, dst)
        pd_staged.append(dst)
    pd_v2 = pd_stage / lab_v2_stage.name
    shutil.copy2(lab_v2_stage, pd_v2)
    pd_res = pd_lib.import_paths(MAIN_COURSE, pd_staged)
    pd_lab_id = next(r.document_id for r in pd_res if "Lab-01" in r.source_path)
    soon = datetime.now(UTC) + timedelta(seconds=1.2)
    pd_sched = pd_lib.import_paths(MAIN_COURSE, [pd_v2], document_id=pd_lab_id, release_at=soon)[0]
    assert pd_sched.status == "scheduled"
    sess, owner = pd_ws.create(None, CreateSession(course_ids=[MAIN_COURSE]))
    time.sleep(1.6)
    # Submit turn directly WITHOUT calling kg_concepts first!
    turn_out = pd_ws.submit(
        owner,
        sess["session_id"],
        TurnRequest(
            client_turn_id="pd-direct-qa-1",
            kind="ask",
            text="极点配置滞后补偿 实验 sweep_v2",
            scope_revision=sess["scope_revision"],
        ),
    )
    # Check active edges immediately after promote_due-triggered turn:
    _, pd_edges_after = pd_ws.kg_store.active_edges_for_courses((MAIN_COURSE,))
    lab_edges_after = [e for e in pd_edges_after if e.document_id == pd_lab_id]
    assert lab_edges_after and all(e.version == 2 for e in lab_edges_after), lab_edges_after

    # (d) Delete NoEdge-Notes.pdf and Lab-01 -> edges immediately vanish!
    assert lib.delete(doc_ids["NoEdge-Notes.pdf"]) is True
    assert lib.delete(doc_ids["Lab-01-Experiments.ipynb"]) is True
    _, edges_after_del = kg.active_edges_for_courses((MAIN_COURSE,))
    assert not any(
        e.document_id in {doc_ids["NoEdge-Notes.pdf"], doc_ids["Lab-01-Experiments.ipynb"]}
        for e in edges_after_del
    ), "Deleted document edges leaked!"

    # Check M5 API 404 isolation at service layer too:
    sess_a, owner_a = ws.create(None, CreateSession(course_ids=[MAIN_COURSE]))
    sess_b, owner_b = ws.create(None, CreateSession(course_ids=[OTHER_COURSE]))
    cross_404 = []
    for label, fn in [
        ("owner_a_query_other_course", lambda: ws.kg_concepts(owner_a, sess_a["session_id"], course_id=OTHER_COURSE)),
        ("owner_a_access_sess_b", lambda: ws.kg_concepts(owner_a, sess_b["session_id"])),
        ("owner_b_query_main_concept", lambda: ws.kg_concept_detail(owner_b, sess_b["session_id"], concepts_v1[0].concept_key)),
    ]:
        try:
            fn()
            cross_404.append((label, "UNEXPECTED_200"))
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status", None)
            cross_404.append((label, status))
            assert status == 404, f"{label} raised {exc}"

    return {
        "m1_v1_accepted_count": len(edge_records_v1),
        "m1_v1_expected_count": len(expected_v1),
        "m1_v1_missing": missing_v1,
        "m1_span_exact_verified": len(span_mismatches) == 0,
        "m1_false_acceptances": false_acceptances,
        "m1_v1_edges": edge_records_v1,
        "m4_v2_accepted_count": len(v2_records),
        "m4_v2_c2_removed": len(c2_v2_demos) == 0,
        "m4_v2_c3_shifted_cell3": any(r["source_id"] == "cell:3" for r in c3_v2_demos),
        "m4_v2_c4_shifted_cell4": any(r["source_id"] == "cell:4" for r in c4_v2_demos),
        "m4_promote_due_direct_qa_v2_edges": len(lab_edges_after),
        "m4_delete_zero_leak": True,
        "m5_service_404": cross_404,
    }


# ---------------------------------------------------------------------------
# M2, M3, M4(browser), M5(browser): Real Chromium Dual-Entry Walkthrough
# ---------------------------------------------------------------------------

def run_browser_gate(archive_dir: Path, work_dir: Path, env_map: dict[str, str]) -> dict[str, Any]:
    web_dir = work_dir / "browser"
    if web_dir.exists():
        shutil.rmtree(web_dir)
    web_dir.mkdir(parents=True, exist_ok=True)
    shots = web_dir / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)
    stage = web_dir / "stage"
    stage.mkdir(parents=True, exist_ok=True)
    for name in (
        "Lecture-01-Core-Theory.pdf",
        "Slides-02-Method-Walkthrough.pptx",
        "Lab-01-Experiments.ipynb",
        "Lab-01-Experiments-v2.ipynb",
        "NoEdge-Notes.pdf",
        "OtherCourse-Distractor.ipynb",
    ):
        shutil.copy2(CORPUS_DIR / name, stage / name)

    results: dict[str, Any] = {}

    # Part A: Offline Zero-LLM Mode verification (Entry 1 navigation without external LLM)
    off_port = _find_free_port()
    off_url = f"http://127.0.0.1:{off_port}"
    off_env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "APP_HOST": "127.0.0.1",
        "APP_PORT": str(off_port),
        "SQLITE_PATH": str(web_dir / "off.sqlite3"),
        "DATA_DIR": str(web_dir / "off_data"),
        "COURSE_ROOT": str(web_dir / "off_courses"),
        "WORKSPACE_COURSE_IDS": f"{MAIN_COURSE},{OTHER_COURSE}",
        "WORKSPACE_ANSWER_ENABLED": "false",
        "WORKSPACE_KG_RETRIEVAL_ENABLED": "true",
        "WORKSPACE_RETRIEVAL_INCLUDE_ADJACENCY": "true",
        "SOURCE_VISION_ENABLED": "false",
        "DEEPSEEK_API_KEY": "",
        "DASHSCOPE_API_KEY": "",
        "JEV_API_KEY": "",
        "SRAG_SOFFICE": "/opt/homebrew/bin/soffice",
    }
    off_srv = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(off_port)],
        cwd=str(archive_dir),
        env=off_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    bs_off = f"h03-off-{off_port}"
    try:
        _wait_health(off_url)
        _ab(bs_off, "set", "viewport", "1380", "900")
        _ab(bs_off, "open", f"{off_url}/workspace/")
        _wait_for_js(bs_off, "document.getElementById('sessions')?.value || null", desc="off session")
        _ab(bs_off, "eval", "document.getElementById('import-panel').open = true")
        _ab(bs_off, "select", "#import-course", MAIN_COURSE)
        _ab(
            bs_off,
            "upload",
            "#import-files",
            str(stage / "Lecture-01-Core-Theory.pdf"),
            str(stage / "Lab-01-Experiments.ipynb"),
        )
        _ab(bs_off, "click", "#import-button")
        _wait_for_js(
            bs_off,
            "(() => { const li = [...document.querySelectorAll('#import-results li')]; "
            "return li.length === 2 && li.every(x => x.textContent.includes('导入完成')); })()",
            desc="offline imports done",
        )
        _ab(bs_off, "eval", "document.getElementById('kg-panel').open = true")
        off_names = _wait_for_js(
            bs_off,
            "(() => { const els = [...document.querySelectorAll('#kg-list .kg-item-name')]; "
            "return els.length >= 3 ? els.map(x => x.textContent) : null; })()",
            desc="offline kg concepts visible",
        )
        results["offline_zero_llm_nav"] = {"concepts": off_names, "passed": True}
    finally:
        subprocess.run(["agent-browser", "--session", bs_off, "close"], check=False, capture_output=True)
        off_srv.terminate()
        off_srv.wait(timeout=5)

    # Part B: Full Real-LLM Browser Gate (M2 Dual Entry, M3 No-Edge Fallback, M4 Replace + Stale Chip + v1 History, M5 404)
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    on_env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "APP_HOST": "127.0.0.1",
        "APP_PORT": str(port),
        "SQLITE_PATH": str(web_dir / "on.sqlite3"),
        "DATA_DIR": str(web_dir / "on_data"),
        "COURSE_ROOT": str(web_dir / "on_courses"),
        "WORKSPACE_COURSE_IDS": f"{MAIN_COURSE},{OTHER_COURSE}",
        "WORKSPACE_ANSWER_ENABLED": "true",
        "WORKSPACE_KG_RETRIEVAL_ENABLED": "true",
        "WORKSPACE_RETRIEVAL_INCLUDE_ADJACENCY": "true",
        "WORKSPACE_RETRIEVAL_STRATEGY": "s0",
        "WORKSPACE_RETRIEVAL_POOL_CAP": "12",
        "WORKSPACE_RETRIEVAL_BUDGET": "6",
        "WORKSPACE_VECTOR_RETRIEVAL_ENABLED": "false",
        "SOURCE_VISION_ENABLED": "false",
        "DEEPSEEK_API_KEY": env_map["DEEPSEEK_API_KEY"],
        "DEEPSEEK_BASE_URL": env_map.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "DEEPSEEK_QUERY_MODEL": "deepseek-flash",
        "DEEPSEEK_CHAT_MODEL": "deepseek-flash",
        "DASHSCOPE_API_KEY": "",
        "JEV_API_KEY": "",
        "SRAG_SOFFICE": "/opt/homebrew/bin/soffice",
    }
    srv = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(archive_dir),
        env=on_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    bs_a = f"h03-a-{port}"
    bs_b = f"h03-b-{port}"
    try:
        _wait_health(base_url)
        _ab(bs_a, "set", "viewport", "1380", "900")
        _ab(bs_a, "open", f"{base_url}/workspace/")
        sess_a = _wait_for_js(bs_a, "document.getElementById('sessions')?.value || null", desc="session A")

        # Scope session A to cs-ctrl-h ONLY.
        _ab_eval(
            bs_a,
            f"(() => {{ const r = [...document.querySelectorAll('#catalog .catalog-row')]"
            f".find(el => el.textContent.includes('{MAIN_COURSE}')); if (r) r.click(); return !!r; }})()",
        )
        _wait_for_js(
            bs_a,
            f"document.getElementById('scope-summary')?.textContent?.includes('{MAIN_COURSE}')",
            desc="scoped to cs-ctrl-h",
        )

        # Import cs-ctrl-h (4 files) and cs-other-h (1 distractor file).
        _ab(bs_a, "eval", "document.getElementById('import-panel').open = true")
        _ab(bs_a, "select", "#import-course", MAIN_COURSE)
        _ab(
            bs_a,
            "upload",
            "#import-files",
            str(stage / "Lecture-01-Core-Theory.pdf"),
            str(stage / "Slides-02-Method-Walkthrough.pptx"),
            str(stage / "Lab-01-Experiments.ipynb"),
            str(stage / "NoEdge-Notes.pdf"),
        )
        _ab(bs_a, "click", "#import-button")
        _wait_for_js(
            bs_a,
            "(() => { const li = [...document.querySelectorAll('#import-results li')]; "
            "return li.length === 4 && li.every(x => x.textContent.includes('导入完成')); })()",
            timeout=60.0,
            desc="4 main course files imported",
        )

        _ab(bs_a, "select", "#import-course", OTHER_COURSE)
        _ab(bs_a, "upload", "#import-files", str(stage / "OtherCourse-Distractor.ipynb"))
        _ab(bs_a, "click", "#import-button")
        _wait_for_js(
            bs_a,
            "(() => { const li = [...document.querySelectorAll('#import-results li')]; "
            "return li.length === 1 && li.every(x => x.textContent.includes('导入完成')); })()",
            timeout=30.0,
            desc="other course distractor imported",
        )

        # Ensure session A scope stays cs-ctrl-h only!
        scope_check = _ab_eval(
            bs_a,
            f"(async () => {{ const r = await fetch('/api/workspace/sessions/{sess_a}'); "
            "return await r.json(); })()",
        )
        assert scope_check["scope"]["course_ids"] == [MAIN_COURSE], scope_check["scope"]

        # M2 Entry 1: Open #kg-panel -> inspect concept list -> click C1 -> open Lecture page & Lab code/output PNG.
        _ab(bs_a, "eval", "document.getElementById('kg-panel').open = true")
        kg_list = _wait_for_js(
            bs_a,
            "(() => { const items = [...document.querySelectorAll('#kg-list .kg-item')]; "
            "return items.length >= 6 ? items.map(el => ({key: el.dataset.conceptKey, text: el.textContent})) : null; })()",
            desc="Entry 1 concept list loaded",
        )
        joined_list = " ".join(item["text"] for item in kg_list)
        assert "Smith" not in joined_list and "零相位" not in joined_list and "量化舍入" not in joined_list, joined_list

        c1_key = next(item["key"] for item in kg_list if "极点配置滞后补偿" in item["text"])
        c2_key = next(item["key"] for item in kg_list if "扩展卡尔曼协方差膨胀" in item["text"])
        c6_key = next(item["key"] for item in kg_list if "混合灵敏度" in item["text"])

        _ab_click(bs_a, f"#kg-list .kg-item[data-concept-key='{c1_key}']")
        c1_detail_txt = _wait_for_js(
            bs_a,
            "(() => { const d = document.getElementById('kg-detail'); "
            "return d && !d.hidden && d.textContent.includes('Lecture-01-Core-Theory.pdf') "
            "&& d.textContent.includes('Lab-01-Experiments.ipynb') ? d.textContent : null; })()",
            desc="C1 detail loaded via Entry 1",
        )
        assert "0.72" in c1_detail_txt and "4.15%" in c1_detail_txt, c1_detail_txt
        _ab(bs_a, "screenshot", str(shots / "h03-m2-entry1-detail.png"))

        # Click "打开讲义原页"
        _ab_eval(
            bs_a,
            "(() => { const b = [...document.querySelectorAll('#kg-detail button')]"
            ".find(x => x.textContent.trim() === '打开讲义原页'); b.click(); return true; })()",
        )
        _wait_for_js(
            bs_a,
            "(() => { const r = document.getElementById('reader'); const img = r?.querySelector('img'); "
            "const t = document.getElementById('reader-title')?.textContent || ''; "
            "return r && r.open && t.includes('Lecture-01-Core-Theory.pdf') && img && img.complete && img.naturalWidth > 0; })()",
            desc="Entry 1 opened Lecture-01 page 1 in #reader",
        )
        _ab(bs_a, "screenshot", str(shots / "h03-m2-entry1-lecture.png"))

        # Click "打开实验 Cell" and then "查看输出图"
        _ab_eval(
            bs_a,
            "(() => { const b = [...document.querySelectorAll('#kg-detail button')]"
            ".find(x => x.textContent.trim() === '打开实验 Cell'); b.click(); return true; })()",
        )
        _wait_for_js(
            bs_a,
            "(() => { const r = document.getElementById('reader'); "
            "const t = document.getElementById('reader-title')?.textContent || ''; "
            "const body = document.getElementById('reader-body')?.textContent || ''; "
            "return r && r.open && t.includes('Lab-01-Experiments.ipynb') && body.includes('run_loop_trim_p1'); })()",
            desc="Entry 1 opened Lab cell code",
        )
        _ab_eval(
            bs_a,
            "(() => { const b = [...document.querySelectorAll('#kg-detail button')]"
            ".find(x => x.textContent.trim() === '查看输出图'); b.click(); return true; })()",
        )
        _wait_for_js(
            bs_a,
            "(() => { const r = document.getElementById('reader'); const img = r?.querySelector('img'); "
            "const t = document.getElementById('reader-title')?.textContent || ''; "
            "return r && r.open && t.includes('Lab-01-Experiments.ipynb') && img && img.complete && img.naturalWidth > 0; })()",
            desc="Entry 1 opened Lab child_output PNG",
        )
        _ab(bs_a, "screenshot", str(shots / "h03-m2-entry1-lab-png.png"))

        # Clear file scope back to full course cs-ctrl-h
        _ab_eval(bs_a, "(() => { const b = document.getElementById('clear-file'); if (b && !b.hidden) b.click(); return true; })()")
        _wait_for_js(bs_a, "document.getElementById('clear-file').hidden === true", desc="clear file scope")

        # M3: No-Edge concept (C6) explicit notice + fallback search
        _ab_click(bs_a, f"#kg-list .kg-item[data-concept-key='{c6_key}']")
        c6_notice = _wait_for_js(
            bs_a,
            "(() => { const d = document.getElementById('kg-detail'); "
            "return d && d.querySelector('.kg-fallback-search') ? d.textContent : null; })()",
            desc="C6 no-edge explicit notice + fallback button",
        )
        assert "暂无可靠的实验演示关系" in c6_notice, c6_notice
        msg_n_before = _ab_eval(bs_a, "document.querySelectorAll('#messages article').length")
        _ab_click(bs_a, "#kg-detail .kg-fallback-search")
        c6_answer = _wait_for_js(
            bs_a,
            f"(() => {{ const arts = [...document.querySelectorAll('#messages article')]; "
            f"if (arts.length < {msg_n_before + 2}) return null; "
            "const st = document.getElementById('composer-status')?.textContent || ''; "
            "if (st !== '' || document.querySelector('#turn-actions button')) return null; "
            "return arts.at(-1)?.textContent || null; })()",
            timeout=60.0,
            desc="C6 fallback search answered",
        )
        assert "run_hinf_other_x9" not in c6_answer and "1.18" not in c6_answer, c6_answer
        _ab(bs_a, "screenshot", str(shots / "h03-m3-no-edge-fallback.png"))

        # M2 Entry 2: Ask H_Q01 (cross-file) -> verify answer + concept chip -> click chip opens detail & source!
        q1_text = "课程里“极点配置滞后补偿”成立需要满足哪些数学条件？对应实验在什么采样周期下实测得到了怎样的阻尼比、超调量和调节时间？"
        msg_n_before = _ab_eval(bs_a, "document.querySelectorAll('#messages article').length")
        _ab(bs_a, "fill", "#question", q1_text)
        _ab(bs_a, "click", "#send")
        q1_ans = _wait_for_js(
            bs_a,
            f"(() => {{ const arts = [...document.querySelectorAll('#messages article')]; "
            f"if (arts.length < {msg_n_before + 2}) return null; "
            "const st = document.getElementById('composer-status')?.textContent || ''; "
            "if (st !== '' || document.querySelector('#turn-actions button')) return null; "
            "const last = arts.at(-1); "
            "return last.querySelector('.kg-concept-chip') ? last.textContent : null; })()",
            timeout=60.0,
            desc="H_Q01 answer & concept chip rendered",
        )
        assert "0.72" in q1_ans and "4.15" in q1_ans and "185" in q1_ans, q1_ans

        # Also ask H_Q02 on v1 so we have a v1 historical message with C2 chip & v1 citation!
        q2_text = "“扩展卡尔曼协方差膨胀”在什么条件下触发，实验里未膨胀与开启膨胀后的 RMSE 和卡方拒收率分别是多少？"
        msg_n_before = _ab_eval(bs_a, "document.querySelectorAll('#messages article').length")
        _ab(bs_a, "fill", "#question", q2_text)
        _ab(bs_a, "click", "#send")
        q2_ans = _wait_for_js(
            bs_a,
            f"(() => {{ const arts = [...document.querySelectorAll('#messages article')]; "
            f"if (arts.length < {msg_n_before + 2}) return null; "
            "const st = document.getElementById('composer-status')?.textContent || ''; "
            "if (st !== '' || document.querySelector('#turn-actions button')) return null; "
            "return arts.at(-1)?.textContent || null; })()",
            timeout=60.0,
            desc="H_Q02 v1 answer rendered",
        )
        assert "0.418" in q2_ans and "0.064" in q2_ans, q2_ans

        # Click C1 chip inside message -> opens #kg-detail -> click "打开讲义原页"
        _ab(bs_a, "eval", "document.getElementById('kg-detail').hidden = true")
        _ab_click(bs_a, f"#messages .kg-concept-chip[data-concept-key='{c1_key}']")
        _wait_for_js(
            bs_a,
            "document.getElementById('kg-detail')?.textContent?.includes('0.72')",
            desc="Entry 2 message chip opened C1 detail",
        )
        _ab(bs_a, "screenshot", str(shots / "h03-m2-entry2-chip.png"))

        # M4: Replace Lab-01 with v2 via #import-form
        _ab(bs_a, "eval", "document.getElementById('import-panel').open = true")
        _ab(bs_a, "select", "#import-course", MAIN_COURSE)
        lab_doc_info = _ab_eval(
            bs_a,
            f"(async () => {{ const r = await fetch('/api/workspace/sessions/{sess_a}/catalog'); "
            f"const j = await r.json(); const c = j.courses.find(x => x.course_id === '{MAIN_COURSE}'); "
            "return c.documents.find(d => d.file_name === 'Lab-01-Experiments.ipynb'); })()",
        )
        clicked_lab = _ab_eval(
            bs_a,
            f"(() => {{ const r = document.querySelector('#catalog [data-node-id=\"{lab_doc_info['node_id']}\"]'); "
            "if (r) r.click(); return !!r; })()",
        )
        assert clicked_lab, "Lab-01-Experiments.ipynb row not found in #catalog"
        _wait_for_js(bs_a, "!document.getElementById('replace-label').hidden", desc="replace checkbox visible")
        _ab(bs_a, "check", "#replace-document")
        _ab(bs_a, "upload", "#import-files", str(stage / "Lab-01-Experiments-v2.ipynb"))
        _ab(bs_a, "click", "#import-button")
        _wait_for_js(
            bs_a,
            "document.getElementById('import-results')?.textContent?.includes('v2')",
            timeout=30.0,
            desc="Lab-01 v2 uploaded via UI",
        )
        _ab_eval(bs_a, "(() => { const b = document.getElementById('clear-file'); if (b && !b.hidden) b.click(); return true; })()")
        _wait_for_js(bs_a, "document.getElementById('clear-file').hidden === true", desc="restore full course scope")

        # Verify KG detail for C1 now shows v2 (0.84 / 1.42% / 132.0), and C2 has partial fallback (0 demo edges)!
        _ab_click(bs_a, f"#kg-list .kg-item[data-concept-key='{c1_key}']")
        c1_v2_detail = _wait_for_js(
            bs_a,
            "(() => { const t = document.getElementById('kg-detail')?.textContent || ''; "
            "return t.includes('0.84') && t.includes('v2') ? t : null; })()",
            desc="C1 KG detail updated to v2",
        )
        assert "0.72" not in c1_v2_detail, c1_v2_detail

        _ab_click(bs_a, f"#kg-list .kg-item[data-concept-key='{c2_key}']")
        c2_v2_detail = _wait_for_js(
            bs_a,
            "(() => { const d = document.getElementById('kg-detail'); "
            "if (!d || !d.querySelector('.kg-fallback-search')) return null; "
            "const groups = d.querySelectorAll('.kg-group'); "
            "return { full: d.textContent, demo: groups[1]?.textContent || '' }; })()",
            desc="C2 KG detail now reports no demo edge after v2",
        )
        assert "当前版本暂无可靠记录" in c2_v2_detail["demo"], c2_v2_detail
        assert "Lab-01-Experiments" not in c2_v2_detail["demo"], c2_v2_detail
        assert "eval_res_scale_k7" not in c2_v2_detail["demo"], c2_v2_detail

        # Ask H_Q09 on v2 -> verify v2 numbers (0.84 / 1.42% / 132.0) and ZERO v1 numbers!
        q9_text = "更新实验版本后，“极点配置滞后补偿”在新版实验中的采样周期、阻尼比 zeta、超调量和调节时间变成了多少？"
        msg_n_before = _ab_eval(bs_a, "document.querySelectorAll('#messages article').length")
        _ab(bs_a, "fill", "#question", q9_text)
        _ab(bs_a, "click", "#send")
        q9_ans = _wait_for_js(
            bs_a,
            f"(() => {{ const arts = [...document.querySelectorAll('#messages article')]; "
            f"if (arts.length < {msg_n_before + 2}) return null; "
            "const st = document.getElementById('composer-status')?.textContent || ''; "
            "if (st !== '' || document.querySelector('#turn-actions button')) return null; "
            "return arts.at(-1)?.textContent || null; })()",
            timeout=60.0,
            desc="H_Q09 v2 answer rendered",
        )
        assert "0.84" in q9_ans and "1.42" in q9_ans and "132" in q9_ans, q9_ans
        assert "0.72" not in q9_ans and "4.15" not in q9_ans, q9_ans

        # Click historical v1 Lab citation button -> opens v1 historical version in #reader!
        _ab_eval(
            bs_a,
            "(() => { const b = [...document.querySelectorAll('#messages .citations button')]"
            ".find(x => x.textContent.includes('Lab-01-Experiments.ipynb') && x.textContent.includes('v1')); "
            "if (!b) throw new Error('missing Lab v1 citation'); b.click(); return true; })()",
        )
        v1_reader = _wait_for_js(
            bs_a,
            "(() => { const r = document.getElementById('reader'); "
            "const t = document.getElementById('reader-title')?.textContent || ''; "
            "const loc = document.getElementById('reader-location')?.textContent || ''; "
            "const body = document.getElementById('reader-body')?.textContent || ''; "
            "return r && r.open && t.includes('v1') && loc.includes('历史版本') ? body : null; })()",
            desc="historical v1 evidence opened in #reader",
        )
        _ab(bs_a, "screenshot", str(shots / "h03-m4-v1-history.png"))

        # M4 Delete & Stale Historical Chip Graceful Degradation:
        # First ask H_Q08 so an assistant message has a C6 chip anchored to NoEdge-Notes.pdf,
        # then delete NoEdge-Notes.pdf via local library API -> reload -> click stale C6 chip -> graceful fallback, NO global error banner!
        q8_text = "混合灵敏度鲁棒回路整形在 H_inf 范数与低频增益方面需要满足哪些数学条件？课程里是否提供了对应实验代码？"
        msg_n_before = _ab_eval(bs_a, "document.querySelectorAll('#messages article').length")
        _ab(bs_a, "fill", "#question", q8_text)
        _ab(bs_a, "click", "#send")
        _wait_for_js(
            bs_a,
            f"(() => {{ const arts = [...document.querySelectorAll('#messages article')]; "
            f"if (arts.length < {msg_n_before + 2}) return null; "
            "const st = document.getElementById('composer-status')?.textContent || ''; "
            "if (st !== '' || document.querySelector('#turn-actions button')) return null; "
            f"return arts.at(-1).querySelector('.kg-concept-chip[data-concept-key=\"{c6_key}\"]') ? true : null; }})()",
            timeout=60.0,
            desc="H_Q08 rendered C6 chip before delete",
        )
        noedge_doc_info = _ab_eval(
            bs_a,
            f"(async () => {{ const r = await fetch('/api/workspace/sessions/{sess_a}/catalog'); "
            f"const j = await r.json(); const c = j.courses.find(x => x.course_id === '{MAIN_COURSE}'); "
            "return c.documents.find(d => d.file_name === 'NoEdge-Notes.pdf'); })()",
        )
        del_req = urllib.request.Request(
            f"{base_url}/api/rag/local/documents/{noedge_doc_info['document_id']}",
            method="DELETE",
        )
        with urllib.request.urlopen(del_req, timeout=5.0) as del_resp:
            assert del_resp.status == 200
        # Click the rendered C6 chip right after backend deletion -> 404 degrades cleanly in-panel, NO global error banner!
        _ab(bs_a, "eval", "document.getElementById('kg-detail').hidden = true")
        _ab_click(bs_a, f"#messages .kg-concept-chip[data-concept-key='{c6_key}']")
        _wait_for_js(
            bs_a,
            "(() => { const d = document.getElementById('kg-detail'); "
            "const st = document.getElementById('workspace-status'); "
            "return d && !d.hidden && d.querySelector('.kg-fallback-search') !== null "
            "&& st && st.hidden === true; })()",
            desc="deleted concept chip graceful in-panel fallback without global error banner",
        )
        _ab(bs_a, "screenshot", str(shots / "h03-m4-stale-chip-fallback.png"))
        _ab(bs_a, "reload")
        _wait_for_js(
            bs_a,
            f"(() => {{ const list = [...document.querySelectorAll('#kg-list .kg-item')].map(el => el.dataset.conceptKey); "
            f"const txt = [...document.querySelectorAll('#messages article')].at(-1)?.textContent || ''; "
            f"return list.length > 0 && !list.includes('{c6_key}') && txt.includes('此回答的来源已不可访问'); }})()",
            desc="after reload deleted source message inaccessible and C6 absent from #kg-list",
        )

        # M5: Independent Browser Session B + unauth + cross-course 404 checks
        _ab(bs_b, "open", f"{base_url}/workspace/")
        sess_b = _wait_for_js(bs_b, "document.getElementById('sessions')?.value || null", desc="session B")
        cross_statuses = _ab_eval(
            bs_b,
            "(async () => { "
            f"const r1 = await fetch('/api/workspace/sessions/{sess_a}/kg/concepts'); "
            f"const r2 = await fetch('/api/workspace/sessions/{sess_a}/kg/concepts/{c1_key}'); "
            "return [r1.status, r2.status]; })()",
        )
        assert cross_statuses == [404, 404], cross_statuses
        cross_course_status = _ab_eval(
            bs_a,
            "(async () => { "
            f"const r1 = await fetch('/api/workspace/sessions/{sess_a}/kg/concepts?course_id={OTHER_COURSE}'); "
            f"const r2 = await fetch('/api/workspace/sessions/{sess_a}/kg/concepts/{c1_key}?course_id={OTHER_COURSE}'); "
            "return [r1.status, r2.status]; })()",
        )
        assert cross_course_status == [404, 404], cross_course_status
        unauth_status, _ = _http_status(f"{base_url}/api/workspace/sessions/{sess_a}/kg/concepts")
        assert unauth_status == 404, unauth_status

        results["online_browser_gate"] = {
            "session_a": sess_a,
            "session_b": sess_b,
            "entry1_concepts_count": len(kg_list),
            "entry1_lecture_and_lab_png": True,
            "m3_c6_no_edge_fallback_passed": True,
            "m2_entry2_chip_passed": True,
            "m4_v2_update_and_v1_history_passed": True,
            "m4_v1_reader_len": len(v1_reader),
            "m5_cross_session_404": cross_statuses,
            "m5_cross_course_404": cross_course_status,
            "m5_unauth_404": unauth_status,
        }
        return results
    finally:
        for bs in (bs_a, bs_b):
            subprocess.run(["agent-browser", "--session", bs, "close"], check=False, capture_output=True)
        srv.terminate()
        srv.wait(timeout=5)


# ---------------------------------------------------------------------------
# M6: Same-Budget 3-Way Comparison (S0_base vs S0_adj vs S0_kg)
# ---------------------------------------------------------------------------

FACT_REGEX: dict[str, list[tuple[str, str]]] = {
    "H_Q01": [
        ("cond_controllable", r"能控|rank\(C_aug\)|controllab"),
        ("cond_ts", r"0\.05\s*s"),
        ("cond_pole", r"0\.85"),
        ("fn_p1", r"run_loop_trim_p1"),
        ("ts_020", r"0\.020?\s*s"),
        ("zeta_72", r"0\.72"),
        ("overshoot_415", r"4\.15\s*%"),
        ("settling_185", r"185(?:\.0)?"),
    ],
    "H_Q02": [
        ("cond_rk", r"R_k\s*>\s*0|正定"),
        ("cond_chi2", r"7\.81"),
        ("cond_3steps", r"连续\s*3\s*步|3\s*步"),
        ("fn_k7", r"eval_res_scale_k7"),
        ("rmse_unscaled", r"0\.418"),
        ("rmse_scaled", r"0\.064"),
        ("reject_rate", r"2\.30?\s*%"),
    ],
    "H_Q03": [
        ("cond_dist", r"d_max|无穷范数|有界"),
        ("cond_phi", r"2\s*\*\s*eps\s*\*\s*T_s|2\\epsilon T_s|phi\s*>=\s*2"),
        ("fn_m4", r"probe_band_law_m4"),
        ("osc_018", r"0\.018"),
        ("band_024", r"0\.024"),
        ("hz_142", r"14\.2"),
    ],
    "H_Q04": [
        ("formula_r", r"B_-3dB|f_s|1\s*-\s*\(?pi"),
        ("r_09529", r"0\.9529"),
        ("q_800", r"8\.00?"),
        ("fn_b2", r"verify_iir_dip_b2"),
        ("dip_314", r"-?31\.4\s*dB"),
        ("ripple_018", r"0\.18\s*dB"),
    ],
    "H_Q05": [
        ("cond_sat", r"u_max|饱和"),
        ("cond_sign", r"同号|e_k\s*\*\s*u_sat\s*>\s*0"),
        ("clamp_48", r"4\.8\s*%"),
        ("clamp_190", r"190(?:\.0)?\s*ms"),
        ("bc_94", r"9\.4\s*%"),
        ("bc_265", r"265(?:\.0)?\s*ms"),
    ],
    "H_Q06": [
        ("negated", r"没有|并未|并非|不是|仅在否定|does NOT|不演示"),
        ("fft", r"numpy\.fft|fft|耗时|计时"),
        ("us_684", r"68\.4"),
    ],
    "H_Q07": [
        ("unrun", r"未运行|尚未运行|未执行|无保存输出|空输出|NotImplementedError"),
        ("smith_mention", r"仅作背景|仅.*提及|未定义|无对应实验"),
    ],
    "H_Q08": [
        ("norm_gamma", r"\|\|M\(s\)\|\|_inf\s*<\s*gamma|M\(s\).*gamma|20\s*dB/dec|-26\s*dB"),
        ("no_demo", r"没有|未安排|无.*实验"),
    ],
    "H_Q09": [
        ("ts_010", r"0\.010?\s*s"),
        ("ki_195", r"1\.95"),
        ("zeta_84", r"0\.84"),
        ("overshoot_142", r"1\.42\s*%"),
        ("settling_132", r"132(?:\.0)?"),
    ],
    "H_Q10": [
        ("absent", r"没有|未找到|不存在|未检索到|不包含|无相关|并未"),
    ],
}

FORBIDDEN_REGEX: dict[str, list[str]] = {
    "H_Q01": [r"0\.51", r"16\.80%", r"420\.0", r"0\.84", r"1\.42%"],
    "H_Q02": [r"0\.890", r"0\.295", r"11\.40%", r"68\.4\s*us"],
    "H_Q03": [],
    "H_Q04": [r"68\.4\s*us"],
    "H_Q05": [],
    "H_Q06": [],
    "H_Q07": [],
    "H_Q08": [r"run_hinf_other_x9", r"1\.18", r"54\.2"],
    "H_Q09": [r"0\.72", r"4\.15%", r"185\.0", r"0\.51", r"16\.80%"],
    "H_Q10": [r"gamma_achieved\s*=\s*1\.18", r"phase_margin_deg\s*=\s*54\.2", r"16\.80%", r"420\.0"],
}


def _run_arm(
    arm_name: str,
    *,
    include_adjacency: bool,
    kg_enabled: bool,
    work_dir: Path,
    criteria: dict[str, Any],
    env_map: dict[str, str],
) -> dict[str, Any]:
    from app.config import Settings
    from app.rag.workspace import make_workspace
    from app.schemas.workspace import CreateSession, TurnRequest

    arm_dir = work_dir / f"arm_{arm_name}"
    if arm_dir.exists():
        shutil.rmtree(arm_dir)
    arm_dir.mkdir(parents=True, exist_ok=True)
    stage = arm_dir / "courses" / "stage"
    stage.mkdir(parents=True, exist_ok=True)
    for name in (
        "Lecture-01-Core-Theory.pdf",
        "Slides-02-Method-Walkthrough.pptx",
        "Lab-01-Experiments.ipynb",
        "Lab-01-Experiments-v2.ipynb",
        "NoEdge-Notes.pdf",
        "OtherCourse-Distractor.ipynb",
    ):
        shutil.copy2(CORPUS_DIR / name, stage / name)

    settings = Settings(
        sqlite_path=arm_dir / "srag.sqlite3",
        data_dir=arm_dir / "data",
        course_root=arm_dir / "courses",
        workspace_course_ids=f"{MAIN_COURSE},{OTHER_COURSE}",
        workspace_answer_enabled=True,
        workspace_retrieval_strategy="s0",
        workspace_retrieval_pool_cap=12,
        workspace_retrieval_budget=6,
        workspace_input_token_limit=32_768,
        workspace_vector_retrieval_enabled=False,
        workspace_retrieval_include_adjacency=include_adjacency,
        workspace_kg_retrieval_enabled=kg_enabled,
        source_vision_enabled=False,
        deepseek_api_key=env_map["DEEPSEEK_API_KEY"],
        deepseek_base_url=env_map.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_query_model="deepseek-flash",
        deepseek_chat_model="deepseek-flash",
        srag_soffice=Path("/opt/homebrew/bin/soffice"),
    )
    ws = make_workspace(settings)
    lib = ws.library

    res_main = lib.import_paths(
        MAIN_COURSE,
        [
            stage / "Lecture-01-Core-Theory.pdf",
            stage / "Slides-02-Method-Walkthrough.pptx",
            stage / "Lab-01-Experiments.ipynb",
            stage / "NoEdge-Notes.pdf",
        ],
    )
    lib.import_paths(OTHER_COURSE, [stage / "OtherCourse-Distractor.ipynb"])
    lab_doc_id = next(r.document_id for r in res_main if "Lab-01-Experiments.ipynb" in r.source_path)
    doc_name_by_id = {r.document_id: Path(r.source_path).name for r in res_main}

    v1_questions = [q for q in criteria["eval_questions"] if q["snapshot"] == "v1"]
    v2_questions = [q for q in criteria["eval_questions"] if q["snapshot"] == "v2"]

    per_q: list[dict[str, Any]] = []
    latencies: list[float] = []

    def _eval_question(q: dict[str, Any], idx: int) -> dict[str, Any]:
        # Fresh single-turn session per question so history never leaks across questions.
        sess, owner = ws.create(None, CreateSession(course_ids=[MAIN_COURSE]))
        t0 = time.perf_counter()
        out = ws.submit(
            owner,
            sess["session_id"],
            TurnRequest(
                client_turn_id=f"{arm_name}-{q['question_id']}-{idx}",
                kind="ask",
                text=q["question"],
                scope_revision=sess["scope_revision"],
            ),
        )
        dt = time.perf_counter() - t0
        latencies.append(dt)
        turn = out["turn"]
        msg = out.get("message") or {}
        answer_text = "\n".join(
            b.get("text", "") for b in msg.get("blocks", []) if b.get("type") in {"text", "notice"}
        )
        cited_ids = {
            cid
            for b in msg.get("blocks", [])
            if b.get("type") == "text"
            for cid in b.get("citations", [])
        }
        cited_sources = []
        for cid in cited_ids:
            ev = (msg.get("sources") or {}).get(cid)
            if ev:
                src = ev["revision"]["source"]
                cited_sources.append(
                    (doc_name_by_id.get(src["document_id"], src["document_id"]), src["source_id"], src["version"])
                )
        read_sources = []
        for ev in (msg.get("sources") or {}).values():
            src = ev["revision"]["source"]
            read_sources.append(
                (doc_name_by_id.get(src["document_id"], src["document_id"]), src["source_id"], src["version"])
            )

        # Check expected_sources hit in read_sources / cited_sources:
        exp_pairs = [(s["file_name"], s["source_id"]) for s in q["expected_sources"]]
        read_pair_set = {(f, sid) for f, sid, _ in read_sources}
        cited_pair_set = {(f, sid) for f, sid, _ in cited_sources}
        # Special handling for H_Q06 / H_Q07 where refusal without citing cell:8 is also valid
        all_expected_read = all(p in read_pair_set for p in exp_pairs) if exp_pairs else True
        any_expected_cited = (
            all(p in cited_pair_set for p in exp_pairs) if exp_pairs else True
        )

        # Fact matching
        fact_checks = {
            name: bool(re.search(pat, answer_text, re.IGNORECASE))
            for name, pat in FACT_REGEX[q["question_id"]]
        }
        facts_passed = all(fact_checks.values())
        forbidden_hits = [
            pat for pat in FORBIDDEN_REGEX[q["question_id"]] if re.search(pat, answer_text, re.IGNORECASE)
        ]

        # Severe error check: hallucinated forbidden numbers / cross-course leak / unverified citation
        severe_error = bool(forbidden_hits) or turn["status"] != "completed"
        if q["question_id"] in {"H_Q01", "H_Q02", "H_Q03", "H_Q04", "H_Q05", "H_Q09"}:
            passed = facts_passed and all_expected_read and not severe_error
        elif q["question_id"] == "H_Q08":
            passed = facts_passed and ("NoEdge-Notes.pdf", "page:1") in read_pair_set and not severe_error
        elif q["question_id"] in {"H_Q06", "H_Q07", "H_Q10"}:
            passed = facts_passed and not severe_error
        else:
            passed = facts_passed and not severe_error

        trace = turn.get("trace") or {}
        ret = trace.get("retrieval") or {}
        return {
            "question_id": q["question_id"],
            "category": q["category"],
            "snapshot": q["snapshot"],
            "passed": passed,
            "severe_error": severe_error,
            "all_expected_read": all_expected_read,
            "all_expected_cited": any_expected_cited,
            "facts_passed": facts_passed,
            "fact_checks": fact_checks,
            "forbidden_hits": forbidden_hits,
            "latency_s": round(dt, 3),
            "external_call_count": trace.get("external_call_count", 0),
            "kg_trace": ret.get("kg"),
            "read_sources": read_sources,
            "cited_sources": cited_sources,
            "related_concepts": msg.get("related_concepts", []),
            "answer_preview": answer_text[:400],
        }

    for idx, q in enumerate(v1_questions, start=1):
        per_q.append(_eval_question(q, idx))

    # Upgrade Lab-01 -> v2 and evaluate v2 questions
    lib.import_paths(MAIN_COURSE, [stage / "Lab-01-Experiments-v2.ipynb"], document_id=lab_doc_id)
    for idx, q in enumerate(v2_questions, start=1):
        per_q.append(_eval_question(q, idx))

    lat_sorted = sorted(latencies)
    p50 = statistics.median(lat_sorted) if lat_sorted else 0.0
    p95 = lat_sorted[min(len(lat_sorted) - 1, max(0, int(len(lat_sorted) * 0.95) - 1))] if lat_sorted else 0.0
    cross_zero = [r for r in per_q if r["question_id"] in {"H_Q01", "H_Q02", "H_Q03", "H_Q04"}]
    return {
        "arm": arm_name,
        "include_adjacency": include_adjacency,
        "kg_enabled": kg_enabled,
        "total_passed": sum(1 for r in per_q if r["passed"]),
        "total_questions": len(per_q),
        "cross_zero_overlap_passed": sum(1 for r in cross_zero if r["passed"]),
        "cross_zero_overlap_total": len(cross_zero),
        "expected_sources_read_passed": sum(1 for r in per_q if r["all_expected_read"]),
        "severe_errors": sum(1 for r in per_q if r["severe_error"]),
        "p50_latency_s": round(p50, 3),
        "p95_latency_s": round(p95, 3),
        "total_external_calls": sum(r["external_call_count"] for r in per_q),
        "per_question": per_q,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H03 independent blind gate verification.")
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/srag-h03-gate-run"))
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    archive_dir = args.archive_dir.resolve()
    if str(archive_dir) not in sys.path:
        sys.path.insert(0, str(archive_dir))

    env_map = _load_env_file(DEFAULT_ENV_PATH)
    assert env_map.get("DEEPSEEK_API_KEY"), "DEEPSEEK_API_KEY missing from .env"
    criteria = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))

    work_dir = args.work_dir.resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Running M1 + M4 lifecycle + M5 API 404 checks...")
    m1_m4 = run_m1_and_lifecycle(archive_dir, work_dir, criteria)

    print("[2/3] Running real-Chromium dual-entry + fallback + v2 + 404 browser gate...")
    browser_res = run_browser_gate(archive_dir, work_dir, env_map)

    print("[3/3] Running M6 same-budget 3-way comparison (S0_base / S0_adj / S0_kg)...")
    arms = []
    for arm_name, adj, kg_flag in [
        ("S0_base", False, False),
        ("S0_adj", True, False),
        ("S0_kg", True, True),
    ]:
        print(f"  -> Running arm {arm_name} (include_adjacency={adj}, kg_enabled={kg_flag})...")
        arms.append(
            _run_arm(
                arm_name,
                include_adjacency=adj,
                kg_enabled=kg_flag,
                work_dir=work_dir,
                criteria=criteria,
                env_map=env_map,
            )
        )

    payload = {
        "gate_phase": "H03_independent_blind_gate",
        "executed_at": datetime.now(UTC).isoformat(),
        "archive_dir": str(archive_dir),
        "m1_m4_m5_offline": m1_m4,
        "m2_m3_m4_m5_browser": browser_res,
        "m6_three_way_comparison": arms,
    }
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SAVED: {args.out_json}")
    for a in arms:
        print(
            f"ARM {a['arm']}: {a['total_passed']}/{a['total_questions']} "
            f"(cross_zero={a['cross_zero_overlap_passed']}/{a['cross_zero_overlap_total']}, "
            f"severe={a['severe_errors']}, p50={a['p50_latency_s']}s, p95={a['p95_latency_s']}s)"
        )


if __name__ == "__main__":
    main()
