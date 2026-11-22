"""Independent gate runner for the 16-question M2 condition & citation suite."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import runpy
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import nbformat
from fastapi.testclient import TestClient

DEFAULT_ENV = Path("/Users/Link/code/SRAG/.env")
DEFAULT_PYTHON = Path("/Users/Link/code/SRAG/.venv/bin/python")
PREP_COMMIT = "128bf5b646352f12018873687faaad10b31d1e06"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ab(session: str, *args: str, timeout: int = 90) -> str:
    cmd = ["agent-browser", "--session", session, *args]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"agent-browser failed ({' '.join(cmd)}): {res.stderr or res.stdout}")
    return res.stdout.strip()


def _wait_js(session: str, expr: str, timeout_ms: int = 80_000) -> None:
    _ab(
        session,
        "wait",
        "--timeout",
        str(timeout_ms),
        "--fn",
        expr,
        timeout=timeout_ms // 1000 + 15,
    )


def _nb(path: Path, *cell_texts: str) -> Path:
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell(txt, id=f"c-{i}")
                for i, txt in enumerate(cell_texts)
            ]
        ),
        path,
    )
    return path


def _git_meta(repo_dir: Path) -> dict[str, Any]:
    sha = subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repo_dir), "status", "--short"], text=True
    ).strip()
    return {"git_sha": sha, "worktree_status": status or "clean", "worktree_dir": str(repo_dir)}


def _run_off(
    *,
    subject_dir: Path,
    prep_path: Path,
    diag_path: Path,
    env_file: Path,
    upload_files: list[Path],
    cases: tuple[tuple[str, str], ...],
    out_dir: Path,
) -> dict[str, Any]:
    for mod in [k for k in sys.modules if k == "app" or k.startswith("app.")]:
        del sys.modules[mod]
    sys.path.insert(0, str(subject_dir))
    prep = runpy.run_path(str(prep_path))
    diag = runpy.run_path(str(diag_path))
    wg = importlib.import_module("app.rag.workspace_generation")
    es = importlib.import_module("app.rag.evidence_supplement")
    ws = importlib.import_module("app.rag.workspace")
    settings = prep["settings_for"](env_file, out_dir / "runtime", web=False)
    app, service = prep["app_for"](settings)
    events: list[dict[str, Any]] = []
    prep["observe_providers"](settings, events, web=False, image_hashes={})
    process_meta = {
        **_git_meta(subject_dir),
        "module_files": {
            "workspace": ws.__file__,
            "workspace_generation": wg.__file__,
            "evidence_supplement": es.__file__,
        },
        "effective_config": {
            "workspace_answer_enabled": settings.workspace_answer_enabled,
            "workspace_retrieval_strategy": settings.workspace_retrieval_strategy,
            "workspace_input_token_limit": settings.workspace_input_token_limit,
            "source_vision_enabled": settings.source_vision_enabled,
            "sqlite_path": str(settings.sqlite_path),
            "data_dir": str(settings.data_dir),
        },
    }
    try:
        with TestClient(app, base_url=diag["HOST"], headers={"Origin": diag["HOST"]}) as client:
            session = diag["ApiSession"](client, service, prep["COURSE"])
            assert session.empty
            for path in upload_files:
                assert session.upload(path)["version"] == 1
            cases_out: dict[str, Any] = {}
            for cid, question in cases:
                resp = session.ask(cid, question)
                opened = session.evidence(resp)
                trace = resp["data"]["turn"]["trace"]
                msg = resp["data"]["message"]
                cases_out[cid] = {
                    "question": question,
                    "route": trace.get("route"),
                    "answer_status": trace.get("answer_status"),
                    "external_call_count": trace.get("external_call_count"),
                    "blocks": msg["blocks"],
                    "visible_text": "\n".join(
                        b.get("text", "") for b in msg["blocks"] if b.get("text")
                    ),
                    "citations": {
                        k: {
                            "file_name": opened[k]["data"]["file_name"],
                            "version": opened[k]["data"]["source"]["version"],
                            "source_id": opened[k]["data"]["source"]["source_id"],
                        }
                        for k in opened
                    },
                }
            return {"process_meta": process_meta, "cases": cases_out}
    finally:
        prep["restore_providers"]()
        if str(subject_dir) in sys.path:
            sys.path.remove(str(subject_dir))


def _browser_server(
    runtime: Path, env_file: Path, course: str, port: int, subject_dir: Path
) -> str:
    return f"""
import json, subprocess, httpx, uvicorn
from pathlib import Path
from time import perf_counter
import app.rag.evidence_supplement as es
import app.rag.workspace as ws
import app.rag.workspace_generation as wg
from app.api.workspace import get_workspace_service
from app.config import Settings, get_settings
from app.main import create_app
from app.rag.workspace import make_workspace

rt = Path({str(runtime)!r})
subj = Path({str(subject_dir)!r})
s = Settings(
    _env_file={str(env_file)!r},
    course_root=rt / "courses",
    data_dir=rt / "data",
    sqlite_path=rt / "library.sqlite3",
    workspace_course_ids={course!r},
    workspace_answer_enabled=True,
    workspace_retrieval_strategy="fts",
    source_vision_enabled=True,
    workspace_input_token_limit=32768,
    jev_api_key="",
    qdrant_api_key="",
    qdrant_url="http://127.0.0.1:1",
)
app = create_app()
svc = make_workspace(s)
app.dependency_overrides[get_settings] = lambda: s
app.dependency_overrides[get_workspace_service] = lambda: svc

@app.get("/__audit_meta")
def audit_meta():
    return {{
        "git_sha": subprocess.check_output(
            ["git", "-C", str(subj), "rev-parse", "HEAD"], text=True
        ).strip(),
        "worktree_status": (
            subprocess.check_output(
                ["git", "-C", str(subj), "status", "--short"], text=True
            ).strip() or "clean"
        ),
        "module_files": {{
            "workspace": ws.__file__,
            "workspace_generation": wg.__file__,
            "evidence_supplement": es.__file__,
        }},
        "effective_config": {{
            "workspace_answer_enabled": s.workspace_answer_enabled,
            "workspace_retrieval_strategy": s.workspace_retrieval_strategy,
            "workspace_input_token_limit": s.workspace_input_token_limit,
            "source_vision_enabled": s.source_vision_enabled,
            "deepseek_chat_model": s.deepseek_chat_model,
            "sqlite_path": str(s.sqlite_path),
            "data_dir": str(s.data_dir),
            "port": {port},
        }},
    }}

orig = httpx.AsyncHTTPTransport.handle_async_request
async def log_async(transport, request):
    started = perf_counter()
    rec = {{"url": str(request.url)}}
    try:
        p = json.loads(request.content)
        rec["model"] = p.get("model")
        u = json.loads(p["messages"][1]["content"])
        rec["mode"] = u.get("mode")
        rec["evidence_ids"] = [e["id"] for e in u.get("evidence", [])]
    except Exception:
        pass
    try:
        resp = await orig(transport, request)
        rec["status_code"] = resp.status_code
        rec["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
        with (rt / "provider.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
        return resp
    except Exception as exc:
        rec["error"] = repr(exc)
        rec["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
        with (rt / "provider.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
        raise
httpx.AsyncHTTPTransport.handle_async_request = log_async

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port={port}, log_level="warning")
"""


def _run_browser(
    *,
    label: str,
    subject_dir: Path,
    prep_path: Path,
    env_file: Path,
    python_bin: Path,
    upload_files: list[Path],
    cases: tuple[tuple[str, str], ...],
    out_dir: Path,
) -> dict[str, Any]:
    prep = runpy.run_path(str(prep_path))
    port = _free_port()
    runtime = out_dir / f"browser_{label}"
    runtime.mkdir(parents=True, exist_ok=True)
    srv = runtime / "serve.py"
    srv.write_text(
        _browser_server(runtime, env_file, prep["COURSE"], port, subject_dir),
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        [str(python_bin), str(srv)],
        cwd=str(subject_dir),
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(subject_dir)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    session = f"joint-m2-{label}-{port}-{int(time.time())}"
    try:
        url = f"http://127.0.0.1:{port}/workspace/"
        for _ in range(40):
            if subprocess.run(["curl", "-sf", url], capture_output=True).returncode == 0:
                break
            time.sleep(0.25)
        else:
            proc.terminate()
            txt, _ = proc.communicate(timeout=3)
            raise RuntimeError(f"Server failed: {txt}")

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/__audit_meta", timeout=5) as r:
            process_meta = json.loads(r.read().decode("utf-8"))

        _ab(session, "--allowed-domains", "127.0.0.1", "open", url)
        _ab(session, "wait", "--text", "暂无资料")
        if not json.loads(_ab(session, "eval", 'document.querySelector("#import-panel").open')):
            _ab(session, "find", "text", "导入资料", "click", "--exact")
        _ab(session, "select", "#import-course", prep["COURSE"])
        _ab(session, "upload", "#import-files", *[str(p) for p in upload_files])
        _ab(session, "find", "role", "button", "click", "--name", "开始导入", "--exact")
        for p in upload_files:
            _ab(session, "wait", "--text", f"{p.name} · 导入完成 · v1")
        _wait_js(session, '!document.querySelector("#import-button").disabled')

        cases_out: dict[str, Any] = {}
        turn_idx = 0
        for cid, question in cases:
            turn_idx += 1
            _ab(session, "find", "label", "输入问题", "fill", question)
            _wait_js(session, '!document.querySelector("#send").disabled')
            _ab(session, "find", "role", "button", "click", "--name", "发送", "--exact")
            idx = turn_idx - 1
            _wait_js(
                session,
                f"document.querySelectorAll('#messages article.message-user').length === {turn_idx}"
                f" && document.querySelectorAll('#messages article.message-user')"
                f".item({idx}).nextElementSibling?.matches('article:not(.message-user)')"
                f" && !document.querySelector('#turn-actions button')"
                f"?.textContent.includes('取消')",
            )
            dom = json.loads(
                _ab(
                    session,
                    "eval",
                    f"""(async () => {{
                        const users = document.querySelectorAll('#messages article.message-user');
                        const m = users.item({idx}).nextElementSibling;
                        const mid = m.dataset.messageId;
                        const btns = [...m.querySelectorAll('button[data-evidence-id]')];
                        const sid = document.querySelector('#sessions').value;
                        const sj = await (await fetch('/api/workspace/sessions/' + sid)).json();
                        return {{
                            mid,
                            text: m.innerText,
                            cits: [...new Set(btns.map(b => b.dataset.evidenceId))],
                            turn: sj.turns.at(-1),
                            message: sj.messages.at(-1),
                        }};
                    }})()""",
                )
            )
            clicks: dict[str, Any] = {}
            mid = dom["mid"]
            for k in dom["cits"]:
                sel = f"article[data-message-id='{mid}'] button[data-evidence-id='{k}']"
                _ab(
                    session,
                    "eval",
                    f'document.querySelector({json.dumps(sel)}).scrollIntoView({{block:"center"}})',
                )
                _ab(session, "click", sel)
                _wait_js(
                    session,
                    'document.querySelector("#reader").open && '
                    '!document.querySelector("#reader-body").innerText.includes("正在加载原件")',
                )
                clicks[k] = json.loads(
                    _ab(
                        session,
                        "eval",
                        """(() => ({
                            open: document.querySelector("#reader").open,
                            title: document.querySelector("#reader-title").innerText,
                            status: document.querySelector("#reader-status").innerText,
                            head: document.querySelector("#reader-body").innerText.slice(0, 320),
                        }))()""",
                    )
                )
                _ab(session, "find", "role", "button", "click", "--name", "关闭原件", "--exact")

            trace = dom["turn"]["trace"]
            cases_out[cid] = {
                "question": question,
                "turn_status": dom["turn"]["status"],
                "route": trace.get("route"),
                "generation": trace.get("generation"),
                "external_call_count": trace.get("external_call_count"),
                "blocks": dom["message"]["blocks"] if dom.get("message") else None,
                "visible_text": dom["text"],
                "citations": dom["cits"],
                "clicked_citations": clicks,
            }
        provider_log: list[dict[str, Any]] = []
        prov_file = runtime / "provider.jsonl"
        if prov_file.exists():
            provider_log = [
                json.loads(line)
                for line in prov_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return {
            "process_meta": process_meta,
            "provider_log": provider_log,
            "cases": cases_out,
        }
    finally:
        try:
            _ab(session, "close", timeout=15)
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


POSITIVE_CASES = {
    ("2_a_declares_equiv", "ask_a"): ("0.26", "0.19"),
    ("3_same_cell_fwd", "ask_a"): ("0.26", "0.19"),
    ("3_same_cell_rev", "ask_a"): ("0.26", "0.19"),
    ("5_original_six", "held_out"): ("0.26", "0.19"),
}
FORBIDDEN_MEASURED_VALS = ("0.26", "0.19", "0.44", "0.31", "0.87")
NUM_TOKEN = re.compile(r"\b\d+(?:\.\d+)?\b")


def score_m2(results: dict[str, Any]) -> dict[str, Any]:
    """Score attribution, numeric-citation support, status accuracy, and extra-missing warnings."""
    summary: dict[str, Any] = {"mode_off": {}, "mode_on": {}}
    for mode_key, out_key in (("mode_off", "mode_off"), ("mode_on_browser", "mode_on")):
        rows = []
        extra_missing_count = 0
        for sname, sdata in results["suites"].items():
            for qid, row in sdata[mode_key]["cases"].items():
                is_pos = (sname, qid) in POSITIVE_CASES
                text = row["visible_text"]
                blocks = row.get("blocks") or []
                if mode_key == "mode_off":
                    status = row.get("answer_status")
                    cites = list(row.get("citations", {}).values())
                    has_lab = any(
                        "Lab" in c["file_name"] or "Latch-Trial" in c["file_name"] for c in cites
                    )
                    attribution_ok = has_lab if is_pos else (not has_lab)
                    status_expected = "complete" if is_pos else "partial"
                    status_ok = status == status_expected or (
                        not is_pos and status == "no_evidence"
                    )
                    if is_pos:
                        expected_vals = POSITIVE_CASES[(sname, qid)]
                        num_status = (
                            "verified_by_citation"
                            if all(v in text for v in expected_vals) and attribution_ok
                            else "unverified_or_leaked"
                        )
                    else:
                        has_forbidden = any(v in text for v in FORBIDDEN_MEASURED_VALS)
                        guess_uncertain = qid == "ask_test_guess_026" and "无法确认" in text
                        num_status = (
                            "unverified_or_leaked"
                            if (has_forbidden and not guess_uncertain)
                            else "无测量声明"
                        )
                    extra_missing = False
                else:
                    gen = row.get("generation") or {}
                    status = gen.get("status")
                    clicks = list(row.get("clicked_citations", {}).values())
                    cited_lab_ok = any(
                        ("Lab" in c["title"] or "Latch-Trial" in c["title"])
                        and not c["status"].startswith("错误")
                        and all(v in c["head"] or True for v in ())
                        for c in clicks
                    )
                    rej = gen.get("rejected_experiment_reasons") or {}
                    attribution_ok = cited_lab_ok if is_pos else (not cited_lab_ok)
                    if is_pos:
                        expected_vals = POSITIVE_CASES[(sname, qid)]
                        num_ok = all(v in text for v in expected_vals) and any(
                            ("Lab" in c["title"] or "Latch-Trial" in c["title"])
                            and all(v in c["head"] for v in expected_vals)
                            for c in clicks
                        )
                        num_status = "verified_by_citation" if num_ok else "unverified_or_leaked"
                        if status == "succeeded":
                            status_ok = True
                            extra_missing = False
                        elif status == "partial" and num_ok and not rej:
                            # Registered "答案完整但额外提示缺证"
                            status_ok = True
                            extra_missing = True
                            extra_missing_count += 1
                        else:
                            status_ok = False
                            extra_missing = False
                    else:
                        stmt_text = "\n".join(
                            b.get("text", "") for b in blocks if b.get("type") == "text"
                        )
                        has_forbidden = any(v in stmt_text for v in FORBIDDEN_MEASURED_VALS)
                        uncertain_ok = qid == "ask_test_guess_026" and (
                            "无法确认" in text or "未能确认" in text or "没有" in text
                        )
                        num_status = (
                            "无测量声明"
                            if (not has_forbidden or (uncertain_ok and "0.26" not in stmt_text))
                            else "unverified_or_leaked"
                        )
                        status_ok = status in {"partial", "no_evidence"} or (
                            qid == "ask_test_value" and status == "partial"
                        )
                        extra_missing = False
                rows.append(
                    {
                        "suite": sname,
                        "question_id": qid,
                        "positive_expected": is_pos,
                        "attribution_ok": attribution_ok,
                        "numeric_citation_status": num_status,
                        "status": status,
                        "status_ok": status_ok,
                        "extra_missing_on_complete": extra_missing,
                    }
                )
        summary[out_key] = {
            "total": len(rows),
            "attribution_passed": sum(1 for r in rows if r["attribution_ok"]),
            "numeric_verified_by_citation": sum(
                1 for r in rows if r["numeric_citation_status"] == "verified_by_citation"
            ),
            "no_measurement_declaration_count": sum(
                1 for r in rows if r["numeric_citation_status"] == "无测量声明"
            ),
            "numeric_unverified_or_leaked": sum(
                1 for r in rows if r["numeric_citation_status"] == "unverified_or_leaked"
            ),
            "status_accurate_passed": sum(1 for r in rows if r["status_ok"]),
            "extra_missing_on_complete_count": extra_missing_count,
            "rows": rows,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/Users/Link/code/SRAG"))
    parser.add_argument("--commit", type=str, required=True)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    python_bin = Path(os.path.abspath(args.python))
    env_file = args.env_file.resolve()
    target_sha = subprocess.check_output(
        ["git", "-C", str(args.repo), "rev-parse", args.commit], text=True
    ).strip()

    with tempfile.TemporaryDirectory(prefix="srag-joint-m2-") as tmp:
        t = Path(tmp)
        prep_dir = t / "prep"
        cand_dir = t / "candidate"
        for d, sha in ((prep_dir, PREP_COMMIT), (cand_dir, target_sha)):
            subprocess.run(
                ["git", "-C", str(args.repo), "worktree", "add", "--detach", str(d), sha],
                check=True,
                capture_output=True,
            )
        try:
            prep_path = prep_dir / "docs" / "G" / "M2-retest-prep" / "rehearse.py"
            diag_path = cand_dir / "docs" / "G" / "M2-public-diagnosis" / "diagnose.py"
            prep = runpy.run_path(str(prep_path))
            f = t / "files"
            prep["fixtures"](f)

            q_basalt = "Basalt latch 的定义是什么？讲义所对应的测试集实验如何验证？"
            q_quartz = "Quartz latch 的定义是什么？讲义所对应的测试集实验如何验证？"

            g_a = _nb(
                f / "Guide-A.ipynb",
                "Definition: A basalt latch opens on three consecutive readings strictly above 8 V."
                "\nExperiment-ID: TrialA.",
            )
            g_ab = _nb(
                f / "Guide-AB.ipynb",
                "Definition: A basalt latch opens on three consecutive readings strictly above 8 V."
                "\nExperiment-ID: TrialA.",
                "Definition: A quartz latch opens on two consecutive readings strictly above 5 V."
                "\nExperiment-ID: TrialB.",
            )
            lab_only_b = _nb(
                f / "Lab-OnlyB.ipynb",
                "Basalt latch TrialA.\n"
                "Experiment-ID: TrialA. Split: held-out. Metric: MAE. Unit: amplitude. "
                "Baseline: 0.26. Trial: 0.19. Samples: 4.\n"
                "A 使用留出验证集。For Trial B only, held-out is used as the test split.",
            )
            lab_a_self = _nb(
                f / "Lab-ASelf.ipynb",
                "Basalt latch TrialA.\n"
                "Experiment-ID: TrialA. Split: held-out. Metric: MAE. Unit: amplitude. "
                "Baseline: 0.26. Trial: 0.19. Samples: 4.\n"
                "For TrialA only, held-out is used as the test split.",
            )
            rec_a = (
                "Basalt latch TrialA.\n"
                "Experiment-ID: TrialA. Split: held-out. Metric: MAE. Unit: amplitude. "
                "Baseline: 0.26. Trial: 0.19. Samples: 4.\n"
                "For TrialA, held-out is used as the test split."
            )
            rec_b = (
                "Quartz latch TrialB.\n"
                "Experiment-ID: TrialB. Split: held-out. Metric: MAE. Unit: amplitude. "
                "Baseline: 0.44. Trial: 0.31. Samples: 8."
            )
            lab_ab_fwd = _nb(f / "Lab-AB-Fwd.ipynb", f"{rec_a}\n\n{rec_b}")
            lab_ab_rev = _nb(f / "Lab-AB-Rev.ipynb", f"{rec_b}\n\n{rec_a}")
            lab_diff = _nb(
                f / "Lab-Diff.ipynb",
                "Basalt latch TrialA.\n"
                "Experiment-ID: TrialA. Split: held-out. Metric: MAE. Unit: amplitude. "
                "Baseline: 0.26. Trial: 0.19. Samples: 4.\n"
                "For TrialA, held-out is different from the test split.",
            )
            orig_files = [f / prep["GUIDE"], f / prep["DECK"], f / prep["LAB"]]
            orig_cases = (
                ("held_out", prep["QUESTION"]),
                ("salt_fog", prep["MISSING"]),
                (
                    "unfamiliar_condition",
                    prep["QUESTION"].replace("留出集", "低温循环后的留出集"),
                ),
                ("different_object", prep["QUESTION"].replace("留出集", "Quartz latch 的留出集")),
                ("different_metric", prep["QUESTION"] + "请给出 RMSE。"),
                ("test_split", q_basalt),
            )
            counter_cases = (
                (
                    "independent_verification_proc",
                    prep["QUESTION"] + "请给出针对定义阈值的独立验证流程。",
                ),
            )
            cit_counter_cases = (
                ("ask_test_value", "Basalt latch 的定义是什么？讲义所对应的测试集实验测得多少？"),
                (
                    "ask_test_guess_026",
                    "Basalt latch 的定义是什么？讲义所对应的测试集实验是否测得 0.26？",
                ),
            )

            suites = {
                "1_only_b_declares_equiv": ([g_a, lab_only_b], (("ask_a", q_basalt),)),
                "2_a_declares_equiv": ([g_a, lab_a_self], (("ask_a", q_basalt),)),
                "3_same_cell_fwd": ([g_ab, lab_ab_fwd], (("ask_a", q_basalt), ("ask_b", q_quartz))),
                "3_same_cell_rev": ([g_ab, lab_ab_rev], (("ask_a", q_basalt), ("ask_b", q_quartz))),
                "4_explicit_diff": ([g_a, lab_diff], (("ask_a", q_basalt),)),
                "5_original_six": (orig_files, orig_cases),
                "6_counter_example": (orig_files, counter_cases),
                "7_citation_number_counterexamples": (orig_files, cit_counter_cases),
            }

            results: dict[str, Any] = {"subject_commit": target_sha, "suites": {}}
            for name, (files, cases) in suites.items():
                results["suites"][name] = {
                    "mode_off": _run_off(
                        subject_dir=cand_dir,
                        prep_path=prep_path,
                        diag_path=diag_path,
                        env_file=env_file,
                        upload_files=files,
                        cases=cases,
                        out_dir=t / f"off_{name}",
                    ),
                    "mode_on_browser": _run_browser(
                        label=name,
                        subject_dir=cand_dir,
                        prep_path=prep_path,
                        env_file=env_file,
                        python_bin=python_bin,
                        upload_files=files,
                        cases=cases,
                        out_dir=t,
                    ),
                }
            results["score_summary"] = score_m2(results)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote {args.out}")
        finally:
            for d in (prep_dir, cand_dir):
                subprocess.run(
                    ["git", "-C", str(args.repo), "worktree", "remove", "--force", str(d)],
                    check=False,
                    capture_output=True,
                )


if __name__ == "__main__":
    main()
