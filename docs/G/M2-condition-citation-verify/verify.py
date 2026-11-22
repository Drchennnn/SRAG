"""Verify direct citation support for measurement numbers + 14-question regression.

Runs on clean detached worktree of the product fix commit (8d530e4887330f422364375f897b29da5e97a224),
records live process metadata (git_sha, worktree_status, module_files, effective_config),
and evaluates Mode OFF & Mode ON (Chromium + deepseek-flash + citation clicks) across:
- Original 14 questions (13 attribution/regression + 1 independent verification proc counterexample)
- New citation-number counterexamples (hallucinated/guessed 0.26 on test split vs held-out).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
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
    settings = prep["settings_for"](env_file, out_dir / "runtime", web=False)
    app, service = prep["app_for"](settings)
    events: list[dict[str, Any]] = []
    prep["observe_providers"](settings, events, web=False, image_hashes={})
    process_meta = {
        **_git_meta(subject_dir),
        "module_files": {
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


def _browser_server(runtime: Path, env_file: Path, course: str, port: int, subject_dir: Path) -> str:
    return f"""
import json, subprocess, httpx, uvicorn
from pathlib import Path
from time import perf_counter
import app.rag.evidence_supplement as es
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
        "git_sha": subprocess.check_output(["git", "-C", str(subj), "rev-parse", "HEAD"], text=True).strip(),
        "worktree_status": subprocess.check_output(["git", "-C", str(subj), "status", "--short"], text=True).strip() or "clean",
        "module_files": {{
            "workspace_generation": wg.__file__,
            "evidence_supplement": es.__file__,
        }},
        "effective_config": {{
            "workspace_answer_enabled": s.workspace_answer_enabled,
            "workspace_retrieval_strategy": s.workspace_retrieval_strategy,
            "workspace_input_token_limit": s.workspace_input_token_limit,
            "source_vision_enabled": s.source_vision_enabled,
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
    session = f"m2-cit-{label}-{port}-{int(time.time())}"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/Users/Link/code/SRAG-m2-cond-gen-fix"))
    parser.add_argument("--commit", type=str, default="HEAD")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    python_bin = Path(os.path.abspath(args.python))
    env_file = args.env_file.resolve()
    target_sha = subprocess.check_output(
        ["git", "-C", str(args.repo), "rev-parse", args.commit], text=True
    ).strip()

    with tempfile.TemporaryDirectory(prefix="srag-m2-cit-") as tmp:
        t = Path(tmp)
        prep_dir = t / "prep"
        cand_dir = t / "candidate"
        for d, repo_path, sha in (
            (prep_dir, Path("/Users/Link/code/SRAG"), PREP_COMMIT),
            (cand_dir, args.repo, target_sha),
        ):
            subprocess.run(
                ["git", "-C", str(repo_path), "worktree", "add", "--detach", str(d), sha],
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
                ("ask_test_guess_026", "Basalt latch 的定义是什么？讲义所对应的测试集实验是否测得 0.26？"),
            )

            suites = {
                "1_only_b_declares_equiv": ([g_a, lab_only_b], (("ask_a", q_basalt),)),
                "2_a_declares_equiv": ([g_a, lab_a_self], (("ask_a", q_basalt),)),
                "3_same_cell_fwd": (
                    [g_ab, lab_ab_fwd],
                    (("ask_a", q_basalt), ("ask_b", q_quartz)),
                ),
                "3_same_cell_rev": (
                    [g_ab, lab_ab_rev],
                    (("ask_a", q_basalt), ("ask_b", q_quartz)),
                ),
                "4_explicit_diff": ([g_a, lab_diff], (("ask_a", q_basalt),)),
                "5_original_six": (orig_files, orig_cases),
                "6_counter_example": (orig_files, counter_cases),
                "7_citation_number_counterexamples": (orig_files, cit_counter_cases),
            }

            results: dict[str, Any] = {
                "subject_commit": target_sha,
                "suites": {},
            }
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

            orig_14_keys = [
                (s, q)
                for s in (
                    "1_only_b_declares_equiv",
                    "2_a_declares_equiv",
                    "3_same_cell_fwd",
                    "3_same_cell_rev",
                    "4_explicit_diff",
                    "5_original_six",
                    "6_counter_example",
                )
                for q in results["suites"][s]["mode_off"]["cases"]
            ]
            all_keys = [
                (s, q)
                for s in results["suites"]
                for q in results["suites"][s]["mode_off"]["cases"]
            ]
            results["denominators"] = {
                "original_14": len(orig_14_keys),
                "total_with_citation_counterexamples": len(all_keys),
            }

            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote {args.out}")
        finally:
            subprocess.run(
                ["git", "-C", "/Users/Link/code/SRAG", "worktree", "remove", "--force", str(prep_dir)],
                check=False,
                capture_output=True,
            )
            subprocess.run(
                ["git", "-C", str(args.repo), "worktree", "remove", "--force", str(cand_dir)],
                check=False,
                capture_output=True,
            )


if __name__ == "__main__":
    main()
