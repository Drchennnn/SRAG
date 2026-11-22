"""Verify M2 condition-mismatch fix across generation-OFF and real Chromium generation-ON.

Covers:
1. Original 6 public cases (held_out, salt_fog, unfamiliar_condition,
   different_object, different_metric, test_split) in both Mode OFF and Mode ON.
2. New explicit test-split positive case (Split: test, 0.31 -> 0.22, 6 samples).
3. New dual-split disambiguation case (Cell 1 held-out 0.26 -> 0.19 vs Cell 2 test 0.31 -> 0.22).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import socket
import subprocess
import sys
import tempfile
import time
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


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _original_six(prep: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    q = prep["QUESTION"]
    m = prep["MISSING"]
    return (
        ("held_out", q),
        ("salt_fog", m),
        ("unfamiliar_condition", q.replace("留出集", "低温循环后的留出集")),
        ("different_object", q.replace("留出集", "Quartz latch 的留出集")),
        ("different_metric", q + "请给出 RMSE。"),
        ("test_split", "Basalt latch 的定义是什么？讲义所对应的测试集实验如何验证？"),
    )


def _write_new_notebooks(folder: Path) -> tuple[Path, Path]:
    pos_nb = folder / "Latch-Test-Positive.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell(
                    "Basalt latch SlateTrialB6 test split verification.\n"
                    "Experiment-ID: SlateTrialB6. Split: test. Metric: MAE. Unit: amplitude. "
                    "Baseline: 0.31. Trial: 0.22. Samples: 6.\n"
                    "Baseline absolute errors: (0.28, 0.30, 0.33, 0.31, 0.32, 0.32).\n"
                    "Trial absolute errors: (0.20, 0.22, 0.23, 0.21, 0.24, 0.22).",
                    id="test-positive-cell",
                )
            ]
        ),
        pos_nb,
    )
    dual_nb = folder / "Latch-Dual-Splits.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell(
                    "Basalt latch SlateTrialB6 held-out split.\n"
                    "Experiment-ID: SlateTrialB6. Split: held-out. Metric: MAE. Unit: amplitude. "
                    "Baseline: 0.26. Trial: 0.19. Samples: 4.",
                    id="dual-heldout-cell",
                ),
                nbformat.v4.new_markdown_cell(
                    "Basalt latch SlateTrialB6 test split.\n"
                    "Experiment-ID: SlateTrialB6. Split: test. Metric: MAE. Unit: amplitude. "
                    "Baseline: 0.31. Trial: 0.22. Samples: 6.",
                    id="dual-test-cell",
                ),
            ]
        ),
        dual_nb,
    )
    return pos_nb, dual_nb


def _run_off_suite(
    *,
    subject_dir: Path,
    prep_path: Path,
    diag_path: Path,
    env_file: Path,
    upload_files: list[Path],
    cases: tuple[tuple[str, str], ...],
    out_dir: Path,
) -> dict[str, Any]:
    for mod_name in [k for k in sys.modules if k == "app" or k.startswith("app.")]:
        del sys.modules[mod_name]
    sys.path.insert(0, str(subject_dir))
    prep = runpy.run_path(str(prep_path))
    diag = runpy.run_path(str(diag_path))
    settings = prep["settings_for"](env_file, out_dir / "runtime", web=False)
    app, service = prep["app_for"](settings)
    events: list[dict[str, Any]] = []
    prep["observe_providers"](settings, events, web=False, image_hashes={})
    try:
        with TestClient(app, base_url=diag["HOST"], headers={"Origin": diag["HOST"]}) as client:
            session = diag["ApiSession"](client, service, prep["COURSE"])
            assert session.empty
            for path in upload_files:
                assert session.upload(path)["version"] == 1
            cases_out: dict[str, Any] = {}
            for case_id, question in cases:
                resp = session.ask(case_id, question)
                opened = session.evidence(resp)
                trace = resp["data"]["turn"]["trace"]
                msg = resp["data"]["message"]
                cases_out[case_id] = {
                    "question": question,
                    "turn_status": resp["data"]["turn"]["status"],
                    "route": trace.get("route"),
                    "answer_status": trace.get("answer_status"),
                    "external_call_count": trace.get("external_call_count"),
                    "visible_text": "\n".join(
                        b.get("text", "") for b in msg["blocks"] if b.get("text")
                    ),
                    "citations": {
                        cid: {
                            "file_name": opened[cid]["data"]["file_name"],
                            "version": opened[cid]["data"]["source"]["version"],
                            "source_id": opened[cid]["data"]["source"]["source_id"],
                            "text": opened[cid]["data"].get("text", ""),
                        }
                        for cid in opened
                    },
                }
    finally:
        prep["restore_providers"]()
        if str(subject_dir) in sys.path:
            sys.path.remove(str(subject_dir))
    return {
        "mode": "answer_disabled",
        "effective_settings": prep["effective"](settings),
        "cases": cases_out,
    }


def _browser_server_script(
    runtime: Path, env_file: Path, course_id: str, port: int
) -> str:
    return f"""
import json, httpx, uvicorn
from pathlib import Path
from time import perf_counter
from app.api.workspace import get_workspace_service
from app.config import Settings, get_settings
from app.main import create_app
from app.rag.workspace import make_workspace

runtime = Path({str(runtime)!r})
settings = Settings(
    _env_file={str(env_file)!r},
    course_root=runtime / "courses",
    data_dir=runtime / "data",
    sqlite_path=runtime / "library.sqlite3",
    workspace_course_ids={course_id!r},
    workspace_answer_enabled=True,
    workspace_retrieval_strategy="fts",
    source_vision_enabled=True,
    workspace_input_token_limit=32768,
    jev_api_key="",
    qdrant_api_key="",
    qdrant_url="http://127.0.0.1:1",
)
app = create_app()
service = make_workspace(settings)
app.dependency_overrides[get_settings] = lambda: settings
app.dependency_overrides[get_workspace_service] = lambda: service

orig_async = httpx.AsyncHTTPTransport.handle_async_request
async def log_async(transport, request):
    started = perf_counter()
    rec = {{"url": str(request.url), "method": request.method}}
    try:
        payload = json.loads(request.content)
        rec["model"] = payload.get("model")
        rec["request_bytes"] = len(request.content)
        user_msg = json.loads(payload["messages"][1]["content"])
        rec["mode"] = user_msg.get("mode")
        rec["question"] = user_msg.get("question")
        rec["evidence_ids"] = [e["id"] for e in user_msg.get("evidence", [])]
    except Exception as exc:
        rec["parse_error"] = type(exc).__name__
    try:
        resp = await orig_async(transport, request)
        rec["status_code"] = resp.status_code
        rec["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
        with (runtime / "provider.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
        return resp
    except Exception as exc:
        rec["error"] = type(exc).__name__
        rec["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
        with (runtime / "provider.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
        raise

httpx.AsyncHTTPTransport.handle_async_request = log_async

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port={port}, log_level="warning")
"""


def _run_browser_suite(
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
    server_script = runtime / "serve.py"
    server_script.write_text(
        _browser_server_script(runtime, env_file, prep["COURSE"], port), encoding="utf-8"
    )
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(subject_dir),
    }
    proc = subprocess.Popen(
        [str(python_bin), str(server_script)],
        cwd=str(subject_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    session = f"m2-fix-{label}-{port}-{int(time.time())}"
    try:
        base_url = f"http://127.0.0.1:{port}/workspace/"
        for _ in range(40):
            r = subprocess.run(
                ["curl", "--silent", "--fail", base_url],
                capture_output=True,
                timeout=2,
                check=False,
            )
            if r.returncode == 0:
                break
            time.sleep(0.25)
        else:
            proc.terminate()
            txt, _ = proc.communicate(timeout=3)
            raise RuntimeError(f"Server failed on {base_url}: {txt}")

        _ab(session, "--allowed-domains", "127.0.0.1", "open", base_url)
        _ab(session, "set", "viewport", "1340", "860")
        _ab(session, "wait", "--text", "暂无资料")
        user_agent = json.loads(_ab(session, "eval", "navigator.userAgent"))

        panel_open = json.loads(
            _ab(session, "eval", 'document.querySelector("#import-panel").open')
        )
        if not panel_open:
            _ab(session, "find", "text", "导入资料", "click", "--exact")
        _ab(session, "select", "#import-course", prep["COURSE"])
        _ab(session, "upload", "#import-files", *[str(p) for p in upload_files])
        _ab(session, "find", "role", "button", "click", "--name", "开始导入", "--exact")
        for path in upload_files:
            _ab(session, "wait", "--text", f"{path.name} · 导入完成 · v1")
        _wait_js(
            session,
            '!document.querySelector("#import-button").disabled && '
            '!document.querySelector("#import-files").files.length',
        )

        def read_provider() -> list[dict[str, Any]]:
            p = runtime / "provider.jsonl"
            if not p.exists():
                return []
            return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

        cases_out: dict[str, Any] = {}
        turn_idx = 0
        for case_id, question in cases:
            turn_idx += 1
            calls_before = len(read_provider())
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
            dom_state = json.loads(
                _ab(
                    session,
                    "eval",
                    f"""(async () => {{
                        const users = document.querySelectorAll('#messages article.message-user');
                        const m = users.item({idx}).nextElementSibling;
                        const mid = m.dataset.messageId;
                        const btns = [...m.querySelectorAll('button[data-evidence-id]')];
                        const sid = document.querySelector('#sessions')?.value
                            || window.localStorage.getItem('srag-workspace-session');
                        const sessResp = await fetch('/api/workspace/sessions/' + sid);
                        const sessJson = await sessResp.json();
                        const msgJson = (sessJson.messages || []).find(x => x.message_id === mid);
                        const turnJson = (sessJson.turns || []).at(-1);
                        return {{
                            message_id: mid,
                            visible_text: m.innerText,
                            citations: [...new Set(btns.map(b => b.dataset.evidenceId))],
                            trace: turnJson?.trace,
                            status: turnJson?.status,
                            blocks: msgJson?.blocks,
                        }};
                    }})()""",
                )
            )
            clicked_citations: dict[str, Any] = {}
            mid = dom_state["message_id"]
            for cid in dom_state["citations"]:
                sel = f"article[data-message-id='{mid}'] button[data-evidence-id='{cid}']"
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
                clicked_citations[cid] = json.loads(
                    _ab(
                        session,
                        "eval",
                        """(() => ({
                            open: document.querySelector("#reader").open,
                            title: document.querySelector("#reader-title").innerText,
                            location: document.querySelector("#reader-location").innerText,
                            status: document.querySelector("#reader-status").innerText,
                            body_head: document.querySelector("#reader-body")
                                .innerText.slice(0, 300),
                        }))()""",
                    )
                )
                _ab(session, "find", "role", "button", "click", "--name", "关闭原件", "--exact")

            trace = dom_state.get("trace") or {}
            cases_out[case_id] = {
                "question": question,
                "turn_status": dom_state["status"],
                "route": trace.get("route"),
                "generation": trace.get("generation"),
                "external_call_count": trace.get("external_call_count"),
                "provider_calls": read_provider()[calls_before:],
                "visible_text": dom_state["visible_text"],
                "citations": dom_state["citations"],
                "clicked_citations": clicked_citations,
            }

        return {
            "mode": "answer_enabled_browser",
            "user_agent": user_agent,
            "port": port,
            "empty_start": True,
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
    parser.add_argument("--repo", type=Path, default=Path("/Users/Link/code/SRAG"))
    parser.add_argument("--subject-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    subject_dir = args.subject_dir.resolve()
    python_bin = Path(os.path.abspath(args.python))
    with tempfile.TemporaryDirectory(prefix="srag-m2-fix-verify-") as tmp:
        tmp_path = Path(tmp)
        prep_dir = tmp_path / "prep"
        subprocess.run(
            [
                "git",
                "-C",
                str(args.repo),
                "worktree",
                "add",
                "--detach",
                str(prep_dir),
                PREP_COMMIT,
            ],
            check=True,
        )
        try:
            prep_path = prep_dir / "docs" / "G" / "M2-retest-prep" / "rehearse.py"
            diag_path = subject_dir / "docs" / "G" / "M2-public-diagnosis" / "diagnose.py"
            prep = runpy.run_path(str(prep_path))
            files_dir = tmp_path / "fixtures"
            prep["fixtures"](files_dir)
            pos_nb, dual_nb = _write_new_notebooks(files_dir)

            orig_files = [
                files_dir / prep["GUIDE"],
                files_dir / prep["DECK"],
                files_dir / prep["LAB"],
            ]
            orig_cases = _original_six(prep)

            pos_files = [files_dir / prep["GUIDE"], files_dir / prep["DECK"], pos_nb]
            pos_cases = (("explicit_test_positive", orig_cases[5][1]),)

            dual_files = [files_dir / prep["GUIDE"], files_dir / prep["DECK"], dual_nb]
            dual_cases = (
                ("dual_ask_test", orig_cases[5][1]),
                ("dual_ask_heldout", orig_cases[0][1]),
            )

            report = {
                "subject_commit": subprocess.check_output(
                    ["git", "-C", str(subject_dir), "rev-parse", "HEAD"], text=True
                ).strip(),
                "original_six": {
                    "mode_off": _run_off_suite(
                        subject_dir=subject_dir,
                        prep_path=prep_path,
                        diag_path=diag_path,
                        env_file=args.env_file.resolve(),
                        upload_files=orig_files,
                        cases=orig_cases,
                        out_dir=tmp_path / "off_orig",
                    ),
                    "mode_on_browser": _run_browser_suite(
                        label="orig",
                        subject_dir=subject_dir,
                        prep_path=prep_path,
                        env_file=args.env_file.resolve(),
                        python_bin=python_bin,
                        upload_files=orig_files,
                        cases=orig_cases,
                        out_dir=tmp_path,
                    ),
                },
                "explicit_test_positive": {
                    "mode_off": _run_off_suite(
                        subject_dir=subject_dir,
                        prep_path=prep_path,
                        diag_path=diag_path,
                        env_file=args.env_file.resolve(),
                        upload_files=pos_files,
                        cases=pos_cases,
                        out_dir=tmp_path / "off_pos",
                    ),
                    "mode_on_browser": _run_browser_suite(
                        label="pos",
                        subject_dir=subject_dir,
                        prep_path=prep_path,
                        env_file=args.env_file.resolve(),
                        python_bin=python_bin,
                        upload_files=pos_files,
                        cases=pos_cases,
                        out_dir=tmp_path,
                    ),
                },
                "dual_split_disambiguation": {
                    "mode_off": _run_off_suite(
                        subject_dir=subject_dir,
                        prep_path=prep_path,
                        diag_path=diag_path,
                        env_file=args.env_file.resolve(),
                        upload_files=dual_files,
                        cases=dual_cases,
                        out_dir=tmp_path / "off_dual",
                    ),
                    "mode_on_browser": _run_browser_suite(
                        label="dual",
                        subject_dir=subject_dir,
                        prep_path=prep_path,
                        env_file=args.env_file.resolve(),
                        python_bin=python_bin,
                        upload_files=dual_files,
                        cases=dual_cases,
                        out_dir=tmp_path,
                    ),
                },
            }
            args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
            args.out.resolve().write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        finally:
            subprocess.run(
                ["git", "-C", str(args.repo), "worktree", "remove", "--force", str(prep_dir)],
                check=False,
            )


if __name__ == "__main__":
    main()
