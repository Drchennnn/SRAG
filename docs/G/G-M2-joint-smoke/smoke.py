"""Public real-browser smoke test for the G/M2 joint candidate.

1. Imports public Beacon-Sync-Manual.pdf + Beacon-Sync-Log.ipynb into an empty runtime.
2. Turn 1 (course scope): asks definition + companion experiment measurement.
3. Narrows scope to Beacon-Sync-Manual.pdf in UI, then asks comma-connected follow-up.
4. Clicks every citation and inspects topic inheritance, definition citation,
   scope-narrowing notice, and zero leakage of old experiment numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_ENV = Path("/Users/Link/code/SRAG/.env")
DEFAULT_PYTHON = Path("/Users/Link/code/SRAG/.venv/bin/python")
COURSE = "g-m2-joint-smoke"


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


def _server_code(runtime: Path, env_file: Path, port: int, repo_dir: Path) -> str:
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
repo = Path({str(repo_dir)!r})
s = Settings(
    _env_file={str(env_file)!r},
    course_root=rt / "courses",
    data_dir=rt / "data",
    sqlite_path=rt / "library.sqlite3",
    workspace_course_ids={COURSE!r},
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

@app.get("/__smoke_meta")
def smoke_meta():
    sha = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    wt = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True
    ).strip()
    return {{
        "git_sha": sha,
        "worktree_status": wt or "clean",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/Users/Link/code/SRAG-g-m2-joint"))
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    env_file = args.env_file.resolve()
    python_bin = Path(os.path.abspath(args.python))
    fixture_dir = repo / "docs" / "G" / "G28-comma-fix" / "fixture"
    files = [fixture_dir / "Beacon-Sync-Manual.pdf", fixture_dir / "Beacon-Sync-Log.ipynb"]

    with tempfile.TemporaryDirectory(prefix="srag-gm2-smoke-") as tmp:
        rt = Path(tmp)
        port = _free_port()
        srv = rt / "serve.py"
        srv.write_text(_server_code(rt, env_file, port, repo), encoding="utf-8")
        proc = subprocess.Popen(
            [str(python_bin), str(srv)],
            cwd=str(repo),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(repo)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        session = f"gm2-smoke-{port}-{int(time.time())}"
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

            with urllib.request.urlopen(f"http://127.0.0.1:{port}/__smoke_meta", timeout=5) as r:
                meta = json.loads(r.read().decode("utf-8"))

            _ab(session, "--allowed-domains", "127.0.0.1", "open", url)
            _ab(session, "wait", "--text", "暂无资料")
            if not json.loads(_ab(session, "eval", 'document.querySelector("#import-panel").open')):
                _ab(session, "find", "text", "导入资料", "click", "--exact")
            _ab(session, "select", "#import-course", COURSE)
            _ab(session, "upload", "#import-files", *[str(p) for p in files])
            _ab(session, "find", "role", "button", "click", "--name", "开始导入", "--exact")
            for p in files:
                _ab(session, "wait", "--text", f"{p.name} · 导入完成 · v1")
            _wait_js(session, '!document.querySelector("#import-button").disabled')

            turns_plan = [
                (
                    "turn1_course_prime",
                    None,
                    "讲义怎么定义 Barrier Latch？配套实验测得的 mean sync delay 是多少毫秒？",
                ),
                (
                    "turn2_scoped_comma_followup",
                    "Beacon-Sync-Manual.pdf",
                    "只看我选中的讲义，把刚才问过的部分重新梳理一遍，配套实验还能核对吗？",
                ),
            ]
            turns_out: dict[str, Any] = {}
            for idx, (tid, scope_file, question) in enumerate(turns_plan):
                if scope_file:
                    _ab(session, "find", "text", scope_file, "click", "--exact")
                    _wait_js(
                        session,
                        f"document.querySelector('#scope-summary').textContent.includes({json.dumps(scope_file)})",
                    )
                scope_summary = _ab(session, "get", "text", "#scope-summary")
                turn_idx = idx + 1
                _ab(session, "find", "label", "输入问题", "fill", question)
                _wait_js(session, '!document.querySelector("#send").disabled')
                _ab(session, "find", "role", "button", "click", "--name", "发送", "--exact")
                _wait_js(
                    session,
                    f"document.querySelectorAll('#messages article.message-user').length"
                    f" === {turn_idx}"
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
                            const q = '#messages article.message-user';
                            const users = document.querySelectorAll(q);
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
                            """(() => {
                                const body = document.querySelector("#reader-body").innerText;
                                return {
                                    open: document.querySelector("#reader").open,
                                    title: document.querySelector("#reader-title").innerText,
                                    status: document.querySelector("#reader-status").innerText,
                                    head: body.slice(0, 320),
                                };
                            })()""",
                        )
                    )
                    _ab(session, "find", "role", "button", "click", "--name", "关闭原件", "--exact")

                turns_out[tid] = {
                    "scope_summary": scope_summary,
                    "question": question,
                    "turn_status": dom["turn"]["status"],
                    "trace": dom["turn"]["trace"],
                    "blocks": dom["message"]["blocks"] if dom.get("message") else None,
                    "visible_text": dom["text"],
                    "citations": dom["cits"],
                    "clicked_citations": clicks,
                }

            provider_log = [
                json.loads(line)
                for line in (rt / "provider.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            out = {
                "process_meta": meta,
                "provider_log": provider_log,
                "turns": turns_out,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote {args.out}")
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


if __name__ == "__main__":
    main()
