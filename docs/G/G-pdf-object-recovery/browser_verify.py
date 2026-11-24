"""Real-browser verification for G-pdf-object-recovery under formal G01 settings.

Starts FastAPI/uvicorn with workspace_answer_enabled=False, source_vision_enabled=True,
qwen3-vl-plus, fts, empty DB/runtime. Imports both frozen PDFs, executes Q01-Q06 in
Chromium /workspace/, clicks every citation/card, and verifies file, version, page,
and crop region (rect_visible_norm).
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
COURSE = "pdf-obj-browser"


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
import subprocess, uvicorn
from pathlib import Path
import app.rag.parsers.local_slides as ls
import app.rag.workspace as ws
import app.rag.workspace_objects as wo
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
    workspace_answer_enabled=False,
    workspace_retrieval_strategy="fts",
    source_vision_enabled=True,
    workspace_input_token_limit=32768,
    deepseek_api_key="",
    jev_api_key="",
    qdrant_api_key="",
    qdrant_url="http://127.0.0.1:1",
    dashscope_timeout_seconds=20,
)
app = create_app()
svc = make_workspace(s)
app.dependency_overrides[get_settings] = lambda: s
app.dependency_overrides[get_workspace_service] = lambda: svc

@app.get("/__meta")
def meta():
    sha = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    return {{
        "git_sha": sha,
        "module_files": {{
            "local_slides": ls.__file__,
            "workspace": ws.__file__,
            "workspace_objects": wo.__file__,
        }},
        "effective_config": {{
            "workspace_answer_enabled": s.workspace_answer_enabled,
            "source_vision_enabled": s.source_vision_enabled,
            "vision_model": s.dashscope_vision_model,
            "workspace_retrieval_strategy": s.workspace_retrieval_strategy,
            "workspace_input_token_limit": s.workspace_input_token_limit,
        }},
    }}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port={port}, log_level="warning")
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/Users/Link/code/SRAG-pdf-obj"))
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/Users/Link/code/SRAG-pdf-obj/docs/G/G-pdf-object-recovery/post_fix_browser.json"),
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    criteria = json.loads(
        (repo / "docs/G/G-pdf-object-recovery/frozen_criteria.json").read_text(encoding="utf-8")
    )
    files = [repo / info["file"] for info in criteria["pdfs"].values()]
    python_bin = Path(os.path.abspath(args.python))

    with tempfile.TemporaryDirectory(prefix="srag-pdf-obj-web-") as tmp:
        rt = Path(tmp)
        port = _free_port()
        srv = rt / "serve.py"
        srv.write_text(_server_code(rt, args.env_file.resolve(), port, repo), encoding="utf-8")
        proc = subprocess.Popen(
            [str(python_bin), str(srv)],
            cwd=str(repo),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(repo)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        session = f"pdf-obj-{port}-{int(time.time())}"
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

            with urllib.request.urlopen(f"http://127.0.0.1:{port}/__meta", timeout=5) as r:
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

            questions_out: list[dict[str, Any]] = []
            for idx, qitem in enumerate(criteria["questions"]):
                qid = qitem["question_id"]
                prev_sid = _ab(session, "eval", "document.querySelector('#sessions').value")
                _ab(session, "find", "role", "button", "click", "--name", "新建会话", "--exact")
                _wait_js(
                    session,
                    f"document.querySelector('#sessions').value !== {prev_sid}"
                    " && !document.querySelector('#sessions').disabled"
                    " && document.querySelectorAll('#messages article').length === 0",
                )
                _wait_js(
                    session,
                    f"[...document.querySelectorAll('#catalog button')]"
                    f".some(b => b.textContent.includes({json.dumps(qitem['document'])}))",
                )
                _ab(
                    session,
                    "eval",
                    f"""(() => {{
                        const btn = [...document.querySelectorAll('#catalog button')]
                            .find(b => b.textContent.includes({json.dumps(qitem['document'])}));
                        if (!btn) throw new Error('Doc button not found');
                        btn.click();
                    }})()""",
                )
                _wait_js(
                    session,
                    f"document.querySelector('#scope-summary').textContent.includes({json.dumps(qitem['document'])})",
                )
                _ab(session, "find", "label", "输入问题", "fill", qitem["question"])
                _wait_js(session, '!document.querySelector("#send").disabled')
                _ab(session, "find", "role", "button", "click", "--name", "发送", "--exact")
                _wait_js(
                    session,
                    "document.querySelectorAll('#messages article.message-user').length === 1"
                    " && document.querySelector('#messages article.message-user')"
                    ".nextElementSibling?.matches('article:not(.message-user)')"
                    " && !document.querySelector('#turn-actions button')"
                    "?.textContent.includes('取消')",
                )
                dom = json.loads(
                    _ab(
                        session,
                        "eval",
                        """(async () => {
                            const m = document.querySelector('#messages article.message-user').nextElementSibling;
                            const mid = m.dataset.messageId;
                            const btns = [...m.querySelectorAll('button[data-evidence-id]')];
                            const sid = document.querySelector('#sessions').value;
                            const sj = await (await fetch('/api/workspace/sessions/' + sid)).json();
                            return {
                                mid,
                                text: m.innerText,
                                cits: [...new Set(btns.map(b => b.dataset.evidenceId))],
                                turn: sj.turns.at(-1),
                                message: sj.messages.at(-1),
                            };
                        })()""",
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
                                const img = document.querySelector("#reader-body img");
                                return {
                                    open: document.querySelector("#reader").open,
                                    title: document.querySelector("#reader-title").innerText,
                                    status: document.querySelector("#reader-status").innerText,
                                    has_img: Boolean(img && img.complete && img.naturalWidth > 0),
                                    img_width: img ? img.naturalWidth : 0,
                                    img_height: img ? img.naturalHeight : 0,
                                    head: document.querySelector("#reader-body").innerText.slice(0, 200),
                                };
                            })()""",
                        )
                    )
                    _ab(session, "find", "role", "button", "click", "--name", "关闭原件", "--exact")

                msg = dom.get("message") or {}
                displays = [
                    {
                        "evidence_id": b.get("evidence_id"),
                        "display_source": b.get("display_source"),
                        "rect_visible_norm": b.get("rect_visible_norm"),
                        "rendition_kind": b.get("rendition_kind"),
                        "read_status": b.get("read_status"),
                    }
                    for b in msg.get("blocks") or []
                    if b.get("type") == "source_image"
                ]
                trace = dom["turn"].get("trace") or {}
                questions_out.append(
                    {
                        "question_id": qid,
                        "turn_status": dom["turn"]["status"],
                        "route": trace.get("route"),
                        "clarification": msg.get("clarification"),
                        "vision": trace.get("vision"),
                        "visible_text": dom["text"],
                        "citations": dom["cits"],
                        "clicked_citations": clicks,
                        "displays": displays,
                    }
                )

            out = {"process_meta": meta, "questions": questions_out}
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
