"""Reproduce G20 Q09 in real Chromium (/workspace/) on an isolated server."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_ENV = Path("/Users/Link/code/SRAG/.env")
V1_NB = REPO_ROOT / "docs" / "G" / "G20-fixtures" / "v1" / "Mixtral-Router-Verification.ipynb"
V2_NB = REPO_ROOT / "docs" / "G" / "G20-fixtures" / "v2" / "Mixtral-Router-Verification.ipynb"

Q08A = (
    "请展示第 1 个 Cell 的保存输出，并说明实测的平均路由分发延迟、"
    "容量因子以及评估词元数分别是多少？"
)
Q09 = (
    "跟上一版相比，第 1 个 Cell 保存输出里的平均路由分发延迟、"
    "容量因子和评估词元数有什么变化？"
)


def _ab(session: str, *args: str, timeout: int = 120) -> str:
    cmd = ["agent-browser", "--session", session, *args]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"agent-browser failed ({' '.join(cmd)}): {res.stderr or res.stdout}")
    return res.stdout.strip()


def _wait_js(session: str, expr: str, timeout_ms: int = 90_000) -> None:
    _ab(
        session,
        "wait",
        "--timeout",
        str(timeout_ms),
        "--fn",
        expr,
        timeout=timeout_ms // 1000 + 20,
    )


def run_q09_repro(*, port: int, out_path: Path) -> dict[str, Any]:
    python_bin = "/Users/Link/code/SRAG/.venv/bin/python"
    course_id = "g24-repro-course"

    with tempfile.TemporaryDirectory(prefix="srag-g24-repro-") as tmp:
        out = Path(tmp)
        server_script = out / "serve_g24.py"
        server_script.write_text(
            f"""
import json, httpx, uvicorn
from pathlib import Path
from time import perf_counter
from app.api.workspace import get_workspace_service
from app.config import Settings, get_settings
from app.main import create_app
from app.rag.workspace import make_workspace

out = Path({str(out)!r})
settings = Settings(
    _env_file={str(MAIN_ENV)!r},
    app_host="127.0.0.1",
    app_port={port},
    sqlite_path=out / "workspace.sqlite3",
    data_dir=out / "data",
    course_root=out / "courses",
    workspace_course_ids={course_id!r},
    workspace_max_upload_bytes=16 * 1024 * 1024,
    workspace_answer_enabled=True,
    source_vision_enabled=True,
    workspace_retrieval_strategy="fts",
    workspace_input_token_limit=32_768,
)
service = make_workspace(settings)
app = create_app()
app.dependency_overrides[get_settings] = lambda: settings
app.dependency_overrides[get_workspace_service] = lambda: service

orig_async = httpx.AsyncHTTPTransport.handle_async_request

async def log_async(transport, request):
    started = perf_counter()
    raw_body = request.content.decode("utf-8", "replace") if request.content else ""
    try:
        payload = json.loads(raw_body) if raw_body else {{}}
    except Exception:
        payload = {{}}
    rec = {{
        "url": str(request.url),
        "model": payload.get("model"),
        "request_bytes": len(request.content or b""),
    }}
    try:
        resp = await orig_async(transport, request)
        rec["status_code"] = resp.status_code
        rec["elapsed_ms"] = round((perf_counter() - started) * 1000, 1)
        with (out / "provider.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
        return resp
    except Exception as exc:
        rec["error"] = type(exc).__name__
        rec["elapsed_ms"] = round((perf_counter() - started) * 1000, 1)
        with (out / "provider.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
        raise

httpx.AsyncHTTPTransport.handle_async_request = log_async

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port={port}, log_level="warning")
""",
            encoding="utf-8",
        )

        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(REPO_ROOT)}
        proc = subprocess.Popen(
            [python_bin, str(server_script)],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        session = f"g24-repro-{port}-{int(time.time())}"
        try:
            url = f"http://127.0.0.1:{port}/workspace/"
            for _ in range(40):
                r = subprocess.run(
                    ["curl", "--silent", "--fail", url],
                    capture_output=True,
                    timeout=2,
                    check=False,
                )
                if r.returncode == 0:
                    break
                time.sleep(0.25)
            else:
                raise RuntimeError(f"Server failed to start on {url}")

            _ab(session, "--allowed-domains", "127.0.0.1", "open", url)
            _ab(session, "set", "viewport", "1310", "820")
            _ab(session, "wait", "--text", "暂无资料")
            user_agent = json.loads(_ab(session, "eval", "navigator.userAgent"))

            def import_nb(nb_path: Path, *, replace: bool, expect_ver: int) -> None:
                panel_open = json.loads(
                    _ab(session, "eval", 'document.querySelector("#import-panel").open')
                )
                if not panel_open:
                    _ab(session, "find", "text", "导入资料", "click", "--exact")
                _ab(session, "select", "#import-course", course_id)
                if replace:
                    _wait_js(session, '!document.querySelector("#replace-label").hidden')
                    checked = json.loads(
                        _ab(session, "eval", 'document.querySelector("#replace-document").checked')
                    )
                    if not checked:
                        _ab(session, "click", "#replace-document")
                _ab(session, "upload", "#import-files", str(nb_path))
                _ab(session, "find", "role", "button", "click", "--name", "开始导入", "--exact")
                _ab(session, "wait", "--text", f"{nb_path.name} · 导入完成 · v{expect_ver}")
                _wait_js(
                    session,
                    '!document.querySelector("#import-button").disabled && '
                    '!document.querySelector("#import-files").files.length',
                )

            def select_doc(file_name: str) -> None:
                _ab(
                    session,
                    "eval",
                    """(() => {
                        const cb = document.querySelector("#catalog button[data-kind='course']");
                        if (cb && cb.getAttribute("aria-expanded") !== "true") cb.click();
                    })()""",
                )
                _wait_js(
                    session,
                    f"[...document.querySelectorAll('#catalog button')]"
                    f".some(b => b.textContent.includes({file_name!r}))",
                )
                _ab(
                    session,
                    "eval",
                    f"""(() => {{
                        const db = [...document.querySelectorAll('#catalog button')]
                            .find(b => b.textContent.includes({file_name!r}));
                        db.click();
                    }})()""",
                )
                _wait_js(
                    session,
                    f"document.querySelector('#scope-summary')?.textContent.includes({file_name!r})",
                )

            def ask_next(question_text: str, user_index: int) -> dict[str, Any]:
                t0 = time.perf_counter()
                _ab(session, "find", "label", "输入问题", "fill", question_text)
                _wait_js(session, '!document.querySelector("#send").disabled')
                _ab(session, "find", "role", "button", "click", "--name", "发送", "--exact")
                _wait_js(
                    session,
                    f"document.querySelectorAll('#messages article.message-user')"
                    f".length === {user_index + 1} && ("
                    f"document.querySelectorAll('#messages article.message-user')"
                    f".item({user_index})"
                    ".nextElementSibling?.matches('article:not(.message-user)') || "
                    "document.querySelector('#turn-status')?.textContent.includes('处理失败') || "
                    "document.querySelector('#turn-status')?.textContent.includes('需要澄清'))",
                )
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
                dom_msg = json.loads(
                    _ab(
                        session,
                        "eval",
                        f"""(() => {{
                            const m = document.querySelectorAll('#messages article.message-user')
                                .item({user_index})?.nextElementSibling;
                            if (!m?.matches('article:not(.message-user)')) {{
                                return {{
                                    missing: true,
                                    status: document.querySelector('#turn-status')?.textContent,
                                }};
                            }}
                            const btns = [...m.querySelectorAll('button[data-evidence-id]')];
                            return {{
                                id: m.dataset.messageId,
                                text: m.innerText,
                                notices: [...m.querySelectorAll('.notice')].map(n => n.innerText),
                                citations: [...new Set(btns.map(b => b.dataset.evidenceId))],
                            }};
                        }})()""",
                    )
                )
                opened: dict[str, Any] = {}
                for cid in dom_msg.get("citations", []):
                    mid = dom_msg["id"]
                    selector = f"article[data-message-id='{mid}'] button[data-evidence-id='{cid}']"
                    _ab(
                        session,
                        "eval",
                        f'document.querySelector("{selector}").scrollIntoView({{block:"center"}})',
                    )
                    _ab(session, "click", selector)
                    _wait_js(
                        session,
                        'document.querySelector("#reader").open && '
                        '!document.querySelector("#reader-body").innerText.includes("正在加载原件")',
                    )
                    opened[cid] = json.loads(
                        _ab(
                            session,
                            "eval",
                            """(() => ({
                                open: document.querySelector("#reader").open,
                                location: document.querySelector("#reader-location").innerText,
                                title: document.querySelector("#reader-title").innerText,
                                status: document.querySelector("#reader-status").innerText,
                                body_head: (
                                    document.querySelector("#reader-body").innerText.slice(0, 260)
                                ),
                            }))()""",
                        )
                    )
                    _ab(session, "find", "role", "button", "click", "--name", "关闭原件", "--exact")

                sess_payload = json.loads(
                    _ab(
                        session,
                        "eval",
                        """(async () => {
                            const sel = document.querySelector('#sessions');
                            const sid = (sel && sel.value)
                                || (await (await fetch('/api/workspace/sessions')).json())
                                    .sessions[0]?.session_id;
                            return await (await fetch(`/api/workspace/sessions/${sid}`)).json();
                        })()""",
                    )
                )
                return {
                    "elapsed_ms": elapsed_ms,
                    "dom": {**dom_msg, "opened_citations": opened},
                    "turn": sess_payload["turns"][-1] if sess_payload.get("turns") else None,
                    "message_meta": sess_payload["messages"][-1]
                    if sess_payload.get("messages")
                    else None,
                }

            import_nb(V1_NB, replace=False, expect_ver=1)
            select_doc(V1_NB.name)
            q08a = ask_next(Q08A, 0)
            import_nb(V2_NB, replace=True, expect_ver=2)
            q09 = ask_next(Q09, 1)

            provider_log = out / "provider.jsonl"
            provider_calls = (
                [
                    json.loads(line)
                    for line in provider_log.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if provider_log.exists()
                else []
            )
            result = {
                "user_agent": user_agent,
                "q08a": q08a,
                "q09": q09,
                "provider_calls": provider_calls,
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return result
        finally:
            try:
                _ab(session, "close")
            except Exception:
                pass
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18924)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs" / "G" / "G24-基线Q09复现.json",
    )
    args = parser.parse_args()
    res = run_q09_repro(port=args.port, out_path=args.out)
    print(
        json.dumps(
            {
                "q09_route": res["q09"]["turn"]["trace"].get("route"),
                "q09_sources": {
                    k: v["revision"]["source"]["version"]
                    for k, v in res["q09"]["message_meta"]["sources"].items()
                },
                "q09_text": res["q09"]["dom"]["text"],
                "opened_citations": {
                    k: v["location"] for k, v in res["q09"]["dom"]["opened_citations"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
