"""G27 real-Chromium verification for borderless PDF tables, crops, citations, and fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_ENV = Path("/Users/Link/code/SRAG/.env")
PAGED_ATTENTION = REPO_ROOT / "docs/G/G20-fixtures/PagedAttention-2309.06180v1.pdf"
MIXTRAL = REPO_ROOT / "docs/G/G20-fixtures/Mixtral-of-Experts-2401.04088v1.pdf"
MAMBA = REPO_ROOT / "docs/G/G21-fixtures/Mamba-2312.00752v1.pdf"


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


def run_browser_verify(*, port: int, out_path: Path) -> dict[str, Any]:
    python_bin = "/Users/Link/code/SRAG/.venv/bin/python"
    course_id = "g27-table-course"

    with tempfile.TemporaryDirectory(prefix="srag-g27-browser-") as tmp:
        out = Path(tmp)
        server_script = out / "serve_g27.py"
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

orig_sync = httpx.HTTPTransport.handle_request
orig_async = httpx.AsyncHTTPTransport.handle_async_request

def log_sync(transport, request):
    started = perf_counter()
    rec = {{"url": str(request.url), "kind": "sync"}}
    try:
        resp = orig_sync(transport, request)
        rec["status_code"] = resp.status_code
        rec["elapsed_ms"] = round((perf_counter() - started) * 1000, 1)
        with (out / "provider.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
        return resp
    except Exception as exc:
        rec["error"] = type(exc).__name__
        with (out / "provider.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
        raise

async def log_async(transport, request):
    started = perf_counter()
    rec = {{"url": str(request.url), "kind": "async"}}
    try:
        resp = await orig_async(transport, request)
        rec["status_code"] = resp.status_code
        rec["elapsed_ms"] = round((perf_counter() - started) * 1000, 1)
        with (out / "provider.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
        return resp
    except Exception as exc:
        rec["error"] = type(exc).__name__
        with (out / "provider.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
        raise

httpx.HTTPTransport.handle_request = log_sync
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
        session = f"g27-browser-{port}-{int(time.time())}"
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

            def upload_pdf(pdf_path: Path) -> None:
                panel_open = json.loads(
                    _ab(session, "eval", 'document.querySelector("#import-panel").open')
                )
                if not panel_open:
                    _ab(session, "find", "text", "导入资料", "click", "--exact")
                _ab(session, "select", "#import-course", course_id)
                _ab(session, "upload", "#import-files", str(pdf_path))
                _ab(session, "find", "role", "button", "click", "--name", "开始导入", "--exact")
                _ab(session, "wait", "--text", f"{pdf_path.name} · 导入完成 · v1")
                _wait_js(
                    session,
                    '!document.querySelector("#import-button").disabled && '
                    '!document.querySelector("#import-files").files.length',
                )

            def new_clean_session() -> None:
                prev_sid = _ab(session, "eval", "document.querySelector('#sessions')?.value || ''")
                _ab(session, "click", "#new-session")
                _wait_js(
                    session,
                    f"document.querySelector('#sessions')?.value && "
                    f"document.querySelector('#sessions')?.value !== {prev_sid} && "
                    f"document.querySelector('#catalog')?.getAttribute('aria-busy') === 'false'",
                )

            def select_doc_page(file_name: str, page_no: int | None = None) -> None:
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
                        const wantOpen = {json.dumps(page_no is not None)};
                        const isOpen = db.getAttribute("aria-expanded") === "true";
                        const summary = document.querySelector('#scope-summary')?.textContent || "";
                        const scopeHas = summary.includes({file_name!r});
                        if (!scopeHas || (wantOpen && !isOpen)) db.click();
                    }})()""",
                )
                _wait_js(
                    session,
                    f"document.querySelector('#scope-summary')?.textContent.includes({file_name!r})",
                )
                if page_no is not None:
                    label = f"第 {page_no} 页"
                    _wait_js(
                        session,
                        f"[...document.querySelectorAll('#catalog button')]"
                        f".some(b => b.textContent.includes({label!r}))",
                    )
                    _ab(
                        session,
                        "eval",
                        f"[...document.querySelectorAll('#catalog button')]"
                        f".find(b => b.textContent.includes({label!r})).click()",
                    )
                    _wait_js(
                        session,
                        f"!document.querySelector('#location')?.hidden && "
                        f"document.querySelector('#location-title')?.textContent.includes({label!r})",
                    )

            def ask_one(question_text: str) -> dict[str, Any]:
                prev_count = json.loads(
                    _ab(
                        session,
                        "eval",
                        "document.querySelectorAll('#messages article.message-user').length",
                    )
                )
                t0 = time.perf_counter()
                _ab(session, "find", "label", "输入问题", "fill", question_text)
                _wait_js(session, '!document.querySelector("#send").disabled')
                _ab(session, "find", "role", "button", "click", "--name", "发送", "--exact")
                _wait_js(
                    session,
                    f"document.querySelectorAll('#messages article.message-user')"
                    f".length === {prev_count + 1} && ("
                    f"document.querySelectorAll('#messages article.message-user')"
                    f".item({prev_count})"
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
                                .item({prev_count})?.nextElementSibling;
                            if (!m?.matches('article:not(.message-user)')) {{
                                return {{
                                    missing: true,
                                    status: document.querySelector('#turn-status')?.textContent,
                                }};
                            }}
                            const btns = [...m.querySelectorAll('button[data-evidence-id]')];
                            const imgs = [...m.querySelectorAll('img')].map(img => ({{
                                src: img.getAttribute('src'),
                                complete: img.complete,
                                naturalWidth: img.naturalWidth,
                                naturalHeight: img.naturalHeight,
                            }}));
                            return {{
                                id: m.dataset.messageId,
                                text: m.innerText,
                                citations: [...new Set(btns.map(b => b.dataset.evidenceId))],
                                images: imgs,
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
                            """(() => {
                                const hl = document.querySelector(".reader-highlight");
                                return {
                                    open: document.querySelector("#reader").open,
                                    location: document.querySelector("#reader-location").innerText,
                                    title: document.querySelector("#reader-title").innerText,
                                    status: document.querySelector("#reader-status").innerText,
                                    has_highlight: Boolean(hl),
                                    body_head: document.querySelector(
                                        "#reader-body"
                                    ).innerText.slice(0, 260),
                                };
                            })()""",
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
                    "question": question_text,
                    "elapsed_ms": elapsed_ms,
                    "dom": {**dom_msg, "opened_citations": opened},
                    "turn": sess_payload["turns"][-1] if sess_payload.get("turns") else None,
                    "message_meta": sess_payload["messages"][-1]
                    if sess_payload.get("messages")
                    else None,
                }

            for pdf in (PAGED_ATTENTION, MIXTRAL, MAMBA):
                upload_pdf(pdf)

            cases: dict[str, Any] = {}

            # 1. H02: PagedAttention Page 9 Table 1 (borderless booktabs)
            new_clean_session()
            select_doc_page(PAGED_ATTENTION.name, 9)
            cases["h02_paged_attention_p9_table1"] = ask_one(
                "第 9 页表格里，13B、66B 和 175B 三种模型配置分别使用了多少张什么型号的 GPU？"
                "它们留给 KV 缓存的内存大小和最大 KV 缓存槽位数（slots）分别是多少？"
            )

            # 2. Mixtral Page 8 Table 5 (2-row spanner header)
            new_clean_session()
            select_doc_page(MIXTRAL.name, 8)
            cases["mixtral_p8_table5"] = ask_one(
                "第 8 页表格里的 ArXiv 和 DM Mathematics 数据集，"
                "在第 0 层、第 15 层和第 31 层（First choice）"
                "连续两个词元选到同一个专家的重复比例分别是多少？"
            )

            # 3. Unseen Mamba Page 11 Table
            new_clean_session()
            select_doc_page(MAMBA.name, 11)
            cases["mamba_p11_table"] = ask_one(
                "第 11 页表格里 Mamba 模型对应的 Arch.、Layer 和 Acc. 分别是多少？"
            )

            # 4. Page-level fallback / no false table on PagedAttention Page 5 (chart gridlines)
            new_clean_session()
            select_doc_page(PAGED_ATTENTION.name, 5)
            cases["paged_attention_p5_no_false_table"] = ask_one("第 5 页表格里写了什么？")
            cases["paged_attention_p5_page_text"] = ask_one(
                "请说明第 5 页关于 PagedAttention 的内容。"
            )

            # 5. Real vision read on Mixtral Page 6 leaderboard image
            new_clean_session()
            select_doc_page(MIXTRAL.name, 6)
            cases["mixtral_p6_real_vision"] = ask_one(
                "第 6 页的排行榜图里，Mixtral-8x7b-Instruct-v0.1 的 Arena Elo 评分是多少？"
            )

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
                "fixture_sha256": {
                    PAGED_ATTENTION.name: hashlib.sha256(PAGED_ATTENTION.read_bytes()).hexdigest(),
                    MIXTRAL.name: hashlib.sha256(MIXTRAL.read_bytes()).hexdigest(),
                    MAMBA.name: hashlib.sha256(MAMBA.read_bytes()).hexdigest(),
                },
                "cases": cases,
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
    parser.add_argument("--port", type=int, default=18929)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs" / "G" / "G27-实测.json",
    )
    args = parser.parse_args()
    res = run_browser_verify(port=args.port, out_path=args.out)
    summary = {}
    for name, item in res["cases"].items():
        turn = item.get("turn") or {}
        trace = turn.get("trace") or {}
        summary[name] = {
            "route": trace.get("route"),
            "selected_source": trace.get("selected_source"),
            "images": [
                (img["naturalWidth"], img["naturalHeight"])
                for img in item["dom"].get("images", [])
            ],
            "opened_citations": {
                k: {
                    "location": v["location"],
                    "title": v["title"],
                    "has_highlight": v["has_highlight"],
                }
                for k, v in item["dom"].get("opened_citations", {}).items()
            },
        }
    print(
        json.dumps(
            {"summary": summary, "provider_calls": res["provider_calls"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
