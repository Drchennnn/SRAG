"""G24 real-Chromium blind verification with brand-new filenames and natural phrasings."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import nbformat
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_ENV = Path("/Users/Link/code/SRAG/.env")


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


def _plot_png(title: str, x_label: str, color: str) -> str:
    img = Image.new("RGB", (320, 180), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([28, 20, 296, 140], outline="black", width=2)
    draw.line([(40, 120), (140, 80), (270, 42)], fill=color, width=3)
    draw.text((34, 6), title, fill="black")
    draw.text((88, 148), x_label, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_blind_fixtures(fixture_dir: Path) -> dict[str, Path]:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    v1_path = fixture_dir / "Zephyr-Cache-Benchmark-v1.ipynb"
    v2_path = fixture_dir / "Zephyr-Cache-Benchmark-v2.ipynb"
    other_path = fixture_dir / "Cobalt-Quant-Audit.ipynb"

    img_v1_a = _plot_png("KV Hit Curve (v1)", "Cache Size [MB]", "navy")
    img_v1_b = _plot_png("P95 Tail Trace (v1)", "Concurrency [reqs]", "darkred")
    img_v2_a = _plot_png("KV Hit Curve (v2)", "Window Slots [blocks]", "teal")
    img_v2_b = _plot_png("P95 Tail Trace (v2)", "Batch Queue [items]", "purple")

    nb_v1 = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "run_kv_cache_eval(batch_tokens=16384)",
                id="kv-cache-bench",
                execution_count=1,
                outputs=[
                    nbformat.v4.new_output(
                        "stream",
                        name="stdout",
                        text=(
                            "Dataset: Zephyr-Cache-16k\n"
                            "Cache hit rate: 78.4%\n"
                            "P95 decode latency: 18.6 ms\n"
                            "Throughput: 1420 tok/s\n"
                        ),
                    ),
                    nbformat.v4.new_output("display_data", data={"image/png": img_v1_a}),
                    nbformat.v4.new_output("display_data", data={"image/png": img_v1_b}),
                ],
            ),
            nbformat.v4.new_code_cell(
                "run_eviction_audit()",
                id="eviction-policy-v1",
                execution_count=2,
                outputs=[
                    nbformat.v4.new_output(
                        "stream",
                        name="stdout",
                        text="Eviction policy: LRU\nEviction overhead: 2.4 ms\n",
                    )
                ],
            ),
        ]
    )
    nb_v2 = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "run_kv_cache_eval(batch_tokens=32768)",
                id="kv-cache-bench",
                execution_count=1,
                outputs=[
                    nbformat.v4.new_output(
                        "stream",
                        name="stdout",
                        text=(
                            "Dataset: Zephyr-Cache-32k\n"
                            "Cache hit rate: 91.2%\n"
                            "P95 decode latency: 11.4 ms\n"
                            "Throughput: 2380 tok/s\n"
                        ),
                    ),
                    nbformat.v4.new_output("display_data", data={"image/png": img_v2_a}),
                    nbformat.v4.new_output("display_data", data={"image/png": img_v2_b}),
                ],
            ),
            nbformat.v4.new_code_cell(
                "run_eviction_audit()",
                id="eviction-policy-v2",
                execution_count=2,
                outputs=[
                    nbformat.v4.new_output(
                        "stream",
                        name="stdout",
                        text="Eviction policy: Adaptive-LFU\nEviction overhead: 1.1 ms\n",
                    )
                ],
            ),
        ]
    )
    nb_other = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "run_quant_audit()",
                id="quant-check",
                execution_count=1,
                outputs=[
                    nbformat.v4.new_output(
                        "stream",
                        name="stdout",
                        text="Quantization mode: INT4-AWQ\nPerplexity delta: +0.08\n",
                    )
                ],
            )
        ]
    )
    nbformat.write(nb_v1, v1_path)
    nbformat.write(nb_v2, v2_path)
    nbformat.write(nb_other, other_path)
    return {"v1": v1_path, "v2": v2_path, "other": other_path}


def run_blind_verify(*, port: int, out_path: Path) -> dict[str, Any]:
    python_bin = "/Users/Link/code/SRAG/.venv/bin/python"
    course_id = "g24-blind-course"

    with tempfile.TemporaryDirectory(prefix="srag-g24-blind-") as tmp:
        out = Path(tmp)
        fixtures = _build_blind_fixtures(out / "fixtures")
        # Upload under identical filename Zephyr-Cache-Benchmark.ipynb across v1 -> v2
        upload_v1 = out / "upload_v1" / "Zephyr-Cache-Benchmark.ipynb"
        upload_v2 = out / "upload_v2" / "Zephyr-Cache-Benchmark.ipynb"
        upload_v1.parent.mkdir(parents=True, exist_ok=True)
        upload_v2.parent.mkdir(parents=True, exist_ok=True)
        upload_v1.write_bytes(fixtures["v1"].read_bytes())
        upload_v2.write_bytes(fixtures["v2"].read_bytes())

        server_script = out / "serve_g24_blind.py"
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
        session = f"g24-blind-{port}-{int(time.time())}"
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

            def clear_file_scope() -> None:
                _ab(
                    session,
                    "eval",
                    """(() => {
                        const cf = document.querySelector('#clear-file');
                        if (cf && !cf.hidden) cf.click();
                    })()""",
                )
                _wait_js(session, "document.querySelector('#clear-file')?.hidden !== false")

            def new_clean_session() -> None:
                prev_sid = _ab(session, "eval", "document.querySelector('#sessions')?.value || ''")
                _ab(session, "click", "#new-session")
                _wait_js(
                    session,
                    f"document.querySelector('#sessions')?.value && "
                    f"document.querySelector('#sessions')?.value !== {prev_sid} && "
                    f"document.querySelector('#catalog')?.getAttribute('aria-busy') === 'false'",
                )

            def ask_next(question_text: str) -> dict[str, Any]:
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
                    "question": question_text,
                    "elapsed_ms": elapsed_ms,
                    "dom": {**dom_msg, "opened_citations": opened},
                    "turn": sess_payload["turns"][-1] if sess_payload.get("turns") else None,
                    "message_meta": sess_payload["messages"][-1]
                    if sess_payload.get("messages")
                    else None,
                }

            # 1. Import Zephyr v1 and Cobalt v1
            import_nb(upload_v1, replace=False, expect_ver=1)
            import_nb(fixtures["other"], replace=False, expect_ver=1)
            select_doc("Zephyr-Cache-Benchmark.ipynb")

            # Warm up v1 citations in Cell 1 & Cell 2
            t_v1_cell1 = ask_next(
                "请展示第 1 个 Cell 的保存输出，并说明命中率、P95 延迟和吞吐量分别是多少？"
            )
            t_v1_cell2 = ask_next("第 2 个 Cell 保存输出里的淘汰策略和耗时是什么？")

            # Replace Zephyr with v2 in-place
            import_nb(upload_v2, replace=True, expect_ver=2)

            # Blind cases on Session 1
            cases: dict[str, Any] = {
                "warmup_v1_cell1": t_v1_cell1,
                "warmup_v1_cell2": t_v1_cell2,
                # Meaning 1a: Natural phrasing without "新版", explicit cell
                "meaning1_natural_explicit": ask_next(
                    "跟之前那个版本比起来，第 1 个 Cell 保存输出里的命中率和 P95 延迟怎么变了？"
                ),
                # Meaning 1b: Natural implicit phrasing without cell location
                "meaning1_natural_implicit": ask_next("和上一版相比有什么变化？"),
                # Meaning 2a: Only look back at old version (image follow-up -> historical_only)
                "meaning2_old_image_only": ask_next("刚才旧版展示的第 2 张图横轴是什么？"),
                # Meaning 2b: Only look back at old version (text only -> not version_comparison)
                "meaning2_old_text_only": ask_next(
                    "之前那版第 1 个 Cell 保存输出里的 P95 延迟是多少？"
                ),
                # Meaning 3: Compare two figures inside current version (must stay on v2)
                "meaning3_current_two_images": ask_next("比较当前版里的两张图片有什么区别"),
                # Meaning 4a: Mentions "旧版" with explicit negated comparison
                "meaning4_negated_compare": ask_next(
                    "不用和旧版对比，直接告诉我现在第 1 个 Cell 保存输出的命中率是多少？"
                ),
                # Meaning 4b: Mentions "旧版" as background + "变化趋势"
                "meaning4_background_trend": ask_next(
                    "我们已经换掉旧版了，说说当前版第 1 个 Cell 里延迟的变化趋势"
                ),
                # Unclear object correspondence: Cell 2 changed cell_id in v2
                "boundary_unclear_cell2": ask_next(
                    "跟上一版相比，第 2 个 Cell 的保存输出有什么变化？"
                ),
            }

            # Scope switch: clear file to course-wide (2 files, no focus)
            clear_file_scope()
            cases["boundary_scope_course_wide"] = ask_next("跟上一版相比有什么变化？")

            # Scope switch: select Cobalt-Quant-Audit.ipynb (different file)
            select_doc("Cobalt-Quant-Audit.ipynb")
            cases["boundary_scope_other_file"] = ask_next("跟上一版相比有什么变化？")

            # Missing history: new clean session (v2 only, no v1 citations)
            new_clean_session()
            select_doc("Zephyr-Cache-Benchmark.ipynb")
            cases["boundary_missing_history_new_session"] = ask_next(
                "跟上一版相比，第 1 个 Cell 保存输出有什么变化？"
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
    parser.add_argument("--port", type=int, default=18926)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs" / "G" / "G24-盲验实测.json",
    )
    args = parser.parse_args()
    res = run_blind_verify(port=args.port, out_path=args.out)
    summary = {}
    for name, item in res["cases"].items():
        sources = (item.get("message_meta") or {}).get("sources") or {}
        summary[name] = {
            "route": ((item.get("turn") or {}).get("trace") or {}).get("route"),
            "status": (item.get("turn") or {}).get("status"),
            "notices": item["dom"].get("notices"),
            "source_versions": {k: v["revision"]["source"]["version"] for k, v in sources.items()},
            "opened_citations": {
                k: {
                    "location": v["location"],
                    "title": v["title"],
                    "body_head": v["body_head"][:80],
                }
                for k, v in item["dom"].get("opened_citations", {}).items()
            },
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
