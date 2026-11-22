"""G25 real Chromium + real DashScope (qwen3-vl-plus) end-to-end verification script.

Verifies:
1. G20 Q03 fix (Mixtral page 5 right subplot crossing midline -> page:5:image:1) & Q10 ambiguity.
2. All 6 new public layout scenarios across SensorLab-Layouts-CCBY4.pdf &
   SensorLab-Overlap-CCBY4.pptx:
   - Normal dual figures (page 1 left & right)
   - Right figure axis label crossing midline (page 2 right)
   - Left-right mirrored layout with left legend crossing midline (page 3 left & right)
   - Top-bottom stacked layout (page 4 top, bottom, and horizontal-direction refusal)
   - Single centered figure (page 5 directional refusal & plain figure read)
   - Clearly overlapping PPTX picture shapes (slide 1 object_ambiguous)
3. Four-way identity invariant on every answered image turn:
   (a) selected_source == display.source
   (b) wire_image_sha256 sent to qwen3-vl-plus == display.byte_sha256 == actual_input_sha256
   (c) answer citation revision.source == selected_source & media_url sha256 == wire_image_sha256
   (d) real Chromium citation click & "回原件" highlight open #reader for the same object/version.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV = Path("/Users/Link/code/SRAG/.env")
DEFAULT_PYTHON = Path("/Users/Link/code/SRAG/.venv/bin/python")


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


def _server_script(out: Path, env_file: Path, course_id: str, port: int) -> str:
    return f"""
import base64, hashlib, json, httpx, uvicorn
from pathlib import Path
from time import perf_counter
from app.api.workspace import get_workspace_service
from app.config import Settings, get_settings
from app.main import create_app
from app.rag.workspace import make_workspace

out = Path({str(out)!r})
settings = Settings(
    _env_file={str(env_file)!r},
    course_root=out / "courses",
    data_dir=out / "data",
    sqlite_path=out / "library.sqlite3",
    workspace_course_ids={course_id!r},
    workspace_max_upload_bytes=10 * 1024 * 1024,
    workspace_answer_enabled=True,
    workspace_retrieval_strategy="fts",
    source_vision_enabled=True,
)
app = create_app()
service = make_workspace(settings)
app.dependency_overrides[get_settings] = lambda: settings
app.dependency_overrides[get_workspace_service] = lambda: service

orig_sync = httpx.HTTPTransport.handle_request
def log_sync(transport, request):
    started = perf_counter()
    rec = {{"url": str(request.url), "method": request.method}}
    try:
        payload = json.loads(request.content)
        rec["model"] = payload.get("model")
        content = payload.get("input", {{}}).get("messages", [{{}}])[0].get("content", [])
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("image"), str):
                uri = part["image"]
                _, _, b64 = uri.partition(";base64,")
                raw_img = base64.b64decode(b64)
                rec["wire_image_bytes"] = len(raw_img)
                rec["wire_image_sha256"] = hashlib.sha256(raw_img).hexdigest()
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                rec["prompt_tail"] = part["text"][-180:]
    except Exception as exc:
        rec["parse_error"] = type(exc).__name__
    try:
        resp = orig_sync(transport, request)
        rec["status_code"] = resp.status_code
        rec["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
        with (out / "provider.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
        return resp
    except Exception as exc:
        rec["error"] = type(exc).__name__
        rec["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
        with (out / "provider.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
        raise

httpx.HTTPTransport.handle_request = log_sync

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port={port}, log_level="warning")
"""


def run_verification(
    *,
    code_dir: Path,
    env_file: Path,
    python_bin: Path,
    out_json: Path,
    port: int | None = None,
) -> dict[str, Any]:
    assigned_port = port or _free_port()
    course_id = "g25-browser-course"
    mixtral_pdf = code_dir / "docs" / "G" / "G20-fixtures" / "Mixtral-of-Experts-2401.04088v1.pdf"
    layouts_pdf = code_dir / "docs" / "G" / "G25-fixtures" / "SensorLab-Layouts-CCBY4.pdf"
    overlap_pptx = code_dir / "docs" / "G" / "G25-fixtures" / "SensorLab-Overlap-CCBY4.pptx"

    with tempfile.TemporaryDirectory(prefix="srag-g25-browser-") as tmp:
        out = Path(tmp)
        server_script = out / "serve_g25.py"
        server_script.write_text(
            _server_script(out, env_file, course_id, assigned_port), encoding="utf-8"
        )
        env = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(code_dir),
        }
        proc = subprocess.Popen(
            [str(python_bin), str(server_script)],
            cwd=str(code_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        session = f"g25-{assigned_port}-{int(time.time())}"
        try:
            base_url = f"http://127.0.0.1:{assigned_port}/workspace/"
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
                out_text, _ = proc.communicate(timeout=3)
                raise RuntimeError(f"Server failed to start on {base_url}: {out_text}")

            _ab(session, "--allowed-domains", "127.0.0.1", "open", base_url)
            _ab(session, "set", "viewport", "1340", "860")
            _ab(session, "wait", "--text", "暂无资料")
            user_agent = json.loads(_ab(session, "eval", "navigator.userAgent"))

            def upload_doc(file_path: Path) -> None:
                panel_open = json.loads(
                    _ab(session, "eval", 'document.querySelector("#import-panel").open')
                )
                if not panel_open:
                    _ab(session, "find", "text", "导入资料", "click", "--exact")
                _ab(session, "select", "#import-course", course_id)
                _ab(session, "upload", "#import-files", str(file_path))
                _ab(session, "find", "role", "button", "click", "--name", "开始导入", "--exact")
                _ab(session, "wait", "--text", f"{file_path.name} · 导入完成 · v1")
                _wait_js(
                    session,
                    '!document.querySelector("#import-button").disabled && '
                    '!document.querySelector("#import-files").files.length',
                )

            def select_catalog_doc(file_name: str) -> None:
                _ab(
                    session,
                    "eval",
                    f"""(() => {{
                        const btn = [...document.querySelectorAll('#catalog button')]
                            .find(b => b.textContent.includes({json.dumps(file_name)}));
                        if (!btn) throw new Error('Doc missing: ' + {json.dumps(file_name)});
                        btn.click();
                    }})()""",
                )
                _wait_js(
                    session,
                    f"document.querySelector('#location-title')?.textContent.includes("
                    f"{json.dumps(file_name)})",
                )

            turn_counter = 0

            def read_provider_log() -> list[dict[str, Any]]:
                log_path = out / "provider.jsonl"
                if not log_path.exists():
                    return []
                return [
                    json.loads(line)
                    for line in log_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]

            def ask_and_verify(
                case_id: str,
                question: str,
                *,
                expect_source_id: str | None = None,
                expect_notice: str | None = None,
                expect_keywords: tuple[str, ...] = (),
            ) -> dict[str, Any]:
                nonlocal turn_counter
                turn_counter += 1
                calls_before = len(read_provider_log())
                _ab(session, "find", "label", "输入问题", "fill", question)
                _wait_js(session, '!document.querySelector("#send").disabled')
                _ab(session, "find", "role", "button", "click", "--name", "发送", "--exact")
                idx = turn_counter - 1
                _wait_js(
                    session,
                    f"document.querySelectorAll('#messages article.message-user').length"
                    f" === {turn_counter} && ("
                    f"document.querySelectorAll('#messages article.message-user')"
                    f".item({idx}).nextElementSibling?.matches('article:not(.message-user)')"
                    f" && !document.querySelector('#turn-actions button')"
                    f"?.textContent.includes('取消'))",
                )

                # Inspect DOM message + fetch full API session state via browser cookie
                dom_state = json.loads(
                    _ab(
                        session,
                        "eval",
                        f"""(async () => {{
                            const users = document.querySelectorAll(
                                '#messages article.message-user'
                            );
                            const m = users.item({idx}).nextElementSibling;
                            const mid = m.dataset.messageId;
                            const btns = [...m.querySelectorAll('button[data-evidence-id]')];
                            const cards = [...m.querySelectorAll('figure.source-card')];
                            const sid = document.querySelector('#sessions')?.value
                                || window.localStorage.getItem('srag-workspace-session');
                            const sessResp = await fetch('/api/workspace/sessions/' + sid);
                            const sessJson = await sessResp.json();
                            const msgJson = (sessJson.messages || []).find(
                                x => x.message_id === mid
                            );
                            const turnJson = (sessJson.turns || []).at(-1);
                            return {{
                                message_id: mid,
                                text: m.innerText,
                                notices: [...m.querySelectorAll('.notice-block')].map(
                                    n => n.innerText
                                ),
                                citations: [...new Set(btns.map(b => b.dataset.evidenceId))],
                                card_count: cards.length,
                                turn_trace: turnJson?.trace,
                                msg_blocks: msgJson?.blocks,
                                msg_sources: msgJson?.sources,
                            }};
                        }})()""",
                    )
                )

                # Compute sha256 of the inline media URL in browser context if source_image exists
                preview_checks: dict[str, Any] = {}
                img_blocks = [
                    b
                    for b in (dom_state.get("msg_blocks") or [])
                    if b.get("type") == "source_image"
                ]
                if img_blocks:
                    blk = img_blocks[0]
                    cid = blk["evidence_id"]
                    mid = dom_state["message_id"]
                    media_url = blk["media_url"]
                    media_check = json.loads(
                        _ab(
                            session,
                            "eval",
                            f"""(async () => {{
                                const r = await fetch({json.dumps(media_url)});
                                const buf = await r.arrayBuffer();
                                const hashBuf = await crypto.subtle.digest('SHA-256', buf);
                                const hex = [...new Uint8Array(hashBuf)]
                                    .map(x => x.toString(16).padStart(2, '0')).join('');
                                return {{status: r.status, bytes: buf.byteLength, sha256: hex}};
                            }})()""",
                        )
                    )
                    preview_checks["media_fetch"] = media_check

                    # 1) Click citation button to open #reader
                    cit_sel = f"article[data-message-id='{mid}'] button[data-evidence-id='{cid}']"
                    _ab(
                        session,
                        "eval",
                        f'document.querySelector({json.dumps(cit_sel)}).scrollIntoView({{block:"center"}})',
                    )
                    _ab(session, "click", cit_sel)
                    _wait_js(
                        session,
                        'document.querySelector("#reader").open && '
                        'document.querySelector("#reader img")?.complete && '
                        'document.querySelector("#reader img").naturalWidth > 0',
                    )
                    cit_reader = json.loads(
                        _ab(
                            session,
                            "eval",
                            """(() => {
                                const r = document.querySelector("#reader");
                                const img = r.querySelector("img");
                                return {
                                    open: r.open,
                                    title: document.querySelector("#reader-title").innerText,
                                    location: document.querySelector("#reader-location").innerText,
                                    status: document.querySelector("#reader-status").innerText,
                                    img_src: img?.getAttribute("src"),
                                    natural_w: img?.naturalWidth,
                                    natural_h: img?.naturalHeight,
                                };
                            })()""",
                        )
                    )
                    preview_checks["citation_reader"] = cit_reader
                    _ab(session, "find", "role", "button", "click", "--name", "关闭原件", "--exact")

                    # 2) Click "回原件" on card to verify original page + highlight rect
                    orig_btn_sel = (
                        f"article[data-message-id='{mid}'] figure.source-card button"
                    )
                    _ab(
                        session,
                        "eval",
                        f"""(() => {{
                            const btn = [...document.querySelectorAll({json.dumps(orig_btn_sel)})]
                                .find(b => b.textContent.trim() === '回原件');
                            if (!btn) throw new Error('回原件 button missing');
                            btn.scrollIntoView({{block: 'center'}});
                            btn.click();
                        }})()""",
                    )
                    _wait_js(
                        session,
                        'document.querySelector("#reader").open && '
                        'document.querySelector("#reader img")?.complete && '
                        'document.querySelector("#reader img").naturalWidth > 0',
                    )
                    orig_reader = json.loads(
                        _ab(
                            session,
                            "eval",
                            """(() => {
                                const hl = document.querySelector("#reader .reader-highlight");
                                const img = document.querySelector("#reader img");
                                return {
                                    open: document.querySelector("#reader").open,
                                    location: document.querySelector("#reader-location").innerText,
                                    status: document.querySelector("#reader-status").innerText,
                                    img_src: img?.getAttribute("src"),
                                    highlight_style: hl ? {
                                        left: hl.style.left,
                                        top: hl.style.top,
                                        width: hl.style.width,
                                        height: hl.style.height,
                                    } : null,
                                };
                            })()""",
                        )
                    )
                    preview_checks["original_page_reader"] = orig_reader
                    _ab(session, "find", "role", "button", "click", "--name", "关闭原件", "--exact")

                provider_logs = read_provider_log()[calls_before:]
                trace = dom_state.get("turn_trace") or {}
                invariants_ok = True
                failures: list[str] = []

                if expect_source_id is not None:
                    selected_id = (trace.get("selected_source") or {}).get("source_id")
                    display = trace.get("display") or {}
                    display_id = (display.get("source") or {}).get("source_id")
                    display_sha = display.get("byte_sha256")
                    wire_sha = provider_logs[0].get("wire_image_sha256") if provider_logs else None
                    ev_id = img_blocks[0]["evidence_id"] if img_blocks else None
                    ev = (dom_state.get("msg_sources") or {}).get(ev_id or "", {})
                    ev_read = (ev.get("reads") or [{}])[0]
                    read_sha = ev_read.get("actual_input_sha256")
                    cit_source = (ev.get("revision") or {}).get("source") or {}
                    media_sha = (preview_checks.get("media_fetch") or {}).get("sha256")
                    cit_src = (preview_checks.get("citation_reader") or {}).get("img_src", "")

                    if selected_id != expect_source_id:
                        failures.append(f"selected_source={selected_id} != {expect_source_id}")
                    if display_id != expect_source_id:
                        failures.append(f"display_source={display_id} != {expect_source_id}")
                    if not wire_sha or wire_sha != display_sha or wire_sha != read_sha:
                        failures.append(
                            f"sha mismatch wire={wire_sha} display={display_sha} read={read_sha}"
                        )
                    if media_sha != wire_sha:
                        failures.append(f"media_sha={media_sha} != wire={wire_sha}")
                    if cit_source != trace.get("selected_source"):
                        failures.append("citation revision.source != selected_source")
                    if not img_blocks:
                        failures.append("missing source_image block")
                    else:
                        if img_blocks[0]["display_id"] not in cit_src:
                            failures.append(f"reader src {cit_src} missing display_id")
                    orig_reader = preview_checks.get("original_page_reader") or {}
                    if not orig_reader.get("highlight_style"):
                        failures.append("missing original page highlight")
                    low_text = dom_state["text"].casefold()
                    for kw in expect_keywords:
                        if kw.casefold() not in low_text:
                            failures.append(f"missing keyword {kw!r} in answer")
                    invariants_ok = not failures
                else:
                    codes = [b.get("code") for b in (dom_state.get("msg_blocks") or [])]
                    if expect_notice not in codes:
                        failures.append(f"expected notice {expect_notice}, got {codes}")
                    if provider_logs:
                        failures.append(f"unexpected external calls: {len(provider_logs)}")
                    invariants_ok = not failures

                return {
                    "case_id": case_id,
                    "question": question,
                    "expect_source_id": expect_source_id,
                    "expect_notice": expect_notice,
                    "invariants_ok": invariants_ok,
                    "failures": failures,
                    "trace": trace,
                    "provider_calls": provider_logs,
                    "answer_text": dom_state["text"],
                    "citations": dom_state["citations"],
                    "preview_checks": preview_checks,
                }

            # Upload all 3 fixtures cleanly
            upload_doc(mixtral_pdf)
            upload_doc(layouts_pdf)
            upload_doc(overlap_pptx)

            cases: list[dict[str, Any]] = []

            # Group 1: G20 Mixtral Q03 + Q10
            select_catalog_doc(mixtral_pdf.name)
            cases.append(
                ask_and_verify(
                    "G20_Q03_right_cross",
                    (
                        "第 5 页右侧图的横轴和纵轴分别表示什么？"
                        "图中纵轴的刻度范围大约从多少到多少，曲线随上下文长度增加呈现什么趋势？"
                    ),
                    expect_source_id="page:5:image:1",
                    expect_keywords=("context", "perplexity"),
                )
            )
            cases.append(
                ask_and_verify(
                    "G20_Q10_ambiguous",
                    "第 5 页的插图里具体展示了什么结果？",
                    expect_notice="object_ambiguous",
                )
            )

            # Group 2: New Public SensorLab-Layouts-CCBY4.pdf (Pages 1-5)
            select_catalog_doc(layouts_pdf.name)
            cases.append(
                ask_and_verify(
                    "NewPDF_P1_normal_right",
                    "第 1 页右图的横轴和纵轴分别是什么？",
                    expect_source_id="page:1:image:2",
                    expect_keywords=("current", "ripple"),
                )
            )
            cases.append(
                ask_and_verify(
                    "NewPDF_P2_right_label_cross",
                    "第 2 页右图的横轴、纵轴及刻度范围是什么？",
                    expect_source_id="page:2:image:2",
                    expect_keywords=("batch", "throughput"),
                )
            )
            cases.append(
                ask_and_verify(
                    "NewPDF_P3_mirrored_left",
                    "第 3 页左图的横轴和纵轴分别是什么？",
                    expect_source_id="page:3:image:1",
                    expect_keywords=("freq", "gain"),
                )
            )
            cases.append(
                ask_and_verify(
                    "NewPDF_P3_mirrored_right",
                    "第 3 页右图的横轴和纵轴分别是什么？",
                    expect_source_id="page:3:image:2",
                    expect_keywords=("bias", "floor"),
                )
            )
            cases.append(
                ask_and_verify(
                    "NewPDF_P4_stacked_top",
                    "第 4 页上图的横轴和纵轴分别是什么？",
                    expect_source_id="page:4:image:1",
                    expect_keywords=("wavenumber", "absorbance"),
                )
            )
            cases.append(
                ask_and_verify(
                    "NewPDF_P4_stacked_bottom",
                    "第 4 页下图的横轴和纵轴分别是什么？",
                    expect_source_id="page:4:image:2",
                    expect_keywords=("sample", "phase"),
                )
            )
            cases.append(
                ask_and_verify(
                    "NewPDF_P4_stacked_ask_right_refused",
                    "第 4 页右侧图的纵轴是什么？",
                    expect_notice="object_not_found",
                )
            )
            cases.append(
                ask_and_verify(
                    "NewPDF_P5_centered_ask_right_refused",
                    "第 5 页右图的纵轴是什么？",
                    expect_notice="object_not_found",
                )
            )
            cases.append(
                ask_and_verify(
                    "NewPDF_P5_centered_plain_read",
                    "第 5 页的插图里横轴和纵轴分别是什么？",
                    expect_source_id="page:5:image:1",
                    expect_keywords=("step", "residual"),
                )
            )

            # Group 3: New Public SensorLab-Overlap-CCBY4.pptx
            select_catalog_doc(overlap_pptx.name)
            cases.append(
                ask_and_verify(
                    "NewPPTX_overlap_ask_right_ambiguous",
                    "第 1 页右侧图的纵轴是什么？",
                    expect_notice="object_ambiguous",
                )
            )

            report = {
                "code_dir": str(code_dir),
                "port": assigned_port,
                "user_agent": user_agent,
                "total_cases": len(cases),
                "passed_cases": sum(1 for c in cases if c["invariants_ok"]),
                "all_passed": all(c["invariants_ok"] for c in cases),
                "cases": cases,
            }
            out_json.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return report
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
    parser.add_argument("--code-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_ROOT / "docs" / "G" / "G25-网页实测.json",
    )
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    report = run_verification(
        code_dir=args.code_dir.resolve(),
        env_file=args.env_file.resolve(),
        python_bin=Path(os.path.abspath(args.python)),
        out_json=args.out.resolve(),
        port=args.port,
    )
    print(
        json.dumps(
            {
                "all_passed": report["all_passed"],
                "passed_cases": report["passed_cases"],
                "total_cases": report["total_cases"],
                "failures": [
                    {"case_id": c["case_id"], "failures": c["failures"]}
                    for c in report["cases"]
                    if not c["invariants_ok"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
