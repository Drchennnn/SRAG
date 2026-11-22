"""G22 Step 3 & Step 4 real-Chromium blind evaluation on QLoRA (v1) + G20 Q05/Q06."""

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
CRITERIA_PATH = REPO_ROOT / "docs" / "G" / "G22-盲测冻结判据.json"
QLORA_PDF = REPO_ROOT / "docs" / "G" / "G22-fixtures" / "QLoRA-2305.14314v1.pdf"
DPO_PDF = REPO_ROOT / "docs" / "G" / "G20-fixtures" / "Direct-Preference-Optimization-2305.18290v3.pdf"
MIXTRAL_PDF = REPO_ROOT / "docs" / "G" / "G20-fixtures" / "Mixtral-of-Experts-2401.04088v1.pdf"

G20_Q05_TEXT = (
    "在这份论文的附录实现细节里，默认使用的优化器、学习率、线性预热步数和批次大小分别是多少？"
    "默认的温度惩罚系数是多少，在做文本摘要实验时改成了多少？"
)
G20_Q06_TEXT = (
    "Mixtral 和 DPO 这两份资料里，Mixtral-8x7B-Instruct 微调时用了哪两种方法？"
    "而在 DPO 论文附录里 TL;DR 摘要的 β 参数和默认 β 分别是多少？"
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


def _run_browser_arm(
    *,
    arm_name: str,
    code_dir: Path,
    env_file: Path,
    port: int,
    strategy: str,
    questions: list[dict[str, Any]],
    include_g20_cross: bool = False,
) -> dict[str, Any]:
    python_bin = str(REPO_ROOT / ".venv" / "bin" / "python")
    course_id = "g22-blind-course"

    with tempfile.TemporaryDirectory(prefix=f"srag-g22-{arm_name}-") as tmp:
        out = Path(tmp)
        server_script = out / "serve_g22.py"
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
    _env_file={str(env_file)!r},
    app_host="127.0.0.1",
    app_port={port},
    sqlite_path=out / "workspace.sqlite3",
    data_dir=out / "data",
    course_root=out / "courses",
    workspace_course_ids={course_id!r},
    workspace_max_upload_bytes=16 * 1024 * 1024,
    workspace_answer_enabled=True,
    source_vision_enabled=True,
    workspace_retrieval_strategy={strategy!r},
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

        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(code_dir)}
        proc = subprocess.Popen(
            [python_bin, str(server_script)],
            cwd=str(code_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        session = f"g22-{arm_name}-{port}-{int(time.time())}"
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

            def upload_doc(pdf_path: Path) -> None:
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

            def select_doc(file_name: str, *, page_no: int | None = None) -> None:
                _ab(
                    session,
                    "eval",
                    f"""(() => {{
                        const cb = document.querySelector("#catalog button[data-kind='course']");
                        if (cb && cb.getAttribute("aria-expanded") !== "true") cb.click();
                    }})()""",
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
                        const scopeHas = (document.querySelector('#scope-summary')?.textContent || "")
                            .includes({file_name!r});
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

            def select_course_scope() -> None:
                _ab(
                    session,
                    "eval",
                    """(() => {
                        const cf = document.querySelector('#clear-file');
                        if (cf && !cf.hidden) cf.click();
                    })()""",
                )
                _wait_js(
                    session,
                    "document.querySelector('#clear-file')?.hidden !== false",
                )

            def ask_one(question_text: str) -> dict[str, Any]:
                t0 = time.perf_counter()
                _ab(session, "find", "label", "输入问题", "fill", question_text)
                _wait_js(session, '!document.querySelector("#send").disabled')
                _ab(session, "find", "role", "button", "click", "--name", "发送", "--exact")
                _wait_js(
                    session,
                    "document.querySelectorAll('#messages article.message-user').length === 1 && ("
                    "document.querySelectorAll('#messages article.message-user').item(0)"
                    ".nextElementSibling?.matches('article:not(.message-user)') || "
                    "document.querySelector('#turn-status')?.textContent.includes('处理失败') || "
                    "document.querySelector('#turn-status')?.textContent.includes('需要澄清'))",
                )
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
                dom_msg = json.loads(
                    _ab(
                        session,
                        "eval",
                        """(() => {
                            const m = document.querySelectorAll('#messages article.message-user')
                                .item(0)?.nextElementSibling;
                            if (!m?.matches('article:not(.message-user)')) {
                                return {
                                    missing: true,
                                    status: document.querySelector('#turn-status')?.textContent,
                                };
                            }
                            const btns = [...m.querySelectorAll('button[data-evidence-id]')];
                            return {
                                id: m.dataset.messageId,
                                text: m.innerText,
                                notices: [...m.querySelectorAll('.notice')].map(n => n.innerText),
                                citations: [...new Set(btns.map(b => b.dataset.evidenceId))],
                            };
                        })()""",
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
                                body_head: document.querySelector("#reader-body").innerText.slice(0, 260),
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
                    "message_meta": sess_payload["messages"][-1] if sess_payload.get("messages") else None,
                }

            upload_doc(QLORA_PDF)
            cases: dict[str, Any] = {}
            for q in questions:
                qid = q["question_id"]
                new_clean_session()
                focus_page = 15 if q.get("explicit_page_reference") else None
                select_doc(QLORA_PDF.name, page_no=focus_page)
                cases[qid] = _compact_case(ask_one(q["question"]))

            if include_g20_cross:
                upload_doc(DPO_PDF)
                upload_doc(MIXTRAL_PDF)
                new_clean_session()
                select_doc(DPO_PDF.name)
                cases["G20_Q05"] = _compact_case(ask_one(G20_Q05_TEXT))
                new_clean_session()
                select_course_scope()
                cases["G20_Q06"] = _compact_case(ask_one(G20_Q06_TEXT))

            provider_calls = []
            provider_log = out / "provider.jsonl"
            if provider_log.exists():
                for line in provider_log.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        provider_calls.append(json.loads(line))
            return {
                "arm": arm_name,
                "strategy": strategy,
                "user_agent": user_agent,
                "cases": cases,
                "provider_calls": provider_calls,
            }
        finally:
            try:
                _ab(session, "close")
            except Exception:
                pass
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


def _compact_case(raw: dict[str, Any]) -> dict[str, Any]:
    turn = raw.get("turn") or {}
    trace = turn.get("trace") or (turn.get("outcome") or {}).get("trace") or {}
    msg = raw.get("message_meta") or {}
    sources = msg.get("sources") or {}
    doc_names = {
        ref["document_id"]: ref["file_name"] for ref in msg.get("document_refs") or []
    }
    if not doc_names:
        for src in sources.values():
            fn = src.get("file_name")
            did = (src.get("revision") or {}).get("source", {}).get("document_id")
            if did and fn:
                doc_names[did] = fn
    candidates = [
        {
            "rank": idx,
            "file_name": c.get("file_name"),
            "source_id": (c.get("source") or {}).get("source_id"),
            "match_kind": c.get("match_kind"),
            "reasons": [r.get("code") for r in c.get("reasons") or []],
        }
        for idx, c in enumerate(trace.get("candidates") or [], 1)
    ]
    reading = trace.get("reading") or {}
    read_ranges = [
        {
            "file_name": doc_names.get((r.get("source") or {}).get("document_id"), ""),
            "source_id": (r.get("source") or {}).get("source_id"),
            "start": r.get("start"),
            "end": r.get("end"),
        }
        for r in reading.get("read_ranges") or []
    ]
    return {
        "elapsed_ms": raw.get("elapsed_ms"),
        "dom": raw.get("dom"),
        "search_status": trace.get("search_status"),
        "candidates": candidates,
        "retrieval": trace.get("retrieval"),
        "reading": {
            "full_text_scope": reading.get("full_text_scope"),
            "estimated_message_input": reading.get("estimated_message_input"),
            "preferred_source_ids": [p.get("source_id") for p in reading.get("preferred") or []],
            "read_ranges": read_ranges,
        },
        "generation": trace.get("generation"),
        "external_call_count_retrieval": trace.get("external_call_count", 0),
        "cited_sources": {
            eid: {
                "file_name": src.get("file_name"),
                "version": ((src.get("reads") or [{}])[0].get("actual_source") or {}).get("version"),
                "source_id": ((src.get("reads") or [{}])[0].get("actual_source") or {}).get("source_id"),
                "locator": src.get("locator"),
                "text_head": ((src.get("reads") or [{}])[0].get("text") or "")[:200],
            }
            for eid, src in sources.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-sha", default="eba76b49f7e8f2ba9402c2e1613b580ed7dbbd76")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs" / "G" / "G22-盲测浏览器实测.json",
    )
    args = parser.parse_args()

    criteria = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
    questions = criteria["questions"]

    with tempfile.TemporaryDirectory(prefix="srag-g22-base-") as base_tmp:
        base_dir = Path(base_tmp)
        subprocess.run(
            f"git -C {REPO_ROOT} archive {args.baseline_sha} | tar -x -C {base_dir}",
            shell=True,
            check=True,
        )
        baseline_arm = _run_browser_arm(
            arm_name="baseline_eba76b4",
            code_dir=base_dir,
            env_file=args.env_file,
            port=19521,
            strategy="fts",
            questions=questions,
            include_g20_cross=False,
        )

    bilingual_arm = _run_browser_arm(
        arm_name="g22_bilingual",
        code_dir=REPO_ROOT,
        env_file=args.env_file,
        port=19522,
        strategy="fts",
        questions=questions,
        include_g20_cross=True,
    )

    jev_arm = _run_browser_arm(
        arm_name="g22_bilingual_jev",
        code_dir=REPO_ROOT,
        env_file=args.env_file,
        port=19523,
        strategy="jev",
        questions=questions[:5],
        include_g20_cross=False,
    )

    report = {
        "task_id": "G22",
        "baseline_commit": args.baseline_sha,
        "qlora_sha256": hashlib.sha256(QLORA_PDF.read_bytes()).hexdigest(),
        "arms": {
            "baseline_eba76b4": baseline_arm,
            "g22_bilingual": bilingual_arm,
            "g22_bilingual_jev": jev_arm,
        },
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for arm_key, arm_data in report["arms"].items():
        print(f"=== {arm_key} (calls={len(arm_data['provider_calls'])}) ===")
        for qid, case in arm_data["cases"].items():
            read_p = [f"{r['file_name'][:8]}:{r['source_id']}" for r in case["reading"]["read_ranges"]]
            cites = {
                k: v.get("location") for k, v in (case["dom"].get("opened_citations") or {}).items()
            }
            print(
                f"  {qid}: ms={case['elapsed_ms']} status={case['search_status']} "
                f"read={read_p} cites={cites}"
            )


if __name__ == "__main__":
    main()
