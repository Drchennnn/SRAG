"""G21 browser verification: baseline (6b3e81a) vs G21 on Q01, blind Mamba PDF, and over-limit."""

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

Q01_TEXT = (
    "综合这份课件，用中文解释稀疏混合专家层（SMoE）是怎么根据门控网络计算输出的？"
    "每个词元激活几个专家、总参数量和激活参数量分别是多少？"
    "在多卡专家并行（Expert Parallelism）时会带来什么挑战？"
)

MAMBA_BLIND_QUESTION = (
    "综合这份课件，用中文解释传统结构化状态空间模型（S4 等）的线性时不变（LTI）性质"
    "为什么难以解决选择性复制（Selective Copying）和归纳头（Induction Heads）任务？"
    "Mamba（S6）在算法 2 中把哪些参数改成了输入 x 的函数、失去了哪种计算模式？"
    "Mamba 相比 Transformer 的推理吞吐量提升了多少倍，Mamba-3B 的表现如何？"
)


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


def _run_browser_arm(
    *,
    arm_name: str,
    code_dir: Path,
    env_file: Path,
    port: int,
    input_token_limit: int = 32_768,
    over_limit_only: bool = False,
) -> dict[str, Any]:
    python_bin = str(REPO_ROOT / ".venv" / "bin" / "python")
    course_id = "g21-budget-course"
    mixtral_pdf = REPO_ROOT / "docs" / "G" / "G20-fixtures" / "Mixtral-of-Experts-2401.04088v1.pdf"
    mamba_pdf = REPO_ROOT / "docs" / "G" / "G21-fixtures" / "Mamba-2312.00752v1.pdf"

    with tempfile.TemporaryDirectory(prefix=f"srag-g21-{arm_name}-") as tmp:
        out = Path(tmp)
        server_script = out / "serve_g21.py"
        server_script.write_text(
            f"""
import hashlib, json, httpx, uvicorn
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
    workspace_answer_enabled=True,
    workspace_retrieval_strategy="fts",
    source_vision_enabled=True,
    workspace_input_token_limit={input_token_limit},
)
app = create_app()
service = make_workspace(settings)
app.dependency_overrides[get_settings] = lambda: settings
app.dependency_overrides[get_workspace_service] = lambda: service

orig_async = httpx.AsyncHTTPTransport.handle_async_request
async def log_async(transport, request):
    started = perf_counter()
    payload = json.loads(request.content)
    user_prompt = json.loads(payload["messages"][1]["content"])
    rec = {{
        "url": str(request.url),
        "model": payload.get("model"),
        "max_tokens": payload.get("max_tokens"),
        "request_bytes": len(request.content),
        "question": user_prompt.get("question"),
        "mode": user_prompt.get("mode"),
        "coverage": user_prompt.get("coverage"),
        "evidence_ids": [e["id"] for e in user_prompt.get("evidence", [])],
        "evidence_sources": [
            {{
                "id": e["id"],
                "file_name": e["file_name"],
                "source": e["source"],
                "locator": e["locator"],
                "text_len": len(e["reads"][0]["text"]) if e.get("reads") else 0,
                "text_sha256": (
                    hashlib.sha256(e["reads"][0]["text"].encode()).hexdigest()
                    if e.get("reads")
                    else ""
                ),
            }}
            for e in user_prompt.get("evidence", [])
        ],
    }}
    try:
        resp = await orig_async(transport, request)
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

httpx.AsyncHTTPTransport.handle_async_request = log_async

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port={port}, log_level="warning")
""",
            encoding="utf-8",
        )

        env = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(code_dir),
        }
        proc = subprocess.Popen(
            [python_bin, str(server_script)],
            cwd=str(code_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        session = f"g21-{arm_name}-{port}-{int(time.time())}"
        try:
            url = f"http://127.0.0.1:{port}/workspace/"
            for _ in range(40):
                try:
                    r = subprocess.run(
                        ["curl", "--silent", "--fail", url],
                        capture_output=True,
                        timeout=2,
                        check=False,
                    )
                    if r.returncode == 0:
                        break
                except Exception:
                    pass
                time.sleep(0.25)
            else:
                raise RuntimeError(f"Server failed to start on {url}")

            _ab(session, "--allowed-domains", "127.0.0.1", "open", url)
            _ab(session, "set", "viewport", "1310", "790")
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

            def ask_in_browser(question_text: str, turn_num: int) -> dict[str, Any]:
                _ab(session, "find", "label", "输入问题", "fill", question_text)
                _wait_js(session, '!document.querySelector("#send").disabled')
                _ab(session, "find", "role", "button", "click", "--name", "发送", "--exact")
                idx = turn_num - 1
                _wait_js(
                    session,
                    f"document.querySelectorAll('#messages article.message-user').length"
                    f" === {turn_num} && ("
                    f"document.querySelectorAll('#messages article.message-user')"
                    f".item({idx}).nextElementSibling?.matches('article:not(.message-user)') || "
                    f"document.querySelector('#turn-status')?.textContent.includes('处理失败') || "
                    f"document.querySelector('#turn-status')?.textContent.includes('需要澄清'))",
                )
                dom_msg = json.loads(
                    _ab(
                        session,
                        "eval",
                        f"""(() => {{
                            const users = document.querySelectorAll(
                                '#messages article.message-user'
                            );
                            const user = users.item({idx});
                            const m = user?.nextElementSibling;
                            if (!m?.matches('article:not(.message-user)')) {{
                                return {{
                                    missing: true,
                                    notices: document.querySelector('#turn-status')?.textContent,
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
                opened_citations = {}
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
                        '(!document.querySelector("#reader img") || '
                        'document.querySelector("#reader img").complete) && '
                        '!document.querySelector("#reader-body").innerText.includes("正在加载原件")',
                    )
                    reader_state = json.loads(
                        _ab(
                            session,
                            "eval",
                            """(() => {
                                const r = document.querySelector("#reader");
                                const img = r.querySelector("img");
                                const body = document.querySelector("#reader-body").innerText;
                                const pic = img ? img.closest(".reader-picture") : null;
                                return {
                                    open: r.open,
                                    location: document.querySelector("#reader-location").innerText,
                                    title: document.querySelector("#reader-title").innerText,
                                    status: document.querySelector("#reader-status").innerText,
                                    body_head: body.slice(0, 240),
                                    body_len: body.length,
                                    image_displayed: Boolean(
                                        img && img.naturalWidth > 0 && !pic?.hidden
                                    ),
                                };
                            })()""",
                        )
                    )
                    opened_citations[cid] = reader_state
                    _ab(
                        session,
                        "find",
                        "role",
                        "button",
                        "click",
                        "--name",
                        "关闭原件",
                        "--exact",
                    )
                dom_msg["opened_citations"] = opened_citations
                return dom_msg

            results: dict[str, Any] = {
                "arm": arm_name,
                "input_token_limit": input_token_limit,
                "user_agent": user_agent,
                "cases": {},
            }

            upload_doc(mixtral_pdf)
            _ab(
                session,
                "find",
                "role",
                "button",
                "click",
                "--name",
                f"{mixtral_pdf.name} v1",
                "--exact",
            )

            if over_limit_only:
                dom_over = ask_in_browser(Q01_TEXT, 1)
                results["cases"]["Q01_over_limit"] = {"dom": dom_over}
            else:
                dom_q01 = ask_in_browser(Q01_TEXT, 1)
                results["cases"]["Q01"] = {"dom": dom_q01}

                upload_doc(mamba_pdf)
                _ab(
                    session,
                    "find",
                    "role",
                    "button",
                    "click",
                    "--name",
                    f"{mamba_pdf.name} v1",
                    "--exact",
                )
                dom_mamba = ask_in_browser(MAMBA_BLIND_QUESTION, 2)
                results["cases"]["Blind_Mamba"] = {"dom": dom_mamba}

            timeline = json.loads(
                _ab(
                    session,
                    "eval",
                    """(async () => {
                        const list = await (await fetch("/api/workspace/sessions")).json();
                        const sid = list.sessions[0].session_id;
                        return await (await fetch(`/api/workspace/sessions/${sid}`)).json();
                    })()""",
                )
            )
            results["session_id"] = timeline["session_id"]
            results["turns"] = timeline["turns"]
            results["messages_meta"] = [
                {
                    "message_id": m.get("message_id"),
                    "turn_id": m.get("turn_id"),
                    "role": m.get("role"),
                    "block_types": [
                        {"type": b["type"], "code": b.get("code"), "citations": b.get("citations")}
                        for b in m.get("blocks", [])
                    ],
                    "source_keys": {
                        k: {
                            "file_name": v["file_name"],
                            "source": v["revision"]["source"],
                            "locator": v["locator"],
                            "read_status": v["read_status"],
                            "text_len": len(v["reads"][0]["text"]) if v.get("reads") else 0,
                        }
                        for k, v in m.get("sources", {}).items()
                    },
                }
                for m in timeline.get("messages", [])
            ]
            provider_calls = []
            provider_log = out / "provider.jsonl"
            if provider_log.exists():
                for line in provider_log.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        provider_calls.append(json.loads(line))
            results["provider_calls"] = provider_calls
            return results
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


def _run_g20_plus_1600_probe_comparison() -> dict[str, Any]:
    """Compare G20 C1 fixed +1600 probe on Q01 with G21 exact budget accounting."""
    from app.rag.workspace_reading import estimate_input_budget

    old_reserve = 1800 + estimate_input_budget([Q01_TEXT, None])
    c1_fixed_reserve = old_reserve + 1600
    return {
        "baseline_old_reserve": old_reserve,
        "baseline_old_window_budget": 32_768 - old_reserve,
        "baseline_q01_window_estimated": 30_662,
        "baseline_q01_generation_estimated": 33_939,
        "g20_c1_fixed_plus_1600_reserve": c1_fixed_reserve,
        "g20_c1_fixed_window_budget": 32_768 - c1_fixed_reserve,
        "g20_c1_q01_generation_estimated": 32_227,
        "g20_c1_unused_budget_slack_bytes": 32_768 - 32_227,
        "why_fixed_1600_is_insufficient": (
            "固定 +1600 字节不随历史上下文（previous_context 最高 2600 字符）、"
            "多来源元数据（12+ 段 file_name/locator/reads 结构）"
            "以及原文中密集的引号、反斜线和换行符动态变化："
            "在短文本无转义时白白浪费约 541–1600 字节阅读窗口，"
            "而在长历史或密集引号/反斜线/多行代码时仍可能再次突破 32768 触发 budget_exceeded。"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-sha", default="6b3e81adf0eb949162dcb7e52ee3bf17ba3e4846")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    default_out = REPO_ROOT / "docs" / "G" / "G21-浏览器与盲验实测.json"
    parser.add_argument("--out", type=Path, default=default_out)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="srag-g21-baseline-archive-") as base_tmp:
        base_dir = Path(base_tmp)
        subprocess.run(
            f"git -C {REPO_ROOT} archive {args.baseline_sha} | tar -x -C {base_dir}",
            shell=True,
            check=True,
        )
        baseline_res = _run_browser_arm(
            arm_name="baseline_6b3e81a",
            code_dir=base_dir,
            env_file=args.env_file,
            port=19421,
            input_token_limit=32_768,
        )

    candidate_res = _run_browser_arm(
        arm_name="candidate_g21",
        code_dir=REPO_ROOT,
        env_file=args.env_file,
        port=19422,
        input_token_limit=32_768,
    )

    over_limit_res = _run_browser_arm(
        arm_name="candidate_g21_over_limit",
        code_dir=REPO_ROOT,
        env_file=args.env_file,
        port=19423,
        input_token_limit=500,
        over_limit_only=True,
    )

    c1_comparison = _run_g20_plus_1600_probe_comparison()
    mixtral_file = REPO_ROOT / "docs" / "G" / "G20-fixtures" / "Mixtral-of-Experts-2401.04088v1.pdf"
    mamba_file = REPO_ROOT / "docs" / "G" / "G21-fixtures" / "Mamba-2312.00752v1.pdf"

    report = {
        "task_id": "G21",
        "baseline_commit": args.baseline_sha,
        "g20_diagnosis_commit": "1de0cf37b06ce5a79e36135712cd6ca32e1c0f37",
        "fixtures": {
            "Mixtral-of-Experts-2401.04088v1.pdf": {
                "sha256": hashlib.sha256(mixtral_file.read_bytes()).hexdigest(),
                "license": "CC-BY-4.0 (arXiv:2401.04088v1)",
            },
            "Mamba-2312.00752v1.pdf": {
                "sha256": hashlib.sha256(mamba_file.read_bytes()).hexdigest(),
                "license": "CC-BY-4.0 (arXiv:2312.00752v1)",
            },
        },
        "g20_plus_1600_probe_comparison": c1_comparison,
        "baseline_browser_run": baseline_res,
        "candidate_browser_run": candidate_res,
        "candidate_over_limit_browser_run": over_limit_res,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote G21 browser verification report to {args.out}")


if __name__ == "__main__":
    main()
