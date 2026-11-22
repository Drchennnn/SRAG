"""G26 real Chromium /workspace/ verification on GaLore G_Q02 and frozen FOCCUS Q01-Q08."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/Users/Link/code/SRAG/.venv/bin/python"
ENV_FILE = Path("/Users/Link/code/SRAG/.env")
GALORE_PDF = REPO_ROOT / "docs/G/G23-fixtures/GaLore-2403.03507v2.pdf"
FOCCUS_PDF = REPO_ROOT / "docs/G/G26-fixtures/FOCCUS-D2.2-CCBY4.pdf"
CRITERIA_PATH = REPO_ROOT / "docs/G/G26-盲测冻结判据.json"
OUT_PATH = REPO_ROOT / "docs/G/G26-浏览器实测.json"


def _ab(session: str, *args: str, timeout: int = 120) -> str:
    res = subprocess.run(
        ["agent-browser", "--session", session, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"agent-browser failed ({' '.join(args)}): {res.stderr or res.stdout}")
    return res.stdout.strip()


def _wait_js(session: str, expr: str, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _ab(session, "eval", f"Boolean({expr})") == "true":
            return
        time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for: {expr}")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    port = _free_port()
    criteria = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
    galore_q02 = (
        "在文末附录补充的显存与吞吐量实测表中，在两百五十六个词元批大小下，"
        "未启用与启用逐层权重更新时，八位投影法与八位基准优化器各自占用的显存、"
        "每秒词元吞吐量以及每秒样本吞吐量分别是多少？"
    )

    with tempfile.TemporaryDirectory(prefix="srag-g26-browser-") as tmp:
        out = Path(tmp)
        srv = out / "serve_g26.py"
        srv.write_text(
            f"""
import uvicorn
from pathlib import Path
from app.api.workspace import get_workspace_service
from app.config import Settings, get_settings
from app.main import create_app
from app.rag.workspace import make_workspace

out = Path({str(out)!r})
settings = Settings(
    _env_file={str(ENV_FILE)!r},
    app_host="127.0.0.1",
    app_port={port},
    sqlite_path=out / "workspace.sqlite3",
    data_dir=out / "data",
    course_root=out / "courses",
    workspace_course_ids="g26-verify-course",
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
uvicorn.run(app, host="127.0.0.1", port={port}, log_level="warning")
""",
            encoding="utf-8",
        )

        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(REPO_ROOT)}
        proc = subprocess.Popen(
            [PYTHON, str(srv)],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        session = f"g26-verify-{port}-{int(time.time())}"
        try:
            url = f"http://127.0.0.1:{port}/workspace/"
            for _ in range(40):
                if (
                    subprocess.run(
                        ["curl", "--silent", "--fail", url], capture_output=True, timeout=2
                    ).returncode
                    == 0
                ):
                    break
                time.sleep(0.25)

            _ab(session, "--allowed-domains", "127.0.0.1", "open", url)
            _ab(session, "set", "viewport", "1310", "820")
            _ab(session, "wait", "--text", "暂无资料")

            def upload_doc(pdf_path: Path) -> None:
                if not json.loads(
                    _ab(session, "eval", 'document.querySelector("#import-panel").open')
                ):
                    _ab(session, "find", "text", "导入资料", "click", "--exact")
                _ab(session, "select", "#import-course", "g26-verify-course")
                _ab(session, "upload", "#import-files", str(pdf_path))
                _ab(session, "find", "role", "button", "click", "--name", "开始导入", "--exact")
                _ab(session, "wait", "--text", f"{pdf_path.name} · 导入完成 · v1")
                _wait_js(
                    session,
                    '!document.querySelector("#import-button").disabled && '
                    '!document.querySelector("#import-files").files.length',
                )

            def new_clean_session(file_name: str) -> None:
                prev_sid = _ab(session, "eval", "document.querySelector('#sessions')?.value || ''")
                _ab(session, "click", "#new-session")
                _wait_js(
                    session,
                    f"document.querySelector('#sessions')?.value && "
                    f"document.querySelector('#sessions')?.value !== {prev_sid}",
                )
                buttons = "[...document.querySelectorAll('#catalog button')]"
                match = f"b => b.textContent.includes({file_name!r})"
                _ab(
                    session,
                    "eval",
                    "(() => { "
                    "const cb = document.querySelector(\"#catalog button[data-kind='course']\"); "
                    'if (cb && cb.getAttribute("aria-expanded") !== "true") cb.click(); })()',
                )
                _wait_js(
                    session,
                    f"{buttons}.some({match})",
                )
                _ab(
                    session,
                    "eval",
                    f"{buttons}.find({match}).click()",
                )
                _wait_js(
                    session,
                    f"document.querySelector('#scope-summary')?.textContent.includes({file_name!r})",
                )

            def ask_turn(question: str, expected_user_count: int) -> dict[str, Any]:
                users = "document.querySelectorAll('#messages article.message-user')"
                t0 = time.perf_counter()
                _ab(session, "find", "label", "输入问题", "fill", question)
                _wait_js(session, '!document.querySelector("#send").disabled')
                _ab(session, "find", "role", "button", "click", "--name", "发送", "--exact")
                _wait_js(
                    session,
                    f"{users}.length === {expected_user_count} && ("
                    f"{users}.item({expected_user_count - 1})"
                    ".nextElementSibling?.matches('article:not(.message-user)') || "
                    "document.querySelector('#turn-status')?.textContent.includes('处理失败') || "
                    "document.querySelector('#turn-status')?.textContent.includes('需要澄清'))",
                )
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
                msg_info = json.loads(
                    _ab(
                        session,
                        "eval",
                        f"""(() => {{
                            const m = document.querySelectorAll('#messages article.message-user')
                                .item({expected_user_count - 1})?.nextElementSibling;
                            if (!m?.matches('article:not(.message-user)')) {{
                                const status = document.querySelector('#turn-status')?.textContent;
                                return {{missing: true, status}};
                            }}
                            const btns = [...m.querySelectorAll('button[data-evidence-id]')];
                            return {{
                                id: m.dataset.messageId,
                                text: m.innerText,
                                citations: btns.map(b => b.dataset.evidenceId),
                            }};
                        }})()""",
                    )
                )
                opened: dict[str, Any] = {}
                for ev_id in msg_info.get("citations") or []:
                    _ab(
                        session,
                        "eval",
                        f'document.querySelector(\'article[data-message-id="{msg_info["id"]}"] '
                        f'button[data-evidence-id="{ev_id}"]\').click()',
                    )
                    _wait_js(session, "!document.querySelector('#reader')?.hidden")
                    opened[ev_id] = json.loads(
                        _ab(
                            session,
                            "eval",
                            "({title: document.querySelector('#reader-title')?.textContent, "
                            "location: document.querySelector('#reader-location')?.textContent})",
                        )
                    )
                tl = json.loads(
                    _ab(
                        session,
                        "eval",
                        "(async () => { const sid = document.querySelector('#sessions').value; "
                        "return await (await fetch(`/api/workspace/sessions/${sid}`)).json(); })()",
                    )
                )
                last_turn = tl["turns"][-1]
                last_msg = tl["messages"][-1]
                return {
                    "question": question,
                    "elapsed_ms": elapsed_ms,
                    "turn_status": last_turn.get("status"),
                    "trace": last_turn.get("trace"),
                    "blocks": last_msg.get("blocks"),
                    "sources": last_msg.get("sources"),
                    "opened_citations": opened,
                    "dom_text": msg_info.get("text"),
                }

            upload_doc(GALORE_PDF)
            upload_doc(FOCCUS_PDF)

            results: dict[str, Any] = {"cases": {}}
            new_clean_session(GALORE_PDF.name)
            results["cases"]["GaLore_G_Q02"] = ask_turn(galore_q02, 1)

            for q in criteria["questions"]:
                qid = q["question_id"]
                new_clean_session(FOCCUS_PDF.name)
                turns = q.get("turns") or [q["question"]]
                if len(turns) == 1:
                    results["cases"][qid] = ask_turn(turns[0], 1)
                else:
                    t1 = ask_turn(turns[0], 1)
                    t2 = ask_turn(turns[1], 2)
                    results["cases"][qid] = {"turn_1": t1, "turn_2": t2}

            OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            for k, v in results["cases"].items():
                item = v["turn_2"] if "turn_2" in v else v
                tr = item.get("trace") or {}
                notices = [b.get("code") for b in (item.get("blocks") or []) if b.get("code")]
                opened = {
                    ek: ev.get("title") for ek, ev in (item.get("opened_citations") or {}).items()
                }
                print(
                    f"{k}: status={item.get('turn_status')} route={tr.get('route')} "
                    f"sel={(tr.get('selected_source') or {}).get('source_id')} "
                    f"notices={notices} opened={opened} ms={item.get('elapsed_ms')}"
                )
        finally:
            subprocess.run(["agent-browser", "--session", session, "close"], capture_output=True)
            proc.terminate()
            proc.wait(timeout=5)


if __name__ == "__main__":
    main()
