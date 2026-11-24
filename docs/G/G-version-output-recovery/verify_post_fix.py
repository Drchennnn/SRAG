"""Post-fix real-provider (G01 config) and real-Chromium (/workspace/) verification."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_fixtures import generate_all  # noqa: E402
from repro_baseline import BASE, COURSE, DEFAULT_ENV, HOST, _snap  # noqa: E402

from app.api.workspace import get_workspace_service, router  # noqa: E402
from app.config import Settings  # noqa: E402
from app.rag.workspace import make_workspace  # noqa: E402


def _distinguish_displayed_vs_read(snap: dict[str, Any]) -> dict[str, Any]:
    displayed = [
        d["display_source"] for d in snap.get("display_order", []) if d.get("display_source")
    ]
    read_sources = [
        {
            "source": ev["source"],
            "method": r["method"],
            "status": r["status"],
        }
        for ev in snap.get("actual_reads", [])
        for r in ev.get("reads", [])
        if r.get("status") in {"succeeded", "partial"}
    ]
    return {
        "displayed_objects": displayed,
        "actually_read_objects": read_sources,
    }


def run_api_post_fix(work_dir: Path, env_file: Path = DEFAULT_ENV) -> dict[str, Any]:
    fixtures = generate_all(work_dir / "fixtures")
    settings = Settings(
        sqlite_path=work_dir / "library.sqlite3",
        data_dir=work_dir / "data",
        course_root=work_dir / "originals",
        workspace_course_ids=COURSE,
        workspace_retrieval_strategy="fts",
        workspace_input_token_limit=32768,
        workspace_answer_enabled=False,
        source_vision_enabled=True,
        deepseek_api_key="",
        qdrant_api_key="",
        qdrant_url="http://127.0.0.1:1",
        _env_file=env_file,
    )
    service = make_workspace(settings)

    vision_calls: list[dict[str, Any]] = []
    orig_process = service.vision.process_current

    def observed_process(request, *, expected_input_sha256=None):
        res = orig_process(request, expected_input_sha256=expected_input_sha256)
        vision_calls.append(
            {
                "source": request.source.model_dump(),
                "question": request.question,
                "status": res.status,
                "external_call_count": res.external_call_count,
                "read_text": res.reads[0].text if res.reads else None,
            }
        )
        return res

    service.vision.process_current = observed_process  # type: ignore[method-assign]
    app = FastAPI()
    app.include_router(router, prefix=BASE)
    app.dependency_overrides[get_workspace_service] = lambda: service

    results: dict[str, Any] = {
        "git_sha": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "config": {
            "workspace_retrieval_strategy": settings.workspace_retrieval_strategy,
            "workspace_input_token_limit": settings.workspace_input_token_limit,
            "workspace_answer_enabled": settings.workspace_answer_enabled,
            "source_vision_enabled": settings.source_vision_enabled,
        },
        "scenarios": {},
        "vision_calls": vision_calls,
    }

    with TestClient(app, base_url=HOST, headers={"Origin": HOST}) as client:

        def new_session() -> str:
            r = client.post(f"{BASE}/sessions", json={"course_ids": [COURSE], "document_ids": []})
            assert r.status_code == 201, r.text
            return r.json()["session_id"]

        def upload(
            sid: str, path: Path, doc_id: str | None = None, name: str = "Bench-Drift.ipynb"
        ) -> dict[str, Any]:
            params = {"course_id": COURSE, "filename": name}
            if doc_id:
                params["document_id"] = doc_id
            r = client.post(
                f"{BASE}/sessions/{sid}/import",
                params=params,
                headers={"content-type": "application/octet-stream", "origin": HOST},
                content=path.read_bytes(),
            )
            assert r.status_code == 200, r.text
            return r.json()

        def ask(sid: str, q: str, rev: int, cid: str, anchor: dict | None = None) -> dict[str, Any]:
            body: dict[str, Any] = {
                "kind": "ask", "text": q, "scope_revision": rev, "client_turn_id": cid
            }
            if anchor is not None:
                body["anchor"] = anchor
            r = client.post(f"{BASE}/sessions/{sid}/turns", json=body)
            assert r.status_code == 200, r.text
            return r.json()

        def focus_cell(sid: str, doc_id: str, cell_index: int, rev: int) -> tuple[str, int]:
            cat = client.get(f"{BASE}/sessions/{sid}/catalog").json()
            stack = list(cat["courses"][0]["documents"])
            node_id = None
            while stack:
                item = stack.pop(0)
                if (
                    item["document_id"] == doc_id
                    and item["kind"] == "cell"
                    and item["locator"].get("cell_index") == cell_index
                ):
                    node_id = item["node_id"]
                    break
                stack.extend(item.get("children", []))
            assert node_id is not None
            r = client.put(
                f"{BASE}/sessions/{sid}/focus", json={"node_id": node_id, "scope_revision": rev}
            )
            assert r.status_code == 200, r.text
            return node_id, r.json()["scope_revision"]

        # Problem 2: Multi-output cell -> clarify -> choose -> read
        sid_p2 = new_session()
        v1 = upload(sid_p2, fixtures["v1"], name="Bench-P2.ipynb")
        cell_nid, rev = focus_cell(sid_p2, v1["document_id"], 1, 1)
        p2_a = ask(sid_p2, "当前这个 Cell 的输出图像横轴是什么？", rev, "p2-img-q")
        p2_a_pick = ask(
            sid_p2,
            p2_a["options"][1]["label"],
            rev,
            "p2-img-pick",
            anchor={
                "type": "clarification",
                "turn_id": p2_a["turn"]["turn_id"],
                "option_id": p2_a["options"][1]["option_id"],
            },
        )
        p2_b = ask(sid_p2, "当前这个 Cell 的输出结果显示什么？", rev, "p2-txt-q")
        p2_b_pick = ask(
            sid_p2,
            p2_b["options"][2]["label"],
            rev,
            "p2-txt-pick",
            anchor={
                "type": "clarification",
                "turn_id": p2_b["turn"]["turn_id"],
                "option_id": p2_b["options"][2]["option_id"],
            },
        )
        s_a = _snap("p2_img_q", p2_a, cell_nid)
        s_ap = _snap("p2_img_pick", p2_a_pick, cell_nid)
        s_b = _snap("p2_out_q", p2_b, cell_nid)
        s_bp = _snap("p2_out_pick", p2_b_pick, cell_nid)
        results["scenarios"]["problem2_multi_output_focus"] = {
            "img_question": {**s_a, **_distinguish_displayed_vs_read(s_a)},
            "img_selected": {**s_ap, **_distinguish_displayed_vs_read(s_ap)},
            "out_question": {**s_b, **_distinguish_displayed_vs_read(s_b)},
            "out_selected": {**s_bp, **_distinguish_displayed_vs_read(s_bp)},
        }

        # Problem 1: 3 intents + controls WITHOUT PUT /focus
        for intent_name, q_text in (
            ("read_new_counterpart", "最新版里刚才读过的那张图现在的峰值是多少？"),
            ("recall_old_read", "回看刚才读过的旧版那张图的内容。"),
            ("compare_both_versions", "和上一版相比，刚才读过的那张图有什么变化？"),
            ("only_old_control", "只看旧版第一张图，刚才展示的第 1 张图是什么？"),
            ("ordinary_stale_focus", "看当前 Cell"),
        ):
            sid = new_session()
            doc_v1 = upload(sid, fixtures["v1"], name=f"Bench-P1-{intent_name}.ipynb")
            nid_v1, r_v1 = focus_cell(sid, doc_v1["document_id"], 1, 1)
            t1_show = ask(sid, "展示这个 Cell 的两张图", r_v1, f"{intent_name}-show")
            t1_read = ask(sid, "刚才第一张图是什么？", r_v1, f"{intent_name}-read")
            doc_v2 = upload(
                sid, fixtures["v2_same"], doc_id=doc_v1["document_id"],
                name=f"Bench-P1-{intent_name}.ipynb",
            )
            assert doc_v2["version"] == 2
            t2 = ask(sid, q_text, r_v1, f"{intent_name}-after-v2")
            sn1, sn2, sn3 = (
                _snap("turn1_show", t1_show, nid_v1),
                _snap("turn1_read", t1_read, nid_v1),
                _snap(intent_name, t2, nid_v1),
            )
            results["scenarios"][f"p1_{intent_name}"] = {
                "turn1_show": {**sn1, **_distinguish_displayed_vs_read(sn1)},
                "turn1_read": {**sn2, **_distinguish_displayed_vs_read(sn2)},
                "turn2_after_v2": {**sn3, **_distinguish_displayed_vs_read(sn3)},
            }

        # Reordered outputs v2 & deleted outputs v2 + multi-display-no-read
        for variant_name, v2_key in (
            ("reordered_v2", "v2_reordered"),
            ("deleted_v2", "v2_deleted"),
        ):
            sid = new_session()
            d1 = upload(sid, fixtures["v1"], name=f"Bench-{variant_name}.ipynb")
            nid, r1 = focus_cell(sid, d1["document_id"], 1, 1)
            ask(sid, "展示这个 Cell 的两张图", r1, f"{variant_name}-show")
            ask(sid, "刚才第一张图是什么？", r1, f"{variant_name}-read")
            upload(
                sid,
                fixtures[v2_key],
                doc_id=d1["document_id"],
                name=f"Bench-{variant_name}.ipynb",
            )
            t2 = ask(
                sid,
                "最新版里刚才读过的那张图现在的峰值是多少？",
                r1,
                f"{variant_name}-after",
            )
            sn = _snap(variant_name, t2, nid)
            entry: dict[str, Any] = {"clarification": {**sn, **_distinguish_displayed_vs_read(sn)}}
            if t2.get("options"):
                picked = ask(
                    sid,
                    t2["options"][1]["label"],
                    r1,
                    f"{variant_name}-pick",
                    anchor={
                        "type": "clarification",
                        "turn_id": t2["turn"]["turn_id"],
                        "option_id": t2["options"][1]["option_id"],
                    },
                )
                snp = _snap(f"{variant_name}_picked", picked, nid)
                entry["picked"] = {**snp, **_distinguish_displayed_vs_read(snp)}
            results["scenarios"][f"p1_{variant_name}"] = entry

        sid_combo = new_session()
        doc_c1 = upload(sid_combo, fixtures["v1"], name="Bench-P1-combo.ipynb")
        nid_c1, rev_c1 = focus_cell(sid_combo, doc_c1["document_id"], 1, 1)
        t_show = ask(sid_combo, "展示这个 Cell 的两张图", rev_c1, "combo-show")
        upload(
            sid_combo,
            fixtures["v2_same"],
            doc_id=doc_c1["document_id"],
            name="Bench-P1-combo.ipynb",
        )
        t_cmp = ask(sid_combo, "和上一版相比，刚才那张图有什么变化？", rev_c1, "combo-cmp")
        results["scenarios"]["p1_multi_display_no_read_compare"] = {
            "turn1_show": _snap("combo_show", t_show, nid_c1),
            "turn2_cmp": _snap("combo_cmp", t_cmp, nid_c1),
        }

    return results


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


def run_browser_verify(*, port: int, env_file: Path = DEFAULT_ENV) -> dict[str, Any]:
    python_bin = "/Users/Link/code/SRAG/.venv/bin/python"
    with tempfile.TemporaryDirectory(prefix="g-ver-browser-") as tmp:
        out = Path(tmp)
        fixtures = generate_all(out / "fixtures")
        server_script = out / "serve_browser.py"
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
    workspace_course_ids={COURSE!r},
    workspace_max_upload_bytes=16 * 1024 * 1024,
    workspace_answer_enabled=False,
    source_vision_enabled=True,
    workspace_retrieval_strategy="fts",
    workspace_input_token_limit=32_768,
    deepseek_api_key="",
    qdrant_api_key="",
    qdrant_url="http://127.0.0.1:1",
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
        session = f"g-ver-{port}-{int(time.time())}"
        try:
            url = f"http://127.0.0.1:{port}/workspace/"
            for _ in range(40):
                r = subprocess.run(
                    ["curl", "--silent", "--fail", url], capture_output=True, timeout=2
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

            def upload_nb(nb_path: Path, filename: str, *, replace: bool = False) -> None:
                is_open = json.loads(
                    _ab(session, "eval", 'document.querySelector("#import-panel").open')
                )
                if not is_open:
                    _ab(session, "find", "text", "导入资料", "click", "--exact")
                _ab(session, "select", "#import-course", COURSE)
                if replace:
                    _ab(
                        session,
                        "eval",
                        "document.querySelector('#replace-document').checked = true",
                    )
                else:
                    _ab(
                        session,
                        "eval",
                        "document.querySelector('#replace-document').checked = false",
                    )
                target = out / "fixtures" / filename
                target.write_bytes(nb_path.read_bytes())
                _ab(session, "upload", "#import-files", str(target))
                _ab(session, "find", "role", "button", "click", "--name", "开始导入", "--exact")
                expected_ver = "v2" if replace else "v1"
                _ab(session, "wait", "--text", f"{filename} · 导入完成 · {expected_ver}")
                _wait_js(
                    session,
                    '!document.querySelector("#import-button").disabled && '
                    '!document.querySelector("#import-files").files.length',
                )

            def new_clean_session() -> str:
                prev_sid = _ab(session, "eval", "document.querySelector('#sessions')?.value || ''")
                _ab(session, "click", "#new-session")
                _wait_js(
                    session,
                    f"document.querySelector('#sessions')?.value && "
                    f"document.querySelector('#sessions')?.value !== {prev_sid} && "
                    f"document.querySelector('#catalog')?.getAttribute('aria-busy') === 'false'",
                )
                return json.loads(_ab(session, "eval", "document.querySelector('#sessions').value"))

            def focus_doc_and_cell(file_name: str, cell_num: int | None = None) -> None:
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
                        const wantOpen = {json.dumps(cell_num is not None)};
                        const isOpen = db.getAttribute("aria-expanded") === "true";
                        const summaryTxt = document.querySelector('#scope-summary')?.textContent;
                        const scopeHas = (summaryTxt || "").includes({file_name!r});
                        if (!scopeHas || (wantOpen && !isOpen)) db.click();
                    }})()""",
                )
                _wait_js(
                    session,
                    f"document.querySelector('#scope-summary')?.textContent.includes({file_name!r})",
                )
                if cell_num is not None:
                    # Expand intermediate section nodes inside the document tree if present.
                    target_cell = f"Cell {cell_num}"
                    for _ in range(3):
                        _ab(
                            session,
                            "eval",
                            f"""(() => {{
                                const db = [...document.querySelectorAll('#catalog button')]
                                    .find(b => b.textContent.includes({file_name!r}));
                                const li = db?.closest('li');
                                if (!li) return;
                                const sel = "button[aria-expanded='false']";
                                for (const childBtn of li.querySelectorAll(sel)) {{
                                    if (!childBtn.textContent.includes({target_cell!r})) {{
                                        childBtn.click();
                                    }}
                                }}
                            }})()""",
                        )
                    label = f"Cell {cell_num}"
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

            def current_session_state() -> dict[str, Any]:
                return json.loads(
                    _ab(
                        session,
                        "eval",
                        """(async () => {
                            const sid = document.querySelector('#sessions').value;
                            return await (await fetch(`/api/workspace/sessions/${sid}`)).json();
                        })()""",
                    )
                )

            def ask_ui(question_text: str) -> dict[str, Any]:
                prev_u = json.loads(
                    _ab(
                        session,
                        "eval",
                        "document.querySelectorAll('#messages article.message-user').length",
                    )
                )
                _ab(session, "find", "label", "输入问题", "fill", question_text)
                _wait_js(session, '!document.querySelector("#send").disabled')
                _ab(session, "find", "role", "button", "click", "--name", "发送", "--exact")
                _wait_js(
                    session,
                    f"(() => {{ const s = '#messages article.message-user';"
                    f"const u = document.querySelectorAll(s);"
                    f"if (u.length < {prev_u + 1}) return false;"
                    f"const last = u.item({prev_u});"
                    f"const busy = document.querySelector('#composer-status')"
                    f"?.textContent.includes('正在处理');"
                    f"if (busy) return false;"
                    f"const sib = last?.nextElementSibling;"
                    f"return Boolean(sib?.matches('article:not(.message-user)') "
                    f"|| document.querySelector('#turn-status')?.textContent.includes('需要澄清') "
                    f"|| document.querySelector('#turn-status')?.textContent.includes('处理失败'));"
                    f" }})()",
                )
                return _inspect_latest_turn()

            def _inspect_latest_turn() -> dict[str, Any]:
                state = current_session_state()
                last_turn = state["turns"][-1]
                last_msg = state["messages"][-1] if state.get("messages") else None
                opts = json.loads(
                    _ab(
                        session,
                        "eval",
                        "[...document.querySelectorAll('.clarification-actions button')]"
                        ".map(b => b.textContent)",
                    )
                )
                opened: dict[str, Any] = {}
                if last_msg:
                    mid = last_msg["message_id"]
                    cids = json.loads(
                        _ab(
                            session,
                            "eval",
                            f"[...document.querySelectorAll(\"article[data-message-id='{mid}'] "
                            f"button[data-evidence-id]\")].map(b => b.dataset.evidenceId)",
                        )
                    )
                    for cid in dict.fromkeys(cids):
                        sel = f"article[data-message-id='{mid}'] button[data-evidence-id='{cid}']"
                        _ab(session, "click", sel)
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
                                    img_src: document.querySelector("#reader-body img")
                                        ?.getAttribute("src") || null,
                                }))()""",
                            )
                        )
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
                return {
                    "turn": last_turn,
                    "message": last_msg,
                    "clarification_buttons": opts,
                    "opened_citations": opened,
                }

            # --- 1. Browser Problem 2 (focused multi-output Cell -> clarify -> click -> cite)
            upload_nb(fixtures["v1"], "Browser-P2.ipynb")
            focus_doc_and_cell("Browser-P2.ipynb", cell_num=2)
            p2_clarify = ask_ui("当前这个 Cell 的输出图像横轴是什么？")
            assert p2_clarify["turn"]["status"] == "clarification", p2_clarify
            assert len(p2_clarify["clarification_buttons"]) == 2, p2_clarify
            prev_u_p2 = json.loads(
                _ab(
                    session,
                    "eval",
                    "document.querySelectorAll('#messages article.message-user').length",
                )
            )
            _ab(
                session,
                "eval",
                "document.querySelectorAll('.clarification-actions button')[1].click()",
            )
            _wait_js(session, '!document.querySelector("#send").disabled')
            _ab(session, "find", "role", "button", "click", "--name", "发送", "--exact")
            _wait_js(
                session,
                f"(() => {{ const u = document.querySelectorAll('#messages article.message-user');"
                f"if (u.length < {prev_u_p2 + 1}) return false;"
                f"const busy = document.querySelector('#composer-status')"
                f"?.textContent.includes('正在处理');"
                f"if (busy) return false;"
                f"return Boolean(u.item({prev_u_p2})"
                f"?.nextElementSibling?.matches('article:not(.message-user)')); }})()",
            )
            p2_chosen = _inspect_latest_turn()
            assert p2_chosen["turn"]["status"] == "completed", p2_chosen
            p2_ev = next(iter(p2_chosen["message"]["sources"].values()))
            assert p2_ev["revision"]["source"]["source_id"] == "cell:1:output:1", p2_ev
            assert p2_chosen["opened_citations"], p2_chosen

            # --- 2. Browser Problem 1 (NO PUT /focus after v2 publish!) ---
            sid_p1 = new_clean_session()
            upload_nb(fixtures["v1"], "Browser-P1.ipynb")
            focus_doc_and_cell("Browser-P1.ipynb", cell_num=2)
            p1_show = ask_ui("展示这个 Cell 的两张图")
            assert p1_show["turn"]["status"] == "completed", p1_show
            assert len(p1_show["message"]["visible_display_ids"]) == 2, p1_show
            p1_read = ask_ui("刚才第一张图是什么？")
            assert p1_read["turn"]["status"] == "completed", p1_read
            assert (
                p1_read["turn"]["resolution"]["actual_source"]["source_id"] == "cell:1:output:0"
            ), p1_read
            doc_id_p1 = current_session_state()["scope"]["document_ids"][0]
            # Publish v2 via formal /import inside browser cookie WITHOUT calling PUT /focus
            v2_b64 = base64.b64encode(fixtures["v2_same"].read_bytes()).decode("ascii")
            pub_res = json.loads(
                _ab(
                    session,
                    "eval",
                    f"""(async () => {{
                        const raw = Uint8Array.from(atob({v2_b64!r}), c => c.charCodeAt(0));
                        const q = new URLSearchParams({{
                            course_id: {COURSE!r},
                            filename: "Browser-P1.ipynb",
                            document_id: {doc_id_p1!r}
                        }});
                        const r = await fetch(`/api/workspace/sessions/{sid_p1}/import?${{q}}`, {{
                            method: "POST",
                            headers: {{"content-type": "application/octet-stream"}},
                            body: raw,
                        }});
                        return {{ok: r.ok, status: r.status, body: await r.json()}};
                    }})()""",
                )
            )
            assert pub_res["ok"] and pub_res["body"]["version"] == 2, pub_res
            p1_new_v2 = ask_ui("最新版里刚才读过的那张图现在的峰值是多少？")
            assert p1_new_v2["turn"]["status"] == "completed", p1_new_v2
            p1_ev = next(iter(p1_new_v2["message"]["sources"].values()))
            assert p1_ev["revision"]["source"]["version"] == 2, p1_ev
            assert p1_ev["revision"]["source"]["source_id"] == "cell:1:output:0", p1_ev
            assert p1_new_v2["opened_citations"], p1_new_v2

            # --- 3. Separate compatibility regression: natural web auto-rebind after UI replace
            new_clean_session()
            upload_nb(fixtures["v1"], "Browser-AutoRebind.ipynb")
            focus_doc_and_cell("Browser-AutoRebind.ipynb", cell_num=None)
            upload_nb(fixtures["v2_same"], "Browser-AutoRebind.ipynb", replace=True)
            rebind_state = current_session_state()
            assert rebind_state["focus"]["version"] == 2, rebind_state
            rebind_ask = ask_ui("第 2 个 Cell 的输出 summary 写了什么？")
            assert rebind_ask["turn"]["status"] == "completed", rebind_ask

            provider_calls = []
            pfile = out / "provider.jsonl"
            if pfile.exists():
                provider_calls = [
                    json.loads(line) for line in pfile.read_text().splitlines() if line
                ]

            return {
                "git_sha": subprocess.check_output(
                    ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
                ).strip(),
                "user_agent": user_agent,
                "problem2": {"clarify": p2_clarify, "after_select": p2_chosen},
                "problem1_no_put_focus": {
                    "turn1_show": p1_show,
                    "turn1_read": p1_read,
                    "turn2_new_counterpart_v2": p1_new_v2,
                },
                "natural_web_auto_rebind": {
                    "scope_after_replace": rebind_state["scope"],
                    "focus_after_replace": rebind_state.get("focus"),
                    "followup_ask": rebind_ask,
                },
                "provider_calls": provider_calls,
            }
        finally:
            try:
                _ab(session, "close")
            except Exception:
                pass
            proc.terminate()
            proc.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--port", type=int, default=18923)
    parser.add_argument("--browser-only", action="store_true")
    args = parser.parse_args()

    if not args.browser_only:
        with tempfile.TemporaryDirectory(prefix="g-ver-post-") as tmp:
            api_res = run_api_post_fix(Path(tmp))
        (args.out_dir / "post_fix_traces.json").write_text(
            json.dumps(api_res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote {args.out_dir / 'post_fix_traces.json'}")

    browser_res = run_browser_verify(port=args.port)
    (args.out_dir / "browser_verify.json").write_text(
        json.dumps(browser_res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out_dir / 'browser_verify.json'}")


if __name__ == "__main__":
    main()
