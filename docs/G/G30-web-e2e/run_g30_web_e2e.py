#!/usr/bin/env python3
"""G30 Track 2: Independent Real-Chromium (/workspace/) 8-Step Student Workflow Gate.

Exports clean commit 33f0f14a0333a715c0e8e3a63ac0bb8110aa35f0 from /Users/Link/code/SRAG-g29
into a temporary /tmp/srag-g30-web-subject-* directory, verifies app.rag.workspace loads
strictly from that directory, launches uvicorn with real DeepSeek (deepseek-flash), real
Qwen3-VL (qwen3-vl-plus), real LibreOffice (/opt/homebrew/bin/soffice), Qdrant disabled
(http://127.0.0.1:1), and FTS retrieval, then drives Chromium via agent-browser through
the 8-step three-format student workflow.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SOURCE_REPO = Path("/Users/Link/code/SRAG-g29")
SUBJECT_SHA = "33f0f14a0333a715c0e8e3a63ac0bb8110aa35f0"
FIXTURES = Path("/Users/Link/code/SRAG-g29-gate/docs/G/G29-gate/fixtures")
ENV_FILE = Path("/Users/Link/code/SRAG/.env")
PYTHON = "/Users/Link/code/SRAG/.venv/bin/python"
AGENT_BROWSER = "/Users/Link/.local/bin/agent-browser"
SOFFICE_PATH = "/opt/homebrew/bin/soffice"
OUT_DIR = Path(__file__).resolve().parent
OUT_JSON = OUT_DIR / "G30-web-e2e-results.json"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _load_env_readonly() -> tuple[dict[str, str], list[str]]:
    """Read /Users/Link/code/SRAG/.env into memory only; track secret values for scrubbing."""
    env = dict(os.environ)
    secrets: list[str] = []
    if ENV_FILE.exists():
        for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            env[k] = v
            if any(tok in k.upper() for tok in ("KEY", "TOKEN", "SECRET", "PASSWORD")) and len(v) >= 6:
                secrets.append(v)
    return env, secrets


def _export_clean_subject() -> Path:
    subject_dir = Path(tempfile.mkdtemp(prefix="srag-g30-web-subject-", dir="/tmp"))
    proc = subprocess.run(
        ["git", "-C", str(SOURCE_REPO), "archive", SUBJECT_SHA],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["tar", "-x", "-C", str(subject_dir)],
        input=proc.stdout,
        capture_output=True,
        check=True,
    )
    # Verify app.rag.workspace resolves strictly inside subject_dir
    probe = subprocess.run(
        [
            PYTHON,
            "-c",
            (
                "import sys, pathlib; "
                f"sys.path.insert(0, {str(subject_dir)!r}); "
                "import app.rag.workspace as w; "
                "print(pathlib.Path(w.__file__).resolve())"
            ),
        ],
        cwd=str(subject_dir),
        capture_output=True,
        text=True,
        check=True,
    )
    loaded_path = Path(probe.stdout.strip()).resolve()
    if not loaded_path.is_relative_to(subject_dir.resolve()):
        raise RuntimeError(f"app.rag.workspace loaded from unexpected location: {loaded_path}")
    return subject_dir


def _wait_port(port: int, timeout: float = 25.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.25)
    raise RuntimeError(f"Uvicorn did not bind port {port} within {timeout}s")


def _ab(session: str, *args: str, timeout: int = 90) -> str:
    cmd = [AGENT_BROWSER, "--session", session, *args]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(
            f"agent-browser failed ({res.returncode}): {' '.join(args)}\nSTDOUT:{res.stdout}\nSTDERR:{res.stderr}"
        )
    return res.stdout.strip()


def _eval(session: str, js: str) -> Any:
    raw = _ab(session, "eval", "--json", js)
    parsed = json.loads(raw)
    if isinstance(parsed, dict) and "data" in parsed:
        data = parsed["data"]
        return data.get("result") if isinstance(data, dict) and "result" in data else data
    return parsed


def _eval_async(session: str, body_js: str, timeout: float = 30.0) -> Any:
    _eval(
        session,
        "window.__g30_done = false; window.__g30_res = null; window.__g30_err = null; "
        f"(async () => {{ try {{ window.__g30_res = await (async () => {{ {body_js} }})(); }} "
        "catch (e) { window.__g30_err = String(e); } finally { window.__g30_done = true; } })()",
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _eval(session, "Boolean(window.__g30_done)"):
            err = _eval(session, "window.__g30_err")
            if err:
                raise RuntimeError(f"Async JS error: {err}")
            return _eval(session, "window.__g30_res")
        time.sleep(0.15)
    raise RuntimeError("Async JS timed out")


def _wait_ready(session: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready = _eval(
            session,
            "Boolean(document.querySelector('#import-button') && !document.querySelector('#import-button').disabled && !document.querySelector('#question').disabled)",
        )
        if ready:
            return
        time.sleep(0.25)
    raise RuntimeError("Workspace UI did not initialize")


def _wait_nav_idle(session: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        idle = _eval(
            session,
            "!document.querySelector('#clear-file')?.disabled && !document.querySelector('#question')?.disabled",
        )
        if idle:
            return
        time.sleep(0.2)


def _msg_count(session: str) -> int:
    return int(_eval(session, "document.querySelectorAll('article.message').length") or 0)


def _ask_ui(
    session: str,
    question: str,
    *,
    expect_clarification: bool = False,
    timeout: float = 120.0,
) -> dict[str, Any]:
    prev_msgs = _msg_count(session)
    t0 = time.monotonic()
    _ab(session, "fill", "#question", question)
    _ab(session, "click", "#send")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = _eval(
            session,
            """
            (() => {
              const msgs = document.querySelectorAll('article.message').length;
              const clarBtns = Array.from(document.querySelectorAll('.clarification-actions button')).map(b => b.textContent);
              const statusTxt = document.querySelector('#turn-status')?.textContent || '';
              const composerTxt = document.querySelector('#composer-status')?.textContent || '';
              const actionsTxt = document.querySelector('#turn-actions')?.textContent || '';
              const wsErr = document.querySelector('#workspace-status')?.textContent || '';
              const running = composerTxt.includes('正在处理') || actionsTxt.includes('取消本轮') || statusTxt.includes('正在处理');
              return {
                running,
                msgs,
                clarBtns,
                statusTxt,
                composerTxt,
                wsErr,
              };
            })()
            """,
        )
        if expect_clarification:
            if not st["running"] and len(st["clarBtns"]) >= 2 and not st["wsErr"]:
                return {"elapsed_s": round(time.monotonic() - t0, 3), "state": st}
            if not st["running"] and ("处理失败" in st["statusTxt"] or st["wsErr"]):
                raise RuntimeError(f"Turn failed while expecting clarification for {question!r}: {st}")
        else:
            if not st["running"] and st["msgs"] >= prev_msgs + 2:
                return {"elapsed_s": round(time.monotonic() - t0, 3), "state": st}
            if not st["running"] and ("处理失败" in st["statusTxt"] or st["wsErr"]):
                raise RuntimeError(f"Turn failed for {question!r}: {st}")
        time.sleep(0.4)
    raise TimeoutError(f"Turn timed out after {timeout}s for question: {question!r}; last_state={st!r}")


def _click_clarification_and_send(
    session: str,
    matcher_js: str,
    *,
    timeout: float = 120.0,
) -> dict[str, Any]:
    prev_msgs = _msg_count(session)
    t0 = time.monotonic()
    chosen = _eval(
        session,
        f"""
        (() => {{
          const btns = Array.from(document.querySelectorAll('.clarification-actions button'));
          const target = ({matcher_js})(btns);
          if (!target) return null;
          target.click();
          return target.textContent;
        }})()
        """,
    )
    if not chosen:
        raise RuntimeError("Matching clarification button not found")
    _ab(session, "click", "#send")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = _eval(
            session,
            """
            (() => {
              const msgs = document.querySelectorAll('article.message').length;
              const statusTxt = document.querySelector('#turn-status')?.textContent || '';
              const composerTxt = document.querySelector('#composer-status')?.textContent || '';
              const actionsTxt = document.querySelector('#turn-actions')?.textContent || '';
              const wsErr = document.querySelector('#workspace-status')?.textContent || '';
              const running = composerTxt.includes('正在处理') || actionsTxt.includes('取消本轮') || statusTxt.includes('正在处理');
              return {running, msgs, statusTxt, composerTxt, wsErr};
            })()
            """,
        )
        if not st["running"] and st["msgs"] >= prev_msgs + 2:
            return {"elapsed_s": round(time.monotonic() - t0, 3), "chosen_option": chosen, "state": st}
        if not st["running"] and ("处理失败" in st["statusTxt"] or st["wsErr"]):
            raise RuntimeError(f"Clarification resume failed: {st}")
        time.sleep(0.4)
    raise TimeoutError(f"Clarification resume timed out after {timeout}s")


def _wait_last_message_images(session: str, expected_count: int = 1, timeout: float = 20.0) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        imgs = _eval(
            session,
            """
            (() => {
              const arts = Array.from(document.querySelectorAll('article.message'));
              const last = arts[arts.length - 1];
              if (!last) return [];
              return Array.from(last.querySelectorAll('.source-card img')).map(i => ({
                complete: i.complete,
                nw: i.naturalWidth,
                nh: i.naturalHeight,
                src: i.getAttribute('src')
              }));
            })()
            """,
        )
        if len(imgs) >= expected_count and all(i["complete"] and i["nw"] > 0 for i in imgs):
            return imgs
        time.sleep(0.25)
    raise TimeoutError(f"Expected {expected_count} loaded source-card image(s), got: {imgs}")


def _read_last_message_dom(session: str) -> dict[str, Any]:
    return _eval(
        session,
        """
        (() => {
          const arts = Array.from(document.querySelectorAll('article.message'));
          const last = arts[arts.length - 1];
          const textBlocks = Array.from(last.querySelectorAll('.text-block, .notice-block')).map(n => n.textContent);
          const cites = Array.from(last.querySelectorAll('.citations button')).map(b => ({
            label: b.textContent,
            evidence_id: b.dataset.evidenceId || null,
          }));
          const cards = Array.from(last.querySelectorAll('.source-card')).map(c => ({
            display_id: c.dataset.displayId,
            caption: c.querySelector('.source-caption')?.textContent || '',
            details: c.querySelector('.source-details')?.textContent || '',
            img: (() => {
              const i = c.querySelector('img');
              return i ? {complete: i.complete, nw: i.naturalWidth, nh: i.naturalHeight, src: i.getAttribute('src')} : null;
            })(),
          }));
          return {
            message_id: last.dataset.messageId,
            text: textBlocks.join('\\n'),
            full_text: last.textContent,
            cites,
            cards,
          };
        })()
        """,
    )


def _open_reader_by_click(session: str, selector_or_fn: str, *, is_js_fn: bool = False, expect_img: bool = True) -> dict[str, Any]:
    if is_js_fn:
        clicked = _eval(session, f"(() => {{ const el = ({selector_or_fn})(); if (!el) return false; el.click(); return true; }})()")
        if not clicked:
            raise RuntimeError("Target element to open #reader not found")
    else:
        _ab(session, "click", selector_or_fn)

    deadline = time.monotonic() + 15.0
    last_state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last_state = _eval(
            session,
            """
            (() => {
              const dlg = document.querySelector('#reader');
              const img = dlg?.querySelector('img');
              const hl = dlg?.querySelector('.reader-highlight');
              const excerpt = dlg?.querySelector('.reader-excerpt')?.textContent || '';
              return {
                open: Boolean(dlg?.open),
                title: document.querySelector('#reader-title')?.textContent || '',
                location: document.querySelector('#reader-location')?.textContent || '',
                status: document.querySelector('#reader-status')?.textContent || '',
                img_present: Boolean(img),
                img_loaded: Boolean(img && img.complete && img.naturalWidth > 0),
                img_size: img ? [img.naturalWidth, img.naturalHeight] : null,
                img_src: img ? img.getAttribute('src') : null,
                has_highlight: Boolean(hl),
                highlight_style: hl ? {
                  left: hl.style.left, top: hl.style.top, width: hl.style.width, height: hl.style.height
                } : null,
                excerpt_len: excerpt.length,
                body_text: document.querySelector('#reader-body')?.textContent?.slice(0, 300) || ''
              };
            })()
            """,
        )
        if last_state["open"]:
            if not expect_img or last_state["img_loaded"] or (not last_state["img_present"] and last_state["excerpt_len"] > 0):
                return last_state
        time.sleep(0.25)
    raise TimeoutError(f"#reader did not settle as expected: {last_state}")


def _close_reader(session: str) -> None:
    is_open = _eval(session, "Boolean(document.querySelector('#reader')?.open)")
    if is_open:
        _ab(session, "click", "#close-reader")
        time.sleep(0.2)


def _check_session_urls(session: str, urls: list[str]) -> list[dict[str, Any]]:
    return _eval_async(
        session,
        f"""
        const urls = {json.dumps(urls)};
        const out = [];
        for (const u of urls) {{
          const r = await fetch(u, {{credentials: 'same-origin', cache: 'no-store'}});
          const buf = new Uint8Array(await r.arrayBuffer());
          let sha256 = null;
          if (r.ok && buf.length > 0 && crypto?.subtle) {{
            const digest = await crypto.subtle.digest('SHA-256', buf);
            sha256 = Array.from(new Uint8Array(digest)).map(b => b.toString(16).padStart(2, '0')).join('');
          }}
          out.push({{
            url: u,
            status: r.status,
            content_type: r.headers.get('content-type'),
            bytes: buf.length,
            sha256,
          }});
        }}
        return out;
        """,
    )


def _fetch_session_snapshot(session: str, sid: str) -> dict[str, Any]:
    return _eval_async(
        session,
        f"return await fetch('/api/workspace/sessions/' + encodeURIComponent({sid!r}), {{credentials: 'same-origin', cache: 'no-store'}}).then(r => r.json());",
    )


def _collect_message_endpoints(session_snap: dict[str, Any], sid: str, message_id: str) -> dict[str, Any]:
    msg = next(m for m in session_snap["messages"] if m["message_id"] == message_id)
    sources = msg.get("sources") or {}
    ev_urls: list[str] = []
    pv_urls: list[str] = []
    media_urls: list[str] = []
    eids: list[str] = []
    visible = set(msg.get("visible_display_ids") or [])

    def _add_eid(eid: str | None) -> None:
        if not eid or eid not in sources or eid in eids:
            return
        eids.append(eid)
        ev = sources[eid]
        reads = ev.get("reads") or []
        has_text = any(
            r.get("method") in {"native_text", "native_table", "ocr"}
            and r.get("status") in {"succeeded", "partial"}
            and r.get("text")
            for r in reads
        )
        has_vision = any(
            r.get("method") == "vision"
            and r.get("status") in {"succeeded", "partial"}
            and r.get("actual_input_sha256")
            for r in reads
        )
        if has_text:
            ev_urls.append(f"/api/workspace/sessions/{sid}/messages/{message_id}/evidence/{eid}")
        if has_text or has_vision:
            pv_urls.append(f"/api/workspace/sessions/{sid}/messages/{message_id}/evidence/{eid}/preview")

    for block in msg.get("blocks", []):
        if block.get("type") == "source_image" and block.get("display_id") in visible:
            if block.get("media_url"):
                media_urls.append(block["media_url"])
            _add_eid(block.get("evidence_id"))
        for cit in block.get("citations", []):
            eid = cit if isinstance(cit, str) else cit.get("evidence_id")
            _add_eid(eid)
    return {"evidence_urls": ev_urls, "preview_urls": pv_urls, "media_urls": media_urls, "message": msg}


def _click_catalog_row(session: str, matcher_js: str) -> str:
    label = _eval(
        session,
        f"""
        (() => {{
          const rows = Array.from(document.querySelectorAll('#catalog .catalog-row'));
          const target = ({matcher_js})(rows);
          if (!target) return {{found: false, available: rows.map(r => ({{title: r.title, text: r.querySelector('.catalog-label')?.textContent || ''}}))}};
          target.click();
          return {{found: true, text: target.textContent}};
        }})()
        """,
    )
    if not label or not label.get("found"):
        raise RuntimeError(f"Catalog row not found. Available: {label}")
    _wait_nav_idle(session)
    time.sleep(0.3)
    return str(label["text"])


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return round(s[0], 3)
    k = (len(s) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return round(s[int(k)], 3)
    return round(s[f] * (c - k) + s[c] * (k - f), 3)


def _scrub_secrets(obj: Any, secrets: list[str]) -> Any:
    text = json.dumps(obj, ensure_ascii=False)
    for sec in secrets:
        if sec in text:
            raise AssertionError("Secret leakage detected in output JSON!")
    return obj


def main() -> None:
    env_loaded, secrets = _load_env_readonly()
    subject_dir = _export_clean_subject()
    run_dir = Path(tempfile.mkdtemp(prefix="srag-g30-web-run-", dir="/tmp"))
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"

    env = dict(env_loaded)
    env.update(
        {
            "PYTHONPATH": str(subject_dir),
            "DATA_DIR": str(run_dir / "data"),
            "SQLITE_PATH": str(run_dir / "library.sqlite3"),
            "COURSE_ROOT": str(run_dir / "courses"),
            "WORKSPACE_COURSE_IDS": "g30-course",
            "WORKSPACE_ANSWER_ENABLED": "true",
            "WORKSPACE_RETRIEVAL_STRATEGY": "fts",
            "SOURCE_VISION_ENABLED": "true",
            "QDRANT_URL": "http://127.0.0.1:1",
            "SRAG_SOFFICE": SOFFICE_PATH,
        }
    )

    server = subprocess.Popen(
        [
            PYTHON,
            "-c",
            (
                "import sys, pathlib, uvicorn; "
                f"sys.path.insert(0, {str(subject_dir)!r}); "
                "import app.rag.workspace as w; "
                f"assert pathlib.Path(w.__file__).resolve().is_relative_to(pathlib.Path({str(subject_dir)!r}).resolve()), w.__file__; "
                f"print('VERIFIED_WORKSPACE_FILE=' + str(pathlib.Path(w.__file__).resolve()), flush=True); "
                f"uvicorn.run('app.main:app', host='127.0.0.1', port={port}, log_level='warning')"
            ),
        ],
        cwd=str(subject_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    session_a = f"g30-web-a-{os.getpid()}"
    session_b = f"g30-web-b-{os.getpid()}"
    turn_latencies: list[float] = []
    report: dict[str, Any] = {
        "gate": "G30-track2-web-e2e",
        "subject_sha": SUBJECT_SHA,
        "subject_dir": str(subject_dir),
        "base_url": base_url,
        "effective_config": {
            "WORKSPACE_COURSE_IDS": "g30-course",
            "WORKSPACE_ANSWER_ENABLED": True,
            "WORKSPACE_RETRIEVAL_STRATEGY": "fts",
            "SOURCE_VISION_ENABLED": True,
            "QDRANT_URL": "http://127.0.0.1:1",
            "SRAG_SOFFICE": SOFFICE_PATH,
        },
        "steps": {},
    }

    try:
        _wait_port(port)
        verified_line = server.stdout.readline().strip() if server.stdout else ""
        report["verified_loaded_module"] = verified_line

        _ab(session_a, "open", f"{base_url}/workspace/")
        _wait_ready(session_a)

        sid_a = _eval_async(
            session_a,
            "const d = await fetch('/api/workspace/sessions', {credentials:'same-origin'}).then(r=>r.json()); return (d.sessions || [])[0]?.session_id;",
        )
        report["session_a_id"] = sid_a

        print("[Step 1] Uploading 3 formats via UI...", flush=True)
        # =====================================================================
        # Step 1: Import PDF, PPTX, Notebook v1 via empty-library Web UI
        # =====================================================================
        _eval(session_a, "(() => { const p = document.querySelector('#import-panel'); if (p) p.open = true; })()")
        _ab(
            session_a,
            "upload",
            "#import-files",
            str(FIXTURES / "Speculative-Decoding-2211.17192v2.pdf"),
            str(FIXTURES / "Week06-Thermal-Cache-Lab.pptx"),
            str(FIXTURES / "notebook/v1/Lab06-Thermal-Sweep.ipynb"),
        )
        _ab(session_a, "click", "#import-button")
        deadline = time.monotonic() + 90.0
        import_items: list[str] = []
        while time.monotonic() < deadline:
            import_items = _eval(
                session_a,
                "Array.from(document.querySelectorAll('#import-results li')).map(li => li.textContent)",
            )
            if len(import_items) == 3 and all("导入中" not in x for x in import_items):
                break
            time.sleep(0.5)
        time.sleep(0.5)
        catalog_docs = _eval(
            session_a,
            """
            Array.from(document.querySelectorAll('#catalog .catalog-row')).map(r => ({
              label: r.querySelector('.catalog-label')?.textContent || '',
              version: r.querySelector('.catalog-version')?.textContent || null,
              node_id: r.dataset.nodeId || null,
              title: r.title || ''
            }))
            """,
        )
        expected_files = {
            "Speculative-Decoding-2211.17192v2.pdf",
            "Week06-Thermal-Cache-Lab.pptx",
            "Lab06-Thermal-Sweep.ipynb",
        }
        doc_rows = {d["label"]: d["version"] for d in catalog_docs if d["label"] in expected_files}
        step1_pass = len(doc_rows) == 3 and all(v == "v1" for v in doc_rows.values())
        report["steps"]["step1_web_import_3formats"] = {
            "passed": step1_pass,
            "import_results": import_items,
            "catalog_docs": doc_rows,
        }
        assert step1_pass, f"Step 1 failed: {report['steps']['step1_web_import_3formats']}"
        print(f"[Step 1 PASS] {doc_rows}", flush=True)

        print("[Step 2] Asking Chinese synthesis question on PDF...", flush=True)
        # =====================================================================
        # Step 2: Long English PDF Chinese synthesis QA + click citation -> #reader
        # =====================================================================
        q2 = "请用中文概括 Speculative Decoding 的核心机制，以及它能够带来推理加速的关键条件是什么？"
        t2 = _ask_ui(session_a, q2)
        turn_latencies.append(t2["elapsed_s"])
        dom2 = _read_last_message_dom(session_a)
        reader2 = _open_reader_by_click(
            session_a,
            "() => { const arts = Array.from(document.querySelectorAll('article.message')); return arts[arts.length - 1]?.querySelector('.citations button'); }",
            is_js_fn=True,
            expect_img=True,
        )
        _close_reader(session_a)
        snap2 = _fetch_session_snapshot(session_a, sid_a)
        ep2 = _collect_message_endpoints(snap2, sid_a, dom2["message_id"])
        ev2_checks = _check_session_urls(session_a, ep2["evidence_urls"])
        has_zh = bool(re.search(r"[\u4e00-\u9fff]{15,}", dom2["text"]))
        has_mech = any(k in dom2["text"] for k in ("草稿", "近似模型", "draft", "目标模型", "并行", "验证"))
        reader2_ok = (
            reader2["open"]
            and "Speculative-Decoding-2211.17192v2.pdf" in reader2["title"]
            and "v1" in reader2["title"]
            and "第" in reader2["title"]
            and "页" in reader2["title"]
        )
        step2_pass = (
            has_zh
            and has_mech
            and len(dom2["cites"]) >= 1
            and reader2_ok
            and all(c["status"] == 200 for c in ev2_checks)
        )
        report["steps"]["step2_pdf_zh_synthesis"] = {
            "passed": step2_pass,
            "elapsed_s": t2["elapsed_s"],
            "answer_preview": dom2["text"][:350],
            "citations": dom2["cites"],
            "reader": reader2,
            "evidence_http": ev2_checks,
        }
        assert step2_pass, f"Step 2 failed: {report['steps']['step2_pdf_zh_synthesis']}"
        print(f"[Step 2 PASS] elapsed={t2['elapsed_s']}s", flush=True)

        print("[Step 3] Testing PDF Table 1, Q04 clarification, Eq(1)/Figure 5 /evidence 200...", flush=True)

        # =====================================================================
        # Step 3: PDF complex objects
        #   3a: borderless booktabs table + aligned #reader highlight
        #   3b: focused two-figure p5 Q04 clarification -> click button -> real VLM -> #reader crop
        #   3c: caption / formula /evidence HTTP 200 + #reader open
        # =====================================================================
        # 3a: Table 1 p5
        q3a = "在 Speculative-Decoding-2211.17192v2.pdf 第 5 页 Table 1 中，当 α=0.90 且 γ=10 时，SPEED 和 OPERATIONS 分别是多少？该表格的前提假设是什么？"
        t3a = _ask_ui(session_a, q3a)
        turn_latencies.append(t3a["elapsed_s"])
        dom3a = _read_last_message_dom(session_a)
        reader3a_cite = _open_reader_by_click(
            session_a,
            "() => { const arts = Array.from(document.querySelectorAll('article.message')); return arts[arts.length - 1]?.querySelector('.citations button'); }",
            is_js_fn=True,
            expect_img=True,
        )
        _close_reader(session_a)
        reader3a_page = _open_reader_by_click(
            session_a,
            "() => { const arts = Array.from(document.querySelectorAll('article.message')); const last = arts[arts.length - 1]; return Array.from(last.querySelectorAll('.source-actions button')).find(b => b.textContent === '回原件') || last.querySelector('.citations button'); }",
            is_js_fn=True,
            expect_img=True,
        )
        _close_reader(session_a)
        hl = reader3a_page.get("highlight_style") or {}
        hl_ok = False
        if reader3a_page["has_highlight"] and hl:
            vals = [float(hl[k].rstrip("%")) for k in ("left", "top", "width", "height")]
            hl_ok = (0 <= vals[0] < 100 and 0 <= vals[1] < 100 and vals[2] > 1 and vals[3] > 1 and vals[0] + vals[2] <= 100.5 and vals[1] + vals[3] <= 100.5)
        snap3a = _fetch_session_snapshot(session_a, sid_a)
        ep3a = _collect_message_endpoints(snap3a, sid_a, dom3a["message_id"])
        ep3a_http = _check_session_urls(session_a, ep3a["media_urls"] + ep3a["evidence_urls"] + ep3a["preview_urls"])
        s3a_ok = (
            "6.86" in dom3a["text"]
            and "1.60" in dom3a["text"]
            and reader3a_cite["open"]
            and reader3a_cite["img_loaded"]
            and reader3a_page["open"]
            and reader3a_page["img_loaded"]
            and hl_ok
            and all(c["status"] == 200 for c in ep3a_http)
        )

        # 3b: Focus page 5 in catalog UI -> Q04 clarification -> click button -> real VLM -> #reader crop
        _click_catalog_row(session_a, "rows => rows.find(r => r.querySelector('.catalog-label')?.textContent === 'Speculative-Decoding-2211.17192v2.pdf')")
        _click_catalog_row(session_a, "rows => rows.find(r => { const t = r.querySelector('.catalog-label')?.textContent || ''; return r.title === '页' && (t === '第 5 页' || t.startsWith('第 5 页 ·')); })")
        _close_reader(session_a)

        t3b_1 = _ask_ui(session_a, "这张图里的曲线趋势是什么？", expect_clarification=True)
        turn_latencies.append(t3b_1["elapsed_s"])
        clar_btns_3b = t3b_1["state"]["clarBtns"]
        t3b_2 = _click_clarification_and_send(session_a, "btns => btns[0]")
        turn_latencies.append(t3b_2["elapsed_s"])
        imgs3b = _wait_last_message_images(session_a, expected_count=1)
        dom3b = _read_last_message_dom(session_a)
        reader3b = _open_reader_by_click(
            session_a,
            "() => { const arts = Array.from(document.querySelectorAll('article.message')); const last = arts[arts.length - 1]; return Array.from(last.querySelectorAll('.source-actions button')).find(b => b.textContent === '查看图片') || last.querySelector('.citations button'); }",
            is_js_fn=True,
            expect_img=True,
        )
        _close_reader(session_a)
        snap3b = _fetch_session_snapshot(session_a, sid_a)
        ep3b = _collect_message_endpoints(snap3b, sid_a, dom3b["message_id"])
        ep3b_http = _check_session_urls(session_a, ep3b["media_urls"] + ep3b["evidence_urls"] + ep3b["preview_urls"])
        s3b_ok = (
            len(clar_btns_3b) == 2
            and len(imgs3b) == 1
            and reader3b["open"]
            and reader3b["img_loaded"]
            and all(c["status"] == 200 for c in ep3b_http)
        )

        # 3c: Formula Eq. (1) & Figure 5 caption -> /evidence 200 + #reader open
        # Click document node so scope is PDF (not focused on p5)
        _click_catalog_row(session_a, "rows => rows.find(r => r.querySelector('.catalog-label')?.textContent === 'Speculative-Decoding-2211.17192v2.pdf')")
        _close_reader(session_a)
        q3c_formula = "Speculative-Decoding-2211.17192v2.pdf 第 3 页公式 (1) 的数学表达式是什么？它依赖什么假设？"
        t3c_f = _ask_ui(session_a, q3c_formula)
        turn_latencies.append(t3c_f["elapsed_s"])
        dom3c_f = _read_last_message_dom(session_a)
        reader3c_f = _open_reader_by_click(
            session_a,
            "() => { const arts = Array.from(document.querySelectorAll('article.message')); return arts[arts.length - 1]?.querySelector('.citations button'); }",
            is_js_fn=True,
            expect_img=True,
        )
        _close_reader(session_a)

        q3c_caption = "Speculative-Decoding-2211.17192v2.pdf 第 6 页 Figure 5 的图注内容是什么？其中紫色、蓝色、黄色和橙色块分别对应什么？"
        t3c_c = _ask_ui(session_a, q3c_caption)
        turn_latencies.append(t3c_c["elapsed_s"])
        dom3c_c = _read_last_message_dom(session_a)
        reader3c_c = _open_reader_by_click(
            session_a,
            "() => { const arts = Array.from(document.querySelectorAll('article.message')); return arts[arts.length - 1]?.querySelector('.citations button'); }",
            is_js_fn=True,
            expect_img=True,
        )
        _close_reader(session_a)

        snap3c = _fetch_session_snapshot(session_a, sid_a)
        ep3c_f = _collect_message_endpoints(snap3c, sid_a, dom3c_f["message_id"])
        ep3c_c = _collect_message_endpoints(snap3c, sid_a, dom3c_c["message_id"])
        ev3c_http = _check_session_urls(session_a, ep3c_f["evidence_urls"] + ep3c_c["evidence_urls"])
        s3c_ok = (
            len(ev3c_http) >= 2
            and all(c["status"] == 200 for c in ev3c_http)
            and reader3c_f["open"]
            and reader3c_c["open"]
        )

        step3_pass = s3a_ok and s3b_ok and s3c_ok
        report["steps"]["step3_pdf_complex_objects"] = {
            "passed": step3_pass,
            "table_3a": {"passed": s3a_ok, "elapsed_s": t3a["elapsed_s"], "answer_preview": dom3a["text"][:300], "reader_cite": reader3a_cite, "reader_full_page": reader3a_page, "http_checks": ep3a_http},
            "q04_clarify_3b": {
                "passed": s3b_ok,
                "clarify_elapsed_s": t3b_1["elapsed_s"],
                "resume_elapsed_s": t3b_2["elapsed_s"],
                "clarification_buttons": clar_btns_3b,
                "chosen": t3b_2["chosen_option"],
                "answer_preview": dom3b["text"][:300],
                "imgs": imgs3b,
                "reader": reader3b,
                "http_checks": ep3b_http,
            },
            "formula_caption_3c": {
                "passed": s3c_ok,
                "formula_elapsed_s": t3c_f["elapsed_s"],
                "caption_elapsed_s": t3c_c["elapsed_s"],
                "formula_preview": dom3c_f["text"][:250],
                "caption_preview": dom3c_c["text"][:250],
                "formula_reader": reader3c_f,
                "caption_reader": reader3c_c,
                "evidence_http": ev3c_http,
            },
        }
        assert step3_pass, f"Step 3 failed: {report['steps']['step3_pdf_complex_objects']}"
        print("[Step 3 PASS]", flush=True)

        print("[Step 4] Testing PPTX Slide 2 table & Slide 3 chart...", flush=True)

        # =====================================================================
        # Step 4: Real LibreOffice PPTX full flow (Slide 2 table + Slide 3 chart)
        # =====================================================================
        _ab(session_a, "click", "#clear-file")
        _wait_nav_idle(session_a)
        time.sleep(0.3)

        # 4a: Slide 2 table
        q4a = "Week06-Thermal-Cache-Lab.pptx 第 2 页表格中，哪个负载的 L2 Miss Rate (%) 最高、哪个负载的 Peak Tj (°C) 最高？请写出对应数值、单位以及 Sustained Clock 和 Energy-Delay。"
        t4a = _ask_ui(session_a, q4a)
        turn_latencies.append(t4a["elapsed_s"])
        dom4a = _read_last_message_dom(session_a)
        reader4a = _open_reader_by_click(
            session_a,
            "() => { const arts = Array.from(document.querySelectorAll('article.message')); return arts[arts.length - 1]?.querySelector('.citations button'); }",
            is_js_fn=True,
            expect_img=True,
        )
        _close_reader(session_a)
        snap4a = _fetch_session_snapshot(session_a, sid_a)
        ep4a = _collect_message_endpoints(snap4a, sid_a, dom4a["message_id"])
        ep4a_http = _check_session_urls(session_a, ep4a["evidence_urls"] + ep4a["preview_urls"])
        s4a_ok = (
            "GraphBFS-CSR" in dom4a["text"]
            and "11.4" in dom4a["text"]
            and "SparseFFT-16K" in dom4a["text"]
            and "91.8" in dom4a["text"]
            and "2.95" in dom4a["text"]
            and "21.6" in dom4a["text"]
            and reader4a["open"]
            and reader4a["img_loaded"]
            and all(c["status"] == 200 for c in ep4a_http)
        )

        # 4b: Slide 3 embedded curve chart (natural phrasing)
        q4b = "Week06-Thermal-Cache-Lab.pptx 第3页图片里曲线的横纵轴和趋势是什么？"
        t4b = _ask_ui(session_a, q4b)
        turn_latencies.append(t4b["elapsed_s"])
        imgs4b = _wait_last_message_images(session_a, expected_count=1)
        dom4b = _read_last_message_dom(session_a)
        reader4b = _open_reader_by_click(
            session_a,
            "() => { const arts = Array.from(document.querySelectorAll('article.message')); const last = arts[arts.length - 1]; return Array.from(last.querySelectorAll('.source-actions button')).find(b => b.textContent === '查看图片') || last.querySelector('.citations button'); }",
            is_js_fn=True,
            expect_img=True,
        )
        _close_reader(session_a)
        snap4b = _fetch_session_snapshot(session_a, sid_a)
        ep4b = _collect_message_endpoints(snap4b, sid_a, dom4b["message_id"])
        ep4b_http = _check_session_urls(session_a, ep4b["media_urls"] + ep4b["evidence_urls"] + ep4b["preview_urls"])
        s4b_ok = (
            ("LRU" in dom4b["text"] or "FIFO" in dom4b["text"])
            and len(imgs4b) == 1
            and reader4b["open"]
            and reader4b["img_loaded"]
            and all(c["status"] == 200 for c in ep4b_http)
        )

        step4_pass = s4a_ok and s4b_ok
        report["steps"]["step4_pptx_libreoffice"] = {
            "passed": step4_pass,
            "slide2_table_4a": {
                "passed": s4a_ok,
                "elapsed_s": t4a["elapsed_s"],
                "answer_preview": dom4a["text"][:320],
                "reader": reader4a,
                "http_checks": ep4a_http,
            },
            "slide3_chart_4b": {
                "passed": s4b_ok,
                "elapsed_s": t4b["elapsed_s"],
                "answer_preview": dom4b["text"][:320],
                "imgs": imgs4b,
                "reader": reader4b,
                "http_checks": ep4b_http,
            },
        }
        assert step4_pass, f"Step 4 failed: {report['steps']['step4_pptx_libreoffice']}"
        print("[Step 4 PASS]", flush=True)

        print("[Step 5] Testing Notebook multi-output -> v2_reordered -> old-only -> bilateral...", flush=True)

        # =====================================================================
        # Step 5: Notebook multi-output clarification -> replace v2_reordered ->
        #         "只看旧版" -> "对比两版" bilateral recovery
        # =====================================================================
        _click_catalog_row(session_a, "rows => rows.find(r => r.querySelector('.catalog-label')?.textContent === 'Lab06-Thermal-Sweep.ipynb')")
        _click_catalog_row(session_a, "rows => rows.find(r => r.title === '章节' && (r.querySelector('.catalog-label')?.textContent || '').includes('Thermal Sweep'))")
        _click_catalog_row(session_a, "rows => rows.find(r => { const t = r.querySelector('.catalog-label')?.textContent || ''; return r.title === 'Cell' && (t === 'Cell 2' || t.startsWith('Cell 2 ·')); })")
        _close_reader(session_a)

        # 5a: Focus Cell 2 -> ask figure -> clarification -> click output 1 (temp curve) -> v1 read
        t5a_1 = _ask_ui(session_a, "这张图里的横纵轴单位和峰值数值是多少？", expect_clarification=True)
        turn_latencies.append(t5a_1["elapsed_s"])
        clar_btns_5a = t5a_1["state"]["clarBtns"]
        t5a_2 = _click_clarification_and_send(session_a, "btns => btns.find(b => b.textContent.includes('输出 1')) || btns[0]")
        turn_latencies.append(t5a_2["elapsed_s"])
        imgs5a = _wait_last_message_images(session_a, expected_count=1)
        dom5a = _read_last_message_dom(session_a)
        snap5a = _fetch_session_snapshot(session_a, sid_a)
        ep5a = _collect_message_endpoints(snap5a, sid_a, dom5a["message_id"])
        ep5a_http = _check_session_urls(session_a, ep5a["media_urls"] + ep5a["evidence_urls"] + ep5a["preview_urls"])
        v1_media_url = ep5a["media_urls"][0]
        v1_media_sha_before_reload = next(c["sha256"] for c in ep5a_http if c["url"] == v1_media_url)
        v1_msg_id = dom5a["message_id"]
        s5a_ok = (
            len(clar_btns_5a) >= 2
            and "91.8" in dom5a["text"]
            and all(c["status"] == 200 for c in ep5a_http)
        )

        # 5b: Replace with v2_reordered via UI (#replace-document + #import-files)
        _eval(session_a, "(() => { const p = document.querySelector('#import-panel'); if (p) p.open = true; })()")
        _eval(session_a, "(() => { const cb = document.querySelector('#replace-document'); if (cb && !cb.checked) cb.click(); })()")
        _ab(session_a, "upload", "#import-files", str(FIXTURES / "notebook/v2_reordered/Lab06-Thermal-Sweep.ipynb"))
        _ab(session_a, "click", "#import-button")
        deadline = time.monotonic() + 60.0
        replace_res: list[str] = []
        while time.monotonic() < deadline:
            replace_res = _eval(session_a, "Array.from(document.querySelectorAll('#import-results li')).map(li => li.textContent)")
            if replace_res and any("v2" in x and "导入中" not in x for x in replace_res):
                break
            time.sleep(0.4)
        time.sleep(0.4)
        nb_ver_after = _eval(
            session_a,
            "Array.from(document.querySelectorAll('#catalog .catalog-row')).find(r => r.querySelector('.catalog-label')?.textContent === 'Lab06-Thermal-Sweep.ipynb')?.querySelector('.catalog-version')?.textContent",
        )
        s5b_ok = nb_ver_after == "v2"

        # 5c: "只看旧版刚才读过的那张图..." -> strictly v1 (0 version leakage)
        t5c = _ask_ui(session_a, "只看旧版刚才读过的那张图，它的横轴单位和峰值温度是多少？")
        turn_latencies.append(t5c["elapsed_s"])
        dom5c = _read_last_message_dom(session_a)
        reader5c = _open_reader_by_click(
            session_a,
            "() => { const arts = Array.from(document.querySelectorAll('article.message')); return arts[arts.length - 1]?.querySelector('.citations button'); }",
            is_js_fn=True,
            expect_img=True,
        )
        _close_reader(session_a)
        snap5c = _fetch_session_snapshot(session_a, sid_a)
        ep5c = _collect_message_endpoints(snap5c, sid_a, dom5c["message_id"])
        ep5c_http = _check_session_urls(session_a, ep5c["media_urls"] + ep5c["evidence_urls"] + ep5c["preview_urls"])
        ev_versions_5c = [
            ev.get("revision", {}).get("source", {}).get("version")
            for ev in (ep5c["message"].get("sources") or {}).values()
        ]
        s5c_ok = (
            "91.8" in dom5c["text"]
            and "83.4" not in dom5c["text"]
            and "min" not in dom5c["text"].lower()
            and ev_versions_5c == [1]
            and reader5c["open"]
            and reader5c["img_loaded"]
            and len(ep5c_http) >= 1
            and all(c["status"] == 200 for c in ep5c_http)
            and any(c["sha256"] == v1_media_sha_before_reload for c in ep5c_http)
        )

        # 5d: "对比最新版和旧版刚才读过的图的纵轴单位和数值" -> reordered clarification -> bilateral v1+v2
        t5d_1 = _ask_ui(session_a, "对比最新版和旧版刚才读过的图的纵轴单位和数值", expect_clarification=True)
        turn_latencies.append(t5d_1["elapsed_s"])
        clar_btns_5d = t5d_1["state"]["clarBtns"]
        t5d_2 = _click_clarification_and_send(session_a, "btns => btns.find(b => b.textContent.includes('输出 3')) || btns[btns.length - 1]")
        turn_latencies.append(t5d_2["elapsed_s"])
        imgs5d = _wait_last_message_images(session_a, expected_count=2)
        dom5d = _read_last_message_dom(session_a)
        snap5d = _fetch_session_snapshot(session_a, sid_a)
        ep5d = _collect_message_endpoints(snap5d, sid_a, dom5d["message_id"])
        ep5d_http = _check_session_urls(session_a, ep5d["media_urls"] + ep5d["evidence_urls"] + ep5d["preview_urls"])
        ev_versions_5d = sorted(
            {
                ev.get("revision", {}).get("source", {}).get("version")
                for ev in (ep5d["message"].get("sources") or {}).values()
            }
        )
        s5d_ok = (
            len(clar_btns_5d) >= 2
            and len(dom5d["cards"]) == 2
            and len(imgs5d) == 2
            and ev_versions_5d == [1, 2]
            and "91.8" in dom5d["text"]
            and "83.4" in dom5d["text"]
            and len(ep5d["media_urls"]) == 2
            and len(ep5d["preview_urls"]) == 2
            and all(c["status"] == 200 for c in ep5d_http)
        )

        step5_pass = s5a_ok and s5b_ok and s5c_ok and s5d_ok
        report["steps"]["step5_notebook_version_bilateral"] = {
            "passed": step5_pass,
            "v1_multi_output_5a": {
                "passed": s5a_ok,
                "clarify_elapsed_s": t5a_1["elapsed_s"],
                "resume_elapsed_s": t5a_2["elapsed_s"],
                "clarification_buttons": clar_btns_5a,
                "chosen": t5a_2["chosen_option"],
                "answer_preview": dom5a["text"][:280],
                "imgs": imgs5a,
            },
            "replace_v2_reordered_5b": {
                "passed": s5b_ok,
                "import_results": replace_res,
                "catalog_version": nb_ver_after,
            },
            "old_only_control_5c": {
                "passed": s5c_ok,
                "elapsed_s": t5c["elapsed_s"],
                "evidence_versions": ev_versions_5c,
                "answer_preview": dom5c["text"][:280],
                "reader": reader5c,
                "http_checks": ep5c_http,
            },
            "bilateral_recovery_5d": {
                "passed": s5d_ok,
                "clarify_elapsed_s": t5d_1["elapsed_s"],
                "resume_elapsed_s": t5d_2["elapsed_s"],
                "clarification_buttons": clar_btns_5d,
                "chosen": t5d_2["chosen_option"],
                "card_count": len(dom5d["cards"]),
                "evidence_versions": ev_versions_5d,
                "answer_preview": dom5d["text"][:450],
                "http_checks": ep5d_http,
            },
        }
        assert step5_pass, f"Step 5 failed: {report['steps']['step5_notebook_version_bilateral']}"
        print("[Step 5 PASS]", flush=True)

        print("[Step 6] Testing reload recovery...", flush=True)

        # =====================================================================
        # Step 6: Browser reload recovery & historical v1 citation byte integrity
        # =====================================================================
        msg_count_before_reload = _msg_count(session_a)
        _ab(session_a, "reload")
        _wait_ready(session_a)
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if _msg_count(session_a) >= msg_count_before_reload:
                break
            time.sleep(0.25)
        msg_count_after_reload = _msg_count(session_a)
        reader6 = _open_reader_by_click(
            session_a,
            f"() => document.querySelector(\"article.message[data-message-id='{v1_msg_id}'] .citations button\") || document.querySelector(\"article.message[data-message-id='{v1_msg_id}'] .source-actions button:nth-child(2)\")",
            is_js_fn=True,
            expect_img=True,
        )
        _close_reader(session_a)
        recheck_v1 = _check_session_urls(session_a, [v1_media_url])[0]
        step6_pass = (
            msg_count_after_reload == msg_count_before_reload
            and reader6["open"]
            and reader6["img_loaded"]
            and recheck_v1["status"] == 200
            and recheck_v1["sha256"] == v1_media_sha_before_reload
        )
        report["steps"]["step6_reload_recovery"] = {
            "passed": step6_pass,
            "msg_count_before": msg_count_before_reload,
            "msg_count_after": msg_count_after_reload,
            "v1_reader": reader6,
            "v1_sha256_before": v1_media_sha_before_reload,
            "v1_sha256_after": recheck_v1["sha256"],
        }
        assert step6_pass, f"Step 6 failed: {report['steps']['step6_reload_recovery']}"
        print("[Step 6 PASS]", flush=True)

        print("[Step 7] Testing missing-evidence honesty...", flush=True)

        # =====================================================================
        # Step 7: Missing-evidence honesty boundary (0 hallucination)
        # =====================================================================
        _ab(session_a, "click", "#clear-file")
        _wait_nav_idle(session_a)
        q7 = "请给出这三份资料中明确记载的 GPU L3 共享缓存总容量（MB）以及 T5-11B 在 WMT14 EnFr 上的精确 BLEU 分数。"
        t7 = _ask_ui(session_a, q7)
        turn_latencies.append(t7["elapsed_s"])
        dom7 = _read_last_message_dom(session_a)
        honest_markers = ("未", "没有", "不包含", "未提及", "未给出", "未找到", "无法", "缺少", "不存在")
        step7_pass = any(m in dom7["text"] for m in honest_markers)
        report["steps"]["step7_honest_missing_evidence"] = {
            "passed": step7_pass,
            "elapsed_s": t7["elapsed_s"],
            "answer_text": dom7["text"][:400],
        }
        assert step7_pass, f"Step 7 failed: {report['steps']['step7_honest_missing_evidence']}"
        print("[Step 7 PASS]", flush=True)

        print("[Step 8] Testing cross-session 404 isolation...", flush=True)

        # =====================================================================
        # Step 8: Cross-session isolation (HTTP 404 on /media, /evidence, /preview)
        # =====================================================================
        _ab(session_b, "open", f"{base_url}/workspace/")
        _wait_ready(session_b)
        sid_b = _eval_async(
            session_b,
            "const d = await fetch('/api/workspace/sessions', {credentials:'same-origin'}).then(r=>r.json()); return (d.sessions || [])[0]?.session_id;",
        )
        # Use session_a's /media, /evidence, /preview URLs from session_b (different owner cookie!)
        target_media = ep5d["media_urls"][0]
        target_ev = ep4a["evidence_urls"][0]
        target_ev_pdf = ep2["evidence_urls"][0]
        target_pv = ep5d["preview_urls"][0]
        cross_checks = _check_session_urls(session_b, [target_media, target_ev, target_ev_pdf, target_pv])
        step8_pass = sid_b != sid_a and all(c["status"] == 404 and c["bytes"] < 200 for c in cross_checks)
        report["steps"]["step8_cross_session_404"] = {
            "passed": step8_pass,
            "session_b_id": sid_b,
            "cross_checks": cross_checks,
        }
        assert step8_pass, f"Step 8 failed: {report['steps']['step8_cross_session_404']}"
        print("[Step 8 PASS]", flush=True)

        # =====================================================================
        # Real-call accounting & latency percentiles
        # =====================================================================
        final_snap = _fetch_session_snapshot(session_a, sid_a)
        deepseek_gen_calls = 0
        deepseek_rewrite_calls = 0
        deepseek_gen_tokens = 0
        deepseek_rewrite_tokens = 0
        vision_real_calls = 0
        vision_cache_reuses = 0
        vision_total_tokens = 0
        mock_calls = 0
        seen_vision_keys: set[tuple[str, int, str, str | None]] = set()

        for turn in final_snap.get("turns", []):
            trace = turn.get("trace") or {}
            gen = trace.get("generation") or {}
            if gen.get("status") in {"succeeded", "partial"} and gen.get("model"):
                deepseek_gen_calls += int(gen.get("external_call_count") or 1)
                deepseek_gen_tokens += int(gen.get("total_tokens") or 0)
            if gen.get("execution") == "mock_protocol":
                mock_calls += 1
            ret = trace.get("retrieval") or {}
            rw = ret.get("rewrite") or trace.get("rewrite") or {}
            if rw.get("status") == "succeeded" and rw.get("external_call_count"):
                deepseek_rewrite_calls += int(rw.get("external_call_count") or 0)
                deepseek_rewrite_tokens += int(rw.get("total_tokens") or 0)
            vis = trace.get("vision") or {}
            for vr in vis.get("reads") or []:
                if vr.get("execution") == "real_provider":
                    vision_total_tokens += int((vr.get("usage") or {}).get("total_tokens") or 0)

        for msg in final_snap.get("messages", []):
            for ev in (msg.get("sources") or {}).values():
                for rd in ev.get("reads", []):
                    if rd.get("method") == "vision":
                        orig = rd.get("origin_execution")
                        ex = rd.get("execution")
                        if orig == "mock_protocol" or ex == "mock_protocol":
                            mock_calls += 1
                        elif ex == "real_provider":
                            src = rd.get("actual_source") or {}
                            vkey = (str(src.get("document_id")), int(src.get("version") or 0), str(src.get("source_id")), rd.get("actual_input_sha256"))
                            if vkey not in seen_vision_keys:
                                seen_vision_keys.add(vkey)
                                vision_real_calls += 1
                        elif ex == "cache_reuse":
                            vision_cache_reuses += 1

        passed_steps = sum(1 for s in report["steps"].values() if s["passed"])
        total_steps = len(report["steps"])
        report["summary"] = {
            "passed_steps": passed_steps,
            "total_steps": total_steps,
            "pass_rate": f"{passed_steps}/{total_steps}",
            "severe_errors": 0 if passed_steps == total_steps and mock_calls == 0 else (total_steps - passed_steps + mock_calls),
            "real_call_ledger": {
                "deepseek_model": "deepseek-flash",
                "deepseek_generation_calls": deepseek_gen_calls,
                "deepseek_rewrite_calls": deepseek_rewrite_calls,
                "deepseek_total_calls": deepseek_gen_calls + deepseek_rewrite_calls,
                "deepseek_generation_tokens": deepseek_gen_tokens,
                "deepseek_rewrite_tokens": deepseek_rewrite_tokens,
                "deepseek_total_tokens": deepseek_gen_tokens + deepseek_rewrite_tokens,
                "vision_model": "qwen3-vl-plus",
                "vision_real_provider_calls": vision_real_calls,
                "vision_cache_reuses": vision_cache_reuses,
                "vision_total_tokens": vision_total_tokens,
                "mock_protocol_calls": mock_calls,
            },
            "latency_seconds": {
                "turn_count": len(turn_latencies),
                "values": turn_latencies,
                "p50": _percentile(turn_latencies, 50),
                "p95": _percentile(turn_latencies, 95),
                "max": round(max(turn_latencies), 3) if turn_latencies else None,
            },
        }

        _scrub_secrets(report, secrets)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    except Exception:
        try:
            snap_err = _fetch_session_snapshot(session_a, report.get("session_a_id", "")) if report.get("session_a_id") else None
            if snap_err:
                print("[DIAGNOSTIC LAST TURN]", json.dumps((snap_err.get("turns") or [])[-1:], ensure_ascii=False), flush=True)
        except Exception:
            pass
        raise
    finally:
        for sess in (session_a, session_b):
            try:
                _ab(sess, "close")
            except Exception:
                pass
        server.terminate()
        try:
            out_s, err_s = server.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            out_s, err_s = server.communicate()
        if err_s and err_s.strip():
            print("[SERVER STDERR]\n" + err_s[-2000:], flush=True)
        shutil.rmtree(subject_dir, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
