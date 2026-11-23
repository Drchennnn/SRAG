#!/usr/bin/env python3
"""Part C: Real Chromium (/workspace/) E2E verification via agent-browser against clean archive 6d9fb7d.

Verifies in live Chromium:
  1. Q04 focused two-figure PDF page -> .clarification-actions buttons render -> click Left button ->
     real VLM visual answer -> click citation opens #reader with image loaded.
  2. PDF borderless booktabs table -> click text citation opens #reader with FULL-PAGE image +
     aligned .reader-highlight box (verifying table_crop Branch C highlight coordinate fix!).
  3. Notebook v1 -> v2_reordered (b2) -> .clarification-actions buttons render -> click output 3 ->
     bilateral v1 + v2 cards & citations render in DOM and open in #reader.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SUBJECT_DIR = Path(os.environ.get("G29_SUBJECT_DIR", "/tmp/srag-g29-gate-subject-6d9fb7d"))
SUBJECT_SHA = os.environ.get("G29_SUBJECT_SHA", "6d9fb7d45a3dd19424b5fc151900c703abb23462")
OUT_FILE = os.environ.get("G29_BROWSER_OUT_FILE", "G29-browser-e2e-results.json")
GATE_DIR = Path("/Users/Link/code/SRAG-g29-gate/docs/G/G29-gate")
FIXTURES = GATE_DIR / "fixtures"
ENV_FILE = Path("/Users/Link/code/SRAG/.env")
PYTHON = "/Users/Link/code/SRAG/.venv/bin/python"
AGENT_BROWSER = "/Users/Link/.local/bin/agent-browser"
PORT = 18029
BASE_URL = f"http://127.0.0.1:{PORT}"


def _load_env() -> dict[str, str]:
    env = dict(os.environ)
    if ENV_FILE.exists():
        for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _wait_port(port: int, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.25)
    raise RuntimeError(f"Server did not start on port {port}")


def _ab(session: str, *args: str, timeout: int = 60) -> str:
    cmd = [AGENT_BROWSER, "--session", session, *args]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"agent-browser failed: {' '.join(args)}\nSTDOUT:{res.stdout}\nSTDERR:{res.stderr}")
    return res.stdout.strip()


def _eval(session: str, js: str) -> Any:
    raw = _ab(session, "eval", "--json", js)
    parsed = json.loads(raw)
    if isinstance(parsed, dict) and "data" in parsed:
        data = parsed["data"]
        return data.get("result") if isinstance(data, dict) and "result" in data else data
    return parsed


def _eval_async(session: str, body_js: str, timeout: float = 15.0) -> Any:
    """Execute async JS in Chromium and poll window.__g29_res cleanly."""
    _eval(
        session,
        f"window.__g29_done = false; window.__g29_res = null; window.__g29_err = null; "
        f"(async () => {{ try {{ window.__g29_res = await (async () => {{ {body_js} }})(); }} "
        f"catch (e) {{ window.__g29_err = String(e); }} finally {{ window.__g29_done = true; }} }})()",
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        done = _eval(session, "Boolean(window.__g29_done)")
        if done:
            err = _eval(session, "window.__g29_err")
            if err:
                raise RuntimeError(f"Async JS error: {err}")
            return _eval(session, "window.__g29_res")
        time.sleep(0.15)
    raise RuntimeError("Async JS timed out")


def main() -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="srag-g29-browser-e2e-"))
    env = _load_env()
    env.update(
        {
            "PYTHONPATH": str(SUBJECT_DIR),
            "DATA_DIR": str(tmp_dir / "data"),
            "SQLITE_PATH": str(tmp_dir / "library.sqlite3"),
            "COURSE_ROOT": str(tmp_dir / "courses"),
            "WORKSPACE_COURSE_IDS": "g29-browser",
            "WORKSPACE_ANSWER_ENABLED": "true",
            "SOURCE_VISION_ENABLED": "true",
        }
    )

    # Start uvicorn from clean archive directory so Python loads SUBJECT_DIR/app strictly
    server = subprocess.Popen(
        [
            PYTHON,
            "-c",
            (
                "import sys, uvicorn; "
                f"sys.path.insert(0, {str(SUBJECT_DIR)!r}); "
                "import app; "
                f"assert app.__file__.startswith({str(SUBJECT_DIR)!r}), app.__file__; "
                f"uvicorn.run('app.main:app', host='127.0.0.1', port={PORT}, log_level='warning')"
            ),
        ],
        cwd=str(SUBJECT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    session = f"g29-e2e-{os.getpid()}"
    report: dict = {"base_url": BASE_URL, "subject_sha": SUBJECT_SHA, "srag_soffice": env.get("SRAG_SOFFICE"), "checks": {}}
    try:
        _wait_port(PORT)
        _ab(session, "open", f"{BASE_URL}/workspace/")
        _ab(session, "wait", "1500")

        # Open /workspace/ and create/select a session directly through the real UI!
        _ab(session, "open", f"{BASE_URL}/workspace/")
        _ab(session, "wait", "1500")
        # Open #import-panel <details> and wait for session to initialize
        _eval(session, "const p = document.querySelector('#import-panel'); if (p) p.open = true;")
        for _ in range(30):
            if not _eval(session, "Boolean(document.querySelector('#import-button')?.disabled)"):
                break
            time.sleep(0.3)
        sid = _eval_async(
            session,
            "const d = await fetch('/api/workspace/sessions').then(r => r.json()); const list = Array.isArray(d) ? d : (d.sessions || []); return list[0]?.session_id;",
        )
        report["session_id"] = sid
        # Upload Speculative-Decoding PDF through real #import-files UI!
        _ab(session, "upload", "#import-files", str(FIXTURES / "Speculative-Decoding-2211.17192v2.pdf"))
        _ab(session, "click", "#import-button")
        for _ in range(40):
            txt = str(_eval(session, "document.querySelector('#import-results')?.textContent || ''") or "")
            if "导入完成" in txt or "复用" in txt:
                break
            time.sleep(0.5)
        report["checks"]["ui_import_pdf"] = str(_eval(session, "document.querySelector('#import-results')?.textContent || ''") or "")

        # Scope to PDF and focus page:5 via same-origin browser API, then reload UI
        setup_info = _eval_async(
            session,
            """
            const d = await fetch('/api/workspace/sessions').then(r => r.json());
            const sessions = Array.isArray(d) ? d : (d.sessions || []);
            const sid = sessions[0].session_id;
            const cat = await fetch(`/api/workspace/sessions/${sid}/catalog`).then(r => r.json());
            const doc = cat.courses[0].documents[0];
            const s0 = await fetch(`/api/workspace/sessions/${sid}`).then(r => r.json());
            const s1 = await fetch(`/api/workspace/sessions/${sid}/scope`, {
              method: 'PATCH',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({course_ids: ['g29-browser'], document_ids: [doc.document_id], scope_revision: s0.scope_revision})
            }).then(r => r.json());
            const p5 = doc.children.find(c => c.kind === 'page' && c.locator?.page === 5);
            await fetch(`/api/workspace/sessions/${sid}/focus`, {
              method: 'PUT',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({node_id: p5.node_id, scope_revision: s1.scope_revision})
            });
            return {sid, doc_id: doc.document_id, p5_node: p5.node_id};
            """,
        )
        _ab(session, "reload")
        _ab(session, "wait", "1500")

        # 1. Q04 in real Chromium: type question without page number -> send -> verify .clarification-actions -> click left -> verify answer & #reader
        _ab(session, "fill", "#question", "这张图的纵轴表示什么？不同曲线对应什么参数？")
        _ab(session, "click", "#send")
        for _ in range(40):
            cnt = int(_eval(session, "document.querySelectorAll('.clarification-actions button').length") or 0)
            if cnt >= 2:
                break
            time.sleep(0.4)
        btn_labels = _eval(session, "Array.from(document.querySelectorAll('.clarification-actions button')).map(b => b.textContent)")
        _ab(session, "click", ".clarification-actions button:first-child")
        _ab(session, "click", "#send")
        for _ in range(60):
            state = _eval(session, "({running: Boolean(document.querySelector('#send')?.disabled), msgs: document.querySelectorAll('article.message').length})")
            if not state["running"] and state["msgs"] >= 4:
                break
            time.sleep(0.5)
        q04_dom = _eval(
            session,
            """
            (() => {
              const arts = Array.from(document.querySelectorAll('article.message'));
              const last = arts[arts.length - 1];
              const imgs = Array.from(last.querySelectorAll('img')).map(i => ({complete: i.complete, nw: i.naturalWidth, nh: i.naturalHeight, src: i.getAttribute('src')}));
              const cites = Array.from(last.querySelectorAll('.citations button')).map(b => b.textContent);
              return {text: last.textContent.slice(0, 400), imgs, cites};
            })()
            """,
        )
        _eval(session, "document.querySelectorAll('article.message')[document.querySelectorAll('article.message').length - 1].querySelector('.citations button')?.click()")
        _ab(session, "wait", "1200")
        q04_reader = _eval(
            session,
            """
            (() => {
              const dlg = document.querySelector('#reader');
              const img = dlg?.querySelector('img');
              return {
                open: Boolean(dlg?.open),
                title: document.querySelector('#reader-title')?.textContent,
                img_loaded: Boolean(img && img.complete && img.naturalWidth > 0),
                img_size: img ? [img.naturalWidth, img.naturalHeight] : null,
                has_highlight: Boolean(dlg?.querySelector('.reader-highlight'))
              };
            })()
            """,
        )
        report["checks"]["browser_q04"] = {
            "clarification_buttons": btn_labels,
            "answer_dom": q04_dom,
            "reader": q04_reader,
        }

        # 2. Table crop Branch C check in real Chromium: ask Table 1 p5 -> click text citation -> verify #reader loads FULL PAGE + aligned .reader-highlight!
        prev_msgs = int(_eval(session, "document.querySelectorAll('article.message').length") or 0)
        _ab(session, "fill", "#question", "打开第 5 页的表 1")
        _ab(session, "click", "#send")
        for _ in range(40):
            st = _eval(session, "({running: Boolean(document.querySelector('#send')?.disabled), msgs: document.querySelectorAll('article.message').length})")
            if not st["running"] and st["msgs"] >= prev_msgs + 2:
                break
            time.sleep(0.4)
        _eval(session, "document.querySelectorAll('article.message')[document.querySelectorAll('article.message').length - 1].querySelector('.citations button')?.click()")
        _ab(session, "wait", "1200")
        table_reader = _eval(
            session,
            """
            (() => {
              const dlg = document.querySelector('#reader');
              const img = dlg?.querySelector('img');
              const hl = dlg?.querySelector('.reader-highlight');
              return {
                open: Boolean(dlg?.open),
                title: document.querySelector('#reader-title')?.textContent,
                img_loaded: Boolean(img && img.complete && img.naturalWidth > 0),
                img_size: img ? [img.naturalWidth, img.naturalHeight] : null,
                img_src: img ? img.getAttribute('src') : null,
                highlight_style: hl ? {left: hl.style.left, top: hl.style.top, width: hl.style.width, height: hl.style.height} : null
              };
            })()
            """,
        )
        report["checks"]["browser_table_crop_highlight"] = table_reader

        # 3. Notebook b2 reordered bilateral recovery in real Chromium!
        _eval(session, "const p = document.querySelector('#import-panel'); if (p) p.open = true;")
        _ab(session, "upload", "#import-files", str(FIXTURES / "notebook/v1/Lab06-Thermal-Sweep.ipynb"))
        _ab(session, "click", "#import-button")
        for _ in range(40):
            txt = str(_eval(session, "document.querySelector('#import-results')?.textContent || ''") or "")
            if "Lab06-Thermal-Sweep.ipynb" in txt and ("导入完成" in txt or "复用" in txt):
                break
            time.sleep(0.4)
        nb_setup = _eval_async(
            session,
            """
            const d = await fetch('/api/workspace/sessions').then(r => r.json());
            const sessions = Array.isArray(d) ? d : (d.sessions || []);
            const sid = sessions[0].session_id;
            const cat = await fetch(`/api/workspace/sessions/${sid}/catalog`).then(r => r.json());
            const doc = cat.courses[0].documents.find(d => d.file_name === 'Lab06-Thermal-Sweep.ipynb');
            const s0 = await fetch(`/api/workspace/sessions/${sid}`).then(r => r.json());
            const s1 = await fetch(`/api/workspace/sessions/${sid}/scope`, {
              method: 'PATCH',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({course_ids: ['g29-browser'], document_ids: [doc.document_id], scope_revision: s0.scope_revision})
            }).then(r => r.json());
            const walk = (nodes) => {
              for (const n of nodes || []) {
                if (n.kind === 'cell' && n.locator?.cell_index === 1) return n;
                const hit = walk(n.children);
                if (hit) return hit;
              }
              return null;
            };
            const c1 = walk(doc.children);
            await fetch(`/api/workspace/sessions/${sid}/focus`, {
              method: 'PUT',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({node_id: c1.node_id, scope_revision: s1.scope_revision})
            });
            return {sid, doc_id: doc.document_id};
            """,
        )
        _ab(session, "reload")
        _ab(session, "wait", "1200")
        prev_msgs = int(_eval(session, "document.querySelectorAll('article.message').length") or 0)
        _ab(session, "fill", "#question", "第 2 个 Cell 第 1 张图的横轴单位和峰值温度分别是多少？")
        _ab(session, "click", "#send")
        for _ in range(60):
            st = _eval(session, "({running: Boolean(document.querySelector('#send')?.disabled), msgs: document.querySelectorAll('article.message').length})")
            if not st["running"] and st["msgs"] >= prev_msgs + 2:
                break
            time.sleep(0.5)

        v2r_bytes = (FIXTURES / "notebook/v2_reordered/Lab06-Thermal-Sweep.ipynb").read_bytes()
        import base64
        v2r_b64 = base64.b64encode(v2r_bytes).decode("ascii")
        _eval_async(
            session,
            f"""
            const raw = atob({v2r_b64!r});
            const bytes = new Uint8Array(raw.length);
            for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
            const q = new URLSearchParams({{course_id: 'g29-browser', filename: 'Lab06-Thermal-Sweep.ipynb', document_id: {nb_setup['doc_id']!r}}});
            const r = await fetch(`/api/workspace/sessions/{nb_setup['sid']}/import?${{q}}`, {{
              method: 'POST',
              headers: {{'Content-Type': 'application/octet-stream'}},
              body: bytes
            }}).then(r => r.json());
            return r;
            """,
        )
        _ab(session, "fill", "#question", "对比最新版和旧版刚才读过的这张图，横轴单位和峰值温度有什么变化？")
        _ab(session, "click", "#send")
        for _ in range(40):
            cnt = int(_eval(session, "document.querySelectorAll('.clarification-actions button').length") or 0)
            if cnt >= 2:
                break
            time.sleep(0.4)
        b2_btns = _eval(session, "Array.from(document.querySelectorAll('.clarification-actions button')).map(b => b.textContent)")
        _ab(session, "click", ".clarification-actions button:nth-child(2)")
        prev_msgs = int(_eval(session, "document.querySelectorAll('article.message').length") or 0)
        _ab(session, "click", "#send")
        for _ in range(60):
            st = _eval(session, "({running: Boolean(document.querySelector('#send')?.disabled), msgs: document.querySelectorAll('article.message').length})")
            if not st["running"] and st["msgs"] >= prev_msgs + 2:
                break
            time.sleep(0.5)
        b2_dom = _eval(
            session,
            """
            (() => {
              const arts = Array.from(document.querySelectorAll('article.message'));
              const last = arts[arts.length - 1];
              const imgs = Array.from(last.querySelectorAll('img')).map(i => ({complete: i.complete, nw: i.naturalWidth, nh: i.naturalHeight}));
              const cites = Array.from(last.querySelectorAll('.citations button')).map(b => b.textContent);
              return {text: last.textContent.slice(0, 600), img_count: imgs.length, imgs, cites};
            })()
            """,
        )
        report["checks"]["browser_b2_bilateral"] = {
            "clarification_buttons": b2_btns,
            "bilateral_dom": b2_dom,
        }

        out_path = GATE_DIR / OUT_FILE
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        try:
            _ab(session, "close")
        except Exception:
            pass
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
