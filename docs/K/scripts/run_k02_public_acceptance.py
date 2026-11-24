#!/usr/bin/env python3
"""K02 Public Release Acceptance Runner (C1, C2, C3, C5-Public, C6, C7).

Runs from clean empty DATA_DIR/SQLITE_PATH directories using real FastAPI/Uvicorn,
real Chromium via agent-browser, real LibreOffice (SRAG_SOFFICE), real DeepSeek
(deepseek-flash), and real DashScope (qwen3-vl-plus + S3 embeddings).

Writes desensitized machine evidence to docs/K/K02-公开样例实测.json.
Never hardcodes personal absolute paths or secrets.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = REPO_ROOT / "examples" / "demo-course"
OUT_JSON = REPO_ROOT / "docs" / "K" / "K02-公开样例实测.json"
PYTHON = sys.executable


def _resolve_soffice() -> str:
    env_val = os.environ.get("SRAG_SOFFICE", "").strip()
    if env_val:
        return env_val
    for cand in (
        shutil.which("soffice"),
        "/opt/homebrew/bin/soffice",
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    ):
        if cand and Path(cand).is_file():
            return str(cand)
    return ""


def _resolve_agent_browser() -> str:
    env_val = os.environ.get("AGENT_BROWSER", "").strip()
    if env_val:
        return env_val
    found = shutil.which("agent-browser")
    if found:
        return found
    local_bin = Path.home() / ".local" / "bin" / "agent-browser"
    if local_bin.is_file():
        return str(local_bin)
    raise RuntimeError("agent-browser CLI not found")


def _load_env_secrets() -> tuple[dict[str, str], list[str]]:
    env: dict[str, str] = {}
    secrets: list[str] = []
    candidates = []
    if explicit := os.environ.get("SRAG_ENV_FILE", "").strip():
        candidates.append(Path(explicit))
    candidates.extend([REPO_ROOT / ".env", REPO_ROOT.parent / "SRAG" / ".env"])
    if len(REPO_ROOT.parents) > 1:
        candidates.append(REPO_ROOT.parents[1] / ".env")
    for path in candidates:
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k not in env and v:
                env[k] = v
                if any(tok in k.upper() for tok in ("KEY", "TOKEN", "SECRET", "PASSWORD")) and len(v) >= 6:
                    secrets.append(v)
    for k, v in os.environ.items():
        if v and any(tok in k.upper() for tok in ("KEY", "TOKEN", "SECRET", "PASSWORD")) and len(v) >= 6:
            secrets.append(v)
    return env, list(dict.fromkeys(secrets))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_port(port: int, timeout: float = 25.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise RuntimeError(f"Server failed to bind port {port}")


def _http_json(url: str, *, method: str = "GET", body: Any = None) -> tuple[int, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "ignore")
        try:
            payload = json.loads(raw)
        except Exception:
            payload = raw
        return exc.code, payload


class BrowserSession:
    def __init__(self, ab_bin: str, session_name: str) -> None:
        self.ab_bin = ab_bin
        self.name = session_name

    def cmd(self, *args: str, timeout: int = 90) -> str:
        res = subprocess.run(
            [self.ab_bin, "--session", self.name, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"agent-browser ({self.name}) failed: {' '.join(args)}\n{res.stdout}\n{res.stderr}"
            )
        return res.stdout.strip()

    def eval(self, js: str) -> Any:
        raw = self.cmd("eval", "--json", js)
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "data" in parsed:
            data = parsed["data"]
            return data.get("result") if isinstance(data, dict) and "result" in data else data
        return parsed

    def eval_async(self, body_js: str, timeout: float = 40.0) -> Any:
        self.eval(
            "window.__k02_done=false; window.__k02_res=null; window.__k02_err=null;"
            f"(async()=>{{try{{window.__k02_res=await (async()=>{{{body_js}}})();}}"
            "catch(e){window.__k02_err=String(e);}finally{window.__k02_done=true;}})()"
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.eval("Boolean(window.__k02_done)"):
                err = self.eval("window.__k02_err")
                if err:
                    raise RuntimeError(f"Async JS error: {err}")
                return self.eval("window.__k02_res")
            time.sleep(0.15)
        raise TimeoutError("Async JS timed out")

    def open_workspace(self, base_url: str) -> None:
        self.cmd("open", f"{base_url}/workspace/")
        self.wait_ready()

    def wait_ready(self, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ok = self.eval(
                "Boolean(document.querySelector('#import-button') && !document.querySelector('#import-button').disabled && !document.querySelector('#question').disabled)"
            )
            if ok:
                return
            time.sleep(0.25)
        raise TimeoutError("Workspace UI failed to initialize")

    def wait_nav_idle(self, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ok = self.eval(
                "!document.querySelector('#clear-file')?.disabled && !document.querySelector('#question')?.disabled"
            )
            if ok:
                return
            time.sleep(0.2)

    def session_id(self) -> str:
        return str(
            self.eval_async(
                "const d = await fetch('/api/workspace/sessions', {credentials:'same-origin'}).then(r=>r.json()); return (d.sessions || [])[0]?.session_id;"
            )
        )

    def fetch_snapshot(self, sid: str) -> dict[str, Any]:
        return self.eval_async(
            f"return await fetch('/api/workspace/sessions/' + encodeURIComponent({sid!r}), {{credentials:'same-origin', cache:'no-store'}}).then(r=>r.json());"
        )

    def msg_count(self) -> int:
        return int(self.eval("document.querySelectorAll('article.message').length") or 0)

    def import_files(
        self,
        paths: list[Path],
        *,
        course_id: str | None = None,
        replace: bool = False,
        timeout: float = 90.0,
    ) -> list[str]:
        self.eval("(() => { const p = document.querySelector('#import-panel'); if (p) p.open = true; })()")
        if course_id:
            self.eval(
                f"(() => {{ const s = document.querySelector('#import-course'); if (s) {{ s.value = {course_id!r}; s.dispatchEvent(new Event('change')); }} }})()"
            )
        if replace:
            self.eval(
                "(() => { const cb = document.querySelector('#replace-document'); if (cb) { cb.checked = true; cb.dispatchEvent(new Event('change')); } })()"
            )
        self.cmd("upload", "#import-files", *[str(p) for p in paths])
        self.eval("document.querySelector('#import-button')?.scrollIntoView({block:'center'})")
        self.cmd("click", "#import-button")
        deadline = time.monotonic() + timeout
        items: list[str] = []
        while time.monotonic() < deadline:
            items = self.eval(
                "Array.from(document.querySelectorAll('#import-results li')).map(li => li.textContent)"
            )
            if len(items) == len(paths) and all("导入中" not in x for x in items):
                break
            time.sleep(0.4)
        self.wait_nav_idle()
        time.sleep(0.4)
        return items

    def click_catalog_row(self, matcher_js: str) -> str:
        res = self.eval(
            f"""
            (() => {{
              const rows = Array.from(document.querySelectorAll('#catalog .catalog-row'));
              const target = ({matcher_js})(rows);
              if (!target) return {{found: false, available: rows.map(r => r.textContent)}};
              target.click();
              return {{found: true, text: target.textContent}};
            }})()
            """
        )
        if not res or not res.get("found"):
            raise RuntimeError(f"Catalog row not found: {res}")
        self.wait_nav_idle()
        time.sleep(0.3)
        return str(res["text"])

    def clear_file_scope(self, course_id: str = "demo-course") -> None:
        """Reset document filter to whole-course scope without collapsing the course folder."""
        clicked_clear = self.eval(
            """
            (() => {
              const cf = document.querySelector('#clear-file');
              if (cf && !cf.hidden) { cf.click(); return true; }
              return false;
            })()
            """
        )
        if clicked_clear:
            self.wait_nav_idle()
            time.sleep(0.25)
            return
        res = self.eval(
            f"""
            (() => {{
              const rows = Array.from(document.querySelectorAll('#catalog .catalog-row'));
              const c = rows.find(r => r.title === '课程' && r.textContent.includes({course_id!r}));
              if (!c) return false;
              const isSelected = c.getAttribute('aria-current') === 'location';
              const isOpen = c.getAttribute('aria-expanded') === 'true';
              if (!isSelected || !isOpen) c.click();
              return true;
            }})()
            """
        )
        if not res:
            raise RuntimeError(f"Course row not found: {course_id!r}")
        self.wait_nav_idle()
        time.sleep(0.25)

    def click_doc_child(self, doc_substr: str, *child_substrs: str) -> str:
        """Click a document row by substring, auto-expanding collapsed courses and keeping only its own subtree targeted."""
        need_expand = self.eval(
            f"!Array.from(document.querySelectorAll('#catalog .catalog-row')).some(r => r.title === '资料' && r.textContent.includes({doc_substr!r}))"
        )
        if need_expand:
            self.eval(
                """
                (() => {
                  const collapsed = Array.from(document.querySelectorAll('#catalog .catalog-row'))
                    .find(r => r.title === '课程' && r.getAttribute('aria-expanded') === 'false');
                  if (collapsed) collapsed.click();
                })()
                """
            )
            self.wait_nav_idle()
            time.sleep(0.25)
        res = self.eval(
            f"""
            (() => {{
              const rows = Array.from(document.querySelectorAll('#catalog .catalog-row'));
              const docBtn = rows.find(r => r.title === '资料' && r.textContent.includes({doc_substr!r}))
                || rows.find(r => r.textContent.includes({doc_substr!r}));
              if (!docBtn) return {{found: false, available: rows.map(r => r.textContent)}};
              const needClick = docBtn.getAttribute('aria-current') !== 'location' || docBtn.getAttribute('aria-expanded') !== 'true';
              if (needClick) docBtn.click();
              return {{found: true, clicked: needClick, text: docBtn.textContent}};
            }})()
            """
        )
        if not res or not res.get("found"):
            raise RuntimeError(f"Document row not found ({doc_substr!r}): {res}")
        self.wait_nav_idle()
        time.sleep(0.25)
        last_text = str(res["text"])
        for child_sub in child_substrs:
            cres = self.eval(
                f"""
                (() => {{
                  const rows = Array.from(document.querySelectorAll('#catalog .catalog-row'));
                  const docBtn = rows.find(r => r.title === '资料' && r.textContent.includes({doc_substr!r}))
                    || rows.find(r => r.textContent.includes({doc_substr!r}));
                  const docLi = docBtn?.closest('li');
                  if (!docLi) return {{found: false, reason: 'docLi missing'}};
                  const subRows = Array.from(docLi.querySelectorAll('.catalog-row'));
                  const target = subRows.find(r => r !== docBtn && r.textContent.includes({child_sub!r}));
                  if (!target) return {{found: false, available: subRows.map(r => r.textContent)}};
                  const isExp = target.getAttribute('aria-expanded');
                  if (isExp === 'false' || target.getAttribute('aria-current') !== 'location') {{
                    target.click();
                  }}
                  return {{found: true, text: target.textContent}};
                }})()
                """
            )
            if not cres or not cres.get("found"):
                raise RuntimeError(f"Doc child {child_sub!r} under {doc_substr!r} not found: {cres}")
            self.wait_nav_idle()
            time.sleep(0.25)
            last_text = str(cres["text"])
        return last_text

    def ask_ui(
        self,
        question: str,
        *,
        expect_clarification: bool = False,
        allow_failed: bool = False,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        prev_msgs = self.msg_count()
        t0 = time.monotonic()
        self.eval("document.querySelector('#question')?.scrollIntoView({block:'center'})")
        self.cmd("fill", "#question", question)
        self.eval("document.querySelector('#send')?.scrollIntoView({block:'center'})")
        self.cmd("click", "#send")
        deadline = time.monotonic() + timeout
        st: dict[str, Any] = {}
        while time.monotonic() < deadline:
            st = self.eval(
                """
                (() => {
                  const msgs = document.querySelectorAll('article.message').length;
                  const clarBtns = Array.from(document.querySelectorAll('.clarification-actions button')).map(b => b.textContent);
                  const statusTxt = document.querySelector('#turn-status')?.textContent || '';
                  const composerTxt = document.querySelector('#composer-status')?.textContent || '';
                  const actionsTxt = document.querySelector('#turn-actions')?.textContent || '';
                  const wsErr = document.querySelector('#workspace-status')?.textContent || '';
                  const running = composerTxt.includes('正在处理') || actionsTxt.includes('取消本轮') || statusTxt.includes('正在处理');
                  return {running, msgs, clarBtns, statusTxt, composerTxt, actionsTxt, wsErr};
                })()
                """
            )
            if expect_clarification:
                if not st["running"] and len(st["clarBtns"]) >= 2 and not st["wsErr"]:
                    return {"elapsed_s": round(time.monotonic() - t0, 3), "state": st}
                if not st["running"] and ("处理失败" in st["statusTxt"] or st["wsErr"]):
                    if allow_failed:
                        return {"elapsed_s": round(time.monotonic() - t0, 3), "state": st}
                    raise RuntimeError(f"Turn failed while expecting clarification: {st}")
            else:
                if not st["running"] and (
                    st["msgs"] >= prev_msgs + 2
                    or (allow_failed and ("处理失败" in st["statusTxt"] or bool(st["wsErr"])))
                ):
                    return {"elapsed_s": round(time.monotonic() - t0, 3), "state": st}
                if not st["running"] and ("处理失败" in st["statusTxt"] or st["wsErr"]):
                    raise RuntimeError(f"Turn failed for {question!r}: {st}")
            time.sleep(0.35)
        raise TimeoutError(f"Timed out waiting for turn {question!r}: {st}")

    def click_clarification_and_send(self, matcher_js: str, timeout: float = 120.0) -> dict[str, Any]:
        prev_msgs = self.msg_count()
        t0 = time.monotonic()
        chosen = self.eval(
            f"""
            (() => {{
              const btns = Array.from(document.querySelectorAll('.clarification-actions button'));
              const target = ({matcher_js})(btns);
              if (!target) return null;
              target.click();
              return target.textContent;
            }})()
            """
        )
        if not chosen:
            raise RuntimeError("Clarification option not found")
        self.eval("document.querySelector('#send')?.scrollIntoView({block:'center'})")
        self.cmd("click", "#send")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            st = self.eval(
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
                """
            )
            if not st["running"] and st["msgs"] >= prev_msgs + 2:
                return {"elapsed_s": round(time.monotonic() - t0, 3), "chosen": chosen, "state": st}
            if not st["running"] and ("处理失败" in st["statusTxt"] or st["wsErr"]):
                raise RuntimeError(f"Clarification resume failed: {st}")
            time.sleep(0.35)
        raise TimeoutError("Clarification resume timed out")

    def read_last_message(self) -> dict[str, Any]:
        return self.eval(
            """
            (() => {
              const arts = Array.from(document.querySelectorAll('article.message'));
              const last = arts[arts.length - 1];
              if (!last) return null;
              const textBlocks = Array.from(last.querySelectorAll('.text-block, .notice-block')).map(n => n.textContent);
              const cites = Array.from(last.querySelectorAll('.citations button')).map((b, idx) => ({
                index: idx,
                label: b.textContent,
                evidence_id: b.dataset.evidenceId || null,
              }));
              const cards = Array.from(last.querySelectorAll('.source-card')).map((c, idx) => ({
                index: idx,
                display_id: c.dataset.displayId,
                caption: c.querySelector('.source-caption')?.textContent || '',
                details: c.querySelector('.source-details')?.textContent || '',
                actions: Array.from(c.querySelectorAll('.source-actions button')).map(b => b.textContent),
                img: (() => {
                  const i = c.querySelector('img');
                  return i ? {complete: i.complete, nw: i.naturalWidth, nh: i.naturalHeight, src: i.getAttribute('src')} : null;
                })(),
              }));
              const kgChips = Array.from(last.querySelectorAll('.kg-concept-chip')).map(b => ({
                text: b.textContent,
                concept_key: b.dataset.conceptKey || null,
                stale: b.dataset.stale === 'true',
              }));
              return {
                message_id: last.dataset.messageId,
                text: textBlocks.join('\\n'),
                cites,
                cards,
                kgChips,
              };
            })()
            """
        )

    def wait_last_images(self, count: int = 1, timeout: float = 20.0) -> list[dict[str, Any]]:
        deadline = time.monotonic() + timeout
        imgs: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            imgs = self.eval(
                """
                (() => {
                  const arts = Array.from(document.querySelectorAll('article.message'));
                  const last = arts[arts.length - 1];
                  if (!last) return [];
                  return Array.from(last.querySelectorAll('.source-card img')).map(i => ({
                    complete: i.complete, nw: i.naturalWidth, nh: i.naturalHeight, src: i.getAttribute('src')
                  }));
                })()
                """
            )
            if len(imgs) >= count and all(i["complete"] and i["nw"] > 0 for i in imgs):
                return imgs
            time.sleep(0.25)
        raise TimeoutError(f"Expected {count} loaded images, got {imgs}")

    def open_reader_via(self, js_target_fn: str, *, expect_img: bool = True) -> dict[str, Any]:
        ok = self.eval(
            f"(() => {{ const el = ({js_target_fn})(); if (!el) return false; el.click(); return true; }})()"
        )
        if not ok:
            raise RuntimeError("Target element for #reader not found")
        deadline = time.monotonic() + 15.0
        st: dict[str, Any] = {}
        while time.monotonic() < deadline:
            st = self.eval(
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
                    highlight_style: hl ? {left: hl.style.left, top: hl.style.top, width: hl.style.width, height: hl.style.height} : null,
                    excerpt_len: excerpt.length,
                    body_text: document.querySelector('#reader-body')?.textContent?.slice(0, 220) || ''
                  };
                })()
                """
            )
            if st["open"] and (
                not expect_img or st["img_loaded"] or (not st["img_present"] and len(st["body_text"]) > 0)
            ):
                return st
            time.sleep(0.25)
        raise TimeoutError(f"Reader did not settle: {st}")

    def close_reader(self) -> None:
        if self.eval("Boolean(document.querySelector('#reader')?.open)"):
            self.eval("document.querySelector('#close-reader')?.scrollIntoView({block:'center'})")
            self.cmd("click", "#close-reader")
            time.sleep(0.2)

    def fetch_urls(self, urls: list[str]) -> list[dict[str, Any]]:
        return self.eval_async(
            f"""
            const urls = {json.dumps(urls)};
            const out = [];
            for (const u of urls) {{
              const r = await fetch(u, {{credentials:'same-origin', cache:'no-store'}});
              const buf = new Uint8Array(await r.arrayBuffer());
              let sha256 = null;
              if (r.ok && buf.length > 0 && crypto?.subtle) {{
                const d = await crypto.subtle.digest('SHA-256', buf);
                sha256 = Array.from(new Uint8Array(d)).map(b=>b.toString(16).padStart(2,'0')).join('');
              }}
              out.push({{url: u, status: r.status, bytes: buf.length, sha256}});
            }}
            return out;
            """
        )

    def close(self) -> None:
        try:
            self.cmd("close", timeout=15)
        except Exception:
            pass


def _launch_server(env_overrides: dict[str, str], port: int) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env.update(env_overrides)
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.Popen(
        [
            PYTHON,
            "-c",
            (
                "import uvicorn; "
                f"uvicorn.run('app.main:app', host='127.0.0.1', port={port}, log_level='warning')"
            ),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _wait_port(port)
    return proc


def _stop_server(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()


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


def _check_evidence_records(snap: dict[str, Any], message_id: str) -> dict[str, Any]:
    msg = next(m for m in snap["messages"] if m["message_id"] == message_id)
    turn = next((t for t in snap["turns"] if t["turn_id"] == msg["turn_id"]), {})
    sources = msg.get("sources") or {}
    reads_summary = []
    for eid, ev in sources.items():
        for r in ev.get("reads", []):
            reads_summary.append(
                {
                    "evidence_id": eid,
                    "method": r.get("method"),
                    "status": r.get("status"),
                    "origin_execution": r.get("origin_execution"),
                    "execution": r.get("execution"),
                    "actual_source": r.get("actual_source"),
                    "actual_input_sha256": r.get("actual_input_sha256"),
                }
            )
    blocks = msg.get("blocks") or []
    displays = [
        {
            "display_id": b.get("display_id"),
            "evidence_id": b.get("evidence_id"),
            "rendition_kind": b.get("rendition_kind"),
            "fallback_reason": b.get("fallback_reason"),
            "read_status": b.get("read_status"),
            "display_source": b.get("display_source"),
        }
        for b in blocks
        if b.get("type") == "source_image"
    ]
    notices = [{"code": b.get("code"), "text": b.get("text")} for b in blocks if b.get("type") == "notice"]
    return {
        "turn_status": turn.get("status"),
        "error": turn.get("error"),
        "retryable": turn.get("retryable"),
        "trace": turn.get("trace"),
        "reads": reads_summary,
        "displays": displays,
        "notices": notices,
        "related_concepts": msg.get("related_concepts", []),
    }


def _run_phase1_offline(ab_bin: str, soffice: str) -> dict[str, Any]:
    print("[Phase 1] C1 Offline cold start & no-key capabilities...", flush=True)
    run_dir = Path(tempfile.mkdtemp(prefix="srag-k02-p1-"))
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    server = _launch_server(
        {
            "COURSE_ROOT": str(DEMO_DIR),
            "DATA_DIR": str(run_dir / "data"),
            "SQLITE_PATH": str(run_dir / "srag.sqlite3"),
            "WORKSPACE_COURSE_IDS": "demo-course",
            "WORKSPACE_ANSWER_ENABLED": "false",
            "SOURCE_VISION_ENABLED": "false",
            "DEEPSEEK_API_KEY": "",
            "DASHSCOPE_API_KEY": "",
            "JEV_API_KEY": "",
            "QDRANT_URL": "http://127.0.0.1:1",
            "SRAG_SOFFICE": soffice,
        },
        port,
    )
    bs = BrowserSession(ab_bin, f"k02-p1-{os.getpid()}")
    try:
        health_status, health_body = _http_json(f"{base_url}/api/health")
        bs.open_workspace(base_url)
        empty_catalog = bs.eval("document.querySelector('#catalog')?.textContent || ''")
        empty_kg = bs.eval("document.querySelector('#kg-count')?.textContent || ''")
        ws_err_empty = bs.eval("document.querySelector('#workspace-status')?.hidden")

        imports = bs.import_files(
            [
                DEMO_DIR / "Axis-LoRA-2611.07820v1.pdf",
                DEMO_DIR / "FlashAttention-3-2407.08608v2.pdf",
                DEMO_DIR / "Lecture-01-Retrieval-and-Attention.pptx",
                DEMO_DIR / "v1" / "Lab-01-Retrieval-Bench.ipynb",
            ]
        )
        # Catalog tree & neighbor navigation
        bs.click_doc_child("Lecture-01-Retrieval-and-Attention.pptx", "第 2 页")
        loc_p2 = bs.eval("document.querySelector('#location-title')?.textContent || ''")
        bs.cmd("click", "#next-location")
        bs.wait_nav_idle()
        loc_p3 = bs.eval("document.querySelector('#location-title')?.textContent || ''")
        bs.cmd("click", "#previous-location")
        bs.wait_nav_idle()
        reader_cat = bs.open_reader_via("() => document.querySelector('#preview-location')", expect_img=True)
        bs.close_reader()

        # H KG offline dual-entry navigation
        bs.eval("(() => { const p = document.querySelector('#kg-panel'); if (p) p.open = true; })()")
        kg_count = int(bs.eval("document.querySelector('#kg-count')?.textContent || '0'"))
        bs.eval(
            "(() => { const b = Array.from(document.querySelectorAll('#kg-list .kg-item')).find(x => x.textContent.includes('轴向门控路由')); if (b) b.click(); })()"
        )
        time.sleep(0.4)
        kg_detail_html = bs.eval("document.querySelector('#kg-detail')?.textContent || ''")
        reader_kg_def = bs.open_reader_via(
            "() => document.querySelector('#kg-detail .kg-card-actions button')", expect_img=True
        )
        bs.close_reader()
        reader_kg_out = bs.open_reader_via(
            "() => Array.from(document.querySelectorAll('#kg-detail .kg-child-item button')).find(b => b.textContent.includes('输出图'))",
            expect_img=True,
        )
        bs.close_reader()

        # No-experiment concept fallback in KG
        bs.eval(
            "(() => { const b = Array.from(document.querySelectorAll('#kg-list .kg-item')).find(x => x.textContent.includes('单向边界守卫')); if (b) b.click(); })()"
        )
        time.sleep(0.4)
        no_edge_notice = bs.eval("document.querySelector('#kg-detail .kg-notice')?.textContent || ''")

        # Offline excerpt fallback ask (0 external calls)
        bs.clear_file_scope("demo-course")
        ask_res = bs.ask_ui("什么是 Axis-Gated Routing 轴向门控路由？")
        msg = bs.read_last_message()
        sid = bs.session_id()
        snap = bs.fetch_snapshot(sid)
        ev_info = _check_evidence_records(snap, msg["message_id"])

        passed = (
            health_status == 200
            and health_body.get("status") == "ok"
            and "暂无资料" in empty_catalog
            and empty_kg == "0"
            and bool(ws_err_empty)
            and len(imports) == 4
            and all("导入完成" in x for x in imports)
            and reader_cat["img_loaded"]
            and kg_count >= 3
            and "probe_axis_gate" in kg_detail_html
            and reader_kg_def["img_loaded"]
            and reader_kg_out["img_loaded"]
            and "暂无可靠" in no_edge_notice
            and any(n["code"] == "excerpt_fallback" for n in ev_info["notices"])
            and (ev_info["trace"] or {}).get("external_call_count", 0) == 0
        )
        return {
            "passed": passed,
            "health": health_body,
            "empty_state_ok": "暂无资料" in empty_catalog and empty_kg == "0",
            "imports": imports,
            "catalog_nav": {"p2": loc_p2, "p3": loc_p3, "reader_img": reader_cat["img_size"]},
            "kg_offline": {
                "kg_count": kg_count,
                "def_reader_title": reader_kg_def["title"],
                "out_reader_title": reader_kg_out["title"],
                "no_edge_notice": no_edge_notice,
            },
            "offline_ask": {
                "elapsed_s": ask_res["elapsed_s"],
                "notices": ev_info["notices"],
                "external_call_count": (ev_info["trace"] or {}).get("external_call_count", 0),
            },
        }
    finally:
        bs.close()
        _stop_server(server)
        shutil.rmtree(run_dir, ignore_errors=True)


def _run_phase2_online_main(
    ab_bin: str, soffice: str, loaded_env: dict[str, str]
) -> dict[str, Any]:
    print("[Phase 2] C1-online + C2 + C3 + C5-Public + C6(KG)...", flush=True)
    run_dir = Path(tempfile.mkdtemp(prefix="srag-k02-p2-"))
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    server = _launch_server(
        {
            **loaded_env,
            "COURSE_ROOT": str(DEMO_DIR),
            "DATA_DIR": str(run_dir / "data"),
            "SQLITE_PATH": str(run_dir / "srag.sqlite3"),
            "WORKSPACE_COURSE_IDS": "demo-course,demo-distractor",
            "WORKSPACE_ANSWER_ENABLED": "true",
            "SOURCE_VISION_ENABLED": "true",
            "WORKSPACE_RETRIEVAL_STRATEGY": "s0",
            "WORKSPACE_RETRIEVAL_BUDGET": "6",
            "WORKSPACE_RETRIEVAL_POOL_CAP": "12",
            "WORKSPACE_RETRIEVAL_INCLUDE_ADJACENCY": "true",
            "WORKSPACE_VECTOR_RETRIEVAL_ENABLED": "false",
            "WORKSPACE_KG_RETRIEVAL_ENABLED": "false",
            "QDRANT_URL": "http://127.0.0.1:1",
            "SRAG_SOFFICE": soffice,
        },
        port,
    )
    bs_a = BrowserSession(ab_bin, f"k02-p2a-{os.getpid()}")
    bs_b = BrowserSession(ab_bin, f"k02-p2b-{os.getpid()}")
    latencies: list[float] = []
    results: dict[str, Any] = {"c2": {}, "c3": {}, "c5_public": {}, "c6": {}, "usage": {}}

    try:
        bs_a.open_workspace(base_url)
        sid_a = bs_a.session_id()
        bs_a.import_files(
            [
                DEMO_DIR / "Axis-LoRA-2611.07820v1.pdf",
                DEMO_DIR / "FlashAttention-3-2407.08608v2.pdf",
                DEMO_DIR / "Lecture-01-Retrieval-and-Attention.pptx",
                DEMO_DIR / "v1" / "Lab-01-Retrieval-Bench.ipynb",
            ],
            course_id="demo-course",
        )
        # Select demo-course explicitly so single-course scope is active
        bs_a.clear_file_scope("demo-course")

        # -----------------------------------------------------------------
        # K-Pub-01 (C5-Public + C2-01 Whole PPTX + C6 KG chip)
        # -----------------------------------------------------------------
        print("  -> K-Pub-01 (cross-file PPTX+IPYNB + C2-01)...", flush=True)
        t_p1 = bs_a.ask_ui(
            "请说明 轴向门控路由 的定义，给出配套实验的实测 recall 结果。"
        )
        latencies.append(t_p1["elapsed_s"])
        m_p1 = bs_a.read_last_message()
        snap = bs_a.fetch_snapshot(sid_a)
        ev_p1 = _check_evidence_records(snap, m_p1["message_id"])
        # Click PPTX citation -> #reader
        r_p1_pptx = bs_a.open_reader_via(
            f"() => Array.from(document.querySelectorAll('article.message[data-message-id=\"{m_p1['message_id']}\"] .citations button')).find(b => b.textContent.includes('Lecture-01'))",
            expect_img=True,
        )
        bs_a.close_reader()
        # Click KG chip in message -> #kg-detail
        chip_clicked = bs_a.eval(
            f"""
            (() => {{
              const c = Array.from(document.querySelectorAll('article.message[data-message-id="{m_p1['message_id']}"] .kg-concept-chip')).find(x => x.textContent.includes('轴向门控路由'));
              if (!c) return false;
              c.click();
              return true;
            }})()
            """
        )
        time.sleep(0.4)
        kg_detail_p1 = bs_a.eval("document.querySelector('#kg-detail')?.textContent || ''")
        results["c2"]["C2-01_pptx_whole"] = {
            "passed": r_p1_pptx["open"] and r_p1_pptx["img_loaded"] and "Lecture-01" in r_p1_pptx["title"],
            "reader_title": r_p1_pptx["title"],
            "img_size": r_p1_pptx["img_size"],
        }
        results["c5_public"]["K-Pub-01"] = {
            "passed": (
                all(k in m_p1["text"] for k in ("0.15", "86.4"))
                and len(m_p1["cites"]) >= 2
                and (ev_p1["trace"] or {}).get("generation", {}).get("status") in {"succeeded", "partial"}
            ),
            "elapsed_s": t_p1["elapsed_s"],
            "cites": [c["label"] for c in m_p1["cites"]],
            "kg_chip_clicked": chip_clicked,
            "kg_detail_has_both": "Lecture-01" in kg_detail_p1 and "Lab-01" in kg_detail_p1,
            "generation_status": (ev_p1["trace"] or {}).get("generation", {}).get("status"),
        }

        # -----------------------------------------------------------------
        # K-Pub-02 & C2-03: 2-column PDF formula + left-column figure & highlight
        # -----------------------------------------------------------------
        print("  -> K-Pub-02 / C2-03 (2-col PDF formula + figure crop + highlight)...", flush=True)
        bs_a.click_doc_child("Axis-LoRA-2611.07820v1.pdf", "第 6 页")
        t_p2_f_clar = bs_a.ask_ui("Axis-LoRA 第 6 页的公式是什么？请说明各项含义。", expect_clarification=True)
        latencies.append(t_p2_f_clar["elapsed_s"])
        t_p2_f = bs_a.click_clarification_and_send("btns => btns.find(b => b.textContent.includes('InfoNCE'))")
        latencies.append(t_p2_f["elapsed_s"])
        m_p2_f = bs_a.read_last_message()
        r_p2_f = bs_a.open_reader_via(
            f"() => document.querySelector('article.message[data-message-id=\"{m_p2_f['message_id']}\"] .citations button')",
            expect_img=True,
        )
        bs_a.close_reader()

        # Navigate to Axis-LoRA page 3 for left-column Figure 2 crop & back-to-original highlight
        bs_a.click_doc_child("Axis-LoRA-2611.07820v1.pdf", "第 3 页")
        t_p2_fig = bs_a.ask_ui("第 3 页的图片展示了什么趋势？")
        latencies.append(t_p2_fig["elapsed_s"])
        bs_a.wait_last_images(1)
        m_p2_fig = bs_a.read_last_message()
        snap = bs_a.fetch_snapshot(sid_a)
        ev_p2_fig = _check_evidence_records(snap, m_p2_fig["message_id"])
        r_p2_orig = bs_a.open_reader_via(
            f"() => Array.from(document.querySelectorAll('article.message[data-message-id=\"{m_p2_fig['message_id']}\"] .source-actions button')).find(b => b.textContent === '回原件')",
            expect_img=True,
        )
        bs_a.close_reader()
        hl_left = float((r_p2_orig.get("highlight_style") or {}).get("left", "100%").rstrip("%"))
        results["c2"]["C2-03_two_col_pdf"] = {
            "passed": (
                r_p2_f["open"]
                and r_p2_f["img_loaded"]
                and r_p2_orig["open"]
                and r_p2_orig["has_highlight"]
                and hl_left < 40.0
                and any(r.get("origin_execution") == "real_provider" for r in ev_p2_fig["reads"])
            ),
            "formula_reader": r_p2_f["title"],
            "figure_highlight": r_p2_orig["highlight_style"],
            "figure_reads": ev_p2_fig["reads"],
        }
        results["c5_public"]["K-Pub-02"] = {
            "passed": ("InfoNCE" in m_p2_f["text"] or "KD" in m_p2_f["text"]) and results["c2"]["C2-03_two_col_pdf"]["passed"],
            "elapsed_s": round(t_p2_f_clar["elapsed_s"] + t_p2_f["elapsed_s"] + t_p2_fig["elapsed_s"], 3),
        }

        # -----------------------------------------------------------------
        # K-Pub-03 & C2-02: PPTX native table + 2-col PDF table crop & highlight
        # -----------------------------------------------------------------
        print("  -> K-Pub-03 / C2-02 (PPTX + PDF tables with units)...", flush=True)
        bs_a.click_doc_child("Axis-LoRA-2611.07820v1.pdf", "第 5 页")
        t_pdf_tbl = bs_a.ask_ui("第 5 页表格中 MiniLM 在 TEMP. 和 EVENT 上的数值分别是多少？")
        latencies.append(t_pdf_tbl["elapsed_s"])
        m_pdf_tbl = bs_a.read_last_message()
        r_pdf_tbl_crop = bs_a.open_reader_via(
            f"() => document.querySelector('article.message[data-message-id=\"{m_pdf_tbl['message_id']}\"] .citations button')",
            expect_img=True,
        )
        bs_a.close_reader()
        r_pdf_tbl_orig = bs_a.open_reader_via(
            f"() => Array.from(document.querySelectorAll('article.message[data-message-id=\"{m_pdf_tbl['message_id']}\"] .source-actions button')).find(b => b.textContent === '回原件')",
            expect_img=True,
        )
        bs_a.close_reader()

        bs_a.click_doc_child("Lecture-01-Retrieval-and-Attention.pptx", "第 5 页")
        t_pptx_tbl = bs_a.ask_ui("表 1 中 FA3-Pingpong-FP8 的硬件带宽、端到端延迟、峰值显存和核心温度分别是多少？")
        latencies.append(t_pptx_tbl["elapsed_s"])
        m_pptx_tbl = bs_a.read_last_message()
        r_pptx_tbl = bs_a.open_reader_via(
            f"() => document.querySelector('article.message[data-message-id=\"{m_pptx_tbl['message_id']}\"] .citations button')",
            expect_img=True,
        )
        bs_a.close_reader()
        results["c2"]["C2-02_tables_with_units"] = {
            "passed": (
                "41.7" in m_pdf_tbl["text"]
                and "36.1" in m_pdf_tbl["text"]
                and r_pdf_tbl_crop["img_loaded"]
                and r_pdf_tbl_orig["has_highlight"]
                and all(v in m_pptx_tbl["text"] for v in ("3120", "9.6", "2940", "68.4"))
                and r_pptx_tbl["img_loaded"]
            ),
            "pdf_tbl_highlight": r_pdf_tbl_orig["highlight_style"],
            "pptx_tbl_size": r_pptx_tbl["img_size"],
        }
        results["c5_public"]["K-Pub-03"] = {
            "passed": results["c2"]["C2-02_tables_with_units"]["passed"],
            "elapsed_s": round(t_pdf_tbl["elapsed_s"] + t_pptx_tbl["elapsed_s"], 3),
        }

        # -----------------------------------------------------------------
        # K-Pub-04 & C2-04: Notebook Cell 2 (UI Cell 3) dual PNG clarification + follow-up
        # -----------------------------------------------------------------
        print("  -> K-Pub-04 / C2-04 (Notebook multi-output clarification + followup)...", flush=True)
        bs_a.click_doc_child("Lab-01-Retrieval-Bench.ipynb", "Lab-01 Retrieval", "Cell 3")
        t_nb_clar = bs_a.ask_ui("请读取当前 Cell 输出曲线图的横轴单位和峰值数值。", expect_clarification=True)
        latencies.append(t_nb_clar["elapsed_s"])
        t_nb_res = bs_a.click_clarification_and_send("btns => btns.find(b => b.textContent.includes('输出 2'))")
        latencies.append(t_nb_res["elapsed_s"])
        bs_a.wait_last_images(1)
        m_nb1 = bs_a.read_last_message()
        snap = bs_a.fetch_snapshot(sid_a)
        ev_nb1 = _check_evidence_records(snap, m_nb1["message_id"])

        # Check media & preview SHA-256 symmetry
        media_u = m_nb1["cards"][0]["img"]["src"]
        eid_nb1 = m_nb1["cites"][0]["evidence_id"] if m_nb1["cites"] else ev_nb1["displays"][0]["evidence_id"]
        pv_u = f"/api/workspace/sessions/{sid_a}/messages/{m_nb1['message_id']}/evidence/{eid_nb1}/preview"
        url_checks = bs_a.fetch_urls([media_u, pv_u])

        t_nb_fu = bs_a.ask_ui("刚才那张图的纵轴单位和曲线标签是什么？")
        latencies.append(t_nb_fu["elapsed_s"])
        m_nb2 = bs_a.read_last_message()
        snap = bs_a.fetch_snapshot(sid_a)
        ev_nb2 = _check_evidence_records(snap, m_nb2["message_id"])

        nb_sel = ((ev_nb1["trace"] or {}).get("selected_source") or {}).get("source_id")
        nb_reads = [r for r in ev_nb1["reads"] if r["method"] == "vision"]
        nb_act = (nb_reads[0].get("actual_source") or {}).get("source_id") if nb_reads else None
        results["c2"]["C2-04_notebook_multi_output"] = {
            "passed": (
                nb_sel == "cell:2:output:2"
                and nb_act == "cell:2:output:2"
                and nb_reads[0]["origin_execution"] == "real_provider"
                and len({u["sha256"] for u in url_checks}) == 1
                and all(u["status"] == 200 for u in url_checks)
                and "4.12" in m_nb1["text"]
                and ("RMSE" in m_nb2["text"] or "FP8" in m_nb2["text"])
            ),
            "selected_source": nb_sel,
            "actual_source": nb_act,
            "actual_input_sha256": nb_reads[0].get("actual_input_sha256") if nb_reads else None,
            "origin_execution": nb_reads[0].get("origin_execution") if nb_reads else None,
            "url_checks": url_checks,
            "followup_reads": ev_nb2["reads"],
        }
        results["c5_public"]["K-Pub-04"] = {
            "passed": results["c2"]["C2-04_notebook_multi_output"]["passed"],
            "elapsed_s": round(t_nb_clar["elapsed_s"] + t_nb_res["elapsed_s"] + t_nb_fu["elapsed_s"], 3),
        }

        # -----------------------------------------------------------------
        # K-Pub-05 & C2-05: PPTX Slide 7 right image + Slide 8 grouped unboxed fallback
        # -----------------------------------------------------------------
        print("  -> K-Pub-05 / C2-05 (PPTX right image + grouped unboxed fallback)...", flush=True)
        bs_a.click_doc_child("Lecture-01-Retrieval-and-Attention.pptx", "第 7 页")
        t_s7 = bs_a.ask_ui("第 7 页右侧那张图展示了什么指标和延迟？")
        latencies.append(t_s7["elapsed_s"])
        bs_a.wait_last_images(1)
        m_s7 = bs_a.read_last_message()

        bs_a.click_doc_child("Lecture-01-Retrieval-and-Attention.pptx", "第 8 页")
        t_s8 = bs_a.ask_ui("第 8 页的示意图展示了哪两个阶段？")
        latencies.append(t_s8["elapsed_s"])
        bs_a.wait_last_images(1)
        m_s8 = bs_a.read_last_message()
        snap = bs_a.fetch_snapshot(sid_a)
        ev_s8 = _check_evidence_records(snap, m_s8["message_id"])
        disp_s8 = next(
            (d for d in ev_s8["displays"] if d.get("rendition_kind") == "parent_page_fallback"),
            ev_s8["displays"][0] if ev_s8["displays"] else {},
        )
        disp_id_s8 = disp_s8.get("display_id", "")
        r_s8 = bs_a.open_reader_via(
            f"() => Array.from(document.querySelectorAll('article.message[data-message-id=\"{m_s8['message_id']}\"] .source-card[data-display-id=\"{disp_id_s8}\"] .source-actions button')).find(b => b.textContent === '回原件')",
            expect_img=True,
        )
        bs_a.close_reader()
        results["c2"]["C2-05_unboxed_fallback"] = {
            "passed": (
                disp_s8.get("rendition_kind") == "parent_page_fallback"
                and bool(disp_s8.get("fallback_reason"))
                and r_s8["open"]
                and r_s8["img_loaded"]
                and not r_s8["has_highlight"]
            ),
            "display": disp_s8,
            "reader_has_highlight": r_s8["has_highlight"],
        }
        results["c5_public"]["K-Pub-05"] = {
            "passed": "86.9%" in m_s7["text"] and "11.4" in m_s7["text"] and results["c2"]["C2-05_unboxed_fallback"]["passed"],
            "elapsed_s": round(t_s7["elapsed_s"] + t_s8["elapsed_s"], 3),
        }

        # -----------------------------------------------------------------
        # K-Pub-06 (S0 baseline on pure-diagram Slide 6)
        # -----------------------------------------------------------------
        print("  -> K-Pub-06 (S0 baseline on pure-diagram Slide 6)...", flush=True)
        bs_a.clear_file_scope("demo-course")
        t_p6_s0 = bs_a.ask_ui(
            "异步流水延迟曲线图展示在哪一页幻灯片？请读出图中 4K 和 16K 序列长度对应的全部四个 Step Latency (ms) 实测数值。"
        )
        latencies.append(t_p6_s0["elapsed_s"])
        m_p6_s0 = bs_a.read_last_message()
        results["c5_public"]["K-Pub-06_s0"] = {
            "elapsed_s": t_p6_s0["elapsed_s"],
            "hit_pure_slide_values": all(v in m_p6_s0["text"] for v in ("14.2", "6.8", "48.6", "19.4")),
            "cites": [c["label"] for c in m_p6_s0["cites"]],
        }

        # -----------------------------------------------------------------
        # K-Pub-07 & C3-01: FlashAttention-3 page 6 multi-image clarification
        # -----------------------------------------------------------------
        print("  -> K-Pub-07 / C3-01 (FA3 page 6 clarification)...", flush=True)
        bs_a.click_doc_child("FlashAttention-3-2407.08608v2.pdf", "第 6 页")
        t_p7_clar = bs_a.ask_ui("第 6 页的图片展示了什么调度？", expect_clarification=True)
        latencies.append(t_p7_clar["elapsed_s"])
        t_p7_res = bs_a.click_clarification_and_send("btns => btns[0]")
        latencies.append(t_p7_res["elapsed_s"])
        m_p7 = bs_a.read_last_message()
        results["c3"]["C3-01_clarification"] = {
            "passed": len(t_p7_clar["state"]["clarBtns"]) >= 2 and ("pingpong" in m_p7["text"].lower() or "softmax" in m_p7["text"].lower()),
            "options": t_p7_clar["state"]["clarBtns"],
            "chosen": t_p7_res["chosen"],
        }
        results["c5_public"]["K-Pub-07"] = {
            "passed": results["c3"]["C3-01_clarification"]["passed"],
            "elapsed_s": round(t_p7_clar["elapsed_s"] + t_p7_res["elapsed_s"], 3),
        }

        # -----------------------------------------------------------------
        # K-Pub-08 & C3-02: Scope switch + experiment_outside_scope + no-answer
        # -----------------------------------------------------------------
        print("  -> K-Pub-08 / C3-02 (scope switch + out-of-scope + no fabrication)...", flush=True)
        # First do cross-file -> narrow to PPTX only -> followup asking experiment error to trigger experiment_outside_scope
        bs_a.clear_file_scope("demo-course")
        t_cross = bs_a.ask_ui("请说明 双阶异步流水 的定义，给出配套实验的实测结果。")
        latencies.append(t_cross["elapsed_s"])
        m_cross = bs_a.read_last_message()

        # Switch scope to Lecture-01 PPTX only, then ask about the experiment's RMSE
        bs_a.click_doc_child("Lecture-01-Retrieval-and-Attention.pptx")
        scope_after_switch = bs_a.eval("document.querySelector('#scope-summary')?.textContent || ''")
        t_narrow = bs_a.ask_ui("那这个实验测得的 fp8_rmse 数值是多少？")
        latencies.append(t_narrow["elapsed_s"])
        m_narrow = bs_a.read_last_message()
        snap = bs_a.fetch_snapshot(sid_a)
        ev_narrow = _check_evidence_records(snap, m_narrow["message_id"])

        # Switch to Axis-LoRA PDF only and ask nonexistent metric
        bs_a.click_doc_child("Axis-LoRA-2611.07820v1.pdf")
        t_p8 = bs_a.ask_ui("请给出 Axis-LoRA 论文中测得的 1-bit FP1.58 量化在 H100 8-GPU 集群上的训练能耗焦耳数。")
        latencies.append(t_p8["elapsed_s"])
        m_p8 = bs_a.read_last_message()

        no_fab = ("4.12" not in m_narrow["text"]) and any(
            w in m_p8["text"] for w in ("未", "没有", "无法", "缺少", "不包含")
        )
        results["c3"]["C3-02_scope_switch"] = {
            "passed": "Lecture-01" in scope_after_switch and "4.12" not in m_narrow["text"],
            "scope_summary": scope_after_switch,
            "notices": ev_narrow["notices"],
        }
        results["c5_public"]["K-Pub-08"] = {
            "passed": no_fab,
            "elapsed_s": t_p8["elapsed_s"],
        }

        # -----------------------------------------------------------------
        # K-Pub-09: KG dual entry + R1/R3/R4 zero false edges + fallback
        # -----------------------------------------------------------------
        print("  -> K-Pub-09 (KG dual entry + anti-examples)...", flush=True)
        bs_a.eval("(() => { const p = document.querySelector('#kg-panel'); if (p) p.open = true; })()")
        kg_items = bs_a.eval(
            "Array.from(document.querySelectorAll('#kg-list .kg-item')).map(b => b.textContent)"
        )
        # Entry 1: click message concept chip -> opens #kg-detail -> verify defined_in + child_outputs
        chip_cross = bs_a.eval(
            f"""
            (() => {{
              const c = Array.from(document.querySelectorAll('article.message[data-message-id="{m_cross['message_id']}"] .kg-concept-chip')).find(x => x.textContent.includes('双阶异步流水'));
              if (!c) return false;
              c.click();
              return true;
            }})()
            """
        )
        time.sleep(0.4)
        kg_detail_cross = bs_a.eval("document.querySelector('#kg-detail')?.textContent || ''")
        r_kg_def = bs_a.open_reader_via(
            "() => document.querySelector('#kg-detail .kg-card-actions button')", expect_img=True
        )
        bs_a.close_reader()
        r_kg_txt = bs_a.open_reader_via(
            "() => Array.from(document.querySelectorAll('#kg-detail .kg-child-item button')).find(b => b.textContent === '查看输出')",
            expect_img=False,
        )
        bs_a.close_reader()
        r_kg_img = bs_a.open_reader_via(
            "() => Array.from(document.querySelectorAll('#kg-detail .kg-child-item button')).find(b => b.textContent.includes('输出图'))",
            expect_img=True,
        )
        bs_a.close_reader()
        # Entry 2: #kg-panel no-edge concept fallback
        bs_a.eval(
            "(() => { const b = Array.from(document.querySelectorAll('#kg-list .kg-item')).find(x => x.textContent.includes('单向边界守卫')); if (b) b.click(); })()"
        )
        time.sleep(0.4)
        kg_fallback_notice = bs_a.eval("document.querySelector('#kg-detail .kg-notice')?.textContent || ''")
        results["c5_public"]["K-Pub-09"] = {
            "passed": (
                "128" in m_cross["text"]
                and bool(chip_cross)
                and "bench_fa3_pingpong" in kg_detail_cross
                and r_kg_def["open"]
                and r_kg_def["img_loaded"]
                and "第 3 页" in r_kg_def["title"]
                and r_kg_txt["open"]
                and "662" in r_kg_txt["body_text"]
                and r_kg_img["open"]
                and r_kg_img["img_loaded"]
                and "暂无可靠" in kg_fallback_notice
                and all(
                    "42.0" not in k and "pending_run" not in k and "Outline" not in k
                    for k in kg_items
                )
            ),
            "elapsed_s": t_cross["elapsed_s"],
            "kg_items_count": len(kg_items),
            "def_reader_title": r_kg_def["title"],
            "child_txt_title": r_kg_txt["title"],
            "child_img_title": r_kg_img["title"],
            "fallback_notice": kg_fallback_notice,
        }

        # -----------------------------------------------------------------
        # C3-03: Browser reload persistence
        # -----------------------------------------------------------------
        print("  -> C3-03 (browser reload persistence)...", flush=True)
        pre_reload_count = bs_a.msg_count()
        bs_a.cmd("reload")
        bs_a.wait_ready()
        post_reload_count = bs_a.msg_count()
        hist_check = bs_a.fetch_urls([media_u, pv_u])
        results["c3"]["C3-03_reload"] = {
            "passed": (
                post_reload_count == pre_reload_count
                and all(u["status"] == 200 for u in hist_check)
                and hist_check[0]["sha256"] == url_checks[0]["sha256"]
            ),
            "pre_reload_count": pre_reload_count,
            "post_reload_count": post_reload_count,
        }

        # -----------------------------------------------------------------
        # C3-04: Dual independent browser contexts + strict 404 isolation
        # -----------------------------------------------------------------
        print("  -> C3-04 (dual isolated browser sessions + 404 checks)...", flush=True)
        bs_b.open_workspace(base_url)
        sid_b = bs_b.session_id()
        bs_b.import_files([DEMO_DIR / "Distractor-Notes.pdf"], course_id="demo-distractor")
        bs_b.click_catalog_row("rows => rows.find(r => r.title === '课程' && r.textContent.includes('demo-distractor'))")
        t_b = bs_b.ask_ui("在当前课程中 probe_axis_gate 返回多少百分比？")
        m_b = bs_b.read_last_message()
        cross_404 = bs_b.eval_async(
            f"""
            const paths = [
              '/api/workspace/sessions/' + encodeURIComponent({sid_a!r}),
              '/api/workspace/sessions/' + encodeURIComponent({sid_a!r}) + '/kg/concepts',
              {media_u!r},
              {pv_u!r}
            ];
            const out = [];
            for (const p of paths) {{
              const r = await fetch(p, {{credentials:'same-origin'}});
              out.push({{path: p, status: r.status}});
            }}
            return out;
            """
        )
        results["c3"]["C3-04_dual_session_isolation"] = {
            "passed": (
                "12.5%" in m_b["text"]
                and "86.4%" not in m_b["text"]
                and all(item["status"] == 404 for item in cross_404)
            ),
            "session_b_id": sid_b,
            "session_b_elapsed_s": t_b["elapsed_s"],
            "cross_404": cross_404,
        }

        # -----------------------------------------------------------------
        # K-Pub-10 & C3-05: v1 -> v2 upgrade, v1 historical view, KG invalidation
        # -----------------------------------------------------------------
        print("  -> K-Pub-10 / C3-05 (v1 -> v2 upgrade + old citation + KG stale chip)...", flush=True)
        bs_a.click_doc_child("Lab-01-Retrieval-Bench.ipynb")
        up_v2 = bs_a.import_files(
            [DEMO_DIR / "v2" / "Lab-01-Retrieval-Bench.ipynb"],
            course_id="demo-course",
            replace=True,
        )
        bs_a.clear_file_scope("demo-course")
        t_v2_only = bs_a.ask_ui("最新版 Lab-01-Retrieval-Bench.ipynb 中 probe_axis_gate 的 Recall@10 实测数值是多少？")
        latencies.append(t_v2_only["elapsed_s"])
        m_v2_only = bs_a.read_last_message()

        t_p10 = bs_a.ask_ui("请对比最新版和旧版 Lab-01-Retrieval-Bench.ipynb 中 probe_axis_gate 实测输出的变化。")
        latencies.append(t_p10["elapsed_s"])
        m_p10 = bs_a.read_last_message()

        # Click old v1 image citation from m_nb1 -> #reader must show 历史版本 & same SHA
        r_old_v1 = bs_a.open_reader_via(
            f"() => document.querySelector('article.message[data-message-id=\"{m_nb1['message_id']}\"] .citations button')",
            expect_img=True,
        )
        bs_a.close_reader()
        old_v1_url_check = bs_a.fetch_urls([media_u, pv_u])

        # Click stale concept chip (双阶异步流水) from m_cross -> graceful degrade without #workspace-status banner
        bs_a.eval(
            f"""
            (() => {{
              const c = Array.from(document.querySelectorAll('article.message[data-message-id="{m_cross['message_id']}"] .kg-concept-chip')).find(x => x.textContent.includes('双阶异步流水'));
              if (c) c.click();
            }})()
            """
        )
        time.sleep(0.4)
        kg_after_v2 = bs_a.eval("document.querySelector('#kg-detail')?.textContent || ''")
        ws_err_hidden = bs_a.eval("Boolean(document.querySelector('#workspace-status')?.hidden)")

        results["c3"]["C3-05_v1_to_v2"] = {
            "passed": (
                any("v2" in x for x in up_v2)
                and "91.8" in m_v2_only["text"]
                and "86.4" not in m_v2_only["text"]
                and "86.4" in m_p10["text"]
                and "91.8" in m_p10["text"]
                and r_old_v1["open"]
                and r_old_v1["img_loaded"]
                and all(u["status"] == 200 for u in old_v1_url_check)
                and old_v1_url_check[0]["sha256"] == url_checks[0]["sha256"]
                and ("暂无可靠" in kg_after_v2 or "用常规检索查找" in kg_after_v2)
                and ws_err_hidden
            ),
            "upload_v2": up_v2,
            "old_reader_location": r_old_v1["location"],
            "kg_degraded_cleanly": ws_err_hidden and ("用常规检索查找" in kg_after_v2),
        }
        results["c5_public"]["K-Pub-10"] = {
            "passed": results["c3"]["C3-05_v1_to_v2"]["passed"],
            "elapsed_s": round(t_v2_only["elapsed_s"] + t_p10["elapsed_s"], 3),
        }

        # Aggregate external calls & tokens from final session A + B snapshots
        final_snap_a = bs_a.fetch_snapshot(sid_a)
        final_snap_b = bs_b.fetch_snapshot(sid_b)
        ds_calls = 0
        ds_tokens = 0
        vis_calls = 0
        for snap_x in (final_snap_a, final_snap_b):
            for t in snap_x.get("turns", []):
                tr = t.get("trace") or {}
                gen = tr.get("generation") or {}
                ds_calls += int(gen.get("external_call_count", 0) or 0)
                ds_tokens += int(gen.get("total_tokens", 0) or 0)
                rw = ((tr.get("retrieval") or {}).get("rewrite") or {})
                ds_calls += int(rw.get("external_call_count", 0) or 0)
                ds_tokens += int(rw.get("total_tokens", 0) or 0)
                vis = tr.get("vision") or {}
                vis_calls += sum(1 for r in (vis.get("reads") or []) if r.get("external_call"))
            for m in snap_x.get("messages", []):
                # Also catch vision reads outside turn.trace.vision (e.g., S3 or followup) if not double-counted
                pass
        # Authoritative vision external call count across all messages' reads[*]
        vis_msg_calls = sum(
            1
            for snap_x in (final_snap_a, final_snap_b)
            for m in snap_x.get("messages", [])
            for ev in (m.get("sources") or {}).values()
            for r in (ev.get("reads") or [])
            if r.get("method") == "vision" and r.get("external_call")
        )
        vis_calls = max(vis_calls, vis_msg_calls)
        results["usage"] = {
            "deepseek_calls": ds_calls,
            "deepseek_tokens": ds_tokens,
            "vision_calls": vis_calls,
            "latency_p50_s": _percentile(latencies, 50),
            "latency_p95_s": _percentile(latencies, 95),
            "latency_max_s": round(max(latencies), 3) if latencies else None,
        }
        return results
    finally:
        bs_a.close()
        bs_b.close()
        _stop_server(server)
        shutil.rmtree(run_dir, ignore_errors=True)


def _run_phase3_s3_and_kg_checks(
    ab_bin: str, soffice: str, loaded_env: dict[str, str]
) -> dict[str, Any]:
    print("[Phase 3] C6 S3 (vector + local Qdrant) vs S0 pure diagram + S0+KG check...", flush=True)
    run_dir = Path(tempfile.mkdtemp(prefix="srag-k02-p3-"))
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    server = _launch_server(
        {
            **loaded_env,
            "COURSE_ROOT": str(DEMO_DIR),
            "DATA_DIR": str(run_dir / "data"),
            "SQLITE_PATH": str(run_dir / "srag.sqlite3"),
            "WORKSPACE_COURSE_IDS": "demo-course",
            "WORKSPACE_ANSWER_ENABLED": "true",
            "SOURCE_VISION_ENABLED": "true",
            "WORKSPACE_RETRIEVAL_STRATEGY": "vector",
            "WORKSPACE_VECTOR_RETRIEVAL_ENABLED": "true",
            "QDRANT_LOCAL_PATH": str(run_dir / "qdrant_s3"),
            "WORKSPACE_S3_MAX_VISUAL_READS": "2",
            "WORKSPACE_RETRIEVAL_BUDGET": "6",
            "WORKSPACE_RETRIEVAL_POOL_CAP": "12",
            "SRAG_SOFFICE": soffice,
        },
        port,
    )
    bs = BrowserSession(ab_bin, f"k02-p3-{os.getpid()}")
    try:
        bs.open_workspace(base_url)
        sid = bs.session_id()
        bs.import_files(
            [
                DEMO_DIR / "Axis-LoRA-2611.07820v1.pdf",
                DEMO_DIR / "FlashAttention-3-2407.08608v2.pdf",
                DEMO_DIR / "Lecture-01-Retrieval-and-Attention.pptx",
                DEMO_DIR / "v1" / "Lab-01-Retrieval-Bench.ipynb",
            ],
            course_id="demo-course",
        )
        bs.click_catalog_row("rows => rows.find(r => r.title === '课程' && r.textContent.includes('demo-course'))")
        t_s3 = bs.ask_ui(
            "异步流水延迟曲线图展示在哪一页幻灯片？请读出图中 4K 和 16K 序列长度对应的全部四个 Step Latency (ms) 实测数值。",
            timeout=300.0,
        )
        m_s3 = bs.read_last_message()
        snap = bs.fetch_snapshot(sid)
        ev_s3 = _check_evidence_records(snap, m_s3["message_id"])
        vec_trace = ((ev_s3["trace"] or {}).get("retrieval") or {}).get("vector") or {}
        s3_vis = (ev_s3["trace"] or {}).get("s3_visual_reads") or {}
        s3_passed = (
            all(v in m_s3["text"] for v in ("14.2", "6.8", "48.6", "19.4"))
            and vec_trace.get("status") == "succeeded"
            and (s3_vis.get("accepted_count") or 0) >= 1
        )
        s3_res = {
            "passed": s3_passed,
            "elapsed_s": t_s3["elapsed_s"],
            "vector_status": vec_trace.get("status"),
            "visual_triggered": vec_trace.get("visual_triggered"),
            "s3_visual_accepted": s3_vis.get("accepted_count"),
            "cites": [c["label"] for c in m_s3["cites"]],
        }
    finally:
        bs.close()
        _stop_server(server)
        shutil.rmtree(run_dir, ignore_errors=True)

    # Also verify S0 vs S0+KG (workspace_kg_retrieval_enabled=True) via WorkspaceService on a clean temp DB
    sys.path.insert(0, str(REPO_ROOT))
    from app.config import Settings
    from app.rag.local_library import LocalLibrary
    from app.rag.workspace import make_workspace
    from app.schemas.workspace import CreateSession, TurnRequest

    kg_cmp: dict[str, Any] = {}
    for kg_flag in (False, True):
        with tempfile.TemporaryDirectory(prefix=f"srag-k02-kg-{kg_flag}-") as tmp:
            td = Path(tmp)
            st = Settings(
                _env_file=None,
                course_root=DEMO_DIR,
                data_dir=td,
                sqlite_path=td / "kg.sqlite3",
                workspace_course_ids="demo-course",
                workspace_answer_enabled=True,
                source_vision_enabled=False,
                workspace_retrieval_strategy="s0",
                workspace_retrieval_budget=6,
                workspace_kg_retrieval_enabled=kg_flag,
                deepseek_api_key=loaded_env.get("DEEPSEEK_API_KEY", ""),
                deepseek_base_url=loaded_env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                srag_soffice=Path(soffice) if soffice else None,
            )
            lib = LocalLibrary(st)
            lib.import_paths(
                "demo-course",
                [
                    Path("Lecture-01-Retrieval-and-Attention.pptx"),
                    Path("v1/Lab-01-Retrieval-Bench.ipynb"),
                ],
            )
            svc = make_workspace(st)
            sess, owner = svc.create(None, CreateSession(course_ids=["demo-course"]))
            out = svc.submit(
                owner,
                sess["session_id"],
                TurnRequest(
                    client_turn_id=f"kg-{kg_flag}",
                    kind="ask",
                    text="轴向门控路由对应的实验验证测得了什么结果？",
                    scope_revision=sess["scope_revision"],
                ),
            )
            tr = out["turn"]["trace"]
            txt = "\n".join(b.get("text", "") for b in (out.get("message") or {}).get("blocks", []))
            kg_cmp[f"kg_{kg_flag}"] = {
                "kg_trace": tr.get("kg") or (tr.get("retrieval") or {}).get("kg"),
                "hit_86_4": "86.4%" in txt,
                "generation_status": (tr.get("generation") or {}).get("status"),
            }
    return {"s3_pure_diagram": s3_res, "s0_vs_s0_kg": kg_cmp}


def _run_phase4_c7_faults(
    ab_bin: str, soffice: str, loaded_env: dict[str, str]
) -> dict[str, Any]:
    print("[Phase 4] C7 Fault injection & recovery...", flush=True)
    results: dict[str, Any] = {}
    run_dir = Path(tempfile.mkdtemp(prefix="srag-k02-p4-"))
    data_dir = run_dir / "data"
    db_path = run_dir / "srag.sqlite3"

    # Helper to query SQLite counts safely
    def _sqlite_counts(session_id: str, client_turn_id: str) -> dict[str, int]:
        with sqlite3.connect(db_path) as db:
            db.row_factory = sqlite3.Row
            turns = db.execute(
                "SELECT turn_id, status FROM workspace_turns WHERE session_id=? AND client_turn_id=?",
                (session_id, client_turn_id),
            ).fetchall()
            msgs = db.execute(
                """
                SELECT m.role
                FROM workspace_turns t
                JOIN workspace_messages m ON m.turn_id = t.turn_id
                WHERE t.session_id=? AND t.client_turn_id=?
                """,
                (session_id, client_turn_id),
            ).fetchall()
            half_pub = db.execute(
                """
                SELECT count(*) FROM local_documents d
                JOIN local_versions v ON v.document_id = d.document_id AND v.version = d.current_version
                WHERE v.published_at IS NULL
                """
            ).fetchone()[0]
        return {
            "turn_rows": len(turns),
            "user_msgs": sum(1 for m in msgs if m["role"] == "user"),
            "assistant_msgs": sum(1 for m in msgs if m["role"] == "assistant"),
            "half_published_active": int(half_pub),
        }

    # Prepare corrupt files in temp dir
    corrupt_nb = run_dir / "corrupt.ipynb"
    corrupt_nb.write_bytes(b"{not-valid-json")
    corrupt_in_course = DEMO_DIR / ".tmp-corrupt-ignored"  # Do not write in DEMO_DIR!
    del corrupt_in_course

    # 4a. Server with BAD DeepSeek key + BAD DashScope key + Qdrant down (tests C7-01, C7-02, C7-03)
    port1 = _free_port()
    base1 = f"http://127.0.0.1:{port1}"
    active_srv: subprocess.Popen[str] | None = _launch_server(
        {
            **loaded_env,
            "COURSE_ROOT": str(run_dir),
            "DATA_DIR": str(data_dir),
            "SQLITE_PATH": str(db_path),
            "WORKSPACE_COURSE_IDS": "demo-course",
            "WORKSPACE_ANSWER_ENABLED": "true",
            "SOURCE_VISION_ENABLED": "true",
            "DEEPSEEK_API_KEY": "sk-invalid-k02-key-test",
            "DEEPSEEK_MAX_RETRIES": "0",
            "DASHSCOPE_API_KEY": "sk-invalid-k02-vision",
            "WORKSPACE_RETRIEVAL_STRATEGY": "s0",
            "WORKSPACE_RETRIEVAL_BUDGET": "6",
            "QDRANT_URL": "http://127.0.0.1:1",
            "SRAG_SOFFICE": soffice,
        },
        port1,
    )
    bs = BrowserSession(ab_bin, f"k02-p4-{os.getpid()}")
    try:
        bs.open_workspace(base1)
        sid = bs.session_id()
        # C7-01a: Corrupt fresh import in workspace UI (HTTP 200 status=failed)
        c7_01a_ui = bs.import_files([corrupt_nb], course_id="demo-course")
        # Also verify workspace /import HTTP status code == 200 on corrupt upload
        import_http_status = bs.eval_async(
            f"""
            const f = new File(['{{bad-json'], 'bad.ipynb', {{type: 'application/octet-stream'}});
            const r = await fetch(
              `/api/workspace/sessions/${{encodeURIComponent({sid!r})}}/import?course_id=demo-course&filename=bad.ipynb`,
              {{method:'POST', headers:{{'Content-Type':'application/octet-stream'}}, body:f, credentials:'same-origin'}}
            );
            return {{status: r.status, body: await r.json()}};
            """
        )
        # Verify legacy batch /api/rag/local/import HTTP 400 on corrupt file
        legacy_status, legacy_body = _http_json(
            f"{base1}/api/rag/local/import",
            method="POST",
            body={"course_id": "demo-course", "paths": [corrupt_nb.name]},
        )

        # Upload valid v1 Lab-01 & PPTX
        bs.import_files(
            [DEMO_DIR / "Lecture-01-Retrieval-and-Attention.pptx", DEMO_DIR / "v1" / "Lab-01-Retrieval-Bench.ipynb"],
            course_id="demo-course",
        )
        # C7-01b: Corrupt replace-upgrade on Lab-01 v1 -> v1 remains active & usable
        bs.click_doc_child("Lab-01-Retrieval-Bench.ipynb")
        c7_01b_ui = bs.import_files([corrupt_nb], course_id="demo-course", replace=True)
        catalog_ver_after_fail = bs.eval(
            "Array.from(document.querySelectorAll('#catalog .catalog-row')).find(r => r.textContent.includes('Lab-01-Retrieval-Bench'))?.querySelector('.catalog-version')?.textContent"
        )
        # Upload valid v2 -> succeeds
        c7_01c_ui = bs.import_files(
            [DEMO_DIR / "v2" / "Lab-01-Retrieval-Bench.ipynb"], course_id="demo-course", replace=True
        )

        # C7-02: Invalid DeepSeek key -> excerpt_fallback + duplicate client_turn_id dedup
        bs.clear_file_scope("demo-course")
        bs.ask_ui("什么是 Axis-Gated Routing 轴向门控路由 的正式定义？")
        m_bad_ds = bs.read_last_message()
        snap = bs.fetch_snapshot(sid)
        ev_bad_ds = _check_evidence_records(snap, m_bad_ds["message_id"])

        # Duplicate client_turn_id POST idempotency check
        dedup_res = bs.eval_async(
            f"""
            const cid = 'k02-dedup-turn-1';
            const snap0 = await fetch(`/api/workspace/sessions/${{encodeURIComponent({sid!r})}}`, {{credentials:'same-origin'}}).then(r=>r.json());
            const body = {{client_turn_id: cid, kind: 'ask', text: '什么是双阶异步流水？', scope_revision: snap0.scope_revision}};
            const r1 = await fetch(`/api/workspace/sessions/${{encodeURIComponent({sid!r})}}/turns`, {{
              method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(body), credentials:'same-origin'
            }}).then(r=>r.json());
            const r2 = await fetch(`/api/workspace/sessions/${{encodeURIComponent({sid!r})}}/turns`, {{
              method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(body), credentials:'same-origin'
            }}).then(r=>r.json());
            return {{t1: r1.turn.turn_id, t2: r2.turn.turn_id}};
            """
        )
        dedup_counts = _sqlite_counts(sid, "k02-dedup-turn-1")

        # C7-03: Vision failure via DisplayAnchor ("问这张图") with invalid DashScope key
        bs.click_doc_child("Lecture-01-Retrieval-and-Attention.pptx", "第 7 页")
        bs.ask_ui("第 7 页右侧那张图展示了什么指标？")
        bs.wait_last_images(1)
        m_vis_card = bs.read_last_message()
        clicked_ask_img = bs.eval(
            f"""
            (() => {{
              const b = Array.from(document.querySelectorAll('article.message[data-message-id="{m_vis_card['message_id']}"] .source-actions button'))
                .find(x => x.textContent === '问这张图');
              if (!b || b.disabled) return false;
              b.click();
              return true;
            }})()
            """
        )
        if not clicked_ask_img:
            raise RuntimeError("问这张图 button not found or disabled on C7-03 card")
        bs.ask_ui("这张图的具体 Recall@10 百分比和延迟数值是多少？", allow_failed=True)
        snap = bs.fetch_snapshot(sid)
        vis_turn = snap["turns"][-1]
        vis_client_id = vis_turn["client_turn_id"]
        vis_ui_state = bs.eval(
            "({status: document.querySelector('#turn-status')?.textContent || '', actions: document.querySelector('#turn-actions')?.textContent || ''})"
        )
        vis_msg_text = bs.read_last_message()["text"]

        # 4b. Restart server on SAME port1 and SAME DATA_DIR/SQLITE_PATH with VALID DeepSeek + DashScope keys
        _stop_server(active_srv)
        active_srv = None
        time.sleep(0.3)
        active_srv = _launch_server(
            {
                **loaded_env,
                "COURSE_ROOT": str(run_dir),
                "DATA_DIR": str(data_dir),
                "SQLITE_PATH": str(db_path),
                "WORKSPACE_COURSE_IDS": "demo-course",
                "WORKSPACE_ANSWER_ENABLED": "true",
                "SOURCE_VISION_ENABLED": "true",
                "WORKSPACE_RETRIEVAL_STRATEGY": "s0",
                "WORKSPACE_RETRIEVAL_BUDGET": "6",
                "QDRANT_URL": "http://127.0.0.1:1",
                "SRAG_SOFFICE": soffice,
            },
            port1,
        )
        bs.cmd("reload")
        bs.wait_ready()
        sid = bs.session_id()
        # Click 重试本轮 in UI to recover C7-03!
        clicked_retry = bs.eval(
            "(() => { const b = Array.from(document.querySelectorAll('#turn-actions button')).find(x => x.textContent.includes('重试本轮')); if (!b) return false; b.click(); return true; })()"
        )
        if not clicked_retry:
            raise RuntimeError("重试本轮 button not found after reload")
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            st = bs.eval(
                "({running: (document.querySelector('#composer-status')?.textContent||'').includes('正在处理'), statusTxt: document.querySelector('#turn-status')?.textContent||''})"
            )
            if not st["running"] and not st["statusTxt"]:
                break
            time.sleep(0.35)
        snap_rec = bs.fetch_snapshot(sid)
        rec_turn = next(t for t in snap_rec["turns"] if t["client_turn_id"] == vis_client_id)
        rec_counts = _sqlite_counts(sid, vis_client_id)
        m_rec = bs.read_last_message()

        # Verify DeepSeek recovery after restoring key
        bs.clear_file_scope("demo-course")
        bs.ask_ui("什么是 Axis-Gated Routing 轴向门控路由 的正式定义？")
        m_ds_rec = bs.read_last_message()
        snap_rec = bs.fetch_snapshot(sid)
        ev_ds_rec = _check_evidence_records(snap_rec, m_ds_rec["message_id"])

        # C7-04: Delete Lab-01 while browser holds focus -> stale_reference + original_unavailable + clean catalog
        bs.click_doc_child("Lab-01-Retrieval-Bench.ipynb", "Lab-01 Retrieval", "Cell 3")
        bs.ask_ui("当前 Cell 输出了什么指标？", expect_clarification=True)
        bs.click_clarification_and_send("btns => btns[0]")
        m_before_del = bs.read_last_message()
        snap_before_del = bs.fetch_snapshot(sid)
        del_doc_id = next(
            ev["revision"]["source"]["document_id"]
            for ev in (snap_before_del["messages"][-1].get("sources") or {}).values()
        )
        del_eid = m_before_del["cites"][0]["evidence_id"]
        del_ev_url = f"/api/workspace/sessions/{sid}/messages/{m_before_del['message_id']}/evidence/{del_eid}"

        del_http, del_body = _http_json(
            f"{base1}/api/rag/local/documents/{del_doc_id}", method="DELETE"
        )
        bs.ask_ui("再说明一下当前 Cell 的数值。", allow_failed=True)
        stale_ui_err = bs.eval(
            "(document.querySelector('#turn-status')?.textContent || '') + ' ' + (document.querySelector('#workspace-status')?.textContent || '')"
        ).strip()
        deleted_ev_check = bs.fetch_urls([del_ev_url])
        bs.cmd("reload")
        bs.wait_ready()
        catalog_after_del = bs.eval("document.querySelector('#catalog')?.textContent || ''")

        results["C7-01_corrupt_import"] = {
            "passed": (
                import_http_status["status"] == 200
                and import_http_status["body"].get("error") == "import_failed"
                and legacy_status == 400
                and any("导入失败" in x for x in c7_01a_ui)
                and any("导入失败" in x for x in c7_01b_ui)
                and catalog_ver_after_fail == "v1"
                and any("v2" in x for x in c7_01c_ui)
                and rec_counts["half_published_active"] == 0
            ),
            "workspace_import_http": import_http_status,
            "legacy_import_http": legacy_status,
            "ver_after_corrupt_upgrade": catalog_ver_after_fail,
            "half_published_active": rec_counts["half_published_active"],
        }
        results["C7-02_invalid_model_and_dedup"] = {
            "passed": (
                (ev_bad_ds["trace"] or {}).get("generation", {}).get("status") == "provider_error"
                and any(n["code"] == "excerpt_fallback" for n in ev_bad_ds["notices"])
                and dedup_res["t1"] == dedup_res["t2"]
                and dedup_counts["turn_rows"] == 1
                and dedup_counts["user_msgs"] == 1
                and dedup_counts["assistant_msgs"] == 1
                and (ev_ds_rec["trace"] or {}).get("generation", {}).get("status") in {"succeeded", "partial"}
            ),
            "bad_generation_status": (ev_bad_ds["trace"] or {}).get("generation", {}).get("status"),
            "dedup_counts": dedup_counts,
            "recovered_generation_status": (ev_ds_rec["trace"] or {}).get("generation", {}).get("status"),
        }
        results["C7-03_vision_failure_and_retry"] = {
            "passed": (
                ((vis_turn.get("trace") or {}).get("vision") or {}).get("status") == "failed"
                and vis_turn.get("error") == "vision_failed"
                and vis_turn.get("retryable") is True
                and "处理失败" in vis_ui_state["status"]
                and "重试本轮" in vis_ui_state["actions"]
                and "视觉服务读取失败" in vis_msg_text
                and rec_turn.get("status") == "completed"
                and "86.9%" in m_rec["text"]
                and rec_counts["turn_rows"] == 1
                and rec_counts["user_msgs"] == 1
                and rec_counts["assistant_msgs"] == 1
            ),
            "failed_vision_status": ((vis_turn.get("trace") or {}).get("vision") or {}).get("status"),
            "recovered_status": rec_turn.get("status"),
            "retry_sqlite_counts": rec_counts,
        }
        results["C7-04_delete_stale_reference"] = {
            "passed": (
                del_http == 200
                and del_body.get("deleted") is True
                and "版本已更改" in stale_ui_err
                and deleted_ev_check[0]["status"] in {404, 409}
                and "Lab-01" not in catalog_after_del
            ),
            "stale_ui_err": stale_ui_err,
            "deleted_ev_status": deleted_ev_check[0]["status"],
            "removed_from_catalog_after_reload": "Lab-01" not in catalog_after_del,
        }
        return results
    finally:
        bs.close()
        _stop_server(active_srv)
        shutil.rmtree(run_dir, ignore_errors=True)


def _scrub(obj: Any, secrets: list[str]) -> Any:
    raw = json.dumps(obj, ensure_ascii=False)
    for sec in secrets:
        if sec in raw:
            raise AssertionError("Secret value leaked into public acceptance output!")
    if re.search(r"/Users/|/home/", raw):
        raise AssertionError("Absolute home directory path leaked into public acceptance output!")
    return obj


def main() -> None:
    loaded_env, secrets = _load_env_secrets()
    if not loaded_env.get("DEEPSEEK_API_KEY") or not loaded_env.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY and DASHSCOPE_API_KEY are required for K02 real acceptance")
    soffice = _resolve_soffice()
    if not soffice:
        raise RuntimeError("LibreOffice (soffice) is required for K02 PPTX rendering verification")
    ab_bin = _resolve_agent_browser()

    p1 = _run_phase1_offline(ab_bin, soffice)
    p2 = _run_phase2_online_main(ab_bin, soffice, loaded_env)
    p3 = _run_phase3_s3_and_kg_checks(ab_bin, soffice, loaded_env)
    p4 = _run_phase4_c7_faults(ab_bin, soffice, loaded_env)

    # Combine K-Pub-06 from S0 & S3
    p2["c5_public"]["K-Pub-06"] = {
        "passed": p3["s3_pure_diagram"]["passed"],
        "s0": p2["c5_public"].pop("K-Pub-06_s0"),
        "s3": p3["s3_pure_diagram"],
    }

    report = {
        "suite": "K02-public-release-acceptance",
        "demo_manifest_files": len(json.loads((DEMO_DIR / "MANIFEST.json").read_text())["files"]),
        "C1_offline": p1,
        "C2_sources": p2["c2"],
        "C3_multiturn_and_version": p2["c3"],
        "C5_Public_cases": p2["c5_public"],
        "C6_kg_and_s3": p3,
        "C7_fault_recovery": p4,
        "online_main_usage": p2["usage"],
    }
    _scrub(report, secrets)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {OUT_JSON.relative_to(REPO_ROOT)}", flush=True)
    summary = {
        "C1_offline": p1["passed"],
        "C2": {k: v["passed"] for k, v in p2["c2"].items()},
        "C3": {k: v["passed"] for k, v in p2["c3"].items()},
        "C5_Public": {k: v["passed"] for k, v in p2["c5_public"].items()},
        "C6_S3": p3["s3_pure_diagram"]["passed"],
        "C7": {k: v["passed"] for k, v in p4.items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
