#!/usr/bin/env python3
"""Independent K03 Release Gate Runner.

Executes against a clean `git archive` checkout in `/tmp/srag-k03-gate-subject`
with completely isolated temporary DATA_DIR/SQLITE_PATH/QDRANT_LOCAL_PATH:
- C1: README / .env.example / MANIFEST.json (8 files SHA-256) + offline cold start + online first question
- C2 & C3: Three-format source inspection, multi-output selection, fallback, reload, dual-session 404, v1->v2
- C5: Pre-frozen K03 blind course cs-got-k (K_Q01..K_Q08) with real DeepSeek + qwen3-vl-plus + LibreOffice
- C6: KG dual-entry & v2 invalidation + S0 vs S3 (K_Q05) + S0 vs S0+KG (K_Q03)
- C7: 4 failure injections & recovery with SQLite duplicate/half-publish assertions
"""

from __future__ import annotations

import hashlib
import json
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

GATE_DIR = Path(__file__).resolve().parent
BLIND_DIR = GATE_DIR / "blind_course"
CRITERIA_JSON = GATE_DIR / "K-blind-criteria.json"
OUT_JSON = Path(os.environ.get("K03_OUT_JSON", str(GATE_DIR / "K03-盲测实测.json")))
SUBJECT_ROOT = Path(os.environ.get("K03_SUBJECT_DIR", "/tmp/srag-k03-gate-subject"))
TESTED_COMMIT = os.environ.get("K03_TESTED_COMMIT", "b8faed8c5658e82d8857e55225079dc8642d487f")
TMP_PREFIX = os.environ.get("K03_TMP_PREFIX", "srag-k03-gate-run-")
PYTHON = sys.executable


def _resolve_soffice() -> str:
    val = os.environ.get("SRAG_SOFFICE", "").strip()
    if val:
        return val
    for cand in (shutil.which("soffice"), "/opt/homebrew/bin/soffice"):
        if cand and Path(cand).is_file():
            return str(cand)
    return ""


def _resolve_ab() -> str:
    for cand in (
        os.environ.get("AGENT_BROWSER", "").strip(),
        shutil.which("agent-browser"),
        str(Path.home() / ".local" / "bin" / "agent-browser"),
    ):
        if cand and Path(cand).is_file():
            return str(cand)
    raise RuntimeError("agent-browser not found")


def _load_env_secrets() -> tuple[dict[str, str], list[str]]:
    env: dict[str, str] = {}
    secrets: list[str] = []
    repo_root = GATE_DIR.parents[2]
    candidates = [
        repo_root / ".env",
        repo_root.parent / "SRAG" / ".env",
        GATE_DIR.parents[4] / ".env",
    ]
    if explicit := os.environ.get("SRAG_ENV_FILE", "").strip():
        candidates.insert(0, Path(explicit))
    for path in candidates:
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k not in env and v:
                env[k] = v
                if (
                    any(t in k.upper() for t in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
                    and len(v) >= 6
                ):
                    secrets.append(v)
    for k, v in os.environ.items():
        if (
            v
            and any(t in k.upper() for t in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
            and len(v) >= 6
        ):
            secrets.append(v)
    return env, list(dict.fromkeys(secrets))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_port(port: int, timeout: float = 30.0) -> None:
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
            return exc.code, json.loads(raw)
        except Exception:
            return exc.code, raw


class ServerProc:
    def __init__(self, cwd: Path, env_overrides: dict[str, str]) -> None:
        self.cwd = cwd
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        env = os.environ.copy()
        env.update(env_overrides)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = str(cwd)
        self.env = env
        self.proc: subprocess.Popen[str] | None = None

    def __enter__(self) -> ServerProc:
        self.proc = subprocess.Popen(
            [
                PYTHON,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--log-level",
                "warning",
            ],
            cwd=str(self.cwd),
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _wait_port(self.port)
        return self

    def __exit__(self, *args: Any) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)


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

    def eval_async(self, body_js: str, timeout: float = 45.0) -> Any:
        self.eval(
            "window.__k03_done=false; window.__k03_res=null; window.__k03_err=null;"
            f"(async()=>{{try{{window.__k03_res=await (async()=>{{{body_js}}})();}}"
            "catch(e){window.__k03_err=String(e);}finally{window.__k03_done=true;}})()"
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.eval("Boolean(window.__k03_done)"):
                err = self.eval("window.__k03_err")
                if err:
                    raise RuntimeError(f"Async JS error: {err}")
                return self.eval("window.__k03_res")
            time.sleep(0.15)
        raise TimeoutError("Async JS timed out")

    def open_workspace(self, base_url: str) -> None:
        self.cmd("open", f"{base_url}/workspace/")
        self.wait_ready()

    def wait_ready(self, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ok = self.eval(
                "Boolean(document.querySelector('#import-button') && "
                "!document.querySelector('#import-button').disabled && "
                "!document.querySelector('#question').disabled)"
            )
            if ok:
                return
            time.sleep(0.25)
        raise TimeoutError("Workspace UI failed to initialize")

    def session_id(self) -> str:
        return str(
            self.eval_async(
                "const d = await fetch('/api/workspace/sessions', {credentials:'same-origin'}).then(r=>r.json()); "
                "return (d.sessions || [])[0]?.session_id;"
            )
        )

    def fetch_snapshot(self, sid: str) -> dict[str, Any]:
        return self.eval_async(
            f"return await fetch('/api/workspace/sessions/' + encodeURIComponent({sid!r}), "
            "{credentials:'same-origin', cache:'no-store'}).then(r=>r.json());"
        )

    def import_files(
        self,
        paths: list[Path],
        *,
        course_id: str | None = None,
        replace: bool = False,
        timeout: float = 120.0,
    ) -> list[str]:
        self.eval(
            "(() => { const p = document.querySelector('#import-panel'); if (p) p.open = true; })()"
        )
        if course_id:
            self.eval(
                f"(() => {{ const s = document.querySelector('#import-course'); if (s) {{ s.value = {course_id!r}; "
                "s.dispatchEvent(new Event('change')); } })()"
            )
        self.eval(
            f"(() => {{ const cb = document.querySelector('#replace-document'); if (cb) {{ "
            f"cb.checked = {str(replace).lower()}; cb.dispatchEvent(new Event('change')); }} }})()"
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
        self.wait_ready()
        time.sleep(0.3)
        return items

    def select_course_scope(self, course_id: str) -> None:
        self.eval(
            f"(() => {{ const s = document.querySelector('#import-course'); if (s) {{ s.value = {course_id!r}; "
            "s.dispatchEvent(new Event('change')); } })()"
        )
        time.sleep(0.2)

    def select_doc_in_catalog(self, doc_substr: str) -> None:
        self.eval(
            """(() => {
              const rows = Array.from(document.querySelectorAll('#catalog .catalog-row'));
              const c = rows.find(r => r.title === '课程' && r.getAttribute('aria-expanded') === 'false');
              if (c) c.click();
            })()"""
        )
        time.sleep(0.3)
        self.eval(
            f"""(() => {{
              const rows = Array.from(document.querySelectorAll('#catalog .catalog-row'));
              const hit = rows.find(r => r.title === '资料' && (r.textContent || '').includes({doc_substr!r}))
                || rows.find(r => (r.textContent || '').includes({doc_substr!r}));
              if (!hit) throw new Error('Catalog item not found: ' + {doc_substr!r});
              hit.click();
            }})()"""
        )
        time.sleep(0.4)

    def select_doc_child(self, doc_substr: str, child_substr: str) -> None:
        self.eval(
            """(() => {
              const rows = Array.from(document.querySelectorAll('#catalog .catalog-row'));
              const c = rows.find(r => r.title === '课程' && r.getAttribute('aria-expanded') === 'false');
              if (c) c.click();
            })()"""
        )
        time.sleep(0.3)
        self.eval(
            f"""(() => {{
              const rows = Array.from(document.querySelectorAll('#catalog .catalog-row'));
              const doc = rows.find(r => r.title === '资料' && (r.textContent || '').includes({doc_substr!r}));
              if (!doc) throw new Error('Catalog doc not found: ' + {doc_substr!r});
              if (doc.getAttribute('aria-expanded') !== 'true') doc.click();
            }})()"""
        )
        time.sleep(0.3)
        for _ in range(4):
            found = self.eval(
                f"""(() => {{
                  const rows = Array.from(document.querySelectorAll('#catalog .catalog-row'));
                  const hit = rows.find(r => (r.textContent || '').includes({child_substr!r}));
                  if (hit) {{ hit.click(); return true; }}
                  const sec = rows.find(r => r.getAttribute('aria-expanded') === 'false' && r.title !== '课程' && r.title !== '资料');
                  if (sec) sec.click();
                  return false;
                }})()"""
            )
            if found:
                time.sleep(0.4)
                return
            time.sleep(0.3)
        raise RuntimeError(f"Catalog child not found: {child_substr}")

    def clear_file_scope(self) -> None:
        self.eval(
            "(() => { const b = document.querySelector('#clear-file'); if (b && !b.hidden) b.click(); })()"
        )
        time.sleep(0.3)

    def close_reader(self) -> None:
        self.eval(
            "(() => { const b = document.querySelector('#close-reader'); "
            "const r = document.querySelector('#reader'); "
            "if (r && r.open && b) b.click(); })()"
        )
        time.sleep(0.15)

    def wait_reader_open(self, *, timeout_ms: int = 10000) -> dict[str, Any]:
        return self.eval_async(
            f"""
            const deadline = Date.now() + {timeout_ms};
            while (Date.now() < deadline) {{
              const r = document.querySelector('#reader');
              const img = document.querySelector('#reader .reader-picture img');
              const title = document.querySelector('#reader-title')?.textContent || '';
              if (r && r.open && img && img.complete && img.naturalWidth > 0) {{
                return {{
                  open: true,
                  title,
                  loc: document.querySelector('#reader-location')?.textContent || '',
                  status: document.querySelector('#reader-status')?.textContent || '',
                  imgW: img.naturalWidth,
                  imgH: img.naturalHeight,
                  hasHl: Boolean(document.querySelector('#reader .reader-highlight')),
                }};
              }}
              await new Promise(res => setTimeout(res, 150));
            }}
            const r = document.querySelector('#reader');
            const img = document.querySelector('#reader .reader-picture img');
            return {{
              open: Boolean(r && r.open),
              title: document.querySelector('#reader-title')?.textContent || '',
              loc: document.querySelector('#reader-location')?.textContent || '',
              status: document.querySelector('#reader-status')?.textContent || '',
              imgW: img?.naturalWidth || 0,
              imgH: img?.naturalHeight || 0,
              hasHl: Boolean(document.querySelector('#reader .reader-highlight')),
            }};
            """
        )

    def ask_ui(self, text: str, *, timeout: float = 120.0) -> dict[str, Any]:
        sid = self.session_id()
        prev_snap = self.fetch_snapshot(sid)
        prev_turns = len(prev_snap.get("turns", []))
        t0 = time.monotonic()
        self.close_reader()
        self.eval("document.querySelector('#question')?.scrollIntoView({block:'center'})")
        self.cmd("fill", "#question", text)
        self.eval(
            "(() => { const q = document.querySelector('#question'); "
            "if (q) q.dispatchEvent(new Event('input', {bubbles:true})); })()"
        )
        self.eval("document.querySelector('#send')?.scrollIntoView({block:'center'})")
        disabled = self.eval("Boolean(document.querySelector('#send')?.disabled)")
        if disabled:
            raise RuntimeError(f"#send disabled for question: {text}")
        self.cmd("click", "#send")
        deadline = time.monotonic() + timeout
        snap: dict[str, Any] = prev_snap
        while time.monotonic() < deadline:
            snap = self.fetch_snapshot(sid)
            turns = snap.get("turns", [])
            running = bool(
                self.eval(
                    "Boolean((document.querySelector('#composer-status')?.textContent || '').includes('正在处理') "
                    "|| (document.querySelector('#turn-actions')?.textContent || '').includes('取消本轮'))"
                )
            )
            if len(turns) > prev_turns and turns[-1].get("status") != "running" and not running:
                break
            time.sleep(0.4)
        else:
            raise TimeoutError(f"ask_ui timed out for: {text}")
        elapsed = round(time.monotonic() - t0, 3)
        turns = snap.get("turns", [])
        msgs = snap.get("messages", [])
        last_turn = turns[-1] if turns else None
        last_assistant = next((m for m in reversed(msgs) if m.get("role") == "assistant"), None)
        return {
            "elapsed_s": elapsed,
            "session_id": sid,
            "turn": last_turn,
            "assistant": last_assistant,
            "snapshot": snap,
        }

    def close(self) -> None:
        try:
            self.cmd("close")
        except Exception:
            pass


def _answer_text(msg: dict[str, Any] | None) -> str:
    if not msg:
        return ""
    parts = [b.get("text", "") for b in msg.get("blocks", []) if b.get("type") == "text"]
    return "\n".join(parts)


def _msg_notices(msg: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not msg:
        return []
    return [
        {"code": b.get("code"), "text": b.get("text")}
        for b in (msg.get("blocks") or [])
        if b.get("type") == "notice"
    ]


def _visual_reads_from_msg(msg: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not msg:
        return []
    out: list[dict[str, Any]] = []
    for eid, src in (msg.get("sources") or {}).items():
        for r in src.get("reads") or []:
            if r.get("method") == "vision":
                out.append({"evidence_id": eid, **r})
    return out


def _verify_demo_manifest(subject_root: Path) -> dict[str, Any]:
    demo_dir = subject_root / "examples" / "demo-course"
    manifest = json.loads((demo_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    checked = []
    for rel, meta in manifest["files"].items():
        p = demo_dir / rel
        raw = p.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        assert len(raw) == meta["bytes"], f"{rel}: bytes mismatch {len(raw)} != {meta['bytes']}"
        assert sha == meta["sha256"], f"{rel}: sha mismatch {sha} != {meta['sha256']}"
        checked.append({"file": rel, "bytes": len(raw), "sha256_prefix": sha[:16]})
    return {"count": len(checked), "files": checked}


def _verify_blind_hashes() -> list[dict[str, Any]]:
    criteria = json.loads(CRITERIA_JSON.read_text(encoding="utf-8"))
    out = []
    for entry in criteria["corpus_files"]:
        p = BLIND_DIR / entry["file_name"]
        raw = p.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        assert len(raw) == entry["bytes"], f"{entry['file_name']}: bytes mismatch"
        assert sha == entry["sha256"], f"{entry['file_name']}: sha256 mismatch"
        out.append({"file_name": entry["file_name"], "bytes": len(raw), "sha256": sha})
    return out


def main() -> None:
    if not SUBJECT_ROOT.is_dir():
        raise FileNotFoundError(f"Clean archive missing at {SUBJECT_ROOT}")
    soffice = _resolve_soffice()
    assert soffice, "SRAG_SOFFICE must be available for K03 real-provider verification"
    ab_bin = _resolve_ab()
    env_secrets, secret_values = _load_env_secrets()
    assert env_secrets.get("DEEPSEEK_API_KEY") and env_secrets.get("DASHSCOPE_API_KEY"), (
        "DEEPSEEK_API_KEY and DASHSCOPE_API_KEY required"
    )

    manifest_check = _verify_demo_manifest(SUBJECT_ROOT)
    blind_hashes = _verify_blind_hashes()

    results: dict[str, Any] = {
        "gate_phase": "K03_independent_release_gate",
        "tested_commit": TESTED_COMMIT,
        "subject_isolated_from_dev_db": True,
        "demo_manifest_check": manifest_check,
        "blind_corpus_hashes": blind_hashes,
        "C1_offline_and_first_question": {},
        "C2_three_format_inspection": {},
        "C3_multiturn_and_versioning": {},
        "C5_blind_8q": {},
        "C6_kg_and_strategy_tradeoffs": {},
        "C7_four_faults": {},
    }

    with tempfile.TemporaryDirectory(prefix=TMP_PREFIX) as td:
        td_path = Path(td)
        # ------------------------------------------------------------------
        # 1. C1 Offline Cold Start on examples/demo-course
        # ------------------------------------------------------------------
        c1_data = td_path / "c1_offline_data"
        c1_data.mkdir(parents=True, exist_ok=True)
        c1_env = {
            "COURSE_ROOT": str(SUBJECT_ROOT / "examples" / "demo-course"),
            "DATA_DIR": str(c1_data),
            "SQLITE_PATH": str(c1_data / "srag.sqlite3"),
            "WORKSPACE_COURSE_IDS": "demo-course",
            "SOURCE_EVIDENCE_COURSE_IDS": "demo-course",
            "SRAG_SOFFICE": soffice,
            "WORKSPACE_ANSWER_ENABLED": "false",
            "SOURCE_VISION_ENABLED": "false",
            "DEEPSEEK_API_KEY": "",
            "DASHSCOPE_API_KEY": "",
        }
        demo_dir = SUBJECT_ROOT / "examples" / "demo-course"
        with ServerProc(SUBJECT_ROOT, c1_env) as srv:
            h_code, h_body = _http_json(f"{srv.base_url}/api/health")
            assert h_code == 200 and h_body.get("status") == "ok"
            ab = BrowserSession(ab_bin, f"k03_c1_{os.getpid()}")
            try:
                ab.open_workspace(srv.base_url)
                empty_txt = ab.eval("document.querySelector('#catalog')?.textContent || ''")
                assert "暂无资料" in empty_txt
                imp = ab.import_files(
                    [
                        demo_dir / "Axis-LoRA-2611.07820v1.pdf",
                        demo_dir / "Lecture-01-Retrieval-and-Attention.pptx",
                        demo_dir / "v1" / "Lab-01-Retrieval-Bench.ipynb",
                    ],
                    course_id="demo-course",
                )
                assert all("导入完成" in x for x in imp), f"Offline import failed: {imp}"
                kg_cnt = int(ab.eval("document.querySelector('#kg-count')?.textContent || '0'"))
                assert kg_cnt >= 3, f"Offline KG count unexpected: {kg_cnt}"
                off_res = ab.ask_ui("什么是 Axis-Gated Routing 轴向门控路由？")
                notices = _msg_notices(off_res["assistant"])
                assert any(n.get("code") == "excerpt_fallback" for n in notices), (
                    f"got notices={notices}"
                )
                assert (off_res["turn"]["trace"] or {}).get("external_call_count", 0) == 0
                print("  [K03] C1 offline passed", flush=True)
                results["C1_offline_and_first_question"]["offline"] = {
                    "passed": True,
                    "health": h_body,
                    "empty_state_ok": True,
                    "imports": imp,
                    "offline_kg_count": kg_cnt,
                    "excerpt_fallback_notices": notices,
                    "external_call_count": 0,
                }
            finally:
                ab.close()

        # Online first question check on demo-course
        c1_on_data = td_path / "c1_online_data"
        c1_on_data.mkdir(parents=True, exist_ok=True)
        c1_on_env = {
            **c1_env,
            "DATA_DIR": str(c1_on_data),
            "SQLITE_PATH": str(c1_on_data / "srag.sqlite3"),
            "WORKSPACE_ANSWER_ENABLED": "true",
            "SOURCE_VISION_ENABLED": "true",
            "DEEPSEEK_API_KEY": env_secrets["DEEPSEEK_API_KEY"],
            "DEEPSEEK_BASE_URL": env_secrets.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "DEEPSEEK_QUERY_MODEL": env_secrets.get("DEEPSEEK_QUERY_MODEL", "deepseek-flash"),
            "DEEPSEEK_CHAT_MODEL": env_secrets.get("DEEPSEEK_CHAT_MODEL", "deepseek-flash"),
            "DASHSCOPE_API_KEY": env_secrets["DASHSCOPE_API_KEY"],
            "DASHSCOPE_VISION_MODEL": env_secrets.get("DASHSCOPE_VISION_MODEL", "qwen3-vl-plus"),
            "WORKSPACE_RETRIEVAL_STRATEGY": "s0",
            "WORKSPACE_RETRIEVAL_BUDGET": "6",
            "WORKSPACE_RETRIEVAL_POOL_CAP": "12",
            "WORKSPACE_RETRIEVAL_INCLUDE_ADJACENCY": "true",
            "WORKSPACE_KG_RETRIEVAL_ENABLED": "false",
            "WORKSPACE_VECTOR_RETRIEVAL_ENABLED": "false",
        }
        with ServerProc(SUBJECT_ROOT, c1_on_env) as srv:
            ab = BrowserSession(ab_bin, f"k03_c1_on_{os.getpid()}")
            try:
                ab.open_workspace(srv.base_url)
                imp = ab.import_files(
                    [
                        demo_dir / "Axis-LoRA-2611.07820v1.pdf",
                        demo_dir / "Lecture-01-Retrieval-and-Attention.pptx",
                    ],
                    course_id="demo-course",
                )
                q1 = ab.ask_ui("什么是 Axis-Gated Routing 轴向门控路由？")
                q1_ans = _answer_text(q1["assistant"])
                has_cit = bool(
                    ab.eval(
                        "(() => { const b = document.querySelector('article.message:last-of-type .citations button'); "
                        "if(!b) return false; b.click(); return true; })()"
                    )
                )
                cit_ok = ab.wait_reader_open() if has_cit else None
                ab.close_reader()
                results["C1_offline_and_first_question"]["online_first_question"] = {
                    "passed": bool(q1_ans)
                    and q1["turn"]["status"] == "completed"
                    and bool(cit_ok and cit_ok.get("open") and cit_ok.get("imgW", 0) > 0),
                    "elapsed_s": q1["elapsed_s"],
                    "answer_preview": q1_ans[:140],
                    "citation_click": cit_ok,
                }
                # C2-05: parent_page_fallback on Lecture-01 page 8 grouped unboxed diagram
                ab.select_doc_child("Lecture-01", "第 8 页")
                q_s8 = ab.ask_ui("第 8 页的示意图展示了哪两个阶段？")
                blocks_s8 = (q_s8["assistant"] or {}).get("blocks", [])
                fb_block = next(
                    (b for b in blocks_s8 if b.get("rendition_kind") == "parent_page_fallback"),
                    None,
                )
                has_fb_btn = bool(
                    ab.eval(
                        "(() => { const btns = Array.from(document.querySelectorAll('article.message:last-of-type .source-actions button')); "
                        "const b = btns.find(x => x.textContent === '回原件'); "
                        "if (!b) return false; b.click(); return true; })()"
                    )
                )
                s8_reader = ab.wait_reader_open() if has_fb_btn else None
                ab.close_reader()
                results["C2_three_format_inspection"]["C2-05_parent_page_fallback"] = {
                    "passed": bool(
                        fb_block
                        and fb_block.get("fallback_reason")
                        and s8_reader
                        and s8_reader.get("open")
                        and s8_reader.get("imgW", 0) > 0
                        and not s8_reader.get("hasHl")
                    ),
                    "rendition_kind": fb_block.get("rendition_kind") if fb_block else None,
                    "fallback_reason": fb_block.get("fallback_reason") if fb_block else None,
                    "reader": s8_reader,
                }
                print("  [K03] C1 online + C2-05 fallback passed", flush=True)
            finally:
                ab.close()

        # ------------------------------------------------------------------
        # 2. C5 Pre-Frozen Blind Course (cs-got-k): K_Q01..K_Q08 + C2/C3/C6
        # ------------------------------------------------------------------
        blind_work = td_path / "blind_upload"
        blind_work.mkdir(parents=True, exist_ok=True)
        # Copy Lab-01 v1 under stable filename Lab-01-GoT-Bench.ipynb
        lab_stable = blind_work / "Lab-01-GoT-Bench.ipynb"
        shutil.copy2(BLIND_DIR / "Lab-01-GoT-Bench-v1.ipynb", lab_stable)
        distractor_src = demo_dir / "Distractor-Notes.pdf"

        blind_data = td_path / "blind_data"
        blind_data.mkdir(parents=True, exist_ok=True)
        blind_db = blind_data / "srag.sqlite3"
        blind_course_root = td_path / "blind_course_root"
        blind_course_root.mkdir(parents=True, exist_ok=True)

        blind_env = {
            "COURSE_ROOT": str(blind_course_root),
            "DATA_DIR": str(blind_data),
            "SQLITE_PATH": str(blind_db),
            "WORKSPACE_COURSE_IDS": "cs-got-k,cs-other-k",
            "SOURCE_EVIDENCE_COURSE_IDS": "cs-got-k,cs-other-k",
            "SRAG_SOFFICE": soffice,
            "WORKSPACE_ANSWER_ENABLED": "true",
            "SOURCE_VISION_ENABLED": "true",
            "DEEPSEEK_API_KEY": env_secrets["DEEPSEEK_API_KEY"],
            "DEEPSEEK_BASE_URL": env_secrets.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "DEEPSEEK_QUERY_MODEL": env_secrets.get("DEEPSEEK_QUERY_MODEL", "deepseek-flash"),
            "DEEPSEEK_CHAT_MODEL": env_secrets.get("DEEPSEEK_CHAT_MODEL", "deepseek-flash"),
            "DASHSCOPE_API_KEY": env_secrets["DASHSCOPE_API_KEY"],
            "DASHSCOPE_VISION_MODEL": env_secrets.get("DASHSCOPE_VISION_MODEL", "qwen3-vl-plus"),
            "WORKSPACE_RETRIEVAL_STRATEGY": "s0",
            "WORKSPACE_RETRIEVAL_BUDGET": "6",
            "WORKSPACE_RETRIEVAL_POOL_CAP": "12",
            "WORKSPACE_RETRIEVAL_INCLUDE_ADJACENCY": "true",
            "WORKSPACE_KG_RETRIEVAL_ENABLED": "false",
            "WORKSPACE_VECTOR_RETRIEVAL_ENABLED": "false",
        }

        criteria = json.loads(CRITERIA_JSON.read_text(encoding="utf-8"))
        q_map = {q["question_id"]: q for q in criteria["blind_questions"]}

        v1_doc_ids: dict[str, str] = {}
        kq06_media_url = ""
        kq06_media_sha = ""
        kq07_v1_msg_id = ""
        kq07_v1_ev_id = ""
        sess_a_id = ""

        with ServerProc(SUBJECT_ROOT, blind_env) as srv:
            ab_a = BrowserSession(ab_bin, f"k03_blind_a_{os.getpid()}")
            ab_b = BrowserSession(ab_bin, f"k03_blind_b_{os.getpid()}")
            try:
                ab_a.open_workspace(srv.base_url)
                imp = ab_a.import_files(
                    [
                        BLIND_DIR / "Paper-01-GoT-2305.16582v2.pdf",
                        BLIND_DIR / "Deck-01-GoT-Architecture.pptx",
                        lab_stable,
                    ],
                    course_id="cs-got-k",
                )
                assert all("导入完成" in x for x in imp), f"Blind import failed: {imp}"
                sess_a_id = ab_a.session_id()
                ab_a.select_course_scope("cs-got-k")
                # Capture document IDs from catalog
                cat_data = ab_a.eval_async(
                    f"return await fetch('/api/workspace/sessions/' + encodeURIComponent({sess_a_id!r}) + '/catalog', "
                    "{credentials:'same-origin'}).then(r=>r.json());"
                )
                for c in cat_data.get("courses", []):
                    for d in c.get("documents", []):
                        v1_doc_ids[d["file_name"]] = d["document_id"]

                # C3-04: Independent session B on cs-other-k + 404 isolation
                ab_b.open_workspace(srv.base_url)
                ab_b.import_files([distractor_src], course_id="cs-other-k")
                sess_b_id = ab_b.session_id()
                ab_b.eval_async(
                    f"return await fetch('/api/workspace/sessions/' + encodeURIComponent({sess_b_id!r}) + '/scope', "
                    "{method:'PATCH', credentials:'same-origin', headers:{'Content-Type':'application/json'}, "
                    "body:JSON.stringify({course_ids:['cs-other-k'], document_ids:[]})}).then(r=>r.json());"
                )
                c3_04_b_ask = ab_b.ask_ui("What percentage is in Distractor-Notes?")
                cross_404 = ab_b.eval_async(
                    f"""
                    const paths = [
                      '/api/workspace/sessions/{sess_a_id}',
                      '/api/workspace/sessions/{sess_a_id}/catalog',
                      '/api/workspace/sessions/{sess_a_id}/kg/concepts'
                    ];
                    const out = [];
                    for (const p of paths) {{
                      const r = await fetch(p, {{credentials:'same-origin'}});
                      out.push({{path: p, status: r.status}});
                    }}
                    return out;
                    """
                )
                results["C3_multiturn_and_versioning"]["C3-04_dual_session_404"] = {
                    "passed": all(x["status"] == 404 for x in cross_404)
                    and "12.5%" in _answer_text(c3_04_b_ask["assistant"]),
                    "b_answer": _answer_text(c3_04_b_ask["assistant"])[:120],
                    "cross_404_checks": cross_404,
                }

                # Ensure session A is scoped to cs-got-k
                ab_a.eval_async(
                    f"return await fetch('/api/workspace/sessions/' + encodeURIComponent({sess_a_id!r}) + '/scope', "
                    "{method:'PATCH', credentials:'same-origin', headers:{'Content-Type':'application/json'}, "
                    "body:JSON.stringify({course_ids:['cs-got-k'], document_ids:[]})}).then(r=>r.json());"
                )

                def _record_q(qid: str, res: dict[str, Any]) -> dict[str, Any]:
                    qmeta = q_map[qid]
                    ans = _answer_text(res["assistant"])
                    blocks = res["assistant"].get("blocks", []) if res["assistant"] else []
                    cits = [c for b in blocks for c in (b.get("citations") or [])]
                    srcs = res["assistant"].get("sources", {}) if res["assistant"] else {}
                    vreads = _visual_reads_from_msg(res["assistant"])
                    req_hits = [
                        f for f in qmeta["required_facts"] if f.casefold() in ans.casefold()
                    ]
                    forb_hits = [
                        f for f in qmeta["forbidden_facts"] if f.casefold() in ans.casefold()
                    ]
                    return {
                        "question_id": qid,
                        "status": res["turn"]["status"],
                        "elapsed_s": res["elapsed_s"],
                        "answer": ans,
                        "citations_count": len(cits),
                        "cited_sources": [
                            {
                                "file_name": s.get("file_name"),
                                "version": s.get("revision", {}).get("source", {}).get("version"),
                                "source_id": s.get("revision", {})
                                .get("source", {})
                                .get("source_id"),
                            }
                            for s in srcs.values()
                        ],
                        "selected_source": res["turn"]["trace"].get("selected_source"),
                        "route": (res["turn"].get("trace") or {}).get("route"),
                        "clarification_options": res["turn"].get("options"),
                        "notices": _msg_notices(res["assistant"]),
                        "visual_reads": vreads,
                        "external_call_count": res["turn"]["trace"].get("external_call_count"),
                        "required_facts_matched": req_hits,
                        "required_facts_total": len(qmeta["required_facts"]),
                        "forbidden_facts_matched": forb_hits,
                    }

                # K_Q01: Two-column borderless tables cross-page + C2-02 reader highlight
                print("  [K03] Running K_Q01..K_Q08 on cs-got-k...", flush=True)
                r_q01 = ab_a.ask_ui(q_map["K_Q01"]["question"])
                results["C5_blind_8q"]["K_Q01"] = _record_q("K_Q01", r_q01)
                print(
                    f"    K_Q01 done: {r_q01['turn']['status']} ({r_q01['elapsed_s']}s)", flush=True
                )
                has_q01_cit = bool(
                    ab_a.eval(
                        "(() => { const cits = Array.from(document.querySelectorAll('article.message:last-of-type .citations button')); "
                        "const cit = cits.find(b => (b.textContent||'').includes('表格')) || cits[0]; "
                        "if(!cit) return false; cit.click(); return true; })()"
                    )
                )
                q01_reader = ab_a.wait_reader_open() if has_q01_cit else None
                ab_a.close_reader()
                results["C2_three_format_inspection"]["C2-02_two_col_table"] = {
                    "passed": bool(
                        q01_reader
                        and q01_reader.get("imgW", 0) > 0
                        and (q01_reader.get("hasHl") or "表格" in (q01_reader.get("title") or ""))
                    ),
                    **(q01_reader or {}),
                }

                # K_Q02: Two-column formulas (page:5 & page:6) + C2-03
                r_q02 = ab_a.ask_ui(q_map["K_Q02"]["question"])
                results["C5_blind_8q"]["K_Q02"] = _record_q("K_Q02", r_q02)
                print(
                    f"    K_Q02 done: {r_q02['turn']['status']} ({r_q02['elapsed_s']}s)", flush=True
                )
                results["C2_three_format_inspection"]["C2-03_two_col_formulas"] = {
                    "passed": r_q02["turn"]["status"] == "completed"
                    and len(results["C5_blind_8q"]["K_Q02"]["required_facts_matched"]) >= 7,
                    "cited_sources": results["C5_blind_8q"]["K_Q02"]["cited_sources"],
                }

                # K_Q03: Default S0 (kg=False) + C2-01 PPTX reader + C6 S0 probe (no 表 1 trigger)
                r_q03 = ab_a.ask_ui(q_map["K_Q03"]["question"])
                results["C5_blind_8q"]["K_Q03_s0"] = _record_q("K_Q03", r_q03)
                print(
                    f"    K_Q03_s0 done: {r_q03['turn']['status']} ({r_q03['elapsed_s']}s)",
                    flush=True,
                )
                r_c6_s0 = ab_a.ask_ui(
                    "课程讲义中“思维图结构编码”需要满足哪三个机制成立条件？它对应的实验结果是什么？"
                )
                c6_s0_ans = _answer_text(r_c6_s0["assistant"])
                has_pptx_cit = bool(
                    ab_a.eval(
                        "(() => { const cits = Array.from(document.querySelectorAll('article.message:last-of-type .citations button')); "
                        "const pptx = cits.find(c => (c.textContent||'').includes('Deck-01')); "
                        "if(!pptx) return false; pptx.click(); return true; })()"
                    )
                )
                pptx_reader = ab_a.wait_reader_open() if has_pptx_cit else None
                ab_a.close_reader()
                results["C2_three_format_inspection"]["C2-01_pptx_whole"] = {
                    "passed": bool(pptx_reader and pptx_reader.get("imgW", 0) > 0),
                    **(pptx_reader or {}),
                }

                # K_Q04: Multi-image directional selection (page:5:shape:6 right chart)
                r_q04 = ab_a.ask_ui(q_map["K_Q04"]["question"])
                results["C5_blind_8q"]["K_Q04"] = _record_q("K_Q04", r_q04)
                print(
                    f"    K_Q04 done: {r_q04['turn']['status']} ({r_q04['elapsed_s']}s)", flush=True
                )

                # K_Q05 under default S0 (expected to miss pure-image Slide 6)
                r_q05_s0 = ab_a.ask_ui(q_map["K_Q05"]["question"])
                results["C5_blind_8q"]["K_Q05_s0"] = _record_q("K_Q05", r_q05_s0)
                print(
                    f"    K_Q05_s0 done: {r_q05_s0['turn']['status']} ({r_q05_s0['elapsed_s']}s)",
                    flush=True,
                )

                # K_Q06: Notebook same-cell 2nd output PNG (cell:1:output:2)
                # Focus cell:1 in catalog first or ask directly
                c301_clar_res: dict[str, Any] | None = None
                r_q06 = ab_a.ask_ui(q_map["K_Q06"]["question"])
                if r_q06["turn"]["status"] == "clarification":
                    opts = r_q06["turn"].get("options", [])
                    opt2 = next(
                        (o for o in opts if "输出 2" in o.get("label", "") or "output:2" in str(o)),
                        opts[-1],
                    )
                    res_clar = ab_a.eval_async(
                        f"""
                        return await fetch('/api/workspace/sessions/' + encodeURIComponent({sess_a_id!r}) + '/turns', {{
                          method: 'POST', credentials: 'same-origin',
                          headers: {{'Content-Type': 'application/json'}},
                          body: JSON.stringify({{
                            client_turn_id: 'k03-q06-clar-' + Date.now(),
                            kind: 'ask',
                            text: {q_map["K_Q06"]["question"]!r},
                            anchor: {{type: 'clarification', turn_id: {r_q06["turn"]["turn_id"]!r}, option_id: {opt2["option_id"]!r}}},
                            scope_revision: {r_q06["snapshot"]["scope_revision"]}
                          }})
                        }}).then(r=>r.json());
                        """
                    )
                    r_q06 = {
                        "elapsed_s": r_q06["elapsed_s"],
                        "session_id": sess_a_id,
                        "turn": res_clar["turn"],
                        "assistant": res_clar["message"],
                        "snapshot": ab_a.fetch_snapshot(sess_a_id),
                    }
                    c301_clar_res = r_q06
                    results["C3_multiturn_and_versioning"]["C3-01_clarification_anchor"] = {
                        "passed": res_clar["turn"]["status"] == "completed",
                        "options_count": len(opts),
                        "chosen_label": opt2.get("label"),
                    }
                else:
                    # Test C3-01 clarification explicitly on Deck-01 page 5 or Lab-01 cell:1 multi-output
                    r_clar = ab_a.ask_ui("Deck-01-GoT-Architecture.pptx 第 5 页那张图展示了什么？")
                    if r_clar["turn"]["status"] != "clarification":
                        ab_a.select_doc_child("Lab-01-GoT-Bench", "Cell 2")
                        r_clar = ab_a.ask_ui("这张输出图展示了什么？")
                    if r_clar["turn"]["status"] == "clarification":
                        opts = r_clar["turn"].get("options", [])
                        opt_r = next(
                            (
                                o
                                for o in opts
                                if "输出 2" in o.get("label", "")
                                or "右" in o.get("label", "")
                                or "output:2" in str(o)
                                or "shape:6" in str(o)
                            ),
                            opts[-1],
                        )
                        res_clar = ab_a.eval_async(
                            f"""
                            return await fetch('/api/workspace/sessions/' + encodeURIComponent({sess_a_id!r}) + '/turns', {{
                              method: 'POST', credentials: 'same-origin',
                              headers: {{'Content-Type': 'application/json'}},
                              body: JSON.stringify({{
                                client_turn_id: 'k03-c301-clar-' + Date.now(),
                                kind: 'ask',
                                text: {r_clar["turn"]["text"]!r},
                                anchor: {{type: 'clarification', turn_id: {r_clar["turn"]["turn_id"]!r}, option_id: {opt_r["option_id"]!r}}},
                                scope_revision: {r_clar["snapshot"]["scope_revision"]}
                              }})
                            }}).then(r=>r.json());
                            """
                        )
                        c301_clar_res = {
                            "turn": res_clar["turn"],
                            "assistant": res_clar["message"],
                        }
                        results["C3_multiturn_and_versioning"]["C3-01_clarification_anchor"] = {
                            "passed": res_clar["turn"]["status"] == "completed",
                            "options_count": len(opts),
                            "chosen_label": opt_r.get("label"),
                        }
                    ab_a.clear_file_scope()
                results["C5_blind_8q"]["K_Q06"] = _record_q("K_Q06", r_q06)
                print(
                    f"    K_Q06 done: {r_q06['turn']['status']} ({r_q06['elapsed_s']}s)", flush=True
                )

                # Verify /media vs /evidence/{img_ev_id}/preview SHA-256 (C2-04)
                c204_target = (
                    r_q06
                    if _visual_reads_from_msg(r_q06["assistant"])
                    else (c301_clar_res or r_q06)
                )
                if c204_target and c204_target.get("assistant"):
                    c204_msg = c204_target["assistant"]
                    kq07_v1_msg_id = c204_msg["message_id"]
                    blocks = c204_msg.get("blocks") or []
                    img_b = next(
                        (
                            b
                            for b in blocks
                            if b.get("type") == "source_image"
                            and (
                                (c204_msg.get("sources") or {})
                                .get(b.get("evidence_id", ""), {})
                                .get("revision", {})
                                .get("source", {})
                                .get("source_id")
                                == "cell:1:output:2"
                            )
                        ),
                        next((b for b in blocks if b.get("type") == "source_image"), None),
                    )
                    if img_b:
                        disp_id = img_b["display_id"]
                        kq07_v1_ev_id = img_b.get("evidence_id") or "ev-1"
                        kq06_media_url = f"/api/workspace/sessions/{sess_a_id}/messages/{kq07_v1_msg_id}/media/{disp_id}?view=inline"
                        prev_url = f"/api/workspace/sessions/{sess_a_id}/messages/{kq07_v1_msg_id}/evidence/{kq07_v1_ev_id}/preview"
                        media_check = ab_a.eval_async(
                            f"""
                            async function sha(url) {{
                              const buf = await (await fetch(url, {{credentials:'same-origin'}})).arrayBuffer();
                              const dig = await crypto.subtle.digest('SHA-256', buf);
                              return {{bytes: buf.byteLength, sha: Array.from(new Uint8Array(dig)).map(b=>b.toString(16).padStart(2,'0')).join('')}};
                            }}
                            const m = await sha({kq06_media_url!r});
                            const p = await sha({prev_url!r});
                            return {{media: m, preview: p, match: m.sha === p.sha}};
                            """
                        )
                        kq06_media_sha = media_check["media"]["sha"]
                        c204_vreads = _visual_reads_from_msg(c204_msg)
                        results["C2_three_format_inspection"]["C2-04_notebook_second_output"] = {
                            "passed": media_check["match"]
                            and bool(c204_vreads)
                            and c204_vreads[0].get("actual_source", {}).get("source_id")
                            == "cell:1:output:2",
                            "source_used": "clarified_cell_focus"
                            if c204_target is c301_clar_res
                            else "direct_K_Q06",
                            "actual_source_id": (
                                c204_vreads[0].get("actual_source", {}).get("source_id")
                                if c204_vreads
                                else None
                            ),
                            **media_check,
                        }

                # K_Q08 (under v1 before upgrading to v2): insufficient evidence + unrun cell:4 + no-edge concept
                r_q08 = ab_a.ask_ui(q_map["K_Q08"]["question"])
                results["C5_blind_8q"]["K_Q08"] = _record_q("K_Q08", r_q08)
                print(
                    f"    K_Q08 done: {r_q08['turn']['status']} ({r_q08['elapsed_s']}s)", flush=True
                )

                # C3-02: Explicit scope switch to Paper-01-GoT only and ask about Lab-01 experiment
                paper_doc_id = v1_doc_ids["Paper-01-GoT-2305.16582v2.pdf"]
                ab_a.select_doc_in_catalog("Paper-01-GoT")
                scoped_snap = ab_a.fetch_snapshot(sess_a_id)
                assert scoped_snap["scope"]["document_ids"] == [paper_doc_id], (
                    f"scope not switched: {scoped_snap['scope']}"
                )
                r_c302 = ab_a.ask_ui(
                    "在当前选定的文件中，run_triplet_hop_k3() 的显存占用和节点准确率是多少？"
                )
                c302_ans = _answer_text(r_c302["assistant"])
                results["C3_multiturn_and_versioning"]["C3-02_explicit_scope_switch"] = {
                    "passed": "18.40" not in c302_ans
                    and "Lab-01" not in c302_ans
                    and "run_triplet_hop_k3" in c302_ans,
                    "scoped_document_ids": scoped_snap["scope"]["document_ids"],
                    "notices": _msg_notices(r_c302["assistant"]),
                    "answer_preview": c302_ans[:160],
                }
                # Restore full course scope via UI button
                ab_a.clear_file_scope()
                restored_snap = ab_a.fetch_snapshot(sess_a_id)
                assert restored_snap["scope"]["document_ids"] == [], (
                    f"scope not restored: {restored_snap['scope']}"
                )

                # C6: KG dual-entry on v1 before upgrade (API + real UI clicks)
                kg_v1 = ab_a.eval_async(
                    f"""
                    const list = await fetch('/api/workspace/sessions/' + encodeURIComponent({sess_a_id!r}) + '/kg/concepts', {{credentials:'same-origin'}}).then(r=>r.json());
                    const c1 = list.concepts.find(c => c.canonical_name === '思维图结构编码');
                    const c3 = list.concepts.find(c => c.canonical_name === '共指实体聚类归并');
                    const d1 = c1 ? await fetch('/api/workspace/sessions/' + encodeURIComponent({sess_a_id!r}) + '/kg/concepts/' + encodeURIComponent(c1.concept_key), {{credentials:'same-origin'}}).then(r=>r.json()) : null;
                    const d3 = c3 ? await fetch('/api/workspace/sessions/' + encodeURIComponent({sess_a_id!r}) + '/kg/concepts/' + encodeURIComponent(c3.concept_key), {{credentials:'same-origin'}}).then(r=>r.json()) : null;
                    return {{concepts: list.concepts, c1_detail: d1, c3_detail: d3}};
                    """
                )
                ab_a.eval(
                    "(() => { const p = document.querySelector('#kg-panel'); if (p) p.open = true; "
                    "const btn = Array.from(document.querySelectorAll('#kg-list button')).find(b => (b.textContent||'').includes('思维图结构编码')); "
                    "if (btn) btn.click(); })()"
                )
                time.sleep(0.5)
                kg_ui_def_clicked = bool(
                    ab_a.eval(
                        "(() => { const b = Array.from(document.querySelectorAll('#kg-detail button')).find(x => (x.textContent||'').includes('讲义')); "
                        "if(!b) return false; b.click(); return true; })()"
                    )
                )
                kg_ui_def_reader = ab_a.wait_reader_open() if kg_ui_def_clicked else None
                ab_a.close_reader()
                kg_ui_out_clicked = bool(
                    ab_a.eval(
                        "(() => { const b = Array.from(document.querySelectorAll('#kg-detail button')).find(x => (x.textContent||'').includes('输出')); "
                        "if(!b) return false; b.click(); return true; })()"
                    )
                )
                kg_ui_out_reader = ab_a.wait_reader_open() if kg_ui_out_clicked else None
                ab_a.close_reader()

                # K_Q07: Replace Lab-01-GoT-Bench.ipynb with v2 via UI
                ab_a.select_doc_in_catalog("Lab-01-GoT-Bench")
                shutil.copy2(BLIND_DIR / "Lab-01-GoT-Bench-v2.ipynb", lab_stable)
                imp_v2 = ab_a.import_files([lab_stable], course_id="cs-got-k", replace=True)
                assert all("导入完成" in x for x in imp_v2), f"v2 upgrade failed: {imp_v2}"
                ab_a.clear_file_scope()
                r_q07 = ab_a.ask_ui(q_map["K_Q07"]["question"])
                results["C5_blind_8q"]["K_Q07"] = _record_q("K_Q07", r_q07)
                print(
                    f"    K_Q07 done: {r_q07['turn']['status']} ({r_q07['elapsed_s']}s)", flush=True
                )
                r_v2_explicit = ab_a.ask_ui(
                    "在 Lab-01-GoT-Bench.ipynb 中，run_triplet_hop_k3() 的节点准确率和图开销是多少？"
                )
                v2_exp_ans = _answer_text(r_v2_explicit["assistant"])
                v2_exp_vers = [
                    s.get("revision", {}).get("source", {}).get("version")
                    for s in (r_v2_explicit["assistant"].get("sources") or {}).values()
                    if s.get("file_name") == "Lab-01-GoT-Bench.ipynb"
                ]

                # C3-03: Reload browser & verify messages + historical v1 media SHA unchanged
                pre_reload_msgs = int(
                    ab_a.eval("document.querySelectorAll('article.message').length")
                )
                ab_a.cmd("reload")
                ab_a.wait_ready()
                post_reload_msgs = int(
                    ab_a.eval("document.querySelectorAll('article.message').length")
                )
                v1_after_reload = ab_a.eval_async(
                    f"""
                    const r = await fetch({kq06_media_url!r}, {{credentials:'same-origin'}});
                    const buf = await r.arrayBuffer();
                    const dig = await crypto.subtle.digest('SHA-256', buf);
                    return {{status: r.status, sha: Array.from(new Uint8Array(dig)).map(b=>b.toString(16).padStart(2,'0')).join('')}};
                    """
                )
                results["C3_multiturn_and_versioning"]["C3-03_reload_persistence"] = {
                    "passed": pre_reload_msgs == post_reload_msgs
                    and pre_reload_msgs > 0
                    and v1_after_reload["status"] == 200,
                    "pre_reload_msgs": pre_reload_msgs,
                    "post_reload_msgs": post_reload_msgs,
                }
                results["C3_multiturn_and_versioning"]["C3-05_v1_to_v2_upgrade"] = {
                    "passed": (
                        "90.81%" in v2_exp_ans
                        and "23.4" in v2_exp_ans
                        and "87.59%" not in v2_exp_ans
                        and all(v == 2 for v in v2_exp_vers if v is not None)
                        and v1_after_reload["status"] == 200
                        and (not kq06_media_sha or v1_after_reload["sha"] == kq06_media_sha)
                    ),
                    "explicit_v2_answer": v2_exp_ans[:200],
                    "explicit_v2_versions": v2_exp_vers,
                    "v1_media_preserved": v1_after_reload,
                    "k_q07_unscoped_route": results["C5_blind_8q"]["K_Q07"]["route"],
                    "k_q07_unscoped_notices": results["C5_blind_8q"]["K_Q07"]["notices"],
                }

                # C6: Check KG v2 invalidation (C2 demonstrated_by removed)
                kg_v2 = ab_a.eval_async(
                    f"""
                    const list = await fetch('/api/workspace/sessions/' + encodeURIComponent({sess_a_id!r}) + '/kg/concepts', {{credentials:'same-origin'}}).then(r=>r.json());
                    const c2 = list.concepts.find(c => c.canonical_name === '模态门控特征融合');
                    const d2 = c2 ? await fetch('/api/workspace/sessions/' + encodeURIComponent({sess_a_id!r}) + '/kg/concepts/' + encodeURIComponent(c2.concept_key), {{credentials:'same-origin'}}).then(r=>r.json()) : null;
                    return {{concepts: list.concepts, c2_detail: d2}};
                    """
                )
                results["C6_kg_and_strategy_tradeoffs"]["kg_verification"] = {
                    "v1_concepts": [
                        {
                            "name": c["canonical_name"],
                            "def": c["defined_in_count"],
                            "demo": c["demonstrated_by_count"],
                        }
                        for c in kg_v1["concepts"]
                    ],
                    "v1_c1_has_child_outputs": bool(
                        kg_v1["c1_detail"]
                        and kg_v1["c1_detail"]["demonstrated_by"]
                        and kg_v1["c1_detail"]["demonstrated_by"][0].get("child_outputs")
                    ),
                    "v1_c3_fallback_to_search": bool(
                        kg_v1["c3_detail"] and kg_v1["c3_detail"].get("fallback_to_search")
                    ),
                    "v2_c2_demo_invalidated": bool(
                        kg_v2["c2_detail"]
                        and len(kg_v2["c2_detail"].get("demonstrated_by", [])) == 0
                        and kg_v2["c2_detail"].get("fallback_to_search")
                    ),
                    "ui_dual_entry_ok": bool(
                        kg_ui_def_reader
                        and kg_ui_def_reader.get("imgW", 0) > 0
                        and kg_ui_out_reader
                        and kg_ui_out_reader.get("imgW", 0) > 0
                    ),
                    "kg_ui_def_reader": kg_ui_def_reader,
                    "kg_ui_out_reader": kg_ui_out_reader,
                }
            finally:
                ab_a.close()
                ab_b.close()

        # ------------------------------------------------------------------
        # 3. C6 Strategy Comparisons on Same Blind Snapshot (S3 for K_Q05, KG-1hop for K_Q03)
        # ------------------------------------------------------------------
        # S0+KG on K_Q03 (using copy of blind_db? Actually Lab-01 is v2 now, so create clean v1 copy for fair S0 vs S0+KG & S3!)
        kg_s3_data = td_path / "kg_s3_data"
        shutil.copytree(blind_data, kg_s3_data)
        # Wait: blind_db currently has Lab-01 v2; that's totally fine for K_Q05 (Deck-01 is unchanged!)
        # For K_Q05 S3 (vector + local Qdrant):
        s3_qdrant = td_path / "k03_s3_qdrant"
        s3_env = {
            **blind_env,
            "DATA_DIR": str(kg_s3_data),
            "SQLITE_PATH": str(kg_s3_data / "srag.sqlite3"),
            "WORKSPACE_RETRIEVAL_STRATEGY": "vector",
            "WORKSPACE_VECTOR_RETRIEVAL_ENABLED": "true",
            "QDRANT_LOCAL_PATH": str(s3_qdrant),
            "WORKSPACE_S3_MAX_VISUAL_READS": "2",
        }
        with ServerProc(SUBJECT_ROOT, s3_env) as srv:
            ab_s3 = BrowserSession(ab_bin, f"k03_s3_{os.getpid()}")
            try:
                ab_s3.open_workspace(srv.base_url)
                sid_s3 = ab_s3.session_id()
                ab_s3.eval_async(
                    f"return await fetch('/api/workspace/sessions/' + encodeURIComponent({sid_s3!r}) + '/scope', "
                    "{method:'PATCH', credentials:'same-origin', headers:{'Content-Type':'application/json'}, "
                    "body:JSON.stringify({course_ids:['cs-got-k'], document_ids:[]})}).then(r=>r.json());"
                )
                r_q05_s3 = ab_s3.ask_ui(q_map["K_Q05"]["question"], timeout=240.0)
                qmeta = q_map["K_Q05"]
                ans_s3 = _answer_text(r_q05_s3["assistant"])
                results["C5_blind_8q"]["K_Q05_s3"] = {
                    "question_id": "K_Q05",
                    "strategy": "vector (S3)",
                    "status": r_q05_s3["turn"]["status"],
                    "elapsed_s": r_q05_s3["elapsed_s"],
                    "answer": ans_s3,
                    "vector_trace": r_q05_s3["turn"]["trace"].get("vector"),
                    "visual_reads": _visual_reads_from_msg(r_q05_s3["assistant"]),
                    "required_facts_matched": [
                        f for f in qmeta["required_facts"] if f.casefold() in ans_s3.casefold()
                    ],
                    "required_facts_total": len(qmeta["required_facts"]),
                }
                results["C6_kg_and_strategy_tradeoffs"]["S0_vs_S3_K_Q05"] = {
                    "s0_matched": len(results["C5_blind_8q"]["K_Q05_s0"]["required_facts_matched"]),
                    "s3_matched": len(results["C5_blind_8q"]["K_Q05_s3"]["required_facts_matched"]),
                    "total": len(qmeta["required_facts"]),
                    "s3_elapsed_s": r_q05_s3["elapsed_s"],
                }
                print(
                    f"    K_Q05_s3 done: {r_q05_s3['turn']['status']} ({r_q05_s3['elapsed_s']}s)",
                    flush=True,
                )
            finally:
                ab_s3.close()

        # S0+KG on K_Q03 (re-import v1 fresh so comparison is fair on v1)
        kg1_data = td_path / "kg1_data"
        kg1_data.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BLIND_DIR / "Lab-01-GoT-Bench-v1.ipynb", lab_stable)
        kg1_env = {
            **blind_env,
            "DATA_DIR": str(kg1_data),
            "SQLITE_PATH": str(kg1_data / "srag.sqlite3"),
            "WORKSPACE_KG_RETRIEVAL_ENABLED": "true",
        }
        with ServerProc(SUBJECT_ROOT, kg1_env) as srv:
            ab_kg = BrowserSession(ab_bin, f"k03_kg1_{os.getpid()}")
            try:
                ab_kg.open_workspace(srv.base_url)
                ab_kg.import_files(
                    [
                        BLIND_DIR / "Paper-01-GoT-2305.16582v2.pdf",
                        BLIND_DIR / "Deck-01-GoT-Architecture.pptx",
                        lab_stable,
                    ],
                    course_id="cs-got-k",
                )
                r_q03_kg = ab_kg.ask_ui(q_map["K_Q03"]["question"])
                ans_kg = _answer_text(r_q03_kg["assistant"])
                results["C5_blind_8q"]["K_Q03_kg"] = {
                    "question_id": "K_Q03",
                    "strategy": "s0+kg",
                    "status": r_q03_kg["turn"]["status"],
                    "elapsed_s": r_q03_kg["elapsed_s"],
                    "answer": ans_kg,
                    "kg_trace": r_q03_kg["turn"]["trace"].get("kg"),
                    "required_facts_matched": [
                        f
                        for f in q_map["K_Q03"]["required_facts"]
                        if f.casefold() in ans_kg.casefold()
                    ],
                    "required_facts_total": len(q_map["K_Q03"]["required_facts"]),
                }
                results["C6_kg_and_strategy_tradeoffs"]["S0_vs_S0KG_K_Q03"] = {
                    "s0_matched": len(results["C5_blind_8q"]["K_Q03_s0"]["required_facts_matched"]),
                    "kg_matched": len(results["C5_blind_8q"]["K_Q03_kg"]["required_facts_matched"]),
                    "total": len(q_map["K_Q03"]["required_facts"]),
                    "kg_external_calls": (r_q03_kg["turn"]["trace"].get("kg") or {}).get(
                        "external_call_count", 0
                    ),
                }
                print(
                    f"    K_Q03_kg done: {r_q03_kg['turn']['status']} ({r_q03_kg['elapsed_s']}s)",
                    flush=True,
                )
                r_c6_kg = ab_kg.ask_ui(
                    "课程讲义中“思维图结构编码”需要满足哪三个机制成立条件？它对应的实验结果是什么？"
                )
                c6_kg_ans = _answer_text(r_c6_kg["assistant"])
                c6_facts = ["512", "150", "GELU", "run_triplet_hop_k3", "87.59%", "19.6", "18.40"]
                results["C6_kg_and_strategy_tradeoffs"]["S0_vs_S0KG_no_table1_probe"] = {
                    "s0_matched": [f for f in c6_facts if f.casefold() in c6_s0_ans.casefold()],
                    "kg_matched": [f for f in c6_facts if f.casefold() in c6_kg_ans.casefold()],
                    "total": len(c6_facts),
                    "kg_trace": r_c6_kg["turn"]["trace"].get("kg"),
                    "kg_external_calls": (r_c6_kg["turn"]["trace"].get("kg") or {}).get(
                        "external_call_count", 0
                    ),
                }
            finally:
                ab_kg.close()

        # ------------------------------------------------------------------
        # 4. C7 Four Failure Injections & Recovery
        # ------------------------------------------------------------------
        c7_data = td_path / "c7_data"
        c7_data.mkdir(parents=True, exist_ok=True)
        c7_db = c7_data / "srag.sqlite3"
        c7_lab = td_path / "c7_upload" / "Lab-01-GoT-Bench.ipynb"
        c7_lab.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BLIND_DIR / "Lab-01-GoT-Bench-v1.ipynb", c7_lab)
        bad_lab = td_path / "c7_upload" / "Broken-Lab.ipynb"
        bad_lab.write_bytes(b"{not valid json!!!")

        c7_env = {
            **blind_env,
            "DATA_DIR": str(c7_data),
            "SQLITE_PATH": str(c7_db),
        }
        # C7-01 & C7-04 with valid keys
        with ServerProc(SUBJECT_ROOT, c7_env) as srv:
            ab_c7 = BrowserSession(ab_bin, f"k03_c7_{os.getpid()}")
            try:
                ab_c7.open_workspace(srv.base_url)
                ab_c7.import_files([c7_lab], course_id="cs-got-k")
                sid_c7 = ab_c7.session_id()
                # C7-01a: broken new import
                imp_bad = ab_c7.import_files([bad_lab], course_id="cs-got-k")
                # C7-01b: broken replace v1
                ab_c7.select_doc_in_catalog("Lab-01-GoT-Bench")
                shutil.copy2(bad_lab, c7_lab)
                imp_bad_rep = ab_c7.import_files([c7_lab], course_id="cs-got-k", replace=True)
                # Verify SQLite has 0 half-published versions & v1 still active
                conn = sqlite3.connect(str(c7_db))
                half_pub = conn.execute(
                    "SELECT count(*) FROM local_versions v JOIN local_documents d ON d.document_id=v.document_id "
                    "WHERE v.published_at IS NULL AND d.current_version = v.version"
                ).fetchone()[0]
                cur_ver = conn.execute(
                    "SELECT current_version FROM local_documents WHERE deleted = 0"
                ).fetchall()
                conn.close()
                # Restore valid v2 & upload
                ab_c7.select_doc_in_catalog("Lab-01-GoT-Bench")
                shutil.copy2(BLIND_DIR / "Lab-01-GoT-Bench-v2.ipynb", c7_lab)
                imp_good_v2 = ab_c7.import_files([c7_lab], course_id="cs-got-k", replace=True)
                results["C7_four_faults"]["C7-01_corrupted_import"] = {
                    "passed": (
                        any("失败" in x for x in imp_bad)
                        and any("失败" in x for x in imp_bad_rep)
                        and half_pub == 0
                        and cur_ver == [(1,)]
                        and all("导入完成" in x for x in imp_good_v2)
                    ),
                    "imp_bad": imp_bad,
                    "imp_bad_rep": imp_bad_rep,
                    "half_published_active": half_pub,
                    "imp_good_v2": imp_good_v2,
                }

                # C7-04: stale reference + delete doc
                cat_c7 = ab_c7.eval_async(
                    f"return await fetch('/api/workspace/sessions/' + encodeURIComponent({sid_c7!r}) + '/catalog', "
                    "{credentials:'same-origin'}).then(r=>r.json());"
                )
                doc_c7 = cat_c7["courses"][0]["documents"][0]
                doc_id_c7 = doc_c7["document_id"]

                def _walk(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
                    out: list[dict[str, Any]] = []
                    for it in items:
                        out.append(it)
                        out.extend(_walk(it.get("children") or []))
                    return out

                all_nodes = _walk(doc_c7.get("nodes") or doc_c7.get("children") or [])
                cell_node = next(
                    (
                        n
                        for n in all_nodes
                        if n.get("kind") == "cell"
                        and (n.get("locator") or {}).get("cell_index") == 1
                    ),
                    next((n for n in all_nodes if n.get("kind") == "cell"), all_nodes[0]),
                )
                cell_node_id = cell_node["node_id"]
                ab_c7.eval_async(
                    f"return await fetch('/api/workspace/sessions/' + encodeURIComponent({sid_c7!r}) + '/focus', "
                    "{method:'PUT', credentials:'same-origin', headers:{'Content-Type':'application/json'}, "
                    f"body:JSON.stringify({{node_id:{cell_node_id!r}, scope_revision:1}})}}).then(r=>r.json());"
                )
                del_status, del_body = _http_json(
                    f"{srv.base_url}/api/rag/local/documents/{doc_id_c7}", method="DELETE"
                )
                stale_res = ab_c7.ask_ui("run_triplet_hop_k3 输出了什么？")
                ab_c7.cmd("reload")
                ab_c7.wait_ready()
                cat_after_del = ab_c7.eval("document.querySelector('#catalog')?.textContent || ''")
                results["C7_four_faults"]["C7-04_delete_stale_ref"] = {
                    "passed": (
                        del_status == 200
                        and stale_res["turn"]["error"] in {"stale_reference", "not_accessible"}
                        and "Lab-01-GoT-Bench" not in cat_after_del
                    ),
                    "delete_status": del_status,
                    "stale_turn_error": stale_res["turn"]["error"],
                    "catalog_cleared": "Lab-01-GoT-Bench" not in cat_after_del,
                }
            finally:
                ab_c7.close()

        # C7-02 (bad DeepSeek key + duplicate client_turn_id) & C7-03 (bad DashScope key + /retry)
        c7_bad_ds_env = {**c7_env, "DEEPSEEK_API_KEY": "sk-invalid-k03-probe"}
        shutil.copy2(BLIND_DIR / "Lab-01-GoT-Bench-v1.ipynb", c7_lab)
        c7_23_data = td_path / "c7_23_data"
        c7_23_data.mkdir(parents=True, exist_ok=True)
        c7_23_db = c7_23_data / "srag.sqlite3"
        c7_bad_ds_env.update({"DATA_DIR": str(c7_23_data), "SQLITE_PATH": str(c7_23_db)})

        c7_sid = ""
        c7_03_turn_id = "k03-c703-retry-turn"
        with ServerProc(SUBJECT_ROOT, c7_bad_ds_env) as srv:
            ab = BrowserSession(ab_bin, f"k03_c72_{os.getpid()}")
            try:
                ab.open_workspace(srv.base_url)
                ab.import_files([c7_lab], course_id="cs-got-k")
                c7_sid = ab.session_id()
                # C7-02: provider_error + excerpt_fallback + duplicate client_turn_id
                dup_cid = "k03-c702-dup"
                r1 = ab.eval_async(
                    f"""
                    const body = JSON.stringify({{client_turn_id:{dup_cid!r}, kind:'ask', text:'run_triplet_hop_k3 结果是什么？', scope_revision:1}});
                    const a = await fetch('/api/workspace/sessions/' + encodeURIComponent({c7_sid!r}) + '/turns', {{method:'POST', credentials:'same-origin', headers:{{'Content-Type':'application/json'}}, body}}).then(r=>r.json());
                    const b = await fetch('/api/workspace/sessions/' + encodeURIComponent({c7_sid!r}) + '/turns', {{method:'POST', credentials:'same-origin', headers:{{'Content-Type':'application/json'}}, body}}).then(r=>r.json());
                    return {{a, b}};
                    """
                )
                gen_st = r1["a"]["turn"]["trace"]["generation"]["status"]
                notices = _msg_notices(r1["a"]["message"])
                conn = sqlite3.connect(str(c7_23_db))
                t_cnt, u_cnt, a_cnt = conn.execute(
                    "SELECT count(DISTINCT t.turn_id), "
                    "sum(CASE WHEN m.role='user' THEN 1 ELSE 0 END), "
                    "sum(CASE WHEN m.role='assistant' THEN 1 ELSE 0 END) "
                    "FROM workspace_turns t LEFT JOIN workspace_messages m ON m.turn_id = t.turn_id "
                    "WHERE t.session_id = ? AND t.client_turn_id = ?",
                    (c7_sid, dup_cid),
                ).fetchone()
                conn.close()
                results["C7_four_faults"]["C7-02_invalid_model_key_dedup"] = {
                    "passed": (
                        gen_st == "provider_error"
                        and any(n.get("code") == "excerpt_fallback" for n in notices)
                        and r1["a"]["turn"]["turn_id"] == r1["b"]["turn"]["turn_id"]
                        and t_cnt == 1
                        and u_cnt == 1
                        and a_cnt == 1
                    ),
                    "generation_status": gen_st,
                    "notices": notices,
                    "dedup_same_turn_id": r1["a"]["turn"]["turn_id"] == r1["b"]["turn"]["turn_id"],
                    "sqlite_counts": {"turns": t_cnt, "user_msgs": u_cnt, "assistant_msgs": a_cnt},
                }
            finally:
                ab.close()

        # C7-03: Valid DeepSeek + Invalid DashScope -> vision_failed -> restore key -> /retry
        c7_bad_vis_env = {
            **c7_env,
            "DATA_DIR": str(c7_23_data),
            "SQLITE_PATH": str(c7_23_db),
            "DASHSCOPE_API_KEY": "sk-invalid-dashscope-k03",
        }
        c7_03_msg_id = ""
        c7_03_disp_id = ""
        c73_sid = ""
        ab_c73 = BrowserSession(ab_bin, f"k03_c73_{os.getpid()}")
        try:
            with ServerProc(SUBJECT_ROOT, c7_bad_vis_env) as srv:
                ab_c73.open_workspace(srv.base_url)
                r_pre = ab_c73.ask_ui("run_triplet_hop_k3 Sweep B 第 2 张输出图展示了什么？")
                c73_sid = r_pre["session_id"]
                c7_03_msg_id = r_pre["assistant"]["message_id"]
                srcs_pre = r_pre["assistant"].get("sources") or {}
                img_blocks = [
                    b for b in r_pre["assistant"]["blocks"] if b.get("type") == "source_image"
                ]
                c7_03_disp_id = next(
                    (
                        b["display_id"]
                        for b in img_blocks
                        if (srcs_pre.get(b.get("evidence_id"), {})
                            .get("revision", {})
                            .get("source", {})
                            .get("source_id")
                            == "cell:1:output:2")
                    ),
                    img_blocks[-1]["display_id"],
                )
                fail_turn = ab_c73.eval_async(
                    f"""
                    return await fetch('/api/workspace/sessions/' + encodeURIComponent({c73_sid!r}) + '/turns', {{
                      method:'POST', credentials:'same-origin', headers:{{'Content-Type':'application/json'}},
                      body: JSON.stringify({{
                        client_turn_id: {c7_03_turn_id!r},
                        kind: 'ask',
                        text: '这张图里 3-hop 的耗时是多少？',
                        anchor: {{type: 'display', message_id: {c7_03_msg_id!r}, display_id: {c7_03_disp_id!r}}},
                        scope_revision: {r_pre["snapshot"]["scope_revision"]}
                      }})
                    }}).then(r=>r.json());
                    """
                )
                assert fail_turn["turn"]["status"] == "failed", f"got fail_turn={fail_turn}"
                assert fail_turn["turn"]["error"] == "vision_failed"
                assert fail_turn["turn"]["trace"]["vision"]["status"] == "failed"

            # Restore valid DashScope key and retry in-place on same browser session (cookie preserved!)
            c7_good_vis_env = {**c7_env, "DATA_DIR": str(c7_23_data), "SQLITE_PATH": str(c7_23_db)}
            with ServerProc(SUBJECT_ROOT, c7_good_vis_env) as srv:
                ab_c73.cmd("open", f"{srv.base_url}/workspace/")
                ab_c73.wait_ready()
                retried = ab_c73.eval_async(
                    f"""
                    return await fetch('/api/workspace/sessions/' + encodeURIComponent({c73_sid!r}) + '/turns/' + encodeURIComponent({c7_03_turn_id!r}) + '/retry', {{
                      method:'POST', credentials:'same-origin'
                    }}).then(r=>r.json());
                    """
                )
                conn = sqlite3.connect(str(c7_23_db))
                t_cnt, u_cnt, a_cnt = conn.execute(
                    "SELECT count(DISTINCT t.turn_id), "
                    "sum(CASE WHEN m.role='user' THEN 1 ELSE 0 END), "
                    "sum(CASE WHEN m.role='assistant' THEN 1 ELSE 0 END) "
                    "FROM workspace_turns t LEFT JOIN workspace_messages m ON m.turn_id = t.turn_id "
                    "WHERE t.session_id = ? AND t.client_turn_id = ?",
                    (c73_sid, c7_03_turn_id),
                ).fetchone()
                conn.close()
                ans_retry = _answer_text(retried["message"])
                results["C7_four_faults"]["C7-03_vision_failed_and_retry"] = {
                    "passed": (
                        retried["turn"]["status"] == "completed"
                        and ("19.6" in ans_retry or "87.59" in ans_retry)
                        and t_cnt == 1
                        and u_cnt == 1
                        and a_cnt == 1
                    ),
                    "retry_status": retried["turn"]["status"],
                    "retry_answer": ans_retry[:160],
                    "sqlite_counts": {"turns": t_cnt, "user_msgs": u_cnt, "assistant_msgs": a_cnt},
                }
        finally:
            ab_c73.close()

    # Final sanitization check before writing OUT_JSON
    dumped = json.dumps(results, ensure_ascii=False, indent=2)
    for sec in secret_values:
        if sec and sec in dumped:
            raise AssertionError("Secret value leaked into K03 output!")
    if re.search(r"/Users/|/home/", dumped):
        raise AssertionError("Absolute path leaked into K03 output!")
    OUT_JSON.write_text(dumped + "\n", encoding="utf-8")
    print("K03_GATE_RUN_FINISHED:", OUT_JSON.name)


if __name__ == "__main__":
    main()
