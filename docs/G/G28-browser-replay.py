"""Replay a frozen public-material gate through Chromium from a clean product worktree."""

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

BROWSER = runpy.run_path(str(Path(__file__).with_name("G25-verify.py")))


def server_script(root, env_file, port, config):
    template = BROWSER["_server_script"](root, env_file, "public", port)
    template = template[: template.index("orig_sync =")]
    return (
        template.replace(
            '    workspace_answer_enabled=True,\n    workspace_retrieval_strategy="fts",\n'
            "    source_vision_enabled=True,",
            "".join(f"    {key}={value!r},\n" for key, value in config.items()).rstrip(),
        )
        + """
def start(request):
    try:
        model = json.loads(request.content).get("model")
    except Exception:
        model = None
    return {"host": request.url.host, "model": model}, perf_counter()

def finish(row, started):
    row["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
    with (out / "provider.jsonl").open("a") as log:
        log.write(json.dumps(row) + "\\n")

sync_transport = httpx.HTTPTransport.handle_request
async_transport = httpx.AsyncHTTPTransport.handle_async_request

def sync_call(transport, request):
    row, started = start(request)
    try:
        response = sync_transport(transport, request)
        row["status"] = response.status_code
        return response
    except Exception as error:
        row["error"] = type(error).__name__
        raise
    finally:
        finish(row, started)

async def async_call(transport, request):
    row, started = start(request)
    try:
        response = await async_transport(transport, request)
        row["status"] = response.status_code
        return response
    except Exception as error:
        row["error"] = type(error).__name__
        raise
    finally:
        finish(row, started)

httpx.HTTPTransport.handle_request = sync_call
httpx.AsyncHTTPTransport.handle_async_request = async_call
fields = ("deepseek_chat_model", "deepseek_query_model", "dashscope_vision_model",
          "deepseek_timeout_seconds", "deepseek_max_retries", "answer_context_tokens",
          "workspace_course_ids", "workspace_max_upload_bytes")
(out / "settings.json").write_text(json.dumps({
    field: getattr(settings, field) for field in fields
}))
"""
        + f'\nuvicorn.run(app, host="127.0.0.1", port={port}, log_level="warning")\n'
    )


def replay(url, session, folder, manifest, log, record, save):
    def ab(*args):
        return BROWSER["_ab"](session, *args)

    def js(expr):
        return json.loads(ab("eval", expr))

    def wait(expr, timeout=90_000):
        BROWSER["_wait_js"](session, expr, timeout)

    ab("--allowed-domains", "127.0.0.1", "open", url)
    ab("wait", "--text", "暂无资料")
    record["empty_library_visible"] = True
    record["browser"] = js("navigator.userAgent")
    for name in manifest["files"]:
        if not js("document.querySelector('#import-panel').open"):
            ab("find", "text", "导入资料", "click", "--exact")
        ab("select", "#import-course", "public")
        ab("upload", "#import-files", str(folder / name))
        ab("click", "#import-button")
        ab("wait", "--text", f"{name} · 导入完成 · v1")
        wait("!document.querySelector('#import-files').files.length")
    last_call = 0

    def calls():
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    for case in manifest["turns"]:
        row = {"id": case["id"], "question": case["question"], "reader": []}
        record["turns"].append(row)
        try:
            if case["new"]:
                previous = js("document.querySelector('#sessions').value")
                ab("click", "#new-session")
                wait(f"document.querySelector('#sessions').value !== {json.dumps(previous)}")
            scope = case["scope"]
            if scope == "course":
                if not js("document.querySelector('#clear-file').hidden"):
                    ab("click", "#clear-file")
                wait("document.querySelector('#scope-summary').textContent.includes('全部资料')")
            else:
                present = (
                    f"[...document.querySelectorAll('#catalog button[data-node-id]')].some("
                    f"button => button.textContent.includes({json.dumps(scope)}))"
                )
                if not js(present):
                    ab("click", "#catalog button[aria-expanded=false]")
                wait(present)
                node_id = js(
                    "[...document.querySelectorAll('#catalog button[data-node-id]')]"
                    f".find(button => button.textContent.includes({json.dumps(scope)}))"
                    ".dataset.nodeId"
                )
                ab("click", f"#catalog button[data-node-id={json.dumps(node_id)}]")
                wait(
                    f"document.querySelector('#scope-summary').textContent.includes({json.dumps(scope)})"
                )
            row["visible_scope"] = js("document.querySelector('#scope-summary').innerText")
            count = js("document.querySelectorAll('article.message-user').length")
            ab("find", "label", "输入问题", "fill", case["question"])
            ab("click", "#send")
            wait(
                f"document.querySelectorAll('article.message-user').length === {count + 1} && "
                "Array.from(document.querySelectorAll('article.message-user')).at(-1)"
                ".nextElementSibling?.dataset.messageId"
            )
            row.update(
                js("""(async () => {
                const article = Array.from(document.querySelectorAll('article.message-user'))
                    .at(-1).nextElementSibling;
                const sid = document.querySelector('#sessions').value;
                const response = await fetch('/api/workspace/sessions/' + sid);
                const data = await response.json();
                const id = article.dataset.messageId;
                return {message: data.messages.find(m => m.message_id === id),
                    turn: data.turns.at(-1), scope: data.scope,
                    visible: article.innerText, message_id: id,
                    citations: Array.from(new Set(Array.from(article.querySelectorAll(
                        'button[data-evidence-id]'), node => node.dataset.evidenceId)))};
            })()""")
            )
            save()
            for eid in row["citations"]:
                selector = (
                    f'article[data-message-id="{row["message_id"]}"] '
                    f'button[data-evidence-id="{eid}"]'
                )
                clicked = {"evidence_id": eid}
                row["reader"].append(clicked)
                try:
                    js(
                        f"document.querySelector({json.dumps(selector)})"
                        ".scrollIntoView({block: 'center', behavior: 'instant'}) ?? true"
                    )
                    ab("click", selector)
                    wait(
                        "document.querySelector('#reader').open && ("
                        "document.querySelector('#reader img')?.naturalWidth > 0 || "
                        "((!document.querySelector('#reader img') || "
                        "document.querySelector('#reader img').complete) && "
                        "document.querySelector('#reader-body .text-block')?.textContent.trim()))",
                        12_000,
                    )
                    clicked.update(
                        js("""(() => {
                        const reader = document.querySelector('#reader');
                        return {title: document.querySelector('#reader-title').innerText,
                            location: document.querySelector('#reader-location').innerText,
                            text: reader.querySelector('#reader-body .text-block')?.innerText,
                            status: document.querySelector('#reader-status').innerText,
                            preview_notice: reader.querySelector(
                                '#reader-body [role=status]:not([hidden])')?.innerText,
                            pixels: reader.querySelector('img')?.naturalWidth};
                    })()""")
                    )
                except RuntimeError as error:
                    clicked["error"] = str(error)
                finally:
                    if js("document.querySelector('#reader').open"):
                        ab("press", "Escape")
                        wait("!document.querySelector('#reader').open")
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            row["error"] = str(error)
        finally:
            all_calls = calls()
            row["provider_calls"] = all_calls[last_call:]
            last_call = len(all_calls)
            save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("subject", "manifest", "env-file", "out"):
        parser.add_argument("--" + argument, type=Path, required=True)
    args = parser.parse_args()
    subject = args.subject.resolve()
    manifest = json.loads(args.manifest.read_text())
    folder = args.manifest.resolve().parent
    for name, metadata in manifest["files"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != metadata["sha256"]:
            raise SystemExit(f"Unexpected content hash for {name}")

    def git(*params):
        return subprocess.check_output(["git", "-C", str(subject), *params], text=True).strip()

    if git("status", "--porcelain"):
        raise SystemExit("Subject worktree is dirty")
    record = {
        "subject": git("rev-parse", "HEAD"),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "config": manifest["config"],
        "turns": [],
    }

    def save():
        args.out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")

    with tempfile.TemporaryDirectory(prefix="srag-g28-gate-") as tmp:
        root = Path(tmp)
        port = BROWSER["_free_port"]()
        session = f"g28-independent-{port}"
        script = root / "server.py"
        script.write_text(server_script(root, args.env_file.resolve(), port, manifest["config"]))
        env = {**os.environ, "PYTHONPATH": str(subject), "PYTHONDONTWRITEBYTECODE": "1"}
        proc = subprocess.Popen([sys.executable, str(script)], cwd=subject, env=env)
        try:
            url = f"http://127.0.0.1:{port}/workspace/"
            for _ in range(100):
                try:
                    with urllib.request.urlopen(url, timeout=1):
                        break
                except OSError as error:
                    if proc.poll() is not None:
                        raise RuntimeError(f"Server exited with {proc.returncode}") from error
                    time.sleep(0.2)
            else:
                raise RuntimeError("Server did not become ready")
            record["settings"] = json.loads((root / "settings.json").read_text())
            replay(url, session, folder, manifest, root / "provider.jsonl", record, save)
        finally:
            save()
            proc.terminate()
            proc.wait(timeout=15)
            subprocess.run(["agent-browser", "--session", session, "close"], capture_output=True)


if __name__ == "__main__":
    main()
