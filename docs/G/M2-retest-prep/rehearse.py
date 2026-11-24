"""Exercise the production workspace with fresh synthetic files and seal public exports."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import platform
import runpy
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BROWSER = runpy.run_path(str(ROOT / "docs/G/G25-verify.py"))
GUIDE = "Basalt-Latch-Guide.pdf"
LAB = "Latch-Trial.ipynb"
PLOTS = "Latch-Plots.ipynb"
DECK = "Latch-Check.pptx"
COURSE = "public"
QUESTION = "Basalt latch 的定义是什么？讲义所对应的留出集实验如何验证？"
MISSING = "Basalt latch 的定义是什么？讲义所对应的盐雾实验如何验证？"
SHOW = "请展示第 1 个 Cell 的保存输出。"
VISION = "刚才第二张图，横轴的单位是什么？"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def notebook(path, source, outputs=()):
    cell = {"cell_type": "markdown", "id": "public-note", "source": source, "metadata": {}}
    if outputs:
        cell.update(cell_type="code", execution_count=1, outputs=list(outputs))
    dump(path, {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [cell]})


def plot(label):
    image = Image.new("RGB", (680, 370), "white")
    draw = ImageDraw.Draw(image)
    draw.line([(80, 55), (80, 245), (620, 245)], width=5, fill="#233049")
    draw.line([(110, 203), (240, 120), (390, 140), (570, 69)], width=7, fill="#156a63")
    draw.text((340, 308), label, font=ImageFont.load_default(size=45), anchor="mm", fill="#233049")
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def fixtures(folder):
    folder.mkdir()
    builder = runpy.run_path(str(ROOT / "docs/G/G14-fixtures.py"))["make_pdf"]
    builder.__globals__["PAGES"] = (
        (
            "Basalt latch calibration",
            "Definition: A basalt latch opens on three consecutive readings strictly above 8 V.",
            "Experiment-ID: SlateTrialB6.",
            "Only dry bench observations are included in this guide.",
        ),
    )
    builder(folder / GUIDE)
    notebook(
        folder / LAB,
        "Experiment-ID: SlateTrialB6. Split: held-out. Metric: MAE. Unit: amplitude. "
        "Baseline: 0.26. Trial: 0.19. Samples: 4.\n"
        "Baseline absolute errors: (0.21, 0.25, 0.30, 0.28).\n"
        "Trial absolute errors: (0.15, 0.17, 0.22, 0.22).",
    )
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[5])
    slide.shapes.title.text = "Basalt handling checklist"
    table = slide.shapes.add_table(2, 2, Inches(1), Inches(2), Inches(7), Inches(1)).table
    for idx, value in enumerate(("Check", "Status", "Dry bench", "Recorded")):
        table.cell(idx // 2, idx % 2).text = value
    deck.save(folder / DECK)
    images = {}
    for version, label in ((1, "Voltage [mV]"), (2, "Displacement [mm]")):
        location = folder / f"v{version}"
        location.mkdir()
        output = []
        for label_for_plot in ("Time [ms]", label):
            data = plot(label_for_plot)
            images[sha(data)] = label_for_plot
            output.append(
                {
                    "output_type": "display_data",
                    "metadata": {},
                    "data": {"image/png": base64.b64encode(data).decode(), "text/plain": "<Chart>"},
                }
            )
        notebook(location / PLOTS, "# saved public plot outputs", output)
    return images


def settings_for(env_file, runtime, *, web):
    from app.config import Settings

    overrides = (
        {}
        if web
        else {
            "deepseek_api_key": "",
            "dashscope_vision_model": "qwen3-vl-plus",
            "dashscope_timeout_seconds": 20,
        }
    )
    return Settings(
        _env_file=env_file,
        course_root=runtime / "originals",
        data_dir=runtime / "data",
        sqlite_path=runtime / "library.sqlite3",
        workspace_course_ids=COURSE,
        workspace_answer_enabled=web,
        source_vision_enabled=True,
        workspace_retrieval_strategy="fts",
        jev_api_key="",
        qdrant_api_key="",
        qdrant_url="http://127.0.0.1:1",
        **overrides,
    )


def effective(settings):
    names = (
        "workspace_answer_enabled",
        "workspace_retrieval_strategy",
        "workspace_input_token_limit",
        "answer_context_tokens",
        "source_vision_enabled",
        "dashscope_vision_model",
        "dashscope_timeout_seconds",
        "deepseek_chat_model",
        "deepseek_query_model",
        "deepseek_timeout_seconds",
        "jev_model",
        "jev_timeout_seconds",
        "retrieval_enable_visual",
    )
    flags = {name: getattr(settings, name) for name in names}
    return flags | {
        f"{provider}_configured": bool(getattr(settings, f"{provider}_api_key").strip())
        for provider in ("deepseek", "dashscope", "jev", "qdrant")
    }


def app_for(settings):
    from app.api.workspace import get_workspace_service
    from app.config import get_settings
    from app.main import create_app
    from app.rag.workspace import make_workspace

    app = create_app()
    workspace = make_workspace(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_workspace_service] = lambda: workspace
    return app, workspace


def observe_providers(settings, events, *, web, image_hashes, event_file=None):
    """Record actual transport attempts for these synthetic runs, without headers or key values."""
    from app.rag.source_vision import (
        _DASHSCOPE_DEFAULT_BASE_URL,
        _DASHSCOPE_MULTIMODAL_PATH,
        _prompt,
    )

    dash = httpx.URL(
        (settings.dashscope_base_url.rstrip("/") or _DASHSCOPE_DEFAULT_BASE_URL)
        + _DASHSCOPE_MULTIMODAL_PATH
    )
    deep = httpx.URL(settings.deepseek_base_url.rstrip("/") + "/v1/chat/completions")
    sync = httpx.HTTPTransport.handle_request
    asynchronous = httpx.AsyncHTTPTransport.handle_async_request

    def save(entry):
        events.append(entry)
        if event_file is not None:
            with event_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def event(request):
        payload = json.loads(request.content)
        entry = {"request_sha256": sha(request.content), "model": payload.get("model")}
        if request.url == dash:
            image, prompt = payload["input"]["messages"][0]["content"]
            digest = sha(base64.b64decode(image["image"].split(",", 1)[1], validate=True))
            if digest in image_hashes and prompt["text"] == _prompt("directed_question", VISION):
                return entry | {"provider": "dashscope", "actual_input_sha256": digest}
        if request.url == deep and web:
            user = payload["messages"][-1]["content"]
            try:
                user = json.loads(user)
            except ValueError:
                user = {}
            evidence_ids = (
                [item["id"] for item in user.get("evidence", [])] if isinstance(user, dict) else []
            )
            return entry | {"provider": "deepseek", "input_evidence_ids": evidence_ids}
        save(entry | {"outcome": "blocked", "provider_host": request.url.host})
        raise RuntimeError("Unexpected outbound request in public rehearsal")

    def finish(entry, elapsed, response=None):
        if response is not None:
            entry["http_status"] = response.status_code
            try:
                entry["usage"] = response.json().get("usage")
            except ValueError:
                entry["usage"] = None
        entry["elapsed_ms"] = round(elapsed * 1000, 3)
        save(entry)

    def handle_sync(transport, request):
        entry, started = event(request), time.perf_counter()
        response = None
        try:
            response = sync(transport, request)
            response.read()
            return response
        finally:
            finish(entry, time.perf_counter() - started, response)

    async def handle_async(transport, request):
        entry, started = event(request), time.perf_counter()
        response = None
        try:
            response = await asynchronous(transport, request)
            await response.aread()
            return response
        finally:
            finish(entry, time.perf_counter() - started, response)

    httpx.HTTPTransport.handle_request = handle_sync
    httpx.AsyncHTTPTransport.handle_async_request = handle_async


def run_asgi_case(out, env_file, files, image_hashes, name):
    from fastapi.testclient import TestClient

    runtime = out / f"asgi-{name}"
    settings = settings_for(env_file, runtime, web=False)
    before = {
        "sqlite": settings.sqlite_path.exists(),
        "originals": settings.course_root.exists(),
        "data": settings.data_dir.exists(),
    }
    app, workspace = app_for(settings)
    events = []
    observe_providers(settings, events, web=False, image_hashes=image_hashes)
    ledger = []
    root = "http://127.0.0.1:18886"
    with TestClient(app, base_url=root, headers={"Origin": root}) as client:

        def request(method, path, **params):
            started = time.perf_counter()
            response = client.request(method, path, **params)
            is_json = response.headers.get("content-type", "").startswith("application/json")
            row = {
                "method": method,
                "path": str(response.request.url.path),
                "http_status": response.status_code,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            }
            if "json" in params:
                row["request"] = params["json"]
            if "content" in params:
                row["upload"] = params["params"] | {"sha256": sha(params["content"])}
            if is_json:
                row["response"] = response.json()
            else:
                row.update(sha256=sha(response.content), bytes=len(response.content))
            ledger.append(row)
            return response

        session = request("POST", "/api/workspace/sessions", json={}).json()
        route = f"/api/workspace/sessions/{session['session_id']}"
        catalog = request("GET", route + "/catalog").json()

        def upload(path, document_id=None):
            params = {"course_id": COURSE, "filename": path.name}
            if document_id:
                params["document_id"] = document_id
            return request(
                "POST",
                route + "/import",
                params=params,
                content=path.read_bytes(),
                headers={"Content-Type": "application/octet-stream"},
            ).json()

        def ask(text, turn_id):
            before_turn = request("GET", route).json()
            response = request(
                "POST",
                route + "/turns",
                json={
                    "kind": "ask",
                    "scope_revision": before_turn["scope_revision"],
                    "client_turn_id": turn_id,
                    "text": text,
                },
            ).json()
            message = response.get("message") or {}
            cited = {
                key
                for block in message.get("blocks", [])
                if block["type"] == "text"
                for key in block.get("citations", [])
            }
            for key in sorted(cited):
                citation = f"{route}/messages/{message['message_id']}/evidence/{key}"
                request("GET", citation)
                request("GET", citation + "/preview")
            return message

        if name == "visual_version":
            doc_id = upload(files / "v1" / PLOTS)["document_id"]
            revision = request(
                "PATCH",
                route + "/scope",
                json={
                    "course_ids": [COURSE],
                    "document_ids": [doc_id],
                    "scope_revision": session["scope_revision"],
                },
            ).json()["scope_revision"]
            current = request("GET", route + "/catalog").json()
            node = next(
                d["node_id"]
                for c in current["courses"]
                for d in c["documents"]
                if d["document_id"] == doc_id
            )
            request("PUT", route + "/focus", json={"node_id": node, "scope_revision": revision})
            ask(SHOW, "show-v1")
            old = ask(VISION, "vision-v1")
            upload(files / "v2" / PLOTS, doc_id)
            ask(SHOW, "show-v2")
            ask(VISION, "vision-v2")
            for block in old.get("blocks", []):
                for eid in block.get("citations", []):
                    request("GET", f"{route}/messages/{old['message_id']}/evidence/{eid}/preview")
        else:
            upload(files / GUIDE)
            upload(files / DECK)
            upload(files / LAB)
            ask(QUESTION if name == "cross_file" else MISSING, name)
        export = request("GET", route).json()
    return {
        "storage_before": before,
        "empty_catalog": catalog,
        "settings": effective(settings),
        "scope": sorted(workspace.allowed_courses),
        "http": ledger,
        "provider": events,
        "workspace_export": export,
    }


def run_asgi(out, env_file, files, image_hashes):
    result = {}
    for name in ("cross_file", "missing_condition", "visual_version"):
        try:
            result[name] = run_asgi_case(out, env_file, files, image_hashes, name)
        finally:
            restore_providers()
    dump(out / "asgi.json", result)


def restore_providers():
    httpx.HTTPTransport.handle_request = SYNC_ORIGINAL
    httpx.AsyncHTTPTransport.handle_async_request = ASYNC_ORIGINAL


SYNC_ORIGINAL = httpx.HTTPTransport.handle_request
ASYNC_ORIGINAL = httpx.AsyncHTTPTransport.handle_async_request


def serve_web(runtime, env_file, port):
    import uvicorn

    settings = settings_for(env_file, runtime, web=True)
    app, workspace = app_for(settings)
    events = []
    observe_providers(
        settings, events, web=True, image_hashes={}, event_file=runtime / "provider.jsonl"
    )
    dump(
        runtime / "settings.json",
        effective(settings) | {"scope": sorted(workspace.allowed_courses)},
    )
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    finally:
        dump(runtime / "provider.json", events)


def run_web(out, env_file, files, subject):
    runtime = out / "web"
    runtime.mkdir()
    port = BROWSER["_free_port"]()
    url = f"http://127.0.0.1:{port}/workspace/"
    session = f"m2-prep-{port}"
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "_serve",
            str(runtime),
            str(env_file),
            str(port),
        ],
        cwd=subject,
        env={**os.environ, "PYTHONPATH": str(subject), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    result = None
    try:
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError("Browser workspace exited")
            try:
                with urllib.request.urlopen(url, timeout=1):
                    break
            except OSError:
                time.sleep(0.2)
        else:
            raise RuntimeError("Browser workspace did not start")
        probe = runpy.run_path(str(ROOT / "docs/G/G28-verify.py"))
        result = probe["browser_run"](
            url, session, [files / GUIDE, files / DECK, files / LAB], QUESTION, None
        )
        response = BROWSER["_ab"](
            session,
            "eval",
            """(async()=>{
          const id = document.querySelector('#sessions').value;
          const base = '/api/workspace/sessions/' + id;
          return {browser:navigator.userAgent,
                  session:await (await fetch(base)).json(),
                  catalog:await (await fetch(base+'/catalog')).json()};
        })()""",
        )
        result["export"] = json.loads(response)
    finally:
        process.terminate()
        process.wait(timeout=15)
        subprocess.run(
            ["agent-browser", "--session", session, "close"], check=False, capture_output=True
        )
    if result is not None:
        result["settings"] = json.loads((runtime / "settings.json").read_text())
        provider_log = runtime / "provider.jsonl"
        result["provider"] = (
            [json.loads(line) for line in provider_log.read_text().splitlines()]
            if provider_log.exists()
            else []
        )
        dump(out / "web.json", result)


def seal(out):
    entries = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {str(p.relative_to(out)): sha(p.read_bytes()) for p in entries}
    dump(out / "seal.json", manifest)
    for path in [*entries, out / "seal.json"]:
        path.chmod(0o444)
    print(f"seal_sha256={sha((out / 'seal.json').read_bytes())} files={len(entries)}")


def verify(out):
    entries = json.loads((out / "seal.json").read_text())
    actual = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}
    if actual != set(entries) | {"seal.json"}:
        raise RuntimeError("Unexpected archive members")
    for name, digest in entries.items():
        path = out / name
        if sha(path.read_bytes()) != digest or path.stat().st_mode & 0o222:
            raise RuntimeError(f"Invalid sealed member {name}")
    if (out / "seal.json").stat().st_mode & 0o222:
        raise RuntimeError("Writable seal")
    print(f"verified={len(entries)} seal_sha256={sha((out / 'seal.json').read_bytes())}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_serve":
        serve_web(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    for argument in ("subject", "env-file", "out"):
        run.add_argument("--" + argument, required=True, type=Path)
    run.add_argument("--reuse-asgi", type=Path, help="read a public rehearsal checkpoint")
    audit = commands.add_parser("verify")
    audit.add_argument("out", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.command == "verify":
        verify(out)
        return
    subject, env_file = args.subject.resolve(), args.env_file.resolve()
    git = ["git", "-C", str(subject)]
    if subprocess.check_output([*git, "status", "--porcelain"]).strip():
        raise RuntimeError("Use a clean product checkout")
    subject_sha = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    out.mkdir()
    files = out / "fixtures"
    if args.reuse_asgi:
        checkpoint = args.reuse_asgi.resolve()
        previous = json.loads((checkpoint / "metadata.json").read_text())
        if previous["subject_sha"] != subject_sha:
            raise RuntimeError("The checkpoint has a different product SHA")
        for name, digest in previous["fixture_files"].items():
            if sha((checkpoint / "fixtures" / name).read_bytes()) != digest:
                raise RuntimeError(f"Changed public fixture: {name}")
        shutil.copytree(checkpoint / "fixtures", files)
        shutil.copyfile(checkpoint / "asgi.json", out / "asgi.json")
        image_hashes = previous["image_sha_and_expected_label"]
    else:
        image_hashes = fixtures(files)
    sys.path.insert(0, str(subject))
    from app.config import Settings

    dump(
        out / "metadata.json",
        {
            "subject_sha": subject_sha,
            "asgi_from_checkpoint": (
                sha((out / "asgi.json").read_bytes()) if args.reuse_asgi else None
            ),
            "runner_sha256": sha(Path(__file__).read_bytes()),
            "browser_driver": subprocess.check_output(
                ["agent-browser", "--version"], text=True
            ).strip(),
            "python": platform.python_version(),
            "os": platform.platform(),
            "app_factory_defaults": effective(Settings(_env_file=None)),
            "env_file_settings": effective(Settings(_env_file=env_file)),
            "fixture_files": {
                str(p.relative_to(files)): sha(p.read_bytes())
                for p in files.rglob("*")
                if p.is_file()
            },
            "image_sha_and_expected_label": image_hashes,
            "os_page_cache_controlled": False,
        },
    )
    if not args.reuse_asgi:
        run_asgi(out, env_file, files, image_hashes)
    run_web(out, env_file, files, subject)
    seal(out)


if __name__ == "__main__":
    main()
