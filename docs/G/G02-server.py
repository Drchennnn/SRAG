"""Run G02 against synthetic public originals and audit the real workspace traffic."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from time import perf_counter
from urllib.parse import parse_qs

import httpx
import nbformat
import uvicorn

from app.api.workspace import get_workspace_service
from app.config import Settings, get_settings
from app.main import create_app
from app.rag.source_vision import _prompt
from app.rag.workspace import make_workspace

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PATH = ROOT / "docs/F/样例/SignalTrialA7.json"
PUBLIC = ROOT / "docs/A/样例"
COURSES = "signals,signal-demo,signal-incomplete"
QUESTION = "刚才第二张图"
DIRECTED = "请描述图中实际可见的内容，尤其是坐标轴名称和单位（如有）。"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _record(target: Path, entry: dict) -> None:
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")


def _save_notebook(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cell = nbformat.v4.new_markdown_cell(text, id="evidence")
    nbformat.write(nbformat.v4.new_notebook(cells=[cell]), path)


def prepare(directory: Path) -> None:
    sample = json.loads(SAMPLE_PATH.read_text())
    for item in sample["documents"]:
        name, text = item["file_name"], item["text"]
        _save_notebook(directory / "complete" / name, text)
        if item["role"] == "lab":
            # Keep the named experiment, but no trial result, including the raw observations.
            incomplete = text.splitlines()[0].replace(" Trial: 0.18.", "")
            _save_notebook(directory / "incomplete" / name, incomplete)
            revision = text.replace("Trial: 0.18.", "Trial: 0.16.").replace(
                "(0.16, 0.20, 0.18, 0.18)", "(0.14, 0.18, 0.16, 0.16)"
            )
            _save_notebook(directory / "v2" / name, revision)
        else:
            _save_notebook(directory / "incomplete" / name, text)


def _audit_httpx(service, target: Path) -> None:
    """Observe the actual synchronous HTTP payload, then let the real transport send it."""
    handle = httpx.HTTPTransport.handle_request
    approved_png = json.loads((PUBLIC / "Signal-lab.ipynb").read_text())["cells"][1]["outputs"][2][
        "data"
    ]["image/png"]
    if isinstance(approved_png, list):
        approved_png = "".join(approved_png)
    approved_sha = _sha(base64.b64decode(approved_png))
    provider = service.vision.provider
    endpoint = httpx.URL(provider.base_url + "/services/aigc/multimodal-generation/generation")

    def observe(transport, request):
        if not service.vision.enabled or request.url != endpoint or request.method != "POST":
            _record(target, {"event": "blocked_unexpected_httpx_request"})
            raise RuntimeError("unexpected external request in public G02 audit")
        payload = json.loads(request.content)
        messages = payload["input"]["messages"]
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise RuntimeError("unexpected model message count")
        image, prompt = messages[0]["content"]
        if (
            set(image) != {"image"}
            or set(prompt) != {"text"}
            or prompt["text"] != _prompt("directed_question", DIRECTED)
            or not image["image"].startswith("data:image/png;base64,")
            or _sha(base64.b64decode(image["image"].split(",", 1)[1])) != approved_sha
        ):
            _record(target, {"event": "blocked_unexpected_vision_payload"})
            raise RuntimeError("unexpected model message content")
        started = perf_counter()
        record = {
            "event": "real_vision_attempt",
            "image_count": 1,
            "image_sha256": approved_sha,
            "message_count": 1,
            "normalized_question_sha256": _sha(DIRECTED.encode()),
            "model": payload["model"],
        }
        try:
            response = handle(transport, request)
            record["http_status"] = response.status_code
            return response
        except Exception as error:
            record["failure"] = type(error).__name__
            raise
        finally:
            record["response_headers_ms"] = round((perf_counter() - started) * 1000, 3)
            _record(target, record)

    httpx.HTTPTransport.handle_request = observe


def app_for(directory: Path, soffice: Path | None, real_vision: bool, env_file: Path | None):
    directory.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        _env_file=env_file if real_vision else None,
        course_root=directory / "courses",
        data_dir=directory / "data",
        sqlite_path=directory / "library.sqlite3",
        workspace_course_ids=COURSES,
        srag_soffice=soffice,
        source_vision_enabled=real_vision,
        deepseek_api_key="",
        qdrant_url="http://127.0.0.1:1",
        qdrant_api_key="",
        dashscope_timeout_seconds=20,
        **({} if real_vision else {"dashscope_api_key": ""}),
    )
    service = make_workspace(settings)
    _audit_httpx(service, directory / "provider.jsonl")
    (directory / "config.json").write_text(
        json.dumps(
            {
                "vision_enabled": real_vision,
                "vision_credential_present": bool(settings.dashscope_api_key),
                "vision_model": settings.dashscope_vision_model,
                "vision_timeout_seconds": settings.dashscope_timeout_seconds,
                "course_ids": COURSES.split(","),
                "data_and_cache_initially_empty": True,
                "process_restarted_between_probes": False,
                "os_page_cache_controlled": False,
            }
        ),
        encoding="utf-8",
    )
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_workspace_service] = lambda: service

    async def traced(scope, receive, send):
        path, method = scope.get("path", ""), scope.get("method")
        category = None
        if scope["type"] == "http" and path.startswith("/api/workspace/sessions/"):
            if method == "POST" and path.endswith("/turns"):
                category = "turn"
            elif method == "POST" and path.endswith("/import"):
                category = "import"
            elif method == "GET" and "/messages/" in path and "/evidence/" in path:
                category = "citation"
            elif method == "GET" and "/messages/" in path and "/media/" in path:
                category = "media"
        if category is None:
            return await app(scope, receive, send)
        started = perf_counter()
        body, outgoing = bytearray(), bytearray()
        size, status = 0, 0

        async def read():
            nonlocal size
            event = await receive()
            chunk = event.get("body", b"")
            size += len(chunk)
            if category == "turn":
                body.extend(chunk)
            return event

        async def respond(event):
            nonlocal status
            if event["type"] == "http.response.start":
                status = event["status"]
            if event["type"] == "http.response.body":
                outgoing.extend(event.get("body", b""))
                if not event.get("more_body"):
                    result = {"sha256": _sha(outgoing), "bytes": len(outgoing)}
                    if category != "media":
                        result = json.loads(outgoing)
                    _record(
                        directory / "http.jsonl",
                        {
                            "category": category,
                            "status": status,
                            "path": path,
                            "query": parse_qs(scope["query_string"].decode()),
                            "request": json.loads(body) if body else None,
                            "request_bytes": size,
                            "response": result,
                            "http_ms": round((perf_counter() - started) * 1000, 3),
                        },
                    )
            await send(event)

        await app(scope, read, respond)

    return traced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18831)
    parser.add_argument("--soffice", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--real-vision", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.data_dir / "fixtures")
        return
    if (args.data_dir / "library.sqlite3").exists():
        parser.error("G02 requires an empty workspace")
    prepare(args.data_dir / "fixtures")
    asgi = app_for(args.data_dir, args.soffice, args.real_vision, args.env_file)
    uvicorn.run(asgi, host="127.0.0.1", port=args.port, log_level="warning", lifespan="off")


if __name__ == "__main__":
    main()
