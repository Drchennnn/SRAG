"""Record a public-fixture G18 browser replay with all provider transports disabled."""

import argparse
import base64
import json
import runpy
from hashlib import sha256
from pathlib import Path

import httpx
import uvicorn
from starlette.responses import JSONResponse

from app.api.workspace import get_workspace_service
from app.config import Settings, get_settings
from app.main import create_app
from app.rag import source_evidence
from app.rag.source_vision import VisionReply
from app.rag.workspace import make_workspace

COURSE = "srag-g18-public"
PDF = "Northpoint-Completion-Briefing.pdf"
PPTX = "Drum-Readings.pptx"
NOTEBOOK = "Drum-Trend.ipynb"


def fixtures(out):
    suite = Path(__file__).parent
    runpy.run_path(str(suite / "G10-probe.py"))["build"](out)
    runpy.run_path(str(suite / "G16-browser-fixtures.py"))["build"](out)
    paths = [out / "fixtures" / name for name in (f"v1/{PDF}", f"v2/{PDF}", PPTX, f"v1/{NOTEBOOK}")]
    return {path.read_bytes() for path in paths}


def replay_vision(service, out):
    notebook = json.loads((out / "fixtures/v1" / NOTEBOOK).read_text(encoding="utf-8"))
    image = base64.b64decode(notebook["cells"][0]["outputs"][1]["data"]["image/png"])
    service.vision.enabled = True
    service.vision._provider_configured = True
    service.vision.model = "g18-offline-fixture"
    service.vision.provider_name = "g18-offline-fixture"

    def read(*, image: bytes, media_type, purpose, question):
        assert image == saved_image and media_type == "image/png"
        assert purpose == "directed_question"
        append(out / "simulated-vision.jsonl", {
            "mode": "synthetic_provider_reply",
            "question": question,
            "input_sha256": sha256(image).hexdigest(),
        })
        return VisionReply("succeeded", "横轴为 Cycle，单位是 minutes（分钟）。", (), None, {})

    saved_image = image
    service.vision.provider.read = read


def append(path, record):
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")


def build(out, soffice, without_soffice):
    fixture_bytes = fixtures(out)
    settings = Settings(
        _env_file=None,
        course_root=out / "courses",
        data_dir=out / "data",
        sqlite_path=out / "library.sqlite3",
        workspace_course_ids=COURSE,
        dashscope_api_key="",
        deepseek_api_key="",
        jev_api_key="",
        qdrant_api_key="",
        qdrant_url="http://127.0.0.1:1",
        source_vision_enabled=False,
        workspace_answer_enabled=True,
        srag_soffice=soffice,
    )
    if without_soffice:
        source_evidence._soffice = lambda _settings: (None, "")

    def blocked(_transport, request):
        append(out / "provider.jsonl", {"scheme": request.url.scheme, "sent": False})
        raise RuntimeError("G18 offline browser replay blocked outgoing traffic")

    async def blocked_async(transport, request):
        blocked(transport, request)

    httpx.HTTPTransport.handle_request = blocked
    httpx.AsyncHTTPTransport.handle_async_request = blocked_async
    app = create_app()
    service = make_workspace(settings)
    replay_vision(service, out)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_workspace_service] = lambda: service

    async def traced(scope, receive, send):
        method, path = scope.get("method"), scope.get("path", "")
        if not path.startswith("/api/workspace/sessions/"):
            return await app(scope, receive, send)
        end = path.rsplit("/", 1)[-1]
        reads = method == "GET" and any(key in path for key in ("/evidence/", "/media"))
        writes = method == "POST" and end in {"turns", "import"}
        if not (reads or writes):
            return await app(scope, receive, send)

        record = {"method": method, "path": path}
        if writes:
            body = b""
            while True:
                event = await receive()
                if event.get("type") != "http.request":
                    return
                body += event.get("body", b"")
                if len(body) > 200_000:
                    return await JSONResponse({"detail": "Only public fixture inputs"}, 413)(
                        scope, receive, send
                    )
                if not event.get("more_body"):
                    break
            if end == "import" and body not in fixture_bytes:
                return await JSONResponse({"detail": "Only public fixture inputs"}, 400)(
                    scope, receive, send
                )
            if end == "turns":
                record["turn"] = json.loads(body)

            async def replay():
                nonlocal body
                event = {"type": "http.request", "body": body}
                body = b""
                return event
        else:
            replay = receive
        response_body = bytearray()

        async def capture(event):
            if event["type"] == "http.response.start":
                record["status"] = event["status"]
                headers = dict(event["headers"])
                record["content_type"] = headers.get(b"content-type", b"").decode()
            elif event["type"] == "http.response.body":
                response_body.extend(event.get("body", b""))
                if not event.get("more_body"):
                    if record["content_type"].startswith("application/json"):
                        record["response"] = json.loads(response_body)
                    else:
                        record["size"] = len(response_body)
                        record["sha256"] = sha256(response_body).hexdigest()
                    append(out / "http.jsonl", record)
            await send(event)

        return await app(scope, replay, capture)

    return traced


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--soffice", default=None)
    parser.add_argument("--without-soffice", action="store_true")
    args = parser.parse_args()
    uvicorn.run(
        build(args.out, args.soffice, args.without_soffice),
        host="127.0.0.1", port=args.port, log_level="warning",
    )
