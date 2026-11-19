"""Passively record real workspace requests and restrict all outbound probe traffic."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from contextvars import ContextVar
from pathlib import Path
from time import perf_counter
from urllib.parse import parse_qs

import httpx
import uvicorn
from PIL import Image

from app.api.workspace import get_workspace_service
from app.config import Settings, get_settings
from app.main import create_app
from app.rag.source_vision import _prompt
from app.rag.workspace import make_workspace

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.json").read_text())
CASE_BY_TEXT = {case["text"]: case for case in PROTOCOL["cases"] if not case.get("warm_repeat")}
CURRENT_QUESTION: ContextVar[str | None] = ContextVar("current_public_browser_question", default=None)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write(path: Path, row: dict) -> None:
    path.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n")


def color_pixels(image: Image.Image, rgb: list[int]) -> int:
    return sum(abs(r - rgb[0]) < 8 and abs(g - rgb[1]) < 8 and abs(b - rgb[2]) < 8
               for r, g, b in image.convert("RGB").resize((150, 80)).getdata())


def assert_only_expected_crop(image: bytes, case: dict) -> None:
    with Image.open(io.BytesIO(image)) as picture:
        side = case["expected_side"]
        other = "left" if side == "right" else "right"
        truth = PROTOCOL["figure_truth"]
        assert color_pixels(picture, truth[side]["rgb"]) > 75, "expected_crop_absent"
        assert color_pixels(picture, truth[other]["rgb"]) < 30, "neighbor_in_request"


def make_app(args):
    directory = args.out
    settings = Settings(
        _env_file=args.env_file,
        sqlite_path=directory / "library.sqlite3",
        data_dir=directory / "data",
        course_root=directory / "courses",
        workspace_course_ids=",".join(PROTOCOL["courses"]),
        source_vision_enabled=args.env_file is not None,
        dashscope_timeout_seconds=25,
        deepseek_api_key="",
        qdrant_api_key="",
        qdrant_url="http://127.0.0.1:1",
        srag_soffice=args.soffice,
        **({} if args.env_file else {"dashscope_api_key": ""}),
    )
    service = make_workspace(settings)
    original = httpx.HTTPTransport.handle_request
    provider = service.vision.provider
    endpoint = httpx.URL(provider.base_url + "/services/aigc/multimodal-generation/generation")
    if args.env_file and not provider.api_key:
        raise RuntimeError("missing configured real visual provider credential")

    def vision_guard(transport, request):
        question = CURRENT_QUESTION.get()
        case = CASE_BY_TEXT.get(question, {})
        started = perf_counter()
        entry = {"case": case.get("id"), "path_kind": "vision", "is_external": False}
        try:
            assert args.env_file and request.url == endpoint and request.method == "POST", "unexpected_endpoint"
            assert case.get("kind") == "pdf", "question_does_not_authorize_image"
            payload = json.loads(request.content)
            assert payload["model"] == provider.model, "unexpected_model"
            messages = payload["input"]["messages"]
            assert len(messages) == 1 and messages[0]["role"] == "user", "unexpected_message_shape"
            image, prompt = messages[0]["content"]
            assert set(image) == {"image"} and set(prompt) == {"text"}, "unexpected_text_or_image_shape"
            assert prompt["text"] == _prompt("directed_question", question), "question_does_not_match_browser_input"
            image_uri = image["image"]
            assert image_uri.startswith("data:image/png;base64,"), "unexpected_mimetype"
            picture = base64.b64decode(image_uri.split(",", 1)[1], validate=True)
            entry["image_sha256"] = sha(picture)
            assert_only_expected_crop(picture, case)
            entry.update(event="real_vision_attempt", model=payload["model"], is_external=True)
            response = original(transport, request)
            entry["http_status"] = response.status_code
            return response
        except Exception as error:
            entry["event"] = "real_vision_attempt" if entry["is_external"] else "blocked_before_egress"
            entry["failure"] = type(error).__name__
            if isinstance(error, AssertionError):
                entry["reason"] = str(error)
            raise
        finally:
            entry["headers_elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
            append(directory / "provider.jsonl", entry)

    async def reject_async_egress(_transport, _request):
        append(directory / "provider.jsonl", {"event": "blocked_before_egress", "path_kind": "async", "is_external": False})
        raise RuntimeError("The independent probe permits only the guarded sync image provider")

    httpx.HTTPTransport.handle_request = vision_guard
    httpx.AsyncHTTPTransport.handle_async_request = reject_async_egress
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_workspace_service] = lambda: service
    write(directory / "config.json", {
        "vision_enabled": args.env_file is not None,
        "vision_model": provider.model if args.env_file else None,
        "revision": args.revision,
        "soffice": args.soffice is not None,
    })

    async def traced(scope, receive, send):
        path, method = scope.get("path", ""), scope.get("method", "")
        category = None
        if path.startswith("/api/workspace/sessions/"):
            if method == "POST" and path.endswith("/turns"):
                category = "turn"
            elif method == "POST" and path.endswith("/import"):
                category = "import"
            elif method == "GET" and "/messages/" in path and "/media/" in path:
                category = "media"
            elif method == "GET" and "/messages/" in path and "/evidence/" in path:
                category = "citation"
        if category is None:
            return await app(scope, receive, send)

        chunks = []
        request_payload = None
        if category == "turn":
            while True:
                event = await receive()
                chunks.append(event)
                if not event.get("more_body"):
                    break
            try:
                request_payload = json.loads(b"".join(event.get("body", b"") for event in chunks))
            except (json.JSONDecodeError, ValueError):
                pass
        token = CURRENT_QUESTION.set((request_payload or {}).get("text"))
        sent = bytearray()
        status = 0
        started = perf_counter()

        async def replay():
            return chunks.pop(0) if chunks else await receive()

        async def respond(event):
            nonlocal status
            if event["type"] == "http.response.start":
                status = event["status"]
            if event["type"] == "http.response.body":
                sent.extend(event.get("body", b""))
                if not event.get("more_body"):
                    try:
                        response = json.loads(sent) if category != "media" else None
                    except (json.JSONDecodeError, ValueError):
                        response = None
                    media = {"sha256": sha(sent), "bytes": len(sent)} if category == "media" else None
                    if category == "media" and status == 200:
                        with Image.open(io.BytesIO(sent)) as picture:
                            media["dimensions"] = picture.size
                            media["panel_pixels"] = {
                                side: color_pixels(picture, truth["rgb"])
                                for side, truth in PROTOCOL["figure_truth"].items()
                            }
                        (directory / (media["sha256"] + ".png")).write_bytes(sent)
                    append(directory / "http.jsonl", {
                        "category": category,
                        "status": status,
                        "path": path,
                        "query": parse_qs(scope["query_string"].decode()),
                        "request": request_payload,
                        "response": media if category == "media" else response,
                        "elapsed_ms": round((perf_counter() - started) * 1000, 3),
                    })
            await send(event)

        try:
            await app(scope, replay, respond)
        finally:
            CURRENT_QUESTION.reset(token)

    return traced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--soffice")
    args = parser.parse_args()
    uvicorn.run(make_app(args), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
