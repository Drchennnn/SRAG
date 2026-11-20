"""Run G19 against a frozen checkout using only the independent synthetic fixture."""

import argparse
import base64
import hashlib
import json
from contextvars import ContextVar
from pathlib import Path
from time import perf_counter

import httpx
import uvicorn
from fixtures import COURSE, GUIDE, QUESTIONS, build
from starlette.responses import JSONResponse

from app.api.workspace import get_workspace_service
from app.config import Settings, get_settings
from app.main import create_app
from app.rag.parsers.local_notebook import parse_local_notebook
from app.rag.parsers.local_slides import parse_local_slides
from app.rag.source_vision import _DASHSCOPE_DEFAULT_BASE_URL, _DASHSCOPE_MULTIMODAL_PATH, _prompt
from app.rag.workspace import make_workspace
from app.stores.structure_snapshot import SnapshotRequest

TURN = ContextVar("g19_public_turn", default=None)


def append(path, data):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False) + "\n")


def install_guards(out, settings, service, files):
    sync = httpx.HTTPTransport.handle_request
    asynchronous = httpx.AsyncHTTPTransport.handle_async_request
    deepseek_url = httpx.URL(settings.deepseek_base_url.rstrip("/") + "/v1/chat/completions")
    vision_base = settings.dashscope_base_url.rstrip("/") or _DASHSCOPE_DEFAULT_BASE_URL
    vision_url = httpx.URL(vision_base + _DASHSCOPE_MULTIMODAL_PATH)
    paths = {
        path: int(path.parent.name[1:]) if path.parent.name != "fixtures" else 1 for path in files
    }
    allowed_text = {}
    allowed_images = set()
    for path, version in paths.items():
        parser = parse_local_notebook if path.suffix == ".ipynb" else parse_local_slides
        for source in parser(path).sources:
            allowed_text[(path.name, version, source.source_id)] = source.text
        if path.suffix == ".ipynb":
            for cell in json.loads(path.read_text())["cells"]:
                for entry in cell.get("outputs", []):
                    if image := entry.get("data", {}).get("image/png"):
                        allowed_images.add(hashlib.sha256(base64.b64decode(image)).hexdigest())

    def deny(reason):
        append(out / "provider.jsonl", {"turn": TURN.get(), "sent": False, "reason": reason})
        raise RuntimeError("G19 guard rejected a nonpublic request")

    def authorize(request, endpoint, key):
        required = f"Bearer {key}"
        if (
            not TURN.get()
            or request.url != endpoint
            or endpoint.scheme != "https"
            or request.method != "POST"
            or not key
            or request.headers.get("Authorization") != required
            or key.encode() in request.content
        ):
            deny("unexpected_destination_or_payload")
        return json.loads(request.content)

    def record(request, provider):
        return {
            "turn": TURN.get(),
            "sent": True,
            "provider": provider,
            "request_sha256": hashlib.sha256(request.content).hexdigest(),
        }

    def sync_guard(transport, request):
        payload = authorize(request, vision_url, settings.dashscope_api_key)
        turn = TURN.get()
        if (
            turn["case"] not in {"second", "pdf_visual"}
            or payload["model"] != settings.dashscope_vision_model
        ):
            deny("unexpected_vision_turn")
        (message,) = payload["input"]["messages"]
        attachment, prompt = message["content"]
        if prompt["text"] != _prompt("directed_question", QUESTIONS[turn["case"]]):
            deny("unexpected_vision_question")
        header, encoded = attachment["image"].split(",", 1)
        if header != "data:image/png;base64":
            deny("unexpected_vision_media")
        digest = hashlib.sha256(base64.b64decode(encoded, validate=True)).hexdigest()
        admitted = digest in allowed_images
        if not admitted and turn["case"] == "pdf_visual":
            snapshot = service.snapshots.read(SnapshotRequest((COURSE,), ()))
            docs = {doc.document_id: doc for doc in snapshot.documents if doc.file_name == GUIDE}
            for source in snapshot.sources:
                if source.document_id in docs and source.locator.get("page") == 4:
                    rendition = service._render_request(docs[source.document_id], source)
                    rendered = service.evidence.render_current(rendition)
                    if rendered.display and digest == rendered.display.byte_sha256:
                        admitted = True
        if not admitted:
            deny("not_a_public_fixture_image")
        event = record(request, "dashscope") | {"model": payload["model"], "image_sha256": digest}
        started = perf_counter()
        try:
            response = sync(transport, request)
            event["status_code"] = response.status_code
            return response
        except httpx.HTTPError as error:
            event["failure_type"] = type(error).__name__
            raise
        finally:
            event["headers_ms"] = round((perf_counter() - started) * 1000, 3)
            append(out / "provider.jsonl", event)

    async def async_guard(transport, request):
        payload = authorize(request, deepseek_url, settings.deepseek_api_key)
        if payload["model"] != settings.deepseek_chat_model:
            deny("unexpected_answer_model")
        system, user = payload["messages"]
        prompt = json.loads(user["content"])
        if (
            system["role"] != "system"
            or user["role"] != "user"
            or prompt.get("question") != TURN.get()["text"]
        ):
            deny("unexpected_answer_question")
        for supplied in prompt["evidence"]:
            source = supplied["source"]
            chosen = allowed_text.get(
                (supplied["file_name"], source["version"], source["source_id"])
            )
            if not chosen or not all(
                read["text"].strip().removesuffix("…") in chosen for read in supplied["reads"]
            ):
                deny("not_a_public_source_excerpt")
        event = record(request, "deepseek") | {"model": payload["model"], "model_input": prompt}
        started = perf_counter()
        try:
            response = await asynchronous(transport, request)
            event["status_code"] = response.status_code
            return response
        except httpx.HTTPError as error:
            event["failure_type"] = type(error).__name__
            raise
        finally:
            event["headers_ms"] = round((perf_counter() - started) * 1000, 3)
            append(out / "provider.jsonl", event)

    httpx.HTTPTransport.handle_request = sync_guard
    httpx.AsyncHTTPTransport.handle_async_request = async_guard


def build_app(out, env, soffice):
    manifest = build(out)
    files = [out / path for path in manifest["files"]]
    allowed_uploads = {path.read_bytes() for path in files}
    settings = Settings(
        _env_file=env,
        course_root=out / "courses",
        data_dir=out / "data",
        sqlite_path=out / "library.sqlite3",
        workspace_course_ids=COURSE,
        workspace_answer_enabled=True,
        workspace_retrieval_strategy="fts",
        source_vision_enabled=True,
        srag_soffice=soffice,
        jev_api_key="",
        qdrant_api_key="",
        qdrant_url="http://127.0.0.1:1",
    )
    if not settings.deepseek_api_key.strip() or not settings.dashscope_api_key.strip():
        raise RuntimeError("G19 requires configured DeepSeek and DashScope in the local dotenv")
    app = create_app()
    service = make_workspace(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_workspace_service] = lambda: service
    install_guards(out, settings, service, files)
    cases = {question: case for case, question in QUESTIONS.items()}

    async def guarded(scope, receive, send):
        path = scope.get("path", "")
        method = scope.get("method")
        if not path.startswith("/api/workspace/sessions"):
            return await app(scope, receive, send)
        writing = method == "POST" and path.rsplit("/", 1)[-1] in {"import", "turns"}
        recording = writing or method == "GET" and any(s in path for s in ("/evidence/", "/media"))
        if not recording:
            return await app(scope, receive, send)
        ledger = {"method": method, "path": path}
        replay = receive
        token = TURN.set(None)
        started = perf_counter()
        try:
            if writing:
                body = bytearray()
                while True:
                    chunk = await receive()
                    if chunk["type"] != "http.request":
                        return
                    body.extend(chunk.get("body", b""))
                    if len(body) > 1_000_000:
                        return await JSONResponse({"detail": "only synthetic input"}, 413)(
                            scope, receive, send
                        )
                    if not chunk.get("more_body"):
                        break

                async def replay():
                    data = bytes(body)
                    body.clear()
                    return {"type": "http.request", "body": data}

                if path.endswith("/import"):
                    if bytes(body) not in allowed_uploads:
                        return await JSONResponse({"detail": "only synthetic documents"}, 403)(
                            scope, receive, send
                        )
                else:
                    turn = json.loads(body)
                    case = cases.get(turn.get("text"))
                    if not case or turn.get("anchor") is not None or turn.get("kind") != "ask":
                        return await JSONResponse(
                            {"detail": "only ordinary fixture questions"}, 403
                        )(scope, receive, send)
                    TURN.set(
                        {
                            "case": case,
                            "text": turn["text"],
                            "client_turn_id": turn["client_turn_id"],
                        }
                    )
                    ledger["turn"] = turn | {"case": case}
            response = bytearray()

            async def capture(event):
                if event["type"] == "http.response.start":
                    ledger["status_code"] = event["status"]
                    ledger["content_type"] = (
                        dict(event["headers"]).get(b"content-type", b"").decode()
                    )
                elif event["type"] == "http.response.body":
                    response.extend(event.get("body", b""))
                    if not event.get("more_body"):
                        if ledger["content_type"].startswith("application/json"):
                            ledger["response"] = json.loads(response)
                        else:
                            ledger["bytes"] = len(response)
                            ledger["sha256"] = hashlib.sha256(response).hexdigest()
                        ledger["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
                        append(out / "http.jsonl", ledger)
                await send(event)

            return await app(scope, replay, capture)
        finally:
            TURN.reset(token)

    return guarded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--soffice", type=Path, required=True)
    args = parser.parse_args()
    uvicorn.run(
        build_app(args.out, args.env_file, args.soffice),
        host="127.0.0.1",
        port=args.port,
        log_level="warning",
    )
