"""Observe the G14 public baseline through production FastAPI and Chromium."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
from time import perf_counter

import httpx
import uvicorn

from app.api.workspace import get_workspace_service
from app.config import Settings, get_settings
from app.main import create_app
from app.rag.parsers.local_slides import parse_local_slides
from app.rag.workspace import make_workspace

BASELINE = "82c5a40d3de6224c238ab461bc5d670c5005ba39"
FIXTURE = runpy.run_path(str(Path(__file__).with_name("G14-fixtures.py")))


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append(path: Path, value) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False) + "\n")


def make_app(out: Path):
    outbound = out / "outbound.jsonl"
    outbound.touch(exist_ok=False)

    def block(interface):
        append(outbound, {"attempt": interface, "sent": False})
        raise RuntimeError("G14 baseline forbids outbound requests")

    def audit(event, args):
        if event == "socket.connect" and isinstance(args[1], tuple):
            block("socket")

    def sync_guard(transport, request):
        block("httpx_sync")

    async def async_guard(transport, request):
        block("httpx_async")

    sys.addaudithook(audit)
    httpx.HTTPTransport.handle_request = sync_guard
    httpx.AsyncHTTPTransport.handle_async_request = async_guard
    settings = Settings(
        _env_file=None,
        workspace_course_ids=FIXTURE["COURSE"],
        course_root=out / "courses",
        data_dir=out / "data",
        sqlite_path=out / "library.sqlite3",
        deepseek_api_key="g14-isolated-test-sentinel",
        dashscope_api_key="",
        qdrant_api_key="",
        source_vision_enabled=False,
    )
    service = make_workspace(settings)
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_workspace_service] = lambda: service
    write(out / "config.json", {
        "subject": BASELINE,
        "deepseek_key_present": bool(settings.deepseek_api_key),
        "deepseek_key_source": "isolated sentinel",
        "real_provider_allowed": False,
        "vision_enabled": settings.source_vision_enabled,
        "request_guards": ["httpx_sync", "httpx_async", "socket.connect"],
    })

    async def trace(scope, receive, send):
        method, path = scope.get("method"), scope.get("path", "")
        if not path.startswith("/api/workspace/sessions/"):
            return await app(scope, receive, send)
        log_request = (method, path.rsplit("/", 1)[1]) in {("POST", "turns"), ("PUT", "focus")}
        if not log_request and not (method == "GET" and "/evidence/" in path):
            return await app(scope, receive, send)

        request = bytearray()
        response = bytearray()
        status = 0
        start = perf_counter()

        async def observe_request():
            event = await receive()
            request.extend(event.get("body", b""))
            return event

        async def observe_response(event):
            nonlocal status
            if event["type"] == "http.response.start":
                status = event["status"]
            if event["type"] == "http.response.body":
                response.extend(event.get("body", b""))
                if not event.get("more_body"):
                    append(out / "http.jsonl", {
                        "operation": "citation" if method == "GET" else path.rsplit("/", 1)[1],
                        "status": status,
                        "request": json.loads(request) if request else None,
                        "response": json.loads(response),
                        "elapsed_ms": round((perf_counter() - start) * 1000, 3),
                    })
            await send(event)

        await app(scope, observe_request, observe_response)

    return trace


def verify(out: Path) -> None:
    original = out / FIXTURE["FILENAME"]
    parsed = parse_local_slides(original)
    originals = {item.locator["page"]: item for item in parsed.sources if item.kind == "page"}
    assert set(originals) == set(range(1, 9))
    originals_by_id = {item.source_id: item for item in originals.values()}
    for page, phrases in FIXTURE["EXPECTED"].items():
        assert all(phrase in originals[page].text for phrase in phrases)

    log = [json.loads(line) for line in (out / "http.jsonl").read_text().splitlines()]
    config = json.loads((out / "config.json").read_text())
    outbound = (out / "outbound.jsonl").read_text().splitlines()
    assert config["deepseek_key_present"] and not outbound
    turns = [item for item in log if item["operation"] == "turns"]
    assert len(turns) == 2
    assert any(item["operation"] == "focus" and item["status"] == 200 for item in log)
    results = []
    for idx, item in enumerate(turns):
        dom = json.loads((out / f"turn-{idx+1}.json").read_text())
        expected = FIXTURE["QUESTIONS"][idx]
        req = item["request"]
        response = item["response"]
        turn, message = response["turn"], response["message"]
        assert item["status"] == 200 and req["kind"] == "ask" and req["text"] == expected
        assert req.get("anchor") is None
        assert turn["status"] == "completed"
        assert message["message_id"] == dom["message_id"] and dom["focus"] == original.name
        trace = turn["trace"]
        assert trace["search_status"] == "no_text_anchor"
        assert len(trace["effective_scope"]["document_ids"]) == 1
        assert trace["effective_scope"]["course_ids"] == [FIXTURE["COURSE"]]
        candidates = trace["candidates"]
        assert len(candidates) == 8
        assert all(c["match_kind"] == "browse" for c in candidates)
        assert [c["reasons"][0]["code"] for c in candidates] == ["page_browse_fallback"] * 8
        reads = [read for evidence in message["sources"].values() for read in evidence["reads"]]
        assert [r["actual_source"]["source_id"] for r in reads] == ["page:1", "page:2"]
        texts = [block["text"] for block in message["blocks"] if block["type"] == "text"]
        assert len(texts) == 2
        for text, read in zip(texts, reads, strict=True):
            source = originals_by_id[read["actual_source"]["source_id"]]
            assert text.split("\n", 1)[1] == source.text.strip()
            assert read["method"] == "native_text" and read["status"] == "succeeded"
        assert all(
            block["read_status"] == "not_attempted"
            for block in message["blocks"] if block["type"] == "source_image"
        )
        results.append({
            "request": req, "response": response, "dom": dom, "text_blocks": texts,
            "elapsed_ms": item["elapsed_ms"],
        })
    assert results[0]["text_blocks"] == results[1]["text_blocks"]
    citations = [item for item in log if item["operation"] == "citation"]
    assert len(citations) == 2 and all(item["status"] == 200 for item in citations)
    assert all(item["response"]["source"]["source_id"] == "page:1" for item in citations)
    write(out / "G14-browser-baseline.json", {
        "fixture": original.name,
        "fixture_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "pages": len(originals),
        "required_phrases_by_page": FIXTURE["EXPECTED"],
        "browser": json.loads((out / "browser.json").read_text()),
        "config": config,
        "outbound_attempts": len(outbound),
        "outbound_sent": 0,
        "turns": results,
        "opened_citations": citations,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--port", type=int)
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.port:
        uvicorn.run(make_app(args.out), host="127.0.0.1", port=args.port, log_level="warning")
    else:
        verify(args.out)
