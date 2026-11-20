"""Capture the staged gold and browser runs on a public-only SRAG copy."""

from __future__ import annotations

import argparse
import json
import re
import runpy
from contextvars import ContextVar
from pathlib import Path
from time import perf_counter
from urllib.parse import parse_qs

import httpx
import uvicorn
from starlette.responses import JSONResponse

from app.api.workspace import get_workspace_service
from app.config import Settings, get_settings
from app.main import create_app
from app.rag.parsers.local_slides import parse_local_slides
from app.rag.workspace import make_workspace
from app.schemas.local import SourceKey
from app.schemas.turn_evidence import SourceRevision, TurnEvidence, TurnRead

FIXTURE = runpy.run_path(str(Path(__file__).with_name("G16-browser-fixtures.py")))
CONTEXT: ContextVar[str | None] = ContextVar("public_g16_turn", default=None)
CLIENT_TURN: ContextVar[str | None] = ContextVar("public_g16_client_turn_id", default=None)
GOLD_PAGES = (3, 5, 7, 8)


def write(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append(path: Path, data) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def settings_for(out: Path, env: Path | None) -> Settings:
    options = {
        "_env_file": env,
        "course_root": out / "courses",
        "data_dir": out / "data",
        "sqlite_path": out / "library.sqlite3",
        "workspace_course_ids": FIXTURE["COURSE"],
        "dashscope_api_key": "",
        "jev_api_key": "",
        "qdrant_api_key": "",
        "source_vision_enabled": False,
        "qdrant_url": "http://127.0.0.1:1",
        "workspace_answer_enabled": True,
    }
    if env is None:
        options["deepseek_api_key"] = ""
    settings = Settings(**options)
    if env is not None and not settings.deepseek_api_key.strip():
        raise RuntimeError("DeepSeek isn't configured for this public probe")
    return settings


def prepare_transport(out: Path, settings: Settings) -> None:
    original = httpx.AsyncHTTPTransport.handle_async_request
    endpoint = httpx.URL(settings.deepseek_base_url.rstrip("/") + "/v1/chat/completions")
    allowed_models = {settings.deepseek_chat_model, settings.deepseek_query_model}
    path = out / "provider.jsonl"
    public_text = {
        (version, item.source_id): item.text
        for version in (1, 2)
        for item in parse_local_slides(
            out / "fixtures" / f"v{version}" / FIXTURE["FILENAME"]
        ).sources
        if item.kind == "page"
    }

    def blocked(reason: str) -> None:
        append(path, {"turn": CONTEXT.get(), "sent": False, "blocked": reason})
        raise RuntimeError("G16 public replay blocked a provider request")

    def guard_sync(transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        blocked("unexpected_sync_provider")

    async def guard_async(transport: httpx.AsyncHTTPTransport, request: httpx.Request):
        if not (CONTEXT.get() and settings.deepseek_api_key.strip()):
            blocked("no_public_turn_or_key")
        if request.method != "POST" or request.url != endpoint or request.url.scheme != "https":
            blocked("unexpected_endpoint")
        if request.headers.get("Authorization") != f"Bearer {settings.deepseek_api_key}":
            blocked("unexpected_authentication")
        payload = json.loads(request.content)
        messages = payload.get("messages", [])
        if (
            payload.get("model") not in allowed_models
            or [item.get("role") for item in messages] != ["system", "user"]
            or any(not isinstance(item.get("content"), str) for item in messages)
        ):
            blocked("unexpected_model_or_nontest_payload")
        content = "\n".join(message["content"] for message in messages)
        request_data = json.loads(messages[1]["content"])
        question = FIXTURE["QUESTIONS"]["first" if CONTEXT.get() == "fixed_gold" else CONTEXT.get()]
        if request_data.get("question") != question:
            blocked("unexpected_question")
        if settings.deepseek_api_key in content:
            blocked("credential_in_content")
        input_evidence = []
        fixture_reads = request_data.get("evidence")
        if not isinstance(fixture_reads, list) or not fixture_reads:
            blocked("missing_public_evidence")
        for item in fixture_reads:
            source = item["source"]
            key = (source["version"], source["source_id"])
            reads = item["reads"]
            if (
                item["file_name"] != FIXTURE["FILENAME"]
                or key not in public_text
                or not reads
                or any(
                    read["method"] != "native_text"
                    or not read["text"].strip()
                    or read["text"].removesuffix("…") not in public_text[key]
                    for read in reads
                )
            ):
                blocked("nonfixture_source_text")
            input_evidence.append({"id": item["id"], "source": source, "locator": item["locator"]})
        started = perf_counter()
        evidence_markers = {
            str(page): [phrase for phrase in FIXTURE["DATES"][page] if phrase in content]
            for page in GOLD_PAGES
        }
        record = {
            "turn": CONTEXT.get(),
            "client_turn_id": CLIENT_TURN.get(),
            "sent": True,
            "model": payload["model"],
            "request_sha256": FIXTURE["sha256"](request.content),
            "evidence_markers": evidence_markers,
            "input_evidence": input_evidence,
            "has_previous_context": bool(request_data["previous_context"]),
            "generation_mode": request_data["mode"],
            "message_roles": [message["role"] for message in messages],
        }
        try:
            response = await original(transport, request)
            record["status"] = response.status_code
            return response
        except httpx.TransportError as error:
            record["failure_type"] = type(error).__name__
            raise
        finally:
            record["headers_ms"] = round((perf_counter() - started) * 1000, 3)
            append(path, record)

    httpx.AsyncHTTPTransport.handle_async_request = guard_async
    httpx.HTTPTransport.handle_request = guard_sync


def run_gold(out: Path, settings: Settings):
    from app.rag.workspace_generation import generate_response

    path = out / "fixtures/v1" / FIXTURE["FILENAME"]
    parsed = parse_local_slides(path)
    supplied = {}
    for source in parsed.sources:
        if source.kind != "page" or source.locator["page"] not in GOLD_PAGES:
            continue
        key = SourceKey(document_id="public-g16-gold", version=1, source_id=source.source_id)
        identity = f"ev-{len(supplied) + 1}"
        supplied[identity] = TurnEvidence(
            evidence_id=identity,
            revision=SourceRevision(
                source=key,
                course_id=FIXTURE["COURSE"],
                publication_epoch=1,
                content_sha256=FIXTURE["sha256"](path.read_bytes()),
            ),
            file_name=FIXTURE["FILENAME"],
            locator={"page": source.locator["page"]},
            read_status="succeeded",
            reads=[
                TurnRead(
                    actual_source=key, method="native_text", status="succeeded", text=source.text
                )
            ],
        )
    token = CONTEXT.set("fixed_gold")
    try:
        answer = generate_response(settings, FIXTURE["QUESTIONS"]["first"], supplied, mode="ask")
    finally:
        CONTEXT.reset(token)
    statements = [
        {"text": item.text, "citations": list(item.citations)} for item in answer.statements
    ]
    write(
        out / "gold.json",
        {
            "provided_evidence": {
                name: item.model_dump(mode="json") for name, item in supplied.items()
            },
            "response": {
                "status": answer.status,
                "statements": statements,
                "missing": answer.missing,
                "trace": answer.trace,
            },
        },
    )
    validate_generation_status(answer.status, answer.missing, allow_unprovided=True)
    verify_gold("\n".join(item["text"] for item in statements))
    verify_cited_facts(statements, {name: item.locator["page"] for name, item in supplied.items()})


def build_browser_app(out: Path, settings: Settings):
    expected_pdf = {
        (out / "fixtures" / version / FIXTURE["FILENAME"]).read_bytes() for version in ("v1", "v2")
    }
    service = make_workspace(settings)
    app = create_app()
    app.dependency_overrides[get_workspace_service] = lambda: service
    app.dependency_overrides[get_settings] = lambda: settings
    case_by_question = {text: case for case, text in FIXTURE["QUESTIONS"].items()}

    async def traced(scope, receive, send):
        method, path = scope.get("method"), scope.get("path", "")
        if not path.startswith("/api/workspace/sessions/"):
            return await app(scope, receive, send)
        operation = path.rsplit("/", 1)[1]
        label = None
        if method == "POST" and operation in {"turns", "import"}:
            body = bytearray()
            while True:
                event = await receive()
                if event.get("type") != "http.request":
                    return
                body.extend(event.get("body", b""))
                if len(body) > 32768:
                    return await JSONResponse({"detail": "Only public fixture inputs"}, 413)(
                        scope, receive, send
                    )
                if not event.get("more_body"):
                    break
            request = json.loads(body) if operation == "turns" else None
            if operation == "turns":
                label = case_by_question.get(request.get("text"))
                admitted = (
                    label is not None and request.get("kind") == "ask" and not request.get("anchor")
                )
            else:
                query = parse_qs(scope["query_string"].decode())
                admitted = (
                    bytes(body) in expected_pdf
                    and query.get("filename") == [FIXTURE["FILENAME"]]
                    and query.get("course_id") == [FIXTURE["COURSE"]]
                )
            if not admitted:
                return await JSONResponse({"detail": "Only public fixture inputs"}, 403)(
                    scope, receive, send
                )
        elif method == "GET" and "/evidence/" in path:
            operation, request, event = "citation", None, None
        else:
            return await app(scope, receive, send)

        started = perf_counter()
        response = bytearray()
        status = 0

        async def replay_body():
            nonlocal event
            if event is not None:
                event = None
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        async def receive_result(part):
            nonlocal status
            if part["type"] == "http.response.start":
                status = part["status"]
            if part["type"] == "http.response.body":
                response.extend(part.get("body", b""))
                if not part.get("more_body"):
                    append(
                        out / "http.jsonl",
                        {
                            "operation": operation,
                            "path": path,
                            "case": label,
                            "status": status,
                            "request": request,
                            "elapsed_ms": round((perf_counter() - started) * 1000, 3),
                            "response": json.loads(response),
                        },
                    )
            await send(part)

        token = CONTEXT.set(label)
        local_turn = CLIENT_TURN.set(request.get("client_turn_id") if label else None)
        try:
            return await app(scope, replay_body, receive_result)
        finally:
            CLIENT_TURN.reset(local_turn)
            CONTEXT.reset(token)

    return traced


MONTHS = {"January": 1, "May": 5, "June": 6, "July": 7, "August": 8, "September": 9, "December": 12}
_CHINESE_COUNTS = dict(zip("零一二三四五六七八九十", range(11), strict=True)) | {"两": 2}
_ENGLISH_COUNTS = dict(
    zip("zero one two three four five six seven eight nine ten".split(), range(11), strict=True)
)
_COUNT = (
    r"(?<![\d第])(?P<count>\d{1,2}(?!\d)|[一二两三四五六七八九]?十[一二三四五六七八九]?"
    r"|[零一二两三四五六七八九])"
)
_ROUND = (
    r"(?:(?:(?:结业|毕业|授奖)(?:[/／](?:结业|毕业|授奖))?\s*)?(?:轮次|批次|档期)"
    r"|(?:结业|毕业|授奖)(?:时间(?:选项)?|日期|选项))"
)
_OPTION_COUNTS = (
    re.compile(rf"{_ROUND}\s*(?:一共|总共|共有|共计|有|分成|分为|是|为)\s*{_COUNT}"),
    re.compile(rf"{_COUNT}\s*(?:个|种|项|场|类|档)?\s*(?:可选|可供选择的|正式|不同)?\s*{_ROUND}"),
    re.compile(
        r"\b(?P<count>\d{1,2}|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(?:different\s+)?(?:award|completion|graduation)\s+(?:rounds?|options?|windows?)\b",
        re.IGNORECASE,
    ),
)
_AFTER_EXCLUSION = re.compile(r"错过|排除|扣除|剔除|剩(?:余|下)|还剩|remaining|\bexcluding\b", re.I)


def verify_option_count(text: str, expected_count: int) -> None:
    for clause in re.split(r"[。！？；;\n]", text):
        for pattern in _OPTION_COUNTS:
            for match in pattern.finditer(clause):
                if _AFTER_EXCLUSION.search(clause[: match.start()]):
                    continue
                raw = match["count"].casefold()
                if raw.isdecimal():
                    count = int(raw)
                elif "十" in raw and raw != "十":
                    tens, ones = raw.split("十", 1)
                    count = 10 * _CHINESE_COUNTS[tens or "一"] + _CHINESE_COUNTS[ones or "零"]
                else:
                    count = (_CHINESE_COUNTS | _ENGLISH_COUNTS)[raw]
                assert count == expected_count, (
                    f"stated {count} total options, expected {expected_count}",
                    clause.strip(),
                )


def has_date(text: str, date: str) -> bool:
    month, day = date.split()
    number = MONTHS[month]
    return date in text or bool(re.search(rf"(?<!\d)0?{number}\s*月\s*0?{day}(?!\d)", text))


def verify_gold(text: str) -> None:
    verify_option_count(text, len(FIXTURE["AWARD_ROUNDS"]))
    assert len(re.findall(r"[\u3400-\u9fff]", text)) >= 20
    assert all(has_date(text, date) for dates in FIXTURE["DATES"].values() for date in dates)
    assert "雇主" in text or "实习单位" in text or "用人单位" in text or "公司" in text
    assert re.search(r"(?:无法|不能|不可能|没法|来不及|无法在).*?(?:正式|证明|证书)", text)
    assert re.search(
        r"(?:资格.{0,5}(?:信|函|证明)|eligibility letter).{0,25}"
        r"(?:不(?:是|等同|能代替)|不能).{0,25}(?:正式|证书|证明|award certificate)",
        text,
        re.IGNORECASE,
    )


def validate_generation_status(status: str, missing: str, *, allow_unprovided: bool) -> None:
    if status == "partial" and allow_unprovided:
        # The fixture cannot establish third-party decisions or administrative details.
        assert re.search(
            r"申请|处理|雇主|实习单位|推迟|eligibility letter|资格", missing, re.IGNORECASE
        )
    else:
        assert status == "succeeded"


def verify_cited_facts(statements: list[dict], evidence_pages: dict[str, int]) -> None:
    for page in GOLD_PAGES:
        texts = "\n".join(
            item["text"]
            for item in statements
            if any(evidence_pages[citation] == page for citation in item["citations"])
        )
        assert all(has_date(texts, date) for date in FIXTURE["DATES"][page]), (page, texts)


def cited_records(message: dict) -> dict:
    cited = {
        citation
        for block in message["blocks"]
        if block["type"] == "text"
        for citation in block["citations"]
    }
    sources = message["sources"]
    assert cited and cited <= sources.keys()
    records = {key: sources[key] for key in cited}
    for item in records.values():
        source = item["revision"]["source"]
        assert any(
            read["actual_source"] == source and read["status"] in {"succeeded", "partial"}
            for read in item["reads"]
        )
    return records


def texts(message: dict) -> str:
    return "\n".join(block["text"] for block in message["blocks"] if block["type"] == "text")


def verify_browser(out: Path, subject: str, *, live: bool) -> None:
    actual = load_events(out / "http.jsonl")
    provider = load_events(out / "provider.jsonl")
    evidence = {
        "subject": subject,
        "live": live,
        "fixtures": json.loads((out / "fixtures.json").read_text()),
        "browser": json.loads((out / "browser.json").read_text()),
        "gold": json.loads((out / "gold.json").read_text()) if live else None,
        "provider": provider,
        "http": actual,
        "browser_turns": [
            json.loads((out / f"turn-{step}.json").read_text()) for step in range(1, 6)
        ],
        "browser_readers": {
            path.stem: json.loads(path.read_text())
            for path in sorted(out.glob("reader-*-ev-*.json"))
        },
    }
    write(out / "G16-browser-result.json", evidence)
    imports = [entry for entry in actual if entry["operation"] == "import"]
    assert [entry["response"]["version"] for entry in imports] == [1, 2]
    assert len({entry["response"]["document_id"] for entry in imports}) == 1
    turns = [entry for entry in actual if entry["operation"] == "turns"]
    assert all(event["status"] == 200 for event in turns)
    assert all(
        event["request"]["kind"] == "ask" and not event["request"].get("anchor") for event in turns
    )
    if not live:
        assert not provider
        first = turns[0]["response"]["message"]
        assert any("摘录" in (block.get("text") or "") for block in first["blocks"])
        return

    assert evidence["browser_turns"][2]["focus"].startswith("第 2 页")
    assert evidence["browser_turns"][3]["focus"].startswith("第 5 页")
    assert [turn["case"] for turn in turns] == ["first", "follow", "first", "current", "compare"]
    assert all(event["sent"] and event["status"] == 200 for event in provider)
    assert {event["turn"] for event in provider} >= {
        "fixed_gold",
        "first",
        "follow",
        "current",
        "compare",
    }
    for turn in turns:
        case = turn["case"]
        response = turn["response"]
        result, message = response["turn"], response["message"]
        assert result["status"] == "completed"
        trace = result["trace"]
        observed = [
            event
            for event in provider
            if event["client_turn_id"] == turn["request"]["client_turn_id"]
        ]
        assert trace["external_call_count"] >= 1 and observed
        records = cited_records(message)
        answer = texts(message)
        if case == "compare":
            assert trace["route"] == "version_comparison"
            versions = {record["revision"]["source"]["version"] for record in records.values()}
            assert versions == {1, 2}
            dates = (*FIXTURE["DATES"][5], *FIXTURE["CURRENT_DATES"])
            assert all(has_date(answer, date) for date in dates)
        else:
            assert trace["route"] == "generated_answer"
        missing = "\n".join(
            block["text"] for block in message["blocks"] if block.get("code") == "missing_evidence"
        )
        validate_generation_status(
            trace["generation"]["status"], missing, allow_unprovided=case == "first"
        )
        assert trace["generation"]["provider"] == "deepseek"
        if case == "first":
            verify_gold(answer)
            assert {record["locator"]["page"] for record in records.values()} >= set(GOLD_PAGES)
            verify_cited_facts(
                [block for block in message["blocks"] if block["type"] == "text"],
                {name: item["locator"]["page"] for name, item in records.items()},
            )
            assert any(all(event["evidence_markers"].values()) for event in observed)
        elif case == "follow":
            verify_option_count(answer, len(FIXTURE["AWARD_ROUNDS"]))
            assert len(re.findall(r"[\u3400-\u9fff]", answer)) >= 20
            assert len(re.findall(r"[。！？]", answer)) == 2
            assert has_date(answer, "July 15") or "实习" in answer
            assert trace["reading"]["mode"] == "previous_citations"
            assert any(event["has_previous_context"] for event in observed)
        elif case == "current":
            assert all(has_date(answer, date) for date in FIXTURE["CURRENT_DATES"])
            assert {
                (record["revision"]["source"]["version"], record["locator"]["page"])
                for record in records.values()
            } == {(2, 5)}
    opened = [entry for entry in actual if entry["operation"] == "citation"]
    assert all(entry["status"] == 200 for entry in opened)

    def key(item):
        return item["response"]["source"]["version"], item["response"]["locator"]["page"]

    assert {key(entry) for entry in opened} >= {(1, page) for page in GOLD_PAGES} | {(2, 5)}
    compare_message = turns[-1]["response"]["message"]["message_id"]
    assert {key(entry) for entry in opened if f"/messages/{compare_message}/" in entry["path"]} == {
        (1, 5),
        (2, 5),
    }
    compare_readers = [
        reader
        for name, reader in evidence["browser_readers"].items()
        if name.startswith("reader-5-")
    ]
    assert {reader["location"] for reader in compare_readers} == {"历史版本", "当前版本"}
    for opening in opened:
        version, page = key(opening)
        if page == 5:
            dates = FIXTURE["DATES"][5] if version == 1 else FIXTURE["CURRENT_DATES"]
            assert all(date in opening["response"]["text"] for date in dates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--env-file", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gold", action="store_true")
    mode.add_argument("--port", type=int)
    mode.add_argument("--verify", type=str, metavar="SUBJECT")
    args = parser.parse_args()
    if args.verify:
        verify_browser(args.out, args.verify, live=args.env_file is not None)
        raise SystemExit(0)
    settings = settings_for(args.out, args.env_file)
    prepare_transport(args.out, settings)
    if args.gold:
        run_gold(args.out, settings)
    else:
        uvicorn.run(build_browser_app(args.out, settings), host="127.0.0.1", port=args.port)
