"""Synthetic public G05 workspace, traffic collector, and baseline assertions."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path
from time import perf_counter
from urllib.parse import parse_qs

import httpx
import nbformat
import uvicorn
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches

from app.api.workspace import get_workspace_service
from app.config import Settings, get_settings
from app.main import create_app
from app.rag.source_vision import _prompt
from app.rag.workspace import make_workspace

COURSES = ["harbor-panels", "pulse-evidence"]
BOOK = "HarborTrace.ipynb"
LECTURE = "PulseGate-lecture.ipynb"
TABLE = "PulseGate-limits.pptx"
OUTPUT = "PulseGate-run.ipynb"
LEFT = "cell:1:output:0"
RIGHT = "cell:1:output:1"
FIRST = "HarborTrace.ipynb 第 2 个 Cell 右侧琥珀子图的横轴刻度单位是什么？"
FOLLOW = "刚才第二张图"
COMPARE = (
    "请同时比较上一版和当前版 HarborTrace.ipynb 的第 2 个 Cell，概述两版琥珀色子图上的刻度单位。"
)
TWO = "What is the definition of a pulse gate, and how did the held-out experiment validate it?"
THREE = (
    TWO + " Use the threshold from the separate limit table and compare it with the saved output."
)
DIRECTED = "请描述图中实际可见的内容，尤其是坐标轴名称和单位（如有）。"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def append(path: Path, payload) -> None:
    with path.open("a") as output:
        output.write(json.dumps(payload, ensure_ascii=False) + "\n")


def panel(side: str, unit: str, color: str) -> str:
    image = Image.new("RGB", (640, 350), "#fffef7")
    draw = ImageDraw.Draw(image)
    large = ImageFont.load_default(size=31)
    small = ImageFont.load_default(size=22)
    draw.text((38, 18), f"{side.upper()} HARBOR PANEL", fill=color, font=large)
    draw.line([(93, 72), (93, 258), (578, 258)], fill="black", width=3)
    draw.line([(93, 237), (230, 183), (389, 205), (565, 124)], fill=color, width=6)
    for i, x in enumerate((93, 230, 389, 565)):
        draw.line([(x, 258), (x, 270)], fill="black", width=2)
        draw.text((x - 5, 274), str(i * 5), fill="black", font=small)
    draw.text((224, 310), f"Elapsed time [{unit}]", fill="black", font=small)
    draw.text((6, 145), "Flow [L/s]", fill="black", font=small)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode()


def notebook(path: Path, cells) -> None:
    nbformat.write(nbformat.v4.new_notebook(cells=cells), path)


def build(directory: Path) -> dict:
    fixtures = directory / "fixtures"
    fixtures.mkdir()
    image_hashes = {}
    for version, unit in ((1, "seconds"), (2, "minutes")):
        version_dir = fixtures / f"v{version}"
        version_dir.mkdir()
        plots = [panel("left", "hours", "#0055aa"), panel("right", unit, "#a86100")]
        outputs = [
            nbformat.v4.new_output("display_data", data={"image/png": encoded}) for encoded in plots
        ]
        notebook(
            version_dir / BOOK,
            [
                nbformat.v4.new_markdown_cell("# Harbor pump study", id="summary"),
                nbformat.v4.new_code_cell(
                    f"# Harbor panels: blue left then amber right\n# Edition {version}",
                    id="panels",
                    execution_count=1,
                    outputs=outputs,
                ),
            ],
        )
        image_hashes[str(version)] = {
            key: sha(base64.b64decode(data)) for key, data in zip((LEFT, RIGHT), plots, strict=True)
        }

    lecture = (
        "Definition: A pulse gate passes only pulses above its operating threshold.\n"
        "Experiment-ID: TidalRunQ9."
    )
    log = (
        "Experiment-ID: TidalRunQ9. Split: held-out. Metric: MAE. Unit: amplitude. "
        "Baseline: 0.42. Trial: 0.29. Samples: 7."
    )
    notebook(
        fixtures / LECTURE,
        [nbformat.v4.new_markdown_cell(lecture, id="definition")],
    )
    notebook(
        fixtures / OUTPUT,
        [
            nbformat.v4.new_code_cell(
                "run_measurements()",
                id="run",
                execution_count=1,
                outputs=[nbformat.v4.new_output("stream", name="stdout", text=log)],
            )
        ],
    )
    slide_deck = Presentation()
    slide = slide_deck.slides.add_slide(slide_deck.slide_layouts[6])
    table = slide.shapes.add_table(2, 3, Inches(0.65), Inches(1.2), Inches(8), Inches(1.2))
    table.name = "Pulse gate operating limit table"
    rows = (("Series", "Gate setting (mA)", "Dataset"), ("Pulse gate", "17.5", "held-out"))
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            table.table.cell(row_index, col_index).text = value
    slide_deck.save(fixtures / TABLE)
    manifest = {
        "image_sha256": image_hashes,
        "image_units": {"1": "seconds", "2": "minutes"},
        "definition": lecture,
        "run_output": log,
        "table_cells": rows,
        "questions": dict(first=FIRST, follow=FOLLOW, compare=COMPARE, two=TWO, three=THREE),
        "fixture_sha256": {
            str(p.relative_to(fixtures)): sha(p.read_bytes())
            for p in sorted(fixtures.rglob("*"))
            if p.is_file()
        },
    }
    write(directory / "fixtures.json", manifest)
    return manifest


def make_app(directory: Path, args):
    manifest = build(directory)
    settings = Settings(
        _env_file=args.env_file if args.real_vision else None,
        sqlite_path=directory / "library.sqlite3",
        data_dir=directory / "data",
        course_root=directory / "courses",
        workspace_course_ids=",".join(COURSES),
        source_vision_enabled=args.real_vision,
        dashscope_timeout_seconds=25,
        deepseek_api_key="",
        qdrant_api_key="",
        qdrant_url="http://127.0.0.1:1",
        srag_soffice=args.soffice,
        **({} if args.real_vision else {"dashscope_api_key": ""}),
    )
    service = make_workspace(settings)
    original = httpx.HTTPTransport.handle_request
    provider = service.vision.provider
    endpoint = httpx.URL(provider.base_url + "/services/aigc/multimodal-generation/generation")
    expected = manifest["image_sha256"]["1"][RIGHT]

    def vision_only(transport, request):
        if not args.real_vision or request.url != endpoint or request.method != "POST":
            append(directory / "provider.jsonl", {"event": "blocked_non_probe_request"})
            raise RuntimeError("G05 allows only the explicitly selected public image")
        payload = json.loads(request.content)
        messages = payload["input"]["messages"]
        image, prompt = messages[0]["content"]
        raw = image["image"]
        assert len(messages) == 1 and messages[0]["role"] == "user"
        assert set(image) == {"image"} and set(prompt) == {"text"}
        assert prompt["text"] == _prompt("directed_question", DIRECTED)
        assert raw.startswith("data:image/png;base64,")
        assert sha(base64.b64decode(raw.split(",", 1)[1])) == expected
        started = perf_counter()
        entry = {
            "event": "real_vision_attempt",
            "image_sha256": expected,
            "model": payload["model"],
        }
        try:
            response = original(transport, request)
            entry["http_status"] = response.status_code
            return response
        except Exception as error:
            entry["failure"] = type(error).__name__
            raise
        finally:
            entry["http_headers_ms"] = round((perf_counter() - started) * 1000, 3)
            append(directory / "provider.jsonl", entry)

    httpx.HTTPTransport.handle_request = vision_only
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_workspace_service] = lambda: service
    write(directory / "config.json", {"vision_enabled": args.real_vision, "courses": COURSES})

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
        sent = bytearray()
        received = bytearray()
        status = 0

        async def read():
            event = await receive()
            if category == "turn":
                received.extend(event.get("body", b""))
            return event

        async def respond(event):
            nonlocal status
            if event["type"] == "http.response.start":
                status = event["status"]
            if event["type"] == "http.response.body":
                sent.extend(event.get("body", b""))
                if not event.get("more_body"):
                    append(
                        directory / "http.jsonl",
                        {
                            "category": category,
                            "status": status,
                            "path": path,
                            "query": parse_qs(scope["query_string"].decode()),
                            "request": json.loads(received) if received else None,
                            "response": {"sha256": sha(sent), "bytes": len(sent)}
                            if category == "media"
                            else json.loads(sent),
                        },
                    )
            await send(event)

        await app(scope, read, respond)

    return traced


def load(directory: Path, name: str):
    return json.loads((directory / (name + ".json")).read_text())


def candidate_keys(turn):
    return {item["source"]["source_id"] for item in turn["trace"]["candidates"]}


def verify(directory: Path) -> None:
    manifest = load(directory, "fixtures")
    config = load(directory, "config")
    with (directory / "http.jsonl").open() as log:
        http = [json.loads(line) for line in log]
    turns = [event for event in http if event["category"] == "turn"]
    assert len(turns) == (7 if config["vision_enabled"] else 6)
    assert all(event["status"] == 200 for event in turns)
    assert all(event["request"]["kind"] == "ask" for event in turns)
    assert all(event["request"].get("anchor") is None for event in turns)
    assert all(
        "source" not in event["request"] and "source_id" not in event["request"] for event in turns
    )
    by_message = {event["response"]["message"]["message_id"]: event for event in turns}
    names = ["first", "follow", "compare", "two", "three", "table"]
    if config["vision_enabled"]:
        names.append("follow-cache")
    cases = {}
    for name in names:
        browser = load(directory, name)
        event = by_message[browser["message_id"]]
        cases[name] = {
            "request": event["request"],
            "http_status": event["status"],
            "browser": browser,
            "turn": event["response"]["turn"],
            "answer": event["response"]["message"],
        }
    for name, question in (
        ("first", FIRST),
        ("compare", COMPARE),
        ("follow", FOLLOW),
        ("two", TWO),
        ("three", THREE),
    ):
        assert cases[name]["request"]["text"] == question

    first, follow, compare = (cases[key] for key in ("first", "follow", "compare"))
    for case, version in ((first, 1), (compare, 2)):
        sources = case["answer"]["sources"]
        assert [entry["revision"]["source"]["version"] for entry in sources.values()] == [
            version
        ] * 3
        assert [entry["revision"]["source"]["source_id"] for entry in sources.values()] == [
            "cell:1",
            LEFT,
            RIGHT,
        ]
        assert {"cell:1", LEFT, RIGHT} <= candidate_keys(case["turn"])
        assert case["turn"]["resolution"] is None
        assert all(
            not entry["reads"]
            for entry in sources.values()
            if entry["read_status"] == "not_attempted"
        )
        assert sum(len(entry["reads"]) for entry in sources.values()) == 1
        assert "Elapsed time [" not in case["browser"]["text"]
        assert len(case["browser"]["displays"]) == 2
        assert len(case["browser"]["citations"]) == 1
        assert case["turn"]["trace"]["display_attempts"]["displayed"] == 2
        assert case["turn"]["trace"].get("external_call_count", 0) == 0
    resolution = follow["turn"]["resolution"]
    assert resolution["method"] == "visible_ordinal" and resolution["ordinal"] == 2
    assert resolution["display_id"] == first["browser"]["displays"][1]["display_id"]
    assert resolution["actual_source"]["source_id"] == RIGHT
    assert resolution["actual_source"]["version"] == 1

    provider_file = directory / "provider.jsonl"
    provider = []
    if provider_file.exists():
        with provider_file.open() as log:
            provider = [json.loads(line) for line in log]
    follow_reads = list(follow["answer"]["sources"].values())[0]["reads"]
    if config["vision_enabled"]:
        assert len(provider) == 1 and provider[0].get("http_status") == 200
        assert follow["turn"]["trace"]["external_call_count"] == 1
        assert len(follow_reads) == 1
        read = follow_reads[0]
        assert read["method"] == "vision" and read["origin_execution"] == "real_provider"
        assert read["actual_source"] == resolution["actual_source"]
        assert read["actual_input_sha256"] == manifest["image_sha256"]["1"][RIGHT]
        assert read["status"] in {"succeeded", "partial"}
        assert any(word in read["text"].casefold() for word in ("second", "秒", "[s]", "（s）"))
        cached = cases["follow-cache"]["turn"]["trace"]
        assert cached["external_call_count"] == 0
        assert cached["vision"]["reads"][0]["cache_hit"] is True
    else:
        assert provider == []
        assert follow_reads == []
        assert follow["turn"]["trace"]["vision"]["capability_reason"] == "provider_disabled"

    historical_control = {}
    for name, version in (("open-old", 1), ("open-current", 2)):
        shown = load(directory, name)
        route, _, _ = shown["media"].partition("?")
        attempts = [
            row
            for row in http
            if row["category"] == "media" and row["path"] == route and row["status"] == 200
        ]
        assert len(attempts) >= 1
        # The final observed request is from reopening the selected original after publication.
        received = attempts[-1]["response"]
        assert "v" + str(version) in shown["title"] and shown["width"] == 640
        assert received["sha256"] == manifest["image_sha256"][str(version)][RIGHT]
        displayed = first if version == 1 else compare
        assert shown["media"] == displayed["browser"]["displays"][1]["url"]
        historical_control[name] = {
            "browser": shown,
            "actual_source": displayed["answer"]["blocks"][-1]["display_source"],
            "media": received,
        }
    second_version = historical_control["open-current"]["actual_source"]
    original_version = historical_control["open-old"]["actual_source"]
    assert original_version == {**second_version, "version": 1}
    assert (
        historical_control["open-old"]["media"]["sha256"]
        != historical_control["open-current"]["media"]["sha256"]
    )

    for name in ("two", "three"):
        case = cases[name]
        trace = case["turn"]["trace"]
        assert trace["route"] == "evidence_answer" and trace["answer_status"] == "complete"
        assert trace["requirements"] == ["definition", "experiment"]
        assert trace["budget"] == 2 and trace["supplement_count"] == 1
        assert trace["term"]["word"] == "TidalRunQ9"
        excerpt = manifest["definition"]
        assert trace["term"]["excerpt_sha256"] == sha(excerpt.encode())
        assert excerpt[trace["term"]["start"] : trace["term"]["end"]] == trace["term"]["word"]
        assert trace["first_missing"] == ["experiment"]
        assert TABLE in [doc["file_name"] for doc in trace["snapshot"]]
        assert TABLE not in [v["file_name"] for v in case["answer"]["sources"].values()]
        assert [
            [v["file_name"], v["revision"]["source"]["source_id"], v["read_status"]]
            for v in case["answer"]["sources"].values()
        ] == [[LECTURE, "cell:0", "succeeded"], [OUTPUT, "cell:0:output:0", "succeeded"]]
        assert "0.42" in case["browser"]["text"] and "0.29" in case["browser"]["text"]
        assert "17.5" not in case["browser"]["text"]
        assert len(case["browser"]["citations"]) == 2

    table = cases["table"]
    native_table_ids = [
        item["evidence_id"]
        for item in table["answer"]["sources"].values()
        if item["file_name"] == TABLE
        and any(
            read["method"] == "native_table"
            and "17.5" in read["text"]
            and read["status"] == "succeeded"
            for read in item["reads"]
        )
    ]
    assert len(native_table_ids) == 1
    assert any(
        item["type"] == "text" and item["citations"] == native_table_ids and "17.5" in item["text"]
        for item in table["answer"]["blocks"]
    )

    worktree = Path(__file__).resolve().parents[2]
    public_paths = [
        "app/rag/workspace.py",
        "app/rag/evidence_supplement.py",
        "app/rag/evidence_answer.py",
        "docs/G/G05-probe.py",
        "docs/G/G05-browser.sh",
    ]
    evidence = {
        "schema": "g05-diagnosis-v1",
        "subject": os.environ.get("G05_SUBJECT_SHA", "96113392f9403f505d4e3416cf39956704abd11f"),
        "source_sha256": {name: sha((worktree / name).read_bytes()) for name in public_paths},
        "agent_browser": subprocess.check_output(["agent-browser", "--version"], text=True).strip(),
        "chrome": load(directory, "browser"),
        "fixtures": manifest,
        "config": config,
        "imports": [row["response"] for row in http if row["category"] == "import"],
        "cases": cases,
        "historical_control": historical_control,
        "provider": provider,
        "first_divergence": {
            "first": "target_selection_uses_parent_candidate_and_both_preview_outputs",
            "compare": "scope_pins_current_version_only_before_candidate_retrieval",
            "three": "missing_table_requirement_before_budget_and_candidate_admission",
        },
    }
    write(directory / "G05-实测.json", evidence)
    print(f"G05 browser verified: {directory / 'G05-实测.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18841)
    parser.add_argument("--soffice", type=Path)
    parser.add_argument("--real-vision", action="store_true")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.data_dir)
        return
    if (args.data_dir / "library.sqlite3").exists():
        parser.error("G05 requires an empty workspace")
    uvicorn.run(make_app(args.data_dir, args), host="127.0.0.1", port=args.port, lifespan="off")


if __name__ == "__main__":
    main()
