"""Replay a G06 public first question against the recorded workspace API."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import uvicorn

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("g05_public_probe", HERE / "G05-probe.py")
assert spec is not None and spec.loader is not None
g05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g05)

QUESTION = "HarborTrace.ipynb Cell 2 第 2 张图的横轴刻度单位是什么？"
TABLE = "PulseGate-limits.pptx 第 1 页表格里的 Pulse gate 设定值是多少？"


def read_turns(directory: Path):
    with (directory / "http.jsonl").open() as stream:
        http = [json.loads(line) for line in stream]
    found = {}
    for row in http:
        if row["category"] != "turn":
            continue
        assert row["status"] == 200
        assert row["request"]["kind"] == "ask" and row["request"].get("anchor") is None
        assert not {"source", "source_id", "display_id", "target"} & row["request"].keys()
        message = row["response"]["message"]
        found[message["message_id"]] = row
    return http, found


def checked_answer(directory: Path, turns, name: str, question: str):
    browser = g05.load(directory, name)
    entry = turns[browser["message_id"]]
    assert entry["request"]["text"] == question
    result = entry["response"]
    trace = result["turn"]["trace"]
    assert result["turn"]["status"] == "completed"
    assert trace["route"] == "target_object"
    source = trace["selected_source"]
    sources = result["message"]["sources"]
    assert len(sources) == 1
    evidence = next(iter(sources.values()))
    assert evidence["revision"]["source"] == source
    assert source in trace["object_candidates"]
    answer = {
        "browser": browser,
        "request": entry["request"],
        "http_status": entry["status"],
        "turn": result["turn"],
        "message": result["message"],
    }
    return answer, source, evidence, trace


def verify_image(directory: Path, http, turns, name, real, cold):
    answer, source, evidence, trace = checked_answer(directory, turns, name, QUESTION)
    assert source["source_id"] == g05.RIGHT and source["version"] == 1
    blocks = answer["message"]["blocks"]
    (image,) = (b for b in blocks if b["type"] == "source_image")
    assert image["display_source"] == source and image["evidence_id"] == evidence["evidence_id"]
    assert image["fallback_reason"] is None and image["rect_visible_norm"] is None
    assert image["width"] == answer["browser"]["width"] == 640
    assert image["height"] == answer["browser"]["height"] == 350
    assert image["media_url"] == answer["browser"]["media_url"]
    expected = g05.load(directory, "fixtures")["image_sha256"]["1"][g05.RIGHT]
    matches = [
        row
        for row in http
        if row["category"] == "media" and row["path"] == image["media_url"].partition("?")[0]
    ]
    assert any(row["status"] == 200 and row["response"]["sha256"] == expected for row in matches)
    if real:
        (read,) = evidence["reads"]
        assert read["method"] == "vision" and read["actual_source"] == source
        assert read["origin_execution"] == "real_provider"
        assert read["actual_input_sha256"] == expected and read["actual_input_rect"] is None
        assert read["actual_input_media_type"] == "image/png"
        assert read["status"] in {"succeeded", "partial"}
        cite = [
            b for b in blocks if b["type"] == "text" and b["citations"] == [evidence["evidence_id"]]
        ]
        assert len(cite) == 1 and read["text"] == cite[0]["text"]
        assert any(word in read["text"].casefold() for word in ("second", "秒", "[s]", "（s）"))
        assert read["text"] in answer["browser"]["text"]
        (provider_read,) = trace["vision"]["reads"]
        assert provider_read["execution"] == ("real_provider" if cold else "cache_reuse")
        assert provider_read["cache_hit"] is not cold
    else:
        assert not evidence["reads"] and evidence["read_status"] == "not_attempted"
        assert trace["vision"]["capability_reason"] == "provider_disabled"
        assert not any(b["type"] == "text" and b["citations"] for b in blocks)
        assert any(b["type"] == "notice" and b["code"] == "object_unreadable" for b in blocks)
    assert trace["external_call_count"] == (1 if real and cold else 0)
    answer["browser_png_sha256"] = expected
    return answer


def verify(directory: Path):
    real = g05.load(directory, "config")["vision_enabled"]
    http, turns = read_turns(directory)
    assert len(turns) == (3 if real else 2)
    answers = {"first": verify_image(directory, http, turns, "first", real, cold=True)}
    if real:
        answers["repeat"] = verify_image(directory, http, turns, "repeat", real, cold=False)
    table, source, evidence, trace = checked_answer(directory, turns, "table", TABLE)
    (read,) = evidence["reads"]
    assert read["method"] == "native_table" and read["status"] == "succeeded"
    assert read["actual_source"] == source and "17.5" in read["text"]
    assert source["source_id"].startswith("page:1:shape:")
    assert trace["external_call_count"] == 0 and "17.5" in table["browser"]["text"]
    answers["table"] = table

    calls = directory / "provider.jsonl"
    provider = (
        [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    )
    if real:
        (call,) = provider
        assert call["event"] == "real_vision_attempt" and call["http_status"] == 200
        assert call["image_sha256"] == answers["first"]["browser_png_sha256"]
    else:
        assert provider == []
    repo = HERE.parents[1]
    paths = (
        "app/rag/workspace.py",
        "app/rag/workspace_objects.py",
        "app/rag/source_vision.py",
        "docs/G/G05-probe.py",
        "docs/G/G06-probe.py",
        "docs/G/G06-browser.sh",
    )
    summary = {
        "schema": "g06-public-browser-v1",
        "subject_base": os.environ.get("G06_SUBJECT_BASE"),
        "files_sha256": {path: g05.sha((repo / path).read_bytes()) for path in paths},
        "agent_browser": subprocess.check_output(["agent-browser", "--version"], text=True).strip(),
        "chrome": g05.load(directory, "browser"),
        "vision_enabled": real,
        "provider_attempts": provider,
        "answers": answers,
    }
    path = directory / "G06-首问实测.json"
    g05.write(path, summary)
    print(f"G06 browser verified: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18854)
    parser.add_argument("--soffice", type=Path)
    parser.add_argument("--real-vision", action="store_true")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.data_dir)
        return
    if (args.data_dir / "library.sqlite3").exists():
        parser.error("G06 requires an empty workspace")
    if args.real_vision and (args.env_file is None or not args.env_file.is_file()):
        parser.error("real vision requires the authorized local configuration file")
    # Reuse the public-only provider guard with the exact first question. It checks image bytes
    # against the right v1 output and checks that the request sends a single user message.
    g05.DIRECTED = QUESTION
    uvicorn.run(
        g05.make_app(args.data_dir, args),
        host="127.0.0.1",
        port=args.port,
        lifespan="off",
    )


if __name__ == "__main__":
    main()
