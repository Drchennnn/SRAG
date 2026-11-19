"""Collect and verify the public G07 browser replay with vision disabled."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import nbformat
import uvicorn

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("g05_public_probe", HERE / "G05-probe.py")
assert spec is not None and spec.loader is not None
g05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g05)

BASENAME = "VersionDeck.ipynb"
HISTORY = "刚才第二张图"
CURRENT = "第二张图"
AMBIGUOUS = "那张图是哪一张？"
PILOT = "Pilot amplitude frequency"
COMPARE = "比较 VersionDeck.ipynb 上一版和当前版 Cell 1 保存输出。"
OUTPUT = "cell:0:output:0"
EXPECTED = {1: "2.5 kHz", 2: "4.5 kHz"}


def prepare_versions(path: Path) -> None:
    for version, value in EXPECTED.items():
        cell = nbformat.v4.new_code_cell(
            "summarize()",
            id="stability",
            execution_count=1,
            outputs=[nbformat.v4.new_output("stream", name="stdout", text=f"{PILOT}: {value}")],
        )
        target = path / "fixtures" / f"v{version}" / BASENAME
        g05.notebook(target, [cell])


def verify(path: Path) -> None:
    with (path / "http.jsonl").open() as stream:
        http = [json.loads(line) for line in stream]
    turns = {
        row["response"]["message"]["message_id"]: row
        for row in http
        if row["category"] == "turn" and row["status"] == 200
    }
    assert len(turns) == 6
    cases = {}
    for label, question, kind in (
        ("history-empty", HISTORY, "ask"),
        ("current", CURRENT, "ask"),
        ("ambiguous", AMBIGUOUS, "ask"),
        ("choice", AMBIGUOUS, "ask"),
        ("version-history", PILOT, "search"),
        ("compare", COMPARE, "ask"),
    ):
        browser = g05.load(path, label)
        event = turns[browser["message_id"]]
        request = event["request"]
        assert request["text"] == question and request["kind"] == kind
        assert not {"source", "source_id", "display_id", "target"} & request.keys()
        assert bool(request.get("anchor")) == (label == "choice")
        cases[label] = {
            "request": request,
            "turn": event["response"]["turn"],
            "message": event["response"]["message"],
            "browser": browser,
        }

    no_history = cases["history-empty"]
    assert no_history["turn"]["status"] == "clarification"
    assert no_history["turn"]["options"] == [] and no_history["message"]["sources"] == {}
    assert not no_history["browser"]["images"]
    for label, method in (("current", "current_focus"), ("choice", "current_choice")):
        case = cases[label]
        turn = case["turn"]
        selected = turn["resolution"]["actual_source"]
        assert turn["resolution"]["method"] == method and turn["status"] == "completed"
        assert selected["version"] == 1 and selected["source_id"] == g05.RIGHT
        (evidence,) = case["message"]["sources"].values()
        assert evidence["revision"]["source"] == selected
        assert evidence["read_status"] == "not_attempted" and evidence["reads"] == []
        assert turn["trace"]["external_call_count"] == 0
        (drawn,) = case["browser"]["images"]
        assert drawn["width"] == 640
        expected_sha = g05.load(path, "fixtures")["image_sha256"]["1"][g05.RIGHT]
        media = [
            row["response"]
            for row in http
            if row["category"] == "media"
            and row["status"] == 200
            and row["path"] == drawn["url"].partition("?")[0]
        ]
        assert any(item["sha256"] == expected_sha for item in media)
        case["browser_png_sha256"] = expected_sha
    setup = g05.load(path, "focus-setup")
    assert setup["focus"].startswith("Cell 2") and setup["history_images"] == 0

    ambiguity = cases["ambiguous"]
    options = ambiguity["turn"]["options"]
    assert ambiguity["turn"]["status"] == "clarification" and len(options) == 2
    assert not ambiguity["message"]["sources"]
    before = g05.load(path, "options-before-refresh")
    after = g05.load(path, "options-after-refresh")
    assert before == after == [option["label"] for option in options]
    assert cases["choice"]["request"]["anchor"] == {
        "type": "clarification",
        "turn_id": ambiguity["turn"]["turn_id"],
        "option_id": options[1]["option_id"],
    }

    compare = cases["compare"]
    trace = compare["turn"]["trace"]
    assert trace["route"] == "version_comparison" and trace["external_call_count"] == 0
    assert trace["historical_read_status"] == trace["current_read_status"] == "succeeded"
    assert trace["historical_message_id"] == cases["version-history"]["message"]["message_id"]
    before_sources = cases["version-history"]["message"]["sources"].values()
    assert any(row["revision"]["source"] == trace["historical_source"] for row in before_sources)
    sources = {
        row["revision"]["source"]["version"]: row for row in compare["message"]["sources"].values()
    }
    assert sources.keys() == EXPECTED.keys()
    text_blocks = [block for block in compare["message"]["blocks"] if block["type"] == "text"]
    assert {c for block in text_blocks for c in block["citations"]} == {
        row["evidence_id"] for row in sources.values()
    }
    citations = {}
    for version, value in EXPECTED.items():
        read = sources[version]["reads"][0]
        key = sources[version]["revision"]["source"]
        assert read["actual_source"] == key and key["source_id"] == OUTPUT
        assert key == trace["historical_source" if version == 1 else "current_source"]
        assert read["method"] == "native_text" and read["status"] == "succeeded"
        assert value in read["text"] and value in compare["browser"]["text"]
        message_id = compare["message"]["message_id"]
        evidence_id = sources[version]["evidence_id"]
        suffix = f"/messages/{message_id}/evidence/{evidence_id}"
        (opened,) = [
            row for row in http if row["category"] == "citation" and row["path"].endswith(suffix)
        ]
        assert opened["status"] == 200 and opened["response"]["source"] == key
        assert opened["response"]["is_current"] == (version == 2)
        assert value in opened["response"]["text"]
        browser = g05.load(path, f"citation-v{version}")
        assert f"v{version}" in browser["title"] and value in browser["body"]
        assert browser["location"] == ("历史版本" if version == 1 else "当前版本")
        citations[str(version)] = {"browser": browser, "get": opened}
    assert trace["historical_source"]["document_id"] == trace["current_source"]["document_id"]
    assert not (path / "provider.jsonl").exists()

    base = HERE.parents[1]
    paths = (
        "app/rag/workspace.py",
        "app/rag/workspace_versions.py",
        "app/rag/workspace_initial_focus.py",
        "app/web/workspace/workspace.js",
        "docs/G/G05-probe.py",
        "docs/G/G07-probe.py",
        "docs/G/G07-browser.sh",
    )
    result = {
        "schema": "g07-public-browser-v1",
        "subject_base": os.environ.get("G07_SUBJECT_BASE"),
        "files_sha256": {name: g05.sha((base / name).read_bytes()) for name in paths},
        "synthetic_versions": {
            str(version): {
                "file": BASENAME,
                "sha256": g05.sha((path / "fixtures" / f"v{version}" / BASENAME).read_bytes()),
                "saved_text": f"{PILOT}: {value}",
            }
            for version, value in EXPECTED.items()
        },
        "agent_browser": subprocess.check_output(["agent-browser", "--version"], text=True).strip(),
        "chrome": g05.load(path, "browser"),
        "external_calls": 0,
        "focus_setup": setup,
        "cases": cases,
        "clarification_persisted": {"before_refresh": before, "after_refresh": after},
        "version_citation_roundtrip": citations,
    }
    g05.write(path / "G07-实测.json", result)
    print(f"G07 browser verified: {path / 'G07-实测.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18859)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.data_dir)
        return
    if (args.data_dir / "library.sqlite3").exists():
        parser.error("G07 requires an empty workspace")
    args.real_vision = False
    args.env_file = None
    args.soffice = None
    app = g05.make_app(args.data_dir, args)
    prepare_versions(args.data_dir)
    uvicorn.run(app, host="127.0.0.1", port=args.port, lifespan="off")


if __name__ == "__main__":
    main()
