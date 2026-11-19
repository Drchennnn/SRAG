"""Check synthetic G02 browser traffic, byte identities and the independently visible UI."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "docs/A/样例"
SAMPLE = json.loads((ROOT / "docs/F/样例/SignalTrialA7.json").read_text())
PROBES = (
    ("Signal-reference.pdf", "lookahead", {"page": 1}, "带原生文字的 PDF 页"),
    ("Signal-slides.pptx", "Median filter experiment", {"page": 1}, "带原生文字的幻灯片"),
    ("Signal-lab.ipynb", "Median filter", {"cell_index": 0}, "Notebook Markdown Cell"),
)
F_PATHS = (
    ("app/rag/workspace.py"),
    ("app/rag/evidence_supplement.py"),
    ("app/rag/evidence_answer.py"),
    ("app/stores/workspace.py"),
    ("app/api/workspace.py"),
    ("app/web/workspace/workspace.js"),
    ("docs/G/G02-server.py"),
    ("docs/G/G02-browser.sh"),
    ("docs/G/G02-verify.py"),
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def timed_turn(row: dict, ui: dict[str, dict]) -> dict:
    response = row["response"]
    answer = response["message"]
    return {
        "http_ms": row["http_ms"],
        "browser_visible_ms": ui[answer["message_id"]]["visible_ms"],
        "request": row["request"],
        "status": response["turn"]["status"],
        "external_call_count": response["turn"]["trace"].get("external_call_count", 0),
    }


def _matching_probe(row: dict, document_id: str, query: str, locator: dict) -> dict:
    assert row["request"]["kind"] == "search" and row["request"]["text"] == query
    assert row["response"]["turn"]["status"] == "completed"
    message = row["response"]["message"]
    for evidence in message["sources"].values():
        source = evidence["revision"]["source"]
        read = evidence["reads"][0]
        if (
            source["document_id"] == document_id
            and source["version"] == 1
            and all(evidence["locator"].get(key) == val for key, val in locator.items())
            and query.casefold() in read["text"].casefold()
            and read["actual_source"] == source
            and read["method"] == "native_text"
            and evidence["read_status"] == "succeeded"
        ):
            return evidence
    raise AssertionError(f"no verified first text hit for {query}")


def check_imports(ledger, imports, turns, perf, ui):
    results = []
    for index, (name, query, locator, type_label) in enumerate(PROBES):
        upload = imports[index]
        assert upload["query"] == {"course_id": ["signals"], "filename": [name]}
        original = (FIXTURES / name).read_bytes()
        doc = upload["response"]
        assert doc["status"] == "published" and doc["version"] == 1
        assert sha(original) == doc["content_sha256"]
        assert upload["request_bytes"] == len(original)
        cold, hot = turns[2 * index : 2 * index + 2]
        cold_evidence = _matching_probe(cold, doc["document_id"], query, locator)
        hot_evidence = _matching_probe(hot, doc["document_id"], query, locator)
        assert cold_evidence["revision"] == hot_evidence["revision"]
        cold_ui = ui[cold["response"]["message"]["message_id"]]
        assert perf["imports"][index]["files"] == [name]
        results.append(
            {
                "file": name,
                "file_bytes": len(original),
                "source_type": type_label,
                "source": cold_evidence["revision"]["source"],
                "locator": cold_evidence["locator"],
                "import_http_ms": upload["http_ms"],
                "browser_import_submit_to_first_visible_hit_ms": round(
                    cold_ui["end"] - perf["imports"][index]["start"], 2
                ),
                "cold_query": timed_turn(cold, ui),
                "warm_query": timed_turn(hot, ui),
            }
        )
    reused = imports[3]
    assert reused["response"]["status"] == "skipped"
    assert reused["response"]["content_sha256"] == imports[0]["response"]["content_sha256"]
    assert reused["response"]["document_id"] == imports[0]["response"]["document_id"]
    assert reused["response"]["version"] == 1 and reused["response"]["publication_epoch"] == 2
    assert next(i for i, row in enumerate(ledger) if row is turns[0]) > next(
        i for i, row in enumerate(ledger) if row is imports[0]
    )
    return results, {
        "published_identity": reused["response"],
        "http_ms": reused["http_ms"],
    }


def _saved_png(path: Path, output_index: int) -> bytes:
    value = load(path)["cells"][1]["outputs"][output_index]["data"]["image/png"]
    return base64.b64decode("".join(value) if isinstance(value, list) else value)


def check_images(out, imports, turns, media, ui, config):
    notebook = load(out / "notebook-v1.json")
    user_open = load(out / "image-original.json")
    historical = load(out / "historical-image.json")
    shown = turns[7]["response"]["message"]
    assert shown["message_id"] == notebook["message_id"]
    blocks = [block for block in shown["blocks"] if block["type"] == "source_image"]
    assert [item["display_id"] for item in notebook["displays"]] == shown["visible_display_ids"]
    assert [item["ordinal"] for item in notebook["displays"]] == [1, 2]
    assert len(blocks) == 2
    for idx, (dom, source) in enumerate(zip(notebook["displays"], blocks, strict=True), 1):
        assert dom["img_width"] == 440
        assert dom["display_id"] == source["display_id"] and dom["url"] == source["media_url"]
        assert source["display_source"]["source_id"] == f"cell:1:output:{idx}"
        assert source["display_source"]["version"] == 1
        actual_hash = sha(_saved_png(FIXTURES / "Signal-lab.ipynb", idx))
        matching = [v for v in media if v["path"] == dom["url"].split("?")[0]]
        assert len(matching) >= 2 and all(v["response"]["sha256"] == actual_hash for v in matching)
        dom["sha256"] = actual_hash
        dom["source"] = source["display_source"]
    second = notebook["displays"][1]
    assert user_open["media"] == historical["media"] == second["url"]
    assert user_open["width"] == historical["width"] == 440
    assert user_open["height"] == historical["height"] == 260
    assert all(
        "v1" in entry["title"] and "输出 2" in entry["title"] for entry in [user_open, historical]
    )
    real_calls = rows(out / "provider.jsonl")
    followups = []
    for row in turns[8:10]:
        request, answer, turn = row["request"], row["response"]["message"], row["response"]["turn"]
        assert set(request) == {"client_turn_id", "kind", "text", "scope_revision"}
        assert request["text"] == "刚才第二张图" and request["kind"] == "ask"
        assert turn["status"] in {"completed", "failed"}
        resolved = turn["resolution"]
        assert resolved["method"] == "visible_ordinal" and resolved["ordinal"] == 2
        assert resolved["message_id"] == notebook["message_id"]
        assert resolved["display_id"] == second["display_id"]
        assert resolved["actual_source"] == second["source"]
        evidence = answer["sources"]["ev-1"]
        for actual in evidence["reads"]:
            assert actual["actual_source"] == second["source"]
            assert actual["actual_input_sha256"] == second["sha256"]
        text = "\n".join(block["text"] for block in answer["blocks"])
        followups.append(
            {
                **timed_turn(row, ui),
                "resolution": resolved,
                "read_status": evidence["read_status"],
                "state": turn["trace"]["vision"],
                "actual_reads": [
                    {key: val for key, val in read.items() if key != "text"}
                    for read in evidence["reads"]
                ],
                "response_text": text,
            }
        )
    real_on = config["vision_enabled"]
    assert len(real_calls) == sum(followup["external_call_count"] for followup in followups)
    if real_on:
        assert config["vision_credential_present"] and real_calls
        assert all(row["event"] == "real_vision_attempt" for row in real_calls)
        assert all(row["image_count"] == row["message_count"] == 1 for row in real_calls)
        assert all(row["image_sha256"] == second["sha256"] for row in real_calls)
        cold, warm = followups
        if cold["read_status"] in {"succeeded", "partial"}:
            assert cold["external_call_count"] == 1
            assert (
                "frequency" in cold["response_text"].casefold()
                and "hz" in cold["response_text"].casefold()
            )
            assert warm["external_call_count"] == 0
            assert warm["state"]["reads"][0]["cache_hit"] is True
            assert warm["response_text"] == cold["response_text"]
    else:
        assert all(followup["read_status"] == "not_attempted" for followup in followups)
    revision = next(
        row
        for row in imports
        if row["response"]["file_name"] == "Signal-lab.ipynb" and row["response"]["version"] == 2
    )
    assert revision["response"]["document_id"] == second["source"]["document_id"]
    assert revision["response"]["content_sha256"] == sha(
        (FIXTURES / "revisions/v2/Signal-lab.ipynb").read_bytes()
    )
    new_png_sha256 = sha(_saved_png(FIXTURES / "revisions/v2/Signal-lab.ipynb", 2))
    assert second["sha256"] != new_png_sha256
    return {
        "notebook": notebook,
        "browser_open_at_v1": user_open,
        "browser_open_after_publish_and_reload": historical,
        "v2_publish": revision["response"],
        "v2_second_image_original_sha256": new_png_sha256,
        "turns": followups,
        "actual_provider_attempts": real_calls,
        "read_failures": sum(followup["read_status"] == "failed" for followup in followups),
        "cache_hits": sum(
            row["cache_hit"] for followup in followups for row in followup["state"]["reads"]
        ),
    }


def _cell_text(path: Path) -> str:
    value = load(path)["cells"][0]["source"]
    return "".join(value) if isinstance(value, list) else value


def check_cross_file(out, turns, citations, ui):
    results = []
    expected = [
        ("signal-demo", "supplemented", "complete", ["ev-1", "ev-3"], "0.18"),
        ("signal-demo", "supplemented", "complete", ["ev-1", "ev-3"], "0.16"),
        ("signal-incomplete", "still_missing", "partial", ["ev-1"], None),
    ]
    for row, (course, reason, answer_status, ids, trial) in zip(turns[10:], expected, strict=True):
        response = row["response"]
        answer, trace = response["message"], response["turn"]["trace"]
        assert set(row["request"]) == {"client_turn_id", "kind", "text", "scope_revision"}
        assert row["request"]["text"] == SAMPLE["question"] and row["request"]["kind"] == "ask"
        assert trace["route"] == "evidence_answer" and trace["answer_status"] == answer_status
        assert trace["first_supported"] == ["definition"] and trace["first_missing"] == [
            "experiment"
        ]
        assert trace["budget"] == 2 and trace["supplement_count"] == 1
        assert trace["external_call_count"] == 0
        assert trace["stop_reason"] == reason and trace["cited_evidence_ids"] == ids
        assert set(answer["sources"]) == set(ids)
        assert [item for block in answer["blocks"] for item in block.get("citations", [])] == ids
        terms = trace["term"]
        assert terms["word"] == "SignalTrialA7" and terms["course_id"] == course
        read = answer["sources"][terms["evidence_id"]]["reads"][0]
        assert read["method"] == "native_text" and read["status"] == "succeeded"
        assert terms["source"] == read["actual_source"]
        assert read["text"][terms["start"] : terms["end"]] == terms["word"]
        assert sha(read["text"].encode()) == terms["excerpt_sha256"]
        first, extra = trace["hops"]
        assert first["kind"] == "first" and extra["kind"] == "literal_supplement"
        assert extra["query"] == terms["word"] and extra["scope"]["course_ids"] == [course]
        assert extra["scope"]["document_ids"] == first["scope"]["document_ids"]
        assert [hit["file_name"] for hit in first["candidates"]] == [
            "Median-lecture.ipynb",
            "Median-drills.ipynb",
        ]
        assert {hit["course_id"] for hit in extra["candidates"]} == {course}
        if trial is None:
            assert trace["missing"] == ["experiment"]
            assert trace["replacements"] == []
            assert not any(
                "0.18" in block["text"] or "0.16" in block["text"] for block in answer["blocks"]
            )
            assert any(hit["reason"] == "experiment_fields_missing" for hit in extra["candidates"])
        else:
            assert trace["missing"] == [] and len(trace["replacements"]) == 1
            replaced = trace["replacements"][0]
            assert replaced["removed"]["source"] == first["candidates"][1]["source"]
            assert replaced["added"]["source"] == answer["sources"]["ev-3"]["revision"]["source"]
            assert any(trial in block["text"] for block in answer["blocks"])
        results.append(
            {
                **timed_turn(row, ui),
                "message_id": answer["message_id"],
                "turn_trace": trace,
                "visible_answer_blocks": answer["blocks"],
                "cited_sources": answer["sources"],
            }
        )
    expected_citations = [
        ("definition-citation", 0, "ev-1", "complete", True),
        ("experiment-citation", 0, "ev-3", "complete", True),
        ("current-experiment", 1, "ev-3", "v2", True),
        ("historical-experiment", 0, "ev-3", "complete", False),
    ]
    reader_results = []
    for row, (name, index, evidence_id, fixture, current) in zip(
        citations, expected_citations, strict=True
    ):
        dom = load(out / f"{name}.json")
        stored = results[index]["cited_sources"][evidence_id]
        actual = row["response"]
        assert row["path"].endswith(
            f"/messages/{results[index]['message_id']}/evidence/{evidence_id}"
        )
        assert actual["source"] == stored["revision"]["source"]
        assert actual["is_current"] is current and actual["locator"]["cell_index"] == 0
        assert actual["file_name"] == stored["file_name"]
        assert dom["title"] == f"{actual['file_name']} · v{actual['source']['version']} · Cell 1"
        assert (
            dom["text"]
            == actual["text"]
            == _cell_text(out / "fixtures" / fixture / actual["file_name"])
        )
        assert ("历史版本" in dom["location"]) is (not current)
        reader_results.append({"operation": name, "dom": dom, "http": row})
    incomplete = load(out / "incomplete-visible.json")
    assert incomplete["buttons"] == ["Median-lecture.ipynb · v1 · Cell 1"]
    assert "原文不足以确认" in incomplete["text"]
    return {"turns": results, "citation_readers": reader_results, "incomplete_ui": incomplete}


def _runtime(perf: dict, config: dict) -> dict:
    git_sha = os.environ.get("G02_SUBJECT_SHA")
    if git_sha is None and (ROOT / ".git").exists():
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return {
        "git_checkout_parent_or_exact_archive": git_sha,
        "product_and_runner_sha256": {path: sha((ROOT / path).read_bytes()) for path in F_PATHS},
        "system": platform.platform(),
        "python": platform.python_version(),
        "agent_browser": subprocess.check_output(["agent-browser", "--version"], text=True).strip(),
        "browser_user_agent": perf["agent"],
        "settings": config,
    }


def main() -> None:
    output = Path(sys.argv[1])
    perf, config = load(output / "visual-times.json"), load(output / "config.json")
    ui = {item["message_id"]: item for item in perf["turns"]}
    ledger = rows(output / "http.jsonl")
    groups = defaultdict(list)
    for entry in ledger:
        assert entry["status"] == 200, entry
        groups[entry["category"]].append(entry)
    turns, imports = groups["turn"], groups["import"]
    assert len(turns) == len(ui) == 13
    assert len(imports) == 12
    first_hits, same_content = check_imports(ledger, imports, turns, perf, ui)
    definition = turns[6]["response"]["message"]
    assert any(
        evidence["file_name"] == "Signal-reference.pdf"
        and "median" in evidence["reads"][0]["text"].casefold()
        for evidence in definition["sources"].values()
    )
    report = {
        "schema": "g02-browser-evidence-v1",
        "runtime": _runtime(perf, config),
        "measurements": {
            "http_ms": "服务收到 HTTP 到最后一个响应块",
            "browser_visible_ms": "当前轮浏览器提交到首条助手消息进 DOM",
            "browser_import_submit_to_first_visible_hit_ms": "含浏览器切换文件和提交首查",
            "provider_response_headers_ms": "实际出站到模型 HTTP 响应头",
            "cold_warm_samples_per_format": 1,
            "token_amounts": "服务商返回的 usage；货币价格无实测账单",
        },
        "first_text_hits": first_hits,
        "repeat_same_content": same_content,
        "definition_before_image": {
            **timed_turn(turns[6], ui),
            "sources": definition["sources"],
        },
        "image_roundtrip": check_images(output, imports, turns, groups["media"], ui, config),
        "cross_file_roundtrip": check_cross_file(output, turns, groups["citation"], ui),
    }
    assert sum(
        row["response"]["turn"]["trace"].get("external_call_count", 0) for row in turns
    ) == len(report["image_roundtrip"]["actual_provider_attempts"])
    (output / "G02-实测.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("G02 browser verified")


if __name__ == "__main__":
    main()
