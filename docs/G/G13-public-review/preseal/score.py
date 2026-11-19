"""Evaluate the predeclared independent browser cases without altering either run."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.json").read_text())
FILES = {
    ("Azimuth-Workshop.pdf", 1): "Azimuth-Workshop.pdf",
    ("Beacon-Inspection.pptx", 1): "Beacon-Inspection.pptx",
    ("BenchRetest.ipynb", 1): "v1/BenchRetest.ipynb",
    ("BenchRetest.ipynb", 2): "v2/BenchRetest.ipynb",
}
HEXES = {key: hashlib.sha256((HERE / "materials" / path).read_bytes()).hexdigest()
         for key, path in FILES.items()}
PASS, FAIL, NA, UNTESTED = "pass", "fail", "not_applicable", "not_measured"
READ_OK = {"succeeded", "partial"}


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def standard(value):
    return PASS if value else FAIL


def actual(evidence):
    return evidence.get("revision", {}).get("source", {})


def valid_revision(evidence, file, version):
    revision = evidence.get("revision", {})
    return evidence.get("file_name") == file and revision.get("source", {}).get("version") == version and revision.get("content_sha256") == HEXES[(file, version)]


def equivalent_location(found, expected):
    return bool(isinstance(found, list) and len(found) == 4 and all(abs(a - b) < 0.025 for a, b in zip(found, expected)))


def highlight_location(value):
    if not value or len(value) != 4:
        return None
    left, top, width, height = (float(x.removesuffix("%")) / 100 for x in value)
    return [left, top, left + width, top + height]


def load_media(http, path, mode):
    rows = [row["response"] for row in http if row["category"] == "media" and row["path"] == (path or "").partition("?")[0]
            and row["status"] == 200 and (row["query"].get("view") == ["page"]) == (mode == "page")]
    return rows[-1] if rows else {}


def reader_matches(reader, filename, version):
    title = reader.get("title") or ""
    location = reader.get("location") or ""
    return filename in (title + location) and f"v{version}" in (title + location)


def resolve_first_failure(axes):
    return next((key for key in ("normal_entry", "object", "display", "read", "version", "citation", "answer")
                 if axes[key] == FAIL), None)


def score_case(directory, case, turn_rows, http, calls, real):
    name, kind = case["id"], case["kind"]
    browser = read_json(directory / f"{name}.json", {})
    message_id = browser.get("message_id")
    turn_row = turn_rows.get(message_id, {})
    result = turn_row.get("response") or {}
    turn, message = result.get("turn") or {}, result.get("message") or {}
    trace = turn.get("trace") or {}
    request = turn_row.get("request") or {}
    normal = (turn_row.get("status") == 200 and request.get("kind") == "ask"
              and request.get("text") == case["text"] and not request.get("anchor")
              and "source" not in request and "source_id" not in request and "display_id" not in request)
    axes = {key: NA for key in ("normal_entry", "object", "display", "read", "version", "citation", "answer")}
    axes["normal_entry"] = standard(normal)
    evidences = list((message.get("sources") or {}).values())
    read_records = [read for evidence in evidences for read in evidence.get("reads", [])]
    reads_succeeded = [read for read in read_records if read.get("status") in READ_OK]
    overstated = any(read.get("method") == "vision" and read.get("origin_execution") != "real_provider" for read in reads_succeeded)
    image_blocks = [block for block in message.get("blocks", []) if block.get("type") == "source_image"]
    browser_answer = "\n".join(browser.get("answer", []))
    completed = turn.get("status") in {"completed", "clarification"}
    source = trace.get("selected_source")
    case_calls = [call for call in calls if call.get("case") == name]
    severe = []
    if overstated:
        severe.append("falsely_marked_actual_read")
    if any(call["event"] == "blocked_before_egress" for call in case_calls):
        severe.append("unsafe_egress_attempt")

    if kind in {"pdf", "table"}:
        filename = "Azimuth-Workshop.pdf" if kind == "pdf" else "Beacon-Inspection.pptx"
        expected = (PROTOCOL["figure_truth"][case["expected_side"]] if kind == "pdf" else PROTOCOL["table_truth"])
        evidence = evidences[0] if len(evidences) == 1 else {}
        locator = evidence.get("locator", {})
        axes["version"] = standard(valid_revision(evidence, filename, 1))
        block = image_blocks[0] if len(image_blocks) == 1 else {}
        in_ui = browser.get("images", [])
        inline = load_media(http, block.get("media_url"), mode="inline")
        page = load_media(http, block.get("media_url"), mode="page")
        source_linked = bool(source and source == actual(evidence))
        crop_attached = bool(block and source_linked and block.get("display_source") == source
                             and block.get("evidence_id") == evidence.get("evidence_id")
                             and locator.get("page") == 1 and not block.get("fallback_reason")
                             and equivalent_location(block.get("rect_visible_norm"), expected["rect"]))
        inline_reader = read_json(directory / f"{name}-image-1.json", {})
        page_reader = read_json(directory / f"{name}-page-1.json", {})
        inline_success = bool(inline.get("sha256") and len(in_ui) == 1 and in_ui[0].get("width")
                              and inline_reader.get("width") and reader_matches(inline_reader, filename, 1))
        page_success = bool(page.get("sha256") and page_reader.get("width") and reader_matches(page_reader, filename, 1)
                            and equivalent_location(highlight_location(page_reader.get("rect")), expected["rect"]))
        if kind == "pdf":
            side = case["expected_side"]
            other = "left" if side == "right" else "right"
            crop_pixels = inline.get("panel_pixels", {})
            meaning = (crop_pixels.get(side, 0) > 75 and crop_pixels.get(other, 0) < 30)
            axes["object"] = standard(source_linked and locator.get("object_type") in {"image", "picture"}
                                      and (meaning or equivalent_location(block.get("rect_visible_norm"), expected["rect"])))
            axes["display"] = standard(crop_attached and meaning and inline_success)
            visual = [read for read in reads_succeeded if read.get("method") == "vision"]
            actual_matches = (len(visual) == 1 and visual[0].get("actual_source") == source
                              and visual[0].get("origin_execution") == "real_provider"
                              and visual[0].get("actual_input_sha256") == inline.get("sha256")
                              and equivalent_location(visual[0].get("actual_input_rect"), expected["rect"]))
            if not real:
                axes["read"] = UNTESTED
            else:
                axes["read"] = standard(actual_matches and (not case.get("warm_repeat") or (
                    visual[0].get("execution") == "cache_reuse" and trace.get("external_call_count") == 0
                )))
            if visual and not actual_matches:
                severe.append("actual_read_linked_to_wrong_object")
            axes["citation"] = standard(crop_attached and inline_success and page_success)
            unit = {"mrad": r"\bmrad\b|毫弧度", "deg": r"\bdeg(?:rees?)?\b|\(°\)|[计为是]\s*度"}[case["expected_unit"]]
            alt = {"mrad": r"\bdeg(?:rees?)?\b|\(°\)|[计为是]\s*度", "deg": r"\bmrad\b|毫弧度"}[case["expected_unit"]]
            correct = bool(re.search(unit, browser_answer, re.I) and not re.search(alt, browser_answer, re.I))
            axes["answer"] = standard(completed and correct)
            if browser_answer and re.search(alt, browser_answer, re.I):
                severe.append("wrong_side_definitive_answer")
            elif correct and not visual:
                severe.append("image_answer_without_actual_read")
        else:
            axes["object"] = standard(source_linked and locator.get("object_type") == "table" and locator.get("page") == 1)
            axes["display"] = standard(crop_attached and inline_success)
            native = [read for read in reads_succeeded if read.get("method") == "native_table" and read.get("actual_source") == source]
            headers = expected["column_headers"]
            native_grounded = len(native) == 1 and all(item.casefold() in native[0]["text"].casefold() for item in headers)
            axes["read"] = standard(native_grounded and not any(read.get("method") == "vision" for read in read_records)
                                    and trace.get("external_call_count") == 0)
            axes["citation"] = standard(crop_attached and inline_success and page_success
                                        and valid_text_open(directory, name, http, message_id, evidence, 1))
            actual_target = re.search(r"\b54\s*(?:[（(]\s*)?(?:ms\b|毫秒)", browser_answer, re.I)
            native_exact = re.search(r"North beacon[^\n]*\b1\.6\b[^\n]*\b54\b", browser_answer, re.I)
            correct = bool(actual_target or (native_exact and "settle time (ms)" in browser_answer.casefold()))
            axes["answer"] = standard(completed and correct)
            if re.search(r"\b(?:72|1\.6|3\.8)\s*(?:[（(]\s*)?(?:ms\b|毫秒)", browser_answer, re.I) and not correct:
                severe.append("wrong_table_row_or_unit")

    if kind == "ambiguity":
        axes["object"] = standard(not source)
        axes["display"] = standard(not image_blocks)
        axes["read"] = standard(not read_records and trace.get("external_call_count", 0) == 0)
        asks_details = bool(browser.get("clarification") or re.search(
            r"哪张|哪幅|哪一|多[个幅张]|请.*(?:指定|补充|明确)|位置|图序", "\n".join(browser.get("notices", []))))
        axes["answer"] = standard(completed and asks_details and not browser_answer)
        if browser_answer and re.search(r"\bmrad\b|毫弧度|\bdeg\b|(?:横轴|单位).{0,9}度", browser_answer, re.I):
            severe.append("ambiguity_answered_as_single_definitive_object")

    if kind == "missing":
        axes["object"] = standard(not source or any(e.get("locator", {}).get("object_type") == "table" for e in evidences))
        axes["read"] = standard(not any(read.get("method") == "vision" for read in read_records)
                                and trace.get("external_call_count", 0) == 0)
        answer = browser_answer + "\n" + "\n".join(browser.get("notices", []))
        acknowledges_missing = re.search(r"(?:缺少|无|没有|未见|没找到|查不到|没(?:法|有)|未(?:读到|找到|提供)|不能|无法|不支持|证据不足)", answer)
        unsupported_value = re.search(r"\d+(?:\.\d+)?\s*(?:毫瓦|mW\b)", browser_answer, re.I)
        axes["answer"] = standard(completed and bool(acknowledges_missing) and not unsupported_value)
        if unsupported_value:
            severe.append("missing_supported_as_measured_value")

    if kind in {"history", "history_setup"}:
        versions = [1] if kind == "history_setup" else [1, 2]
        referents = {actual(evidence).get("version"): evidence for evidence in evidences
                     if evidence.get("file_name") == "BenchRetest.ipynb"}
        axes["object"] = standard(all(referents.get(v, {}).get("locator", {}).get("cell_index") == 0 and
                                      "output" in actual(referents[v]).get("source_id", "") for v in versions))
        axes["read"] = standard(all(any(
            record["method"] == "native_text" and record.get("status") in READ_OK
            and record.get("actual_source") == actual(referents.get(v, {}))
            and PROTOCOL["history_truth"][str(v)] in record["text"]
            for record in referents.get(v, {}).get("reads", [])) for v in versions))
        axes["version"] = standard(all(valid_revision(referents.get(v, {}), "BenchRetest.ipynb", v) for v in versions)
                                   and len({actual(referents.get(v, {})).get("document_id") for v in versions}) == 1)
        axes["citation"] = standard(all(valid_text_open(directory, name, http, message_id, referents.get(v, {}), v,
                                                      PROTOCOL["history_truth"][str(v)]) for v in versions))
        axes["answer"] = standard(completed and all(PROTOCOL["history_truth"][str(v)] in browser_answer for v in versions))
        if kind == "history":
            for v in versions:
                line = next((line for line in browser_answer.splitlines()
                             if re.search(rf"(?:\bv{v}\b|第?\s*{v}\s*版)", line, re.I)), "")
                if line and PROTOCOL["history_truth"][str(3-v)] in line and PROTOCOL["history_truth"][str(v)] not in line:
                    severe.append("historical_and_current_values_misattributed")
        for row in http:
            if row["category"] != "citation" or row["status"] != 200 or f"/messages/{message_id}/" not in row["path"]:
                continue
            for version, evidence in referents.items():
                if row["path"].endswith("/" + evidence["evidence_id"]):
                    if row["response"].get("source") != actual(evidence):
                        severe.append("citation_opens_wrong_historical_version")

    critical_axes = [value for value in axes.values() if value != NA]
    overall = FAIL if FAIL in critical_axes else (UNTESTED if UNTESTED in critical_axes else PASS)
    return {"case": name, "axes": axes, "overall": overall,
            "first_failure": resolve_first_failure(axes), "serious_errors": sorted(set(severe)),
            "message_id": message_id, "elapsed_ms": turn_row.get("elapsed_ms"),
            "external_call_count": trace.get("external_call_count"), "vision_trace": trace.get("vision"),
            "candidate_sources": trace.get("object_candidates"), "selected_source": source,
            "sources": evidences, "displays": image_blocks, "browser_answer": browser.get("answer"),
            "browser_notices": browser.get("notices")}


def valid_text_open(directory, name, http, message_id, evidence, version, value=None):
    target = evidence.get("evidence_id")
    if not target:
        return False
    for row in http:
        if row["category"] != "citation" or row["status"] != 200:
            continue
        if not row["path"].endswith(f"/messages/{message_id}/evidence/{target}"):
            continue
        response = row["response"] or {}
        if response.get("source") != actual(evidence) or (value and value not in response.get("text", "")):
            continue
        if name == "notebook-both" and response.get("is_current") is not (version == 2):
            continue
        for path in directory.glob(name + "-citation-*.json"):
            reader = read_json(path)
            if reader_matches(reader, evidence["file_name"], version) and (not value or value in reader.get("body", "")):
                return True
    return False


def score(directory):
    http = read_lines(directory / "http.jsonl")
    calls = read_lines(directory / "provider.jsonl")
    requests = {((row.get("response") or {}).get("message") or {}).get("message_id"): row
                for row in http if row["category"] == "turn"}
    config = read_json(directory / "config.json")
    rows = [score_case(directory, case, requests, http, calls, config["vision_enabled"]) for case in PROTOCOL["cases"]]
    blocked = [call for call in calls if call["event"] == "blocked_before_egress"]
    return {"runtime": config, "browser": read_json(directory / "browser.json"), "cases": rows,
            "provider_events": calls, "guard_blocked_attempts": len(blocked),
            "total_provider_calls": sum(call.get("is_external", False) for call in calls)}


def gate(baseline, candidate):
    left, right = baseline["runtime"], candidate["runtime"]
    config_equal = all(left[k] == right[k] for k in ("vision_enabled", "vision_model", "soffice"))
    improved, lost, unresolved = [], [], []
    severe = []
    for before, after in zip(baseline["cases"], candidate["cases"], strict=True):
        assert before["case"] == after["case"]
        for axis, before_status in before["axes"].items():
            after_status = after["axes"][axis]
            if before_status == FAIL and after_status == PASS:
                improved.append([before["case"], axis])
            if before_status == PASS and after_status != PASS:
                lost.append([before["case"], axis])
        if before["overall"] == FAIL and after["overall"] != PASS:
            unresolved.append(after["case"])
        severe.extend([[after["case"], reason] for reason in after["serious_errors"]
                       if reason not in before["serious_errors"]])
    baseline_passes = sum(case["overall"] == PASS for case in baseline["cases"])
    candidate_passes = sum(case["overall"] == PASS for case in candidate["cases"])
    positive_axis = any(axis in {"object", "display", "read", "citation"} for _, axis in improved)
    normal_complete = all(case["axes"]["normal_entry"] == PASS for run in (baseline, candidate) for case in run["cases"])
    admissible = (config_equal and normal_complete and positive_axis and candidate_passes >= baseline_passes and not severe
                  and candidate["guard_blocked_attempts"] <= baseline["guard_blocked_attempts"])
    return {"comparable_settings": config_equal, "all_normal_entries_captured": normal_complete,
            "baseline_passes": baseline_passes, "candidate_passes": candidate_passes,
            "gains": improved, "regressions": lost, "new_serious_errors": severe,
            "unresolved": unresolved, "second_suite_gate_pass": admissible}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    baseline, candidate = score(args.baseline), score(args.candidate)
    result = {"baseline": baseline, "candidate": candidate, "gate": gate(baseline, candidate)}
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["gate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
