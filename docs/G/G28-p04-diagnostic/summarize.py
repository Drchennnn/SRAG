"""Report generation stages for the synthetic public fixture only."""

import argparse
import hashlib
import json
from pathlib import Path


def summarize(path):
    raw = path.read_bytes()
    trace = json.loads(raw)
    stages = trace["generation_stages"]
    by_stage = {}
    for stage in stages:
        by_stage.setdefault(stage["stage"], []).append(stage["value"])

    def event(name, index):
        events = by_stage.get(name)
        return events[index] if events else None

    completed = []
    for index, row in enumerate(trace["turns"]):
        model = event("model_input", index)
        messages = model["messages"]
        prompt = json.loads(messages[1]["content"])
        wire = event("outgoing_generation_payload", index)
        before_client = event("provider_content_before_client_normalization", index)
        before_parse = event("provider_reply", index)
        parsed = event("parsed_reply", index)
        filtered = event("technical_filter", index)
        assert row["question"] == prompt["question"], row["id"]
        if wire is not None:
            assert wire["payload"]["messages"] == messages, row["id"]
            assert before_client["content"].strip() == before_parse, row["id"]
        target = [e for e in prompt["evidence"] if e["source"]["source_id"] == "page:9"]
        contents = row["message"]["blocks"]
        final = [{k: b[k] for k in ("text", "citations")} for b in contents if b["type"] == "text"]
        assert parsed["statements"] == filtered["before"] and filtered["after"] == final, row["id"]
        clicks = [
            {
                "evidence_id": item["evidence_id"],
                **{
                    key: item[key]
                    for key in ("title", "location", "pixels", "first_click", "error")
                    if key in item
                },
            }
            for item in row["reader"]
        ]
        completed.append(
            {
                "id": row["id"],
                "question": prompt["question"],
                "mode": prompt["mode"],
                "previous_context_keys": sorted(prompt["previous_context"]),
                "previous_question_names_target": (
                    "Acceptance Grid" in prompt["previous_context"].get("previous_question", "")
                ),
                "source_count": len(prompt["evidence"]),
                "includes_notebook_output": any(
                    e["source"]["source_id"] == "cell:0:output:0" for e in prompt["evidence"]
                ),
                "target_fragments": [{"id": e["id"], "reads": e["reads"]} for e in target],
                "wire_body_sha256": wire["body_sha256"] if wire else None,
                "wire_messages_match_generation_input": True if wire else None,
                "wire_parameter_max_tokens": wire["payload"]["max_tokens"] if wire else None,
                "provider_content": before_client["content"] if before_client else before_parse,
                "strict_parser_retained_count": len(parsed["statements"]),
                "technical_filter_removed_count": len(filtered["before"]) - len(filtered["after"]),
                "final_statements_match_postfilter": True,
                "notices": [b for b in contents if b["type"] == "notice"],
                "actual_citation_clicks": clicks,
                "provider_http_statuses": [c.get("status") for c in row["provider_calls"]],
            }
        )
    return {
        "subject": trace["subject"],
        "manifest_sha256": trace["manifest_sha256"],
        "raw_trace_sha256": hashlib.sha256(raw).hexdigest(),
        "turns": completed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("supplement", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "first": summarize(args.first),
                "wire_supplement": summarize(args.supplement),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
