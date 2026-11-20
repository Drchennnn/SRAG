"""Collect the independent browser replay as JSON using only public fixture ledgers."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def by_path(events, url):
    path = urlsplit(url).path
    return [
        result
        for item in events
        if item["path"] == path and item["method"] == "GET"
        for result in [
            {
                "status_code": item["status_code"],
                "content_type": item["content_type"],
                **{key: item[key] for key in ("sha256", "bytes") if key in item},
            }
        ]
    ]


def collect(folder):
    events = load_lines(folder / "http.jsonl")
    calls = load_lines(folder / "provider.jsonl")
    manifest = load(folder / "manifest.json")
    turns = []
    for count, event in enumerate((item for item in events if "turn" in item), 1):
        request = event["turn"]
        case = request["case"]
        assert request["text"] == manifest["questions"][case]
        assert request["kind"] == "ask" and request.get("anchor") is None
        response = event["response"]
        turn = response.get("turn", {})
        message = response.get("message") or {}
        dom_path = folder / f"{count}-{case}.json"
        dom = load(dom_path) if dom_path.exists() else None
        if dom is not None and not dom.get("missing") and dom["id"] != message.get("message_id"):
            dom = (
                None  # The original harness indexed assistant replies before tracking failed turns.
            )
        actual_visible = folder / f"{count}-{case}-actual-visible.json"
        visible = load(actual_visible) if actual_visible.exists() else []
        for card in visible:
            card["http"] = by_path(events, card["url"])
        open_paths = sorted(folder.glob(f"{count}-{case}*-citation-*.json")) if dom else []
        opened = []
        for path in open_paths:
            pane = load(path)
            pane["evidence_id"] = path.stem.rsplit("-citation-", 1)[1]
            pane["record"] = path.stem
            pane["media"] = by_path(events, pane["url"]) if pane.get("url") else []
            opened.append(pane)
        turns.append(
            {
                "case": case,
                "question": request["text"],
                "client_turn_id": request["client_turn_id"],
                "scope_revision": request["scope_revision"],
                "turn_status": turn.get("status"),
                "error": turn.get("error"),
                "http_elapsed_ms": event["elapsed_ms"],
                "trace": turn.get("trace"),
                "resolution": turn.get("resolution"),
                "message_id": message.get("message_id"),
                "blocks": message.get("blocks", []),
                "sources": message.get("sources", {}),
                "visible_display_ids": message.get("visible_display_ids", []),
                "actual_visible_cards": visible,
                "dom": dom,
                "opened_citations": opened,
                "provider_calls": [
                    call
                    for call in calls
                    if call["turn"] and call["turn"]["client_turn_id"] == request["client_turn_id"]
                ],
            }
        )
    imports = [e["response"] for e in events if e["path"].endswith("/import")]
    return {
        "subject": (folder / "subject.txt").read_text(encoding="utf-8").strip(),
        "chromium": load(folder / "chromium.json"),
        "http_ledger_sha256": hashlib.sha256((folder / "http.jsonl").read_bytes()).hexdigest(),
        "provider_ledger_sha256": hashlib.sha256(
            (folder / "provider.jsonl").read_bytes()
        ).hexdigest(),
        "fixture": manifest,
        "imports": imports,
        "provider_counts": {
            service: sum(call["sent"] and call.get("provider") == service for call in calls)
            for service in ("deepseek", "dashscope")
        },
        "blocked_attempts": sum(not call["sent"] for call in calls),
        "turns": turns,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    args.result.write_text(
        json.dumps(collect(args.folder), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
