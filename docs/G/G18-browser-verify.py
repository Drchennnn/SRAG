"""Evaluate the G18 browser DOM and recorded HTTP bytes without accessing the service."""

import argparse
import base64
import json
from hashlib import sha256
from pathlib import Path


def check(out, subject, no_renderer):
    def load(path):
        return json.loads((out / path).read_text(encoding="utf-8"))

    events = [json.loads(line) for line in (out / "http.jsonl").read_text().splitlines()]
    turns = [event for event in events if "turn" in event]
    assert [event["turn"]["kind"] for event in turns] == [
        "ask", "ask", "summary", "search", "ask", "search", "ask", "ask"
    ]
    assert all(event["status"] == 200 for event in turns)
    assert all(not event["turn"].get("anchor") for event in turns[:7])
    assert turns[7]["turn"]["anchor"]["type"] == "display"
    assert [load(f"message-{i}.json")["draft"] for i in (3, 4)] == ["等一下继续问这份文件。"] * 2
    assert load("composer.json") == {"buttons": 1, "modes": 0}
    assert "August 23" in load("message-1.json")["text"]
    assert "August 23" in load("message-2.json")["text"]
    assert "Cedar" in load("message-6.json")["text"]
    assert not (out / "provider.jsonl").exists()

    def reader(case, file, *, rendered=True, read=False):
        pane = load(f"reader-{case}.json")
        status = "图像已读取" if read else "图像未读取"
        assert pane["open"] and file in pane["title"] and status in pane["status"]
        path = pane["chosen"].split("?", 1)[0]
        matching = [event for event in events if event["path"] == path]
        assert matching, path
        event = matching[-1]
        if rendered:
            assert min(pane["image"]) > 0 and event["status"] == 200
            assert event["content_type"].startswith("image/png") and event["size"] > 0
        else:
            assert pane["image"] == [0, 0] and event["status"] == 409
            assert "暂时无法渲染" in pane["body"]
        return {"pane": pane, "http": event}

    pdf = "Northpoint-Completion-Briefing.pdf · v1 · 第 5 页"
    catalog = reader("pdf-tree", pdf)
    citation = reader("pdf-citation", pdf)
    historical = reader("pdf-historical", pdf)
    assert catalog["pane"]["layout"] == "sticky"
    assert catalog["pane"]["status"] == "仅展示 · 图像未读取"
    narrow = citation["pane"]
    assert narrow["details"] == "当前版本"
    assert narrow["viewport"] == [680, 700] and narrow["layout"] == "fixed"
    assert narrow["width"] <= 680 and not narrow["horizontalOverflow"]
    assert abs(narrow["scroll"] - load("scroll-before-narrow.json")) <= 2
    assert historical["pane"]["details"] == "历史版本"
    evidence_path = citation["http"]["path"].removesuffix("/preview")
    text_versions = [event["response"] for event in events if event["path"] == evidence_path]
    assert [item["is_current"] for item in text_versions] == [True, False]
    assert text_versions[0]["source"] == text_versions[1]["source"]
    assert text_versions[0]["source"]["version"] == 1
    pdf_renders = [event for event in events if event["path"] == citation["http"]["path"]]
    assert len(pdf_renders) == 2 and pdf_renders[0]["sha256"] == pdf_renders[1]["sha256"]

    pptx = "Drum-Readings.pptx · v1 · 第 1 页"
    tree = reader("pptx-tree", pptx, rendered=not no_renderer)
    crop = reader("pptx-crop", pptx, rendered=not no_renderer)
    table = reader("pptx-citation", pptx, rendered=not no_renderer)
    if not no_renderer:
        assert crop["http"]["sha256"] != tree["http"]["sha256"]
        assert crop["pane"]["image"][0] < tree["pane"]["image"][0]
    assert "Cedar" in table["pane"]["body"] and "Frequency" in table["pane"]["body"]
    notebook = reader("notebook-output", "Drum-Trend.ipynb · v1 · Cell 1 · 输出 1")
    image = load("fixtures/v1/Drum-Trend.ipynb")["cells"][0]["outputs"][1]["data"]["image/png"]
    saved_image = base64.b64decode("".join(image))
    assert notebook["http"]["sha256"] == sha256(saved_image).hexdigest()
    follow = reader(
        "notebook-follow-citation", "Drum-Trend.ipynb · v1 · Cell 1 · 输出 1", read=True
    )
    assert follow["http"]["sha256"] == notebook["http"]["sha256"]
    read_records = [
        json.loads(line) for line in (out / "simulated-vision.jsonl").read_text().splitlines()
    ]
    assert len(read_records) == 1 and read_records[0]["input_sha256"] == follow["http"]["sha256"]
    follow_response = turns[7]["response"]
    follow_message = follow_response["message"]
    follow_blocks = follow_message["blocks"]
    assert all(block["type"] != "source_image" for block in follow_blocks)
    follow_read = follow_message["sources"]["ev-1"]
    assert follow_read["reads"][0]["actual_input_sha256"] == follow["http"]["sha256"]
    isolated = load("session-b.json")
    original = load("session-a.json")
    assert isolated["session"] != original and isolated["empty"] and not isolated["preview"]

    imports = [event["response"] for event in events if event["path"].endswith("/import")]
    assert len(imports) == 4 and [event["version"] for event in imports] == [1, 2, 1, 1]
    assert imports[0]["document_id"] == imports[1]["document_id"]
    result = {
        "subject": subject,
        "chromium": load("browser.json"),
        "renderer": "unavailable" if no_renderer else "libreoffice",
        "outgoing_provider_calls": 0,
        "simulated_vision_reads": read_records,
        "turns": [
            {"request": event["turn"], "dom": load(f"message-{i}.json"), "status": event["status"]}
            for i, event in enumerate(turns, 1)
        ],
        "catalog_pdf": catalog,
        "citation_pdf_narrow": citation,
        "citation_pdf_narrow_scroll_before": load("scroll-before-narrow.json"),
        "citation_pdf_historical": historical,
        "pdf_source_before_and_after": text_versions,
        "catalog_pptx": tree,
        "catalog_pptx_crop": crop,
        "citation_pptx": table,
        "notebook_output": notebook,
        "notebook_follow_up_citation": follow,
        "notebook_follow_up_replayed_read": {
            "execution": "synthetic_provider_reply",
            "resolution": follow_response["turn"]["resolution"],
            "evidence": follow_read,
            "display_attached_to_reply": False,
        },
        "cross_session": {"original": original, "new": isolated},
        "imports": imports,
    }
    target = out / "G18-browser-result.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("subject")
    parser.add_argument("no_renderer", choices=("true", "false"))
    args = parser.parse_args()
    check(args.out, args.subject, args.no_renderer == "true")
