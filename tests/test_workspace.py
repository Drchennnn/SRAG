import base64
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.stores.workspace as workspace_store
from app.api.workspace import get_workspace_service, router
from app.config import Settings
from app.rag.workspace import Draft, make_workspace

_PUBLIC = Path("docs/A/样例")
_BASE = "/api/workspace"
_HOST = "http://127.0.0.1:8001"
_COOKIE = "srag_workspace_owner"
_COURSE = "public-signals"


@pytest.fixture
def system(tmp_path):
    settings = Settings(
        sqlite_path=tmp_path / "db.sqlite3",
        data_dir=tmp_path / "data",
        course_root=tmp_path / "originals",
        workspace_course_ids=f"{_COURSE},other",
        workspace_max_upload_bytes=120 * 1024,
        source_vision_enabled=False,
        dashscope_api_key="",
        _env_file=None,
    )
    service = make_workspace(settings)
    app = FastAPI()
    app.include_router(router, prefix=_BASE)
    app.dependency_overrides[get_workspace_service] = lambda: service
    with TestClient(app, base_url=_HOST) as client:
        yield settings, service, app, client


def _create(client, **payload):
    response = client.post(
        f"{_BASE}/sessions",
        json=payload or {"course_ids": [_COURSE]},
        headers={"origin": _HOST},
    )
    assert response.status_code == 201, response.text
    return f"{_BASE}/sessions/{response.json()['session_id']}"


def _upload(client, session, path: Path, *, filename=None, document_id=None, course=_COURSE):
    params = {"course_id": course, "filename": filename or path.name}
    if document_id:
        params["document_id"] = document_id
    response = client.post(
        session + "/import",
        params=params,
        headers={"content-type": "application/octet-stream", "origin": _HOST},
        content=path.read_bytes(),
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "published", data
    assert "source_path" not in data
    return data


def _flat(items):
    for item in items:
        yield item
        yield from _flat(item["children"])


def _node(client, session, document_id, kind, **locator):
    response = client.get(session + "/catalog")
    assert response.status_code == 200, response.text
    for course in response.json()["courses"]:
        for doc in course["documents"]:
            if doc["document_id"] == document_id:
                for item in [doc, *_flat(doc["nodes"])]:
                    if item["kind"] == kind and all(
                        item["locator"].get(key) == value for key, value in locator.items()
                    ):
                        return item
    raise AssertionError((document_id, kind, locator))


def _focus(client, session, node, revision=1):
    response = client.put(
        session + "/focus", json={"node_id": node["node_id"], "scope_revision": revision}
    )
    assert response.status_code == 200, response.text
    return response.json()["scope_revision"]


def _ask(client, session, text, revision=1, *, client_id=None, anchor=None, kind="ask"):
    request = {
        "client_turn_id": client_id or text,
        "scope_revision": revision,
        "kind": kind,
        "text": text,
    }
    if anchor:
        request["anchor"] = anchor
    response = client.post(session + "/turns", json=request)
    assert response.status_code == 200, response.text
    return response.json()


def _notebook_case(client):
    session = _create(client)
    imported = _upload(client, session, _PUBLIC / "Signal-lab.ipynb")
    node = _node(client, session, imported["document_id"], "cell", cell_index=1)
    revision = _focus(client, session, node)
    mail = _ask(client, session, "显示本 Cell 的两张原件图片", revision)
    assert mail["turn"]["status"] == "completed", mail
    images = [block for block in mail["message"]["blocks"] if block["type"] == "source_image"]
    assert len(images) == 2
    return session, imported, node, revision, mail, images


def _published_bytes(version, index):
    p = _PUBLIC / ("Signal-lab.ipynb" if version == 1 else "revisions/v2/Signal-lab.ipynb")
    return base64.b64decode(
        json.loads(p.read_text())["cells"][1]["outputs"][index]["data"]["image/png"]
    )


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def test_real_upload_and_two_followup_paths_preserve_order_and_offline_state(system):
    _, _, _, client = system
    session, imported, node, revision, first, images = _notebook_case(client)
    assert node["parent_id"] and node["previous_id"]
    assert [item["display_id"] for item in images] == first["message"]["visible_display_ids"]
    assert len(set(first["message"]["visible_display_ids"])) == 2
    assert [item["display_source"]["source_id"] for item in images] == [
        "cell:1:output:1",
        "cell:1:output:2",
    ]
    assert all(item["read_status"] == "not_attempted" for item in images)
    for index, item in enumerate(images, 1):
        media = client.get(item["media_url"])
        assert media.headers["content-type"] == "image/png"
        assert media.headers["cache-control"] == "private, no-store"
        assert _sha(media.content) == _sha(_published_bytes(1, index))
    anchor = {
        "type": "display",
        "message_id": first["message"]["message_id"],
        "display_id": images[1]["display_id"],
    }
    card = _ask(client, session, "横轴是什么？", revision, anchor=anchor)
    natural = _ask(client, session, "刚才第二张图", revision)
    assert card["turn"]["resolution"]["method"] == "display_anchor"
    assert natural["turn"]["resolution"] == {
        "method": "visible_ordinal",
        "resolution": "visible_ordinal",
        "message_id": anchor["message_id"],
        "display_id": anchor["display_id"],
        "ordinal": 2,
        "actual_source": {
            "source_id": "cell:1:output:2",
            "version": 1,
            "document_id": imported["document_id"],
        },
    }
    assert natural["turn"]["trace"]["external_call_count"] == 0
    assert natural["message"]["sources"]["ev-1"]["read_status"] == "not_attempted"
    restored = client.get(session).json()
    assert restored["messages"][1]["visible_display_ids"] == first["message"]["visible_display_ids"]
    assert restored["focus"]["node_id"] == node["node_id"]


def test_bootstrap_cross_origin_isolation_and_restored_clarification(system):
    _, _, app, client = system
    assert client.get(_BASE + "/sessions").status_code == 404
    for headers in (
        {"origin": "https://outside.invalid"},
        {"sec-fetch-site": "cross-site"},
        {"host": "rebinding.invalid"},
        {"authorization": "Bearer invalid"},
    ):
        response = client.post(_BASE + "/sessions", json={}, headers=headers)
        assert response.status_code == 404
        assert "set-cookie" not in response.headers
    session, _, _, revision, first, images = _notebook_case(client)
    other_session = _create(client, course_ids=["other"])
    anchors = {
        "type": "display",
        "message_id": first["message"]["message_id"],
        "display_id": images[0]["display_id"],
    }
    denied = client.post(
        other_session + "/turns",
        json={
            "text": "读这张图",
            "client_turn_id": "foreign",
            "kind": "ask",
            "scope_revision": 1,
            "anchor": anchors,
        },
    )
    assert denied.status_code == 404 and denied.json()["detail"]["code"] == "not_accessible"
    with TestClient(app, base_url=_HOST) as independent:
        own = _create(independent)
        assert own != session
        assert independent.get(session).status_code == 404
        assert independent.get(images[0]["media_url"]).status_code == 404
        assert len(independent.get(_BASE + "/sessions").json()["sessions"]) == 1
    assert len(client.get(_BASE + "/sessions").json()["sessions"]) == 2

    clarification = _ask(client, session, "那张图的单位？", revision)
    assert clarification["turn"]["status"] == "clarification"
    assert len(clarification["options"]) == 2
    restored = client.get(session).json()["turns"][-1]
    assert restored["options"] == clarification["options"]
    assert restored["turn_id"] == clarification["message"]["turn_id"]
    resolved = _ask(
        client,
        session,
        "选择第二项",
        revision,
        anchor={
            "type": "clarification",
            "turn_id": restored["turn_id"],
            "option_id": restored["options"][1]["option_id"],
        },
    )
    assert resolved["turn"]["resolution"]["method"] == "clarification_choice"
    assert resolved["turn"]["resolution"]["actual_source"]["source_id"] == "cell:1:output:2"


def test_old_directory_ids_fail_after_release_same_content_epoch_and_scope_switch(system):
    settings, service, _, client = system
    session, imported, node, revision, _, images = _notebook_case(client)
    original = client.get(images[1]["media_url"]).content
    updated = _upload(
        client,
        session,
        _PUBLIC / "revisions/v2/Signal-lab.ipynb",
        document_id=imported["document_id"],
    )
    new_node = _node(client, session, updated["document_id"], "cell", cell_index=1)
    assert new_node["node_id"] != node["node_id"] and new_node["version"] == 2
    assert (
        client.put(
            session + "/focus", json={"node_id": node["node_id"], "scope_revision": revision}
        ).status_code
        == 404
    )
    assert client.get(session + f"/catalog/{node['node_id']}/media").status_code == 404
    stale = _ask(client, session, "看当前 Cell", revision)
    assert stale["turn"]["error"] == "stale_reference"
    assert client.get(images[1]["media_url"]).content == original
    history = _ask(client, session, "刚才第二张图", revision)
    assert history["turn"]["resolution"]["actual_source"]["version"] == 1
    assert history["message"]["blocks"][0]["code"] == "historical_only"

    lecture = _upload(client, session, _PUBLIC / "Signal-reference.pdf")
    patched = client.patch(
        session + "/scope",
        json={
            "scope_revision": revision,
            "course_ids": [_COURSE],
            "document_ids": [lecture["document_id"]],
        },
    ).json()
    assert patched["focus"] is None
    assert client.get(images[1]["media_url"]).content == original
    new_summary = _ask(client, session, "总结当前资料", patched["scope_revision"], kind="summary")
    assert new_summary["message"]["sources"]
    assert all(
        ev["revision"]["source"]["document_id"] == lecture["document_id"]
        for ev in new_summary["message"]["sources"].values()
    )
    assert service.library.delete(updated["document_id"])
    path = settings.course_root / "same-v2.ipynb"
    path.write_bytes((_PUBLIC / "revisions/v2/Signal-lab.ipynb").read_bytes())
    again = service.library.import_paths(_COURSE, [path], document_id=updated["document_id"])[0]
    assert again.version == 2
    assert again.publication_epoch > updated["publication_epoch"]
    replacement = _node(client, session, updated["document_id"], "cell", cell_index=1)
    assert replacement["node_id"] != new_node["node_id"]


def test_revocation_masks_turn_evidence_trace_options_and_media(system):
    _, service, _, client = system
    session, imported, _, revision, first, images = _notebook_case(client)
    ordinary = _ask(client, session, "刚才第二张图", revision)
    clarification = _ask(client, session, "那张图", revision)
    assert ordinary["turn"]["resolution"] and clarification["options"]
    assert service.library.delete(imported["document_id"])
    assert client.get(images[1]["media_url"]).status_code == 404
    restored = client.get(session)
    assert imported["document_id"] not in restored.text
    assert "Signal-lab.ipynb" not in restored.text
    assert images[1]["display_id"] not in restored.text
    assert "Two saved views" not in restored.text
    for turn in restored.json()["turns"]:
        assert turn["resolution"] is None and turn["trace"] is None and turn["options"] == []
    assert client.get(session + "/turns/刚才第二张图").json()["turn"]["error"] == "not_accessible"
    assert first["turn"]["trace"] is not None


def test_only_successful_image_blocks_count_even_if_source_is_repeated(system, monkeypatch):
    _, service, _, client = system
    real = service.evidence.render_current

    def obsolete(request):
        result = real(request)
        if request.source.source_id == "cell:1:output:1":
            return result.model_copy(update={"status": "rendition_unavailable", "display": None})
        return result

    monkeypatch.setattr(service.evidence, "render_current", obsolete)
    original = Draft.add_image

    def append_twice(self, evidence, result, request, label):
        original(self, evidence, result, request, label)
        if result.display is not None:
            original(self, evidence, result, request, label)

    monkeypatch.setattr(Draft, "add_image", append_twice)
    session, _, _, revision, first, images = _notebook_case(client)
    assert any(item["type"] == "notice" for item in first["message"]["blocks"])
    assert images[0]["display_id"] != images[1]["display_id"]
    assert images[0]["display_source"] == images[1]["display_source"]
    follow = _ask(client, session, "刚才第二张图", revision)
    assert follow["turn"]["resolution"]["display_id"] == images[1]["display_id"]


def test_retry_and_cancel_keep_one_user_message_and_ignore_late_results(system, monkeypatch):
    _, service, _, client = system
    session = _create(client)
    _upload(client, session, _PUBLIC / "Signal-lab.ipynb")
    entered, release = Event(), Event()
    query = service.searcher.search

    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=3)
        return query(*args, **kwargs)

    monkeypatch.setattr(service.searcher, "search", slow)
    request = {"text": "NotebookImage", "client_turn_id": "in-flight", "scope_revision": 1}
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(client.post, session + "/turns", json=request)
        try:
            assert entered.wait(timeout=3)
            dup = client.post(session + "/turns", json=request)
            assert dup.status_code == 202
            different = client.post(session + "/turns", json={**request, "text": "other"})
            assert different.status_code == 409
            gone = client.post(session + "/turns/in-flight/cancel").json()
            assert gone["turn"]["status"] == "cancelled"
        finally:
            release.set()
        assert pending.result(timeout=3).json()["turn"]["status"] == "cancelled"
    assert len(client.get(session).json()["messages"]) == 1

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("test-controlled outage")

    monkeypatch.setattr(service.searcher, "search", unavailable)
    failed = _ask(client, session, "NotebookImage", client_id="retry", kind="search")
    assert failed["turn"]["status"] == "failed" and failed["turn"]["retryable"]
    monkeypatch.setattr(service.searcher, "search", query)
    again = _ask(client, session, "NotebookImage", client_id="retry", kind="search")
    assert again == failed
    retried = client.post(session + "/turns/retry/retry").json()
    assert retried["turn"]["status"] == "completed"
    assert retried["turn"]["turn_id"] == failed["turn"]["turn_id"]
    timeline = client.get(session).json()
    assert [item["role"] for item in timeline["messages"]] == ["user", "user", "assistant"]


def test_interrupted_attempt_lease_recovers_and_cannot_commit_late(system, monkeypatch):
    _, service, _, client = system
    session = _create(client)
    request = {
        "text": "NotebookImage",
        "client_turn_id": "dead",
        "scope_revision": 1,
        "kind": "search",
    }
    from app.schemas.workspace import TurnRequest

    old, started = service.store.start(
        client.cookies.get(_COOKIE), session.rsplit("/", 1)[1], TurnRequest.model_validate(request)
    )
    assert started
    now = workspace_store.time()
    monkeypatch.setattr(workspace_store, "time", lambda: now + 301)
    result = client.get(session + "/turns/dead").json()
    assert result["turn"]["error"] == "interrupted"
    assert client.post(session + "/turns/dead/retry").json()["turn"]["status"] == "completed"
    late = service.store.finish(
        old,
        "failed",
        {"error": "synthetic_late"},
        None,
        (),
        (),
        service.allowed_courses,
    )
    assert late.status == "completed" and "error" not in late.outcome


def test_upload_rejects_escape_large_body_and_unauthorized_course(system):
    settings, _, _, client = system
    session = _create(client)
    for name, body, course, status in (
        ("../escape.pdf", b"test", _COURSE, 422),
        ("escape.svg", b"<svg/>", _COURSE, 422),
        ("valid.pdf", b"x" * (settings.workspace_max_upload_bytes + 1), _COURSE, 413),
        ("valid.pdf", b"test", "unlisted", 404),
        ("valid.pdf", b"", _COURSE, 422),
    ):
        result = client.post(
            session + "/import",
            params={"filename": name, "course_id": course},
            headers={"Content-Type": "application/octet-stream"},
            content=body,
        )
        assert result.status_code == status, result.text
    assert not list(settings.course_root.rglob("*"))


def test_terminal_and_http_use_the_same_service_and_store(system):
    settings, _, _, client = system
    session = _create(client)
    imported = _upload(client, session, _PUBLIC / "Signal-lab.ipynb")
    env = {
        **os.environ,
        "SQLITE_PATH": str(settings.sqlite_path),
        "DATA_DIR": str(settings.data_dir),
        "COURSE_ROOT": str(settings.course_root),
        "WORKSPACE_COURSE_IDS": _COURSE,
        "SOURCE_VISION_ENABLED": "false",
        "DASHSCOPE_API_KEY": "",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.workspace_chat",
            "--json",
            "--course",
            _COURSE,
            "--file",
            imported["document_id"],
            "--kind",
            "summary",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    output = json.loads(result.stdout)
    assert output["turn"]["status"] == "completed"
    assert output["message"]["sources"]
    assert not (settings.data_dir / "workspace_cli_owner").stat().st_mode & 0o077
    assert len(client.get(_BASE + "/sessions").json()["sessions"]) == 1


def test_explicit_page_overrides_existing_focus_when_choosing_media(system):
    _, _, _, client = system
    session = _create(client)
    imported = _upload(client, session, _PUBLIC / "Signal-reference.pdf")
    first_page = _node(client, session, imported["document_id"], "page", page=1)
    revision = _focus(client, session, first_page)
    other_page = _ask(client, session, "打开第2页", revision)
    images = [item for item in other_page["message"]["blocks"] if item["type"] == "source_image"]
    assert images[0]["display_source"]["source_id"] == "page:2"
    assert client.get(images[0]["media_url"]).content


def test_summary_shows_partial_when_a_base_page_has_only_scan(system, tmp_path):
    from PIL import Image
    from pypdfium2 import PdfDocument

    _, _, _, client = system
    image_page = tmp_path / "image-only.pdf"
    Image.new("RGB", (120, 80), "white").save(image_page, format="PDF")
    mixed = tmp_path / "mixed.pdf"
    with (
        PdfDocument(_PUBLIC / "Signal-reference.pdf") as text_pdf,
        PdfDocument(image_page) as scan_pdf,
        PdfDocument.new() as two_pages,
    ):
        two_pages.import_pages(text_pdf, pages=[0])
        two_pages.import_pages(scan_pdf)
        two_pages.save(mixed)

    session = _create(client)
    imported = _upload(client, session, mixed)
    summary = _ask(client, session, "总结这份资料", kind="summary")
    trace = summary["turn"]["trace"]
    assert trace["full_text_scope"] is False and trace["base_source_count"] == 2
    assert trace["uncovered_sources"] == [
        {
            "source": {
                "document_id": imported["document_id"],
                "source_id": "page:2",
                "version": 1,
            },
            "kind": "page",
            "read_status": "not_attempted",
            "reason": "needs_ocr",
        }
    ]
    assert "部分来源" in summary["message"]["blocks"][0]["text"]
    assert any(
        item["read_status"] in {"succeeded", "partial"}
        for item in summary["message"]["sources"].values()
    )


def test_bad_notebook_png_bytes_leave_room_for_later_real_display(system, tmp_path):
    from copy import deepcopy

    _, _, _, client = system
    data = json.loads((_PUBLIC / "Signal-lab.ipynb").read_text(encoding="utf-8"))
    outputs = data["cells"][1]["outputs"]
    bad = deepcopy(outputs[1])
    bad["data"]["image/png"] = base64.b64encode(b"declared PNG with corrupt payload").decode()
    data["cells"][1]["outputs"] = [bad, deepcopy(bad), outputs[1], outputs[2]]
    fixture = tmp_path / "mixed-images.ipynb"
    fixture.write_text(json.dumps(data), encoding="utf-8")
    session = _create(client)
    saved = _upload(client, session, fixture)
    cell = _node(client, session, saved["document_id"], "cell", cell_index=1)
    revision = _focus(client, session, cell)
    first = _ask(client, session, "显示本 Cell 的图片", revision)
    assert first["turn"]["status"] == "completed", first
    stats = first["turn"]["trace"]["display_attempts"]
    assert (stats["eligible"], stats["attempted"], stats["failed"], stats["displayed"]) == (
        4,
        4,
        2,
        2,
    )
    assert stats["attempt_limit"] > stats["display_limit"]
    assert [
        block["display_source"]["source_id"]
        for block in first["message"]["blocks"]
        if block["type"] == "source_image"
    ] == ["cell:1:output:2", "cell:1:output:3"]
    assert len(first["message"]["visible_display_ids"]) == 2
    second = _ask(client, session, "刚才第二张图", revision)
    assert second["turn"]["resolution"]["actual_source"]["source_id"] == "cell:1:output:3"


def test_summary_reports_clipped_and_budget_skipped_base_cells(system, tmp_path):
    _, _, _, client = system
    data = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "markdown",
                "id": f"cell-{number}",
                "metadata": {},
                "source": [
                    f"Section {number}\n"
                    + ("full source content " * 150 if number == 0 else "Short text.")
                ],
            }
            for number in range(7)
        ],
    }
    fixture = tmp_path / "excerpt-budget.ipynb"
    fixture.write_text(json.dumps(data), encoding="utf-8")
    session = _create(client)
    _upload(client, session, fixture)
    result = _ask(client, session, "总结这份资料", kind="summary")
    trace = result["turn"]["trace"]
    assert trace["full_text_scope"] is False and trace["base_source_count"] == 7
    assert [
        (gap["source"]["source_id"], gap["read_status"], gap["reason"])
        for gap in trace["uncovered_sources"]
    ] == [
        ("cell:0", "partial", "excerpt_truncated"),
        ("cell:6", "not_attempted", "source_budget"),
    ]


def _publish_supplement_case(client, session, tmp_path, *, incomplete=False):
    import nbformat

    sample = json.loads(Path("docs/F/样例/SignalTrialA7.json").read_text())
    published = {}
    for item in sample["documents"]:
        path = tmp_path / item["file_name"]
        text = item["text"]
        if incomplete and item["role"] == "lab":
            text = text.replace(" Trial: 0.18.", "")
        cell = nbformat.v4.new_markdown_cell(text, id="evidence")
        nbformat.write(nbformat.v4.new_notebook(cells=[cell]), path)
        published[item["role"]] = (path, _upload(client, session, path)["document_id"])
    return sample["question"], published


@pytest.mark.parametrize("incomplete", [False, True])
def test_normal_web_question_persists_only_grounded_final_citations(system, tmp_path, incomplete):
    _, _, app, client = system
    session = _create(client)
    question, files = _publish_supplement_case(client, session, tmp_path, incomplete=incomplete)
    result = _ask(client, session, question)
    assert result["turn"]["status"] == "completed", result
    trace = result["turn"]["trace"]
    assert trace["route"] == "evidence_answer"
    assert trace["external_call_count"] == 0
    assert trace["first_supported"] == ["definition"]
    assert trace["first_missing"] == ["experiment"]
    assert trace["supplement_count"] == 1
    assert trace["term"]["word"] == "SignalTrialA7"
    assert len(trace["final_evidence_ids"]) == trace["budget"] == 2
    assert "reads" not in trace
    assert trace["missing"] == (["experiment"] if incomplete else [])
    answer = result["message"]
    refs = answer["sources"]
    assert len(refs) == (1 if incomplete else 2)
    assert set(refs) == set(trace["cited_evidence_ids"])
    assert set(refs) == {
        citation for block in answer["blocks"] for citation in block.get("citations", [])
    }
    if incomplete:
        assert any("不足" in block["text"] for block in answer["blocks"])
    for key, evidence in refs.items():
        endpoint = session + f"/messages/{answer['message_id']}/evidence/{key}"
        original = client.get(endpoint)
        assert original.status_code == 200
        assert original.headers["cache-control"] == "private, no-store"
        saved = original.json()
        assert saved["source"] == evidence["revision"]["source"]
        assert saved["locator"]["cell_index"] == 0 and saved["is_current"]
        assert saved["text"] == evidence["reads"][0]["text"]
        with TestClient(app, base_url=_HOST) as other:
            _create(other)
            assert other.get(endpoint).status_code == 404
        docid = saved["source"]["document_id"]
        path = next(path for path, published_id in files.values() if published_id == docid)
        data = json.loads(path.read_text())
        data["cells"][0]["source"].append("\n新版内容")
        path.write_text(json.dumps(data, ensure_ascii=False))
        assert _upload(client, session, path, document_id=docid)["version"] == 2
        assert client.get(endpoint).json() == {**saved, "is_current": False}
    unknown = session + f"/messages/{answer['message_id']}/evidence/not-a-citation"
    assert client.get(unknown).status_code == 404
    narrowed = _ask(client, session, f"{question} 只查 {files['lecture'][0].name}")
    assert narrowed["turn"]["trace"]["effective_scope"]["document_ids"] == [files["lecture"][1]]
    assert narrowed["turn"]["trace"]["missing"] == ["experiment"]


def test_supplement_respects_same_name_clarification_and_explicit_location(system, tmp_path):
    _, _, _, client = system
    session = _create(client, course_ids=[_COURSE, "other"])
    question, files = _publish_supplement_case(client, session, tmp_path)
    lecture, _ = files["lecture"]
    _upload(client, session, lecture, course="other")
    ambiguous = _ask(client, session, f"{question} {lecture.name}")
    assert ambiguous["turn"]["status"] == "clarification"
    assert len(ambiguous["turn"]["options"]) == 2
    directed = _ask(client, session, f"{question} {lecture.name} Cell 1", kind="ask")
    assert directed["turn"]["status"] == "clarification"
    assert len(directed["turn"]["options"]) == 2


def test_supplement_rechecks_snapshot_at_message_commit(system, tmp_path, monkeypatch):
    settings, service, _, client = system
    session = _create(client)
    question, files = _publish_supplement_case(client, session, tmp_path)
    path, docid = files["lab"]
    original = service.answerer.answer

    def publish_while_finishing(*args, **kwargs):
        answer = original(*args, **kwargs)
        data = json.loads(path.read_text())
        data["cells"][0]["source"].append("\n新版内容")
        replacement = settings.course_root / path.name
        replacement.write_text(json.dumps(data, ensure_ascii=False))
        assert (
            service.library.import_paths(_COURSE, [replacement], document_id=docid)[0].version == 2
        )
        return answer

    monkeypatch.setattr(service.answerer, "answer", publish_while_finishing)
    result = _ask(client, session, question)
    assert result["turn"]["status"] == "failed"
    assert result["turn"]["error"] == "stale_reference"
    assert result["message"] is None


def test_supplement_does_not_expand_an_unresolved_explicit_file(system, tmp_path):
    _, _, _, client = system
    session = _create(client)
    question, files = _publish_supplement_case(client, session, tmp_path)
    result = _ask(client, session, f"{question} 只查 Missing.ipynb")
    assert result["message"]["blocks"][0]["code"] == "scope_not_found"
    lecture = files["lecture"][0].name
    lab = files["lab"][0].name
    result = _ask(
        client, session, f"虽有 {files['notes'][0].name}，{question} 仅在『{lecture}』与『{lab}』"
    )
    assert result["turn"]["trace"]["missing"] == []
    assert set(result["turn"]["trace"]["effective_scope"]["document_ids"]) == {
        files["lecture"][1],
        files["lab"][1],
    }
    result = _ask(client, session, f"{question} 只查 {lecture} 和 Missing.ipynb")
    assert result["message"]["blocks"][0]["code"] == "scope_not_found"
    scope = client.patch(
        session + "/scope",
        json={"course_ids": [_COURSE], "document_ids": [files["lecture"][1]], "scope_revision": 1},
    ).json()
    result = _ask(
        client, session, f"{question} 只查 {files['lab'][0].name}", scope["scope_revision"]
    )
    assert result["message"]["blocks"][0]["code"] == "scope_not_found"
    assert not result["message"]["sources"]


def test_explicit_no_experiment_does_not_trigger_web_supplement(system, tmp_path, monkeypatch):
    _, service, _, client = system
    session = _create(client)
    _publish_supplement_case(client, session, tmp_path)

    def unexpected(*args, **kwargs):
        pytest.fail("the user explicitly omitted the experiment")

    monkeypatch.setattr(service.answerer, "answer", unexpected)
    for query in (
        "中值滤波 median filter 的定义是什么？不要查实验。",
        "请给出中值滤波的定义。别解释实验。",
        "请给出中值滤波的定义并别解释实验。",
        "请给出中值滤波的定义，可别提实验。",
        "只要中值滤波的定义别查实验。",
        "What is the definition of median filter? Do not include any experiment.",
    ):
        result = _ask(client, session, query)
        assert result["turn"]["status"] == "completed"


def test_combined_web_question_allows_explicit_request_for_each_part(system, tmp_path):
    _, _, _, client = system
    session = _create(client)
    _publish_supplement_case(client, session, tmp_path)
    result = _ask(client, session, "请分别给出中值滤波的定义和留出集实验验证。")
    trace = result["turn"]["trace"]
    assert trace["route"] == "evidence_answer"


@pytest.fixture
def public_object_files(tmp_path):
    import runpy

    fixture = runpy.run_path("docs/G/G05-probe.py")
    return tmp_path / "fixtures", fixture["build"](tmp_path)


def test_first_in_file_image_question_selects_one_exact_output(system, public_object_files):
    _, _, _, client = system
    originals, public = public_object_files
    session = _create(client)
    imported = _upload(client, session, originals / "v1" / "HarborTrace.ipynb")
    result = _ask(client, session, "HarborTrace.ipynb Cell 2 第 2 张图的横轴刻度单位是什么？")
    trace = result["turn"]["trace"]
    selected = {
        "document_id": imported["document_id"],
        "version": 1,
        "source_id": "cell:1:output:1",
    }
    assert trace["selected_source"] == selected
    assert trace["object_candidates"] == [selected]
    assert trace["read_status"] == "not_attempted"
    assert trace["external_call_count"] == 0
    displays = [block for block in result["message"]["blocks"] if block["type"] == "source_image"]
    assert len(displays) == 1
    assert displays[0]["display_source"] == selected
    assert (
        _sha(client.get(displays[0]["media_url"]).content)
        == public["image_sha256"]["1"]["cell:1:output:1"]
    )


def test_unknown_side_and_unreadable_slot_never_substitute_neighbor(system, public_object_files):
    _, _, _, client = system
    originals, _ = public_object_files
    session = _create(client)
    source = originals / "v1" / "HarborTrace.ipynb"
    data = json.loads(source.read_text())
    data["cells"][1]["outputs"][1]["data"]["image/png"] = "bad image"
    source.write_text(json.dumps(data))
    _upload(client, session, source)
    ambiguous = _ask(client, session, "HarborTrace.ipynb Cell 2 的右侧琥珀子图刻度单位是什么？")
    assert not ambiguous["message"]["sources"]
    assert ambiguous["turn"]["trace"]["external_call_count"] == 0
    broken = _ask(client, session, "HarborTrace.ipynb 第 2 个 Cell 第 2 张图的横轴单位是什么？")
    assert broken["turn"]["trace"]["selected_source"]["source_id"] == "cell:1:output:1"
    assert broken["turn"]["trace"]["external_call_count"] == 0
    assert all(block["type"] != "source_image" for block in broken["message"]["blocks"])
    assert any(block.get("code") == "object_unreadable" for block in broken["message"]["blocks"])


def test_first_table_question_reads_exact_native_table(system, public_object_files):
    _, _, _, client = system
    originals, _ = public_object_files
    session = _create(client)
    _upload(client, session, originals / "PulseGate-limits.pptx")
    result = _ask(
        client, session, "PulseGate-limits.pptx 第 1 页 表格里的 Pulse gate 设定值是多少？"
    )
    assert result["turn"]["trace"]["selected_source"]["source_id"] == "page:1:shape:2"
    assert result["turn"]["trace"]["external_call_count"] == 0
    (item,) = result["message"]["sources"].values()
    (read,) = item["reads"]
    assert read["method"] == "native_table"
    assert read["actual_source"] == item["revision"]["source"]
    assert "17.5" in read["text"]


@pytest.mark.skipif(not os.getenv("SRAG_SOFFICE"), reason="SRAG_SOFFICE not configured")
def test_first_table_answer_shows_the_same_object_without_vision(system, public_object_files):
    _, service, app, client = system
    service.evidence.soffice_path = Path(os.environ["SRAG_SOFFICE"])
    originals, _ = public_object_files
    session = _create(client)
    _upload(client, session, originals / "PulseGate-limits.pptx")
    result = _ask(client, session, "核对 PulseGate-limits.pptx 第 1 页原表的设定值与单位。")
    message = result["message"]
    (evidence_id, item), = message["sources"].items()
    text, image = message["blocks"]
    assert text["citations"] == [evidence_id] == [image["evidence_id"]]
    assert item["reads"][0]["method"] == "native_table"
    source = item["revision"]["source"]
    assert item["reads"][0]["actual_source"] == source == image["display_source"]
    assert "17.5" in text["text"] and "mA" in text["text"]
    assert image["rendition_kind"] == "object_crop" and image["rect_visible_norm"]
    assert image["read_status"] == "not_attempted"
    assert result["turn"]["trace"]["external_call_count"] == 0
    assert client.get(image["media_url"]).headers["content-type"] == "image/png"
    endpoint = session + f"/messages/{message['message_id']}/evidence/{evidence_id}"
    assert client.get(endpoint).json()["source"] == image["display_source"]
    with TestClient(app, base_url=_HOST) as other:
        _create(other)
        assert other.get(endpoint).status_code == 404
        assert other.get(image["media_url"]).status_code == 404


def test_stale_first_image_keeps_call_count_without_publishing_source(
    system, public_object_files, monkeypatch
):
    from app.rag.source_evidence import EvidenceResult

    _, service, _, client = system
    originals, _ = public_object_files
    session = _create(client)
    _upload(client, session, originals / "v1" / "HarborTrace.ipynb")

    def changed_during_read(request, *, expected_input_sha256):
        assert expected_input_sha256
        return EvidenceResult(
            status="stale_reference", requested_source=request.source, external_call_count=1
        )

    monkeypatch.setattr(service.vision, "process_current", changed_during_read)
    result = _ask(client, session, "HarborTrace.ipynb Cell 2 第 2 张图的横轴刻度单位是什么？")
    assert result["turn"]["error"] == "stale_reference"
    assert result["message"] is None
    assert set(result["turn"]["trace"]) == {"external_call_count", "vision"}
    assert result["turn"]["trace"]["external_call_count"] == 1


def test_first_image_question_does_not_substitute_another_named_file(system, public_object_files):
    _, _, _, client = system
    originals, _ = public_object_files
    session = _create(client)
    _upload(client, session, originals / "v1" / "HarborTrace.ipynb")
    for question in (
        "MissingNotebook.ipynb 的 Cell 2 第 2 张图是什么单位？请不要用 HarborTrace.ipynb 代替。",
        "not-HarborTrace.ipynb Cell 2 第 2 张图的单位是什么？",
        "请不要用 HarborTrace.ipynb，Cell 2 第 2 张图的单位是什么？",
    ):
        result = _ask(client, session, question)
        assert result["message"]["blocks"][0]["code"] == "scope_not_found"
        assert not result["message"]["sources"]


@pytest.mark.parametrize("with_table", [True, False])
def test_normal_three_material_question_uses_exact_roles(system, public_object_files, with_table):
    _, _, _, client = system
    originals, public = public_object_files
    session = _create(client)
    for name in ("PulseGate-lecture.ipynb", "PulseGate-run.ipynb"):
        _upload(client, session, originals / name)
    if with_table:
        _upload(client, session, originals / "PulseGate-limits.pptx")
    result = _ask(client, session, public["questions"]["three"], client_id="three-materials")
    assert result["turn"]["status"] == "completed"
    trace = result["turn"]["trace"]
    assert trace["route"] == "evidence_composition"
    assert trace["supplement_count"] <= 1
    assert trace["external_call_count"] == 0
    message = result["message"]
    sources = message["sources"]
    assert len(sources) == (3 if with_table else 2)
    assert {item["revision"]["source"]["source_id"] for item in sources.values()} == (
        {"cell:0", "cell:0:output:0", "page:1:shape:2"}
        if with_table
        else {"cell:0", "cell:0:output:0"}
    )
    for ref in sources:
        endpoint = session + f"/messages/{message['message_id']}/evidence/{ref}"
        opened = client.get(endpoint)
        assert opened.status_code == 200
        assert opened.json()["source"] == sources[ref]["revision"]["source"]
    if with_table:
        assert trace["comparison"] == "unverified_alignment"
    else:
        assert all("17.5" not in block["text"] for block in message["blocks"])


def test_composition_does_not_replace_explicit_page_by_another_page(system, public_object_files):
    _, _, _, client = system
    originals, public = public_object_files
    session = _create(client)
    for name in ("PulseGate-lecture.ipynb", "PulseGate-run.ipynb", "PulseGate-limits.pptx"):
        _upload(client, session, originals / name)
    for page in ("page 99", "第 99 页"):
        result = _ask(
            client,
            session,
            f"{public['questions']['three']} Only use {page} of PulseGate-limits.pptx.",
            client_id=f"restricted-{page}",
        )
        assert result["turn"]["status"] == "clarification"
        assert not result["message"]["sources"]
