"""Historical citation image verification through the production workspace router."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_workspace import _ask, _create, _notebook_case, _upload
from test_workspace import system as workspace_fixture

from app.rag import source_evidence
from app.rag.source_vision import VisionReply

preview_system = workspace_fixture


def cited_path(session, answer, *, page):
    message = answer["message"]
    used = {
        citation
        for block in message["blocks"]
        if block["type"] == "text"
        for citation in block["citations"]
    }
    for identifier, evidence in message["sources"].items():
        if identifier in used and evidence["locator"].get("page") == page:
            return f"{session}/messages/{message['message_id']}/evidence/{identifier}"
    raise AssertionError(f"no cited source for page {page}")


def test_pdf_preview_pinned_to_citation_and_owner(preview_system, tmp_path):
    _, service, app, client = preview_system
    original = runpy.run_path("docs/G/G16-browser-fixtures.py")
    original["build"](tmp_path)
    session = _create(client)
    file = original["FILENAME"]
    saved = _upload(client, session, tmp_path / "fixtures/v1" / file)
    path = cited_path(session, _ask(client, session, "August 23", kind="search"), page=5)
    before = client.get(path)
    assert before.json()["is_current"]
    assert set(before.json()) == {"source", "file_name", "locator", "text", "is_current"}
    image = client.get(path + "/preview")
    assert image.status_code == 200 and image.headers["Content-Type"] == "image/png"
    assert image.headers["Cache-Control"] == "private, no-store"
    assert image.headers["X-Content-Type-Options"] == "nosniff"
    replacement = _upload(
        client, session, tmp_path / "fixtures/v2" / file, document_id=saved["document_id"]
    )
    assert replacement["version"] == 2
    assert client.get(path).json()["source"]["version"] == 1
    assert not client.get(path).json()["is_current"]
    assert client.get(path + "/preview").content == image.content
    latest = cited_path(session, _ask(client, session, "August 21", kind="search"), page=5)
    assert client.get(latest + "/preview").content != image.content

    other_session = _create(client)
    cross_session = path.replace(session + "/", other_session + "/", 1)
    assert client.get(cross_session + "/preview").status_code == 404
    assert client.get(path.rsplit("/", 1)[0] + "/ev-999/preview").status_code == 404
    assert (
        client.get(path + "/preview", headers={"Origin": "https://outside.invalid"}).status_code
        == 404
    )
    with TestClient(app, base_url="http://127.0.0.1:8001") as other:
        _create(other)
        assert other.get(path + "/preview").status_code == 404
    assert service.library.store.delete_local(saved["document_id"])
    assert client.get(path + "/preview").status_code == 404


def test_visual_followup_uses_exact_read_bytes(preview_system, monkeypatch):
    _, service, _, client = preview_system
    session, imported, _, revision, initial, images = _notebook_case(client)
    cited_image = images[1]
    original = client.get(cited_image["media_url"])
    assert original.status_code == 200
    read_images = []

    def simulated_provider(*, image, **_kwargs):
        read_images.append(image)
        return VisionReply("succeeded", "可核对的模拟视觉回复。", (), None, {})

    monkeypatch.setattr(service.vision, "enabled", True)
    monkeypatch.setattr(service.vision, "_provider_configured", True)
    monkeypatch.setattr(service.vision.provider, "read", simulated_provider)
    anchor = {
        "type": "display",
        "message_id": initial["message"]["message_id"],
        "display_id": cited_image["display_id"],
    }
    follow = _ask(client, session, "坐标的单位是什么？", revision, anchor=anchor)
    message = follow["message"]
    assert not message["visible_display_ids"]
    assert message["blocks"][0]["citations"] == ["ev-1"]
    assert read_images == [original.content]
    path = f"{session}/messages/{message['message_id']}/evidence/ev-1"
    assert client.get(path).status_code == 404
    assert client.get(path + "/preview").content == original.content
    assert client.get(path.replace("/ev-1", "/ev-2") + "/preview").status_code == 404
    assert client.get(path.replace(session, _create(client)) + "/preview").status_code == 404
    assert service.library.store.delete_local(imported["document_id"])
    assert client.get(path + "/preview").status_code == 404
    assert read_images == [original.content]


@pytest.mark.parametrize("renderer", ["libreoffice", "unavailable"])
def test_pptx_citation_preview_has_fallback(preview_system, tmp_path, monkeypatch, renderer):
    _, service, _, client = preview_system
    if renderer == "libreoffice":
        executable = os.environ.get("SRAG_SOFFICE")
        if not executable or not Path(executable).is_file():
            pytest.skip("Set SRAG_SOFFICE to exercise genuine PPTX page rendering")
        service.evidence.soffice_path = Path(executable)
    else:
        monkeypatch.setattr(source_evidence, "_soffice", lambda _path: (None, ""))

    fixture = runpy.run_path("docs/G/G10-probe.py")
    fixture["build"](tmp_path)
    session = _create(client)
    _upload(client, session, tmp_path / "fixtures/Drum-Readings.pptx")
    path = cited_path(session, _ask(client, session, "Cedar", kind="search"), page=1)
    text = client.get(path)
    assert text.status_code == 200 and "Cedar" in text.json()["text"]
    preview = client.get(path + "/preview")
    if renderer == "libreoffice":
        assert preview.status_code == 200 and preview.content.startswith(b"\x89PNG")
    else:
        assert preview.status_code == 409
        assert preview.json()["detail"]["code"] == "rendition_unavailable"
        assert client.get(path).json() == text.json()
