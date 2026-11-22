"""G25 end-to-end HTTP/workspace tests for same-page object positioning & ambiguity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.workspace import get_workspace_service, router
from app.config import Settings
from app.rag.source_vision import VisionReply
from app.rag.workspace import make_workspace

_ROOT = Path(__file__).resolve().parents[1]
_BASE = "/api/workspace"
_HOST = "http://127.0.0.1:8001"
_COURSE = "g25-e2e-course"
_MIXTRAL_PDF = _ROOT / "docs/G/G20-fixtures/Mixtral-of-Experts-2401.04088v1.pdf"
_LAYOUTS_PDF = _ROOT / "docs/G/G25-fixtures/SensorLab-Layouts-CCBY4.pdf"
_OVERLAP_PPTX = _ROOT / "docs/G/G25-fixtures/SensorLab-Overlap-CCBY4.pptx"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def g25_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    captured_vision_calls: list[dict] = []

    def fake_vision_read(
        self, *, image: bytes, media_type: str, purpose: str, question: str | None
    ):
        digest = _sha256(image)
        captured_vision_calls.append(
            {
                "bytes": len(image),
                "sha256": digest,
                "media_type": media_type,
                "purpose": purpose,
                "question": question,
            }
        )
        return VisionReply(
            status="succeeded",
            text=f"Mock read for sha={digest[:12]}",
            missing=(),
            request_id=f"req-{len(captured_vision_calls)}",
            usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )

    from app.rag.source_vision import DashScopeVisionProvider

    monkeypatch.setattr(DashScopeVisionProvider, "read", fake_vision_read)

    settings = Settings(
        sqlite_path=tmp_path / "g25.sqlite3",
        data_dir=tmp_path / "data",
        course_root=tmp_path / "courses",
        workspace_course_ids=_COURSE,
        workspace_max_upload_bytes=10 * 1024 * 1024,
        source_vision_enabled=True,
        dashscope_api_key="synthetic-g25-key",
        workspace_answer_enabled=False,
        _env_file=None,
    )
    service = make_workspace(settings)
    app = FastAPI()
    app.include_router(router, prefix=_BASE)
    app.dependency_overrides[get_workspace_service] = lambda: service
    with TestClient(app, base_url=_HOST) as client:
        yield client, service, captured_vision_calls


def _create_session(client: TestClient) -> str:
    resp = client.post(
        f"{_BASE}/sessions",
        json={"course_ids": [_COURSE]},
        headers={"origin": _HOST},
    )
    assert resp.status_code == 201, resp.text
    return f"{_BASE}/sessions/{resp.json()['session_id']}"


def _upload_doc(client: TestClient, session_url: str, path: Path) -> str:
    resp = client.post(
        f"{session_url}/import",
        params={"course_id": _COURSE, "filename": path.name},
        headers={"content-type": "application/octet-stream", "origin": _HOST},
        content=path.read_bytes(),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "published"
    catalog = client.get(f"{session_url}/catalog").json()
    for course in catalog["courses"]:
        for doc in course["documents"]:
            if doc["file_name"] == path.name:
                return doc["document_id"]
    raise AssertionError(f"Uploaded doc {path.name} missing from catalog")


def _select_doc(client: TestClient, session_url: str, doc_id: str) -> int:
    current = client.get(session_url).json()
    resp = client.patch(
        f"{session_url}/scope",
        json={
            "course_ids": [_COURSE],
            "document_ids": [doc_id],
            "scope_revision": current["scope_revision"],
        },
        headers={"origin": _HOST},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["scope_revision"]


def _ask(client: TestClient, session_url: str, text: str, scope_rev: int, turn_id: str) -> dict:
    resp = client.post(
        f"{session_url}/turns",
        json={"client_turn_id": turn_id, "text": text, "scope_revision": scope_rev},
        headers={"origin": _HOST},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_g25_reproduces_q03_fix_and_preserves_q10_ambiguity(g25_client):
    if not _MIXTRAL_PDF.exists():
        pytest.skip("G20 Mixtral fixture missing")
    client, _, vision_calls = g25_client
    session_url = _create_session(client)
    doc_id = _upload_doc(client, session_url, _MIXTRAL_PDF)
    rev = _select_doc(client, session_url, doc_id)

    # Q03: right subplot whose left y-axis crosses x=0.5 must resolve to page:5:image:1
    q03 = _ask(
        client,
        session_url,
        "第 5 页右侧图的横轴和纵轴分别表示什么？图中纵轴的刻度范围大约从多少到多少？",
        rev,
        "q03",
    )
    trace = q03["turn"]["trace"]
    assert trace["route"] == "target_object"
    assert trace["selected_source"]["source_id"] == "page:5:image:1"
    assert trace["external_call_count"] == 1
    assert len(vision_calls) == 1

    msg = q03["message"]
    (img_block,) = [b for b in msg["blocks"] if b["type"] == "source_image"]
    (txt_block,) = [b for b in msg["blocks"] if b["type"] == "text"]
    ev_id = img_block["evidence_id"]
    assert txt_block["citations"] == [ev_id]
    ev = msg["sources"][ev_id]
    assert ev["revision"]["source"] == trace["selected_source"]
    assert ev["reads"][0]["actual_input_sha256"] == vision_calls[0]["sha256"]
    media_bytes = client.get(img_block["media_url"]).content
    assert _sha256(media_bytes) == vision_calls[0]["sha256"]

    # Q10: non-directional page 5 question must remain object_ambiguous with 0 calls
    q10 = _ask(client, session_url, "第 5 页的插图里具体展示了什么结果？", rev, "q10")
    assert q10["turn"]["trace"]["external_call_count"] == 0
    assert len(q10["turn"]["trace"]["object_candidates"]) == 2
    assert q10["message"]["blocks"][0]["code"] == "object_ambiguous"


def test_g25_all_six_public_layout_scenarios_end_to_end(g25_client):
    client, _, vision_calls = g25_client
    session_url = _create_session(client)

    pdf_id = _upload_doc(client, session_url, _LAYOUTS_PDF)
    rev = _select_doc(client, session_url, pdf_id)

    cases = [
        # (turn_id, question, expected_source_id_or_none, expected_notice_code_or_none)
        ("p1_right", "第 1 页右图的横轴和纵轴分别是什么？", "page:1:image:2", None),
        ("p1_left", "第 1 页左图的横轴和纵轴分别是什么？", "page:1:image:1", None),
        ("p2_right_cross", "第 2 页右图的横轴和纵轴分别是什么？", "page:2:image:2", None),
        ("p2_left", "第 2 页左图的横轴是什么？", "page:2:image:1", None),
        ("p3_mirror_left", "第 3 页左图的横轴和纵轴分别是什么？", "page:3:image:1", None),
        ("p3_mirror_right", "第 3 页右图的横轴是什么？", "page:3:image:2", None),
        ("p4_top", "第 4 页上图的横轴和纵轴分别是什么？", "page:4:image:1", None),
        ("p4_bottom", "第 4 页下图的横轴和纵轴分别是什么？", "page:4:image:2", None),
        ("p4_stacked_right", "第 4 页右侧图的纵轴是什么？", None, "object_not_found"),
        ("p5_centered_right", "第 5 页右图的纵轴是什么？", None, "object_not_found"),
        ("p5_centered_plain", "第 5 页的插图里横轴和纵轴是什么？", "page:5:image:1", None),
    ]

    for turn_id, question, expected_source_id, expected_notice in cases:
        calls_before = len(vision_calls)
        res = _ask(client, session_url, question, rev, turn_id)
        trace = res["turn"]["trace"]
        msg = res["message"]
        if expected_source_id is not None:
            assert trace["selected_source"]["source_id"] == expected_source_id, (turn_id, trace)
            assert trace["external_call_count"] == 1
            assert len(vision_calls) == calls_before + 1
            sent_sha = vision_calls[-1]["sha256"]
            (img_block,) = [b for b in msg["blocks"] if b["type"] == "source_image"]
            (txt_block,) = [b for b in msg["blocks"] if b["type"] == "text"]
            ev = msg["sources"][img_block["evidence_id"]]
            assert txt_block["citations"] == [img_block["evidence_id"]]
            assert ev["revision"]["source"]["source_id"] == expected_source_id
            assert ev["reads"][0]["actual_input_sha256"] == sent_sha
            assert _sha256(client.get(img_block["media_url"]).content) == sent_sha
        else:
            assert len(vision_calls) == calls_before
            assert trace["external_call_count"] == 0
            assert msg["blocks"][0]["code"] == expected_notice

    # Overlap PPTX scenario: two clearly overlapping pictures on slide 1 must ask user
    pptx_id = _upload_doc(client, session_url, _OVERLAP_PPTX)
    rev_pptx = _select_doc(client, session_url, pptx_id)
    calls_before = len(vision_calls)
    overlap_res = _ask(
        client, session_url, "第 1 页右侧图的纵轴是什么？", rev_pptx, "pptx_overlap_right"
    )
    assert len(vision_calls) == calls_before
    assert overlap_res["turn"]["trace"]["external_call_count"] == 0
    assert len(overlap_res["turn"]["trace"]["object_candidates"]) == 2
    assert overlap_res["message"]["blocks"][0]["code"] == "object_ambiguous"

    # Also verify spec JSON exists and matches generated PDF sha256
    spec = json.loads((_ROOT / "docs/G/G25-判据与标注.json").read_text(encoding="utf-8"))
    assert spec["pdf"]["sha256"] == _sha256(_LAYOUTS_PDF.read_bytes())
