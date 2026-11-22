"""G26 tests for table question intent classification, routing, follow-up, and ambiguity."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pptx import Presentation
from pptx.util import Inches

from app.api.workspace import get_workspace_service, router
from app.config import Settings
from app.rag.workspace import make_workspace
from app.rag.workspace_objects import ObjectIntent, object_intent

_BASE = "/api/workspace"
_HOST = "http://127.0.0.1:8001"
_COURSE = "g26-test-course"


def _make_two_table_pptx(path: Path) -> Path:
    prs = Presentation()
    blank = prs.slide_layouts[6]

    slide1 = prs.slides.add_slide(blank)
    t1 = slide1.shapes.add_table(3, 3, Inches(0.5), Inches(0.8), Inches(8.0), Inches(2.5)).table
    rows1 = [
        ["Method", "Memory (GB)", "Latency (ms)"],
        ["AlphaQuant", "12.4", "18.2"],
        ["BaseAdam", "26.0", "21.5"],
    ]
    for r, row in enumerate(rows1):
        for c, val in enumerate(row):
            t1.cell(r, c).text = val

    slide2 = prs.slides.add_slide(blank)
    t2 = slide2.shapes.add_table(3, 3, Inches(0.5), Inches(0.8), Inches(8.0), Inches(2.5)).table
    rows2 = [
        ["Dataset", "Accuracy (%)", "Throughput (samples/s)"],
        ["BetaBench", "84.5", "1120"],
        ["GammaEval", "79.1", "980"],
    ]
    for r, row in enumerate(rows2):
        for c, val in enumerate(row):
            t2.cell(r, c).text = val

    prs.save(path)
    return path


def _make_single_table_pptx(path: Path) -> Path:
    prs = Presentation()
    slide1 = prs.slides.add_slide(prs.slide_layouts[6])
    t1 = slide1.shapes.add_table(2, 2, Inches(0.5), Inches(0.8), Inches(6.0), Inches(2.0)).table
    for r, row in enumerate([["Metric", "Unit"], ["Current", "mA"]]):
        for c, val in enumerate(row):
            t1.cell(r, c).text = val
    prs.save(path)
    return path


@pytest.fixture
def g26_client(tmp_path: Path):
    settings = Settings(
        sqlite_path=tmp_path / "g26.sqlite3",
        data_dir=tmp_path / "data",
        course_root=tmp_path / "courses",
        workspace_course_ids=_COURSE,
        workspace_max_upload_bytes=10 * 1024 * 1024,
        source_vision_enabled=False,
        workspace_answer_enabled=False,
        _env_file=None,
    )
    service = make_workspace(settings)
    app = FastAPI()
    app.include_router(router, prefix=_BASE)
    app.dependency_overrides[get_workspace_service] = lambda: service
    with TestClient(app, base_url=_HOST) as client:
        yield client


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


def _ask(client: TestClient, session_url: str, text: str) -> dict:
    current = client.get(session_url).json()
    resp = client.post(
        f"{session_url}/turns",
        json={
            "client_turn_id": f"turn-{len(text)}-{abs(hash(text)) % 100000}",
            "kind": "ask",
            "text": text,
            "scope_revision": current["scope_revision"],
        },
        headers={"origin": _HOST},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_g26_object_intent_distinguishes_facts_from_explicit_table_targets():
    # Category 1: natural factual/comparison questions mentioning tables without targeting an object
    assert object_intent("实测表中，哪组的内存开销更低，差多少？") is None
    assert (
        object_intent("对比表格里的海岸观测数据产品，哪些同时使用了 netCDF 和 shapefile？")
        is None
    )
    assert object_intent("在文末附录补充的显存与吞吐量实测表中，八位投影法占多少显存？") is None

    # Category 2: explicit object-targeting commands
    assert object_intent("打开第 4 页的表 1") == ObjectIntent(
        kind="table", ordinal=1, order="tables"
    )
    assert object_intent("查看第 5 页第 1 张表格的原表") == ObjectIntent(
        kind="table", ordinal=1, order="tables"
    )

    # Category 3 & 4: demonstrative/anaphoric table targets & preserved existing boundaries
    assert object_intent("打开那张表") == ObjectIntent(kind="table")
    assert object_intent("打开包含海冰变化（Sea ice changes）的那张表") == ObjectIntent(
        kind="table"
    )
    assert object_intent("刚才那张表里，第二列是什么单位？") == ObjectIntent(kind="table")
    assert object_intent("打开原表") is None
    assert object_intent("请展示第 2 张图？") is None


def test_g26_factual_table_question_bypasses_object_ambiguity(
    g26_client: TestClient, tmp_path: Path
):
    pptx = _make_two_table_pptx(tmp_path / "metrics.pptx")
    session_url = _create_session(g26_client)
    _upload_doc(g26_client, session_url, pptx)

    res = _ask(g26_client, session_url, "实测表中，AlphaQuant 和 BaseAdam 哪组的内存开销更低？")
    trace = res["turn"]["trace"]
    assert trace.get("route") != "target_object"
    assert not any(b.get("code") == "object_ambiguous" for b in res["message"]["blocks"])
    assert any(
        "12.4" in (r.get("text") or "")
        for ev in res["message"]["sources"].values()
        for r in ev.get("reads", [])
    )


def test_g26_explicit_table_open_and_session_followup(g26_client: TestClient, tmp_path: Path):
    pptx = _make_two_table_pptx(tmp_path / "metrics.pptx")
    session_url = _create_session(g26_client)
    _upload_doc(g26_client, session_url, pptx)

    turn1 = _ask(g26_client, session_url, "打开第 1 页的表 1")
    assert turn1["turn"]["trace"]["route"] == "target_object"
    assert turn1["turn"]["trace"]["selected_source"]["source_id"] == "page:1:shape:2"

    turn2 = _ask(g26_client, session_url, "刚才那张表里，第二列是什么单位？")
    assert turn2["turn"]["trace"]["route"] == "followup_table"
    assert turn2["turn"]["trace"]["selected_source"]["source_id"] == "page:1:shape:2"
    (ev,) = turn2["message"]["sources"].values()
    assert ev["revision"]["source"]["source_id"] == "page:1:shape:2"
    assert ev["revision"]["source"]["version"] == 1
    assert "GB" in ev["reads"][0]["text"]


def test_g26_table_ambiguity_vs_distinguishable_context(g26_client: TestClient, tmp_path: Path):
    multi = _make_two_table_pptx(tmp_path / "multi.pptx")
    s1 = _create_session(g26_client)
    _upload_doc(g26_client, s1, multi)

    # Ambiguous when two tables exist and no distinguishing clue is given
    ambig = _ask(g26_client, s1, "打开那张表")
    assert ambig["turn"]["trace"]["route"] == "target_object"
    blocks = ambig["message"]["blocks"]
    assert any(b.get("code") == "object_ambiguous" for b in blocks)
    assert "第 1 页" in blocks[0]["text"] and "第 2 页" in blocks[0]["text"]

    # Distinguishable by content -> opens unique table without asking
    dist = _ask(g26_client, s1, "打开包含 BetaBench 的那张表")
    assert dist["turn"]["trace"]["route"] == "target_object"
    assert dist["turn"]["trace"]["selected_source"]["source_id"] == "page:2:shape:2"

    # Single table in scope -> opens directly without asking
    single = _make_single_table_pptx(tmp_path / "single.pptx")
    s2 = _create_session(g26_client)
    single_id = _upload_doc(g26_client, s2, single)
    current_s2 = g26_client.get(s2).json()
    scope_resp = g26_client.patch(
        f"{s2}/scope",
        json={
            "course_ids": [_COURSE],
            "document_ids": [single_id],
            "scope_revision": current_s2["scope_revision"],
        },
        headers={"origin": _HOST},
    )
    assert scope_resp.status_code == 200, scope_resp.text
    direct = _ask(g26_client, s2, "打开那张表")
    assert direct["turn"]["trace"]["route"] == "target_object"
    assert direct["turn"]["trace"]["selected_source"]["source_id"] == "page:1:shape:2"
