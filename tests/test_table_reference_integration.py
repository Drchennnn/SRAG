"""Follow-up tables must be visible to the learner and in the current scope."""

from pathlib import Path

from fastapi.testclient import TestClient
from test_workspace_table_routing_g26 import (
    _COURSE,
    _HOST,
    _ask,
    _create_session,
    _make_single_table_pptx,
    _make_two_table_pptx,
    _upload_doc,
)
from test_workspace_table_routing_g26 import g26_client as workspace_fixture

from app.api.workspace import get_workspace_service
from app.rag.workspace_generation import GeneratedStatement, Generation

g26_client = workspace_fixture


def _generate_with_citations(client, monkeypatch, *, cite_all):
    service = client.app.dependency_overrides[get_workspace_service]()
    monkeypatch.setattr(service.settings, "workspace_answer_enabled", True)
    monkeypatch.setattr(service.settings, "deepseek_api_key", "offline-test")

    def generate(settings, question, evidence, **kwargs):
        tables = [
            key
            for key, item in evidence.items()
            if any(read.method == "native_table" for read in item.reads)
        ]
        citations = tables if cite_all else tables[:1]
        return Generation(
            "succeeded",
            (GeneratedStatement("见表格。", tuple(citations)),),
            "",
            {"external_call_count": 0},
            dict(evidence),
        )

    monkeypatch.setattr("app.rag.workspace.generate_response", generate)


def test_followup_uses_cited_table_among_uncited_model_inputs(
    g26_client: TestClient, tmp_path: Path, monkeypatch
):
    url = _create_session(g26_client)
    _upload_doc(g26_client, url, _make_two_table_pptx(tmp_path / "results.pptx"))
    _generate_with_citations(g26_client, monkeypatch, cite_all=False)
    response = g26_client.post(
        url + "/turns",
        json={
            "client_turn_id": "table-summary",
            "kind": "summary",
            "text": "Summarize the results.",
            "scope_revision": g26_client.get(url).json()["scope_revision"],
        },
        headers={"origin": _HOST},
    )
    assert response.status_code == 200
    first = response.json()
    msg = first["message"]
    tables = {
        key: item
        for key, item in msg["sources"].items()
        if any(read["method"] == "native_table" for read in item["reads"])
    }
    assert len(tables) == 2
    cited = [key for b in msg["blocks"] for key in b.get("citations", [])]
    assert len(cited) == 1
    assert not msg["visible_display_ids"]

    follow = _ask(g26_client, url, "刚才那张表里，第二列是什么单位？")
    assert follow["turn"]["trace"]["selected_source"] == tables[cited[0]]["revision"]["source"]


def test_multiple_visible_tables_outside_current_scope_do_not_disclose_labels(
    g26_client: TestClient, tmp_path: Path, monkeypatch
):
    url = _create_session(g26_client)
    _upload_doc(g26_client, url, _make_two_table_pptx(tmp_path / "prior-tables.pptx"))
    _generate_with_citations(g26_client, monkeypatch, cite_all=True)
    first = _ask(g26_client, url, "Compare both benchmark results.")
    assert len(first["message"]["blocks"][0]["citations"]) == 2
    ambiguous = _ask(g26_client, url, "刚才那张表的单位是什么？")
    assert ambiguous["turn"]["status"] == "clarification"
    prompt = ambiguous["message"]["blocks"][0]["text"]
    assert "第 1 页" in prompt and "第 2 页" in prompt
    other = _upload_doc(g26_client, url, _make_single_table_pptx(tmp_path / "other.pptx"))
    response = g26_client.patch(
        url + "/scope",
        json={
            "course_ids": [_COURSE],
            "document_ids": [other],
            "scope_revision": g26_client.get(url).json()["scope_revision"],
        },
        headers={"origin": _HOST},
    )
    assert response.status_code == 200
    follow = _ask(g26_client, url, "刚才那张表里，第二列是什么单位？")
    blocks = follow["message"]["blocks"]
    assert all("prior-tables.pptx" not in block["text"] for block in blocks)
    assert any(block.get("code") == "outside_current_scope" for block in blocks)
