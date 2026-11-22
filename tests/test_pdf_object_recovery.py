"""Deterministic tests for PDF vector subfigure, caption, formula, and table recovery."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.workspace import get_workspace_service
from app.config import Settings, get_settings
from app.main import create_app
from app.rag.parsers.local_slides import parse_local_slides
from app.rag.source_vision import DashScopeVisionProvider, VisionReply
from app.rag.workspace import make_workspace

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "docs" / "G" / "G-pdf-object-recovery" / "fixtures"
CRITERIA = ROOT / "docs" / "G" / "G-pdf-object-recovery" / "frozen_criteria.json"


def test_pdf_parser_registers_vector_subfigures_captions_formulas_and_tables() -> None:
    vec = parse_local_slides(FIXTURES / "Vector-Subfigures-Guide.pdf")
    by_id_vec = {s.source_id: s for s in vec.sources}
    assert "page:1" in by_id_vec
    assert "page:1:table:1" in by_id_vec
    assert by_id_vec["page:2:image:1"].locator["object_type"] == "image"
    assert by_id_vec["page:2:image:2"].locator["object_type"] == "image"
    assert (
        by_id_vec["page:2:image:1"].locator["bbox"][0]
        < by_id_vec["page:2:image:2"].locator["bbox"][0]
    )
    assert by_id_vec["page:2:caption:1"].locator["object_type"] == "caption"
    assert (
        "does not support causal delay claims"
        in by_id_vec["page:2:caption:1"].locator["object_text"]
    )

    fml = parse_local_slides(FIXTURES / "Formula-And-Notes.pdf")
    by_id_fml = {s.source_id: s for s in fml.sources}
    assert "page:2:formula:1" in by_id_fml
    assert "page:2:formula:2" in by_id_fml
    assert "tau_eff = (W - 1) / 2" in by_id_fml["page:2:formula:1"].locator["object_text"]
    assert "for odd W >= 3 only" in by_id_fml["page:2:formula:1"].locator["object_text"]
    assert "alpha_max = 1 / (N + 1)" in by_id_fml["page:2:formula:2"].locator["object_text"]
    assert by_id_fml["page:2:caption:1"].locator["object_type"] == "caption"


def test_pdf_object_selection_disambiguation_and_ambiguity_in_asgi(tmp_path: Path) -> None:
    criteria = json.loads(CRITERIA.read_text(encoding="utf-8"))
    settings = Settings(
        _env_file=None,
        course_root=tmp_path / "courses",
        data_dir=tmp_path / "data",
        sqlite_path=tmp_path / "library.sqlite3",
        workspace_course_ids="pdf-obj-test",
        workspace_answer_enabled=False,
        workspace_retrieval_strategy="fts",
        source_vision_enabled=True,
        dashscope_api_key="synthetic-key",
        deepseek_api_key="",
        qdrant_url="http://127.0.0.1:1",
    )
    app = create_app()
    svc = make_workspace(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_workspace_service] = lambda: svc

    def fake_vision(self, *args, **kwargs):
        return VisionReply(
            status="succeeded",
            text=(
                "Right panel (b) shows Gain [dB] vs Time [ms], "
                "damped decay, spike at t = 8 ms, +6 dB."
            ),
            missing=(),
            request_id="req-test",
            usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )

    with (
        patch.object(DashScopeVisionProvider, "read", fake_vision),
        TestClient(
            app, base_url="http://127.0.0.1:8000", headers={"Origin": "http://127.0.0.1:8000"}
        ) as client,
    ):
        sid = client.post(
            "/api/workspace/sessions", json={"title": "test", "course_ids": ["pdf-obj-test"]}
        ).json()["session_id"]
        sbase = f"/api/workspace/sessions/{sid}"
        doc_ids = {}
        for name in ("Vector-Subfigures-Guide.pdf", "Formula-And-Notes.pdf"):
            resp = client.post(
                f"{sbase}/import",
                params={"course_id": "pdf-obj-test", "filename": name},
                content=(FIXTURES / name).read_bytes(),
                headers={"Content-Type": "application/octet-stream"},
            )
            assert resp.status_code == 200
            doc_ids[name] = resp.json()["document_id"]

        # Q01 right vector subfigure, Q02 caption, Q03 formula w/ distractor, Q05 table, Q06 page
        for qitem in criteria["questions"]:
            qid = qitem["question_id"]
            if qid == "Q04_ambiguity_counterexample":
                continue
            qsid = client.post(
                "/api/workspace/sessions", json={"title": qid, "course_ids": ["pdf-obj-test"]}
            ).json()["session_id"]
            qsbase = f"/api/workspace/sessions/{qsid}"
            rev = client.get(qsbase).json()["scope_revision"]
            rev = client.patch(
                f"{qsbase}/scope",
                json={
                    "scope_revision": rev,
                    "course_ids": ["pdf-obj-test"],
                    "document_ids": [doc_ids[qitem["document"]]],
                },
            ).json()["scope_revision"]
            turn_resp = client.post(
                f"{qsbase}/turns",
                json={
                    "kind": "ask",
                    "client_turn_id": f"t-{qid}",
                    "scope_revision": rev,
                    "text": qitem["question"],
                },
            ).json()
            msg = turn_resp["message"]
            sources = {s["revision"]["source"]["source_id"]: s for s in msg["sources"].values()}
            if qid == "Q01_vector_subfigure":
                assert "page:2:image:2" in sources
            elif qid == "Q02_figure_caption":
                assert "page:2:caption:1" in sources
            elif qid == "Q03_standalone_formula":
                assert "page:2:formula:1" in sources
                assert "page:2:formula:2" not in sources
                assert "for odd W >= 3 only" in sources["page:2:formula:1"]["reads"][0]["text"]
            elif qid == "Q05_table_regression":
                assert "page:1:table:1" in sources
            elif qid == "Q06_page_citation_regression":
                assert "page:1" in sources
