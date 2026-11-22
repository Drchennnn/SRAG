"""Deterministic tests for condition-mismatch enforcement under text generation ON/OFF."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import nbformat
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.workspace import get_workspace_service, router
from app.clients.deepseek import DeepSeekClient
from app.config import Settings
from app.rag.evidence_supplement import classify_experiment_evidence
from app.rag.workspace import make_workspace
from app.rag.workspace_generation import generate_response
from app.schemas.local import SourceKey
from app.schemas.turn_evidence import SourceRevision, TurnEvidence, TurnRead

_BASE = "/api/workspace"
_HOST = "http://127.0.0.1:8001"
_COURSE = "m2-cond-gen-test"


def _ev(
    ev_id: str,
    doc_id: str,
    file_name: str,
    source_id: str,
    text: str,
    *,
    page: int | None = None,
    cell_index: int | None = None,
) -> TurnEvidence:
    locator: dict = {}
    if page is not None:
        locator["page"] = page
    if cell_index is not None:
        locator["cell_index"] = cell_index
    key = SourceKey(document_id=doc_id, version=1, source_id=source_id)
    return TurnEvidence(
        evidence_id=ev_id,
        revision=SourceRevision(
            source=key,
            course_id=_COURSE,
            publication_epoch=1,
            content_sha256="a" * 64,
        ),
        file_name=file_name,
        locator=locator,
        read_status="succeeded",
        reads=[TurnRead(actual_source=key, method="native_text", status="succeeded", text=text)],
    )


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=tmp_path / "db.sqlite3",
        data_dir=tmp_path / "data",
        course_root=tmp_path / "originals",
        workspace_course_ids=_COURSE,
        workspace_answer_enabled=True,
        deepseek_api_key="synthetic-key",
        source_vision_enabled=False,
        _env_file=None,
    )


def test_generation_drops_mismatched_experiment_claims_and_keeps_definition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    ev1 = _ev(
        "ev-1",
        "doc-guide",
        "Basalt-Latch-Guide.pdf",
        "page:1",
        "Definition: A basalt latch opens on three consecutive readings strictly above 8 V.\n"
        "Experiment-ID: SlateTrialB6.",
        page=1,
    )
    ev2 = _ev(
        "ev-2",
        "doc-lab",
        "Latch-Trial.ipynb",
        "cell:0",
        "Experiment-ID: SlateTrialB6. Split: held-out. Metric: MAE. Unit: amplitude. "
        "Baseline: 0.26. Trial: 0.19. Samples: 4.",
        cell_index=0,
    )

    async def fake_answer(self, messages, **kwargs):
        # Simulate LLM falsely claiming held-out 0.26 -> 0.19 answers the test/salt-fog query.
        return {
            "content": json.dumps(
                {
                    "statements": [
                        {
                            "text": "Basalt latch 在连续三次读数大于 8 V 时开启。",
                            "citations": ["ev-1"],
                        },
                        {"text": "实验 MAE 从 0.26 降至 0.19。", "citations": ["ev-2"]},
                    ],
                    "missing": "",
                }
            )
        }

    monkeypatch.setattr(DeepSeekClient, "generate_answer", fake_answer)
    for question in (
        "Basalt latch 的定义是什么？讲义所对应的测试集实验如何验证？",
        "Basalt latch 的定义是什么？讲义所对应的盐雾实验如何验证？",
        "Basalt latch 的定义是什么？讲义所对应的低温循环后的留出集实验如何验证？",
        "Basalt latch 的定义是什么？讲义所对应的 Quartz latch 的留出集实验如何验证？",
        "Basalt latch 的定义是什么？讲义所对应的留出集实验如何验证？请给出 RMSE。",
    ):
        res = generate_response(
            settings,
            question,
            {"ev-1": ev1, "ev-2": ev2},
            mode="definition_and_experiment",
        )
        assert res.status == "partial", question
        assert len(res.statements) == 1
        assert res.statements[0].citations == ("ev-1",)
        assert "0.26" not in res.statements[0].text
        assert res.trace["unverified_experiment_statements"] == 1
        assert "ev-2" in res.trace["rejected_experiment_reasons"]
        assert res.missing


def test_explicit_heldout_test_equivalence_vs_unmarked_or_negated() -> None:
    defn = _ev(
        "ev-1",
        "doc-guide",
        "Basalt-Latch-Guide.pdf",
        "page:1",
        "Definition: A basalt latch opens on three consecutive readings strictly above 8 V.\n"
        "Experiment-ID: SlateTrialB6.",
        page=1,
    )
    q = "Basalt latch 的定义是什么？讲义所对应的测试集实验如何验证？"

    # Unmarked held-out must be rejected for test-split question
    plain_heldout = _ev(
        "ev-2",
        "doc-lab",
        "Latch-Trial.ipynb",
        "cell:0",
        "Experiment-ID: SlateTrialB6. Split: held-out. Metric: MAE. Unit: amplitude. "
        "Baseline: 0.26. Trial: 0.19. Samples: 4.",
        cell_index=0,
    )
    sup, rej = classify_experiment_evidence({"ev-1": defn, "ev-2": plain_heldout}, q)
    assert sup == set() and rej == {"ev-2": "experiment_split_mismatch"}

    # Negated equivalence must also be rejected
    neg_heldout = _ev(
        "ev-2",
        "doc-lab",
        "Latch-Trial.ipynb",
        "cell:0",
        "Experiment-ID: SlateTrialB6. Split: held-out. Metric: MAE. Unit: amplitude. "
        "Baseline: 0.26. Trial: 0.19. Samples: 4.\n"
        "The held-out set is not used as the test split.",
        cell_index=0,
    )
    sup_neg, rej_neg = classify_experiment_evidence({"ev-1": defn, "ev-2": neg_heldout}, q)
    assert sup_neg == set() and rej_neg == {"ev-2": "experiment_split_mismatch"}

    # Explicit equivalence in source text must be accepted
    equiv_heldout = _ev(
        "ev-2",
        "doc-lab",
        "Latch-Trial.ipynb",
        "cell:0",
        "Experiment-ID: SlateTrialB6. Split: held-out. Metric: MAE. Unit: amplitude. "
        "Baseline: 0.26. Trial: 0.19. Samples: 4.\n"
        "Note: the held-out split serves as the test split for SlateTrialB6.",
        cell_index=0,
    )
    sup_eq, rej_eq = classify_experiment_evidence({"ev-1": defn, "ev-2": equiv_heldout}, q)
    assert sup_eq == {"ev-2"} and rej_eq == {}


def test_dual_split_selects_only_matching_split_and_handles_provider_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    service = make_workspace(settings)
    app = FastAPI()
    app.include_router(router, prefix=_BASE)
    app.dependency_overrides[get_workspace_service] = lambda: service

    # Build guide + dual-split notebook (cell 0: held-out 0.26->0.19, cell 1: test 0.31->0.22)
    guide_nb = tmp_path / "Basalt-Guide.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell(
                    "Definition: A basalt latch opens on three consecutive readings > 8 V.\n"
                    "Experiment-ID: SlateTrialB6.",
                    id="guide",
                )
            ]
        ),
        guide_nb,
    )
    dual_nb = tmp_path / "Latch-Dual.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell(
                    "Basalt latch SlateTrialB6 held-out run.\n"
                    "Experiment-ID: SlateTrialB6. Split: held-out. Metric: MAE. Unit: amplitude. "
                    "Baseline: 0.26. Trial: 0.19. Samples: 4.",
                    id="heldout-cell",
                ),
                nbformat.v4.new_markdown_cell(
                    "Basalt latch SlateTrialB6 test split run.\n"
                    "Experiment-ID: SlateTrialB6. Split: test. Metric: MAE. Unit: amplitude. "
                    "Baseline: 0.31. Trial: 0.22. Samples: 6.",
                    id="test-cell",
                ),
            ]
        ),
        dual_nb,
    )

    async def dual_answer(self, messages, **kwargs):
        req = json.loads(messages[-1]["content"])
        by_cell = {
            item["locator"].get("cell_index"): item["id"]
            for item in req["evidence"]
            if item["file_name"] == "Latch-Dual.ipynb"
        }
        guide_id = next(
            item["id"] for item in req["evidence"] if item["file_name"] == "Basalt-Guide.ipynb"
        )
        statements = [{"text": "Basalt latch 定义。", "citations": [guide_id]}]
        if 0 in by_cell:
            statements.append(
                {"text": "留出集 MAE 从 0.26 降至 0.19。", "citations": [by_cell[0]]}
            )
        if 1 in by_cell:
            statements.append(
                {"text": "测试集 MAE 从 0.31 降至 0.22。", "citations": [by_cell[1]]}
            )
        return {"content": json.dumps({"statements": statements, "missing": ""})}

    monkeypatch.setattr(DeepSeekClient, "generate_answer", dual_answer)
    with TestClient(app, base_url=_HOST, headers={"Origin": _HOST}) as client:
        sess = client.post(f"{_BASE}/sessions", json={"course_ids": [_COURSE]}).json()
        sid = f"{_BASE}/sessions/{sess['session_id']}"
        for path in (guide_nb, dual_nb):
            resp = client.post(
                f"{sid}/import",
                params={"course_id": _COURSE, "filename": path.name},
                headers={"content-type": "application/octet-stream"},
                content=path.read_bytes(),
            )
            assert resp.status_code == 200

        r_test = client.post(
            f"{sid}/turns",
            json={
                "client_turn_id": "t-test",
                "text": "Basalt latch 的定义是什么？讲义所对应的测试集实验如何验证？",
                "scope_revision": 1,
            },
        ).json()
        assert r_test["turn"]["trace"]["generation"]["status"] == "succeeded"
        texts_test = [b["text"] for b in r_test["message"]["blocks"] if b["type"] == "text"]
        assert any("0.31" in t and "0.22" in t for t in texts_test)
        assert all("0.26" not in t and "0.19" not in t for t in texts_test)

        # Provider failure must safely fall back to excerpt without claiming support
        async def fail_provider(self, messages, **kwargs):
            raise httpx.ConnectError("simulated provider outage")

        monkeypatch.setattr(DeepSeekClient, "generate_answer", fail_provider)
        r_fail = client.post(
            f"{sid}/turns",
            json={
                "client_turn_id": "t-fail",
                "text": "Basalt latch 的定义是什么？讲义所对应的测试集实验如何验证？",
                "scope_revision": 1,
            },
        ).json()
        assert r_fail["turn"]["trace"]["generation"]["status"] == "provider_error"
        assert any(
            b.get("code") == "excerpt_fallback" for b in r_fail["message"]["blocks"]
        )
