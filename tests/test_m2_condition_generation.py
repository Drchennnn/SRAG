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


def test_split_equivalence_attribution_rejects_other_experiment_and_preserves_order_invariance():
    from app.rag.evidence_supplement import _condition_problem

    q_a = "实验 A 的测试集 MAE 是多少？"
    fields = {"split": "held-out", "metric": "MAE"}
    assert (
        _condition_problem(
            q_a,
            fields,
            context_texts=(
                "A 使用留出验证集。For Trial B only, held-out is used as the test split.",
            ),
        )
        == "experiment_split_mismatch"
    )
    assert (
        _condition_problem(
            q_a,
            fields,
            context_texts=(
                "A 使用留出验证集。For Trial A only, held-out is used as the test split.",
            ),
        )
        is None
    )
    assert (
        _condition_problem(
            q_a,
            fields,
            context_texts=("For Trial A, held-out is different from the test split.",),
        )
        == "experiment_split_mismatch"
    )

    # Same cell containing both TrialA and TrialB in both orders (A then B vs B then A)
    def_a = _ev(
        "ev-ga",
        "doc-g",
        "Guide.pdf",
        "page:1",
        "Definition: A basalt latch opens on three consecutive readings > 8 V.\n"
        "Experiment-ID: TrialA.",
        page=1,
    )
    def_b = _ev(
        "ev-gb",
        "doc-g",
        "Guide.pdf",
        "page:2",
        "Definition: A quartz latch opens on two consecutive readings > 5 V.\n"
        "Experiment-ID: TrialB.",
        page=2,
    )
    rec_a = (
        "Experiment-ID: TrialA. Split: held-out. Metric: MAE. Unit: amplitude. "
        "Baseline: 0.26. Trial: 0.19. Samples: 4. "
        "For TrialA only, held-out is used as the test split."
    )
    rec_b = (
        "Experiment-ID: TrialB. Split: held-out. Metric: MAE. Unit: amplitude. "
        "Baseline: 0.44. Trial: 0.31. Samples: 8."
    )
    q_basalt = "Basalt latch 的定义是什么？讲义所对应的测试集实验如何验证？"
    q_quartz = "Quartz latch 的定义是什么？讲义所对应的测试集实验如何验证？"

    for cell_text in (f"{rec_a}\n\n{rec_b}", f"{rec_b}\n\n{rec_a}"):
        ev_cell = _ev("ev-lab", "doc-l", "Dual-Cell.ipynb", "cell:0", cell_text, cell_index=0)
        sup_a, rej_a = classify_experiment_evidence(
            {"ev-ga": def_a, "ev-gb": def_b, "ev-lab": ev_cell}, q_basalt
        )
        assert sup_a == {"ev-lab"} and rej_a == {}
        sup_b, rej_b = classify_experiment_evidence(
            {"ev-ga": def_a, "ev-gb": def_b, "ev-lab": ev_cell}, q_quartz
        )
        assert sup_b == set() and rej_b == {"ev-lab": "experiment_split_mismatch"}

    # Unattributed generic sentence in a multi-experiment block must not apply to either
    shared_block = (
        "Experiment-ID: TrialA. Split: held-out. Metric: MAE. Unit: amplitude. "
        "Baseline: 0.26. Trial: 0.19. Samples: 4.\n"
        "Experiment-ID: TrialB. Split: held-out. Metric: MAE. Unit: amplitude. "
        "Baseline: 0.44. Trial: 0.31. Samples: 8.\n"
        "Held-out is used as the test split."
    )
    ev_shared = _ev("ev-lab", "doc-l", "Shared.ipynb", "cell:0", shared_block, cell_index=0)
    sup_s, rej_s = classify_experiment_evidence(
        {"ev-ga": def_a, "ev-lab": ev_shared}, q_basalt
    )
    assert sup_s == set() and rej_s == {"ev-lab": "experiment_split_mismatch"}


def test_rejected_experiment_numbers_stripped_from_missing_and_uncited_statements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay public different_metric output: MAE 0.26/0.19 must not leak without citations."""
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
    ev5 = _ev(
        "ev-5",
        "doc-lab",
        "Latch-Trial.ipynb",
        "cell:0",
        "Experiment-ID: SlateTrialB6. Split: held-out. Metric: MAE. Unit: amplitude. "
        "Baseline: 0.26. Trial: 0.19. Samples: 4.",
        cell_index=0,
    )

    async def fake_answer(self, messages, **kwargs):
        return {
            "content": json.dumps(
                {
                    "statements": [
                        {
                            "text": "Basalt latch 的定义是：连续三次读数严格高于 8 V 时开启。",
                            "citations": ["ev-1"],
                        },
                        {
                            "text": "现有证据只给出 MAE（基线 0.26、试验 0.19）。",
                            "citations": ["ev-1"],
                        },
                    ],
                    "missing": (
                        "缺少留出集实验的 RMSE 数值；"
                        "现有证据只给出 MAE（基线 0.26、试验 0.19，单位 amplitude）。"
                    ),
                }
            )
        }

    monkeypatch.setattr(DeepSeekClient, "generate_answer", fake_answer)
    res = generate_response(
        settings,
        "Basalt latch 的定义是什么？讲义所对应的留出集实验如何验证？请给出 RMSE。",
        {"ev-1": ev1, "ev-5": ev5},
        mode="definition_and_experiment",
    )
    assert res.status == "partial"
    assert res.trace["rejected_experiment_reasons"] == {"ev-5": "experiment_metric_mismatch"}
    assert res.trace["unsupported_literal_statements"] == 1
    assert [s.citations for s in res.statements] == [("ev-1",)]
    assert "0.26" not in res.missing and "0.19" not in res.missing
    assert "RMSE" in res.missing
    for s in res.statements:
        assert "0.26" not in s.text and "0.19" not in s.text


def test_extra_verification_demand_cleared_when_answered_but_kept_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay public same_cell_rev/held_out false-partial outputs and counter-example."""
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
    ev4 = _ev(
        "ev-4",
        "doc-lab",
        "Latch-Trial.ipynb",
        "cell:0",
        "Experiment-ID: SlateTrialB6. Split: held-out. Metric: MAE. Unit: amplitude. "
        "Baseline: 0.26. Trial: 0.19. Samples: 4.",
        cell_index=0,
    )

    replays = iter(
        [
            (
                "Basalt latch 的定义是什么？讲义所对应的留出集实验如何验证？",
                "现有证据未说明该留出集实验如何验证 basalt latch 的定义，"
                "也未说明实验条件与“连续三次读数严格高于 8 V”这一触发条件之间的关系。",
                "succeeded",
            ),
            (
                "Basalt latch 的定义是什么？讲义所对应的留出集实验如何验证？",
                "未找到讲义中针对 Basalt latch 的独立验证流程或额外实验说明；"
                "现有证据仅给出 TrialA 的 held-out 指标数值。",
                "succeeded",
            ),
            (
                "Basalt latch 的定义是什么？讲义所对应的留出集实验如何验证？"
                "请给出针对定义阈值的独立验证流程。",
                "未找到讲义中针对 Basalt latch 的独立验证流程或额外实验说明。",
                "partial",
            ),
        ]
    )

    for question, missing_text, expected_status in replays:
        async def fake_answer(self, messages, _m=missing_text, **kwargs):
            return {
                "content": json.dumps(
                    {
                        "statements": [
                            {
                                "text": "Basalt latch 的定义是：连续三次读数严格高于 8 V 时开启。",
                                "citations": ["ev-1"],
                            },
                            {
                                "text": (
                                    "留出集实验 SlateTrialB6 的指标为 MAE，"
                                    "基线 0.26，试验 0.19。"
                                ),
                                "citations": ["ev-4"],
                            },
                        ],
                        "missing": _m,
                    }
                )
            }

        monkeypatch.setattr(DeepSeekClient, "generate_answer", fake_answer)
        res = generate_response(
            settings,
            question,
            {"ev-1": ev1, "ev-4": ev4},
            mode="definition_and_experiment",
        )
        assert res.status == expected_status
        if expected_status == "succeeded":
            assert res.missing == ""
        else:
            assert "独立验证流程" in res.missing
