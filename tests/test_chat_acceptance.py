"""验收报告必须把范围越界和不完整总结判为失败。"""

from unittest.mock import Mock

from app.rag.chat import ChatResult
from app.schemas.rag import SearchHit, SearchResponse, SearchScope, SourceReference
from evals.run_chat_acceptance import _turn_record


def test_acceptance_rejects_citation_legal_but_out_of_scope_answer() -> None:
    hit = SearchHit(
        chunk_id="chunk",
        section_id="section",
        representation_type="text_content",
        score=1,
        content="事实",
        source=SourceReference(course_id="other", document_id="other", file="other.pdf", version=1),
    )
    result = ChatResult(
        query="问题",
        answer="事实[1]",
        search=SearchResponse(query="问题", hits=[hit]),
        evidence=[hit],
        cited_numbers=[1],
        citation_warning=None,
        model_tokens=1,
        model_called=True,
    )
    turn = {
        "turn_id": "scope",
        "query": "问题",
        "expected": {
            "routes": ["knowledge"],
            "retrieval_performed": True,
            "model_called": True,
            "require_citations": True,
        },
    }
    row = _turn_record("session", turn, Mock(scope=SearchScope(course_ids=["allowed"])), result, 1)
    assert row["status"] == "failed"
    assert row["mechanical_evaluation"]["checks"]["citation_numbers_legal"]
    assert not row["mechanical_evaluation"]["checks"]["scope_respected"]


def test_acceptance_rejects_partial_summary_even_with_expected_route() -> None:
    result = ChatResult(
        query="总结",
        answer="总结",
        search=SearchResponse(query="总结", hits=[]),
        evidence=[],
        cited_numbers=[],
        citation_warning=None,
        model_tokens=1,
        route="document_summary",
        retrieval_performed=False,
        model_called=True,
        coverage={"read_chunks": 5, "total_chunks": 10},
    )
    turn = {
        "turn_id": "summary",
        "query": "总结",
        "expected": {
            "routes": ["document_summary"],
            "retrieval_performed": False,
            "model_called": True,
        },
    }
    row = _turn_record("session", turn, Mock(scope=SearchScope()), result, 1)
    assert row["status"] == "failed"
    assert not row["mechanical_evaluation"]["checks"]["summary_read_coverage"]


def test_summary_counts_errors_and_only_applicable_checks() -> None:
    from evals.run_chat_acceptance import _summarize

    report = _summarize(
        [
            {
                "status": "passed",
                "latency_ms": 10,
                "model_tokens": 20,
                "mechanical_evaluation": {
                    "passed": True,
                    "checks": {"route": True, "citation_numbers_legal": True},
                    "applicable_checks": ["route", "citation_numbers_legal"],
                    "source_groups": [{"recalled": True}],
                },
            },
            {"status": "error", "latency_ms": 40},
        ],
        expected_turn_count=3,
    )
    assert report["completion_rate"] == 1 / 3
    assert report["mechanical_pass_rate"] == 1 / 2
    assert report["checks"]["route"] == {
        "applicable_turns": 1,
        "passed_turns": 1,
        "pass_rate": 1.0,
    }
    assert report["source_group_recall"] == 1.0
    assert report["failed_turns"] == 1
    assert report["latency_p50_ms"] == 10
    assert report["latency_p95_ms"] == 40
    empty = _summarize([])
    assert empty["completion_rate"] is None
    assert empty["mechanical_pass_rate"] is None
    assert empty["source_group_recall"] is None
    assert empty["latency_p95_ms"] is None


def test_environment_gate_cannot_pass_on_changed_or_missing_snapshot() -> None:
    from evals.run_chat_acceptance import _automatic_pass_gates

    frozen = dict.fromkeys(("sources", "sqlite_index", "qdrant_index", "visual_text_cache"), False)
    args = {
        "execution_succeeded": True,
        "attempted_turns": 2,
        "expected_turns": 2,
        "failed_turns": 0,
    }
    assert _automatic_pass_gates(**args, environment_changed=frozen)["stage_automatic_passed"]
    for snapshot in ({}, {**frozen, "visual_text_cache": True}):
        result = _automatic_pass_gates(**args, environment_changed=snapshot)
        assert result["overall_passed"]
        assert not result["frozen_environment_passed"]
        assert not result["stage_automatic_passed"]
    partial = _automatic_pass_gates(**{**args, "attempted_turns": 1}, environment_changed=frozen)
    assert not partial["overall_passed"]
