"""追问题集与指标计算的可复核性测试。"""

import json
from pathlib import Path

from app.rag.query import construct_query
from app.schemas.rag import SearchHit, SearchRequest, SourceReference
from evals.run_query_followup import DEFAULT_CASES, load_cases, source_rank, summarize
from evals.run_query_intent import summarize as summarize_intent

HOLDOUT_CASES = Path("evals/query_followup_holdout_cases.jsonl")
FILE_DIALOGUE_CASES = Path("evals/query_dialogue_file_cases.jsonl")
NOTEBOOK_DIALOGUE_CASES = Path("evals/query_notebook_dialogue_cases.jsonl")
NOTEBOOK_HOLDOUT_CASES = Path("evals/query_notebook_holdout_cases.jsonl")


def test_followup_cases_are_disjoint_from_earlier_slide_targets() -> None:
    cases = load_cases(DEFAULT_CASES)
    old_targets: set[tuple[str, int]] = set()
    for path in Path("evals").glob("*cases.jsonl"):
        if path == DEFAULT_CASES:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            case = json.loads(line)
            old_targets.update(
                (case["document_id"], page)
                for page in case.get("pages", [])
                if "document_id" in case
            )

    assert len(cases) == 16
    assert {(case.document_id, page) for case in cases for page in case.pages}.isdisjoint(
        old_targets
    )
    assert all(
        construct_query(
            SearchRequest(
                query=case.query,
                query_construction="contextual",
                history=case.history,
            )
        ).method
        == "history_expanded"
        for case in cases
        if case.kind == "followup"
    )


def test_summary_separates_followup_gain_from_false_expansion() -> None:
    rows = [
        {
            "kind": "followup",
            "raw_rank": None,
            "contextual_rank": 2,
            "query_method": "history_expanded",
        },
        {
            "kind": "followup",
            "raw_rank": 1,
            "contextual_rank": 7,
            "query_method": "history_expanded",
        },
        {
            "kind": "independent",
            "raw_rank": 1,
            "contextual_rank": 7,
            "query_method": "history_expanded",
        },
        {
            "kind": "independent",
            "raw_rank": 2,
            "contextual_rank": 2,
            "query_method": "raw",
        },
    ]

    metrics = summarize(rows)

    assert metrics["followup_raw_recall@5"] == 0.5
    assert metrics["followup_contextual_recall@5"] == 0.5
    assert metrics["followup_gained@5"] == 1
    assert metrics["followup_lost@5"] == 1
    assert metrics["false_expansion_rate"] == 0.5
    assert metrics["independent_harmed@5"] == 1


def test_holdout_has_new_targets_and_expected_query_intents() -> None:
    holdout = load_cases(HOLDOUT_CASES)
    prior_targets: set[tuple[str, int]] = set()
    for path in Path("evals").glob("*cases.jsonl"):
        if path == HOLDOUT_CASES:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            case = json.loads(line)
            prior_targets.update(
                (case["document_id"], page)
                for page in case.get("pages", [])
                if "document_id" in case
            )

    assert len(holdout) == 16
    assert {(case.document_id, page) for case in holdout for page in case.pages}.isdisjoint(
        prior_targets
    )
    for case in holdout:
        method = construct_query(
            SearchRequest(
                query=case.query,
                query_construction="contextual",
                history=case.history,
            )
        ).method
        assert method == ("history_expanded" if case.kind == "followup" else "raw")


def test_file_dialogue_cases_use_new_query_documents_and_multiturn_history() -> None:
    cases = load_cases(FILE_DIALOGUE_CASES)
    old_document_ids = {
        case.document_id for path in (DEFAULT_CASES, HOLDOUT_CASES) for case in load_cases(path)
    }

    assert len(cases) == 16
    assert {case.document_id for case in cases}.isdisjoint(old_document_ids)
    assert all(len(case.history) >= 3 for case in cases)
    assert all(any(turn.role == "assistant" for turn in case.history) for case in cases)


def test_notebook_dialogue_cases_use_distinct_documents_and_cell_sources() -> None:
    cases = load_cases(NOTEBOOK_DIALOGUE_CASES)
    prior_documents = {
        case.document_id
        for path in (DEFAULT_CASES, HOLDOUT_CASES, FILE_DIALOGUE_CASES)
        for case in load_cases(path)
    }

    assert len(cases) == 16
    assert {case.document_id for case in cases}.isdisjoint(prior_documents)
    assert all(case.cell_ids and not case.pages for case in cases)
    assert all(len(case.history) >= 3 for case in cases)

    case = cases[0]
    hit = SearchHit(
        chunk_id="chunk-1",
        section_id="section-1",
        representation_type="text",
        score=1.0,
        content="example",
        source=SourceReference(
            course_id=case.course_id,
            document_id=case.document_id,
            file="Lecture1a.ipynb",
            cell_ids=[case.cell_ids[0]],
        ),
    )
    assert source_rank([hit], case) == 1


def test_notebook_holdout_uses_new_cell_targets() -> None:
    development = load_cases(NOTEBOOK_DIALOGUE_CASES)
    holdout = load_cases(NOTEBOOK_HOLDOUT_CASES)
    development_targets = {
        (case.document_id, cell_id) for case in development for cell_id in case.cell_ids
    }
    holdout_targets = {(case.document_id, cell_id) for case in holdout for cell_id in case.cell_ids}

    assert len(holdout) == 8
    assert holdout_targets.isdisjoint(development_targets)


def test_intent_summary_counts_misses_and_retrieval_harm_separately() -> None:
    rows = [
        {
            "kind": "followup",
            "rule_method": "raw",
            "model_method": "history_expanded",
            "raw_rank": None,
            "rule_rank": None,
            "model_rank": 1,
            "usage": {"total_tokens": 20},
            "attempts": 2,
            "deepseek_latency_ms": 100,
        },
        {
            "kind": "independent",
            "rule_method": "raw",
            "model_method": "history_expanded",
            "raw_rank": 2,
            "rule_rank": 2,
            "model_rank": None,
            "usage": {"total_tokens": 30},
            "attempts": 1,
            "deepseek_latency_ms": 200,
        },
    ]

    metrics = summarize_intent(rows)

    assert metrics["model_followup_recall@5"] == 1.0
    assert metrics["model_false_expansion"] == 1
    assert metrics["model_independent_harmed@5"] == 1
    assert metrics["deepseek_total_tokens"] == 50
    assert metrics["deepseek_retries"] == 1
    assert metrics["mean_deepseek_latency_ms"] == 150
