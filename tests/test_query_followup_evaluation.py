"""追问题集与指标计算的可复核性测试。"""

from pathlib import Path

from app.schemas.rag import SearchHit, SourceReference
from evals.run_query_followup import load_cases, source_rank, summarize
from evals.run_query_intent import summarize as summarize_intent

NOTEBOOK_DIALOGUE_CASES = Path("evals/query_notebook_dialogue_cases.jsonl")


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


def test_source_rank_matches_notebook_cells_and_document() -> None:
    cases = load_cases(NOTEBOOK_DIALOGUE_CASES)

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
    hit.source.cell_ids = ["unrelated-cell"]
    assert source_rank([hit], case) is None
    hit.source.cell_ids = [case.cell_ids[0]]
    hit.source.document_id = "unrelated-document"
    assert source_rank([hit], case) is None


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
