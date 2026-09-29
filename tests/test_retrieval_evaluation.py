from pathlib import Path

import pytest

from app.evaluation.retrieval import (
    ExpectedSource,
    RetrievalCase,
    aggregate_metrics,
    evaluate_case,
    load_cases,
    load_corpus,
    source_matches,
)
from app.schemas.rag import SearchHit, SourceReference


def _case(*, expected_sources: list[ExpectedSource] | None = None) -> RetrievalCase:
    return RetrievalCase(
        case_id="case-1",
        split="dev",
        course_id="CS5489",
        query="LASSO 为什么能做特征选择？",
        reference_answer="L1 正则可将部分权重压到零。",
        expected_sources=expected_sources
        or [
            ExpectedSource(
                source_path="CS5489 Machine learning/Lecture4/Lecture4a.ipynb",
                section_path=["LASSO"],
                cell_ids=["cell-63"],
            )
        ],
    )


def _hit(
    *,
    file: str = "/Users/test/Course/CS5489 Machine learning/Lecture4/Lecture4a.ipynb",
    section_path: list[str] | None = None,
    cell_ids: list[str] | None = None,
) -> SearchHit:
    return SearchHit(
        chunk_id="chunk-1",
        section_id="section-1",
        representation_type="text_content",
        score=0.8,
        content="LASSO uses an L1 penalty.",
        source=SourceReference(
            course_id="CS5489",
            document_id="doc-1",
            file=file,
            section_path=section_path or ["LASSO"],
            cell_ids=cell_ids or ["cell-63"],
        ),
    )


def test_source_matching_requires_requested_granularity() -> None:
    expected = _case().expected_sources[0]

    assert source_matches(_hit(), expected, level="file")
    assert source_matches(_hit(), expected, level="section")
    assert source_matches(_hit(), expected, level="cell")
    assert not source_matches(_hit(section_path=["Ridge Regression"]), expected, level="section")
    assert not source_matches(_hit(cell_ids=["cell-64"]), expected, level="cell")


def test_aggregate_metrics_uses_first_relevant_rank_and_source_levels() -> None:
    case = _case()
    evaluation = evaluate_case(
        case,
        [
            _hit(section_path=["Ridge Regression"], cell_ids=["cell-42"]),
            _hit(),
        ],
        latency_ms=25.0,
    )

    metrics = aggregate_metrics([case], [evaluation], ks=(1, 3))

    assert metrics["recall@1"] == 0.0
    assert metrics["recall@3"] == 1.0
    assert metrics["mrr@3"] == 0.5
    assert metrics["ndcg@3"] == pytest.approx(1 / 1.5849625007)
    assert metrics["file_hit_rate@3"] == 1.0
    assert metrics["section_hit_rate@3"] == 1.0
    assert metrics["cell_hit_rate@3"] == 1.0
    assert metrics["latency_ms"] == {
        "mean": 25.0,
        "p50": 25.0,
        "p95": 25.0,
        "max": 25.0,
    }


def test_duplicate_hits_do_not_receive_duplicate_relevance_credit() -> None:
    case = _case()

    evaluation = evaluate_case(case, [_hit(), _hit()], latency_ms=1.0)

    assert evaluation.relevant_ranks == (1,)
    assert aggregate_metrics([case], [evaluation], ks=(2,))["recall@2"] == 1.0


def test_load_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    case_json = _case().model_dump_json()
    path = tmp_path / "cases.jsonl"
    path.write_text(f"{case_json}\n{case_json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="重复 case_id"):
        load_cases(path)


def test_load_corpus_rejects_duplicate_course_paths(tmp_path: Path) -> None:
    item = {
        "course_id": "CS5489",
        "source_path": "Lecture1.ipynb",
        "sha256": "a" * 64,
    }
    path = tmp_path / "corpus.json"
    path.write_text(f"[{item!r}, {item!r}]".replace("'", '"'), encoding="utf-8")

    with pytest.raises(ValueError, match="重复课程/路径"):
        load_corpus(path)
