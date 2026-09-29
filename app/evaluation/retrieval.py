"""Notebook 检索基线的数据契约、来源匹配与指标计算。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import fmean

from pydantic import BaseModel, Field

from app.schemas.rag import SearchHit


class ExpectedSource(BaseModel):
    """一个可被检索结果满足的人工标注来源。"""

    source_path: str
    section_path: list[str] = Field(min_length=1)
    cell_ids: list[str] = Field(min_length=1)
    relevance: int = Field(default=3, ge=1, le=3)


class RetrievalCase(BaseModel):
    """一条检索评估样本。"""

    case_id: str
    split: str
    course_id: str
    query: str
    reference_answer: str
    expected_sources: list[ExpectedSource] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


class CorpusDocument(BaseModel):
    """评估语料中一份被冻结内容的 Notebook。"""

    course_id: str
    source_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    case_id: str
    split: str
    latency_ms: float
    relevant_ranks: tuple[int, ...]
    matched_source_indexes: tuple[int, ...]
    file_hit: bool
    section_hit: bool
    cell_hit: bool


def load_cases(path: Path) -> list[RetrievalCase]:
    """读取 JSONL，并拒绝重复 case_id 或空文件。"""

    cases: list[RetrievalCase] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue
            case = RetrievalCase.model_validate_json(line)
            if case.case_id in seen_ids:
                raise ValueError(f"{path}:{line_number} 存在重复 case_id：{case.case_id}")
            seen_ids.add(case.case_id)
            cases.append(case)
    if not cases:
        raise ValueError(f"评估集为空：{path}")
    return cases


def load_corpus(path: Path) -> list[CorpusDocument]:
    """读取评估语料清单，并拒绝重复的课程/路径组合。"""

    raw_documents = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_documents, list) or not raw_documents:
        raise ValueError(f"评估语料清单为空：{path}")
    documents = [CorpusDocument.model_validate(item) for item in raw_documents]
    keys = [(item.course_id, item.source_path) for item in documents]
    if len(set(keys)) != len(keys):
        raise ValueError(f"评估语料清单存在重复课程/路径：{path}")
    return documents


def evaluate_case(
    case: RetrievalCase,
    hits: list[SearchHit],
    *,
    latency_ms: float,
) -> CaseEvaluation:
    """按文件、章节和 cell 严格匹配一条检索结果。"""

    unmatched = set(range(len(case.expected_sources)))
    relevant_ranks: list[int] = []
    matched_source_indexes: list[int] = []
    for rank, hit in enumerate(hits, start=1):
        candidates = [
            index
            for index in unmatched
            if source_matches(hit, case.expected_sources[index], level="cell")
        ]
        if not candidates:
            continue
        matched_index = max(
            candidates,
            key=lambda index: case.expected_sources[index].relevance,
        )
        unmatched.remove(matched_index)
        relevant_ranks.append(rank)
        matched_source_indexes.append(matched_index)

    return CaseEvaluation(
        case_id=case.case_id,
        split=case.split,
        latency_ms=latency_ms,
        relevant_ranks=tuple(relevant_ranks),
        matched_source_indexes=tuple(matched_source_indexes),
        file_hit=any(
            source_matches(hit, expected, level="file")
            for hit in hits
            for expected in case.expected_sources
        ),
        section_hit=any(
            source_matches(hit, expected, level="section")
            for hit in hits
            for expected in case.expected_sources
        ),
        cell_hit=bool(relevant_ranks),
    )


def source_matches(
    hit: SearchHit,
    expected: ExpectedSource,
    *,
    level: str,
) -> bool:
    """判断命中是否到达指定来源粒度。"""

    if level not in {"file", "section", "cell"}:
        raise ValueError(f"未知来源匹配粒度：{level}")
    if not _path_ends_with(hit.source.file, expected.source_path):
        return False
    if level == "file":
        return True
    if hit.source.section_path != expected.section_path:
        return False
    if level == "section":
        return True
    return bool(set(hit.source.cell_ids).intersection(expected.cell_ids))


def aggregate_metrics(
    cases: list[RetrievalCase],
    evaluations: list[CaseEvaluation],
    *,
    ks: tuple[int, ...] = (1, 3, 5, 8),
) -> dict[str, object]:
    """计算 macro Recall@K、MRR、nDCG、来源命中率和延迟。"""

    if len(cases) != len(evaluations):
        raise ValueError("评估样本与结果数量不一致")
    if not cases:
        raise ValueError("没有可聚合的评估结果")
    if any(k < 1 for k in ks):
        raise ValueError("K 必须大于等于 1")

    by_id = {evaluation.case_id: evaluation for evaluation in evaluations}
    if len(by_id) != len(evaluations):
        raise ValueError("评估结果包含重复 case_id")
    missing = [case.case_id for case in cases if case.case_id not in by_id]
    if missing:
        raise ValueError(f"缺少评估结果：{', '.join(missing)}")

    ordered = [by_id[case.case_id] for case in cases]
    recall = {
        f"recall@{k}": fmean(
            _case_recall(case, evaluation, k)
            for case, evaluation in zip(cases, ordered, strict=True)
        )
        for k in ks
    }
    max_k = max(ks)
    mrr = fmean(
        1.0 / evaluation.relevant_ranks[0]
        if evaluation.relevant_ranks and evaluation.relevant_ranks[0] <= max_k
        else 0.0
        for evaluation in ordered
    )
    ndcg = fmean(
        _case_ndcg(case, evaluation, max_k)
        for case, evaluation in zip(cases, ordered, strict=True)
    )
    latencies = sorted(evaluation.latency_ms for evaluation in ordered)
    return {
        "case_count": len(cases),
        **recall,
        f"mrr@{max_k}": mrr,
        f"ndcg@{max_k}": ndcg,
        f"file_hit_rate@{max_k}": fmean(float(item.file_hit) for item in ordered),
        f"section_hit_rate@{max_k}": fmean(float(item.section_hit) for item in ordered),
        f"cell_hit_rate@{max_k}": fmean(float(item.cell_hit) for item in ordered),
        "latency_ms": {
            "mean": fmean(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": latencies[-1],
        },
    }


def dump_json(path: Path, payload: object) -> None:
    """以稳定、可审阅的格式写入评估结果。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _case_recall(case: RetrievalCase, evaluation: CaseEvaluation, k: int) -> float:
    retrieved = sum(rank <= k for rank in evaluation.relevant_ranks)
    return retrieved / len(case.expected_sources)


def _case_ndcg(case: RetrievalCase, evaluation: CaseEvaluation, k: int) -> float:
    relevance_by_index = [source.relevance for source in case.expected_sources]
    dcg = sum(
        (2**relevance_by_index[source_index] - 1) / math.log2(rank + 1)
        for rank, source_index in zip(
            evaluation.relevant_ranks,
            evaluation.matched_source_indexes,
            strict=True,
        )
        if rank <= k
    )
    ideal = sum(
        (2**relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(
            sorted(relevance_by_index, reverse=True)[:k],
            start=1,
        )
    )
    return dcg / ideal if ideal else 0.0


def _path_ends_with(actual: str, expected: str) -> bool:
    actual_parts = PurePosixPath(actual.replace("\\", "/")).parts
    expected_parts = PurePosixPath(expected.replace("\\", "/")).parts
    return len(actual_parts) >= len(expected_parts) and (
        actual_parts[-len(expected_parts) :] == expected_parts
    )


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
