"""索引评估语料并运行 Notebook dense retrieval baseline。"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings, get_settings
from app.evaluation.retrieval import (
    CaseEvaluation,
    RetrievalCase,
    aggregate_metrics,
    dump_json,
    evaluate_case,
    load_cases,
)
from app.rag.indexer import DocumentIndexer
from app.rag.retriever import HybridRetriever
from app.schemas.rag import IndexRequest, SearchRequest, SearchScope
from app.stores.qdrant import QdrantStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "evals" / "retrieval_cases.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "evals" / "results" / "dense_baseline.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--split",
        choices=("all", "dev", "holdout"),
        default="all",
        help="选择评估集切分；默认运行全部样本。",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="评估前按数据集列出的路径执行幂等增量索引。",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k 必须大于等于 1")
    cases = load_cases(args.cases)
    if args.split != "all":
        cases = [case for case in cases if case.split == args.split]
    if not cases:
        raise SystemExit(f"切分 {args.split!r} 中没有评估样本")

    settings = get_settings()
    if args.index:
        await index_corpus(settings, cases)

    retriever = HybridRetriever.from_settings(settings)
    case_results: list[dict[str, object]] = []
    evaluations = []
    try:
        for case in cases:
            started = time.perf_counter()
            response = await retriever.search(
                SearchRequest(
                    query=case.query,
                    scope=SearchScope(course_ids=[case.course_id]),
                    top_k=args.top_k,
                )
            )
            latency_ms = (time.perf_counter() - started) * 1000
            evaluation = evaluate_case(case, response.hits, latency_ms=latency_ms)
            evaluations.append(evaluation)
            case_results.append(
                {
                    "case_id": case.case_id,
                    "split": case.split,
                    "query": case.query,
                    "latency_ms": round(latency_ms, 3),
                    "relevant_ranks": list(evaluation.relevant_ranks),
                    "file_hit": evaluation.file_hit,
                    "section_hit": evaluation.section_hit,
                    "cell_hit": evaluation.cell_hit,
                    "top_hits": [
                        {
                            "rank": rank,
                            "score": round(hit.score, 6),
                            "source_path": _relative_source(settings, hit.source.file),
                            "section_path": hit.source.section_path,
                            "cell_ids": hit.source.cell_ids,
                        }
                        for rank, hit in enumerate(response.hits, start=1)
                    ],
                }
            )
    finally:
        if isinstance(retriever.vector_store, QdrantStore):
            await retriever.vector_store.close()

    metrics = aggregate_metrics(cases, evaluations, ks=_metric_ks(args.top_k))
    metrics_by_split = _aggregate_by_split(cases, evaluations, args.top_k)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "retrieval_method": "raw_query_text_dense_cosine",
        "embedding_model": settings.dashscope_embedding_model,
        "embedding_dimension": settings.dashscope_embedding_dimension,
        "top_k": args.top_k,
        "split": args.split,
        "metrics": metrics,
        "metrics_by_split": metrics_by_split,
        "cases": case_results,
    }
    dump_json(args.output, payload)
    print(f"评估结果已写入 {args.output}")
    print_metrics(metrics)


async def index_corpus(settings: Settings, cases: list[RetrievalCase]) -> None:
    """按课程聚合基线文件，并复用正式索引生命周期。"""

    course_paths: dict[str, set[Path]] = defaultdict(set)
    course_root = settings.course_root.expanduser().resolve()
    for case in cases:
        for expected in case.expected_sources:
            path = (course_root / expected.source_path).resolve()
            if not path.is_relative_to(course_root):
                raise ValueError(f"评估语料路径越出 COURSE_ROOT：{expected.source_path}")
            if path.suffix.lower() != ".ipynb" or not path.is_file():
                raise FileNotFoundError(f"评估 Notebook 不存在：{path}")
            course_paths[case.course_id].add(path)

    indexer = DocumentIndexer.from_settings(settings)
    try:
        for course_id, paths in sorted(course_paths.items()):
            status = await indexer.index(
                IndexRequest(course_id=course_id, paths=sorted(paths))
            )
            if status.failed_files or status.errors:
                raise RuntimeError(
                    f"课程 {course_id} 的评估语料索引失败：{status.errors}"
                )
            print(
                f"{course_id}: indexed={status.indexed_files}, "
                f"skipped={status.skipped_files}"
            )
    finally:
        if isinstance(indexer.text_store, QdrantStore):
            await indexer.text_store.close()


def print_metrics(metrics: dict[str, object]) -> None:
    for key, value in metrics.items():
        if key == "latency_ms":
            latency = value
            assert isinstance(latency, dict)
            print(
                "latency_ms: "
                + ", ".join(f"{name}={number:.2f}" for name, number in latency.items())
            )
        elif isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def _metric_ks(top_k: int) -> tuple[int, ...]:
    return tuple(k for k in (1, 3, 5, 8) if k <= top_k) or (top_k,)


def _aggregate_by_split(
    cases: list[RetrievalCase],
    evaluations: list[CaseEvaluation],
    top_k: int,
) -> dict[str, dict[str, object]]:
    case_by_id = {case.case_id: case for case in cases}
    split_cases: dict[str, list[RetrievalCase]] = defaultdict(list)
    split_evaluations: dict[str, list[CaseEvaluation]] = defaultdict(list)
    for evaluation in evaluations:
        case = case_by_id[evaluation.case_id]
        split_cases[case.split].append(case)
        split_evaluations[case.split].append(evaluation)
    return {
        split: aggregate_metrics(
            split_cases[split],
            split_evaluations[split],
            ks=_metric_ks(top_k),
        )
        for split in sorted(split_cases)
    }


def _relative_source(settings: Settings, source_path: str) -> str:
    path = Path(source_path).expanduser().resolve()
    root = settings.course_root.expanduser().resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


if __name__ == "__main__":
    asyncio.run(main())
