"""运行检索策略消融实验：对比 dense、sparse、RRF、outline、rerank。"""

from __future__ import annotations

import argparse
import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import models

from app.clients.dashscope import DashScopeClient
from app.config import Settings, get_settings
from app.evaluation.retrieval import (
    aggregate_metrics,
    dump_json,
    evaluate_case,
    load_cases,
)
from app.rag.fusion import merge_and_deduplicate, reciprocal_rank_fusion
from app.schemas.embedding import SparseValue
from app.stores.qdrant import QdrantStore
from app.stores.sqlite import MetadataStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "evals" / "retrieval_cases.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "evals" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--split",
        choices=("all", "dev", "holdout"),
        default="all",
    )
    parser.add_argument("--index", action="store_true")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["dense", "sparse", "rrf", "rrf+outline", "rrf+outline+rerank"],
        default=["dense", "rrf", "rrf+outline", "rrf+outline+rerank"],
        help="要评估的检索策略",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    if args.split != "all":
        cases = [case for case in cases if case.split == args.split]
    if not cases:
        raise SystemExit(f"切分 {args.split!r} 中没有评估样本")

    settings = get_settings()
    if args.index:
        await index_corpus(settings, cases)

    # 初始化组件
    dashscope = DashScopeClient(settings)
    qdrant = QdrantStore(settings)
    metadata_store = MetadataStore(settings)
    metadata_store.initialize()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for strategy in args.strategies:
            print(f"\n=== 评估策略: {strategy} ===")
            results = await evaluate_strategy(
                strategy=strategy,
                cases=cases,
                top_k=args.top_k,
                dashscope=dashscope,
                qdrant=qdrant,
                metadata_store=metadata_store,
                settings=settings,
            )
            output_file = args.output_dir / f"{strategy.replace('+', '_')}_baseline.json"
            dump_json(output_file, results)
            print(f"结果已写入 {output_file}")
            print_metrics(results["metrics"])
    finally:
        await qdrant.close()


async def evaluate_strategy(
    *,
    strategy: str,
    cases: list,
    top_k: int,
    dashscope: DashScopeClient,
    qdrant: QdrantStore,
    metadata_store: MetadataStore,
    settings: Settings,
) -> dict:
    """评估单个检索策略。"""
    case_results = []
    evaluations = []

    for case in cases:
        started = time.perf_counter()

        # 获取可见版本
        visible = metadata_store.visible_versions(course_ids=[case.course_id])
        visible_versions = [(v.document_id, v.version) for v in visible]
        if not visible_versions:
            continue

        # Query embedding
        query_vector = await dashscope.embed_query(case.query)

        # 根据策略执行检索
        if strategy == "dense":
            points = await qdrant.search_text_dense(
                query_vector.dense,
                limit=top_k,
                course_ids=[case.course_id],
                visible_versions=visible_versions,
            )
        elif strategy == "sparse":
            if not query_vector.sparse:
                continue
            sparse_vector = _to_sparse_vector(query_vector.sparse)
            points = await qdrant.search_text_sparse(
                sparse_vector,
                limit=top_k,
                course_ids=[case.course_id],
                visible_versions=visible_versions,
            )
        elif strategy == "rrf":
            recall_limit = min(top_k * 3, 50)
            dense_points = await qdrant.search_text_dense(
                query_vector.dense,
                limit=recall_limit,
                course_ids=[case.course_id],
                visible_versions=visible_versions,
            )
            sparse_points = []
            if query_vector.sparse:
                sparse_vector = _to_sparse_vector(query_vector.sparse)
                sparse_points = await qdrant.search_text_sparse(
                    sparse_vector,
                    limit=recall_limit,
                    course_ids=[case.course_id],
                    visible_versions=visible_versions,
                )
            points = reciprocal_rank_fusion([dense_points, sparse_points], k=60)[:top_k]
        elif strategy == "rrf+outline":
            recall_limit = min(top_k * 3, 50)
            dense_points = await qdrant.search_text_dense(
                query_vector.dense,
                limit=recall_limit,
                course_ids=[case.course_id],
                visible_versions=visible_versions,
            )
            sparse_points = []
            if query_vector.sparse:
                sparse_vector = _to_sparse_vector(query_vector.sparse)
                sparse_points = await qdrant.search_text_sparse(
                    sparse_vector,
                    limit=recall_limit,
                    course_ids=[case.course_id],
                    visible_versions=visible_versions,
                )
            outline_points = await qdrant.search_document_outline(
                query_vector.dense,
                limit=min(10, recall_limit),
                course_ids=[case.course_id],
                visible_versions=visible_versions,
            )
            text_fused = reciprocal_rank_fusion([dense_points, sparse_points], k=60)
            points = merge_and_deduplicate(text_fused, outline_points)[:top_k]
        elif strategy == "rrf+outline+rerank":
            recall_limit = min(top_k * 3, 50)
            dense_points = await qdrant.search_text_dense(
                query_vector.dense,
                limit=recall_limit,
                course_ids=[case.course_id],
                visible_versions=visible_versions,
            )
            sparse_points = []
            if query_vector.sparse:
                sparse_vector = _to_sparse_vector(query_vector.sparse)
                sparse_points = await qdrant.search_text_sparse(
                    sparse_vector,
                    limit=recall_limit,
                    course_ids=[case.course_id],
                    visible_versions=visible_versions,
                )
            outline_points = await qdrant.search_document_outline(
                query_vector.dense,
                limit=min(10, recall_limit),
                course_ids=[case.course_id],
                visible_versions=visible_versions,
            )
            text_fused = reciprocal_rank_fusion([dense_points, sparse_points], k=60)
            all_candidates = merge_and_deduplicate(text_fused, outline_points)

            # 评估失败必须显式报错，不能把未重排的候选标记为 rerank 结果。
            if all_candidates:
                documents = [str((p.payload or {}).get("content", "")) for p in all_candidates]
                ranked_indices = await dashscope.rerank(case.query, documents, top_n=top_k)
                points = [
                    models.ScoredPoint(
                        id=all_candidates[idx].id,
                        score=score,
                        payload=all_candidates[idx].payload,
                        version=all_candidates[idx].version,
                    )
                    for idx, score in ranked_indices
                ]
            else:
                points = []
        else:
            raise ValueError(f"未知策略：{strategy}")

        latency_ms = (time.perf_counter() - started) * 1000

        # 转换为 SearchHit
        from app.rag.retriever import _to_search_hit

        hits = [_to_search_hit(p) for p in points]
        evaluation = evaluate_case(case, hits, latency_ms=latency_ms)
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
            }
        )

    metrics = aggregate_metrics(cases, evaluations, ks=_metric_ks(top_k))
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "retrieval_method": strategy,
        "embedding_model": settings.dashscope_embedding_model,
        "top_k": top_k,
        "split": "all",
        "metrics": metrics,
        "cases": case_results,
    }


async def index_corpus(settings: Settings, cases: list) -> None:
    """索引评估语料。"""
    from evals.run_retrieval import index_corpus as _index

    await _index(settings, cases)


def print_metrics(metrics: dict) -> None:
    """打印主要指标。"""
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


def _metric_ks(top_k: int) -> tuple[int, ...]:
    return tuple(k for k in (1, 3, 5, 8) if k <= top_k) or (top_k,)


def _to_sparse_vector(sparse_values: list[SparseValue]) -> models.SparseVector:
    """将 SparseValue 列表转换为 Qdrant SparseVector。"""
    if not sparse_values:
        return models.SparseVector(indices=[], values=[])
    sorted_values = sorted(sparse_values, key=lambda v: v.index)
    return models.SparseVector(
        indices=[v.index for v in sorted_values],
        values=[v.value for v in sorted_values],
    )


if __name__ == "__main__":
    asyncio.run(main())
