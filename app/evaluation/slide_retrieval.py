"""固定页码题集：比较幻灯片 dense、文本混合、视觉和三路融合检索。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from pydantic import BaseModel, Field
from qdrant_client import models

from app.clients.dashscope import DashScopeClient
from app.config import get_settings
from app.rag.fusion import visual_query_intent
from app.schemas.embedding import SparseValue
from app.stores.qdrant import QdrantStore
from app.stores.sqlite import MetadataStore


class SlideCase(BaseModel):
    case_id: str
    query: str = Field(min_length=1)
    document_id: str
    pages: list[int] = Field(min_length=1)
    tag: str


def load_cases(path: Path) -> list[SlideCase]:
    cases = [
        SlideCase.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("题集为空或存在重复 case_id")
    return cases


def rank_pages(*lists: list[models.ScoredPoint], k: int = 60) -> list[tuple[str, int]]:
    """按 document_id+页码合并表示，避免同页文本与视觉结果重复。"""

    scores: dict[tuple[str, int], float] = defaultdict(float)
    for points in lists:
        seen: set[tuple[str, int]] = set()
        for rank, point in enumerate(points, start=1):
            payload = point.payload or {}
            page = payload.get("page")
            document_id = payload.get("document_id")
            if not isinstance(page, int) or not isinstance(document_id, str):
                continue
            key = (document_id, page)
            if key not in seen:
                scores[key] += 1 / (k + rank)
                seen.add(key)
    return sorted(scores, key=lambda key: (-scores[key], key))


def summarize(rows: list[dict], method: str) -> dict:
    ranks = [row["ranks"][method] for row in rows]
    result = {
        f"recall@{cutoff}": sum(rank is not None and rank <= cutoff for rank in ranks) / len(ranks)
        for cutoff in (1, 3, 5, 8)
    }
    result["mrr@8"] = sum(1 / rank for rank in ranks if rank is not None and rank <= 8) / len(ranks)
    result["mean_case_latency_ms"] = round(fmean(row["latency_ms"] for row in rows), 1)
    result["by_tag_recall@5"] = {
        tag: sum(
            row["ranks"][method] is not None and row["ranks"][method] <= 5
            for row in rows
            if row["tag"] == tag
        )
        / sum(row["tag"] == tag for row in rows)
        for tag in sorted({row["tag"] for row in rows})
    }
    return result


def _sparse_vector(values: list[SparseValue]) -> models.SparseVector:
    ordered = sorted(values, key=lambda value: value.index)
    return models.SparseVector(
        indices=[value.index for value in ordered], values=[value.value for value in ordered]
    )


async def evaluate(cases_path: Path, output_path: Path, *, course_id: str = "CS5187") -> dict:
    settings = get_settings()
    cases = load_cases(cases_path)
    metadata = MetadataStore(settings)
    metadata.initialize()
    visible = [
        (item.document_id, item.version)
        for item in metadata.visible_versions(course_ids=[course_id])
    ]
    missing = {case.document_id for case in cases} - {item[0] for item in visible}
    if missing:
        raise ValueError(f"题集文档未索引或未解封：{sorted(missing)}")
    store = QdrantStore(settings)
    embedding = DashScopeClient(settings)
    rows: list[dict] = []
    try:
        for index, case in enumerate(cases, start=1):
            started = time.perf_counter()
            text_query = await embedding.embed_query(case.query)
            visual_query = await embedding.embed_visual_query(case.query)
            dense = await store.search_text_dense(
                text_query.dense, limit=50, course_ids=[course_id], visible_versions=visible
            )
            sparse = (
                await store.search_text_sparse(
                    _sparse_vector(text_query.sparse),
                    limit=50,
                    course_ids=[course_id],
                    visible_versions=visible,
                )
                if text_query.sparse
                else []
            )
            visual = await store.search_visual_fusion(
                visual_query.vectors[0].dense,
                limit=50,
                course_ids=[course_id],
                visible_versions=visible,
            )
            ranked = {
                "dense": rank_pages(dense),
                "text_rrf": rank_pages(dense, sparse),
                "visual": rank_pages(visual),
                "fusion_rrf": rank_pages(dense, sparse, visual),
            }
            routed_to_visual = visual_query_intent(case.query)
            ranked["auto_route"] = ranked["visual" if routed_to_visual else "text_rrf"]
            expected = {(case.document_id, page) for page in case.pages}
            ranks = {
                method: next(
                    (rank for rank, page in enumerate(pages, start=1) if page in expected), None
                )
                for method, pages in ranked.items()
            }
            rows.append(
                {
                    "case_id": case.case_id,
                    "tag": case.tag,
                    "auto_mode": "visual" if routed_to_visual else "text",
                    "query": case.query,
                    "expected": sorted(case.pages),
                    "ranks": ranks,
                    "top5": {
                        method: [{"document_id": doc, "page": page} for doc, page in pages[:5]]
                        for method, pages in ranked.items()
                    },
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                }
            )
            print(
                f"{index}/{len(cases)} {case.case_id}: "
                + " ".join(f"{name}={rank or '-'}" for name, rank in ranks.items()),
                flush=True,
            )
    finally:
        await store.close()
    report = {
        "course_id": course_id,
        "case_count": len(cases),
        "case_file_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "corpus_versions": visible,
        "models": {
            "text": settings.dashscope_embedding_model,
            "visual": settings.dashscope_visual_embedding_model,
        },
        "metrics": {
            method: summarize(rows, method)
            for method in ("dense", "text_rrf", "visual", "fusion_rrf", "auto_route")
        },
        "cases": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="运行课程幻灯片页码召回评估")
    parser.add_argument("--course-id", default="CS5187", help="限定评估的课程")
    parser.add_argument("--cases", type=Path, default=Path("evals/slide_cases.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("evals/results/slide_development_visual_route.json")
    )
    args = parser.parse_args()
    report = asyncio.run(evaluate(args.cases, args.output, course_id=args.course_id))
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"报告：{args.output}")


if __name__ == "__main__":
    main()
