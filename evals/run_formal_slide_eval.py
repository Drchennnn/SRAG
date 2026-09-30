"""Evaluate page recall through the same retriever used by /api/rag/search."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

from app.config import get_settings
from app.evaluation.slide_retrieval import load_cases, summarize
from app.rag.retriever import HybridRetriever
from app.schemas.rag import SearchRequest, SearchScope


async def evaluate(cases_path: Path, output_path: Path, course_id: str | None) -> dict:
    cases = load_cases(cases_path)
    retriever = HybridRetriever.from_settings(get_settings())
    rows: list[dict] = []
    try:
        for index, case in enumerate(cases, 1):
            started = time.perf_counter()
            result = await retriever.search(
                SearchRequest(
                    query=case.query,
                    scope=SearchScope(course_ids=[course_id] if course_id else []),
                    top_k=8,
                    retrieval_mode="auto",
                )
            )
            hits = [(hit.source.document_id, hit.source.page) for hit in result.hits]
            expected = {(case.document_id, page) for page in case.pages}
            rank = next(
                (position for position, hit in enumerate(hits, 1) if hit in expected), None
            )
            rows.append(
                {
                    "case_id": case.case_id,
                    "tag": case.tag,
                    "query": case.query,
                    "expected_document_id": case.document_id,
                    "expected_pages": case.pages,
                    "mode_used": result.mode_used,
                    "ranks": {"formal": rank},
                    "top5": [
                        {"document_id": document_id, "page": page}
                        for document_id, page in hits[:5]
                    ],
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                }
            )
            print(f"{index}/{len(cases)} {case.case_id}: rank={rank or '-'}", flush=True)
    finally:
        await retriever.vector_store.close()
    report = {
        "course_id": course_id or "all",
        "case_count": len(cases),
        "case_file_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "metric": summarize(rows, "formal"),
        "cases": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="评估正式检索入口的页码召回")
    parser.add_argument("--course-id", help="可选；不指定时检索全部课程")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(evaluate(args.cases, args.output, args.course_id))
    print(json.dumps(report["metric"], ensure_ascii=False, indent=2))
    print(f"报告：{args.output}")


if __name__ == "__main__":
    main()
