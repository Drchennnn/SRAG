"""对固定追问题集比较原文与近期上下文构造后的正式检索结果。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.config import get_settings
from app.rag.retriever import HybridRetriever
from app.schemas.rag import ConversationTurn, SearchHit, SearchRequest, SearchScope
from app.stores.sqlite import MetadataStore

DEFAULT_CASES = Path("evals/query_notebook_holdout_cases.jsonl")
DEFAULT_OUTPUT = Path("data/evals/query_notebook_holdout_current_rule.json")
TOP_K = 8


class FollowupCase(BaseModel):
    case_id: str
    kind: Literal["followup", "independent"]
    course_id: str
    history: list[ConversationTurn] = Field(min_length=1, max_length=6)
    query: str
    document_id: str
    pages: list[int] = Field(default_factory=list)
    cell_ids: list[str] = Field(default_factory=list)
    evidence: str

    @model_validator(mode="after")
    def require_one_source_type(self) -> FollowupCase:
        if bool(self.pages) == bool(self.cell_ids):
            raise ValueError("每题必须且只能标注页码或 cell_ids")
        return self


def load_cases(path: Path) -> list[FollowupCase]:
    cases = [
        FollowupCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("题集为空或存在重复 case_id")
    if {case.kind for case in cases} != {"followup", "independent"}:
        raise ValueError("题集必须同时包含追问与独立问题")
    return cases


def source_rank(hits: list[SearchHit], case: FollowupCase) -> int | None:
    return next(
        (
            rank
            for rank, hit in enumerate(hits, start=1)
            if hit.source.document_id == case.document_id
            and (
                hit.source.page in case.pages
                if case.pages
                else bool(set(hit.source.cell_ids) & set(case.cell_ids))
                or hit.source.cell_id in case.cell_ids
            )
        ),
        None,
    )


def recall(rows: list[dict], key: str, cutoff: int) -> float:
    return sum(row[key] is not None and row[key] <= cutoff for row in rows) / len(rows)


def summarize(rows: list[dict]) -> dict:
    followups = [row for row in rows if row["kind"] == "followup"]
    independent = [row for row in rows if row["kind"] == "independent"]
    if not followups or not independent:
        raise ValueError("评估结果必须同时包含追问与独立问题")
    return {
        "followup_count": len(followups),
        "followup_expansion_rate": sum(
            row["query_method"] == "history_expanded" for row in followups
        )
        / len(followups),
        "followup_raw_recall@5": recall(followups, "raw_rank", 5),
        "followup_contextual_recall@5": recall(followups, "contextual_rank", 5),
        "followup_gained@5": sum(
            (row["raw_rank"] is None or row["raw_rank"] > 5)
            and row["contextual_rank"] is not None
            and row["contextual_rank"] <= 5
            for row in followups
        ),
        "followup_lost@5": sum(
            row["raw_rank"] is not None
            and row["raw_rank"] <= 5
            and (row["contextual_rank"] is None or row["contextual_rank"] > 5)
            for row in followups
        ),
        "independent_count": len(independent),
        "false_expansion_rate": sum(
            row["query_method"] == "history_expanded" for row in independent
        )
        / len(independent),
        "independent_raw_recall@5": recall(independent, "raw_rank", 5),
        "independent_contextual_recall@5": recall(independent, "contextual_rank", 5),
        "independent_harmed@5": sum(
            row["raw_rank"] is not None
            and row["raw_rank"] <= 5
            and (row["contextual_rank"] is None or row["contextual_rank"] > 5)
            for row in independent
        ),
    }


async def evaluate(cases_path: Path, output_path: Path, *, oracle_missed: bool = False) -> dict:
    cases = load_cases(cases_path)
    settings = get_settings()
    metadata = MetadataStore(settings)
    metadata.initialize()
    visible = {
        item.document_id: item.version
        for course_id in {case.course_id for case in cases}
        for item in metadata.visible_versions(course_ids=[course_id])
    }
    missing = {case.document_id for case in cases} - visible.keys()
    if missing:
        raise ValueError(f"题集文档未索引或未解封：{sorted(missing)}")

    retriever = HybridRetriever.from_settings(settings)
    rows: list[dict] = []
    try:
        for number, case in enumerate(cases, start=1):
            scope = SearchScope(course_ids=[case.course_id])
            started = time.perf_counter()
            raw = await retriever.search(
                SearchRequest(
                    query=case.query,
                    scope=scope,
                    top_k=TOP_K,
                    retrieval_mode="text",
                    query_construction="raw",
                )
            )
            raw_latency_ms = round((time.perf_counter() - started) * 1000)
            started = time.perf_counter()
            contextual = await retriever.search(
                SearchRequest(
                    query=case.query,
                    scope=scope,
                    top_k=TOP_K,
                    retrieval_mode="text",
                    query_construction="contextual",
                    history=case.history,
                )
            )
            contextual_latency_ms = round((time.perf_counter() - started) * 1000)
            oracle_expanded_rank = None
            if oracle_missed and case.kind == "followup" and contextual.query_method == "raw":
                previous_user = next(
                    (
                        turn.content.strip()
                        for turn in reversed(case.history)
                        if turn.role == "user"
                    ),
                    "",
                )
                if previous_user:
                    oracle = await retriever.search(
                        SearchRequest(
                            query=f"{previous_user}\n追问：{case.query}",
                            scope=scope,
                            top_k=TOP_K,
                            retrieval_mode="text",
                            query_construction="raw",
                        )
                    )
                    oracle_expanded_rank = source_rank(oracle.hits, case)
            row = {
                "case_id": case.case_id,
                "kind": case.kind,
                "query": case.query,
                "history": [turn.model_dump() for turn in case.history],
                "expected_document_id": case.document_id,
                "expected_pages": case.pages,
                "expected_cell_ids": case.cell_ids,
                "raw_rank": source_rank(raw.hits, case),
                "contextual_rank": source_rank(contextual.hits, case),
                "oracle_expanded_rank": oracle_expanded_rank,
                "query_method": contextual.query_method,
                "retrieval_query": contextual.retrieval_query,
                "raw_top5": [[hit.source.document_id, hit.source.page] for hit in raw.hits[:5]],
                "contextual_top5": [
                    [hit.source.document_id, hit.source.page] for hit in contextual.hits[:5]
                ],
                "raw_top5_cells": [hit.source.cell_ids for hit in raw.hits[:5]],
                "contextual_top5_cells": [hit.source.cell_ids for hit in contextual.hits[:5]],
                "raw_latency_ms": raw_latency_ms,
                "contextual_latency_ms": contextual_latency_ms,
            }
            rows.append(row)
            print(
                f"{number}/{len(cases)} {case.case_id}: "
                f"raw={row['raw_rank'] or '-'} contextual={row['contextual_rank'] or '-'}",
                flush=True,
            )
    finally:
        await retriever.vector_store.close()

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "case_file_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "corpus_versions": visible,
        "text_embedding_model": settings.dashscope_embedding_model,
        "retrieval_mode": "text",
        "top_k": TOP_K,
        "oracle_missed": oracle_missed,
        "metrics": summarize(rows),
        "cases": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--oracle-missed", action="store_true")
    args = parser.parse_args()
    report = asyncio.run(evaluate(args.cases, args.output, oracle_missed=args.oracle_missed))
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"报告：{args.output}")


if __name__ == "__main__":
    main()
