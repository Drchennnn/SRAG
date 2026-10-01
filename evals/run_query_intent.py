"""离线比较显式 contextual 模式的规则判断与 DeepSeek 追问分类。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from app.clients.deepseek import DeepSeekClient
from app.config import get_settings
from app.rag.retriever import HybridRetriever
from app.schemas.rag import SearchRequest, SearchScope
from app.stores.sqlite import MetadataStore
from evals.run_query_followup import TOP_K, load_cases, source_rank

DEFAULT_CASES = Path("evals/query_notebook_holdout_cases.jsonl")
DEFAULT_BASELINE = Path("evals/results/query_notebook_holdout_current_rule.json")
DEFAULT_OUTPUT = Path("evals/results/query_intent_holdout.json")


def summarize(rows: list[dict]) -> dict:
    followups = [row for row in rows if row["kind"] == "followup"]
    independent = [row for row in rows if row["kind"] == "independent"]
    if not followups or not independent:
        raise ValueError("必须同时包含追问与独立问题")

    def recall(group: list[dict], key: str) -> float:
        return sum(row[key] is not None and row[key] <= 5 for row in group) / len(group)

    return {
        "followup_count": len(followups),
        "independent_count": len(independent),
        "rule_missed_followup": sum(row["rule_method"] == "raw" for row in followups),
        "model_missed_followup": sum(row["model_method"] == "raw" for row in followups),
        "rule_false_expansion": sum(
            row["rule_method"] == "history_expanded" for row in independent
        ),
        "model_false_expansion": sum(
            row["model_method"] == "history_expanded" for row in independent
        ),
        "rule_followup_recall@5": recall(followups, "rule_rank"),
        "model_followup_recall@5": recall(followups, "model_rank"),
        "rule_independent_recall@5": recall(independent, "rule_rank"),
        "model_independent_recall@5": recall(independent, "model_rank"),
        "model_independent_harmed@5": sum(
            row["raw_rank"] is not None
            and row["raw_rank"] <= 5
            and (row["model_rank"] is None or row["model_rank"] > 5)
            for row in independent
        ),
        "decision_changes": sum(row["rule_method"] != row["model_method"] for row in rows),
        "deepseek_total_tokens": sum(row["usage"].get("total_tokens", 0) for row in rows),
        "deepseek_retries": sum(row.get("attempts", 1) - 1 for row in rows),
        "mean_deepseek_latency_ms": round(
            sum(row.get("deepseek_latency_ms", 0) for row in rows) / len(rows)
        ),
    }


async def evaluate(cases_path: Path, baseline_path: Path, output_path: Path) -> dict:
    cases = load_cases(cases_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    case_sha = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    if baseline.get("case_file_sha256") != case_sha:
        raise ValueError("基线报告与题集哈希不一致")
    baseline_rows = {row["case_id"]: row for row in baseline["cases"]}
    if set(baseline_rows) != {case.case_id for case in cases}:
        raise ValueError("基线报告与题集 case_id 不一致")

    settings = get_settings()
    metadata = MetadataStore(settings)
    visible = {
        item.document_id: item.version
        for course_id in {case.course_id for case in cases}
        for item in metadata.visible_versions(course_ids=[course_id])
    }
    if baseline.get("corpus_versions") != visible:
        raise ValueError("索引可见版本与基线报告不一致")

    client = DeepSeekClient(settings)
    retriever = HybridRetriever.from_settings(settings)
    rows: list[dict] = []
    try:
        for number, case in enumerate(cases, start=1):
            previous_user = next(
                (turn.content.strip() for turn in reversed(case.history) if turn.role == "user"),
                "",
            )
            if not previous_user:
                raise ValueError(f"{case.case_id} 缺少上一条用户问题")
            started = time.perf_counter()
            decision = await client.classify_context_need(previous_user, case.query)
            latency_ms = round((time.perf_counter() - started) * 1000)
            model_method = "history_expanded" if decision["needs_context"] else "raw"
            rule = baseline_rows[case.case_id]
            if model_method == "raw":
                model_rank = rule["raw_rank"]
            elif rule["query_method"] == "history_expanded":
                model_rank = rule["contextual_rank"]
            else:
                expanded = await retriever.search(
                    SearchRequest(
                        query=f"{previous_user}\n追问：{case.query}",
                        scope=SearchScope(course_ids=[case.course_id]),
                        top_k=TOP_K,
                        retrieval_mode="text",
                        query_construction="raw",
                    )
                )
                model_rank = source_rank(expanded.hits, case)
            row = {
                "case_id": case.case_id,
                "kind": case.kind,
                "rule_method": rule["query_method"],
                "model_method": model_method,
                "raw_rank": rule["raw_rank"],
                "rule_rank": rule["contextual_rank"],
                "model_rank": model_rank,
                "reason": decision["reason"],
                "usage": decision["usage"],
                "attempts": decision["attempts"],
                "deepseek_latency_ms": latency_ms,
            }
            rows.append(row)
            print(
                f"{number}/{len(cases)} {case.case_id}: "
                f"rule={row['rule_method']} model={model_method} rank={model_rank or '-'}",
                flush=True,
            )
    finally:
        await retriever.vector_store.close()

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "case_file_sha256": case_sha,
        "baseline_report": baseline_path.as_posix(),
        "corpus_versions": visible,
        "deepseek_model": settings.deepseek_query_model,
        "text_embedding_model": settings.dashscope_embedding_model,
        "top_k": TOP_K,
        "metrics": summarize(rows),
        "cases": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = asyncio.run(evaluate(args.cases, args.baseline, args.output))
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"报告：{args.output}")


if __name__ == "__main__":
    main()
