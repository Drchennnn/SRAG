"""运行冻结的跨文件终端问答验收集，并保存可复核的逐轮证据。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
import time
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from typing import Any

from app.clients.dashscope import DashScopeClient
from app.clients.deepseek import DeepSeekClient
from app.config import Settings, get_settings
from app.rag.chat import ChatResult, ChatSession
from app.rag.retriever import HybridRetriever
from app.rag.summary import DocumentSummarizer
from app.rag.vision import VisualTextReader
from app.schemas.rag import DocumentSummary, SearchHit, SearchScope
from app.stores.qdrant import QdrantStore
from app.stores.sqlite import MetadataStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "evals" / "chat_acceptance_cases.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evals" / "chat_acceptance_report.json"
_INDEX_TABLES = ("document_versions", "sections", "content_blocks", "assets", "chunks")


class NoWriteVisualStore:
    """评测时记录派生点发布意图，但不改变待测 Qdrant 索引。"""

    def __init__(self) -> None:
        self.attempts: list[dict[str, Any]] = []

    async def upsert_visual_text(
        self,
        document: DocumentSummary,
        hit: SearchHit,
        vector: Any,
        *,
        cache_key: str,
        embedding_model: str,
    ) -> None:
        self.attempts.append(
            {
                "document_id": document.document_id,
                "version": document.version,
                "page": hit.source.page,
                "origin_chunk_id": hit.origin_chunk_id,
                "cache_key": cache_key,
                "embedding_model": embedding_model,
            }
        )


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_suite(path: Path) -> dict[str, Any]:
    suite = json.loads(path.read_text(encoding="utf-8"))
    if suite.get("schema_version") != 1 or not isinstance(suite.get("sessions"), list):
        raise ValueError("验收题集 schema_version 或 sessions 无效")
    sessions = suite["sessions"]
    session_ids = [item.get("session_id") for item in sessions]
    if (
        not sessions
        or len(session_ids) != len(set(session_ids))
        or any(not item for item in session_ids)
    ):
        raise ValueError("验收题集 session_id 为空或重复")
    turns = [turn for session in sessions for turn in session.get("turns", [])]
    if not turns:
        raise ValueError("验收集必须包含问题")
    turn_ids = [turn.get("turn_id") for turn in turns]
    if len(turn_ids) != len(set(turn_ids)) or any(not item for item in turn_ids):
        raise ValueError("验收题集 turn_id 为空或重复")
    for session in sessions:
        SearchScope.model_validate(session.get("initial_scope", {}))
        if not session.get("turns"):
            raise ValueError(f"会话 {session['session_id']} 没有问题")
        for turn in session["turns"]:
            if not str(turn.get("query", "")).strip() or not isinstance(turn.get("expected"), dict):
                raise ValueError(f"题目 {turn.get('turn_id')} 缺少 query 或 expected")
    return suite


def _validate_frozen_sources(suite: dict[str, Any], documents: list[DocumentSummary]) -> None:
    visible = {(item.document_id, item.version) for item in documents}
    expected = {
        (source["document_id"], source["version"])
        for session in suite["sessions"]
        for turn in session["turns"]
        for group in turn["expected"].get("source_groups", [])
        for source in group.get("any_of", [])
    }
    missing = expected - visible
    if missing:
        raise ValueError(f"冻结来源与当前可见文档版本不一致：{sorted(missing)}")


def _source_snapshot(documents: list[DocumentSummary]) -> dict[str, Any]:
    records = []
    for document in sorted(documents, key=lambda item: item.document_id):
        path = Path(document.source_path).expanduser()
        records.append(
            {
                "course_id": document.course_id,
                "document_id": document.document_id,
                "version": document.version,
                "source_path": document.source_path,
                "source_exists": path.is_file(),
                "source_sha256": _file_sha256(path) if path.is_file() else None,
                "indexed_at": document.indexed_at.isoformat(),
            }
        )
    return {
        "documents": records,
        "sha256": hashlib.sha256(_canonical_bytes(records)).hexdigest(),
    }


def _sqlite_index_snapshot(settings: Settings, documents: list[DocumentSummary]) -> dict[str, Any]:
    digest = hashlib.sha256()
    row_count = 0
    database = settings.sqlite_path.expanduser().resolve()
    with sqlite3.connect(database, timeout=30) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        for document in sorted(documents, key=lambda item: item.document_id):
            document_rows = connection.execute(
                "SELECT * FROM documents WHERE document_id=? ORDER BY document_id",
                (document.document_id,),
            ).fetchall()
            for row in document_rows:
                digest.update(_canonical_bytes(["documents", dict(row)]))
                row_count += 1
            for table in _INDEX_TABLES:
                rows = connection.execute(
                    f"SELECT * FROM {table} WHERE document_id=? AND version=? ORDER BY rowid",
                    (document.document_id, document.version),
                ).fetchall()
                for row in rows:
                    digest.update(_canonical_bytes([table, dict(row)]))
                    row_count += 1
        connection.rollback()
    return {
        "path": str(database),
        "logical_row_count": row_count,
        "logical_sha256": digest.hexdigest(),
    }


def _cache_snapshot(cache_dir: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = []
    if cache_dir.is_dir():
        for path in sorted(item for item in cache_dir.rglob("*") if item.is_file()):
            relative = path.relative_to(cache_dir).as_posix()
            file_hash = _file_sha256(path)
            files.append({"path": relative, "bytes": path.stat().st_size, "sha256": file_hash})
            digest.update(_canonical_bytes(files[-1]))
    return {
        "directory": str(cache_dir.resolve()),
        "file_count": len(files),
        "sha256": digest.hexdigest(),
        "files": files,
    }


async def _qdrant_snapshot(store: QdrantStore) -> dict[str, Any]:
    info = await store.client.get_collection(store.collection)
    records: list[tuple[str, str]] = []
    offset = None
    while True:
        page, offset = await store.client.scroll(
            collection_name=store.collection,
            limit=128,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for point in page:
            point_body = {
                "id": str(point.id),
                "payload": point.payload,
                "vector": point.vector,
            }
            records.append(
                (str(point.id), hashlib.sha256(_canonical_bytes(point_body)).hexdigest())
            )
        if offset is None:
            break
    records.sort()
    fingerprint = hashlib.sha256(_canonical_bytes(records)).hexdigest()
    return {
        "url": store.url,
        "collection": store.collection,
        "status": str(info.status),
        "reported_points_count": info.points_count,
        "scrolled_points_count": len(records),
        "points_sha256": fingerprint,
        "schema": _jsonable(info.config.params),
    }


async def _environment_snapshot(
    settings: Settings,
    documents: list[DocumentSummary],
    vector_store: QdrantStore,
) -> dict[str, Any]:
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "sources": _source_snapshot(documents),
        "sqlite_index": _sqlite_index_snapshot(settings, documents),
        "qdrant_index": await _qdrant_snapshot(vector_store),
        "visual_text_cache": _cache_snapshot(settings.data_dir / "visual_text"),
    }


def _hit_record(hit: SearchHit) -> dict[str, Any]:
    return {
        "course_id": hit.source.course_id,
        "document_id": hit.source.document_id,
        "version": hit.source.version,
        "file": hit.source.file,
        "page": hit.source.page,
        "cell_ids": hit.source.cell_ids,
        "section_path": hit.source.section_path,
        "chunk_id": hit.chunk_id,
        "origin_chunk_id": hit.origin_chunk_id,
        "representation_type": hit.representation_type,
        "evidence_origin": hit.evidence_origin,
        "score": hit.score,
    }


def _matches_source(hit: SearchHit, source: dict[str, Any]) -> bool:
    if hit.source.document_id != source["document_id"]:
        return False
    if hit.source.version != source["version"]:
        return False
    pages = source.get("pages", [])
    if pages and hit.source.page not in pages:
        return False
    cell_ids = set(source.get("cell_ids", []))
    return not cell_ids or bool(cell_ids.intersection(hit.source.cell_ids))


def _evaluate_expectations(result: ChatResult, expected: dict[str, Any]) -> dict[str, Any]:
    invalid_citations = [
        number for number in result.cited_numbers if not 1 <= number <= len(result.evidence)
    ]
    source_groups = []
    for group in expected.get("source_groups", []):
        matched = [
            _hit_record(hit)
            for hit in result.evidence
            if any(_matches_source(hit, source) for source in group.get("any_of", []))
        ]
        source_groups.append(
            {"label": group.get("label", ""), "recalled": bool(matched), "matches": matched}
        )
    query_methods = expected.get("query_methods")
    checks = {
        "route": result.route in expected.get("routes", []),
        "retrieval_performed": result.retrieval_performed == expected.get("retrieval_performed"),
        "model_called": result.model_called == expected.get("model_called"),
        "query_method": query_methods is None or result.search.query_method in query_methods,
        "all_source_groups_recalled": all(group["recalled"] for group in source_groups),
        "citation_numbers_legal": not invalid_citations,
        "required_citation_present": not expected.get("require_citations", False)
        or bool(result.cited_numbers),
        "answer_contains": all(
            text.casefold() in result.answer.casefold()
            for text in expected.get("answer_contains", [])
        ),
    }
    applicable_checks = ["citation_numbers_legal"]
    if expected.get("routes"):
        applicable_checks.append("route")
    if "retrieval_performed" in expected:
        applicable_checks.append("retrieval_performed")
    if "model_called" in expected:
        applicable_checks.append("model_called")
    if query_methods is not None:
        applicable_checks.append("query_method")
    if expected.get("source_groups"):
        applicable_checks.append("all_source_groups_recalled")
    if expected.get("require_citations", False):
        applicable_checks.append("required_citation_present")
    if expected.get("answer_contains"):
        applicable_checks.append("answer_contains")
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "applicable_checks": applicable_checks,
        "source_groups": source_groups,
        "invalid_citations": invalid_citations,
    }


def _turn_record(
    session_id: str,
    turn: dict[str, Any],
    session: ChatSession,
    result: ChatResult,
    latency_ms: float,
) -> dict[str, Any]:
    evaluation = _evaluate_expectations(result, turn["expected"])
    evaluation["checks"]["scope_respected"] = all(
        (not session.scope.course_ids or hit.source.course_id in session.scope.course_ids)
        and (not session.scope.document_ids or hit.source.document_id in session.scope.document_ids)
        and (
            not result.effective_scope
            or not result.effective_scope.course_ids
            or hit.source.course_id in result.effective_scope.course_ids
        )
        and (
            not result.effective_scope
            or not result.effective_scope.document_ids
            or hit.source.document_id in result.effective_scope.document_ids
        )
        for hit in result.evidence
    )
    evaluation["applicable_checks"].append("scope_respected")
    if result.coverage:
        evaluation["checks"]["summary_read_coverage"] = (
            result.coverage["read_chunks"] == result.coverage["total_chunks"]
        )
        evaluation["applicable_checks"].append("summary_read_coverage")
    evaluation["passed"] = all(evaluation["checks"].values())
    return {
        "session_id": session_id,
        "turn_id": turn["turn_id"],
        "tags": turn.get("tags", []),
        "query": turn["query"],
        "answer": result.answer,
        "status": "passed" if evaluation["passed"] else "failed",
        "route": result.route,
        "retrieval_performed": result.retrieval_performed,
        "model_called": result.model_called,
        "query_method": result.search.query_method,
        "retrieval_mode": result.search.mode_used,
        "retrieval_query": result.search.retrieval_query,
        "requested_scope": session.scope.model_dump(),
        "effective_scope": (
            result.effective_scope.model_dump() if result.effective_scope else None
        ),
        "search_sources": [_hit_record(hit) for hit in result.search.hits],
        "evidence_sources": [_hit_record(hit) for hit in result.evidence],
        "coverage": result.coverage,
        "cited_numbers": result.cited_numbers,
        "citation_warning": result.citation_warning,
        "reading_warnings": list(result.reading_warnings),
        "latency_ms": round(latency_ms, 3),
        "model_tokens": result.model_tokens,
        "expected": turn["expected"],
        "mechanical_evaluation": evaluation,
    }


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    """返回 nearest-rank 分位数；空样本返回 None。"""

    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _summarize(
    rows: list[dict[str, Any]], expected_turn_count: int | None = None
) -> dict[str, Any]:
    expected_turn_count = len(rows) if expected_turn_count is None else expected_turn_count
    completed = [row for row in rows if row.get("status") != "error"]
    evaluated = [row for row in completed if "mechanical_evaluation" in row]
    groups = [group for row in evaluated for group in row["mechanical_evaluation"]["source_groups"]]
    check_names = sorted(
        {
            check
            for row in evaluated
            for check in row["mechanical_evaluation"].get(
                "applicable_checks", row["mechanical_evaluation"]["checks"]
            )
        }
    )
    checks = {}
    for check in check_names:
        applicable = [
            row
            for row in evaluated
            if check
            in row["mechanical_evaluation"].get(
                "applicable_checks", row["mechanical_evaluation"]["checks"]
            )
        ]
        passed = sum(bool(row["mechanical_evaluation"]["checks"].get(check)) for row in applicable)
        checks[check] = {
            "applicable_turns": len(applicable),
            "passed_turns": passed,
            "pass_rate": passed / len(applicable) if applicable else None,
        }
    latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    mechanically_passed = sum(bool(row["mechanical_evaluation"]["passed"]) for row in evaluated)
    return {
        "turn_count": expected_turn_count,
        "attempted_turns": len(rows),
        "completed_turns": len(completed),
        "completion_rate": (len(completed) / expected_turn_count if expected_turn_count else None),
        "mechanically_evaluated_turns": len(evaluated),
        "mechanically_passed_turns": mechanically_passed,
        "mechanical_pass_rate": (mechanically_passed / len(rows) if rows else None),
        "passed_turns": sum(row.get("status") == "passed" for row in rows),
        "failed_turns": sum(row.get("status") in {"failed", "error"} for row in rows),
        "checks": checks,
        "source_group_count": len(groups),
        "source_groups_recalled": sum(group["recalled"] for group in groups),
        "source_group_recall": (
            sum(group["recalled"] for group in groups) / len(groups) if groups else None
        ),
        "citation_legal_turns": sum(
            row["mechanical_evaluation"]["checks"]["citation_numbers_legal"] for row in evaluated
        ),
        "total_latency_ms": round(sum(row.get("latency_ms", 0) for row in rows), 3),
        "latency_turns": len(latencies),
        "latency_percentile_method": "nearest_rank",
        "latency_p50_ms": _nearest_rank(latencies, 0.50),
        "latency_p95_ms": _nearest_rank(latencies, 0.95),
        "total_model_tokens": sum(row.get("model_tokens", 0) for row in rows),
        "metric_boundary": (
            "完成率以冻结题集总轮数为分母；逐轮机械通过率以已尝试轮数为分母，异常计未通过；"
            "check 通过率只统计该 check 适用的轮次。以上均不是回答事实准确率。"
        ),
    }


def _automatic_pass_gates(
    *,
    execution_succeeded: bool,
    attempted_turns: int,
    expected_turns: int,
    failed_turns: int,
    environment_changed: dict[str, bool],
) -> dict[str, bool]:
    """分开报告逐轮机械验收与冻结环境约束。"""

    overall_passed = execution_succeeded and attempted_turns == expected_turns and failed_turns == 0
    required = {"sources", "sqlite_index", "qdrant_index", "visual_text_cache"}
    frozen_environment_passed = required <= environment_changed.keys() and all(
        changed is False for changed in environment_changed.values()
    )
    return {
        "overall_passed": overall_passed,
        "frozen_environment_passed": frozen_environment_passed,
        "stage_automatic_passed": overall_passed and frozen_environment_passed,
    }


async def evaluate(
    cases_path: Path,
    output_path: Path,
    *,
    top_k: int = 6,
    retrieval_mode: str = "text",
) -> dict[str, Any]:
    suite = load_suite(cases_path)
    settings = get_settings()
    if not settings.dashscope_api_key:
        raise RuntimeError("尚未在 .env 中配置 DASHSCOPE_API_KEY")
    answer_provider = DeepSeekClient(settings)
    answer_reviewer = DeepSeekClient(settings, answer_thinking=True)
    answer_provider.require_api_key()
    catalog = MetadataStore(settings)
    catalog.initialize()
    documents = catalog.list_documents()
    _validate_frozen_sources(suite, documents)

    retriever = HybridRetriever.from_settings(settings)
    visual_store = QdrantStore(settings)
    visual_publish = NoWriteVisualStore()
    dashscope = DashScopeClient(settings)
    before = await _environment_snapshot(settings, documents, visual_store)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "run_status": "running",
                "overall_passed": False,
                "frozen_environment_passed": False,
                "stage_automatic_passed": False,
                "suite_id": suite["suite_id"],
                "case_file_sha256": _file_sha256(cases_path),
                "environment_before": before,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    execution_error: Exception | None = None
    expected_turn_count = sum(len(item["turns"]) for item in suite["sessions"])
    try:
        for session_case in suite["sessions"]:
            session = ChatSession(
                retriever,
                answer_provider,
                catalog,
                scope=SearchScope.model_validate(session_case["initial_scope"]),
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                query_construction="contextual",
                summarizer=DocumentSummarizer(catalog, answer_provider, reviewer=answer_reviewer),
                answer_reviewer=answer_reviewer,
                visual_reader=VisualTextReader(
                    catalog, dashscope, visual_publish, settings.data_dir / "visual_text"
                ),
            )
            for turn in session_case["turns"]:
                started = time.perf_counter()
                try:
                    result = await session.ask(turn["query"])
                except Exception as error:
                    rows.append(
                        {
                            "session_id": session_case["session_id"],
                            "turn_id": turn["turn_id"],
                            "tags": turn.get("tags", []),
                            "query": turn["query"],
                            "status": "error",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                            "expected": turn["expected"],
                        }
                    )
                    execution_error = error
                    break
                rows.append(
                    _turn_record(
                        session_case["session_id"],
                        turn,
                        session,
                        result,
                        (time.perf_counter() - started) * 1000,
                    )
                )
                print(
                    f"{len(rows)}/{expected_turn_count} {turn['turn_id']}: {rows[-1]['status']}",
                    flush=True,
                )
            if execution_error is not None:
                break
    finally:
        after = await _environment_snapshot(settings, catalog.list_documents(), visual_store)
        await retriever.vector_store.close()
        await visual_store.close()

    summary = _summarize(rows, expected_turn_count)
    environment_changed = {
        "sources": before["sources"]["sha256"] != after["sources"]["sha256"],
        "sqlite_index": before["sqlite_index"]["logical_sha256"]
        != after["sqlite_index"]["logical_sha256"],
        "qdrant_index": before["qdrant_index"]["points_sha256"]
        != after["qdrant_index"]["points_sha256"],
        "visual_text_cache": before["visual_text_cache"]["sha256"]
        != after["visual_text_cache"]["sha256"],
    }
    pass_gates = _automatic_pass_gates(
        execution_succeeded=execution_error is None,
        attempted_turns=len(rows),
        expected_turns=expected_turn_count,
        failed_turns=summary["failed_turns"],
        environment_changed=environment_changed,
    )
    report = {
        "run_status": "finished",
        "generated_at": datetime.now(UTC).isoformat(),
        "suite_id": suite["suite_id"],
        "case_file": str(cases_path.resolve()),
        "case_file_sha256": _file_sha256(cases_path),
        "implementation_sha256": hashlib.sha256(
            _canonical_bytes(
                {
                    str(path.relative_to(PROJECT_ROOT)): _file_sha256(path)
                    for path in sorted((PROJECT_ROOT / "app").rglob("*.py"))
                }
            )
        ).hexdigest(),
        "models": {
            "chat": settings.deepseek_chat_model,
            "text_embedding": settings.dashscope_embedding_model,
            "visual_embedding": settings.dashscope_visual_embedding_model,
            "vision": settings.dashscope_vision_model,
            "answer_review_thinking": True,
            "answer_review_reasoning_effort": "low",
        },
        "top_k": top_k,
        "retrieval_mode": retrieval_mode,
        "query_construction": "contextual",
        "visual_index_write_policy": "disabled_during_evaluation",
        "visual_index_publish_attempts": visual_publish.attempts,
        "automatic_checks_boundary": (
            "本报告只把路由、范围、预期来源召回、引用编号合法性和运行错误作为机械验收；"
            "不把模型自评或自动分数当作回答事实、公式、图表解释正确性的结论。"
        ),
        "environment_before": before,
        "environment_after": after,
        "environment_changed": environment_changed,
        "summary": summary,
        **pass_gates,
        "turns": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if execution_error is not None:
        raise RuntimeError(
            f"验收运行在 {rows[-1]['turn_id']} 失败；部分报告已写入 {output_path}"
        ) from execution_error
    return report


def validate_only(cases_path: Path) -> dict[str, Any]:
    suite = load_suite(cases_path)
    settings = get_settings()
    catalog = MetadataStore(settings)
    catalog.initialize()
    documents = catalog.list_documents()
    _validate_frozen_sources(suite, documents)
    return {
        "suite_id": suite["suite_id"],
        "turn_count": sum(len(item["turns"]) for item in suite["sessions"]),
        "session_count": len(suite["sessions"]),
        "case_file_sha256": _file_sha256(cases_path),
        "visible_document_count": len(documents),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--mode", choices=("text", "auto", "visual"), default="text")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只校验冻结题集和当前 SQLite 文档版本，不调用模型或 Qdrant",
    )
    args = parser.parse_args()
    if not 1 <= args.top_k <= 50:
        parser.error("--top-k 必须在 1 到 50 之间")
    if args.validate_only:
        print(json.dumps(validate_only(args.cases), ensure_ascii=False, indent=2))
        return
    report = asyncio.run(
        evaluate(args.cases, args.output, top_k=args.top_k, retrieval_mode=args.mode)
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"逐轮报告：{args.output}")
    if not report["stage_automatic_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
