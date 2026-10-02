"""评估连续用户提问的追问决策；输入和逐题结果默认保存在 Git 忽略目录。"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.clients.deepseek import DeepSeekClient
from app.config import get_settings
from app.rag.query import construct_query
from app.schemas.rag import ConversationTurn, SearchRequest

DEFAULT_CASES = Path("data/evals/real_dialogues.jsonl")
DEFAULT_OUTPUT = Path("data/evals/real_dialogue_report.json")


class DialogueTurn(BaseModel):
    query: str = Field(min_length=1, max_length=2000, pattern=r"\S")
    assistant_reply: str | None = Field(default=None, min_length=1, pattern=r"\S")
    kind: Literal["followup", "independent", "unclear"] | None = None
    tags: list[str] = Field(default_factory=list)


class DialogueSession(BaseModel):
    session_id: str = Field(min_length=1, pattern=r"\S")
    course_id: str | None = None
    source_file: str | None = None
    turns: list[DialogueTurn] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_labels(self) -> DialogueSession:
        if self.turns[0].kind is not None:
            raise ValueError("首轮问题不应标注 kind")
        if any(turn.kind is None for turn in self.turns[1:]):
            raise ValueError("后续每轮问题必须标注 kind")
        return self


def load_sessions(path: Path) -> list[DialogueSession]:
    sessions = [
        DialogueSession.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not sessions or len({session.session_id for session in sessions}) != len(sessions):
        raise ValueError("对话为空或 session_id 重复")
    return sessions


def summarize(rows: list[dict], method: str) -> dict:
    followups = [row for row in rows if row["kind"] == "followup"]
    independent = [row for row in rows if row["kind"] == "independent"]
    unclear = [row for row in rows if row["kind"] == "unclear"]
    false_expansions = sum(row[method] == "history_expanded" for row in independent)
    missed_followups = sum(row[method] == "raw" for row in followups)
    return {
        "followup_count": len(followups),
        "independent_count": len(independent),
        "unclear_count": len(unclear),
        "answer_observed_count": sum(
            bool(row.get("previous_assistant_reply_present")) for row in rows
        ),
        "missed_followups": missed_followups,
        "missed_followup_rate": missed_followups / len(followups) if followups else None,
        "false_expansions": false_expansions,
        "false_expansion_rate": false_expansions / len(independent) if independent else None,
    }


async def evaluate(
    cases_path: Path,
    output_path: Path,
    *,
    compare_model: bool = False,
    client: DeepSeekClient | None = None,
) -> dict:
    sessions = load_sessions(cases_path)
    if compare_model and client is None:
        client = DeepSeekClient(get_settings())
    rows: list[dict] = []
    for session in sessions:
        for turn_index, turn in enumerate(session.turns[1:], start=2):
            previous_turn = session.turns[turn_index - 2]
            previous_user = previous_turn.query.strip()
            rule = construct_query(
                SearchRequest(
                    query=turn.query,
                    query_construction="contextual",
                    history=[ConversationTurn(role="user", content=previous_user)],
                )
            )
            row = {
                "session_id": session.session_id,
                "course_id": session.course_id,
                "source_file": session.source_file,
                "turn_index": turn_index,
                "previous_user": previous_user,
                "previous_assistant_reply_present": previous_turn.assistant_reply is not None,
                "query": turn.query.strip(),
                "kind": turn.kind,
                "tags": turn.tags,
                "rule_method": rule.method,
            }
            if compare_model:
                assert client is not None
                started = time.perf_counter()
                decision = await client.classify_context_need(previous_user, turn.query)
                row.update(
                    model_method="history_expanded" if decision["needs_context"] else "raw",
                    model_reason=decision["reason"],
                    model_latency_ms=round((time.perf_counter() - started) * 1000),
                    model_attempts=decision["attempts"],
                    model_tokens=decision["usage"].get("total_tokens", 0),
                )
            rows.append(row)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": cases_path.as_posix(),
        "session_count": len(sessions),
        "rule": summarize(rows, "rule_method"),
        "by_tag": {
            tag: {"rule": summarize([row for row in rows if tag in row["tags"]], "rule_method")}
            for tag in sorted({tag for row in rows for tag in row["tags"]})
        },
        "cases": rows,
    }
    if compare_model:
        report["model"] = summarize(rows, "model_method")
        report["model_tokens"] = sum(row["model_tokens"] for row in rows)
        for tag, metrics in report["by_tag"].items():
            metrics["model"] = summarize(
                [row for row in rows if tag in row["tags"]], "model_method"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compare-model", action="store_true", help="额外调用 DeepSeek 分类 API")
    args = parser.parse_args()
    report = asyncio.run(evaluate(args.cases, args.output, compare_model=args.compare_model))
    print(json.dumps({key: report[key] for key in ("session_count", "rule")}, ensure_ascii=False))
    if args.compare_model:
        print(json.dumps(report["model"], ensure_ascii=False))
    print(f"逐题报告：{args.output}")


if __name__ == "__main__":
    main()
