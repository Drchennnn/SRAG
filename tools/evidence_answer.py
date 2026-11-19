"""从已发布 SQLite 资料生成离线的定义与实验回答。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from app.api.structure_search import get_structure_course_allowlist
from app.config import get_settings
from app.rag.evidence_answer import EvidenceAnswer, EvidenceAnswerer
from app.rag.evidence_supplement import SupplementRetriever
from app.rag.local_library import LocalLibrary
from app.rag.structure_search import StructureScope, StructureSearchRequest
from app.stores.structure_snapshot import StructureSnapshotStore


def _visible(text: str) -> str:
    return " ".join("".join(char if char.isprintable() else " " for char in text).split())


def format_text(answer: EvidenceAnswer) -> str:
    headings = {"definition": "定义", "experiment": "实验比较"}
    lines = []
    for section in answer.sections:
        lines.append(headings[section.requirement])
        if section.message:
            lines.append(f"  {section.message}")
        for statement in section.statements:
            citation = statement.citation
            revision = citation.revision
            source = citation.actual_source
            lines.append(f"  {_visible(citation.evidence_id)} · {_visible(statement.text)}")
            label = _visible(f"{revision.course_id} · {citation.file_name} · {source.document_id}")
            lines.append(f"    {label} · v{source.version}/{_visible(source.source_id)}")
            lines.append(f"    原文 [{citation.start}, {citation.end}) {_visible(citation.quote)}")
    if answer.message:
        lines.append(answer.message)
    lines.append(f"停止原因 {_visible(answer.stop_reason)} · 最终证据预算 {answer.budget}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="包含定义或实验比较的问句")
    parser.add_argument("--course", action="append", required=True, help="课程 ID，可重复")
    parser.add_argument("--file", action="append", default=[], help="文件名，可重复")
    parser.add_argument("--document", action="append", default=[], help="资料 ID，可重复")
    parser.add_argument("--strategy", choices=("single", "supplement"), default="supplement")
    parser.add_argument("--top-k", type=int, default=2, help="最终证据预算，1–50")
    parser.add_argument("--json", action="store_true", help="输出序列化回答")
    parser.add_argument("--trace", action="store_true", help="以 JSON 同时展示获准的逐跳轨迹")
    args = parser.parse_args(argv)

    allowed = get_structure_course_allowlist()
    courses = list(dict.fromkeys(course.strip() for course in args.course if course.strip()))
    if not courses or not set(courses).issubset(allowed):
        parser.error("所选课程须全部在本机 SRAG_STRUCTURE_COURSES 授权范围内")
    if not args.question.strip() or not 1 <= args.top_k <= 50:
        parser.error("请输入问句，并将最终证据预算设为 1–50")

    settings = get_settings()
    library = LocalLibrary(settings)
    answerer = EvidenceAnswerer(
        SupplementRetriever(library.store, StructureSnapshotStore(settings))
    )
    request = StructureSearchRequest(
        query=args.question,
        scope=StructureScope(course_ids=courses, document_ids=args.document, file_names=args.file),
        top_k=args.top_k,
        include_adjacency=False,
        include_page_fallback=False,
    )
    result = answerer.answer(
        request, allowed_course_ids=get_structure_course_allowlist, strategy=args.strategy
    )
    if args.trace:
        print(result.model_dump_json(indent=2))
    elif args.json:
        print(result.answer.model_dump_json(indent=2))
    else:
        try:
            text = format_text(result.answer)
        except (TypeError, ValueError):
            fallback = {
                "message": "文本输出格式化失败，保留已核对的结构化回答。",
                "answer": result.answer.model_dump(mode="json"),
            }
            text = json.dumps(fallback, ensure_ascii=True, indent=2)
        print(text)
    return 1 if result.answer.status == "unavailable" else 0


if __name__ == "__main__":
    raise SystemExit(main())
