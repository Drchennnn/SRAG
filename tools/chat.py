"""在终端体验课程检索与 DeepSeek 文字回答。"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.clients.dashscope import DashScopeClient
from app.clients.deepseek import DeepSeekClient
from app.config import get_settings
from app.rag.chat import ChatResult, ChatSession, _source_location
from app.rag.retriever import HybridRetriever
from app.rag.summary import DocumentSummarizer
from app.rag.vision import VisualTextReader
from app.schemas.rag import DocumentSummary, SearchScope
from app.stores.qdrant import QdrantStore
from app.stores.sqlite import MetadataStore


def _course_documents(catalog: MetadataStore, course_id: str) -> list[DocumentSummary]:
    return [item for item in catalog.list_documents() if item.course_id == course_id]


def _resolve_document(catalog: MetadataStore, course_id: str, selector: str) -> DocumentSummary:
    documents = _course_documents(catalog, course_id)
    if selector.isdecimal():
        number = int(selector)
        if 1 <= number <= len(documents):
            return documents[number - 1]
    matches = [
        item
        for item in documents
        if item.document_id == selector
        or item.source_path == selector
        or item.source_path.endswith("/" + selector)
    ]
    if len(matches) != 1:
        raise ValueError(f"文件匹配数为 {len(matches)}；请用 /files 查看编号或完整路径")
    return matches[0]


def _print_files(catalog: MetadataStore, course_id: str) -> None:
    for index, item in enumerate(_course_documents(catalog, course_id), start=1):
        print(f"{index}. {item.source_path} (v{item.version})")


def _write_turn(
    path: Path, session: ChatSession, result: ChatResult, catalog: MetadataStore
) -> None:
    record = {
        "at": datetime.now(UTC).isoformat(),
        "query": result.query,
        "answer": result.answer,
        "route": result.route,
        "retrieval_performed": result.retrieval_performed,
        "query_context": asdict(result.query_context) if result.query_context else None,
        "scope": session.scope.model_dump(),
        "effective_scope": result.effective_scope.model_dump() if result.effective_scope else None,
        "visible_versions": [
            {"document_id": item.document_id, "version": item.version}
            for item in catalog.visible_versions(
                course_ids=session.scope.course_ids,
                document_ids=session.scope.document_ids,
            )
        ],
        "query_construction": session.query_construction,
        "retrieval_query": result.search.retrieval_query,
        "query_method": result.search.query_method,
        "retrieval_mode": result.search.mode_used,
        "hits": [hit.model_dump(mode="json") for hit in result.search.hits],
        "evidence_count": len(result.evidence),
        "cited_numbers": result.cited_numbers,
        "citation_warning": result.citation_warning,
        "model_tokens": result.model_tokens,
        "model_called": result.model_called,
        "coverage": result.coverage,
        "reading_warnings": result.reading_warnings,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _print_result(result: ChatResult) -> None:
    print(f"\n任务：{result.route}")
    if result.retrieval_performed:
        print(f"检索：{result.search.query_method} / {result.search.mode_used}")
        print(f"实际检索文本：{result.search.retrieval_query}")
    else:
        print(
            "本轮直接读取指定文档内容，未调用向量检索。"
            if result.model_called
            else "本轮直接处理，未调用模型或向量检索。"
        )
    for warning in result.reading_warnings:
        print(f"读取提示：{warning}")
    print(f"\n助手：{result.answer}")
    if result.evidence:
        print("\n本轮证据：")
        for number, hit in enumerate(result.evidence, start=1):
            if result.coverage and number not in result.cited_numbers:
                continue
            print(f"  [{number}] {Path(hit.source.file).name} · {_source_location(hit)}")
            if hit.source.image_path and not result.coverage:
                print(f"      截图：{hit.source.image_path}")
    if result.citation_warning:
        print(f"引用检查：{result.citation_warning}")
    print()


def _handle_command(line: str, session: ChatSession, catalog: MetadataStore) -> bool:
    command, _, argument = line.partition(" ")
    argument = argument.strip()
    course_id = session.scope.course_ids[0]
    if command == "/files":
        _print_files(catalog, course_id)
    elif command == "/scope":
        print(
            session.scope.model_dump(), f"contextual={session.query_construction == 'contextual'}"
        )
    elif command == "/course":
        if not argument or not _course_documents(catalog, argument):
            raise ValueError("课程不存在或没有可见文件")
        session.set_scope(SearchScope(course_ids=[argument]))
        print(f"已切换到课程 {argument}；短期对话上下文已清空")
    elif command == "/file":
        document = _resolve_document(catalog, course_id, argument)
        session.set_scope(SearchScope(course_ids=[course_id], document_ids=[document.document_id]))
        print(f"已限定文件 {document.source_path}；短期对话上下文已清空")
    elif command == "/all":
        session.set_scope(SearchScope(course_ids=[course_id]))
        print(f"已使用课程 {course_id} 的全部可见文件；短期对话上下文已清空")
    elif command == "/context":
        if argument not in {"on", "off"}:
            raise ValueError("用法：/context on 或 /context off")
        session.query_construction = "contextual" if argument == "on" else "raw"
        print(f"追问补全：{argument}")
    elif command == "/mode":
        if argument not in {"text", "auto", "visual"}:
            raise ValueError("用法：/mode text、/mode auto 或 /mode visual")
        session.retrieval_mode = argument
        print(f"检索模式：{argument}")
    elif command == "/help":
        print("/files 列文件；/file 编号或路径 限定文件；/all 使用整门课；/course 课程号 切课")
        print(
            "/context on|off 追问补全；/mode text|auto|visual 检索模式；/scope 看范围；/quit 退出"
        )
    elif command == "/quit":
        return False
    else:
        raise ValueError("未知命令；输入 /help 查看可用命令")
    return True


async def run(args: argparse.Namespace) -> None:
    settings = get_settings()
    if not settings.dashscope_api_key:
        raise RuntimeError("尚未在 .env 中配置 DASHSCOPE_API_KEY")
    answer_provider = DeepSeekClient(settings)
    answer_provider.require_api_key()
    catalog = MetadataStore(settings)
    catalog.initialize()
    if not _course_documents(catalog, args.course):
        raise ValueError(f"课程 {args.course} 没有可见的入库文件")
    scope = SearchScope(course_ids=[args.course])
    if args.file:
        document = _resolve_document(catalog, args.course, args.file)
        scope.document_ids = [document.document_id]
    retriever = HybridRetriever.from_settings(settings)
    visual_store = QdrantStore(settings)
    session = ChatSession(
        retriever,
        answer_provider,
        catalog,
        scope=scope,
        top_k=args.top_k,
        retrieval_mode=args.mode,
        query_construction="contextual" if args.contextual else "raw",
        summarizer=DocumentSummarizer(catalog, answer_provider),
        visual_reader=VisualTextReader(
            catalog, DashScopeClient(settings), visual_store, settings.data_dir / "visual_text"
        ),
    )
    log_path = args.log or Path(
        f"data/chat_sessions/session_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}.jsonl"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"范围：{session.scope.model_dump()}；日志：{log_path}")
    print("输入 /help 查看命令，/quit 退出。公式/图表按需识别最多两页并缓存，图片可手动查看。")
    try:
        while True:
            try:
                line = input("你> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            try:
                if line.startswith("/"):
                    if not _handle_command(line, session, catalog):
                        break
                    continue
                result = await session.ask(line)
                _write_turn(log_path, session, result, catalog)
                _print_result(result)
            except Exception as exc:
                print(f"本轮未完成：{exc}")
    finally:
        await retriever.vector_store.close()
        await visual_store.close()
    print(f"对话日志：{log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", required=True, help="已入库课程 ID，例如 CS5187")
    parser.add_argument("--file", help="可见文件的编号、document_id、文件名或完整路径")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--mode", choices=["text", "auto", "visual"], default="text")
    context_flags = parser.add_mutually_exclusive_group()
    context_flags.add_argument(
        "--contextual", dest="contextual", action="store_true", help="开启追问补全（终端默认）"
    )
    context_flags.add_argument(
        "--raw", dest="contextual", action="store_false", help="关闭追问补全，便于对比"
    )
    parser.set_defaults(contextual=True)
    parser.add_argument("--log", type=Path, help="本地 JSONL 日志路径")
    args = parser.parse_args()
    if not 1 <= args.top_k <= 50:
        parser.error("--top-k 必须在 1 到 50 之间")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
