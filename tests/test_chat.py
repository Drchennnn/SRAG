"""终端对话的范围、证据与历史边界。"""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest

from app.rag.chat import ChatResult, ChatSession
from app.rag.summary import DocumentSummarizer
from app.schemas.rag import (
    DocumentContent,
    DocumentSummary,
    SearchHit,
    SearchResponse,
    SearchScope,
    SourceReference,
)
from app.stores.sqlite import VisibleVersion
from tools.chat import _handle_command, _resolve_document, _write_turn


def _document(name: str = "Lecture2-1.pdf") -> DocumentSummary:
    return DocumentSummary(
        document_id="doc-1",
        course_id="CS5187",
        source_path=f"/courses/CS5187/{name}",
        version=1,
        indexed_at=datetime.now(UTC),
    )


def _hit() -> SearchHit:
    return SearchHit(
        chunk_id="chunk-1",
        section_id="section-1",
        representation_type="text",
        score=0.9,
        content="Sobel 算子计算图像梯度。",
        source=SourceReference(
            course_id="CS5187",
            document_id="doc-1",
            file="/courses/CS5187/Lecture2-1.pdf",
            page=12,
        ),
    )


@pytest.mark.asyncio
async def test_chat_uses_current_evidence_and_recent_history() -> None:
    retriever = Mock()
    retriever.search = AsyncMock(
        side_effect=[
            SearchResponse(query="Sobel 是什么？", retrieval_query="Sobel 是什么？", hits=[_hit()]),
            SearchResponse(
                query="它有什么用？",
                retrieval_query="Sobel 是什么？\n追问：它有什么用？",
                hits=[_hit()],
            ),
        ]
    )
    answer = Mock()
    answer.generate_answer = AsyncMock(
        side_effect=[
            {"content": "Sobel 用于估计梯度。[1]", "usage": {"total_tokens": 30}},
            {"content": "它可用于边缘检测。[1]", "usage": {"total_tokens": 35}},
        ]
    )
    catalog = Mock(list_documents=Mock(return_value=[_document()]))
    session = ChatSession(
        retriever,
        answer,
        catalog,
        scope=SearchScope(course_ids=["CS5187"], document_ids=["doc-1"]),
        query_construction="contextual",
    )

    first = await session.ask("Sobel 是什么？")
    second = await session.ask("它有什么用？")

    assert first.cited_numbers == [1]
    assert first.model_tokens == 30
    assert second.citation_warning is None
    request = retriever.search.await_args.args[0]
    assert request.query == "Sobel 是什么？\n追问：它有什么用？"
    assert request.history == []
    assert request.query_construction == "raw"
    assert second.query == "它有什么用？"
    assert second.search.query_method == "history_expanded"
    assert second.route == "follow_up"
    prompt = answer.generate_answer.await_args.args[0][1]["content"]
    assert "Sobel 用于估计梯度。[1]" in prompt
    assert "第 12 页" in prompt
    assert "/courses/CS5187/Lecture2-1.pdf" in prompt


@pytest.mark.asyncio
async def test_chat_no_evidence_does_not_call_model_and_scope_reset() -> None:
    retriever = Mock()
    retriever.search = AsyncMock(return_value=SearchResponse(query="无答案？", hits=[]))
    answer = Mock(generate_answer=AsyncMock())
    catalog = Mock(list_documents=Mock(return_value=[_document()]))
    session = ChatSession(retriever, answer, catalog, scope=SearchScope(course_ids=["CS5187"]))

    result = await session.ask("无答案？")
    assert "没有找到" in result.answer
    answer.generate_answer.assert_not_awaited()
    assert len(session.turns) == 1

    session.set_scope(SearchScope(course_ids=["CS5489"]))
    assert session.turns == []
    assert session.context is None


@pytest.mark.asyncio
async def test_chat_flags_invalid_citation_number() -> None:
    retriever = Mock(search=AsyncMock(return_value=SearchResponse(query="问题", hits=[_hit()])))
    answer = Mock(
        generate_answer=AsyncMock(
            return_value={"content": "错误引用。[7]", "usage": {"total_tokens": 20}}
        )
    )
    catalog = Mock(list_documents=Mock(return_value=[_document()]))
    session = ChatSession(retriever, answer, catalog, scope=SearchScope(course_ids=["CS5187"]))

    result = await session.ask("问题")

    assert result.citation_warning == "回答含不存在的证据编号：[7]"


def test_cli_file_selection_is_course_limited_and_ambiguous_names_fail() -> None:
    first = _document()
    second = first.model_copy(
        update={"document_id": "doc-2", "source_path": "/courses/CS5187/other/Lecture2-1.pdf"}
    )
    catalog = Mock(list_documents=Mock(return_value=[first, second]))

    assert _resolve_document(catalog, "CS5187", "1").document_id == "doc-1"
    with pytest.raises(ValueError, match="匹配数为 2"):
        _resolve_document(catalog, "CS5187", "Lecture2-1.pdf")

    session = ChatSession(Mock(), Mock(), catalog, scope=SearchScope(course_ids=["CS5187"]))
    _handle_command("/file 2", session, catalog)
    assert session.scope.document_ids == ["doc-2"]
    _handle_command("/all", session, catalog)
    assert session.scope.document_ids == []


def test_chat_log_keeps_answer_scope_and_visible_version(tmp_path) -> None:
    catalog = Mock()
    catalog.list_documents.return_value = [_document()]
    catalog.visible_versions.return_value = [VisibleVersion(document_id="doc-1", version=1)]
    session = ChatSession(
        Mock(), Mock(), catalog, scope=SearchScope(course_ids=["CS5187"], document_ids=["doc-1"])
    )
    result = ChatResult(
        query="Sobel 是什么？",
        answer="Sobel 用于梯度估计。[1]",
        search=SearchResponse(
            query="Sobel 是什么？", retrieval_query="Sobel 是什么？", hits=[_hit()]
        ),
        evidence=[_hit()],
        cited_numbers=[1],
        citation_warning=None,
        model_tokens=42,
    )
    path = tmp_path / "session.jsonl"

    _write_turn(path, session, result, catalog)

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["answer"] == result.answer
    assert row["visible_versions"] == [{"document_id": "doc-1", "version": 1}]
    assert row["hits"][0]["source"]["page"] == 12
    assert row["route"] == "knowledge"
    assert row["retrieval_performed"] is True


@pytest.mark.asyncio
async def test_history_and_catalog_skip_retrieval_model_and_citation_check() -> None:
    retriever = Mock(search=AsyncMock())
    answer = Mock(generate_answer=AsyncMock())
    other = _document().model_copy(update={"course_id": "CS5489", "document_id": "other"})
    catalog = Mock(list_documents=Mock(return_value=[_document(), other]))
    session = ChatSession(retriever, answer, catalog, scope=SearchScope(course_ids=["CS5187"]))
    session.turns = [(f"知识点问题 {index}", "旧回答 [9]") for index in range(1, 6)]

    history = await session.ask("我第一个问题是什么")
    listing = await session.ask("有哪些已入库的课件？")
    previous = await session.ask("你刚才的回答是什么")

    assert "知识点问题 1" in history.answer
    assert "无法保证" not in history.answer
    assert "Lecture2-1.pdf" in listing.answer
    assert "CS5489" not in listing.answer
    assert previous.answer.endswith(listing.answer)
    for result in (history, listing, previous):
        assert result.retrieval_performed is False
        assert result.evidence == []
        assert result.citation_warning is None
        assert result.model_tokens == 0
    retriever.search.assert_not_awaited()
    answer.generate_answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_real_kernel_followup_and_topic_switch_use_fresh_evidence() -> None:
    old_hit = _hit()
    new_hit = old_hit.model_copy(update={"chunk_id": "new-chunk", "content": "新版本梯度解释"})
    document = _document()
    catalog = Mock(list_documents=Mock(return_value=[document]))
    retriever = Mock(
        search=AsyncMock(
            side_effect=[
                SearchResponse(query="Sobel", hits=[old_hit]),
                SearchResponse(query="追问", hits=[new_hit]),
                SearchResponse(query="换题", hits=[new_hit]),
            ]
        )
    )
    answer = Mock(
        generate_answer=AsyncMock(
            side_effect=[
                {"content": "Sobel 的两个方向核分别得到 Gx 和 Gy。[1]"},
                {"content": "方向核计算梯度。[1]"},
                {"content": "主点位于相机成像平面。[1]"},
            ]
        )
    )
    session = ChatSession(retriever, answer, catalog, scope=SearchScope(course_ids=["CS5187"]))

    await session.ask("sobel operator 是什么？以及在哪")
    catalog.list_documents.return_value = [document.model_copy(update={"version": 2})]
    followup = await session.ask("这两个方向的核有什么用？课件如果没有你自己回答")
    switched = await session.ask("这个主点定义在哪页？")

    request = retriever.search.await_args_list[1].args[0]
    assert request.query == "sobel operator 是什么？以及在哪\n追问：这两个方向的核有什么用"
    assert request.scope.document_ids == []
    assert request.scope.course_ids == ["CS5187"]
    assert followup.evidence == [new_hit]
    assert followup.query_context.sources[0].version == 1
    assert followup.query_context.entities == ("Sobel", "Gx", "Gy")
    prompt = answer.generate_answer.await_args_list[1].args[0][1]["content"]
    assert "内容：新版本梯度解释" in prompt
    assert "内容：Sobel 算子计算图像梯度。" not in prompt
    assert switched.route == "knowledge"
    assert retriever.search.await_args_list[2].args[0].query == "这个主点定义在哪页？"


@pytest.mark.asyncio
async def test_summary_is_explicitly_pending_and_does_not_search_random_pages() -> None:
    retriever = Mock(search=AsyncMock())
    answer = Mock(generate_answer=AsyncMock())
    catalog = Mock(list_documents=Mock(return_value=[_document()]))
    session = ChatSession(retriever, answer, catalog, scope=SearchScope(course_ids=["CS5187"]))

    result = await session.ask("给我简要总结一下 lectrue 2-1")

    assert result.route == "document_summary"
    assert "尚未实现" in result.answer
    assert result.retrieval_performed is False
    retriever.search.assert_not_awaited()
    answer.generate_answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_summary_reads_full_file_instead_of_retrieval_top_k(tmp_path) -> None:
    document = _document()
    hits = [_hit().model_copy(update={"chunk_id": f"chunk-{i}"}) for i in range(9)]
    content = DocumentContent(document=document, title="Lecture2-1", outline=["Sobel"], hits=hits)
    catalog = Mock(
        list_documents=Mock(return_value=[document]), read_document=Mock(return_value=content)
    )
    answer = Mock(generate_answer=AsyncMock(return_value={"content": "末尾知识点。[9]"}))
    retriever = Mock(search=AsyncMock())
    session = ChatSession(
        retriever,
        answer,
        catalog,
        scope=SearchScope(course_ids=["CS5187"]),
        summarizer=DocumentSummarizer(catalog, answer),
    )
    result = await session.ask("总一下 lecture2-1，顺便说一下有哪些重点")
    assert result.model_called and not result.retrieval_performed
    assert len(result.evidence) == 9 and result.coverage["read_chunks"] == 9
    assert result.citation_warning is None and result.cited_numbers == [9]
    retriever.search.assert_not_awaited()
    path = tmp_path / "chat.jsonl"
    catalog.visible_versions.return_value = []
    _write_turn(path, session, result, catalog)
    saved = json.loads(path.read_text())
    assert saved["coverage"]["read_chunks"] == 9 and saved["model_called"]


@pytest.mark.asyncio
async def test_summary_does_not_choose_between_different_formats() -> None:
    pdf = _document()
    pptx = _document("Lecture2-1.pptx").model_copy(update={"document_id": "doc-2"})
    catalog = Mock(list_documents=Mock(return_value=[pdf, pptx]))
    summarizer = Mock(summarize=AsyncMock())
    session = ChatSession(
        Mock(), Mock(), catalog, scope=SearchScope(course_ids=["CS5187"]), summarizer=summarizer
    )
    result = await session.ask("总结 lecture2-1")
    assert "无法唯一定位" in result.answer
    assert "Lecture2-1.pdf" in result.answer and "Lecture2-1.pptx" in result.answer
    summarizer.summarize.assert_not_awaited()


@pytest.mark.asyncio
async def test_chat_uses_visual_extraction_in_answer_and_logs_reading_failure() -> None:
    original = _hit()
    extracted = original.model_copy(
        update={"content": "G(x)=exp(-x^2)。", "evidence_origin": "visual_extract"}
    )
    visual = Mock(enrich=AsyncMock(return_value=([extracted], ["第 15 页视觉补充失败"])))
    answer = Mock(generate_answer=AsyncMock(return_value={"content": "公式 G(x)=exp(-x^2)。[1]"}))
    retriever = Mock(search=AsyncMock(return_value=SearchResponse(query="Q", hits=[original])))
    session = ChatSession(
        retriever,
        answer,
        Mock(list_documents=Mock(return_value=[_document()])),
        scope=SearchScope(course_ids=["CS5187"]),
        visual_reader=visual,
    )
    result = await session.ask("高斯公式是什么")
    assert result.search.hits == [extracted]
    assert result.reading_warnings == ("第 15 页视觉补充失败",)
    prompt = answer.generate_answer.await_args.args[0]
    assert "G(x)=exp(-x^2)" in prompt[1]["content"]
    assert "visual_extract" in prompt[1]["content"]


@pytest.mark.asyncio
async def test_new_extrinsic_topic_is_used_by_matrix_followup() -> None:
    retriever = Mock(search=AsyncMock(return_value=SearchResponse(query="Q", hits=[_hit()])))
    answer = Mock(
        generate_answer=AsyncMock(
            side_effect=[
                {"content": "内参矩阵 K。[1]"},
                {"content": "外参包含 R 和 T。[1]"},
                {"content": "外参是世界坐标到相机坐标的变换。[1]"},
            ]
        )
    )
    session = ChatSession(
        retriever,
        answer,
        Mock(list_documents=Mock(return_value=[_document()])),
        scope=SearchScope(course_ids=["CS5187"]),
    )
    await session.ask("Intrinsic parameters 是什么？")
    switched = await session.ask("Extrinsic parameters 又是什么？")
    result = await session.ask("它的矩阵公式是什么？")
    assert switched.route == "knowledge" and switched.search.query_method == "raw"
    assert result.route == "follow_up"
    assert (
        result.search.retrieval_query
        == "Extrinsic parameters 又是什么？\n追问：它的矩阵公式是什么？"
    )


@pytest.mark.asyncio
async def test_derived_formula_and_original_chunk_are_not_duplicate_evidence() -> None:
    original = _hit()
    derived = original.model_copy(
        update={
            "chunk_id": "visual-text-key",
            "content": "公式内容",
            "origin_chunk_id": original.chunk_id,
            "evidence_origin": "visual_extract",
        }
    )
    retriever = Mock(
        search=AsyncMock(return_value=SearchResponse(query="Q", hits=[derived, original]))
    )
    session = ChatSession(
        retriever,
        Mock(generate_answer=AsyncMock(return_value={"content": "公式。[1]"})),
        Mock(list_documents=Mock(return_value=[_document()])),
        scope=SearchScope(course_ids=["CS5187"]),
    )
    result = await session.ask("公式是什么？")
    assert result.evidence == [derived]


@pytest.mark.asyncio
async def test_document_deletion_while_answering_does_not_return_stale_content() -> None:
    hit = _hit().model_copy(update={"source": _hit().source.model_copy(update={"version": 1})})
    catalog = Mock(list_documents=Mock(return_value=[]))
    retriever = Mock(search=AsyncMock(return_value=SearchResponse(query="Q", hits=[hit])))
    session = ChatSession(
        retriever,
        Mock(generate_answer=AsyncMock(return_value={"content": "旧信息。[1]"})),
        catalog,
        scope=SearchScope(course_ids=["CS5187"]),
    )
    with pytest.raises(ValueError, match="不可见"):
        await session.ask("Sobel 是什么？")
    assert not session.turns


@pytest.mark.asyncio
async def test_explicit_file_then_page_correction_reads_exact_page_without_vector_search():
    document = _document()
    other = _document("Lecture1-1.pdf").model_copy(update={"document_id": "other"})
    page14 = _hit().model_copy(update={"source": _hit().source.model_copy(update={"page": 14})})
    catalog = Mock(
        list_documents=Mock(return_value=[document, other]),
        read_document=Mock(
            return_value=DocumentContent(
                document=document, title="Edges", outline=[], hits=[page14], pages=[14]
            )
        ),
    )
    retriever = Mock(search=AsyncMock(return_value=SearchResponse(query="Q", hits=[_hit()])))
    visual = Mock(enrich=AsyncMock(side_effect=lambda q, hits, scope: (hits, [])))
    session = ChatSession(
        retriever,
        Mock(generate_answer=AsyncMock(return_value={"content": "解释。[1]"})),
        catalog,
        scope=SearchScope(course_ids=["CS5187"]),
        visual_reader=visual,
    )
    await session.ask("lecture2-1 的 smooth first 部分的图是什么意思？")
    assert retriever.search.await_args.args[0].scope.document_ids == [document.document_id]
    correction = await session.ask("并不是，是 14 页的图，给我解释一下")
    assert correction.route == "document_page" and not correction.retrieval_performed
    assert correction.effective_scope.document_ids == [document.document_id]
    assert correction.evidence == [page14]
    assert retriever.search.await_count == 1
    assert session.scope.document_ids == []  # 本轮约束不会永久改变用户范围。
    catalog.read_document.assert_called_once_with(
        "doc-1", 1, course_ids=["CS5187"], document_ids=["doc-1"]
    )


@pytest.mark.asyncio
async def test_page_without_file_focus_asks_for_document_instead_of_guessing():
    documents = [
        _document(),
        _document("Lecture1-1.pdf").model_copy(update={"document_id": "other"}),
    ]
    retriever = Mock(search=AsyncMock())
    session = ChatSession(
        retriever,
        Mock(),
        Mock(list_documents=Mock(return_value=documents)),
        scope=SearchScope(course_ids=["CS5187"]),
    )
    result = await session.ask("给我解释第 14 页的图")
    assert "页码需要对应文件" in result.answer
    retriever.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_ambiguous_file_in_knowledge_question_does_not_guess_between_formats():
    documents = [
        _document("Lecture1-1.pdf"),
        _document("Lecture1-1.pptx").model_copy(update={"document_id": "other"}),
    ]
    retriever = Mock(search=AsyncMock())
    session = ChatSession(
        retriever,
        Mock(),
        Mock(list_documents=Mock(return_value=documents)),
        scope=SearchScope(course_ids=["CS5187"]),
    )
    result = await session.ask("lecture1-1 的成绩占比是什么？")
    assert "无法唯一定位" in result.answer
    retriever.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_scope_switch_prevents_followup_from_using_previous_course() -> None:
    retriever = Mock(search=AsyncMock(return_value=SearchResponse(query="Q", hits=[_hit()])))
    answer = Mock(generate_answer=AsyncMock(return_value={"content": "Sobel 梯度。[1]"}))
    catalog = Mock(list_documents=Mock(return_value=[_document()]))
    session = ChatSession(retriever, answer, catalog, scope=SearchScope(course_ids=["CS5187"]))
    await session.ask("Sobel 是什么？")

    session.set_scope(SearchScope(course_ids=["CS5489"]))
    result = await session.ask("这两个方向的核有什么用？")

    assert result.route == "knowledge"
    request = retriever.search.await_args.args[0]
    assert "Sobel" not in request.query
    assert request.scope.course_ids == ["CS5489"]
