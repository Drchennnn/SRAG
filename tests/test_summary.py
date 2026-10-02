"""整份文档总结的定位、分批和可见性测试。"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest

from app.rag.summary import DocumentSummarizer, resolve_document
from app.schemas.rag import (
    DocumentContent,
    DocumentSummary,
    SearchHit,
    SearchScope,
    SourceReference,
)


def _document(name: str, *, document_id: str | None = None, version: int = 1) -> DocumentSummary:
    return DocumentSummary(
        document_id=document_id or name,
        course_id="CS5187",
        source_path=f"/course/{name}",
        version=version,
        indexed_at=datetime.now(UTC),
    )


def _hit(number: int, content: str, *, page: int | None = None) -> SearchHit:
    return SearchHit(
        chunk_id=f"chunk-{number}",
        section_id=f"section-{number}",
        representation_type="text_content",
        score=0,
        content=content,
        source=SourceReference(
            course_id="CS5187",
            document_id="doc-1",
            file="/course/Lecture2-1.pdf",
            section_path=[f"主题 {number}"],
            cell_ids=[] if page else [f"cell-{number}"],
            page=page,
        ),
    )


class _Catalog:
    def __init__(self, content: DocumentContent) -> None:
        self.content = content
        self.visible = True
        self.read_calls: list[tuple[str, int, list[str], list[str]]] = []

    def read_document(
        self,
        document_id: str,
        version: int,
        *,
        course_ids: list[str],
        document_ids: list[str],
    ) -> DocumentContent:
        self.read_calls.append((document_id, version, course_ids, document_ids))
        return self.content

    def list_documents(self) -> list[DocumentSummary]:
        return [self.content.document] if self.visible else []


class _Model:
    def __init__(
        self, *, long_initial_notes: bool = False, hide_catalog: _Catalog | None = None
    ) -> None:
        self.long_initial_notes = long_initial_notes
        self.hide_catalog = hide_catalog
        self.prompts: list[str] = []

    async def generate_answer(self, messages: list[dict[str, str]]) -> dict:
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        identifiers = " ".join(dict.fromkeys(re.findall(r"\[\d+\]", prompt)))
        if prompt.startswith("文件：") and self.long_initial_notes:
            content = f"{identifiers} " + "中间笔记" * 300
        elif prompt.startswith("压缩以下"):
            content = f"压缩笔记 {identifiers}"
        elif prompt.startswith("合并以下"):
            content = f"最终总结 {identifiers}"
            if self.hide_catalog is not None:
                self.hide_catalog.visible = False
        else:
            content = f"总结 {identifiers}"
        return {"content": content, "usage": {"total_tokens": 7}}


def test_resolve_document_keeps_same_stem_formats_distinct() -> None:
    pdf = _document("Lecture2-1.pdf")
    pptx = _document("Lecture2-1.pptx")
    other = _document("Lecture2-2.pdf")

    assert resolve_document("概括 lecture 2-1", [pdf, pptx, other]) == [pdf, pptx]
    assert resolve_document("概括 Lecture2-1.pdf", [pdf, pptx, other]) == [pdf]
    assert resolve_document("概括 Lectrue_2-1.pptx", [pdf, pptx, other]) == [pptx]
    assert resolve_document("概括不存在的课件.pdf", [pdf, pptx, other]) == []
    assert resolve_document("概括当前课件", [pdf, pptx]) == [pdf, pptx]


def test_resolve_document_uses_longest_path_suffix_for_duplicate_basenames() -> None:
    week_1 = _document("Lecture2-1.pdf", document_id="week-1").model_copy(
        update={"source_path": "/course/CS5187/Week01/Lecture2-1.pdf"}
    )
    week_2 = _document("Lecture2-1.pdf", document_id="week-2").model_copy(
        update={"source_path": "/archive/CS5187_Vision/Week02/Lecture2-1.pdf"}
    )

    assert resolve_document("总结 Week02/Lecture2-1.pdf", [week_1, week_2]) == [week_2]
    assert resolve_document("总结 `/course/CS5187/Week01/Lecture2-1.pdf`", [week_1, week_2]) == [
        week_1
    ]
    assert resolve_document("总结 Lecture2-1.pdf", [week_1, week_2]) == [week_1, week_2]


@pytest.mark.asyncio
async def test_summarize_reads_every_chunk_and_preserves_source_numbers() -> None:
    document = _document("Lecture2-1.pdf", document_id="doc-1")
    hits = [
        _hit(1, "第一主题 " + "甲" * 1650, page=1),
        _hit(2, "第二主题 " + "乙" * 1650, page=2),
        _hit(3, "第三主题 " + "丙" * 1650, page=3),
    ]
    content = DocumentContent(
        document=document,
        title="Edge Detection",
        outline=["Gaussian", "Sobel", "Canny"],
        hits=hits,
        pages=[1, 2, 3],
        image_only_pages=[2],
    )
    catalog = _Catalog(content)
    model = _Model(long_initial_notes=True)

    result = await DocumentSummarizer(catalog, model, batch_chars=2000).summarize(
        document,
        SearchScope(course_ids=["CS5187"], document_ids=["doc-1"]),
    )

    assert catalog.read_calls == [("doc-1", 1, ["CS5187"], ["doc-1"])]
    initial_prompts = [prompt for prompt in model.prompts if prompt.startswith("文件：")]
    assert len(initial_prompts) == 3
    assert all(f"[{number}]" in "\n".join(initial_prompts) for number in range(1, 4))
    assert any(prompt.startswith("压缩以下") for prompt in model.prompts)
    assert all(f"[{number}]" in result.answer for number in range(1, 4))
    assert result.evidence == hits
    assert result.coverage == {
        "document_id": "doc-1",
        "version": 1,
        "total_chunks": 3,
        "read_chunks": 3,
        "pages": [1, 2, 3],
        "image_only_pages": [2],
        "batches": 3,
    }
    assert result.model_tokens == len(model.prompts) * 7
    assert "3 页" in result.answer
    assert "仅有标题/无正文的页面：2" in result.answer


@pytest.mark.asyncio
async def test_summarize_rejects_version_change_during_generation() -> None:
    document = _document("Lecture2-1.pdf", document_id="doc-1")
    content = DocumentContent(
        document=document,
        title="Lecture",
        outline=["Topic"],
        hits=[_hit(1, "甲" * 1650, page=1), _hit(2, "乙" * 1650, page=2)],
        pages=[1, 2],
    )
    catalog = _Catalog(content)
    model = _Model(hide_catalog=catalog)

    with pytest.raises(ValueError, match="已更新或不可见"):
        await DocumentSummarizer(catalog, model, batch_chars=2000).summarize(
            document,
            SearchScope(course_ids=["CS5187"], document_ids=["doc-1"]),
        )


@pytest.mark.asyncio
async def test_summarize_rejects_content_from_another_version() -> None:
    document = _document("Lecture2-1.pdf", document_id="doc-1")
    wrong = DocumentContent(
        document=document.model_copy(update={"version": 2}),
        title="Lecture",
        outline=[],
        hits=[_hit(1, "topic", page=1)],
        pages=[1],
    )
    catalog = _Catalog(wrong)

    with pytest.raises(ValueError, match="版本不符"):
        await DocumentSummarizer(catalog, _Model()).summarize(
            document,
            SearchScope(course_ids=["CS5187"], document_ids=["doc-1"]),
            content=wrong,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope",
    [
        SearchScope(course_ids=["OTHER"], document_ids=[]),
        SearchScope(course_ids=[], document_ids=["other-doc"]),
    ],
)
async def test_summarize_revalidates_scope_for_preloaded_content(
    scope: SearchScope,
) -> None:
    document = _document("Lecture2-1.pdf", document_id="doc-1")
    content = DocumentContent(
        document=document,
        title="Lecture",
        outline=[],
        hits=[_hit(1, "topic", page=1)],
        pages=[1],
    )

    with pytest.raises(ValueError, match="当前总结范围"):
        await DocumentSummarizer(_Catalog(content), _Model()).summarize(
            document,
            scope,
            content=content,
        )
