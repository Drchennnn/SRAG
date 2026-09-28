"""Notebook 结构化切块的边界与追溯测试。"""

from app.rag.chunker import NotebookChunker, estimate_token_count
from app.schemas.document import BlockKind, ContentBlock, ParsedDocument


def _sample_document() -> ParsedDocument:
    return ParsedDocument(
        document_id="doc-gradient",
        source_path="course/gradient_descent.ipynb",
        title="Lecture 3 - Optimization",
        blocks=[
            ContentBlock(
                block_id="intro",
                document_id="doc-gradient",
                kind=BlockKind.MARKDOWN,
                text="# 梯度下降\n梯度下降通过损失函数的梯度更新参数。",
                section_path=["梯度下降"],
                heading_level=1,
                cell_id="intro",
            ),
            ContentBlock(
                block_id="train:code",
                document_id="doc-gradient",
                kind=BlockKind.CODE,
                text="weights -= learning_rate * gradient(weights)",
                section_path=["梯度下降"],
                cell_id="train",
            ),
            ContentBlock(
                block_id="train:output:0",
                document_id="doc-gradient",
                kind=BlockKind.OUTPUT,
                text="最终损失: 0.031",
                section_path=["梯度下降"],
                cell_id="train",
                related_block_id="train:code",
            ),
            ContentBlock(
                block_id="rate",
                document_id="doc-gradient",
                kind=BlockKind.MARKDOWN,
                text="# 学习率\n学习率过大会震荡，过小会减慢收敛。",
                section_path=["学习率"],
                heading_level=1,
                cell_id="rate",
            ),
        ],
    )


def test_chunker_keeps_sections_and_execution_groups_intact() -> None:
    chunker = NotebookChunker(
        soft_min_tokens=10,
        target_tokens=45,
        max_tokens=80,
        overlap_tokens=8,
    )
    document = _sample_document()

    result = chunker.chunk(document)
    repeated = chunker.chunk(document)

    assert [item.chunk_id for item in result.chunks] == [
        item.chunk_id for item in repeated.chunks
    ]
    assert [section.section_path for section in result.sections] == [["梯度下降"], ["学习率"]]
    assert all(chunk.token_count <= 80 for chunk in result.chunks)
    assert all(chunk.section_path in (["梯度下降"], ["学习率"]) for chunk in result.chunks)
    execution_group = next(group for group in result.groups if group.kind.value == "execution")
    assert execution_group.block_ids == ["train:code", "train:output:0"]
    assert any(
        {"train:code", "train:output:0"} <= set(chunk.block_ids)
        for chunk in result.chunks
    )
    assert "梯度下降" in result.outline_text
    assert "学习率" in result.outline_text
    if len(result.chunks) > 1:
        assert result.chunks[0].next_chunk_id == result.chunks[1].chunk_id
        assert result.chunks[1].previous_chunk_id == result.chunks[0].chunk_id


def test_token_estimator_supports_chinese_code_and_numbers() -> None:
    assert estimate_token_count("学习率 learning_rate = 0.01") >= 6
