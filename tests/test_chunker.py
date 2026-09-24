"""父子切块器的结构与边界测试。"""

from app.rag.chunker import ParentChildChunker, estimate_token_count
from app.schemas.document import BlockKind, ContentBlock, ParsedDocument


def _sample_document() -> ParsedDocument:
    explanation = (
        "梯度下降通过损失函数的梯度更新参数。学习率决定每次更新的步长。"
        "学习率过大会造成震荡，过小则会让收敛速度变慢。"
    )
    return ParsedDocument(
        document_id="doc-gradient",
        source_path="course/gradient_descent.ipynb",
        title="机器学习基础",
        blocks=[
            ContentBlock(
                block_id="cell-1",
                document_id="doc-gradient",
                kind=BlockKind.MARKDOWN,
                text="\n\n".join([explanation] * 5),
                section_path=["优化方法", "梯度下降"],
                cell_id="cell-1",
                cell_index=0,
            ),
            ContentBlock(
                block_id="cell-2",
                document_id="doc-gradient",
                kind=BlockKind.CODE,
                text="for step in range(100):\n    weights -= learning_rate * gradient(weights)",
                section_path=["优化方法", "梯度下降"],
                cell_id="cell-2",
                cell_index=1,
            ),
            ContentBlock(
                block_id="cell-2-output",
                document_id="doc-gradient",
                kind=BlockKind.OUTPUT,
                text="最终损失: 0.031",
                section_path=["优化方法", "梯度下降"],
                cell_id="cell-2",
                cell_index=1,
                related_block_id="cell-2",
            ),
        ],
    )


def test_chunker_builds_deterministic_parent_child_relations() -> None:
    chunker = ParentChildChunker(
        parent_min_tokens=60,
        parent_max_tokens=120,
        child_min_tokens=25,
        child_max_tokens=55,
        child_overlap_tokens=8,
    )
    document = _sample_document()

    parents, children = chunker.chunk(document)
    repeated_parents, repeated_children = chunker.chunk(document)

    assert len(parents) >= 2
    assert len(children) >= len(parents)
    assert [item.parent_id for item in parents] == [item.parent_id for item in repeated_parents]
    assert [item.chunk_id for item in children] == [item.chunk_id for item in repeated_children]
    assert {child.parent_id for child in children} <= {parent.parent_id for parent in parents}
    assert all(child.text.startswith("[课程位置]") for child in children)
    assert all(child.token_count <= 55 for child in children)
    assert all(parent.token_count <= 120 for parent in parents)
    assert all(child.block_ids for child in children)
    assert any(child.block_ids == ["cell-1"] for child in children)


def test_token_estimator_supports_chinese_code_and_numbers() -> None:
    text = "学习率 learning_rate = 0.01"

    assert estimate_token_count(text) >= 6
