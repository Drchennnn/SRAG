"""测试上下文回取和相邻 chunk 扩展。"""

import pytest
from qdrant_client import models

from app.rag.context import ExpandedContext, _estimate_tokens, _order_chunks, expand_with_neighbors
from app.schemas.rag import SearchHit, SourceReference


class FakeChunkStore:
    def __init__(self, chunks: dict[str, models.ScoredPoint]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[list[str], str, int]] = []

    async def get_chunks_by_ids(
        self,
        chunk_ids: list[str],
        *,
        document_id: str,
        version: int,
    ) -> list[models.ScoredPoint]:
        self.calls.append((chunk_ids, document_id, version))
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]


def test_estimate_tokens():
    """测试 token 估算。"""
    assert _estimate_tokens("") == 0
    assert _estimate_tokens("你好") > 0
    assert _estimate_tokens("hello world") > 0
    # 中英文混合
    text = "这是一段测试文本 with some English words"
    assert _estimate_tokens(text) > 0


def test_order_chunks_simple():
    """测试简单的 chunk 顺序排列。"""
    chunk1 = models.ScoredPoint(
        id="1",
        version=0,
        score=1.0,
        payload={"chunk_id": "c1", "next_chunk_id": "c2"},
    )
    chunk2 = models.ScoredPoint(
        id="2",
        version=0,
        score=1.0,
        payload={"chunk_id": "c2", "previous_chunk_id": "c1", "next_chunk_id": "c3"},
    )
    chunk3 = models.ScoredPoint(
        id="3",
        version=0,
        score=1.0,
        payload={"chunk_id": "c3", "previous_chunk_id": "c2"},
    )

    result = _order_chunks([chunk3, chunk1, chunk2], current_chunk_id="c2")

    assert len(result) == 3
    assert result[0].id == "1"  # c1 在前
    assert result[1].id == "2"  # c2 在中间
    assert result[2].id == "3"  # c3 在后


def test_order_chunks_missing_current():
    """当前 chunk 不在列表中时降级处理。"""
    chunk1 = models.ScoredPoint(
        id="1",
        version=0,
        score=1.0,
        payload={"chunk_id": "c1"},
    )
    result = _order_chunks([chunk1], current_chunk_id="c999")
    assert result == [chunk1]  # 返回原始顺序


@pytest.mark.asyncio
async def test_expand_with_neighbors_no_budget():
    """预算为 0 时只保留空上下文。"""
    hit = SearchHit(
        chunk_id="c1",
        section_id="s1",
        representation_type="text_content",
        score=0.9,
        content="测试内容",
        source=SourceReference(
            course_id="CS001",
            document_id="doc1",
            file="test.ipynb",
        ),
    )
    store = FakeChunkStore({})

    result = await expand_with_neighbors([hit], store, context_budget_tokens=0)

    assert len(result) == 1
    assert result[0].original_hit == hit
    assert result[0].expanded_chunks == []


@pytest.mark.asyncio
async def test_expand_with_neighbors_no_hits():
    """空 hits 列表返回空结果。"""
    store = FakeChunkStore({})
    result = await expand_with_neighbors([], store)
    assert result == []


def test_expanded_context_get_combined_content():
    """测试合并 chunk 内容。"""
    hit = SearchHit(
        chunk_id="c1",
        section_id="s1",
        representation_type="text_content",
        score=0.9,
        content="原始内容",
        source=SourceReference(
            course_id="CS001",
            document_id="doc1",
            file="test.ipynb",
        ),
    )
    chunk1 = models.ScoredPoint(
        id="1",
        version=0,
        score=1.0,
        payload={"chunk_id": "c1", "content": "第一段"},
    )
    chunk2 = models.ScoredPoint(
        id="2",
        version=0,
        score=1.0,
        payload={"chunk_id": "c2", "content": "第二段"},
    )

    context = ExpandedContext(hit, [chunk1, chunk2], total_tokens=100)
    combined = context.get_combined_content()

    assert "第一段" in combined
    assert "第二段" in combined
    assert combined == "第一段\n\n第二段"
