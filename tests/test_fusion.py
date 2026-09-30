"""测试 RRF 融合和跨表示去重逻辑。"""

from qdrant_client import models

from app.rag.fusion import (
    deduplicate_by_chunk,
    merge_and_deduplicate,
    prefer_visual_pages,
    reciprocal_rank_fusion,
    visual_query_intent,
)


def test_rrf_single_list():
    """单个列表不需要融合，直接返回。"""
    points = [
        models.ScoredPoint(id="1", version=0, score=0.9, payload={"chunk_id": "c1"}),
        models.ScoredPoint(id="2", version=0, score=0.8, payload={"chunk_id": "c2"}),
    ]
    result = reciprocal_rank_fusion([points])
    assert result == points


def test_rrf_fusion():
    """RRF 融合两个列表。"""
    list1 = [
        models.ScoredPoint(id="1", version=0, score=0.9, payload={"chunk_id": "c1"}),
        models.ScoredPoint(id="2", version=0, score=0.8, payload={"chunk_id": "c2"}),
    ]
    list2 = [
        models.ScoredPoint(id="2", version=0, score=0.95, payload={"chunk_id": "c2"}),
        models.ScoredPoint(id="3", version=0, score=0.85, payload={"chunk_id": "c3"}),
    ]
    result = reciprocal_rank_fusion([list1, list2], k=60)

    # c2 在两个列表中都出现，RRF 得分应该最高
    assert len(result) == 3
    assert str(result[0].id) == "2"  # c2 应该排第一
    assert result[0].score > 0  # RRF 得分


def test_deduplicate_by_chunk():
    """按 chunk_id 去重，保留最高得分。"""
    points = [
        models.ScoredPoint(id="1", version=0, score=0.9, payload={"chunk_id": "c1"}),
        models.ScoredPoint(id="2", version=0, score=0.8, payload={"chunk_id": "c1"}),  # 重复
        models.ScoredPoint(id="3", version=0, score=0.7, payload={"chunk_id": "c2"}),
    ]
    result = deduplicate_by_chunk(points)

    assert len(result) == 2
    assert str(result[0].id) == "1"  # 保留 c1 的最高得分
    assert result[0].score == 0.9


def test_deduplicate_outline():
    """document outline 没有 chunk_id，使用 representation_id 去重。"""
    points = [
        models.ScoredPoint(
            id="1",
            version=0,
            score=0.9,
            payload={"representation_id": "r1", "representation_type": "document_outline"},
        ),
        models.ScoredPoint(
            id="2",
            version=0,
            score=0.8,
            payload={"representation_id": "r1", "representation_type": "document_outline"},
        ),
    ]
    result = deduplicate_by_chunk(points)

    assert len(result) == 1
    assert str(result[0].id) == "1"


def test_merge_and_deduplicate():
    """合并多个列表并去重。"""
    list1 = [
        models.ScoredPoint(id="1", version=0, score=0.9, payload={"chunk_id": "c1"}),
    ]
    list2 = [
        # 重复，更高分
        models.ScoredPoint(id="2", version=0, score=0.95, payload={"chunk_id": "c1"}),
        models.ScoredPoint(id="3", version=0, score=0.8, payload={"chunk_id": "c2"}),
    ]
    result = merge_and_deduplicate(list1, list2)

    assert len(result) == 2
    assert str(result[0].id) == "2"  # c1 保留更高得分
    assert result[0].score == 0.95


def test_visual_query_intent_distinguishes_page_image_from_topic() -> None:
    assert visual_query_intent("哪页照片里有河流、红玫瑰和雪山？")
    assert visual_query_intent("Find the diagram of camera projection")
    assert not visual_query_intent("图像梯度的方向如何计算？")


def test_visual_first_keeps_image_only_page_and_deduplicates_text() -> None:
    visual = [
        models.ScoredPoint(
            id=1, version=1, score=0.9, payload={"document_id": "doc", "page": 8, "chunk_id": "c8"}
        ),
    ]
    text = [
        models.ScoredPoint(
            id=2, version=1, score=0.8, payload={"document_id": "doc", "page": 8, "chunk_id": "c8"}
        ),
        models.ScoredPoint(
            id=3, version=1, score=0.7, payload={"document_id": "doc", "page": 9, "chunk_id": "c9"}
        ),
    ]
    result = prefer_visual_pages(visual, text, top_k=2)
    assert [point.id for point in result] == [1, 3]
