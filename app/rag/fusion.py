"""实现 Reciprocal Rank Fusion (RRF) 和跨表示候选融合与去重。"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from qdrant_client import models


class ScoredCandidate(Protocol):
    """支持按 chunk_id 去重和得分融合的候选协议。"""

    @property
    def id(self) -> str: ...

    @property
    def score(self) -> float: ...


def reciprocal_rank_fusion(
    ranked_lists: list[list[models.ScoredPoint]],
    *,
    k: int = 60,
) -> list[models.ScoredPoint]:
    """使用 RRF 融合多个召回结果列表。

    RRF 公式：score(chunk) = Σ 1 / (k + rank_i)
    其中 rank_i 是该 chunk 在第 i 个列表中的排名（从 1 开始）。

    Args:
        ranked_lists: 多个召回结果列表，每个列表按原始得分降序排列
        k: RRF 平滑参数，默认 60

    Returns:
        按 RRF 得分降序排列的融合结果
    """
    if not ranked_lists:
        return []
    if len(ranked_lists) == 1:
        return ranked_lists[0]

    # 收集每个 point ID 的 RRF 得分和首次出现的 point 对象
    rrf_scores: dict[str, float] = defaultdict(float)
    point_by_id: dict[str, models.ScoredPoint] = {}

    for ranked_list in ranked_lists:
        for rank, point in enumerate(ranked_list, start=1):
            point_id = str(point.id)
            rrf_scores[point_id] += 1.0 / (k + rank)
            if point_id not in point_by_id:
                point_by_id[point_id] = point

    # 按 RRF 得分降序排列，保留原始 point 结构但更新 score
    fused = [
        models.ScoredPoint(
            id=point.id,
            score=rrf_scores[str(point.id)],
            payload=point.payload,
            version=point.version,
            vector=point.vector,
        )
        for point_id, point in point_by_id.items()
    ]
    fused.sort(key=lambda p: p.score, reverse=True)
    return fused


def deduplicate_by_chunk(
    candidates: list[models.ScoredPoint],
) -> list[models.ScoredPoint]:
    """按 chunk_id 去重，保留得分最高的候选。

    对于 document outline 表示（没有 chunk_id），使用 representation_id 去重。
    """
    seen_chunks: dict[str, models.ScoredPoint] = {}

    for point in candidates:
        payload = point.payload or {}
        chunk_id = payload.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id:
            key = chunk_id
        else:
            # document outline 或其他没有 chunk_id 的表示
            representation_id = payload.get("representation_id")
            key = str(representation_id) if representation_id else str(point.id)

        existing = seen_chunks.get(key)
        if existing is None or point.score > existing.score:
            seen_chunks[key] = point

    # 保持原始排序（按得分降序）
    return [seen_chunks[key] for key in seen_chunks]


def merge_and_deduplicate(
    *ranked_lists: list[models.ScoredPoint],
) -> list[models.ScoredPoint]:
    """合并多个已排序列表并按 chunk 去重，保留最高得分。"""
    all_candidates: list[models.ScoredPoint] = []
    for ranked_list in ranked_lists:
        all_candidates.extend(ranked_list)

    # 按得分降序排列
    all_candidates.sort(key=lambda p: p.score, reverse=True)

    # 按 chunk 去重
    return deduplicate_by_chunk(all_candidates)
