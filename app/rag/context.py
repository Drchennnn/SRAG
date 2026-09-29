"""上下文回取、相邻 chunk 扩展和预算管理。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from qdrant_client import models

from app.schemas.rag import SearchHit


class ChunkStore(Protocol):
    """支持按 chunk_id 批量获取的存储接口。"""

    async def get_chunks_by_ids(
        self,
        chunk_ids: Sequence[str],
        *,
        document_id: str,
        version: int,
    ) -> list[models.ScoredPoint]: ...


class ExpandedContext:
    """扩展后的上下文，包含原始 hit、相邻 chunks 和资源信息。"""

    def __init__(
        self,
        original_hit: SearchHit,
        expanded_chunks: list[models.ScoredPoint],
        total_tokens: int,
    ) -> None:
        self.original_hit = original_hit
        self.expanded_chunks = expanded_chunks
        self.total_tokens = total_tokens

    def get_combined_content(self) -> str:
        """合并所有 chunk 的内容为单一文本。"""
        contents = []
        for point in self.expanded_chunks:
            payload = point.payload or {}
            content = payload.get("content")
            if isinstance(content, str) and content.strip():
                contents.append(content.strip())
        return "\n\n".join(contents)


async def expand_with_neighbors(
    hits: list[SearchHit],
    chunk_store: ChunkStore,
    *,
    context_budget_tokens: int = 8000,
    max_neighbors: int = 2,
) -> list[ExpandedContext]:
    """为每个 hit 回取相邻 chunk 并管理总上下文预算。

    Args:
        hits: 检索返回的候选列表
        chunk_store: 支持 get_chunks_by_ids 的存储
        context_budget_tokens: 最终上下文的总 token 预算
        max_neighbors: 每个 hit 最多回取前后各多少个相邻 chunk

    Returns:
        扩展后的上下文列表，按原始得分排序
    """
    if not hits:
        return []

    # 按文档分组，批量获取相邻 chunks
    by_document: dict[tuple[str, int], list[SearchHit]] = {}
    for hit in hits:
        # 从 payload 中提取 version（假设在原始 point 的 payload 中）
        # 这里简化处理，假设同一次查询的 hits 版本一致
        document_id = hit.source.document_id
        version = 1  # 实际应从 metadata_store 或 hit payload 获取
        key = (document_id, version)
        if key not in by_document:
            by_document[key] = []
        by_document[key].append(hit)

    # 收集所有需要获取的 chunk_ids（包括原始 chunk 和相邻 chunk）
    expanded_contexts: list[ExpandedContext] = []
    used_tokens = 0

    for hit in hits:
        if used_tokens >= context_budget_tokens:
            # 预算用尽，后续 hits 只保留原始 chunk
            expanded_contexts.append(
                ExpandedContext(
                    original_hit=hit,
                    expanded_chunks=[],
                    total_tokens=0,
                )
            )
            continue

        # 收集相邻 chunk_ids
        neighbor_ids = _get_neighbor_chunk_ids(hit, max_neighbors=max_neighbors)
        if not neighbor_ids:
            # 没有相邻 chunk 信息，只保留原始内容
            chunk_tokens = _estimate_tokens(hit.content)
            expanded_contexts.append(
                ExpandedContext(
                    original_hit=hit,
                    expanded_chunks=[],
                    total_tokens=chunk_tokens,
                )
            )
            used_tokens += chunk_tokens
            continue

        # 批量获取相邻 chunks
        document_id = hit.source.document_id
        version = 1  # 实际应从 metadata_store 获取
        try:
            neighbor_points = await chunk_store.get_chunks_by_ids(
                neighbor_ids,
                document_id=document_id,
                version=version,
            )

            # 按原始顺序排列（previous -> current -> next）
            ordered_points = _order_chunks(neighbor_points, hit.chunk_id)

            # 计算扩展后的 tokens
            total_content = "\n\n".join(
                str((p.payload or {}).get("content", ""))
                for p in ordered_points
            )
            chunk_tokens = _estimate_tokens(total_content)

            # 检查预算
            if used_tokens + chunk_tokens > context_budget_tokens:
                # 预算不足，只保留原始 chunk
                chunk_tokens = _estimate_tokens(hit.content)
                expanded_contexts.append(
                    ExpandedContext(
                        original_hit=hit,
                        expanded_chunks=[],
                        total_tokens=chunk_tokens,
                    )
                )
            else:
                expanded_contexts.append(
                    ExpandedContext(
                        original_hit=hit,
                        expanded_chunks=ordered_points,
                        total_tokens=chunk_tokens,
                    )
                )
            used_tokens += chunk_tokens

        except Exception:
            # 获取失败时降级到原始内容
            chunk_tokens = _estimate_tokens(hit.content)
            expanded_contexts.append(
                ExpandedContext(
                    original_hit=hit,
                    expanded_chunks=[],
                    total_tokens=chunk_tokens,
                )
            )
            used_tokens += chunk_tokens

    return expanded_contexts


def _get_neighbor_chunk_ids(hit: SearchHit, *, max_neighbors: int) -> list[str]:
    """从 hit 中提取相邻 chunk 的 IDs。

    注意：当前 SearchHit 没有直接存储 previous_chunk_id 和 next_chunk_id，
    需要从原始 ScoredPoint payload 中提取，或者在 retriever 中补充这些信息。
    这里返回空列表作为占位符。
    """
    # TODO: 需要在 SearchHit 或其 source 中添加 previous_chunk_id 和 next_chunk_id
    # 或者在召回时保留原始 payload 信息
    return []


def _order_chunks(
    points: list[models.ScoredPoint],
    current_chunk_id: str,
) -> list[models.ScoredPoint]:
    """按 previous -> current -> next 顺序排列 chunks。"""
    if not points:
        return []

    # 构建 chunk_id -> point 映射
    by_id = {str((p.payload or {}).get("chunk_id", "")): p for p in points}

    # 找到当前 chunk 和相邻 chunks
    current = by_id.get(current_chunk_id)
    if not current:
        return points  # 降级：无法排序时返回原始顺序

    # 向前追溯
    prev_chunks = []
    prev_id = (current.payload or {}).get("previous_chunk_id")
    while prev_id and isinstance(prev_id, str):
        prev_point = by_id.get(prev_id)
        if not prev_point:
            break
        prev_chunks.insert(0, prev_point)
        prev_id = (prev_point.payload or {}).get("previous_chunk_id")

    # 向后追溯
    next_chunks = []
    next_id = (current.payload or {}).get("next_chunk_id")
    while next_id and isinstance(next_id, str):
        next_point = by_id.get(next_id)
        if not next_point:
            break
        next_chunks.append(next_point)
        next_id = (next_point.payload or {}).get("next_chunk_id")

    return prev_chunks + [current] + next_chunks


def _estimate_tokens(text: str) -> int:
    """粗略估算文本的 token 数量（中英文混合）。

    简化估算：中文 1 字符 ≈ 1 token，英文 1 单词 ≈ 1.3 tokens
    """
    if not text:
        return 0

    # 简单估算：字符数 / 2（考虑中英文混合和空格）
    return max(1, len(text) // 2)
