"""基于文档结构构建父块与子块。"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass

from app.config import Settings
from app.schemas.document import (
    BlockKind,
    ChildChunk,
    ContentBlock,
    ParentChunk,
    ParsedDocument,
)

_TOKEN_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[^\s]"
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])|(?<=[.!?])\s+")

_BLOCK_LABELS = {
    BlockKind.MARKDOWN: "说明",
    BlockKind.CODE: "代码",
    BlockKind.OUTPUT: "运行输出",
    BlockKind.TEXT: "正文",
    BlockKind.TABLE: "表格",
    BlockKind.FIGURE: "图表",
}


def estimate_token_count(text: str) -> int:
    """在不加载本地模型 Tokenizer 的前提下估算 Token 数量。

    中文字符、英文单词、数字和标点分别计为一个估算 Token。实际计费 Token
    以 DashScope API 返回的 usage 为准。
    """

    return len(_TOKEN_PATTERN.findall(text))


@dataclass(frozen=True, slots=True)
class _Fragment:
    text: str
    block_ids: tuple[str, ...]
    section_path: tuple[str, ...]

    @property
    def token_count(self) -> int:
        return estimate_token_count(self.text)


class ParentChildChunker:
    """构建可追溯的父级上下文和可检索子块。"""

    def __init__(
        self,
        *,
        parent_min_tokens: int = 1000,
        parent_max_tokens: int = 1800,
        child_min_tokens: int = 300,
        child_max_tokens: int = 600,
        child_overlap_tokens: int = 60,
    ) -> None:
        if not 0 < parent_min_tokens <= parent_max_tokens:
            raise ValueError("父块 Token 范围无效")
        if not 0 < child_min_tokens <= child_max_tokens:
            raise ValueError("子块 Token 范围无效")
        if not 0 <= child_overlap_tokens < child_max_tokens:
            raise ValueError("子块重叠 Token 数必须小于子块最大 Token 数")

        self.parent_min_tokens = parent_min_tokens
        self.parent_max_tokens = parent_max_tokens
        self.child_min_tokens = child_min_tokens
        self.child_max_tokens = child_max_tokens
        self.child_overlap_tokens = child_overlap_tokens

    @classmethod
    def from_settings(cls, settings: Settings) -> ParentChildChunker:
        """从项目配置创建切块器。"""

        return cls(
            parent_min_tokens=settings.parent_chunk_min_tokens,
            parent_max_tokens=settings.parent_chunk_max_tokens,
            child_min_tokens=settings.child_chunk_min_tokens,
            child_max_tokens=settings.child_chunk_max_tokens,
            child_overlap_tokens=settings.child_chunk_overlap_tokens,
        )

    def chunk(self, document: ParsedDocument) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """将统一文档结构转换为父块和可检索子块。"""

        fragments = [
            fragment
            for block in document.blocks
            for fragment in self._fragment_block(block)
            if fragment.text.strip()
        ]
        parent_groups = self._build_parents(document, fragments)
        parents = [parent for parent, _ in parent_groups]
        children = [
            child
            for parent_index, (parent, group) in enumerate(parent_groups)
            for child in self._build_children(document, parent, group, parent_index)
        ]
        return parents, children

    def _fragment_block(self, block: ContentBlock) -> list[_Fragment]:
        rendered = self._render_block(block)
        if estimate_token_count(rendered) <= self.parent_max_tokens:
            return [
                _Fragment(
                    text=rendered,
                    block_ids=(block.block_id,),
                    section_path=tuple(block.section_path),
                )
            ]

        pieces = _split_text(rendered, self.parent_max_tokens, overlap_tokens=0)
        return [
            _Fragment(
                text=piece,
                block_ids=(block.block_id,),
                section_path=tuple(block.section_path),
            )
            for piece in pieces
        ]

    def _build_parents(
        self,
        document: ParsedDocument,
        fragments: list[_Fragment],
    ) -> list[tuple[ParentChunk, list[_Fragment]]]:
        groups: list[list[_Fragment]] = []
        current: list[_Fragment] = []
        current_tokens = 0

        for fragment in fragments:
            section_changed = bool(current) and fragment.section_path != current[-1].section_path
            exceeds_limit = bool(current) and (
                current_tokens + fragment.token_count > self.parent_max_tokens
            )
            section_is_complete = section_changed and current_tokens >= self.parent_min_tokens

            if exceeds_limit or section_is_complete:
                groups.append(current)
                current = []
                current_tokens = 0

            current.append(fragment)
            current_tokens += fragment.token_count

        if current:
            groups.append(current)

        parents: list[tuple[ParentChunk, list[_Fragment]]] = []
        for index, group in enumerate(groups):
            text = "\n\n".join(fragment.text for fragment in group).strip()
            block_ids = _unique(block_id for fragment in group for block_id in fragment.block_ids)
            section_path = _common_prefix([fragment.section_path for fragment in group])
            parent = ParentChunk(
                parent_id=_stable_id("parent", document.document_id, index, text),
                document_id=document.document_id,
                text=text,
                block_ids=block_ids,
                section_path=section_path,
                token_count=estimate_token_count(text),
            )
            parents.append((parent, group))
        return parents

    def _build_children(
        self,
        document: ParsedDocument,
        parent: ParentChunk,
        fragments: list[_Fragment],
        parent_index: int,
    ) -> list[ChildChunk]:
        prefix_tokens = max(
            (
                estimate_token_count(_section_prefix(document.title, list(fragment.section_path)))
                for fragment in fragments
            ),
            default=0,
        )
        content_limit = max(1, self.child_max_tokens - prefix_tokens)
        content_min = max(1, self.child_min_tokens - prefix_tokens)
        content_overlap = min(self.child_overlap_tokens, max(0, content_limit - 1))
        child_fragments = [
            _Fragment(
                text=piece,
                block_ids=fragment.block_ids,
                section_path=fragment.section_path,
            )
            for fragment in fragments
            for piece in _split_text(fragment.text, content_limit, overlap_tokens=0)
        ]
        groups = _pack_fragments(
            child_fragments,
            max_tokens=content_limit,
            min_tokens=content_min,
            overlap_tokens=content_overlap,
        )

        children: list[ChildChunk] = []
        for child_index, group in enumerate(groups):
            piece = "\n\n".join(fragment.text for fragment in group).strip()
            section_path = _common_prefix([fragment.section_path for fragment in group])
            section_prefix = _section_prefix(document.title, section_path)
            text = f"{section_prefix}\n\n{piece}" if section_prefix else piece
            children.append(
                ChildChunk(
                    chunk_id=_stable_id(
                        "child",
                        document.document_id,
                        parent_index,
                        child_index,
                        text,
                    ),
                    parent_id=parent.parent_id,
                    document_id=document.document_id,
                    text=text,
                    block_ids=_unique(
                        block_id for fragment in group for block_id in fragment.block_ids
                    ),
                    section_path=section_path,
                    token_count=estimate_token_count(text),
                )
            )
        return children

    @staticmethod
    def _render_block(block: ContentBlock) -> str:
        label = _BLOCK_LABELS[block.kind]
        body = block.text.strip()
        asset_lines = [
            f"资源：{asset.caption or asset.path}（{asset.media_type}）" for asset in block.assets
        ]
        if asset_lines:
            body = "\n".join(part for part in (body, *asset_lines) if part)
        if block.kind is BlockKind.CODE:
            return f"[{label}]\n```python\n{body}\n```"
        return f"[{label}]\n{body}".strip()


def _section_prefix(title: str, section_path: list[str]) -> str:
    parts = [part for part in (title, *section_path) if part]
    return f"[课程位置] {' > '.join(parts)}" if parts else ""


def _stable_id(kind: str, *parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"{kind}_{digest}"


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _common_prefix(paths: list[tuple[str, ...]]) -> list[str]:
    if not paths:
        return []
    prefix: list[str] = []
    for parts in zip(*paths, strict=False):
        if len(set(parts)) != 1:
            break
        prefix.append(parts[0])
    return prefix


def _split_text(
    text: str,
    max_tokens: int,
    *,
    min_tokens: int = 1,
    overlap_tokens: int = 0,
) -> list[str]:
    units = [unit for unit in _logical_units(text, max_tokens) if unit.strip()]
    if not units:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = estimate_token_count(unit)
        if current and current_tokens + unit_tokens > max_tokens:
            chunks.append("\n\n".join(current).strip())
            current = _overlap_tail(current, overlap_tokens)
            current_tokens = sum(estimate_token_count(item) for item in current)
            while current and current_tokens + unit_tokens > max_tokens:
                removed = current.pop(0)
                current_tokens -= estimate_token_count(removed)
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append("\n\n".join(current).strip())

    if len(chunks) > 1 and estimate_token_count(chunks[-1]) < min_tokens:
        combined = f"{chunks[-2]}\n\n{chunks[-1]}"
        if estimate_token_count(combined) <= max_tokens:
            chunks[-2:] = [combined]
    return chunks


def _pack_fragments(
    fragments: list[_Fragment],
    *,
    max_tokens: int,
    min_tokens: int,
    overlap_tokens: int,
) -> list[list[_Fragment]]:
    if not fragments:
        return []

    groups: list[list[_Fragment]] = []
    current: list[_Fragment] = []
    current_tokens = 0

    for fragment in fragments:
        if current and current_tokens + fragment.token_count > max_tokens:
            groups.append(current)
            current = _overlap_fragment_tail(current, overlap_tokens)
            current_tokens = sum(item.token_count for item in current)
            while current and current_tokens + fragment.token_count > max_tokens:
                removed = current.pop(0)
                current_tokens -= removed.token_count
        current.append(fragment)
        current_tokens += fragment.token_count

    if current:
        groups.append(current)

    if len(groups) > 1 and sum(item.token_count for item in groups[-1]) < min_tokens:
        combined_tokens = sum(item.token_count for item in (*groups[-2], *groups[-1]))
        if combined_tokens <= max_tokens:
            groups[-2:] = [[*groups[-2], *groups[-1]]]
    return groups


def _logical_units(text: str, max_tokens: int) -> list[str]:
    paragraphs = re.split(r"\n{2,}", text.strip())
    units: list[str] = []
    for paragraph in paragraphs:
        if estimate_token_count(paragraph) <= max_tokens:
            units.append(paragraph)
            continue
        for line in paragraph.splitlines():
            if estimate_token_count(line) <= max_tokens:
                units.append(line)
                continue
            for sentence in _SENTENCE_BOUNDARY.split(line):
                if not sentence.strip():
                    continue
                if estimate_token_count(sentence) <= max_tokens:
                    units.append(sentence)
                else:
                    units.extend(_hard_split(sentence, max_tokens))
    return units


def _hard_split(text: str, max_tokens: int) -> list[str]:
    matches = list(_TOKEN_PATTERN.finditer(text))
    if not matches:
        return []
    pieces: list[str] = []
    for start in range(0, len(matches), max_tokens):
        group = matches[start : start + max_tokens]
        pieces.append(text[group[0].start() : group[-1].end()].strip())
    return pieces


def _overlap_tail(units: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0:
        return []
    result: list[str] = []
    total = 0
    for unit in reversed(units):
        tokens = estimate_token_count(unit)
        if result and total + tokens > overlap_tokens:
            break
        result.append(unit)
        total += tokens
        if total >= overlap_tokens:
            break
    return list(reversed(result))


def _overlap_fragment_tail(fragments: list[_Fragment], overlap_tokens: int) -> list[_Fragment]:
    if overlap_tokens <= 0:
        return []
    result: list[_Fragment] = []
    total = 0
    for fragment in reversed(fragments):
        if result and total + fragment.token_count > overlap_tokens:
            break
        result.append(fragment)
        total += fragment.token_count
        if total >= overlap_tokens:
            break
    return list(reversed(result))
