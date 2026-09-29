"""基于 Notebook 标题、cell 和执行关系构建结构化内容块。"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.schemas.document import (
    BlockKind,
    CellGroup,
    CellGroupKind,
    ContentBlock,
    NotebookChunk,
    NotebookChunkingResult,
    ParsedDocument,
    SectionRecord,
)

_TOKEN_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[^\s]"
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])|(?<=[.!?])\s+")
_ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_MEMORY_ADDRESS_PATTERN = re.compile(r"0x[0-9A-Fa-f]{6,}")

_BLOCK_LABELS = {
    BlockKind.MARKDOWN: "说明",
    BlockKind.CODE: "代码",
    BlockKind.OUTPUT: "运行输出",
    BlockKind.TEXT: "正文",
    BlockKind.TABLE: "表格",
    BlockKind.FIGURE: "图表",
}


def estimate_token_count(text: str) -> int:
    """估算中英文、代码和标点组成的 Token 数。"""

    return len(_TOKEN_PATTERN.findall(text))


@dataclass(frozen=True, slots=True)
class _GroupPiece:
    display_text: str
    embedding_text: str
    group_ids: tuple[str, ...]
    block_ids: tuple[str, ...]
    cell_ids: tuple[str, ...]
    asset_ids: tuple[str, ...]
    kinds: tuple[CellGroupKind, ...]

    @property
    def token_count(self) -> int:
        return estimate_token_count(self.embedding_text)


class NotebookChunker:
    """构建标题段、cell group 与自适应 Notebook chunk。"""

    def __init__(
        self,
        *,
        soft_min_tokens: int = 120,
        target_tokens: int = 280,
        max_tokens: int = 450,
        overlap_tokens: int = 40,
    ) -> None:
        if not 0 < soft_min_tokens <= target_tokens <= max_tokens:
            raise ValueError("Notebook Chunk Token 范围无效")
        if not 0 <= overlap_tokens < max_tokens:
            raise ValueError("Notebook Chunk 重叠 Token 数必须小于最大 Token 数")
        self.soft_min_tokens = soft_min_tokens
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    @classmethod
    def from_settings(cls, settings: Settings) -> NotebookChunker:
        return cls(
            soft_min_tokens=settings.notebook_chunk_soft_min_tokens,
            target_tokens=settings.notebook_chunk_target_tokens,
            max_tokens=settings.notebook_chunk_max_tokens,
            overlap_tokens=settings.notebook_chunk_overlap_tokens,
        )

    def chunk(self, document: ParsedDocument) -> NotebookChunkingResult:
        """按标题段组织 cell group，并在段内进行自适应切块。"""

        groups = [
            group
            if group.section_path
            else group.model_copy(update={"section_path": [document.title]})
            for group in self._build_groups(document)
        ]
        section_groups = self._group_sections(groups)
        sections: list[SectionRecord] = []
        chunks: list[NotebookChunk] = []

        for section_index, (section_path, grouped) in enumerate(section_groups):
            section_id = _stable_id(
                "section",
                document.document_id,
                section_index,
                " > ".join(section_path),
            )
            section_chunks = self._build_section_chunks(
                document=document,
                section_id=section_id,
                section_path=section_path,
                groups=grouped,
            )
            sections.append(
                SectionRecord(
                    section_id=section_id,
                    document_id=document.document_id,
                    heading=section_path[-1] if section_path else None,
                    section_path=list(section_path),
                    group_ids=[group.group_id for group in grouped],
                    chunk_ids=[chunk.chunk_id for chunk in section_chunks],
                )
            )
            chunks.extend(section_chunks)

        linked_chunks = [
            chunk.model_copy(
                update={
                    "previous_chunk_id": chunks[index - 1].chunk_id if index else None,
                    "next_chunk_id": (
                        chunks[index + 1].chunk_id if index + 1 < len(chunks) else None
                    ),
                }
            )
            for index, chunk in enumerate(chunks)
        ]
        return NotebookChunkingResult(
            document_id=document.document_id,
            outline_text=_build_outline(document, sections),
            groups=groups,
            sections=sections,
            chunks=linked_chunks,
        )

    def _build_groups(self, document: ParsedDocument) -> list[CellGroup]:
        groups: list[CellGroup] = []
        blocks = document.blocks
        index = 0
        while index < len(blocks):
            block = blocks[index]
            grouped = [block]
            if block.kind is BlockKind.CODE:
                next_index = index + 1
                while next_index < len(blocks):
                    candidate = blocks[next_index]
                    if candidate.related_block_id != block.block_id:
                        break
                    grouped.append(candidate)
                    next_index += 1
                index = next_index
            elif block.related_block_id:
                next_index = index + 1
                while next_index < len(blocks):
                    candidate = blocks[next_index]
                    if candidate.related_block_id != block.related_block_id:
                        break
                    grouped.append(candidate)
                    next_index += 1
                index = next_index
            else:
                index += 1
            groups.append(self._make_group(document, grouped, len(groups)))
        return groups

    @staticmethod
    def _make_group(
        document: ParsedDocument,
        blocks: list[ContentBlock],
        group_index: int,
    ) -> CellGroup:
        display_text = "\n\n".join(_render_block(block) for block in blocks).strip()
        embedding_text = _clean_embedding_text(display_text)
        kind = _group_kind(blocks)
        block_ids = [block.block_id for block in blocks]
        cell_ids = _unique(block.cell_id for block in blocks if block.cell_id)
        asset_ids = _unique(asset.asset_id for block in blocks for asset in block.assets)
        section_path = _common_prefix([tuple(block.section_path) for block in blocks])
        return CellGroup(
            group_id=_stable_id(
                "group", document.document_id, group_index, *block_ids, embedding_text
            ),
            document_id=document.document_id,
            kind=kind,
            display_text=display_text,
            embedding_text=embedding_text,
            block_ids=block_ids,
            cell_ids=cell_ids,
            section_path=section_path,
            asset_ids=asset_ids,
            token_count=estimate_token_count(embedding_text),
        )

    @staticmethod
    def _group_sections(
        groups: list[CellGroup],
    ) -> list[tuple[tuple[str, ...], list[CellGroup]]]:
        sections: list[tuple[tuple[str, ...], list[CellGroup]]] = []
        for group in groups:
            section_path = tuple(group.section_path)
            if sections and sections[-1][0] == section_path:
                sections[-1][1].append(group)
            else:
                sections.append((section_path, [group]))
        return sections

    def _build_section_chunks(
        self,
        *,
        document: ParsedDocument,
        section_id: str,
        section_path: tuple[str, ...],
        groups: list[CellGroup],
    ) -> list[NotebookChunk]:
        prefix = _embedding_prefix(document, section_path)
        prefix_tokens = estimate_token_count(prefix)
        content_max = max(1, self.max_tokens - prefix_tokens)
        content_target = max(1, self.target_tokens - prefix_tokens)
        content_soft_min = max(1, self.soft_min_tokens - prefix_tokens)

        pieces = [
            piece
            for group in groups
            for piece in self._split_group(group, content_max)
            if piece.embedding_text.strip()
        ]
        packed: list[list[_GroupPiece]] = []
        current: list[_GroupPiece] = []
        current_tokens = 0
        for piece in pieces:
            exceeds_max = current and current_tokens + piece.token_count > content_max
            reached_target = current and current_tokens >= content_target
            if exceeds_max or reached_target:
                packed.append(current)
                current = []
                current_tokens = 0
            current.append(piece)
            current_tokens += piece.token_count
        if current:
            packed.append(current)

        if len(packed) > 1:
            last_tokens = sum(piece.token_count for piece in packed[-1])
            previous_tokens = sum(piece.token_count for piece in packed[-2])
            if last_tokens < content_soft_min and previous_tokens + last_tokens <= content_max:
                packed[-2:] = [[*packed[-2], *packed[-1]]]

        chunks: list[NotebookChunk] = []
        for chunk_index, chunk_pieces in enumerate(packed):
            display_text = "\n\n".join(piece.display_text for piece in chunk_pieces).strip()
            body = "\n\n".join(piece.embedding_text for piece in chunk_pieces).strip()
            embedding_text = f"{prefix}\n\n{body}" if prefix else body
            kinds = {kind for piece in chunk_pieces for kind in piece.kinds}
            kind = next(iter(kinds)) if len(kinds) == 1 else CellGroupKind.MIXED
            chunks.append(
                NotebookChunk(
                    chunk_id=_stable_id(
                        "chunk",
                        document.document_id,
                        section_id,
                        chunk_index,
                        embedding_text,
                    ),
                    section_id=section_id,
                    document_id=document.document_id,
                    kind=kind,
                    display_text=display_text,
                    embedding_text=embedding_text,
                    group_ids=_unique(
                        group_id for piece in chunk_pieces for group_id in piece.group_ids
                    ),
                    block_ids=_unique(
                        block_id for piece in chunk_pieces for block_id in piece.block_ids
                    ),
                    cell_ids=_unique(
                        cell_id for piece in chunk_pieces for cell_id in piece.cell_ids
                    ),
                    section_path=list(section_path),
                    asset_ids=_unique(
                        asset_id for piece in chunk_pieces for asset_id in piece.asset_ids
                    ),
                    token_count=estimate_token_count(embedding_text),
                )
            )
        return chunks

    def _split_group(self, group: CellGroup, max_tokens: int) -> list[_GroupPiece]:
        if group.token_count <= max_tokens:
            return [_group_piece(group)]
        texts = split_text_by_tokens(
            group.display_text,
            max_tokens,
            overlap_tokens=min(self.overlap_tokens, max(0, max_tokens - 1)),
        )
        return [
            _GroupPiece(
                display_text=text,
                embedding_text=_clean_embedding_text(text),
                group_ids=(group.group_id,),
                block_ids=tuple(group.block_ids),
                cell_ids=tuple(group.cell_ids),
                asset_ids=tuple(group.asset_ids) if index == 0 else (),
                kinds=(group.kind,),
            )
            for index, text in enumerate(texts)
        ]


def _group_piece(group: CellGroup) -> _GroupPiece:
    return _GroupPiece(
        display_text=group.display_text,
        embedding_text=group.embedding_text,
        group_ids=(group.group_id,),
        block_ids=tuple(group.block_ids),
        cell_ids=tuple(group.cell_ids),
        asset_ids=tuple(group.asset_ids),
        kinds=(group.kind,),
    )


def _group_kind(blocks: list[ContentBlock]) -> CellGroupKind:
    kinds = {block.kind for block in blocks}
    if BlockKind.CODE in kinds:
        return CellGroupKind.EXECUTION
    if kinds == {BlockKind.TABLE}:
        return CellGroupKind.TABLE
    if kinds == {BlockKind.FIGURE}:
        return CellGroupKind.FIGURE
    if kinds <= {BlockKind.MARKDOWN, BlockKind.TEXT}:
        return CellGroupKind.NARRATIVE
    return CellGroupKind.MIXED


def _render_block(block: ContentBlock) -> str:
    label = _BLOCK_LABELS[block.kind]
    body = block.text.strip()
    resources = [
        f"资源：{asset.caption or Path(asset.path).name}（{asset.media_type}）"
        for asset in block.assets
    ]
    if resources:
        body = "\n".join(part for part in (body, *resources) if part)
    if block.kind is BlockKind.CODE:
        return f"[{label}]\n```python\n{body}\n```"
    return f"[{label}]\n{body}".strip()


def _clean_embedding_text(text: str) -> str:
    text = _ANSI_PATTERN.sub("", text)
    text = _MEMORY_ADDRESS_PATTERN.sub("<memory-address>", text)
    lines: list[str] = []
    blank = False
    for line in text.splitlines():
        stripped = line.rstrip()
        if not stripped:
            if not blank:
                lines.append("")
            blank = True
            continue
        lines.append(stripped)
        blank = False
    return "\n".join(lines).strip()


def _embedding_prefix(document: ParsedDocument, section_path: tuple[str, ...]) -> str:
    parts = [f"文件：{Path(document.source_path).name}", f"主题：{document.title}"]
    if section_path:
        parts.append(f"章节：{' > '.join(section_path)}")
    return "\n".join(parts)


def _build_outline(document: ParsedDocument, sections: list[SectionRecord]) -> str:
    headings = _unique(
        " > ".join(section.section_path)
        for section in sections
        if section.section_path
    )
    outline_blocks = [
        _clean_embedding_text(block.text)
        for block in document.blocks
        if block.kind is BlockKind.MARKDOWN
        and block.section_path
        and block.section_path[-1].strip().lower() in {"outline", "目录", "大纲"}
    ]
    parts = [
        f"文件：{Path(document.source_path).name}",
        f"主题：{document.title}",
    ]
    if headings:
        parts.append("章节：\n" + "\n".join(f"- {heading}" for heading in headings))
    if outline_blocks:
        parts.append("课程大纲：\n" + "\n\n".join(outline_blocks))
    return "\n\n".join(parts)


def split_text_by_tokens(
    text: str, max_tokens: int, *, overlap_tokens: int = 0
) -> list[str]:
    """按段落、行和句子边界切分文本，必要时按 Token 硬切。"""
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
                current_tokens -= estimate_token_count(current.pop(0))
        current.append(unit)
        current_tokens += unit_tokens
    if current:
        chunks.append("\n\n".join(current).strip())
    return chunks


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
    return [
        text[group[0].start() : group[-1].end()].strip()
        for start in range(0, len(matches), max_tokens)
        if (group := matches[start : start + max_tokens])
    ]


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


def _common_prefix(paths: list[tuple[str, ...]]) -> list[str]:
    if not paths:
        return []
    prefix: list[str] = []
    for parts in zip(*paths, strict=False):
        if len(set(parts)) != 1:
            break
        prefix.append(parts[0])
    return prefix


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _stable_id(kind: str, *parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"{kind}_{digest}"
