"""将 PDF 页与 PPTX 幻灯片按统一规则切块。"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from app.config import Settings
from app.rag.chunker import estimate_token_count, split_text_by_tokens
from app.schemas.document import (
    Slide,
    SlideChunk,
    SlideChunkingResult,
    SlideDeck,
    SlideElementKind,
    SlideSection,
)

_SPACE_PATTERN = re.compile(r"\s+")


class SlideChunker:
    """保留页边界，仅对超长单页做页内切分。"""

    def __init__(self, *, max_tokens: int = 650, overlap_tokens: int = 50) -> None:
        if max_tokens <= 0:
            raise ValueError("幻灯片 Chunk 最大 Token 数必须大于 0")
        if not 0 <= overlap_tokens < max_tokens:
            raise ValueError("幻灯片 Chunk 重叠 Token 数必须小于最大值")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    @classmethod
    def from_settings(cls, settings: Settings) -> SlideChunker:
        return cls(
            max_tokens=settings.slide_chunk_max_tokens,
            overlap_tokens=settings.slide_chunk_overlap_tokens,
        )

    def chunk(self, deck: SlideDeck) -> SlideChunkingResult:
        section_groups = _group_sections(deck.slides, deck.title)
        sections: list[SlideSection] = []
        chunks: list[SlideChunk] = []

        for section_index, (title, slides) in enumerate(section_groups):
            section_id = _stable_id(
                "slide-section", deck.document_id, section_index, title
            )
            section_chunks = [
                chunk
                for slide in slides
                for chunk in self._chunk_slide(deck, slide, section_id)
            ]
            sections.append(
                SlideSection(
                    section_id=section_id,
                    document_id=deck.document_id,
                    title=title,
                    slide_ids=[slide.slide_id for slide in slides],
                    chunk_ids=[chunk.chunk_id for chunk in section_chunks],
                )
            )
            chunks.extend(section_chunks)

        linked = [
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
        return SlideChunkingResult(
            document_id=deck.document_id,
            outline_text=_build_outline(deck, sections),
            sections=sections,
            chunks=linked,
        )

    def _chunk_slide(
        self, deck: SlideDeck, slide: Slide, section_id: str
    ) -> list[SlideChunk]:
        body = _render_slide(slide)
        content_limit = max(1, self.max_tokens - estimate_token_count(slide.title or ""))
        parts = split_text_by_tokens(
            body,
            content_limit,
            overlap_tokens=min(self.overlap_tokens, max(0, content_limit - 1)),
        ) or [""]
        element_ids = [element.element_id for element in slide.elements]
        asset_ids = [
            asset.asset_id
            for asset in slide.assets
            if asset.role.value != "decorative"
        ]
        chunks: list[SlideChunk] = []
        for index, part in enumerate(parts):
            embedding_text = slide_embedding_text(slide.title, part)
            chunks.append(
                SlideChunk(
                    chunk_id=_stable_id(
                        "slide-chunk",
                        deck.document_id,
                        slide.slide_id,
                        index,
                        embedding_text,
                    ),
                    section_id=section_id,
                    document_id=deck.document_id,
                    slide_id=slide.slide_id,
                    slide_no=slide.slide_no,
                    title=slide.title,
                    display_text=part,
                    embedding_text=embedding_text,
                    element_ids=element_ids,
                    asset_ids=asset_ids if index == 0 else [],
                    token_count=estimate_token_count(embedding_text),
                    split_index=index,
                    split_count=len(parts),
                )
            )
        return chunks


def _render_slide(slide: Slide) -> str:
    labels = {
        SlideElementKind.TITLE: "标题",
        SlideElementKind.TEXT: "正文",
        SlideElementKind.TABLE: "表格",
        SlideElementKind.IMAGE: "图片",
        SlideElementKind.CHART: "图表",
        SlideElementKind.FORMULA: "公式",
        SlideElementKind.OTHER: "其他",
    }
    rendered: list[str] = []
    for element in slide.elements:
        text = element.text.strip()
        if not text or (element.kind is SlideElementKind.TITLE and text == slide.title):
            continue
        rendered.append(f"[{labels[element.kind]}]\n{text}")
    return "\n\n".join(rendered).strip()


def _group_sections(
    slides: list[Slide], fallback_title: str
) -> list[tuple[str, list[Slide]]]:
    groups: list[tuple[str, list[Slide]]] = []
    for slide in slides:
        title = slide.title.strip() if slide.title and slide.title.strip() else fallback_title
        normalized = _SPACE_PATTERN.sub(" ", title).casefold()
        if groups and _SPACE_PATTERN.sub(" ", groups[-1][0]).casefold() == normalized:
            groups[-1][1].append(slide)
        else:
            groups.append((title, [slide]))
    return groups


def slide_embedding_text(title: str | None, body: str) -> str:
    """只把可检索的页标题和正文送入文本向量模型。"""

    return "\n\n".join(part for part in ((title or "").strip(), body.strip()) if part)


def _build_outline(deck: SlideDeck, sections: list[SlideSection]) -> str:
    lines = [f"文件：{Path(deck.source_path).name}", f"主题：{deck.title}"]
    lines.extend(f"- {section.title}" for section in sections if section.slide_ids)
    return "\n".join(lines)


def _stable_id(kind: str, *parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"{kind}_{digest}"
