"""Notebook 解析、结构分组与切块使用的数据结构。"""

from enum import StrEnum

from pydantic import BaseModel, Field


class BlockKind(StrEnum):
    MARKDOWN = "markdown"
    CODE = "code"
    OUTPUT = "output"
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


class CellGroupKind(StrEnum):
    NARRATIVE = "narrative"
    EXECUTION = "execution"
    TABLE = "table"
    FIGURE = "figure"
    MIXED = "mixed"


class AssetRef(BaseModel):
    """指向已提取图片或原始本地资源。"""

    asset_id: str
    media_type: str
    path: str
    caption: str | None = None


class ContentBlock(BaseModel):
    """解析器输出的最小可溯源内容单元。"""

    block_id: str
    document_id: str
    kind: BlockKind
    text: str = ""
    section_path: list[str] = Field(default_factory=list)
    heading_level: int | None = None
    cell_id: str | None = None
    cell_index: int | None = None
    page: int | None = None
    related_block_id: str | None = None
    assets: list[AssetRef] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    """与文件格式无关的解析结果。"""

    document_id: str
    source_path: str
    title: str
    blocks: list[ContentBlock]


class CellGroup(BaseModel):
    """不可随意拆散的 Markdown 单元或代码执行单元。"""

    group_id: str
    document_id: str
    kind: CellGroupKind
    display_text: str
    embedding_text: str
    block_ids: list[str]
    cell_ids: list[str] = Field(default_factory=list)
    section_path: list[str] = Field(default_factory=list)
    asset_ids: list[str] = Field(default_factory=list)
    token_count: int


class SectionRecord(BaseModel):
    """Notebook 标题段的结构记录，不重复保存大段正文。"""

    section_id: str
    document_id: str
    heading: str | None = None
    section_path: list[str] = Field(default_factory=list)
    group_ids: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)


class NotebookChunk(BaseModel):
    """Notebook 中实际参与文本向量化的结构化内容块。"""

    chunk_id: str
    section_id: str
    document_id: str
    kind: CellGroupKind
    display_text: str
    embedding_text: str
    group_ids: list[str]
    block_ids: list[str]
    cell_ids: list[str] = Field(default_factory=list)
    section_path: list[str] = Field(default_factory=list)
    asset_ids: list[str] = Field(default_factory=list)
    token_count: int
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None


class NotebookChunkingResult(BaseModel):
    """Notebook 结构化切块的完整结果。"""

    document_id: str
    outline_text: str
    groups: list[CellGroup]
    sections: list[SectionRecord]
    chunks: list[NotebookChunk]
