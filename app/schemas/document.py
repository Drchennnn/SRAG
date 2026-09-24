"""所有文档解析器统一输出的数据结构。"""

from enum import StrEnum

from pydantic import BaseModel, Field


class BlockKind(StrEnum):
    MARKDOWN = "markdown"
    CODE = "code"
    OUTPUT = "output"
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


class AssetRef(BaseModel):
    """指向已提取图片、表格或其他原始资源的引用。"""

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


class ParentChunk(BaseModel):
    """在 SQLite 中只保存一次的父级上下文块。"""

    parent_id: str
    document_id: str
    text: str
    block_ids: list[str]
    section_path: list[str] = Field(default_factory=list)
    token_count: int


class ChildChunk(BaseModel):
    """存入 Qdrant 的可检索子块表示。"""

    chunk_id: str
    parent_id: str
    document_id: str
    text: str
    block_ids: list[str]
    section_path: list[str] = Field(default_factory=list)
    representation: str = "content"
    token_count: int
