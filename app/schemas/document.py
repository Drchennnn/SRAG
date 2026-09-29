"""Notebook 与幻灯片资料的解析、切块数据结构。"""

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


class SlideSourceFormat(StrEnum):
    """幻灯片型文档的原始格式。"""

    PDF = "pdf"
    PPTX = "pptx"


class SlideElementKind(StrEnum):
    """幻灯片中可按顺序追溯的内容类型。"""

    TITLE = "title"
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CHART = "chart"
    FORMULA = "formula"
    OTHER = "other"


class SlideAssetRole(StrEnum):
    """资源在检索中的信息价值。"""

    UNKNOWN = "unknown"
    DECORATIVE = "decorative"
    INFORMATIVE = "informative"


class NormalizedBoundingBox(BaseModel):
    """左上角为原点的 0..1 归一化坐标。"""

    left: float = Field(ge=0, le=1)
    top: float = Field(ge=0, le=1)
    right: float = Field(ge=0, le=1)
    bottom: float = Field(ge=0, le=1)

    @property
    def area_ratio(self) -> float:
        return max(0.0, self.right - self.left) * max(0.0, self.bottom - self.top)


class SlideAsset(BaseModel):
    """从单页中提取的图片或其他视觉资源。"""

    asset_id: str
    media_type: str
    path: str
    sha256: str
    bbox: NormalizedBoundingBox | None = None
    caption: str | None = None
    role: SlideAssetRole = SlideAssetRole.UNKNOWN
    role_reasons: list[str] = Field(default_factory=list)


class SlideElement(BaseModel):
    """单页内的文本、表格、图片或图表元素。"""

    element_id: str
    kind: SlideElementKind
    text: str = ""
    bbox: NormalizedBoundingBox | None = None
    asset_id: str | None = None


class Slide(BaseModel):
    """统一表示 PDF 页或 PPTX 幻灯片。"""

    slide_id: str
    document_id: str
    slide_no: int = Field(ge=1)
    title: str | None = None
    elements: list[SlideElement] = Field(default_factory=list)
    assets: list[SlideAsset] = Field(default_factory=list)
    page_image_path: str | None = None
    previous_slide_id: str | None = None
    next_slide_id: str | None = None
    visual_embedding_candidate: bool = False


class SlideDeck(BaseModel):
    """PDF/PPTX 共用的页级预处理结果。"""

    document_id: str
    source_path: str
    source_format: SlideSourceFormat
    title: str
    slides: list[Slide]


class SlideSection(BaseModel):
    """连续同标题页组成的轻量章节索引。"""

    section_id: str
    document_id: str
    title: str
    slide_ids: list[str]
    chunk_ids: list[str]


class SlideChunk(BaseModel):
    """实际参与文本向量化的幻灯片内容块。"""

    chunk_id: str
    section_id: str
    document_id: str
    slide_id: str
    slide_no: int
    title: str | None = None
    display_text: str
    embedding_text: str
    element_ids: list[str] = Field(default_factory=list)
    asset_ids: list[str] = Field(default_factory=list)
    token_count: int
    split_index: int = 0
    split_count: int = 1
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None


class SlideChunkingResult(BaseModel):
    """幻灯片页级切块的完整结果。"""

    document_id: str
    outline_text: str
    sections: list[SlideSection]
    chunks: list[SlideChunk]
