"""切块和向量化完成后、写入存储前的数据结构。"""

from enum import StrEnum

from pydantic import BaseModel, Field

from app.schemas.document import CellGroup, NotebookChunk, SectionRecord
from app.schemas.embedding import SparseValue


class RepresentationType(StrEnum):
    DOCUMENT_OUTLINE = "document_outline"
    TEXT_CONTENT = "text_content"
    VISUAL_FUSION = "visual_fusion"


class TextRepresentation(BaseModel):
    representation_id: str
    representation_type: RepresentationType
    source_id: str
    document_id: str
    text: str


class VisualRepresentation(BaseModel):
    representation_id: str
    representation_type: RepresentationType = RepresentationType.VISUAL_FUSION
    source_id: str
    document_id: str
    chunk_id: str
    image_path: str
    media_type: str
    context_text: str


class EmbeddedTextRepresentation(BaseModel):
    representation: TextRepresentation
    dense: list[float]
    sparse: list[SparseValue] = Field(default_factory=list)


class EmbeddedVisualRepresentation(BaseModel):
    representation: VisualRepresentation
    dense: list[float]


class PreparedDocumentIndex(BaseModel):
    """已经可以交给 SQLite 与 Qdrant 的单文档索引数据包。"""

    document_id: str
    source_path: str
    title: str
    groups: list[CellGroup]
    sections: list[SectionRecord]
    chunks: list[NotebookChunk]
    text_representations: list[EmbeddedTextRepresentation]
    visual_representations: list[EmbeddedVisualRepresentation] = Field(default_factory=list)
    skipped_visual_asset_ids: list[str] = Field(default_factory=list)
    text_embedding_model: str
    text_embedding_dimension: int
    visual_embedding_model: str
    visual_embedding_dimension: int
    text_total_tokens: int = 0
    visual_total_tokens: int = 0
    request_ids: list[str] = Field(default_factory=list)
