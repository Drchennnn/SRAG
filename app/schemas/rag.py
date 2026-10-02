"""索引与检索的请求和响应结构。"""

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class JobState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class IndexRequest(BaseModel):
    course_id: str
    paths: list[Path] = Field(min_length=1)
    release_at: datetime | None = None


class SlideIndexRequest(BaseModel):
    """导入已在 Colab 预处理并向量化的 PDF/PPTX。"""

    course_id: str
    source_path: Path
    bundle_dir: Path
    embeddings_path: Path
    release_at: datetime | None = None
    force: bool = False


class IndexStatus(BaseModel):
    job_id: str
    state: JobState
    indexed_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    errors: dict[str, str] = Field(default_factory=dict)


class SearchScope(BaseModel):
    course_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000, pattern=r"\S")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, pattern=r"\S")
    scope: SearchScope = Field(default_factory=SearchScope)
    top_k: int = Field(default=8, ge=1, le=50)
    retrieval_mode: Literal["auto", "text", "visual"] = "auto"
    query_construction: Literal["raw", "contextual"] = "raw"
    history: list[ConversationTurn] = Field(default_factory=list, max_length=6)


class SourceReference(BaseModel):
    course_id: str
    document_id: str
    file: str
    section_path: list[str] = Field(default_factory=list)
    cell_ids: list[str] = Field(default_factory=list)
    cell_id: str | None = None
    cell_index: int | None = None
    page: int | None = None
    image_path: str | None = None
    version: int | None = None


class SearchHit(BaseModel):
    chunk_id: str
    section_id: str
    representation_type: str
    score: float
    content: str
    source: SourceReference
    evidence_origin: Literal["course_text", "visual_extract"] = "course_text"
    origin_chunk_id: str | None = None


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    mode_used: Literal["text", "visual"] = "text"
    retrieval_query: str | None = None
    query_method: Literal["raw", "history_expanded"] = "raw"


class DocumentSummary(BaseModel):
    document_id: str
    course_id: str
    source_path: str
    version: int
    release_at: datetime | None = None
    indexed_at: datetime


class DocumentContent(BaseModel):
    """当前可见文档的完整文字结构；已读页面不等于已理解图片。"""

    document: DocumentSummary
    title: str
    outline: list[str]
    hits: list[SearchHit]
    pages: list[int] = Field(default_factory=list)
    image_only_pages: list[int] = Field(default_factory=list)


class DeleteDocumentResponse(BaseModel):
    document_id: str
    deleted: bool
