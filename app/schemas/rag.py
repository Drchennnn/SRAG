"""索引与检索的请求和响应结构。"""

from datetime import datetime
from enum import StrEnum
from pathlib import Path

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


class IndexStatus(BaseModel):
    job_id: str
    state: JobState
    indexed_files: int = 0
    failed_files: int = 0
    errors: dict[str, str] = Field(default_factory=dict)


class SearchScope(BaseModel):
    course_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    scope: SearchScope = Field(default_factory=SearchScope)
    top_k: int = Field(default=8, ge=1, le=50)


class SourceReference(BaseModel):
    course_id: str
    document_id: str
    file: str
    section_path: list[str] = Field(default_factory=list)
    cell_ids: list[str] = Field(default_factory=list)
    cell_id: str | None = None
    cell_index: int | None = None
    page: int | None = None


class SearchHit(BaseModel):
    chunk_id: str
    section_id: str
    representation_type: str
    score: float
    content: str
    source: SourceReference


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


class DocumentSummary(BaseModel):
    document_id: str
    course_id: str
    source_path: str
    version: int
    release_at: datetime | None = None
    indexed_at: datetime


class DeleteDocumentResponse(BaseModel):
    document_id: str
    deleted: bool
