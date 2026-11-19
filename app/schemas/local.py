"""Source structures shared by local parsers and the SQLite publisher."""

from typing import Any, Literal

from pydantic import BaseModel, Field

SourceKind = Literal["page", "page_object", "cell", "notebook_output"]


class SourceDraft(BaseModel):
    source_id: str
    kind: SourceKind
    ordinal: int = Field(ge=0)
    parent_id: str | None = None
    title: str = ""
    text: str = ""
    locator: dict[str, Any] = Field(default_factory=dict)
    state: str = "available"


class ParsedSource(BaseModel):
    title: str
    sources: list[SourceDraft]


class SourceKey(BaseModel):
    document_id: str
    version: int = Field(gt=0)
    source_id: str


class LocalHit(SourceKey):
    course_id: str
    publication_epoch: int
    content_sha256: str
    source_path: str
    locator: dict[str, Any]
    origin: str
    title: str
    text: str
    score: float


class ImportResult(BaseModel):
    status: Literal["published", "scheduled", "skipped", "backfilled", "failed"]
    source_path: str
    document_id: str | None = None
    version: int | None = None
    publication_epoch: int | None = None
    content_sha256: str | None = None
    source_count: int = 0
    searchable_count: int = 0
    error: str | None = None
