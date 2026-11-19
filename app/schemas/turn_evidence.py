"""Versioned evidence exchanged by the workspace and bounded retrieval."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.local import SourceKey

ReadStatus = Literal["not_attempted", "succeeded", "partial", "failed"]
ReadMethod = Literal["native_text", "native_table", "ocr", "vision"]
ReadExecution = Literal["real_provider", "cache_reuse", "mock_protocol"]

_SHA256 = r"^[0-9a-f]{64}$"


class TurnScope(BaseModel):
    """The effective course and document IDs among the pinned publications."""

    course_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)

    @field_validator("course_ids", "document_ids")
    @classmethod
    def validate_ids(cls, ids: list[str]) -> list[str]:
        if any(not value.strip() for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("scope IDs must be nonempty and unique")
        return ids


class SourceRevision(BaseModel):
    source: SourceKey
    course_id: str = Field(min_length=1)
    publication_epoch: int = Field(ge=0)
    content_sha256: str = Field(pattern=_SHA256)


class TurnRead(BaseModel):
    actual_source: SourceKey
    method: ReadMethod
    status: ReadStatus
    text: str = ""
    warnings: list[str] = Field(default_factory=list)
    execution: ReadExecution | None = None
    origin_execution: Literal["real_provider", "mock_protocol"] | None = None
    actual_input_sha256: str | None = Field(default=None, pattern=_SHA256)
    actual_input_media_type: str | None = None
    actual_input_rect: tuple[float, float, float, float] | None = None
    model: str | None = None
    prompt_version: str | None = None
    question_sha256: str | None = Field(default=None, pattern=_SHA256)


class TurnEvidence(BaseModel):
    """One requested object and its actual reads. Display order belongs to E."""

    evidence_id: str = Field(min_length=1)
    revision: SourceRevision
    file_name: str
    locator: dict[str, Any] = Field(default_factory=dict)
    read_status: ReadStatus = "not_attempted"
    reads: list[TurnRead] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_actual_revisions(self) -> "TurnEvidence":
        expected = self.revision.source
        for read in self.reads:
            actual = read.actual_source
            if (actual.document_id, actual.version) != (expected.document_id, expected.version):
                raise ValueError("a read must stay in the requested document version")
        return self
