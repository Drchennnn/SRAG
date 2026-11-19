"""Requests and shared errors for the local workspace."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]


class WorkspaceError(Exception):
    def __init__(self, code: str, status: int = 409) -> None:
        super().__init__(code)
        self.code = code
        self.status = status


def inaccessible() -> WorkspaceError:
    return WorkspaceError("not_accessible", 404)


class WorkspaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateSession(WorkspaceRequest):
    title: str = Field(default="", max_length=100)
    course_ids: list[Identifier] = Field(default_factory=list, max_length=100)
    document_ids: list[Identifier] = Field(default_factory=list, max_length=100)


class UpdateScope(WorkspaceRequest):
    course_ids: list[Identifier] = Field(min_length=1, max_length=100)
    document_ids: list[Identifier] = Field(default_factory=list, max_length=100)
    scope_revision: int = Field(ge=1)


class UpdateFocus(WorkspaceRequest):
    node_id: Identifier | None
    scope_revision: int = Field(ge=1)


class DisplayAnchor(WorkspaceRequest):
    type: Literal["display"]
    message_id: Identifier
    display_id: Identifier


class ClarificationAnchor(WorkspaceRequest):
    type: Literal["clarification"]
    turn_id: Identifier
    option_id: Identifier


class TurnRequest(WorkspaceRequest):
    client_turn_id: Identifier
    kind: Literal["ask", "search", "summary"] = "ask"
    text: str = Field(default="", max_length=4000)
    anchor: Annotated[DisplayAnchor | ClarificationAnchor, Field(discriminator="type")] | None = (
        None
    )
    scope_revision: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_action(self) -> TurnRequest:
        self.text = self.text.strip()
        if self.kind == "search" and not self.text:
            raise ValueError("search requires a query")
        if self.kind == "ask" and not self.text and self.anchor is None:
            raise ValueError("ask requires text or an explicit anchor")
        if self.kind != "ask" and self.anchor is not None:
            raise ValueError("only ask accepts an anchor")
        return self
