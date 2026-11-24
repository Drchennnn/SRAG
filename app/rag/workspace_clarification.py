"""Stage 3: Structured, resumable clarification options and security-verified continuation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from app.rag.workspace_evidence import format_object_option_label
from app.schemas.local import SourceKey
from app.schemas.turn_evidence import TurnEvidence
from app.schemas.workspace import ClarificationAnchor, DisplayAnchor, WorkspaceError, inaccessible
from app.stores.structure_snapshot import SnapshotDocument, SnapshotSource

if TYPE_CHECKING:
    from app.rag.workspace import Catalog, WorkspaceService
    from app.rag.workspace_context import TurnContext
    from app.stores.workspace import Session, Turn


class HistoricalComparisonRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str
    evidence_id: str
    display_id: str | None = None
    source: SourceKey
    course_id: str
    publication_epoch: int
    content_sha256: str


class ComparisonResumeContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_question: str
    historical_ref: HistoricalComparisonRef
    is_image: bool
    attribute: str | None = None
    inherited_image: bool = False


class ClarificationOptionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    option_id: str
    label: str
    kind: Literal["display", "document"]
    resume_action: Literal[
        "follow_display",
        "read_object",
        "retrieve_document",
        "compare_versions",
    ]
    message_id: str | None = None
    display_id: str | None = None
    node_id: str | None = None
    target_source: SourceKey | None = None
    current_object: bool = False
    comparison_context: ComparisonResumeContext | None = None


def normalize_option_record(raw: Mapping[str, Any]) -> ClarificationOptionRecord:
    """Parse structured or legacy persisted clarification options with backward compatibility."""
    if "resume_action" in raw:
        return ClarificationOptionRecord.model_validate(raw)
    kind = raw.get("kind")
    if kind == "display":
        return ClarificationOptionRecord(
            option_id=raw["option_id"],
            label=raw["label"],
            kind="display",
            resume_action="follow_display",
            message_id=raw.get("message_id"),
            display_id=raw.get("display_id"),
        )
    is_current_obj = bool(raw.get("current_object"))
    return ClarificationOptionRecord(
        option_id=raw["option_id"],
        label=raw["label"],
        kind="document",
        resume_action="read_object" if is_current_obj else "retrieve_document",
        node_id=raw.get("node_id"),
        current_object=is_current_obj,
    )


def project_public_options(options: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Expose only public option_id/label/kind fields to API consumers."""
    return [
        {key: item[key] for key in ("option_id", "label", "kind")}
        for item in options
        if "option_id" in item and "label" in item and "kind" in item
    ]


def build_clarification(
    service: WorkspaceService,
    turn: Turn,
    text: str,
    options: Sequence[Mapping[str, Any]],
    *,
    refs: Sequence[dict[str, Any]] = (),
    historical_refs: Sequence[dict[str, Any]] = (),
    notice_code: str = "clarification",
    trace: dict[str, Any] | None = None,
) -> Turn:
    """Persist a clarification turn with separate current/historical access refs."""
    from app.rag.workspace import Draft

    pub_refs = list(refs)
    hist_refs = list(historical_refs)
    draft = Draft(turn.session_id)
    draft.notice(notice_code, text)
    outcome: dict[str, Any] = {
        "dependencies": [*pub_refs, *hist_refs],
        "options": [dict(item) for item in options],
    }
    if trace is not None:
        outcome["trace"] = trace
    return service.store.finish(
        turn,
        "clarification",
        outcome,
        draft.answer(),
        (),
        pub_refs,
        service.allowed_courses,
        historical_refs=hist_refs,
    )


def build_object_clarification_options(
    candidates: Sequence[SnapshotSource],
    docs: Mapping[str, SnapshotDocument],
    catalog: Catalog,
    *,
    is_displayable_fn: Any,
    base_label_fn: Any,
    doc_ref_fn: Any,
) -> list[dict[str, Any]]:
    """Build clickable read_object ClarificationOptionRecord dicts for up to 8 candidates."""
    nodes_by_key = {
        (entry.source.document_id, entry.source.version, entry.source.source_id): entry
        for entry in catalog.nodes.values()
        if entry.source is not None
    }
    options: list[dict[str, Any]] = []
    for source in candidates[:8]:
        doc = docs.get(source.document_id)
        entry = nodes_by_key.get((source.document_id, source.version, source.source_id))
        if doc is None or entry is None or doc_ref_fn(entry.document) != doc_ref_fn(doc):
            continue
        is_displayable = bool(is_displayable_fn(source))
        is_image_object = (
            source.kind == "notebook_output" and is_displayable
        ) or (
            source.kind == "page_object"
            and source.locator.get("object_type") in {"image", "picture", "chart"}
        ) or source.kind == "page"
        label = format_object_option_label(
            source,
            doc,
            candidates,
            base_label=base_label_fn(source),
            is_displayable=is_displayable,
        )
        record = ClarificationOptionRecord(
            option_id=uuid4().hex,
            label=label,
            kind="document",
            resume_action="read_object",
            node_id=entry.entry["node_id"],
            target_source=SourceKey(
                document_id=source.document_id,
                version=source.version,
                source_id=source.source_id,
            ),
            current_object=is_image_object,
        )
        options.append(record.model_dump(mode="json"))
    return options


def resume_clarification(
    service: WorkspaceService,
    owner: str,
    turn: Turn,
    ctx: TurnContext,
    anchor: ClarificationAnchor,
) -> Turn:
    """Re-verify ownership, scope, revision, and versions before resuming a clarified action."""
    from app.rag.workspace import _in_scope
    from app.rag.workspace_initial_focus import answer_current_image

    session: Session = ctx.session
    # Check 1: Session & owner binding, scope revision, and prior turn accessibility.
    previous = service.store.clarification(session.session_id, anchor.turn_id)
    if (
        previous is None
        or previous.request.scope_revision != session.scope_revision
        or not service._turn_access(previous)
    ):
        raise inaccessible()

    # Check 2: Option membership in the exact clarification turn.
    raw_option = next(
        (
            item
            for item in previous.outcome.get("options", [])
            if item.get("option_id") == anchor.option_id
        ),
        None,
    )
    if raw_option is None:
        raise inaccessible()
    try:
        option = normalize_option_record(raw_option)
    except ValueError as exc:
        raise inaccessible() from exc

    # Check 3: Course authorization re-check.
    if not set(session.course_ids) <= service.allowed_courses:
        raise inaccessible()

    if option.resume_action == "follow_display":
        if not option.message_id or not option.display_id:
            raise inaccessible()
        selected = DisplayAnchor(
            type="display", message_id=option.message_id, display_id=option.display_id
        )
        return service._follow_display(
            turn,
            session,
            selected,
            "clarification_choice",
            previous.request.text,
            owner=owner,
        )

    if option.resume_action == "compare_versions":
        return _resume_compare_versions(service, owner, turn, session, anchor, option)

    # Check 4: Current node resolution, scope membership, and target_source drift check.
    if not option.node_id:
        raise inaccessible()
    node = service._node(session, option.node_id, stale=option.target_source is not None)
    if not _in_scope(session, node.document):
        raise inaccessible()
    if option.target_source is not None:
        if node.source is None or node.key() != option.target_source:
            raise WorkspaceError("stale_reference")

    if option.resume_action == "read_object":
        if option.current_object:
            return answer_current_image(
                service, turn, session, node, previous.request.text, "current_choice"
            )
        if node.source is not None and node.source.kind == "page_object":
            return service._answer_selected_text_object(
                turn, session, node, previous.request.text
            )
        return service._retrieve(
            turn,
            session,
            previous.request.text,
            forced=node,
            kind=previous.request.kind,
            owner=owner,
        )

    if option.resume_action == "retrieve_document" and node.source is None:
        from app.rag.workspace_versions import compare_versions, is_version_comparison

        if is_version_comparison(previous.request.text):
            return compare_versions(
                service,
                owner,
                turn,
                session,
                question=previous.request.text,
                forced_file_name=node.document.file_name,
            )

    return service._retrieve(
        turn,
        session,
        previous.request.text,
        forced=node,
        kind=previous.request.kind,
        owner=owner,
    )


def _resume_compare_versions(
    service: WorkspaceService,
    owner: str,
    turn: Turn,
    session: Session,
    anchor: ClarificationAnchor,
    option: ClarificationOptionRecord,
) -> Turn:
    from app.rag.workspace import _in_scope
    from app.rag.workspace_versions import (
        _as_snapshot,
        _completed_history,
        _Previous,
        execute_paired_comparison,
    )

    comp = option.comparison_context
    if comp is None or not option.node_id:
        raise inaccessible()
    href = comp.historical_ref

    # Check 3b: Historical course authorization and scope membership.
    if href.course_id not in service.allowed_courses or href.course_id not in session.course_ids:
        raise inaccessible()

    # Check 4: Re-resolve v2 current node and verify scope/version/target_source.
    node = service._node(session, option.node_id, stale=True)
    if not _in_scope(session, node.document):
        raise inaccessible()
    if node.source is None:
        raise WorkspaceError("stale_reference")
    if option.target_source is not None and node.key() != option.target_source:
        raise WorkspaceError("stale_reference")
    if (
        node.document.document_id != href.source.document_id
        or node.document.version == href.source.version
    ):
        raise WorkspaceError("stale_reference")

    # Check 5: Re-verify v1 historical message belongs to same owner & same session & completed.
    stored_hist = service.store.completed_answer(owner, session.session_id, href.message_id)
    if stored_hist is None or not service._turn_access(stored_hist[0]):
        raise inaccessible()
    _, hist_msg = stored_hist
    ev_payload = (hist_msg.get("sources") or {}).get(href.evidence_id)
    if ev_payload is None:
        raise inaccessible()
    ev = TurnEvidence.model_validate(ev_payload)
    rev = ev.revision
    if (
        rev.source != href.source
        or rev.course_id != href.course_id
        or rev.publication_epoch != href.publication_epoch
        or rev.content_sha256 != href.content_sha256
    ):
        raise inaccessible()

    # Check 6: Re-verify v1 archived original bytes & attachment ownership/integrity.
    original = service._historical_source(
        {"course_id": href.course_id, "content_sha256": href.content_sha256}, href.source
    )
    if original.source is None:
        raise WorkspaceError("original_unavailable", 409)
    attachment = None
    if comp.is_image:
        if not href.display_id:
            raise inaccessible()
        attachment = service._attachment(session.session_id, href.message_id, href.display_id)
        if (
            attachment.description.get("fallback_reason") is not None
            or SourceKey.model_validate(attachment.description["display_source"]) != href.source
        ):
            raise inaccessible()

    # Check 7: Reconstruct paired historical/current objects and execute bilateral comparison.
    past = _Previous(
        message_id=href.message_id,
        evidence=ev,
        source=_as_snapshot(original.source, href.course_id, href.source),
        original=original,
        attachment=attachment,
    )
    history = _completed_history(service, owner, turn, session)
    return execute_paired_comparison(
        service,
        owner,
        turn,
        session,
        past=past,
        document=node.document,
        selected=node.source,
        history=history,
        question=comp.original_question,
        is_image=comp.is_image,
        attribute=comp.attribute,
        inherited_image=comp.inherited_image,
        resumed_from_turn_id=anchor.turn_id,
    )
