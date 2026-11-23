"""Stage 1: Read-only Scope, Focus, and History state resolution for workspace turns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from app.rag.workspace_evidence import readable_source_text
from app.schemas.workspace import DisplayAnchor, WorkspaceError, inaccessible
from app.stores.structure_snapshot import (
    SnapshotDocument,
    SnapshotRequest,
    SnapshotSource,
    StructureSnapshot,
)

if TYPE_CHECKING:
    from app.rag.workspace import Catalog, CatalogNode, WorkspaceService
    from app.stores.workspace import Session, Turn


FocusStatus = Literal["none", "valid", "stale", "out_of_scope"]


@dataclass(frozen=True)
class FocusState:
    status: FocusStatus
    node: CatalogNode | None = None
    source: SnapshotSource | None = None
    child_images: tuple[SnapshotSource, ...] = ()
    child_outputs: tuple[SnapshotSource, ...] = ()
    stale_focus_id: str | None = None


@dataclass(frozen=True)
class HistorySummary:
    completed_turns: dict[str, Turn]
    messages: tuple[dict[str, Any], ...]
    boundary: int
    latest_visible_message: dict[str, Any] | None
    latest_single_vision_read: DisplayAnchor | None


@dataclass(frozen=True)
class TurnContext:
    owner: str
    turn: Turn
    session: Session
    scope_snapshot: StructureSnapshot
    documents: dict[str, SnapshotDocument]
    catalog: Catalog
    focus_state: FocusState
    history: HistorySummary


def resolve_turn_context(
    service: WorkspaceService,
    owner: str,
    turn: Turn,
    session: Session,
) -> TurnContext:
    """Assemble immutable turn context without guessing intent or triggering external I/O."""
    from app.rag.workspace import _can_display, _doc_ref, _in_scope

    if session.scope_revision != turn.request.scope_revision:
        raise WorkspaceError("scope_changed")
    if not set(session.course_ids) <= service.allowed_courses:
        raise inaccessible()

    scope_snapshot = service.snapshots.read(
        SnapshotRequest(tuple(session.course_ids), tuple(session.document_ids))
    )
    documents = {doc.document_id: doc for doc in scope_snapshot.documents}
    catalog = service._catalog(session.session_id)
    focus_state = _resolve_focus_state(
        session, catalog, scope_snapshot, documents, _in_scope, _doc_ref, _can_display
    )
    history = _resolve_history(service, owner, turn, session)
    return TurnContext(
        owner=owner,
        turn=turn,
        session=session,
        scope_snapshot=scope_snapshot,
        documents=documents,
        catalog=catalog,
        focus_state=focus_state,
        history=history,
    )


def _resolve_focus_state(
    session: Session,
    catalog: Catalog,
    snapshot: StructureSnapshot,
    documents: dict[str, SnapshotDocument],
    in_scope_fn: Any,
    doc_ref_fn: Any,
    can_display_fn: Any,
) -> FocusState:
    if not session.focus_id:
        return FocusState(status="none")
    node = catalog.nodes.get(session.focus_id)
    if node is None:
        return FocusState(status="stale", stale_focus_id=session.focus_id)
    if not in_scope_fn(session, node.document):
        return FocusState(status="out_of_scope", stale_focus_id=session.focus_id)
    current_doc = documents.get(node.document.document_id)
    if current_doc is None or doc_ref_fn(current_doc) != doc_ref_fn(node.document):
        return FocusState(status="stale", stale_focus_id=session.focus_id)

    focus_source = node.source
    doc_sources = [
        src
        for src in snapshot.sources
        if src.document_id == node.document.document_id and src.version == node.document.version
    ]
    child_images: list[SnapshotSource] = []
    child_outputs: list[SnapshotSource] = []
    if focus_source is not None and focus_source.kind == "page":
        child_images = sorted(
            (
                src
                for src in doc_sources
                if src.parent_id == focus_source.source_id
                and src.kind == "page_object"
                and src.locator.get("object_type") in {"image", "picture", "chart"}
            ),
            key=lambda s: s.ordinal,
        )
    elif focus_source is not None and focus_source.kind == "cell":
        outputs = sorted(
            (
                src
                for src in doc_sources
                if src.parent_id == focus_source.source_id
                and src.kind == "notebook_output"
                and (can_display_fn(src) or bool(readable_source_text(src).strip()))
            ),
            key=lambda s: (s.locator.get("output_index", s.ordinal), s.ordinal),
        )
        child_outputs = outputs
        child_images = [src for src in outputs if can_display_fn(src)]
    elif focus_source is None:
        child_images = sorted(
            (
                src
                for src in doc_sources
                if (
                    src.kind == "page_object"
                    and src.locator.get("object_type") in {"image", "picture", "chart"}
                )
                or (src.kind == "notebook_output" and can_display_fn(src))
            ),
            key=lambda s: s.ordinal,
        )

    return FocusState(
        status="valid",
        node=node,
        source=focus_source,
        child_images=tuple(child_images),
        child_outputs=tuple(child_outputs),
    )


def _resolve_history(
    service: WorkspaceService,
    owner: str,
    turn: Turn,
    session: Session,
) -> HistorySummary:
    _, turns, messages = service.store.timeline(owner, session.session_id)
    completed_turns = {
        item.turn_id: item
        for item in turns
        if item.status == "completed" and service._turn_access(item)
    }
    boundary = next(
        (
            message["ordinal"]
            for message in messages
            if message["turn_id"] == turn.turn_id and message["role"] == "user"
        ),
        len(messages) + 1,
    )
    latest_visible = next(
        (
            item
            for item in reversed(messages)
            if item["role"] == "assistant"
            and item["turn_id"] in completed_turns
            and item["ordinal"] < boundary
            and item.get("visible_display_ids")
        ),
        None,
    )
    latest_single_read = _latest_single_read_anchor(messages, completed_turns, boundary)
    return HistorySummary(
        completed_turns=completed_turns,
        messages=tuple(messages),
        boundary=boundary,
        latest_visible_message=latest_visible,
        latest_single_vision_read=latest_single_read,
    )


def _latest_single_read_anchor(
    messages: list[dict[str, Any]],
    completed_turns: dict[str, Turn],
    boundary: int,
) -> DisplayAnchor | None:
    for item in reversed(messages):
        if (
            item["role"] != "assistant"
            or item["turn_id"] not in completed_turns
            or item["ordinal"] >= boundary
        ):
            continue
        read_sources = [
            (eid, ev)
            for eid, ev in (item.get("sources") or {}).items()
            if any(
                r.get("method") == "vision" and r.get("status") in {"succeeded", "partial"}
                for r in ev.get("reads", [])
            )
        ]
        if len(read_sources) != 1:
            if item.get("visible_display_ids"):
                return None
            continue
        eid, _ = read_sources[0]
        block = next(
            (
                b
                for b in item.get("blocks", [])
                if b.get("type") == "source_image" and b.get("evidence_id") == eid
            ),
            None,
        )
        if block and block.get("display_id") in item.get("visible_display_ids", []):
            return DisplayAnchor(
                type="display",
                message_id=item["message_id"],
                display_id=block["display_id"],
            )
        prev_turn = completed_turns.get(item["turn_id"])
        res = (prev_turn.outcome.get("resolution") or {}) if prev_turn else {}
        if isinstance(res.get("message_id"), str) and isinstance(res.get("display_id"), str):
            return DisplayAnchor(
                type="display",
                message_id=res["message_id"],
                display_id=res["display_id"],
            )
        return None
    return None
