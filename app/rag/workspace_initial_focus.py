"""Resolve local image references separately from historical message ordinals."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.rag.workspace_clarification import build_clarification, build_object_clarification_options
from app.rag.workspace_objects import ObjectIntent, object_choices, object_intent
from app.schemas.workspace import WorkspaceError, inaccessible
from app.stores.structure_snapshot import SnapshotRequest
from app.stores.workspace import Session, Turn

if TYPE_CHECKING:
    from app.rag.workspace import CatalogNode, WorkspaceService


def resolve_current_image(service: WorkspaceService, turn: Turn, session: Session) -> Turn:
    from app.rag.workspace import _can_display, _doc_ref, _effective, _in_scope, _key, _label

    if not set(session.course_ids) <= service.allowed_courses:
        raise inaccessible()
    focus = service._node(session, session.focus_id, stale=True) if session.focus_id else None
    if focus and not _in_scope(session, focus.document):
        raise inaccessible()
    document_ids = (focus.document.document_id,) if focus else tuple(session.document_ids)
    snapshot = service.snapshots.read(SnapshotRequest(tuple(session.course_ids), document_ids))
    docs = {doc.document_id: doc for doc in snapshot.documents}
    refs = [_doc_ref(doc) for doc in docs.values()]
    if snapshot.resolution_status != "resolved":
        if document_ids:
            raise WorkspaceError("stale_reference")
        return build_clarification(service, turn, "请先导入或选定带图片的原件。", [], refs=refs)
    if focus and _doc_ref(focus.document) != _doc_ref(docs[focus.document.document_id]):
        raise WorkspaceError("stale_reference")

    current = focus.source if focus else None
    intent = object_intent(turn.request.text, include_reference=True) or ObjectIntent(kind="image")
    if intent.kind != "image":
        return build_clarification(service, turn, "请明确要读取的图片。", [], refs=refs)
    all_images = object_choices(ObjectIntent(kind="image"), snapshot.sources, focus=current)
    choices = object_choices(intent, snapshot.sources, focus=current)
    distinct_docs = {source.document_id for source in all_images}
    catalog = service._catalog(session.session_id)
    nodes = {
        (entry.source.document_id, entry.source.version, entry.source.source_id): entry
        for entry in catalog.nodes.values()
        if entry.source
    }
    if len(choices) == 1 and len(distinct_docs) == 1:
        source = choices[0]
        node = nodes.get((source.document_id, source.version, source.source_id))
        if not node or _doc_ref(node.document) != _doc_ref(docs[source.document_id]):
            raise WorkspaceError("stale_reference")
        return answer_current_image(
            service, turn, session, node, turn.request.text, "current_focus"
        )

    display_candidates = choices if len(choices) > 1 else all_images
    options = build_object_clarification_options(
        display_candidates,
        docs,
        catalog,
        is_displayable_fn=_can_display,
        base_label_fn=_label,
        doc_ref_fn=_doc_ref,
    )
    trace = {
        "route": "target_object",
        "clarification_reason": "object_ambiguous" if len(options) >= 2 else "no_unique_image",
        "effective_scope": _effective(docs),
        "object_candidates": [_key(source).model_dump() for source in display_candidates],
        "external_call_count": 0,
    }
    prompt = (
        "当前范围内有多个候选图片，请选择你要读取的具体图片。"
        if len(options) >= 2
        else "请明确要读取的当前文件或图片。"
    )
    return build_clarification(
        service,
        turn,
        prompt,
        options,
        refs=refs,
        notice_code="object_ambiguous" if len(options) >= 2 else "clarification",
        trace=trace,
    )


def answer_current_image(
    service: WorkspaceService,
    turn: Turn,
    session: Session,
    node: CatalogNode,
    question: str,
    method: str,
) -> Turn:
    from app.rag.workspace import _BARE_REFERENCE, Draft, _doc_ref, _in_scope

    if node.source is None or node.document.course_id not in service.allowed_courses:
        raise inaccessible()
    if not _in_scope(session, node.document):
        raise inaccessible()
    if _BARE_REFERENCE.fullmatch(question):
        question = "请描述图中实际可见的内容，尤其是坐标轴名称和单位（如有）。"
    trace = {
        "route": "target_object",
        "selected_source": node.key().model_dump(),
        "external_call_count": 0,
    }
    draft = Draft(session.session_id)
    if error := service._read_selected_image(
        turn, draft, node.document, node.source, question, trace
    ):
        return service._finish_invalid_read(turn, error, trace)
    return service._complete(
        turn,
        draft,
        refs=[_doc_ref(node.document)],
        resolution={
            "method": method,
            "resolution": method,
            "actual_source": node.key().model_dump(),
        },
        trace=trace,
    )
