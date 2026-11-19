"""Resolve local image references separately from historical message ordinals."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from app.rag.workspace_objects import ObjectIntent, object_choices, object_intent
from app.schemas.workspace import WorkspaceError, inaccessible
from app.stores.structure_snapshot import SnapshotRequest
from app.stores.workspace import Session, Turn

if TYPE_CHECKING:
    from app.rag.workspace import CatalogNode, WorkspaceService


def resolve_current_image(service: WorkspaceService, turn: Turn, session: Session) -> Turn:
    from app.rag.workspace import _doc_ref, _in_scope, _label

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
        return service._clarify(turn, "请先导入或选定带图片的原件。", [], refs)
    if focus and _doc_ref(focus.document) != _doc_ref(docs[focus.document.document_id]):
        raise WorkspaceError("stale_reference")

    current = focus.source if focus else None
    intent = object_intent(turn.request.text, include_reference=True) or ObjectIntent(kind="image")
    if intent.kind != "image":
        return service._clarify(turn, "请明确要读取的图片。", [], refs)
    all_images = object_choices(ObjectIntent(kind="image"), snapshot.sources, focus=current)
    choices = object_choices(intent, snapshot.sources, focus=current)
    distinct_docs = {source.document_id for source in all_images}
    catalog_nodes = service._catalog(session.session_id).nodes.values()
    nodes = {
        (entry.source.document_id, entry.source.version, entry.source.source_id): entry
        for entry in catalog_nodes
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

    options = []
    for source in (choices if len(choices) > 1 else all_images)[:8]:
        entry = nodes.get((source.document_id, source.version, source.source_id))
        if entry and _doc_ref(entry.document) == _doc_ref(docs[source.document_id]):
            options.append(
                {
                    "option_id": uuid4().hex,
                    "label": (
                        f"{entry.document.file_name} · v{entry.document.version} · {_label(source)}"
                    ),
                    "kind": "document",
                    "node_id": entry.entry["node_id"],
                    "current_object": True,
                }
            )
    return service._clarify(turn, "请明确要读取的当前文件或图片。", options, refs)


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
