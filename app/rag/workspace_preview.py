"""Preview the original page or exact image behind a saved workspace citation."""

from __future__ import annotations

from app.rag.source_evidence import EvidenceRequest, HistoricalSourceGrant
from app.rag.workspace import WorkspaceService, _image_response
from app.rag.workspace_evidence import (
    has_native_table_read,
    has_native_text_read,
    verified_attachment_preview,
)
from app.schemas.turn_evidence import TurnEvidence
from app.schemas.workspace import WorkspaceError, inaccessible


def render_citation_preview(
    service: WorkspaceService, owner: str, session_id: str, message_id: str, evidence_id: str
) -> tuple[bytes, str]:
    stored = service.store.completed_answer(owner, session_id, message_id)
    if stored is None:
        raise inaccessible()
    turn, message = stored
    payload = message["sources"].get(evidence_id)
    if payload is None or not service._turn_access(turn):
        raise inaccessible()

    visible = set(message.get("visible_display_ids") or ())
    cited_in_text = any(
        block["type"] == "text" and evidence_id in block.get("citations", [])
        for block in message["blocks"]
    )
    attached_block = next(
        (
            block
            for block in message["blocks"]
            if block.get("type") == "source_image"
            and block.get("evidence_id") == evidence_id
            and block.get("display_id") in visible
        ),
        None,
    )
    if not cited_in_text and attached_block is None:
        raise inaccessible()

    evidence = TurnEvidence.model_validate(payload)
    revision = evidence.revision
    key = revision.source

    # Branch A: Same-message visible attachment for pure visual/image evidence or non-table
    # object_crop (captions/formulas where workspace.js opens openImageEvidence directly).
    # Native-table citations (including table_crop) intentionally bypass Branch A and use
    # Branch C (original_page) so workspace.js full-page normalized bbox highlights align.
    native_text = has_native_text_read(evidence)
    native_table = has_native_table_read(evidence)
    if attached_block is not None and (
        not native_text
        or (attached_block.get("rendition_kind") == "object_crop" and not native_table)
    ):
        return verified_attachment_preview(
            service,
            owner,
            session_id,
            message_id,
            attached_block["display_id"],
            evidence,
            turn,
            require_vision_read=False,
        )

    # Branch B: Historical display follow-up (_follow_display) referencing a prior message's image.
    resolution = turn.outcome.get("resolution") or {}
    origin_message = resolution.get("message_id")
    display_id = resolution.get("display_id")
    if origin_message and display_id and any(read.method == "vision" for read in evidence.reads):
        return verified_attachment_preview(
            service,
            owner,
            session_id,
            origin_message,
            display_id,
            evidence,
            turn,
            require_vision_read=True,
        )

    # Branch C: Native text / native table citations render original_page or notebook_output.
    authorized = service.historical_evidence(owner, session_id, message_id, evidence_id)
    if key.model_dump() != authorized["source"]:
        raise inaccessible()
    locator = authorized["locator"]
    if "page" in locator:
        rendition = "original_page"
    elif "output_index" in locator:
        rendition = "notebook_output"
    else:
        raise WorkspaceError("rendition_unavailable", 409)
    request = EvidenceRequest(
        source=key,
        expected_content_sha256=revision.content_sha256,
        rendition=rendition,
    )
    grant = HistoricalSourceGrant(
        key, (revision.course_id,), revision.content_sha256, message_id, evidence_id
    )
    result = service.evidence.render_historical(request, grant)
    if not service._turn_access(turn):
        raise inaccessible()
    return _image_response(result)
