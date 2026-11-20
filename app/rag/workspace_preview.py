"""Preview the original page or exact image behind a saved workspace citation."""

from hashlib import sha256

from app.rag.source_evidence import EvidenceRequest, HistoricalSourceGrant
from app.rag.workspace import WorkspaceService, _image_response
from app.schemas.local import SourceKey
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
    cited = any(
        block["type"] == "text" and evidence_id in block.get("citations", [])
        for block in message["blocks"]
    )
    if payload is None or not cited or not service._turn_access(turn):
        raise inaccessible()
    evidence = TurnEvidence.model_validate(payload)
    revision = evidence.revision
    key = revision.source
    resolution = turn.outcome.get("resolution") or {}
    origin_message = resolution.get("message_id")
    display_id = resolution.get("display_id")
    if origin_message and display_id and any(read.method == "vision" for read in evidence.reads):
        image = _read_image(service, owner, session_id, origin_message, display_id, evidence)
        if not service._turn_access(turn):
            raise inaccessible()
        return image

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


def _read_image(
    service: WorkspaceService,
    owner: str,
    session_id: str,
    origin_message: str,
    display_id: str,
    evidence: TurnEvidence,
) -> tuple[bytes, str]:
    """Resolve a follow-up citation back to the exact bytes used for its read."""
    revision = evidence.revision
    attached = service._attachment(session_id, origin_message, display_id)
    service.store.display_ordinal(session_id, origin_message, display_id)
    details = attached.description
    if (
        SourceKey.model_validate(details["display_source"]) != revision.source
        or details["course_id"] != revision.course_id
        or details["content_sha256"] != revision.content_sha256
        or details["publication_epoch"] != revision.publication_epoch
    ):
        raise inaccessible()
    expected = next(
        (
            read.actual_input_sha256
            for read in evidence.reads
            if read.method == "vision"
            and read.status in {"succeeded", "partial"}
            and read.origin_execution == "real_provider"
            and read.actual_source == revision.source
            and read.actual_input_media_type == details["mime_type"]
            and read.actual_input_sha256 == details["byte_sha256"]
        ),
        None,
    )
    if not expected:
        raise inaccessible()
    image, mime_type = service.historical_media(
        owner, session_id, origin_message, display_id, view="inline"
    )
    if sha256(image).hexdigest() != expected or mime_type != details["mime_type"]:
        raise WorkspaceError("rendition_unavailable", 409)
    return image, mime_type
