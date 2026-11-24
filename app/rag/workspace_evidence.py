"""Unified readable-text extraction, option labeling, and verified attachment preview."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from typing import TYPE_CHECKING

from app.schemas.local import SourceDraft, SourceKey
from app.schemas.turn_evidence import TurnEvidence, TurnRead
from app.schemas.workspace import WorkspaceError, inaccessible
from app.stores.structure_snapshot import SnapshotDocument, SnapshotSource

if TYPE_CHECKING:
    from app.rag.workspace import WorkspaceService
    from app.stores.workspace import Turn


def readable_source_text(source: SourceDraft | SnapshotSource) -> str:
    """Return authoritative native text, including locator['object_text'] for captions/formulas."""
    if source.text.strip():
        return source.text
    if (
        source.kind == "page_object"
        and source.locator.get("object_type") in {"caption", "formula"}
        and isinstance(object_text := source.locator.get("object_text"), str)
    ):
        return object_text
    return ""


def has_native_text_read(evidence: TurnEvidence) -> bool:
    return any(
        read.method in {"native_text", "native_table", "ocr"}
        and read.status in {"succeeded", "partial"}
        and bool(read.text)
        for read in evidence.reads
    )


def has_native_table_read(evidence: TurnEvidence) -> bool:
    return any(
        read.method == "native_table"
        and read.status in {"succeeded", "partial"}
        and bool(read.text)
        for read in evidence.reads
    )


def verified_vision_read(evidence: TurnEvidence) -> TurnRead | None:
    revision = evidence.revision
    return next(
        (
            read
            for read in evidence.reads
            if read.method == "vision"
            and read.status in {"succeeded", "partial"}
            and read.origin_execution == "real_provider"
            and read.actual_source == revision.source
            and bool(read.actual_input_sha256)
        ),
        None,
    )


def verified_attachment_preview(
    service: WorkspaceService,
    owner: str,
    session_id: str,
    attachment_message_id: str,
    display_id: str,
    evidence: TurnEvidence,
    turn: Turn,
    *,
    require_vision_read: bool = False,
) -> tuple[bytes, str]:
    """Return verified attachment bytes matching evidence revision and optional vision hash."""
    revision = evidence.revision
    attached = service._attachment(session_id, attachment_message_id, display_id)
    service.store.display_ordinal(session_id, attachment_message_id, display_id)
    details = attached.description
    if (
        SourceKey.model_validate(details["display_source"]) != revision.source
        or details["course_id"] != revision.course_id
        or details["content_sha256"] != revision.content_sha256
        or details["publication_epoch"] != revision.publication_epoch
    ):
        raise inaccessible()

    expected_sha = details["byte_sha256"]
    vision = verified_vision_read(evidence)
    if require_vision_read:
        if (
            vision is None
            or vision.actual_input_media_type != details["mime_type"]
            or vision.actual_input_sha256 != expected_sha
        ):
            raise inaccessible()
    elif vision is not None:
        if (
            vision.actual_input_media_type != details["mime_type"]
            or vision.actual_input_sha256 != expected_sha
        ):
            raise inaccessible()

    image, mime_type = service.historical_media(
        owner, session_id, attachment_message_id, display_id, view="inline"
    )
    if sha256(image).hexdigest() != expected_sha or mime_type != details["mime_type"]:
        raise WorkspaceError("rendition_unavailable", 409)
    if not service._turn_access(turn):
        raise inaccessible()
    return image, mime_type


def _valid_norm_bbox(source: SnapshotSource) -> tuple[float, float, float, float] | None:
    bbox = source.locator.get("bbox")
    if (
        isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and all(type(v) in {int, float} and 0 <= v <= 1 for v in bbox)
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    ):
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return None


def _spatial_tag(
    source: SnapshotSource, same_page_peers: Sequence[SnapshotSource]
) -> str | None:
    if len(same_page_peers) < 2:
        return None
    indexed: list[tuple[SnapshotSource, tuple[float, float, float, float]]] = []
    for peer in same_page_peers:
        rect = _valid_norm_bbox(peer)
        if rect is None:
            return None
        indexed.append((peer, rect))
    if len(indexed) == 2:
        (s0, r0), (s1, r1) = indexed
        cx0, cx1 = (r0[0] + r0[2]) / 2, (r1[0] + r1[2]) / 2
        cy0, cy1 = (r0[1] + r0[3]) / 2, (r1[1] + r1[3]) / 2
        h_inter = max(0.0, min(r0[2], r1[2]) - max(r0[0], r1[0]))
        v_inter = max(0.0, min(r0[3], r1[3]) - max(r0[1], r1[1]))
        min_w = min(r0[2] - r0[0], r1[2] - r1[0])
        min_h = min(r0[3] - r0[1], r1[3] - r1[1])
        h_overlap = h_inter / min_w if min_w > 0 else 1.0
        v_overlap = v_inter / min_h if min_h > 0 else 1.0
        if h_overlap < 0.20 and (cx0 < 0.5 <= cx1 or cx1 < 0.5 <= cx0 or abs(cx0 - cx1) >= 0.25):
            left_src = s0 if cx0 <= cx1 else s1
            return "左图" if source.source_id == left_src.source_id else "右图"
        if v_overlap < 0.20 and (cy0 < 0.5 <= cy1 or cy1 < 0.5 <= cy0 or abs(cy0 - cy1) >= 0.20):
            top_src = s0 if cy0 <= cy1 else s1
            return "上图" if source.source_id == top_src.source_id else "下图"
    ordered = sorted(indexed, key=lambda pair: (pair[1][1], pair[1][0], pair[0].ordinal))
    for pos, (peer, _) in enumerate(ordered, start=1):
        if peer.source_id == source.source_id:
            return f"图 {pos}"
    return None


def format_object_option_label(
    source: SnapshotSource,
    doc: SnapshotDocument,
    peers: Sequence[SnapshotSource],
    *,
    base_label: str,
    is_displayable: bool,
) -> str:
    """Build a human-distinguishable clarification label with spatial or output cues."""
    if source.kind == "notebook_output":
        cell_num = int(source.locator.get("cell_index", 0)) + 1
        out_idx = int(source.locator.get("output_index", 0))
        if is_displayable:
            detail = "图像"
        else:
            snippet = (readable_source_text(source).strip().splitlines() or ["文本输出"])[0][:24]
            detail = snippet
        return f"{doc.file_name} · v{doc.version} · Cell {cell_num} · 输出 {out_idx}（{detail}）"

    page = source.locator.get("page")
    page_part = f"第 {page} 页" if isinstance(page, int) else None
    object_type = source.locator.get("object_type")
    if source.kind == "page_object" and object_type in {"image", "picture", "chart"}:
        same_page_peers = [
            peer
            for peer in peers
            if peer.document_id == source.document_id
            and peer.version == source.version
            and peer.locator.get("page") == page
            and peer.kind == "page_object"
            and peer.locator.get("object_type") in {"image", "picture", "chart"}
        ]
        tag = _spatial_tag(source, same_page_peers)
        title = source.title.strip()
        parts = [f"{doc.file_name} · v{doc.version}"]
        if page_part:
            parts.append(page_part)
        if tag:
            parts.append(tag)
        if title:
            parts.append(title)
        elif not tag:
            parts.append(base_label)
        return " · ".join(parts)

    parts = [f"{doc.file_name} · v{doc.version}"]
    if page_part:
        parts.append(page_part)
    parts.append(base_label)
    return " · ".join(parts)
