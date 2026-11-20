"""Budgeted reads from a pinned, authorized publication snapshot."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from app.schemas.local import SourceKey
from app.schemas.workspace import WorkspaceError
from app.stores.local_library import LocalLibraryStore
from app.stores.structure_snapshot import SnapshotDocument, SnapshotSource, StructureSnapshot

_OUTLINE_LIMIT = 48
_TRACE_LIMIT = 100
_VISUAL_OBJECTS = frozenset({"picture", "image", "chart", "figure"})
_POSITION_FIELDS = ("page", "cell_index", "output_index", "section_path", "bbox")
_Key = tuple[str, int, str]


@dataclass(frozen=True)
class ReadingCursor:
    scope_sha256: str
    source_index: int
    offset: int
    prioritized: bool


@dataclass(frozen=True)
class ReadingSegment:
    document: SnapshotDocument
    source: SnapshotSource
    start: int
    end: int
    text: str
    method: Literal["native_text", "native_table", "ocr"]
    table_headers: tuple[str, ...]


@dataclass(frozen=True)
class ReadingWindow:
    segments: tuple[ReadingSegment, ...]
    next_cursor: ReadingCursor | None
    outline: tuple[dict[str, Any], ...]
    outline_total: int
    skipped: tuple[dict[str, Any], ...]
    unread_visual: tuple[dict[str, Any], ...]
    budget: int
    estimated_input: int
    full_text_scope: bool
    prioritized: bool

    def trace(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "estimated_input": self.estimated_input,
            "prioritized": self.prioritized,
            "full_text_scope": self.full_text_scope,
            "outline": list(self.outline),
            "outline_total": self.outline_total,
            "read_ranges": [
                {
                    **_descriptor(segment.source),
                    "start": segment.start,
                    "end": segment.end,
                    "method": segment.method,
                    "text_sha256": _sha(segment.text),
                }
                for segment in self.segments
            ],
            "skipped": list(self.skipped[:_TRACE_LIMIT]),
            "skipped_total": len(self.skipped),
            "unread_visual": list(self.unread_visual[:_TRACE_LIMIT]),
            "unread_visual_total": len(self.unread_visual),
            # A vector diagram may not have its own registered object.
            "visual_read_status": "not_attempted",
            "next_cursor": asdict(self.next_cursor) if self.next_cursor else None,
        }


@dataclass(frozen=True)
class _Readable:
    source: SnapshotSource
    text: str
    method: Literal["native_text", "native_table", "ocr"]
    headers: tuple[str, ...] = ()

    @property
    def metadata_size(self) -> int:
        return estimate_input_budget({**_descriptor(self.source), "headers": self.headers}) + 48


def estimate_input_budget(value: Any) -> int:
    """Conservative UTF-8 JSON sizing for byte-level model input, not billed tokens."""
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def read_window(
    store: LocalLibraryStore,
    snapshot: StructureSnapshot,
    *,
    context_tokens: int,
    output_reserve: int,
    prompt_reserve: int = 0,
    preferred_sources: Sequence[SourceKey] = (),
    cursor: ReadingCursor | None = None,
    active: Callable[[], bool] = lambda: True,
) -> ReadingWindow:
    """Read native or persisted OCR text without claiming to read images in this turn."""
    budget = context_tokens - output_reserve - prompt_reserve
    if output_reserve < 0 or prompt_reserve < 0 or budget < 0:
        raise ValueError("invalid reading budget")
    if snapshot.resolution_status != "resolved":
        raise WorkspaceError("stale_reference")

    documents = {document.document_id: document for document in snapshot.documents}
    sources = sorted(snapshot.sources, key=_position)
    available = _available_documents(store, documents, sources, active)
    preferred = {_key(item) for item in preferred_sources}
    candidates, skipped, unread = _readable_sources(store, sources, available, preferred)
    outline = [_descriptor(source) for source in sources if source.kind in {"page", "cell"}]
    outline = [entry for entry in outline if entry["title"]]
    anchor_names = {_key(item.source) for item in candidates if _key(item.source) in preferred}
    # The cursor retains the first call's order even when its next window has a different budget.
    prioritized = (
        cursor.prioritized
        if cursor
        else bool(anchor_names)
        and sum(item.metadata_size + estimate_input_budget(item.text) for item in candidates)
        > budget
    )
    if prioritized:
        anchor_names = _expand_priority(candidates, anchor_names)
        candidates.sort(key=lambda item: _key(item.source) not in anchor_names)
    signature = _sha(
        json.dumps(
            [
                [
                    (doc.document_id, doc.version, doc.publication_epoch, doc.content_sha256)
                    for doc in snapshot.documents
                ],
                sorted(preferred),
                [(_key(item.source), _sha(item.text)) for item in candidates],
            ]
        )
    )
    index, offset = _resume(candidates, cursor, signature)
    segments: list[ReadingSegment] = []
    used = 0
    while index < len(candidates):
        if not active():
            raise WorkspaceError("interrupted")
        item = candidates[index]
        capacity = budget - used - item.metadata_size
        end = _fit(item.text, offset, capacity)
        if end == offset:
            if not segments:
                raise ValueError("reading budget cannot fit the next text segment and its source")
            break
        text = item.text[offset:end]
        segments.append(
            ReadingSegment(
                documents[item.source.document_id],
                item.source,
                offset,
                end,
                text,
                item.method,
                item.headers,
            )
        )
        used += item.metadata_size + estimate_input_budget(text)
        if end < len(item.text):
            offset = end
            break
        index += 1
        offset = 0
    next_cursor = (
        ReadingCursor(signature, index, offset, prioritized) if index < len(candidates) else None
    )
    if next_cursor:
        skipped.extend(
            {**_descriptor(item.source), "reason": "window_budget"} for item in candidates[index:]
        )
    if cursor:
        skipped.extend(
            {**_descriptor(item.source), "reason": "before_cursor"}
            for item in candidates[: cursor.source_index]
        )
    return ReadingWindow(
        tuple(sorted(segments, key=lambda item: _position(item.source))),
        next_cursor,
        tuple(_sample_outline(outline)),
        len(outline),
        tuple(skipped),
        tuple(unread),
        budget,
        used,
        cursor is None
        and next_cursor is None
        and all(
            item["reason"] in {"duplicate_page_text", "image_only", "empty_output"}
            for item in skipped
        ),
        prioritized,
    )


def _available_documents(store, documents, sources, active) -> frozenset[str]:
    checked = set()
    available = set()
    for source in sources:
        if source.document_id in checked:
            continue
        if not active():
            raise WorkspaceError("interrupted")
        document = documents[source.document_id]
        raw = store.read_source(_source_key(source), course_ids=(document.course_id,))
        if (
            raw.content_sha256 != document.content_sha256
            or raw.publication_epoch != document.publication_epoch
            or not raw.is_current
        ):
            raise WorkspaceError("stale_reference")
        if raw.status == "available":
            available.add(source.document_id)
        checked.add(source.document_id)
    return frozenset(available)


def _readable_sources(store, sources, available, preferred):
    by_key = {_key(source): source for source in sources}
    result = []
    skipped = []
    unread = []
    for source in sources:
        visual = _has_visual(source)
        if visual:
            unread.append(_descriptor(source))
        text = source.text
        if source.document_id not in available:
            reason = "original_unavailable"
        elif not text.strip():
            ocr = (
                store.read_ocr_text(_source_key(source), course_ids=(source.course_id,))
                if source.kind == "page" and source.state == "needs_ocr"
                else ""
            )
            if ocr:
                result.append(_Readable(source, ocr, "ocr"))
                continue
            reason = (
                "needs_ocr"
                if source.state == "needs_ocr"
                else (
                    "image_only"
                    if visual
                    else ("empty_output" if source.kind == "notebook_output" else "no_text")
                )
            )
        else:
            object_type = source.locator.get("object_type")
            parent = by_key.get((source.document_id, source.version, source.parent_id))
            duplicate = (
                source.locator.get("format") == "pptx"
                and object_type in {"title", "text"}
                and parent
                and source.text in parent.text
                and _key(source) not in preferred
            )
            if not duplicate:
                headers = tuple(source.locator.get("columns", ())) if object_type == "table" else ()
                result.append(
                    _Readable(
                        source,
                        text,
                        "native_table" if object_type == "table" else "native_text",
                        headers,
                    )
                )
                continue
            reason = "duplicate_page_text"
        skipped.append({**_descriptor(source), "reason": reason})
    return result, skipped, unread


def _expand_priority(candidates: Sequence[_Readable], anchors: set[_Key]) -> set[_Key]:
    groups = {
        (
            item.source.document_id,
            item.source.version,
            item.source.parent_id or item.source.source_id,
        )
        for item in candidates
        if _key(item.source) in anchors
    }
    return {
        _key(item.source)
        for item in candidates
        if (
            item.source.document_id,
            item.source.version,
            item.source.parent_id or item.source.source_id,
        )
        in groups
    }


def _resume(candidates, cursor, signature):
    if cursor is None:
        return 0, 0
    if (
        cursor.scope_sha256 != signature
        or not 0 <= cursor.source_index < len(candidates)
        or not 0 <= cursor.offset < len(candidates[cursor.source_index].text)
    ):
        raise WorkspaceError("stale_reference")
    return cursor.source_index, cursor.offset


def _fit(text: str, start: int, budget: int) -> int:
    if estimate_input_budget(text[start:]) <= budget:
        return len(text)
    lo, hi = start, min(len(text), start + max(budget - 2, 0))
    while lo < hi:
        middle = (lo + hi + 1) // 2
        if estimate_input_budget(text[start:middle]) <= budget:
            lo = middle
        else:
            hi = middle - 1
    # Prefer a whole line (especially a table row), with a hard split for oversized rows.
    edge = text.rfind("\n", start + (lo - start) // 2, lo)
    return edge + 1 if edge >= start else lo


def _sample_outline(outline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(outline) <= _OUTLINE_LIMIT:
        return outline
    return [
        outline[index * (len(outline) - 1) // (_OUTLINE_LIMIT - 1)]
        for index in range(_OUTLINE_LIMIT)
    ]


def _has_visual(source: SnapshotSource) -> bool:
    return (
        source.state == "needs_ocr"
        or source.locator.get("object_type") in _VISUAL_OBJECTS
        or any(
            str(item.get("mime_type", "")).startswith("image/")
            for item in source.locator.get("representations", [])
            if isinstance(item, dict)
        )
    )


def _descriptor(source: SnapshotSource) -> dict[str, Any]:
    return {
        "source": _source_key(source).model_dump(),
        "title": source.title,
        "kind": source.kind,
        "locator": {
            name: source.locator[name] for name in _POSITION_FIELDS if name in source.locator
        },
    }


def _position(source: SnapshotSource) -> tuple[str, int, int]:
    return source.document_id, source.locator.get("page", 0), source.ordinal


def _key(source: SnapshotSource | SourceKey) -> _Key:
    return source.document_id, source.version, source.source_id


def _source_key(source: SnapshotSource) -> SourceKey:
    return SourceKey(
        document_id=source.document_id, version=source.version, source_id=source.source_id
    )


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
