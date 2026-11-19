"""Validated, read-only inputs for migrating legacy SQLite and slide bundles."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings
from app.schemas.document import SlideChunkingResult, SlideDeck
from app.schemas.local import ParsedSource, SourceDraft

_FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_FORMATS = {"ipynb", "pdf", "pptx"}


@dataclass(frozen=True, slots=True)
class LegacySnapshot:
    """A checked legacy version plus inert payloads used only for migration mapping."""

    document_id: str
    version: int
    course_id: str
    source_path: Path
    content_sha256: str
    document_format: str
    title: str
    release_at: datetime | None
    _blocks: tuple[dict[str, Any], ...] = field(default=(), repr=False)
    _chunks: tuple[dict[str, Any], ...] = field(default=(), repr=False)
    _elements: tuple[dict[str, Any], ...] = field(default=(), repr=False)
    _assets: tuple[dict[str, Any], ...] = field(default=(), repr=False)


def load_legacy_sqlite(settings: Settings, document_id: str) -> LegacySnapshot:
    """Read one visible legacy active version in a single read transaction."""

    clean_document_id = document_id.strip()
    if not clean_document_id:
        raise ValueError("document_id must not be empty")
    database = settings.sqlite_path.expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f"legacy SQLite database does not exist: {database}")

    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=30.0)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN")
        rows = connection.execute(
            """
            SELECT d.document_id, d.course_id, d.source_path, d.title, d.document_type,
                   v.version, v.content_hash, v.release_at
            FROM documents AS d
            JOIN document_versions AS v ON v.document_id = d.document_id
            WHERE d.document_id = ?
              AND d.deleted_at IS NULL
              AND v.state = 'active'
              AND (v.release_at IS NULL OR v.release_at <= ?)
            ORDER BY v.version DESC
            """,
            (clean_document_id, _serialize_datetime(datetime.now(UTC))),
        ).fetchall()
        if not rows:
            raise ValueError("legacy document has no currently visible active version")
        if len(rows) != 1:
            raise ValueError("legacy document has multiple visible active versions")
        row = rows[0]
        parameters = (clean_document_id, row["version"])
        blocks = _read_payload_rows(
            connection,
            "SELECT * FROM content_blocks WHERE document_id=? AND version=? ORDER BY rowid",
            parameters,
        )
        chunks = _read_payload_rows(
            connection,
            "SELECT * FROM chunks WHERE document_id=? AND version=? ORDER BY rowid",
            parameters,
        )
        asset_rows = tuple(
            dict(item)
            for item in connection.execute(
                "SELECT * FROM assets WHERE document_id=? AND version=? ORDER BY asset_id",
                parameters,
            ).fetchall()
        )
        connection.commit()
    finally:
        connection.close()

    content_sha256 = _checked_sha256(row["content_hash"])
    source_path = Path(row["source_path"])
    document_format = _normalized_format(row["document_type"], source_path)
    block_pages = {
        block_id: _page_number(block)
        for block in blocks
        if (block_id := _identifier(block, "block_id")) is not None
    }
    assets = tuple(
        {**asset, "_legacy_page": block_pages.get(str(asset.get("block_id")))}
        for asset in asset_rows
    )
    elements = tuple(
        _page_child(payload, block, "elements")
        for block in blocks
        for payload in [block["payload"]]
        for item in payload.get("elements", [])
        if isinstance(item, dict)
        for payload in [item]
    )
    nested_assets = tuple(
        _page_child(payload, block, "assets")
        for block in blocks
        for payload in [block["payload"]]
        for item in payload.get("assets", [])
        if isinstance(item, dict)
        for payload in [item]
    )
    return LegacySnapshot(
        document_id=row["document_id"],
        version=row["version"],
        course_id=row["course_id"],
        source_path=source_path,
        content_sha256=content_sha256,
        document_format=document_format,
        title=row["title"],
        release_at=_parse_datetime(row["release_at"]),
        _blocks=blocks,
        _chunks=chunks,
        _elements=elements,
        _assets=(*assets, *nested_assets),
    )


def load_legacy_bundle(
    *,
    course_id: str,
    source_path: Path,
    bundle_dir: Path,
    content_sha256: str,
    version: int = 1,
) -> LegacySnapshot:
    """Load and validate a portable old slide bundle without opening asset paths."""

    clean_course_id = course_id.strip()
    if not clean_course_id:
        raise ValueError("course_id must not be empty")
    if version < 1:
        raise ValueError("version must be at least 1")
    checked_hash = _checked_sha256(content_sha256)
    source = source_path.expanduser().resolve()
    bundle = bundle_dir.expanduser().resolve()
    deck_path = bundle / "deck.json"
    chunks_path = bundle / "chunks.json"
    if not deck_path.is_file() or not chunks_path.is_file():
        raise FileNotFoundError("legacy bundle requires deck.json and chunks.json")

    deck = SlideDeck.model_validate_json(deck_path.read_text(encoding="utf-8"))
    chunking = SlideChunkingResult.model_validate_json(chunks_path.read_text(encoding="utf-8"))
    expected_document_id = f"document_{checked_hash[:24]}"
    document_format = _normalized_format(deck.source_format.value, source)
    if (
        deck.document_id != expected_document_id
        or chunking.document_id != expected_document_id
        or Path(deck.source_path).name != source.name
        or deck.source_format.value != document_format
    ):
        raise ValueError("legacy bundle identity, filename, format, or pinned hash does not match")
    _validate_bundle_relations(deck, chunking)

    blocks = tuple(slide.model_dump(mode="json") for slide in deck.slides)
    elements = tuple(
        {
            **element.model_dump(mode="json"),
            "_legacy_page": slide.slide_no,
            "_legacy_slide_id": slide.slide_id,
        }
        for slide in deck.slides
        for element in slide.elements
    )
    assets = tuple(
        {
            **asset.model_dump(mode="json"),
            "_legacy_page": slide.slide_no,
            "_legacy_slide_id": slide.slide_id,
        }
        for slide in deck.slides
        for asset in slide.assets
    )
    return LegacySnapshot(
        document_id=expected_document_id,
        version=version,
        course_id=clean_course_id,
        source_path=source,
        content_sha256=checked_hash,
        document_format=document_format,
        title=deck.title,
        release_at=None,
        _blocks=blocks,
        _chunks=tuple(chunk.model_dump(mode="json") for chunk in chunking.chunks),
        _elements=elements,
        _assets=assets,
    )


def map_legacy_sources(
    snapshot: LegacySnapshot, native: ParsedSource | None
) -> tuple[ParsedSource, dict[str, list[str]]]:
    """Map legacy IDs to verified native sources or conservative page/cell fallbacks."""

    parsed = native if native is not None else _legacy_fallback_sources(snapshot)
    by_id = {source.source_id: source for source in parsed.sources}
    if len(by_id) != len(parsed.sources):
        raise ValueError("native ParsedSource contains duplicate source_id values")
    mapping: dict[str, list[str]] = {}

    if snapshot.document_format in {"pdf", "pptx"}:
        _map_slide_sources(snapshot, parsed, by_id, mapping)
    elif snapshot.document_format == "ipynb":
        _map_notebook_sources(snapshot, parsed, by_id, mapping)
    else:  # defensive guard for manually constructed snapshots
        raise ValueError(f"unsupported legacy format: {snapshot.document_format}")
    return parsed, mapping


def _map_slide_sources(
    snapshot: LegacySnapshot,
    parsed: ParsedSource,
    by_id: dict[str, SourceDraft],
    mapping: dict[str, list[str]],
) -> None:
    block_pages: dict[str, int] = {}
    for block in snapshot._blocks:
        block_id = _identifier(block, "block_id", "slide_id")
        page = _page_number(block)
        if block_id and page is not None:
            block_pages[block_id] = page
            mapping[f"block:{block_id}"] = _existing(by_id, [f"page:{page}"])

    element_maps: dict[str, list[str]] = {}
    asset_elements: dict[str, str] = {}
    for element in snapshot._elements:
        element_id = _identifier(element, "element_id")
        page = _page_number(element)
        if not element_id or page is None:
            continue
        exact = _match_native_object(parsed, page, element)
        targets = _existing(by_id, [exact or f"page:{page}"])
        element_maps[element_id] = targets
        mapping[f"element:{element_id}"] = targets
        if (asset_id := _identifier(element, "asset_id")) is not None:
            asset_elements[asset_id] = element_id

    for asset in snapshot._assets:
        asset_id = _identifier(asset, "asset_id")
        if not asset_id:
            continue
        element_id = asset.get("element_id") or asset_elements.get(asset_id)
        page = _page_number(asset)
        targets = element_maps.get(str(element_id), []) if element_id else []
        if not targets and page is not None:
            targets = _existing(by_id, [f"page:{page}"])
        mapping[f"asset:{asset_id}"] = targets

    for chunk in snapshot._chunks:
        chunk_id = _identifier(chunk, "chunk_id")
        if not chunk_id:
            continue
        targets = _mapped_children(mapping, "element", chunk.get("element_ids"))
        page = _page_number(chunk)
        if not targets and page is not None:
            targets = _existing(by_id, [f"page:{page}"])
        mapping[f"chunk:{chunk_id}"] = targets


def _map_notebook_sources(
    snapshot: LegacySnapshot,
    parsed: ParsedSource,
    by_id: dict[str, SourceDraft],
    mapping: dict[str, list[str]],
) -> None:
    native_cells_by_original_id: dict[str, list[str]] = {}
    for source in parsed.sources:
        if source.kind == "cell" and (cell_id := source.locator.get("cell_id")):
            native_cells_by_original_id.setdefault(str(cell_id), []).append(source.source_id)

    block_maps: dict[str, list[str]] = {}
    for block in snapshot._blocks:
        payload = block.get("payload", block)
        block_id = _identifier(block, "block_id") or _identifier(payload, "block_id")
        if not block_id:
            continue
        cell_index = payload.get("cell_index", block.get("cell_index"))
        cell_source_id = f"cell:{cell_index}" if isinstance(cell_index, int) else None
        targets: list[str] = []
        if cell_source_id and cell_source_id in by_id:
            output_index = _legacy_output_index(block_id)
            output_source_id = (
                f"{cell_source_id}:output:{output_index}" if output_index is not None else None
            )
            targets = _existing(by_id, [output_source_id or cell_source_id])
            if not targets:
                targets = _existing(by_id, [cell_source_id])
        if not targets:
            original_cell_id = payload.get("cell_id", block.get("cell_id"))
            candidates = native_cells_by_original_id.get(str(original_cell_id), [])
            if len(candidates) == 1:
                targets = candidates
        block_maps[block_id] = targets
        mapping[f"block:{block_id}"] = targets

    for asset in snapshot._assets:
        asset_id = _identifier(asset, "asset_id")
        if asset_id:
            mapping[f"asset:{asset_id}"] = block_maps.get(str(asset.get("block_id")), [])

    for chunk in snapshot._chunks:
        payload = chunk.get("payload", chunk)
        chunk_id = _identifier(chunk, "chunk_id") or _identifier(payload, "chunk_id")
        if not chunk_id:
            continue
        targets = _mapped_children(mapping, "block", payload.get("block_ids"))
        if not targets:
            for cell_id in payload.get("cell_ids", []):
                candidates = native_cells_by_original_id.get(str(cell_id), [])
                if len(candidates) == 1:
                    targets.extend(candidates)
        mapping[f"chunk:{chunk_id}"] = _unique(targets)


def _legacy_fallback_sources(snapshot: LegacySnapshot) -> ParsedSource:
    if snapshot.document_format in {"pdf", "pptx"}:
        pages = sorted(
            {
                page
                for item in (*snapshot._blocks, *snapshot._chunks)
                if (page := _page_number(item)) is not None
            }
        )
        sources = []
        for ordinal, page in enumerate(pages):
            page_text = _legacy_page_text(snapshot, page)
            sources.append(
                SourceDraft(
                    source_id=f"page:{page}",
                    kind="page",
                    ordinal=ordinal,
                    title=_first_line(page_text),
                    text=page_text,
                    locator={
                        "format": snapshot.document_format,
                        "page": page,
                        "text_origin": "legacy_text",
                        "position_precision": "page_only",
                    },
                    state="legacy_fallback",
                )
            )
        return ParsedSource(title=snapshot.title, sources=sources)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for block in snapshot._blocks:
        payload = block.get("payload", block)
        cell_index = payload.get("cell_index", block.get("cell_index"))
        if isinstance(cell_index, int):
            grouped.setdefault(cell_index, []).append(payload)
    sources = []
    for ordinal, (cell_index, blocks) in enumerate(sorted(grouped.items())):
        text = "\n\n".join(
            value for block in blocks if (value := str(block.get("text") or "").strip())
        )
        cell_ids = {str(block["cell_id"]) for block in blocks if block.get("cell_id")}
        sources.append(
            SourceDraft(
                source_id=f"cell:{cell_index}",
                kind="cell",
                ordinal=ordinal,
                title=_first_line(text),
                text=text,
                locator={
                    "cell_index": cell_index,
                    "cell_id": next(iter(cell_ids)) if len(cell_ids) == 1 else None,
                    "text_origin": "legacy_text",
                    "position_precision": "cell_only",
                },
                state="legacy_fallback",
            )
        )
    return ParsedSource(title=snapshot.title, sources=sources)


def _validate_bundle_relations(deck: SlideDeck, chunking: SlideChunkingResult) -> None:
    slides = _unique_by_id(deck.slides, "slide_id")
    chunks = _unique_by_id(chunking.chunks, "chunk_id")
    sections = _unique_by_id(chunking.sections, "section_id")
    elements: dict[str, Any] = {}
    assets: dict[str, Any] = {}
    pages: set[int] = set()
    for slide in deck.slides:
        if slide.document_id != deck.document_id or slide.slide_no in pages:
            raise ValueError("legacy deck slide document IDs and page numbers must be unique")
        pages.add(slide.slide_no)
        slide_asset_ids = {item.asset_id for item in slide.assets}
        for element in slide.elements:
            if element.element_id in elements:
                raise ValueError("legacy element IDs must be unique")
            elements[element.element_id] = element
            if element.asset_id and element.asset_id not in slide_asset_ids:
                raise ValueError("legacy element references an asset outside its page")
        for asset in slide.assets:
            if asset.asset_id in assets:
                raise ValueError("legacy asset IDs must be unique")
            assets[asset.asset_id] = asset
    for section in chunking.sections:
        if section.document_id != deck.document_id:
            raise ValueError("legacy section document_id does not match deck")
        if any(slide_id not in slides for slide_id in section.slide_ids):
            raise ValueError("legacy section references a missing slide")
        if any(chunk_id not in chunks for chunk_id in section.chunk_ids):
            raise ValueError("legacy section references a missing chunk")
    for chunk in chunking.chunks:
        slide = slides.get(chunk.slide_id)
        section = sections.get(chunk.section_id)
        if (
            chunk.document_id != deck.document_id
            or slide is None
            or chunk.slide_no != slide.slide_no
            or section is None
            or chunk.chunk_id not in section.chunk_ids
            or chunk.slide_id not in section.slide_ids
            or any(
                element_id not in {item.element_id for item in slide.elements}
                for element_id in chunk.element_ids
            )
            or any(
                asset_id not in {item.asset_id for item in slide.assets}
                for asset_id in chunk.asset_ids
            )
        ):
            raise ValueError("legacy chunk has inconsistent document, page, section, or child IDs")
        if chunk.previous_chunk_id and chunk.previous_chunk_id not in chunks:
            raise ValueError("legacy chunk references a missing previous chunk")
        if chunk.next_chunk_id and chunk.next_chunk_id not in chunks:
            raise ValueError("legacy chunk references a missing next chunk")
    for slide in deck.slides:
        if slide.previous_slide_id and slide.previous_slide_id not in slides:
            raise ValueError("legacy slide references a missing previous slide")
        if slide.next_slide_id and slide.next_slide_id not in slides:
            raise ValueError("legacy slide references a missing next slide")


def _read_payload_rows(
    connection: sqlite3.Connection, query: str, parameters: tuple[str, int]
) -> tuple[dict[str, Any], ...]:
    result = []
    for row in connection.execute(query, parameters).fetchall():
        item = dict(row)
        try:
            payload = json.loads(item["payload_json"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("legacy SQLite contains invalid payload_json") from exc
        if not isinstance(payload, dict):
            raise ValueError("legacy SQLite payload_json must contain an object")
        item["payload"] = payload
        result.append(item)
    return tuple(result)


def _page_child(payload: dict[str, Any], block: dict[str, Any], key: str) -> dict[str, Any]:
    return {
        **payload,
        "_legacy_page": _page_number(block),
        "_legacy_slide_id": block.get("block_id") or block.get("slide_id"),
        "_legacy_child_kind": key,
    }


def _match_native_object(parsed: ParsedSource, page: int, element: dict[str, Any]) -> str | None:
    old_kind = str(element.get("kind") or "").casefold()
    expected_kind = {"image": "picture"}.get(old_kind, old_kind)
    old_text = _normalize_text(str(element.get("text") or ""))
    old_bbox = _bbox_values(element.get("bbox"))
    candidates = []
    for source in parsed.sources:
        if source.kind != "page_object" or source.locator.get("page") != page:
            continue
        object_type = str(source.locator.get("object_type") or "").casefold()
        if expected_kind and object_type != expected_kind:
            continue
        if old_text and _normalize_text(source.text) != old_text:
            continue
        native_bbox = _bbox_values(source.locator.get("bbox"))
        if old_bbox is not None and (
            native_bbox is None
            or any(abs(a - b) > 0.02 for a, b in zip(old_bbox, native_bbox, strict=True))
        ):
            continue
        candidates.append(source.source_id)
    return candidates[0] if len(candidates) == 1 else None


def _legacy_page_text(snapshot: LegacySnapshot, page: int) -> str:
    parts = []
    for block in snapshot._blocks:
        if _page_number(block) != page:
            continue
        payload = block.get("payload", block)
        parts.extend(
            str(element.get("text") or "").strip()
            for element in payload.get("elements", [])
            if isinstance(element, dict) and str(element.get("text") or "").strip()
        )
    if not parts:
        parts.extend(
            str(chunk.get("payload", chunk).get("display_text") or "").strip()
            for chunk in snapshot._chunks
            if _page_number(chunk) == page
            and str(chunk.get("payload", chunk).get("display_text") or "").strip()
        )
    return "\n\n".join(_unique(parts))


def _normalized_format(document_type: Any, source_path: Path) -> str:
    raw = str(document_type or "").casefold().lstrip(".")
    raw = "ipynb" if raw == "notebook" else raw
    suffix = source_path.suffix.casefold().lstrip(".")
    if raw not in _SUPPORTED_FORMATS:
        raw = suffix
    if raw not in _SUPPORTED_FORMATS or (suffix in _SUPPORTED_FORMATS and suffix != raw):
        raise ValueError("legacy document format is unsupported or conflicts with source filename")
    return raw


def _checked_sha256(value: Any) -> str:
    digest = str(value or "").casefold()
    if not _FULL_SHA256.fullmatch(digest):
        raise ValueError("legacy content hash must be a complete SHA-256 digest")
    return digest


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _serialize_datetime(value: datetime) -> str:
    aware = value if value.tzinfo else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _identifier(item: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _page_number(item: dict[str, Any]) -> int | None:
    payload = item.get("payload", item)
    for value in (
        item.get("_legacy_page"),
        payload.get("slide_no"),
        item.get("page"),
    ):
        if isinstance(value, int) and value >= 1:
            return value
    return None


def _legacy_output_index(block_id: str) -> int | None:
    marker = ":output:"
    if marker not in block_id:
        return None
    value = block_id.rsplit(marker, 1)[1]
    return int(value) if value.isascii() and value.isdigit() else None


def _mapped_children(mapping: dict[str, list[str]], prefix: str, values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return _unique(target for value in values for target in mapping.get(f"{prefix}:{value}", []))


def _existing(by_id: dict[str, SourceDraft], source_ids: list[str]) -> list[str]:
    return [source_id for source_id in source_ids if source_id in by_id]


def _bbox_values(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        values = [value.get(key) for key in ("left", "top", "right", "bottom")]
    elif isinstance(value, list | tuple) and len(value) == 4:
        values = list(value)
    else:
        return None
    if not all(isinstance(item, int | float) for item in values):
        return None
    return tuple(float(item) for item in values)  # type: ignore[return-value]


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _first_line(value: str) -> str:
    return next((line.strip() for line in value.splitlines() if line.strip()), "")


def _unique_by_id(values: list[Any], attribute: str) -> dict[str, Any]:
    result = {getattr(value, attribute): value for value in values}
    if len(result) != len(values):
        raise ValueError(f"legacy {attribute} values must be unique")
    return result


def _unique(values: Any) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
