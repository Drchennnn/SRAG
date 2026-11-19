"""Consistent read-only snapshots for structure search."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.stores.local_visibility import promote_due
from app.stores.sqlite import MetadataStore


@dataclass(frozen=True)
class SnapshotRequest:
    course_ids: tuple[str, ...]
    document_ids: tuple[str, ...] = ()
    file_names: tuple[str, ...] = ()
    fts_match: str | None = None
    fts_limit: int = 0
    require_single_document: bool = False


@dataclass(frozen=True)
class SnapshotDocument:
    course_id: str
    document_id: str
    version: int
    publication_epoch: int
    content_sha256: str
    source_path: str
    file_name: str
    title: str
    document_format: str


@dataclass(frozen=True)
class SnapshotSource:
    course_id: str
    document_id: str
    version: int
    source_id: str
    parent_id: str | None
    ordinal: int
    kind: str
    title: str
    text: str
    locator: dict[str, Any]
    state: str


@dataclass(frozen=True)
class SnapshotFtsMatch:
    origin: str
    title: str
    body: str
    rank: float


@dataclass(frozen=True)
class SnapshotFtsSeed:
    document_id: str
    version: int
    source_id: str
    source_rowid: int
    rank: float
    matches: tuple[SnapshotFtsMatch, ...]


@dataclass(frozen=True)
class StructureSnapshot:
    resolution_status: Literal["resolved", "scope_not_found", "ambiguous_scope"]
    resolution_misses: tuple[str, ...]
    documents: tuple[SnapshotDocument, ...]
    sources: tuple[SnapshotSource, ...]
    fts_seeds: tuple[SnapshotFtsSeed, ...]
    name_matches: tuple[tuple[str, tuple[str, ...]], ...]
    searchable_source_count: int
    fts_scan_truncated: bool


class StructureSnapshotStore(MetadataStore):
    """Read a fixed B publication snapshot without changing B's store surface."""

    def read(self, request: SnapshotRequest) -> StructureSnapshot:
        if not request.course_ids:
            return StructureSnapshot("scope_not_found", ("empty_scope",), (), (), (), (), 0, False)
        with self._connect() as db:
            promote_due(db, _now())
            db.execute("BEGIN")
            course_marks = ",".join("?" for _ in request.course_ids)
            operands: list[object] = list(request.course_ids)
            document_sql = ""
            if request.document_ids:
                marks = ",".join("?" for _ in request.document_ids)
                document_sql = f" AND d.document_id IN ({marks})"
                operands.extend(request.document_ids)
            rows = db.execute(
                f"""
                SELECT d.course_id, d.document_id, d.current_version AS version,
                       d.publication_epoch, d.source_path, v.content_sha256,
                       v.title, v.document_format
                FROM local_documents d
                JOIN local_versions v ON v.document_id = d.document_id
                    AND v.version = d.current_version AND v.published_at IS NOT NULL
                WHERE d.deleted = 0 AND d.current_version IS NOT NULL
                    AND d.course_id IN ({course_marks}) {document_sql}
                ORDER BY d.course_id, d.document_id
                """,
                operands,
            ).fetchall()
            all_documents = tuple(_document(row) for row in rows)
            selected, name_matches = _select_names(all_documents, request.file_names)
            missing_documents = tuple(
                f"document_not_found:{document_id}"
                for document_id in request.document_ids
                if document_id not in {document.document_id for document in all_documents}
            )
            missing_names = tuple(
                f"file_not_found:{name}" for name, matches in name_matches if not matches
            )
            if missing_documents or missing_names or not selected:
                misses = (*missing_documents, *missing_names) or ("empty_scope",)
                return StructureSnapshot(
                    "scope_not_found", misses, selected, (), (), name_matches, 0, False
                )
            ambiguous = any(len(matches) > 1 for _, matches in name_matches) or (
                request.require_single_document and len(selected) > 1
            )
            if ambiguous:
                return StructureSnapshot(
                    "ambiguous_scope", (), selected, (), (), name_matches, 0, False
                )

            document_ids = tuple(document.document_id for document in selected)
            marks = ",".join("?" for _ in document_ids)
            source_rows = db.execute(
                f"""
                SELECT d.course_id, s.*
                FROM local_sources s
                JOIN local_documents d ON d.document_id = s.document_id
                    AND d.current_version = s.version AND d.deleted = 0
                WHERE s.document_id IN ({marks})
                ORDER BY d.course_id, d.document_id, s.ordinal, s.source_id
                """,
                document_ids,
            ).fetchall()
            sources = tuple(_source(row) for row in source_rows)
            searchable = db.execute(
                f"""
                SELECT count(DISTINCT f.source_rowid)
                FROM local_fts f
                JOIN local_sources s ON s.id = f.source_rowid
                JOIN local_documents d ON d.document_id = s.document_id
                    AND d.current_version = s.version AND d.deleted = 0
                WHERE s.document_id IN ({marks})
                """,
                document_ids,
            ).fetchone()[0]
            seeds, scan_truncated = _fts_seeds(
                db,
                document_ids=document_ids,
                match=request.fts_match,
                limit=request.fts_limit,
            )
            return StructureSnapshot(
                "resolved",
                (),
                selected,
                sources,
                seeds,
                name_matches,
                searchable,
                scan_truncated,
            )


def _fts_seeds(db, *, document_ids: tuple[str, ...], match: str | None, limit: int):
    if not match or limit <= 0:
        return (), False
    marks = ",".join("?" for _ in document_ids)
    page_size = max(32, limit * 2)
    scan_limit = max(64, limit * 8)
    selected: dict[int, tuple[float, str, int, str]] = {}
    offset = 0
    exhausted = False
    while len(selected) < limit and offset < scan_limit:
        rows = db.execute(
            f"""
            SELECT f.source_rowid, s.document_id, s.version, s.source_id,
                   bm25(local_fts) AS rank
            FROM local_fts f
            JOIN local_sources s ON s.id = f.source_rowid
            JOIN local_documents d ON d.document_id = s.document_id
                AND d.current_version = s.version AND d.deleted = 0
            WHERE local_fts MATCH ? AND s.document_id IN ({marks})
            ORDER BY rank, s.document_id, s.ordinal, f.rowid
            LIMIT ? OFFSET ?
            """,
            [match, *document_ids, page_size, offset],
        ).fetchall()
        if not rows:
            exhausted = True
            break
        for row in rows:
            selected.setdefault(
                row["source_rowid"],
                (row["rank"], row["document_id"], row["version"], row["source_id"]),
            )
            if len(selected) == limit:
                break
        offset += len(rows)
        if len(rows) < page_size:
            exhausted = True
            break
    if not selected:
        return (), False

    rowids = tuple(selected)
    row_marks = ",".join("?" for _ in rowids)
    match_rows = db.execute(
        f"""
        SELECT f.source_rowid, f.origin, f.title, f.body, bm25(local_fts) AS rank
        FROM local_fts f
        WHERE local_fts MATCH ? AND f.source_rowid IN ({row_marks})
        ORDER BY rank, f.rowid
        """,
        [match, *rowids],
    ).fetchall()
    matches: dict[int, list[SnapshotFtsMatch]] = {rowid: [] for rowid in rowids}
    for row in match_rows:
        matches[row["source_rowid"]].append(
            SnapshotFtsMatch(row["origin"], row["title"], row["body"], row["rank"])
        )
    seeds = tuple(
        SnapshotFtsSeed(
            document_id=document_id,
            version=version,
            source_id=source_id,
            source_rowid=rowid,
            rank=rank,
            matches=tuple(matches[rowid]),
        )
        for rowid, (rank, document_id, version, source_id) in selected.items()
    )
    return seeds, not exhausted and offset >= scan_limit and len(seeds) < limit


def _select_names(
    documents: tuple[SnapshotDocument, ...], file_names: tuple[str, ...]
) -> tuple[tuple[SnapshotDocument, ...], tuple[tuple[str, tuple[str, ...]], ...]]:
    if not file_names:
        return documents, ()
    normalized = tuple(
        dict.fromkeys(_normalize_name(Path(name).name) for name in file_names if name.strip())
    )
    mapping = []
    selected_ids: set[str] = set()
    for name in normalized:
        matches = tuple(
            document.document_id
            for document in documents
            if _normalize_name(document.file_name) == name
        )
        mapping.append((name, matches))
        selected_ids.update(matches)
    return tuple(d for d in documents if d.document_id in selected_ids), tuple(mapping)


def _document(row) -> SnapshotDocument:
    source_path = row["source_path"]
    return SnapshotDocument(
        course_id=row["course_id"],
        document_id=row["document_id"],
        version=row["version"],
        publication_epoch=row["publication_epoch"],
        content_sha256=row["content_sha256"],
        source_path=source_path,
        file_name=Path(source_path).name,
        title=row["title"],
        document_format=row["document_format"],
    )


def _source(row) -> SnapshotSource:
    return SnapshotSource(
        course_id=row["course_id"],
        document_id=row["document_id"],
        version=row["version"],
        source_id=row["source_id"],
        parent_id=row["parent_id"],
        ordinal=row["ordinal"],
        kind=row["kind"],
        title=row["title"],
        text=row["text"],
        locator=json.loads(row["locator_json"]),
        state=row["state"],
    )


def _normalize_name(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


def _now() -> float:
    from datetime import UTC, datetime

    return datetime.now(UTC).timestamp()
