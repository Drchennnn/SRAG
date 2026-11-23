"""Atomic storage of original bytes, versioned sources and local FTS entries."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from sqlite3 import Connection, Row
from typing import Literal
from uuid import uuid4

from app.schemas.local import ImportResult, LocalHit, ParsedSource, SourceDraft, SourceKey
from app.schemas.rag import SearchRequest
from app.stores.local_kg import (
    KG_SCHEMA,
    invalidate_document_kg_in_tx,
    sync_course_kg_in_tx,
)
from app.stores.local_visibility import promote_due
from app.stores.sqlite import MetadataStore, _serialize_datetime

_SCHEMA = """
CREATE TABLE IF NOT EXISTS local_documents (
    document_id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    current_version INTEGER,
    publication_epoch INTEGER NOT NULL DEFAULT 0,
    deleted INTEGER NOT NULL DEFAULT 0 CHECK (deleted IN (0, 1)),
    pending_version INTEGER,
    pending_at REAL,
    CHECK ((pending_version IS NULL) = (pending_at IS NULL)),
    FOREIGN KEY(document_id, current_version) REFERENCES local_versions(document_id, version),
    FOREIGN KEY(document_id, pending_version) REFERENCES local_versions(document_id, version)
);
CREATE TABLE IF NOT EXISTS local_paths (
    course_id TEXT NOT NULL,
    canonical_path TEXT NOT NULL,
    document_id TEXT NOT NULL REFERENCES local_documents(document_id),
    PRIMARY KEY(course_id, canonical_path)
);
CREATE TABLE IF NOT EXISTS local_versions (
    document_id TEXT NOT NULL REFERENCES local_documents(document_id),
    version INTEGER NOT NULL CHECK(version > 0),
    content_sha256 TEXT NOT NULL,
    source_path TEXT NOT NULL,
    document_format TEXT NOT NULL,
    title TEXT NOT NULL,
    original BLOB,
    published_at REAL,
    PRIMARY KEY(document_id, version),
    UNIQUE(document_id, content_sha256)
);
CREATE TABLE IF NOT EXISTS local_sources (
    id INTEGER PRIMARY KEY,
    document_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    source_id TEXT NOT NULL,
    parent_id TEXT,
    ordinal INTEGER NOT NULL,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    locator_json TEXT NOT NULL,
    state TEXT NOT NULL,
    UNIQUE(document_id, version, source_id),
    FOREIGN KEY(document_id, version) REFERENCES local_versions(document_id, version),
    FOREIGN KEY(document_id, version, parent_id)
        REFERENCES local_sources(document_id, version, source_id)
);
CREATE VIRTUAL TABLE IF NOT EXISTS local_fts USING fts5(
    title, body, source_rowid UNINDEXED, origin UNINDEXED, tokenize='unicode61'
);
CREATE TABLE IF NOT EXISTS local_legacy_versions (
    legacy_document_id TEXT NOT NULL,
    legacy_version INTEGER NOT NULL,
    course_id TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    document_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    PRIMARY KEY(legacy_document_id, legacy_version),
    FOREIGN KEY(document_id, version) REFERENCES local_versions(document_id, version)
);
CREATE TABLE IF NOT EXISTS local_legacy_keys (
    legacy_document_id TEXT NOT NULL,
    legacy_version INTEGER NOT NULL,
    legacy_key TEXT NOT NULL,
    source_rowid INTEGER NOT NULL REFERENCES local_sources(id),
    PRIMARY KEY(legacy_document_id, legacy_version, legacy_key, source_rowid),
    FOREIGN KEY(legacy_document_id, legacy_version)
        REFERENCES local_legacy_versions(legacy_document_id, legacy_version)
);
"""


class PublicationConflict(ValueError):
    """The authoritative document changed while a snapshot was being prepared."""


@dataclass(frozen=True)
class ImportPlan:
    course_id: str
    source_path: str
    document_id: str | None
    epoch: int | None
    move: bool


@dataclass(frozen=True)
class LegacyBinding:
    document_id: str
    version: int
    keys: dict[str, list[str]]
    from_sqlite: bool = False
    refine: bool = False


@dataclass(frozen=True)
class HistoricalSource:
    status: Literal["available", "original_unavailable", "not_accessible"]
    source: SourceDraft | None = None
    content_sha256: str | None = None
    original: bytes | None = None
    publication_epoch: int | None = None
    is_current: bool = False


class LocalLibraryStore(MetadataStore):
    def initialize(self) -> None:
        super().initialize()
        with self._connect() as db:
            db.executescript(_SCHEMA + KG_SCHEMA)

    def prepare(self, course_id: str, path: Path, *, document_id: str | None = None) -> ImportPlan:
        course_id = course_id.strip()
        if not course_id:
            raise ValueError("course_id 不能为空")
        source_path = str(path)
        with self._connect() as db:
            promote_due(db, _now())
            existing = _path_owner(db, course_id, source_path)
            if document_id is not None:
                selected = _get_document(db, document_id)
                if selected is None or selected["course_id"] != course_id:
                    raise ValueError("资料不存在或不属于该课程")
                if existing is not None and existing["document_id"] != document_id:
                    raise PublicationConflict("所选路径已登记给其他资料")
                existing = selected
            if existing is None:
                return ImportPlan(course_id, source_path, None, None, False)
            return ImportPlan(
                course_id,
                source_path,
                existing["document_id"],
                existing["publication_epoch"],
                document_id is not None and source_path != existing["source_path"],
            )

    def publish(
        self,
        plan: ImportPlan,
        parsed: ParsedSource,
        *,
        original: bytes | None,
        content_sha256: str,
        document_format: str,
        release_at: datetime | None = None,
        legacy: LegacyBinding | None = None,
    ) -> ImportResult:
        if not re.fullmatch(r"[0-9a-f]{64}", content_sha256):
            raise ValueError("需要已核对的完整 SHA-256")
        if original is not None and hashlib.sha256(original).hexdigest() != content_sha256:
            raise ValueError("原件 SHA-256 与登记不符")
        if original is None and legacy is None:
            raise ValueError("本地发布需要原件")
        _validate_sources(parsed.sources)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cutoff = _now()
            promote_due(db, cutoff)
            if legacy and legacy.from_sqlite:
                _validate_legacy_snapshot(db, plan, legacy, content_sha256, cutoff)
            document = _claim_plan(db, plan, legacy)
            document_id = document["document_id"]
            version_row = db.execute(
                "SELECT version, original IS NOT NULL AS archived FROM local_versions "
                "WHERE document_id = ? AND content_sha256 = ?",
                (document_id, content_sha256),
            ).fetchone()
            version = (
                version_row["version"]
                if version_row
                else _insert_version(
                    db, document, plan, parsed, content_sha256, document_format, original
                )
            )
            added_images = False
            if version_row is None or (not version_row["archived"] and original is not None):
                _save_sources(db, document_id, version, parsed.sources)
                if version_row is not None:
                    db.execute(
                        "UPDATE local_versions SET original = ?, title = ? "
                        "WHERE document_id = ? AND version = ?",
                        (original, parsed.title, document_id, version),
                    )
            elif document_format == "pdf" and original is not None and legacy is None:
                added_images = _append_missing_pdf_images(db, document_id, version, parsed.sources)

            if legacy:
                _bind_legacy(db, document_id, version, plan, legacy, content_sha256)
            result = _activate(db, document, plan, version, cutoff, release_at, legacy is not None)
            if added_images and result == "skipped" and not plan.move:
                # Activation already invalidates snapshots on version or path changes.
                db.execute(
                    "UPDATE local_documents SET publication_epoch = publication_epoch + 1 "
                    "WHERE document_id = ?",
                    (document_id,),
                )
            if result != "skipped" or added_images or plan.move:
                sync_course_kg_in_tx(db, plan.course_id, now=cutoff)
            counts = db.execute(
                """
                SELECT count(*) AS source_count,
                       sum(CASE WHEN trim(text) <> '' THEN 1 ELSE 0 END) AS searchable_count
                FROM local_sources WHERE document_id = ? AND version = ?
                """,
                (document_id, version),
            ).fetchone()
            return ImportResult(
                status=result,
                source_path=plan.source_path,
                document_id=document_id,
                version=version,
                publication_epoch=_get_document(db, document_id)["publication_epoch"],
                content_sha256=content_sha256,
                source_count=counts["source_count"],
                searchable_count=counts["searchable_count"] or 0,
            )

    def search_local(self, request: SearchRequest) -> list[LocalHit]:
        terms = list(dict.fromkeys(re.findall(r"[^\W_]+", request.query)))
        if not terms:
            return []
        match = " OR ".join(f'"{term}"' for term in terms)
        scope, operands = _scope_sql(request.scope.course_ids, request.scope.document_ids)
        with self._connect() as db:
            promote_due(db, _now())
            rows = db.execute(
                f"""
                SELECT d.document_id, d.course_id, d.source_path, d.publication_epoch,
                       s.version, s.source_id, s.locator_json, v.content_sha256,
                       local_fts.origin, local_fts.title, local_fts.body, bm25(local_fts) AS rank
                FROM local_fts
                JOIN local_sources s ON s.id = local_fts.source_rowid
                JOIN local_documents d ON d.document_id = s.document_id
                    AND d.current_version = s.version AND d.deleted = 0
                JOIN local_versions v ON v.document_id = s.document_id AND v.version = s.version
                WHERE local_fts MATCH ? {scope}
                ORDER BY rank, d.document_id, s.ordinal, local_fts.rowid
                LIMIT ?
                """,
                [match, *operands, request.top_k],
            ).fetchall()
        return [
            LocalHit(
                document_id=row["document_id"],
                course_id=row["course_id"],
                version=row["version"],
                source_id=row["source_id"],
                publication_epoch=row["publication_epoch"],
                content_sha256=row["content_sha256"],
                source_path=row["source_path"],
                locator=json.loads(row["locator_json"]),
                origin=row["origin"],
                title=row["title"],
                text=row["body"],
                score=-row["rank"],
            )
            for row in rows
        ]

    def delete_local(self, document_id: str) -> bool:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT course_id FROM local_documents WHERE document_id = ? AND deleted = 0",
                (document_id,),
            ).fetchone()
            if row is None:
                return False
            cursor = db.execute(
                """
                UPDATE local_documents
                SET deleted = 1, current_version = NULL, pending_version = NULL, pending_at = NULL,
                    publication_epoch = publication_epoch + 1
                WHERE document_id = ? AND deleted = 0
                """,
                (document_id,),
            )
            if cursor.rowcount == 1:
                invalidate_document_kg_in_tx(db, document_id, reason="document_deleted")
                sync_course_kg_in_tx(db, row["course_id"], now=_now())
                return True
            return False

    def current_sources(self, document_id: str, *, course_ids: Sequence[str]) -> list[SourceDraft]:
        if not course_ids:
            return []
        scope, params = _scope_sql(course_ids, [document_id])
        with self._connect() as db:
            promote_due(db, _now())
            rows = db.execute(
                f"""
                SELECT s.* FROM local_sources s
                JOIN local_documents d ON d.document_id = s.document_id
                    AND d.current_version = s.version AND d.deleted = 0
                WHERE 1 {scope} ORDER BY s.ordinal, s.source_id
                """,
                params,
            ).fetchall()
        return [_source(row) for row in rows]

    def read_source(self, key: SourceKey, *, course_ids: Sequence[str]) -> HistoricalSource:
        if not course_ids:
            return HistoricalSource("not_accessible")
        scope, params = _scope_sql(course_ids, [key.document_id])
        with self._connect() as db:
            promote_due(db, _now())
            row = db.execute(
                f"""
                SELECT s.*, v.content_sha256, v.original, d.publication_epoch, d.current_version
                FROM local_sources s
                JOIN local_documents d ON d.document_id = s.document_id AND d.deleted = 0
                JOIN local_versions v ON v.document_id = s.document_id AND v.version = s.version
                WHERE s.version = ? AND s.source_id = ? AND v.published_at IS NOT NULL
                    AND d.current_version IS NOT NULL {scope}
                """,
                [key.version, key.source_id, *params],
            ).fetchone()
        if row is None:
            return HistoricalSource("not_accessible")
        original = row["original"]
        digest = row["content_sha256"]
        epoch = row["publication_epoch"]
        is_current = row["current_version"] == key.version
        if original is None or hashlib.sha256(original).hexdigest() != digest:
            return HistoricalSource(
                "original_unavailable",
                _source(row),
                digest,
                publication_epoch=epoch,
                is_current=is_current,
            )
        return HistoricalSource("available", _source(row), digest, original, epoch, is_current)

    def legacy_sources(
        self,
        legacy_document_id: str,
        legacy_version: int,
        legacy_key: str,
        *,
        course_ids: Sequence[str],
    ) -> list[SourceKey]:
        if not course_ids:
            return []
        scope, params = _scope_sql(course_ids, [])
        with self._connect() as db:
            promote_due(db, _now())
            rows = db.execute(
                f"""
                SELECT s.document_id, s.version, s.source_id
                FROM local_legacy_keys k
                JOIN local_sources s ON s.id = k.source_rowid
                JOIN local_documents d ON d.document_id = s.document_id AND d.deleted = 0
                JOIN local_versions v ON v.document_id = s.document_id AND v.version = s.version
                WHERE k.legacy_document_id = ? AND k.legacy_version = ? AND k.legacy_key = ?
                    AND v.published_at IS NOT NULL AND d.current_version IS NOT NULL {scope}
                ORDER BY s.ordinal, s.source_id
                """,
                [legacy_document_id, legacy_version, legacy_key, *params],
            ).fetchall()
        return [SourceKey(**dict(row)) for row in rows]

    def write_enhancement(
        self,
        key: SourceKey,
        *,
        course_ids: Sequence[str],
        content_sha256: str,
        publication_epoch: int,
        text: str,
        origin: Literal["enhancement", "ocr"] = "enhancement",
    ) -> bool:
        if not course_ids or not text.strip():
            return False
        scope, params = _scope_sql(course_ids, [key.document_id])
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            promote_due(db, _now())
            row = db.execute(
                f"""
                SELECT s.id, s.title, s.kind, s.state, s.text, v.document_format
                FROM local_sources s
                JOIN local_documents d ON d.document_id = s.document_id
                    AND d.current_version = s.version AND d.deleted = 0
                JOIN local_versions v ON v.document_id = s.document_id AND v.version = s.version
                WHERE s.version = ? AND s.source_id = ? AND v.content_sha256 = ?
                    AND d.publication_epoch = ? AND v.original IS NOT NULL {scope}
                """,
                [key.version, key.source_id, content_sha256, publication_epoch, *params],
            ).fetchone()
            if row is None:
                return False
            if origin == "ocr":
                if (
                    row["document_format"] != "pdf"
                    or row["kind"] != "page"
                    or row["state"] != "needs_ocr"
                    or row["text"].strip()
                ):
                    return False
                # Retain the exact derivative previously referenced by a completed answer.
                saved = db.execute(
                    "SELECT body FROM local_fts WHERE rowid = ? AND origin = 'ocr'", (-row["id"],)
                ).fetchone()
                if saved:
                    return saved["body"] == text.strip()
            _save_fts(db, row["id"], row["title"], text.strip(), origin)
            return True

    def read_ocr_text(self, key: SourceKey, *, course_ids: Sequence[str]) -> str:
        """Re-open a saved OCR derivative under the same version access rules as its page."""
        if not course_ids:
            return ""
        scope, args = _scope_sql(course_ids, [key.document_id])
        with self._connect() as db:
            promote_due(db, _now())
            row = db.execute(
                f"""
                SELECT f.body FROM local_sources s
                JOIN local_documents d ON d.document_id = s.document_id AND d.deleted = 0
                JOIN local_versions v ON v.document_id = s.document_id AND v.version = s.version
                JOIN local_fts f ON f.rowid = -s.id AND f.origin = 'ocr'
                WHERE s.source_id = ? AND s.version = ? AND v.published_at IS NOT NULL
                    AND v.original IS NOT NULL AND d.current_version IS NOT NULL {scope}
                """,
                [key.source_id, key.version, *args],
            ).fetchone()
        return row["body"] if row else ""


def _now() -> float:
    return datetime.now(UTC).timestamp()


def _get_document(db: Connection, document_id: str) -> Row | None:
    return db.execute(
        "SELECT * FROM local_documents WHERE document_id = ?", (document_id,)
    ).fetchone()


def _path_owner(db: Connection, course_id: str, path: str) -> Row | None:
    return db.execute(
        "SELECT d.* FROM local_paths p JOIN local_documents d USING(document_id) "
        "WHERE p.course_id = ? AND p.canonical_path = ?",
        (course_id, path),
    ).fetchone()


def _claim_plan(db: Connection, plan: ImportPlan, legacy: LegacyBinding | None) -> Row:
    owner = _path_owner(db, plan.course_id, plan.source_path)
    if plan.document_id is None:
        if owner is not None:
            raise PublicationConflict("路径已由其他操作登记，请重试")
        if legacy:
            document_id = legacy.document_id
            if _get_document(db, document_id) is not None:
                raise PublicationConflict("旧资料 ID 已绑定其他路径或课程")
        else:
            document_id = f"local_{uuid4().hex}"
            while _get_document(db, document_id) is not None:
                document_id = f"local_{uuid4().hex}"
        db.execute(
            "INSERT INTO local_documents(document_id, course_id, source_path) VALUES(?, ?, ?)",
            (document_id, plan.course_id, plan.source_path),
        )
    else:
        document_id = plan.document_id
        doc = _get_document(db, document_id)
        if (
            doc is None
            or doc["course_id"] != plan.course_id
            or doc["publication_epoch"] != plan.epoch
            or (owner is not None and owner["document_id"] != document_id)
        ):
            raise PublicationConflict("资料在准备期间已变化，请重试")
    if owner is None:
        db.execute(
            "INSERT INTO local_paths(course_id, canonical_path, document_id) VALUES(?, ?, ?)",
            (plan.course_id, plan.source_path, document_id),
        )
    return _get_document(db, document_id)


def _validate_sources(sources: list[SourceDraft]) -> None:
    parents: dict[str, SourceDraft] = {}
    for source in sources:
        parent = parents.get(source.parent_id)
        if not source.source_id or source.source_id in parents:
            raise ValueError("来源 ID 重复或为空")
        if source.kind == "notebook_output" and (parent is None or parent.kind != "cell"):
            raise ValueError("Notebook 输出必须关联同版 Cell")
        if source.kind == "notebook_output" and source.locator.get(
            "cell_index"
        ) != parent.locator.get("cell_index"):
            raise ValueError("Notebook 输出与父 Cell 位置不符")
        if source.kind == "page_object" and (
            parent is None or parent.kind not in {"page", "page_object"}
        ):
            raise ValueError("页内对象必须关联同版父页或父对象")
        if source.kind == "page_object" and source.locator.get("page") != parent.locator.get(
            "page"
        ):
            raise ValueError("页内对象与父页位置不符")
        if source.kind in {"page", "cell"} and source.parent_id is not None:
            raise ValueError("页或 Cell 不可使用对象父键")
        parents[source.source_id] = source


def _insert_version(
    db: Connection,
    document: Row,
    plan: ImportPlan,
    parsed: ParsedSource,
    digest: str,
    document_format: str,
    original: bytes | None,
) -> int:
    document_id = document["document_id"]
    version = db.execute(
        "SELECT COALESCE(max(version), 0) + 1 FROM local_versions WHERE document_id = ?",
        (document_id,),
    ).fetchone()[0]
    db.execute(
        """
        INSERT INTO local_versions(document_id, version, content_sha256, source_path,
                                   document_format, title, original)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (document_id, version, digest, plan.source_path, document_format, parsed.title, original),
    )
    return version


def _save_sources(
    db: Connection, document_id: str, version: int, sources: list[SourceDraft]
) -> None:
    for item in sources:
        cursor = db.execute(
            """
            INSERT INTO local_sources(document_id, version, source_id, parent_id, ordinal, kind,
                                      title, text, locator_json, state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id, version, source_id) DO UPDATE SET
                title = excluded.title, text = excluded.text, ordinal = excluded.ordinal,
                locator_json = excluded.locator_json, state = excluded.state
            RETURNING id
            """,
            (
                document_id,
                version,
                item.source_id,
                item.parent_id,
                item.ordinal,
                item.kind,
                item.title,
                item.text,
                json.dumps(item.locator, ensure_ascii=False),
                item.state,
            ),
        )
        source_rowid = cursor.fetchone()[0]
        db.execute("DELETE FROM local_fts WHERE rowid = ?", (source_rowid * 2,))
        if item.text.strip():
            origin = item.locator.get("text_origin", "native_text")
            _save_fts(db, source_rowid, item.title, item.text, origin)


def _append_missing_pdf_images(
    db: Connection, document_id: str, version: int, sources: list[SourceDraft]
) -> bool:
    candidates = [
        item
        for item in sources
        if item.kind == "page_object"
        and (
            (
                not item.text.strip()
                and item.locator.get("object_type") == "image"
                and item.locator.get("position_basis")
                in {"pdfium_native_image_extents_v1", "pdfium_native_vector_extents_v1"}
            )
            or (
                item.text.strip()
                and item.locator.get("object_type") == "table"
                and item.locator.get("position_basis") == "axis_aligned_native_rule_grid_v1"
            )
            or (
                not item.text.strip()
                and item.locator.get("object_type") in {"caption", "formula"}
                and item.locator.get("position_basis") == "pdfium_native_text_block_v1"
            )
        )
    ]
    if not candidates:
        return False

    rows = db.execute(
        """
        SELECT source_id, kind, ordinal, json_extract(locator_json, '$.page') AS page
        FROM local_sources WHERE document_id = ? AND version = ?
        """,
        (document_id, version),
    ).fetchall()
    existing = {row["source_id"] for row in rows}
    pages = {row["source_id"]: row["page"] for row in rows if row["kind"] == "page"}
    start = max((row["ordinal"] for row in rows), default=-1) + 1
    missing = (
        item
        for item in candidates
        if item.source_id not in existing
        and item.parent_id in pages
        and pages[item.parent_id] == item.locator.get("page")
    )
    additions = [
        candidate.model_copy(update={"ordinal": ordinal})
        for ordinal, candidate in enumerate(missing, start)
    ]
    _save_sources(db, document_id, version, additions)
    added_types = {
        obj_type
        for item in additions
        if isinstance(obj_type := item.locator.get("object_type"), str)
        and obj_type in {"table", "image", "caption", "formula"}
    }
    if added_types:
        # Preserve existing source_id keys and FTS rows; reorder ordinals only within each
        # object type's own slot pool so visual reading order matches a cold import.
        ordinals = {row["source_id"]: row["ordinal"] for row in rows}
        ordinals.update((item.source_id, item.ordinal) for item in additions)
        for obj_type in sorted(added_types):
            type_ids = [
                item.source_id
                for item in candidates
                if item.locator.get("object_type") == obj_type and item.source_id in ordinals
            ]
            slots = sorted(ordinals[source_id] for source_id in type_ids)
            db.executemany(
                "UPDATE local_sources SET ordinal = ? "
                "WHERE document_id = ? AND version = ? AND source_id = ?",
                (
                    (slot, document_id, version, source_id)
                    for source_id, slot in zip(type_ids, slots, strict=True)
                    if slot != ordinals[source_id]
                ),
            )
    return bool(additions)


def _save_fts(db: Connection, rowid: int, title: str, text: str, origin: str) -> None:
    # Native and visual derivatives keep their existing positive row IDs.
    fts_id = -rowid if origin == "ocr" else 2 * rowid + (origin == "enhancement")
    db.execute(
        "INSERT OR REPLACE INTO local_fts(rowid, title, body, source_rowid, origin) "
        "VALUES (?, ?, ?, ?, ?)",
        (fts_id, title, text, rowid, origin),
    )


def _source(row: Row) -> SourceDraft:
    return SourceDraft(
        source_id=row["source_id"],
        kind=row["kind"],
        parent_id=row["parent_id"],
        ordinal=row["ordinal"],
        title=row["title"],
        text=row["text"],
        locator=json.loads(row["locator_json"]),
        state=row["state"],
    )


def _scope_sql(course_ids: Sequence[str], document_ids: Sequence[str]) -> tuple[str, list[str]]:
    conditions = []
    parameters = []
    for column, values in (("course_id", course_ids), ("document_id", document_ids)):
        if values:
            holders = ", ".join("?" for _ in values)
            conditions.append(f" AND d.{column} IN ({holders})")
            parameters.extend(values)
    return "".join(conditions), parameters


def _validate_legacy_snapshot(
    db: Connection,
    plan: ImportPlan,
    legacy: LegacyBinding,
    digest: str,
    cutoff: float,
) -> None:
    current = db.execute(
        """
        SELECT d.course_id, d.source_path, v.content_hash, v.version
        FROM documents d JOIN document_versions v USING(document_id)
        WHERE d.document_id = ? AND d.deleted_at IS NULL AND v.state = 'active'
          AND (v.release_at IS NULL OR v.release_at <= ?)
        """,
        (legacy.document_id, _serialize_datetime(datetime.fromtimestamp(cutoff, UTC))),
    ).fetchone()
    if current is None or (
        current["course_id"],
        current["source_path"],
        current["content_hash"],
        current["version"],
    ) != (plan.course_id, plan.source_path, digest, legacy.version):
        raise PublicationConflict("旧索引在回填期间已变化，请重试")


def _bind_legacy(
    db: Connection,
    document_id: str,
    version: int,
    plan: ImportPlan,
    legacy: LegacyBinding,
    digest: str,
) -> None:
    identity = (legacy.document_id, legacy.version)
    existing = db.execute(
        "SELECT course_id, content_sha256, document_id, version FROM local_legacy_versions "
        "WHERE legacy_document_id = ? AND legacy_version = ?",
        identity,
    ).fetchone()
    target = (plan.course_id, digest, document_id, version)
    other_owner = db.execute(
        "SELECT 1 FROM local_legacy_versions WHERE legacy_document_id = ? "
        "AND (course_id <> ? OR document_id <> ?)",
        (legacy.document_id, plan.course_id, document_id),
    ).fetchone()
    if other_owner is not None:
        raise PublicationConflict("旧资料已有不同课程或资料身份")
    if existing is not None:
        if tuple(existing) != target:
            raise PublicationConflict("旧 ID/版号已有不同归属或原件")
        if not legacy.refine:
            return
    else:
        db.execute(
            "INSERT INTO local_legacy_versions VALUES (?, ?, ?, ?, ?, ?)",
            (*identity, *target),
        )
    for key, sources in legacy.keys.items():
        target_rows = []
        for source_id in sources:
            row = db.execute(
                "SELECT id, kind FROM local_sources "
                "WHERE document_id=? AND version=? AND source_id=?",
                (document_id, version, source_id),
            ).fetchone()
            if row is None:
                raise ValueError(f"旧来源映射指向未登记对象：{source_id}")
            target_rows.append(row)
        if existing is not None:
            if not _refine_alias(db, (*identity, key), target_rows):
                continue
        db.executemany(
            "INSERT OR IGNORE INTO local_legacy_keys VALUES (?, ?, ?, ?)",
            [(*identity, key, row["id"]) for row in target_rows],
        )


def _refine_alias(db: Connection, key: tuple[str, int, str], targets: list[Row]) -> bool:
    """Upgrade a previous parent-only link after verifying the exact original."""

    precise = {"page_object", "notebook_output"}
    old = db.execute(
        """
        SELECT s.kind FROM local_legacy_keys k JOIN local_sources s ON s.id = k.source_rowid
        WHERE k.legacy_document_id = ? AND k.legacy_version = ? AND k.legacy_key = ?
        """,
        key,
    ).fetchall()
    if not old:
        return True
    if any(row["kind"] in precise for row in old) or not any(
        row["kind"] in precise for row in targets
    ):
        return False
    db.execute(
        "DELETE FROM local_legacy_keys "
        "WHERE legacy_document_id = ? AND legacy_version = ? AND legacy_key = ?",
        key,
    )
    return True


def _activate(
    db: Connection,
    document: Row,
    plan: ImportPlan,
    version: int,
    cutoff: float,
    release_at: datetime | None,
    backfill: bool,
) -> Literal["published", "scheduled", "skipped", "backfilled"]:
    document_id = document["document_id"]
    if backfill and (
        document["current_version"] is not None
        or document["deleted"]
        or document["pending_version"] is not None
    ):
        if document["pending_version"] != version:
            db.execute(
                "UPDATE local_versions SET published_at = COALESCE(published_at, ?) "
                "WHERE document_id = ? AND version = ?",
                (cutoff, document_id, version),
            )
        return "backfilled"
    due = release_at.replace(tzinfo=UTC) if release_at and release_at.tzinfo is None else release_at
    if (
        not document["deleted"]
        and document["current_version"] == version
        and document["pending_version"] is None
    ):
        if plan.move:
            db.execute(
                "UPDATE local_documents SET source_path = ?, "
                "publication_epoch = publication_epoch + 1 "
                "WHERE document_id = ?",
                (plan.source_path, document_id),
            )
        return "skipped"
    if due is not None and due.timestamp() > cutoff:
        if (
            document["pending_version"] == version
            and document["pending_at"] == due.timestamp()
            and not plan.move
        ):
            return "scheduled"
        db.execute(
            """
            UPDATE local_documents SET pending_version = ?, pending_at = ?, deleted = 0,
                   source_path = ?, publication_epoch = publication_epoch + 1 WHERE document_id = ?
            """,
            (
                version,
                due.timestamp(),
                plan.source_path if plan.move else document["source_path"],
                document_id,
            ),
        )
        return "scheduled"
    db.execute(
        "UPDATE local_versions SET published_at = COALESCE(published_at, ?) "
        "WHERE document_id = ? AND version = ?",
        (cutoff, document_id, version),
    )
    db.execute(
        """
        UPDATE local_documents
        SET current_version = ?, deleted = 0, pending_version = NULL, pending_at = NULL,
            source_path = ?, publication_epoch = publication_epoch + 1 WHERE document_id = ?
        """,
        (version, plan.source_path if plan.move else document["source_path"], document_id),
    )
    return "published"
