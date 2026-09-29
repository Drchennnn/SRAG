"""文档版本、结构关系与索引任务的 SQLite 存储。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings
from app.schemas.document import ParsedDocument
from app.schemas.indexing import PreparedDocumentIndex
from app.schemas.rag import DocumentSummary, IndexStatus, JobState

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class VersionPlan:
    """一次文件索引对应的版本决策。"""

    document_id: str
    version: int
    should_index: bool
    previous_active_version: int | None = None


@dataclass(frozen=True)
class VisibleVersion:
    """查询时允许访问的文档版本。"""

    document_id: str
    version: int


class MetadataStore:
    """使用短连接管理 SQLite；每个写操作都在独立事务中完成。"""

    def __init__(self, settings: Settings) -> None:
        self.path: Path = settings.sqlite_path
        self.asset_root: Path = settings.data_dir / "assets"

    def initialize(self) -> None:
        """创建数据库和当前 schema migration。"""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    course_id TEXT NOT NULL,
                    source_path TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_documents_course
                ON documents(course_id, deleted_at);

                CREATE TABLE IF NOT EXISTS document_versions (
                    document_id TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK(version >= 1),
                    content_hash TEXT NOT NULL,
                    release_at TEXT,
                    state TEXT NOT NULL
                        CHECK(state IN ('building', 'active', 'superseded', 'failed')),
                    created_at TEXT NOT NULL,
                    indexed_at TEXT,
                    error TEXT,
                    PRIMARY KEY(document_id, version),
                    FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_versions_visibility
                ON document_versions(state, release_at);

                CREATE TABLE IF NOT EXISTS content_blocks (
                    document_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    block_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    cell_id TEXT,
                    cell_index INTEGER,
                    page INTEGER,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(document_id, version, block_id),
                    FOREIGN KEY(document_id, version)
                        REFERENCES document_versions(document_id, version) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS assets (
                    document_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    asset_id TEXT NOT NULL,
                    block_id TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    caption TEXT,
                    PRIMARY KEY(document_id, version, asset_id),
                    FOREIGN KEY(document_id, version)
                        REFERENCES document_versions(document_id, version) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS cell_groups (
                    document_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    group_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(document_id, version, group_id),
                    FOREIGN KEY(document_id, version)
                        REFERENCES document_versions(document_id, version) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS sections (
                    document_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    section_id TEXT NOT NULL,
                    heading TEXT,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(document_id, version, section_id),
                    FOREIGN KEY(document_id, version)
                        REFERENCES document_versions(document_id, version) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    document_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    chunk_id TEXT NOT NULL,
                    section_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    previous_chunk_id TEXT,
                    next_chunk_id TEXT,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(document_id, version, chunk_id),
                    FOREIGN KEY(document_id, version)
                        REFERENCES document_versions(document_id, version) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS index_jobs (
                    job_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    indexed_files INTEGER NOT NULL DEFAULT 0,
                    skipped_files INTEGER NOT NULL DEFAULT 0,
                    failed_files INTEGER NOT NULL DEFAULT 0,
                    errors_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO schema_migrations(version, applied_at)
                VALUES (?, ?)
                """,
                (_SCHEMA_VERSION, _utc_now()),
            )

    def create_job(self, job_id: str) -> None:
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO index_jobs(
                    job_id, state, indexed_files, skipped_files, failed_files,
                    errors_json, created_at, updated_at
                ) VALUES (?, ?, 0, 0, 0, '{}', ?, ?)
                """,
                (job_id, JobState.RUNNING.value, now, now),
            )

    def finish_job(self, status: IndexStatus) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE index_jobs
                SET state = ?, indexed_files = ?, skipped_files = ?,
                    failed_files = ?, errors_json = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    status.state.value,
                    status.indexed_files,
                    status.skipped_files,
                    status.failed_files,
                    json.dumps(status.errors, ensure_ascii=False),
                    _utc_now(),
                    status.job_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"索引任务不存在：{status.job_id}")

    def get_job(self, job_id: str) -> IndexStatus | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM index_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return IndexStatus(
            job_id=row["job_id"],
            state=JobState(row["state"]),
            indexed_files=row["indexed_files"],
            skipped_files=row["skipped_files"],
            failed_files=row["failed_files"],
            errors=json.loads(row["errors_json"]),
        )

    def reserve_version(
        self,
        document: ParsedDocument,
        *,
        course_id: str,
        content_hash: str,
        release_at: datetime | None,
    ) -> VersionPlan:
        """登记文档并决定跳过或创建 building 版本。"""

        clean_course_id = course_id.strip()
        if not clean_course_id:
            raise ValueError("course_id 不能为空")
        release_value = _serialize_datetime(release_at) if release_at else None
        now = _utc_now()

        with self._connect() as connection:
            existing_path = connection.execute(
                "SELECT document_id, course_id FROM documents WHERE source_path = ?",
                (document.source_path,),
            ).fetchone()
            if existing_path is not None and existing_path["document_id"] != document.document_id:
                raise ValueError(f"文件路径已属于其他文档：{document.source_path}")

            course_changed = (
                existing_path is not None and existing_path["course_id"] != clean_course_id
            )
            connection.execute(
                """
                INSERT INTO documents(
                    document_id, course_id, source_path, title, document_type,
                    created_at, updated_at, deleted_at
                ) VALUES (?, ?, ?, ?, 'notebook', ?, ?, NULL)
                ON CONFLICT(document_id) DO UPDATE SET
                    course_id = excluded.course_id,
                    source_path = excluded.source_path,
                    title = excluded.title,
                    updated_at = excluded.updated_at,
                    deleted_at = NULL
                """,
                (
                    document.document_id,
                    clean_course_id,
                    document.source_path,
                    document.title,
                    now,
                    now,
                ),
            )

            active = connection.execute(
                """
                SELECT version, content_hash
                FROM document_versions
                WHERE document_id = ? AND state = 'active'
                ORDER BY version DESC
                LIMIT 1
                """,
                (document.document_id,),
            ).fetchone()
            if (
                active is not None
                and active["content_hash"] == content_hash
                and not course_changed
            ):
                connection.execute(
                    """
                    UPDATE document_versions
                    SET release_at = ?
                    WHERE document_id = ? AND version = ?
                    """,
                    (release_value, document.document_id, active["version"]),
                )
                return VersionPlan(
                    document_id=document.document_id,
                    version=active["version"],
                    should_index=False,
                    previous_active_version=active["version"],
                )

            connection.execute(
                """
                UPDATE document_versions
                SET state = 'failed', error = '被后续重试替代'
                WHERE document_id = ? AND state = 'building'
                """,
                (document.document_id,),
            )
            next_version = connection.execute(
                """
                SELECT COALESCE(MAX(version), 0) + 1 AS next_version
                FROM document_versions
                WHERE document_id = ?
                """,
                (document.document_id,),
            ).fetchone()["next_version"]
            connection.execute(
                """
                INSERT INTO document_versions(
                    document_id, version, content_hash, release_at, state,
                    created_at, indexed_at, error
                ) VALUES (?, ?, ?, ?, 'building', ?, NULL, NULL)
                """,
                (
                    document.document_id,
                    next_version,
                    content_hash,
                    release_value,
                    now,
                ),
            )
            return VersionPlan(
                document_id=document.document_id,
                version=next_version,
                should_index=True,
                previous_active_version=active["version"] if active is not None else None,
            )

    def save_structure(
        self,
        document: ParsedDocument,
        prepared: PreparedDocumentIndex,
        *,
        version: int,
    ) -> None:
        """保存指定 building 版本的 block、asset、group、section 与 chunk。"""

        with self._connect() as connection:
            state = connection.execute(
                """
                SELECT state FROM document_versions
                WHERE document_id = ? AND version = ?
                """,
                (document.document_id, version),
            ).fetchone()
            if state is None or state["state"] != "building":
                raise ValueError("只能为 building 版本保存结构")

            for table in ("content_blocks", "assets", "cell_groups", "sections", "chunks"):
                connection.execute(
                    f"DELETE FROM {table} WHERE document_id = ? AND version = ?",
                    (document.document_id, version),
                )

            for block in document.blocks:
                connection.execute(
                    """
                    INSERT INTO content_blocks(
                        document_id, version, block_id, kind, cell_id,
                        cell_index, page, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document.document_id,
                        version,
                        block.block_id,
                        block.kind.value,
                        block.cell_id,
                        block.cell_index,
                        block.page,
                        _model_json(block),
                    ),
                )
                for asset in block.assets:
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO assets(
                            document_id, version, asset_id, block_id,
                            media_type, path, caption
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            document.document_id,
                            version,
                            asset.asset_id,
                            block.block_id,
                            asset.media_type,
                            asset.path,
                            asset.caption,
                        ),
                    )

            connection.executemany(
                """
                INSERT INTO cell_groups(
                    document_id, version, group_id, kind, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        document.document_id,
                        version,
                        group.group_id,
                        group.kind.value,
                        _model_json(group),
                    )
                    for group in prepared.groups
                ],
            )
            connection.executemany(
                """
                INSERT INTO sections(
                    document_id, version, section_id, heading, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        document.document_id,
                        version,
                        section.section_id,
                        section.heading,
                        _model_json(section),
                    )
                    for section in prepared.sections
                ],
            )
            connection.executemany(
                """
                INSERT INTO chunks(
                    document_id, version, chunk_id, section_id, kind,
                    previous_chunk_id, next_chunk_id, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        document.document_id,
                        version,
                        chunk.chunk_id,
                        chunk.section_id,
                        chunk.kind.value,
                        chunk.previous_chunk_id,
                        chunk.next_chunk_id,
                        _model_json(chunk),
                    )
                    for chunk in prepared.chunks
                ],
            )

    def activate_version(self, document_id: str, version: int) -> None:
        """原子切换当前 active 版本。"""

        now = _utc_now()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT state FROM document_versions
                WHERE document_id = ? AND version = ?
                """,
                (document_id, version),
            ).fetchone()
            if row is None or row["state"] != "building":
                raise ValueError("只能激活 building 版本")
            connection.execute(
                """
                UPDATE document_versions
                SET state = 'superseded'
                WHERE document_id = ? AND state = 'active'
                """,
                (document_id,),
            )
            connection.execute(
                """
                UPDATE document_versions
                SET state = 'active', indexed_at = ?, error = NULL
                WHERE document_id = ? AND version = ?
                """,
                (now, document_id, version),
            )
            connection.execute(
                "UPDATE documents SET updated_at = ? WHERE document_id = ?",
                (now, document_id),
            )

    def fail_version(self, document_id: str, version: int, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE document_versions
                SET state = 'failed', error = ?
                WHERE document_id = ? AND version = ? AND state = 'building'
                """,
                (error[:4000], document_id, version),
            )

    def visible_versions(
        self,
        *,
        course_ids: list[str] | tuple[str, ...] = (),
        document_ids: list[str] | tuple[str, ...] = (),
        at: datetime | None = None,
    ) -> list[VisibleVersion]:
        """返回当前时刻允许查询的 active 版本。"""

        conditions = [
            "d.deleted_at IS NULL",
            "v.state = 'active'",
            "(v.release_at IS NULL OR v.release_at <= ?)",
        ]
        parameters: list[Any] = [_serialize_datetime(at or datetime.now(UTC))]
        clean_course_ids = _clean_values(course_ids)
        clean_document_ids = _clean_values(document_ids)
        if clean_course_ids:
            placeholders = ", ".join("?" for _ in clean_course_ids)
            conditions.append(f"d.course_id IN ({placeholders})")
            parameters.extend(clean_course_ids)
        if clean_document_ids:
            placeholders = ", ".join("?" for _ in clean_document_ids)
            conditions.append(f"d.document_id IN ({placeholders})")
            parameters.extend(clean_document_ids)

        query = f"""
            SELECT d.document_id, v.version
            FROM documents d
            JOIN document_versions v ON v.document_id = d.document_id
            WHERE {' AND '.join(conditions)}
            ORDER BY d.document_id
        """
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            VisibleVersion(document_id=row["document_id"], version=row["version"])
            for row in rows
        ]

    def list_documents(self, *, at: datetime | None = None) -> list[DocumentSummary]:
        """列出当前可查询的文档版本。"""

        cutoff = _serialize_datetime(at or datetime.now(UTC))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT d.document_id, d.course_id, d.source_path,
                       v.version, v.release_at, v.indexed_at
                FROM documents d
                JOIN document_versions v ON v.document_id = d.document_id
                WHERE d.deleted_at IS NULL
                  AND v.state = 'active'
                  AND (v.release_at IS NULL OR v.release_at <= ?)
                ORDER BY d.course_id, d.source_path
                """,
                (cutoff,),
            ).fetchall()
        return [
            DocumentSummary(
                document_id=row["document_id"],
                course_id=row["course_id"],
                source_path=row["source_path"],
                version=row["version"],
                release_at=_parse_datetime(row["release_at"]),
                indexed_at=_parse_datetime(row["indexed_at"]) or datetime.now(UTC),
            )
            for row in rows
        ]

    def has_document(self, document_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM documents
                WHERE document_id = ? AND deleted_at IS NULL
                """,
                (document_id,),
            ).fetchone()
        return row is not None

    def prune_superseded_structure(self, document_id: str, *, keep_version: int) -> int:
        """删除旧版本结构，并清理不再被 active 版本引用的受管资产文件。"""

        with self._connect() as connection:
            keep_paths = {
                row["path"]
                for row in connection.execute(
                    """
                    SELECT path FROM assets
                    WHERE document_id = ? AND version = ?
                    """,
                    (document_id, keep_version),
                ).fetchall()
            }
            stale_paths = [
                row["path"]
                for row in connection.execute(
                    """
                    SELECT path FROM assets
                    WHERE document_id = ? AND version != ?
                    """,
                    (document_id, keep_version),
                ).fetchall()
                if row["path"] not in keep_paths
            ]
            for table in ("content_blocks", "assets", "cell_groups", "sections", "chunks"):
                connection.execute(
                    f"DELETE FROM {table} WHERE document_id = ? AND version != ?",
                    (document_id, keep_version),
                )
        return self.delete_managed_asset_files(stale_paths)

    def asset_paths(self, document_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT path FROM assets WHERE document_id = ?",
                (document_id,),
            ).fetchall()
        return [row["path"] for row in rows]

    def delete_managed_asset_files(self, paths: list[str]) -> int:
        """只删除 data/assets 内的派生文件，跳过课程目录中的原始资源。"""

        asset_root = self.asset_root.expanduser().resolve()
        deleted = 0
        for raw_path in dict.fromkeys(paths):
            path = Path(raw_path).expanduser().resolve()
            if path == asset_root or asset_root not in path.parents:
                continue
            if path.is_file():
                path.unlink()
                deleted += 1
            parent = path.parent
            while parent != asset_root and asset_root in parent.parents:
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
        return deleted

    def delete_document(self, document_id: str) -> bool:
        """硬删除文档元数据；结构表通过外键级联删除。"""

        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (document_id,),
            )
        return cursor.rowcount == 1

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()


def _model_json(model: Any) -> str:
    return json.dumps(model.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))


def _clean_values(values: list[str] | tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _utc_now() -> str:
    return _serialize_datetime(datetime.now(UTC))


def _serialize_datetime(value: datetime) -> str:
    aware = value if value.tzinfo else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
