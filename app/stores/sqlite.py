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
from app.schemas.document import ParsedDocument, SlideChunkingResult, SlideDeck
from app.schemas.indexing import PreparedDocumentIndex
from app.schemas.rag import (
    DocumentContent,
    DocumentSummary,
    IndexStatus,
    JobState,
    SearchHit,
    SourceReference,
)

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
        document_type: str = "notebook",
        force: bool = False,
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
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(document_id) DO UPDATE SET
                    course_id = excluded.course_id,
                    source_path = excluded.source_path,
                    title = excluded.title,
                    document_type = excluded.document_type,
                    updated_at = excluded.updated_at,
                    deleted_at = NULL
                """,
                (
                    document.document_id,
                    clean_course_id,
                    document.source_path,
                    document.title,
                    document_type,
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
                and not force
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

    def save_slide_structure(
        self,
        deck: SlideDeck,
        chunking: SlideChunkingResult,
        *,
        version: int,
        asset_paths: dict[str, str],
    ) -> None:
        """保存页、章节、chunk 与已复制到受管目录的资源路径。"""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT state FROM document_versions WHERE document_id = ? AND version = ?",
                (deck.document_id, version),
            ).fetchone()
            if row is None or row["state"] != "building":
                raise ValueError("只能为 building 版本保存幻灯片结构")
            for slide in deck.slides:
                connection.execute(
                    """INSERT INTO content_blocks(document_id, version, block_id, kind,
                       cell_id, cell_index, page, payload_json)
                       VALUES (?, ?, ?, 'slide', NULL, NULL, ?, ?)""",
                    (deck.document_id, version, slide.slide_id, slide.slide_no, _model_json(slide)),
                )
                for asset in slide.assets:
                    connection.execute(
                        """INSERT INTO assets(document_id, version, asset_id, block_id,
                           media_type, path, caption) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            deck.document_id,
                            version,
                            asset.asset_id,
                            slide.slide_id,
                            asset.media_type,
                            asset_paths[asset.path],
                            asset.caption,
                        ),
                    )
                if slide.page_image_path:
                    connection.execute(
                        """INSERT INTO assets(document_id, version, asset_id, block_id,
                           media_type, path, caption) VALUES (?, ?, ?, ?, 'image/png', ?, NULL)""",
                        (
                            deck.document_id,
                            version,
                            f"{slide.slide_id}:page",
                            slide.slide_id,
                            asset_paths[slide.page_image_path],
                        ),
                    )
            connection.executemany(
                """INSERT INTO sections(document_id, version, section_id, heading, payload_json)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (
                        deck.document_id,
                        version,
                        section.section_id,
                        section.title,
                        _model_json(section),
                    )
                    for section in chunking.sections
                ],
            )
            connection.executemany(
                """INSERT INTO chunks(document_id, version, chunk_id, section_id, kind,
                   previous_chunk_id, next_chunk_id, payload_json)
                   VALUES (?, ?, ?, ?, 'slide', ?, ?, ?)""",
                [
                    (
                        deck.document_id,
                        version,
                        chunk.chunk_id,
                        chunk.section_id,
                        chunk.previous_chunk_id,
                        chunk.next_chunk_id,
                        _model_json(chunk),
                    )
                    for chunk in chunking.chunks
                ],
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
            WHERE {" AND ".join(conditions)}
            ORDER BY d.document_id
        """
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            VisibleVersion(document_id=row["document_id"], version=row["version"]) for row in rows
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

    def read_document(
        self, document_id: str, version: int, *, course_ids: list[str], document_ids: list[str]
    ) -> DocumentContent:
        """在同一读事务中核对范围/发布日期/版本，按页或 cell 顺序读取全部 chunk。"""

        with self._connect() as connection:
            connection.execute("BEGIN")
            row = connection.execute(
                """SELECT d.*, v.version, v.release_at, v.indexed_at
                   FROM documents d JOIN document_versions v ON v.document_id=d.document_id
                   WHERE d.document_id=? AND v.version=? AND d.deleted_at IS NULL
                     AND v.state='active' AND (v.release_at IS NULL OR v.release_at<=?)""",
                (document_id, version, _utc_now()),
            ).fetchone()
            if (
                row is None
                or (course_ids and row["course_id"] not in course_ids)
                or (document_ids and document_id not in document_ids)
            ):
                raise ValueError("文档不在当前范围、不可见或版本已变化；请重新选择")
            parameters = (document_id, version)
            blocks = connection.execute(
                "SELECT * FROM content_blocks WHERE document_id=? AND version=?",
                parameters,
            ).fetchall()
            assets = connection.execute(
                "SELECT * FROM assets WHERE document_id=? AND version=? ORDER BY asset_id",
                parameters,
            ).fetchall()
            chunks = [
                json.loads(item["payload_json"])
                for item in connection.execute(
                    "SELECT payload_json FROM chunks WHERE document_id=? AND version=?",
                    parameters,
                ).fetchall()
            ]
            headings = {
                item["section_id"]: item["heading"] or ""
                for item in connection.execute(
                    "SELECT section_id, heading FROM sections WHERE document_id=? AND version=?",
                    parameters,
                ).fetchall()
            }
        cell_order = {
            item["cell_id"]: item["cell_index"] or 0 for item in blocks if item["cell_id"]
        }
        asset_paths = {item["asset_id"]: item["path"] for item in assets}
        slide_images = {
            item["block_id"]: item["path"] for item in assets if item["asset_id"].endswith(":page")
        }
        chunks.sort(
            key=lambda chunk: (
                chunk.get(
                    "slide_no",
                    min((cell_order.get(cell, 0) for cell in chunk.get("cell_ids", [])), default=0),
                ),
                chunk.get("split_index", 0),
                chunk["chunk_id"],
            )
        )
        if chunks and "slide_no" not in chunks[0]:
            # Notebook 同一 cell 可以拆出多块，哈希 ID 不表示正文顺序。
            by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
            ordered, visited = [], set()
            roots = [chunk for chunk in chunks if chunk.get("previous_chunk_id") not in by_id]
            for root in roots:
                chunk = root
                while chunk and chunk["chunk_id"] not in visited:
                    visited.add(chunk["chunk_id"])
                    ordered.append(chunk)
                    chunk = by_id.get(chunk.get("next_chunk_id"))
            if len(ordered) != len(chunks):
                raise ValueError("Notebook chunk 顺序链不完整或成环，不能生成完整总结")
            chunks = ordered
        hits = []
        empty_pages = set()
        text_pages = set()
        for chunk in chunks:
            heading = chunk.get("title") or headings.get(chunk["section_id"], "")
            body = chunk["display_text"].strip()
            page = chunk.get("slide_no")
            if page is not None and not body:
                empty_pages.add(page)
            elif page is not None:
                text_pages.add(page)
            image_path = slide_images.get(chunk.get("slide_id"))
            if image_path is None:
                image_path = next(
                    (asset_paths[key] for key in chunk.get("asset_ids", []) if key in asset_paths),
                    None,
                )
            hits.append(
                SearchHit(
                    chunk_id=chunk["chunk_id"],
                    section_id=chunk["section_id"],
                    representation_type="text_content",
                    score=0.0,
                    content="\n".join(part for part in (heading, body) if part)
                    or "（本页未提取到文字）",
                    source=SourceReference(
                        course_id=row["course_id"],
                        document_id=document_id,
                        file=row["source_path"],
                        section_path=chunk.get("section_path") or ([heading] if heading else []),
                        cell_ids=chunk.get("cell_ids", []),
                        page=page,
                        image_path=image_path,
                        version=version,
                    ),
                )
            )
        pages = sorted({item["page"] for item in blocks if item["page"] is not None})
        chunk_pages = {hit.source.page for hit in hits}
        if not hits or (pages and not set(pages) <= chunk_pages):
            raise ValueError("文档结构缺少 chunk 或页码覆盖不完整，不能生成完整总结")
        return DocumentContent(
            document=DocumentSummary(
                document_id=document_id,
                course_id=row["course_id"],
                source_path=row["source_path"],
                version=version,
                release_at=_parse_datetime(row["release_at"]),
                indexed_at=_parse_datetime(row["indexed_at"]) or datetime.now(UTC),
            ),
            title=row["title"],
            outline=list(
                dict.fromkeys(
                    heading for hit in hits for heading in hit.source.section_path if heading
                )
            ),
            hits=hits,
            pages=pages,
            image_only_pages=sorted(empty_pages - text_pages),
        )

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
