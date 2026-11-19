"""Local import orchestration. Parsing runs from the same bytes SQLite archives."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from app.config import Settings
from app.rag.local_migration import (
    LegacySnapshot,
    load_legacy_bundle,
    load_legacy_sqlite,
    map_legacy_sources,
)
from app.rag.parsers.local_notebook import parse_local_notebook
from app.rag.parsers.local_slides import parse_local_slides
from app.schemas.local import ImportResult, LocalHit, ParsedSource
from app.schemas.rag import SearchRequest
from app.stores.local_library import LegacyBinding, LocalLibraryStore

_SUFFIXES = {".pdf", ".pptx", ".ipynb"}


class LocalLibrary:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.course_root = settings.course_root.expanduser().resolve()
        self.store = LocalLibraryStore(settings)
        self.store.initialize()

    def import_paths(
        self,
        course_id: str,
        paths: list[Path],
        *,
        document_id: str | None = None,
        release_at: datetime | None = None,
    ) -> list[ImportResult]:
        if not course_id.strip() or not paths:
            raise ValueError("请选择课程与文件或目录")
        if document_id is not None and len(paths) != 1:
            raise ValueError("显式资料 ID 只能用于单文件更新")
        selected: dict[Path, None] = {}
        errors: list[ImportResult] = []
        for path in paths:
            try:
                source = self._selected_path(path)
                if source.is_dir() and document_id is not None:
                    raise ValueError("显式资料 ID 只能用于单文件更新")
                selected.update(dict.fromkeys(self._files(source)))
            except (OSError, ValueError) as error:
                errors.append(_failed(path, error))

        results = []
        for path in selected:
            try:
                plan = self.store.prepare(course_id, path, document_id=document_id)
                raw = path.read_bytes()
                parsed = _parse_snapshot(path, raw)
                result = self.store.publish(
                    plan,
                    parsed,
                    original=raw,
                    content_sha256=hashlib.sha256(raw).hexdigest(),
                    document_format=path.suffix.casefold()[1:],
                    release_at=release_at,
                )
            except Exception as error:
                result = _failed(path, error)
            results.append(result)
        return [*results, *errors]

    def search(self, request: SearchRequest) -> list[LocalHit]:
        return self.store.search_local(request)

    def delete(self, document_id: str) -> bool:
        return self.store.delete_local(document_id)

    def migrate_legacy(self, document_id: str) -> ImportResult:
        return self._migrate(load_legacy_sqlite(self.settings, document_id), from_sqlite=True)

    def import_legacy_bundle(
        self,
        *,
        course_id: str,
        source_path: Path,
        bundle_dir: Path,
        content_sha256: str,
        version: int = 1,
    ) -> ImportResult:
        snapshot = load_legacy_bundle(
            course_id=course_id,
            source_path=source_path,
            bundle_dir=bundle_dir,
            content_sha256=content_sha256,
            version=version,
        )
        return self._migrate(snapshot, from_sqlite=False)

    def _migrate(self, snapshot: LegacySnapshot, *, from_sqlite: bool) -> ImportResult:
        plan = self.store.prepare(snapshot.course_id, snapshot.source_path.absolute())
        original, native = self._legacy_original(snapshot)
        parsed, keys = map_legacy_sources(snapshot, native)
        if not parsed.sources:
            raise ValueError("旧索引没有可核对的页或 Cell")
        return self.store.publish(
            plan,
            parsed,
            original=original,
            content_sha256=snapshot.content_sha256,
            document_format=snapshot.document_format,
            release_at=snapshot.release_at,
            legacy=LegacyBinding(
                snapshot.document_id, snapshot.version, keys, from_sqlite, refine=native is not None
            ),
        )

    def _legacy_original(
        self, snapshot: LegacySnapshot
    ) -> tuple[bytes | None, ParsedSource | None]:
        try:
            path = self._selected_path(snapshot.source_path)
            raw = path.read_bytes()
        except (OSError, ValueError):
            return None, None
        if hashlib.sha256(raw).hexdigest() != snapshot.content_sha256:
            return None, None
        try:
            return raw, _parse_snapshot(snapshot.source_path, raw)
        except (OSError, ValueError, RuntimeError):
            return raw, None

    def _selected_path(self, path: Path) -> Path:
        path = path.expanduser()
        selected = (self.course_root / path).resolve(strict=True)
        if not selected.is_relative_to(self.course_root):
            raise ValueError("所选路径位于 COURSE_ROOT 之外")
        return selected

    def _files(self, source: Path) -> Iterable[Path]:
        if source.is_file() and source.suffix.casefold() in _SUFFIXES:
            return [source]
        if not source.is_dir():
            raise ValueError("请选择 PDF、PPTX、IPYNB 或包含这些文件的目录")
        files = [
            path
            for path in sorted(source.rglob("*"))
            if path.suffix.casefold() in _SUFFIXES
            and not path.is_symlink()
            and path.is_file()
            and path.resolve().is_relative_to(self.course_root)
        ]
        if not files:
            raise ValueError("目录没有可导入的 PDF、PPTX 或 IPYNB")
        return files


def _parse_snapshot(path: Path, raw: bytes) -> ParsedSource:
    with TemporaryDirectory(prefix="srag-local-import-") as directory:
        snapshot = Path(directory) / path.name
        snapshot.write_bytes(raw)
        if path.suffix.casefold() == ".ipynb":
            return parse_local_notebook(snapshot)
        return parse_local_slides(snapshot)


def _failed(path: Path, error: Exception) -> ImportResult:
    return ImportResult(
        status="failed", source_path=str(path), error=str(error) or type(error).__name__
    )
