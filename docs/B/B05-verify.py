"""Independent B05 verification for local import, publication, and migration."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import httpx

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "docs" / "A" / "样例"
MANIFEST = ROOT / "docs" / "B" / "B04-送审版本.json"
FORMATS = {
    "pdf": ("Signal-reference.pdf", "lookahead", "page:1"),
    "pptx": ("Signal-slides.pptx", "Peak", "page:1:shape:3"),
    "ipynb": ("Signal-lab.ipynb", "NotebookImage", "cell:1"),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_note(path: Path, token: str) -> bytes:
    data = json.dumps(
        {
            "nbformat": 4,
            "nbformat_minor": 4,
            "metadata": {},
            "cells": [{"cell_type": "markdown", "metadata": {}, "source": f"# {token}"}],
        },
        separators=(",", ":"),
    ).encode()
    path.write_bytes(data)
    return data


def settings(root: Path):
    from app.config import Settings

    course = root / "course"
    course.mkdir(parents=True, exist_ok=True)
    return Settings(
        _env_file=None,
        course_root=course,
        data_dir=root / "data",
        sqlite_path=root / "library.sqlite3",
        dashscope_api_key="",
        deepseek_api_key="",
        qdrant_url="http://127.0.0.1:1",
        rag_search_backend="local",
    )


def search(library, query: str, document_id: str | None = None):
    from app.schemas.rag import SearchRequest, SearchScope

    return library.search(
        SearchRequest(
            query=query,
            scope=SearchScope(
                course_ids=["b05"],
                document_ids=[document_id] if document_id else [],
            ),
        )
    )


def read(library, document_id: str, version: int, source_id: str = "cell:0"):
    from app.schemas.local import SourceKey

    return library.store.read_source(
        SourceKey(document_id=document_id, version=version, source_id=source_id),
        course_ids=["b05"],
    )


class NetworkGuard:
    def __init__(self) -> None:
        self.attempts: list[str] = []

    def connect(self, sock, address):
        if sock.family in {socket.AF_INET, socket.AF_INET6}:
            self.attempts.append(repr(address))
            raise AssertionError(f"outbound socket blocked: {address!r}")
        return self.original_connect(sock, address)

    def connect_ex(self, sock, address):
        if sock.family in {socket.AF_INET, socket.AF_INET6}:
            self.attempts.append(repr(address))
            raise AssertionError(f"outbound socket blocked: {address!r}")
        return self.original_connect_ex(sock, address)

    def __enter__(self):
        self.original_connect = socket.socket.connect
        self.original_connect_ex = socket.socket.connect_ex
        self.stack = ExitStack()
        self.stack.enter_context(mock.patch.object(socket.socket, "connect", self.connect))
        self.stack.enter_context(mock.patch.object(socket.socket, "connect_ex", self.connect_ex))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


async def benchmark_child(kind: str, root: Path, output: Path) -> None:
    filename, query, expected_source = FORMATS[kind]
    source = SAMPLES / filename
    course = root / "course"
    course.mkdir(parents=True)
    cold_path = course / f"cold-{filename}"
    warm_path = course / f"warm-{filename}"
    shutil.copyfile(source, cold_path)
    shutil.copyfile(source, warm_path)
    raw = source.read_bytes()

    environment = {
        "COURSE_ROOT": str(course),
        "DATA_DIR": str(root / "data"),
        "SQLITE_PATH": str(root / "library.sqlite3"),
        "DASHSCOPE_API_KEY": "",
        "DEEPSEEK_API_KEY": "",
        "QDRANT_API_KEY": "",
        "QDRANT_URL": "http://127.0.0.1:1",
        "RAG_SEARCH_BACKEND": "local",
    }
    os.environ.update(environment)

    from app.clients.dashscope import DashScopeClient
    from app.clients.deepseek import DeepSeekClient
    from app.config import get_settings
    from app.main import create_app
    from app.stores.qdrant import QdrantStore

    get_settings.cache_clear()
    constructors = {"dashscope": 0, "deepseek": 0, "qdrant": 0}

    def denied(name: str):
        def constructor(*_args, **_kwargs):
            constructors[name] += 1
            raise AssertionError(f"external constructor reached: {name}")

        return constructor

    guard = NetworkGuard()
    app = create_app()
    rows = []
    with (
        guard,
        mock.patch.object(DashScopeClient, "__init__", denied("dashscope")),
        mock.patch.object(DeepSeekClient, "__init__", denied("deepseek")),
        mock.patch.object(QdrantStore, "__init__", denied("qdrant")),
    ):
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://b05") as client:
                for temperature, path in (("cold", cold_path), ("warm", warm_path)):
                    started = time.perf_counter_ns()
                    imported = await client.post(
                        "/api/rag/local/import",
                        json={"course_id": "b05", "paths": [path.name]},
                    )
                    imported_ns = time.perf_counter_ns()
                    imported.raise_for_status()
                    item = imported.json()[0]
                    searched = await client.post(
                        "/api/rag/local/search",
                        json={
                            "query": query,
                            "scope": {
                                "course_ids": ["b05"],
                                "document_ids": [item["document_id"]],
                            },
                            "top_k": 20,
                        },
                    )
                    first_hit_ns = time.perf_counter_ns()
                    searched.raise_for_status()
                    hits = searched.json()["hits"]
                    exact = [
                        hit
                        for hit in hits
                        if hit["document_id"] == item["document_id"]
                        and hit["version"] == item["version"]
                        and hit["source_id"] == expected_source
                    ]
                    assert exact, (kind, expected_source, hits)
                    assert item["status"] == "published"
                    assert item["version"] == 1
                    assert item["content_sha256"] == digest(raw)
                    rows.append(
                        {
                            "temperature": temperature,
                            "path": path.name,
                            "status": item["status"],
                            "document_id": item["document_id"],
                            "version": item["version"],
                            "publication_epoch": item["publication_epoch"],
                            "content_sha256": item["content_sha256"],
                            "source_id": expected_source,
                            "source_count": item["source_count"],
                            "searchable_count": item["searchable_count"],
                            "import_ms": round((imported_ns - started) / 1_000_000, 3),
                            "first_hit_ms": round((first_hit_ns - started) / 1_000_000, 3),
                            "hit_text": exact[0]["text"][:160],
                        }
                    )
    assert rows[0]["document_id"] != rows[1]["document_id"]
    assert rows[0]["content_sha256"] == rows[1]["content_sha256"]
    assert constructors == {"dashscope": 0, "deepseek": 0, "qdrant": 0}
    assert not guard.attempts
    output.write_text(
        json.dumps(
            {
                "format": kind,
                "file": filename,
                "bytes": len(raw),
                "sha256": digest(raw),
                "query": query,
                "expected_source": expected_source,
                "runs": rows,
                "external_constructor_attempts": constructors,
                "outbound_socket_attempts": guard.attempts,
                "credentials": {"dashscope": "empty", "deepseek": "empty"},
                "qdrant_url": environment["QDRANT_URL"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def crash_child(root: Path, path: Path, marker: Path) -> None:
    from app.rag.local_library import LocalLibrary
    from app.stores import local_library as storage

    library = LocalLibrary(settings(root))
    original = storage._activate

    def crash_after_activate(db, *args, **kwargs):
        result = original(db, *args, **kwargs)
        current = db.execute("SELECT current_version FROM local_documents LIMIT 1").fetchone()[0]
        bodies = [row[0] for row in db.execute("SELECT body FROM local_fts")]
        payload = json.dumps(
            {
                "inside_transaction_current_version": current,
                "inside_transaction_new_fts": any("newfaultword" in body for body in bodies),
                "activate_result": result,
            }
        ).encode()
        with marker.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os._exit(73)

    storage._activate = crash_after_activate
    library.import_paths("b05", [path])
    raise AssertionError("crash point was not reached")


def fault_replays(root: Path) -> dict:
    from app.rag.local_library import LocalLibrary
    from app.schemas.local import SourceKey

    fault_root = root / "fault"
    config = settings(fault_root)
    path = config.course_root / "fault.ipynb"
    old = write_note(path, "oldfaultword")
    library = LocalLibrary(config)
    first = library.import_paths("b05", [path])[0]
    write_note(path, "newfaultword")
    marker = root / "crash-marker.json"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--crash-child",
            "--root",
            str(fault_root),
            "--path",
            str(path),
            "--out",
            str(marker),
        ],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 73
    marker_data = json.loads(marker.read_text())
    assert marker_data["inside_transaction_current_version"] == 2
    assert marker_data["inside_transaction_new_fts"] is True
    independent = LocalLibrary(config)
    assert search(independent, "oldfaultword", first.document_id)[0].version == 1
    assert not search(independent, "newfaultword", first.document_id)
    assert read(independent, first.document_id, 1).original == old
    assert read(independent, first.document_id, 2).status == "not_accessible"
    with sqlite3.connect(config.sqlite_path) as db:
        persisted = {
            "versions": db.execute("SELECT count(*) FROM local_versions").fetchone()[0],
            "sources": db.execute("SELECT count(*) FROM local_sources").fetchone()[0],
            "fts": db.execute("SELECT count(*) FROM local_fts").fetchone()[0],
        }
    assert persisted == {"versions": 1, "sources": 1, "fts": 1}
    retried = independent.import_paths("b05", [path])[0]
    assert retried.status == "published" and retried.version == 2
    assert search(independent, "newfaultword", first.document_id)[0].version == 2
    assert not search(independent, "oldfaultword", first.document_id)
    assert read(independent, first.document_id, 1).original == old

    stale_root = root / "stale"
    stale_config = settings(stale_root)
    stale_path = stale_config.course_root / "stale.ipynb"
    current_bytes = write_note(stale_path, "stalebaseword")
    stale = LocalLibrary(stale_config)
    published = stale.import_paths("b05", [stale_path])[0]
    key = SourceKey(
        document_id=published.document_id, version=published.version, source_id="cell:0"
    )
    ticket = {
        "course_ids": ["b05"],
        "content_sha256": published.content_sha256,
        "publication_epoch": published.publication_epoch,
    }
    assert stale.store.write_enhancement(key, **ticket, text="acceptedextra")
    assert search(stale, "acceptedextra", published.document_id)[0].origin == "enhancement"
    assert stale.delete(published.document_id)
    after_delete_history = read(stale, published.document_id, published.version).status
    assert after_delete_history == "not_accessible"
    assert not search(stale, "stalebaseword acceptedextra", published.document_id)
    rejected_after_delete = stale.store.write_enhancement(key, **ticket, text="latedeleteword")
    assert rejected_after_delete is False
    reimported = stale.import_paths("b05", [stale_path])[0]
    assert reimported.version == published.version
    assert reimported.publication_epoch > published.publication_epoch
    rejected_after_reimport = stale.store.write_enhancement(key, **ticket, text="latereimportword")
    assert rejected_after_reimport is False
    assert not search(stale, "latedeleteword latereimportword", published.document_id)
    assert search(stale, "stalebaseword", published.document_id)[0].version == published.version
    assert read(stale, published.document_id, published.version).original == current_bytes
    return {
        "publish_interruption": {
            "child_exit": result.returncode,
            "inside_transaction": marker_data,
            "persisted_after_crash": persisted,
            "old_search_after_crash": True,
            "new_search_after_crash": False,
            "v1_history_after_crash": "available",
            "v2_history_after_crash": "not_accessible",
            "retry": {"status": retried.status, "version": retried.version},
        },
        "stale_enhancement": {
            "positive_control": True,
            "history_after_delete": after_delete_history,
            "rejected_after_delete": not rejected_after_delete,
            "reimport_version": reimported.version,
            "reimport_epoch_increased": True,
            "rejected_after_reimport": not rejected_after_reimport,
            "current_native_search": True,
            "history_after_reimport": "available",
        },
    }


def legacy_replays(root: Path) -> dict:
    from app.rag.chunker import NotebookChunker
    from app.rag.local_library import LocalLibrary
    from app.rag.parsers.notebook import NotebookParser
    from app.rag.parsers.pptx import PPTXParser
    from app.rag.slide_chunker import SlideChunker
    from app.schemas.indexing import PreparedDocumentIndex
    from app.schemas.local import SourceKey
    from app.stores.sqlite import MetadataStore

    sqlite_root = root / "legacy-sqlite"
    config = settings(sqlite_root)
    source = config.course_root / "legacy.ipynb"
    archive = write_note(source, "legacysqlword")
    old_store = MetadataStore(config)
    old_store.initialize()
    parsed = NotebookParser(config.data_dir / "assets").parse(source)
    chunks = NotebookChunker().chunk(parsed)
    prepared = PreparedDocumentIndex(
        document_id=parsed.document_id,
        source_path=parsed.source_path,
        title=parsed.title,
        groups=chunks.groups,
        sections=chunks.sections,
        chunks=chunks.chunks,
        text_representations=[],
        text_embedding_model="unused",
        text_embedding_dimension=1,
        visual_embedding_model="unused",
        visual_embedding_dimension=1,
    )
    old_version = old_store.reserve_version(
        parsed,
        course_id="b05",
        content_hash=digest(archive),
        release_at=None,
    ).version
    old_store.save_structure(parsed, prepared, version=old_version)
    old_store.activate_version(parsed.document_id, old_version)
    library = LocalLibrary(config)
    migrated = library.migrate_legacy(parsed.document_id)
    assert search(library, "legacysqlword", migrated.document_id)[0].version == migrated.version
    assert read(library, migrated.document_id, migrated.version).original == archive
    write_note(source, "newlocalsqlword")
    newer = library.import_paths("b05", [source])[0]
    assert not search(library, "legacysqlword", migrated.document_id)
    assert search(library, "newlocalsqlword", migrated.document_id)[0].version == newer.version
    assert not old_store.visible_versions()
    source.write_bytes(archive)
    cutback = library.import_paths("b05", [source])[0]
    assert cutback.version == migrated.version
    assert old_store.visible_versions()[0].version == old_version

    bundle_root = root / "legacy-bundle"
    bundle_config = settings(bundle_root)
    slide = bundle_config.course_root / "Signal-slides.pptx"
    shutil.copyfile(SAMPLES / slide.name, slide)
    slide_bytes = slide.read_bytes()
    deck = PPTXParser(bundle_root / "old-prep", render_slides=False).parse(slide)
    slide_chunks = SlideChunker().chunk(deck)
    bundle = bundle_root / "bundle"
    bundle.mkdir()
    (bundle / "deck.json").write_text(deck.model_dump_json())
    (bundle / "chunks.json").write_text(slide_chunks.model_dump_json())
    table_key = "element:" + next(
        element.element_id for element in deck.slides[0].elements if element.kind == "table"
    )
    slide.unlink()
    bundle_library = LocalLibrary(bundle_config)
    backfilled = bundle_library.import_legacy_bundle(
        course_id="b05",
        source_path=slide,
        bundle_dir=bundle,
        content_sha256=digest(slide_bytes),
        version=7,
    )
    fallback_hit = search(bundle_library, "Peak", backfilled.document_id)[0]
    assert fallback_hit.origin == "legacy_text" and fallback_hit.source_id == "page:1"
    missing = bundle_library.store.read_source(
        SourceKey(
            document_id=backfilled.document_id,
            version=backfilled.version,
            source_id="page:1",
        ),
        course_ids=["b05"],
    )
    assert missing.status == "original_unavailable"
    shutil.copyfile(SAMPLES / slide.name, slide)
    hydrated = bundle_library.import_paths("b05", [slide])[0]
    assert hydrated.version == backfilled.version
    remigrated = bundle_library.import_legacy_bundle(
        course_id="b05",
        source_path=slide,
        bundle_dir=bundle,
        content_sha256=digest(slide_bytes),
        version=7,
    )
    exact = bundle_library.store.legacy_sources(deck.document_id, 7, table_key, course_ids=["b05"])
    assert [key.source_id for key in exact] == ["page:1:shape:3"]
    exact_history = bundle_library.store.read_source(exact[0], course_ids=["b05"])
    assert exact_history.status == "available" and exact_history.original == slide_bytes
    return {
        "legacy_sqlite": {
            "document_id": migrated.document_id,
            "migrated_version": migrated.version,
            "new_version": newer.version,
            "local_new_search": True,
            "legacy_hidden_while_new_current": True,
            "cutback_version": cutback.version,
            "legacy_visible_after_cutback": True,
            "historical_original": "available",
        },
        "legacy_bundle": {
            "document_id": backfilled.document_id,
            "missing_original_search_origin": fallback_hit.origin,
            "missing_original_source": fallback_hit.source_id,
            "missing_original_history": missing.status,
            "hydrated_version": hydrated.version,
            "remigrate_status": remigrated.status,
            "refined_source": exact[0].source_id,
            "refined_history": exact_history.status,
        },
    }


def verify_manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    mismatches = []
    for relative, expected in manifest["files"].items():
        actual = digest((ROOT / relative).read_bytes())
        if actual != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": actual})
    actual_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "manifest_sha256": digest(MANIFEST.read_bytes()),
        "listed_files": len(manifest["files"]),
        "mismatches": mismatches,
        "manifest_head": manifest["head"],
        "actual_head": actual_head,
    }


def environment() -> dict:
    packages = {}
    for name in ("fastapi", "httpx", "pydantic", "pypdfium2", "python-pptx", "nbformat"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "sqlite": sqlite3.sqlite_version,
        "packages": packages,
        "timing_clock": "time.perf_counter_ns monotonic",
        "cold_definition": (
            "fresh interpreter and empty per-format SQLite; OS page cache not flushed"
        ),
        "warm_definition": "same process and SQLite, same bytes at a distinct path/document",
        "timing_start": "immediately before POST /api/rag/local/import",
        "timing_stop": "after POST /api/rag/local/search returns the expected exact version/source",
    }


def main(output: Path) -> None:
    started = datetime.now(UTC)
    with tempfile.TemporaryDirectory(prefix="srag-b05-") as temporary:
        root = Path(temporary)
        benchmarks = []
        for kind in FORMATS:
            child_root = root / kind
            child_root.mkdir()
            child_output = root / f"{kind}.json"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--bench-child",
                    kind,
                    "--root",
                    str(child_root),
                    "--out",
                    str(child_output),
                ],
                cwd=ROOT,
                check=True,
                env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"},
            )
            benchmarks.append(json.loads(child_output.read_text()))
        faults = fault_replays(root)
        legacy = legacy_replays(root)
    result = {
        "verdict": "PASS",
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "environment": environment(),
        "source_version": verify_manifest(),
        "benchmarks": benchmarks,
        "fault_replays": faults,
        "legacy_replays": legacy,
    }
    assert result["source_version"]["manifest_sha256"] == (
        "985033c43c08dca5761df7e34d623c531b708a55714f9c1bcb9ca963543e2d5d"
    )
    assert not result["source_version"]["mismatches"]
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"B05 PASS: evidence written to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-child", choices=FORMATS)
    parser.add_argument("--crash-child", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--path", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "B" / "B05-实测.json")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.bench_child:
        asyncio.run(benchmark_child(arguments.bench_child, arguments.root, arguments.out))
    elif arguments.crash_child:
        crash_child(arguments.root, arguments.path, arguments.out)
    else:
        main(arguments.out)
