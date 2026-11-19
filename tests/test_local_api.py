from __future__ import annotations

import json
import shutil

import httpx

import app.api.rag as rag_api
from app.api.local_library import get_local_library
from app.api.rag import _local_hit_to_legacy
from app.config import Settings
from app.main import create_app
from app.rag.local_library import LocalLibrary
from app.schemas.local import ImportResult, LocalHit
from app.schemas.rag import SearchResponse
from app.stores.local_library import PublicationConflict


class FakeLocalLibrary:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.deleted = True

    def import_paths(self, course_id, paths, *, document_id=None, release_at=None):
        self.calls.append(("import", (course_id, paths, document_id, release_at)))
        return [
            ImportResult(
                status="published",
                source_path=str(paths[0]),
                document_id=document_id or "local_pdf",
                version=1,
                publication_epoch=1,
                content_sha256="a" * 64,
                source_count=2,
                searchable_count=1,
            ),
            ImportResult(
                status="failed",
                source_path=str(paths[1]),
                error="unsupported",
            ),
        ]

    def search(self, request):
        self.calls.append(("search", request))
        return [
            LocalHit(
                document_id="local_pdf",
                course_id="CS5489",
                version=2,
                source_id="page:3",
                locator={"page": 3},
                publication_epoch=4,
                content_sha256="b" * 64,
                source_path="notes.pdf",
                origin="native_text",
                title="Median filter",
                text="window size is five samples",
                score=0.75,
            )
        ]

    def migrate_legacy(self, document_id):
        self.calls.append(("migrate", document_id))
        return ImportResult(
            status="backfilled",
            source_path="legacy.ipynb",
            document_id=document_id,
            version=1,
            publication_epoch=1,
            content_sha256="c" * 64,
            source_count=3,
            searchable_count=2,
        )

    def import_legacy_bundle(self, **kwargs):
        self.calls.append(("bundle", kwargs))
        return ImportResult(
            status="backfilled",
            source_path=str(kwargs["source_path"]),
            document_id="legacy_slides",
            version=kwargs["version"],
            publication_epoch=1,
            content_sha256=kwargs["content_sha256"],
            source_count=2,
            searchable_count=2,
        )

    def delete(self, document_id):
        self.calls.append(("delete", document_id))
        deleted, self.deleted = self.deleted, False
        return deleted


class FakeLegacyRetriever:
    def __init__(self) -> None:
        self.vector_store = object()
        self.requests = []

    async def search(self, request):
        self.requests.append(request)
        return SearchResponse(query=request.query, hits=[], mode_used="visual")


def test_legacy_citation_filters_locator_metadata_without_using_opaque_source_id() -> None:
    hit = LocalHit(
        document_id="local_notebook",
        course_id="CS5489",
        version=1,
        source_id="cell:0",
        locator={
            "cell_id": 17,
            "cell_index": True,
            "section_path": ["  Root  ", "", 5, " Topic "],
        },
        publication_epoch=1,
        content_sha256="f" * 64,
        source_path="notebook.ipynb",
        origin="native_text",
        title="Opaque source",
        text="content",
        score=1.0,
    )

    source = _local_hit_to_legacy(hit).source

    assert source.cell_id is None
    assert source.cell_ids == []
    assert source.cell_index is None
    assert source.section_path == ["Root", "Topic"]


async def test_local_routes_expose_full_results_and_partial_file_failures() -> None:
    app = create_app()
    library = FakeLocalLibrary()
    app.dependency_overrides[get_local_library] = lambda: library

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        imported = await client.post(
            "/api/rag/local/import",
            json={"course_id": "CS5489", "paths": ["one.pdf", "bad.txt"]},
        )
        searched = await client.post(
            "/api/rag/local/search",
            json={"query": "window", "scope": {"course_ids": ["CS5489"]}},
        )
        migrated = await client.post("/api/rag/local/migrate", json={"document_id": "legacy_doc"})
        bundle = await client.post(
            "/api/rag/local/migrate/bundle",
            json={
                "course_id": "CS5489",
                "source_path": "slides.pptx",
                "bundle_dir": "bundle",
                "content_sha256": "D" * 64,
                "version": 3,
            },
        )
        deleted = await client.delete("/api/rag/local/documents/local_pdf")
        missing = await client.delete("/api/rag/local/documents/local_pdf")

    assert imported.status_code == 200
    assert [item["status"] for item in imported.json()] == ["published", "failed"]
    assert searched.status_code == 200
    hit = searched.json()["hits"][0]
    assert (hit["document_id"], hit["version"], hit["source_id"]) == (
        "local_pdf",
        2,
        "page:3",
    )
    assert hit["publication_epoch"] == 4
    assert migrated.json()["status"] == "backfilled"
    assert bundle.json()["content_sha256"] == "d" * 64
    assert deleted.json() == {"document_id": "local_pdf", "deleted": True}
    assert missing.status_code == 404


async def test_local_routes_validate_single_document_path_and_bundle_hash() -> None:
    app = create_app()
    app.dependency_overrides[get_local_library] = FakeLocalLibrary

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        multiple = await client.post(
            "/api/rag/local/import",
            json={
                "course_id": "CS5489",
                "paths": ["one.pdf", "two.pdf"],
                "document_id": "stable",
            },
        )
        bad_hash = await client.post(
            "/api/rag/local/migrate/bundle",
            json={
                "course_id": "CS5489",
                "source_path": "slides.pptx",
                "bundle_dir": "bundle",
                "content_sha256": "short",
            },
        )

    assert multiple.status_code == 422
    assert bad_hash.status_code == 422


async def test_local_route_maps_missing_path_and_publication_conflict() -> None:
    class MissingLibrary(FakeLocalLibrary):
        def import_paths(self, *args, **kwargs):
            raise FileNotFoundError("missing.pdf")

        def migrate_legacy(self, document_id):
            raise PublicationConflict(f"changed: {document_id}")

    app = create_app()
    app.dependency_overrides[get_local_library] = MissingLibrary
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        missing = await client.post(
            "/api/rag/local/import",
            json={"course_id": "CS5489", "paths": ["missing.pdf"]},
        )
        conflict = await client.post("/api/rag/local/migrate", json={"document_id": "legacy_doc"})

    assert missing.status_code == 404
    assert conflict.status_code == 409


async def test_local_import_returns_4xx_when_every_selected_path_fails() -> None:
    class FailedLibrary(FakeLocalLibrary):
        def import_paths(self, course_id, paths, *, document_id=None, release_at=None):
            return [
                ImportResult(status="failed", source_path=str(path), error="outside COURSE_ROOT")
                for path in paths
            ]

    app = create_app()
    app.dependency_overrides[get_local_library] = FailedLibrary
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/rag/local/import",
            json={"course_id": "CS5489", "paths": ["outside.pdf"]},
        )

    assert response.status_code == 400
    assert response.json()["detail"]["results"][0]["status"] == "failed"


async def test_old_search_uses_local_adapter_offline_and_visual_stays_legacy(
    monkeypatch,
) -> None:
    local = FakeLocalLibrary()
    legacy = FakeLegacyRetriever()
    settings = Settings(rag_search_backend="local")
    monkeypatch.setattr(rag_api, "get_settings", lambda: settings)
    monkeypatch.setattr(rag_api, "LocalLibrary", lambda _: local)
    monkeypatch.setattr(
        rag_api.HybridRetriever,
        "from_settings",
        classmethod(lambda cls, _: legacy),
    )
    app = create_app()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        text = await client.post(
            "/api/rag/search",
            json={"query": "window", "retrieval_mode": "text"},
        )
        visual = await client.post(
            "/api/rag/search",
            json={"query": "show the figure", "retrieval_mode": "visual"},
        )

    assert text.status_code == 200
    text_hit = text.json()["hits"][0]
    assert text_hit["chunk_id"] == "local:local_pdf:2:page:3"
    assert text_hit["source"]["page"] == 3
    assert text_hit["source"]["version"] == 2
    assert text_hit["source"]["section_path"] == ["Median filter"]
    assert text.json()["mode_used"] == "text"
    assert visual.status_code == 200
    assert visual.json()["mode_used"] == "visual"
    assert len(legacy.requests) == 1
    assert legacy.requests[0].retrieval_mode == "visual"


async def test_old_search_defaults_and_cuts_back_to_legacy(monkeypatch) -> None:
    legacy = FakeLegacyRetriever()
    settings = Settings()
    assert settings.rag_search_backend == "legacy"
    monkeypatch.setattr(rag_api, "get_settings", lambda: settings)
    monkeypatch.setattr(
        rag_api,
        "LocalLibrary",
        lambda _: (_ for _ in ()).throw(AssertionError("local library constructed")),
    )
    monkeypatch.setattr(
        rag_api.HybridRetriever,
        "from_settings",
        classmethod(lambda cls, _: legacy),
    )
    app = create_app()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/rag/search", json={"query": "legacy"})

    assert response.status_code == 200
    assert len(legacy.requests) == 1
    assert legacy.requests[0].query == "legacy"


async def test_real_local_pdf_import_and_both_search_routes_are_offline(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "reference.pdf"
    shutil.copyfile("docs/A/样例/Signal-reference.pdf", source)
    settings = Settings(
        course_root=tmp_path,
        data_dir=tmp_path / "data",
        sqlite_path=tmp_path / "local.sqlite3",
        rag_search_backend="local",
    )
    library = LocalLibrary(settings)
    app = create_app()
    app.dependency_overrides[get_local_library] = lambda: library
    monkeypatch.setattr(rag_api, "get_settings", lambda: settings)
    monkeypatch.setattr(
        rag_api.HybridRetriever,
        "from_settings",
        classmethod(
            lambda cls, _: (_ for _ in ()).throw(
                AssertionError("legacy clients constructed for local text search")
            )
        ),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        imported = await client.post(
            "/api/rag/local/import",
            json={"course_id": "signals", "paths": ["reference.pdf"]},
        )
        local_search = await client.post("/api/rag/local/search", json={"query": "future samples"})
        old_search = await client.post(
            "/api/rag/search",
            json={"query": "future samples", "retrieval_mode": "text"},
        )

    assert imported.status_code == 200
    assert imported.json()[0]["status"] == "published"
    assert imported.json()[0]["searchable_count"] == 2
    assert local_search.status_code == 200
    assert local_search.json()["hits"][0]["source_id"] == "page:1"
    assert old_search.status_code == 200
    assert old_search.json()["hits"][0]["source"]["page"] == 1
    assert old_search.json()["hits"][0]["source"]["version"] == 1


async def test_real_notebook_legacy_citations_use_locator_identity_and_sections(
    tmp_path, monkeypatch
) -> None:
    notebook = tmp_path / "cells.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 4,
                "metadata": {},
                "cells": [
                    {
                        "id": "intro-original",
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": ["# Root\n", "## Section One\n", "alphasentinel"],
                    },
                    {
                        "cell_type": "code",
                        "metadata": {},
                        "execution_count": None,
                        "source": ["betasentinel = 1"],
                        "outputs": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        course_root=tmp_path,
        data_dir=tmp_path / "data",
        sqlite_path=tmp_path / "local.sqlite3",
        rag_search_backend="local",
    )
    library = LocalLibrary(settings)
    app = create_app()
    app.dependency_overrides[get_local_library] = lambda: library
    monkeypatch.setattr(rag_api, "get_settings", lambda: settings)
    monkeypatch.setattr(
        rag_api.HybridRetriever,
        "from_settings",
        classmethod(
            lambda cls, _: (_ for _ in ()).throw(
                AssertionError("legacy clients constructed for local text search")
            )
        ),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        imported = await client.post(
            "/api/rag/local/import",
            json={"course_id": "notebooks", "paths": ["cells.ipynb"]},
        )
        with_id = await client.post("/api/rag/search", json={"query": "alphasentinel"})
        without_id = await client.post("/api/rag/search", json={"query": "betasentinel"})

    assert imported.status_code == 200
    assert imported.json()[0]["status"] == "published"
    identified_source = with_id.json()["hits"][0]["source"]
    assert identified_source["cell_id"] == "intro-original"
    assert identified_source["cell_ids"] == ["intro-original"]
    assert identified_source["cell_index"] == 0
    assert identified_source["section_path"] == ["Root", "Section One"]
    index_only_source = without_id.json()["hits"][0]["source"]
    assert index_only_source["cell_id"] is None
    assert index_only_source["cell_ids"] == []
    assert index_only_source["cell_index"] == 1
    assert index_only_source["section_path"] == ["Root", "Section One"]
