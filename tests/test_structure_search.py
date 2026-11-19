"""C02 scope, direct-location, and bilingual lexical search acceptance tests."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api import structure_search as structure_api
from app.config import Settings
from app.rag.local_library import LocalLibrary
from app.rag.structure_search import StructureSearcher, StructureSearchRequest
from app.schemas.local import ParsedSource, SourceDraft
from app.stores.structure_snapshot import SnapshotRequest, StructureSnapshotStore

PUBLIC_SAMPLES = Path(__file__).resolve().parents[1] / "docs" / "A" / "样例"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    course_root = tmp_path / "courses"
    course_root.mkdir()
    return Settings(
        _env_file=None,
        sqlite_path=tmp_path / "library.sqlite3",
        data_dir=tmp_path / "data",
        course_root=course_root,
        dashscope_api_key="",
        deepseek_api_key="",
        qdrant_url="http://127.0.0.1:1",
    )


@pytest.fixture
def library(settings: Settings) -> LocalLibrary:
    return LocalLibrary(settings)


def _publish(
    library: LocalLibrary,
    *,
    course_id: str,
    relative_path: str,
    sources: list[SourceDraft],
    raw: bytes | None = None,
    document_id: str | None = None,
):
    path = library.course_root / course_id / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = raw or f"{course_id}:{relative_path}:{len(sources)}".encode()
    plan = library.store.prepare(course_id, path, document_id=document_id)
    return library.store.publish(
        plan,
        ParsedSource(title=Path(relative_path).stem, sources=sources),
        original=payload,
        content_sha256=hashlib.sha256(payload).hexdigest(),
        document_format=Path(relative_path).suffix.lstrip(".") or "ipynb",
    )


def _cell(index: int, text: str, *, cell_id: str | None = None) -> SourceDraft:
    locator = {"format": "ipynb", "cell_index": index}
    if cell_id is not None:
        locator["cell_id"] = cell_id
    return SourceDraft(
        source_id=f"cell:{index}",
        kind="cell",
        ordinal=index,
        title=f"cell {index}",
        text=text,
        locator=locator,
    )


def _page(page: int, text: str, *, ordinal: int = 0) -> SourceDraft:
    return SourceDraft(
        source_id=f"page:{page}",
        kind="page",
        ordinal=ordinal,
        title=f"page {page}",
        text=text,
        locator={"format": "pdf", "page": page},
    )


def _searcher(settings: Settings) -> StructureSearcher:
    return StructureSearcher(StructureSnapshotStore(settings))


class _RecordingStore(StructureSnapshotStore):
    last_request: SnapshotRequest | None = None

    def read(self, request: SnapshotRequest):
        self.last_request = request
        return super().read(request)


def _request(query: str, courses: list[str], **scope) -> StructureSearchRequest:
    return StructureSearchRequest(
        query=query,
        scope={"course_ids": courses, **scope},
        include_adjacency=False,
    )


def _app(searcher: StructureSearcher) -> FastAPI:
    app = FastAPI()
    app.include_router(structure_api.router, prefix="/api/rag/structure")
    app.dependency_overrides[structure_api.get_structure_searcher] = lambda: searcher
    return app


def test_http_allowlist_defaults_closed_and_never_leaks_absolute_paths(
    library: LocalLibrary, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _publish(
        library,
        course_id="trusted",
        relative_path="private/folder/lesson.ipynb",
        sources=[_cell(0, "ridge regression")],
    )
    app = _app(_searcher(settings))
    monkeypatch.delenv("SRAG_STRUCTURE_COURSES", raising=False)
    with TestClient(app) as client:
        denied = client.post(
            "/api/rag/structure/search",
            json={"query": "ridge", "scope": {"course_ids": ["trusted"]}},
        )
        assert denied.status_code == 403

        app.dependency_overrides[structure_api.get_structure_course_allowlist] = lambda: frozenset(
            {"trusted"}
        )
        response = client.post(
            "/api/rag/structure/search",
            json={
                "query": "ridge",
                "scope": {"course_ids": ["trusted"], "document_ids": [result.document_id]},
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["snapshot"][0]["file_name"] == "lesson.ipynb"
    serialized = json.dumps(payload, ensure_ascii=False)
    assert str(library.course_root) not in serialized
    assert "private/folder" not in serialized


def test_bilingual_glossary_crosses_one_language_sources_only(
    library: LocalLibrary, settings: Settings
) -> None:
    english = _publish(
        library,
        course_id="demo",
        relative_path="english.ipynb",
        sources=[_cell(0, "ridge regression controls coefficient size")],
    )
    chinese = _publish(
        library,
        course_id="demo",
        relative_path="chinese.ipynb",
        sources=[_cell(0, "岭回归")],
    )
    searcher = _searcher(settings)
    zh_to_en = searcher.search(
        _request("岭回归", ["demo"], document_ids=[english.document_id]),
        allowed_course_ids=frozenset({"demo"}),
    )
    en_to_zh = searcher.search(
        _request("ridge regression", ["demo"], document_ids=[chinese.document_id]),
        allowed_course_ids=frozenset({"demo"}),
    )
    assert [item.source.document_id for item in zh_to_en.candidates] == [english.document_id]
    assert [item.source.document_id for item in en_to_zh.candidates] == [chinese.document_id]
    assert {variant.reason for variant in zh_to_en.query_trace.variants} == {"raw", "glossary"}
    assert zh_to_en.query_trace.glossary_version == "c02-public-v1"


def test_abbreviation_and_dotted_code_require_original_text_verification(
    library: LocalLibrary, settings: Settings
) -> None:
    result = _publish(
        library,
        course_id="demo",
        relative_path="code.ipynb",
        sources=[
            _cell(0, "fast Fourier transform uses np.linalg.solve(x, y)"),
            _cell(1, "np and linalg and solve are separate tokens"),
        ],
    )
    searcher = _searcher(settings)
    fft = searcher.search(
        _request("FFT", ["demo"], document_ids=[result.document_id]),
        allowed_course_ids=frozenset({"demo"}),
    )
    code = searcher.search(
        _request("np.linalg.solve", ["demo"], document_ids=[result.document_id]),
        allowed_course_ids=frozenset({"demo"}),
    )
    assert fft.status == "ok"
    assert "fast fourier transform" in {v.text.casefold() for v in fft.query_trace.variants}
    kinds = {candidate.source.source_id: candidate.match_kind for candidate in code.candidates}
    assert kinds["cell:0"] == "exact_literal"
    assert kinds["cell:1"] == "token_component"


def test_same_name_is_ambiguous_across_courses_and_options_are_safe(
    library: LocalLibrary, settings: Settings
) -> None:
    first = _publish(
        library,
        course_id="course-a",
        relative_path="unit/common.ipynb",
        sources=[_cell(0, "sharedword alpha")],
    )
    second = _publish(
        library,
        course_id="course-b",
        relative_path="other/common.ipynb",
        sources=[_cell(0, "sharedword beta")],
    )
    response = _searcher(settings).search(
        _request(
            "sharedword",
            ["course-a", "course-b"],
            file_names=["/workspace/private/common.ipynb"],
        ),
        allowed_course_ids=frozenset({"course-a", "course-b"}),
    )
    assert response.status == "ambiguous_scope"
    assert {option.document_id for option in response.options} == {
        first.document_id,
        second.document_id,
    }
    assert {option.file_name for option in response.options} == {"common.ipynb"}
    assert response.scope_trace.requested_file_names == ["common.ipynb"]
    assert response.candidates == []
    assert response.coverage.enumerated_source_count == 0
    assert response.coverage.fts_seed_count == 0
    assert all(not hasattr(option, "source_path") for option in response.options)


def test_direct_page_and_cell_locations_are_typed_and_budgeted(
    library: LocalLibrary, settings: Settings
) -> None:
    page_result = _publish(
        library,
        course_id="demo",
        relative_path="slides.pdf",
        sources=[
            _page(1, "page text", ordinal=0),
            SourceDraft(
                source_id="page:1:table:1",
                kind="page_object",
                ordinal=1,
                parent_id="page:1",
                title="table",
                text="value",
                locator={"format": "pdf", "page": 1, "object_type": "table"},
            ),
            _page(2, "second page", ordinal=2),
        ],
    )
    cell_result = _publish(
        library,
        course_id="demo",
        relative_path="lab.ipynb",
        sources=[
            _cell(0, "intro", cell_id="intro"),
            _cell(1, "plot", cell_id="plot-cell"),
            SourceDraft(
                source_id="cell:1:output:0",
                kind="notebook_output",
                ordinal=2,
                parent_id="cell:1",
                text="saved output",
                locator={"format": "ipynb", "cell_index": 1, "cell_id": "plot-cell"},
            ),
        ],
    )
    recording_store = _RecordingStore(settings)
    searcher = StructureSearcher(recording_store)
    page = searcher.search(
        StructureSearchRequest(
            query="page text",
            scope={"course_ids": ["demo"], "document_ids": [page_result.document_id]},
            location={"page": 1},
            top_k=1,
        ),
        allowed_course_ids=frozenset({"demo"}),
    )
    cell = searcher.search(
        StructureSearchRequest(
            query="",
            scope={"course_ids": ["demo"], "document_ids": [cell_result.document_id]},
            location={"cell_id": "plot-cell"},
        ),
        allowed_course_ids=frozenset({"demo"}),
    )
    assert [candidate.source.source_id for candidate in page.candidates] == ["page:1"]
    assert page.coverage.enumerated_source_count == 3
    assert page.coverage.evidence_inspected_count == 1
    assert page.coverage.returned_count == 1 and not page.coverage.truncated
    assert page.coverage.full_file_read is False
    assert page.candidates[0].location_text_matched is True
    assert [candidate.source.source_id for candidate in cell.candidates] == ["cell:1"]
    assert all(candidate.reasons[0].code == "direct_location" for candidate in cell.candidates)
    assert recording_store.last_request is not None
    assert recording_store.last_request.fts_match is None
    assert recording_store.last_request.fts_limit == 0


def test_invalid_location_and_missing_location_are_explicit(
    library: LocalLibrary, settings: Settings
) -> None:
    result = _publish(
        library,
        course_id="demo",
        relative_path="one.pdf",
        sources=[_page(1, "only")],
    )
    with pytest.raises(ValidationError):
        StructureSearchRequest(
            query="", scope={"course_ids": ["demo"]}, location={"page_range": [2, 1]}
        )
    with pytest.raises(ValidationError):
        StructureSearchRequest(query="", scope={"course_ids": ["demo"]})
    response = _searcher(settings).search(
        StructureSearchRequest(
            query="",
            scope={"course_ids": ["demo"], "document_ids": [result.document_id]},
            location={"page": 9},
        ),
        allowed_course_ids=frozenset({"demo"}),
    )
    assert response.status == "location_not_found"
    assert response.candidates == []
    assert "available_pages=[1, 1]" in response.misses[0]


def test_two_reads_each_keep_one_published_version(
    library: LocalLibrary, settings: Settings
) -> None:
    first = _publish(
        library,
        course_id="demo",
        relative_path="versioned.ipynb",
        sources=[_cell(0, "oldversionword")],
        raw=b"version-one",
    )
    searcher = _searcher(settings)
    old = searcher.search(
        _request("oldversionword", ["demo"], document_ids=[first.document_id]),
        allowed_course_ids=frozenset({"demo"}),
    )
    second = _publish(
        library,
        course_id="demo",
        relative_path="versioned.ipynb",
        sources=[_cell(0, "newversionword")],
        raw=b"version-two",
        document_id=first.document_id,
    )
    new = searcher.search(
        _request("newversionword", ["demo"], document_ids=[first.document_id]),
        allowed_course_ids=frozenset({"demo"}),
    )
    stale = searcher.search(
        _request("oldversionword", ["demo"], document_ids=[first.document_id]),
        allowed_course_ids=frozenset({"demo"}),
    )
    assert first.version == 1 and second.version == 2
    assert {item.version for item in old.snapshot} == {1}
    assert {item.source.version for item in old.candidates} == {1}
    assert {item.version for item in new.snapshot} == {2}
    assert {item.source.version for item in new.candidates} == {2}
    assert stale.status == "empty" and stale.candidates == []


def test_public_three_format_probes_keep_exact_source_keys(
    library: LocalLibrary, settings: Settings
) -> None:
    selected = []
    for name in ("Signal-reference.pdf", "Signal-slides.pptx", "Signal-lab.ipynb"):
        target = library.course_root / name
        shutil.copy2(PUBLIC_SAMPLES / name, target)
        selected.append(target)
    imported = library.import_paths("public", selected)
    by_name = {Path(item.source_path).name: item for item in imported}
    searcher = _searcher(settings)
    probes = (
        ("lookahead", "Signal-reference.pdf", "page:1"),
        ("Peak", "Signal-slides.pptx", "page:1:shape:3"),
        ("NotebookImage", "Signal-lab.ipynb", "cell:1"),
    )
    for query, file_name, source_id in probes:
        imported_document = by_name[file_name]
        response = searcher.search(
            _request(query, ["public"], document_ids=[imported_document.document_id]),
            allowed_course_ids=frozenset({"public"}),
        )
        assert response.status == "ok"
        assert response.candidates[0].source.model_dump() == {
            "document_id": imported_document.document_id,
            "version": 1,
            "source_id": source_id,
        }


@pytest.mark.parametrize(
    ("query", "detected"),
    [("not FFT", True), ("without FFT", True), ("不是 FFT", True), ("notebook", False)],
)
def test_negation_trace_preserves_related_evidence(
    library: LocalLibrary, settings: Settings, query: str, detected: bool
) -> None:
    _publish(
        library,
        course_id="demo",
        relative_path="signals.ipynb",
        sources=[_cell(0, "FFT notebook")],
    )
    result = _searcher(settings).search(
        _request(query, ["demo"]), allowed_course_ids=frozenset({"demo"})
    )
    assert result.query_trace.negation_detected is detected
    assert result.candidates[0].source.source_id == "cell:0"
