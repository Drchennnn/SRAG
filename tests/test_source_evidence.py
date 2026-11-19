from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock

import httpx
import pytest
from PIL import Image

import app.rag.source_evidence as source_evidence_module
from app.api.source_evidence import get_source_evidence_service
from app.config import Settings
from app.main import create_app
from app.rag.local_library import LocalLibrary
from app.rag.source_evidence import (
    EvidenceRequest,
    HistoricalSourceGrant,
    RegionComponent,
    RegionRegistry,
    RegisteredRegion,
    SourceEvidenceService,
    _pdf_rect_to_visible,
)
from app.schemas.local import ParsedSource, SourceDraft, SourceKey
from app.stores.local_library import HistoricalSource

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "docs" / "A" / "样例"


@pytest.fixture
def library(tmp_path: Path) -> LocalLibrary:
    course = tmp_path / "course"
    course.mkdir()
    return LocalLibrary(
        Settings(
            _env_file=None,
            sqlite_path=tmp_path / "library.sqlite3",
            data_dir=tmp_path / "data",
            course_root=course,
            dashscope_api_key="",
            deepseek_api_key="",
            qdrant_url="http://127.0.0.1:1",
        )
    )


def _copy_import(library: LocalLibrary, name: str, *, document_id: str | None = None):
    target = library.course_root / name
    source = SAMPLES / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return library.import_paths("demo", [target], document_id=document_id)[0]


def _service(
    library: LocalLibrary,
    tmp_path: Path,
    *,
    regions: RegionRegistry | None = None,
    courses: tuple[str, ...] = ("demo",),
    max_inline_bytes: int = 8 * 1024 * 1024,
    soffice: Path | None = None,
) -> SourceEvidenceService:
    return SourceEvidenceService(
        library.store,
        cache_dir=tmp_path / "renditions",
        current_course_ids=courses,
        regions=regions,
        max_inline_bytes=max_inline_bytes,
        soffice_path=soffice,
    )


def _request(result, source_id: str, rendition: str, **kwargs) -> EvidenceRequest:
    return EvidenceRequest(
        source=SourceKey(
            document_id=result.document_id,
            version=result.version,
            source_id=source_id,
        ),
        expected_content_sha256=result.content_sha256,
        expected_publication_epoch=result.publication_epoch,
        rendition=rendition,
        **kwargs,
    )


def _image(display) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(display.data_base64))).convert("RGB")


def test_pdf_page_is_rendered_from_archived_bytes_and_cache_is_verified(
    library: LocalLibrary, tmp_path: Path
) -> None:
    imported = _copy_import(library, "Signal-reference.pdf")
    request = _request(imported, "page:1", "original_page")
    service = _service(library, tmp_path)

    first = service.render_current(request)
    second = service.render_current(request)

    assert first.status == second.status == "available"
    assert first.display.display_source == request.source
    assert first.display.actual_source == request.source
    assert first.display.media_type == "image/png"
    assert first.display.cache_hit is False and second.display.cache_hit is True
    assert first.display.byte_sha256 == second.display.byte_sha256
    assert _image(first.display).size == (first.display.width, first.display.height)


def test_concurrent_http_requests_share_one_process_singleflight(
    library: LocalLibrary, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported = _copy_import(library, "Signal-reference.pdf")
    request = _request(imported, "page:1", "original_page")
    original_render = source_evidence_module._render_pdf_page
    render_entered = Event()
    release_render = Event()
    second_dependency_created = Event()
    guard = Lock()
    render_calls = 0
    dependency_calls = 0

    def counted_render(*args, **kwargs):
        nonlocal render_calls
        with guard:
            render_calls += 1
        render_entered.set()
        assert release_render.wait(3)
        return original_render(*args, **kwargs)

    def new_service_per_request():
        nonlocal dependency_calls
        with guard:
            dependency_calls += 1
            if dependency_calls == 2:
                second_dependency_created.set()
        return _service(library, tmp_path)

    monkeypatch.setattr(source_evidence_module, "_render_pdf_page", counted_render)
    app = create_app()
    app.dependency_overrides[get_source_evidence_service] = new_service_per_request

    async def concurrent_posts():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            first = asyncio.create_task(
                client.post("/api/rag/local/evidence/current", json=request.model_dump(mode="json"))
            )
            assert await asyncio.to_thread(render_entered.wait, 3)
            second = asyncio.create_task(
                client.post("/api/rag/local/evidence/current", json=request.model_dump(mode="json"))
            )
            assert await asyncio.to_thread(second_dependency_created.wait, 3)
            release_render.set()
            concurrent = await asyncio.gather(first, second)
            subsequent = await client.post(
                "/api/rag/local/evidence/current", json=request.model_dump(mode="json")
            )
            return [*concurrent, subsequent]

    import asyncio

    responses = asyncio.run(concurrent_posts())
    bodies = [response.json() for response in responses]

    assert [response.status_code for response in responses] == [200, 200, 200]
    assert render_calls == 1
    assert {body["display"]["byte_sha256"] for body in bodies} == {
        "749c18cf9310c973a0fdb2b4ef933babaac4a68e4ebc0df92673122bfd73ecb6"
    }
    assert sorted(body["display"]["cache_hit"] for body in bodies) == [False, True, True]


def test_different_rendition_keys_render_in_parallel(
    library: LocalLibrary, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported = _copy_import(library, "Signal-reference.pdf")
    service = _service(library, tmp_path)
    requests = [
        _request(imported, "page:1", "original_page"),
        _request(imported, "page:2", "original_page"),
    ]
    original_render = source_evidence_module._render_pdf_page
    both_rendering = Barrier(2, timeout=3)
    guard = Lock()
    active = 0
    max_active = 0

    def concurrent_render(*args, **kwargs):
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        both_rendering.wait()
        try:
            return original_render(*args, **kwargs)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(source_evidence_module, "_render_pdf_page", concurrent_render)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(service.render_current, requests))

    assert [result.status for result in results] == ["available", "available"]
    assert max_active == 2


def test_singleflight_releases_after_owner_failure(
    library: LocalLibrary, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported = _copy_import(library, "Signal-reference.pdf")
    request = _request(imported, "page:1", "original_page")
    service = _service(library, tmp_path)
    original_render = source_evidence_module._render_pdf_page
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("deterministic render failure")
        return original_render(*args, **kwargs)

    monkeypatch.setattr(source_evidence_module, "_render_pdf_page", fail_once)

    failed = service.render_current(request)
    recovered = service.render_current(request)
    cached = service.render_current(request)

    assert failed.status == "failed"
    assert recovered.status == cached.status == "available"
    assert calls == 2
    assert recovered.display.cache_hit is False and cached.display.cache_hit is True


def test_rotated_native_region_includes_target_caption_and_invalidates_changed_rect(
    library: LocalLibrary, tmp_path: Path
) -> None:
    original = _rotated_pdf()
    path = library.course_root / "rotated-paper.pdf"
    path.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    plan = library.store.prepare("demo", path)
    imported = library.store.publish(
        plan,
        ParsedSource(
            title="Rotated paper",
            sources=[
                SourceDraft(
                    source_id="page:1",
                    kind="page",
                    ordinal=0,
                    locator={
                        "format": "pdf",
                        "page": 1,
                        "rotation_degrees_clockwise": 90,
                        "visible_page_bbox_pdf_units": [10, 10, 190, 90],
                    },
                )
            ],
        ),
        original=original,
        content_sha256=digest,
        document_format="pdf",
    )
    key = SourceKey(document_id=imported.document_id, version=1, source_id="page:1")
    first_region = RegisteredRegion(
        region_id="paper-figure",
        source=key,
        content_sha256=digest,
        coordinate_frame="pdf_page_units_bottom_left_v1",
        target_rect=(50, 30, 100, 60),
        components=[
            RegionComponent(
                role="caption",
                rect=(50, 15, 100, 25),
                coordinate_frame="pdf_page_units_bottom_left_v1",
            )
        ],
        provenance="trusted_corpus_registration",
        revision=1,
    )
    request = _request(
        imported,
        "page:1",
        "object_crop",
        registered_region_id="paper-figure",
    )
    service = _service(library, tmp_path, regions=RegionRegistry([first_region]))

    first = service.render_current(request)
    cached = service.render_current(request)
    image = _image(first.display)
    colors = list(image.get_flattened_data())

    assert first.status == "available" and first.display.rendition_kind == "object_crop"
    assert first.display.display_source == key
    assert first.display.rect_visible_norm == pytest.approx((0.0425, 0.2022, 0.645, 0.52), abs=0.01)
    assert any(red > 180 and blue < 80 for red, _, blue in colors)
    assert any(blue > 180 and red < 80 for red, _, blue in colors)
    assert cached.display.cache_hit is True

    changed = first_region.model_copy(
        update={"revision": 2, "target_rect": (45.0, 30.0, 100.0, 60.0)}
    )
    changed_service = _service(library, tmp_path, regions=RegionRegistry([changed]))
    rerendered = changed_service.render_current(request)
    assert rerendered.status == "available"
    assert rerendered.display.cache_hit is False
    assert rerendered.display.rect_visible_norm != first.display.rect_visible_norm


def test_unreliable_object_bbox_falls_back_to_actual_parent_page(
    library: LocalLibrary, tmp_path: Path
) -> None:
    original = _rotated_pdf(rotation=0)
    path = library.course_root / "unreliable.pdf"
    path.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    imported = library.store.publish(
        library.store.prepare("demo", path),
        ParsedSource(
            title="Unreliable object",
            sources=[
                SourceDraft(
                    source_id="page:1",
                    kind="page",
                    ordinal=0,
                    locator={
                        "format": "pdf",
                        "page": 1,
                        "rotation_degrees_clockwise": 0,
                        "visible_page_bbox_pdf_units": [10, 10, 190, 90],
                    },
                ),
                SourceDraft(
                    source_id="page:1:group:1",
                    kind="page_object",
                    ordinal=1,
                    parent_id="page:1",
                    title="Nested group",
                    text="native fallback text",
                    locator={"format": "pdf", "page": 1, "position_basis": "unknown"},
                ),
                SourceDraft(
                    source_id="page:1:group:1:object:1",
                    kind="page_object",
                    ordinal=2,
                    parent_id="page:1:group:1",
                    title="Nested target",
                    text="target native text",
                    locator={"format": "pdf", "page": 1, "position_basis": "unknown"},
                ),
            ],
        ),
        original=original,
        content_sha256=digest,
        document_format="pdf",
    )
    request = _request(imported, "page:1:group:1:object:1", "object_crop")

    result = _service(library, tmp_path).render_current(request)

    parent = request.source.model_copy(update={"source_id": "page:1"})
    assert result.status == "available"
    assert result.requested_source == request.source
    assert result.display.display_source == parent
    assert result.display.actual_source == parent
    assert result.display.rendition_kind == "parent_page_fallback"
    assert result.display.fallback_reason == "unreliable_bbox"


def test_notebook_reads_exact_output_mime_without_executing_or_substituting(
    library: LocalLibrary, tmp_path: Path
) -> None:
    imported = _copy_import(library, "Signal-lab.ipynb")
    service = _service(library, tmp_path)
    request = _request(
        imported,
        "cell:1:output:2",
        "notebook_output",
        preferred_mime="image/png",
    )

    first = service.render_current(request)
    cached = service.render_current(request)
    missing = service.render_current(request.model_copy(update={"preferred_mime": "image/jpeg"}))
    text_output = service.render_current(_request(imported, "cell:1:output:0", "notebook_output"))

    assert first.status == "available" and first.display.display_source == request.source
    assert (
        first.display.byte_sha256
        == "ce741007281a939680e7a21020fc80c34abfcb10d6d76d13e7984bc9d5743aa5"
    )
    assert first.display.renderer == "notebook-original"
    assert cached.display.cache_hit is True
    assert missing.status == "image_unavailable"
    assert missing.warnings == ["preferred_mime_unavailable"]
    assert text_output.status == "image_unavailable"


def test_notebook_svg_is_never_inlined_or_substituted(
    tmp_path: Path, library: LocalLibrary
) -> None:
    png_stream = io.BytesIO()
    Image.new("RGB", (8, 6), "green").save(png_stream, format="PNG")
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "id": "unsafe-svg",
                "cell_type": "code",
                "metadata": {},
                "execution_count": 1,
                "source": "display(saved_output)",
                "outputs": [
                    {
                        "output_type": "display_data",
                        "metadata": {},
                        "data": {
                            "image/png": base64.b64encode(png_stream.getvalue()).decode("ascii"),
                            "image/svg+xml": "<svg><script>alert(1)</script></svg>",
                        },
                    }
                ],
            }
        ],
    }
    path = library.course_root / "saved-output.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    imported = library.import_paths("demo", [path])[0]
    service = _service(library, tmp_path)

    default = service.render_current(_request(imported, "cell:0:output:0", "notebook_output"))
    explicit_svg = service.render_current(
        _request(
            imported,
            "cell:0:output:0",
            "notebook_output",
            preferred_mime="image/svg+xml",
        )
    )

    assert default.status == "available" and default.display.media_type == "image/png"
    assert explicit_svg.status == "image_unavailable"
    assert explicit_svg.display is None
    assert explicit_svg.warnings == ["svg_rasterizer_unavailable"]


def test_corrupt_notebook_image_returns_explicit_unavailable(
    tmp_path: Path, library: LocalLibrary
) -> None:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "id": "broken-image",
                "cell_type": "code",
                "metadata": {},
                "execution_count": 1,
                "source": "display(saved_output)",
                "outputs": [
                    {
                        "output_type": "display_data",
                        "metadata": {},
                        "data": {"image/png": "not-valid-base64!"},
                    }
                ],
            }
        ],
    }
    path = library.course_root / "broken-output.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    imported = library.import_paths("demo", [path])[0]

    result = _service(library, tmp_path).render_current(
        _request(imported, "cell:0:output:0", "notebook_output")
    )

    assert result.status == "image_unavailable"
    assert result.display is None
    assert result.warnings == ["output_has_no_available_image"]


def test_current_scope_hash_epoch_and_historical_grant_are_independent(
    library: LocalLibrary, tmp_path: Path
) -> None:
    first = _copy_import(library, "Signal-lab.ipynb")
    path = library.course_root / "Signal-lab.ipynb"
    shutil.copy2(SAMPLES / "revisions" / "v2" / "Signal-lab.ipynb", path)
    second = library.import_paths("demo", [path], document_id=first.document_id)[0]
    old_request = _request(first, "cell:1:output:2", "notebook_output")

    stale = _service(library, tmp_path).render_current(old_request)
    denied = _service(library, tmp_path, courses=("other",)).render_current(
        _request(second, "cell:1:output:2", "notebook_output")
    )
    historical = _service(library, tmp_path).render_historical(
        old_request.model_copy(update={"expected_publication_epoch": None}),
        HistoricalSourceGrant(
            source=old_request.source,
            course_ids=("demo",),
            content_sha256=first.content_sha256,
            message_id="message-1",
            evidence_id="evidence-1",
        ),
    )

    assert stale.status == "stale_reference"
    assert denied.status == "not_accessible" and denied.content_sha256 is None
    assert historical.status == "available"
    assert (
        historical.display.byte_sha256
        == "ce741007281a939680e7a21020fc80c34abfcb10d6d76d13e7984bc9d5743aa5"
    )
    assert historical.content_sha256 == first.content_sha256


def test_empty_allowlist_and_inline_cap_fail_closed(library: LocalLibrary, tmp_path: Path) -> None:
    imported = _copy_import(library, "Signal-reference.pdf")
    request = _request(imported, "page:1", "original_page")

    closed = _service(library, tmp_path, courses=()).render_current(request)
    capped = _service(library, tmp_path, max_inline_bytes=32).render_current(request)

    assert closed.status == "not_accessible" and closed.content_sha256 is None
    assert capped.status == "rendition_unavailable"
    assert capped.display is None and capped.warnings == ["inline_limit_exceeded"]
    assert capped.native_fallback.source == request.source
    assert capped.native_fallback.locator == {"format": "pdf", "page": 1}


def test_failed_or_oversize_delivery_rechecks_authorization(tmp_path: Path) -> None:
    original = _rotated_pdf(rotation=0)
    digest = hashlib.sha256(original).hexdigest()
    source = SourceDraft(
        source_id="page:1",
        kind="page",
        ordinal=0,
        title="Authorized once",
        text="native fallback",
        locator={
            "format": "pdf",
            "page": 1,
            "rotation_degrees_clockwise": 0,
            "visible_page_bbox_pdf_units": [10, 10, 190, 90],
        },
    )

    class RevokedAfterFirstRead:
        calls = 0

        def read_source(self, key, *, course_ids):
            del key, course_ids
            self.calls += 1
            if self.calls == 1:
                return HistoricalSource("available", source, digest, original, 1, True)
            return HistoricalSource("not_accessible")

    request = EvidenceRequest(
        source=SourceKey(document_id="doc", version=1, source_id="page:1"),
        expected_content_sha256=digest,
        expected_publication_epoch=1,
        rendition="original_page",
    )
    service = SourceEvidenceService(
        RevokedAfterFirstRead(),
        cache_dir=tmp_path / "cache",
        current_course_ids=["demo"],
        max_inline_bytes=32,
    )

    result = service.render_current(request)

    assert result.status == "not_accessible"
    assert result.content_sha256 is None
    assert result.native_fallback is None


def test_cross_page_object_is_not_stitched_or_mislabelled(tmp_path: Path) -> None:
    original = _rotated_pdf(rotation=0)
    digest = hashlib.sha256(original).hexdigest()
    child = SourceDraft(
        source_id="page:2:object:1",
        kind="page_object",
        ordinal=1,
        parent_id="page:1",
        title="Cross-page object",
        text="native text only",
        locator={"format": "pdf", "page": 2, "position_basis": "unknown"},
    )
    parent = SourceDraft(
        source_id="page:1",
        kind="page",
        ordinal=0,
        locator={
            "format": "pdf",
            "page": 1,
            "rotation_degrees_clockwise": 0,
            "visible_page_bbox_pdf_units": [10, 10, 190, 90],
        },
    )

    class InconsistentArchive:
        def read_source(self, key, *, course_ids):
            assert course_ids == ("demo",)
            source = child if key.source_id == child.source_id else parent
            return HistoricalSource("available", source, digest, original, 1, True)

    request = EvidenceRequest(
        source=SourceKey(document_id="doc", version=1, source_id=child.source_id),
        expected_content_sha256=digest,
        expected_publication_epoch=1,
        rendition="object_crop",
    )

    result = SourceEvidenceService(
        InconsistentArchive(),
        cache_dir=tmp_path / "cache",
        current_course_ids=["demo"],
    ).render_current(request)

    assert result.status == "rendition_unavailable"
    assert result.display is None
    assert result.warnings == ["cross_page_object_unsupported"]
    assert result.native_fallback.source == request.source


def test_started_pptx_conversion_timeout_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*args, **kwargs):
        del args, kwargs
        raise subprocess.TimeoutExpired(cmd="soffice", timeout=60)

    monkeypatch.setattr(source_evidence_module.subprocess, "run", timeout)

    with pytest.raises(Exception) as failure:
        source_evidence_module._convert_pptx(b"public fixture bytes", Path("/fake/soffice"))

    assert getattr(failure.value, "status", None) == "rendition_unavailable"
    assert getattr(failure.value, "reason", None) == "converter_timeout"


def test_api_forbids_scope_path_and_arbitrary_bbox(library: LocalLibrary, tmp_path: Path) -> None:
    imported = _copy_import(library, "Signal-reference.pdf")
    service = _service(library, tmp_path)
    app = create_app()
    app.dependency_overrides[get_source_evidence_service] = lambda: service
    payload = _request(imported, "page:1", "original_page").model_dump(mode="json")
    payload["course_ids"] = ["demo"]
    payload["bbox"] = [0, 0, 1, 1]
    payload["path"] = "/tmp/source.pdf"

    async def call():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post("/api/rag/local/evidence/current", json=payload)

    import asyncio

    response = asyncio.run(call())
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert {item["loc"][-1] for item in errors} == {"course_ids", "bbox", "path"}


def test_source_key_rejects_hidden_path_field() -> None:
    with pytest.raises(ValueError, match="source only accepts"):
        EvidenceRequest.model_validate(
            {
                "source": {
                    "document_id": "doc",
                    "version": 1,
                    "source_id": "page:1",
                    "path": "/tmp/private.pdf",
                },
                "expected_content_sha256": "a" * 64,
                "expected_publication_epoch": 1,
                "rendition": "original_page",
            }
        )


def test_base_install_can_import_main_without_pillow() -> None:
    script = r"""
import importlib.abc
import sys

class RejectPillow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "PIL" or fullname.startswith("PIL."):
            raise ModuleNotFoundError("Pillow intentionally unavailable")
        return None

sys.meta_path.insert(0, RejectPillow())
from app.main import create_app
from app.rag.source_evidence import _inspect_raster

app = create_app()
assert app.title == "SRAG"
try:
    _inspect_raster(b"not-an-image", "image/png")
except Exception as error:
    assert getattr(error, "status", None) == "image_unavailable"
    assert getattr(error, "reason", None) == "pillow_unavailable"
else:
    raise AssertionError("image operation unexpectedly succeeded without Pillow")
print("main-import-ok;pillow=image_unavailable")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.stdout.strip() == "main-import-ok;pillow=image_unavailable"


def test_pdf_rotation_transform_matches_signed_formula() -> None:
    rect = _pdf_rect_to_visible((50, 30, 100, 60), (10, 10, 190, 90), 90)
    assert rect == pytest.approx((0.25, 40 / 180, 0.625, 90 / 180))
    assert _pdf_rect_to_visible((50, 30, 100, 60), (10, 10, 190, 90), 0) == pytest.approx(
        (40 / 180, 30 / 80, 90 / 180, 60 / 80)
    )


@pytest.mark.skipif(not os.getenv("SRAG_SOFFICE"), reason="SRAG_SOFFICE not configured")
def test_public_pptx_full_page_and_table_crop_with_real_libreoffice(
    library: LocalLibrary, tmp_path: Path
) -> None:
    soffice = Path(os.environ["SRAG_SOFFICE"])
    imported = _copy_import(library, "Signal-slides.pptx")
    service = _service(library, tmp_path, soffice=soffice)

    page = service.render_current(_request(imported, "page:1", "original_page"))
    table_request = _request(imported, "page:1:shape:3", "object_crop")
    table = service.render_current(table_request)

    assert page.status == table.status == "available"
    assert page.display.renderer == table.display.renderer == "libreoffice+pypdfium2"
    assert page.display.display_source.source_id == "page:1"
    assert table.display.display_source == table_request.source
    assert table.display.rect_visible_norm == pytest.approx((0.08, 0.28, 0.92, 0.92))
    assert table.display.width < page.display.width and table.display.height < page.display.height


def test_pptx_converter_absence_is_explicit(library: LocalLibrary, tmp_path: Path) -> None:
    imported = _copy_import(library, "Signal-slides.pptx")
    service = _service(library, tmp_path, soffice=tmp_path / "missing-soffice")

    result = service.render_current(_request(imported, "page:1", "original_page"))

    if (
        shutil.which("soffice")
        or Path("/Applications/LibreOffice.app/Contents/MacOS/soffice").exists()
    ):
        assert result.status == "available"
    else:
        assert result.status == "rendition_unavailable"
        assert result.warnings == ["converter_unavailable"]


def _rotated_pdf(*, rotation: int = 90) -> bytes:
    content = b"1 0 0 rg 50 30 50 30 re f 0 0 1 rg 50 15 50 10 re f"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] "
            b"/CropBox [10 10 190 90] /Rotate "
            + str(rotation).encode("ascii")
            + b" /Resources << >> /Contents 4 0 R >>"
        ),
        b"<< /Length "
        + str(len(content)).encode("ascii")
        + b" >>\nstream\n"
        + content
        + b"\nendstream",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, value in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(value)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode(
            "ascii"
        )
    )
    return bytes(output)
