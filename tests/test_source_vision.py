from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import shutil
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock

import httpx
import pytest

from app.api.source_evidence import get_source_vision_service
from app.config import Settings
from app.main import create_app
from app.rag.local_library import LocalLibrary
from app.rag.source_evidence import (
    EvidenceRequest,
    ReadRecord,
    RegionRegistry,
    RegisteredRegion,
    SourceEvidenceService,
)
from app.rag.source_vision import SourceVisionService, VisionReply, _cache_key
from app.schemas.local import ParsedSource, SourceDraft, SourceKey
from app.schemas.rag import SearchRequest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "docs" / "A" / "样例" / "Signal-reference.pdf"


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


def _import_pdf(library: LocalLibrary, *, document_id: str | None = None):
    target = library.course_root / "signal.pdf"
    if not target.exists():
        shutil.copy2(SAMPLE, target)
    return target, library.import_paths("demo", [target], document_id=document_id)[0]


def _request(imported, source_id: str = "page:1", **updates) -> EvidenceRequest:
    values = {
        "source": SourceKey(
            document_id=imported.document_id,
            version=imported.version,
            source_id=source_id,
        ),
        "expected_content_sha256": imported.content_sha256,
        "expected_publication_epoch": imported.publication_epoch,
        "rendition": "original_page",
        "read_intent": "directed_question",
        "question": "What is the visible chart title?",
    }
    values.update(updates)
    return EvidenceRequest(**values)


def _evidence(
    library: LocalLibrary,
    tmp_path: Path,
    *,
    regions: RegionRegistry | None = None,
) -> SourceEvidenceService:
    return SourceEvidenceService(
        library.store,
        cache_dir=tmp_path / "renditions",
        current_course_ids=["demo"],
        regions=regions,
    )


def _success(request: httpx.Request, *, answer: str = "Signal comparison") -> httpx.Response:
    return httpx.Response(
        200,
        request=request,
        json={
            "output": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": [{"text": json.dumps({"answer": answer, "uncertain": []})}]
                        },
                    }
                ]
            },
            "usage": {"input_tokens": 11, "output_tokens": 4},
            "request_id": "mock-request",
        },
    )


def _vision(
    evidence: SourceEvidenceService,
    library: LocalLibrary,
    tmp_path: Path,
    handler,
    *,
    cache_name: str = "vision",
    model: str = "qwen-vl-max",
    prompt_version: str = "source-directed-v1",
) -> SourceVisionService:
    return SourceVisionService(
        evidence,
        library.store,
        cache_dir=tmp_path / cache_name,
        enabled=True,
        api_key="",
        base_url="https://mock.invalid/api/v1",
        model=model,
        timeout_seconds=2,
        max_image_bytes=8 * 1024 * 1024,
        execution_kind="mock_protocol",
        transport=httpx.MockTransport(handler),
        directed_prompt_version=prompt_version,
    )


def _hits(library: LocalLibrary, query: str):
    return library.search(SearchRequest(query=query, course_ids=["demo"], top_k=20))


def test_disabled_and_unconfigured_provider_leave_image_unread(
    library: LocalLibrary, tmp_path: Path
) -> None:
    _, imported = _import_pdf(library)
    evidence = _evidence(library, tmp_path)
    request = _request(imported)

    disabled = SourceVisionService(
        evidence,
        library.store,
        cache_dir=tmp_path / "disabled",
        enabled=False,
        api_key="",
        base_url="",
        model="qwen-vl-max",
        timeout_seconds=2,
        max_image_bytes=8 * 1024 * 1024,
    ).process_current(request)
    unconfigured = SourceVisionService(
        evidence,
        library.store,
        cache_dir=tmp_path / "unconfigured",
        enabled=True,
        api_key="",
        base_url="",
        model="qwen-vl-max",
        timeout_seconds=2,
        max_image_bytes=8 * 1024 * 1024,
    ).process_current(request)

    assert disabled.read_status == unconfigured.read_status == "not_attempted"
    assert disabled.capability_reason == "provider_disabled"
    assert unconfigured.capability_reason == "provider_not_configured"
    assert disabled.external_call_count == unconfigured.external_call_count == 0
    assert disabled.display is not None and unconfigured.display is not None


@pytest.mark.parametrize(
    ("base_url", "api_key"),
    [
        ("not-a-valid-url", "credential"),
        ("https://user:password@example.invalid/api/v1", "credential"),
        ("https://example.invalid/api/v1?redirect=other", "credential"),
        ("https://example.invalid/api/v1#fragment", "credential"),
        ("https://example.invalid/api/v1", "credential\nsecond-header: value"),
        ("https://example.invalid/api/v1", "非ASCII凭据"),
    ],
)
def test_invalid_real_provider_configuration_is_not_counted_or_sent(
    library: LocalLibrary,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    base_url: str,
    api_key: str,
) -> None:
    _, imported = _import_pdf(library)

    def forbidden_post(*args, **kwargs):
        del args, kwargs
        raise AssertionError("invalid local configuration must not reach Client.post")

    monkeypatch.setattr(httpx.Client, "post", forbidden_post)
    result = SourceVisionService(
        _evidence(library, tmp_path),
        library.store,
        cache_dir=tmp_path / "invalid-provider",
        enabled=True,
        api_key=api_key,
        base_url=base_url,
        model="qwen-vl-max",
        timeout_seconds=0.1,
        max_image_bytes=8 * 1024 * 1024,
    ).process_current(_request(imported))

    assert result.status == "available" and result.display is not None
    assert result.read_status == "not_attempted"
    assert result.capability_reason == "provider_not_configured"
    assert result.reads == [] and result.external_call_count == 0


def test_valid_real_endpoint_connection_failure_counts_attempt(
    library: LocalLibrary, tmp_path: Path
) -> None:
    _, imported = _import_pdf(library)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        unused_port = probe.getsockname()[1]
    result = SourceVisionService(
        _evidence(library, tmp_path),
        library.store,
        cache_dir=tmp_path / "connection-failure",
        enabled=True,
        api_key="test-only",
        base_url=f"http://127.0.0.1:{unused_port}/api/v1",
        model="qwen-vl-max",
        timeout_seconds=0.1,
        max_image_bytes=8 * 1024 * 1024,
    ).process_current(_request(imported))

    assert result.status == "available"
    assert result.read_status == "failed" and result.capability_reason is None
    assert result.reads[0].missing_or_error in (
        ["provider_ConnectError"],
        ["provider_timeout"],
    )
    assert result.external_call_count == 1


def test_mock_protocol_receives_exact_display_bytes_and_directed_read_is_not_indexed(
    library: LocalLibrary, tmp_path: Path
) -> None:
    _, imported = _import_pdf(library)
    evidence = _evidence(library, tmp_path)
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload)
        return _success(request, answer="visiononlytoken9f4c")

    service = _vision(evidence, library, tmp_path, handler)
    request = _request(imported)
    first = service.process_current(request)
    second = service.process_current(request)

    assert len(seen) == 1
    image_items = seen[0]["input"]["messages"][0]["content"]
    assert [set(item) for item in image_items] == [{"image"}, {"text"}]
    encoded = image_items[0]["image"].split(",", 1)[1]
    sent = base64.b64decode(encoded, validate=True)
    assert hashlib.sha256(sent).hexdigest() == first.display.byte_sha256
    assert len(sent) == first.display.byte_length
    assert first.read_status == second.read_status == "not_attempted"
    assert first.capability_reason == second.capability_reason == "mock_only"
    assert first.reads[0].execution == "mock_protocol"
    assert second.reads[0].execution == "cache_reuse"
    assert first.external_call_count == second.external_call_count == 0
    assert not _hits(library, "visiononlytoken9f4c")


def test_mock_generic_transcription_is_never_written_to_fts(
    library: LocalLibrary, tmp_path: Path
) -> None:
    _, imported = _import_pdf(library)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "output": {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": [
                                    {
                                        "text": json.dumps(
                                            {
                                                "text": "mockgenericword81a2",
                                                "formulas": [],
                                                "symbols": [],
                                                "descriptions": [],
                                                "uncertain": [],
                                            }
                                        )
                                    }
                                ]
                            },
                        }
                    ]
                }
            },
        )

    result = _vision(_evidence(library, tmp_path), library, tmp_path, handler).process_current(
        _request(imported, read_intent="generic_transcription", question=None)
    )
    assert result.reads[0].status == "succeeded"
    assert result.reads[0].text == "mockgenericword81a2"
    assert not _hits(library, "mockgenericword81a2")


def test_question_model_prompt_and_execution_are_cache_dimensions(
    library: LocalLibrary, tmp_path: Path
) -> None:
    _, imported = _import_pdf(library)
    evidence = _evidence(library, tmp_path)
    display = evidence.render_current(
        _request(imported).model_copy(update={"read_intent": "display_only", "question": None})
    ).display
    first = _request(imported)
    second = first.model_copy(update={"question": "What unit is shown?"})
    common = dict(
        purpose="directed_question",
        provider="dashscope-native",
        question_sha=None,
    )
    keys = {
        _cache_key(
            first,
            display,
            execution_kind="mock_protocol",
            model="m1",
            prompt_version="p1",
            **common,
        ),
        _cache_key(
            second,
            display,
            execution_kind="mock_protocol",
            model="m1",
            prompt_version="p1",
            **(common | {"question_sha": hashlib.sha256(second.question.encode()).hexdigest()}),
        ),
        _cache_key(
            first,
            display,
            execution_kind="mock_protocol",
            model="m2",
            prompt_version="p1",
            **common,
        ),
        _cache_key(
            first,
            display,
            execution_kind="mock_protocol",
            model="m1",
            prompt_version="p2",
            **common,
        ),
        _cache_key(
            first,
            display,
            execution_kind="real_provider",
            model="m1",
            prompt_version="p1",
            **common,
        ),
    }
    assert len(keys) == 5


def test_version_and_epoch_changes_do_not_reuse_visual_cache(
    library: LocalLibrary, tmp_path: Path
) -> None:
    path, first_import = _import_pdf(library)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _success(request, answer=f"call-{calls}")

    service = _vision(_evidence(library, tmp_path), library, tmp_path, handler)
    first = service.process_current(_request(first_import))
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    second_import = library.import_paths("demo", [path], document_id=first_import.document_id)[0]
    second = service.process_current(_request(second_import))
    assert calls == 2
    assert first.reads[0].text == "call-1" and second.reads[0].text == "call-2"
    assert service.process_current(_request(first_import)).status == "stale_reference"

    assert library.delete(second_import.document_id)
    revived = library.import_paths("demo", [path], document_id=second_import.document_id)[0]
    assert revived.version == second_import.version
    assert revived.publication_epoch > second_import.publication_epoch
    third = service.process_current(_request(revived))
    assert calls == 3 and third.reads[0].text == "call-3"


def test_registered_region_change_invalidates_visual_input_and_cache(
    library: LocalLibrary, tmp_path: Path
) -> None:
    _, imported = _import_pdf(library)
    source = SourceKey(
        document_id=imported.document_id, version=imported.version, source_id="page:1"
    )
    region = RegisteredRegion(
        region_id="public-chart",
        source=source,
        content_sha256=imported.content_sha256,
        coordinate_frame="visible_page_norm_v1",
        target_rect=(0.05, 0.05, 0.55, 0.55),
        provenance="trusted_corpus_registration",
        revision=1,
    )
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _success(request)

    request = _request(
        imported,
        rendition="object_crop",
        registered_region_id="public-chart",
    )
    first = _vision(
        _evidence(library, tmp_path, regions=RegionRegistry([region])),
        library,
        tmp_path,
        handler,
    ).process_current(request)
    changed = region.model_copy(update={"revision": 2, "target_rect": (0.15, 0.15, 0.75, 0.75)})
    second = _vision(
        _evidence(library, tmp_path, regions=RegionRegistry([changed])),
        library,
        tmp_path,
        handler,
    ).process_current(request)
    assert calls == 2
    assert first.display.rect_visible_norm != second.display.rect_visible_norm
    assert first.display.byte_sha256 != second.display.byte_sha256


def test_visual_singleflight_is_shared_across_service_instances(
    library: LocalLibrary, tmp_path: Path
) -> None:
    _, imported = _import_pdf(library)
    evidence = _evidence(library, tmp_path)
    request = _request(imported)
    evidence.render_current(
        request.model_copy(update={"read_intent": "display_only", "question": None})
    )
    entered = Event()
    release = Event()
    guard = Lock()
    calls = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        with guard:
            calls += 1
        entered.set()
        assert release.wait(3)
        return _success(http_request)

    services = [
        _vision(evidence, library, tmp_path, handler),
        _vision(evidence, library, tmp_path, handler),
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(services[0].process_current, request)
        assert entered.wait(3)
        second = pool.submit(services[1].process_current, request)
        release.set()
        results = [first.result(timeout=5), second.result(timeout=5)]

    assert calls == 1
    assert {result.reads[0].execution for result in results} == {
        "mock_protocol",
        "cache_reuse",
    }


def test_mismatched_persisted_record_is_discarded_as_cache_miss(
    library: LocalLibrary, tmp_path: Path
) -> None:
    _, imported = _import_pdf(library)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _success(request, answer=f"read-{calls}")

    service = _vision(_evidence(library, tmp_path), library, tmp_path, handler)
    request = _request(imported)
    first = service.process_current(request)
    cache_file = next((tmp_path / "vision").glob("*.json"))
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    payload["requested_source"]["source_id"] = "page:2"
    cache_file.write_text(json.dumps(payload), encoding="utf-8")

    second = service.process_current(request)
    assert calls == 2
    assert first.reads[0].text == "read-1"
    assert second.reads[0].text == "read-2"
    assert second.reads[0].execution == "mock_protocol"


@pytest.mark.parametrize(
    "payload",
    [
        [],
        None,
        {"output": {"choices": []}},
        {
            "output": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": [{"text": '{"answer":"ok","uncertain":[]}'}]},
                    }
                ]
            },
            "usage": [],
        },
        {
            "output": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": [{"text": None}]},
                    }
                ]
            }
        },
        {
            "output": {
                "choices": [
                    {
                        "finish_reason": [],
                        "message": {"content": [{"text": "{}"}]},
                    }
                ]
            }
        },
        {
            "output": {
                "choices": [
                    {
                        "finish_reason": {},
                        "message": {"content": [{"text": "{}"}]},
                    }
                ]
            }
        },
    ],
)
def test_malformed_provider_responses_are_failed_and_not_cached(
    library: LocalLibrary, tmp_path: Path, payload
) -> None:
    _, imported = _import_pdf(library)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, request=request, json=payload)

    service = _vision(_evidence(library, tmp_path), library, tmp_path, handler)
    first = service.process_current(_request(imported))
    second = service.process_current(_request(imported))
    assert calls == 2
    assert first.reads[0].status == second.reads[0].status == "failed"
    assert first.reads[0].text == second.reads[0].text == ""


def test_provider_http_failure_is_not_cached(library: LocalLibrary, tmp_path: Path) -> None:
    _, imported = _import_pdf(library)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, request=request, text="private provider body")

    service = _vision(_evidence(library, tmp_path), library, tmp_path, handler)
    results = [service.process_current(_request(imported)) for _ in range(2)]
    assert calls == 2
    assert [result.reads[0].missing_or_error for result in results] == [
        ["provider_http_503"],
        ["provider_http_503"],
    ]
    assert all("private provider body" not in json.dumps(result.model_dump()) for result in results)


def test_malformed_response_after_version_change_returns_redacted_stale(
    library: LocalLibrary, tmp_path: Path
) -> None:
    path, imported = _import_pdf(library)

    def handler(request: httpx.Request) -> httpx.Response:
        path.write_bytes(path.read_bytes() + b"\nmalformed-race")
        replacement = library.import_paths("demo", [path], document_id=imported.document_id)[0]
        assert replacement.version > imported.version
        return httpx.Response(
            200,
            request=request,
            json={
                "output": {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": [{"text": None}]},
                        }
                    ]
                }
            },
        )

    result = _vision(_evidence(library, tmp_path), library, tmp_path, handler).process_current(
        _request(imported)
    )
    assert result.status == "stale_reference"
    assert result.display is None and result.reads == []
    assert result.external_call_count == 0


def test_source_change_during_success_or_failure_returns_redacted_stale_with_call_count(
    library: LocalLibrary, tmp_path: Path
) -> None:
    path, imported = _import_pdf(library)
    current = imported

    for outcome in ("success", "failure"):
        request = _request(current)
        replacements = []
        service = SourceVisionService(
            _evidence(library, tmp_path),
            library.store,
            cache_dir=tmp_path / f"race-{outcome}",
            enabled=True,
            api_key="test-only",
            base_url="https://unused.invalid/api/v1",
            model="qwen-vl-max",
            timeout_seconds=2,
            max_image_bytes=8 * 1024 * 1024,
        )

        def change_then_return(
            *, _request=request, _outcome=outcome, _replacements=replacements, **kwargs
        ):
            del kwargs
            path.write_bytes(path.read_bytes() + f"\n{_outcome}".encode())
            replacement = library.import_paths(
                "demo", [path], document_id=_request.source.document_id
            )[0]
            assert replacement.version > _request.source.version
            _replacements.append(replacement)
            if _outcome == "failure":
                raise httpx.ReadTimeout("unit timeout")
            return VisionReply("succeeded", "must not leak", (), "request-id", {})

        service.provider.read = change_then_return
        result = service.process_current(request)
        assert result.status == "stale_reference"
        assert result.display is None and result.reads == []
        assert result.read_status == "not_attempted"
        assert result.external_call_count == 1
        current = replacements[0]


def test_generic_real_origin_writes_actual_source_and_late_delete_is_rejected(
    library: LocalLibrary, tmp_path: Path
) -> None:
    _, imported = _import_pdf(library)
    request = _request(
        imported,
        read_intent="generic_transcription",
        question=None,
    )
    service = SourceVisionService(
        _evidence(library, tmp_path),
        library.store,
        cache_dir=tmp_path / "generic",
        enabled=False,
        api_key="",
        base_url="",
        model="qwen-vl-max",
        timeout_seconds=2,
        max_image_bytes=8 * 1024 * 1024,
    )
    record = ReadRecord(
        requested_source=request.source,
        actual_source=request.source,
        actual_input_sha256="a" * 64,
        actual_input_media_type="image/png",
        purpose="generic_transcription",
        execution="real_provider",
        origin_execution="real_provider",
        provider="dashscope-native",
        model="qwen-vl-max",
        prompt_version="source-generic-v1",
        status="succeeded",
        text="uniquevisualword",
        external_call=True,
    )
    assert service._publish_generic(record, request)
    assert _hits(library, "uniquevisualword")[0].source_id == request.source.source_id
    assert library.delete(imported.document_id)
    assert not service._publish_generic(record, request)
    assert not _hits(library, "uniquevisualword")


def test_fallback_requests_do_not_share_record_identity(
    library: LocalLibrary, tmp_path: Path
) -> None:
    original = SAMPLE.read_bytes()
    path = library.course_root / "fallback.pdf"
    path.write_bytes(original)
    imported = library.store.publish(
        library.store.prepare("demo", path),
        ParsedSource(
            title="Fallback",
            sources=[
                SourceDraft(
                    source_id="page:1",
                    kind="page",
                    ordinal=0,
                    locator={"format": "pdf", "page": 1},
                ),
                SourceDraft(
                    source_id="page:1:object:1",
                    kind="page_object",
                    ordinal=1,
                    parent_id="page:1",
                    locator={"format": "pdf", "page": 1, "position_basis": "unknown"},
                ),
                SourceDraft(
                    source_id="page:1:object:2",
                    kind="page_object",
                    ordinal=2,
                    parent_id="page:1",
                    locator={"format": "pdf", "page": 1, "position_basis": "unknown"},
                ),
            ],
        ),
        original=original,
        content_sha256=hashlib.sha256(original).hexdigest(),
        document_format="pdf",
    )
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _success(request)

    service = _vision(_evidence(library, tmp_path), library, tmp_path, handler)
    requests = [
        _request(imported, source_id, rendition="object_crop")
        for source_id in ("page:1:object:1", "page:1:object:2")
    ]
    results = [service.process_current(request) for request in requests]
    cached = service.process_current(requests[0])
    assert calls == 2
    assert [result.reads[0].requested_source for result in results] == [
        request.source for request in requests
    ]
    assert results[0].reads[0].actual_source == results[1].reads[0].actual_source
    assert cached.reads[0].requested_source == requests[0].source
    assert cached.reads[0].execution == "cache_reuse"


def test_mock_transport_contract_prevents_false_real_or_credentialed_mock(
    library: LocalLibrary, tmp_path: Path
) -> None:
    evidence = _evidence(library, tmp_path)
    transport = httpx.MockTransport(_success)
    common = dict(
        evidence=evidence,
        writer=library.store,
        cache_dir=tmp_path / "contract",
        enabled=True,
        base_url="https://mock.invalid",
        model="qwen-vl-max",
        timeout_seconds=2,
        max_image_bytes=1024,
    )
    with pytest.raises(ValueError, match="does not accept"):
        SourceVisionService(
            **common,
            api_key="secret",
            execution_kind="real_provider",
            transport=transport,
        )
    with pytest.raises(ValueError, match="must not receive"):
        SourceVisionService(
            **common,
            api_key="secret",
            execution_kind="mock_protocol",
            transport=transport,
        )
    with pytest.raises(ValueError, match="requires"):
        SourceVisionService(
            **common,
            api_key="",
            execution_kind="mock_protocol",
            transport=None,
        )


def test_current_http_route_is_backward_compatible_and_accepts_read_intent(
    library: LocalLibrary, tmp_path: Path
) -> None:
    _, imported = _import_pdf(library)
    service = _vision(
        _evidence(library, tmp_path),
        library,
        tmp_path,
        lambda request: _success(request),
    )
    app = create_app()
    app.dependency_overrides[get_source_vision_service] = lambda: service
    display_request = _request(imported).model_copy(
        update={"read_intent": "display_only", "question": None}
    )
    read_request = _request(imported)

    async def call():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await asyncio.gather(
                client.post(
                    "/api/rag/local/evidence/current",
                    json=display_request.model_dump(mode="json"),
                ),
                client.post(
                    "/api/rag/local/evidence/current",
                    json=read_request.model_dump(mode="json"),
                ),
            )

    display_response, read_response = asyncio.run(call())
    assert display_response.status_code == read_response.status_code == 200
    display_body = display_response.json()
    read_body = read_response.json()
    assert display_body["status"] == "available"
    assert display_body["read_status"] == "not_attempted" and display_body["reads"] == []
    assert read_body["capability_reason"] == "mock_only"
    assert read_body["reads"][0]["text"] == "Signal comparison"
