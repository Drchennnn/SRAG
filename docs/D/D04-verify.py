#!/usr/bin/env python3
"""Public, bounded D04 verifier for source display and visual reading.

Run from the repository root. By default it performs only local display/mock/fault
checks. ``--real`` additionally makes at most four no-retry DashScope calls over the
three public samples using the existing environment configuration. Credentials are
never serialized.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageDraw

from app.api.source_evidence import get_source_vision_service
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
)
from app.rag.source_vision import SourceVisionService
from app.schemas.local import ParsedSource, SourceDraft, SourceKey

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "docs" / "A" / "样例"
COURSE_ID = "public-d04"
D03_BASE = "12264dab2d85db195eea44f7591e4df524494ae7"
MAX_REAL_PROVIDER_CALLS = 4


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha(path.read_bytes())


def _artifact_reference(path: Path, output_dir: Path) -> str:
    if output_dir.is_relative_to(ROOT):
        return path.relative_to(ROOT).as_posix()
    return path.relative_to(output_dir).as_posix()


def _resolve_artifact(reference: str, output_dir: Path) -> Path:
    if output_dir.is_relative_to(ROOT):
        return ROOT / reference
    return output_dir / reference


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _request(imported: Any, source_id: str, rendition: str, **updates: Any) -> EvidenceRequest:
    return EvidenceRequest(
        source=SourceKey(
            document_id=imported.document_id,
            version=imported.version,
            source_id=source_id,
        ),
        expected_content_sha256=imported.content_sha256,
        expected_publication_epoch=imported.publication_epoch,
        rendition=rendition,
        **updates,
    )


def _evidence(
    library: LocalLibrary,
    root: Path,
    *,
    regions: RegionRegistry | None = None,
    soffice: Path | None = None,
) -> SourceEvidenceService:
    return SourceEvidenceService(
        library.store,
        cache_dir=root / "renditions",
        current_course_ids=(COURSE_ID,),
        regions=regions,
        soffice_path=soffice,
    )


def _vision(
    evidence: SourceEvidenceService,
    library: LocalLibrary,
    root: Path,
    *,
    cache_name: str,
    enabled: bool,
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
    execution_kind: str = "real_provider",
    transport: httpx.BaseTransport | None = None,
) -> SourceVisionService:
    return SourceVisionService(
        evidence,
        library.store,
        cache_dir=root / cache_name,
        enabled=enabled,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=timeout,
        max_image_bytes=8 * 1024 * 1024,
        execution_kind=execution_kind,
        transport=transport,
    )


async def _post(
    app: Any,
    holder: dict[str, SourceVisionService],
    service: SourceVisionService,
    request: EvidenceRequest,
) -> tuple[dict[str, Any], float]:
    holder["service"] = service
    started = time.perf_counter()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://d04.local"
    ) as client:
        response = await client.post(
            "/api/rag/local/evidence/current", json=request.model_dump(mode="json")
        )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    return response.json(), elapsed


def _call(
    app: Any,
    holder: dict[str, SourceVisionService],
    service: SourceVisionService,
    request: EvidenceRequest,
) -> tuple[dict[str, Any], float]:
    return asyncio.run(_post(app, holder, service, request))


def _compact(
    name: str,
    request: EvidenceRequest,
    body: dict[str, Any],
    elapsed: float,
    output_dir: Path,
    *,
    provenance: str,
) -> dict[str, Any]:
    compact = {
        "name": name,
        "provenance": provenance,
        "elapsed_seconds": round(elapsed, 4),
        "request": request.model_dump(mode="json"),
        "status": body["status"],
        "read_status": body["read_status"],
        "capability_reason": body["capability_reason"],
        "external_call_count": body["external_call_count"],
        "content_sha256": body.get("content_sha256"),
        "warnings": body.get("warnings", []),
        "reads": body.get("reads", []),
    }
    display = body.get("display")
    if display:
        raw = base64.b64decode(display["data_base64"], validate=True)
        assert len(raw) == display["byte_length"]
        assert _sha(raw) == display["byte_sha256"]
        extension = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}[
            display["media_type"]
        ]
        artifact = output_dir / f"{name}.{extension}"
        artifact.write_bytes(raw)
        display = {key: value for key, value in display.items() if key != "data_base64"}
        display["artifact"] = _artifact_reference(artifact, output_dir)
        compact["display"] = display
    else:
        compact["display"] = None
    return compact


def _mock_response(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content)
    prompt = payload["input"]["messages"][0]["content"][1]["text"]
    if "Answer only this question" in prompt:
        answer = {"answer": "mock protocol only", "uncertain": []}
    else:
        answer = {
            "text": "mock transcription only",
            "formulas": [],
            "symbols": [],
            "descriptions": [],
            "uncertain": [],
        }
    return httpx.Response(
        200,
        request=request,
        json={
            "output": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": [{"text": json.dumps(answer)}]},
                    }
                ]
            },
            "request_id": "mock-d04",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )


def _region(
    region_id: str,
    key: SourceKey,
    digest: str,
    target: tuple[float, float, float, float],
    caption: tuple[float, float, float, float],
    x_axis: tuple[float, float, float, float],
    revision: int = 1,
) -> RegisteredRegion:
    return RegisteredRegion(
        region_id=region_id,
        source=key,
        content_sha256=digest,
        coordinate_frame="visible_page_norm_v1",
        target_rect=target,
        components=[
            RegionComponent(role="caption", rect=caption, coordinate_frame="visible_page_norm_v1"),
            RegionComponent(role="x_axis", rect=x_axis, coordinate_frame="visible_page_norm_v1"),
        ],
        provenance="trusted_corpus_registration",
        revision=revision,
    )


def _enhancement(library: LocalLibrary, key: SourceKey) -> dict[str, Any] | None:
    settings = library.settings
    with sqlite3.connect(settings.sqlite_path) as db:
        row = db.execute(
            """
            SELECT f.body, f.origin
            FROM local_fts f
            JOIN local_sources s ON s.id = f.source_rowid
            WHERE s.document_id = ? AND s.version = ? AND s.source_id = ?
              AND f.origin = 'enhancement'
            """,
            (key.document_id, key.version, key.source_id),
        ).fetchone()
    return {"body": row[0], "origin": row[1]} if row else None


def _highlight(full_page: Path, rect: list[float], target: Path) -> None:
    with Image.open(full_page) as source:
        image = source.convert("RGB")
    left, top, right, bottom = rect
    box = (
        round(left * image.width),
        round(top * image.height),
        round(right * image.width),
        round(bottom * image.height),
    )
    draw = ImageDraw.Draw(image)
    width = max(4, round(min(image.size) * 0.006))
    for offset in range(width):
        draw.rectangle(
            (box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset),
            outline=(220, 25, 25),
        )
    image.save(target, format="PNG")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="perform at most four real calls")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "D" / "D04-evidence")
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "docs" / "D" / "D04-实测.json"
    )
    parser.add_argument("--soffice", type=Path, default=Path(os.environ.get("SRAG_SOFFICE", "")))
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    runtime = Settings()
    if args.real and not runtime.dashscope_api_key:
        raise SystemExit("--real requires an existing DashScope credential")
    soffice = args.soffice if str(args.soffice) and args.soffice.is_file() else None

    with tempfile.TemporaryDirectory(prefix="srag-d04-") as directory:
        temp = Path(directory)
        course = temp / "course"
        course.mkdir()
        settings = Settings(
            _env_file=None,
            sqlite_path=temp / "d04.sqlite3",
            data_dir=temp / "data",
            course_root=course,
            dashscope_api_key="",
            deepseek_api_key="",
            qdrant_url="http://127.0.0.1:1",
        )
        library = LocalLibrary(settings)
        imported: dict[str, Any] = {}
        paths: dict[str, Path] = {}
        for label, name in (
            ("pdf", "Signal-reference.pdf"),
            ("pptx", "Signal-slides.pptx"),
            ("notebook", "Signal-lab.ipynb"),
        ):
            target = course / name
            shutil.copy2(SAMPLES / name, target)
            result = library.import_paths(COURSE_ID, [target])[0]
            assert result.status in {"published", "backfilled", "skipped"}
            imported[label] = result
            paths[label] = target

        pdf = imported["pdf"]
        pdf_key = SourceKey(document_id=pdf.document_id, version=1, source_id="page:2")
        right = _region(
            "pdf-right",
            pdf_key,
            pdf.content_sha256,
            (0.53, 0.20, 0.92, 0.51),
            (0.54, 0.507, 0.91, 0.536),
            (0.61, 0.45, 0.83, 0.50),
        )
        left = _region(
            "pdf-left",
            pdf_key,
            pdf.content_sha256,
            (0.08, 0.20, 0.47, 0.51),
            (0.10, 0.507, 0.45, 0.536),
            (0.22, 0.45, 0.36, 0.50),
        )
        registry = RegionRegistry([left, right])
        evidence = _evidence(library, temp, regions=registry, soffice=soffice)
        disabled = _vision(
            evidence,
            library,
            temp,
            cache_name="disabled",
            enabled=False,
            api_key="",
            base_url="",
            model=runtime.dashscope_vision_model,
            timeout=15,
        )

        app = create_app()
        holder: dict[str, SourceVisionService] = {"service": disabled}
        app.dependency_overrides[get_source_vision_service] = lambda: holder["service"]
        scenarios: list[dict[str, Any]] = []

        display_requests = {
            "pdf-page2-full": _request(pdf, "page:2", "original_page"),
            "pdf-left-crop": _request(
                pdf, "page:2", "object_crop", registered_region_id="pdf-left"
            ),
            "pptx-page1-full": _request(imported["pptx"], "page:1", "original_page"),
            "notebook-v1-output1-unread": _request(
                imported["notebook"], "cell:1:output:1", "notebook_output"
            ),
            "notebook-v1-output2-unread": _request(
                imported["notebook"], "cell:1:output:2", "notebook_output"
            ),
        }
        for name, request in display_requests.items():
            body, elapsed = _call(app, holder, disabled, request)
            scenarios.append(
                _compact(name, request, body, elapsed, output_dir, provenance="display_only")
            )

        mock_calls = 0

        def counted_mock(request: httpx.Request) -> httpx.Response:
            nonlocal mock_calls
            mock_calls += 1
            return _mock_response(request)

        mock_service = _vision(
            evidence,
            library,
            temp,
            cache_name="mock-vision",
            enabled=True,
            api_key="",
            base_url="https://mock.invalid/api/v1",
            model=runtime.dashscope_vision_model,
            timeout=15,
            execution_kind="mock_protocol",
            transport=httpx.MockTransport(counted_mock),
        )
        right_request = _request(
            pdf,
            "page:2",
            "object_crop",
            registered_region_id="pdf-right",
            read_intent="directed_question",
            question=(
                "Which panel is shown, what are both axis labels, and what does its caption state?"
            ),
        )
        for name in ("pdf-right-mock", "pdf-right-mock-cache"):
            body, elapsed = _call(app, holder, mock_service, right_request)
            scenarios.append(
                _compact(
                    name,
                    right_request,
                    body,
                    elapsed,
                    output_dir,
                    provenance="mock_transport",
                )
            )

        changed_right = right.model_copy(
            update={"revision": 2, "target_rect": (0.55, 0.20, 0.92, 0.51)}
        )
        changed_evidence = _evidence(
            library, temp, regions=RegionRegistry([changed_right]), soffice=soffice
        )
        changed_mock = _vision(
            changed_evidence,
            library,
            temp,
            cache_name="mock-vision",
            enabled=True,
            api_key="",
            base_url="https://mock.invalid/api/v1",
            model=runtime.dashscope_vision_model,
            timeout=15,
            execution_kind="mock_protocol",
            transport=httpx.MockTransport(counted_mock),
        )
        body, elapsed = _call(app, holder, changed_mock, right_request)
        scenarios.append(
            _compact(
                "pdf-right-region-rev2",
                right_request,
                body,
                elapsed,
                output_dir,
                provenance="mock_transport_region_change",
            )
        )

        original = (SAMPLES / "Signal-reference.pdf").read_bytes()
        fallback_path = course / "fallback.pdf"
        fallback_path.write_bytes(original)
        fallback = library.store.publish(
            library.store.prepare(COURSE_ID, fallback_path),
            ParsedSource(
                title="D04 fallback",
                sources=[
                    SourceDraft(
                        source_id="page:1",
                        kind="page",
                        ordinal=0,
                        locator={
                            "format": "pdf",
                            "page": 1,
                            "rotation_degrees_clockwise": 0,
                            "visible_page_bbox_pdf_units": [0, 0, 612, 792],
                        },
                    ),
                    SourceDraft(
                        source_id="page:1:object:unreliable",
                        kind="page_object",
                        ordinal=1,
                        parent_id="page:1",
                        locator={"format": "pdf", "page": 1, "position_basis": "unknown"},
                    ),
                ],
            ),
            original=original,
            content_sha256=_sha(original),
            document_format="pdf",
        )
        fallback_evidence = _evidence(library, temp, soffice=soffice)
        fallback_mock = _vision(
            fallback_evidence,
            library,
            temp,
            cache_name="fallback-mock",
            enabled=True,
            api_key="",
            base_url="https://mock.invalid/api/v1",
            model=runtime.dashscope_vision_model,
            timeout=15,
            execution_kind="mock_protocol",
            transport=httpx.MockTransport(counted_mock),
        )
        fallback_request = _request(
            fallback,
            "page:1:object:unreliable",
            "object_crop",
            read_intent="directed_question",
            question="What is visible on the fallback page?",
        )
        body, elapsed = _call(app, holder, fallback_mock, fallback_request)
        scenarios.append(
            _compact(
                "pdf-unreliable-parent-fallback",
                fallback_request,
                body,
                elapsed,
                output_dir,
                provenance="mock_transport_fault_fixture",
            )
        )

        real_scenarios: list[dict[str, Any]] = []
        real_provider_calls = 0
        notebook_v1 = imported["notebook"]
        if args.real:
            real = _vision(
                evidence,
                library,
                temp,
                cache_name="real-vision",
                enabled=True,
                api_key=runtime.dashscope_api_key,
                base_url=runtime.dashscope_base_url,
                model=runtime.dashscope_vision_model,
                timeout=min(float(runtime.dashscope_timeout_seconds), 15.0),
            )
            real_requests = [
                (
                    "pptx-table-real-generic",
                    _request(
                        imported["pptx"],
                        "page:1:shape:3",
                        "object_crop",
                        read_intent="generic_transcription",
                    ),
                    "real_provider_public_pptx",
                ),
                ("pdf-right-real-directed", right_request, "real_provider_public_pdf"),
                (
                    "notebook-v1-output1-real-directed",
                    _request(
                        notebook_v1,
                        "cell:1:output:1",
                        "notebook_output",
                        read_intent="directed_question",
                        question="What exact x-axis label and unit are visible?",
                    ),
                    "real_provider_public_notebook",
                ),
            ]
            for name, request, provenance in real_requests:
                assert real_provider_calls < MAX_REAL_PROVIDER_CALLS
                body, elapsed = _call(app, holder, real, request)
                real_provider_calls += body["external_call_count"]
                item = _compact(name, request, body, elapsed, output_dir, provenance=provenance)
                real_scenarios.append(item)
                scenarios.append(item)

            notebook_path = paths["notebook"]
            shutil.copy2(SAMPLES / "revisions" / "v2" / "Signal-lab.ipynb", notebook_path)
            notebook_v2 = library.import_paths(
                COURSE_ID, [notebook_path], document_id=notebook_v1.document_id
            )[0]
            v2_request = _request(
                notebook_v2,
                "cell:1:output:2",
                "notebook_output",
                read_intent="directed_question",
                question="What exact x-axis label and unit are visible?",
            )
            assert real_provider_calls < MAX_REAL_PROVIDER_CALLS
            body, elapsed = _call(app, holder, real, v2_request)
            real_provider_calls += body["external_call_count"]
            item = _compact(
                "notebook-v2-output2-real-directed",
                v2_request,
                body,
                elapsed,
                output_dir,
                provenance="real_provider_public_notebook",
            )
            real_scenarios.append(item)
            scenarios.append(item)
            assert real_provider_calls <= MAX_REAL_PROVIDER_CALLS

            stale_request = _request(notebook_v1, "cell:1:output:2", "notebook_output")
            stale_body, elapsed = _call(app, holder, real, stale_request)
            scenarios.append(
                _compact(
                    "notebook-v1-current-after-v2",
                    stale_request,
                    stale_body,
                    elapsed,
                    output_dir,
                    provenance="current_version_guard",
                )
            )
            historical_request = stale_request.model_copy(
                update={"expected_publication_epoch": None}
            )
            historical = evidence.render_historical(
                historical_request,
                HistoricalSourceGrant(
                    source=historical_request.source,
                    course_ids=(COURSE_ID,),
                    content_sha256=notebook_v1.content_sha256,
                    message_id="public-d04-message",
                    evidence_id="public-d04-evidence",
                ),
            )
            historical_body = historical.model_dump(mode="json")
            scenarios.append(
                _compact(
                    "notebook-v1-internal-history",
                    historical_request,
                    historical_body,
                    0.0,
                    output_dir,
                    provenance="trusted_internal_historical_grant",
                )
            )

        ppt_key = SourceKey(
            document_id=imported["pptx"].document_id,
            version=imported["pptx"].version,
            source_id="page:1:shape:3",
        )
        enhancement_before_delete = _enhancement(library, ppt_key)
        assert library.delete(imported["pptx"].document_id)
        late_write = library.store.write_enhancement(
            ppt_key,
            course_ids=(COURSE_ID,),
            content_sha256=imported["pptx"].content_sha256,
            publication_epoch=imported["pptx"].publication_epoch,
            text="late-write-must-not-appear-d04",
        )

        race_path = course / "race.pdf"
        shutil.copy2(SAMPLES / "Signal-reference.pdf", race_path)
        race_import = library.import_paths(COURSE_ID, [race_path])[0]
        race_evidence = _evidence(library, temp, soffice=soffice)

        def race_handler(request: httpx.Request) -> httpx.Response:
            nonlocal mock_calls
            mock_calls += 1
            race_path.write_bytes(race_path.read_bytes() + b"\npublic-d04-race")
            changed = library.import_paths(
                COURSE_ID, [race_path], document_id=race_import.document_id
            )[0]
            assert changed.version > race_import.version
            return _mock_response(request)

        race_service = _vision(
            race_evidence,
            library,
            temp,
            cache_name="race-mock",
            enabled=True,
            api_key="",
            base_url="https://mock.invalid/api/v1",
            model=runtime.dashscope_vision_model,
            timeout=15,
            execution_kind="mock_protocol",
            transport=httpx.MockTransport(race_handler),
        )
        race_request = _request(
            race_import,
            "page:1",
            "original_page",
            read_intent="directed_question",
            question="What is the title?",
        )
        race_body, elapsed = _call(app, holder, race_service, race_request)
        scenarios.append(
            _compact(
                "pdf-version-race-redaction",
                race_request,
                race_body,
                elapsed,
                output_dir,
                provenance="mock_transport_version_race",
            )
        )

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            unused_port = probe.getsockname()[1]
        loopback = _vision(
            evidence,
            library,
            temp,
            cache_name="loopback-failure",
            enabled=True,
            api_key="public-test-only",
            base_url=f"http://127.0.0.1:{unused_port}/api/v1",
            model=runtime.dashscope_vision_model,
            timeout=0.2,
        )
        loop_body, elapsed = _call(app, holder, loopback, right_request)
        scenarios.append(
            _compact(
                "local-loopback-real-transport-failure",
                right_request,
                loop_body,
                elapsed,
                output_dir,
                provenance="local_loopback_no_supplier",
            )
        )

        full = next(item for item in scenarios if item["name"] == "pdf-page2-full")
        right_item = next(
            item
            for item in scenarios
            if item["name"] in ({"pdf-right-real-directed"} if args.real else {"pdf-right-mock"})
        )
        highlight = output_dir / "pdf-page2-right-highlight.png"
        _highlight(
            _resolve_artifact(full["display"]["artifact"], output_dir),
            right_item["display"]["rect_visible_norm"],
            highlight,
        )

        summary = {
            "schema": "srag-d04-public-v1",
            "workspace": {
                "head": _git("rev-parse", "HEAD"),
                "d03_base": D03_BASE,
                "d_code_diff_from_base": _git(
                    "diff",
                    "--name-only",
                    f"{D03_BASE}..HEAD",
                    "--",
                    "app/rag/source_evidence.py",
                    "app/rag/source_vision.py",
                    "app/api/source_evidence.py",
                    "app/config.py",
                    "app/main.py",
                    "app/stores/local_library.py",
                ).splitlines(),
                "hashes": {
                    path: _file_sha(ROOT / path)
                    for path in (
                        "app/rag/source_evidence.py",
                        "app/rag/source_vision.py",
                        "app/api/source_evidence.py",
                        "app/config.py",
                        "app/main.py",
                        "app/stores/local_library.py",
                    )
                },
            },
            "runtime": {
                "real_mode": args.real,
                "credential_configured": bool(runtime.dashscope_api_key),
                "provider_model": runtime.dashscope_vision_model if args.real else None,
                "timeout_seconds": min(float(runtime.dashscope_timeout_seconds), 15.0),
                "soffice_configured": bool(soffice),
            },
            "artifact_paths": {
                "reference_base": (
                    "repository_root" if output_dir.is_relative_to(ROOT) else "artifact_root"
                ),
                "artifact_root": (
                    output_dir.relative_to(ROOT).as_posix()
                    if output_dir.is_relative_to(ROOT)
                    else str(output_dir)
                ),
            },
            "originals": {
                label: {
                    "repo_path": (SAMPLES / path.name).relative_to(ROOT).as_posix(),
                    "document_id": imported[label].document_id,
                    "version": imported[label].version,
                    "publication_epoch": imported[label].publication_epoch,
                    "content_sha256": imported[label].content_sha256,
                }
                for label, path in paths.items()
            },
            "counts": {
                "authorized_real_provider_limit": MAX_REAL_PROVIDER_CALLS,
                "real_supplier_external_calls": real_provider_calls,
                "real_supplier_success_or_partial": sum(
                    item["read_status"] in {"succeeded", "partial"} for item in real_scenarios
                ),
                "real_supplier_failed": sum(
                    item["read_status"] == "failed" for item in real_scenarios
                ),
                "mock_transport_handler_calls": mock_calls,
                "local_loopback_connection_attempts": loop_body["external_call_count"],
                "display_only_unread": sum(
                    item["provenance"] == "display_only" and item["read_status"] == "not_attempted"
                    for item in scenarios
                ),
            },
            "checks": {
                "mock_cache_reused": next(
                    item for item in scenarios if item["name"] == "pdf-right-mock-cache"
                )["reads"][0]["execution"]
                == "cache_reuse",
                "region_change_invalidated_input": next(
                    item for item in scenarios if item["name"] == "pdf-right-mock"
                )["display"]["byte_sha256"]
                != next(item for item in scenarios if item["name"] == "pdf-right-region-rev2")[
                    "display"
                ]["byte_sha256"],
                "fallback_actual_is_parent": next(
                    item for item in scenarios if item["name"] == "pdf-unreliable-parent-fallback"
                )["display"]["actual_source"]["source_id"]
                == "page:1",
                "race_redacted": race_body["status"] == "stale_reference"
                and race_body["display"] is None
                and race_body["reads"] == [],
                "late_write_rejected": late_write is False,
                "historical_v1_available": (
                    not args.real
                    or next(
                        item for item in scenarios if item["name"] == "notebook-v1-internal-history"
                    )["status"]
                    == "available"
                ),
                "real_generic_fts_origin": (
                    not args.real
                    or (
                        enhancement_before_delete is not None
                        and enhancement_before_delete["origin"] == "enhancement"
                    )
                ),
            },
            "enhancement_before_delete": enhancement_before_delete,
            "late_write_after_delete": late_write,
            "highlight": {
                "artifact": _artifact_reference(highlight, output_dir),
                "rect_visible_norm": right_item["display"]["rect_visible_norm"],
                "sha256": _file_sha(highlight),
            },
            "scenarios": scenarios,
            "provenance_note": (
                "Only provenance values prefixed real_provider_public_* may contact the configured "
                "supplier. mock_transport, local_loopback, display_only and internal grant "
                "scenarios are not supplier model reads."
            ),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            summary_path.relative_to(ROOT)
            if summary_path.is_relative_to(ROOT)
            else summary_path
        )
        print(json.dumps(summary["counts"], ensure_ascii=False, sort_keys=True))
        print(json.dumps(summary["checks"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
