"""Authorized, version-exact rendering of archived local source bytes."""

from __future__ import annotations

import base64
import binascii
import hashlib
import importlib.metadata
import io
import json
import os
import shutil
import subprocess
import tempfile
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.rag.parsers.local_slides import _PDFIUM_LOCK
from app.schemas.local import SourceDraft, SourceKey
from app.stores.local_library import HistoricalSource

EvidenceStatus = Literal[
    "available",
    "partial",
    "original_unavailable",
    "not_accessible",
    "stale_reference",
    "rendition_unavailable",
    "image_unavailable",
    "failed",
]
RenditionKind = Literal["original_page", "object_crop", "notebook_output"]
CoordinateFrame = Literal["visible_page_norm_v1", "pdf_page_units_bottom_left_v1"]

_HASH_PATTERN = r"^[0-9a-f]{64}$"
_DIRECT_RASTER_TYPES = {"image/png", "image/jpeg", "image/webp"}
_TRANSCODE_RASTER_TYPES = {"image/bmp", "image/tiff"}
_IMAGE_PRIORITY = (
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "image/svg+xml",
)
_MAX_IMAGE_PIXELS = 50_000_000


class SourceReader(Protocol):
    def read_source(self, key: SourceKey, *, course_ids: Sequence[str]) -> HistoricalSource: ...


class EvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: SourceKey
    expected_content_sha256: str = Field(pattern=_HASH_PATTERN)
    expected_publication_epoch: int | None = Field(default=None, ge=0)
    rendition: RenditionKind
    preferred_mime: str | None = None
    registered_region_id: str | None = Field(default=None, min_length=1, max_length=160)
    read_intent: Literal["display_only", "generic_transcription", "directed_question"] = (
        "display_only"
    )
    question: str | None = Field(default=None, max_length=4000)

    @field_validator("source", mode="before")
    @classmethod
    def source_has_only_public_key_fields(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and set(value) != {"document_id", "version", "source_id"}:
            raise ValueError("source only accepts document_id, version, and source_id")
        return value

    @model_validator(mode="after")
    def directed_question_requires_question(self) -> EvidenceRequest:
        if self.read_intent == "directed_question":
            if not self.question or not self.question.strip():
                raise ValueError("directed_question requires a non-empty question")
            self.question = self.question.strip()
        elif self.question is not None:
            raise ValueError("question is only accepted for directed_question")
        return self


class RegionComponent(BaseModel):
    role: Literal["target", "legend", "x_axis", "y_axis", "caption", "table_header"]
    rect: tuple[float, float, float, float]
    coordinate_frame: CoordinateFrame


class RegisteredRegion(BaseModel):
    region_id: str = Field(min_length=1, max_length=160)
    source: SourceKey
    content_sha256: str = Field(pattern=_HASH_PATTERN)
    coordinate_frame: CoordinateFrame
    target_rect: tuple[float, float, float, float]
    components: list[RegionComponent] = Field(default_factory=list)
    provenance: Literal["published_locator", "trusted_corpus_registration"]
    revision: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_rectangles(self) -> RegisteredRegion:
        _validate_rect(self.target_rect, self.coordinate_frame)
        for component in self.components:
            _validate_rect(component.rect, component.coordinate_frame)
        return self


class RegionRegistry:
    """Backend-owned region records; public requests may only reference an existing id."""

    def __init__(self, regions: Sequence[RegisteredRegion] = ()) -> None:
        self._regions: dict[str, RegisteredRegion] = {}
        for region in regions:
            self.register(region)

    def register(self, region: RegisteredRegion) -> None:
        if region.region_id in self._regions:
            raise ValueError(f"duplicate registered region: {region.region_id}")
        self._regions[region.region_id] = region

    def get(self, region_id: str) -> RegisteredRegion | None:
        return self._regions.get(region_id)


@dataclass(frozen=True)
class HistoricalSourceGrant:
    """Server-created proof that a saved message reference may read one historical source."""

    source: SourceKey
    course_ids: tuple[str, ...]
    content_sha256: str
    message_id: str
    evidence_id: str


class DisplayRecord(BaseModel):
    display_source: SourceKey
    actual_source: SourceKey
    rendition_kind: Literal[
        "original_page", "object_crop", "notebook_output", "parent_page_fallback"
    ]
    media_type: Literal["image/png", "image/jpeg", "image/webp"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    byte_length: int = Field(gt=0)
    byte_sha256: str = Field(pattern=_HASH_PATTERN)
    data_base64: str
    rect_visible_norm: tuple[float, float, float, float] | None = None
    renderer: str
    renderer_version: str
    cache_hit: bool
    fallback_reason: str | None = None


class NativeFallback(BaseModel):
    source: SourceKey
    kind: str
    title: str
    text: str
    locator: dict[str, Any]
    state: str


class ReadRecord(BaseModel):
    requested_source: SourceKey
    actual_source: SourceKey
    actual_input_sha256: str = Field(pattern=_HASH_PATTERN)
    actual_input_media_type: str
    actual_input_rect: tuple[float, float, float, float] | None = None
    method: Literal["vision"] = "vision"
    purpose: Literal["generic_transcription", "directed_question"]
    execution: Literal["real_provider", "cache_reuse", "mock_protocol"]
    origin_execution: Literal["real_provider", "mock_protocol"]
    provider: str
    model: str
    prompt_version: str
    question_sha256: str | None = None
    status: Literal["succeeded", "partial", "failed"]
    text: str = ""
    missing_or_error: list[str] = Field(default_factory=list)
    cache_hit: bool = False
    external_call: bool = False
    request_id: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)


class EvidenceResult(BaseModel):
    status: EvidenceStatus
    requested_source: SourceKey
    content_sha256: str | None = None
    display: DisplayRecord | None = None
    read_status: Literal["not_attempted", "succeeded", "partial", "failed"] = "not_attempted"
    capability_reason: (
        Literal[
            "not_requested",
            "provider_disabled",
            "provider_not_configured",
            "input_unsupported",
            "mock_only",
        ]
        | None
    ) = "not_requested"
    reads: list[ReadRecord] = Field(default_factory=list)
    external_call_count: int = Field(default=0, ge=0)
    native_fallback: NativeFallback | None = None
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _Access:
    course_ids: tuple[str, ...]
    current: bool
    epoch: int | None


@dataclass(frozen=True)
class _Target:
    page: int | None
    display_source: SourceKey
    rect: tuple[float, float, float, float] | None
    rendition_kind: Literal[
        "original_page", "object_crop", "notebook_output", "parent_page_fallback"
    ]
    fallback_reason: str | None
    page_source: SourceDraft | None
    region_revision: int | None = None
    region_digest: str | None = None


@dataclass(frozen=True)
class _Artifact:
    data: bytes
    media_type: Literal["image/png", "image/jpeg", "image/webp"]
    width: int
    height: int
    renderer: str
    renderer_version: str
    cache_hit: bool = False


@dataclass
class _FlightLock:
    lock: Lock
    users: int = 0


_FLIGHT_GUARD = Lock()
_FLIGHTS: dict[tuple[str, str], _FlightLock] = {}


class SourceEvidenceService:
    def __init__(
        self,
        store: SourceReader,
        *,
        cache_dir: Path,
        current_course_ids: Sequence[str],
        soffice_path: Path | None = None,
        max_inline_bytes: int = 8 * 1024 * 1024,
        regions: RegionRegistry | None = None,
        render_dpi: int = 144,
    ) -> None:
        self.store = store
        self.cache_dir = cache_dir
        self.current_course_ids = tuple(
            dict.fromkeys(item.strip() for item in current_course_ids if item.strip())
        )
        self.soffice_path = soffice_path
        self.max_inline_bytes = max_inline_bytes
        self.regions = regions or RegionRegistry()
        self.render_dpi = render_dpi
        self._flight_namespace = str(cache_dir.expanduser().resolve())
        if max_inline_bytes <= 0:
            raise ValueError("max_inline_bytes must be positive")
        if render_dpi <= 0:
            raise ValueError("render_dpi must be positive")

    def render_current(self, request: EvidenceRequest) -> EvidenceResult:
        if request.expected_publication_epoch is None:
            return self._result("stale_reference", request, warnings=["publication_epoch_required"])
        if not self.current_course_ids:
            return self._result("not_accessible", request)
        return self._render(
            request,
            _Access(self.current_course_ids, True, request.expected_publication_epoch),
        )

    def is_current_authorized(
        self,
        key: SourceKey,
        *,
        content_sha256: str,
        publication_epoch: int,
    ) -> bool:
        request = EvidenceRequest(
            source=key,
            expected_content_sha256=content_sha256,
            expected_publication_epoch=publication_epoch,
            rendition="original_page",
        )
        snapshot, failure = self._authorized_snapshot(
            request, _Access(self.current_course_ids, True, publication_epoch)
        )
        return snapshot is not None and failure is None

    def render_historical(
        self, request: EvidenceRequest, grant: HistoricalSourceGrant
    ) -> EvidenceResult:
        if (
            grant.source != request.source
            or grant.content_sha256 != request.expected_content_sha256
            or not grant.course_ids
            or not grant.message_id
            or not grant.evidence_id
        ):
            return self._result("not_accessible", request)
        return self._render(request, _Access(grant.course_ids, False, None))

    def _render(self, request: EvidenceRequest, access: _Access) -> EvidenceResult:
        snapshot, failure = self._authorized_snapshot(request, access)
        if failure is not None:
            return failure
        assert (
            snapshot is not None and snapshot.source is not None and snapshot.original is not None
        )

        try:
            target = self._target(request, snapshot, access)
            artifact = self._artifact(request, snapshot, target)
        except _EvidenceFailure as exc:
            return self._failure_after_recheck(request, access, snapshot, exc.status, exc.reason)
        except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            return self._failure_after_recheck(
                request,
                access,
                snapshot,
                "failed",
                f"render_failed:{type(exc).__name__}",
            )

        if len(artifact.data) > self.max_inline_bytes:
            return self._failure_after_recheck(
                request,
                access,
                snapshot,
                "rendition_unavailable",
                "inline_limit_exceeded",
            )

        _, final_failure = self._authorized_snapshot(request, access)
        if final_failure is not None:
            return final_failure
        digest = hashlib.sha256(artifact.data).hexdigest()
        return EvidenceResult(
            status="available",
            requested_source=request.source,
            content_sha256=snapshot.content_sha256,
            display=DisplayRecord(
                display_source=target.display_source,
                actual_source=target.display_source,
                rendition_kind=target.rendition_kind,
                media_type=artifact.media_type,
                width=artifact.width,
                height=artifact.height,
                byte_length=len(artifact.data),
                byte_sha256=digest,
                data_base64=base64.b64encode(artifact.data).decode("ascii"),
                rect_visible_norm=target.rect,
                renderer=artifact.renderer,
                renderer_version=artifact.renderer_version,
                cache_hit=artifact.cache_hit,
                fallback_reason=target.fallback_reason,
            ),
        )

    def _authorized_snapshot(
        self, request: EvidenceRequest, access: _Access
    ) -> tuple[HistoricalSource | None, EvidenceResult | None]:
        snapshot = self.store.read_source(request.source, course_ids=access.course_ids)
        if snapshot.status == "not_accessible":
            return None, self._result("not_accessible", request)
        if snapshot.content_sha256 != request.expected_content_sha256:
            status: EvidenceStatus = "stale_reference" if access.current else "not_accessible"
            return None, self._result(status, request)
        if access.current and (
            not snapshot.is_current or snapshot.publication_epoch != access.epoch
        ):
            return None, self._result("stale_reference", request)
        if snapshot.status == "original_unavailable" or snapshot.original is None:
            return None, self._result(
                "original_unavailable",
                request,
                snapshot.content_sha256,
                native_fallback=_native_fallback(request.source, snapshot.source),
            )
        if hashlib.sha256(snapshot.original).hexdigest() != snapshot.content_sha256:
            return None, self._result(
                "original_unavailable",
                request,
                snapshot.content_sha256,
                native_fallback=_native_fallback(request.source, snapshot.source),
                warnings=["archived_original_hash_mismatch"],
            )
        return snapshot, None

    def _failure_after_recheck(
        self,
        request: EvidenceRequest,
        access: _Access,
        snapshot: HistoricalSource,
        status: EvidenceStatus,
        reason: str,
    ) -> EvidenceResult:
        _, final_failure = self._authorized_snapshot(request, access)
        if final_failure is not None:
            return final_failure
        return self._result(
            status,
            request,
            snapshot.content_sha256,
            native_fallback=_native_fallback(request.source, snapshot.source),
            warnings=[reason],
        )

    def _target(
        self, request: EvidenceRequest, snapshot: HistoricalSource, access: _Access
    ) -> _Target:
        source = snapshot.source
        assert source is not None
        if request.rendition == "notebook_output":
            if source.kind != "notebook_output":
                raise _EvidenceFailure("image_unavailable", "source_is_not_notebook_output")
            return _Target(None, request.source, None, "notebook_output", None, None)

        fmt = source.locator.get("format")
        if fmt not in {"pdf", "pptx"}:
            raise _EvidenceFailure("rendition_unavailable", "source_has_no_page_rendition")
        page_source = source
        page_key = request.source
        visited = {source.source_id}
        while page_source.kind == "page_object":
            if not page_source.parent_id or page_source.parent_id in visited:
                raise _EvidenceFailure("rendition_unavailable", "invalid_page_ancestor_chain")
            visited.add(page_source.parent_id)
            page_key = request.source.model_copy(update={"source_id": page_source.parent_id})
            parent = self.store.read_source(page_key, course_ids=access.course_ids)
            if (
                parent.status != "available"
                or parent.content_sha256 != snapshot.content_sha256
                or parent.source is None
                or parent.original is None
                or (
                    access.current
                    and (not parent.is_current or parent.publication_epoch != access.epoch)
                )
            ):
                raise _EvidenceFailure("stale_reference", "parent_page_not_available")
            page_source = parent.source
        if page_source.kind != "page":
            raise _EvidenceFailure("rendition_unavailable", "ancestor_is_not_page")

        page = page_source.locator.get("page")
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            raise _EvidenceFailure("rendition_unavailable", "invalid_page_locator")
        source_page = source.locator.get("page")
        if source.kind == "page_object" and source_page != page:
            raise _EvidenceFailure("rendition_unavailable", "cross_page_object_unsupported")
        if request.rendition == "original_page":
            return _Target(page, page_key, None, "original_page", None, page_source)

        region = None
        if request.registered_region_id:
            region = self.regions.get(request.registered_region_id)
            if (
                region is None
                or region.source != request.source
                or region.content_sha256 != snapshot.content_sha256
            ):
                raise _EvidenceFailure("not_accessible", "registered_region_not_available")
        rect = (
            self._region_rect(region, page_source)
            if region
            else self._source_rect(source, page_source)
        )
        if rect is None:
            return _Target(
                page,
                page_key,
                None,
                "parent_page_fallback",
                "unreliable_bbox",
                page_source,
            )
        return _Target(
            page,
            request.source,
            _pad_rect(rect, 0.02),
            "object_crop",
            None,
            page_source,
            region.revision if region else None,
            _digest(region.model_dump(mode="json")) if region else None,
        )

    def _region_rect(
        self, region: RegisteredRegion, page_source: SourceDraft
    ) -> tuple[float, float, float, float]:
        rectangles = [
            _to_visible(region.target_rect, region.coordinate_frame, page_source.locator),
            *[
                _to_visible(component.rect, component.coordinate_frame, page_source.locator)
                for component in region.components
            ],
        ]
        return (
            min(rect[0] for rect in rectangles),
            min(rect[1] for rect in rectangles),
            max(rect[2] for rect in rectangles),
            max(rect[3] for rect in rectangles),
        )

    @staticmethod
    def _source_rect(
        source: SourceDraft, page_source: SourceDraft
    ) -> tuple[float, float, float, float] | None:
        locator = source.locator
        if source.kind != "page_object":
            return None
        if (
            locator.get("format") == "pptx"
            and locator.get("position_basis") != "native_shape_extents_v1"
        ):
            return None
        bbox = locator.get("bbox")
        if isinstance(bbox, list) and locator.get("coordinate_frame") == "visible_page_norm_v1":
            rect = tuple(bbox)
            try:
                _validate_rect(rect, "visible_page_norm_v1")
            except ValueError:
                return None
            return rect  # type: ignore[return-value]
        native = locator.get("native_bbox")
        if (
            isinstance(native, list)
            and locator.get("native_coordinate_frame") == "pdf_page_units_bottom_left_v1"
        ):
            try:
                return _to_visible(
                    tuple(native), "pdf_page_units_bottom_left_v1", page_source.locator
                )
            except ValueError:
                return None
        return None

    def _artifact(
        self, request: EvidenceRequest, snapshot: HistoricalSource, target: _Target
    ) -> _Artifact:
        assert snapshot.original is not None and snapshot.source is not None
        if target.rendition_kind == "notebook_output":
            renderer, version = _notebook_renderer(snapshot.source, request.preferred_mime)
            key = self._cache_key(request, snapshot, target, renderer, version)
            return self._singleflight_artifact(
                key,
                lambda: _artifact_from_notebook(
                    snapshot.original, snapshot.source, request.preferred_mime
                ),
            )
        fmt = target.page_source.locator.get("format") if target.page_source else None
        if fmt == "pdf":
            renderer, version = "pypdfium2", _package_version("pypdfium2")
            key = self._cache_key(request, snapshot, target, renderer, version)
            return self._singleflight_artifact(
                key,
                lambda: _artifact_from_pdf(
                    snapshot.original, target.page or 0, self.render_dpi, target.rect
                ),
            )
        if fmt == "pptx":
            soffice, lo_version = _soffice(self.soffice_path)
            if soffice is None:
                raise _EvidenceFailure("rendition_unavailable", "converter_unavailable")
            renderer = "libreoffice+pypdfium2"
            version = f"{lo_version}|pypdfium2-{_package_version('pypdfium2')}"
            key = self._cache_key(request, snapshot, target, renderer, version)
            return self._singleflight_artifact(
                key,
                lambda: _artifact_from_pptx(
                    snapshot.original,
                    soffice,
                    target,
                    self.render_dpi,
                    renderer,
                    version,
                ),
            )
        raise _EvidenceFailure("rendition_unavailable", "unsupported_document_format")

    def _singleflight_artifact(self, key: str, produce: Callable[[], _Artifact]) -> _Artifact:
        cached = self._read_cache(key)
        if cached is not None:
            return cached
        with self._singleflight(key):
            cached = self._read_cache(key)
            if cached is not None:
                return cached
            artifact = produce()
            self._write_cache(key, artifact)
            return artifact

    @contextmanager
    def _singleflight(self, key: str):
        flight_key = (self._flight_namespace, key)
        with _FLIGHT_GUARD:
            flight = _FLIGHTS.get(flight_key)
            if flight is None:
                flight = _FlightLock(Lock())
                _FLIGHTS[flight_key] = flight
            flight.users += 1
        flight.lock.acquire()
        try:
            yield
        finally:
            flight.lock.release()
            with _FLIGHT_GUARD:
                flight.users -= 1
                if flight.users == 0 and _FLIGHTS.get(flight_key) is flight:
                    del _FLIGHTS[flight_key]

    def _cache_key(
        self,
        request: EvidenceRequest,
        snapshot: HistoricalSource,
        target: _Target,
        renderer: str,
        version: str,
    ) -> str:
        source = snapshot.source
        assert source is not None
        return _digest(
            {
                "schema": "source-rendition-v1",
                "source": request.source.model_dump(mode="json"),
                "content_sha256": snapshot.content_sha256,
                "locator": source.locator,
                "rendition": target.rendition_kind,
                "page": target.page,
                "rect": target.rect,
                "region_revision": target.region_revision,
                "region_digest": target.region_digest,
                "preferred_mime": request.preferred_mime,
                "renderer": renderer,
                "renderer_version": version,
                "dpi": self.render_dpi,
            }
        )

    def _read_cache(self, key: str) -> _Artifact | None:
        path = self.cache_dir / f"{key}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            data = base64.b64decode(payload["data_base64"], validate=True)
            if (
                len(data) > self.max_inline_bytes
                or hashlib.sha256(data).hexdigest() != payload["sha256"]
            ):
                raise ValueError("invalid cached artifact")
            width, height = _inspect_raster(data, payload["media_type"])
            if (width, height) != (payload["width"], payload["height"]):
                raise ValueError("cached dimensions differ")
            return _Artifact(
                data,
                payload["media_type"],
                width,
                height,
                payload["renderer"],
                payload["renderer_version"],
                True,
            )
        except FileNotFoundError:
            return None
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, binascii.Error, OSError):
            path.unlink(missing_ok=True)
            return None

    def _write_cache(self, key: str, artifact: _Artifact) -> None:
        if len(artifact.data) > self.max_inline_bytes:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "media_type": artifact.media_type,
            "width": artifact.width,
            "height": artifact.height,
            "sha256": hashlib.sha256(artifact.data).hexdigest(),
            "renderer": artifact.renderer,
            "renderer_version": artifact.renderer_version,
            "data_base64": base64.b64encode(artifact.data).decode("ascii"),
        }
        fd, temporary = tempfile.mkstemp(prefix=f".{key}.", suffix=".tmp", dir=self.cache_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(
                    payload, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                stream.flush()
                os.fsync(stream.fileno())
            Path(temporary).replace(self.cache_dir / f"{key}.json")
        finally:
            Path(temporary).unlink(missing_ok=True)

    @staticmethod
    def _result(
        status: EvidenceStatus,
        request: EvidenceRequest,
        content_sha256: str | None = None,
        *,
        native_fallback: NativeFallback | None = None,
        warnings: list[str] | None = None,
    ) -> EvidenceResult:
        return EvidenceResult(
            status=status,
            requested_source=request.source,
            content_sha256=content_sha256,
            native_fallback=native_fallback,
            warnings=warnings or [],
        )


class _EvidenceFailure(Exception):
    def __init__(self, status: EvidenceStatus, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _native_fallback(key: SourceKey, source: SourceDraft | None) -> NativeFallback | None:
    if source is None:
        return None
    safe_keys = {
        "format",
        "page",
        "cell_index",
        "cell_id",
        "output_index",
        "section_path",
        "object_type",
        "columns",
        "rows",
        "content_state",
        "outputs_state",
    }
    locator = {name: value for name, value in source.locator.items() if name in safe_keys}
    return NativeFallback(
        source=key,
        kind=source.kind,
        title=source.title[:500],
        text=source.text[:4000],
        locator=locator,
        state=source.state,
    )


def _validate_rect(rect: Sequence[float], frame: CoordinateFrame) -> None:
    if len(rect) != 4 or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) for value in rect
    ):
        raise ValueError("region rectangle must contain four finite numbers")
    left, top_or_bottom, right, bottom_or_top = (float(value) for value in rect)
    if not all(
        value == value and abs(value) != float("inf")
        for value in (left, top_or_bottom, right, bottom_or_top)
    ):
        raise ValueError("region rectangle contains non-finite values")
    if right <= left or bottom_or_top <= top_or_bottom:
        raise ValueError("region rectangle is degenerate")
    if frame == "visible_page_norm_v1" and not (
        0 <= left < right <= 1 and 0 <= top_or_bottom < bottom_or_top <= 1
    ):
        raise ValueError("normalized region is outside the visible page")


def _to_visible(
    rect: Sequence[float], frame: CoordinateFrame, page_locator: Mapping[str, Any]
) -> tuple[float, float, float, float]:
    _validate_rect(rect, frame)
    if frame == "visible_page_norm_v1":
        return tuple(float(value) for value in rect)  # type: ignore[return-value]
    crop = page_locator.get("visible_page_bbox_pdf_units")
    rotation = page_locator.get("rotation_degrees_clockwise")
    if not isinstance(crop, list) or len(crop) != 4 or rotation not in {0, 90, 180, 270}:
        raise ValueError("PDF page transform is unavailable")
    return _pdf_rect_to_visible(tuple(float(value) for value in rect), tuple(crop), rotation)


def _pdf_rect_to_visible(
    rect: tuple[float, float, float, float],
    crop: tuple[float, float, float, float],
    rotation: int,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = crop
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0 or rotation not in {0, 90, 180, 270}:
        raise ValueError("invalid PDF crop or rotation")
    left, bottom, right, top = rect
    left, right = max(left, x0), min(right, x1)
    bottom, top = max(bottom, y0), min(top, y1)
    if right <= left or top <= bottom:
        raise ValueError("native rectangle is outside the crop box")

    def point(x_value: float, y_value: float) -> tuple[float, float]:
        x_screen, y_screen = x_value - x0, y1 - y_value
        if rotation == 0:
            return x_screen / width, y_screen / height
        if rotation == 90:
            return (height - y_screen) / height, x_screen / width
        if rotation == 180:
            return (width - x_screen) / width, (height - y_screen) / height
        return y_screen / height, (width - x_screen) / width

    points = [point(x_value, y_value) for x_value in (left, right) for y_value in (bottom, top)]
    return (
        min(item[0] for item in points),
        min(item[1] for item in points),
        max(item[0] for item in points),
        max(item[1] for item in points),
    )


def _pad_rect(
    rect: tuple[float, float, float, float], padding: float
) -> tuple[float, float, float, float]:
    return (
        max(0.0, rect[0] - padding),
        max(0.0, rect[1] - padding),
        min(1.0, rect[2] + padding),
        min(1.0, rect[3] + padding),
    )


def _render_pdf_page(
    pdf_bytes: bytes,
    page_number: int,
    dpi: int,
    rect: tuple[float, float, float, float] | None,
) -> tuple[bytes, int, int]:
    if page_number < 1:
        raise ValueError("page number must be positive")
    _pillow("rendition_unavailable")
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("PDF rendering requires pypdfium2") from exc
    with tempfile.TemporaryDirectory(prefix="srag-source-pdf-") as directory:
        path = Path(directory) / "source.pdf"
        path.write_bytes(pdf_bytes)
        with _PDFIUM_LOCK:
            document = pdfium.PdfDocument(path)
            try:
                if page_number > len(document):
                    raise ValueError("page is outside the archived document")
                page = document[page_number - 1]
                try:
                    bitmap = page.render(scale=dpi / 72)
                    try:
                        image = bitmap.to_pil().convert("RGB")
                        image.load()
                    finally:
                        close = getattr(bitmap, "close", None)
                        if close:
                            close()
                finally:
                    page.close()
            finally:
                document.close()
    if rect is not None:
        left = int(rect[0] * image.width)
        top = int(rect[1] * image.height)
        right = min(image.width, max(left + 1, int(rect[2] * image.width + 0.999999)))
        bottom = min(image.height, max(top + 1, int(rect[3] * image.height + 0.999999)))
        image = image.crop((left, top, right, bottom))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue(), image.width, image.height


def _crop_png(data: bytes, rect: tuple[float, float, float, float]) -> tuple[bytes, int, int]:
    image_module = _pillow("rendition_unavailable")
    with image_module.open(io.BytesIO(data)) as source:
        image = source.convert("RGB")
    left = int(rect[0] * image.width)
    top = int(rect[1] * image.height)
    right = min(image.width, max(left + 1, int(rect[2] * image.width + 0.999999)))
    bottom = min(image.height, max(top + 1, int(rect[3] * image.height + 0.999999)))
    image = image.crop((left, top, right, bottom))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue(), image.width, image.height


def _artifact_from_pdf(
    original: bytes,
    page: int,
    dpi: int,
    rect: tuple[float, float, float, float] | None,
) -> _Artifact:
    data, width, height = _render_pdf_page(original, page, dpi, rect)
    return _Artifact(
        data,
        "image/png",
        width,
        height,
        "pypdfium2",
        _package_version("pypdfium2"),
    )


def _artifact_from_pptx(
    original: bytes,
    soffice: Path,
    target: _Target,
    dpi: int,
    renderer: str,
    version: str,
) -> _Artifact:
    _pillow("rendition_unavailable")
    pdf = _convert_pptx(original, soffice)
    _verify_pptx_page_count(original, pdf)
    page_data, page_width, page_height = _render_pdf_page(pdf, target.page or 0, dpi, None)
    _verify_slide_aspect(target.page_source, page_width, page_height)
    if target.rect:
        data, width, height = _crop_png(page_data, target.rect)
    else:
        data, width, height = page_data, page_width, page_height
    return _Artifact(data, "image/png", width, height, renderer, version)


def _soffice(configured: Path | None) -> tuple[Path | None, str]:
    candidates: list[Path] = []
    if configured:
        candidates.append(configured.expanduser())
    located = shutil.which("soffice")
    if located:
        candidates.append(Path(located))
    candidates.extend(
        [
            Path("/Applications/LibreOffice.app/Contents/MacOS/soffice"),
            Path("/usr/lib/libreoffice/program/soffice"),
        ]
    )
    for candidate in candidates:
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        try:
            completed = subprocess.run(
                [str(candidate), "--version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        version = (completed.stdout or completed.stderr).strip() or "unknown"
        return candidate, version
    return None, "unavailable"


def _convert_pptx(original: bytes, soffice: Path) -> bytes:
    with tempfile.TemporaryDirectory(prefix="srag-source-pptx-") as directory:
        root = Path(directory)
        source = root / "source.pptx"
        output = root / "out"
        profile = root / "profile"
        output.mkdir()
        profile.mkdir()
        source.write_bytes(original)
        command = [
            str(soffice),
            "--headless",
            "--nologo",
            "--nodefault",
            "--nolockcheck",
            "--nofirststartwizard",
            f"-env:UserInstallation={profile.resolve().as_uri()}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(output),
            str(source),
        ]
        try:
            completed = subprocess.run(command, check=False, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired as exc:
            raise _EvidenceFailure("rendition_unavailable", "converter_timeout") from exc
        converted = output / "source.pdf"
        if completed.returncode != 0 or not converted.is_file() or not converted.stat().st_size:
            raise _EvidenceFailure("rendition_unavailable", "converter_failed")
        return converted.read_bytes()


def _verify_pptx_page_count(pptx: bytes, pdf: bytes) -> None:
    import re
    import zipfile

    with zipfile.ZipFile(io.BytesIO(pptx)) as archive:
        slides = [
            name
            for name in archive.namelist()
            if re.fullmatch(r"ppt/slides/slide[1-9][0-9]*\.xml", name)
        ]
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PDF rendering requires pypdfium2") from exc
    with _PDFIUM_LOCK:
        document = pdfium.PdfDocument(pdf)
        try:
            pages = len(document)
        finally:
            document.close()
    if not slides or pages != len(slides):
        raise _EvidenceFailure("rendition_unavailable", "converted_page_count_mismatch")


def _verify_slide_aspect(source: SourceDraft | None, width: int, height: int) -> None:
    if source is None:
        return
    size = source.locator.get("slide_size_emu")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or not all(isinstance(item, int) for item in size)
    ):
        if source.kind == "page_object":
            raise _EvidenceFailure("rendition_unavailable", "slide_dimensions_unavailable")
        return
    expected = size[0] / size[1]
    actual = width / height
    if abs(actual - expected) / expected > 0.02:
        raise _EvidenceFailure("rendition_unavailable", "converted_aspect_mismatch")


def _notebook_record(source: SourceDraft, preferred_mime: str | None) -> dict[str, Any]:
    representations = source.locator.get("representations")
    if not isinstance(representations, list):
        raise _EvidenceFailure("image_unavailable", "output_has_no_representations")
    records = {
        item.get("mime_type"): item
        for item in representations
        if isinstance(item, dict) and item.get("status") == "available"
    }
    if preferred_mime:
        record = records.get(preferred_mime)
        if record is None:
            raise _EvidenceFailure("image_unavailable", "preferred_mime_unavailable")
    else:
        record = next((records[mime] for mime in _IMAGE_PRIORITY if mime in records), None)
        if record is None:
            raise _EvidenceFailure("image_unavailable", "output_has_no_available_image")
    return record


def _notebook_renderer(source: SourceDraft, preferred_mime: str | None) -> tuple[str, str]:
    mime = _notebook_record(source, preferred_mime).get("mime_type")
    if mime in _TRANSCODE_RASTER_TYPES:
        return "pillow-rasterize", _package_version("Pillow")
    if mime in _DIRECT_RASTER_TYPES:
        return "notebook-original", "v1"
    if mime == "image/svg+xml":
        return "svg-unavailable", "v1"
    raise _EvidenceFailure("image_unavailable", "unsupported_image_representation")


def _notebook_image(
    original: bytes, source: SourceDraft, preferred_mime: str | None
) -> tuple[bytes, Literal["image/png", "image/jpeg", "image/webp"], int, int, str, str]:
    notebook = json.loads(original.decode("utf-8"))
    record = _notebook_record(source, preferred_mime)
    mime = record.get("mime_type")
    pointer = record.get("json_pointer")
    if not isinstance(pointer, str) or mime not in {
        *_DIRECT_RASTER_TYPES,
        *_TRANSCODE_RASTER_TYPES,
        "image/svg+xml",
    }:
        raise _EvidenceFailure("image_unavailable", "unsupported_image_representation")
    value = _json_pointer(notebook, pointer)
    if mime == "image/svg+xml":
        raw = _coerce_text(value).encode("utf-8")
    else:
        try:
            raw = base64.b64decode("".join(_coerce_text(value).split()), validate=True)
        except (ValueError, binascii.Error) as exc:
            raise _EvidenceFailure("image_unavailable", "invalid_image_data") from exc
    if len(raw) != record.get("byte_length") or hashlib.sha256(raw).hexdigest() != record.get(
        "sha256"
    ):
        raise _EvidenceFailure("image_unavailable", "representation_hash_mismatch")
    if mime == "image/svg+xml":
        raise _EvidenceFailure("image_unavailable", "svg_rasterizer_unavailable")
    width, height = _inspect_raster(raw, mime)
    if mime in _TRANSCODE_RASTER_TYPES:
        image_module = _pillow("image_unavailable")
        with image_module.open(io.BytesIO(raw)) as image:
            converted = image.convert("RGB")
            stream = io.BytesIO()
            converted.save(stream, format="PNG")
            raw = stream.getvalue()
            width, height = converted.size
        return raw, "image/png", width, height, "pillow-rasterize", _package_version("Pillow")
    return raw, mime, width, height, "notebook-original", "v1"


def _artifact_from_notebook(
    original: bytes, source: SourceDraft, preferred_mime: str | None
) -> _Artifact:
    data, media_type, width, height, renderer, version = _notebook_image(
        original, source, preferred_mime
    )
    return _Artifact(data, media_type, width, height, renderer, version)


def _inspect_raster(data: bytes, media_type: str) -> tuple[int, int]:
    image_module = _pillow("image_unavailable")
    expected = {
        "image/png": "PNG",
        "image/jpeg": "JPEG",
        "image/webp": "WEBP",
        "image/bmp": "BMP",
        "image/tiff": "TIFF",
    }.get(media_type)
    if expected is None:
        raise ValueError("unsupported raster media type")
    with warnings.catch_warnings():
        warnings.simplefilter("error", image_module.DecompressionBombWarning)
        with image_module.open(io.BytesIO(data)) as image:
            if image.format != expected:
                raise ValueError("image bytes do not match declared MIME")
            image.load()
            width, height = image.size
    if width <= 0 or height <= 0 or width * height > _MAX_IMAGE_PIXELS:
        raise ValueError("image dimensions are invalid or too large")
    return width, height


def _pillow(status: EvidenceStatus) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised in isolated subprocess
        raise _EvidenceFailure(status, "pillow_unavailable") from exc
    return Image


def _json_pointer(document: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must be absolute")
    current = document
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(token)]
        elif isinstance(current, dict):
            current = current[token]
        else:
            raise KeyError(pointer)
    return current


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    raise ValueError("image representation is not text")


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
