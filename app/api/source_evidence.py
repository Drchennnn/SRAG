"""Current-version source image endpoint with a server-owned course allowlist."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends

from app.config import get_settings
from app.rag.local_library import LocalLibrary
from app.rag.source_evidence import EvidenceRequest, EvidenceResult, SourceEvidenceService
from app.rag.source_vision import SourceVisionService

router = APIRouter()


def get_source_evidence_service() -> SourceEvidenceService:
    settings = get_settings()
    course_ids = [
        item.strip() for item in settings.source_evidence_course_ids.split(",") if item.strip()
    ]
    soffice = Path(settings.srag_soffice) if settings.srag_soffice else None
    return SourceEvidenceService(
        LocalLibrary(settings).store,
        cache_dir=settings.data_dir / "source_evidence_cache",
        current_course_ids=course_ids,
        soffice_path=soffice,
        max_inline_bytes=settings.source_evidence_max_inline_bytes,
    )


def get_source_vision_service(
    evidence: Annotated[SourceEvidenceService, Depends(get_source_evidence_service)],
) -> SourceVisionService:
    settings = get_settings()
    return SourceVisionService(
        evidence,
        evidence.store,
        cache_dir=settings.data_dir / "source_vision_cache",
        enabled=settings.source_vision_enabled,
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        model=settings.dashscope_vision_model,
        timeout_seconds=settings.dashscope_timeout_seconds,
        max_image_bytes=settings.dashscope_visual_max_image_bytes,
    )


@router.post(
    "/current",
    response_model=EvidenceResult,
    summary="读取当前版本的原件页、裁图或 Notebook 输出",
)
async def render_current_source(
    request: EvidenceRequest,
    service: Annotated[SourceVisionService, Depends(get_source_vision_service)],
) -> EvidenceResult:
    """The request cannot supply a course scope, filesystem path, or arbitrary rectangle."""

    return await asyncio.to_thread(service.process_current, request)
