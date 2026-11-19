"""Current-version source image endpoint with a server-owned course allowlist."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends

from app.config import get_settings
from app.rag.local_library import LocalLibrary
from app.rag.source_evidence import EvidenceRequest, EvidenceResult, SourceEvidenceService

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


@router.post(
    "/current",
    response_model=EvidenceResult,
    summary="读取当前版本的原件页、裁图或 Notebook 输出",
)
async def render_current_source(
    request: EvidenceRequest,
    service: Annotated[SourceEvidenceService, Depends(get_source_evidence_service)],
) -> EvidenceResult:
    """The request cannot supply a course scope, filesystem path, or arbitrary rectangle."""

    return await asyncio.to_thread(service.render_current, request)
