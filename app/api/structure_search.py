"""HTTP surface for scoped lexical and direct-location structure search."""

from __future__ import annotations

import os
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.rag.local_library import LocalLibrary
from app.rag.structure_search import (
    StructureSearcher,
    StructureSearchRequest,
    StructureSearchResponse,
)
from app.stores.structure_snapshot import StructureSnapshotStore

router = APIRouter()


def get_structure_course_allowlist() -> frozenset[str]:
    """Return the server-trusted course scope. Empty means closed."""

    raw = os.getenv("SRAG_STRUCTURE_COURSES", "")
    return frozenset(item.strip() for item in raw.split(",") if item.strip())


def get_structure_searcher(
    settings: Annotated[Settings, Depends(get_settings)],
) -> StructureSearcher:
    """Initialize the local schema without constructing model or vector clients."""

    LocalLibrary(settings)
    return StructureSearcher(StructureSnapshotStore(settings))


@router.post(
    "/search",
    response_model=StructureSearchResponse,
    summary="Search the current local structure snapshot",
)
def search_structure(
    request: StructureSearchRequest,
    searcher: Annotated[StructureSearcher, Depends(get_structure_searcher)],
    allowlist: Annotated[frozenset[str], Depends(get_structure_course_allowlist)],
) -> StructureSearchResponse:
    if not allowlist.intersection(request.scope.course_ids):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="no requested course is allowed for structure search",
        )
    return searcher.search(request, allowed_course_ids=allowlist)
