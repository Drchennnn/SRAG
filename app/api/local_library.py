"""Local SQLite import, migration, search, and deletion endpoints."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from app.config import get_settings
from app.rag.local_library import LocalLibrary
from app.schemas.local import ImportResult, LocalHit
from app.schemas.rag import DeleteDocumentResponse, SearchRequest
from app.stores.local_library import PublicationConflict

router = APIRouter()


class LocalImportRequest(BaseModel):
    course_id: str = Field(min_length=1, pattern=r"\S")
    paths: list[Path] = Field(min_length=1)
    document_id: str | None = Field(default=None, min_length=1, pattern=r"\S")
    release_at: datetime | None = None

    @model_validator(mode="after")
    def document_id_requires_one_path(self) -> LocalImportRequest:
        if self.document_id is not None and len(self.paths) != 1:
            raise ValueError("document_id may only be used with one selected path")
        return self


class LocalMigrationRequest(BaseModel):
    document_id: str = Field(min_length=1, pattern=r"\S")


class LocalBundleMigrationRequest(BaseModel):
    course_id: str = Field(min_length=1, pattern=r"\S")
    source_path: Path
    bundle_dir: Path
    content_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    version: int = Field(default=1, ge=1)


class LocalSearchResponse(BaseModel):
    query: str
    hits: list[LocalHit]


def get_local_library() -> LocalLibrary:
    """Create the local library without constructing model or vector clients."""

    return LocalLibrary(get_settings())


@router.post(
    "/import",
    response_model=list[ImportResult],
    summary="导入本地 PDF、PPTX 或 Notebook",
)
def import_local_paths(
    request: LocalImportRequest,
    library: Annotated[LocalLibrary, Depends(get_local_library)],
) -> list[ImportResult]:
    try:
        results = library.import_paths(
            request.course_id,
            request.paths,
            document_id=request.document_id,
            release_at=request.release_at,
        )
    except PublicationConflict as exc:
        raise _conflict(exc) from exc
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        raise _client_error(exc) from exc
    if results and all(result.status == "failed" for result in results):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "all selected paths failed",
                "results": [result.model_dump(mode="json") for result in results],
            },
        )
    return results


@router.post("/search", response_model=LocalSearchResponse, summary="搜索当前本地文字版本")
def search_local_sources(
    request: SearchRequest,
    library: Annotated[LocalLibrary, Depends(get_local_library)],
) -> LocalSearchResponse:
    try:
        hits = library.search(request)
    except ValueError as exc:
        raise _client_error(exc) from exc
    return LocalSearchResponse(query=request.query, hits=hits)


@router.post("/migrate", response_model=ImportResult, summary="迁移现有旧资料")
def migrate_legacy_document(
    request: LocalMigrationRequest,
    library: Annotated[LocalLibrary, Depends(get_local_library)],
) -> ImportResult:
    try:
        return library.migrate_legacy(request.document_id)
    except PublicationConflict as exc:
        raise _conflict(exc) from exc
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise _client_error(exc) from exc


@router.post("/migrate/bundle", response_model=ImportResult, summary="迁移旧课件包")
def migrate_legacy_bundle(
    request: LocalBundleMigrationRequest,
    library: Annotated[LocalLibrary, Depends(get_local_library)],
) -> ImportResult:
    try:
        result = library.import_legacy_bundle(
            course_id=request.course_id,
            source_path=request.source_path,
            bundle_dir=request.bundle_dir,
            content_sha256=request.content_sha256.casefold(),
            version=request.version,
        )
    except PublicationConflict as exc:
        raise _conflict(exc) from exc
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        raise _client_error(exc) from exc
    return result


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteDocumentResponse,
    summary="撤销本地资料可见性",
)
def delete_local_document(
    document_id: str,
    library: Annotated[LocalLibrary, Depends(get_local_library)],
) -> DeleteDocumentResponse:
    if not library.delete(document_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="本地资料不存在")
    return DeleteDocumentResponse(document_id=document_id, deleted=True)


def _client_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (FileNotFoundError, KeyError)):
        code = status.HTTP_404_NOT_FOUND
    elif isinstance(exc, PermissionError):
        code = status.HTTP_403_FORBIDDEN
    else:
        code = status.HTTP_400_BAD_REQUEST
    return HTTPException(status_code=code, detail=str(exc))


def _conflict(exc: PublicationConflict) -> HTTPException:
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
