"""Same-origin browser workspace API."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.routing import APIRoute

from app.config import Settings, get_settings
from app.rag.workspace import WorkspaceService, make_workspace
from app.rag.workspace_preview import render_citation_preview
from app.schemas.workspace import (
    CreateSession,
    TurnRequest,
    UpdateFocus,
    UpdateScope,
    WorkspaceError,
    inaccessible,
)

_COOKIE = "srag_workspace_owner"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_SUFFIXES = {".pdf", ".pptx", ".ipynb"}
_MEDIA_HEADERS = {"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"}


class WorkspaceRoute(APIRoute):
    def get_route_handler(self):
        handle = super().get_route_handler()

        async def guarded(request: Request) -> Response:
            try:
                _same_origin(request)
                response = await handle(request)
                response.headers["Cache-Control"] = "private, no-store"
                return response
            except WorkspaceError as error:
                headers = {"Cache-Control": "private, no-store"}
                raise HTTPException(
                    error.status, detail={"code": error.code}, headers=headers
                ) from error

        return guarded


router = APIRouter(route_class=WorkspaceRoute)


def _same_origin(request: Request) -> None:
    if request.url.hostname not in _LOCAL_HOSTS:
        raise inaccessible()
    origin = request.headers.get("origin")
    origin_from_request = f"{request.url.scheme}://{request.url.netloc}"
    if origin is not None and origin != origin_from_request:
        raise inaccessible()
    if request.headers.get("sec-fetch-site") in {"same-site", "cross-site"}:
        raise inaccessible()


def _owner(request: Request) -> str | None:
    cookie = request.cookies.get(_COOKIE)
    supplied = request.headers.get("authorization")
    if supplied is not None:
        prefix, _, credential = supplied.partition(" ")
        if cookie is not None or prefix.casefold() != "bearer" or not credential:
            raise inaccessible()
        return credential
    return cookie


def _required_owner(request: Request) -> str:
    owner = _owner(request)
    if owner is None:
        raise inaccessible()
    return owner


def get_workspace_service(settings: Annotated[Settings, Depends(get_settings)]) -> WorkspaceService:
    return make_workspace(settings)


Service = Annotated[WorkspaceService, Depends(get_workspace_service)]
Owner = Annotated[str, Depends(_required_owner)]


@router.post("/sessions", status_code=201)
def create_session(
    payload: CreateSession,
    request: Request,
    response: Response,
    service: Service,
):
    if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/json":
        raise WorkspaceError("invalid_content_type", 415)
    supplied = _owner(request)
    session, token = service.create(supplied, payload)
    if supplied is None:
        response.set_cookie(
            _COOKIE,
            token,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
            path="/api/workspace",
        )
    return session


@router.get("/sessions")
def list_sessions(owner: Owner, service: Service):
    return service.list_sessions(owner)


@router.get("/sessions/{session_id}")
def get_session(session_id: str, owner: Owner, service: Service):
    return service.session(owner, session_id)


@router.patch("/sessions/{session_id}/scope")
def update_scope(session_id: str, payload: UpdateScope, owner: Owner, service: Service):
    return service.update_scope(owner, session_id, payload)


@router.put("/sessions/{session_id}/focus")
def update_focus(session_id: str, payload: UpdateFocus, owner: Owner, service: Service):
    return service.update_focus(owner, session_id, payload)


@router.get("/sessions/{session_id}/catalog")
def catalog(session_id: str, owner: Owner, service: Service):
    return service.catalog(owner, session_id)


@router.get("/sessions/{session_id}/catalog/{node_id}/media")
def catalog_media(session_id: str, node_id: str, owner: Owner, service: Service):
    image, mime_type = service.catalog_media(owner, session_id, node_id)
    return Response(image, media_type=mime_type, headers=_MEDIA_HEADERS)


@router.post("/sessions/{session_id}/import")
async def import_file(
    session_id: str,
    request: Request,
    owner: Owner,
    service: Service,
    content_type: Annotated[str | None, Header()] = None,
    course_id: Annotated[str, Query(min_length=1, max_length=160)] = "",
    filename: Annotated[str, Query(min_length=1, max_length=200)] = "",
    document_id: Annotated[str | None, Query(min_length=1, max_length=160)] = None,
):
    if not content_type or content_type.split(";", 1)[0].strip() != "application/octet-stream":
        raise WorkspaceError("invalid_content_type", 415)
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in filename)
        or Path(filename).suffix.casefold() not in _SUFFIXES
    ):
        raise WorkspaceError("invalid_filename", 422)
    await run_in_threadpool(service.authorize_import, owner, session_id, course_id, document_id)
    maximum = service.settings.workspace_max_upload_bytes
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            length = int(declared)
        except ValueError as exc:
            raise WorkspaceError("invalid_request", 422) from exc
        if length > maximum:
            raise WorkspaceError("file_too_large", 413)
    root = service.library.course_root
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".workspace-upload-", dir=root) as temp:
        path = Path(temp) / filename
        size = 0
        with path.open("xb") as target:
            async for chunk in request.stream():
                size += len(chunk)
                if size > maximum:
                    raise WorkspaceError("file_too_large", 413)
                target.write(chunk)
        if not size:
            raise WorkspaceError("empty_file", 422)
        return await run_in_threadpool(
            service.import_file,
            owner,
            session_id,
            course_id,
            path,
            document_id,
        )


@router.post("/sessions/{session_id}/turns")
def submit_turn(
    session_id: str,
    payload: TurnRequest,
    response: Response,
    owner: Owner,
    service: Service,
):
    result = service.submit(owner, session_id, payload)
    if payload.anchor is not None and result["turn"]["error"] == "not_accessible":
        raise inaccessible()
    if result["turn"]["status"] == "running":
        response.status_code = 202
    return result


@router.get("/sessions/{session_id}/turns/{client_turn_id}")
def get_turn(session_id: str, client_turn_id: str, owner: Owner, service: Service):
    return service.get_turn(owner, session_id, client_turn_id)


@router.post("/sessions/{session_id}/turns/{client_turn_id}/cancel")
def cancel_turn(session_id: str, client_turn_id: str, owner: Owner, service: Service):
    return service.cancel(owner, session_id, client_turn_id)


@router.post("/sessions/{session_id}/turns/{client_turn_id}/retry")
def retry_turn(
    session_id: str,
    client_turn_id: str,
    response: Response,
    owner: Owner,
    service: Service,
):
    result = service.retry(owner, session_id, client_turn_id)
    if result["turn"]["status"] == "running":
        response.status_code = 202
    return result


@router.get("/sessions/{session_id}/messages/{message_id}/media/{display_id}")
def historical_media(
    session_id: str,
    message_id: str,
    display_id: str,
    owner: Owner,
    service: Service,
    view: Literal["inline", "page"] = "inline",
):
    image, mime_type = service.historical_media(
        owner, session_id, message_id, display_id, view=view
    )
    return Response(image, media_type=mime_type, headers=_MEDIA_HEADERS)


@router.get("/sessions/{session_id}/messages/{message_id}/evidence/{evidence_id}")
def historical_evidence(
    session_id: str,
    message_id: str,
    evidence_id: str,
    response: Response,
    owner: Owner,
    service: Service,
):
    payload = service.historical_evidence(owner, session_id, message_id, evidence_id)
    response.headers.update(_MEDIA_HEADERS)
    return payload


@router.get("/sessions/{session_id}/messages/{message_id}/evidence/{evidence_id}/preview")
def historical_citation_preview(
    session_id: str, message_id: str, evidence_id: str, owner: Owner, service: Service
):
    image, mime_type = render_citation_preview(service, owner, session_id, message_id, evidence_id)
    return Response(image, media_type=mime_type, headers=_MEDIA_HEADERS)
