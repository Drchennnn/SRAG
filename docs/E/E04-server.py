"""Run the real workspace against public fixtures and record browser turn payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from app.api.workspace import get_workspace_service
from app.config import Settings, get_settings
from app.main import create_app
from app.rag.local_library import LocalLibrary
from app.rag.source_evidence import RegionComponent, RegionRegistry, RegisteredRegion
from app.rag.workspace import make_workspace
from app.schemas.local import SourceKey
from app.stores.structure_snapshot import SnapshotRequest, StructureSnapshotStore

PAPER_HASH = "c8f123efffba2ce032c10243996fa3139a368bc68a38bced6c5e1a81e51e0cf3"
COURSES = ("signals", "isolation")
ROOT = Path(__file__).resolve().parents[2]


def _payload(data: bytearray):
    try:
        return json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def run_workspace(data_dir: Path, trace: Path, soffice: Path | None, real_vision: bool):
    data_dir.mkdir(parents=True, exist_ok=True)
    courses = data_dir / "courses"
    courses.mkdir(exist_ok=True)
    trace.parent.mkdir(parents=True, exist_ok=True)
    key_override = {} if real_vision else {"dashscope_api_key": "", "deepseek_api_key": ""}
    settings = Settings(
        _env_file=ROOT / ".env" if real_vision else None,
        course_root=courses,
        data_dir=data_dir / "data",
        sqlite_path=data_dir / "library.sqlite3",
        workspace_course_ids=",".join(COURSES),
        srag_soffice=soffice,
        source_vision_enabled=real_vision,
        dashscope_timeout_seconds=15,
        qdrant_url="http://127.0.0.1:1",
        **key_override,
    )
    LocalLibrary(settings)
    snapshots = StructureSnapshotStore(settings)
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings

    def workspace():
        regions = RegionRegistry()
        mappings = {}
        for document in snapshots.read(SnapshotRequest(COURSES)).documents:
            if document.content_sha256 != PAPER_HASH:
                continue
            key = SourceKey(
                document_id=document.document_id, version=document.version, source_id="page:2"
            )
            region_id = f"public-right-{document.document_id}-v{document.version}"
            regions.register(
                RegisteredRegion(
                    region_id=region_id,
                    source=key,
                    content_sha256=PAPER_HASH,
                    coordinate_frame="visible_page_norm_v1",
                    target_rect=(0.53, 0.20, 0.92, 0.51),
                    components=[
                        RegionComponent(
                            role="caption",
                            rect=(0.54, 0.507, 0.91, 0.536),
                            coordinate_frame="visible_page_norm_v1",
                        ),
                        RegionComponent(
                            role="x_axis",
                            rect=(0.61, 0.45, 0.83, 0.50),
                            coordinate_frame="visible_page_norm_v1",
                        ),
                    ],
                    provenance="trusted_corpus_registration",
                    revision=1,
                )
            )
            mappings[(key.document_id, key.version, key.source_id)] = region_id
        return make_workspace(settings, regions=regions, region_ids=mappings)

    app.dependency_overrides[get_workspace_service] = workspace

    async def traced(scope, receive, send):
        path = scope.get("path", "")
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or not path.startswith("/api/workspace/sessions/")
            or not path.endswith("/turns")
        ):
            return await app(scope, receive, send)
        request_bytes = bytearray()
        response_bytes = bytearray()
        status_code = 0

        async def request_events():
            event = await receive()
            request_bytes.extend(event.get("body", b""))
            return event

        async def response_events(event):
            nonlocal status_code
            if event["type"] == "http.response.start":
                status_code = event["status"]
            elif event["type"] == "http.response.body":
                response_bytes.extend(event.get("body", b""))
                if not event.get("more_body"):
                    entry = {
                        "path": path,
                        "status": status_code,
                        "request": _payload(request_bytes),
                        "response": _payload(response_bytes),
                    }
                    with trace.open("a", encoding="utf-8") as log:
                        log.write(
                            json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
                        )
            await send(event)

        await app(scope, request_events, response_events)

    return traced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--port", default=18804, type=int)
    parser.add_argument("--soffice", type=Path)
    parser.add_argument("--real-vision", action="store_true")
    args = parser.parse_args()
    asgi = run_workspace(args.data_dir, args.trace, args.soffice, args.real_vision)
    uvicorn.run(asgi, host="127.0.0.1", port=args.port, log_level="warning", lifespan="off")


if __name__ == "__main__":
    main()
