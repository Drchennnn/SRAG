"""Step B: Run G20 frozen questions on the fixed G19 product baseline (d993748) from an empty library.

Executes the exact browser workspace HTTP API flow (/api/workspace/sessions/*) against an isolated
checkout of d993748c01fabaff26586908de9bf72499534012, preserving all raw responses, citations,
media byte digests, and outbound provider calls (DeepSeek & DashScope).
Also supports running the historical A03-style switch combination (--answer-enabled false) in a
separate directory for configuration contrast.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import subprocess
import sys
from contextvars import ContextVar
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import httpx

TURN_CTX: ContextVar[dict[str, Any] | None] = ContextVar("g20_turn_ctx", default=None)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_clean_checkout(repo_root: Path, commit_sha: str, target_dir: Path) -> str:
    resolved = (
        subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", commit_sha])
        .decode("utf-8")
        .strip()
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    archive = subprocess.Popen(
        ["git", "-C", str(repo_root), "archive", resolved],
        stdout=subprocess.PIPE,
    )
    subprocess.check_call(["tar", "-x", "-C", str(target_dir)], stdin=archive.stdout)
    if archive.wait() != 0:
        raise RuntimeError(f"git archive failed for {resolved}")
    return resolved


def _install_provider_recorder(settings: Any, provider_events: list[dict[str, Any]]) -> Any:
    orig_sync = httpx.HTTPTransport.handle_request
    orig_async = httpx.AsyncHTTPTransport.handle_async_request

    def sync_wrapper(transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        ctx = TURN_CTX.get()
        payload = json.loads(request.content)
        image_sha = None
        try:
            messages = payload.get("input", {}).get("messages", [])
            for msg in messages:
                for part in msg.get("content", []):
                    if isinstance(part, dict) and "image" in part:
                        _, b64 = part["image"].split(",", 1)
                        image_sha = _sha256_bytes(base64.b64decode(b64))
        except Exception:
            pass
        event: dict[str, Any] = {
            "turn": ctx,
            "provider": "dashscope" if "dashscope" in str(request.url) else "http_sync",
            "url": str(request.url),
            "model": payload.get("model"),
            "image_sha256": image_sha,
            "request_sha256": _sha256_bytes(request.content),
        }
        started = perf_counter()
        try:
            response = orig_sync(transport, request)
            event["status_code"] = response.status_code
            try:
                response.read()
                body = response.json()
                event["usage"] = body.get("usage")
            except Exception:
                pass
            return response
        except Exception as exc:
            event["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            event["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
            provider_events.append(event)

    async def async_wrapper(
        transport: httpx.AsyncHTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        ctx = TURN_CTX.get()
        payload = json.loads(request.content)
        input_evidence_ids = None
        mode = None
        try:
            messages = payload.get("messages", [])
            if len(messages) >= 2:
                user_obj = json.loads(messages[1]["content"])
                if isinstance(user_obj, dict) and "evidence" in user_obj:
                    input_evidence_ids = [item.get("id") for item in user_obj["evidence"]]
                    mode = user_obj.get("mode")
        except Exception:
            pass
        event: dict[str, Any] = {
            "turn": ctx,
            "provider": (
                "deepseek"
                if "deepseek" in str(request.url) or "/chat/completions" in str(request.url)
                else ("jev" if "systemone" in str(request.url) else "http_async")
            ),
            "url": str(request.url),
            "model": payload.get("model"),
            "mode": mode,
            "input_evidence_ids": input_evidence_ids,
            "request_sha256": _sha256_bytes(request.content),
        }
        started = perf_counter()
        try:
            response = await orig_async(transport, request)
            event["status_code"] = response.status_code
            try:
                await response.aread()
                body = response.json()
                event["usage"] = body.get("usage")
            except Exception:
                pass
            return response
        except Exception as exc:
            event["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            event["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
            provider_events.append(event)

    httpx.HTTPTransport.handle_request = sync_wrapper
    httpx.AsyncHTTPTransport.handle_async_request = async_wrapper

    def restore() -> None:
        httpx.HTTPTransport.handle_request = orig_sync
        httpx.AsyncHTTPTransport.handle_async_request = orig_async

    return restore


async def _run_suite(
    app: Any,
    service: Any,
    annotations: dict[str, Any],
    fixture_dir: Path,
    provider_events: list[dict[str, Any]],
) -> dict[str, Any]:
    course_id = annotations["course_id"]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://127.0.0.1:18886",
        headers={"Origin": "http://127.0.0.1:18886"},
    ) as client:
        # 1. Create workspace session
        res = await client.post("/api/workspace/sessions", json={})
        assert res.status_code == 201, res.text
        session = res.json()
        session_id = session["session_id"]

        # 2. Import initial documents (Mixtral PDF v1, DPO PDF v1, Notebook v1)
        imports: list[dict[str, Any]] = []
        doc_ids_by_name: dict[str, str] = {}
        initial_uploads = [
            ("Mixtral-of-Experts-2401.04088v1.pdf", fixture_dir / "Mixtral-of-Experts-2401.04088v1.pdf"),
            (
                "Direct-Preference-Optimization-2305.18290v3.pdf",
                fixture_dir / "Direct-Preference-Optimization-2305.18290v3.pdf",
            ),
            (
                "Mixtral-Router-Verification.ipynb",
                fixture_dir / "v1" / "Mixtral-Router-Verification.ipynb",
            ),
        ]
        for fname, fpath in initial_uploads:
            raw = fpath.read_bytes()
            imp_res = await client.post(
                f"/api/workspace/sessions/{session_id}/import",
                params={"course_id": course_id, "filename": fname},
                content=raw,
                headers={"Content-Type": "application/octet-stream"},
            )
            assert imp_res.status_code == 200, imp_res.text
            imp_data = imp_res.json()
            imports.append(imp_data)
            doc_ids_by_name[fname] = imp_data["document_id"]

        async def get_session_and_catalog() -> tuple[dict[str, Any], dict[str, Any]]:
            s_res = await client.get(f"/api/workspace/sessions/{session_id}")
            c_res = await client.get(f"/api/workspace/sessions/{session_id}/catalog")
            assert s_res.status_code == 200 and c_res.status_code == 200
            return s_res.json(), c_res.json()

        def find_doc_node_id(catalog_payload: dict[str, Any], doc_id: str) -> str:
            for course in catalog_payload.get("courses", []):
                for doc in course.get("documents", []):
                    if doc["document_id"] == doc_id:
                        return doc["node_id"]
            raise KeyError(f"Document node not found in catalog: {doc_id}")

        async def select_document(file_name: str) -> dict[str, Any]:
            doc_id = doc_ids_by_name[file_name]
            sess, cat = await get_session_and_catalog()
            if sess["scope"]["document_ids"] != [doc_id]:
                sc_res = await client.patch(
                    f"/api/workspace/sessions/{session_id}/scope",
                    json={
                        "course_ids": [course_id],
                        "document_ids": [doc_id],
                        "scope_revision": sess["scope_revision"],
                    },
                )
                assert sc_res.status_code == 200, sc_res.text
                sess = sc_res.json()
            _, cat = await get_session_and_catalog()
            node_id = find_doc_node_id(cat, doc_id)
            fc_res = await client.put(
                f"/api/workspace/sessions/{session_id}/focus",
                json={
                    "node_id": node_id,
                    "scope_revision": sess["scope_revision"],
                },
            )
            assert fc_res.status_code == 200, fc_res.text
            return fc_res.json()

        async def select_course_scope() -> dict[str, Any]:
            sess, _ = await get_session_and_catalog()
            if sess["scope"]["document_ids"]:
                sc_res = await client.patch(
                    f"/api/workspace/sessions/{session_id}/scope",
                    json={
                        "course_ids": [course_id],
                        "document_ids": [],
                        "scope_revision": sess["scope_revision"],
                    },
                )
                assert sc_res.status_code == 200, sc_res.text
                return sc_res.json()
            return sess

        # Snapshot inventory after v1 import
        from app.stores.structure_snapshot import SnapshotRequest

        v1_snapshot = service.snapshots.read(SnapshotRequest((course_id,), ()))
        v1_inventory = [
            {
                "file_name": next(
                    d.file_name for d in v1_snapshot.documents if d.document_id == src.document_id
                ),
                "document_id": src.document_id,
                "version": src.version,
                "source_id": src.source_id,
                "parent_id": src.parent_id,
                "kind": src.kind,
                "object_type": src.locator.get("object_type"),
                "bbox": src.locator.get("bbox"),
                "text_chars": len(src.text),
                "state": src.state,
            }
            for src in v1_snapshot.sources
        ]

        turn_records: list[dict[str, Any]] = []
        v2_inventory: list[dict[str, Any]] = []

        for q_spec in annotations["questions"]:
            qid = q_spec["question_id"]
            turn_idx = q_spec["turn_index"]
            action = q_spec["scope_action"]

            if action.startswith("select_document:"):
                target_file = action.split(":", 1)[1]
                sess = await select_document(target_file)
            elif action == "course_scope":
                sess = await select_course_scope()
            elif action == "keep_previous_scope":
                sess, _ = await get_session_and_catalog()
            elif action == "update_notebook_to_v2_and_ask":
                nb_doc_id = doc_ids_by_name["Mixtral-Router-Verification.ipynb"]
                raw_v2 = (fixture_dir / "v2" / "Mixtral-Router-Verification.ipynb").read_bytes()
                imp_v2 = await client.post(
                    f"/api/workspace/sessions/{session_id}/import",
                    params={
                        "course_id": course_id,
                        "filename": "Mixtral-Router-Verification.ipynb",
                        "document_id": nb_doc_id,
                    },
                    content=raw_v2,
                    headers={"Content-Type": "application/octet-stream"},
                )
                assert imp_v2.status_code == 200, imp_v2.text
                imports.append(imp_v2.json())
                # Mirror workspace.js lines 865-868: after replace-document import, if focus became None, re-focus new document root
                sess, cat = await get_session_and_catalog()
                if not sess.get("focus") and nb_doc_id in sess["scope"]["document_ids"]:
                    node_id = find_doc_node_id(cat, nb_doc_id)
                    fc_res = await client.put(
                        f"/api/workspace/sessions/{session_id}/focus",
                        json={
                            "node_id": node_id,
                            "scope_revision": sess["scope_revision"],
                        },
                    )
                    assert fc_res.status_code == 200, fc_res.text
                    sess = fc_res.json()
                v2_snap = service.snapshots.read(SnapshotRequest((course_id,), (nb_doc_id,)))
                v2_inventory = [
                    {
                        "file_name": "Mixtral-Router-Verification.ipynb",
                        "document_id": src.document_id,
                        "version": src.version,
                        "source_id": src.source_id,
                        "parent_id": src.parent_id,
                        "kind": src.kind,
                        "object_type": src.locator.get("object_type"),
                        "text_chars": len(src.text),
                        "state": src.state,
                    }
                    for src in v2_snap.sources
                ]
            else:
                raise ValueError(f"Unknown scope_action: {action}")

            client_turn_id = f"g20-b-turn-{turn_idx:02d}-{qid.lower()}"
            token = TURN_CTX.set(
                {"question_id": qid, "turn_index": turn_idx, "client_turn_id": client_turn_id}
            )
            started = perf_counter()
            try:
                t_res = await client.post(
                    f"/api/workspace/sessions/{session_id}/turns",
                    json={
                        "client_turn_id": client_turn_id,
                        "kind": "ask",
                        "text": q_spec["question"],
                        "scope_revision": sess["scope_revision"],
                        "anchor": None,
                    },
                )
            finally:
                TURN_CTX.reset(token)
            http_elapsed_ms = round((perf_counter() - started) * 1000, 3)
            assert t_res.status_code == 200, f"{qid} HTTP {t_res.status_code}: {t_res.text}"
            envelope = t_res.json()
            turn_data = envelope["turn"]
            message_data = envelope.get("message") or {}
            message_id = message_data.get("message_id")

            # Verify every cited evidence via GET .../evidence/{eid} and .../preview
            opened_citations: list[dict[str, Any]] = []
            cited_ids: list[str] = []
            for blk in message_data.get("blocks", []):
                if blk.get("type") == "text":
                    for cid in blk.get("citations", []):
                        if cid not in cited_ids:
                            cited_ids.append(cid)

            for cid in cited_ids:
                ev_meta = message_data.get("sources", {}).get(cid, {})
                has_text_read = any(
                    r.get("method") in {"native_text", "native_table", "ocr"} and r.get("text")
                    for r in ev_meta.get("reads", [])
                )
                matching_img = next(
                    (
                        b
                        for b in message_data.get("blocks", [])
                        if b.get("type") == "source_image" and b.get("evidence_id") == cid
                    ),
                    None,
                )
                if matching_img and (
                    not has_text_read or matching_img.get("rendition_kind") == "object_crop"
                ):
                    did = matching_img["display_id"]
                    m_res = await client.get(
                        f"/api/workspace/sessions/{session_id}/messages/{message_id}/media/{did}",
                        params={"view": "inline"},
                    )
                    ev_record = {
                        "evidence_id": cid,
                        "open_mode": "image_media_inline",
                        "status_code": m_res.status_code,
                        "source": matching_img.get("display_source"),
                        "media_sha256": (
                            _sha256_bytes(m_res.content) if m_res.status_code == 200 else None
                        ),
                        "media_bytes": len(m_res.content) if m_res.status_code == 200 else 0,
                    }
                elif not has_text_read:
                    prev_res = await client.get(
                        f"/api/workspace/sessions/{session_id}/messages/{message_id}/evidence/{cid}/preview"
                    )
                    ev_record = {
                        "evidence_id": cid,
                        "open_mode": "visual_citation_preview",
                        "status_code": prev_res.status_code,
                        "source": (ev_meta.get("revision") or {}).get("source"),
                        "preview_sha256": (
                            _sha256_bytes(prev_res.content)
                            if prev_res.status_code == 200
                            else None
                        ),
                        "preview_bytes": (
                            len(prev_res.content) if prev_res.status_code == 200 else 0
                        ),
                    }
                else:
                    ev_res = await client.get(
                        f"/api/workspace/sessions/{session_id}/messages/{message_id}/evidence/{cid}"
                    )
                    ev_record = {
                        "evidence_id": cid,
                        "open_mode": "text_evidence",
                        "status_code": ev_res.status_code,
                    }
                    if ev_res.status_code == 200:
                        ev_body = ev_res.json()
                        ev_record.update(
                            source=ev_body["source"],
                            file_name=ev_body["file_name"],
                            locator=ev_body["locator"],
                            is_current=ev_body["is_current"],
                            text_sha256=_sha256_text(ev_body["text"]),
                            text_preview=ev_body["text"][:240],
                        )
                        prev_res = await client.get(
                            f"/api/workspace/sessions/{session_id}/messages/{message_id}/evidence/{cid}/preview"
                        )
                        ev_record["preview_status_code"] = prev_res.status_code
                        if prev_res.status_code == 200:
                            ev_record["preview_sha256"] = _sha256_bytes(prev_res.content)
                            ev_record["preview_bytes"] = len(prev_res.content)
                opened_citations.append(ev_record)

            # Verify every displayed image card via GET .../media/{did}?view=inline
            displayed_media: list[dict[str, Any]] = []
            for blk in message_data.get("blocks", []):
                if blk.get("type") == "source_image":
                    did = blk["display_id"]
                    m_res = await client.get(
                        f"/api/workspace/sessions/{session_id}/messages/{message_id}/media/{did}",
                        params={"view": "inline"},
                    )
                    p_res = await client.get(
                        f"/api/workspace/sessions/{session_id}/messages/{message_id}/media/{did}",
                        params={"view": "page"},
                    )
                    displayed_media.append(
                        {
                            "display_id": did,
                            "evidence_id": blk.get("evidence_id"),
                            "display_source": blk.get("display_source"),
                            "rendition_kind": blk.get("rendition_kind"),
                            "caption": blk.get("caption"),
                            "rect_visible_norm": blk.get("rect_visible_norm"),
                            "read_status": blk.get("read_status"),
                            "inline_status_code": m_res.status_code,
                            "inline_sha256": (
                                _sha256_bytes(m_res.content) if m_res.status_code == 200 else None
                            ),
                            "inline_bytes": len(m_res.content) if m_res.status_code == 200 else 0,
                            "page_status_code": p_res.status_code,
                            "page_sha256": (
                                _sha256_bytes(p_res.content) if p_res.status_code == 200 else None
                            ),
                        }
                    )

            turn_calls = [
                call
                for call in provider_events
                if call.get("turn") and call["turn"].get("client_turn_id") == client_turn_id
            ]
            turn_records.append(
                {
                    "question_id": qid,
                    "turn_index": turn_idx,
                    "category": q_spec["category"],
                    "scope_action": action,
                    "referred_version": q_spec["referred_version"],
                    "question": q_spec["question"],
                    "required_evidence_units": q_spec["required_evidence_units"],
                    "session_scope": sess["scope"],
                    "session_focus": sess.get("focus"),
                    "http_elapsed_ms": http_elapsed_ms,
                    "turn_status": turn_data.get("status"),
                    "turn_error": turn_data.get("error"),
                    "resolution": turn_data.get("resolution"),
                    "options": envelope.get("options", []),
                    "trace": turn_data.get("trace"),
                    "message_id": message_id,
                    "blocks": message_data.get("blocks", []),
                    "sources": message_data.get("sources", {}),
                    "visible_display_ids": message_data.get("visible_display_ids", []),
                    "opened_citations": opened_citations,
                    "displayed_media": displayed_media,
                    "provider_calls": turn_calls,
                }
            )

        return {
            "session_id": session_id,
            "document_ids": doc_ids_by_name,
            "imports": imports,
            "v1_inventory_summary": {
                "total_sources": len(v1_inventory),
                "by_file": {
                    fname: [item for item in v1_inventory if item["file_name"] == fname]
                    for fname in doc_ids_by_name
                },
            },
            "v2_notebook_inventory": v2_inventory,
            "turns": turn_records,
            "provider_summary": {
                "total_calls": len(provider_events),
                "deepseek_calls": sum(1 for e in provider_events if e["provider"] == "deepseek"),
                "dashscope_calls": sum(1 for e in provider_events if e["provider"] == "dashscope"),
                "jev_calls": sum(1 for e in provider_events if e["provider"] == "jev"),
                "http_200_calls": sum(1 for e in provider_events if e.get("status_code") == 200),
                "events": provider_events,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject",
        default="d993748c01fabaff26586908de9bf72499534012",
        help="Git commit SHA of the fixed product baseline",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file with DeepSeek and DashScope credentials",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("docs/G/G20-A-判据与标注.json"),
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path("docs/G/G20-fixtures"),
    )
    parser.add_argument(
        "--answer-enabled",
        choices=("true", "false"),
        default="true",
        help="Whether WORKSPACE_ANSWER_ENABLED is true (default product) or false (historical A03 switch)",
    )
    parser.add_argument(
        "--strategy",
        choices=("fts", "bilingual", "jev", "expanded"),
        default="fts",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/G/G20-B-基线实测.json"),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    annotations = json.loads((repo_root / args.annotations).read_text(encoding="utf-8"))
    fixture_dir = (repo_root / args.fixture_dir).resolve()
    env_path = (repo_root / args.env_file).resolve()

    with TemporaryDirectory(prefix="srag-g20-b-") as tmp_str:
        tmp = Path(tmp_str)
        checkout_dir = tmp / "checkout"
        resolved_sha = _extract_clean_checkout(repo_root, args.subject, checkout_dir)

        # Put clean checkout first on sys.path so dirty working tree files are never imported
        sys.path.insert(0, str(checkout_dir))

        from app.api.workspace import get_workspace_service
        from app.config import Settings, get_settings
        from app.main import create_app
        from app.rag.workspace import make_workspace

        settings = Settings(
            _env_file=env_path,
            course_root=tmp / "courses",
            data_dir=tmp / "data",
            sqlite_path=tmp / "data" / "library.sqlite3",
            workspace_course_ids=annotations["course_id"],
            workspace_answer_enabled=(args.answer_enabled == "true"),
            workspace_retrieval_strategy=args.strategy,
            source_vision_enabled=True,
            qdrant_url="http://127.0.0.1:1",
            qdrant_api_key="",
        )
        if args.answer_enabled == "true" and not settings.deepseek_api_key.strip():
            raise RuntimeError("DEEPSEEK_API_KEY is required for workspace_answer_enabled=true")
        if not settings.dashscope_api_key.strip():
            raise RuntimeError("DASHSCOPE_API_KEY is required for source_vision_enabled=true")

        app = create_app()
        service = make_workspace(settings)
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[get_workspace_service] = lambda: service

        provider_events: list[dict[str, Any]] = []
        restore = _install_provider_recorder(settings, provider_events)
        try:
            suite_result = asyncio.run(
                _run_suite(app, service, annotations, fixture_dir, provider_events)
            )
        finally:
            restore()

        output_payload = {
            "task_id": "G20",
            "step": "B",
            "subject_commit": resolved_sha,
            "config": {
                "workspace_course_ids": settings.workspace_course_ids,
                "workspace_answer_enabled": settings.workspace_answer_enabled,
                "workspace_retrieval_strategy": settings.workspace_retrieval_strategy,
                "source_vision_enabled": settings.source_vision_enabled,
                "deepseek_chat_model": settings.deepseek_chat_model,
                "dashscope_vision_model": settings.dashscope_vision_model,
                "answer_context_tokens": settings.answer_context_tokens,
                "workspace_input_token_limit": settings.workspace_input_token_limit,
            },
            **suite_result,
        }
        out_path = (repo_root / args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "subject_commit": resolved_sha,
                    "out": str(out_path),
                    "turns": len(suite_result["turns"]),
                    "provider_summary": {
                        k: v
                        for k, v in suite_result["provider_summary"].items()
                        if k != "events"
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
