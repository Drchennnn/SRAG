"""Layer-by-layer ASGI diagnostic & comparison runner for G-pdf-object-recovery.

Records for every frozen question under G01 formal settings
(workspace_answer_enabled=False, source_vision_enabled=True, qwen3-vl-plus, fts):
1. Parser object registration (all snapshot sources & locators)
2. Retrieval & selection (route, clarification, selected sources)
3. Actual reading (native_text / vision reads & external call counts)
4. Citation & display opening (evidence endpoint + media_url crop check)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi.testclient import TestClient

from app.api.workspace import get_workspace_service
from app.config import Settings, get_settings
from app.main import create_app
from app.rag.workspace import make_workspace

BASE = "/api/workspace/sessions"
HOST = "http://127.0.0.1:8000"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_suite(repo_dir: Path, criteria_path: Path, env_file: Path, out_path: Path) -> None:
    criteria = json.loads(criteria_path.read_text(encoding="utf-8"))
    git_sha = subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
    ).strip()

    with tempfile.TemporaryDirectory(prefix="srag-pdf-obj-asgi-") as tmp:
        rt = Path(tmp)
        settings = Settings(
            _env_file=env_file,
            course_root=rt / "courses",
            data_dir=rt / "data",
            sqlite_path=rt / "library.sqlite3",
            workspace_course_ids="pdf-obj-recovery",
            workspace_answer_enabled=False,
            workspace_retrieval_strategy="fts",
            source_vision_enabled=True,
            deepseek_api_key="",
            qdrant_api_key="",
            qdrant_url="http://127.0.0.1:1",
            dashscope_timeout_seconds=20,
        )
        app = create_app()
        service = make_workspace(settings)
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[get_workspace_service] = lambda: service

        with TestClient(app, base_url=HOST, headers={"Origin": HOST}) as client:
            boot = client.post(
                BASE, json={"title": "PDF object recovery", "course_ids": ["pdf-obj-recovery"]}
            )
            assert boot.status_code == 201
            sid = boot.json()["session_id"]
            sbase = f"{BASE}/{sid}"

            doc_ids: dict[str, str] = {}
            for name, info in criteria["pdfs"].items():
                pdf_path = repo_dir / info["file"]
                raw = pdf_path.read_bytes()
                assert _sha256(raw) == info["sha256"]
                imp = client.post(
                    f"{sbase}/import",
                    params={"course_id": "pdf-obj-recovery", "filename": name},
                    content=raw,
                    headers={"Content-Type": "application/octet-stream"},
                )
                assert imp.status_code == 200, imp.text
                doc_ids[name] = imp.json()["document_id"]

            # Layer 1: inspect registered objects in snapshot
            snap = service.snapshots.read_course("pdf-obj-recovery") if hasattr(service.snapshots, "read_course") else None
            if snap is None:
                from app.stores.structure_snapshot import SnapshotRequest
                snap = service.snapshots.read(SnapshotRequest(("pdf-obj-recovery",), tuple(doc_ids.values())))
            doc_names = {d.document_id: d.file_name for d in snap.documents}
            registered: dict[str, list[dict[str, Any]]] = {}
            for src in snap.sources:
                registered.setdefault(doc_names.get(src.document_id, src.document_id), []).append(
                    {
                        "source_id": src.source_id,
                        "kind": src.kind,
                        "parent_id": src.parent_id,
                        "title": src.title,
                        "text_head": src.text[:120],
                        "object_type": src.locator.get("object_type"),
                        "bbox": src.locator.get("bbox"),
                        "position_basis": src.locator.get("position_basis"),
                    }
                )

            results: list[dict[str, Any]] = []
            for qitem in criteria["questions"]:
                qid = qitem["question_id"]
                doc_name = qitem["document"]
                doc_id = doc_ids[doc_name]
                # Give each frozen question its own isolated session
                q_boot = client.post(
                    BASE, json={"title": qid, "course_ids": ["pdf-obj-recovery"]}
                )
                assert q_boot.status_code == 201
                qsbase = f"{BASE}/{q_boot.json()['session_id']}"
                sess = client.get(qsbase).json()
                rev = sess["scope_revision"]
                sc = client.patch(
                    f"{qsbase}/scope",
                    json={
                        "scope_revision": rev,
                        "course_ids": ["pdf-obj-recovery"],
                        "document_ids": [doc_id],
                    },
                )
                assert sc.status_code == 200, sc.text
                rev = sc.json()["scope_revision"]

                # Optional page focus if target_page_hint is given (e.g. Q04 ambiguity on page 2)
                if "target_page_hint" in qitem:
                    cat = client.get(f"{qsbase}/catalog").json()
                    page_no = qitem["target_page_hint"]
                    node = next(
                        n
                        for c in cat["courses"]
                        for d in c["documents"]
                        if d["document_id"] == doc_id
                        for n in d["nodes"]
                        if n.get("locator", {}).get("page") == page_no
                    )
                    foc = client.put(
                        f"{qsbase}/focus", json={"node_id": node["node_id"], "scope_revision": rev}
                    )
                    assert foc.status_code == 200, foc.text
                    rev = foc.json()["scope_revision"]

                resp = client.post(
                    f"{qsbase}/turns",
                    json={
                        "kind": "ask",
                        "client_turn_id": uuid4().hex,
                        "scope_revision": rev,
                        "text": qitem["question"],
                    },
                )
                assert resp.status_code == 200, resp.text
                body = resp.json()
                turn = body["turn"]
                msg = body.get("message") or {}
                trace = turn.get("trace") or {}

                # Reopen citations & displays
                opened_cits: dict[str, Any] = {}
                for blk in msg.get("blocks") or []:
                    for cid in blk.get("citations") or []:
                        if cid in opened_cits:
                            continue
                        ev = client.get(f"{qsbase}/messages/{msg['message_id']}/evidence/{cid}")
                        ev_json = ev.json() if ev.status_code == 200 else {"error": ev.text}
                        opened_cits[cid] = {
                            "http_status": ev.status_code,
                            "file_name": ev_json.get("file_name"),
                            "source": ev_json.get("source"),
                            "locator": ev_json.get("locator"),
                            "has_preview": ev_json.get("has_preview"),
                            "text_head": (ev_json.get("text") or "")[:160],
                        }
                displays: list[dict[str, Any]] = []
                for blk in msg.get("blocks") or []:
                    if blk.get("type") == "source_image":
                        mresp = client.get(blk["media_url"])
                        displays.append(
                            {
                                "evidence_id": blk.get("evidence_id"),
                                "display_source": blk.get("display_source"),
                                "rect_visible_norm": blk.get("rect_visible_norm"),
                                "rendition_kind": blk.get("rendition_kind"),
                                "read_status": blk.get("read_status"),
                                "media_http_status": mresp.status_code,
                                "media_bytes": len(mresp.content) if mresp.is_success else 0,
                            }
                        )

                selected_sources = {
                    k: {
                        "source": s["revision"]["source"],
                        "object_type": s["locator"].get("object_type"),
                        "bbox": s["locator"].get("bbox"),
                        "reads": [
                            {
                                "method": r["method"],
                                "status": r["status"],
                                "text_head": (r.get("text") or "")[:160],
                            }
                            for r in s.get("reads") or []
                        ],
                    }
                    for k, s in (msg.get("sources") or {}).items()
                }

                results.append(
                    {
                        "question_id": qid,
                        "category": qitem["category"],
                        "question": qitem["question"],
                        "turn_status": turn.get("status"),
                        "route": trace.get("route"),
                        "answer_status": trace.get("answer_status"),
                        "clarification": msg.get("clarification"),
                        "object_intent": trace.get("object_intent"),
                        "object_resolution": trace.get("object_resolution"),
                        "vision": trace.get("vision"),
                        "selected_sources": selected_sources,
                        "blocks": msg.get("blocks"),
                        "opened_citations": opened_cits,
                        "displays": displays,
                    }
                )

    out_obj = {
        "git_sha": git_sha,
        "config": {
            "workspace_answer_enabled": False,
            "source_vision_enabled": True,
            "vision_model": "qwen3-vl-plus",
            "workspace_retrieval_strategy": "fts",
        },
        "registered_objects": registered,
        "questions": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=Path("/Users/Link/code/SRAG-pdf-obj"))
    p.add_argument(
        "--criteria",
        type=Path,
        default=Path("/Users/Link/code/SRAG-pdf-obj/docs/G/G-pdf-object-recovery/frozen_criteria.json"),
    )
    p.add_argument("--env-file", type=Path, default=Path("/Users/Link/code/SRAG/.env"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    run_suite(args.repo, args.criteria, args.env_file, args.out)
