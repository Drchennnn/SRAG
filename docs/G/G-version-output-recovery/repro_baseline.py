"""Freeze baseline trajectories on d4c8597 under G01 config/budget for Problems 1 & 2."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_fixtures import generate_all  # noqa: E402

from app.api.workspace import get_workspace_service, router  # noqa: E402
from app.config import Settings  # noqa: E402
from app.rag.workspace import make_workspace  # noqa: E402

COURSE = "public"
HOST = "http://127.0.0.1:8001"
BASE = "/api/workspace"
DEFAULT_ENV = Path("/Users/Link/code/SRAG/.env")


def _snap(turn_id: str, resp_json: dict[str, Any], focus_id: str | None) -> dict[str, Any]:
    turn = resp_json["turn"]
    msg = resp_json.get("message") or {}
    trace = turn.get("trace") or {}
    blocks = msg.get("blocks") or []
    sources = msg.get("sources") or {}
    displays = [
        {
            "display_id": b.get("display_id"),
            "evidence_id": b.get("evidence_id"),
            "display_source": b.get("display_source"),
        }
        for b in blocks
        if b.get("type") == "source_image"
    ]
    actual_reads = [
        {
            "evidence_id": eid,
            "source": ev["revision"]["source"],
            "reads": [
                {
                    "method": r["method"],
                    "status": r["status"],
                    "actual_source": r["actual_source"],
                    "text": r["text"],
                }
                for r in ev.get("reads", [])
            ],
        }
        for eid, ev in sources.items()
    ]
    return {
        "case_id": turn_id,
        "focus_id_sent": focus_id,
        "turn_status": turn.get("status"),
        "turn_error": turn.get("error"),
        "route": trace.get("route"),
        "external_call_count": trace.get("external_call_count"),
        "candidates_count": len(trace.get("candidates", [])),
        "options": turn.get("options", []),
        "resolution": turn.get("resolution"),
        "display_order": displays,
        "visible_display_ids": msg.get("visible_display_ids", []),
        "actual_reads": actual_reads,
        "notices": [
            {"code": b.get("code"), "text": b.get("text")} for b in blocks if b.get("code")
        ],
        "texts": [b.get("text") for b in blocks if b.get("type") == "text"],
        "citations": [
            {"evidence_id": eid, "source": ev["revision"]["source"]}
            for eid, ev in sources.items()
        ],
    }


def run_suite(work_dir: Path, env_file: Path = DEFAULT_ENV) -> dict[str, Any]:
    fixtures = generate_all(work_dir / "fixtures")
    settings = Settings(
        sqlite_path=work_dir / "library.sqlite3",
        data_dir=work_dir / "data",
        course_root=work_dir / "originals",
        workspace_course_ids=COURSE,
        workspace_retrieval_strategy="fts",
        workspace_input_token_limit=32768,
        workspace_answer_enabled=False,
        source_vision_enabled=True,
        deepseek_api_key="",
        qdrant_api_key="",
        qdrant_url="http://127.0.0.1:1",
        _env_file=env_file,
    )
    service = make_workspace(settings)

    vision_calls: list[dict[str, Any]] = []
    orig_process = service.vision.process_current

    def observed_process(request, *, expected_input_sha256=None):
        res = orig_process(request, expected_input_sha256=expected_input_sha256)
        vision_calls.append(
            {
                "source": request.source.model_dump(),
                "question": request.question,
                "status": res.status,
                "external_call_count": res.external_call_count,
                "read_text": res.reads[0].text if res.reads else None,
            }
        )
        return res

    service.vision.process_current = observed_process  # type: ignore[method-assign]
    app = FastAPI()
    app.include_router(router, prefix=BASE)
    app.dependency_overrides[get_workspace_service] = lambda: service

    results: dict[str, Any] = {
        "git_sha": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "config": {
            "workspace_retrieval_strategy": settings.workspace_retrieval_strategy,
            "workspace_input_token_limit": settings.workspace_input_token_limit,
            "workspace_answer_enabled": settings.workspace_answer_enabled,
            "source_vision_enabled": settings.source_vision_enabled,
        },
        "scenarios": {},
        "vision_calls": vision_calls,
    }

    with TestClient(app, base_url=HOST, headers={"Origin": HOST}) as client:

        def new_session() -> str:
            r = client.post(f"{BASE}/sessions", json={"course_ids": [COURSE], "document_ids": []})
            assert r.status_code == 201, r.text
            return r.json()["session_id"]

        def upload(
            sid: str, path: Path, doc_id: str | None = None, name: str = "Bench-Drift.ipynb"
        ) -> dict[str, Any]:
            params = {"course_id": COURSE, "filename": name}
            if doc_id:
                params["document_id"] = doc_id
            r = client.post(
                f"{BASE}/sessions/{sid}/import",
                params=params,
                headers={"content-type": "application/octet-stream", "origin": HOST},
                content=path.read_bytes(),
            )
            assert r.status_code == 200, r.text
            return r.json()

        def ask(sid: str, q: str, rev: int, cid: str, anchor: dict | None = None) -> dict[str, Any]:
            body: dict[str, Any] = {
                "kind": "ask", "text": q, "scope_revision": rev, "client_turn_id": cid
            }
            if anchor is not None:
                body["anchor"] = anchor
            r = client.post(f"{BASE}/sessions/{sid}/turns", json=body)
            assert r.status_code == 200, r.text
            return r.json()

        def focus_cell(sid: str, doc_id: str, cell_index: int, rev: int) -> tuple[str, int]:
            cat = client.get(f"{BASE}/sessions/{sid}/catalog").json()
            stack = list(cat["courses"][0]["documents"])
            node_id = None
            while stack:
                item = stack.pop(0)
                if (
                    item["document_id"] == doc_id
                    and item["kind"] == "cell"
                    and item["locator"].get("cell_index") == cell_index
                ):
                    node_id = item["node_id"]
                    break
                stack.extend(item.get("children", []))
            assert node_id is not None
            r = client.put(
                f"{BASE}/sessions/{sid}/focus", json={"node_id": node_id, "scope_revision": rev}
            )
            assert r.status_code == 200, r.text
            return node_id, r.json()["scope_revision"]

        # --- Problem 2: Focus multi-output cell without specifying which output ---
        sid_p2 = new_session()
        v1 = upload(sid_p2, fixtures["v1"], name="Bench-P2.ipynb")
        cell_nid, rev = focus_cell(sid_p2, v1["document_id"], 1, 1)
        p2_a = ask(sid_p2, "当前这个 Cell 的输出图像横轴是什么？", rev, "p2-img-q")
        p2_b = ask(sid_p2, "当前这个 Cell 的输出结果显示什么？", rev, "p2-txt-q")
        results["scenarios"]["problem2_multi_output_focus"] = {
            "img_question": _snap("p2_img_q", p2_a, cell_nid),
            "out_question": _snap("p2_out_q", p2_b, cell_nid),
        }

        # --- Problem 1: Multi-display turn + single actual read, publish v2 WITHOUT PUT /focus ---
        for intent_name, q_text in (
            ("read_new_counterpart", "最新版里刚才读过的那张图现在的峰值是多少？"),
            ("recall_old_read", "回看刚才读过的旧版那张图的内容。"),
            ("compare_both_versions", "和上一版相比，刚才读过的那张图有什么变化？"),
            ("only_old_control", "只看旧版第一张图，刚才展示的第 1 张图是什么？"),
            ("ordinary_stale_focus", "看当前 Cell"),
        ):
            sid = new_session()
            doc_v1 = upload(sid, fixtures["v1"], name=f"Bench-P1-{intent_name}.ipynb")
            nid_v1, r_v1 = focus_cell(sid, doc_v1["document_id"], 1, 1)
            t1_show = ask(sid, "展示这个 Cell 的两张图", r_v1, f"{intent_name}-show")
            t1_read = ask(sid, "刚才第一张图是什么？", r_v1, f"{intent_name}-read")
            doc_v2 = upload(
                sid, fixtures["v2_same"], doc_id=doc_v1["document_id"],
                name=f"Bench-P1-{intent_name}.ipynb",
            )
            assert doc_v2["version"] == 2
            t2 = ask(sid, q_text, r_v1, f"{intent_name}-after-v2")
            results["scenarios"][f"p1_{intent_name}"] = {
                "turn1_show": _snap("turn1_show", t1_show, nid_v1),
                "turn1_read": _snap("turn1_read", t1_read, nid_v1),
                "turn2_after_v2": _snap(intent_name, t2, nid_v1),
            }

        # Turn 1 has 2 visible displays + 0 vision read -> v2 without PUT /focus
        sid_combo = new_session()
        doc_c1 = upload(sid_combo, fixtures["v1"], name="Bench-P1-combo.ipynb")
        nid_c1, rev_c1 = focus_cell(sid_c1 := sid_combo, doc_c1["document_id"], 1, 1)
        t_show = ask(sid_c1, "展示这个 Cell 的两张图", rev_c1, "combo-show")
        doc_c2 = upload(
            sid_c1, fixtures["v2_same"], doc_id=doc_c1["document_id"], name="Bench-P1-combo.ipynb"
        )
        assert doc_c2["version"] == 2
        t_cmp_after_show = ask(sid_c1, "和上一版相比，刚才那张图有什么变化？", rev_c1, "combo-cmp")
        results["scenarios"]["p1_multi_display_no_read_compare"] = {
            "turn1_show": _snap("combo_show", t_show, nid_c1),
            "turn2_cmp": _snap("combo_cmp", t_cmp_after_show, nid_c1),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="g-ver-base-") as tmp:
        res = run_suite(Path(tmp))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
