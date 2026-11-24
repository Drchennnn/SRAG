#!/usr/bin/env python3
"""Independent G29 Phase 3 blind audit runner against clean archive 6d9fb7d.

Runs:
  Part A: 68-object PDF registration Recall / Precision / FPR audit on 2 unseen real CC BY 4.0 PDFs.
  Part B: All 17 frozen blind cases (PDF, PPTX, dual-version/reordered Notebook, preview symmetry,
          table_crop full-page preview, cross-session 404, stale_reference) using real DeepSeek
          and real qwen3-vl-plus loaded read-only from /Users/Link/code/SRAG/.env.

NEVER modifies production code, NEVER accesses A03 private files, NEVER writes secrets to output.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SUBJECT_DIR = Path(os.environ.get("G29_SUBJECT_DIR", "/tmp/srag-g29-gate-subject-6d9fb7d"))
SUBJECT_SHA = os.environ.get("G29_SUBJECT_SHA", "6d9fb7d45a3dd19424b5fc151900c703abb23462")
OUT_FILE = os.environ.get("G29_OUT_FILE", "G29-blind-audit-results.json")
GATE_DIR = Path("/Users/Link/code/SRAG-g29-gate/docs/G/G29-gate")
FIXTURES = GATE_DIR / "fixtures"
ENV_FILE = Path("/Users/Link/code/SRAG/.env")

# Force module resolution strictly from the clean git archive
sys.path.insert(0, str(SUBJECT_DIR))

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

import app as srag_app  # noqa: E402
import app.rag.parsers.local_slides as local_slides  # noqa: E402
from app.api.workspace import get_workspace_service  # noqa: E402
from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.rag.workspace import WorkspaceService, make_workspace  # noqa: E402

assert str(Path(srag_app.__file__).resolve()).startswith(str(SUBJECT_DIR.resolve())), (
    f"Loaded app from wrong path: {srag_app.__file__}"
)
assert str(Path(local_slides.__file__).resolve()).startswith(str(SUBJECT_DIR.resolve())), (
    f"Loaded local_slides from wrong path: {local_slides.__file__}"
)

HOST_ORIGIN = "http://localhost"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_env_readonly() -> dict[str, str]:
    env: dict[str, str] = {}
    if ENV_FILE.exists():
        for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def _bbox_contain(inner: list[float], outer: list[float]) -> float:
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ai = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return inter / ai if ai > 0 else 0.0


def _token_overlap(gt_text: str, pred_text: str) -> float:
    toks_gt = set(re.findall(r"[a-z0-9]+", gt_text.lower()))
    toks_pr = set(re.findall(r"[a-z0-9]+", pred_text.lower()))
    if not toks_gt:
        return 0.0
    return len(toks_gt & toks_pr) / len(toks_gt)


def run_part_a_pdf_audit(criteria: dict[str, Any]) -> dict[str, Any]:
    """Audit all 68 ground-truth objects across both unseen real CC BY 4.0 PDFs."""
    inv = criteria["pdf_object_registration_audit"]["ground_truth_inventory"]
    audit: dict[str, Any] = {"per_pdf": {}, "by_class": {}, "traps": []}

    totals = {
        k: {"gt": 0, "tp": 0, "fp": 0, "fn": 0, "optional_ignored": 0}
        for k in ("visual_panels", "tables", "figure_captions", "numbered_equations")
    }

    for pdf_name in ("Speculative-Decoding-2211.17192v2.pdf", "YaRN-2309.00071v3.pdf"):
        gt = inv[pdf_name]
        parsed = local_slides.parse_local_slides(FIXTURES / pdf_name)
        objs = [s for s in parsed.sources if s.kind == "page_object"]
        pdf_res: dict[str, Any] = {
            "total_registered_page_objects": len(objs),
            "classes": {},
            "sample_failures": [],
        }

        # 1. Visual panels
        reg_imgs = [s for s in objs if s.locator.get("object_type") == "image"]
        matched_gt = set()
        tp_img, fp_img = 0, 0
        for s in reg_imgs:
            p = s.locator.get("page")
            bb = s.locator.get("bbox") or [0, 0, 0, 0]
            hit = None
            for g in gt["visual_panels"]:
                if g["page"] == p and (_bbox_iou(bb, g["bbox"]) >= 0.45 or _bbox_contain(g["bbox"], bb) >= 0.65):
                    hit = g["id"]
                    break
            if hit and hit not in matched_gt:
                matched_gt.add(hit)
                tp_img += 1
            else:
                fp_img += 1
                pdf_res["sample_failures"].append({"class": "image", "kind": "FP", "source_id": s.source_id, "page": p, "bbox": bb})
        fn_img = len(gt["visual_panels"]) - tp_img
        pdf_res["classes"]["visual_panels"] = {"gt": len(gt["visual_panels"]), "registered": len(reg_imgs), "tp": tp_img, "fp": fp_img, "fn": fn_img}

        # 2. Tables
        reg_tabs = [s for s in objs if s.locator.get("object_type") == "table"]
        gt_tabs_by_page: dict[int, list[dict[str, Any]]] = {}
        for g in gt["tables"]:
            gt_tabs_by_page.setdefault(g["page"], []).append(g)
        tp_tab, fp_tab = 0, 0
        used_tabs: dict[int, int] = {}
        for s in reg_tabs:
            p = s.locator.get("page")
            avail = len(gt_tabs_by_page.get(p, []))
            used = used_tabs.get(p, 0)
            if used < avail:
                used_tabs[p] = used + 1
                tp_tab += 1
            else:
                fp_tab += 1
                pdf_res["sample_failures"].append({"class": "table", "kind": "FP", "source_id": s.source_id, "page": p})
        fn_tab = len(gt["tables"]) - tp_tab
        pdf_res["classes"]["tables"] = {"gt": len(gt["tables"]), "registered": len(reg_tabs), "tp": tp_tab, "fp": fp_tab, "fn": fn_tab}

        # 3. Figure Captions
        reg_caps = [s for s in objs if s.locator.get("object_type") == "caption"]
        matched_caps = set()
        tp_cap, fp_cap = 0, 0
        for s in reg_caps:
            p = s.locator.get("page")
            txt = (s.locator.get("object_text") or s.text or "").strip()
            hit = None
            best_ov = 0.0
            for g in gt["figure_captions"]:
                if g["page"] != p:
                    continue
                # Extract Figure N prefix
                m_gt = re.search(r"Figure\s+\d+", g["anchor"])
                m_pr = re.search(r"Figure\s+\d+", txt)
                ov = _token_overlap(g["anchor"], txt)
                if m_gt and m_pr and m_gt.group(0) == m_pr.group(0) and ov >= 0.60:
                    if ov > best_ov:
                        best_ov = ov
                        hit = g["id"]
            if hit and hit not in matched_caps:
                matched_caps.add(hit)
                tp_cap += 1
            else:
                fp_cap += 1
                pdf_res["sample_failures"].append({
                    "class": "caption",
                    "kind": "FP_or_truncated",
                    "source_id": s.source_id,
                    "page": p,
                    "bbox": [round(v, 3) for v in (s.locator.get("bbox") or [])],
                    "text_excerpt": txt[:100],
                })
        fn_cap = len(gt["figure_captions"]) - tp_cap
        for g in gt["figure_captions"]:
            if g["id"] not in matched_caps:
                pdf_res["sample_failures"].append({"class": "caption", "kind": "FN_missed_or_truncated", "gt_id": g["id"], "page": g["page"]})
        pdf_res["classes"]["figure_captions"] = {"gt": len(gt["figure_captions"]), "registered": len(reg_caps), "tp": tp_cap, "fp": fp_cap, "fn": fn_cap}

        # 4. Numbered Standalone Equations
        reg_eqs = [s for s in objs if s.locator.get("object_type") == "formula"]
        if pdf_name == "Speculative-Decoding-2211.17192v2.pdf":
            gt_eq_count = len(gt["numbered_equations"])
            # Check Eq. (1) on p3: must contain both LHS and RHS or (1)
            tp_eq, fp_eq, opt_eq = 0, 0, 0
            matched_eq1 = False
            for s in reg_eqs:
                p = s.locator.get("page")
                txt = (s.locator.get("object_text") or s.text or "").strip()
                # Check if matches Eq. (1) cleanly (has E(# generated tokens) AND (1 - alpha) or (1))
                if p == 3 and "generated tokens" in txt and ("1 − α" in txt or "1 - α" in txt or "(1)" in txt):
                    if not matched_eq1:
                        matched_eq1 = True
                        tp_eq += 1
                    else:
                        fp_eq += 1
                elif p in (4, 11, 12) and len(txt) >= 24 and "=" in txt and not txt.startswith(("=", ")")) and "Algorithm" not in txt:
                    # Per frozen Phase 1 criteria: intact unnumbered display equations on p4/p11/p12 are allow_optional
                    opt_eq += 1
                else:
                    fp_eq += 1
                    pdf_res["sample_failures"].append({
                        "class": "formula",
                        "kind": "FP_or_broken_fragment",
                        "source_id": s.source_id,
                        "page": p,
                        "bbox": [round(v, 3) for v in (s.locator.get("bbox") or [])],
                        "text_excerpt": txt[:90],
                    })
            fn_eq = gt_eq_count - tp_eq
        else:
            gt_by_page = gt["numbered_equations_by_page"]
            gt_eq_count = sum(len(v) for v in gt_by_page.values())
            tp_eq, fp_eq, opt_eq = 0, 0, 0
            covered_eq_nums: set[str] = set()
            used_per_page: dict[int, int] = {}
            for s in reg_eqs:
                p = s.locator.get("page")
                txt = (s.locator.get("object_text") or s.text or "").strip()
                looks_truncated = (
                    len(txt) < 10
                    or txt.startswith(("=", ")", "D|", "16k-token", "ppl(t)", "Let "))
                    or txt.endswith(("=", "ln(s", "·", "−", "fW", "x", "\uf8f4"))
                    or "mparing" in txt
                )
                page_nums = gt_by_page.get(str(p), [])
                matched_nums = [num for num in page_nums if num in txt]
                avail = len(page_nums)
                used = used_per_page.get(p, 0)
                if matched_nums and not looks_truncated:
                    # Exact numbered equation match (may cover 1 or 2 paired align equations)
                    new_nums = [n for n in matched_nums if n not in covered_eq_nums]
                    if new_nums:
                        covered_eq_nums.update(new_nums)
                        used_per_page[p] = used + len(new_nums)
                        tp_eq += len(new_nums)
                    else:
                        fp_eq += 1
                elif avail > 0 and not looks_truncated and used < avail and "=" in txt and not txt.startswith("Wq,Wk"):
                    used_per_page[p] = used + 1
                    tp_eq += 1
                else:
                    fp_eq += 1
                    pdf_res["sample_failures"].append({
                        "class": "formula",
                        "kind": "FP_or_sliced_fragment",
                        "source_id": s.source_id,
                        "page": p,
                        "bbox": [round(v, 3) for v in (s.locator.get("bbox") or [])],
                        "text_excerpt": txt[:90],
                    })
            fn_eq = gt_eq_count - tp_eq

        pdf_res["classes"]["numbered_equations"] = {
            "gt": gt_eq_count,
            "registered": len(reg_eqs),
            "tp": tp_eq,
            "fp": fp_eq,
            "fn": fn_eq,
            "optional_ignored": opt_eq,
        }

        for k in totals:
            for m in ("gt", "tp", "fp", "fn"):
                totals[k][m] += pdf_res["classes"][k][m]
            totals[k]["optional_ignored"] += pdf_res["classes"][k].get("optional_ignored", 0)

        audit["per_pdf"][pdf_name] = pdf_res

    # Compute per-class & overall micro metrics
    micro_gt = sum(v["gt"] for v in totals.values())
    micro_tp = sum(v["tp"] for v in totals.values())
    micro_fp = sum(v["fp"] for v in totals.values())
    micro_fn = sum(v["fn"] for v in totals.values())

    for k, v in totals.items():
        rec = v["tp"] / v["gt"] if v["gt"] else 1.0
        prec = v["tp"] / (v["tp"] + v["fp"]) if (v["tp"] + v["fp"]) else 0.0
        audit["by_class"][k] = {
            **v,
            "recall": round(rec, 4),
            "precision": round(prec, 4),
            "fpr": round(1.0 - prec, 4) if (v["tp"] + v["fp"]) else 0.0,
        }

    micro_rec = micro_tp / micro_gt if micro_gt else 0.0
    micro_prec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    audit["micro_overall"] = {
        "gt": micro_gt,
        "tp": micro_tp,
        "fp": micro_fp,
        "fn": micro_fn,
        "recall": round(micro_rec, 4),
        "precision": round(micro_prec, 4),
        "fpr": round(1.0 - micro_prec, 4),
    }

    # Check red-line triggers (< 70%)
    red_lines = []
    if micro_rec < 0.70:
        red_lines.append(f"overall_micro_recall={micro_rec:.1%} < 70%")
    if micro_prec < 0.70:
        red_lines.append(f"overall_micro_precision={micro_prec:.1%} < 70% (FPR={1-micro_prec:.1%} > 30%)")
    if audit["by_class"]["visual_panels"]["recall"] < 0.70:
        red_lines.append("visual_panels_recall < 70%")
    if audit["by_class"]["tables"]["recall"] < 0.70:
        red_lines.append("tables_recall < 70%")
    if audit["by_class"]["figure_captions"]["precision"] < 0.70:
        red_lines.append(f"figure_captions_precision={audit['by_class']['figure_captions']['precision']:.1%} < 70%")
    if audit["by_class"]["numbered_equations"]["precision"] < 0.70:
        red_lines.append(f"numbered_equations_precision={audit['by_class']['numbered_equations']['precision']:.1%} < 70%")
    if audit["by_class"]["numbered_equations"]["recall"] < 0.70:
        red_lines.append(f"numbered_equations_recall={audit['by_class']['numbered_equations']['recall']:.1%} < 70%")
    audit["red_line_triggered"] = bool(red_lines)
    audit["red_line_reasons"] = red_lines
    return audit


def _make_client(work_dir: Path, env: dict[str, str], *, answer_enabled: bool, vision_enabled: bool) -> tuple[TestClient, WorkspaceService]:
    settings = Settings(
        _env_file=None,
        data_dir=work_dir / "data",
        sqlite_path=work_dir / "library.sqlite3",
        course_root=work_dir / "courses",
        workspace_course_ids="g29-blind",
        workspace_answer_enabled=answer_enabled,
        source_vision_enabled=vision_enabled,
        deepseek_api_key=env.get("DEEPSEEK_API_KEY", ""),
        deepseek_base_url=env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_query_model=env.get("DEEPSEEK_QUERY_MODEL", "deepseek-chat"),
        dashscope_api_key=env.get("DASHSCOPE_API_KEY", ""),
        dashscope_base_url=env.get("DASHSCOPE_BASE_URL", ""),
        dashscope_vision_model=env.get("DASHSCOPE_VISION_MODEL", "qwen3-vl-plus"),
    )
    app = create_app()
    svc = make_workspace(settings)
    app.dependency_overrides[get_workspace_service] = lambda: svc
    client = TestClient(app, base_url=HOST_ORIGIN)
    return client, svc


def _create_session(client: TestClient) -> tuple[str, int]:
    r = client.post(
        "/api/workspace/sessions",
        json={"course_ids": ["g29-blind"], "document_ids": []},
        headers={"origin": HOST_ORIGIN},
    )
    assert r.status_code == 201, r.text
    d = r.json()
    return d["session_id"], d["scope_revision"]


def _import_file(client: TestClient, sid: str, path: Path, document_id: str | None = None) -> dict[str, Any]:
    params = {"course_id": "g29-blind", "filename": path.name}
    if document_id:
        params["document_id"] = document_id
    r = client.post(
        f"/api/workspace/sessions/{sid}/import",
        params=params,
        content=path.read_bytes(),
        headers={"origin": HOST_ORIGIN, "content-type": "application/octet-stream"},
    )
    assert r.status_code == 200, r.text
    return r.json()


def _set_scope(client: TestClient, sid: str, doc_ids: list[str]) -> int:
    s = client.get(f"/api/workspace/sessions/{sid}").json()
    r = client.patch(
        f"/api/workspace/sessions/{sid}/scope",
        json={"course_ids": ["g29-blind"], "document_ids": doc_ids, "scope_revision": s["scope_revision"]},
        headers={"origin": HOST_ORIGIN},
    )
    assert r.status_code == 200, r.text
    return r.json()["scope_revision"]


def _find_catalog_node(client: TestClient, sid: str, doc_id: str, source_id: str) -> str:
    cat = client.get(f"/api/workspace/sessions/{sid}/catalog").json()
    want_kind, want_key, want_val = None, None, None
    if source_id.startswith("page:"):
        want_kind, want_key, want_val = "page", "page", int(source_id.split(":")[1])
    elif source_id.startswith("cell:"):
        want_kind, want_key, want_val = "cell", "cell_index", int(source_id.split(":")[1])

    def walk(nodes):
        for n in nodes:
            if n.get("document_id") == doc_id and n.get("kind") == want_kind:
                loc = n.get("locator") or {}
                if loc.get(want_key) == want_val:
                    return n["node_id"]
            if n.get("children"):
                found = walk(n["children"])
                if found:
                    return found
        return None

    for course in cat.get("courses", []):
        for doc in course.get("documents", []):
            if doc.get("document_id") == doc_id:
                found = walk(doc.get("children", []))
                if found:
                    return found
    raise RuntimeError(f"Catalog node not found: {doc_id} {source_id}")


def _put_focus(client: TestClient, sid: str, node_id: str) -> dict[str, Any]:
    s = client.get(f"/api/workspace/sessions/{sid}").json()
    r = client.put(
        f"/api/workspace/sessions/{sid}/focus",
        json={"node_id": node_id, "scope_revision": s["scope_revision"]},
        headers={"origin": HOST_ORIGIN},
    )
    assert r.status_code == 200, r.text
    return r.json()


def _ask(client: TestClient, sid: str, text: str, turn_key: str, anchor: dict[str, Any] | None = None) -> dict[str, Any]:
    s = client.get(f"/api/workspace/sessions/{sid}").json()
    payload: dict[str, Any] = {
        "client_turn_id": turn_key,
        "kind": "ask",
        "text": text,
        "scope_revision": s["scope_revision"],
    }
    if anchor is not None:
        payload["anchor"] = anchor
    t0 = time.monotonic()
    r = client.post(f"/api/workspace/sessions/{sid}/turns", json=payload, headers={"origin": HOST_ORIGIN})
    elapsed = round(time.monotonic() - t0, 2)
    return {"http_status": r.status_code, "elapsed_s": elapsed, "body": r.json()}


def _check_endpoints(client: TestClient, sid: str, msg: dict[str, Any] | None) -> dict[str, Any]:
    """Check /media/{did}, /evidence/{eid}, and /evidence/{eid}/preview symmetry for a message."""
    if not msg:
        return {"media_checks": [], "evidence_checks": [], "symmetry_checks": []}
    mid = msg["message_id"]
    blocks = msg.get("blocks", [])
    sources = msg.get("sources", {})
    visible = set(msg.get("visible_display_ids") or [])
    report: dict[str, Any] = {"media_checks": [], "evidence_checks": [], "symmetry_checks": []}

    img_by_eid: dict[str, dict[str, Any]] = {}
    for b in blocks:
        if b.get("type") == "source_image" and b.get("display_id") in visible:
            did = b["display_id"]
            eid = b["evidence_id"]
            img_by_eid[eid] = b
            rm = client.get(f"/api/workspace/sessions/{sid}/messages/{mid}/media/{did}?view=inline")
            img_sha = _sha256(rm.content) if rm.status_code == 200 else None
            w, h = (Image.open(io.BytesIO(rm.content)).size if rm.status_code == 200 and rm.headers.get("content-type", "").startswith("image/") else (0, 0))
            report["media_checks"].append({
                "display_id": did,
                "evidence_id": eid,
                "rendition_kind": b.get("rendition_kind"),
                "status": rm.status_code,
                "content_type": rm.headers.get("content-type"),
                "sha256": img_sha,
                "size": [w, h],
            })

    cited_eids = []
    for b in blocks:
        if b.get("type") == "text":
            for c in b.get("citations", []):
                eid = c if isinstance(c, str) else c.get("evidence_id")
                if eid and eid not in cited_eids:
                    cited_eids.append(eid)
    all_eids = list(dict.fromkeys(list(img_by_eid.keys()) + cited_eids))
    for eid in all_eids:
        ev = sources.get(eid) or {}
        reads = ev.get("reads") or []
        has_native = any(r.get("method") in {"native_text", "native_table", "ocr"} and r.get("text") for r in reads)
        has_table = any(r.get("method") == "native_table" and r.get("text") for r in reads)
        vis_reads = [r for r in reads if r.get("method") == "vision"]

        re_ev = client.get(f"/api/workspace/sessions/{sid}/messages/{mid}/evidence/{eid}")
        re_pv = client.get(f"/api/workspace/sessions/{sid}/messages/{mid}/evidence/{eid}/preview")
        pv_sha = _sha256(re_pv.content) if re_pv.status_code == 200 else None
        pw, ph = (Image.open(io.BytesIO(re_pv.content)).size if re_pv.status_code == 200 and re_pv.headers.get("content-type", "").startswith("image/") else (0, 0))

        report["evidence_checks"].append({
            "evidence_id": eid,
            "source": (ev.get("revision") or {}).get("source"),
            "has_native_read": has_native,
            "has_native_table_read": has_table,
            "vision_statuses": [r.get("status") for r in vis_reads],
            "vision_input_sha256": [r.get("actual_input_sha256") for r in vis_reads],
            "evidence_http": re_ev.status_code,
            "evidence_text_len": len(re_ev.json().get("text", "")) if re_ev.status_code == 200 else 0,
            "preview_http": re_pv.status_code,
            "preview_sha256": pv_sha,
            "preview_size": [pw, ph],
        })

        if eid in img_by_eid:
            mrec = next(x for x in report["media_checks"] if x["evidence_id"] == eid)
            rk = img_by_eid[eid].get("rendition_kind")
            if has_table:
                # table_crop must return full original_page on /preview (larger than crop or matching ?view=page)
                rm_page = client.get(f"/api/workspace/sessions/{sid}/messages/{mid}/media/{mrec['display_id']}?view=page")
                page_sha = _sha256(rm_page.content) if rm_page.status_code == 200 else None
                report["symmetry_checks"].append({
                    "evidence_id": eid,
                    "mode": "table_crop_full_page_preview",
                    "media_inline_200": mrec["status"] == 200,
                    "preview_200": re_pv.status_code == 200,
                    "preview_matches_full_page": pv_sha == page_sha,
                    "crop_size": mrec["size"],
                    "preview_size": [pw, ph],
                })
            else:
                report["symmetry_checks"].append({
                    "evidence_id": eid,
                    "mode": f"inline_symmetry({rk})",
                    "media_200": mrec["status"] == 200,
                    "preview_200": re_pv.status_code == 200,
                    "sha_match": mrec["sha256"] == pv_sha and pv_sha is not None,
                    "vision_sha_match": (not vis_reads) or all(r.get("actual_input_sha256") == pv_sha for r in vis_reads if r.get("actual_input_sha256")),
                })
    return report


def _summarize_turn(resp: dict[str, Any], checks: dict[str, Any] | None = None) -> dict[str, Any]:
    body = resp["body"] or {}
    turn = body.get("turn") or {}
    msg = body.get("message") or {}
    texts = [b.get("text", "") for b in msg.get("blocks", []) if b.get("type") in {"text", "notice"}]
    codes = [b.get("code") for b in msg.get("blocks", []) if b.get("type") == "notice"]
    return {
        "http_status": resp["http_status"],
        "elapsed_s": resp["elapsed_s"],
        "turn_id": turn.get("turn_id"),
        "turn_status": turn.get("status"),
        "turn_error": turn.get("error"),
        "options": body.get("options") or turn.get("options") or [],
        "notice_codes": codes,
        "answer_text": "\n".join(texts)[:1500],
        "trace": turn.get("trace"),
        "checks": checks,
    }


def run_part_b_cases(env: dict[str, str]) -> dict[str, Any]:
    results: dict[str, Any] = {"product_mode": {}, "no_gen_mode": {}}

    # ==================== 1. PRODUCT MODE (DeepSeek + qwen3-vl-plus) ====================
    tmp_prod = Path(tempfile.mkdtemp(prefix="srag-g29-blind-prod-"))
    try:
        client, _ = _make_client(tmp_prod, env, answer_enabled=True, vision_enabled=True)
        sid, _ = _create_session(client)

        sd_imp = _import_file(client, sid, FIXTURES / "Speculative-Decoding-2211.17192v2.pdf")
        yarn_imp = _import_file(client, sid, FIXTURES / "YaRN-2309.00071v3.pdf")
        pptx_imp = _import_file(client, sid, FIXTURES / "Week06-Thermal-Cache-Lab.pptx")
        sd_id, yarn_id, pptx_id = sd_imp["document_id"], yarn_imp["document_id"], pptx_imp["document_id"]

        # --- G29-PDF-01: Speculative Decoding p5 right vector figure ---
        _set_scope(client, sid, [sd_id])
        r = _ask(client, sid, "第 5 页右边的图画了哪两类曲线？当 α = 0.90 且 γ = 10 时，这两条曲线对应的纵轴数值大约分别是多少？", "pdf-01")
        results["product_mode"]["G29-PDF-01"] = _summarize_turn(r, _check_endpoints(client, sid, r["body"]["message"]))

        # --- G29-PDF-02: Q04 class (focus p5, ask without page number -> clarify -> select left Figure 3) ---
        p5_node = _find_catalog_node(client, sid, sd_id, "page:5")
        _put_focus(client, sid, p5_node)
        r1 = _ask(client, sid, "这张图的纵轴表示什么？不同曲线对应什么参数？", "pdf-02-t1")
        opts = r1["body"].get("options") or r1["body"]["turn"].get("options") or []
        t2_sum = None
        if len(opts) >= 2:
            left_opt = next((o for o in opts if "左图" in o["label"]), opts[0])
            r2 = _ask(
                client, sid, left_opt["label"], "pdf-02-t2",
                anchor={"type": "clarification", "turn_id": r1["body"]["turn"]["turn_id"], "option_id": left_opt["option_id"]},
            )
            t2_sum = _summarize_turn(r2, _check_endpoints(client, sid, r2["body"]["message"]))
        results["product_mode"]["G29-PDF-02"] = {"turn1": _summarize_turn(r1), "turn2": t2_sum}

        # --- G29-PDF-04: Figure 5 caption + GET /evidence/{eid} 200 ---
        sid_sd2, _ = _create_session(client)
        _set_scope(client, sid_sd2, [sd_id])
        r = _ask(client, sid_sd2, "第 6 页 Figure 5 的图注如何说明顶行、中间行和底行的配置，以及紫色、蓝色、黄色和橙色块各代表哪个模型？", "pdf-04")
        results["product_mode"]["G29-PDF-04"] = _summarize_turn(r, _check_endpoints(client, sid_sd2, r["body"]["message"]))

        # --- G29-PDF-05: Eq (1) p3 + GET /evidence/{eid} 200 ---
        r = _ask(client, sid_sd2, "第 3 页公式 (1) 给出的期望生成 token 数表达式是什么？该公式依赖什么前提假设？", "pdf-05")
        results["product_mode"]["G29-PDF-05"] = _summarize_turn(r, _check_endpoints(client, sid_sd2, r["body"]["message"]))

        # --- G29-PDF-07a & 07b: Booktabs Tables p5 & p6 (also verify table_crop preview!) ---
        r7a = _ask(client, sid_sd2, "第 5 页 Table 1 是在什么假设条件下统计的？当 α = 0.8、γ = 5 和 α = 0.9、γ = 10 时，OPERATIONS 和 SPEED 分别是多少？", "pdf-07a")
        r7b = _ask(client, sid_sd2, "打开第 6 页的表 1，并指出 EnDe 任务在 temp=0 时 T5-small 与 T5-large 的 α 和 SPEED 分别是多少？", "pdf-07b")
        results["product_mode"]["G29-PDF-07a"] = _summarize_turn(r7a, _check_endpoints(client, sid_sd2, r7a["body"]["message"]))
        results["product_mode"]["G29-PDF-07b"] = _summarize_turn(r7b, _check_endpoints(client, sid_sd2, r7b["body"]["message"]))

        # --- G29-PDF-03a & 03b: YaRN p8 right plot & p7 focused ambiguity ---
        sid_y, _ = _create_session(client)
        _set_scope(client, sid_y, [yarn_id])
        r3a = _ask(client, sid_y, "第 8 页右边的图纵轴指标是什么？当 Context Size 超过 25000 时，NTK-aware 曲线的变化趋势是什么？", "pdf-03a")
        results["product_mode"]["G29-PDF-03a"] = _summarize_turn(r3a, _check_endpoints(client, sid_y, r3a["body"]["message"]))

        p7_node = _find_catalog_node(client, sid_y, yarn_id, "page:7")
        _put_focus(client, sid_y, p7_node)
        r3b1 = _ask(client, sid_y, "当前页这张图的纵轴 Loss 刻度范围是从多少到多少？", "pdf-03b-t1")
        opts3b = r3b1["body"].get("options") or r3b1["body"]["turn"].get("options") or []
        r3b2_sum = None
        if len(opts3b) >= 2:
            right_opt = next((o for o in opts3b if "右图" in o["label"]), opts3b[1])
            r3b2 = _ask(
                client, sid_y, right_opt["label"], "pdf-03b-t2",
                anchor={"type": "clarification", "turn_id": r3b1["body"]["turn"]["turn_id"], "option_id": right_opt["option_id"]},
            )
            r3b2_sum = _summarize_turn(r3b2, _check_endpoints(client, sid_y, r3b2["body"]["message"]))
        results["product_mode"]["G29-PDF-03b"] = {"turn1": _summarize_turn(r3b1), "turn2": r3b2_sum}

        # --- G29-PDF-06a & 06b: YaRN Eq (11) & Eq (15) + /evidence 200 ---
        sid_y2, _ = _create_session(client)
        _set_scope(client, sid_y2, [yarn_id])
        r6a = _ask(client, sid_y2, "第 5 页公式 (11) 的分段函数 γ(r) 在 r < α、r > β 和中间区间分别取什么值？论文对 Llama 系列推荐的 α 和 β 是多少？", "pdf-06a")
        r6b = _ask(client, sid_y2, "第 6 页公式 (15) 推荐的温度缩放公式是什么？它适用于哪些模型，且是通过哪几组模型拟合得到的？", "pdf-06b")
        results["product_mode"]["G29-PDF-06a"] = _summarize_turn(r6a, _check_endpoints(client, sid_y2, r6a["body"]["message"]))
        results["product_mode"]["G29-PDF-06b"] = _summarize_turn(r6b, _check_endpoints(client, sid_y2, r6b["body"]["message"]))

        # --- G29-PDF-08a & 08b: YaRN Table 1 & Table 4 ---
        r8a = _ask(client, sid_y2, "第 8 页 Table 1 中，13B 模型在 4k×16（400步）与 4k×32（400+200步）两种配置下，评估窗口为 65536 和 131072 时的困惑度分别是多少？", "pdf-08a")
        r8b = _ask(client, sid_y2, "第 9 页 Table 4 对比了不同扩展方法的 A100 训练时间，Llama 2 YaRN（4k×32, 128k）、(Together.ai, 2023) 和 (Xiong et al., 2023) 的训练时间分别是多少？", "pdf-08b")
        results["product_mode"]["G29-PDF-08a"] = _summarize_turn(r8a, _check_endpoints(client, sid_y2, r8a["body"]["message"]))
        results["product_mode"]["G29-PDF-08b"] = _summarize_turn(r8b, _check_endpoints(client, sid_y2, r8b["body"]["message"]))

        # --- G29-PPTX-01..03 ---
        sid_p, _ = _create_session(client)
        _set_scope(client, sid_p, [pptx_id])
        rp1 = _ask(client, sid_p, "第 2 页表格里哪个负载的 L2 Miss Rate 最高、数值是多少？哪个负载 Peak Tj 最高，其 Sustained Clock 和 Energy-Delay（含单位）分别是多少？", "pptx-01")
        rp2 = _ask(client, sid_p, "第 3 页实验图的横轴和纵轴分别是什么？在 8 entries 拐点处 LRU 和 FIFO 的 L2 Miss Rate 分别是多少？", "pptx-02")
        rp3 = _ask(client, sid_p, "看第 4 页的架构图，DVFS 的 P0、P1、P2 三个档位对应的电压轨分别是多少伏？热传感器的采样周期 dt 是多少？", "pptx-03")
        results["product_mode"]["G29-PPTX-01"] = _summarize_turn(rp1, _check_endpoints(client, sid_p, rp1["body"]["message"]))
        results["product_mode"]["G29-PPTX-02"] = _summarize_turn(rp2, _check_endpoints(client, sid_p, rp2["body"]["message"]))
        results["product_mode"]["G29-PPTX-03"] = _summarize_turn(rp3, _check_endpoints(client, sid_p, rp3["body"]["message"]))

        # --- G29-NB-01: v1 -> v2 without PUT /focus ---
        sid_nb1, _ = _create_session(client)
        nb_v1_imp = _import_file(client, sid_nb1, FIXTURES / "notebook/v1/Lab06-Thermal-Sweep.ipynb")
        nb1_doc = nb_v1_imp["document_id"]
        _set_scope(client, sid_nb1, [nb1_doc])
        c1_node = _find_catalog_node(client, sid_nb1, nb1_doc, "cell:1")
        _put_focus(client, sid_nb1, c1_node)
        r_v1 = _ask(client, sid_nb1, "第 2 个 Cell 第 1 张图的横轴单位和峰值温度分别是多少？", "nb01-v1")
        v1_checks = _check_endpoints(client, sid_nb1, r_v1["body"]["message"])
        # Upgrade to v2 WITHOUT PUT /focus
        _import_file(client, sid_nb1, FIXTURES / "notebook/v2/Lab06-Thermal-Sweep.ipynb", document_id=nb1_doc)
        r_nb1a = _ask(client, sid_nb1, "换版后最新版刚才读过的这张图的横轴单位和峰值温度变成了什么？", "nb01-a")
        r_nb1b = _ask(client, sid_nb1, "只看旧版刚才读过的那张图，它的横轴单位和峰值温度是多少？", "nb01-b")
        r_nb1c = _ask(client, sid_nb1, "对比最新版和旧版刚才读过的这张图，横轴单位和峰值温度发生了什么变化？", "nb01-c")
        results["product_mode"]["G29-NB-01"] = {
            "v1_read": _summarize_turn(r_v1, v1_checks),
            "read_new_no_focus": _summarize_turn(r_nb1a, _check_endpoints(client, sid_nb1, r_nb1a["body"]["message"])),
            "old_only_control": _summarize_turn(r_nb1b, _check_endpoints(client, sid_nb1, r_nb1b["body"]["message"])),
            "bilateral_compare": _summarize_turn(r_nb1c, _check_endpoints(client, sid_nb1, r_nb1c["body"]["message"])),
        }

        # --- G29-NB-02: v1 -> v2_reordered (b2 bilateral recovery!) ---
        sid_nb2, _ = _create_session(client)
        nb_v1_imp2 = _import_file(client, sid_nb2, FIXTURES / "notebook/v1/Lab06-Thermal-Sweep.ipynb")
        nb2_doc = nb_v1_imp2["document_id"]
        _set_scope(client, sid_nb2, [nb2_doc])
        c1_node2 = _find_catalog_node(client, sid_nb2, nb2_doc, "cell:1")
        _put_focus(client, sid_nb2, c1_node2)
        r_b2_v1 = _ask(client, sid_nb2, "第 2 个 Cell 第 1 张图的横轴单位和峰值温度分别是多少？", "nb02-v1")
        # Upgrade to v2_reordered WITHOUT PUT /focus
        _import_file(client, sid_nb2, FIXTURES / "notebook/v2_reordered/Lab06-Thermal-Sweep.ipynb", document_id=nb2_doc)
        r_b2_t1 = _ask(client, sid_nb2, "对比最新版和旧版刚才读过的这张图，横轴单位和峰值温度有什么变化？", "nb02-t1")
        opts_b2 = r_b2_t1["body"].get("options") or r_b2_t1["body"]["turn"].get("options") or []
        r_b2_t2_sum = None
        if len(opts_b2) >= 2:
            # Temp chart is output 3 in v2_reordered
            temp_opt = next((o for o in opts_b2 if "输出 3" in o["label"]), opts_b2[-1])
            r_b2_t2 = _ask(
                client, sid_nb2, temp_opt["label"], "nb02-t2",
                anchor={"type": "clarification", "turn_id": r_b2_t1["body"]["turn"]["turn_id"], "option_id": temp_opt["option_id"]},
            )
            r_b2_t2_sum = _summarize_turn(r_b2_t2, _check_endpoints(client, sid_nb2, r_b2_t2["body"]["message"]))
        results["product_mode"]["G29-NB-02"] = {
            "v1_read": _summarize_turn(r_b2_v1, _check_endpoints(client, sid_nb2, r_b2_v1["body"]["message"])),
            "reordered_clarify": _summarize_turn(r_b2_t1),
            "bilateral_recovered": r_b2_t2_sum,
        }

        # --- G29-NB-03: Focused multi-output Cell clarification (text vs image) ---
        sid_nb3, _ = _create_session(client)
        nb_v2_imp3 = _import_file(client, sid_nb3, FIXTURES / "notebook/v2/Lab06-Thermal-Sweep.ipynb")
        nb3_doc = nb_v2_imp3["document_id"]
        _set_scope(client, sid_nb3, [nb3_doc])
        c1_node3 = _find_catalog_node(client, sid_nb3, nb3_doc, "cell:1")
        _put_focus(client, sid_nb3, c1_node3)
        r_nb3_t1 = _ask(client, sid_nb3, "查看当前单元格输出里的关键指标和峰值结果", "nb03-t1")
        opts_nb3 = r_nb3_t1["body"].get("options") or r_nb3_t1["body"]["turn"].get("options") or []
        nb3_text_sum, nb3_img_sum = None, None
        if opts_nb3:
            txt_opt = next((o for o in opts_nb3 if "输出 0" in o["label"]), None)
            img_opt = next((o for o in opts_nb3 if "输出 2" in o["label"]), None)
            if txt_opt:
                r_txt = _ask(
                    client, sid_nb3, txt_opt["label"], "nb03-t2-text",
                    anchor={"type": "clarification", "turn_id": r_nb3_t1["body"]["turn"]["turn_id"], "option_id": txt_opt["option_id"]},
                )
                nb3_text_sum = _summarize_turn(r_txt, _check_endpoints(client, sid_nb3, r_txt["body"]["message"]))
            if img_opt:
                r_img = _ask(
                    client, sid_nb3, img_opt["label"], "nb03-t2-img",
                    anchor={"type": "clarification", "turn_id": r_nb3_t1["body"]["turn"]["turn_id"], "option_id": img_opt["option_id"]},
                )
                nb3_img_sum = _summarize_turn(r_img, _check_endpoints(client, sid_nb3, r_img["body"]["message"]))
        results["product_mode"]["G29-NB-03"] = {
            "clarify": _summarize_turn(r_nb3_t1),
            "select_text": nb3_text_sum,
            "select_image": nb3_img_sum,
        }

        # --- G29-CTR-02: Cross-session 404 isolation (all 4 vectors!) ---
        # Create second owner client (separate cookies!)
        client_b = TestClient(client.app, base_url=HOST_ORIGIN)
        sid_b, _ = _create_session(client_b)
        msg_a = r_v1["body"]["message"]
        mid_a = msg_a["message_id"]
        did_a = msg_a["visible_display_ids"][0]
        eid_a = list(msg_a["sources"].keys())[0]
        x_media = client_b.get(f"/api/workspace/sessions/{sid_nb1}/messages/{mid_a}/media/{did_a}")
        x_ev = client_b.get(f"/api/workspace/sessions/{sid_nb1}/messages/{mid_a}/evidence/{eid_a}")
        x_pv = client_b.get(f"/api/workspace/sessions/{sid_nb1}/messages/{mid_a}/evidence/{eid_a}/preview")
        x_anchor = _ask(
            client_b, sid_b, "这张图横轴是什么？", "ctr02-xanchor",
            anchor={"type": "display", "message_id": mid_a, "display_id": did_a},
        )
        # Also cross-session clarification anchor replay
        x_clar = None
        if opts_b2:
            x_clar = _ask(
                client_b, sid_b, "replay", "ctr02-xclar",
                anchor={"type": "clarification", "turn_id": r_b2_t1["body"]["turn"]["turn_id"], "option_id": opts_b2[0]["option_id"]},
            )
        results["product_mode"]["G29-CTR-02"] = {
            "cross_media_status": x_media.status_code,
            "cross_evidence_status": x_ev.status_code,
            "cross_preview_status": x_pv.status_code,
            "cross_display_anchor_status": x_anchor["http_status"],
            "cross_clarification_anchor_status": x_clar["http_status"] if x_clar else None,
        }

        # --- G29-CTR-03: stale_reference controls ---
        # 1) Stale focus after v1->v2 upgrade asking "看当前 Cell"
        r_stale_focus = _ask(client, sid_nb1, "看当前 Cell 的内容", "ctr03-stale-focus")
        # 2) Stale clarification anchor after publishing a v3 version
        if opts_b2:
            # Upgrade nb2_doc to v2 (becomes version 3) then replay old b2 clarification anchor
            _import_file(client, sid_nb2, FIXTURES / "notebook/v2/Lab06-Thermal-Sweep.ipynb", document_id=nb2_doc)
            r_stale_clar = _ask(
                client, sid_nb2, "replay-after-v3", "ctr03-stale-clar",
                anchor={"type": "clarification", "turn_id": r_b2_t1["body"]["turn"]["turn_id"], "option_id": opts_b2[0]["option_id"]},
            )
        else:
            r_stale_clar = None
        results["product_mode"]["G29-CTR-03"] = {
            "stale_focus": _summarize_turn(r_stale_focus),
            "stale_clarification_after_v3": _summarize_turn(r_stale_clar) if r_stale_clar else None,
        }
    finally:
        shutil.rmtree(tmp_prod, ignore_errors=True)

    # ==================== 2. NO-GEN MODE (workspace_answer_enabled=False, source_vision_enabled=True) ====================
    tmp_nogen = Path(tempfile.mkdtemp(prefix="srag-g29-blind-nogen-"))
    try:
        client, _ = _make_client(tmp_nogen, env, answer_enabled=False, vision_enabled=True)
        sid, _ = _create_session(client)
        sd_id = _import_file(client, sid, FIXTURES / "Speculative-Decoding-2211.17192v2.pdf")["document_id"]
        _set_scope(client, sid, [sd_id])

        # Q04 in no-gen mode
        p5_node = _find_catalog_node(client, sid, sd_id, "page:5")
        _put_focus(client, sid, p5_node)
        rn_q04 = _ask(client, sid, "这张图的纵轴表示什么？", "ng-q04-t1")
        opts_ng = rn_q04["body"].get("options") or rn_q04["body"]["turn"].get("options") or []
        rn_q04_t2 = None
        if len(opts_ng) >= 2:
            right_opt = next((o for o in opts_ng if "右图" in o["label"]), opts_ng[1])
            r2 = _ask(
                client, sid, right_opt["label"], "ng-q04-t2",
                anchor={"type": "clarification", "turn_id": rn_q04["body"]["turn"]["turn_id"], "option_id": right_opt["option_id"]},
            )
            rn_q04_t2 = _summarize_turn(r2, _check_endpoints(client, sid, r2["body"]["message"]))
        results["no_gen_mode"]["G29-PDF-02-nogen"] = {"turn1": _summarize_turn(rn_q04), "turn2": rn_q04_t2}

        # b2 in no-gen mode
        sid_nb, _ = _create_session(client)
        nb_doc = _import_file(client, sid_nb, FIXTURES / "notebook/v1/Lab06-Thermal-Sweep.ipynb")["document_id"]
        _set_scope(client, sid_nb, [nb_doc])
        _put_focus(client, sid_nb, _find_catalog_node(client, sid_nb, nb_doc, "cell:1"))
        r_v1 = _ask(client, sid_nb, "第 2 个 Cell 第 1 张图的横轴单位是什么？", "ng-b2-v1")
        _import_file(client, sid_nb, FIXTURES / "notebook/v2_reordered/Lab06-Thermal-Sweep.ipynb", document_id=nb_doc)
        r_t1 = _ask(client, sid_nb, "对比最新版和旧版刚才读过的这张图的横轴单位有什么变化？", "ng-b2-t1")
        opts_b2_ng = r_t1["body"].get("options") or r_t1["body"]["turn"].get("options") or []
        r_t2_sum = None
        if len(opts_b2_ng) >= 2:
            temp_opt = next((o for o in opts_b2_ng if "输出 3" in o["label"]), opts_b2_ng[-1])
            r_t2 = _ask(
                client, sid_nb, temp_opt["label"], "ng-b2-t2",
                anchor={"type": "clarification", "turn_id": r_t1["body"]["turn"]["turn_id"], "option_id": temp_opt["option_id"]},
            )
            r_t2_sum = _summarize_turn(r_t2, _check_endpoints(client, sid_nb, r_t2["body"]["message"]))
        results["no_gen_mode"]["G29-NB-02-nogen"] = {
            "v1_read": _summarize_turn(r_v1, _check_endpoints(client, sid_nb, r_v1["body"]["message"])),
            "reordered_clarify": _summarize_turn(r_t1),
            "bilateral_recovered": r_t2_sum,
        }
    finally:
        shutil.rmtree(tmp_nogen, ignore_errors=True)

    return results


def main() -> None:
    criteria = json.loads((GATE_DIR / "G29-blind-criteria.json").read_text(encoding="utf-8"))
    env = _load_env_readonly()

    print("=== Running Part A: 68-object PDF registration audit ===")
    part_a = run_part_a_pdf_audit(criteria)
    print(json.dumps({"by_class": part_a["by_class"], "micro_overall": part_a["micro_overall"], "red_line_triggered": part_a["red_line_triggered"], "red_line_reasons": part_a["red_line_reasons"]}, indent=2, ensure_ascii=False))

    print("\n=== Running Part B: 17 frozen blind cases (Real DeepSeek + Real qwen3-vl-plus) ===")
    part_b = run_part_b_cases(env)

    out = {
        "subject_sha": SUBJECT_SHA,
        "gate_freeze_sha": "a2ad63c6a683b52a2084c247f7b2520955274f00",
        "loaded_app_file": str(Path(srag_app.__file__).resolve()),
        "srag_soffice": os.environ.get("SRAG_SOFFICE"),
        "part_a_pdf_68_object_audit": part_a,
        "part_b_blind_cases": part_b,
    }
    out_path = GATE_DIR / OUT_FILE
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved sanitized audit results to {out_path}")


if __name__ == "__main__":
    main()
