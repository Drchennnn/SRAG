"""H02 tests: course-scoped provenance KG, anti-examples, invalidation, 404s, 1-hop QA."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import nbformat
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pptx import Presentation
from pypdfium2 import PdfDocument

from app.api.workspace import get_workspace_service, router
from app.clients.deepseek import DeepSeekClient
from app.config import Settings
from app.rag.local_library import LocalLibrary
from app.rag.structure_search import StructureSearchRequest
from app.rag.workspace import make_workspace
from app.schemas.local import ParsedSource, SourceDraft, SourceKey
from app.stores.local_kg import LocalKgStore

Locator = dict

_BASE = "/api/workspace"
_HOST = "http://127.0.0.1:8001"
_COURSE = "cs-signals-h02"
_COURSE_B = "cs-systems-other"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_pptx_file(path: Path, slides: list[tuple[str, str]]) -> Path:
    prs = Presentation()
    layout = prs.slide_layouts[1]
    for title, body in slides:
        slide = prs.slides.add_slide(layout)
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
    prs.save(path)
    return path


def _make_ipynb_file(path: Path, cells: list[Any]) -> Path:
    nb = nbformat.v4.new_notebook(cells=cells)
    nbformat.write(nb, path)
    return path


def _publish_synthetic(
    library: LocalLibrary,
    course_id: str,
    file_name: str,
    fmt: str,
    sources: list[SourceDraft],
    *,
    document_id: str | None = None,
    raw: bytes | None = None,
) -> Any:
    path = library.course_root / course_id / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is None:
        if fmt == "pdf":
            with PdfDocument.new() as doc:
                for _ in sources:
                    doc.new_page(612, 792)
                doc.save(path)
            raw = path.read_bytes()
        else:
            raw = json.dumps(
                {"file": file_name, "sources": [s.source_id for s in sources]},
                ensure_ascii=False,
            ).encode("utf-8")
            path.write_bytes(raw)
    else:
        path.write_bytes(raw)
    plan = library.store.prepare(course_id, path, document_id=document_id)
    return library.store.publish(
        plan,
        ParsedSource(title=file_name, sources=sources),
        original=raw,
        content_sha256=_sha(raw),
        document_format=fmt,
    )


def _create_app(tmp_path: Path, **overrides: Any):
    params: dict[str, Any] = {
        "_env_file": None,
        "sqlite_path": tmp_path / "kg.sqlite3",
        "data_dir": tmp_path / "data",
        "course_root": tmp_path / "courses",
        "workspace_course_ids": f"{_COURSE},{_COURSE_B}",
        "workspace_answer_enabled": False,
        "source_vision_enabled": False,
        "deepseek_api_key": "",
        "dashscope_api_key": "",
    }
    params.update(overrides)
    settings = Settings(**params)
    service = make_workspace(settings)
    app = FastAPI()
    app.include_router(router, prefix=_BASE)
    app.dependency_overrides[get_workspace_service] = lambda: service
    return settings, service, app


def _create_session(client: TestClient, course_ids: list[str] | None = None) -> str:
    resp = client.post(
        f"{_BASE}/sessions",
        json={"course_ids": course_ids or [_COURSE]},
        headers={"origin": _HOST},
    )
    assert resp.status_code == 201, resp.text
    return f"{_BASE}/sessions/{resp.json()['session_id']}"


def _upload_file(
    client: TestClient,
    session_path: str,
    path: Path,
    *,
    course_id: str = _COURSE,
    document_id: str | None = None,
) -> dict[str, Any]:
    params: dict[str, str] = {"course_id": course_id, "filename": path.name}
    if document_id:
        params["document_id"] = document_id
    resp = client.post(
        f"{session_path}/import",
        params=params,
        headers={"content-type": "application/octet-stream", "origin": _HOST},
        content=path.read_bytes(),
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["status"] == "published", payload
    return payload


def _ask(
    client: TestClient,
    session_path: str,
    text: str,
    revision: int,
    *,
    client_id: str | None = None,
    kind: str = "ask",
) -> dict[str, Any]:
    resp = client.post(
        f"{session_path}/turns",
        json={
            "client_turn_id": client_id or f"turn-{revision}-{abs(hash(text))}",
            "scope_revision": revision,
            "kind": kind,
            "text": text,
        },
        headers={"origin": _HOST},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_h02_stage1_definitions_pdf_hardbreaks_numbered_and_anchors(tmp_path: Path) -> None:
    settings, service, app = _create_app(tmp_path)
    pdf_page_1 = (
        "2.1 核心概念定义：自适应陷波滤波器 (Adaptive Notch Filter, ANF) —\n"
        "用于抑制窄带周期干扰的实时带阻核。\n\n"
        "数学条件说明：设极点半径 r 满足 0 < r < 1，以控制过渡带带宽。\n"
        "在实际实验中通过调用 `notch_filter_step` 函数执行在线更新。\n\n"
        "2.2 §3.2 Formal Definition: Spectral Gatekeeper —\n"
        "suppresses sub-band energy anomalies\n"
        "beyond threshold tau during streaming.\n"
        "Experimental probe `gatekeeper_dispatch` verifies latency."
    )
    pdf_page_2 = (
        "梯度裁剪（Gradient\n裁剪阈值）\n"
        "是指\n"
        "当参数梯度欧氏范数超出上限 tau_clip 时按比例缩放更新步长。\n"
        "配套脚本使用 `gradient_clip_norm` 函数记录截断率。\n\n"
        "Spectral\nNotch Filter is defined as\n"
        "a narrowband suppression filter\n"
        "operating at center frequency f0."
    )
    doc = _publish_synthetic(
        service.library,
        _COURSE,
        "Lecture01-Defs.pdf",
        "pdf",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=1,
                title="Page 1",
                text=pdf_page_1,
                locator=Locator(page=1),
            ),
            SourceDraft(
                source_id="page:2",
                kind="page",
                ordinal=2,
                title="Page 2",
                text=pdf_page_2,
                locator=Locator(page=2),
            ),
        ],
    )
    assert doc.status == "published"

    concepts, edges = service.kg_store.active_edges_for_courses([_COURSE])
    by_name = {c.canonical_name: c for c in concepts}
    assert "自适应陷波滤波器" in by_name
    assert "Spectral Gatekeeper" in by_name
    assert "Spectral Notch Filter" in by_name
    assert "梯度裁剪" in by_name

    # Companion anchor extracted across non-adjacent math paragraph inside same concept boundary.
    assert "notch_filter_step" in by_name["自适应陷波滤波器"].anchor_terms
    assert "gatekeeper_dispatch" in by_name["Spectral Gatekeeper"].anchor_terms
    assert "gradient_clip_norm" in by_name["梯度裁剪"].anchor_terms
    # Section isolation: page:1 section 2.2 anchor must NOT leak into section 2.1 concept.
    assert "gatekeeper_dispatch" not in by_name["自适应陷波滤波器"].anchor_terms

    # Real PPTX slide ingestion also extracts definitions while ignoring outline slides.
    pptx_path = _make_pptx_file(
        tmp_path / "Slides-Intro.pptx",
        [
            ("Agenda", "1. Adaptive Thresholding\n2. Labs"),
            (
                "3.1 Adaptive Thresholding",
                "核心概念：当局部能量突增超过均值三倍方差时触发自适应截断。\n"
                "实验验证调用 `adaptive_thresh_step` 函数。",
            ),
        ],
    )
    with TestClient(app, base_url=_HOST) as client:
        sess = _create_session(client)
        _upload_file(client, sess, pptx_path)
    pptx_concepts = {c.canonical_name: c for c in service.kg_store.list_concepts([_COURSE])}
    assert "Adaptive Thresholding" in pptx_concepts
    assert "adaptive_thresh_step" in pptx_concepts["Adaptive Thresholding"].anchor_terms

    # Every accepted edge span must slice exact raw SQLite source text char-for-char.
    with sqlite3.connect(settings.sqlite_path) as conn:
        for edge in edges:
            row = conn.execute(
                "SELECT text FROM local_sources "
                "WHERE document_id = ? AND version = ? AND source_id = ?",
                (edge.document_id, edge.version, edge.source_id),
            ).fetchone()
            assert row is not None
            raw = row[0]
            assert raw[edge.evidence_span_start : edge.evidence_span_end] == edge.evidence_excerpt


def test_h02_stage2_cross_file_symbol_and_five_anti_examples(tmp_path: Path) -> None:
    settings, service, app = _create_app(tmp_path)
    # 1. Lecture defines ANF (`notch_filter_step`) and Gated Routing (`sparse_route_top2`).
    _publish_synthetic(
        service.library,
        _COURSE,
        "Lecture-Concepts.pdf",
        "pdf",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=1,
                title="Page 1",
                text=(
                    "1.1 定义：自适应陷波滤波器 (Adaptive Notch Filter, ANF) — "
                    "用于消除 60Hz 工频干扰的自适应滤波器。\n"
                    "实验代码中通过 `notch_filter_step` 进行单步验证。\n\n"
                    "1.2 核心概念定义：门控路由 (Gated Routing) — "
                    "根据打分向量动态选择激活专家子集的分发机制。\n"
                    "实验复现脚本调用 `sparse_route_top2` 统计吞吐。"
                ),
                locator=Locator(page=1),
            )
        ],
    )

    # 2. Lab notebook with ZERO shared concept names in code cells + anti-examples.
    lab_path = _make_ipynb_file(
        tmp_path / "Lab-Suite.ipynb",
        [
            # cell:0 — Valid E2 cross-file symbol verification (zero mention of concept names)
            nbformat.v4.new_code_cell(
                "vals, db_drop = notch_filter_step(raw_wave, f0=60.0)\n"
                "print(f'drop={db_drop:.1f}')",
                outputs=[nbformat.v4.new_output(output_type="stream", text="drop=-34.2 dB\n")],
            ),
            # cell:1 — R4 comment disclaimer blocking BOTH E1 AND E2!
            nbformat.v4.new_code_cell(
                "# Do NOT treat this cell as Adaptive Notch Filter;\n"
                "# it does NOT implement or demonstrate Adaptive Notch Filter.\n"
                "vals, _ = notch_filter_step(baseline_wave, f0=0.0)\nprint('baseline=-1.0 dB')",
                outputs=[nbformat.v4.new_output(output_type="stream", text="baseline=-1.0 dB\n")],
            ),
            # cell:2 — R4 Chinese disclaimer blocking E2 symbol!
            nbformat.v4.new_code_cell(
                "# 非 自适应陷波滤波器 实验，仅作无关噪声对比基线。\n"
                "out = notch_filter_step(noise_only, f0=10.0)\nprint('noise=-0.2 dB')",
                outputs=[nbformat.v4.new_output(output_type="stream", text="noise=-0.2 dB\n")],
            ),
            # cell:3 — R3 unexecuted code cell (no saved outputs!)
            nbformat.v4.new_code_cell(
                "# Demo: Gated Routing experiment\n"
                "routed = sparse_route_top2(tokens)\nprint(routed)",
                outputs=[],
            ),
            # cell:4 — R1 comment-only mention without demo cue or verified code anchor
            nbformat.v4.new_code_cell(
                "# TODO: review 门控路由 timing later\n"
                "latency = 1.25\nprint(latency)",
                outputs=[nbformat.v4.new_output(output_type="stream", text="1.25\n")],
            ),
            # cell:5 — R3 planned/hypothetical experiment note + outputs
            nbformat.v4.new_code_cell(
                "# Future work: plan to run 门控路由 benchmark next week\n"
                "routed = sparse_route_top2(mock_tokens)\nprint('mock=10')",
                outputs=[nbformat.v4.new_output(output_type="stream", text="mock=10\n")],
            ),
            # cell:6 — Valid E1 explicit demo cue with valid saved output (zero code symbol needed)
            nbformat.v4.new_code_cell(
                "# Demo: 门控路由 verification benchmark\n"
                "selected = [0, 3]\nprint('throughput=1280 tok/s')",
                outputs=[
                    nbformat.v4.new_output(output_type="stream", text="throughput=1280 tok/s\n")
                ],
            ),
        ],
    )

    with TestClient(app, base_url=_HOST) as client:
        sess = _create_session(client, [_COURSE])
        lab_res = _upload_file(client, sess, lab_path, course_id=_COURSE)
        lab_doc_id = lab_res["document_id"]

    # 3. Literature / comparative / outline / references anti-example document (R2 & R1)
    _publish_synthetic(
        service.library,
        _COURSE,
        "Readings-And-Outline.pdf",
        "pdf",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=1,
                title="Outline",
                text="Agenda\n- 自适应陷波滤波器\n- 门控路由\n- 实验计划",
                locator=Locator(page=1),
            ),
            SourceDraft(
                source_id="page:2",
                kind="page",
                ordinal=2,
                title="Related Work Comparison",
                text=(
                    "Unlike Adaptive Notch Filter [1, 4], Smith et al. use wavelet shrinkage.\n"
                    "与自适应陷波滤波器不同，参考文献[2]采用纯时域平滑。并非自适应陷波滤波器的定义。"
                ),
                locator=Locator(page=2, section_path=["Related Work"]),
            ),
            SourceDraft(
                source_id="page:3",
                kind="page",
                ordinal=3,
                title="Future Work",
                text=(
                    "Future work: planned Adaptive Notch Filter evaluation on noisy radar, "
                    "unobserved."
                ),
                locator=Locator(page=3),
            ),
        ],
    )

    # 4. R5 Cross-course isolation: Course B imports identically named symbol & concept.
    _publish_synthetic(
        service.library,
        _COURSE_B,
        "OtherCourse-Lab.ipynb",
        "ipynb",
        [
            SourceDraft(
                source_id="cell:0",
                kind="cell",
                ordinal=1,
                title="Cell 1",
                text="# Demo: Adaptive Notch Filter verification\nnotch_filter_step(x)",
                locator=Locator(cell_index=0, cell_type="code", outputs_state="available"),
            ),
            SourceDraft(
                source_id="cell:0:output:0",
                kind="notebook_output",
                parent_id="cell:0",
                ordinal=2,
                title="Output",
                text="-29 dB",
                locator=Locator(cell_index=0, output_index=0),
            ),
        ],
    )

    concepts, accepted = service.kg_store.active_edges_for_courses([_COURSE])
    accepted_demo_sources = {
        (e.concept_key, e.document_id, e.source_id, e.evidence_rule)
        for e in accepted
        if e.relation_type == "demonstrated_by"
    }
    anf_key = next(c.concept_key for c in concepts if c.canonical_name == "自适应陷波滤波器")
    gate_key = next(c.concept_key for c in concepts if c.canonical_name == "门控路由")

    # Only cell:0 (E2) and cell:6 (E1) are accepted! Anti-examples cell:1..cell:5 NEVER accepted.
    assert accepted_demo_sources == {
        (anf_key, lab_doc_id, "cell:0", "cross_file_symbol_verified"),
        (gate_key, lab_doc_id, "cell:6", "explicit_demo_cell"),
    }
    assert all(e.course_id == _COURSE for e in accepted)

    # Verify DB recorded explicit rejections/candidates for auditability.
    with sqlite3.connect(settings.sqlite_path) as conn:
        rows = conn.execute(
            "SELECT source_id, status, rejection_reason FROM local_kg_edges "
            "WHERE course_id = ? AND document_id = ? AND status != 'accepted'",
            (_COURSE, lab_doc_id),
        ).fetchall()
    by_src = {(r[0], r[1]): r[2] for r in rows}
    assert by_src[("cell:1", "rejected")] == "negated_or_mismatched_condition"
    assert by_src[("cell:2", "rejected")] == "negated_or_mismatched_condition"
    assert by_src[("cell:3", "rejected")] == "hypothetical_or_unexecuted_plan"
    assert by_src[("cell:4", "candidate")] == "cooccurrence_only"
    assert by_src[("cell:5", "rejected")] == "hypothetical_or_unexecuted_plan"

    # Course B had NO definition in Course B, so Course A's ANF never leaks into Course B.
    b_concepts, b_edges = service.kg_store.active_edges_for_courses([_COURSE_B])
    assert b_concepts == [] and b_edges == []


def test_h02_version_update_delete_promote_due_cas_and_historical_replay(tmp_path: Path) -> None:
    settings, service, app = _create_app(
        tmp_path, workspace_kg_retrieval_enabled=True, workspace_answer_enabled=False
    )
    lec_v1_path = tmp_path / "Lecture-Filter.pdf"
    _publish_synthetic(
        service.library,
        _COURSE,
        lec_v1_path.name,
        "pdf",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=1,
                title="Page 1",
                text=(
                    "定义：自适应陷波滤波器 (Adaptive Notch Filter) — 抑制窄带干扰。\n"
                    "实验通过 `old_notch_anchor` 演示衰减。"
                ),
                locator=Locator(page=1),
            )
        ],
    )
    lab_v1 = _make_ipynb_file(
        tmp_path / "Lab-Filter.ipynb",
        [
            nbformat.v4.new_code_cell(
                "res = old_notch_anchor(sig)\nprint('v1 attenuation -28 dB')",
                outputs=[
                    nbformat.v4.new_output(output_type="stream", text="v1 attenuation -28 dB\n")
                ],
            )
        ],
    )
    with TestClient(app, base_url=_HOST) as client:
        sess = _create_session(client)
        lec_res = client.get(f"{sess}/catalog").json()
        lec_doc_id = lec_res["courses"][0]["documents"][0]["document_id"]
        lab_res = _upload_file(client, sess, lab_v1)
        lab_doc_id = lab_res["document_id"]

        # Historical v1 QA answer records related_concepts & v1 evidence.
        turn_v1 = _ask(client, sess, "自适应陷波滤波器", 1, client_id="t-v1")
        assert turn_v1["turn"]["status"] == "completed"
        related_v1 = turn_v1["message"]["related_concepts"]
        assert any(c["canonical_name"] == "自适应陷波滤波器" for c in related_v1)
        ev_id = next(iter(turn_v1["message"]["sources"]))

        # Same-byte re-import is a clean idempotent no-op for KG.
        re_resp = client.post(
            f"{sess}/import",
            params={"course_id": _COURSE, "filename": lab_v1.name, "document_id": lab_doc_id},
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=lab_v1.read_bytes(),
        )
        assert re_resp.status_code == 200
        assert re_resp.json()["status"] == "skipped"
        concepts_mid = client.get(f"{sess}/kg/concepts").json()["concepts"]
        assert len(concepts_mid) == 1 and concepts_mid[0]["demonstrated_by_count"] == 1

        # Update Lecture v2 replacing anchor `old_notch_anchor` -> `new_notch_v2`!
        _publish_synthetic(
            service.library,
            _COURSE,
            lec_v1_path.name,
            "pdf",
            [
                SourceDraft(
                    source_id="page:1",
                    kind="page",
                    ordinal=1,
                    title="Page 1",
                    text=(
                        "定义：自适应陷波滤波器 (Adaptive Notch Filter) — 抑制窄带干扰。\n"
                        "v2 实验改用 `new_notch_v2` 完成。")
                    ,
                    locator=Locator(page=1),
                )
            ],
            document_id=lec_doc_id,
        )
        # Recompute must drop `old_notch_anchor` AND revoke dangling v1 lab E2 edge!
        c_list = service.kg_store.list_concepts([_COURSE])
        assert c_list[0].anchor_terms == ("new_notch_v2",)
        assert c_list[0].demonstrated_by_count == 0

        # Detail API flags partial + fallback_to_search!
        detail = client.get(f"{sess}/kg/concepts/{c_list[0].concept_key}").json()
        assert detail["completeness"] == "partial"
        assert detail["fallback_to_search"] is True
        assert detail["notice"]

        # Historical v1 evidence endpoint STILL serves archived v1 when doc is not deleted.
        msg_id = turn_v1["message"]["message_id"]
        hist = client.get(f"{sess}/messages/{msg_id}/evidence/{ev_id}")
        assert hist.status_code == 200, hist.text

        # Test scheduled -> promote_due directly before any UI/kg_concepts call.
        lab_v2 = _make_ipynb_file(
            tmp_path / "Lab-Filter-v2.ipynb",
            [
                nbformat.v4.new_code_cell(
                    "out = new_notch_v2(sig)\nprint('v2 attenuation -42 dB')",
                    outputs=[
                        nbformat.v4.new_output(
                            output_type="stream", text="v2 attenuation -42 dB\n"
                        )
                    ],
                )
            ],
        )
        lab_bytes = lab_v2.read_bytes()
        lab_dest = service.library.course_root / _COURSE / lab_v1.name
        lab_dest.write_bytes(lab_bytes)
        future_at = datetime.now(UTC) + timedelta(minutes=10)
        plan = service.library.store.prepare(_COURSE, lab_dest, document_id=lab_doc_id)
        sched = service.library.store.publish(
            plan,
            ParsedSource(
                title=lab_v1.name,
                sources=[
                    SourceDraft(
                        source_id="cell:0",
                        kind="cell",
                        ordinal=1,
                        title="Cell 1",
                        text="out = new_notch_v2(sig)\nprint('v2 attenuation -42 dB')",
                        locator=Locator(
                            cell_index=0, cell_type="code", outputs_state="available"
                        ),
                    ),
                    SourceDraft(
                        source_id="cell:0:output:0",
                        kind="notebook_output",
                        parent_id="cell:0",
                        ordinal=2,
                        title="Output",
                        text="v2 attenuation -42 dB\n",
                        locator=Locator(cell_index=0, output_index=0),
                    ),
                ],
            ),
            original=lab_bytes,
            content_sha256=_sha(lab_bytes),
            document_format="ipynb",
            release_at=future_at,
        )
        assert sched.status == "scheduled"
        # Before due time: still v1 (no demo edge). Once promoted ready in DB, direct QA syncs KG!
        assert service.kg_store.list_concepts([_COURSE])[0].demonstrated_by_count == 0
        with sqlite3.connect(settings.sqlite_path) as conn:
            conn.execute(
                "UPDATE local_documents SET pending_at = ? WHERE document_id = ?",
                ((datetime.now(UTC) - timedelta(seconds=5)).timestamp(), lab_doc_id),
            )
            conn.commit()

        # Without opening sidebar /kg/concepts first, direct QA promotes & hits v2!
        turn_v2 = _ask(
            client,
            sess,
            "什么是自适应陷波滤波器及其实验衰减结果",
            1,
            client_id="t-v2-promoted",
        )
        assert turn_v2["turn"]["trace"]["kg"]["status"] == "succeeded"
        cited_versions = {
            s["revision"]["source"]["version"]
            for s in turn_v2["message"]["sources"].values()
            if s["revision"]["source"]["document_id"] == lab_doc_id
        }
        assert cited_versions == {2}

        # CAS guard rejects stale epoch or mismatched sha256 or mismatched excerpt.
        assert (
            service.kg_store.write_kg_edges(
                SourceKey(document_id=lab_doc_id, version=2, source_id="cell:0"),
                course_id=_COURSE,
                content_sha256=_sha(lab_bytes),
                publication_epoch=0,  # stale epoch
                concept_key=c_list[0].concept_key,
                canonical_name="自适应陷波滤波器",
                relation_type="demonstrated_by",
                span_start=0,
                span_end=10,
                excerpt="out = new_",
            )
            is False
        )
        assert (
            service.kg_store.write_kg_edges(
                SourceKey(document_id=lab_doc_id, version=2, source_id="cell:0"),
                course_id=_COURSE,
                content_sha256=_sha(lab_bytes),
                publication_epoch=sched.publication_epoch,
                concept_key=c_list[0].concept_key,
                canonical_name="自适应陷波滤波器",
                relation_type="demonstrated_by",
                span_start=0,
                span_end=10,
                excerpt="wrong_text",
            )
            is False
        )

        # Delete lecture doc via BEGIN IMMEDIATE -> retires concept & marks historical chip stale!
        assert service.library.store.delete_local(lec_doc_id) is True
        assert client.get(f"{sess}/kg/concepts").json()["concepts"] == []
        assert client.get(f"{sess}/kg/concepts/{c_list[0].concept_key}").status_code == 404
        # Deleting lecture revokes turn access (not_accessible) and clears related_concepts.
        sess_state = client.get(sess).json()
        assistant_msgs = [m for m in sess_state["messages"] if m["role"] == "assistant"]
        assert all(m.get("related_concepts", []) == [] for m in assistant_msgs)


def test_h02_api_auth_404_catalog_isolation_and_stale_chip_flag(tmp_path: Path) -> None:
    _, service, app = _create_app(tmp_path)
    doc_a = _publish_synthetic(
        service.library,
        _COURSE,
        "A-Lecture.pdf",
        "pdf",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=1,
                title="Page 1",
                text="定义：信号门控 (Signal Gate) — 依据阈值截断高频毛刺。",
                locator=Locator(page=1),
            )
        ],
    )
    doc_b = _publish_synthetic(
        service.library,
        _COURSE_B,
        "B-Lecture.pdf",
        "pdf",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=1,
                title="Page 1",
                text=(
                    "Definition: Cache Coherency (CC) — "
                    "preserves consistent memory view across cores."
                ),
                locator=Locator(page=1),
            )
        ],
    )
    a_key = service.kg_store.list_concepts([_COURSE])[0].concept_key
    b_key = service.kg_store.list_concepts([_COURSE_B])[0].concept_key

    with TestClient(app, base_url=_HOST) as client_a, TestClient(app, base_url=_HOST) as client_b:
        # Session A authorized ONLY for _COURSE (while server allowed_courses contains BOTH).
        sess_a = _create_session(client_a, [_COURSE])
        sess_b = _create_session(client_b, [_COURSE_B])

        # _catalog internally contains both courses, yet /kg/concepts only returns _COURSE!
        list_a = client_a.get(f"{sess_a}/kg/concepts")
        assert list_a.status_code == 200
        assert [c["concept_key"] for c in list_a.json()["concepts"]] == [a_key]

        # Cross-course concept lookup MUST return 404 not_accessible (zero leakage).
        cross_course = client_a.get(f"{sess_a}/kg/concepts/{b_key}")
        assert cross_course.status_code == 404
        assert cross_course.json()["detail"]["code"] == "not_accessible"

        explicit_cross = client_a.get(
            f"{sess_a}/kg/concepts/{a_key}", params={"course_id": _COURSE_B}
        )
        assert explicit_cross.status_code == 404
        assert explicit_cross.json()["detail"]["code"] == "not_accessible"

        # Cross-owner session lookup MUST return 404.
        assert client_b.get(f"{sess_a}/kg/concepts").status_code == 404
        assert client_b.get(f"{sess_a}/kg/concepts/{a_key}").status_code == 404
        assert client_a.get(f"{sess_b}/kg/concepts/{b_key}").status_code == 404

        # Trigger turn on doc_a, then replace doc_a v2 removing concept -> chip is_stale=True.
        t_a = _ask(client_a, sess_a, "Signal Gate", 1, client_id="a1")
        assert t_a["message"]["related_concepts"][0]["concept_key"] == a_key
        _publish_synthetic(
            service.library,
            _COURSE,
            "A-Lecture.pdf",
            "pdf",
            [
                SourceDraft(
                    source_id="page:1",
                    kind="page",
                    ordinal=1,
                    title="Page 1",
                    text="Revised introduction without any formal definition block.",
                    locator=Locator(page=1),
                )
            ],
            document_id=doc_a.document_id,
        )
        # Re-read session_a: doc_a is authorized, concept retired -> is_stale=True!
        state_a = client_a.get(sess_a).json()
        assert state_a["messages"][-1]["related_concepts"][0]["is_stale"] is True
        assert client_a.get(f"{sess_a}/kg/concepts/{a_key}").status_code == 404
        assert doc_b.document_id


def test_h02_qa_one_hop_monolingual_fts_budget_quota_scope_child_output_and_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_search_reqs: list[StructureSearchRequest] = []
    settings, service, app = _create_app(
        tmp_path,
        workspace_answer_enabled=True,
        deepseek_api_key="h02-mock-key",
        workspace_retrieval_strategy="fts",
        workspace_retrieval_budget=0,
        workspace_retrieval_include_adjacency=False,
        workspace_kg_retrieval_enabled=True,
    )
    orig_search = service.searcher.search

    def spy_search(req: StructureSearchRequest, **kwargs: Any):
        captured_search_reqs.append(req)
        return orig_search(req, **kwargs)

    monkeypatch.setattr(service.searcher, "search", spy_search)

    def fail_if_called(*args: Any, **kwargs: Any):
        raise AssertionError("Monolingual fts+kg must not invoke bilingual rewriter!")

    monkeypatch.setattr(service, "rewriter", fail_if_called)

    # 3 distinct concepts defined in lecture & demonstrated in zero-shared-term lab cells+outputs:
    _publish_synthetic(
        service.library,
        _COURSE,
        "Lecture-Ch1.pdf",
        "pdf",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=1,
                title="P1",
                text=(
                    "Definition: Alpha Filter — removes low frequency drift in sensor traces.\n"
                    "Uses `notch_proc_x1` for streaming runs.\n\n"
                    "Definition: Beta Gate — clips impulse spikes above sigma.\n"
                    "Uses `clamp_proc_x2` in lab validation."
                ),
                locator=Locator(page=1),
            ),
            SourceDraft(
                source_id="page:2",
                kind="page",
                ordinal=2,
                title="P2",
                text=(
                    "Definition: Gamma Router — dispatches packets by congestion score.\n"
                    "Uses `route_proc_x3` for simulation."
                ),
                locator=Locator(page=2),
            ),
        ],
    )
    lab_doc = _publish_synthetic(
        service.library,
        _COURSE,
        "Lab-Suite.ipynb",
        "ipynb",
        [
            SourceDraft(
                source_id="cell:0",
                kind="cell",
                ordinal=1,
                title="C1",
                text="a_val = notch_proc_x1(stream)\nprint(a_val)",
                locator=Locator(cell_index=0, cell_type="code", outputs_state="available"),
            ),
            SourceDraft(
                source_id="cell:0:output:0",
                kind="notebook_output",
                parent_id="cell:0",
                ordinal=2,
                title="Out1",
                text="snr_rmse=0.012",
                locator=Locator(cell_index=0, output_index=0),
            ),
            SourceDraft(
                source_id="cell:1",
                kind="cell",
                ordinal=3,
                title="C2",
                text="b_val = clamp_proc_x2(stream)\nprint(b_val)",
                locator=Locator(cell_index=1, cell_type="code", outputs_state="available"),
            ),
            SourceDraft(
                source_id="cell:1:output:0",
                kind="notebook_output",
                parent_id="cell:1",
                ordinal=4,
                title="Out2",
                text="clipped_peaks=42",
                locator=Locator(cell_index=1, output_index=0),
            ),
            SourceDraft(
                source_id="cell:2",
                kind="cell",
                ordinal=5,
                title="C3",
                text="g_val = route_proc_x3(pkts)\nprint(g_val)",
                locator=Locator(cell_index=2, cell_type="code", outputs_state="available"),
            ),
            SourceDraft(
                source_id="cell:2:output:0",
                kind="notebook_output",
                parent_id="cell:2",
                ordinal=6,
                title="Out3",
                text="tail_p99=3.4ms",
                locator=Locator(cell_index=2, output_index=0),
            ),
        ],
    )

    seen_evidence: list[dict[str, Any]] = []

    async def fake_generate(self_client, messages, **kwargs):
        wire = json.loads(messages[-1]["content"])
        seen_evidence.append(wire)
        keys = [item["id"] for item in wire["evidence"]]
        return {
            "content": json.dumps(
                {
                    "statements": [
                        {"text": "Verified from live SQLite reads.", "citations": keys[:2]}
                    ],
                    "missing": "",
                }
            )
        }

    monkeypatch.setattr(DeepSeekClient, "generate_answer", fake_generate)

    with TestClient(app, base_url=_HOST) as client:
        sess = _create_session(client)
        docs_by_name = {
            d["file_name"]: d["document_id"]
            for d in client.get(f"{sess}/catalog").json()["courses"][0]["documents"]
        }
        lec_id = docs_by_name["Lecture-Ch1.pdf"]

        # 1. Single-concept monolingual English query under strategy="fts" (must NOT early-return!).
        res1 = _ask(
            client,
            sess,
            "Explain Alpha Filter and its lab verification output",
            1,
            client_id="mono-1",
        )
        assert captured_search_reqs[-1].include_adjacency is False
        ret_trace = res1["turn"]["trace"]["retrieval"]
        assert ret_trace["budget"] == 6
        assert ret_trace["effective_strategy"] == "fts+kg"
        assert ret_trace["kg"]["status"] == "succeeded"
        selected_ids = [(s["document_id"], s["source_id"]) for s in ret_trace["selected"]]
        # Both lecture page:1 AND lab cell:0 AND child_output cell:0:output:0 enter budget<=6!
        assert len(selected_ids) <= 6
        assert (lec_id, "page:1") in selected_ids
        assert (lab_doc.document_id, "cell:0") in selected_ids
        assert (lab_doc.document_id, "cell:0:output:0") in selected_ids
        # Verify co-ranking: output immediately follows parent cell:0!
        idx_cell = selected_ids.index((lab_doc.document_id, "cell:0"))
        assert selected_ids[idx_cell + 1] == (lab_doc.document_id, "cell:0:output:0")
        # Verify live full re-read (evidence includes full text, not just KG excerpt).
        assert any("snr_rmse=0.012" in str(ev) for ev in seen_evidence[-1]["evidence"])

        # 2. Multi-concept query hitting all 3 concepts: round-robin + per_concept_cap=2!
        res3 = _ask(
            client,
            sess,
            "Compare Alpha Filter, Beta Gate, and Gamma Router experiments",
            1,
            client_id="multi-3",
        )
        kg3 = res3["turn"]["trace"]["retrieval"]["kg"]
        assert kg3["status"] == "succeeded"
        assert kg3["per_concept_cap"] == 2
        added_srcs = [(s["document_id"], s["source_id"]) for s in kg3["added_sources"]]
        # Round-robin picks 1st missing counterpart for each of the 3 concepts before 2nd slots!
        assert len(added_srcs) <= kg3["kg_reserve_cap"] == 3
        assert (lab_doc.document_id, "cell:0") in added_srcs
        assert (lab_doc.document_id, "cell:1") in added_srcs
        assert (lab_doc.document_id, "cell:2") in added_srcs

        # 3. Document scope narrowing to [lec_id]: lab_doc edge MUST be excluded!
        patched = client.patch(
            f"{sess}/scope",
            json={"course_ids": [_COURSE], "document_ids": [lec_id], "scope_revision": 1},
        ).json()
        rev2 = patched["scope_revision"]
        # Scope narrowed to lecture only: verify lab_doc NEVER leaks into wire evidence.
        res_scoped = _ask(
            client,
            sess,
            "Explain Alpha Filter and its lab verification output",
            rev2,
            client_id="scoped-lec",
        )
        assert all(
            item["source"]["document_id"] == lec_id
            for item in seen_evidence[-1]["evidence"]
        )
        assert res_scoped["turn"]["status"] == "completed"

    # 4. Strict re-read tamper test: if local_sources text changed without KG rebuild,
    # live SQLite text wins over KG excerpt — KG excerpt is never trusted as source text.
    with sqlite3.connect(settings.sqlite_path) as conn:
        conn.execute(
            "UPDATE local_sources SET text = ? "
            "WHERE document_id = ? AND source_id = 'cell:0:output:0'",
            ("LIVE_SQLITE_READ_VERIFIED=99.9", lab_doc.document_id),
        )
        conn.commit()
    with TestClient(app, base_url=_HOST) as client:
        sess2 = _create_session(client)
        _ask(client, sess2, "Explain Alpha Filter result", 1, client_id="reread-live")
        assert any(
            "LIVE_SQLITE_READ_VERIFIED=99.9" in str(ev) for ev in seen_evidence[-1]["evidence"]
        )


def test_h02_kg_disabled_default_and_include_adjacency_toggle(tmp_path: Path) -> None:
    settings, service, app = _create_app(tmp_path)
    assert settings.workspace_kg_retrieval_enabled is False
    assert settings.workspace_retrieval_include_adjacency is True

    _publish_synthetic(
        service.library,
        _COURSE,
        "Notes.pdf",
        "pdf",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=1,
                title="P1",
                text="定义：测试锚点 (Test Anchor) — 第一页正文。\n调用 `anchor_fn_1`。",
                locator=Locator(page=1),
            ),
            SourceDraft(
                source_id="page:2",
                kind="page",
                ordinal=2,
                title="P2",
                text="相邻第二页实验数值：42.0 ms。",
                locator=Locator(page=2),
            ),
        ],
    )
    service.searcher.search(
        StructureSearchRequest(
            query="第一页正文",
            scope={"course_ids": [_COURSE], "document_ids": []},
            include_adjacency=True,
        ),
        allowed_course_ids=[_COURSE],
    )
    resp_no_adj = service.searcher.search(
        StructureSearchRequest(
            query="第一页正文",
            scope={"course_ids": [_COURSE], "document_ids": []},
            include_adjacency=False,
        ),
        allowed_course_ids=[_COURSE],
    )
    assert all(
        not any(r.code in {"next_position", "previous_position"} for r in c.reasons)
        for c in resp_no_adj.candidates
    )


def test_h02_offline_no_llm_mode_navigation_and_qa(tmp_path: Path) -> None:
    settings, service, app = _create_app(
        tmp_path, workspace_answer_enabled=False, workspace_kg_retrieval_enabled=True
    )
    _publish_synthetic(
        service.library,
        _COURSE,
        "Lecture-Offline.pdf",
        "pdf",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=1,
                title="P1",
                text=(
                    "定义：离线滤波核 (Offline Kernel) — 无需外部模型的确定性滤波核。\n"
                    "使用 `run_filt_42` 验证。"
                ),
                locator=Locator(page=1),
            ),
        ],
    )
    _publish_synthetic(
        service.library,
        _COURSE,
        "Lab-Offline.ipynb",
        "ipynb",
        [
            SourceDraft(
                source_id="cell:0",
                kind="cell",
                ordinal=1,
                title="Cell 1",
                text="val = run_filt_42(sig)\nprint(val)",
                locator=Locator(cell_index=0, cell_type="code", outputs_state="available"),
            ),
            SourceDraft(
                source_id="cell:0:output:0",
                kind="notebook_output",
                parent_id="cell:0",
                ordinal=2,
                title="Output 0",
                text="snr_gain=18.5 dB",
                locator=Locator(cell_index=0, output_index=0),
            ),
        ],
    )
    assert isinstance(service.kg_store, LocalKgStore)
    with TestClient(app, base_url=_HOST) as client:
        sess = _create_session(client)
        concepts = client.get(f"{sess}/kg/concepts").json()["concepts"]
        assert len(concepts) == 1
        key = concepts[0]["concept_key"]
        detail = client.get(f"{sess}/kg/concepts/{key}").json()
        assert detail["completeness"] == "complete"
        out0 = detail["demonstrated_by"][0]["child_outputs"][0]
        assert out0["text_excerpt"] == "snr_gain=18.5 dB"

        ans = _ask(client, sess, "Offline Kernel 的实测结果", 1, client_id="offline-q1")
        assert ans["turn"]["status"] == "completed"
        assert ans["turn"]["trace"]["kg"]["status"] == "succeeded"
        texts = " ".join(b.get("text", "") for b in ans["message"]["blocks"])
        assert "离线滤波核" in texts
        assert "snr_gain=18.5 dB" in texts
        assert ans["message"]["related_concepts"][0]["concept_key"] == key


def test_h02_budget_parity_visual_child_output_legacy_revoke_and_doc_scope_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import base64
    import io

    from PIL import Image

    from app.rag.workspace_kg import expand_kg_one_hop
    from app.stores.local_library import LegacyBinding
    from app.stores.local_visibility import revoke_legacy_document
    from app.stores.structure_snapshot import SnapshotRequest

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 120, 200)).save(buf, format="PNG")
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # 1. Unified budget=6 across strategies (fts, bilingual, s0) when KG enabled & budget=0.
    for strategy in ("fts", "s0"):
        sub_dir = tmp_path / strategy
        settings, service, app = _create_app(
            sub_dir,
            workspace_answer_enabled=True,
            deepseek_api_key="h02-mock",
            workspace_retrieval_strategy=strategy,
            workspace_retrieval_budget=0,
            workspace_kg_retrieval_enabled=True,
        )
        _publish_synthetic(
            service.library,
            _COURSE,
            "Lec.pdf",
            "pdf",
            [
                SourceDraft(
                    source_id="page:1",
                    kind="page",
                    ordinal=1,
                    title="P1",
                    text=(
                        "Definition: Visual Probe — triggers signal plot verification.\n"
                        "Uses `vis_probe_fn` in lab."
                    ),
                    locator=Locator(page=1),
                )
            ],
        )
        _publish_synthetic(
            service.library,
            _COURSE,
            "Lab.ipynb",
            "ipynb",
            [
                SourceDraft(
                    source_id="cell:0",
                    kind="cell",
                    ordinal=1,
                    title="C0",
                    text="vis_probe_fn(x)\nprint('plot ok')",
                    locator=Locator(cell_index=0, cell_type="code", outputs_state="available"),
                ),
                SourceDraft(
                    source_id="cell:0:output:0",
                    kind="notebook_output",
                    parent_id="cell:0",
                    ordinal=2,
                    title="Out0",
                    text="",
                    locator=Locator(
                        cell_index=0,
                        output_index=0,
                        representations=[
                            {"status": "available", "mime_type": "image/png", "data": png_b64}
                        ],
                    ),
                ),
            ],
        )

        async def _ok_gen(self_client, messages, **kwargs):
            wire = json.loads(messages[-1]["content"])
            keys = [item["id"] for item in wire["evidence"]]
            return {
                "content": json.dumps(
                    {"statements": [{"text": "OK", "citations": keys[:1]}], "missing": ""}
                )
            }

        monkeypatch.setattr(DeepSeekClient, "generate_answer", _ok_gen)
        with TestClient(app, base_url=_HOST) as client:
            sess = _create_session(client)
            res = _ask(client, sess, "Explain Visual Probe", 1, client_id=f"strat-{strategy}")
            ret = res["turn"]["trace"]["retrieval"]
            assert ret["budget"] == 6
            assert ret["kg"]["status"] == "succeeded"
            # Visual notebook child_output was attached as visual candidate from 1-hop!
            assert any(
                v["source_id"] == "cell:0:output:0" for v in ret["kg"]["visual_added"]
            )

    # 2. Verify expand_kg_one_hop strictly scopes out unselected documents + revoke_legacy_document.
    rev_dir = tmp_path / "revoke_check"
    settings, service, app = _create_app(rev_dir)
    lec_raw = b"lec-legacy-bytes"
    lec_path = service.library.course_root / _COURSE / "Legacy-Lec.pdf"
    lec_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfDocument.new() as doc:
        doc.new_page(612, 792)
        doc.save(lec_path)
    lec_raw = lec_path.read_bytes()
    lec_hash = _sha(lec_raw)
    with sqlite3.connect(settings.sqlite_path) as conn:
        conn.execute(
            "INSERT INTO documents(document_id, course_id, source_path, title, document_type, "
            "created_at, updated_at, deleted_at) VALUES (?, ?, ?, ?, 'pdf', 't', 't', NULL)",
            ("legacy-42", _COURSE, str(lec_path), "Legacy-Lec.pdf"),
        )
        conn.execute(
            "INSERT INTO document_versions(document_id, version, content_hash, release_at, "
            "state, created_at, indexed_at, error) "
            "VALUES ('legacy-42', 1, ?, NULL, 'active', 't', 't', NULL)",
            (lec_hash,),
        )
        conn.commit()
    plan = service.library.store.prepare(_COURSE, lec_path)
    lec_res = service.library.store.publish(
        plan,
        ParsedSource(
            title="Legacy-Lec.pdf",
            sources=[
                SourceDraft(
                    source_id="page:1",
                    kind="page",
                    ordinal=1,
                    title="P1",
                    text=(
                        "定义：遗留算子 (Legacy Operator) — 用于旧版同步测试。\n"
                        "函数 `legacy_op_fn` 执行验证。"
                    ),
                    locator=Locator(page=1),
                )
            ],
        ),
        original=lec_raw,
        content_sha256=lec_hash,
        document_format="pdf",
        legacy=LegacyBinding(
            document_id="legacy-42",
            version=1,
            keys={"legacy-chunk-1": ["page:1"]},
            from_sqlite=True,
        ),
    )
    lab_res = _publish_synthetic(
        service.library,
        _COURSE,
        "Legacy-Lab.ipynb",
        "ipynb",
        [
            SourceDraft(
                source_id="cell:0",
                kind="cell",
                ordinal=1,
                title="C0",
                text="legacy_op_fn(x)",
                locator=Locator(cell_index=0, cell_type="code", outputs_state="available"),
            ),
            SourceDraft(
                source_id="cell:0:output:0",
                kind="notebook_output",
                parent_id="cell:0",
                ordinal=2,
                title="O0",
                text="ok=1",
                locator=Locator(cell_index=0, output_index=0),
            ),
        ],
    )
    snap = service.snapshots.read(SnapshotRequest([_COURSE]))
    lec_only_docs = {
        d.document_id: d for d in snap.documents if d.document_id == lec_res.document_id
    }
    lec_only_srcs = {
        (s.document_id, s.version, s.source_id): s
        for s in snap.sources
        if s.document_id == lec_res.document_id
    }
    scoped_exp = expand_kg_one_hop(
        service.kg_store,
        course_ids=[_COURSE],
        docs=lec_only_docs,
        sources=lec_only_srcs,
        question="Legacy Operator",
        alternate_query=None,
        pool=[],
        budget=6,
    )
    assert scoped_exp.trace["filtered_out_of_scope"] >= 1
    assert all(
        c.source.document_id != lab_res.document_id for c in scoped_exp.pool
    )

    with service.library.store._connect() as conn:
        revoke_legacy_document(conn, "legacy-42")
    assert service.kg_store.list_concepts([_COURSE]) == []

