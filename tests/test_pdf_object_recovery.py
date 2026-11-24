"""Deterministic tests for PDF vector subfigure, caption, formula, table recovery & Q04."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.workspace import get_workspace_service
from app.config import Settings, get_settings
from app.main import create_app
from app.rag.parsers.local_slides import parse_local_slides
from app.rag.source_vision import DashScopeVisionProvider, VisionReply
from app.rag.workspace import make_workspace


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _t(x: float, y: float, text: str, size: float = 11.0) -> str:
    return f"BT /F1 {size:.1f} Tf 1 0 0 1 {x:.1f} {y:.1f} Tm ({_escape(text)}) Tj ET"


def _line(x0: float, y0: float, x1: float, y1: float, width: float = 1.0) -> str:
    return f"{width:.1f} w {x0:.1f} {y0:.1f} m {x1:.1f} {y1:.1f} l S"


def _rect(x: float, y: float, w: float, h: float, width: float = 1.0) -> str:
    return f"{width:.1f} w {x:.1f} {y:.1f} {w:.1f} {h:.1f} re S"


def _polyline(pts: list[tuple[float, float]], width: float = 1.5) -> str:
    head = f"{width:.1f} w {pts[0][0]:.1f} {pts[0][1]:.1f} m"
    segs = " ".join(f"{x:.1f} {y:.1f} l" for x, y in pts[1:])
    return f"{head} {segs} S"


def _build_multi_page_pdf(pages: list[str], *, width: int = 612, height: int = 792) -> bytes:
    """Build a minimal valid multi-page PDF with Helvetica font."""
    objects: list[bytes] = []
    # 1: Catalog, 2: Pages, 3: Font
    page_ids: list[int] = []
    content_ids: list[int] = []
    next_id = 4
    for _ in pages:
        page_ids.append(next_id)
        content_ids.append(next_id + 1)
        next_id += 2

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for _pid, cid, stream_text in zip(page_ids, content_ids, pages, strict=True):
        stream = stream_text.encode("latin-1")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {cid} 0 R >>".encode()
        )
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj_id, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{obj_id} 0 obj\n".encode())
        out.extend(body)
        out.extend(b"\nendobj\n")
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode())
    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    )
    out.extend(trailer.encode())
    return bytes(out)


def _write_vector_subfigures_pdf(path: Path) -> None:
    # Page 1: Section separator line + Table 1 (horizontal rules) + section 1 text
    p1 = "\n".join(
        [
            _t(50, 740, "Section 1: Bench Overview", 14),
            _line(50, 725, 560, 725, 1.0),  # Full-width section divider
            _t(
                50,
                695,
                "Default sampling interval 0.25 ms, impedance 50-ohm, ambient 23.0 C.",
            ),
            _t(50, 650, "Table 1: Channel Calibration Parameters", 12),
            _line(50, 635, 560, 635, 1.2),
            _t(60, 616, "Channel"),
            _t(260, 616, "Nominal Range"),
            _line(50, 608, 560, 608, 0.8),
            _t(60, 588, "CH-A (Step)"),
            _t(260, 588, "0.0 V to 5.0 V"),
            _t(60, 568, "CH-B (Decay)"),
            _t(260, 568, "-12.0 dB to +6.0 dB"),
            _line(50, 558, 560, 558, 1.2),
        ]
    )
    # Page 2: Two side-by-side vector subfigures + Figure 1 caption
    p2 = "\n".join(
        [
            _t(50, 740, "Section 2: Dual Vector Waveforms", 14),
            # Left subfigure (a): x in [45, 285], y in [340, 620]
            _rect(45, 340, 240, 280),
            _t(60, 600, "(a) Left Panel: Step Response", 11),
            _line(70, 380, 260, 380),
            _line(70, 380, 70, 570),
            _polyline([(70, 390), (110, 390), (130, 530), (190, 530), (255, 530)]),
            _t(120, 355, "Time [ms]", 9),
            # Right subfigure (b): x in [325, 565], y in [340, 620]
            _rect(325, 340, 240, 280),
            _t(340, 600, "(b) Right Panel: Damped Decay", 11),
            _line(350, 380, 540, 380),
            _line(350, 380, 350, 570),
            _polyline([(350, 410), (410, 545), (450, 460), (490, 430), (535, 420)]),
            _t(400, 355, "Time [ms]", 9),
            _t(355, 545, "Gain [dB] +6 dB @ 8 ms", 9),
            # Caption below subfigures
            _t(
                50,
                305,
                "Figure 1: Qualitative bench waveforms. Only a schematic illustration;",
                10,
            ),
            _t(50, 290, "does not support causal delay claims.", 10),
        ]
    )
    path.write_bytes(_build_multi_page_pdf([p1, p2]))


def _write_formula_and_notes_pdf(path: Path) -> None:
    p1 = "\n".join(
        [
            _t(50, 730, "Background Notes", 13),
            _rect(60, 620, 480, 60),
            _t(80, 650, "E_loss = 0.5 * k * x^2", 11),
            _t(50, 560, "Figure 2: Reference spring energy baseline.", 10),
        ]
    )
    p2 = "\n".join(
        [
            _t(50, 740, "Effective Group Delay (Standalone Formula):", 12),
            _rect(60, 610, 480, 85),
            _t(80, 668, "tau_eff = (W - 1) / 2", 12),
            _t(80, 642, "Condition: for odd W >= 3 only", 10),
            _rect(60, 470, 480, 75),
            _t(80, 518, "alpha_max = 1 / (N + 1)", 12),
            _t(80, 494, "Condition: exponential moving average", 10),
            _t(50, 320, "Figure 3: Peak-hold envelope under dry bench conditions.", 10),
        ]
    )
    path.write_bytes(_build_multi_page_pdf([p1, p2]))


def test_pdf_parser_registers_vector_subfigures_captions_formulas_and_tables(
    tmp_path: Path,
) -> None:
    vec_pdf = tmp_path / "Vector-Subfigures-Guide.pdf"
    fml_pdf = tmp_path / "Formula-And-Notes.pdf"
    _write_vector_subfigures_pdf(vec_pdf)
    _write_formula_and_notes_pdf(fml_pdf)

    vec = parse_local_slides(vec_pdf)
    by_id_vec = {s.source_id: s for s in vec.sources}
    assert "page:1" in by_id_vec
    assert "page:1:table:1" in by_id_vec
    assert by_id_vec["page:2:image:1"].locator["object_type"] == "image"
    assert by_id_vec["page:2:image:2"].locator["object_type"] == "image"
    assert (
        by_id_vec["page:2:image:1"].locator["bbox"][0]
        < by_id_vec["page:2:image:2"].locator["bbox"][0]
    )
    assert by_id_vec["page:2:caption:1"].locator["object_type"] == "caption"
    assert by_id_vec["page:2:caption:1"].text == ""
    assert (
        "does not support causal delay claims"
        in by_id_vec["page:2:caption:1"].locator["object_text"]
    )

    fml = parse_local_slides(fml_pdf)
    by_id_fml = {s.source_id: s for s in fml.sources}
    assert "page:2:formula:1" in by_id_fml
    assert "page:2:formula:2" in by_id_fml
    assert by_id_fml["page:2:formula:1"].text == ""
    assert "tau_eff = (W - 1) / 2" in by_id_fml["page:2:formula:1"].locator["object_text"]
    assert "for odd W >= 3 only" in by_id_fml["page:2:formula:1"].locator["object_text"]
    assert "alpha_max = 1 / (N + 1)" in by_id_fml["page:2:formula:2"].locator["object_text"]
    assert by_id_fml["page:2:caption:1"].locator["object_type"] == "caption"


def test_pdf_q01_to_q06_including_q04_focus_disambiguation_and_evidence_200(tmp_path: Path) -> None:
    vec_pdf = tmp_path / "Vector-Subfigures-Guide.pdf"
    fml_pdf = tmp_path / "Formula-And-Notes.pdf"
    _write_vector_subfigures_pdf(vec_pdf)
    _write_formula_and_notes_pdf(fml_pdf)

    settings = Settings(
        _env_file=None,
        course_root=tmp_path / "courses",
        data_dir=tmp_path / "data",
        sqlite_path=tmp_path / "library.sqlite3",
        workspace_course_ids="pdf-obj-test",
        workspace_answer_enabled=False,
        workspace_retrieval_strategy="fts",
        source_vision_enabled=True,
        dashscope_api_key="synthetic-key",
        deepseek_api_key="",
        qdrant_url="http://127.0.0.1:1",
    )
    app = create_app()
    svc = make_workspace(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_workspace_service] = lambda: svc

    def fake_vision(self, *args, **kwargs):
        return VisionReply(
            status="succeeded",
            text="Right panel (b) shows Gain [dB] vs Time [ms], damped decay, +6 dB @ 8 ms.",
            missing=(),
            request_id="req-test",
            usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )

    with (
        patch.object(DashScopeVisionProvider, "read", fake_vision),
        TestClient(
            app, base_url="http://127.0.0.1:8000", headers={"Origin": "http://127.0.0.1:8000"}
        ) as client,
    ):
        sid = client.post(
            "/api/workspace/sessions", json={"title": "test", "course_ids": ["pdf-obj-test"]}
        ).json()["session_id"]
        sbase = f"/api/workspace/sessions/{sid}"
        doc_ids = {}
        for pdf_path in (vec_pdf, fml_pdf):
            resp = client.post(
                f"{sbase}/import",
                params={"course_id": "pdf-obj-test", "filename": pdf_path.name},
                content=pdf_path.read_bytes(),
                headers={"Content-Type": "application/octet-stream"},
            )
            assert resp.status_code == 200
            doc_ids[pdf_path.name] = resp.json()["document_id"]

        # Q01: Right vector subfigure -> vision read + both /media and /preview return 200
        q1_resp = client.post(
            f"{sbase}/turns",
            json={
                "kind": "ask",
                "client_turn_id": "q01",
                "scope_revision": 1,
                "text": (
                    "In Vector-Subfigures-Guide.pdf on page 2, "
                    "what does the right subfigure (b) show?"
                ),
            },
        ).json()
        assert q1_resp["turn"]["status"] == "completed"
        assert q1_resp["turn"]["trace"]["selected_source"]["source_id"] == "page:2:image:2"
        q1_msg = q1_resp["message"]
        q1_img = next(b for b in q1_msg["blocks"] if b["type"] == "source_image")
        m1 = client.get(q1_img["media_url"])
        p1 = client.get(
            f"{sbase}/messages/{q1_msg['message_id']}/evidence/{q1_img['evidence_id']}/preview"
        )
        assert m1.status_code == 200 and p1.status_code == 200
        assert hashlib.sha256(m1.content).hexdigest() == hashlib.sha256(p1.content).hexdigest()

        # Q02: Figure 1 caption -> GET /evidence/{eid} 200 AND /preview 200 (object_crop)
        q2_resp = client.post(
            f"{sbase}/turns",
            json={
                "kind": "ask",
                "client_turn_id": "q02",
                "scope_revision": 1,
                "text": (
                    "What does the Figure 1 caption on page 2 "
                    "of Vector-Subfigures-Guide.pdf state?"
                ),
            },
        ).json()
        assert q2_resp["turn"]["status"] == "completed"
        assert q2_resp["turn"]["trace"]["selected_source"]["source_id"] == "page:2:caption:1"
        q2_msg = q2_resp["message"]
        ev2 = client.get(f"{sbase}/messages/{q2_msg['message_id']}/evidence/ev-1")
        assert ev2.status_code == 200, ev2.text
        assert "does not support causal delay claims" in ev2.json()["text"]
        assert ev2.json()["has_preview"] is True
        pv2 = client.get(f"{sbase}/messages/{q2_msg['message_id']}/evidence/ev-1/preview")
        assert pv2.status_code == 200

        # Q03: tau_eff formula (excluding distractor alpha_max) -> GET /evidence/{eid} 200
        q3_resp = client.post(
            f"{sbase}/turns",
            json={
                "kind": "ask",
                "client_turn_id": "q03",
                "scope_revision": 1,
                "text": (
                    "In Formula-And-Notes.pdf on page 2, "
                    "what is the formula for tau_eff and its condition?"
                ),
            },
        ).json()
        assert q3_resp["turn"]["status"] == "completed"
        assert q3_resp["turn"]["trace"]["selected_source"]["source_id"] == "page:2:formula:1"
        q3_msg = q3_resp["message"]
        ev3 = client.get(f"{sbase}/messages/{q3_msg['message_id']}/evidence/ev-1")
        assert ev3.status_code == 200, ev3.text
        assert "tau_eff = (W - 1) / 2" in ev3.json()["text"]
        assert "odd W >= 3 only" in ev3.json()["text"]
        assert "alpha_max" not in ev3.json()["text"]

        # Q04: Focus on page:2 + demonstrative question -> left/right options -> pick right
        q4_sid = client.post(
            "/api/workspace/sessions", json={"title": "q04", "course_ids": ["pdf-obj-test"]}
        ).json()["session_id"]
        q4_base = f"/api/workspace/sessions/{q4_sid}"
        cat = client.get(f"{q4_base}/catalog").json()
        p2_node = next(
            n
            for c in cat["courses"]
            for d in c["documents"]
            if d["document_id"] == doc_ids["Vector-Subfigures-Guide.pdf"]
            for n in d["nodes"]
            if n.get("locator", {}).get("page") == 2
        )
        foc = client.put(
            f"{q4_base}/focus", json={"node_id": p2_node["node_id"], "scope_revision": 1}
        )
        assert foc.status_code == 200
        rev = foc.json()["scope_revision"]

        q4_clarify = client.post(
            f"{q4_base}/turns",
            json={
                "kind": "ask",
                "client_turn_id": "q04-ask",
                "scope_revision": rev,
                "text": "这张图里的曲线趋势是什么？",
            },
        ).json()
        assert q4_clarify["turn"]["status"] == "clarification"
        assert q4_clarify["message"]["blocks"][0]["code"] == "object_ambiguous"
        assert q4_clarify["turn"]["trace"]["route"] == "target_object"
        assert q4_clarify["turn"]["trace"]["clarification_reason"] == "object_ambiguous"
        assert q4_clarify["turn"]["trace"]["external_call_count"] == 0
        assert len(q4_clarify["options"]) == 2
        labels = [opt["label"] for opt in q4_clarify["options"]]
        assert any("左图" in lbl and "Left Panel" in lbl for lbl in labels)
        assert any("右图" in lbl and "Right Panel" in lbl for lbl in labels)

        # Click right subfigure option -> reads page:2:image:2
        right_opt = next(opt for opt in q4_clarify["options"] if "右图" in opt["label"])
        q4_done = client.post(
            f"{q4_base}/turns",
            json={
                "kind": "ask",
                "client_turn_id": "q04-pick",
                "scope_revision": rev,
                "text": right_opt["label"],
                "anchor": {
                    "type": "clarification",
                    "turn_id": q4_clarify["turn"]["turn_id"],
                    "option_id": right_opt["option_id"],
                },
            },
        ).json()
        assert q4_done["turn"]["status"] == "completed"
        assert q4_done["turn"]["resolution"]["actual_source"]["source_id"] == "page:2:image:2"
        q4_msg = q4_done["message"]
        q4_img = next(b for b in q4_msg["blocks"] if b["type"] == "source_image")
        assert client.get(q4_img["media_url"]).status_code == 200
        assert (
            client.get(
                f"{q4_base}/messages/{q4_msg['message_id']}/evidence/{q4_img['evidence_id']}/preview"
            ).status_code
            == 200
        )

        # Q05: Table 1 -> /evidence 200, /media crop 200, /preview original_page 200
        q5_resp = client.post(
            f"{sbase}/turns",
            json={
                "kind": "ask",
                "client_turn_id": "q05",
                "scope_revision": 1,
                "text": (
                    "In Table 1 of Vector-Subfigures-Guide.pdf on page 1, "
                    "what are the nominal ranges?"
                ),
            },
        ).json()
        assert q5_resp["turn"]["status"] == "completed"
        assert q5_resp["turn"]["trace"]["selected_source"]["source_id"] == "page:1:table:1"
        q5_msg = q5_resp["message"]
        q5_img = next(b for b in q5_msg["blocks"] if b["type"] == "source_image")
        q5_crop = client.get(q5_img["media_url"])
        q5_page_prev = client.get(
            f"{sbase}/messages/{q5_msg['message_id']}/evidence/ev-1/preview"
        )
        assert q5_crop.status_code == 200 and q5_page_prev.status_code == 200
        # Branch C renders full original_page for native_table (taller than the table crop!)
        assert q5_crop.content != q5_page_prev.content
