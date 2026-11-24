"""G29 dedicated regression suite: 4 contract cracks, b2 security checks, and PDF guardrails."""

from __future__ import annotations

import hashlib
import io
import sqlite3
from pathlib import Path

import nbformat
from PIL import Image
from test_pdf_object_recovery import _build_multi_page_pdf, _line, _polyline, _rect, _t
from test_workspace import _ask, _create, _focus, _node, _upload
from test_workspace import system as system  # noqa: F401
from test_workspace_versions_recovery import _patch_fake_vision, _write_nb

from app.config import Settings
from app.rag.local_library import LocalLibrary
from app.rag.parsers import local_slides
from app.rag.parsers.local_slides import parse_local_slides
from app.rag.workspace_clarification import ClarificationOptionRecord, normalize_option_record
from app.rag.workspace_objects import object_choices, object_intent
from app.schemas.local import SourceKey
from app.stores.structure_snapshot import SnapshotRequest, StructureSnapshotStore


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _img_size(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as img:
        return img.size


def _build_pdf_with_form_xobject(
    *,
    form_stream: bytes,
    form_bbox: tuple[float, float, float, float],
    page_extra: str = "",
) -> bytes:
    """Build a valid PDF containing a top-level Form XObject (like \\includegraphics{fig.pdf})."""
    x0, y0, x1, y1 = form_bbox
    page_stream = f"q 1 0 0 1 0 0 cm /Fm1 Do Q\n{page_extra}".encode("latin-1")
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> /XObject << /Fm1 6 0 R >> >> "
            b"/Contents 5 0 R >>"
        ),
        (
            b"<< /Length "
            + str(len(page_stream)).encode()
            + b" >>\nstream\n"
            + page_stream
            + b"\nendstream"
        ),
        (
            f"<< /Type /XObject /Subtype /Form /BBox [{x0} {y0} {x1} {y1}] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Length {len(form_stream)} >>\nstream\n"
        ).encode("latin-1")
        + form_stream
        + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{idx} 0 obj\n".encode())
        out.extend(body)
        out.extend(b"\nendobj\n")
    xref = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode())
    out.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(out)


def test_pdf_guardrails_t1_t2_t3_header_footer_dual_booktabs_and_algorithm_rejection(
    tmp_path: Path,
) -> None:
    """T1 strips header/footer rules; T2 splits & extracts both tables; T3 rejects Algorithm box."""
    p1 = "\n".join(
        [
            # Header decorative rule near top 6% (y=768, > 0.12*792 above table 1)
            _line(60, 768, 550, 768, 0.8),
            _t(60, 772, "IEEE Conference Header", 9),
            # Table 1 (booktabs, 3 rules)
            _t(60, 650, "Table 1: First Model Comparison", 11),
            _line(60, 636, 550, 636, 1.2),
            _t(70, 618, "Model"),
            _t(260, 618, "Acc"),
            _line(60, 610, 550, 610, 0.8),
            _t(70, 592, "Alpha"),
            _t(260, 592, "91.2"),
            _t(70, 574, "Beta"),
            _t(260, 574, "93.4"),
            _line(60, 566, 550, 566, 1.2),
            # Intervening caption + prose between two same-width tables
            _t(60, 520, "Table 2: Second Latency Comparison", 11),
            # Table 2 (same width [60, 550], 3 rules)
            _line(60, 506, 550, 506, 1.2),
            _t(70, 488, "Backend"),
            _t(260, 488, "Latency"),
            _line(60, 480, 550, 480, 0.8),
            _t(70, 462, "CPU"),
            _t(260, 462, "18 ms"),
            _t(70, 444, "GPU"),
            _t(260, 444, "4 ms"),
            _line(60, 436, 550, 436, 1.2),
            # Footer rule near bottom 7% (y=42)
            _line(60, 42, 550, 42, 0.8),
            _t(60, 26, "1 Footnote text here", 8),
        ]
    )
    # Page 2: LaTeX Algorithm pseudocode block (3 horizontal rules, sequential line numbers)
    p2 = "\n".join(
        [
            _line(60, 650, 550, 650, 1.2),
            _t(70, 632, "Algorithm 1: Greedy Search"),
            _line(60, 624, 550, 624, 0.8),
            _t(70, 606, "1:"),
            _t(120, 606, "Initialize state s = 0"),
            _t(70, 588, "2:"),
            _t(120, 588, "Update s = step(s)"),
            _t(70, 570, "3:"),
            _t(120, 570, "Return s"),
            _line(60, 560, 550, 560, 1.2),
        ]
    )
    pdf_path = tmp_path / "tables_and_algo.pdf"
    pdf_path.write_bytes(_build_multi_page_pdf([p1, p2]))

    parsed = parse_local_slides(pdf_path)
    p1_tables = [
        s
        for s in parsed.sources
        if s.locator.get("page") == 1 and s.locator.get("object_type") == "table"
    ]
    p2_tables = [
        s
        for s in parsed.sources
        if s.locator.get("page") == 2 and s.locator.get("object_type") == "table"
    ]
    assert len(p1_tables) == 2
    assert "Alpha" in p1_tables[0].text and "Latency" in p1_tables[1].text
    assert p2_tables == []


def test_pdf_guardrails_v3_v5_form_xobject_bleed_and_framed_textbox_rejection(
    tmp_path: Path,
) -> None:
    """V5 registers slightly bleeding Form XObject plots; V3 rejects framed prose textboxes."""
    # Form XObject plot slightly bleeding left (-2 pt -> >99% inside page_bbox)
    plot_stream = "\n".join(
        [
            _rect(-2, 420, 240, 200),
            _line(20, 450, 210, 450),
            _line(20, 450, 20, 590),
            _polyline([(20, 460), (60, 510), (100, 560), (140, 520), (180, 575), (205, 580)]),
            _t(50, 600, "Form Plot Curve", 10),
        ]
    ).encode("latin-1")
    # Simple 4-edge framed prose box on right side (must be rejected by V3 as image!)
    framed_prose = "\n".join(
        [
            _rect(320, 420, 240, 180),
            _line(320, 575, 560, 575),
            _t(330, 582, "Theorem 1. Convergence Bound.", 10),
            _t(330, 550, "Under mild smoothness assumptions on the loss", 10),
            _t(330, 534, "function, gradient descent converges monotonically.", 10),
            _t(330, 518, "All iterates stay inside a bounded compact set.", 10),
        ]
    )
    pdf_path = tmp_path / "form_and_textbox.pdf"
    pdf_path.write_bytes(
        _build_pdf_with_form_xobject(
            form_stream=plot_stream,
            form_bbox=(-2, 420, 238, 620),
            page_extra=framed_prose,
        )
    )
    parsed = parse_local_slides(pdf_path)
    imgs = [s for s in parsed.sources if s.locator.get("object_type") == "image"]
    assert len(imgs) == 1
    assert imgs[0].locator["position_basis"] == "pdfium_native_vector_extents_v1"
    assert imgs[0].locator["bbox"][0] == 0.0  # clamped cleanly to visible page left
    assert imgs[0].locator["bbox"][2] < 0.5  # only the Form plot, NOT the framed theorem box


def test_pdf_guardrails_x1_x2_x3_two_column_caption_eq_number_reassoc_and_prose_rejection(
    tmp_path: Path,
) -> None:
    """X1 splits two columns & re-associates (1); X2 rejects inline lr=1e-4 & merges fractions."""
    # Two-column layout: left col [40, 285], gutter ~306, right col [325, 570]
    p1 = "\n".join(
        [
            # Left column body paragraphs (4+ boxes to activate two-column gutter detection)
            _t(45, 720, "Left column intro paragraph line one here.", 10),
            _t(45, 704, "Left column intro paragraph line two here.", 10),
            # Left column inline equation in prose (MUST NOT become a standalone formula!)
            _t(45, 670, "We set lr = 1e-4 and batch_size = 32 in our", 10),
            _t(45, 654, "experiments across every evaluation benchmark.", 10),
            # Left column centered display equation with right-aligned (1) (gap > 50 pt!)
            _t(110, 610, "L(theta) = sum_i ||y_i - f(x_i)||^2", 10),
            _t(268, 610, "(1)", 10),
            # Left column fraction-like multi-line display formula
            _t(130, 560, "W - 1", 10),
            _t(95, 548, "tau_eff = -----", 10),
            _t(138, 536, "2", 10),
            _t(95, 522, "Condition: for odd W >= 3 only", 9),
            # Left column caption at y=470
            _t(45, 470, "Figure 1: Left column caption text.", 10),
            _t(45, 456, "Applies only to the left panel.", 10),
            # Right column text at same y coordinates (MUST NOT merge into left caption/eq!)
            _t(330, 720, "Right column paragraph line one here.", 10),
            _t(330, 704, "Right column paragraph line two here.", 10),
            _t(330, 670, "Right column discussion continues smoothly.", 10),
            _t(330, 610, "Right column unrelated text at y=610.", 10),
            _t(330, 470, "Right column prose must not leak into Figure 1.", 10),
            _t(330, 456, "Second right column line at caption height.", 10),
        ]
    )
    pdf_path = tmp_path / "two_column_academic.pdf"
    pdf_path.write_bytes(_build_multi_page_pdf([p1]))

    parsed = parse_local_slides(pdf_path)
    captions = [s for s in parsed.sources if s.locator.get("object_type") == "caption"]
    formulas = [s for s in parsed.sources if s.locator.get("object_type") == "formula"]

    assert len(captions) == 1
    cap_txt = captions[0].locator["object_text"]
    assert "Left column caption text" in cap_txt
    assert "Applies only to the left panel" in cap_txt
    assert "Right column" not in cap_txt

    formula_texts = [f.locator["object_text"] for f in formulas]
    assert not any("lr = 1e-4" in t for t in formula_texts)
    assert any("L(theta)" in t and "(1)" in t and "Right column" not in t for t in formula_texts)
    assert any("tau_eff" in t and "W - 1" in t and "odd W >= 3" in t for t in formula_texts)


def test_same_byte_reimport_appends_objects_preserves_keys_and_visual_ordinal_order(
    tmp_path: Path, monkeypatch
) -> None:
    """Re-importing same-byte PDF keeps legacy keys & FTS untouched while restoring visual order."""
    pdf = tmp_path / "reimport_compat.pdf"
    p1 = "\n".join(
        [
            _t(50, 740, "Header Overview", 12),
            _rect(60, 560, 480, 60),
            _t(80, 586, "tau_eff = (W - 1) / 2", 11),
            _t(50, 480, "Figure 1: Schematic only.", 10),
        ]
    )
    pdf.write_bytes(_build_multi_page_pdf([p1]))

    settings = Settings(
        _env_file=None, sqlite_path=tmp_path / "lib.db", data_dir=tmp_path, course_root=tmp_path
    )
    library = LocalLibrary(settings)

    with monkeypatch.context() as legacy:
        legacy.setattr(local_slides, "_pdf_text_object_sources", lambda *args, **kwargs: [])
        (old,) = library.import_paths("demo", [pdf])
    assert old.source_count == 1
    assert old.searchable_count == 1

    (updated,) = library.import_paths("demo", [pdf])
    (again,) = library.import_paths("demo", [pdf])
    assert updated.version == again.version == old.version
    assert old.publication_epoch + 1 == updated.publication_epoch == again.publication_epoch
    assert updated.source_count == again.source_count == 3
    # caption/formula keep SourceDraft.text="" so searchable_count & FTS rows remain unchanged!
    assert updated.searchable_count == old.searchable_count == 1

    with sqlite3.connect(settings.sqlite_path) as conn:
        fts_count = conn.execute("SELECT count(*) FROM local_fts").fetchone()[0]
    assert fts_count == 1

    snap = StructureSnapshotStore(settings).read(SnapshotRequest(course_ids=("demo",)))
    fml = object_choices(object_intent("公式 tau_eff 是什么？"), snap.sources)
    assert len(fml) == 1 and fml[0].source_id == "page:1:formula:1"


def test_b2_security_checks_and_unread_not_attempted_floor(system, tmp_path, monkeypatch) -> None:
    """Verify b2 7-point security checks, legacy option compat, and not_attempted floor."""
    settings, service, _, client = system
    settings.source_vision_enabled = True
    _patch_fake_vision(service, monkeypatch)

    # Legacy option normalization backward compatibility
    legacy_display = normalize_option_record(
        {
            "option_id": "o1",
            "label": "图1",
            "kind": "display",
            "message_id": "m1",
            "display_id": "d1",
        }
    )
    assert legacy_display.resume_action == "follow_display"
    legacy_obj = normalize_option_record(
        {
            "option_id": "o2",
            "label": "对象",
            "kind": "document",
            "node_id": "n1",
            "current_object": True,
        }
    )
    assert legacy_obj.resume_action == "read_object"
    legacy_doc = normalize_option_record(
        {"option_id": "o3", "label": "文件", "kind": "document", "node_id": "n2"}
    )
    assert legacy_doc.resume_action == "retrieve_document"

    v1_path = tmp_path / "Sec-v1.ipynb"
    v2_reord = tmp_path / "Sec-v2.ipynb"
    v3_path = tmp_path / "Sec-v3.ipynb"
    _write_nb(v1_path, "v1")
    _write_nb(v2_reord, "v2r", reordered=True)
    _write_nb(v3_path, "v3")

    sess_a = _create(client)
    doc1 = _upload(client, sess_a, v1_path, filename="Sec.ipynb")
    cell_node = _node(client, sess_a, doc1["document_id"], "cell", cell_index=1)
    rev_a = _focus(client, sess_a, cell_node)
    _ask(client, sess_a, "展示这个 Cell 的两张图", rev_a, client_id="sec-show")
    _ask(client, sess_a, "刚才第一张图是什么？", rev_a, client_id="sec-read")
    _upload(client, sess_a, v2_reord, filename="Sec.ipynb", document_id=doc1["document_id"])

    # Trigger b2 clarification with general attribute=None wording ("有什么变化")
    clarify = _ask(
        client, sess_a, "对比最新版和上一版刚才读过的图有什么变化？", rev_a, client_id="sec-b2"
    )
    assert clarify["turn"]["status"] == "clarification"
    assert len(clarify["options"]) == 2
    # Public options must NOT leak internal resume fields!
    assert all(set(opt) == {"option_id", "label", "kind"} for opt in clarify["options"])

    # Cross-session replay of sess_a's ClarificationAnchor -> HTTP 404
    sess_b = _create(client)
    cross_resp = client.post(
        f"{sess_b}/turns",
        json={
            "kind": "ask",
            "client_turn_id": "cross-b2",
            "scope_revision": 1,
            "text": "选择第2项",
            "anchor": {
                "type": "clarification",
                "turn_id": clarify["turn"]["turn_id"],
                "option_id": clarify["options"][1]["option_id"],
            },
        },
    )
    assert cross_resp.status_code == 404

    # Unknown option_id -> HTTP 404
    bad_opt = client.post(
        f"{sess_a}/turns",
        json={
            "kind": "ask",
            "client_turn_id": "bad-opt",
            "scope_revision": rev_a,
            "text": "选择",
            "anchor": {
                "type": "clarification",
                "turn_id": clarify["turn"]["turn_id"],
                "option_id": "nonexistent-option",
            },
        },
    )
    assert bad_opt.status_code == 404

    # Valid resume with attribute=None succeeds bilaterally (v1 + v2, both succeeded!)
    resumed = _ask(
        client,
        sess_a,
        "选第2项",
        rev_a,
        client_id="sec-b2-ok",
        anchor={
            "type": "clarification",
            "turn_id": clarify["turn"]["turn_id"],
            "option_id": clarify["options"][1]["option_id"],
        },
    )
    assert resumed["turn"]["status"] == "completed"
    assert resumed["turn"]["trace"]["route"] == "version_comparison"
    sides = {
        ev["revision"]["source"]["version"]: ev for ev in resumed["message"]["sources"].values()
    }
    assert set(sides) == {1, 2}
    assert sides[1]["read_status"] == "succeeded"
    assert sides[2]["read_status"] == "succeeded"

    # Cross-session access to resumed /preview and /media -> HTTP 404
    resumed_mid = resumed["message"]["message_id"]
    for blk in resumed["message"]["blocks"]:
        if blk["type"] == "source_image":
            foreign_media = blk["media_url"].replace(sess_a, sess_b, 1)
            assert client.get(foreign_media).status_code == 404
            assert (
                client.get(
                    f"{sess_b}/messages/{resumed_mid}/evidence/{blk['evidence_id']}/preview"
                ).status_code
                == 404
            )

    # Publish v3 after clarification -> replaying v2 clarification fails with stale_reference
    _upload(client, sess_a, v3_path, filename="Sec.ipynb", document_id=doc1["document_id"])
    stale_resume = _ask(
        client,
        sess_a,
        "重放旧v2选项",
        rev_a,
        client_id="sec-b2-stale",
        anchor={
            "type": "clarification",
            "turn_id": clarify["turn"]["turn_id"],
            "option_id": clarify["options"][1]["option_id"],
        },
    )
    assert stale_resume["turn"]["status"] == "failed"
    assert stale_resume["turn"]["error"] == "stale_reference"

    # Safety floor: v1 image ONLY displayed (never visually read) -> historical_image_unread!
    sess_unread = _create(client)
    nb_one_v1 = tmp_path / "One-v1.ipynb"
    nb_one_v2 = tmp_path / "One-v2.ipynb"
    for p, tag in ((nb_one_v1, "v1"), (nb_one_v2, "v2")):
        _write_nb(p, tag)
        # Keep only 1 output image so Turn 1 displays 1 image without reading it
        nb = nbformat.read(p, as_version=4)
        nb.cells[1].outputs = nb.cells[1].outputs[:1]
        nbformat.write(nb, p)
    u1 = _upload(client, sess_unread, nb_one_v1, filename="Unread.ipynb")
    u_cell = _node(client, sess_unread, u1["document_id"], "cell", cell_index=1)
    u_rev = _focus(client, sess_unread, u_cell)
    shown_only = _ask(client, sess_unread, "展示这个 Cell 的原图", u_rev, client_id="unread-show")
    assert len(shown_only["message"]["visible_display_ids"]) == 1
    img_blocks = [b for b in shown_only["message"]["blocks"] if b["type"] == "source_image"]
    assert len(img_blocks) == 1
    assert shown_only["message"]["sources"][img_blocks[0]["evidence_id"]]["read_status"] == (
        "not_attempted"
    )
    _upload(client, sess_unread, nb_one_v2, filename="Unread.ipynb", document_id=u1["document_id"])
    cmp_unread = _ask(
        client,
        sess_unread,
        "比较 Unread.ipynb 上一版和当前版 Cell 2 第 1 张图的纵轴单位。",
        u_rev,
        client_id="unread-cmp",
    )
    assert cmp_unread["turn"]["trace"]["historical_read_status"] == "not_attempted"
    assert any(b.get("code") == "historical_image_unread" for b in cmp_unread["message"]["blocks"])
    # Ensure ClarificationOptionRecord enforces extra="forbid"
    assert ClarificationOptionRecord.model_config.get("extra") == "forbid"
    assert SourceKey(document_id="d", version=1, source_id="s").version == 1


def _word_tokens(x_start: float, y: float, sentence: str, size: int = 10) -> list[str]:
    """Emit word-level BT/ET objects simulating real LaTeX PDFium token granularity."""
    cmds: list[str] = []
    cursor = x_start
    for word in sentence.split():
        cmds.append(_t(cursor, y, word, size))
        cursor += max(8.0, len(word) * size * 0.52) + size * 0.35
    return cmds


def test_old_version_isolation_and_four_turn_v1_v2_only_old_compare(
    system, tmp_path, monkeypatch
) -> None:
    """Turn 1 v1 read -> v2 upload -> Turn 2 v2 read -> Turn 3 only-old v1 -> Turn 4 compare."""
    settings, service, _, client = system
    settings.source_vision_enabled = True
    _patch_fake_vision(service, monkeypatch)

    v1_path = tmp_path / "Chain-v1.ipynb"
    v2_path = tmp_path / "Chain-v2.ipynb"
    _write_nb(v1_path, "v1")
    _write_nb(v2_path, "v2")

    sess = _create(client)
    doc1 = _upload(client, sess, v1_path, filename="Chain.ipynb")
    cell_node = _node(client, sess, doc1["document_id"], "cell", cell_index=1)
    rev = _focus(client, sess, cell_node)

    # Turn 1: show v1 images & read v1 output:0
    _ask(client, sess, "展示这个 Cell 的两张图", rev, client_id="c4-show-v1")
    t1 = _ask(client, sess, "刚才第一张图的纵轴单位是什么？", rev, client_id="c4-read-v1")
    assert t1["turn"]["status"] == "completed"
    t1_ev = next(iter(t1["message"]["sources"].values()))
    assert t1_ev["revision"]["source"]["version"] == 1

    # Upgrade to v2 without PUT /focus
    _upload(client, sess, v2_path, filename="Chain.ipynb", document_id=doc1["document_id"])

    # Turn 2: ask current version (v2) -> reads v2
    t2 = _ask(client, sess, "最新版刚才读过的那张图是什么？", rev, client_id="c4-read-v2")
    assert t2["turn"]["status"] == "completed"
    t2_vers = {ev["revision"]["source"]["version"] for ev in t2["message"]["sources"].values()}
    assert t2_vers == {2}

    # Turn 3: explicitly ask ONLY old version -> MUST return v1, NEVER v2!
    t3 = _ask(client, sess, "只看旧版刚才读过的那张图是什么？", rev, client_id="c4-only-old")
    assert t3["turn"]["status"] == "completed"
    t3_vers = {ev["revision"]["source"]["version"] for ev in t3["message"]["sources"].values()}
    assert t3_vers == {1}
    assert any("v1" in blk["text"] for blk in t3["message"]["blocks"] if blk["type"] == "text")

    # Turn 4: compare latest and old -> MUST NOT lose v1 history!
    t4 = _ask(
        client,
        sess,
        "对比最新版和旧版刚才读过的这张图有什么变化？",
        rev,
        client_id="c4-compare",
    )
    assert t4["turn"]["status"] == "completed"
    assert t4["turn"]["trace"]["route"] == "version_comparison"
    sides = {ev["revision"]["source"]["version"]: ev for ev in t4["message"]["sources"].values()}
    assert set(sides) == {1, 2}
    assert sides[1]["read_status"] == "succeeded"
    assert sides[2]["read_status"] == "succeeded"

    # Direct Turn 2 (v2 read) -> Turn 3 (compare without intermediate only-old turn)
    sess_skip = _create(client)
    d_s = _upload(client, sess_skip, v1_path, filename="Skip.ipynb")
    c_s = _node(client, sess_skip, d_s["document_id"], "cell", cell_index=1)
    r_s = _focus(client, sess_skip, c_s)
    _ask(client, sess_skip, "展示这个 Cell 的两张图", r_s, client_id="sk-show")
    _ask(client, sess_skip, "刚才第一张图的纵轴单位是什么？", r_s, client_id="sk-v1")
    _upload(client, sess_skip, v2_path, filename="Skip.ipynb", document_id=d_s["document_id"])
    t_v2 = _ask(client, sess_skip, "最新版刚才读过的那张图是什么？", r_s, client_id="sk-v2")
    assert {ev["revision"]["source"]["version"] for ev in t_v2["message"]["sources"].values()} == {
        2
    }
    t_cmp = _ask(
        client,
        sess_skip,
        "对比最新版和旧版刚才读过的这张图有什么变化？",
        r_s,
        client_id="sk-cmp",
    )
    assert t_cmp["turn"]["status"] == "completed"
    assert t_cmp["turn"]["trace"]["route"] == "version_comparison"
    cmp_sides = {
        ev["revision"]["source"]["version"]: ev for ev in t_cmp["message"]["sources"].values()
    }
    assert set(cmp_sides) == {1, 2}
    assert cmp_sides[1]["read_status"] == "succeeded"
    assert cmp_sides[2]["read_status"] == "succeeded"

    # No-old-history safety: session that only ever read v2 asks "只看旧版" -> MUST NOT leak v2!
    sess_no_old = _create(client)
    d_no = _upload(client, sess_no_old, v2_path, filename="OnlyV2.ipynb")
    c_no = _node(client, sess_no_old, d_no["document_id"], "cell", cell_index=1)
    r_no = _focus(client, sess_no_old, c_no)
    _ask(client, sess_no_old, "展示这个 Cell 的两张图", r_no, client_id="no-show")
    _ask(client, sess_no_old, "刚才第一张图是什么？", r_no, client_id="no-read")
    only_old_missing = _ask(
        client, sess_no_old, "只看旧版刚才读过的那张图是什么？", r_no, client_id="no-old"
    )
    assert only_old_missing["turn"]["status"] == "clarification"
    assert (
        only_old_missing["turn"]["trace"]["clarification_reason"] == "missing_old_version_history"
    )
    assert not only_old_missing["message"]["sources"]


def test_pdf_single_column_word_tokens_fraction_eq_algorithm_and_two_col_full_caption(
    tmp_path: Path,
) -> None:
    """Word tokens never split at x=0.50; fractions & eq (1) merge; fragments/Algo rejected."""
    # Page 1: Single-column LaTeX style with 1-inch margins [72, 540] and word-level BT/ET tokens
    p1_cmds: list[str] = []
    # Body paragraph that wraps mid-sentence onto "Figure 2. We observe..." (MUST NOT be a caption!)
    p1_cmds.extend(
        _word_tokens(
            72,
            730,
            "In our empirical evaluation across diverse benchmarks we summarize all curves in",
            10,
        )
    )
    p1_cmds.extend(
        _word_tokens(
            72,
            716,
            "Figure 2. We observe that convergence remains stable across every tested seed.",
            10,
        )
    )
    # Full-width genuine caption spanning well past x=306 (MUST stay intact, NOT chopped at x=0.50!)
    p1_cmds.extend(
        _word_tokens(
            72,
            660,
            "Figure 1: Complete single-column caption spanning cleanly past the page midline.",
            10,
        )
    )
    p1_cmds.extend(
        _word_tokens(
            72,
            646,
            "Second caption line verifying full width preservation without half-line splits.",
            10,
        )
    )
    # Centered display fraction equation with right-margin (1) at 1-inch margin (x=522)
    p1_cmds.extend(_word_tokens(185, 580, "Attention(Q, K, V) =", 10))
    p1_cmds.append(_t(315, 590, "Q K^T", 10))
    p1_cmds.append(_line(308, 584, 368, 584, 0.7))
    p1_cmds.append(_t(314, 571, "sqrt(d_k)", 10))
    p1_cmds.append(_t(522, 580, "(1)", 10))

    # Stray broken math fragments (MUST ALL be rejected as standalone formulas!)
    p1_cmds.append(_t(280, 520, "=", 10))
    p1_cmds.append(_t(260, 495, "(x) =", 10))
    p1_cmds.append(_t(260, 470, "= 1)", 10))

    # Rule-bounded Algorithm 1 pseudocode float containing "=" assignments (MUST NOT emit formulas!)
    p1_cmds.append(_line(72, 420, 540, 420, 1.0))
    p1_cmds.extend(_word_tokens(76, 404, "Algorithm 1: Gradient Descent Update", 10))
    p1_cmds.append(_line(72, 396, 540, 396, 0.7))
    p1_cmds.extend(_word_tokens(76, 380, "1: theta = theta - lr * grad", 10))
    p1_cmds.extend(_word_tokens(76, 364, "2: loss = evaluate(theta)", 10))
    p1_cmds.extend(_word_tokens(76, 348, "3: return theta", 10))
    p1_cmds.append(_line(72, 338, 540, 338, 1.0))

    # Page 2: True two-column layout WITH a full-width spanning Figure 2 caption across gutter!
    p2_cmds: list[str] = []
    # Full-width spanning caption at top (word-level tokens crossing x=306)
    p2_cmds.extend(
        _word_tokens(
            52,
            710,
            "Figure 2: Full-width spanning caption across both columns in a two-column paper.",
            10,
        )
    )
    # 5 left-column lines and 5 right-column lines cleanly separated by a ~45pt gutter
    for idx in range(5):
        y = 650 - idx * 18
        p2_cmds.extend(
            _word_tokens(52, y, f"Left column paragraph line {idx} with normal text.", 10)
        )
        p2_cmds.extend(
            _word_tokens(332, y, f"Right column paragraph line {idx} with normal text.", 10)
        )

    pdf_path = tmp_path / "latex_layout_guardrails.pdf"
    pdf_path.write_bytes(_build_multi_page_pdf(["\n".join(p1_cmds), "\n".join(p2_cmds)]))
    parsed = parse_local_slides(pdf_path)

    p1_captions = [
        s
        for s in parsed.sources
        if s.locator.get("page") == 1 and s.locator.get("object_type") == "caption"
    ]
    p1_formulas = [
        s
        for s in parsed.sources
        if s.locator.get("page") == 1 and s.locator.get("object_type") == "formula"
    ]
    p2_captions = [
        s
        for s in parsed.sources
        if s.locator.get("page") == 2 and s.locator.get("object_type") == "caption"
    ]

    # Page 1: Exactly 1 intact Figure 1 caption (inline "Figure 2." wrap rejected, no x=0.50 split!)
    assert len(p1_captions) == 1
    cap1 = p1_captions[0].locator["object_text"]
    assert "Figure 1:" in cap1 and "page midline" in cap1 and "half-line splits" in cap1
    assert p1_captions[0].locator["bbox"][2] > 0.65

    # Page 1: Exactly 1 complete fraction formula with (1) attached; zero fragments or Algo lines!
    assert len(p1_formulas) == 1
    f1 = p1_formulas[0].locator["object_text"]
    assert "Attention" in f1 and "sqrt(d_k)" in f1 and "(1)" in f1
    assert not any("theta" in f.locator["object_text"] for f in p1_formulas)
    assert not any("loss" in f.locator["object_text"] for f in p1_formulas)

    # Page 2: Spanning Figure 2 caption stays intact across the two-column gutter
    assert len(p2_captions) == 1
    cap2 = p2_captions[0].locator["object_text"]
    assert "Figure 2:" in cap2 and "two-column paper" in cap2
    assert p2_captions[0].locator["bbox"][2] > 0.65


def test_same_page_multi_table_global_label_mapping(system, tmp_path) -> None:
    """Asking '第 2 页 Table 4' on a page with 3 tables (Table 2/3/4) maps to global Table 4."""
    _, _, _, client = system
    # Page 1: Table 1
    p1 = "\n".join(
        [
            _t(60, 680, "Table 1: Baseline Scores", 11),
            _line(60, 666, 520, 666, 1.2),
            _t(70, 648, "Model"),
            _t(260, 648, "F1"),
            _line(60, 640, 520, 640, 0.8),
            _t(70, 622, "Base"),
            _t(260, 622, "80.1"),
            _t(70, 604, "Large"),
            _t(260, 604, "84.5"),
            _line(60, 596, 520, 596, 1.2),
        ]
    )
    # Page 2: 3 tables globally numbered Table 2, Table 3, Table 4
    p2 = "\n".join(
        [
            _t(60, 720, "Table 2: Ablation Results", 11),
            _line(60, 706, 520, 706, 1.2),
            _t(70, 688, "Variant"),
            _t(260, 688, "Gain"),
            _line(60, 680, 520, 680, 0.8),
            _t(70, 662, "NoNorm"),
            _t(260, 662, "+1.2"),
            _t(70, 644, "FullNorm"),
            _t(260, 644, "+3.4"),
            _line(60, 636, 520, 636, 1.2),
            _t(60, 580, "Table 3: Throughput Metrics", 11),
            _line(60, 566, 520, 566, 1.2),
            _t(70, 548, "Batch"),
            _t(260, 548, "QPS"),
            _line(60, 540, 520, 540, 0.8),
            _t(70, 522, "B16"),
            _t(260, 522, "420"),
            _t(70, 504, "B32"),
            _t(260, 504, "790"),
            _line(60, 496, 520, 496, 1.2),
            _t(60, 440, "Table 4: Cross-Domain Transfer", 11),
            _line(60, 426, 520, 426, 1.2),
            _t(70, 408, "Domain"),
            _t(260, 408, "AUC"),
            _line(60, 400, 520, 400, 0.8),
            _t(70, 382, "Medical"),
            _t(260, 382, "95.8"),
            _t(70, 364, "Finance"),
            _t(260, 364, "94.2"),
            _line(60, 356, 520, 356, 1.2),
        ]
    )
    pdf_path = tmp_path / "MultiTablePaper.pdf"
    pdf_path.write_bytes(_build_multi_page_pdf([p1, p2]))

    sess = _create(client)
    _upload(client, sess, pdf_path, filename="MultiTablePaper.pdf")

    # "第 2 页 Table 4" -> ordinal=4 > 3 local tables -> MUST resolve to Table 4 (Medical/Finance)!
    q_t4 = _ask(
        client, sess, "MultiTablePaper.pdf 第 2 页 Table 4 写了什么？", 1, client_id="mt-t4"
    )
    assert q_t4["turn"]["status"] == "completed"
    t4_src = next(iter(q_t4["message"]["sources"].values()))
    assert t4_src["revision"]["source"]["source_id"] == "page:2:table:3"
    assert "Medical" in t4_src["reads"][0]["text"] and "95.8" in t4_src["reads"][0]["text"]

    # "第 2 页 Table 2" -> MUST resolve to Table 2 (page:2:table:1), NOT local 2nd table!
    q_t2 = _ask(
        client, sess, "MultiTablePaper.pdf 第 2 页 Table 2 写了什么？", 1, client_id="mt-t2"
    )
    assert q_t2["turn"]["status"] == "completed"
    t2_src = next(iter(q_t2["message"]["sources"].values()))
    assert t2_src["revision"]["source"]["source_id"] == "page:2:table:1"
    assert "FullNorm" in t2_src["reads"][0]["text"]

    # "第 2 页第 2 张表" (pure local ordinal, no Table M) -> resolves to local 2nd table (Table 3)
    q_loc2 = _ask(
        client, sess, "MultiTablePaper.pdf 第 2 页第 2 张表写了什么？", 1, client_id="mt-loc2"
    )
    assert q_loc2["turn"]["status"] == "completed"
    loc2_src = next(iter(q_loc2["message"]["sources"].values()))
    assert loc2_src["revision"]["source"]["source_id"] == "page:2:table:2"
    assert "QPS" in loc2_src["reads"][0]["text"]
