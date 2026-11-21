from __future__ import annotations

import base64
import io
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from pptx import Presentation
from pptx.util import Inches

from app.config import Settings
from app.rag.local_library import LocalLibrary
from app.rag.parsers.pdf import PDFParser
from app.rag.structure_search import StructureScope, StructureSearcher, StructureSearchRequest
from app.rag.workspace_generation import (
    _CHAT_FRAME_RESERVE,
    _LOCATOR_FIELDS,
    OUTPUT_RESERVE,
    estimate_prompt_reserve,
)
from app.rag.workspace_reading import (
    estimate_embedded_json_budget,
    estimate_embedded_text_budget,
    estimate_input_budget,
    read_window,
)
from app.schemas.local import SourceKey
from app.schemas.rag import SearchRequest
from app.schemas.workspace import WorkspaceError
from app.stores.structure_snapshot import SnapshotRequest, StructureSnapshotStore

COURSE = "g15-public"
G14_FIXTURE = Path(__file__).parents[1] / "docs/G/G14-fixtures.py"


@pytest.fixture
def library(tmp_path):
    return LocalLibrary(
        Settings(
            course_root=tmp_path,
            data_dir=tmp_path,
            sqlite_path=tmp_path / "library.db",
            _env_file=None,
        )
    )


@pytest.fixture
def lecture(library, tmp_path):
    fixture = runpy.run_path(str(G14_FIXTURE))
    path = tmp_path / fixture["FILENAME"]
    fixture["make_pdf"](path)
    return path, _publish(library, path), fixture["EXPECTED"]


def _publish(library, path, document_id=None):
    (result,) = library.import_paths(COURSE, [path], document_id=document_id)
    assert result.status == "published"
    return result


def _snapshot(library, *documents):
    return StructureSnapshotStore(library.settings).read(SnapshotRequest((COURSE,), documents))


def test_short_file_reads_full_native_pages_with_exact_offsets(library, lecture):
    _, result, expected = lecture
    snapshot = _snapshot(library, result.document_id)
    window = read_window(library.store, snapshot, context_tokens=30_000, output_reserve=2048)

    assert window.full_text_scope and window.next_cursor is None
    assert [part.source.source_id for part in window.segments] == [f"page:{i}" for i in range(1, 9)]
    for part in window.segments:
        assert part.text == part.source.text[part.start : part.end]
        assert part.document.document_id == result.document_id and part.document.version == 1
        assert all(term in part.text for term in expected.get(part.source.locator["page"], ()))
    assert window.trace()["visual_read_status"] == "not_attempted"
    assert window.trace()["outline_total"] == 8


def test_long_cursor_reads_related_section_first_without_gaps(library, lecture):
    _, result, _ = lecture
    snapshot = _snapshot(library, result.document_id)
    preferred = [SourceKey(document_id=result.document_id, version=1, source_id="page:8")]
    inputs = dict(
        context_tokens=800, output_reserve=150, prompt_reserve=100, preferred_sources=preferred
    )
    window = read_window(library.store, snapshot, **inputs)
    assert window.prioritized and not window.full_text_scope
    assert window.segments[0].source.source_id == "page:8"
    assert window.outline[-1]["source"]["source_id"] == "page:8"
    seen = {}
    for _ in range(30):
        assert window.estimated_input <= window.budget
        assert window.segments
        for part in window.segments:
            saved = seen.setdefault(part.source.source_id, "")
            assert part.start == len(saved)
            seen[part.source.source_id] = saved + part.text
        if not window.next_cursor:
            break
        window = read_window(library.store, snapshot, cursor=window.next_cursor, **inputs)
    assert seen == {source.source_id: source.text for source in snapshot.sources}


def test_cursor_and_old_publication_rejected_after_replacement(library, lecture):
    path, result, _ = lecture
    snapshot = _snapshot(library, result.document_id)
    inputs = dict(context_tokens=800, output_reserve=150)
    first = read_window(library.store, snapshot, **inputs)
    assert first.next_cursor
    path.write_bytes(path.read_bytes() + b"\n% version two\n")
    assert _publish(library, path, result.document_id).version == 2
    with pytest.raises(WorkspaceError, match="stale_reference"):
        read_window(library.store, snapshot, **inputs)
    with pytest.raises(WorkspaceError, match="stale_reference"):
        read_window(
            library.store,
            _snapshot(library, result.document_id),
            cursor=first.next_cursor,
            **inputs,
        )


def test_impossibly_small_context_does_not_return_a_stuck_cursor(library, lecture):
    _, result, _ = lecture
    with pytest.raises(ValueError, match="cannot fit"):
        read_window(
            library.store,
            _snapshot(library, result.document_id),
            context_tokens=101,
            output_reserve=100,
        )


def test_pptx_rows_and_notebook_outputs_have_separate_source_keys(library, tmp_path):
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[5])
    slide.shapes.title.text = "Calibration workshop"
    slide.shapes.add_textbox(
        Inches(1), Inches(4), Inches(7), Inches(1)
    ).text = "Figure 1. Distinguish cold-start measurements from steady state."
    table = slide.shapes.add_table(2, 2, Inches(1), Inches(2), Inches(7), Inches(1)).table
    table.cell(0, 0).text = "Trial"
    table.cell(0, 1).text = "Nominal rate [Hz]"
    table.cell(1, 0).text = "Blue"
    table.cell(1, 1).text = "47"
    pptx_file = tmp_path / "calibration.pptx"
    deck.save(pptx_file)
    speaker = _publish(library, pptx_file)

    image = io.BytesIO()
    Image.new("RGB", (10, 10)).save(image, format="PNG")
    notebook_file = tmp_path / "readback.ipynb"
    notebook_file.write_text(
        json.dumps(
            {
                "cells": [
                    {"id": "section", "cell_type": "markdown", "source": "# Baseline protocol"},
                    {
                        "id": "measure",
                        "cell_type": "code",
                        "source": "collect()",
                        "outputs": [
                            {"output_type": "stream", "name": "stdout", "text": "Mean: 47 Hz"},
                            {
                                "output_type": "display_data",
                                "data": {"image/png": base64.b64encode(image.getvalue()).decode()},
                            },
                            {"output_type": "stream", "name": "stdout", "text": "Variance: 4 Hz^2"},
                        ],
                    },
                ]
            }
        )
    )
    notebook = _publish(library, notebook_file)
    snapshot = _snapshot(library, speaker.document_id, notebook.document_id)
    result = read_window(library.store, snapshot, context_tokens=30_000, output_reserve=2048)

    (native_table,) = [part for part in result.segments if part.method == "native_table"]
    assert native_table.table_headers == ("Trial", "Nominal rate [Hz]")
    assert "| Blue | 47 |" in native_table.text and native_table.source.locator["page"] == 1
    slide_page = next(
        part
        for part in result.segments
        if part.document == native_table.document and part.source.kind == "page"
    )
    assert "Figure 1." in slide_page.text
    outputs = [part for part in result.segments if part.source.kind == "notebook_output"]
    assert [part.source.source_id for part in outputs] == ["cell:1:output:0", "cell:1:output:2"]
    assert all(part.source.parent_id == "cell:1" for part in outputs)
    trace = result.trace()
    assert trace["unread_visual"][0]["source"]["source_id"] == "cell:1:output:1"
    assert trace["unread_visual"][0]["locator"] == {
        "cell_index": 1,
        "output_index": 1,
        "section_path": ["Baseline protocol"],
    }


def test_ocr_is_separate_from_native_and_visual_enhancement(library, tmp_path):
    source = tmp_path / "scan.pdf"
    Image.new("RGB", (210, 180), "white").save(source)
    result = _publish(library, source)
    key = SourceKey(document_id=result.document_id, version=1, source_id="page:1")
    window = read_window(
        library.store,
        _snapshot(library, result.document_id),
        context_tokens=4096,
        output_reserve=1000,
    )
    assert not window.segments and window.trace()["skipped"][0]["reason"] == "needs_ocr"
    pin = dict(
        course_ids=(COURSE,),
        content_sha256=result.content_sha256,
        publication_epoch=result.publication_epoch,
    )
    assert library.store.write_enhancement(key, text="visual keyword", **pin)
    assert library.store.write_enhancement(
        key, text="vernier scale 36 millimetres", origin="ocr", **pin
    )
    assert not library.store.write_enhancement(
        key, text="a changed derivative", origin="ocr", **pin
    )
    window = read_window(
        library.store,
        _snapshot(library, result.document_id),
        context_tokens=4096,
        output_reserve=1000,
    )
    (segment,) = window.segments
    assert segment.method == "ocr" and segment.text == "vernier scale 36 millimetres"
    assert not segment.source.text and window.trace()["visual_read_status"] == "not_attempted"
    hits = library.search(SearchRequest(query="vernier", course_id=COURSE))
    assert hits[0].origin == "ocr"
    assert (
        library.search(SearchRequest(query="visual", course_id=COURSE))[0].origin == "enhancement"
    )
    structure = StructureSearcher(StructureSnapshotStore(library.settings)).search(
        StructureSearchRequest(query="vernier", scope=StructureScope(course_ids=[COURSE])),
        allowed_course_ids=frozenset({COURSE}),
    )
    assert structure.candidates[0].reasons[0].code == "fts_ocr"
    assert not library.store.read_ocr_text(key, course_ids=("outside",))
    source.write_bytes(source.read_bytes() + b"\n% v2\n")
    _publish(library, source, result.document_id)
    assert library.store.read_ocr_text(key, course_ids=(COURSE,)) == segment.text
    assert not library.search(SearchRequest(query="vernier", course_id=COURSE))
    assert not library.store.write_enhancement(key, text=segment.text, origin="ocr", **pin)
    assert library.delete(result.document_id)
    assert not library.store.read_ocr_text(key, course_ids=(COURSE,))


def test_delayed_docling_write_is_discarded_after_release_changes(library, tmp_path, monkeypatch):
    path = tmp_path / "scan.pdf"
    Image.new("RGB", (210, 180), "white").save(path)
    result = _publish(library, path)
    committed = path.read_bytes()
    # Changing the selectable path alone must not change the archived bytes sent to OCR.
    path.write_bytes(committed + b"\n% replacement\n")

    def replace_while_running(_self, staged_path):
        assert staged_path.read_bytes() == committed
        assert _publish(library, path, result.document_id).version == 2
        item = SimpleNamespace(text="archived computed result", element_id="doc-slide-ocr-0001")
        return SimpleNamespace(slides=[SimpleNamespace(slide_no=1, elements=[item])])

    monkeypatch.setattr(PDFParser, "parse", replace_while_running)
    report = library.enhance_pdf_ocr(COURSE, result.document_id)
    assert report["pages"] == {"page:1": "stale_or_unavailable"}
    assert not library.search(SearchRequest(query="computed", course_id=COURSE))


def test_reading_window_matches_final_message_budget_across_escaping_and_boundaries(
    library, tmp_path
):
    from app.rag.workspace_generation import _build_messages

    cells = [
        {
            "id": f"c-{idx}",
            "cell_type": "markdown",
            "metadata": {},
            "source": (
                f"# 第 {idx} 节 \"引号\" 与 \\反斜线\n"
                f"公式：G(x) = \\sum_{{i=0}}^{{n-1}} \"w_i\" \\cdot E_i(x)\n"
                f"多行中文说明 A_{idx}\n多行中文说明 B_{idx}\n"
            ),
        }
        for idx in range(12)
    ]
    nb_path = tmp_path / "复杂_\"引号\"_\\反斜线_课件.ipynb"
    nb_path.write_text(
        json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": cells}),
        encoding="utf-8",
    )
    published = _publish(library, nb_path)
    snapshot = _snapshot(library, published.document_id)
    question = "请解释 \"G(x)\" 公式中 \\sum 的含义？\n并列出各节结论。"
    context = {
        "previous_question": "上一轮 \"问题\" \\路径\n第二行" * 15,
        "previous_answer": "上一轮 \"回答\" \\反斜线\n多行内容" * 30,
    }
    coverage_true = {"full_text_scope": True, "unread_visual_total": 0}
    coverage_false = {"full_text_scope": False, "unread_visual_total": 0}
    reserve_true = estimate_prompt_reserve(
        question, mode="ask", context=context, coverage=coverage_true
    )
    reserve_false = estimate_prompt_reserve(
        question, mode="ask", context=context, coverage=coverage_false
    )

    full_window = read_window(
        library.store,
        snapshot,
        context_tokens=50_000,
        output_reserve=OUTPUT_RESERVE,
        prompt_reserve=reserve_true,
    )
    assert full_window.full_text_scope and full_window.next_cursor is None
    assert len(full_window.segments) == 12

    def build_evidence_payload(segments):
        payload = []
        for idx, seg in enumerate(segments, 1):
            payload.append(
                {
                    "id": f"ev-{idx}",
                    "source": {
                        "document_id": seg.source.document_id,
                        "version": seg.source.version,
                        "source_id": seg.source.source_id,
                    },
                    "file_name": seg.document.file_name,
                    "locator": {
                        k: v for k, v in seg.source.locator.items() if k in _LOCATOR_FIELDS
                    },
                    "reads": [{"method": seg.method, "status": "succeeded", "text": seg.text}],
                }
            )
        return payload

    exact_messages = _build_messages(
        question,
        mode="ask",
        context=context,
        coverage=coverage_true,
        evidence=build_evidence_payload(full_window.segments),
    )
    exact_full_total = estimate_input_budget(exact_messages) + _CHAT_FRAME_RESERVE
    assert reserve_true + full_window.estimated_input == exact_full_total

    # Boundary: exact full-text budget and +1 byte both fit 100% without truncation.
    for limit in (exact_full_total, exact_full_total + 1):
        win = read_window(
            library.store,
            snapshot,
            context_tokens=limit + OUTPUT_RESERVE,
            output_reserve=OUTPUT_RESERVE,
            prompt_reserve=reserve_true,
        )
        assert win.full_text_scope and win.next_cursor is None
        assert reserve_true + win.estimated_input == exact_full_total <= limit

    # Boundary - 1 byte: must shrink the last segment or drop trailing source and stay <= limit.
    win_below = read_window(
        library.store,
        snapshot,
        context_tokens=(exact_full_total - 1) + OUTPUT_RESERVE,
        output_reserve=OUTPUT_RESERVE,
        prompt_reserve=reserve_false,
    )
    assert not win_below.full_text_scope and win_below.next_cursor is not None
    msgs_below = _build_messages(
        question,
        mode="ask",
        context=context,
        coverage=coverage_false,
        evidence=build_evidence_payload(win_below.segments),
    )
    actual_below = estimate_input_budget(msgs_below) + _CHAT_FRAME_RESERVE
    assert reserve_false + win_below.estimated_input == actual_below <= exact_full_total - 1
    assert estimate_embedded_json_budget({"a": '1"2\\3\n'}) > 0
    assert estimate_embedded_text_budget('中文"\\n') > 0
