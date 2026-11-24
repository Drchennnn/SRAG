"""Offline regression tests for cross-version recovery, b2 resume, and multi-output cells."""

from __future__ import annotations

import base64
import hashlib
import json
from io import BytesIO
from pathlib import Path

import nbformat
from PIL import Image, ImageDraw
from test_workspace import _ask, _create, _focus, _node, _upload
from test_workspace import system as system  # noqa: F401

from app.rag.source_evidence import EvidenceResult, ReadRecord
from app.schemas.local import SourceKey


def _png_b64(text: str, fill: tuple[int, int, int] = (40, 110, 180)) -> str:
    img = Image.new("RGB", (320, 160), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(10, 10), (310, 150)], fill=fill, outline=(10, 10, 10))
    draw.text((24, 64), text, fill=(255, 255, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _write_nb(
    path: Path, version_tag: str, *, reordered: bool = False, deleted: bool = False
) -> None:
    img_a = _png_b64(f"Plot A {version_tag}: 120 ops/s", (30, 90, 180))
    img_b = _png_b64(f"Plot B {version_tag}: 18 ms", (160, 70, 50))
    if deleted:
        outputs = [
            nbformat.v4.new_output(
                "stream", name="stdout", text=f"only text remains {version_tag}\n"
            )
        ]
    elif reordered:
        img_a_v1 = _png_b64("Plot A v1: 120 ops/s", (30, 90, 180))
        img_b_v1 = _png_b64("Plot B v1: 18 ms", (160, 70, 50))
        outputs = [
            nbformat.v4.new_output("display_data", data={"image/png": img_b_v1}),
            nbformat.v4.new_output("display_data", data={"image/png": img_a_v1}),
            nbformat.v4.new_output(
                "stream", name="stdout", text=f"summary {version_tag}: reordered\n"
            ),
        ]
    else:
        outputs = [
            nbformat.v4.new_output("display_data", data={"image/png": img_a}),
            nbformat.v4.new_output("display_data", data={"image/png": img_b}),
            nbformat.v4.new_output("stream", name="stdout", text=f"summary {version_tag}: ok\n"),
        ]
    cells = [
        nbformat.v4.new_markdown_cell(f"# Drift {version_tag}\nGate rule applies.", id="md-0"),
        nbformat.v4.new_code_cell("render()", id="code-1", execution_count=1, outputs=outputs),
    ]
    nbformat.write(nbformat.v4.new_notebook(cells=cells), path)


def _patch_fake_vision(service, monkeypatch):
    def fake_process(request, *, expected_input_sha256=None):
        disp = service.evidence.render_current(request)
        actual = disp.display.actual_source if disp.display else request.source
        label = f"[v{actual.version} {actual.source_id}] read ok for: {request.question[:40]}"
        return EvidenceResult(
            status="available",
            requested_source=request.source,
            content_sha256=disp.content_sha256,
            display=disp.display,
            read_status="succeeded",
            capability_reason=None,
            reads=[
                ReadRecord(
                    requested_source=request.source,
                    actual_source=SourceKey.model_validate(actual.model_dump()),
                    actual_input_sha256=(
                        disp.display.byte_sha256
                        if disp.display
                        else request.expected_content_sha256
                    ),
                    actual_input_media_type="image/png",
                    purpose="directed_question",
                    execution="real_provider",
                    origin_execution="real_provider",
                    provider="fake",
                    model="fake-vl",
                    prompt_version="test-v1",
                    question_sha256=(
                        hashlib.sha256(request.question.strip().encode()).hexdigest()
                        if request.question
                        else None
                    ),
                    status="succeeded",
                    text=label,
                )
            ],
            external_call_count=1,
        )

    monkeypatch.setattr(service.vision, "process_current", fake_process)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_problem1_three_intents_and_controls_without_put_focus(system, tmp_path, monkeypatch):
    settings, service, _, client = system
    settings.source_vision_enabled = True
    _patch_fake_vision(service, monkeypatch)

    v1_path = tmp_path / "Bench-v1.ipynb"
    v2_path = tmp_path / "Bench-v2.ipynb"
    _write_nb(v1_path, "v1")
    _write_nb(v2_path, "v2")

    for intent, query, check in (
        ("new_counterpart", "最新版里刚才读过的那张图现在的峰值是多少？", "counterpart"),
        ("new_counterpart_bare", "最新版里刚才读过的图是什么内容？", "counterpart"),
        ("recall_old", "回看刚才读过的旧版那张图的内容。", "recall"),
        ("recall_old_bare", "回看刚才读过的旧版图的内容。", "recall"),
        ("compare_both", "和上一版相比，刚才读过的那张图有什么变化？", "compare"),
        ("only_old", "只看旧版第一张图，刚才展示的第 1 张图是什么？", "historical_notice"),
        ("stale_ctrl", "看当前 Cell", "stale"),
    ):
        sess = _create(client)
        fname = f"Bench-{intent}.ipynb"
        doc1 = _upload(client, sess, v1_path, filename=fname)
        cell_node = _node(client, sess, doc1["document_id"], "cell", cell_index=1)
        rev = _focus(client, sess, cell_node)
        show = _ask(client, sess, "展示这个 Cell 的两张图", rev, client_id=f"{intent}-show")
        assert len(show["message"]["visible_display_ids"]) == 2
        read1 = _ask(client, sess, "刚才第一张图是什么？", rev, client_id=f"{intent}-read")
        assert read1["turn"]["status"] == "completed"
        assert read1["turn"]["resolution"]["actual_source"]["source_id"] == "cell:1:output:0"
        assert "[v1 cell:1:output:0]" in json.dumps(read1, ensure_ascii=False)

        doc2 = _upload(client, sess, v2_path, filename=fname, document_id=doc1["document_id"])
        assert doc2["version"] == 2
        res = _ask(client, sess, query, rev, client_id=f"{intent}-after")

        if check == "counterpart":
            assert res["turn"]["status"] == "completed", res
            assert res["turn"]["resolution"]["actual_source"]["version"] == 2
            assert res["turn"]["resolution"]["actual_source"]["source_id"] == "cell:1:output:0"
            msg = res["message"]
            img_blk = next(b for b in msg["blocks"] if b["type"] == "source_image")
            ev_id = img_blk["evidence_id"]
            media_resp = client.get(img_blk["media_url"])
            prev_resp = client.get(f"{sess}/messages/{msg['message_id']}/evidence/{ev_id}/preview")
            assert media_resp.status_code == 200
            assert prev_resp.status_code == 200
            assert _sha(media_resp.content) == _sha(prev_resp.content)
        elif check == "recall":
            assert res["turn"]["status"] == "completed", res
            assert any(b.get("code") == "historical_only" for b in res["message"]["blocks"])
            dumped = json.dumps(res["message"]["blocks"], ensure_ascii=False)
            assert "[v1 cell:1:output:0]" in dumped
        elif check == "compare":
            assert res["turn"]["status"] == "completed", res
            assert res["turn"]["trace"]["route"] == "version_comparison"
            versions = {
                ev["revision"]["source"]["version"]
                for ev in res["message"]["sources"].values()
            }
            assert versions == {1, 2}
            msg = res["message"]
            for blk in msg["blocks"]:
                if blk["type"] == "source_image":
                    m_resp = client.get(blk["media_url"])
                    p_resp = client.get(
                        f"{sess}/messages/{msg['message_id']}/evidence/{blk['evidence_id']}/preview"
                    )
                    assert m_resp.status_code == 200
                    assert p_resp.status_code == 200
                    assert _sha(m_resp.content) == _sha(p_resp.content)
        elif check == "historical_notice":
            assert res["turn"]["status"] == "completed", res
            assert any(b.get("code") == "historical_only" for b in res["message"]["blocks"])
        elif check == "stale":
            assert res["turn"]["status"] == "failed", res
            assert res["turn"]["error"] == "stale_reference"


def test_problem1_reordered_b2_bilateral_resume_deleted_and_cross_session_isolation(
    system, tmp_path, monkeypatch
):
    settings, service, _, client = system
    settings.source_vision_enabled = True
    _patch_fake_vision(service, monkeypatch)

    v1_path = tmp_path / "Reord-v1.ipynb"
    v2_reord = tmp_path / "Reord-v2.ipynb"
    v2_del = tmp_path / "Del-v2.ipynb"
    _write_nb(v1_path, "v1")
    _write_nb(v2_reord, "v2r", reordered=True)
    _write_nb(v2_del, "v2d", deleted=True)

    # 1. Reordered v2 -> clarify with full 2 options; selecting resumes bilateral v1+v2 compare!
    sess = _create(client)
    doc1 = _upload(client, sess, v1_path, filename="Reord.ipynb")
    cell_node = _node(client, sess, doc1["document_id"], "cell", cell_index=1)
    rev = _focus(client, sess, cell_node)
    _ask(client, sess, "展示这个 Cell 的两张图", rev, client_id="reord-show")
    _ask(client, sess, "刚才第一张图是什么？", rev, client_id="reord-read")
    _upload(client, sess, v2_reord, filename="Reord.ipynb", document_id=doc1["document_id"])

    clarify_new = _ask(
        client, sess, "最新版里刚才读过的那张图现在的峰值是多少？", rev, client_id="reord-new"
    )
    assert clarify_new["turn"]["status"] == "clarification", clarify_new
    assert len(clarify_new["options"]) == 2, clarify_new

    clarify_cmp = _ask(
        client, sess, "对比最新版和上一版刚才读过的图的纵轴单位和数值", rev, client_id="reord-cmp"
    )
    assert clarify_cmp["turn"]["status"] == "clarification", clarify_cmp
    assert len(clarify_cmp["options"]) == 2, clarify_cmp

    # Select option 2 on clarify_cmp (b2 bilateral resume -> BOTH v1 & v2!)
    chosen_cmp = _ask(
        client,
        sess,
        "选择第2项",
        rev,
        client_id="reord-choose-cmp",
        anchor={
            "type": "clarification",
            "turn_id": clarify_cmp["turn"]["turn_id"],
            "option_id": clarify_cmp["options"][1]["option_id"],
        },
    )
    assert chosen_cmp["turn"]["status"] == "completed", chosen_cmp
    assert chosen_cmp["turn"]["trace"]["route"] == "version_comparison"
    resumed_id = chosen_cmp["turn"]["trace"]["resumed_from_clarification"]
    assert resumed_id == clarify_cmp["turn"]["turn_id"]
    msg = chosen_cmp["message"]
    sides = {
        ev["revision"]["source"]["version"]: (eid, ev)
        for eid, ev in msg["sources"].items()
    }
    assert set(sides) == {1, 2}
    assert sides[1][1]["read_status"] == "succeeded"
    assert sides[2][1]["read_status"] == "succeeded"
    assert sides[2][1]["revision"]["source"]["source_id"] == "cell:1:output:1"
    for blk in msg["blocks"]:
        if blk["type"] == "source_image":
            m_resp = client.get(blk["media_url"])
            p_resp = client.get(
                f"{sess}/messages/{msg['message_id']}/evidence/{blk['evidence_id']}/preview"
            )
            assert m_resp.status_code == 200
            assert p_resp.status_code == 200
            assert _sha(m_resp.content) == _sha(p_resp.content)

    # Select option 2 on clarify_new (read_object -> v2 single read)
    chosen_new = _ask(
        client,
        sess,
        "选择第2项",
        rev,
        client_id="reord-choose-new",
        anchor={
            "type": "clarification",
            "turn_id": clarify_new["turn"]["turn_id"],
            "option_id": clarify_new["options"][1]["option_id"],
        },
    )
    assert chosen_new["turn"]["status"] == "completed", chosen_new
    assert "[v2 cell:1:output:1]" in str(chosen_new["message"]["blocks"])

    # 2. Deleted output in v2 -> clear clarification notice, no silent wrong-object read
    sess_del = _create(client)
    d1 = _upload(client, sess_del, v1_path, filename="Del.ipynb")
    cnode = _node(client, sess_del, d1["document_id"], "cell", cell_index=1)
    rev_d = _focus(client, sess_del, cnode)
    _ask(client, sess_del, "展示这个 Cell 的两张图", rev_d, client_id="del-show")
    _ask(client, sess_del, "刚才第一张图是什么？", rev_d, client_id="del-read")
    _upload(client, sess_del, v2_del, filename="Del.ipynb", document_id=d1["document_id"])
    after_del = _ask(
        client, sess_del, "最新版里刚才读过的那张图现在的峰值是多少？", rev_d, client_id="del-after"
    )
    assert after_del["turn"]["status"] == "clarification", after_del
    assert "删除" in str(after_del["message"]["blocks"])

    # 3. Cross-session isolation: fresh session with same course has zero history of prior read
    sess_other = _create(client)
    iso = _ask(
        client, sess_other, "最新版里刚才读过的那张图现在的峰值是多少？", 1, client_id="iso-1"
    )
    assert iso["turn"]["status"] != "failed"
    assert "[v1" not in str(iso["message"])
    assert "[v2" not in str(iso["message"])


def test_problem2_focused_multi_output_cell_clarifies_and_reads_selected_output(
    system, tmp_path, monkeypatch
):
    settings, service, _, client = system
    settings.source_vision_enabled = True
    _patch_fake_vision(service, monkeypatch)

    nb_path = tmp_path / "MultiOut.ipynb"
    _write_nb(nb_path, "v1")
    sess = _create(client)
    doc = _upload(client, sess, nb_path, filename="MultiOut.ipynb")
    cell_node = _node(client, sess, doc["document_id"], "cell", cell_index=1)
    rev = _focus(client, sess, cell_node)

    preview = _ask(client, sess, "显示本 Cell 的两张原件图片", rev, client_id="p2-preview")
    assert preview["turn"]["status"] == "completed"
    assert len(preview["message"]["visible_display_ids"]) == 2

    img_q = _ask(client, sess, "当前这个 Cell 的输出图像横轴是什么？", rev, client_id="p2-img-q")
    assert img_q["turn"]["status"] == "clarification", img_q
    assert len(img_q["options"]) == 2

    picked_img = _ask(
        client,
        sess,
        img_q["options"][1]["label"],
        rev,
        client_id="p2-pick-img",
        anchor={
            "type": "clarification",
            "turn_id": img_q["turn"]["turn_id"],
            "option_id": img_q["options"][1]["option_id"],
        },
    )
    assert picked_img["turn"]["status"] == "completed", picked_img
    evs = list(picked_img["message"]["sources"].values())
    assert len(evs) == 1
    assert evs[0]["revision"]["source"]["source_id"] == "cell:1:output:1"
    assert evs[0]["reads"][0]["method"] == "vision"
    assert evs[0]["reads"][0]["status"] == "succeeded"

    out_q = _ask(client, sess, "当前这个 Cell 的输出结果显示什么？", rev, client_id="p2-out-q")
    assert out_q["turn"]["status"] == "clarification", out_q
    assert len(out_q["options"]) == 3

    picked_txt = _ask(
        client,
        sess,
        out_q["options"][2]["label"],
        rev,
        client_id="p2-pick-txt",
        anchor={
            "type": "clarification",
            "turn_id": out_q["turn"]["turn_id"],
            "option_id": out_q["options"][2]["option_id"],
        },
    )
    assert picked_txt["turn"]["status"] == "completed", picked_txt
    evs_txt = list(picked_txt["message"]["sources"].values())
    assert len(evs_txt) == 1
    assert evs_txt[0]["revision"]["source"]["source_id"] == "cell:1:output:2"
    assert evs_txt[0]["reads"][0]["method"] == "native_text"
    assert "summary v1: ok" in str(picked_txt["message"]["blocks"])
