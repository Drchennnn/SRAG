"""G24 tests for natural cross-version questions and four easily confused intents."""

from __future__ import annotations

import base64
from pathlib import Path

import nbformat
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_workspace import _BASE, _COURSE, _HOST, _ask, _create, _focus, _node, _upload

from app.api.workspace import get_workspace_service, router
from app.config import Settings
from app.rag.workspace import make_workspace
from app.rag.workspace_versions import is_version_comparison

_PNG_1X1_A = base64.b64encode(
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
    b"\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
).decode("ascii")
_PNG_1X1_B = base64.b64encode(
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x80\x00\x00"
    b"\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
).decode("ascii")


def _write_notebook(
    path: Path,
    result: str,
    *,
    cell_id: str = "kv-cache-bench",
    with_images: bool = False,
    extra_cell: tuple[str, str] | None = None,
) -> None:
    outputs = [nbformat.v4.new_output("stream", name="stdout", text=result)]
    if with_images:
        outputs.extend(
            [
                nbformat.v4.new_output("display_data", data={"image/png": _PNG_1X1_A}),
                nbformat.v4.new_output("display_data", data={"image/png": _PNG_1X1_B}),
            ]
        )
    cells = [
        nbformat.v4.new_code_cell(
            "run_bench()", id=cell_id, execution_count=1, outputs=outputs
        )
    ]
    if extra_cell is not None:
        cells.append(
            nbformat.v4.new_code_cell(
                "run_extra()",
                id=extra_cell[0],
                execution_count=2,
                outputs=[nbformat.v4.new_output("stream", name="stdout", text=extra_cell[1])],
            )
        )
    nbformat.write(nbformat.v4.new_notebook(cells=cells), path)


def _sides(message: dict) -> dict[int, tuple[str, dict]]:
    return {
        item["revision"]["source"]["version"]: (evidence_id, item)
        for evidence_id, item in message["sources"].items()
    }


@pytest.fixture
def g24_system(tmp_path):
    settings = Settings(
        sqlite_path=tmp_path / "g24.sqlite3",
        data_dir=tmp_path / "data",
        course_root=tmp_path / "originals",
        workspace_course_ids=f"{_COURSE},other",
        workspace_max_upload_bytes=512 * 1024,
        source_vision_enabled=False,
        dashscope_api_key="",
        _env_file=None,
    )
    service = make_workspace(settings)
    app = FastAPI()
    app.include_router(router, prefix=_BASE)
    app.dependency_overrides[get_workspace_service] = lambda: service
    with TestClient(app, base_url=_HOST) as client:
        yield settings, service, app, client


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # 1. 比较当前版和上一版（无需同时写“新版”和“旧版”）
        (
            "跟上一版相比，第 1 个 Cell 保存输出里的平均路由分发延迟、"
            "容量因子和评估词元数有什么变化？",
            True,
        ),
        ("跟上一版相比有什么变化？", True),
        ("和上一版比呢？", True),
        ("跟之前那个版本比起来，第 1 个 Cell 保存输出里的命中率怎么变了？", True),
        ("与上一个版本相比，输出延迟变了多少？", True),
        ("较之前的版本有哪些改动？", True),
        ("比上一版快了吗？", True),
        ("和 v1 相比，第 1 个 Cell 的保存输出有什么不同？", True),
        ("前后两版对比一下有什么变化？", True),
        ("新旧版本有什么区别？", True),
        ("跟上一版相比，哪些参数没有变化？", True),
        ("What changed compared to the previous version?", True),
        ("How does this differ from the prior edition?", True),
        ("Compare with the old version.", True),
        # 2. 只回看旧版
        ("旧版第 1 个 Cell 保存输出里的延迟是多少？", False),
        ("上一版中评估词元数是多少？", False),
        ("上一版呢？", False),
        ("回看旧版那张图", False),
        ("刚才旧版展示的那张图横轴是什么？", False),
        ("查看先前版本的第 1 个 Cell 输出", False),
        ("What was the value in the old version?", False),
        ("Show me the output from the previous version.", False),
        ("What was in the old version of compare-results.ipynb?", False),
        # 3. 比较当前版里的两张图（以及单一旧版内两张图）
        ("比较当前版里的两张图", False),
        ("对比这一版第 1 张图和第 2 张图的横轴单位", False),
        ("比较最新版第 5 页的左图和右图", False),
        ("Compare the two figures in the current version", False),
        ("比较 v2 里的两张图", False),
        ("比较上一版里的两张图", False),
        ("对比旧版中图 1 和图 2 的区别", False),
        ("Compare the two figures in the old version", False),
        # 4. 提到了“旧版”但没有要求比较
        ("之前旧版已经看过，现在只看当前第 1 个 Cell 的保存输出", False),
        ("不用和旧版对比，只看当前版的第 1 个 Cell 输出", False),
        ("不要比较上一版，告诉我现在第 1 个 Cell 的延迟", False),
        ("忽略旧版差异，直接说明当前版结果", False),
        ("我们已经从旧版升级到最新版了，请总结当前第 1 个 Cell 的结果", False),
        ("旧版里比较重要的结论是什么？", False),
        ("我们已经换掉旧版了，说说当前版第 1 个 Cell 里延迟的变化趋势", False),
        ("我已经看过旧版，这次只问当前版里的不同配置有哪些", False),
        ("旧版中参数不同时模型怎么处理？", False),
        ("Don't compare with the old version; what is the current latency?", False),
        ("Without comparing to the previous version, summarize the current cell.", False),
    ],
)
def test_natural_version_comparison_intent_classification(question: str, expected: bool) -> None:
    assert is_version_comparison(question) is expected


def test_natural_comparison_reads_both_versions_and_survives_stale_focus(g24_system, tmp_path):
    _, _, _, client = g24_system
    session = _create(client)
    nb_path = tmp_path / "Zephyr-Cache-Benchmark.ipynb"
    _write_notebook(
        nb_path,
        "hit_rate=78.4% p95_latency=18.6 ms batch_tokens=16384",
        with_images=True,
    )
    first = _upload(client, session, nb_path)
    doc_id = first["document_id"]
    cell_node = _node(client, session, doc_id, "cell", cell_index=0)
    rev = _focus(client, session, cell_node)

    turn1 = _ask(
        client, session, "请展示第 1 个 Cell 的保存输出并说明命中率和 P95 延迟", rev
    )
    assert turn1["turn"]["status"] == "completed"
    assert len(turn1["message"]["visible_display_ids"]) == 2

    # Update to v2 WITHOUT re-focusing (focus_id remains stale from v1!) and scope to single doc
    scope_res = client.patch(
        f"{session}/scope",
        json={"course_ids": [_COURSE], "document_ids": [doc_id], "scope_revision": rev},
    )
    assert scope_res.status_code == 200
    rev = scope_res.json()["scope_revision"]
    # Set v1 cell focus again then replace with v2 to simulate stale focus_id
    rev = _focus(client, session, cell_node, rev)
    _write_notebook(
        nb_path,
        "hit_rate=91.2% p95_latency=11.4 ms batch_tokens=32768",
        with_images=True,
    )
    second = _upload(client, session, nb_path, document_id=doc_id)
    assert second["version"] == 2

    # Natural comparison WITHOUT naming "新版" and WITHOUT explicit cell location
    compared = _ask(
        client,
        session,
        "跟上一版相比有什么变化？",
        rev,
        client_id="g24-natural-implicit",
    )
    assert compared["turn"]["trace"]["route"] == "version_comparison"
    msg = compared["message"]
    versions = {item["revision"]["source"]["version"] for item in msg["sources"].values()}
    assert versions == {1, 2}
    # Both v1 and v2 citations must open cleanly with accurate is_current flags
    for eid, ev in msg["sources"].items():
        resp = client.get(f"{session}/messages/{msg['message_id']}/evidence/{eid}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        expected_ver = ev["revision"]["source"]["version"]
        assert body["source"]["version"] == expected_ver
        assert body["is_current"] is (expected_ver == 2)
        if expected_ver == 1:
            if body["source"]["source_id"] == "cell:0:output:0":
                assert "78.4%" in body["text"]
            else:
                assert "run_bench()" in body["text"]
        else:
            if body["source"]["source_id"] == "cell:0:output:0":
                assert "91.2%" in body["text"]
            else:
                assert "run_bench()" in body["text"]


def test_four_confused_meanings_end_to_end_routes(g24_system, tmp_path):
    _, _, _, client = g24_system
    session = _create(client)
    nb_path = tmp_path / "Zephyr-Cache-Benchmark.ipynb"
    _write_notebook(
        nb_path,
        "hit_rate=78.4% p95_latency=18.6 ms batch_tokens=16384",
        with_images=True,
    )
    first = _upload(client, session, nb_path)
    doc_id = first["document_id"]
    cell_node = _node(client, session, doc_id, "cell", cell_index=0)
    rev = _focus(client, session, cell_node)

    # Turn 1 (v1): displays 2 images and cites cell:0 & cell:0:output:0
    t1 = _ask(client, session, "请展示第 1 个 Cell 的保存输出", rev, client_id="t1-v1")
    assert len(t1["message"]["visible_display_ids"]) == 2

    # Update to v2 and focus v2 document root
    _write_notebook(
        nb_path,
        "hit_rate=91.2% p95_latency=11.4 ms batch_tokens=32768",
        with_images=True,
    )
    second = _upload(client, session, nb_path, document_id=doc_id)
    assert second["version"] == 2
    v2_doc = _node(client, session, doc_id, "document")
    rev = _focus(client, session, v2_doc, rev)

    # 1) Compare current vs previous (natural phrasing, no "新版")
    m1 = _ask(
        client,
        session,
        "跟之前那个版本比起来，第 1 个 Cell 保存输出里的命中率和 P95 延迟怎么变了？",
        rev,
        client_id="m1-compare",
    )
    assert m1["turn"]["trace"]["route"] == "version_comparison"
    sides = _sides(m1["message"])
    assert set(sides) == {1, 2}

    # 2) Only look back at old version (historical_only for image; non-compare for text)
    m2_img = _ask(
        client,
        session,
        "刚才旧版展示的第 2 张图横轴是什么？",
        rev,
        client_id="m2-old-img",
    )
    assert (m2_img["turn"]["trace"] or {}).get("route") != "version_comparison"
    assert m2_img["message"]["blocks"][0]["code"] == "historical_only"

    m2_txt = _ask(
        client,
        session,
        "之前那版第 1 个 Cell 保存输出里的 P95 延迟是多少？",
        rev,
        client_id="m2-old-txt",
    )
    assert (m2_txt["turn"]["trace"] or {}).get("route") != "version_comparison"

    # 3) Compare two figures in current version -> stays in current v2, never reads v1
    m3 = _ask(
        client,
        session,
        "比较当前版里的两张图片有什么区别",
        rev,
        client_id="m3-curr-two-imgs",
    )
    assert (m3["turn"]["trace"] or {}).get("route") != "version_comparison"
    assert all(
        ev["revision"]["source"]["version"] == 2
        for ev in m3["message"]["sources"].values()
    )
    assert any(b.get("code") == "object_ambiguous" for b in m3["message"]["blocks"])

    m3_short = _ask(
        client,
        session,
        "比较当前版里的两张图",
        rev,
        client_id="m3-curr-two-imgs-short",
    )
    assert (m3_short["turn"]["trace"] or {}).get("route") != "version_comparison"
    assert all(
        ev["revision"]["source"]["version"] == 2
        for ev in m3_short["message"]["sources"].values()
    )

    # 4) Mentions "旧版" without requesting comparison -> stays on current v2
    m4_neg = _ask(
        client,
        session,
        "不用和旧版对比，直接告诉我现在第 1 个 Cell 保存输出里的命中率是多少？",
        rev,
        client_id="m4-neg",
    )
    assert (m4_neg["turn"]["trace"] or {}).get("route") != "version_comparison"
    assert all(
        ev["revision"]["source"]["version"] == 2
        for ev in m4_neg["message"]["sources"].values()
    )
    assert any("91.2%" in b.get("text", "") for b in m4_neg["message"]["blocks"])

    m4_bg = _ask(
        client,
        session,
        "我们已经换掉旧版了，说说当前版第 1 个 Cell 里延迟的变化趋势",
        rev,
        client_id="m4-bg",
    )
    assert (m4_bg["turn"]["trace"] or {}).get("route") != "version_comparison"


def test_missing_history_scope_switch_unclear_mapping_and_isolation(g24_system, tmp_path):
    _, service, _, client = g24_system
    sess_a = _create(client)
    sess_b = _create(client)

    nb_a = tmp_path / "Zephyr-Cache-Benchmark.ipynb"
    nb_b = tmp_path / "Cobalt-Quant-Audit.ipynb"
    _write_notebook(
        nb_a,
        "hit_rate=78.4%",
        extra_cell=("eviction-policy", "eviction_v1=lru"),
    )
    _write_notebook(nb_b, "quant_error=0.012")
    doc_a = _upload(client, sess_a, nb_a)
    doc_b = _upload(client, sess_a, nb_b)

    # Before updating to v2, only v1 exists -> missing history
    cell0_a = _node(client, sess_a, doc_a["document_id"], "cell", cell_index=0)
    rev_a = _focus(client, sess_a, cell0_a)
    _ask(client, sess_a, "hit_rate", rev_a, client_id="cite-a-cell0")
    cell1_a = _node(client, sess_a, doc_a["document_id"], "cell", cell_index=1)
    rev_a = _focus(client, sess_a, cell1_a, rev_a)
    _ask(client, sess_a, "eviction_v1", rev_a, client_id="cite-a-cell1")

    only_v1 = _ask(client, sess_a, "跟上一版相比有什么变化？", rev_a, client_id="only-v1")
    assert only_v1["message"]["blocks"][0]["code"] == "comparison_missing_history"

    # Update nb_a to v2, changing cell_1's cell_id so cell 2 correspondence becomes unclear
    _write_notebook(
        nb_a,
        "hit_rate=91.2%",
        extra_cell=("eviction-policy-v2", "eviction_v2=lfu"),
    )
    v2 = _upload(client, sess_a, nb_a, document_id=doc_a["document_id"])
    assert v2["version"] == 2
    v2_doc_node = _node(client, sess_a, doc_a["document_id"], "document")
    rev_a = _focus(client, sess_a, v2_doc_node, rev_a)

    # Unclear object correspondence (cell_id changed in Cell 2) -> comparison_ambiguous
    unclear = _ask(
        client,
        sess_a,
        "跟上一版相比，第 2 个 Cell 的保存输出有什么变化？",
        rev_a,
        client_id="unclear-cell2",
    )
    assert unclear["message"]["blocks"][0]["code"] == "comparison_ambiguous"
    assert not unclear["message"]["sources"]

    # Single-version conflict ("和 v2 相比" when current IS v2) -> comparison_version_mismatch
    mismatch = _ask(
        client,
        sess_a,
        "和 v2 相比，第 1 个 Cell 的保存输出有什么变化？",
        rev_a,
        client_id="mismatch-v2",
    )
    assert mismatch["message"]["blocks"][0]["code"] == "comparison_version_mismatch"

    # Explicit v1 comparison ("和 v1 相比") still works
    explicit_v1 = _ask(
        client,
        sess_a,
        "和 v1 相比，第 1 个 Cell 的保存输出有什么变化？",
        rev_a,
        client_id="explicit-v1",
    )
    assert set(_sides(explicit_v1["message"])) == {1, 2}

    # Session isolation: sess_b cannot use sess_a's v1 citations
    rev_b = _focus(client, sess_b, _node(client, sess_b, doc_a["document_id"], "document"))
    isolated = _ask(
        client,
        sess_b,
        "跟上一版相比，第 1 个 Cell 的保存输出有什么变化？",
        rev_b,
        client_id="sess-b-isolated",
    )
    assert isolated["message"]["blocks"][0]["code"] == "comparison_missing_history"
    assert not isolated["message"]["sources"]

    # Scope switch to Cobalt-Quant-Audit.ipynb -> out-of-scope named file vs missing history
    doc_b_node = _node(client, sess_a, doc_b["document_id"], "document")
    rev_a = _focus(client, sess_a, doc_b_node, rev_a)
    scope_res = client.patch(
        f"{sess_a}/scope",
        json={
            "course_ids": [_COURSE],
            "document_ids": [doc_b["document_id"]],
            "scope_revision": rev_a,
        },
    )
    assert scope_res.status_code == 200
    rev_a = scope_res.json()["scope_revision"]

    switched_named = _ask(
        client,
        sess_a,
        "跟上一版相比，Zephyr-Cache-Benchmark.ipynb 有什么变化？",
        rev_a,
        client_id="switched-named",
    )
    assert switched_named["message"]["blocks"][0]["code"] == "comparison_scope"

    switched_anon = _ask(
        client,
        sess_a,
        "跟上一版相比有什么变化？",
        rev_a,
        client_id="switched-anon",
    )
    assert switched_anon["message"]["blocks"][0]["code"] == "comparison_missing_history"

    # Course-wide scope with 2 files and no focus -> comparison_scope clarification
    course_res = client.patch(
        f"{sess_a}/scope",
        json={"course_ids": [_COURSE], "document_ids": [], "scope_revision": rev_a},
    )
    assert course_res.status_code == 200
    rev_a = course_res.json()["scope_revision"]
    course_wide = _ask(
        client,
        sess_a,
        "跟上一版相比有什么变化？",
        rev_a,
        client_id="course-wide-ambig",
    )
    assert course_wide["message"]["blocks"][0]["code"] == "comparison_scope"

    # Restore focus to Zephyr and test v1 historical original unavailability -> 409/failed
    # Wait: first verify normal comparison still works in course-wide scope when file is named!
    course_named = _ask(
        client,
        sess_a,
        "跟上一版相比，Zephyr-Cache-Benchmark.ipynb 第 1 个 Cell 保存输出有什么变化？",
        rev_a,
        client_id="course-wide-named",
    )
    assert set(_sides(course_named["message"])) == {1, 2}

    # Verify deleted document blocks comparison (historical authorization preserved)
    assert service.library.store.delete_local(doc_a["document_id"])
    deleted = _ask(
        client,
        sess_a,
        "跟上一版相比，Zephyr-Cache-Benchmark.ipynb 第 1 个 Cell 保存输出有什么变化？",
        rev_a,
        client_id="deleted-doc",
    )
    assert deleted["message"]["blocks"][0]["code"] == "comparison_scope"
    assert not deleted["message"]["sources"]


def test_corrupted_historical_original_preserves_409_authorization(g24_system, tmp_path):
    _, service, _, client = g24_system
    session = _create(client)
    nb_path = tmp_path / "Zephyr-Cache-Benchmark.ipynb"
    _write_notebook(nb_path, "hit_rate=78.4%")
    first = _upload(client, session, nb_path)
    doc_id = first["document_id"]
    rev = _focus(client, session, _node(client, session, doc_id, "cell", cell_index=0))
    _ask(client, session, "hit_rate", rev, client_id="v1-read")

    _write_notebook(nb_path, "hit_rate=91.2%")
    _upload(client, session, nb_path, document_id=doc_id)
    rev = _focus(client, session, _node(client, session, doc_id, "document"), rev)

    # Clear v1 archived original blob in SQLite to verify historical sha/original integrity check
    with service.library.store._connect() as connection:
        updated = connection.execute(
            "UPDATE local_versions SET original = NULL WHERE document_id = ? AND version = 1",
            (doc_id,),
        )
        assert updated.rowcount == 1

    failed = _ask(
        client,
        session,
        "跟上一版相比，第 1 个 Cell 保存输出有什么变化？",
        rev,
        client_id="missing-v1-blob",
    )
    assert failed["turn"]["status"] == "failed"
    assert failed["turn"]["error"] == "original_unavailable"
