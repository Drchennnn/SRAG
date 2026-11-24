"""K02-R2 regressions: routing precision, version auto-resolution, S0 vision & search hygiene."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import nbformat
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from pypdfium2 import PdfDocument

from app.api.workspace import get_workspace_service, router
from app.clients.deepseek import DeepSeekClient
from app.config import Settings
from app.rag.local_library import LocalLibrary
from app.rag.source_evidence import DisplayRecord, EvidenceResult
from app.rag.source_vision import VisionReply
from app.rag.structure_search import StructureSearcher, StructureSearchRequest
from app.rag.workspace import (
    WorkspaceService,
    _has_latin_corpus,
    _location,
    _parse_location_num,
    make_workspace,
)
from app.rag.workspace_objects import ObjectIntent, has_visual_question, object_intent
from app.rag.workspace_retrieval import _prioritize_continuations
from app.schemas.local import ParsedSource, SourceDraft
from app.stores.structure_snapshot import (
    SnapshotRequest,
    SnapshotSource,
    StructureSnapshotStore,
)

_BASE = "/api/workspace"
_HOST = "http://127.0.0.1:8001"
_COURSE = "k02-r2-course"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _png_bytes(color: tuple[int, int, int] = (20, 90, 180)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (320, 200), color).save(buf, format="PNG")
    return buf.getvalue()


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    params: dict[str, Any] = {
        "_env_file": None,
        "sqlite_path": tmp_path / "k02_r2.sqlite3",
        "data_dir": tmp_path / "data",
        "course_root": tmp_path / "courses",
        "workspace_course_ids": _COURSE,
        "workspace_answer_enabled": True,
        "deepseek_api_key": "r2-mock-deepseek",
        "dashscope_api_key": "",
        "source_vision_enabled": False,
    }
    params.update(overrides)
    return Settings(**params)


def _client(service: WorkspaceService) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix=_BASE)
    app.dependency_overrides[get_workspace_service] = lambda: service
    return TestClient(app, base_url=_HOST)


def _publish_doc(
    library: LocalLibrary,
    file_name: str,
    fmt: str,
    sources: list[SourceDraft],
    *,
    document_id: str | None = None,
    raw: bytes | None = None,
):
    path = library.course_root / _COURSE / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is None:
        if fmt == "pdf":
            page_count = max(
                1,
                *(
                    int(s.locator.get("page", 1))
                    for s in sources
                    if isinstance(s.locator.get("page"), int)
                ),
            )
            with PdfDocument.new() as doc:
                for _ in range(page_count):
                    doc.new_page(612, 792)
                doc.save(path)
            raw = path.read_bytes()
        else:
            if fmt == "pptx":
                from pptx import Presentation

                prs = Presentation()
                page_count = max(
                    1,
                    *(
                        int(s.locator.get("page", 1))
                        for s in sources
                        if isinstance(s.locator.get("page"), int)
                    ),
                )
                blank = prs.slide_layouts[6]
                for _ in range(page_count):
                    prs.slides.add_slide(blank)
                prs.save(path)
                raw = path.read_bytes()
            else:
                raw = f"{file_name}-{len(sources)}-{document_id or 'v1'}".encode()
                path.write_bytes(raw)
    else:
        path.write_bytes(raw)
    plan = library.store.prepare(_COURSE, path, document_id=document_id)
    return library.store.publish(
        plan,
        ParsedSource(title=file_name, sources=sources),
        original=raw,
        content_sha256=_sha(raw),
        document_format=fmt,
    )


def _create_session(client: TestClient) -> str:
    resp = client.post(
        f"{_BASE}/sessions",
        json={"course_ids": [_COURSE]},
        headers={"origin": _HOST},
    )
    assert resp.status_code == 201, resp.text
    return f"{_BASE}/sessions/{resp.json()['session_id']}"


def _ask(
    client: TestClient, session: str, text: str, rev: int = 1, *, client_id: str | None = None
) -> dict:
    resp = client.post(
        f"{session}/turns",
        json={
            "kind": "ask",
            "client_turn_id": client_id or text[:40],
            "scope_revision": rev,
            "text": text,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# RC1: Tighten TargetObjectRoute false triggering while preserving explicit crops


def test_rc1_object_intent_avoids_formula_derivation_and_composite_hijack():
    assert object_intent("请解释这个优化问题里多个公式是怎么推导出来的？") is None
    assert object_intent("正文中的公式主要体现了什么物理直觉？") is None
    assert object_intent("图注一般需要包含哪些要素？") is None
    assert object_intent("请给出稀疏注意力的定义，并说明对应表格里的实验实测结果") is None
    assert object_intent("解释门控机制的原理和下面表格数据的关系") is None

    assert object_intent("第 6 页的公式是什么？请说明各项含义") == ObjectIntent(
        kind="formula", order="formulas"
    )
    assert object_intent("公式 tau_eff 怎么计算的？") == ObjectIntent(
        kind="formula", order="formulas"
    )
    assert object_intent("请查看图 2 的图注") == ObjectIntent(kind="caption", order="captions")


# RC2: Version comparison auto-resolution without filename + stale focus bypass


def test_rc2_version_comparison_auto_resolves_without_filename(tmp_path: Path):
    service = make_workspace(_settings(tmp_path, deepseek_api_key=""))
    nb_a = tmp_path / "Alpha-Lab.ipynb"
    nb_b = tmp_path / "Beta-Notes.ipynb"

    def write_nb(path: Path, text: str) -> None:
        nbformat.write(
            nbformat.v4.new_notebook(
                cells=[
                    nbformat.v4.new_code_cell(
                        "run()",
                        id="bench-cell",
                        execution_count=1,
                        outputs=[nbformat.v4.new_output("stream", name="stdout", text=text)],
                    )
                ]
            ),
            path,
        )

    write_nb(nb_a, "alpha_metric=0.71\n")
    write_nb(nb_b, "beta_schedule=cosine\n")

    with _client(service) as client:
        session = _create_session(client)
        doc_a = client.post(
            f"{session}/import",
            params={"course_id": _COURSE, "filename": "Alpha-Lab.ipynb"},
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=nb_a.read_bytes(),
        ).json()
        doc_b = client.post(
            f"{session}/import",
            params={"course_id": _COURSE, "filename": "Beta-Notes.ipynb"},
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=nb_b.read_bytes(),
        ).json()
        # Cite v1 of both notebooks in this session before upgrading
        _ask(client, session, "alpha_metric", 1, client_id="cite-a-v1")
        _ask(client, session, "beta_schedule", 1, client_id="cite-b-v1")
        # Upgrade only Alpha-Lab to v2
        write_nb(nb_a, "alpha_metric=0.92\n")
        client.post(
            f"{session}/import",
            params={
                "course_id": _COURSE,
                "filename": "Alpha-Lab.ipynb",
                "document_id": doc_a["document_id"],
            },
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=nb_a.read_bytes(),
        )

        # Focused on single-version Beta-Notes: omitted-filename compare binds Alpha-Lab!
        cat = client.get(f"{session}/catalog").json()
        beta_node = next(
            d
            for c in cat["courses"]
            for d in c["documents"]
            if d["document_id"] == doc_b["document_id"]
        )
        foc = client.put(
            f"{session}/focus", json={"node_id": beta_node["node_id"], "scope_revision": 1}
        ).json()
        rev = foc["scope_revision"]

        single_multi = _ask(
            client, session, "新旧版本相比有什么变化？", rev, client_id="single-multi-auto"
        )
        assert single_multi["turn"]["trace"]["route"] == "version_comparison"
        versions = {
            ev["revision"]["source"]["version"]
            for ev in single_multi["message"]["sources"].values()
        }
        assert versions == {1, 2}

        # Upgrade Beta-Notes to v2 as well (>= 2 multi-version docs)
        write_nb(nb_b, "beta_schedule=linear_warmup\n")
        client.post(
            f"{session}/import",
            params={
                "course_id": _COURSE,
                "filename": "Beta-Notes.ipynb",
                "document_id": doc_b["document_id"],
            },
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=nb_b.read_bytes(),
        )
        rev = client.patch(
            f"{session}/scope",
            json={"course_ids": [_COURSE], "document_ids": [], "scope_revision": rev},
        ).json()["scope_revision"]

        by_symbol = _ask(
            client,
            session,
            "跟上一版相比，beta_schedule 有什么变化？",
            rev,
            client_id="symbol-binds-beta",
        )
        assert by_symbol["turn"]["trace"]["route"] == "version_comparison"
        cited_docs = {ev["file_name"] for ev in by_symbol["message"]["sources"].values()}
        assert cited_docs == {"Beta-Notes.ipynb"}


# RC3a: Unfocused explicit Cell + 第 N 张输出图 / output N resolution


def test_rc3a_unfocused_cell_output_image_resolves_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = _settings(
        tmp_path,
        source_vision_enabled=True,
        dashscope_api_key="r2-mock-dashscope",
        deepseek_api_key="",
    )
    service = make_workspace(settings)
    png1 = _png_bytes((20, 40, 200))
    png2 = _png_bytes((220, 60, 40))

    nb = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("setup = 1", id="c0", execution_count=1),
            nbformat.v4.new_code_cell(
                "plot_curves()",
                id="c1",
                execution_count=2,
                outputs=[
                    nbformat.v4.new_output("stream", name="stdout", text="steps=1000"),
                    nbformat.v4.new_output(
                        "display_data", data={"image/png": base64.b64encode(png1).decode("ascii")}
                    ),
                    nbformat.v4.new_output(
                        "display_data", data={"image/png": base64.b64encode(png2).decode("ascii")}
                    ),
                ],
            ),
        ]
    )
    nb_path = tmp_path / "Eval-Notebook.ipynb"
    nbformat.write(nb, nb_path)

    _publish_doc(
        service.library,
        "Slides.pdf",
        "pdf",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=0,
                title="Slide 1",
                text="General course overview.",
                locator={"format": "pdf", "page": 1},
            )
        ],
    )

    monkeypatch.setattr(service.vision, "enabled", True)
    monkeypatch.setattr(service.vision, "_provider_configured", True)
    monkeypatch.setattr(
        service.vision.provider,
        "read",
        lambda **_kw: VisionReply(
            "succeeded", "第 2 张输出图横轴为 Steps，纵轴为 Accuracy。", (), "req-r2", {}
        ),
    )

    with _client(service) as client:
        session = _create_session(client)
        imp = client.post(
            f"{session}/import",
            params={"course_id": _COURSE, "filename": "Eval-Notebook.ipynb"},
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=nb_path.read_bytes(),
        ).json()
        assert imp["status"] == "published"

        res = _ask(
            client,
            session,
            "第 2 个 Cell 的第 2 张输出图横轴是什么？",
            1,
            client_id="unfocused-cell2-out2",
        )
        assert res["turn"]["status"] == "completed"
        trace = res["turn"]["trace"]
        assert trace["route"] == "target_object"
        assert trace["selected_source"]["source_id"] == "cell:1:output:2"
        ev = next(iter(res["message"]["sources"].values()))
        vis_read = next(r for r in ev["reads"] if r["method"] == "vision")
        assert vis_read["actual_source"]["source_id"] == "cell:1:output:2"
        assert vis_read["origin_execution"] == "real_provider"
        assert vis_read["actual_input_sha256"]

        res_en = _ask(
            client,
            session,
            "第 2 个 Cell output 2 的纵轴是什么？",
            1,
            client_id="unfocused-cell2-output-en",
        )
        assert res_en["turn"]["status"] == "completed"
        assert res_en["turn"]["trace"]["selected_source"]["source_id"] == "cell:1:output:1"

        res_out3 = _ask(
            client,
            session,
            "第 2 个 Cell output 3 的纵轴是什么？",
            1,
            client_id="unfocused-cell2-output3-en",
        )
        assert res_out3["turn"]["status"] == "completed"
        assert res_out3["turn"]["trace"]["selected_source"]["source_id"] == "cell:1:output:2"

        # Adding a second notebook with Cell 1 clarifies ONLY across the 2 notebooks
        nb2_path = tmp_path / "Eval-Notebook-Two.ipynb"
        nbformat.write(nb, nb2_path)
        imp2 = client.post(
            f"{session}/import",
            params={"course_id": _COURSE, "filename": "Eval-Notebook-Two.ipynb"},
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=nb2_path.read_bytes(),
        ).json()
        assert imp2["status"] == "published"
        clar = _ask(
            client,
            session,
            "第 2 个 Cell 第 1 张输出图的图例是什么？",
            1,
            client_id="two-notebooks-clarify",
        )
        assert clar["turn"]["status"] == "clarification"
        labels = [opt["label"] for opt in clar["options"]]
        assert len(labels) == 2
        assert all(".ipynb" in lbl for lbl in labels)


# RC3b: S0 bounded on-demand D03 for visual queries vs zero-overhead text queries


def test_rc3b_s0_visual_reads_bounded_for_visual_queries_zero_for_prose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = _settings(
        tmp_path,
        workspace_retrieval_strategy="s0",
        workspace_vector_retrieval_enabled=False,
        source_vision_enabled=True,
        dashscope_api_key="r2-mock-dashscope",
        workspace_s3_max_visual_reads=2,
    )
    service = make_workspace(settings)
    img_bytes = _png_bytes((30, 160, 90))

    text_doc = _publish_doc(
        service.library,
        "Theory.pdf",
        "pdf",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=0,
                title="Laplace Smoothing",
                text="Naive Bayes Laplace smoothing avoids zero probabilities during training.",
                locator={"format": "pdf", "page": 1},
            )
        ],
    )
    chart_doc = _publish_doc(
        service.library,
        "Curves.pdf",
        "pdf",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=0,
                title="Convergence",
                text="Convergence Curve",
                locator={"format": "pdf", "page": 1},
            ),
            SourceDraft(
                source_id="page:1:image:1",
                kind="page_object",
                parent_id="page:1",
                ordinal=1,
                title="Figure 1",
                text="",
                locator={
                    "format": "pdf",
                    "page": 1,
                    "object_type": "image",
                    "bbox": [0.1, 0.2, 0.9, 0.8],
                    "coordinate_frame": "visible_page_norm_v1",
                    "position_basis": "pdfium_native_image_extents_v1",
                },
            ),
        ],
    )
    _publish_doc(
        service.library,
        "OtherChart.pdf",
        "pdf",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=0,
                title="Throughput",
                text="Throughput Benchmark",
                locator={"format": "pdf", "page": 1},
            ),
            SourceDraft(
                source_id="page:1:image:1",
                kind="page_object",
                parent_id="page:1",
                ordinal=1,
                title="Figure 1",
                text="",
                locator={
                    "format": "pdf",
                    "page": 1,
                    "object_type": "image",
                    "bbox": [0.1, 0.2, 0.9, 0.8],
                    "coordinate_frame": "visible_page_norm_v1",
                    "position_basis": "pdfium_native_image_extents_v1",
                },
            ),
        ],
    )

    orig_render = service.evidence.render_current

    def fake_render(req):
        if req.source.document_id == chart_doc.document_id:
            return EvidenceResult(
                status="available",
                requested_source=req.source,
                display=DisplayRecord(
                    display_source=req.source,
                    actual_source=req.source,
                    rendition_kind="object_crop",
                    media_type="image/png",
                    width=320,
                    height=200,
                    byte_length=len(img_bytes),
                    byte_sha256=_sha(img_bytes),
                    data_base64=base64.b64encode(img_bytes).decode("ascii"),
                    rect_visible_norm=(0.1, 0.2, 0.9, 0.8),
                    renderer="mock_renderer",
                    renderer_version="1.0",
                    cache_hit=False,
                ),
            )
        return orig_render(req)

    vision_calls = []

    def fake_vision_read(*, image, **_kw):
        vision_calls.append(image)
        return VisionReply(
            "succeeded", "曲线图显示学习率 0.01 在第 20 轮收敛最快。", (), "req-s0v", {}
        )

    async def fake_answer(self, messages, **_kw):
        wire = json.loads(messages[1]["content"])
        vis_ids = [
            ev["id"] for ev in wire["evidence"] if any(r["method"] == "vision" for r in ev["reads"])
        ]
        cite = vis_ids[0] if vis_ids else wire["evidence"][0]["id"]
        return {
            "content": json.dumps(
                {
                    "statements": [{"text": "回答已核对证据。", "citations": [cite]}],
                    "missing": "",
                }
            ),
            "usage": {"total_tokens": 24},
        }

    monkeypatch.setattr(service.evidence, "render_current", fake_render)
    monkeypatch.setattr(service.vision, "enabled", True)
    monkeypatch.setattr(service.vision, "_provider_configured", True)
    monkeypatch.setattr(service.vision.provider, "read", fake_vision_read)
    monkeypatch.setattr(DeepSeekClient, "generate_answer", fake_answer)
    assert text_doc.document_id != chart_doc.document_id

    with _client(service) as client:
        session = _create_session(client)
        prose = _ask(
            client,
            session,
            "Naive Bayes 中 Laplace smoothing 的设计意图是什么？",
            client_id="prose-idiom",
        )
        assert prose["turn"]["status"] == "completed"
        assert len(vision_calls) == 0
        assert "s0_visual_reads" not in prose["turn"]["trace"]

        vis = _ask(
            client, session, "Convergence 曲线中哪个学习率的收敛趋势更快？", client_id="s0-visual"
        )
        assert vis["turn"]["status"] == "completed"
        assert 1 <= len(vision_calls) <= 2
        assert vis["turn"]["trace"]["s0_visual_reads"]["accepted_count"] == 1
        assert "s3_visual_reads" not in vis["turn"]["trace"]


# RC4a & RC4b: Cell<->stdout pairing, CJK unspaced substring search & bilingual gate


def test_rc4a_bidirectional_cell_stdout_pairing_and_rc4b_chinese_search(tmp_path: Path):
    settings = _settings(tmp_path, deepseek_api_key="")
    lib = LocalLibrary(settings)
    nb_doc = _publish_doc(
        lib,
        "BayesLab.ipynb",
        "ipynb",
        [
            SourceDraft(
                source_id="cell:0",
                kind="cell",
                ordinal=0,
                title="Cell 0",
                text="def compute_laplace_prior(counts, alpha=1.0):\n    return counts + alpha",
                locator={"format": "ipynb", "cell_index": 0},
            ),
            SourceDraft(
                source_id="cell:0:output:0",
                kind="notebook_output",
                parent_id="cell:0",
                ordinal=1,
                title="Cell 0 plot",
                text="",
                locator={
                    "format": "ipynb",
                    "cell_index": 0,
                    "output_index": 0,
                    "representations": [{"mime_type": "image/png", "status": "available"}],
                },
            ),
            SourceDraft(
                source_id="cell:0:output:1",
                kind="notebook_output",
                parent_id="cell:0",
                ordinal=2,
                title="Cell 0 stdout",
                text="laplace_acc=0.9412",
                locator={"format": "ipynb", "cell_index": 0, "output_index": 1},
            ),
        ],
    )
    notes_doc = _publish_doc(
        lib,
        "Lecture-Bayes.pdf",
        "pdf",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=0,
                title="Page 1",
                text="第10节讨论朴素贝叶斯平滑(Laplace smoothing)与后验修正。",
                locator={"format": "pdf", "page": 1},
            )
        ],
    )

    store = StructureSnapshotStore(settings)
    searcher = StructureSearcher(store)

    zh_res = searcher.search(
        StructureSearchRequest(query="平滑处理", scope={"course_ids": [_COURSE]}),
        allowed_course_ids={_COURSE},
    )
    assert zh_res.status == "ok"
    assert any(hit.source.document_id == notes_doc.document_id for hit in zh_res.candidates)

    mixed_res = searcher.search(
        StructureSearchRequest(query="第10节讲了什么？", scope={"course_ids": [_COURSE]}),
        allowed_course_ids={_COURSE},
    )
    assert mixed_res.status == "ok"
    assert any(hit.source.document_id == notes_doc.document_id for hit in mixed_res.candidates)

    cell_res = searcher.search(
        StructureSearchRequest(query="compute_laplace_prior", scope={"course_ids": [_COURSE]}),
        allowed_course_ids={_COURSE},
    )
    cell_ids = [
        hit.source.source_id
        for hit in cell_res.candidates
        if hit.source.document_id == nb_doc.document_id
    ]
    assert "cell:0" in cell_ids and "cell:0:output:1" in cell_ids

    snap = store.read(SnapshotRequest(course_ids=(_COURSE,)))
    src_by_key = {(s.document_id, s.source_id): s for s in snap.sources}
    src_by_triple = {(s.document_id, s.version, s.source_id): s for s in snap.sources}
    out_hit = next(h for h in cell_res.candidates if h.source.source_id == "cell:0:output:1")
    reordered = _prioritize_continuations(
        [out_hit],
        anchors=3,
        snapshot_sources=src_by_triple,
        docs={d.document_id: d for d in snap.documents},
    )
    assert any(h.source.source_id == "cell:0" for h in reordered)

    zh_only = [src_by_key[(notes_doc.document_id, "page:1")]]
    assert _has_latin_corpus(zh_only) is False
    bilingual_slide = [
        SnapshotSource(
            course_id=_COURSE,
            document_id="bi",
            version=1,
            source_id="page:1",
            parent_id=None,
            ordinal=0,
            kind="page",
            title="Bi",
            text=(
                "机器学习中的 Kernel Trick 和 Support Vector Machine 优化目标："
                "maximize margin subject to KKT complementarity constraints."
            ),
            locator={"page": 1},
            state="available",
        )
    ]
    assert _has_latin_corpus(bilingual_slide) is True


# RC4c: Multi-doc location narrowing & multi-table slide fallback hygiene


def test_rc4c_multidoc_location_narrowing_and_multitable_slide_fallback(tmp_path: Path):
    service = make_workspace(_settings(tmp_path, deepseek_api_key=""))
    _publish_doc(
        service.library,
        "Short-Intro.pdf",
        "pdf",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=0,
                title="Page 1",
                text="Short intro page 1.",
                locator={"format": "pdf", "page": 1},
            )
        ],
    )
    long_doc = _publish_doc(
        service.library,
        "Deep-Lecture.pdf",
        "pdf",
        [
            SourceDraft(
                source_id="page:8",
                kind="page",
                ordinal=7,
                title="Page 8",
                text="第 8 页详细讨论了核函数的正则化系数选择。",
                locator={"format": "pdf", "page": 8},
            )
        ],
    )
    slide_doc = _publish_doc(
        service.library,
        "MultiTable.pptx",
        "pptx",
        [
            SourceDraft(
                source_id="page:2",
                kind="page",
                ordinal=1,
                title="Benchmarks",
                text="硬件对照实验表",
                locator={"format": "pptx", "page": 2},
            ),
            SourceDraft(
                source_id="page:2:table:1",
                kind="page_object",
                parent_id="page:2",
                ordinal=2,
                title="Table A",
                text="Model | Latency (ms)\nBase | 18.2",
                locator={"format": "pptx", "page": 2, "object_type": "table"},
            ),
            SourceDraft(
                source_id="page:2:table:2",
                kind="page_object",
                parent_id="page:2",
                ordinal=3,
                title="Table B",
                text="Model | Memory (MB)\nOpt | 1024",
                locator={"format": "pptx", "page": 2, "object_type": "table"},
            ),
        ],
    )

    with _client(service) as client:
        session = _create_session(client)
        loc_res = _ask(client, session, "第 8 页是怎么选择核函数系数的？", client_id="unique-page8")
        assert loc_res["turn"]["status"] == "completed"
        assert {
            ev["revision"]["source"]["document_id"]
            for ev in loc_res["message"]["sources"].values()
        } == {long_doc.document_id}

        cat = client.get(f"{session}/catalog").json()
        slide_p2 = next(
            n
            for c in cat["courses"]
            for d in c["documents"]
            if d["document_id"] == slide_doc.document_id
            for n in d["nodes"]
            if n["locator"].get("page") == 2
        )
        rev = client.put(
            f"{session}/focus", json={"node_id": slide_p2["node_id"], "scope_revision": 1}
        ).json()["scope_revision"]
        tbl_res = _ask(
            client,
            session,
            "表格里的延迟数值和显存占用分别是多少？",
            rev,
            client_id="multi-table-ask",
        )
        assert tbl_res["turn"]["status"] == "completed"
        assert not any(b.get("code") == "object_ambiguous" for b in tbl_res["message"]["blocks"])


def test_r2_post_audit_refinements(tmp_path: Path):
    # 1. NumPy axis=0/1/2 never falsely triggers S0 vision; x-axis / 纵轴 does
    assert has_visual_question("np.array.sum() 的 axis=0/1/2 是怎么解释的？") is False
    assert has_visual_question("图的 x-axis 和纵轴单位是什么？") is True

    # 2. Chinese numeral page parser parses "第二十四页" -> 24, never "哪一页" -> 1
    assert _parse_location_num("二十四") == 24
    assert _parse_location_num("十一") == 11
    assert _parse_location_num("一百零三") == 103
    assert _location("展示在哪一页幻灯片？", None) is None
    assert _location("第二十四页的表格列了什么？", None).page == 24

    # 3. E2E: unlocated figure doc question with "不同" falls back cleanly;
    # "换文件到... 它要求..." switches without definition clarification;
    # "第二十四页的表格..." resolves across multi-doc course via table object match.
    service = make_workspace(_settings(tmp_path, deepseek_api_key=""))
    pdf_doc = _publish_doc(
        service.library,
        "Edge-Lecture.pdf",
        "pdf",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=0,
                title="Edges",
                text="不同的边缘成因包括表面法向量不连续和投影阴影。",
                locator={"format": "pdf", "page": 1},
            ),
            SourceDraft(
                source_id="page:1:image:1",
                kind="page_object",
                parent_id="page:1",
                ordinal=1,
                title="Fig 1",
                text="",
                locator={
                    "format": "pdf",
                    "page": 1,
                    "object_type": "image",
                    "bbox": [0.1, 0.1, 0.5, 0.5],
                    "coordinate_frame": "visible_page_norm_v1",
                    "position_basis": "pdfium_native_image_extents_v1",
                },
            ),
            SourceDraft(
                source_id="page:2",
                kind="page",
                ordinal=2,
                title="Gradient",
                text="边缘梯度核定义。Definition: gradient.",
                locator={"format": "pdf", "page": 2},
            ),
            SourceDraft(
                source_id="page:2:image:1",
                kind="page_object",
                parent_id="page:2",
                ordinal=3,
                title="Fig 2",
                text="",
                locator={
                    "format": "pdf",
                    "page": 2,
                    "object_type": "image",
                    "bbox": [0.5, 0.1, 0.9, 0.5],
                    "coordinate_frame": "visible_page_norm_v1",
                    "position_basis": "pdfium_native_image_extents_v1",
                },
            ),
            SourceDraft(
                source_id="page:24",
                kind="page",
                ordinal=24,
                title="Page 24 PDF",
                text="General notes on page 24 without any table.",
                locator={"format": "pdf", "page": 24},
            ),
        ],
    )
    deck_doc = _publish_doc(
        service.library,
        "Course-Plan.pptx",
        "pptx",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=0,
                title="Textbook",
                text="Course Textbook: None required. Recommended Reading listed.",
                locator={"format": "pptx", "page": 1},
            ),
            SourceDraft(
                source_id="page:24",
                kind="page",
                ordinal=23,
                title="Grading Table",
                text="评分构成表",
                locator={"format": "pptx", "page": 24},
            ),
            SourceDraft(
                source_id="page:24:table:1",
                kind="page_object",
                parent_id="page:24",
                ordinal=24,
                title="Table 1",
                text="Item | Weight\nHomework | 30%",
                locator={"format": "pptx", "page": 24, "object_type": "table"},
            ),
            SourceDraft(
                source_id="page:24:table:2",
                kind="page_object",
                parent_id="page:24",
                ordinal=25,
                title="Table 2",
                text="Exam | 50%",
                locator={"format": "pptx", "page": 24, "object_type": "table"},
            ),
        ],
    )

    with _client(service) as client:
        session = _create_session(client)
        r_diff = _ask(
            client,
            session,
            "Edge-Lecture.pdf 用示例说明了哪些不同的边缘成因？图中标签有何意义？",
            client_id="f07-diff-causes",
        )
        assert r_diff["turn"]["status"] == "completed"
        assert not any(b.get("code") == "object_ambiguous" for b in r_diff["message"]["blocks"])

        _ask(client, session, "边缘梯度核的定义是什么？", client_id="prior-def")
        r_sw = _ask(
            client,
            session,
            "换文件到 Course-Plan.pptx。它要求购买指定教材吗？",
            client_id="f27-switch-req",
        )
        assert r_sw["turn"]["status"] == "completed"
        assert {
            ev["revision"]["source"]["document_id"] for ev in r_sw["message"]["sources"].values()
        } == {deck_doc.document_id}
        assert pdf_doc.document_id != deck_doc.document_id

        r_cn = _ask(
            client,
            session,
            "第二十四页的表格列了什么？",
            client_id="cn-page-24-tbl",
        )
        assert r_cn["turn"]["status"] == "completed"
        cited_docs = {
            ev["revision"]["source"]["document_id"] for ev in r_cn["message"]["sources"].values()
        }
        assert cited_docs == {deck_doc.document_id}
        assert any(
            ev["locator"].get("page") == 24 for ev in r_cn["message"]["sources"].values()
        )


# K02-R3: Bug A (filter multi_docs cleanly) & Bug B (composite table+exp bypasses target_object)


def test_r3_bug_a_version_comparison_filters_single_version_and_resets_v1_focus(tmp_path: Path):
    import sqlite3

    service = make_workspace(_settings(tmp_path, deepseek_api_key=""))
    nb_lab = tmp_path / "Lab-Metrics.ipynb"

    def _write_lab(metric: str) -> None:
        nbformat.write(
            nbformat.v4.new_notebook(
                cells=[
                    nbformat.v4.new_code_cell(
                        "run_bench()",
                        id="bench-metric",
                        execution_count=1,
                        outputs=[
                            nbformat.v4.new_output("stream", name="stdout", text=metric),
                        ],
                    )
                ]
            ),
            nb_lab,
        )

    _write_lab("recall_at_10=84.2%")

    # 1. Single-version PPTX (3 slides) + PDF (v1) alongside Lab-Metrics
    _publish_doc(
        service.library,
        "Deck-Slides.pptx",
        "pptx",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=0,
                title="Overview",
                text="Course overview.",
                locator={"format": "pptx", "page": 1},
            ),
            SourceDraft(
                source_id="page:3",
                kind="page",
                ordinal=2,
                title="Attention Slide",
                text="Attention slide details on page 3.",
                locator={"format": "pptx", "page": 3},
            ),
        ],
    )
    paper_v1 = _publish_doc(
        service.library,
        "Paper-Notes.pdf",
        "pdf",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=0,
                title="Latency Benchmark",
                text="Paper baseline score=70.1%",
                locator={"format": "pdf", "page": 1},
            ),
            SourceDraft(
                source_id="page:1:table:1",
                kind="page_object",
                parent_id="page:1",
                ordinal=1,
                title="Table 1",
                text="Method | Score\nPaperBase | 70.1",
                locator={"format": "pdf", "page": 1, "object_type": "table"},
            ),
        ],
    )

    with _client(service) as client:
        session = _create_session(client)
        # 0 multi-version docs across multiple single-version docs -> comparison_missing_history
        no_multi = _ask(client, session, "v1 和 v2 相比有什么变化？", 1, client_id="r3-no-multi")
        assert no_multi["message"]["blocks"][0]["code"] == "comparison_missing_history"

        # Publish v1 of Lab-Metrics and cite both Lab-Metrics and Paper-Notes in v1
        doc_lab = client.post(
            f"{session}/import",
            params={"course_id": _COURSE, "filename": "Lab-Metrics.ipynb"},
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=nb_lab.read_bytes(),
        ).json()
        _ask(client, session, "recall_at_10", 1, client_id="r3-cite-lab-v1")
        _ask(client, session, "Latency Benchmark score=70.1%", 1, client_id="r3-cite-paper-v1")

        # Upgrade only Lab-Metrics to v2
        _write_lab("recall_at_10=92.6%")
        client.post(
            f"{session}/import",
            params={
                "course_id": _COURSE,
                "filename": "Lab-Metrics.ipynb",
                "document_id": doc_lab["document_id"],
            },
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=nb_lab.read_bytes(),
        )

        # Edge check: a doc with ONLY 1 published version (version=2) is NOT multi-version
        orphan = _publish_doc(
            service.library,
            "Orphan.pdf",
            "pdf",
            [
                SourceDraft(
                    source_id="page:1",
                    kind="page",
                    ordinal=0,
                    title="Orphan",
                    text="Orphan doc v1.",
                    locator={"format": "pdf", "page": 1},
                ),
            ],
        )
        _publish_doc(
            service.library,
            "Orphan.pdf",
            "pdf",
            [
                SourceDraft(
                    source_id="page:1",
                    kind="page",
                    ordinal=0,
                    title="Orphan",
                    text="Orphan doc v2.",
                    locator={"format": "pdf", "page": 1},
                ),
            ],
            document_id=orphan.document_id,
        )
        with sqlite3.connect(service.settings.sqlite_path) as db:
            db.execute(
                "UPDATE local_versions SET published_at = NULL "
                "WHERE document_id = ? AND version = 1",
                (orphan.document_id,),
            )
        assert service.snapshots.multi_version_document_ids(
            [doc_lab["document_id"], orphan.document_id, paper_v1.document_id]
        ) == {doc_lab["document_id"]}
        assert service.library.store.delete_local(orphan.document_id)

        # Focus single-version Deck-Slides.pptx page 3 -> must NOT poison Lab-Metrics compare
        cat = client.get(f"{session}/catalog").json()
        deck_p3 = next(
            n
            for c in cat["courses"]
            for d in c["documents"]
            if d["file_name"] == "Deck-Slides.pptx"
            for n in d["nodes"]
            if n["locator"].get("page") == 3
        )
        rev = client.put(
            f"{session}/focus", json={"node_id": deck_p3["node_id"], "scope_revision": 1}
        ).json()["scope_revision"]

        for cid, q in (
            ("r3-v1-v2", "v1 和 v2 相比有什么变化？"),
            ("r3-old-new-diff", "新旧版本差异是什么？"),
        ):
            cmp_res = _ask(client, session, q, rev, client_id=cid)
            assert cmp_res["turn"]["trace"]["route"] == "version_comparison"
            assert cmp_res["message"]["blocks"][0].get("code") != "comparison_ambiguous"
            assert {
                ev["revision"]["source"]["version"]
                for ev in cmp_res["message"]["sources"].values()
            } == {1, 2}
            assert {
                ev["file_name"] for ev in cmp_res["message"]["sources"].values()
            } == {"Lab-Metrics.ipynb"}

        # Upgrade Paper-Notes.pdf to v2 with changed title; >=2 multi-version docs clarify those 2
        _publish_doc(
            service.library,
            "Paper-Notes.pdf",
            "pdf",
            [
                SourceDraft(
                    source_id="page:1",
                    kind="page",
                    ordinal=0,
                    title="Latency Benchmark Updated",
                    text="Paper improved score=78.4%",
                    locator={"format": "pdf", "page": 1},
                ),
            ],
            document_id=paper_v1.document_id,
        )
        clar = _ask(client, session, "v1 和 v2 相比有什么变化？", rev, client_id="r3-two-multi")
        assert clar["message"]["blocks"][0]["code"] == "comparison_scope"
        opt_labels = [opt["label"] for opt in clar["options"]]
        assert len(opt_labels) == 2
        assert not any("Deck-Slides.pptx" in lbl for lbl in opt_labels)
        paper_opt = next(opt for opt in clar["options"] if "Paper-Notes.pdf" in opt["label"])
        resumed = client.post(
            f"{session}/turns",
            json={
                "kind": "ask",
                "client_turn_id": "r3-two-multi-resume-pdf",
                "scope_revision": rev,
                "text": paper_opt["label"],
                "anchor": {
                    "type": "clarification",
                    "turn_id": clar["turn"]["turn_id"],
                    "option_id": paper_opt["option_id"],
                },
            },
        ).json()
        assert resumed["turn"]["trace"]["route"] == "version_comparison"
        assert {
            ev["revision"]["source"]["version"]
            for ev in resumed["message"]["sources"].values()
        } == {1, 2}


def test_r3_bug_b_composite_table_queries_bypass_object_ambiguity_and_reach_kg(tmp_path: Path):
    # 1. Unit checks for composite table+code/experiment/concept vs explicit single-table lookups
    for composite_q in (
        "讲义中的原理、Table 1 参数与实验代码输出有什么关联？",
        "请先看表 1 的指标限制，再结合实验代码验证结果是否达标",
        "对比讲义 Table 1 与实验 Notebook 代码输出中的实测数据",
        "请结合定义与表 1 的关系说明限制条件",
        "结合表 1 和实验脚本核对实测结果",
    ):
        assert object_intent(composite_q) is None, composite_q

    for explicit_q, expected_kind in (
        ("打开第 1 页的表 1", "table"),
        ("查看 Table 1", "table"),
        ("打开那张表", "table"),
        ("表 1 第一列是什么单位？", "table"),
        ("请先看表 1 的指标，不用查实验代码", "table"),
    ):
        got = object_intent(explicit_q)
        assert got is not None and got.kind == expected_kind, explicit_q

    # 2. E2E course-wide composite query with >= 2 Table 1 docs + KG 1-hop enabled
    service = make_workspace(
        _settings(
            tmp_path,
            deepseek_api_key="",
            workspace_answer_enabled=True,
            workspace_kg_retrieval_enabled=True,
        )
    )
    _publish_doc(
        service.library,
        "Lecture-Deck.pptx",
        "pptx",
        [
            SourceDraft(
                source_id="page:2",
                kind="page",
                ordinal=1,
                title="2. 轴向稀疏路由 (Axis-Sparse Routing)",
                text=(
                    "定义：轴向稀疏路由 (Axis-Sparse Routing) — 按列裁剪 KV 块。\n"
                    "使用 `probe_axis_sparse` 验证。"
                ),
                locator={"format": "pptx", "page": 2},
            ),
            SourceDraft(
                source_id="page:5",
                kind="page",
                ordinal=4,
                title="硬件对照表",
                text="Table 1: 硬件参数表 Memory_Limit=3000 MB",
                locator={"format": "pptx", "page": 5},
            ),
            SourceDraft(
                source_id="page:5:table:1",
                kind="page_object",
                parent_id="page:5",
                ordinal=5,
                title="Table 1",
                text="Metric | Limit\nPeak Memory (MB) | <= 3000",
                locator={"format": "pptx", "page": 5, "object_type": "table"},
            ),
        ],
    )
    _publish_doc(
        service.library,
        "Reference-Paper.pdf",
        "pdf",
        [
            SourceDraft(
                source_id="page:5",
                kind="page",
                ordinal=0,
                title="Page 5",
                text="Table 1: Paper Baseline Scores.",
                locator={"format": "pdf", "page": 5},
            ),
            SourceDraft(
                source_id="page:5:table:1",
                kind="page_object",
                parent_id="page:5",
                ordinal=1,
                title="Table 1",
                text="Baseline | Accuracy\nPaperBase | 75.0",
                locator={"format": "pdf", "page": 5, "object_type": "table"},
            ),
        ],
    )
    nb_path = tmp_path / "Bench-Run.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_code_cell(
                    "res = probe_axis_sparse(kv_blocks)\nprint(res)",
                    id="sparse-cell",
                    execution_count=1,
                    outputs=[
                        nbformat.v4.new_output(
                            "stream", name="stdout", text="Recall@10=88.5%, peak_memory=2800 MB"
                        ),
                        nbformat.v4.new_output(
                            "display_data",
                            data={
                                "image/png": base64.b64encode(_png_bytes((30, 140, 220))).decode(
                                    "ascii"
                                )
                            },
                        ),
                    ],
                )
            ]
        ),
        nb_path,
    )

    with _client(service) as client:
        session = _create_session(client)
        client.post(
            f"{session}/import",
            params={"course_id": _COURSE, "filename": "Bench-Run.ipynb"},
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=nb_path.read_bytes(),
        )

        # Standalone "查看 Table 1" in multi-table scope STILL hits target_object
        bare_tbl = _ask(client, session, "查看 Table 1", 1, client_id="r3-bare-tbl")
        assert bare_tbl["turn"]["trace"]["route"] == "target_object"
        assert bare_tbl["message"]["blocks"][0]["code"] == "object_ambiguous"

        # Course-wide composite query ("表 1" / "Table 1" + code + experiment output + concept)
        # must bypass target_object and enter retrieval + KG 1-hop!
        comp = _ask(
            client,
            session,
            "请解释 轴向稀疏路由 的定义，结合表 1 的 Peak Memory 限制，"
            "再核对实验代码和输出中的实测数据",
            1,
            client_id="r3-composite-kg",
        )
        assert comp["turn"]["status"] == "completed"
        assert comp["turn"]["trace"].get("route") != "target_object"
        assert not any(
            b.get("code") == "object_ambiguous" for b in comp["message"]["blocks"]
        )
        kg_trace = comp["turn"]["trace"].get("kg") or {}
        assert kg_trace.get("status") in {"succeeded", "already_covered"}
        cited_files = {ev["file_name"] for ev in comp["message"]["sources"].values()}
        assert "Bench-Run.ipynb" in cited_files
        assert "Lecture-Deck.pptx" in cited_files

        # Focused multi-output cell still allows table+code+experiment query to bypass hijack
        cat = client.get(f"{session}/catalog").json()
        bench_cell = next(
            n
            for c in cat["courses"]
            for d in c["documents"]
            if d["file_name"] == "Bench-Run.ipynb"
            for n in d["nodes"]
            if n["kind"] == "cell"
        )
        rev = client.put(
            f"{session}/focus", json={"node_id": bench_cell["node_id"], "scope_revision": 1}
        ).json()["scope_revision"]
        cell_comp = _ask(
            client,
            session,
            "请核对 Table 1 的硬件指标限制与实验代码运行输出",
            rev,
            client_id="r3-cell-focus-composite",
        )
        assert cell_comp["turn"]["status"] == "completed"
        assert cell_comp["turn"]["trace"].get("route") != "target_object"
        assert not any(
            b.get("code") == "object_ambiguous" for b in cell_comp["message"]["blocks"]
        )


# K02-R4: Fix 1 (dedup + mixed Cell text/image compare) & Fix 2 (table weighting in budget=6)


def test_r4_fix1_dedup_and_mixed_cell_text_plus_image_version_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from app.rag.source_evidence import ReadRecord

    settings = _settings(
        tmp_path,
        source_vision_enabled=True,
        dashscope_api_key="r4-mock-dashscope",
        deepseek_api_key="",
    )
    service = make_workspace(settings)
    vision_requests: list[tuple[int, str]] = []

    def fake_vision(request, *, expected_input_sha256: str):
        base = service.evidence.render_current(request)
        display = base.display
        assert display is not None and display.byte_sha256 == expected_input_sha256
        vision_requests.append((request.source.version, request.source.source_id))
        rec = ReadRecord(
            requested_source=request.source,
            actual_source=display.actual_source,
            actual_input_sha256=display.byte_sha256,
            actual_input_media_type=display.media_type,
            actual_input_rect=display.rect_visible_norm,
            purpose="directed_question",
            execution="real_provider",
            origin_execution="real_provider",
            provider="r4-fake",
            model="qwen3-vl-plus",
            prompt_version="r4-v1",
            question_sha256=(
                hashlib.sha256(request.question.strip().encode()).hexdigest()
                if request.question
                else None
            ),
            status="succeeded",
            text=f"v{request.source.version} unit=seconds peak=92",
        )
        return base.model_copy(update={"read_status": "succeeded", "reads": [rec]})

    monkeypatch.setattr(service.vision, "process_current", fake_vision)

    _publish_doc(
        service.library,
        "Lecture-Notes.pdf",
        "pdf",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=0,
                title="Notes",
                text="Single-version reference notes.",
                locator={"format": "pdf", "page": 1},
            )
        ],
    )

    nb_path = tmp_path / "Bench-Lab.ipynb"

    def _write_bench(metric: str, col_a: tuple[int, int, int], col_b: tuple[int, int, int]) -> None:
        nbformat.write(
            nbformat.v4.new_notebook(
                cells=[
                    nbformat.v4.new_markdown_cell("# Header", id="md-0"),
                    nbformat.v4.new_code_cell(
                        "eval_metric = run_bench()\nprint(eval_metric)",
                        id="bench-c1",
                        execution_count=1,
                        outputs=[
                            nbformat.v4.new_output("stream", name="stdout", text=metric),
                            nbformat.v4.new_output(
                                "display_data",
                                data={"image/png": base64.b64encode(_png_bytes(col_a)).decode()},
                            ),
                            nbformat.v4.new_output(
                                "display_data",
                                data={"image/png": base64.b64encode(_png_bytes(col_b)).decode()},
                            ),
                        ],
                    ),
                ]
            ),
            nb_path,
        )

    _write_bench("Recall@10=84.2%", (20, 40, 200), (200, 40, 40))

    with _client(service) as client:
        sess = _create_session(client)
        sess_txt = _create_session(client)
        sess_isolated = _create_session(client)

        imp_v1 = client.post(
            f"{sess}/import",
            params={"course_id": _COURSE, "filename": "Bench-Lab.ipynb"},
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=nb_path.read_bytes(),
        ).json()

        # In sess: read output:2 (2nd image) TWICE + cite stdout + read output:1 once
        _ask(client, sess, "Recall@10", 1, client_id="r4-cite-stdout")
        _ask(client, sess, "第 2 个 Cell 第 1 张图的横轴单位是什么？", 1, client_id="r4-img1")
        _ask(client, sess, "第 2 个 Cell 第 2 张图的横轴单位是什么？", 1, client_id="r4-img2-a")
        r_latest = _ask(
            client, sess, "第 2 个 Cell 第 2 张输出图的横轴单位是什么？", 1, client_id="r4-img2-b"
        )
        latest_msg_id = r_latest["message"]["message_id"]

        # In sess_txt: cite ONLY v1 stdout (never opened v1 image!)
        _ask(client, sess_txt, "Recall@10", 1, client_id="r4-txt-only-v1")

        # Upgrade Bench-Lab to v2
        _write_bench("Recall@10=92.4%", (30, 80, 210), (230, 90, 20))
        client.post(
            f"{sess}/import",
            params={
                "course_id": _COURSE,
                "filename": "Bench-Lab.ipynb",
                "document_id": imp_v1["document_id"],
            },
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=nb_path.read_bytes(),
        )
        vision_requests.clear()

        # 1a + 1b: repeated v1 visual reads deduplicate to latest_msg_id AND include v1+v2 stdout
        cmp_both = _ask(
            client,
            sess,
            "对比 v1 和 v2 的指标与第 2 张输出图横轴单位有什么变化？",
            1,
            client_id="r4-cmp-mixed",
        )
        assert cmp_both["turn"]["status"] == "completed"
        assert cmp_both["turn"]["trace"]["route"] == "version_comparison"
        assert cmp_both["turn"]["trace"]["historical_message_id"] == latest_msg_id
        assert cmp_both["message"]["blocks"][0].get("code") != "comparison_ambiguous"
        assert vision_requests == [(2, "cell:1:output:2")]
        joined_text = "\n".join(b.get("text", "") for b in cmp_both["message"]["blocks"])
        assert "84.2%" in joined_text and "92.4%" in joined_text
        img_blocks = [
            b for b in cmp_both["message"]["blocks"] if b.get("type") == "source_image"
        ]
        assert [b["display_source"]["version"] for b in img_blocks] == [1, 2]

        # 1b snapshot recovery when v1 history only cited same-cell stdout + no axis attr
        vision_requests.clear()
        cmp_rec = _ask(
            client,
            sess_txt,
            "对比 v1 和 v2 的指标数值与第 2 张输出图有什么变化？",
            1,
            client_id="r4-cmp-recovered",
        )
        assert cmp_rec["turn"]["status"] == "completed"
        assert cmp_rec["turn"]["trace"]["route"] == "version_comparison"
        assert cmp_rec["message"]["blocks"][0].get("code") != "comparison_ambiguous"
        assert vision_requests == [(2, "cell:1:output:2")]
        rec_text = "\n".join(b.get("text", "") for b in cmp_rec["message"]["blocks"])
        assert "84.2%" in rec_text and "92.4%" in rec_text
        assert any(
            b.get("code") == "historical_image_unread" for b in cmp_rec["message"]["blocks"]
        )

        # Security check: fresh isolated session without v1 history still rejects
        iso_res = _ask(
            client,
            sess_isolated,
            "对比 v1 和 v2 的指标数值与第 2 张输出图有什么变化？",
            1,
            client_id="r4-iso-reject",
        )
        assert iso_res["message"]["blocks"][0]["code"] == "comparison_missing_history"


def test_r4_fix2_table_prioritization_keeps_matched_tables_within_budget_6(tmp_path: Path):
    from app.rag.structure_search import (
        StructureCandidate,
        StructureScope,
        StructureSearcher,
        StructureSearchRequest,
    )
    from app.rag.workspace_retrieval import plan_retrieval, prioritize_matching_tables
    from app.schemas.local import SourceKey
    from app.stores.local_kg import LocalKgStore

    settings = _settings(
        tmp_path,
        deepseek_api_key="r4-mock",
        workspace_retrieval_strategy="s0",
        workspace_retrieval_budget=6,
        workspace_retrieval_pool_cap=12,
        workspace_kg_retrieval_enabled=True,
    )
    service = make_workspace(settings)

    # Publish 8 verbose prose pages mentioning "AxisGate Throughput" so tables rank past top-6
    pptx_sources: list[SourceDraft] = [
        SourceDraft(
            source_id="page:1",
            kind="page",
            ordinal=0,
            title="1. AxisGate Throughput 概述",
            text=(
                "定义：轴向吞吐调度 (AxisGate Throughput) 控制分块延迟与吞吐。\n"
                "使用 `bench_axis_throughput` 验证。"
            ),
            locator={"format": "pptx", "page": 1},
        )
    ]
    for idx in range(2, 9):
        pptx_sources.append(
            SourceDraft(
                source_id=f"page:{idx}",
                kind="page",
                ordinal=idx - 1,
                title=f"Table 1 Table 2 AxisGate QPS P99 {idx}",
                text=(
                    f"Table 1 Table 2 AxisGate FlashBase QPS P99 metrics comparison {idx}. "
                    "Table 1 and Table 2 AxisGate QPS P99 FlashBase 指标 对比 prose."
                ),
                locator={"format": "pptx", "page": idx},
            )
        )
    pptx_sources.extend(
        [
            SourceDraft(
                source_id="page:9",
                kind="page",
                ordinal=8,
                title="Bench Metrics",
                text="Table 1 Summary",
                locator={"format": "pptx", "page": 9},
            ),
            SourceDraft(
                source_id="page:9:shape:2",
                kind="page_object",
                parent_id="page:9",
                ordinal=9,
                title="Table 1",
                text="Variant | QPS | P99\nAxisGate | 4120 | 7.3",
                locator={"format": "pptx", "page": 9, "object_type": "table"},
            ),
        ]
    )
    pptx_doc = _publish_doc(service.library, "Throughput-Deck.pptx", "pptx", pptx_sources)

    pdf_sources = [
        SourceDraft(
            source_id=f"page:{idx}",
            kind="page",
            ordinal=idx - 1,
            title=f"Table 1 Table 2 Background QPS P99 {idx}",
            text="Table 1 Table 2 AxisGate FlashBase QPS P99 background comparison prose.",
            locator={"format": "pdf", "page": idx},
        )
        for idx in range(1, 4)
    ]
    pdf_sources.extend(
        [
            SourceDraft(
                source_id="page:4",
                kind="page",
                ordinal=3,
                title="Baseline Table",
                text="Table 2 Baseline Metrics",
                locator={"format": "pdf", "page": 4},
            ),
            SourceDraft(
                source_id="page:4:table:1",
                kind="page_object",
                parent_id="page:4",
                ordinal=4,
                title="Table 2",
                text="Variant | QPS | P99\nFlashBase | 2890 | 12.1",
                locator={"format": "pdf", "page": 4, "object_type": "table"},
            ),
        ]
    )
    _publish_doc(service.library, "Baseline-Paper.pdf", "pdf", pdf_sources)

    nb_rel = Path("Throughput-Run.ipynb")
    nb_path = service.library.course_root / _COURSE / nb_rel
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_code_cell(
                    "out = bench_axis_throughput(kv_cache)\nprint(out)",
                    id="tp-c0",
                    execution_count=1,
                    outputs=[
                        nbformat.v4.new_output("stream", name="stdout", text="measured_qps=4088")
                    ],
                )
            ]
        ),
        nb_path,
    )
    _publish_doc(
        service.library,
        "Throughput-Run.ipynb",
        "ipynb",
        [
            SourceDraft(
                source_id="cell:0",
                kind="cell",
                ordinal=0,
                title="Cell 1",
                text="out = bench_axis_throughput(kv_cache)\nprint(out)",
                locator={
                    "format": "ipynb",
                    "cell_index": 0,
                    "cell_id": "tp-c0",
                    "cell_type": "code",
                    "outputs_state": "available",
                },
            ),
            SourceDraft(
                source_id="cell:0:output:0",
                kind="notebook_output",
                parent_id="cell:0",
                ordinal=1,
                title="Output 0",
                text="measured_qps=4088",
                locator={
                    "format": "ipynb",
                    "cell_index": 0,
                    "cell_id": "tp-c0",
                    "output_index": 0,
                    "output_type": "stream",
                    "stream_name": "stdout",
                },
            ),
        ],
        raw=nb_path.read_bytes(),
    )
    # Register KG edge cleanly on local library import/publish
    kg_store = LocalKgStore(settings)
    assert len(kg_store.list_concepts([_COURSE])) >= 1

    snap = service.snapshots.read(SnapshotRequest((_COURSE,), ()))
    searcher = StructureSearcher(service.snapshots)
    scope = StructureScope(
        course_ids=[_COURSE], document_ids=[d.document_id for d in snap.documents]
    )

    # 1. Multi-table metric comparison query keeps both matching tables inside budget=6
    multi_req = StructureSearchRequest(
        query="对比 Table 1 和 Table 2 中 AxisGate 与 FlashBase 的 QPS 和 P99 指标",
        top_k=12,
        include_adjacency=True,
        scope=scope,
    )
    plan_multi = plan_retrieval(
        settings,
        searcher,
        snap,
        multi_req,
        frozenset({_COURSE}),
        strategy="s0",
        budget=6,
        allow_bilingual=False,
        kg_store=kg_store,
        kg_enabled=True,
    )
    assert len(plan_multi.selected) == 6
    assert plan_multi.selected[0].kind == "page"
    selected_tables = [
        c for c in plan_multi.selected if c.locator.get("object_type") == "table"
    ]
    assert len(selected_tables) >= 2
    assert any("4120" in c.text for c in selected_tables)
    assert any("2890" in c.text for c in selected_tables)

    # 2. Composite query ("概念定义 + 表 1 + 实验") preserves definition + Table 1 + KG in budget=6
    comp_req = StructureSearchRequest(
        query="请说明 轴向吞吐调度 的定义，结合 Table 1 的 QPS 限制与实验输出验证是否达标",
        top_k=12,
        include_adjacency=True,
        scope=scope,
    )
    plan_comp = plan_retrieval(
        settings,
        searcher,
        snap,
        comp_req,
        frozenset({_COURSE}),
        strategy="s0",
        budget=6,
        allow_bilingual=False,
        kg_store=kg_store,
        kg_enabled=True,
    )
    assert len(plan_comp.selected) == 6
    assert any(
        c.locator.get("object_type") == "table" and "4120" in c.text for c in plan_comp.selected
    )
    assert any(c.kind == "notebook_output" and "4088" in c.text for c in plan_comp.selected)
    assert any(
        c.source.document_id == pptx_doc.document_id and c.source.source_id == "page:1"
        for c in plan_comp.selected
    )

    # 3. Non-table prose query leaves selection unaltered without injecting tables
    prose_req = StructureSearchRequest(
        query="AxisGate Throughput analysis background",
        top_k=12,
        include_adjacency=True,
        scope=scope,
    )
    plan_prose = plan_retrieval(
        settings,
        searcher,
        snap,
        prose_req,
        frozenset({_COURSE}),
        strategy="s0",
        budget=6,
        allow_bilingual=False,
        kg_store=kg_store,
        kg_enabled=False,
    )
    assert not any(c.locator.get("object_type") == "table" for c in plan_prose.selected)

    # 4. Direct rescue check when matching tables sit at rank 7 & 8 outside budget=6
    synthetic_pool = [
        StructureCandidate(
            source=SourceKey(document_id=pptx_doc.document_id, version=1, source_id=f"page:{i}"),
            course_id=_COURSE,
            file_name="Throughput-Deck.pptx",
            kind="page",
            title=f"Prose {i}",
            text="AxisGate prose chunk",
            locator={"format": "pptx", "page": i},
            state="available",
            text_available=True,
            match_kind="fts",
            reasons=[],
        )
        for i in range(1, 8)
    ] + [
        StructureCandidate(
            source=SourceKey(
                document_id=pptx_doc.document_id, version=1, source_id="page:9:shape:2"
            ),
            course_id=_COURSE,
            file_name="Throughput-Deck.pptx",
            kind="page_object",
            title="Table 1",
            text="Variant | QPS\nAxisGate | 4120",
            locator={"format": "pptx", "page": 9, "object_type": "table"},
            state="available",
            text_available=True,
            match_kind="fts",
            reasons=[],
        ),
        StructureCandidate(
            source=SourceKey(document_id="pdf-other", version=1, source_id="page:4:table:1"),
            course_id=_COURSE,
            file_name="Baseline-Paper.pdf",
            kind="table",
            title="Table 2",
            text="Variant | QPS\nFlashBase | 2890",
            locator={"format": "pdf", "page": 4},
            state="available",
            text_available=True,
            match_kind="fts",
            reasons=[],
        ),
    ]
    rescued = prioritize_matching_tables(
        synthetic_pool,
        question="对比 Table 1 和 Table 2 中的 QPS 指标",
        budget=6,
    )
    assert rescued[0].source.source_id == "page:1"
    assert any(c.source.source_id == "page:9:shape:2" for c in rescued[:6])
    assert any(c.source.source_id == "page:4:table:1" for c in rescued[:6])


def test_r5_chinese_ordinal_output_result_image_and_metric_attribute():
    from app.rag.workspace_versions import _image_attribute

    assert object_intent(
        "对比 v1 和 v2 第 2 张输出图里的耗时是多少 ms",
        include_reference=True,
    ) == ObjectIntent(kind="image", ordinal=2, order="images")
    assert object_intent(
        "对比 v1 和 v2 第二张输出图里的耗时",
        include_reference=True,
    ) == ObjectIntent(kind="image", ordinal=2, order="images")
    assert object_intent(
        "对比 v1 和 v2 第 2 张结果图里的延迟是多少？",
        include_reference=True,
    ) == ObjectIntent(kind="image", ordinal=2, order="images")
    assert object_intent(
        "对比 v1 和 v2 第 2 个输出图里的吞吐变化？",
        include_reference=True,
    ) == ObjectIntent(kind="image", ordinal=2, order="images")
    # Existing "第 N 个输出" with axis preserves order="outputs"
    assert object_intent("第 2 个输出的横轴是什么？") == ObjectIntent(
        kind="image", ordinal=2, order="outputs"
    )
    assert _image_attribute("对比 v1 和 v2 第 2 张输出图里的耗时是多少 ms")
    assert _image_attribute("对比 v1 和 v2 第 2 张结果图里的延迟、准确率和吞吐")
    assert _image_attribute("对比最新版图的纵轴单位和数值") == "纵轴单位"
    assert _image_attribute("请展示第 2 张图") is None


def test_r5_prioritize_matching_tables_prefers_cross_page_complementary_metrics():
    from app.rag.structure_search import StructureCandidate
    from app.rag.workspace_retrieval import prioritize_matching_tables
    from app.schemas.local import SourceKey

    def _cand(
        doc_id: str,
        sid: str,
        kind: str,
        title: str,
        text: str,
        page: int,
        *,
        obj_type: str | None = None,
    ) -> StructureCandidate:
        loc: dict[str, Any] = {"format": "pdf", "page": page}
        if obj_type is not None:
            loc["object_type"] = obj_type
        return StructureCandidate(
            source=SourceKey(document_id=doc_id, version=1, source_id=sid),
            course_id=_COURSE,
            file_name="Paper.pdf",
            kind=kind,
            title=title,
            text=text,
            locator=loc,
            state="available",
            text_available=True,
            match_kind="fts",
            reasons=[],
        )

    prose = [
        _cand("doc-a", f"page:{i}", "page", f"Prose {i}", "metrics prose", i)
        for i in range(1, 5)
    ]
    # Same-page (page=2) has two tables both matching QPS & P99 (score 2);
    # page=3 has another redundant QPS & P99 table (score 2);
    # page=6 has a table matching complementary Latency (score 1).
    tables = [
        _cand(
            "doc-a", "page:2:table:1", "table", "Table 1", "Model | QPS | P99\nA | 4120 | 8.1", 2
        ),
        _cand(
            "doc-a", "page:2:table:2", "table", "Table 1b", "Model | QPS | P99\nB | 3900 | 9.2", 2
        ),
        _cand(
            "doc-a", "page:3:table:1", "table", "Table 3", "Model | QPS | P99\nC | 3800 | 9.5", 3
        ),
        _cand("doc-a", "page:6:table:1", "table", "Table 2", "Model | Latency\nA | 12.4 ms", 6),
    ]
    picked = prioritize_matching_tables(
        [*prose, *tables],
        question="对比表中 QPS、P99 与 Latency 指标",
        budget=6,
    )
    top6_sids = {c.source.source_id for c in picked[:6]}
    assert "page:2:table:1" in top6_sids
    assert "page:6:table:1" in top6_sids
    assert "page:2:table:2" not in top6_sids
    assert "page:3:table:1" not in top6_sids

    # With >=2 explicit table ordinals, still prefer cross-page complementary over page=2 dup
    picked_ord = prioritize_matching_tables(
        [*prose, *tables],
        question="对比 Table 1 和 Table 2 中的 QPS、P99 与 Latency 指标",
        budget=6,
    )
    top6_ord_sids = {c.source.source_id for c in picked_ord[:6]}
    assert "page:2:table:1" in top6_ord_sids
    assert "page:6:table:1" in top6_ord_sids
    assert "page:2:table:2" not in top6_ord_sids


def test_r6_counterpart_pairs_unlabeled_nbformat_44_cells_when_unreordered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from app.rag.source_evidence import ReadRecord

    settings = _settings(
        tmp_path,
        deepseek_api_key="r6-mock",
        dashscope_api_key="r6-mock-dashscope",
        source_vision_enabled=True,
        workspace_answer_enabled=False,
    )
    service = make_workspace(settings)
    vision_requests: list[tuple[int, str]] = []

    def fake_vision(request, *, expected_input_sha256: str):
        base = service.evidence.render_current(request)
        disp = base.display
        assert disp is not None and disp.byte_sha256 == expected_input_sha256
        vision_requests.append((request.source.version, request.source.source_id))
        rec = ReadRecord(
            requested_source=request.source,
            actual_source=disp.actual_source,
            actual_input_sha256=disp.byte_sha256,
            actual_input_media_type=disp.media_type,
            actual_input_rect=disp.rect_visible_norm,
            purpose="directed_question",
            execution="real_provider",
            origin_execution="real_provider",
            provider="r6-fake",
            model="qwen3-vl-plus",
            prompt_version="r6-v1",
            question_sha256=(
                hashlib.sha256(request.question.strip().encode()).hexdigest()
                if request.question
                else None
            ),
            status="succeeded",
            text=f"v{request.source.version} latency=18.2 ms",
        )
        return base.model_copy(update={"read_status": "succeeded", "reads": [rec]})

    monkeypatch.setattr(service.vision, "process_current", fake_vision)
    nb_path = tmp_path / "NoCellId-Bench.ipynb"

    def _write_v44_nb(col_a: tuple[int, int, int], col_b: tuple[int, int, int]) -> None:
        md = nbformat.v4.new_markdown_cell("# Intro")
        code = nbformat.v4.new_code_cell(
            "run_bench()",
            execution_count=1,
            outputs=[
                nbformat.v4.new_output("stream", name="stdout", text="ok"),
                nbformat.v4.new_output(
                    "display_data",
                    data={"image/png": base64.b64encode(_png_bytes(col_a)).decode()},
                ),
                nbformat.v4.new_output(
                    "display_data",
                    data={"image/png": base64.b64encode(_png_bytes(col_b)).decode()},
                ),
            ],
        )
        nb = nbformat.v4.new_notebook(cells=[md, code])
        nb["nbformat_minor"] = 4
        for cell in nb["cells"]:
            cell.pop("id", None)
        nbformat.write(nb, nb_path)

    _write_v44_nb((20, 40, 200), (200, 40, 40))

    with _client(service) as client:
        sess = _create_session(client)
        sess_reord = _create_session(client)
        imp = client.post(
            f"{sess}/import",
            params={"course_id": _COURSE, "filename": "NoCellId-Bench.ipynb"},
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=nb_path.read_bytes(),
        ).json()
        src_v1 = service.library.store.current_sources(imp["document_id"], course_ids=(_COURSE,))
        out2 = next(s for s in src_v1 if s.source_id == "cell:1:output:2")
        assert out2.locator.get("cell_id") is None

        # Read v1 output:2 in sess without writing cell:1 in v1->v2 comparison
        sess_cur = client.get(sess).json()["scope_revision"]
        sess_rev = client.patch(
            f"{sess}/scope",
            json={
                "course_ids": [_COURSE],
                "document_ids": [imp["document_id"]],
                "scope_revision": sess_cur,
            },
        ).json()["scope_revision"]
        _ask(
            client,
            sess,
            "第 2 个 Cell 第 2 张输出图里的耗时是多少 ms？",
            sess_rev,
            client_id="r6-v1",
        )
        _write_v44_nb((30, 80, 210), (220, 90, 20))
        client.post(
            f"{sess}/import",
            params={
                "course_id": _COURSE,
                "filename": "NoCellId-Bench.ipynb",
                "document_id": imp["document_id"],
            },
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=nb_path.read_bytes(),
        )
        vision_requests.clear()
        cmp_ok = _ask(
            client,
            sess,
            "对比 v1 和 v2 第 2 张输出图里的耗时是多少 ms？",
            sess_rev,
            client_id="r6-cmp-ok",
        )
        assert cmp_ok["turn"]["status"] == "completed"
        assert cmp_ok["turn"]["trace"]["route"] == "version_comparison"
        assert cmp_ok["message"]["blocks"][0].get("code") != "comparison_ambiguous"
        assert vision_requests == [(2, "cell:1:output:2")]
        imgs = [b for b in cmp_ok["message"]["blocks"] if b.get("type") == "source_image"]
        assert [b["display_source"]["version"] for b in imgs] == [1, 2]

        # Reordered outputs on a dedicated unlabeled doc (v1 -> v2) still trigger clarification
        _write_v44_nb((20, 40, 200), (200, 40, 40))
        reord_nb = tmp_path / "NoCellId-Reord-Pair.ipynb"
        reord_nb.write_bytes(nb_path.read_bytes())
        imp_reord = client.post(
            f"{sess_reord}/import",
            params={"course_id": _COURSE, "filename": "NoCellId-Reord-Pair.ipynb"},
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=reord_nb.read_bytes(),
        ).json()
        scope_cur = client.get(sess_reord).json()["scope_revision"]
        scope_res = client.patch(
            f"{sess_reord}/scope",
            json={
                "course_ids": [_COURSE],
                "document_ids": [imp_reord["document_id"]],
                "scope_revision": scope_cur,
            },
        )
        assert scope_res.status_code == 200
        reord_rev = scope_res.json()["scope_revision"]
        _ask(
            client,
            sess_reord,
            "第 2 个 Cell 第 2 张输出图里的耗时是多少 ms？",
            reord_rev,
            client_id="r6-reord-v1",
        )
        _write_v44_nb((200, 40, 40), (20, 40, 200))
        client.post(
            f"{sess_reord}/import",
            params={
                "course_id": _COURSE,
                "filename": "NoCellId-Reord-Pair.ipynb",
                "document_id": imp_reord["document_id"],
            },
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=nb_path.read_bytes(),
        )
        reord = _ask(
            client,
            sess_reord,
            "对比上一版和当前版第 2 张输出图里的耗时是多少 ms？",
            reord_rev,
            client_id="r6-cmp-reord",
        )
        assert reord["turn"]["status"] == "clarification"
        assert reord["turn"]["trace"]["clarification_reason"] == "reordered_comparison"


def test_r6_prioritize_matching_tables_uses_parent_text_and_caption():
    from app.rag.structure_search import StructureCandidate
    from app.rag.workspace_retrieval import prioritize_matching_tables
    from app.schemas.local import SourceKey
    from app.stores.structure_snapshot import SnapshotDocument

    docs = {
        "doc-pptx": SnapshotDocument(
            course_id=_COURSE,
            document_id="doc-pptx",
            version=1,
            publication_epoch=1,
            content_sha256="sha-pptx",
            source_path="Deck.pptx",
            file_name="Deck.pptx",
            title="Deck",
            document_format="pptx",
        ),
        "doc-pdf": SnapshotDocument(
            course_id=_COURSE,
            document_id="doc-pdf",
            version=1,
            publication_epoch=1,
            content_sha256="sha-pdf",
            source_path="Paper.pdf",
            file_name="Paper.pdf",
            title="Paper",
            document_format="pdf",
        ),
    }

    def _src(
        doc_id: str,
        sid: str,
        kind: str,
        title: str,
        text: str,
        page: int,
        *,
        obj_type: str | None = None,
    ) -> SnapshotSource:
        loc: dict[str, Any] = {"format": "pdf" if doc_id == "doc-pdf" else "pptx", "page": page}
        if obj_type is not None:
            loc["object_type"] = obj_type
        return SnapshotSource(
            course_id=_COURSE,
            document_id=doc_id,
            version=1,
            source_id=sid,
            parent_id=None,
            ordinal=page,
            kind=kind,
            title=title,
            text=text,
            locator=loc,
            state="available",
        )

    snap_sources = {
        ("doc-pptx", 1, "page:1"): _src(
            "doc-pptx", "page:1", "page", "Slide 1", "LongBench Latency Limit overview", 1
        ),
        ("doc-pptx", 1, "page:2"): _src(
            "doc-pptx", "page:2", "page", "Slide 2", "LongBench Latency Limit context", 2
        ),
        ("doc-pptx", 1, "page:3"): _src(
            "doc-pptx", "page:3", "page", "Slide 3", "LongBench Latency Limit details", 3
        ),
        ("doc-pptx", 1, "page:4"): _src(
            "doc-pptx", "page:4", "page", "Slide 4", "LongBench Latency Limit summary", 4
        ),
        ("doc-pptx", 1, "page:5"): _src(
            "doc-pptx", "page:5", "page", "Slide 5", "Deck Limits", 5
        ),
        ("doc-pptx", 1, "page:5:shape:1"): _src(
            "doc-pptx",
            "page:5:shape:1",
            "page_object",
            "Limit Table",
            "Task | LongBench | Latency | Limit\nRun | Yes | 12ms | 15ms",
            5,
            obj_type="table",
        ),
        ("doc-pptx", 1, "page:6:shape:1"): _src(
            "doc-pptx",
            "page:6:shape:1",
            "page_object",
            "Backup Slide Table",
            "Task | LongBench | Latency | Note\nBackup | Yes | 14ms | Ref",
            6,
            obj_type="table",
        ),
        ("doc-pdf", 1, "page:14"): _src(
            "doc-pdf",
            "page:14",
            "page",
            "Main Benchmark",
            "Results on LongBench main evaluation across models.",
            14,
        ),
        ("doc-pdf", 1, "page:14:caption:1"): _src(
            "doc-pdf",
            "page:14:caption:1",
            "caption",
            "Caption",
            "Table 2: LongBench evaluation comparison.",
            14,
        ),
        ("doc-pdf", 1, "page:14:table:1"): _src(
            "doc-pdf",
            "page:14:table:1",
            "table",
            "Scores",
            "Method | L1 | AVG\nAxis-LoRA | 81.2 | 82.4",
            14,
        ),
    }

    def _cand(doc_id: str, sid: str) -> StructureCandidate:
        s = snap_sources[(doc_id, 1, sid)]
        return StructureCandidate(
            source=SourceKey(document_id=doc_id, version=1, source_id=sid),
            course_id=_COURSE,
            file_name=docs[doc_id].file_name,
            kind=s.kind,
            title=s.title,
            text=s.text,
            locator=s.locator,
            state="available",
            text_available=True,
            match_kind="fts",
            reasons=[],
        )

    # Dense PPTX prose + two PPTX tables sit ahead of PDF table in pool
    pool = [
        _cand("doc-pptx", "page:1"),
        _cand("doc-pptx", "page:2"),
        _cand("doc-pptx", "page:3"),
        _cand("doc-pptx", "page:4"),
        _cand("doc-pptx", "page:5"),
        _cand("doc-pptx", "page:5:shape:1"),
        _cand("doc-pptx", "page:6:shape:1"),
    ]
    picked = prioritize_matching_tables(
        pool,
        question="对比论文中 LongBench 主表的 AVG 与课件限制表中的 Latency 和 Limit 指标",
        snapshot_sources=snap_sources,
        docs=docs,
        budget=6,
    )
    top6_keys = {(c.source.document_id, c.source.source_id) for c in picked[:6]}
    assert ("doc-pptx", "page:5:shape:1") in top6_keys
    assert ("doc-pdf", "page:14:table:1") in top6_keys
    assert ("doc-pptx", "page:6:shape:1") not in top6_keys


def test_r7_counterpart_pairs_regenerated_random_cell_ids_unless_moved_or_reordered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from app.rag.source_evidence import ReadRecord

    settings = _settings(
        tmp_path,
        deepseek_api_key="r7-mock",
        dashscope_api_key="r7-mock-dashscope",
        source_vision_enabled=True,
        workspace_answer_enabled=False,
    )
    service = make_workspace(settings)
    vision_requests: list[tuple[int, str]] = []

    def fake_vision(request, *, expected_input_sha256: str):
        base = service.evidence.render_current(request)
        disp = base.display
        assert disp is not None and disp.byte_sha256 == expected_input_sha256
        vision_requests.append((request.source.version, request.source.source_id))
        rec = ReadRecord(
            requested_source=request.source,
            actual_source=disp.actual_source,
            actual_input_sha256=disp.byte_sha256,
            actual_input_media_type=disp.media_type,
            actual_input_rect=disp.rect_visible_norm,
            purpose="directed_question",
            execution="real_provider",
            origin_execution="real_provider",
            provider="r7-fake",
            model="qwen3-vl-plus",
            prompt_version="r7-v1",
            question_sha256=(
                hashlib.sha256(request.question.strip().encode()).hexdigest()
                if request.question
                else None
            ),
            status="succeeded",
            text=f"v{request.source.version} latency=14.6 ms",
        )
        return base.model_copy(update={"read_status": "succeeded", "reads": [rec]})

    monkeypatch.setattr(service.vision, "process_current", fake_vision)
    nb_path = tmp_path / "Regen45-Bench.ipynb"

    def _write_regen_nb(
        col_a: tuple[int, int, int],
        col_b: tuple[int, int, int],
        *,
        c1_id: str | None = None,
        extra_shifted_id: str | None = None,
    ) -> None:
        c1_kw: dict[str, Any] = {"id": c1_id} if c1_id is not None else {}
        cells = [
            nbformat.v4.new_markdown_cell("# Intro"),
            nbformat.v4.new_code_cell(
                "run_bench()",
                execution_count=1,
                outputs=[
                    nbformat.v4.new_output("stream", name="stdout", text="ok"),
                    nbformat.v4.new_output(
                        "display_data",
                        data={"image/png": base64.b64encode(_png_bytes(col_a)).decode()},
                    ),
                    nbformat.v4.new_output(
                        "display_data",
                        data={"image/png": base64.b64encode(_png_bytes(col_b)).decode()},
                    ),
                ],
                **c1_kw,
            ),
        ]
        if extra_shifted_id is not None:
            cells.append(
                nbformat.v4.new_code_cell(
                    "moved_cell()",
                    id=extra_shifted_id,
                    execution_count=2,
                    outputs=[nbformat.v4.new_output("stream", name="stdout", text="moved")],
                )
            )
        nb = nbformat.v4.new_notebook(cells=cells)
        nbformat.write(nb, nb_path)

    with _client(service) as client:
        # 1. Regenerated nbformat 4.5 random cell_ids (v1 -> v2) pair cleanly when unreordered
        sess = _create_session(client)
        _write_regen_nb((20, 40, 200), (200, 40, 40))
        imp = client.post(
            f"{sess}/import",
            params={"course_id": _COURSE, "filename": "Regen45-Bench.ipynb"},
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=nb_path.read_bytes(),
        ).json()
        v1_srcs = service.library.store.current_sources(imp["document_id"], course_ids=(_COURSE,))
        v1_cid = next(s for s in v1_srcs if s.source_id == "cell:1:output:2").locator["cell_id"]
        assert isinstance(v1_cid, str) and len(v1_cid) == 8

        rev = client.patch(
            f"{sess}/scope",
            json={
                "course_ids": [_COURSE],
                "document_ids": [imp["document_id"]],
                "scope_revision": client.get(sess).json()["scope_revision"],
            },
        ).json()["scope_revision"]
        _ask(client, sess, "第 2 个 Cell 第 2 张输出图里的耗时是多少 ms？", rev, client_id="r7-v1")

        _write_regen_nb((30, 80, 210), (220, 90, 20))
        client.post(
            f"{sess}/import",
            params={
                "course_id": _COURSE,
                "filename": "Regen45-Bench.ipynb",
                "document_id": imp["document_id"],
            },
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=nb_path.read_bytes(),
        )
        v2_srcs = service.library.store.current_sources(imp["document_id"], course_ids=(_COURSE,))
        v2_cid = next(s for s in v2_srcs if s.source_id == "cell:1:output:2").locator["cell_id"]
        assert v2_cid != v1_cid

        vision_requests.clear()
        cmp_ok = _ask(
            client, sess, "对比 v1 和 v2 第 2 张输出图里的耗时是多少 ms？", rev, client_id="r7-ok"
        )
        assert cmp_ok["turn"]["status"] == "completed"
        assert cmp_ok["turn"]["trace"]["route"] == "version_comparison"
        assert cmp_ok["message"]["blocks"][0].get("code") != "comparison_ambiguous"
        assert vision_requests == [(2, "cell:1:output:2")]
        assert [
            b["display_source"]["version"]
            for b in cmp_ok["message"]["blocks"]
            if b.get("type") == "source_image"
        ] == [1, 2]

        # 2. Moved cell (v1_cid now sits at cell 2 in v2) must NOT match cell 1
        sess_moved = _create_session(client)
        _write_regen_nb((20, 40, 200), (200, 40, 40), c1_id="a1b2c3d4")
        imp_m = client.post(
            f"{sess_moved}/import",
            params={"course_id": _COURSE, "filename": "Moved45.ipynb"},
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=nb_path.read_bytes(),
        ).json()
        rev_m = client.patch(
            f"{sess_moved}/scope",
            json={
                "course_ids": [_COURSE],
                "document_ids": [imp_m["document_id"]],
                "scope_revision": client.get(sess_moved).json()["scope_revision"],
            },
        ).json()["scope_revision"]
        _ask(
            client,
            sess_moved,
            "第 2 个 Cell 第 2 张输出图里的耗时是多少 ms？",
            rev_m,
            client_id="r7-m-v1",
        )
        _write_regen_nb(
            (30, 80, 210), (220, 90, 20), c1_id="e5f6a7b8", extra_shifted_id="a1b2c3d4"
        )
        client.post(
            f"{sess_moved}/import",
            params={
                "course_id": _COURSE,
                "filename": "Moved45.ipynb",
                "document_id": imp_m["document_id"],
            },
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=nb_path.read_bytes(),
        )
        moved_res = _ask(
            client,
            sess_moved,
            "对比 v1 和 v2 第 2 张输出图里的耗时是多少 ms？",
            rev_m,
            client_id="r7-m-cmp",
        )
        assert moved_res["message"]["blocks"][0]["code"] == "comparison_ambiguous"

        # 3. Reordered outputs + regenerated random cell_id still trigger reordered clarification
        sess_reord = _create_session(client)
        _write_regen_nb((20, 40, 200), (200, 40, 40), c1_id="1111aaaa")
        imp_r = client.post(
            f"{sess_reord}/import",
            params={"course_id": _COURSE, "filename": "Reord45.ipynb"},
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=nb_path.read_bytes(),
        ).json()
        rev_r = client.patch(
            f"{sess_reord}/scope",
            json={
                "course_ids": [_COURSE],
                "document_ids": [imp_r["document_id"]],
                "scope_revision": client.get(sess_reord).json()["scope_revision"],
            },
        ).json()["scope_revision"]
        _ask(
            client,
            sess_reord,
            "第 2 个 Cell 第 2 张输出图里的耗时是多少 ms？",
            rev_r,
            client_id="r7-r-v1",
        )
        _write_regen_nb((200, 40, 40), (20, 40, 200), c1_id="2222bbbb")
        client.post(
            f"{sess_reord}/import",
            params={
                "course_id": _COURSE,
                "filename": "Reord45.ipynb",
                "document_id": imp_r["document_id"],
            },
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=nb_path.read_bytes(),
        )
        reord_res = _ask(
            client,
            sess_reord,
            "对比上一版和当前版第 2 张输出图里的耗时是多少 ms？",
            rev_r,
            client_id="r7-r-cmp",
        )
        assert reord_res["turn"]["status"] == "clarification"
        assert reord_res["turn"]["trace"]["clarification_reason"] == "reordered_comparison"


def test_r7_prioritize_matching_tables_promotes_up_to_3_complementary_tables_in_budget_6():
    from app.rag.structure_search import StructureCandidate
    from app.rag.workspace_retrieval import prioritize_matching_tables
    from app.schemas.local import SourceKey
    from app.stores.structure_snapshot import SnapshotDocument

    docs = {
        "doc-pptx": SnapshotDocument(
            course_id=_COURSE,
            document_id="doc-pptx",
            version=1,
            publication_epoch=1,
            content_sha256="sha-p",
            source_path="Deck.pptx",
            file_name="Deck.pptx",
            title="Deck",
            document_format="pptx",
        ),
        "doc-pdf": SnapshotDocument(
            course_id=_COURSE,
            document_id="doc-pdf",
            version=1,
            publication_epoch=1,
            content_sha256="sha-d",
            source_path="Paper.pdf",
            file_name="Paper.pdf",
            title="Paper",
            document_format="pdf",
        ),
    }

    def _src(doc_id: str, sid: str, kind: str, title: str, text: str, page: int, **kw):
        loc: dict[str, Any] = {"format": "pdf" if doc_id == "doc-pdf" else "pptx", "page": page}
        if "obj_type" in kw:
            loc["object_type"] = kw["obj_type"]
        return SnapshotSource(
            course_id=_COURSE,
            document_id=doc_id,
            version=1,
            source_id=sid,
            parent_id=None,
            ordinal=page,
            kind=kind,
            title=title,
            text=text,
            locator=loc,
            state="available",
        )

    snaps = {
        ("doc-pptx", 1, f"page:{i}"): _src(
            "doc-pptx", f"page:{i}", "page", f"Slide {i}", "LongBench Latency Limit overview", i
        )
        for i in range(1, 7)
    }
    snaps[("doc-pptx", 1, "page:5:shape:1")] = _src(
        "doc-pptx",
        "page:5:shape:1",
        "page_object",
        "Limit Table",
        "Metric | Latency | Limit\nAxis | 12ms | 15ms",
        5,
        obj_type="table",
    )
    snaps[("doc-pdf", 1, "page:11")] = _src(
        "doc-pdf", "page:11", "page", "Bench 1", "LongBench evaluation results.", 11
    )
    snaps[("doc-pdf", 1, "page:11:table:1")] = _src(
        "doc-pdf",
        "page:11:table:1",
        "table",
        "Table 1",
        "Method | Recall\nAxis | 91.4",
        11,
    )
    snaps[("doc-pdf", 1, "page:14")] = _src(
        "doc-pdf", "page:14", "page", "Bench 2", "RulerBench main results.", 14
    )
    snaps[("doc-pdf", 1, "page:14:table:1")] = _src(
        "doc-pdf",
        "page:14:table:1",
        "table",
        "Table 2",
        "Method | AVG\nAxis | 83.7",
        14,
    )

    def _cand(doc_id: str, sid: str) -> StructureCandidate:
        s = snaps[(doc_id, 1, sid)]
        return StructureCandidate(
            source=SourceKey(document_id=doc_id, version=1, source_id=sid),
            course_id=_COURSE,
            file_name=docs[doc_id].file_name,
            kind=s.kind,
            title=s.title,
            text=s.text,
            locator=s.locator,
            state="available",
            text_available=True,
            match_kind="fts",
            reasons=[],
        )

    pool = [_cand("doc-pptx", f"page:{i}") for i in range(1, 7)] + [
        _cand("doc-pptx", "page:5:shape:1"),
        _cand("doc-pdf", "page:11:table:1"),
        _cand("doc-pdf", "page:14:table:1"),
    ]
    picked = prioritize_matching_tables(
        pool,
        question="对比课件 Latency 和 Limit 限制表、LongBench Recall 表与 RulerBench AVG 主表",
        snapshot_sources=snaps,
        docs=docs,
        budget=6,
    )
    assert picked[0].source.source_id == "page:1"
    top6 = {(c.source.document_id, c.source.source_id) for c in picked[:6]}
    assert ("doc-pptx", "page:5:shape:1") in top6
    assert ("doc-pdf", "page:11:table:1") in top6
    assert ("doc-pdf", "page:14:table:1") in top6


def test_r8_prioritize_matching_tables_triggers_on_tableless_multi_metric_comparison():
    from app.rag.structure_search import StructureCandidate
    from app.rag.workspace_retrieval import prioritize_matching_tables
    from app.schemas.local import SourceKey
    from app.stores.structure_snapshot import SnapshotDocument

    docs = {
        "doc-pptx": SnapshotDocument(
            course_id=_COURSE,
            document_id="doc-pptx",
            version=1,
            publication_epoch=1,
            content_sha256="sha-p8",
            source_path="Deck.pptx",
            file_name="Deck.pptx",
            title="Deck",
            document_format="pptx",
        ),
        "doc-pdf": SnapshotDocument(
            course_id=_COURSE,
            document_id="doc-pdf",
            version=1,
            publication_epoch=1,
            content_sha256="sha-d8",
            source_path="Paper.pdf",
            file_name="Paper.pdf",
            title="Paper",
            document_format="pdf",
        ),
    }

    def _src(doc_id: str, sid: str, kind: str, title: str, text: str, page: int, **kw):
        loc: dict[str, Any] = {"format": "pdf" if doc_id == "doc-pdf" else "pptx", "page": page}
        if "obj_type" in kw:
            loc["object_type"] = kw["obj_type"]
        return SnapshotSource(
            course_id=_COURSE,
            document_id=doc_id,
            version=1,
            source_id=sid,
            parent_id=None,
            ordinal=page,
            kind=kind,
            title=title,
            text=text,
            locator=loc,
            state="available",
        )

    snaps = {
        ("doc-pptx", 1, f"page:{i}"): _src(
            "doc-pptx",
            f"page:{i}",
            "page",
            f"Slide {i}",
            "Model overview parameters accuracy throughput",
            i,
        )
        for i in range(1, 7)
    }
    snaps[("doc-pptx", 1, "page:5:shape:1")] = _src(
        "doc-pptx",
        "page:5:shape:1",
        "page_object",
        "Model Spec",
        "Model | 参数量 | Params\nA | 7B | 7B",
        5,
        obj_type="table",
    )
    snaps[("doc-pdf", 1, "page:11:table:1")] = _src(
        "doc-pdf",
        "page:11:table:1",
        "table",
        "Eval Acc",
        "Model | 准确率 | Accuracy\nA | 89.2% | 89.2%",
        11,
    )
    snaps[("doc-pdf", 1, "page:14:table:1")] = _src(
        "doc-pdf",
        "page:14:table:1",
        "table",
        "Speed Bench",
        "Model | 吞吐 | Throughput\nA | 4120 tok/s | 4120",
        14,
    )

    def _cand(doc_id: str, sid: str) -> StructureCandidate:
        s = snaps[(doc_id, 1, sid)]
        return StructureCandidate(
            source=SourceKey(document_id=doc_id, version=1, source_id=sid),
            course_id=_COURSE,
            file_name=docs[doc_id].file_name,
            kind=s.kind,
            title=s.title,
            text=s.text,
            locator=s.locator,
            state="available",
            text_available=True,
            match_kind="fts",
            reasons=[],
        )

    pool = [_cand("doc-pptx", f"page:{i}") for i in range(1, 7)] + [
        _cand("doc-pptx", "page:5:shape:1"),
        _cand("doc-pdf", "page:11:table:1"),
        _cand("doc-pdf", "page:14:table:1"),
    ]
    # Question has NO word "表"/"table", only comparison cue + multi-metric terms
    picked = prioritize_matching_tables(
        pool,
        question="对比两模型参数量、准确率和吞吐",
        snapshot_sources=snaps,
        docs=docs,
        budget=6,
    )
    assert picked[0].source.source_id == "page:1"
    top6 = {(c.source.document_id, c.source.source_id) for c in picked[:6]}
    assert ("doc-pptx", "page:5:shape:1") in top6
    assert ("doc-pdf", "page:11:table:1") in top6
    assert ("doc-pdf", "page:14:table:1") in top6

    # Unrelated prose comparison without matching tables keeps pool untouched
    unchanged = prioritize_matching_tables(
        pool,
        question="对比两个抽象概念之间的核心定义差异",
        snapshot_sources=snaps,
        docs=docs,
        budget=6,
    )
    assert unchanged == pool


def test_r8_mixed_cell_text_regex_covers_calls_identifiers_and_value_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from app.rag.source_evidence import ReadRecord
    from app.rag.workspace_versions import _MIXED_CELL_TEXT_WITH_IMAGE

    assert _MIXED_CELL_TEXT_WITH_IMAGE.search(
        "对比 v1 和 v2 中 foo_bar() 的 avg_latency 数值变化与第 2 张输出图"
    )
    assert _MIXED_CELL_TEXT_WITH_IMAGE.search("对比第 2 张输出图的指标变化")
    assert _MIXED_CELL_TEXT_WITH_IMAGE.search("输出变化和第 1 张输出图")
    assert _MIXED_CELL_TEXT_WITH_IMAGE.search("对比结果与第 2 张图")
    assert not _MIXED_CELL_TEXT_WITH_IMAGE.search(
        "retrieval_bench.ipynb 第 2 张图的横轴单位是什么"
    )
    assert not _MIXED_CELL_TEXT_WITH_IMAGE.search("对比 v1 和 v2 第 2 张输出图里的耗时是多少 ms？")

    settings = _settings(
        tmp_path,
        deepseek_api_key="",
        dashscope_api_key="r8-mock-dashscope",
        source_vision_enabled=True,
        workspace_answer_enabled=False,
    )
    service = make_workspace(settings)
    vision_requests: list[tuple[int, str]] = []

    def fake_vision(request, *, expected_input_sha256: str):
        base = service.evidence.render_current(request)
        disp = base.display
        assert disp is not None and disp.byte_sha256 == expected_input_sha256
        vision_requests.append((request.source.version, request.source.source_id))
        rec = ReadRecord(
            requested_source=request.source,
            actual_source=disp.actual_source,
            actual_input_sha256=disp.byte_sha256,
            actual_input_media_type=disp.media_type,
            actual_input_rect=disp.rect_visible_norm,
            purpose="directed_question",
            execution="real_provider",
            origin_execution="real_provider",
            provider="r8-fake",
            model="qwen3-vl-plus",
            prompt_version="r8-v1",
            question_sha256=(
                hashlib.sha256(request.question.strip().encode()).hexdigest()
                if request.question
                else None
            ),
            status="succeeded",
            text=f"v{request.source.version} curve peak=88",
        )
        return base.model_copy(update={"read_status": "succeeded", "reads": [rec]})

    monkeypatch.setattr(service.vision, "process_current", fake_vision)
    nb_path = tmp_path / "StageBench.ipynb"

    def _write_nb(metric: str, col_a: tuple[int, int, int], col_b: tuple[int, int, int]) -> None:
        nb = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell("# Stage Bench"),
                nbformat.v4.new_code_cell(
                    "res = bench_stage()\nprint(res)",
                    execution_count=1,
                    outputs=[
                        nbformat.v4.new_output("stream", name="stdout", text=metric),
                        nbformat.v4.new_output(
                            "display_data",
                            data={"image/png": base64.b64encode(_png_bytes(col_a)).decode()},
                        ),
                        nbformat.v4.new_output(
                            "display_data",
                            data={"image/png": base64.b64encode(_png_bytes(col_b)).decode()},
                        ),
                    ],
                ),
            ]
        )
        nbformat.write(nb, nb_path)

    with _client(service) as client:
        sess = _create_session(client)
        _write_nb("p95_lat=18.2", (20, 40, 200), (200, 40, 40))
        imp = client.post(
            f"{sess}/import",
            params={"course_id": _COURSE, "filename": "StageBench.ipynb"},
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=nb_path.read_bytes(),
        ).json()
        rev = client.patch(
            f"{sess}/scope",
            json={
                "course_ids": [_COURSE],
                "document_ids": [imp["document_id"]],
                "scope_revision": client.get(sess).json()["scope_revision"],
            },
        ).json()["scope_revision"]
        # Read v1 output:2 generally (no axis/metric attribute)
        _ask(client, sess, "第 2 个 Cell 第 2 张输出图画了什么？", rev, client_id="r8-v1-img2")

        _write_nb("p95_lat=11.4", (30, 80, 210), (220, 90, 20))
        client.post(
            f"{sess}/import",
            params={
                "course_id": _COURSE,
                "filename": "StageBench.ipynb",
                "document_id": imp["document_id"],
            },
            headers={"content-type": "application/octet-stream", "origin": _HOST},
            content=nb_path.read_bytes(),
        )
        vision_requests.clear()
        res = _ask(
            client,
            sess,
            "对比 v1 和 v2 中 bench_stage() 的 p95_lat 数值变化与第 2 张输出图",
            rev,
            client_id="r8-cmp-mixed",
        )
        assert res["turn"]["status"] == "completed"
        assert res["turn"]["trace"]["route"] == "version_comparison"
        assert vision_requests == [(2, "cell:1:output:2")]
        joined = "\n".join(b.get("text", "") for b in res["message"]["blocks"])
        assert "18.2" in joined and "11.4" in joined
        assert [
            b["display_source"]["version"]
            for b in res["message"]["blocks"]
            if b.get("type") == "source_image"
        ] == [1, 2]
