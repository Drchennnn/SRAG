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
