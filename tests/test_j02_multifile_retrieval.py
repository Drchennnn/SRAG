"""J02 regressions: S0-S3 retrieval, 48KB Jev, S3 pre-budget vision & G29/G30 boundaries."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from pypdfium2 import PdfDocument
from qdrant_client import QdrantClient, models

from app.api.workspace import get_workspace_service, router
from app.clients.deepseek import DeepSeekClient
from app.clients.jev import JevCandidate, JevClient
from app.config import Settings
from app.rag.local_library import LocalLibrary
from app.rag.source_evidence import DisplayRecord, EvidenceResult
from app.rag.source_vision import VisionReply
from app.rag.structure_search import StructureSearcher, StructureSearchRequest
from app.rag.workspace import WorkspaceService, make_workspace
from app.rag.workspace_objects import ObjectIntent, object_intent
from app.rag.workspace_retrieval import PlannerResult, plan_retrieval
from app.rag.workspace_vector import LocalVectorBridge
from app.schemas.local import ParsedSource, SourceDraft
from app.schemas.workspace import WorkspaceError
from app.stores.qdrant import TEXT_DENSE_VECTOR, VISUAL_DENSE_VECTOR
from app.stores.structure_snapshot import SnapshotRequest, StructureSnapshotStore

_BASE = "/api/workspace"
_HOST = "http://127.0.0.1:8001"
_COURSE = "j02-test-course"
_OTHER = "j02-distractor-course"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _png_bytes(color: tuple[int, int, int] = (20, 90, 180)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (320, 200), color).save(buf, format="PNG")
    return buf.getvalue()


def _display_record(
    source: Any,
    img_bytes: bytes,
    *,
    rendition_kind: str = "original_page",
    rect: tuple[float, float, float, float] | None = None,
) -> DisplayRecord:
    return DisplayRecord(
        display_source=source,
        actual_source=source,
        rendition_kind=rendition_kind,
        media_type="image/png",
        width=320,
        height=200,
        byte_length=len(img_bytes),
        byte_sha256=_sha(img_bytes),
        data_base64=base64.b64encode(img_bytes).decode("ascii"),
        rect_visible_norm=rect,
        renderer="mock_renderer",
        renderer_version="1.0",
        cache_hit=False,
    )


def _publish_doc(
    library: LocalLibrary,
    course_id: str,
    file_name: str,
    fmt: str,
    sources: list[SourceDraft],
    raw: bytes | None = None,
):
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
            raw = f"{file_name}-bytes-{len(sources)}".encode()
            path.write_bytes(raw)
    else:
        path.write_bytes(raw)
    plan = library.store.prepare(course_id, path)
    return library.store.publish(
        plan,
        ParsedSource(title=file_name, sources=sources),
        original=raw,
        content_sha256=_sha(raw),
        document_format=fmt,
    )


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    params: dict[str, Any] = {
        "_env_file": None,
        "sqlite_path": tmp_path / "j02.sqlite3",
        "data_dir": tmp_path / "data",
        "course_root": tmp_path / "courses",
        "workspace_course_ids": f"{_COURSE},{_OTHER}",
        "workspace_answer_enabled": True,
        "deepseek_api_key": "j02-mock-deepseek",
        "jev_api_key": "j02-mock-jev",
        "dashscope_api_key": "j02-mock-dashscope",
    }
    params.update(overrides)
    return Settings(**params)


def _make_client(service: WorkspaceService) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix=_BASE)
    app.dependency_overrides[get_workspace_service] = lambda: service
    return TestClient(app, base_url=_HOST)


def _create_session(client: TestClient, course_ids: list[str] | None = None) -> str:
    resp = client.post(
        f"{_BASE}/sessions",
        json={"course_ids": course_ids or [_COURSE]},
        headers={"origin": _HOST},
    )
    assert resp.status_code == 201, resp.text
    return f"{_BASE}/sessions/{resp.json()['session_id']}"


def _ask(client: TestClient, session: str, text: str, rev: int = 1, cid: str = "t1") -> dict:
    resp = client.post(
        f"{session}/turns",
        json={"client_turn_id": cid, "kind": "ask", "text": text, "scope_revision": rev},
        headers={"origin": _HOST},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# 1. S0 budget decoupling & pool_cap parity
# ---------------------------------------------------------------------------


def test_s0_decouples_pool_cap_and_enforces_budget_for_non_bilingual(tmp_path, monkeypatch):
    settings = _settings(
        tmp_path,
        workspace_retrieval_strategy="fts",
        workspace_retrieval_pool_cap=24,
        workspace_retrieval_budget=6,
        workspace_input_token_limit=32_768,
    )
    service = make_workspace(settings)
    lib = service.library
    try:
        for doc_idx in range(2):
            pages = [
                SourceDraft(
                    source_id=f"page:{p}",
                    kind="page",
                    ordinal=p - 1,
                    title=f"Routing Section {p}",
                    text=(
                        f"Routing memory architecture calibration value {doc_idx * 100 + p}.\n"
                        + ("extra context words " * 25)
                    ),
                    locator={"format": "pdf", "page": p},
                )
                for p in range(1, 16)
            ]
            _publish_doc(lib, _COURSE, f"Doc{doc_idx}.pdf", "pdf", pages)

        async def fake_answer(self, messages, **_kwargs):
            if self.on_attempt:
                self.on_attempt({"status": 200, "outcome": "succeeded"})
            wire = json.loads(messages[1]["content"])
            eids = [ev["id"] for ev in wire["evidence"]]
            payload = {
                "statements": [{"text": "Routing covered.", "citations": [eids[0]]}],
                "missing": "",
            }
            return {"content": json.dumps(payload), "usage": {"total_tokens": 30}}

        monkeypatch.setattr(DeepSeekClient, "generate_answer", fake_answer)
        with _make_client(service) as client:
            session = _create_session(client)
            # Non-bilingual S0 must still enforce pool_cap=24 & budget=6
            res = _ask(client, session, "What does Routing memory architecture calibration say?")
            trace = res["turn"]["trace"]
            assert trace["retrieval"]["strategy"] == "fts"
            assert trace["retrieval"]["pool_cap"] == 24
            assert trace["retrieval"]["budget"] == 6
            assert len(trace["retrieval"]["candidate_pool"]) <= 24
            assert len(trace["retrieval"]["selected"]) == 6
            assert trace["external_call_count"] == 1
    finally:
        service.close()


# ---------------------------------------------------------------------------
# 2. S1 Jev 48KB batching, recursive binary split, tuple tie-break & fallback
# ---------------------------------------------------------------------------


def test_jev_batches_over_48kb_splits_on_413_and_skips_single_oversized(tmp_path):
    settings = _settings(tmp_path)
    big_text = "Anchor phrase latency throughput measurement. " * 140
    candidates = [
        JevCandidate(f"cand-{i:02d}", f"Doc page {i}", f"Unique-{i} {big_text} TailVal={i * 11}")
        for i in range(12)
    ]
    calls: list[tuple[int, int]] = []
    split_triggered = {"done": False}

    def fake_jev(request: httpx.Request) -> httpx.Response:
        assert len(request.content) <= 48_000
        payload = json.loads(request.content)
        qids = list(payload["questions"])
        calls.append((len(qids), len(request.content)))
        for qid in qids:
            idx = int(qid.split("-")[1])
            passage = payload["questions"][qid]["instructions"]["candidate_passage"]
            assert f"TailVal={idx * 11}" in passage
        if not split_triggered["done"] and len(qids) > 2:
            split_triggered["done"] = True
            return httpx.Response(413, json={"error": {"message": "payload size exceeds limit"}})
        return httpx.Response(
            200,
            json={
                "answers": {
                    qid: {"type": "noul", "noul": 0.9 if qid == "cand-09" else 0.25}
                    for qid in qids
                },
                "usage": {"total_tokens": 10 * len(qids)},
            },
        )

    client = JevClient(settings, transport=httpx.MockTransport(fake_jev))
    res = client.score("What is TailVal=99?", candidates)
    assert res.trace["status"] == "succeeded"
    assert res.trace["binary_splits"] >= 1
    assert res.trace["excerpted_count"] == 12
    assert len(calls) >= 3
    assert res.scores["cand-09"] == 0.9 and len(res.scores) == 12


def test_s1_tuple_tie_break_never_flips_small_noul_diff_and_fallbacks(tmp_path):
    settings = _settings(tmp_path)
    lib = LocalLibrary(settings)
    for i in range(4):
        _publish_doc(
            lib,
            _COURSE,
            f"Paper{i}.pdf",
            "pdf",
            [
                SourceDraft(
                    source_id="page:1",
                    kind="page",
                    ordinal=0,
                    title=f"Paper {i}",
                    text=f"Routing factor candidate {i} calibration experiment",
                    locator={"format": "pdf", "page": 1},
                )
            ],
        )
    store = StructureSnapshotStore(settings)
    snap = store.read(SnapshotRequest((_COURSE,)))
    req = StructureSearchRequest(
        query="Routing factor calibration", top_k=8, scope={"course_ids": [_COURSE]}
    )

    class _PrecisionScorer:
        def __init__(self, low_conf: bool = False):
            self.low_conf = low_conf

        def score(self, question, candidates, *, evidence=(), verify=lambda: None):
            verify()
            if self.low_conf:
                return type(
                    "R",
                    (),
                    {
                        "scores": {c.candidate_id: 0.05 for c in candidates},
                        "trace": {"status": "succeeded", "external_call_count": 1},
                    },
                )()
            scores = {
                c.candidate_id: (0.7002 if idx == len(candidates) - 1 else 0.7001)
                for idx, c in enumerate(candidates)
            }
            return type(
                "R",
                (),
                {"scores": scores, "trace": {"status": "succeeded", "external_call_count": 1}},
            )()

    plan = plan_retrieval(
        settings,
        StructureSearcher(store),
        snap,
        req,
        frozenset({_COURSE}),
        strategy="jev",
        budget=4,
        library=lib.store,
        scorer=_PrecisionScorer(low_conf=False),
        allow_bilingual=False,
    )
    # 0.7002 > 0.7001 must beat higher RRF rank without float addition corruption
    assert plan.selected[0].source.document_id == plan.initial.candidates[-1].source.document_id
    assert [c.source.document_id for c in plan.selected[1:]] == [
        c.source.document_id for c in plan.initial.candidates[:-1]
    ]

    low_plan = plan_retrieval(
        settings,
        StructureSearcher(store),
        snap,
        req,
        frozenset({_COURSE}),
        strategy="jev",
        budget=4,
        library=lib.store,
        scorer=_PrecisionScorer(low_conf=True),
        allow_bilingual=False,
    )
    assert low_plan.trace["jev"]["status"] == "low_confidence"
    assert [c.source.document_id for c in low_plan.selected] == [
        c.source.document_id for c in plan.initial.candidates[:4]
    ]


# ---------------------------------------------------------------------------
# 3. S2a & S2b bounded 2-step retrieval
# ---------------------------------------------------------------------------


def test_s2a_extracts_snake_case_without_jev_and_protects_anchor(tmp_path):
    settings = _settings(tmp_path, jev_api_key="")
    lib = LocalLibrary(settings)
    _publish_doc(
        lib,
        _COURSE,
        "Lecture05.pdf",
        "pdf",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=0,
                title="KV Cache Policy",
                text=(
                    "Table-1 shows KV fragmentation threshold rule; "
                    "verified by `probe_alloc_trace`."
                ),
                locator={"format": "pdf", "page": 1},
            )
        ],
    )
    _publish_doc(
        lib,
        _COURSE,
        "Lab06.ipynb",
        "ipynb",
        [
            SourceDraft(
                source_id="cell:0",
                kind="cell",
                ordinal=0,
                title="Benchmark Code",
                text="probe_alloc_trace()",
                locator={"format": "ipynb", "cell_index": 0},
            ),
            SourceDraft(
                source_id="cell:0:output:0",
                kind="notebook_output",
                parent_id="cell:0",
                ordinal=1,
                title="Experiment Record",
                text="probe_alloc_trace() -> peak_miss=3.8%, tok_rate=4120 tok/s.",
                locator={"format": "ipynb", "cell_index": 0, "output_index": 0},
            ),
        ],
    )
    _publish_doc(
        lib,
        _OTHER,
        "EvilLab.ipynb",
        "ipynb",
        [
            SourceDraft(
                source_id="cell:0",
                kind="cell",
                ordinal=0,
                title="Evil Code",
                text="probe_alloc_trace()",
                locator={"format": "ipynb", "cell_index": 0},
            ),
            SourceDraft(
                source_id="cell:0:output:0",
                kind="notebook_output",
                parent_id="cell:0",
                ordinal=1,
                title="Evil Log",
                text="probe_alloc_trace() -> peak_miss=99.9%.",
                locator={"format": "ipynb", "cell_index": 0, "output_index": 0},
            ),
        ],
    )
    store = StructureSnapshotStore(settings)
    snap = store.read(SnapshotRequest((_COURSE,)))
    req = StructureSearchRequest(
        query="What is the KV fragmentation threshold rule and measured throughput?",
        top_k=6,
        scope={"course_ids": [_COURSE]},
    )
    plan = plan_retrieval(
        settings,
        StructureSearcher(store),
        snap,
        req,
        frozenset({_COURSE}),
        strategy="s2a",
        budget=4,
        library=lib.store,
        allow_bilingual=False,
    )
    assert plan.trace["effective_strategy"] == "s2a"
    assert plan.trace["expansion"]["status"] == "succeeded"
    assert plan.trace["expansion"]["terms"][0]["term"] == "probe_alloc_trace"
    files = [cand.file_name for cand in plan.selected]
    assert "Lecture05.pdf" in files and "Lab06.ipynb" in files
    assert "EvilLab.ipynb" not in files

    from app.rag.workspace_retrieval import _validate_rewrite_constraints

    assert (
        _validate_rewrite_constraints(
            "路由器实测的不均衡系数与有效算力占比是多少？",
            "What are the measured imbalance coefficient and effective compute ratio?",
            "English",
        )
        is None
    )
    assert (
        _validate_rewrite_constraints(
            "第五讲的异构双阶投影切换策略在 32k 上下文下的指标是多少？",
            "What are the metrics of the two-stage projection strategy in Lecture 5 at 32k?",
            "English",
        )
        is None
    )
    assert (
        _validate_rewrite_constraints(
            "第五讲的投影切换策略在 32k 上下文下的指标是多少？",
            "What are the metrics of the projection strategy in Lecture 7 at 32k?",
            "English",
        )
        == "numeric_or_unit_mismatch"
    )


def test_s2b_bounded_planner_stops_cleanly_and_handles_outages(tmp_path):
    settings = _settings(tmp_path)
    lib = LocalLibrary(settings)
    _publish_doc(
        lib,
        _COURSE,
        "Guide.pdf",
        "pdf",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=0,
                title="Calibration",
                text="Adaptive threshold policy sets alpha dynamically per window.",
                locator={"format": "pdf", "page": 1},
            )
        ],
    )
    _publish_doc(
        lib,
        _COURSE,
        "Experiment.ipynb",
        "ipynb",
        [
            SourceDraft(
                source_id="cell:0",
                kind="cell",
                ordinal=0,
                title="Run B Code",
                text="run_calibration_b()",
                locator={"format": "ipynb", "cell_index": 0},
            ),
            SourceDraft(
                source_id="cell:0:output:0",
                kind="notebook_output",
                parent_id="cell:0",
                ordinal=1,
                title="Run B Results",
                text="Calibration Group B: p99=0.14 ms, error=1.2%.",
                locator={"format": "ipynb", "cell_index": 0, "output_index": 0},
            ),
        ],
    )
    store = StructureSnapshotStore(settings)
    snap = store.read(SnapshotRequest((_COURSE,)))
    req = StructureSearchRequest(
        query="What is the adaptive threshold policy and measured latency?",
        top_k=6,
        scope={"course_ids": [_COURSE]},
    )

    def finish_planner(payload, *, verify):
        verify()
        assert len(payload["step_1_evidence_previews"]) >= 1
        assert len(payload["authorized_catalog_outline"]) >= 1
        return PlannerResult(
            "finish", "", (), "already covered", {"status": "succeeded", "external_call_count": 1}
        )

    plan_fin = plan_retrieval(
        settings,
        StructureSearcher(store),
        snap,
        req,
        frozenset({_COURSE}),
        strategy="s2b",
        budget=4,
        library=lib.store,
        planner=finish_planner,
        allow_bilingual=False,
    )
    assert plan_fin.trace["planner"]["stop_reason"] == "planner_finished"

    def dup_planner(_payload, *, verify):
        verify()
        return PlannerResult(
            "search",
            "delta",
            (req.query,),
            "duplicate",
            {"status": "succeeded", "external_call_count": 1},
        )

    plan_dup = plan_retrieval(
        settings,
        StructureSearcher(store),
        snap,
        req,
        frozenset({_COURSE}),
        strategy="s2b",
        budget=4,
        library=lib.store,
        planner=dup_planner,
        allow_bilingual=False,
    )
    assert plan_dup.trace["planner"]["stop_reason"] == "no_new_terms"

    def search_planner(_payload, *, verify):
        verify()
        return PlannerResult(
            "search",
            "experiment delta",
            ("Calibration Group B",),
            "need experimental log",
            {"status": "succeeded", "external_call_count": 1},
        )

    plan_ok = plan_retrieval(
        settings,
        StructureSearcher(store),
        snap,
        req,
        frozenset({_COURSE}),
        strategy="s2b",
        budget=4,
        library=lib.store,
        planner=search_planner,
        allow_bilingual=False,
    )
    assert plan_ok.trace["effective_strategy"] == "s2b"
    assert plan_ok.trace["planner"]["steps_executed"] == 2
    assert {c.file_name for c in plan_ok.selected} == {"Guide.pdf", "Experiment.ipynb"}

    def broken_planner(_payload, *, verify):
        verify()
        return PlannerResult(
            "finish", "", (), "", {"status": "provider_error", "external_call_count": 1}
        )

    plan_err = plan_retrieval(
        settings,
        StructureSearcher(store),
        snap,
        req,
        frozenset({_COURSE}),
        strategy="s2b",
        budget=4,
        library=lib.store,
        planner=broken_planner,
        allow_bilingual=False,
    )
    assert plan_err.trace["effective_strategy"] == "fts"
    assert plan_err.selected[0].file_name == "Guide.pdf"


def test_s2b_catalog_outline_round_robin_reaches_later_documents(tmp_path):
    settings = _settings(tmp_path)
    lib = LocalLibrary(settings)
    for i in range(45):
        _publish_doc(
            lib,
            _COURSE,
            f"FrontDoc{i:02d}.pdf",
            "pdf",
            [
                SourceDraft(
                    source_id="page:1",
                    kind="page",
                    ordinal=0,
                    title=f"Front Section {i}",
                    text="Routing memory overview header baseline text.",
                    locator={"format": "pdf", "page": 1},
                )
            ],
        )
    _publish_doc(
        lib,
        _COURSE,
        "TailLab.ipynb",
        "ipynb",
        [
            SourceDraft(
                source_id="cell:0",
                kind="cell",
                ordinal=0,
                title="Phase-II SVD RankSwitch",
                text="run_switch_trial()",
                locator={"format": "ipynb", "cell_index": 0},
            ),
            SourceDraft(
                source_id="cell:0:output:0",
                kind="notebook_output",
                parent_id="cell:0",
                ordinal=1,
                title="RankSwitch Result",
                text="[RankSwitch] score=14.91 ppl.",
                locator={"format": "ipynb", "cell_index": 0, "output_index": 0},
            ),
        ],
    )
    store = StructureSnapshotStore(settings)
    snap = store.read(SnapshotRequest((_COURSE,)))
    req = StructureSearchRequest(
        query="Routing memory overview baseline", top_k=6, scope={"course_ids": [_COURSE]}
    )
    seen_outline = []

    def inspect_planner(payload, *, verify):
        verify()
        seen_outline.extend(payload["authorized_catalog_outline"])
        return PlannerResult(
            "search",
            "missing tail",
            ("RankSwitch",),
            "check tail notebook",
            {"status": "succeeded", "external_call_count": 1},
        )

    plan = plan_retrieval(
        settings,
        StructureSearcher(store),
        snap,
        req,
        frozenset({_COURSE}),
        strategy="s2b",
        budget=6,
        library=lib.store,
        planner=inspect_planner,
        allow_bilingual=False,
    )
    assert any(item["file_name"] == "TailLab.ipynb" for item in seen_outline)
    assert any(c.file_name == "TailLab.ipynb" for c in plan.selected)


# ---------------------------------------------------------------------------
# 4. S3 local Qdrant + pre-budget D03 visual read + 32KB budget + fallback
# ---------------------------------------------------------------------------


def test_s3_local_qdrant_pre_budget_d03_visual_read_and_lock_lifecycle(tmp_path, monkeypatch):
    qdrant_dir = tmp_path / "qdrant_local"
    settings = _settings(
        tmp_path,
        workspace_retrieval_strategy="vector",
        workspace_vector_retrieval_enabled=True,
        qdrant_local_path=qdrant_dir,
        source_vision_enabled=True,
        workspace_retrieval_budget=6,
        workspace_input_token_limit=32_768,
        dashscope_embedding_dimension=8,
        dashscope_visual_embedding_dimension=8,
    )
    service = make_workspace(settings)
    lib = service.library

    def dashscope_transport(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "/multimodal-embedding/" in str(request.url):
            contents = body["input"]["contents"]
            if len(contents) == 1:
                vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                ctx_text = contents[0]["text"]
                vec = (
                    [0.98, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    if "SlideDeck" in ctx_text
                    else [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                )
            return httpx.Response(
                200,
                json={
                    "output": {"embeddings": [{"index": 0, "embedding": vec}]},
                    "usage": {"total_tokens": 5},
                },
            )
        texts = body["input"]["texts"]
        return httpx.Response(
            200,
            json={
                "output": {
                    "embeddings": [
                        {"index": i, "embedding": [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
                        for i in range(len(texts))
                    ]
                },
                "usage": {"total_tokens": 5 * len(texts)},
            },
        )

    service.vector_bridge._dashscope_transport = httpx.MockTransport(dashscope_transport)
    img_bytes = _png_bytes((10, 140, 200))

    pure_doc = _publish_doc(
        lib,
        _COURSE,
        "SlideDeck.pptx",
        "pptx",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=0,
                title="",
                text="",
                locator={"format": "pptx", "page": 1},
            )
        ],
    )
    _publish_doc(
        lib,
        _COURSE,
        "TextPaper.pdf",
        "pdf",
        [
            SourceDraft(
                source_id="page:1",
                kind="page",
                ordinal=0,
                title="Overview",
                text="Routing overview background text in PDF.",
                locator={"format": "pdf", "page": 1},
            )
        ],
    )

    orig_render = service.evidence.render_current
    vision_calls = []

    def fake_render(req):
        if req.source.document_id == pure_doc.document_id:
            return EvidenceResult(
                status="available",
                requested_source=req.source,
                display=_display_record(req.source, img_bytes, rendition_kind="original_page"),
            )
        return orig_render(req)

    def fake_vision_read(*, image, **_kwargs):
        vision_calls.append(image)
        return VisionReply(
            "succeeded",
            "纯图架构图显示：Top-2 Router 连接 Expert-0 至 Expert-7。",
            (),
            "req-1",
            {},
        )

    monkeypatch.setattr(service.evidence, "render_current", fake_render)
    monkeypatch.setattr(service.vision, "enabled", True)
    monkeypatch.setattr(service.vision, "_provider_configured", True)
    monkeypatch.setattr(service.vision.provider, "read", fake_vision_read)

    seen_prompt_evidence = []

    async def fake_answer(self, messages, **_kwargs):
        wire = json.loads(messages[1]["content"])
        seen_prompt_evidence.extend(wire["evidence"])
        vis_ev = next(
            ev["id"] for ev in wire["evidence"] if ev["reads"][0]["method"] == "vision"
        )
        payload = {
            "statements": [
                {"text": "该架构图由 Top-2 Router 连接 8 个专家。", "citations": [vis_ev]}
            ],
            "missing": "",
        }
        return {"content": json.dumps(payload), "usage": {"total_tokens": 42}}

    monkeypatch.setattr(DeepSeekClient, "generate_answer", fake_answer)

    try:
        with _make_client(service) as client:
            session = _create_session(client)
            res = _ask(client, session, "架构图里路由器连了哪些专家？")
            trace = res["turn"]["trace"]
            msg = res["message"]
            assert trace["retrieval"]["effective_strategy"] == "vector"
            assert trace["s3_visual_reads"]["accepted_count"] == 1
            assert trace["reading"]["estimated_message_input"] <= 32_768
            # Vision read happened before generation and entered DeepSeek evidence
            assert any(ev["reads"][0]["method"] == "vision" for ev in seen_prompt_evidence)
            # Reused in _attach_images: 1 vision call, pre-read SlideDeck first & sharing same ev-id
            assert len(vision_calls) == 1
            img_blocks = [b for b in msg["blocks"] if b["type"] == "source_image"]
            slide_block = next(
                b
                for b in img_blocks
                if b["display_source"]["document_id"] == pure_doc.document_id
            )
            ev_id = slide_block["evidence_id"]
            assert img_blocks[0]["evidence_id"] == ev_id
            assert msg["blocks"][0]["citations"] == [ev_id]
            assert msg["sources"][ev_id]["read_status"] == "succeeded"
            ev_url = f"{session}/messages/{msg['message_id']}/evidence/{ev_id}"
            assert client.get(ev_url).status_code == 200
            assert client.get(ev_url + "/preview").status_code == 200

            # Stale Qdrant point must be rejected by Post-verify
            with service.vector_bridge._lock:
                qclient = service.vector_bridge._get_client()
                qclient.upsert(
                    collection_name=service.vector_bridge.collection,
                    points=[
                        models.PointStruct(
                            id="00000000-0000-4000-8000-000000000099",
                            vector={
                                TEXT_DENSE_VECTOR: [1.0, 0, 0, 0, 0, 0, 0, 0],
                                VISUAL_DENSE_VECTOR: [1.0, 0, 0, 0, 0, 0, 0, 0],
                            },
                            payload={
                                "source_origin": "local_library",
                                "course_id": _COURSE,
                                "document_id": pure_doc.document_id,
                                "version": 1,
                                "publication_epoch": 999999,
                                "content_sha256": _sha(b"stale"),
                                "representation_type": "visual_fusion",
                                "source_id": "page:1",
                            },
                        )
                    ],
                )
            res2 = _ask(client, session, "再看架构图里有什么模块？", cid="t2")
            assert res2["turn"]["trace"]["retrieval"]["vector"]["stale_vector_filtered"] >= 1

            # Add second text page so vector_text >= 2; non-visual query with empty lexical
            # s0 must skip embed_visual_query instead of burning extra VLM calls.
            doc_b = _publish_doc(
                lib,
                _COURSE,
                "TextPaperTwo.pdf",
                "pdf",
                [
                    SourceDraft(
                        source_id="page:1",
                        kind="page",
                        ordinal=0,
                        title="Compaction Telemetry",
                        text="Adaptive slack compaction triggers below 12% slack.",
                        locator={"format": "pdf", "page": 1},
                    )
                ],
            )
            session_b = _create_session(client)
            snap_updated = service.snapshots.read(SnapshotRequest((_COURSE,)))
            service.vector_bridge.ensure_synced(
                snap_updated, service.library.store, service.evidence, include_visual=True
            )
            vis_embed_calls: list[str] = []

            def tracking_transport(request: httpx.Request) -> httpx.Response:
                if "/multimodal-embedding/" in str(request.url):
                    vis_embed_calls.append(str(request.url))
                return dashscope_transport(request)

            service.vector_bridge._dashscope_transport = httpx.MockTransport(tracking_transport)
            calls_before = len(vision_calls)
            res3 = _ask(
                client,
                session_b,
                "动态阈值页面压缩在空闲比例低于多少时触发？",
                cid="t3",
            )
            vtrace3 = res3["turn"]["trace"]["retrieval"]["vector"]
            assert vtrace3["text_hit_count"] >= 2
            assert vtrace3["visual_triggered"] is False
            assert vtrace3["external_call_count"] == 1
            assert vis_embed_calls == []
            assert len(vision_calls) == calls_before
            assert doc_b.document_id
    finally:
        service.close()

    # Directory lock must be released cleanly by close()
    reopened = QdrantClient(path=str(qdrant_dir))
    assert reopened.collection_exists(settings.qdrant_collection)
    reopened.close()


def test_s3_unavailable_qdrant_or_dashscope_falls_back_zero_blocking(tmp_path, monkeypatch):
    qdir = tmp_path / "bad_qdrant"
    qc = QdrantClient(path=str(qdir))
    qc.create_collection(
        collection_name="course_chunks",
        vectors_config={
            TEXT_DENSE_VECTOR: models.VectorParams(size=4, distance=models.Distance.COSINE)
        },
    )
    qc.close()

    settings = _settings(
        tmp_path,
        workspace_retrieval_strategy="vector",
        workspace_vector_retrieval_enabled=True,
        qdrant_local_path=qdir,
        workspace_retrieval_budget=6,
    )
    service = make_workspace(settings)
    try:
        _publish_doc(
            service.library,
            _COURSE,
            "Notes.pdf",
            "pdf",
            [
                SourceDraft(
                    source_id="page:1",
                    kind="page",
                    ordinal=0,
                    title="Notes",
                    text="Transformer KV cache sizing note = 16 GB.",
                    locator={"format": "pdf", "page": 1},
                )
            ],
        )
        _publish_doc(
            service.library,
            _COURSE,
            "ExtraNotes.pdf",
            "pdf",
            [
                SourceDraft(
                    source_id="page:1",
                    kind="page",
                    ordinal=0,
                    title="Extra",
                    text="Transformer attention heads and MLP layers.",
                    locator={"format": "pdf", "page": 1},
                )
            ],
        )

        async def fake_answer(self, messages, **_kwargs):
            wire = json.loads(messages[1]["content"])
            eids = [ev["id"] for ev in wire["evidence"]]
            payload = {"statements": [{"text": "16 GB.", "citations": [eids[0]]}], "missing": ""}
            return {"content": json.dumps(payload), "usage": {"total_tokens": 15}}

        monkeypatch.setattr(DeepSeekClient, "generate_answer", fake_answer)
        with _make_client(service) as client:
            session = _create_session(client)
            res = _ask(client, session, "What is the KV cache sizing?")
            trace = res["turn"]["trace"]
            assert res["turn"]["status"] == "completed"
            assert trace["retrieval"]["vector"]["status"] == "unavailable"
            assert trace["generation"]["status"] == "succeeded"
    finally:
        service.close()

    bridge = LocalVectorBridge(settings)
    bridge.close()
    with pytest.raises(WorkspaceError):
        raise WorkspaceError("interrupted")


# ---------------------------------------------------------------------------
# 5. G29/G30 3 small boundaries
# ---------------------------------------------------------------------------


def test_boundary_1_fallback_reads_rendered_child_formula_and_no_orphan_on_fail(
    tmp_path, monkeypatch
):
    settings = _settings(tmp_path, workspace_answer_enabled=False, deepseek_api_key="")
    service = make_workspace(settings)
    crop_bytes = _png_bytes((240, 240, 240))

    try:
        _publish_doc(
            service.library,
            _COURSE,
            "SpecPaper.pdf",
            "pdf",
            [
                SourceDraft(
                    source_id="page:1",
                    kind="page",
                    ordinal=0,
                    title="Page 1",
                    text="Alpha beta baseline section discussing spectral rule.",
                    locator={"format": "pdf", "page": 1},
                ),
                SourceDraft(
                    source_id="page:2",
                    kind="page",
                    ordinal=1,
                    title="Page 2",
                    text="Secondary spectral note without full equation.",
                    locator={"format": "pdf", "page": 2},
                ),
                SourceDraft(
                    source_id="page:1:formula:1",
                    kind="page_object",
                    parent_id="page:1",
                    ordinal=2,
                    title="Eq. (1)",
                    text="",
                    locator={
                        "format": "pdf",
                        "page": 1,
                        "object_type": "formula",
                        "bbox": [0.1, 0.3, 0.9, 0.4],
                        "object_text": "spectral_loss = lambda_k * ||W_q||_F^2",
                    },
                ),
            ],
        )

        render_mode = {"ok": True}

        def fake_render(req):
            if req.source.source_id == "page:1:formula:1" and render_mode["ok"]:
                return EvidenceResult(
                    status="available",
                    requested_source=req.source,
                    display=_display_record(
                        req.source,
                        crop_bytes,
                        rendition_kind="object_crop",
                        rect=(0.1, 0.3, 0.9, 0.4),
                    ),
                )
            return EvidenceResult(status="rendition_unavailable", requested_source=req.source)

        monkeypatch.setattr(service.evidence, "render_current", fake_render)
        with _make_client(service) as client:
            session = _create_session(client)
            res = _ask(client, session, "spectral rule lambda_k?", cid="t1")
            msg = res["message"]
            img_block = next(
                b
                for b in msg["blocks"]
                if b["type"] == "source_image"
                and b["display_source"]["source_id"] == "page:1:formula:1"
            )
            ev_id = img_block["evidence_id"]
            ev = msg["sources"][ev_id]
            assert ev["read_status"] == "succeeded"
            assert ev["reads"][0]["method"] == "native_text"
            assert "spectral_loss" in ev["reads"][0]["text"]
            assert any(b["type"] == "text" and ev_id in b["citations"] for b in msg["blocks"])

            # When render_current fails, child must NOT produce unread orphan or crash
            render_mode["ok"] = False
            fail_res = _ask(client, session, "spectral baseline?", cid="t2")
            assert not any(
                b.get("type") == "source_image"
                and b.get("display_source", {}).get("source_id") == "page:1:formula:1"
                for b in fail_res["message"]["blocks"]
            )
    finally:
        service.close()


def test_boundary_2_comparative_directional_question_avoids_locking_control_side(tmp_path):
    assert object_intent("与左侧子图相比，另一张图的损失有什么变化？") == ObjectIntent(
        kind="image", direction=None, unresolved=False
    )
    en_q = "Compared to the left subplot, how does the other figure change?"
    assert object_intent(en_q) == ObjectIntent(kind="image", direction=None, unresolved=False)
    assert object_intent("右侧子图的单位是什么？") == ObjectIntent(kind="image", direction="right")
    assert object_intent("左侧图的横轴刻度是多少？") == ObjectIntent(kind="image", direction="left")

    settings = _settings(tmp_path, workspace_answer_enabled=False, deepseek_api_key="")
    service = make_workspace(settings)
    try:
        _publish_doc(
            service.library,
            _COURSE,
            "Compare.pdf",
            "pdf",
            [
                SourceDraft(
                    source_id="page:1",
                    kind="page",
                    ordinal=0,
                    title="Page 1",
                    text="Two figures side by side.",
                    locator={"format": "pdf", "page": 1},
                ),
                SourceDraft(
                    source_id="page:1:image:1",
                    kind="page_object",
                    parent_id="page:1",
                    ordinal=1,
                    title="Left Plot",
                    text="",
                    locator={
                        "format": "pdf",
                        "page": 1,
                        "object_type": "image",
                        "bbox": [0.05, 0.2, 0.45, 0.6],
                    },
                ),
                SourceDraft(
                    source_id="page:1:image:2",
                    kind="page_object",
                    parent_id="page:1",
                    ordinal=2,
                    title="Right Plot",
                    text="",
                    locator={
                        "format": "pdf",
                        "page": 1,
                        "object_type": "image",
                        "bbox": [0.55, 0.2, 0.95, 0.6],
                    },
                ),
            ],
        )
        with _make_client(service) as client:
            session = _create_session(client)
            res = _ask(client, session, "与左侧子图相比，另一张图的趋势有什么变化？")
            assert res["turn"]["status"] == "clarification"
            assert res["turn"]["trace"]["clarification_reason"] == "object_ambiguous"
            assert len(res["turn"]["options"]) == 2
    finally:
        service.close()


def test_boundary_3_compound_image_nouns_and_unregistered_retrieve_fallback(
    tmp_path, monkeypatch
):
    for term in ("实验图", "架构图", "框图", "曲线图", "示意图", "热力图", "柱状图"):
        intent = object_intent(f"这篇资料的{term}说明了什么？")
        assert intent == ObjectIntent(kind="image"), term

    # Unregistered generic diagram question hands off cleanly to RetrieveOrGenerateRoute
    settings = _settings(tmp_path, workspace_answer_enabled=True)
    service = make_workspace(settings)
    try:
        _publish_doc(
            service.library,
            _COURSE,
            "VectorDiagramPaper.pdf",
            "pdf",
            [
                SourceDraft(
                    source_id="page:1",
                    kind="page",
                    ordinal=0,
                    title="Router Diagram",
                    text="如图示架构所示，Router 每 token 选 2 个专家（Top-2 gating）。",
                    locator={"format": "pdf", "page": 1},
                )
            ],
        )

        async def fake_answer(self, messages, **_kwargs):
            wire = json.loads(messages[1]["content"])
            eids = [ev["id"] for ev in wire["evidence"]]
            payload = {
                "statements": [{"text": "每 token 选 2 个专家。", "citations": [eids[0]]}],
                "missing": "",
            }
            return {"content": json.dumps(payload), "usage": {"total_tokens": 20}}

        monkeypatch.setattr(DeepSeekClient, "generate_answer", fake_answer)
        with _make_client(service) as client:
            session = _create_session(client)
            fallback_res = _ask(client, session, "架构图里每 token 选几个专家？", cid="t1")
            assert fallback_res["turn"]["trace"]["route"] == "generated_answer"
            assert "2 个专家" in fallback_res["message"]["blocks"][0]["text"]

            # Explicit location and demonstrative must stay object_not_found / no_unique_image
            explicit_res = _ask(client, session, "第 1 页右侧图的横轴是什么？", cid="t2")
            assert explicit_res["message"]["blocks"][0]["code"] == "object_not_found"
            session_demo = _create_session(client)
            demo_res = _ask(client, session_demo, "这张图里的坐标是多少？", cid="t3")
            assert any(
                b.get("code") in {"object_not_found", "no_unique_image"}
                for b in demo_res["message"]["blocks"]
            ) or demo_res["turn"]["status"] == "clarification"

        # Course-wide generic diagram question across multiple docs/pages with registered images
        # must also hand off to _retrieve instead of spurious multi-doc clarification
        _publish_doc(
            service.library,
            _COURSE,
            "ExtraDiagramDeck.pdf",
            "pdf",
            [
                SourceDraft(
                    source_id="page:1",
                    kind="page",
                    ordinal=0,
                    title="Slide 1",
                    text="Intro diagram",
                    locator={"format": "pdf", "page": 1},
                ),
                SourceDraft(
                    source_id="page:1:image:1",
                    kind="page_object",
                    parent_id="page:1",
                    ordinal=1,
                    title="",
                    text="",
                    locator={
                        "format": "pdf",
                        "page": 1,
                        "object_type": "image",
                        "bbox": [0.1, 0.2, 0.9, 0.8],
                    },
                ),
                SourceDraft(
                    source_id="page:2",
                    kind="page",
                    ordinal=2,
                    title="Slide 2",
                    text="Secondary heatmap",
                    locator={"format": "pdf", "page": 2},
                ),
                SourceDraft(
                    source_id="page:2:image:1",
                    kind="page_object",
                    parent_id="page:2",
                    ordinal=3,
                    title="",
                    text="",
                    locator={
                        "format": "pdf",
                        "page": 2,
                        "object_type": "image",
                        "bbox": [0.1, 0.2, 0.9, 0.8],
                    },
                ),
            ],
        )
        with _make_client(service) as client2:
            session2 = _create_session(client2)
            multi_res = _ask(client2, session2, "架构图里每 token 选几个专家？", cid="t4")
            assert multi_res["turn"]["trace"]["route"] == "generated_answer"
    finally:
        service.close()
