"""Step C: Single-factor paired experiments targeting only the first breakpoints observed in Step B.

All experiments run in an isolated clean checkout of d993748c01fabaff26586908de9bf72499534012
without modifying the production repository tree.

Experiment groups:
- C1 (Q01: E01, E02, E03): Single-serialization budget alignment between read_window and generate_response.
- C2a (Q02: E04): Unregistered PDF booktabs table -> given correct parent page (page:8) reading.
- C2b (Q03: E05): Registered right-hand subplot page:5:image:1 -> given target object / center-based right-side filter.
- C3a (Q05: E07): Long-PDF pure Chinese recall miss -> C3a-1 (bilingual only) vs C3a-2 (bilingual + C1 budget alignment).
- C3b (Q06: E08, E09): Cross-file candidate pool truncation -> fts vs bilingual (budget=4) vs jev (Zev-1 on exact same bilingual pool).
- C4 (Q07: E10, E11): Natural phrasing misrouted to SupplementRetriever -> C4a (given target objects to existing tool) vs C4b (bounded SAG-style tool selection loop with max_steps=3).
- C5 (Q09: E14, E15): Cross-version natural comparison missed by is_version_comparison -> verify historical v1 grant & route to compare_versions.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import subprocess
import sys
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any
from unittest.mock import patch

import httpx

EXP_CTX: ContextVar[dict[str, Any] | None] = ContextVar("g20_exp_ctx", default=None)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_clean_checkout(repo_root: Path, commit_sha: str, target_dir: Path) -> str:
    resolved = (
        subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", commit_sha])
        .decode("utf-8")
        .strip()
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    archive = subprocess.Popen(
        ["git", "-C", str(repo_root), "archive", resolved],
        stdout=subprocess.PIPE,
    )
    subprocess.check_call(["tar", "-x", "-C", str(target_dir)], stdin=archive.stdout)
    if archive.wait() != 0:
        raise RuntimeError(f"git archive failed for {resolved}")
    return resolved


def _install_provider_recorder(provider_events: list[dict[str, Any]]) -> Any:
    orig_sync = httpx.HTTPTransport.handle_request
    orig_async = httpx.AsyncHTTPTransport.handle_async_request

    def sync_wrapper(transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        ctx = EXP_CTX.get()
        payload = json.loads(request.content)
        url_str = str(request.url)
        provider = (
            "dashscope"
            if "dashscope" in url_str
            else ("jev" if "systemone" in url_str else "http_sync")
        )
        image_sha = None
        if provider == "dashscope":
            try:
                for msg in payload.get("input", {}).get("messages", []):
                    for part in msg.get("content", []):
                        if isinstance(part, dict) and "image" in part:
                            _, b64 = part["image"].split(",", 1)
                            image_sha = _sha256_bytes(base64.b64decode(b64))
            except Exception:
                pass
        event: dict[str, Any] = {
            "experiment": ctx,
            "provider": provider,
            "url": url_str,
            "model": payload.get("model"),
            "image_sha256": image_sha,
            "request_sha256": _sha256_bytes(request.content),
        }
        if provider == "jev":
            event["question_count"] = len(payload.get("questions", {}))
            event["candidate_ids"] = list(payload.get("questions", {}).keys())
        started = perf_counter()
        try:
            response = orig_sync(transport, request)
            event["status_code"] = response.status_code
            try:
                response.read()
                body = response.json()
                if "usage" in body:
                    event["usage"] = body.get("usage")
                if provider == "jev" and "answers" in body:
                    event["answers"] = body["answers"]
            except Exception:
                pass
            return response
        except Exception as exc:
            event["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            event["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
            provider_events.append(event)

    async def async_wrapper(
        transport: httpx.AsyncHTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        ctx = EXP_CTX.get()
        payload = json.loads(request.content)
        url_str = str(request.url)
        provider = (
            "deepseek"
            if "deepseek" in url_str or "/chat/completions" in url_str
            else ("jev" if "systemone" in url_str else "http_async")
        )
        event: dict[str, Any] = {
            "experiment": ctx,
            "provider": provider,
            "url": url_str,
            "model": payload.get("model"),
            "request_sha256": _sha256_bytes(request.content),
        }
        started = perf_counter()
        try:
            response = await orig_async(transport, request)
            event["status_code"] = response.status_code
            try:
                await response.aread()
                body = response.json()
                event["usage"] = body.get("usage")
            except Exception:
                pass
            return response
        except Exception as exc:
            event["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            event["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
            provider_events.append(event)

    httpx.HTTPTransport.handle_request = sync_wrapper
    httpx.AsyncHTTPTransport.handle_async_request = async_wrapper

    def restore() -> None:
        httpx.HTTPTransport.handle_request = orig_sync
        httpx.AsyncHTTPTransport.handle_async_request = orig_async

    return restore


def _aligned_read_window(*args: Any, **kwargs: Any) -> Any:
    """Single-factor fix for C1: align read_window's prompt_reserve (+1600 bytes) with generate_response's double-serialized message framing check (input_budget=32768)."""
    from app.rag.workspace_reading import read_window as orig_read_window

    kwargs = dict(kwargs)
    kwargs["prompt_reserve"] = kwargs.get("prompt_reserve", 0) + 1600
    return orig_read_window(*args, **kwargs)


def _summarize_turn(envelope: dict[str, Any], elapsed_ms: float, calls: list[dict[str, Any]]) -> dict[str, Any]:
    turn = envelope["turn"]
    msg = envelope.get("message") or {}
    blocks = msg.get("blocks", [])
    sources = msg.get("sources", {})
    return {
        "turn_status": turn.get("status"),
        "turn_error": turn.get("error"),
        "http_elapsed_ms": elapsed_ms,
        "trace": turn.get("trace"),
        "message_id": msg.get("message_id"),
        "blocks": blocks,
        "text_blocks": [
            {"text": b["text"], "citations": b.get("citations", [])}
            for b in blocks
            if b.get("type") == "text"
        ],
        "notices": [
            {"code": b.get("code"), "text": b.get("text")}
            for b in blocks
            if b.get("type") == "notice"
        ],
        "images": [
            {
                "display_id": b.get("display_id"),
                "evidence_id": b.get("evidence_id"),
                "display_source": b.get("display_source"),
                "rendition_kind": b.get("rendition_kind"),
                "read_status": b.get("read_status"),
            }
            for b in blocks
            if b.get("type") == "source_image"
        ],
        "sources": {
            k: {
                "source": v["revision"]["source"],
                "file_name": v["file_name"],
                "locator": v["locator"],
                "read_status": v["read_status"],
                "reads": [
                    {
                        "method": r["method"],
                        "status": r["status"],
                        "actual_source": r["actual_source"],
                        "text_chars": len(r.get("text", "")),
                        "text_preview": r.get("text", "")[:200],
                        "actual_input_sha256": r.get("actual_input_sha256"),
                    }
                    for r in v.get("reads", [])
                ],
            }
            for k, v in sources.items()
        },
        "provider_calls": calls,
    }


async def _run_all_paired_experiments(
    tmp: Path,
    env_path: Path,
    annotations: dict[str, Any],
    fixture_dir: Path,
    provider_events: list[dict[str, Any]],
) -> dict[str, Any]:
    from app.api.workspace import get_workspace_service
    from app.clients.deepseek import DeepSeekClient
    from app.config import Settings, get_settings
    from app.main import create_app
    from app.rag.structure_search import StructureLocation, StructureScope, StructureSearchRequest
    from app.rag.workspace import Draft, _doc_ref, _key, make_workspace
    from app.rag.workspace_objects import _on_side as orig_on_side
    from app.rag.workspace_reading import read_window
    from app.rag.workspace_versions import is_version_comparison as orig_is_version_comparison
    from app.schemas.local import SourceKey
    from app.schemas.turn_evidence import TurnRead
    from app.schemas.workspace import TurnRequest
    from app.stores.structure_snapshot import SnapshotRequest

    course_id = annotations["course_id"]
    questions_by_id = {q["question_id"]: q for q in annotations["questions"]}

    def make_isolated_env(name: str, *, strategy: str = "fts") -> tuple[Any, Any, Settings]:
        sub = tmp / name
        settings = Settings(
            _env_file=env_path,
            course_root=sub / "courses",
            data_dir=sub / "data",
            sqlite_path=sub / "data" / "library.sqlite3",
            workspace_course_ids=course_id,
            workspace_answer_enabled=True,
            workspace_retrieval_strategy=strategy,
            source_vision_enabled=True,
            qdrant_url="http://127.0.0.1:1",
            qdrant_api_key="",
        )
        app = create_app()
        service = make_workspace(settings)
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[get_workspace_service] = lambda: service
        return app, service, settings

    async def setup_session_with_files(app: Any) -> tuple[httpx.AsyncClient, str, dict[str, str]]:
        transport = httpx.ASGITransport(app=app)
        client = httpx.AsyncClient(
            transport=transport,
            base_url="http://127.0.0.1:18886",
            headers={"Origin": "http://127.0.0.1:18886"},
        )
        res = await client.post("/api/workspace/sessions", json={})
        assert res.status_code == 201
        session_id = res.json()["session_id"]
        doc_ids: dict[str, str] = {}
        for fname, rel in [
            ("Mixtral-of-Experts-2401.04088v1.pdf", Path("Mixtral-of-Experts-2401.04088v1.pdf")),
            (
                "Direct-Preference-Optimization-2305.18290v3.pdf",
                Path("Direct-Preference-Optimization-2305.18290v3.pdf"),
            ),
            (
                "Mixtral-Router-Verification.ipynb",
                Path("v1") / "Mixtral-Router-Verification.ipynb",
            ),
        ]:
            imp = await client.post(
                f"/api/workspace/sessions/{session_id}/import",
                params={"course_id": course_id, "filename": fname},
                content=(fixture_dir / rel).read_bytes(),
                headers={"Content-Type": "application/octet-stream"},
            )
            assert imp.status_code == 200
            doc_ids[fname] = imp.json()["document_id"]
        return client, session_id, doc_ids

    async def select_doc(client: httpx.AsyncClient, session_id: str, doc_id: str) -> dict[str, Any]:
        s_res = await client.get(f"/api/workspace/sessions/{session_id}")
        sess = s_res.json()
        if sess["scope"]["document_ids"] != [doc_id]:
            sc = await client.patch(
                f"/api/workspace/sessions/{session_id}/scope",
                json={
                    "course_ids": [course_id],
                    "document_ids": [doc_id],
                    "scope_revision": sess["scope_revision"],
                },
            )
            assert sc.status_code == 200
            sess = sc.json()
        cat = (await client.get(f"/api/workspace/sessions/{session_id}/catalog")).json()
        node_id = next(
            d["node_id"]
            for c in cat["courses"]
            for d in c["documents"]
            if d["document_id"] == doc_id
        )
        fc = await client.put(
            f"/api/workspace/sessions/{session_id}/focus",
            json={"node_id": node_id, "scope_revision": sess["scope_revision"]},
        )
        assert fc.status_code == 200
        return fc.json()

    async def ask_turn(
        client: httpx.AsyncClient,
        session_id: str,
        scope_revision: int,
        exp_id: str,
        arm: str,
        qid: str,
        question: str,
    ) -> dict[str, Any]:
        cid = f"g20-c-{exp_id.lower()}-{arm.lower()}-{qid.lower()}"
        before = len(provider_events)
        token = EXP_CTX.set({"exp_id": exp_id, "arm": arm, "question_id": qid, "client_turn_id": cid})
        started = perf_counter()
        try:
            res = await client.post(
                f"/api/workspace/sessions/{session_id}/turns",
                json={
                    "client_turn_id": cid,
                    "kind": "ask",
                    "text": question,
                    "scope_revision": scope_revision,
                    "anchor": None,
                },
            )
        finally:
            EXP_CTX.reset(token)
        elapsed = round((perf_counter() - started) * 1000, 3)
        assert res.status_code == 200, res.text
        calls = provider_events[before:]
        return _summarize_turn(res.json(), elapsed, calls)

    results: dict[str, Any] = {}

    # =========================================================================
    # Experiment C1 (Q01: E01, E02, E03) - Post-reading generation budget mismatch
    # =========================================================================
    app_c1, _, _ = make_isolated_env("c1_q01")
    client_c1, sid_c1, docs_c1 = await setup_session_with_files(app_c1)
    try:
        sess_c1 = await select_doc(client_c1, sid_c1, docs_c1["Mixtral-of-Experts-2401.04088v1.pdf"])
        q01_text = questions_by_id["Q01"]["question"]
        # Control arm: unmodified d993748
        c1_control = await ask_turn(
            client_c1, sid_c1, sess_c1["scope_revision"], "C1", "control_baseline", "Q01", q01_text
        )
        # Treatment arm: single-factor alignment of read_window prompt_reserve (+1600 bytes) with generate_response's 32,768 check
        with patch(
            "app.rag.workspace.read_window",
            side_effect=_aligned_read_window,
        ):
            c1_treatment = await ask_turn(
                client_c1,
                sid_c1,
                sess_c1["scope_revision"],
                "C1",
                "treatment_aligned_prompt_reserve",
                "Q01",
                q01_text,
            )
        results["C1_Q01_post_read_budget_alignment"] = {
            "target_question": "Q01",
            "target_units": ["E01", "E02", "E03"],
            "observed_first_breakpoint": "Layer 5 (答案采用 / read_window prompt_reserve=1800 未计入 generate_response 二次 JSON 转义与每段元数据开销，导致 estimated_input=33939 > input_budget=32768)",
            "single_factor_change": "固定 input_budget=32,768 与 generate_response 校验不变，仅将 read_window 的 prompt_reserve 补齐 +1,600 字节转义预留，使装入窗口的已读段在 generate_response 中 <= 32,768",
            "control": c1_control,
            "treatment": c1_treatment,
        }
    finally:
        await client_c1.aclose()

    # =========================================================================
    # Experiment C2a (Q02: E04) - Unregistered PDF booktabs table -> Given correct parent page (page:8)
    # =========================================================================
    app_c2a, _, _ = make_isolated_env("c2a_q02")
    client_c2a, sid_c2a, docs_c2a = await setup_session_with_files(app_c2a)
    try:
        sess_c2a = await select_doc(client_c2a, sid_c2a, docs_c2a["Mixtral-of-Experts-2401.04088v1.pdf"])
        q02_text = questions_by_id["Q02"]["question"]
        c2a_control = await ask_turn(
            client_c2a, sid_c2a, sess_c2a["scope_revision"], "C2a", "control_baseline", "Q02", q02_text
        )
        # Treatment: per G20 protocol ("对象未登记：先给定正确父页做同题阅读，核对整页可读性与精确对象定位。整页答对只计页级支持"),
        # when target_object finds 0 registered table objects on the explicitly specified page (page:8),
        # read the parent page (page:8) through the existing page reading + generation path.
        with patch("app.rag.workspace.object_intent", return_value=None):
            c2a_treatment = await ask_turn(
                client_c2a,
                sid_c2a,
                sess_c2a["scope_revision"],
                "C2a",
                "treatment_given_parent_page_8",
                "Q02",
                q02_text,
            )
        results["C2a_Q02_unregistered_table_parent_page_read"] = {
            "target_question": "Q02",
            "target_units": ["E04"],
            "observed_first_breakpoint": "Layer 1 (精确对象 page:8:table:1 未登记) + target_object 不回退父页 page:8",
            "single_factor_change": "固定原问题与 location=page:8，在精确表对象未登记时给定父页 page:8 走同题页级阅读（按协议整页答对仅计父页级支持，精确表对象仍记未登记）",
            "control": c2a_control,
            "treatment": c2a_treatment,
        }
    finally:
        await client_c2a.aclose()

    # =========================================================================
    # Experiment C2b (Q03: E05) - Registered right subplot page:5:image:1 filtered out by x0 >= 0.5
    # =========================================================================
    app_c2b, _, _ = make_isolated_env("c2b_q03")
    client_c2b, sid_c2b, docs_c2b = await setup_session_with_files(app_c2b)
    try:
        sess_c2b = await select_doc(client_c2b, sid_c2b, docs_c2b["Mixtral-of-Experts-2401.04088v1.pdf"])
        q03_text = questions_by_id["Q03"]["question"]
        c2b_control = await ask_turn(
            client_c2b, sid_c2b, sess_c2b["scope_revision"], "C2b", "control_baseline", "Q03", q03_text
        )

        def center_aware_on_side(rect: tuple[float, float, float, float], direction: str) -> bool:
            x0, y0, x1, y1 = rect
            if direction == "right":
                return x0 >= 0.5 or ((x0 + x1) / 2 >= 0.55 and x0 >= 0.4)
            if direction == "left":
                return x1 <= 0.5 or ((x0 + x1) / 2 <= 0.45 and x1 <= 0.6)
            return orig_on_side(rect, direction)

        with patch("app.rag.workspace_objects._on_side", side_effect=center_aware_on_side):
            c2b_treatment = await ask_turn(
                client_c2b,
                sid_c2b,
                sess_c2b["scope_revision"],
                "C2b",
                "treatment_select_registered_right_image",
                "Q03",
                q03_text,
            )
            # Also verify no regression on Q10 (ambiguous page 5 query without direction)
            q10_text = questions_by_id["Q10"]["question"]
            c2b_q10_regression_check = await ask_turn(
                client_c2b,
                sid_c2b,
                sess_c2b["scope_revision"],
                "C2b",
                "regression_check_q10_ambiguous",
                "Q10",
                q10_text,
            )
        results["C2b_Q03_right_subplot_selection"] = {
            "target_question": "Q03",
            "target_units": ["E05"],
            "observed_first_breakpoint": "Layer 3 (入选 / workspace_objects._on_side 硬编码 x0 >= 0.5 淘汰了已登记的右图 page:5:image:1 [0.444992, 0.688088, 0.800889, 0.863947])",
            "single_factor_change": "固定原问题、location=page:5 与冷视觉缓存，放宽右图方位判定允许水平中心 >= 0.55 且 x0 >= 0.4 的右半页子图入选（同时核对 Q10 无方位双图仍触发 object_ambiguous）",
            "control": c2b_control,
            "treatment": c2b_treatment,
            "q10_ambiguous_regression_check": c2b_q10_regression_check,
        }
    finally:
        await client_c2b.aclose()

    # =========================================================================
    # Experiment C3a (Q05: E07) - Long PDF pure Chinese recall miss (fts vs bilingual vs bilingual+C1)
    # =========================================================================
    q05_text = questions_by_id["Q05"]["question"]
    # Arm 0: control (fts)
    app_c3a0, _, _ = make_isolated_env("c3a_fts", strategy="fts")
    client_c3a0, sid_c3a0, docs_c3a0 = await setup_session_with_files(app_c3a0)
    try:
        sess_c3a0 = await select_doc(
            client_c3a0, sid_c3a0, docs_c3a0["Direct-Preference-Optimization-2305.18290v3.pdf"]
        )
        c3a_control_fts = await ask_turn(
            client_c3a0, sid_c3a0, sess_c3a0["scope_revision"], "C3a", "control_fts", "Q05", q05_text
        )
    finally:
        await client_c3a0.aclose()

    # Arm 1: single-factor switch to WORKSPACE_RETRIEVAL_STRATEGY=bilingual ONLY
    app_c3a1, _, _ = make_isolated_env("c3a_bilingual", strategy="bilingual")
    client_c3a1, sid_c3a1, docs_c3a1 = await setup_session_with_files(app_c3a1)
    try:
        sess_c3a1 = await select_doc(
            client_c3a1, sid_c3a1, docs_c3a1["Direct-Preference-Optimization-2305.18290v3.pdf"]
        )
        c3a_treatment_bilingual_only = await ask_turn(
            client_c3a1,
            sid_c3a1,
            sess_c3a1["scope_revision"],
            "C3a",
            "treatment_bilingual_only",
            "Q05",
            q05_text,
        )
        # Arm 2: bilingual + C1 prompt_reserve alignment
        with patch(
            "app.rag.workspace.read_window",
            side_effect=_aligned_read_window,
        ):
            c3a_treatment_bilingual_plus_c1 = await ask_turn(
                client_c3a1,
                sid_c3a1,
                sess_c3a1["scope_revision"],
                "C3a",
                "treatment_bilingual_plus_c1_budget",
                "Q05",
                q05_text,
            )
    finally:
        await client_c3a1.aclose()

    results["C3a_Q05_long_doc_bilingual_recall"] = {
        "target_question": "Q05",
        "target_units": ["E07"],
        "observed_first_breakpoint": "Layer 2 (进入候选 / 27 页英文长文在纯中文查询下 FTS 零命中，浏览回退仅取前 12 页且窗口在第 7 页截断，page:20 未进候选)",
        "single_factor_change": "Arm 1 仅将 WORKSPACE_RETRIEVAL_STRATEGY 从 fts 改为 bilingual（验证 page:20 升至候选 #1 并完成 Layer 4 实读，但暴露单文档读满窗口后的 Layer 5 二次 JSON 转义 budget_exceeded 次生断点）；Arm 2 在 bilingual 基础上叠加 C1 窗口转义预留对齐",
        "control_fts": c3a_control_fts,
        "treatment_bilingual_only": c3a_treatment_bilingual_only,
        "treatment_bilingual_plus_c1": c3a_treatment_bilingual_plus_c1,
    }

    # =========================================================================
    # Experiment C3b (Q06: E08, E09) - Cross-file candidate pool truncation: fts vs bilingual vs jev
    # =========================================================================
    q06_text = questions_by_id["Q06"]["question"]
    c3b_arms: dict[str, Any] = {}
    for arm_strategy in ("fts", "bilingual", "jev"):
        app_c3b, _, _ = make_isolated_env(f"c3b_{arm_strategy}", strategy=arm_strategy)
        client_c3b, sid_c3b, _ = await setup_session_with_files(app_c3b)
        try:
            s_res = await client_c3b.get(f"/api/workspace/sessions/{sid_c3b}")
            sess_c3b = s_res.json()
            c3b_arms[arm_strategy] = await ask_turn(
                client_c3b,
                sid_c3b,
                sess_c3b["scope_revision"],
                "C3b",
                f"strategy_{arm_strategy}",
                "Q06",
                q06_text,
            )
        finally:
            await client_c3b.aclose()

    # Sub-experiment in C3b: In unmodified _fuse(original, alternate, 12), DPO page:20 was recalled at Rank #6
    # of alternate_pool, but 2 empty-text structural image neighbors (Mixtral page:6:image:1 and page:2:image:1)
    # appeared in both original and alternate and pushed DPO page:20 to Rank #13 (1 slot outside pool_cap=12).
    # When _fuse excludes empty-text structural image objects so DPO page:20 enters candidate_pool[:12] at Rank #9,
    # compare bilingual (budget=4, truncates Rank #9) vs jev (Zev-1 scoring that exact 12-candidate text pool).
    from app.rag.workspace_retrieval import _fuse as orig_fuse

    def text_only_fuse(original: Any, alternate: Any, pool_cap: int) -> Any:
        orig_text = [c for c in original if c.text.strip()]
        alt_text = [c for c in alternate if c.text.strip()]
        return orig_fuse(orig_text, alt_text, pool_cap)

    for arm_strategy in ("bilingual", "jev"):
        app_c3b_t, _, _ = make_isolated_env(f"c3b_textpool_{arm_strategy}", strategy=arm_strategy)
        client_c3b_t, sid_c3b_t, _ = await setup_session_with_files(app_c3b_t)
        try:
            s_res = await client_c3b_t.get(f"/api/workspace/sessions/{sid_c3b_t}")
            sess_c3b_t = s_res.json()
            with patch("app.rag.workspace_retrieval._fuse", side_effect=text_only_fuse):
                c3b_arms[f"{arm_strategy}_text_pool"] = await ask_turn(
                    client_c3b_t,
                    sid_c3b_t,
                    sess_c3b_t["scope_revision"],
                    "C3b",
                    f"strategy_{arm_strategy}_text_pool",
                    "Q06",
                    q06_text,
                )
        finally:
            await client_c3b_t.aclose()

    bilingual_pool = [
        c["source"]
        for c in (
            (c3b_arms["bilingual"].get("trace") or {})
            .get("retrieval", {})
            .get("candidate_pool", [])
        )
    ]
    jev_pool = [
        c["source"]
        for c in (
            (c3b_arms["jev"].get("trace") or {}).get("retrieval", {}).get("candidate_pool", [])
        )
    ]
    bilingual_text_pool = [
        c["source"]
        for c in (
            (c3b_arms["bilingual_text_pool"].get("trace") or {})
            .get("retrieval", {})
            .get("candidate_pool", [])
        )
    ]
    jev_text_pool = [
        c["source"]
        for c in (
            (c3b_arms["jev_text_pool"].get("trace") or {})
            .get("retrieval", {})
            .get("candidate_pool", [])
        )
    ]
    results["C3b_Q06_cross_file_bilingual_vs_jev"] = {
        "target_question": "Q06",
        "target_units": ["E08", "E09"],
        "observed_first_breakpoint": "fts 下 E09 (DPO page:20) 在 Layer 2 漏召回；切到默认 bilingual 后 alternate_pool 虽在第 6 名召回 DPO page:20，但 _fuse 因保留 2 个无文字图片邻居将其挤到第 13 名（掉出 pool_cap=12）；剔除无文字图片邻居后 DPO page:20 进入 12 候选池（第 9 名），在 bilingual (budget=4) 下于 Layer 3 被截断，再由同池 Jev 排序评测",
        "single_factor_change": "在相同课程快照与 budget=4 下，依次对比 fts、默认 bilingual、默认 jev，以及剔除无文字图片占位后的同 12 文本候选池 bilingual_text_pool vs jev_text_pool (Zev-1)",
        "same_default_candidate_pool_verified": bilingual_pool == jev_pool and len(jev_pool) == 12,
        "same_text_candidate_pool_verified": (
            bilingual_text_pool == jev_text_pool and len(jev_text_pool) == 12
        ),
        "arms": c3b_arms,
    }

    # =========================================================================
    # Experiment C4 (Q07: E10, E11) - Natural phrasing misrouting vs SAG-style bounded tool selection
    # =========================================================================
    q07_text = questions_by_id["Q07"]["question"]
    app_c4, service_c4, settings_c4 = make_isolated_env("c4_q07", strategy="fts")
    client_c4, sid_c4, docs_c4 = await setup_session_with_files(app_c4)
    try:
        s_res = await client_c4.get(f"/api/workspace/sessions/{sid_c4}")
        sess_c4 = s_res.json()
        # Control: baseline (hijacked by asks_for_definition_and_experiment)
        c4_control = await ask_turn(
            client_c4, sid_c4, sess_c4["scope_revision"], "C4", "control_baseline", "Q07", q07_text
        )

        # C4a: Protocol step 1 ("先给定正确对象并验证同一工具能答对")
        # Pass the two target sources (Mixtral page:2 and Notebook v1 cell:0:output:0) to the existing
        # _generate_snapshot tool on the exact same session and snapshot.
        snap_c4 = service_c4.snapshots.read(SnapshotRequest((course_id,), ()))
        target_sources = [
            src
            for src in snap_c4.sources
            if (
                src.document_id == docs_c4["Mixtral-of-Experts-2401.04088v1.pdf"]
                and src.source_id == "page:2"
            )
            or (
                src.document_id == docs_c4["Mixtral-Router-Verification.ipynb"]
                and src.source_id == "cell:0:output:0"
            )
        ]
        before_c4a = len(provider_events)
        token = EXP_CTX.set({"exp_id": "C4", "arm": "C4a_given_target_objects", "question_id": "Q07"})
        started_c4a = perf_counter()
        try:
            # Run through normal HTTP turn with asks_for_definition_and_experiment disabled and searcher returning the two given target objects
            from app.rag.structure_search import CandidateReason, StructureCandidate

            given_candidates = [
                StructureCandidate(
                    source=_key(src),
                    course_id=src.course_id,
                    file_name=next(
                        d.file_name for d in snap_c4.documents if d.document_id == src.document_id
                    ),
                    kind=src.kind,
                    title=src.title,
                    text=src.text,
                    locator=src.locator,
                    state=src.state,
                    text_available=True,
                    match_kind="direct",
                    reasons=[CandidateReason(code="direct_location", seed=_key(src))],
                )
                for src in target_sources
            ]
            orig_search = service_c4.searcher.search

            def mock_given_search(req: Any, *, allowed_course_ids: Any) -> Any:
                base_res = orig_search(req, allowed_course_ids=allowed_course_ids)
                return base_res.model_copy(update={"status": "ok", "candidates": given_candidates})

            with (
                patch("app.rag.workspace.asks_for_definition_and_experiment", return_value=False),
                patch.object(service_c4.searcher, "search", side_effect=mock_given_search),
            ):
                c4a_given_objects = await ask_turn(
                    client_c4,
                    sid_c4,
                    sess_c4["scope_revision"],
                    "C4",
                    "C4a_given_target_objects",
                    "Q07",
                    q07_text,
                )
        finally:
            EXP_CTX.reset(token)

        # C4b: Protocol step 2 ("再只替换动作选择方式，试有步数上限的 SAG 式工具循环。保留相同工具、授权、模型和实读预算")
        # Bounded SAG-style tool selection loop (max_steps=3):
        # The LLM (deepseek-flash) selects actions from {search_course, finish_selection} using bilingual queries
        # over the exact same StructureSearcher, authorized snapshot, and final source budget (budget=4),
        # and then feeds the selected sources into the exact same _generate_snapshot (read_window + generate_response).
        before_c4b = len(provider_events)
        token = EXP_CTX.set({"exp_id": "C4", "arm": "C4b_sag_bounded_tool_loop", "question_id": "Q07"})
        started_c4b = perf_counter()
        sag_loop_trace: list[dict[str, Any]] = []
        try:
            ds_client = DeepSeekClient(settings_c4)
            collected_candidates: dict[tuple[str, int, str], Any] = {}
            catalog_summary = [
                {"file_name": d.file_name, "document_id": d.document_id, "format": d.document_format}
                for d in snap_c4.documents
            ]
            for step_no in range(1, 3):  # max 2 search steps before final read+generate
                tool_prompt = [
                    {
                        "role": "system",
                        "content": (
                            "You are a bounded course retrieval planner (SAG-style tool loop). "
                            "Given the user's Chinese question and available course files, output JSON "
                            '{"action":"search","queries":["english/bilingual query 1","query 2"]} '
                            "to search the FTS index for any missing evidence (such as theoretical formula definition in lecture PDF "
                            "and experimental verification output in Notebook), or "
                            '{"action":"answer"} if all required sources are already in collected_sources.'
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "question": q07_text,
                                "available_files": catalog_summary,
                                "collected_sources": [
                                    {
                                        "file_name": c.file_name,
                                        "source_id": c.source.source_id,
                                        "kind": c.kind,
                                        "preview": c.text[:160],
                                    }
                                    for c in collected_candidates.values()
                                ],
                                "step": step_no,
                                "max_steps": 2,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ]
                reply = await ds_client.generate_answer(
                    tool_prompt, response_format={"type": "json_object"}, temperature=0
                )
                decision = json.loads(reply["content"])
                step_record: dict[str, Any] = {"step": step_no, "decision": decision, "hits": []}
                if decision.get("action") == "answer" and collected_candidates:
                    sag_loop_trace.append(step_record)
                    break
                queries = decision.get("queries") or [q07_text]
                for q_str in queries[:2]:
                    sr = service_c4.searcher.search(
                        StructureSearchRequest(
                            query=str(q_str),
                            scope=StructureScope(course_ids=[course_id]),
                            top_k=6,
                        ),
                        allowed_course_ids=service_c4.allowed_courses,
                    )
                    for cand in sr.candidates:
                        if not cand.text.strip():
                            continue
                        ident = (cand.source.document_id, cand.source.version, cand.source.source_id)
                        step_record["hits"].append(
                            {"query": q_str, "file_name": cand.file_name, "source_id": cand.source.source_id}
                        )
                        if ident not in collected_candidates:
                            collected_candidates[ident] = cand
                sag_loop_trace.append(step_record)
                # Stop early if both PDF and Notebook sources are collected
                files_hit = {c.file_name for c in collected_candidates.values()}
                if (
                    "Mixtral-of-Experts-2401.04088v1.pdf" in files_hit
                    and "Mixtral-Router-Verification.ipynb" in files_hit
                ):
                    break

            # Select up to budget=4 sources preserving multi-file coverage (PDF + Notebook)
            ordered_cands = list(collected_candidates.values())
            nb_cands = [c for c in ordered_cands if c.file_name.endswith(".ipynb")]
            pdf_cands = [c for c in ordered_cands if c.file_name.endswith(".pdf")]
            sag_selected = (pdf_cands[:2] + nb_cands[:2])[:4]

            def mock_sag_search(req: Any, *, allowed_course_ids: Any) -> Any:
                base_res = orig_search(req, allowed_course_ids=allowed_course_ids)
                return base_res.model_copy(update={"status": "ok", "candidates": sag_selected})

            with (
                patch("app.rag.workspace.asks_for_definition_and_experiment", return_value=False),
                patch.object(service_c4.searcher, "search", side_effect=mock_sag_search),
            ):
                c4b_sag_result = await ask_turn(
                    client_c4,
                    sid_c4,
                    sess_c4["scope_revision"],
                    "C4",
                    "C4b_sag_bounded_tool_loop",
                    "Q07",
                    q07_text,
                )
            c4b_sag_result["sag_tool_loop_trace"] = sag_loop_trace
            c4b_sag_result["total_provider_calls_including_planner"] = provider_events[before_c4b:]
        finally:
            EXP_CTX.reset(token)

        results["C4_Q07_action_selection_and_sag_loop"] = {
            "target_question": "Q07",
            "target_units": ["E10", "E11"],
            "observed_first_breakpoint": "自然问法含“定义”与“实验”被 asks_for_definition_and_experiment 劫持入 SupplementRetriever(top_k=2) + 僵化正则，导致 E11 在 Layer 2 被挤出、E10 在 Layer 5 被拒答",
            "single_factor_change": "C4a 给定目标对象 Mixtral page:2 与 Notebook cell:0:output:0 验证现有生成工具能完整答对；C4b 仅替换动作选择为有步数上限 (max_steps=2) 的 SAG 式工具循环，保留相同检索器、授权快照、阅读预算与回答模型",
            "control_baseline": c4_control,
            "C4a_given_target_objects": c4a_given_objects,
            "C4b_sag_bounded_tool_loop": c4b_sag_result,
        }
    finally:
        await client_c4.aclose()

    # =========================================================================
    # Experiment C5 (Q09: E14, E15) - Cross-version natural phrasing comparison
    # =========================================================================
    app_c5, service_c5, _ = make_isolated_env("c5_q09", strategy="fts")
    client_c5, sid_c5, docs_c5 = await setup_session_with_files(app_c5)
    try:
        nb_id = docs_c5["Mixtral-Router-Verification.ipynb"]
        sess_c5 = await select_doc(client_c5, sid_c5, nb_id)
        # First run Q08a on v1 to establish historical turn citation in session history
        q08a_text = questions_by_id["Q08a"]["question"]
        await ask_turn(
            client_c5, sid_c5, sess_c5["scope_revision"], "C5", "setup_q08a_v1", "Q08a", q08a_text
        )
        # Update Notebook in-place to v2
        raw_v2 = (fixture_dir / "v2" / "Mixtral-Router-Verification.ipynb").read_bytes()
        imp_v2 = await client_c5.post(
            f"/api/workspace/sessions/{sid_c5}/import",
            params={
                "course_id": course_id,
                "filename": "Mixtral-Router-Verification.ipynb",
                "document_id": nb_id,
            },
            content=raw_v2,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert imp_v2.status_code == 200
        sess_c5 = await select_doc(client_c5, sid_c5, nb_id)

        # Verify historical v1 object & bytes are still retrievable from LocalLibraryStore
        v1_hist = service_c5.library.store.read_source(
            SourceKey(document_id=nb_id, version=1, source_id="cell:0:output:0"),
            course_ids=(course_id,),
        )
        v1_historical_check = {
            "status": v1_hist.status,
            "is_current": v1_hist.is_current,
            "content_sha256": v1_hist.content_sha256,
            "text_sha256": _sha256_text(v1_hist.source.text) if v1_hist.source else None,
            "text_preview": v1_hist.source.text[:200] if v1_hist.source else None,
        }

        q09_text = questions_by_id["Q09"]["question"]
        # Control: unmodified is_version_comparison misses "跟上一版相比..."
        c5_control = await ask_turn(
            client_c5, sid_c5, sess_c5["scope_revision"], "C5", "control_baseline", "Q09", q09_text
        )

        # Treatment: single-factor relaxation in is_version_comparison so _COMPARE + _OLD routes to compare_versions
        from app.rag.workspace_versions import _COMPARE, _FILE_TOKEN, _OLD

        def natural_is_version_comparison(text: str) -> bool:
            if orig_is_version_comparison(text):
                return True
            request_text = _FILE_TOKEN.sub(" ", text)
            return bool(_COMPARE.search(request_text) and _OLD.search(request_text))

        with patch(
            "app.rag.workspace.is_version_comparison",
            side_effect=natural_is_version_comparison,
        ):
            c5_treatment = await ask_turn(
                client_c5,
                sid_c5,
                sess_c5["scope_revision"],
                "C5",
                "treatment_route_to_compare_versions",
                "Q09",
                q09_text,
            )

        results["C5_Q09_cross_version_natural_comparison"] = {
            "target_question": "Q09",
            "target_units": ["E14", "E15"],
            "observed_first_breakpoint": "is_version_comparison 要求同时含旧版词+新版词或仅限 6 字短句“和旧版相比呢”，导致自然问句“跟上一版相比…”落入单版 _retrieve，未召回历史 v1 证据 E14",
            "single_factor_change": "核验 SQLite 历史库中 v1 cell:0:output:0 原件与字节仍完整可取，仅放宽 is_version_comparison 识别 _OLD + _COMPARE 进入 compare_versions",
            "v1_historical_store_verification": v1_historical_check,
            "control": c5_control,
            "treatment": c5_treatment,
        }
    finally:
        await client_c5.aclose()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject",
        default="d993748c01fabaff26586908de9bf72499534012",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("docs/G/G20-A-判据与标注.json"),
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path("docs/G/G20-fixtures"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/G/G20-C-配对实测.json"),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    annotations = json.loads((repo_root / args.annotations).read_text(encoding="utf-8"))
    fixture_dir = (repo_root / args.fixture_dir).resolve()
    env_path = (repo_root / args.env_file).resolve()

    with TemporaryDirectory(prefix="srag-g20-c-") as tmp_str:
        tmp = Path(tmp_str)
        checkout_dir = tmp / "checkout"
        resolved_sha = _extract_clean_checkout(repo_root, args.subject, checkout_dir)
        sys.path.insert(0, str(checkout_dir))

        provider_events: list[dict[str, Any]] = []
        restore = _install_provider_recorder(provider_events)
        try:
            experiments = asyncio.run(
                _run_all_paired_experiments(
                    tmp, env_path, annotations, fixture_dir, provider_events
                )
            )
        finally:
            restore()

        payload = {
            "task_id": "G20",
            "step": "C",
            "subject_commit": resolved_sha,
            "experiments": experiments,
            "provider_summary": {
                "total_calls": len(provider_events),
                "deepseek_calls": sum(1 for e in provider_events if e["provider"] == "deepseek"),
                "dashscope_calls": sum(1 for e in provider_events if e["provider"] == "dashscope"),
                "jev_calls": sum(1 for e in provider_events if e["provider"] == "jev"),
                "http_200_calls": sum(1 for e in provider_events if e.get("status_code") == 200),
                "events": provider_events,
            },
        }
        out_path = (repo_root / args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "subject_commit": resolved_sha,
                    "out": str(out_path),
                    "experiment_groups": list(experiments.keys()),
                    "provider_summary": {
                        k: v for k, v in payload["provider_summary"].items() if k != "events"
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
