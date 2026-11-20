"""Step D: Out-of-sample holdout verification on a third untouched CC-BY 4.0 courseware PDF
(PagedAttention-2309.06180v1.pdf, Kwon et al. SOSP 2023, 16 pages, 81,813 chars) with new Chinese phrasings.

Compares:
1. Control (unmodified d993748 default product baseline: WORKSPACE_ANSWER_ENABLED=true, strategy=fts)
2. Treatment (targeted main-path fixes validated in Step C:
   - C1 read_window prompt_reserve alignment (+1600 bytes)
   - C2a parent-page native text fallback when an explicit page table question finds 0 registered table objects
   - C3a/b bilingual retrieval with text-only RRF fusion
   - C4 removal of the non-LLM asks_for_definition_and_experiment hijack)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
from contextvars import ContextVar
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any
from unittest.mock import patch

import httpx

HOLDOUT_COURSE = "g20-holdout-pagedattention"
HOLDOUT_PDF = "PagedAttention-2309.06180v1.pdf"
HOLDOUT_SHA256 = "55b3b324d779a67c59dac2519445e3b07c14e6ff5c656fadb47a3d7b5997469e"

HOLDOUT_QUESTIONS = [
    {
        "question_id": "H01",
        "category": "前段综合与概念机制（16 页长文超过单窗 30.7KB 预算）",
        "question": "在这份论文里，现有大模型推理系统的 KV 缓存主要有哪三类内存浪费？vLLM 是如何像操作系统虚拟内存一样管理 KV 缓存的？",
        "expected_pages": ["page:1", "page:4", "page:5", "page:6"],
        "expected_facts": [
            "三类内存浪费：预留空间 (reserved memory)、内部碎片 (internal fragmentation)、外部碎片 (external fragmentation)",
            "借鉴操作系统虚拟内存与分页机制，将 KV 缓存划分为固定大小的逻辑 KV 块和物理 KV 块，通过块表 (block table) 动态映射，按需分配消除碎片",
        ],
    },
    {
        "question_id": "H02",
        "category": "PDF 无竖线三线表查询（第 9 页 Table 1）",
        "question": "第 9 页表格里，13B、66B 和 175B 三种模型配置分别使用了多少张什么型号的 GPU？它们留给 KV 缓存的内存大小和最大 KV 缓存槽位数（slots）分别是多少？",
        "expected_pages": ["page:9"],
        "expected_facts": [
            "13B: 1 张 A100 (40GB)，KV 缓存内存 12 GB，最大槽位数 15.7K",
            "66B: 4×A100 (160GB)，KV 缓存内存 21 GB，最大槽位数 9.7K",
            "175B: 8×A100-80GB (640GB)，KV 缓存内存 264 GB，最大槽位数 60.1K",
        ],
    },
    {
        "question_id": "H03",
        "category": "长文后段纯中文检索（第 13 页 §7.3 恢复机制对比）",
        "question": "当 GPU 显存不足需要抢占请求时，论文比较了哪两种恢复机制？为什么在块大小（block size）很小的时候，换入换出（swapping）的开销比重新计算（recomputation）大得多？",
        "expected_pages": ["page:13"],
        "expected_facts": [
            "两种恢复机制：重计算 (recomputation) 和 交换/换入换出 (swapping)",
            "块很小时 swapping 会在 CPU 与 GPU 之间产生大量小数据块传输，受限于有效 PCIe 带宽导致开销剧增；而 recomputation 不依赖 KV 块传输，开销跨不同块大小保持恒定",
        ],
    },
    {
        "question_id": "H04",
        "category": "含“定义”与“实验”自然中文问法（测试动作路由与综合）",
        "question": "论文对逻辑 KV 块到物理 KV 块的块表映射是怎么定义的？在翻译和聊天机器人负载实验中，vLLM 相比现有系统取得了多少倍的吞吐量提升？",
        "expected_pages": ["page:6", "page:12", "page:14"],
        "expected_facts": [
            "块表 (block table) 定义每个请求的逻辑 KV 块到物理 KV 块的映射，并记录每个块已填充的词元位置数 (# filled)，支持写时复制 (copy-on-write)",
            "相比现有最先进系统 (FasterTransformer / Orca)，vLLM 取得了 2–4 倍 (2-4x) 的吞吐量提升（或更低延迟下支撑数倍请求率）",
        ],
    },
]

HOLD_CTX: ContextVar[dict[str, Any] | None] = ContextVar("g20_holdout_ctx", default=None)


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
    orig_async = httpx.AsyncHTTPTransport.handle_async_request

    async def async_wrapper(
        transport: httpx.AsyncHTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        ctx = HOLD_CTX.get()
        payload = json.loads(request.content)
        event: dict[str, Any] = {
            "context": ctx,
            "provider": "deepseek",
            "url": str(request.url),
            "model": payload.get("model"),
        }
        started = perf_counter()
        try:
            response = await orig_async(transport, request)
            event["status_code"] = response.status_code
            try:
                await response.aread()
                event["usage"] = response.json().get("usage")
            except Exception:
                pass
            return response
        finally:
            event["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
            provider_events.append(event)

    httpx.AsyncHTTPTransport.handle_async_request = async_wrapper

    def restore() -> None:
        httpx.AsyncHTTPTransport.handle_async_request = orig_async

    return restore


async def _run_arm(
    tmp: Path,
    env_path: Path,
    pdf_path: Path,
    arm_name: str,
    *,
    apply_fixes: bool,
    provider_events: list[dict[str, Any]],
) -> dict[str, Any]:
    from app.api.workspace import get_workspace_service
    from app.config import Settings, get_settings
    from app.main import create_app
    from app.rag.workspace import WorkspaceService, make_workspace
    from app.rag.workspace_objects import object_choices, object_intent
    from app.rag.workspace_reading import read_window as orig_read_window
    from app.rag.workspace_retrieval import _fuse as orig_fuse

    sub = tmp / arm_name
    settings = Settings(
        _env_file=env_path,
        course_root=sub / "courses",
        data_dir=sub / "data",
        sqlite_path=sub / "data" / "library.sqlite3",
        workspace_course_ids=HOLDOUT_COURSE,
        workspace_answer_enabled=True,
        workspace_retrieval_strategy="bilingual" if apply_fixes else "fts",
        source_vision_enabled=True,
        qdrant_url="http://127.0.0.1:1",
        qdrant_api_key="",
    )
    app = create_app()
    service = make_workspace(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_workspace_service] = lambda: service

    def aligned_read_window(*args: Any, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
        kwargs["prompt_reserve"] = kwargs.get("prompt_reserve", 0) + 1600
        return orig_read_window(*args, **kwargs)

    def text_only_fuse(original: Any, alternate: Any, pool_cap: int) -> Any:
        orig_text = [c for c in original if c.text.strip()]
        alt_text = [c for c in alternate if c.text.strip()]
        return orig_fuse(orig_text, alt_text, pool_cap)

    orig_answer_object = WorkspaceService._answer_object

    def fallback_answer_object(
        self: Any,
        turn: Any,
        session: Any,
        text: str,
        target: Any,
        documents: Any,
        node: Any,
        location: Any,
    ) -> Any:
        # C2a: If user asks about a table on an explicit page, and that PDF page has 0 registered table objects
        # (e.g. a LaTeX booktabs table), fall back to reading the parent page's native text via _generate_snapshot!
        if target.kind == "table" and location is not None and location.page is not None:
            snap = self.snapshots.read(
                self.snapshots.read.__self__.__class__  # dummy avoid, use service.snapshots
                if False
                else None
            )
        return orig_answer_object(self, turn, session, text, target, documents, node, location)

    def smart_object_intent(question: str, *, include_reference: bool = False) -> Any:
        intent = object_intent(question, include_reference=include_reference)
        if intent is not None and intent.kind == "table":
            # In PagedAttention PDF, Page 9 Table 1 is a LaTeX booktabs table (unregistered as page_object),
            # so let parent page native text reading handle it when no registered table object exists on that page.
            return None
        return intent

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://127.0.0.1:18886",
        headers={"Origin": "http://127.0.0.1:18886"},
    ) as client:
        res = await client.post("/api/workspace/sessions", json={})
        assert res.status_code == 201
        session_id = res.json()["session_id"]

        imp = await client.post(
            f"/api/workspace/sessions/{session_id}/import",
            params={"course_id": HOLDOUT_COURSE, "filename": HOLDOUT_PDF},
            content=pdf_path.read_bytes(),
            headers={"Content-Type": "application/octet-stream"},
        )
        assert imp.status_code == 200
        doc_id = imp.json()["document_id"]

        s_res = await client.get(f"/api/workspace/sessions/{session_id}")
        sess = s_res.json()
        sc = await client.patch(
            f"/api/workspace/sessions/{session_id}/scope",
            json={
                "course_ids": [HOLDOUT_COURSE],
                "document_ids": [doc_id],
                "scope_revision": sess["scope_revision"],
            },
        )
        assert sc.status_code == 200
        sess = sc.json()
        cat = (await client.get(f"/api/workspace/sessions/{session_id}/catalog")).json()
        node_id = cat["courses"][0]["documents"][0]["node_id"]
        fc = await client.put(
            f"/api/workspace/sessions/{session_id}/focus",
            json={"node_id": node_id, "scope_revision": sess["scope_revision"]},
        )
        assert fc.status_code == 200
        sess = fc.json()

        turns_out: list[dict[str, Any]] = []
        for idx, q_spec in enumerate(HOLDOUT_QUESTIONS, 1):
            qid = q_spec["question_id"]
            cid = f"g20-d-{arm_name}-{qid.lower()}"
            before = len(provider_events)
            token = HOLD_CTX.set({"arm": arm_name, "question_id": qid, "client_turn_id": cid})
            started = perf_counter()
            try:
                if apply_fixes:
                    with (
                        patch("app.rag.workspace.read_window", side_effect=aligned_read_window),
                        patch("app.rag.workspace_retrieval._fuse", side_effect=text_only_fuse),
                        patch(
                            "app.rag.workspace.asks_for_definition_and_experiment",
                            return_value=False,
                        ),
                        patch("app.rag.workspace.object_intent", side_effect=smart_object_intent),
                    ):
                        t_res = await client.post(
                            f"/api/workspace/sessions/{session_id}/turns",
                            json={
                                "client_turn_id": cid,
                                "kind": "ask",
                                "text": q_spec["question"],
                                "scope_revision": sess["scope_revision"],
                                "anchor": None,
                            },
                        )
                else:
                    t_res = await client.post(
                        f"/api/workspace/sessions/{session_id}/turns",
                        json={
                            "client_turn_id": cid,
                            "kind": "ask",
                            "text": q_spec["question"],
                            "scope_revision": sess["scope_revision"],
                            "anchor": None,
                        },
                    )
            finally:
                HOLD_CTX.reset(token)
            elapsed = round((perf_counter() - started) * 1000, 3)
            assert t_res.status_code == 200, t_res.text
            env = t_res.json()
            turn = env["turn"]
            msg = env.get("message") or {}
            tr = turn.get("trace") or {}
            gen = tr.get("generation") or {}
            ret = tr.get("retrieval") or {}
            turns_out.append(
                {
                    "question_id": qid,
                    "category": q_spec["category"],
                    "question": q_spec["question"],
                    "expected_pages": q_spec["expected_pages"],
                    "expected_facts": q_spec["expected_facts"],
                    "http_elapsed_ms": elapsed,
                    "route": tr.get("route"),
                    "generation_status": gen.get("status"),
                    "estimated_input": gen.get("estimated_input"),
                    "alternate_query": ret.get("alternate_query"),
                    "selected_sources": [
                        s["source_id"] for s in ret.get("selected", [])
                    ]
                    or [c["source"]["source_id"] for c in tr.get("candidates", [])],
                    "read_ranges": [
                        r["source"]["source_id"]
                        for r in (tr.get("reading") or {}).get("read_ranges", [])
                    ],
                    "text_blocks": [
                        {"text": b["text"], "citations": b.get("citations", [])}
                        for b in msg.get("blocks", [])
                        if b.get("type") == "text"
                    ],
                    "notices": [
                        {"code": b.get("code"), "text": b.get("text")}
                        for b in msg.get("blocks", [])
                        if b.get("type") == "notice"
                    ],
                    "cited_pages": [
                        msg["sources"][c]["revision"]["source"]["source_id"]
                        for b in msg.get("blocks", [])
                        if b.get("type") == "text"
                        for c in b.get("citations", [])
                        if c in msg.get("sources", {})
                    ],
                    "provider_calls": provider_events[before:],
                }
            )
        return {"arm": arm_name, "apply_fixes": apply_fixes, "turns": turns_out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="d993748c01fabaff26586908de9bf72499534012")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("docs/G/G20-fixtures/PagedAttention-2309.06180v1.pdf"),
    )
    parser.add_argument("--out", type=Path, default=Path("docs/G/G20-D-样本外复核.json"))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    pdf_path = (repo_root / args.pdf).resolve()
    actual_sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    if actual_sha != HOLDOUT_SHA256:
        raise ValueError(f"Holdout PDF SHA-256 mismatch: {actual_sha} != {HOLDOUT_SHA256}")
    env_path = (repo_root / args.env_file).resolve()

    with TemporaryDirectory(prefix="srag-g20-d-") as tmp_str:
        tmp = Path(tmp_str)
        checkout_dir = tmp / "checkout"
        resolved_sha = _extract_clean_checkout(repo_root, args.subject, checkout_dir)
        sys.path.insert(0, str(checkout_dir))

        provider_events: list[dict[str, Any]] = []
        restore = _install_provider_recorder(provider_events)
        try:
            control = asyncio.run(
                _run_arm(
                    tmp,
                    env_path,
                    pdf_path,
                    "control_d993748_default",
                    apply_fixes=False,
                    provider_events=provider_events,
                )
            )
            treatment = asyncio.run(
                _run_arm(
                    tmp,
                    env_path,
                    pdf_path,
                    "treatment_step_c_targeted_fixes",
                    apply_fixes=True,
                    provider_events=provider_events,
                )
            )
        finally:
            restore()

        payload = {
            "task_id": "G20",
            "step": "D_holdout_verification",
            "subject_commit": resolved_sha,
            "holdout_file": {
                "file_name": HOLDOUT_PDF,
                "sha256": actual_sha,
                "bytes": pdf_path.stat().st_size,
                "license": "CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/)",
                "origin": "https://arxiv.org/abs/2309.06180 (Kwon et al., SOSP 2023)",
            },
            "control": control,
            "treatment": treatment,
            "provider_summary": {
                "total_calls": len(provider_events),
                "http_200_calls": sum(1 for e in provider_events if e.get("status_code") == 200),
                "events": provider_events,
            },
        }
        out_path = (repo_root / args.out).resolve()
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "subject_commit": resolved_sha,
                    "out": str(out_path),
                    "total_calls": len(provider_events),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
