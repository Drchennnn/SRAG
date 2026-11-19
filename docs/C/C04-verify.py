#!/usr/bin/env python3
"""C04 public, offline, same-budget structure-search review probe."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import sqlite3
import statistics
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import httpx

REVISION = "ccb890e7f0b459e782e5d51491f2074cbf0653bf"
TREE = "454c2f24cdea427f297822233feb83abda055ee1"
EXPECTED_HASHES = {
    "app/rag/structure_search.py": (
        "2a23752195f25605eff484556519d8dce7b14bb0270cda6a373998904d9153c8"
    ),
    "app/stores/structure_snapshot.py": (
        "0662a65cf7ddd9939cea74246a0b9ce2acad43399b45cf96fb9dedfde992d965"
    ),
    "app/api/structure_search.py": (
        "c9324a440e73cbce27eaea7205db00fd768bbefc8e72c3283007b60bc9ff19a4"
    ),
    "app/main.py": "310aa76c29fcbd0411cd89f4a7b83cd81f8f714ef19b696102d00f092306eac3",
}
PUBLIC_SAMPLE_HASHES = {
    "Signal-reference.pdf": ("c8f123efffba2ce032c10243996fa3139a368bc68a38bced6c5e1a81e51e0cf3"),
    "Signal-slides.pptx": ("3d6adae71debca0a67e738d509369e1f916be615a6c44696cbb526140a8debfb"),
    "Signal-lab.ipynb": ("8c4808a38493583f6637b80cd0b30ab0d1a4dba933bb9b4306da7e5933f0f2d6"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _blocked_network() -> Any:
    attempts: list[str] = []
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create = socket.create_connection

    def blocked_connect(_socket: socket.socket, address: Any) -> None:
        attempts.append(f"connect:{address!r}")
        raise AssertionError(f"C04 forbids external socket connect: {address!r}")

    def blocked_connect_ex(_socket: socket.socket, address: Any) -> int:
        attempts.append(f"connect_ex:{address!r}")
        raise AssertionError(f"C04 forbids external socket connect_ex: {address!r}")

    def blocked_create(address: Any, *args: Any, **kwargs: Any) -> None:
        attempts.append(f"create_connection:{address!r}")
        raise AssertionError(f"C04 forbids external create_connection: {address!r}")

    socket.socket.connect = blocked_connect
    socket.socket.connect_ex = blocked_connect_ex
    socket.create_connection = blocked_create
    try:
        yield attempts
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
        socket.create_connection = original_create


def _source_key(result: Any, source_id: str) -> str:
    return f"{result.document_id}@v{result.version}:{source_id}"


def _response_key(source: dict[str, Any]) -> str:
    return f"{source['document_id']}@v{source['version']}:{source['source_id']}"


def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    locator = candidate["locator"]
    kept_locator = {
        key: locator[key]
        for key in (
            "page",
            "cell_index",
            "cell_id",
            "output_index",
            "object_type",
            "shape_id_path",
        )
        if key in locator
    }
    return {
        "key": _response_key(candidate["source"]),
        "source": candidate["source"],
        "course_id": candidate["course_id"],
        "file_name": candidate["file_name"],
        "kind": candidate["kind"],
        "locator": kept_locator,
        "state": candidate["state"],
        "text_available": candidate["text_available"],
        "match_kind": candidate["match_kind"],
        "location_text_matched": candidate["location_text_matched"],
        "fts_origins": candidate["fts_origins"],
        "reasons": [
            {
                "code": reason["code"],
                "seed": _response_key(reason["seed"]),
            }
            for reason in candidate["reasons"]
        ],
    }


def _compact_response(payload: dict[str, Any], elapsed_ms: float) -> dict[str, Any]:
    return {
        "status": payload["status"],
        "elapsed_ms": round(elapsed_ms, 3),
        "snapshot": payload["snapshot"],
        "scope_trace": payload["scope_trace"],
        "query_trace": payload["query_trace"],
        "options": payload["options"],
        "candidates": [_compact_candidate(item) for item in payload["candidates"]],
        "coverage": payload["coverage"],
        "misses": payload["misses"],
    }


def _collapse_runs(responses: list[dict[str, Any]]) -> dict[str, Any]:
    assert responses
    stable = []
    for response in responses:
        item = json.loads(json.dumps(response, ensure_ascii=False))
        item.pop("elapsed_ms")
        stable.append(item)
    assert all(item == stable[0] for item in stable[1:])
    samples = [response["elapsed_ms"] for response in responses]
    result = stable[0]
    result["elapsed_ms"] = round(statistics.median(samples), 3)
    result["elapsed_samples_ms"] = samples
    return result


def _candidate_keys(response: dict[str, Any]) -> list[str]:
    return [item["key"] for item in response["candidates"]]


def _case_evaluation(
    *,
    spec: dict[str, Any],
    lexical: dict[str, Any],
    structured: dict[str, Any],
) -> dict[str, Any]:
    results = {"fts": lexical, "structure": structured}
    evaluation: dict[str, Any] = {
        "snapshot_equal": lexical["snapshot"] == structured["snapshot"],
        "scope_equal": lexical["scope_trace"] == structured["scope_trace"],
        "same_fts_cap": (
            lexical["coverage"]["fts_candidate_cap"]
            == structured["coverage"]["fts_candidate_cap"]
            == min(spec["top_k"] * 3, 150)
        ),
        "same_fts_seed_count": (
            lexical["coverage"]["fts_seed_count"] == structured["coverage"]["fts_seed_count"]
        ),
        "within_budget": {
            name: len(response["candidates"]) <= spec["top_k"] for name, response in results.items()
        },
        "strategies": {},
    }
    for name, response in results.items():
        keys = set(_candidate_keys(response))
        required = set(spec.get("required", {}).get(name, []))
        context = set(spec.get("context", []))
        forbidden = set(spec.get("forbidden", {}).get(name, []))
        evaluation["strategies"][name] = {
            "expected_status": spec["expected_status"][name],
            "status_ok": response["status"] == spec["expected_status"][name],
            "required_hits": sorted(required & keys),
            "required_misses": sorted(required - keys),
            "context_hits": sorted(context & keys),
            "context_misses": sorted(context - keys),
            "forbidden_hits": sorted(forbidden & keys),
        }
    fts_keys = set(_candidate_keys(lexical))
    structure_keys = set(_candidate_keys(structured))
    evaluation["structure_added"] = sorted(structure_keys - fts_keys)
    evaluation["structure_removed"] = sorted(fts_keys - structure_keys)
    evaluation["structure_context_gain"] = len(
        set(spec.get("context", [])) & (structure_keys - fts_keys)
    )
    return evaluation


def _assert_case(record: dict[str, Any]) -> None:
    evaluation = record["evaluation"]
    assert evaluation["snapshot_equal"], record["id"]
    assert evaluation["scope_equal"], record["id"]
    assert evaluation["same_fts_cap"], record["id"]
    assert evaluation["same_fts_seed_count"], record["id"]
    assert all(evaluation["within_budget"].values()), record["id"]
    for strategy in ("fts", "structure"):
        result = evaluation["strategies"][strategy]
        assert result["status_ok"], (record["id"], strategy, result)
        assert not result["required_misses"], (record["id"], strategy, result)
        assert not result["forbidden_hits"], (record["id"], strategy, result)
    if "location" not in record["request"]:
        assert record["fts"]["coverage"]["full_file_read"] is False, record["id"]
        assert record["structure"]["coverage"]["full_file_read"] is False, record["id"]


async def _run(repo_root: Path, output: Path) -> dict[str, Any]:
    code_hashes = {relative: _sha256(repo_root / relative) for relative in EXPECTED_HASHES}
    assert code_hashes == EXPECTED_HASHES, code_hashes
    sample_root = repo_root / "docs" / "A" / "样例"
    samples = {
        name: {
            "sha256": _sha256(sample_root / name),
            "size_bytes": (sample_root / name).stat().st_size,
        }
        for name in PUBLIC_SAMPLE_HASHES
    }
    assert {name: item["sha256"] for name, item in samples.items()} == PUBLIC_SAMPLE_HASHES

    with TemporaryDirectory(prefix="srag-c04-") as temporary:
        temp_root = Path(temporary)
        course_root = temp_root / "courses"
        course_root.mkdir()
        os.environ.update(
            {
                "COURSE_ROOT": str(course_root),
                "DATA_DIR": str(temp_root / "data"),
                "SQLITE_PATH": str(temp_root / "library.sqlite3"),
                "DASHSCOPE_API_KEY": "",
                "DEEPSEEK_API_KEY": "",
                "QDRANT_API_KEY": "",
                "QDRANT_URL": "http://127.0.0.1:1",
                "SRAG_STRUCTURE_COURSES": "public,synthetic,course-a,course-b",
            }
        )

        setup_guard = _blocked_network()
        setup_attempts = setup_guard.__enter__()

        from app.config import get_settings

        get_settings.cache_clear()
        import app.rag.structure_search as structure_search_module
        from app.main import create_app
        from app.rag.local_library import LocalLibrary
        from app.schemas.local import ParsedSource, SourceDraft

        module_path = Path(structure_search_module.__file__).resolve()
        assert module_path.is_relative_to(repo_root), (module_path, repo_root)
        library = LocalLibrary(get_settings())

        print("[C04] 导入三份真实公开样例")
        copied = []
        for name in PUBLIC_SAMPLE_HASHES:
            target = course_root / "public" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sample_root / name, target)
            copied.append(target)
        public_results = library.import_paths("public", copied)
        public_by_name = {Path(item.source_path).name: item for item in public_results}
        assert set(public_by_name) == set(PUBLIC_SAMPLE_HASHES)

        def page(
            number: int,
            text: str,
            *,
            ordinal: int,
            state: str = "available",
        ) -> Any:
            return SourceDraft(
                source_id=f"page:{number}",
                kind="page",
                ordinal=ordinal,
                title=f"page {number}",
                text=text,
                locator={"format": "pdf", "page": number},
                state=state,
            )

        def cell(index: int, text: str, *, cell_id: str | None = None) -> Any:
            locator: dict[str, Any] = {"format": "ipynb", "cell_index": index}
            if cell_id is not None:
                locator["cell_id"] = cell_id
            return SourceDraft(
                source_id=f"cell:{index}",
                kind="cell",
                ordinal=index,
                title=f"cell {index}",
                text=text,
                locator=locator,
            )

        def publish(
            course_id: str,
            relative_path: str,
            sources: list[Any],
        ) -> Any:
            path = course_root / course_id / relative_path
            raw = json.dumps(
                [item.model_dump(mode="json") for item in sources],
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
            plan = library.store.prepare(course_id, path)
            return library.store.publish(
                plan,
                ParsedSource(title=Path(relative_path).stem, sources=sources),
                original=raw,
                content_sha256=hashlib.sha256(raw).hexdigest(),
                document_format=Path(relative_path).suffix.lstrip(".") or "ipynb",
            )

        print("[C04] 发布可公开的隔离故障与边界样例")
        bilingual_en = publish(
            "synthetic",
            "english.ipynb",
            [cell(0, "ridge regression controls coefficient size")],
        )
        bilingual_zh = publish("synthetic", "chinese.ipynb", [cell(0, "岭回归")])
        code_doc = publish(
            "synthetic",
            "code.ipynb",
            [
                cell(0, "fast Fourier transform uses np.linalg.solve(x, y)"),
                cell(1, "np and linalg and solve are separate tokens"),
            ],
        )
        budget_doc = publish(
            "synthetic",
            "budget.pdf",
            [
                page(1, "", ordinal=0),
                SourceDraft(
                    source_id="page:1:shape:1",
                    kind="page_object",
                    ordinal=1,
                    parent_id="page:1",
                    text="budgetword",
                    locator={"format": "pdf", "page": 1, "object_type": "text"},
                ),
                page(2, "budgetword", ordinal=2),
                page(3, "budgetword", ordinal=3),
                page(4, "budgetword", ordinal=4),
            ],
        )
        caption_doc = publish(
            "synthetic",
            "caption.pdf",
            [
                page(1, "", ordinal=0),
                SourceDraft(
                    source_id="page:1:caption:1",
                    kind="page_object",
                    ordinal=1,
                    parent_id="page:1",
                    text="diagramtoken",
                    locator={
                        "format": "pdf",
                        "page": 1,
                        "object_type": "caption",
                        "caption_target_source_id": "page:1:image:1",
                    },
                ),
                SourceDraft(
                    source_id="page:1:image:1",
                    kind="page_object",
                    ordinal=2,
                    parent_id="page:1",
                    text="",
                    locator={"format": "pdf", "page": 1, "object_type": "image"},
                ),
            ],
        )
        scan_doc = publish(
            "synthetic",
            "scan.pdf",
            [
                page(1, "", ordinal=0, state="needs_ocr"),
                page(2, "", ordinal=1, state="needs_ocr"),
            ],
        )
        empty_doc = publish(
            "synthetic",
            "empty.pdf",
            [page(1, "unrelated native words", ordinal=0)],
        )
        course_a = publish("course-a", "unit/common.ipynb", [cell(0, "sharedword alpha")])
        course_b = publish("course-b", "other/common.ipynb", [cell(0, "sharedword beta")])
        app = create_app()
        setup_guard.__exit__(None, None, None)

        pdf = public_by_name["Signal-reference.pdf"]
        ppt = public_by_name["Signal-slides.pptx"]
        notebook = public_by_name["Signal-lab.ipynb"]
        cases: list[dict[str, Any]] = [
            {
                "id": "real_pdf_lookahead",
                "name_zh": "真实 PDF 英文原词",
                "request": {
                    "query": "lookahead",
                    "scope": {"course_ids": ["public"], "document_ids": [pdf.document_id]},
                },
                "top_k": 3,
                "expected_status": {"fts": "ok", "structure": "ok"},
                "required": {
                    "fts": [_source_key(pdf, "page:1")],
                    "structure": [_source_key(pdf, "page:1")],
                },
            },
            {
                "id": "real_ppt_peak",
                "name_zh": "真实 PPTX 表格词",
                "request": {
                    "query": "Peak",
                    "scope": {"course_ids": ["public"], "document_ids": [ppt.document_id]},
                },
                "top_k": 3,
                "expected_status": {"fts": "ok", "structure": "ok"},
                "required": {
                    "fts": [_source_key(ppt, "page:1:shape:3")],
                    "structure": [_source_key(ppt, "page:1:shape:3")],
                },
                "context": [_source_key(ppt, "page:1")],
            },
            {
                "id": "real_notebook_identifier",
                "name_zh": "真实 Notebook 代码标识",
                "request": {
                    "query": "NotebookImage",
                    "scope": {
                        "course_ids": ["public"],
                        "document_ids": [notebook.document_id],
                    },
                },
                "top_k": 4,
                "expected_status": {"fts": "ok", "structure": "ok"},
                "required": {
                    "fts": [_source_key(notebook, "cell:1")],
                    "structure": [_source_key(notebook, "cell:1")],
                },
                "context": [
                    _source_key(notebook, "cell:1:output:0"),
                    _source_key(notebook, "cell:1:output:1"),
                    _source_key(notebook, "cell:1:output:2"),
                ],
            },
            {
                "id": "bilingual_zh_to_en",
                "name_zh": "中文查询英文单语资料",
                "request": {
                    "query": "岭回归",
                    "scope": {
                        "course_ids": ["synthetic"],
                        "document_ids": [bilingual_en.document_id],
                    },
                },
                "top_k": 3,
                "expected_status": {"fts": "ok", "structure": "ok"},
                "required": {
                    "fts": [_source_key(bilingual_en, "cell:0")],
                    "structure": [_source_key(bilingual_en, "cell:0")],
                },
            },
            {
                "id": "bilingual_en_to_zh",
                "name_zh": "英文查询中文单语资料",
                "request": {
                    "query": "ridge regression",
                    "scope": {
                        "course_ids": ["synthetic"],
                        "document_ids": [bilingual_zh.document_id],
                    },
                },
                "top_k": 3,
                "expected_status": {"fts": "ok", "structure": "ok"},
                "required": {
                    "fts": [_source_key(bilingual_zh, "cell:0")],
                    "structure": [_source_key(bilingual_zh, "cell:0")],
                },
            },
            {
                "id": "abbreviation_fft",
                "name_zh": "缩写查询英文全称",
                "request": {
                    "query": "FFT",
                    "scope": {
                        "course_ids": ["synthetic"],
                        "document_ids": [code_doc.document_id],
                    },
                },
                "top_k": 3,
                "expected_status": {"fts": "ok", "structure": "ok"},
                "required": {
                    "fts": [_source_key(code_doc, "cell:0")],
                    "structure": [_source_key(code_doc, "cell:0")],
                },
            },
            {
                "id": "dotted_code_exact",
                "name_zh": "点号代码精确复核",
                "request": {
                    "query": "np.linalg.solve",
                    "scope": {
                        "course_ids": ["synthetic"],
                        "document_ids": [code_doc.document_id],
                    },
                },
                "top_k": 3,
                "expected_status": {"fts": "ok", "structure": "ok"},
                "required": {
                    "fts": [_source_key(code_doc, "cell:0")],
                    "structure": [_source_key(code_doc, "cell:0")],
                },
            },
            {
                "id": "dotted_code_k1_limit",
                "name_zh": "点号代码 K=1 小预算漏检观察",
                "request": {
                    "query": "np.linalg.solve",
                    "scope": {
                        "course_ids": ["synthetic"],
                        "document_ids": [code_doc.document_id],
                    },
                },
                "top_k": 1,
                "expected_status": {"fts": "ok", "structure": "ok"},
                "required": {"fts": [], "structure": []},
                "context": [_source_key(code_doc, "cell:0")],
            },
            {
                "id": "negated_fft",
                "name_zh": "中文否定问句保留相关证据",
                "request": {
                    "query": "哪一页不是在讲 FFT",
                    "scope": {
                        "course_ids": ["synthetic"],
                        "document_ids": [code_doc.document_id],
                    },
                },
                "top_k": 3,
                "expected_status": {"fts": "ok", "structure": "ok"},
                "required": {
                    "fts": [_source_key(code_doc, "cell:0")],
                    "structure": [_source_key(code_doc, "cell:0")],
                },
            },
            {
                "id": "explicit_ppt_page",
                "name_zh": "显式 PPT 页与页内对象",
                "request": {
                    "query": "",
                    "scope": {"course_ids": ["public"], "document_ids": [ppt.document_id]},
                    "location": {"page": 1},
                },
                "top_k": 10,
                "expected_status": {"fts": "ok", "structure": "ok"},
                "required": {
                    "fts": [_source_key(ppt, "page:1")],
                    "structure": [_source_key(ppt, "page:1")],
                },
                "context": [_source_key(ppt, "page:1:shape:3")],
            },
            {
                "id": "explicit_notebook_cell",
                "name_zh": "显式 Cell 与同 Cell 多输出",
                "request": {
                    "query": "",
                    "scope": {
                        "course_ids": ["public"],
                        "document_ids": [notebook.document_id],
                    },
                    "location": {"cell_id": "signal-plots"},
                },
                "top_k": 10,
                "expected_status": {"fts": "ok", "structure": "ok"},
                "required": {
                    "fts": [_source_key(notebook, "cell:1")],
                    "structure": [_source_key(notebook, "cell:1")],
                },
                "context": [
                    _source_key(notebook, "cell:1:output:0"),
                    _source_key(notebook, "cell:1:output:1"),
                    _source_key(notebook, "cell:1:output:2"),
                ],
            },
            {
                "id": "cross_course_ambiguous_name",
                "name_zh": "跨课程同名歧义",
                "request": {
                    "query": "sharedword",
                    "scope": {
                        "course_ids": ["course-a", "course-b"],
                        "file_names": ["common.ipynb"],
                    },
                },
                "top_k": 3,
                "expected_status": {
                    "fts": "ambiguous_scope",
                    "structure": "ambiguous_scope",
                },
                "required": {"fts": [], "structure": []},
            },
            {
                "id": "cross_course_isolated",
                "name_zh": "指定课程后的同名隔离",
                "request": {
                    "query": "sharedword",
                    "scope": {"course_ids": ["course-a"], "file_names": ["common.ipynb"]},
                },
                "top_k": 3,
                "expected_status": {"fts": "ok", "structure": "ok"},
                "required": {
                    "fts": [_source_key(course_a, "cell:0")],
                    "structure": [_source_key(course_a, "cell:0")],
                },
                "forbidden": {
                    "fts": [_source_key(course_b, "cell:0")],
                    "structure": [_source_key(course_b, "cell:0")],
                },
            },
            {
                "id": "registered_caption_target",
                "name_zh": "登记图注引入无文字图像",
                "request": {
                    "query": "diagramtoken",
                    "scope": {
                        "course_ids": ["synthetic"],
                        "document_ids": [caption_doc.document_id],
                    },
                },
                "top_k": 6,
                "expected_status": {"fts": "ok", "structure": "ok"},
                "required": {
                    "fts": [_source_key(caption_doc, "page:1:caption:1")],
                    "structure": [_source_key(caption_doc, "page:1:caption:1")],
                },
                "context": [
                    _source_key(caption_doc, "page:1"),
                    _source_key(caption_doc, "page:1:image:1"),
                ],
            },
            {
                "id": "full_k_tradeoff",
                "name_zh": "满 K 结构增益与种子取舍",
                "request": {
                    "query": "budgetword",
                    "scope": {
                        "course_ids": ["synthetic"],
                        "document_ids": [budget_doc.document_id],
                    },
                },
                "top_k": 3,
                "expected_status": {"fts": "ok", "structure": "ok"},
                "required": {
                    "fts": [_source_key(budget_doc, "page:1:shape:1")],
                    "structure": [_source_key(budget_doc, "page:1:shape:1")],
                },
                "context": [_source_key(budget_doc, "page:1")],
            },
            {
                "id": "pure_image_browse",
                "name_zh": "纯图页无文字锚点",
                "request": {
                    "query": "unseenword",
                    "scope": {
                        "course_ids": ["synthetic"],
                        "document_ids": [scan_doc.document_id],
                    },
                },
                "top_k": 2,
                "expected_status": {"fts": "empty", "structure": "no_text_anchor"},
                "required": {
                    "fts": [],
                    "structure": [
                        _source_key(scan_doc, "page:1"),
                        _source_key(scan_doc, "page:2"),
                    ],
                },
            },
            {
                "id": "single_file_empty_no_fallback",
                "name_zh": "单文件关闭浏览后的空结果",
                "request": {
                    "query": "never-present-token",
                    "scope": {
                        "course_ids": ["synthetic"],
                        "document_ids": [empty_doc.document_id],
                    },
                    "include_page_fallback": False,
                },
                "top_k": 3,
                "expected_status": {"fts": "empty", "structure": "empty"},
                "required": {"fts": [], "structure": []},
            },
            {
                "id": "multi_file_empty",
                "name_zh": "多文件空结果不填首页",
                "request": {
                    "query": "never-present-token",
                    "scope": {
                        "course_ids": ["synthetic"],
                        "document_ids": [scan_doc.document_id, empty_doc.document_id],
                    },
                },
                "top_k": 3,
                "expected_status": {"fts": "empty", "structure": "empty"},
                "required": {"fts": [], "structure": []},
            },
        ]

        records: list[dict[str, Any]] = []
        with _blocked_network() as external_attempts:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://c04-review",
            ) as client:

                async def request(payload: dict[str, Any], adjacency: bool) -> dict[str, Any]:
                    body = json.loads(json.dumps(payload, ensure_ascii=False))
                    body["include_adjacency"] = adjacency
                    started = time.perf_counter_ns()
                    response = await client.post("/api/rag/structure/search", json=body)
                    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
                    assert response.status_code == 200, (response.status_code, response.text)
                    return _compact_response(response.json(), elapsed_ms)

                for spec in cases:
                    print(f"[C04] 配对检索：{spec['name_zh']}")
                    body = {**spec["request"], "top_k": spec["top_k"]}
                    repeated: dict[str, list[dict[str, Any]]] = {"fts": [], "structure": []}
                    for round_index in range(3):
                        order = (
                            (("fts", False), ("structure", True))
                            if round_index % 2 == 0
                            else (("structure", True), ("fts", False))
                        )
                        for name, adjacency in order:
                            repeated[name].append(await request(body, adjacency))
                    lexical = _collapse_runs(repeated["fts"])
                    structured = _collapse_runs(repeated["structure"])
                    record = {
                        "id": spec["id"],
                        "name_zh": spec["name_zh"],
                        "request": body,
                        "fts": lexical,
                        "structure": structured,
                        "evaluation": _case_evaluation(
                            spec=spec,
                            lexical=lexical,
                            structured=structured,
                        ),
                    }
                    _assert_case(record)
                    records.append(record)

        all_external_attempts = [*setup_attempts, *external_attempts]
        assert all_external_attempts == [], all_external_attempts

        by_id = {record["id"]: record for record in records}
        code_key = _source_key(code_doc, "cell:0")
        for strategy in ("fts", "structure"):
            code_candidate = next(
                item
                for item in by_id["dotted_code_exact"][strategy]["candidates"]
                if item["key"] == code_key
            )
            assert code_candidate["match_kind"] == "exact_literal"
            assert by_id["negated_fft"][strategy]["query_trace"]["negation_detected"] is True
            k1_candidates = by_id["dotted_code_k1_limit"][strategy]["candidates"]
            assert [item["key"] for item in k1_candidates] == [_source_key(code_doc, "cell:1")]
            assert k1_candidates[0]["match_kind"] == "token_component"

        assert _candidate_keys(by_id["explicit_notebook_cell"]["fts"]) == [
            _source_key(notebook, "cell:1")
        ]
        assert _candidate_keys(by_id["explicit_notebook_cell"]["structure"]) == [
            _source_key(notebook, "cell:1"),
            _source_key(notebook, "cell:1:output:0"),
            _source_key(notebook, "cell:1:output:1"),
            _source_key(notebook, "cell:1:output:2"),
        ]
        ambiguity = by_id["cross_course_ambiguous_name"]
        for strategy in ("fts", "structure"):
            assert ambiguity[strategy]["candidates"] == []
            assert {item["document_id"] for item in ambiguity[strategy]["options"]} == {
                course_a.document_id,
                course_b.document_id,
            }

        caption_image = _source_key(caption_doc, "page:1:image:1")
        assert caption_image not in _candidate_keys(by_id["registered_caption_target"]["fts"])
        structured_caption = next(
            item
            for item in by_id["registered_caption_target"]["structure"]["candidates"]
            if item["key"] == caption_image
        )
        assert "caption_target" in {item["code"] for item in structured_caption["reasons"]}
        assert structured_caption["text_available"] is False

        assert _candidate_keys(by_id["full_k_tradeoff"]["fts"]) == [
            _source_key(budget_doc, "page:1:shape:1"),
            _source_key(budget_doc, "page:2"),
            _source_key(budget_doc, "page:3"),
        ]
        assert _candidate_keys(by_id["full_k_tradeoff"]["structure"]) == [
            _source_key(budget_doc, "page:1:shape:1"),
            _source_key(budget_doc, "page:2"),
            _source_key(budget_doc, "page:1"),
        ]

        pure_image = by_id["pure_image_browse"]
        assert pure_image["fts"]["candidates"] == []
        assert pure_image["structure"]["misses"] == ["no_text_anchor", "visual_not_read"]
        assert all(
            item["match_kind"] == "browse" and item["text_available"] is False
            for item in pure_image["structure"]["candidates"]
        )
        assert by_id["single_file_empty_no_fallback"]["structure"]["candidates"] == []
        assert by_id["multi_file_empty"]["structure"]["candidates"] == []

        summary: dict[str, Any] = {"case_count": len(records), "strategies": {}}
        for strategy in ("fts", "structure"):
            elapsed = [record[strategy]["elapsed_ms"] for record in records]
            summary["strategies"][strategy] = {
                "sum_case_median_elapsed_ms": round(sum(elapsed), 3),
                "median_case_elapsed_ms": round(statistics.median(elapsed), 3),
                "total_candidates_returned": sum(
                    len(record[strategy]["candidates"]) for record in records
                ),
                "total_evidence_inspected": sum(
                    record[strategy]["coverage"]["evidence_inspected_count"] for record in records
                ),
                "total_adjacency_edges_checked": sum(
                    record[strategy]["coverage"]["checked_adjacency_edges"] for record in records
                ),
            }
        summary["cases_with_structure_context_gain"] = [
            record["id"] for record in records if record["evaluation"]["structure_context_gain"] > 0
        ]
        summary["cases_with_seed_tradeoff"] = [
            record["id"] for record in records if record["evaluation"]["structure_removed"]
        ]
        summary["all_contract_assertions_passed"] = True

        evidence = {
            "schema": "srag-c04-public-trace-v1",
            "revision": REVISION,
            "tree": TREE,
            "generated_at": datetime.now(UTC).isoformat(),
            "runtime": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "sqlite": sqlite3.sqlite_version,
                "fastapi": importlib.metadata.version("fastapi"),
                "httpx": importlib.metadata.version("httpx"),
            },
            "code_hashes": code_hashes,
            "public_samples": samples,
            "isolation": {
                "temporary_database": True,
                "temporary_course_root": True,
                "credentials_configured": False,
                "external_socket_attempts": all_external_attempts,
                "a03_material_read": False,
                "search_surface": "create_app POST /api/rag/structure/search",
            },
            "summary": summary,
            "cases": records,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evidence = asyncio.run(_run(args.repo_root.resolve(), args.out.resolve()))
    print(
        "[C04] PASS："
        f"{evidence['summary']['case_count']} 组配对请求，外连 0，"
        f"结构增益 {len(evidence['summary']['cases_with_structure_context_gain'])} 组，"
        f"种子取舍 {len(evidence['summary']['cases_with_seed_tradeoff'])} 组"
    )
    print(f"[C04] 证据：{args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
