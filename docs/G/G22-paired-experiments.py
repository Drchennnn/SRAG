"""G22 Step 2 & 3 paired experiments on development fixtures (G20 Q05, Q06 & controls)."""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

Q05_TEXT = (
    "在这份论文的附录实现细节里，默认使用的优化器、学习率、线性预热步数和批次大小分别是多少？"
    "默认的温度惩罚系数是多少，在做文本摘要实验时改成了多少？"
)
Q06_TEXT = (
    "Mixtral 和 DPO 这两份资料里，Mixtral-8x7B-Instruct 微调时用了哪两种方法？"
    "而在 DPO 论文附录里 TL;DR 摘要的 β 参数和默认 β 分别是多少？"
)
_HOST = "http://127.0.0.1:8001"
_COURSE = "g22-dev"


def _make_settings(tmp_dir: Path, strategy: str, base: Any) -> Any:
    from app.config import Settings

    return Settings(
        _env_file=str(REPO_ROOT / ".env"),
        sqlite_path=tmp_dir / "db.sqlite3",
        data_dir=tmp_dir / "data",
        course_root=tmp_dir / "originals",
        workspace_course_ids=_COURSE,
        workspace_max_upload_bytes=16 * 1024 * 1024,
        workspace_answer_enabled=True,
        source_vision_enabled=True,
        workspace_retrieval_strategy=strategy,
        workspace_input_token_limit=32_768,
        deepseek_api_key=base.deepseek_api_key,
        jev_api_key=base.jev_api_key,
        dashscope_api_key=base.dashscope_api_key,
    )


def _flat_nodes(nodes: list[dict[str, Any]]):
    for node in nodes:
        yield node
        yield from _flat_nodes(node.get("children", []))


def _find_node(client: Any, session_path: str, doc_id: str, kind: str, **locator: Any) -> dict:
    cat = client.get(session_path + "/catalog").json()
    for course in cat["courses"]:
        for doc in course["documents"]:
            if doc["document_id"] == doc_id:
                for node in [doc, *_flat_nodes(doc["nodes"])]:
                    if node["kind"] == kind and all(
                        node["locator"].get(k) == v for k, v in locator.items()
                    ):
                        return node
    raise RuntimeError(f"node not found: {doc_id} {kind} {locator}")


def _run_current_arm(arm: str, tmp_root: Path) -> dict[str, Any]:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.workspace import get_workspace_service, router
    from app.config import Settings
    from app.rag.workspace import make_workspace
    from app.rag.workspace_retrieval import BilingualRewriter

    base = Settings(_env_file=str(REPO_ROOT / ".env"))
    strategy = "jev" if arm == "jev" else "fts"
    settings = _make_settings(tmp_root, strategy, base)
    svc = make_workspace(settings)
    if arm == "rewrite_fallback":
        svc.rewriter = BilingualRewriter(
            settings, transport=httpx.MockTransport(lambda _req: httpx.Response(503))
        )

    app = FastAPI()
    app.include_router(router, prefix="/api/workspace")
    app.dependency_overrides[get_workspace_service] = lambda: svc

    dpo_path = REPO_ROOT / "docs" / "G" / "G20-fixtures" / "Direct-Preference-Optimization-2305.18290v3.pdf"
    mixtral_path = REPO_ROOT / "docs" / "G" / "G20-fixtures" / "Mixtral-of-Experts-2401.04088v1.pdf"
    nb_path = REPO_ROOT / "docs" / "G" / "G20-fixtures" / "v1" / "Mixtral-Router-Verification.ipynb"
    g14 = runpy.run_path(str(REPO_ROOT / "docs" / "G" / "G14-fixtures.py"))
    short_pdf_path = tmp_root / g14["FILENAME"]
    g14["make_pdf"](short_pdf_path)

    results: dict[str, Any] = {"arm": arm, "cases": {}}
    with TestClient(app, base_url=_HOST) as client:
        setup = client.post(
            "/api/workspace/sessions",
            json={"course_ids": [_COURSE]},
            headers={"origin": _HOST},
        )
        assert setup.status_code == 201, setup.text
        setup_path = f"/api/workspace/sessions/{setup.json()['session_id']}"

        def _upload(p: Path) -> dict[str, Any]:
            r = client.post(
                setup_path + "/import",
                params={"course_id": _COURSE, "filename": p.name},
                headers={"content-type": "application/octet-stream", "origin": _HOST},
                content=p.read_bytes(),
            )
            assert r.status_code == 200, r.text
            return r.json()

        dpo = _upload(dpo_path)
        mixtral = _upload(mixtral_path)
        nb = _upload(nb_path)
        short_doc = _upload(short_pdf_path)

        doc_names = {
            dpo["document_id"]: dpo_path.name,
            mixtral["document_id"]: mixtral_path.name,
            nb["document_id"]: nb_path.name,
            short_doc["document_id"]: short_pdf_path.name,
        }

        def _ask(
            case_id: str,
            doc_ids: list[str],
            question: str,
            *,
            focus_locator: tuple[str, str, dict[str, Any]] | None = None,
        ) -> None:
            sres = client.post(
                "/api/workspace/sessions",
                json={"course_ids": [_COURSE], "document_ids": doc_ids},
                headers={"origin": _HOST},
            )
            assert sres.status_code == 201, sres.text
            sdata = sres.json()
            spath = f"/api/workspace/sessions/{sdata['session_id']}"
            rev = sdata["scope_revision"]
            if focus_locator is not None:
                fdoc, fkind, floc = focus_locator
                node = _find_node(client, spath, fdoc, fkind, **floc)
                fres = client.put(
                    spath + "/focus",
                    json={"node_id": node["node_id"], "scope_revision": rev},
                    headers={"origin": _HOST},
                )
                assert fres.status_code == 200, fres.text
                rev = fres.json()["scope_revision"]
            t0 = perf_counter()
            tres = client.post(
                spath + "/turns",
                json={
                    "client_turn_id": f"{arm}-{case_id}-{uuid.uuid4().hex[:6]}",
                    "scope_revision": rev,
                    "kind": "ask",
                    "text": question,
                },
                headers={"origin": _HOST},
            )
            elapsed_ms = round((perf_counter() - t0) * 1000, 1)
            assert tres.status_code == 200, tres.text
            payload = tres.json()
            trace = payload["turn"]["trace"]
            msg = payload["message"]
            cands = [
                f"{c['file_name']}::{c['source']['source_id']}:{c['match_kind']}"
                for c in trace.get("candidates", [])
            ]
            read_ids = [
                f"{doc_names.get(r['source']['document_id'], r['source']['document_id'])}::{r['source']['source_id']}"
                for r in trace.get("reading", {}).get("read_ranges", [])
            ]
            ev_map = {}
            for eid, src in msg["sources"].items():
                first_read = (src.get("reads") or [{}])[0]
                actual = first_read.get("actual_source") or {}
                ev_map[eid] = (
                    f"{src.get('file_name', '')}::{actual.get('source_id', '')} "
                    f"(v{actual.get('version', '')})"
                )
            results["cases"][case_id] = {
                "elapsed_ms": elapsed_ms,
                "search_status": trace.get("search_status"),
                "candidate_count": len(cands),
                "candidates": cands,
                "retrieval": trace.get("retrieval"),
                "reading_full_text_scope": trace.get("reading", {}).get("full_text_scope"),
                "estimated_message_input": trace.get("reading", {}).get("estimated_message_input"),
                "read_ranges": read_ids,
                "generation_status": trace.get("generation", {}).get("status"),
                "external_call_count_retrieval": trace.get("external_call_count", 0),
                "external_call_count_generation": trace.get("generation", {}).get(
                    "external_call_count", 0
                ),
                "answer_blocks": msg["blocks"],
                "evidence_sources": ev_map,
                "visible_display_ids": msg.get("visible_display_ids", []),
            }

        _ask("Q05_dpo_single", [dpo["document_id"]], Q05_TEXT)
        _ask("Q06_mixtral_dpo_cross", [mixtral["document_id"], dpo["document_id"]], Q06_TEXT)
        if arm == "bilingual":
            _ask(
                "ctrl_short_doc",
                [short_doc["document_id"]],
                "这份课件列了哪些结业时间？",
            )
            _ask(
                "ctrl_explicit_page",
                [dpo["document_id"]],
                "这一页写的优化器和超参数是什么？",
                focus_locator=(dpo["document_id"], "page", {"page": 20}),
            )
            _ask(
                "ctrl_visual_nb",
                [nb["document_id"]],
                "第1个Cell的第一张图横轴是什么？",
                focus_locator=(nb["document_id"], "cell", {"cell_index": 0}),
            )
    return results


def _run_baseline_eba76b4() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="g22-eba76b4-") as tmp:
        tmp_path = Path(tmp)
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        subprocess.run(
            f"git archive eba76b4 | tar -x -C {code_dir}",
            cwd=REPO_ROOT,
            shell=True,
            check=True,
        )
        script = tmp_path / "run_base.py"
        script.write_text(
            f"""
import json, sys, uuid
from pathlib import Path
from time import perf_counter
sys.path.insert(0, {str(code_dir)!r})
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.api.workspace import get_workspace_service, router
from app.config import Settings
from app.rag.workspace import make_workspace

tmp_root = Path({str(tmp_path / 'run')!r})
repo_root = Path({str(REPO_ROOT)!r})
base = Settings(_env_file=str(repo_root / '.env'))
settings = Settings(
    _env_file=str(repo_root / '.env'),
    sqlite_path=tmp_root / 'db.sqlite3',
    data_dir=tmp_root / 'data',
    course_root=tmp_root / 'originals',
    workspace_course_ids={_COURSE!r},
    workspace_max_upload_bytes=16 * 1024 * 1024,
    workspace_answer_enabled=True,
    source_vision_enabled=True,
    workspace_retrieval_strategy='fts',
    workspace_input_token_limit=32768,
    deepseek_api_key=base.deepseek_api_key,
)
svc = make_workspace(settings)
app = FastAPI()
app.include_router(router, prefix='/api/workspace')
app.dependency_overrides[get_workspace_service] = lambda: svc

out = {{'arm': 'baseline_eba76b4_fts', 'cases': {{}}}}
with TestClient(app, base_url={_HOST!r}) as client:
    setup = client.post('/api/workspace/sessions', json={{'course_ids': [{_COURSE!r}]}}, headers={{'origin': {_HOST!r}}}).json()
    spath = f"/api/workspace/sessions/{{setup['session_id']}}"
    def pub(p):
        r = client.post(spath + '/import', params={{'course_id': {_COURSE!r}, 'filename': p.name}}, headers={{'content-type': 'application/octet-stream', 'origin': {_HOST!r}}}, content=p.read_bytes())
        assert r.status_code == 200, r.text
        return r.json()
    dpo = pub(repo_root / 'docs/G/G20-fixtures/Direct-Preference-Optimization-2305.18290v3.pdf')
    mixtral = pub(repo_root / 'docs/G/G20-fixtures/Mixtral-of-Experts-2401.04088v1.pdf')
    for cid, dids, q in [
        ('Q05_dpo_single', [dpo['document_id']], {Q05_TEXT!r}),
        ('Q06_mixtral_dpo_cross', [mixtral['document_id'], dpo['document_id']], {Q06_TEXT!r}),
    ]:
        s = client.post('/api/workspace/sessions', json={{'course_ids': [{_COURSE!r}], 'document_ids': dids}}, headers={{'origin': {_HOST!r}}}).json()
        t0 = perf_counter()
        tres = client.post(f"/api/workspace/sessions/{{s['session_id']}}/turns", json={{'client_turn_id': cid, 'scope_revision': s['scope_revision'], 'kind': 'ask', 'text': q}}, headers={{'origin': {_HOST!r}}})
        ms = round((perf_counter() - t0) * 1000, 1)
        assert tres.status_code == 200, tres.text
        payload = tres.json()
        tr = payload['turn']['trace']
        msg = payload['message']
        doc_map = {{
            dpo['document_id']: 'Direct-Preference-Optimization-2305.18290v3.pdf',
            mixtral['document_id']: 'Mixtral-of-Experts-2401.04088v1.pdf',
        }}
        out['cases'][cid] = {{
            'elapsed_ms': ms,
            'search_status': tr.get('search_status'),
            'candidates': [f"{{c['file_name']}}::{{c['source']['source_id']}}:{{c['match_kind']}}" for c in tr.get('candidates', [])],
            'read_ranges': [f"{{doc_map.get(r['source']['document_id'], r['source']['document_id'])}}::{{r['source']['source_id']}}" for r in tr.get('reading', {{}}).get('read_ranges', [])],
            'generation_status': tr.get('generation', {{}}).get('status'),
            'answer_blocks': msg['blocks'],
        }}
print(json.dumps(out, ensure_ascii=False))
""",
            encoding="utf-8",
        )
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        res = subprocess.run(
            [str(REPO_ROOT / ".venv" / "bin" / "python"), str(script)],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if res.returncode != 0:
            raise RuntimeError(f"baseline failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        return json.loads(res.stdout.splitlines()[-1])


def main() -> None:
    report: dict[str, Any] = {"arms": {}}
    report["arms"]["baseline_eba76b4_fts"] = _run_baseline_eba76b4()
    for arm in ("bilingual", "jev", "rewrite_fallback"):
        with tempfile.TemporaryDirectory(prefix=f"g22-{arm}-") as tmp:
            report["arms"][arm] = _run_current_arm(arm, Path(tmp))

    out_path = REPO_ROOT / "docs" / "G" / "G22-开发对照实测.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for arm_name, data in report["arms"].items():
        print(f"=== {arm_name} ===")
        for cid, case in data["cases"].items():
            read_pages = [
                r.split("::")[-1] + "@" + r.split("::")[0][:6] for r in case["read_ranges"]
            ]
            ans = " | ".join(
                b.get("text", "")[:120] for b in case["answer_blocks"] if b.get("type") == "text"
            )
            print(
                f"  {cid}: status={case['search_status']} ms={case['elapsed_ms']} "
                f"read={read_pages} ans={ans!r}"
            )


if __name__ == "__main__":
    main()
