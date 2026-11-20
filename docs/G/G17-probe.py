"""One pinned public SQLite corpus, four paired SRAG retrieval strategies."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import runpy
import subprocess
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import httpx

from app.clients.jev import JevClient
from app.config import Settings
from app.rag.source_evidence import EvidenceRequest
from app.rag.source_vision import _prompt
from app.rag.structure_search import StructureScope, StructureSearcher, StructureSearchRequest
from app.rag.workspace import Draft, make_workspace
from app.rag.workspace_generation import generate_response
from app.rag.workspace_objects import object_choices, object_intent
from app.rag.workspace_retrieval import (
    BilingualRewriter,
    _candidate_id,
    _jev_candidate,
    plan_retrieval,
)
from app.schemas.local import SourceKey
from app.schemas.turn_evidence import TurnRead
from app.stores.structure_snapshot import SnapshotRequest

FIXTURES = runpy.run_path(str(Path(__file__).with_name("G17-fixtures.py")))
COURSE = FIXTURES["COURSE"]
CASES = FIXTURES["CASES"]
STRATEGIES = ("fts", "bilingual", "jev", "expanded")
BUDGET = 4


@dataclass
class Invocation:
    case: str
    arm: str
    phase: str
    question: str
    evidence: dict | None = None
    image_sha256: str | None = None
    inject_503: bool = False


CURRENT: ContextVar[Invocation | None] = ContextVar("public_g17", default=None)


@contextmanager
def invoking(invocation: Invocation):
    token = CURRENT.set(invocation)
    try:
        yield
    finally:
        CURRENT.reset(token)


def _key(obj):
    return obj.document_id, obj.version, obj.source_id


def _alias(source):
    return SourceKey(
        document_id=source.document_id, version=source.version, source_id=source.source_id
    )


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Egress:
    """Permit only exact new public questions, provenance-pinned text and the selected crop."""

    def __init__(self, settings, snapshot):
        self.settings = settings
        self.sources = {_key(source): source for source in snapshot.sources}
        self.public_passages = {(source.title, source.text) for source in snapshot.sources}
        self.jev_candidates = {
            _candidate_id(_alias(source)): (source.title, source.text)
            for source in snapshot.sources
        }
        self.events = []
        self.deep_url = httpx.URL(settings.deepseek_base_url.rstrip("/") + "/v1/chat/completions")
        jev_base = settings.jev_base_url.rstrip("/")
        self.jev_url = httpx.URL(
            jev_base if jev_base.endswith("/systemone") else jev_base + "/systemone"
        )
        dash = settings.dashscope_base_url or "https://dashscope.aliyuncs.com/api/v1"
        self.vision_url = httpx.URL(
            dash.rstrip("/") + "/services/aigc/multimodal-generation/generation"
        )

    def _check(self, request):
        ctx = CURRENT.get()
        if ctx is None or ctx.question != CASES[ctx.case]["question"]:
            raise ValueError("unknown public probe invocation")
        if request.method != "POST" or request.url.scheme != "https":
            raise ValueError("unexpected request method or scheme")
        payload = json.loads(request.content)
        event = {"case": ctx.case, "arm": ctx.arm, "phase": ctx.phase, "sent": False}
        if request.url == self.deep_url:
            self._auth(request, self.settings.deepseek_api_key)
            event["provider"] = "deepseek"
            self._deep(ctx, payload, event)
        elif request.url == self.jev_url:
            self._auth(request, self.settings.jev_api_key.strip())
            event["provider"] = "jev"
            self._jev(ctx, payload, event)
        elif request.url == self.vision_url:
            self._auth(request, self.settings.dashscope_api_key)
            event["provider"] = "dashscope"
            self._vision(ctx, payload, event)
        else:
            raise ValueError("unexpected provider endpoint")
        return ctx, event

    @staticmethod
    def _auth(request, key):
        if not key.strip() or request.headers.get("Authorization") != f"Bearer {key}":
            raise ValueError("unexpected authentication")
        if key.encode() in request.content:
            raise ValueError("credential in content")

    def _deep(self, ctx, payload, event):
        messages = payload["messages"]
        if [item["role"] for item in messages] != ["system", "user"]:
            raise ValueError("unexpected message roles")
        user = json.loads(messages[1]["content"])
        if ctx.phase == "plan":
            if (
                payload["model"] != self.settings.deepseek_query_model
                or user.get("query") != ctx.question
                or set(user) != {"query", "target"}
                or user["target"] not in {"English", "Chinese"}
            ):
                raise ValueError("unexpected public rewrite")
            event["kind"] = "rewrite"
            return
        if (
            ctx.phase != "answer"
            or payload["model"] != self.settings.deepseek_chat_model
            or user.get("question") != ctx.question
            or user.get("previous_context")
            or payload.get("temperature") != 0
            or payload.get("response_format") != {"type": "json_object"}
        ):
            raise ValueError("unexpected public answer")
        reads = user.get("evidence", [])
        if len(reads) != len(ctx.evidence or {}):
            raise ValueError("unexpected answer evidence count")
        for item in reads:
            pinned = ctx.evidence[item["id"]]
            if (
                item["source"] != pinned.revision.source.model_dump()
                or item["file_name"] != pinned.file_name
                or item["reads"]
                != [
                    {"text": read.text, "method": read.method, "status": read.status}
                    for read in pinned.reads
                ]
            ):
                raise ValueError("answer sent different material than read")
        event.update(kind="answer", evidence_ids=[item["id"] for item in reads])

    def _jev(self, ctx, payload, event):
        state = json.loads(payload["state"])
        if (
            ctx.phase not in {"plan", "retry_probe"}
            or payload["model"] != self.settings.jev_model
            or state["original_question"] != ctx.question
            or not payload["questions"]
        ):
            raise ValueError("unexpected Jev scoring context")
        event_ids = []
        for identifier, question in payload["questions"].items():
            instructions = question["instructions"]
            pair = (instructions["candidate_title"], instructions["candidate_passage"])
            if self.jev_candidates.get(identifier) != pair or question["type"] != "noul":
                raise ValueError("Jev candidate not an original public source")
            event_ids.append(identifier)
        for item in state["previously_retrieved_evidence"]:
            if (item["title"], item["passage"]) not in self.public_passages:
                raise ValueError("Jev history isn't an exact public passage")
        event.update(
            kind="score",
            candidate_ids=event_ids,
            prior_evidence_count=len(state["previously_retrieved_evidence"]),
        )

    def _vision(self, ctx, payload, event):
        messages = payload["input"]["messages"]
        if (
            ctx.case != "caption"
            or ctx.phase != "vision"
            or not ctx.image_sha256
            or payload["model"] != self.settings.dashscope_vision_model
            or len(messages) != 1
            or messages[0]["role"] != "user"
        ):
            raise ValueError("unexpected visual input")
        image, text = messages[0]["content"]
        prefix, data = image["image"].split(",", 1)
        digest = _sha(base64.b64decode(data, validate=True))
        if (
            not prefix.startswith("data:image/")
            or digest != ctx.image_sha256
            or text["text"] != _prompt("directed_question", ctx.question)
        ):
            raise ValueError("visual request lacks exact public crop or question")
        event.update(kind="image", image_sha256=digest)

    def _prepare(self, request):
        try:
            return self._check(request)
        except (ValueError, TypeError, KeyError, AttributeError, IndexError):
            ctx = CURRENT.get()
            self.events.append(
                {
                    "case": ctx.case if ctx else None,
                    "arm": ctx.arm if ctx else None,
                    "phase": ctx.phase if ctx else None,
                    "sent": False,
                    "blocked": True,
                }
            )
            raise RuntimeError("G17 blocked non-public provider input") from None

    @contextmanager
    def gate(self):
        original_async = httpx.AsyncHTTPTransport.handle_async_request
        original_sync = httpx.HTTPTransport.handle_request

        async def guarded_async(transport, request):
            _, event = self._prepare(request)
            start = perf_counter()
            event["sent"] = True
            try:
                response = await original_async(transport, request)
                event["status"] = response.status_code
                return response
            except httpx.TransportError as error:
                event["failure"] = type(error).__name__
                raise
            finally:
                event["headers_ms"] = round((perf_counter() - start) * 1000, 3)
                self.events.append(event)

        def guarded_sync(transport, request):
            ctx, event = self._prepare(request)
            if ctx.inject_503:
                ctx.inject_503 = False
                assert event["provider"] == "jev" and ctx.phase == "retry_probe"
                self.events.append(event | {"status": 503, "injected": True})
                return httpx.Response(503, content=b"injected unavailable")
            start = perf_counter()
            event["sent"] = True
            try:
                response = original_sync(transport, request)
                event["status"] = response.status_code
                return response
            except httpx.TransportError as error:
                event["failure"] = type(error).__name__
                raise
            finally:
                event["headers_ms"] = round((perf_counter() - start) * 1000, 3)
                self.events.append(event)

        with (
            patch.object(httpx.AsyncHTTPTransport, "handle_async_request", guarded_async),
            patch.object(httpx.HTTPTransport, "handle_request", guarded_sync),
        ):
            yield


def _actual_text(service, docs, sources, selected, limit):
    draft = Draft("public-g17")
    reads = []
    for item in selected:
        source = sources[_key(item.source)]
        if not source.text.strip() or len(draft.sources) == limit:
            continue
        doc = docs[source.document_id]
        raw = service.library.store.read_source(item.source, course_ids=(COURSE,))
        if (
            raw.status != "available"
            or not raw.is_current
            or raw.source is None
            or raw.source.text != source.text
            or raw.content_sha256 != doc.content_sha256
            or raw.publication_epoch != doc.publication_epoch
        ):
            raise RuntimeError("frozen source bytes or publication no longer match")
        method = "native_table" if source.locator.get("object_type") == "table" else "native_text"
        read = TurnRead(
            actual_source=item.source, method=method, text=raw.source.text, status="succeeded"
        )
        reference = draft.add_evidence(doc, source, [read])
        reads.append(
            {
                "id": reference.evidence_id,
                "file_name": doc.file_name,
                "source": item.source.model_dump(),
                "locator": source.locator,
                "method": method,
                "text": read.text,
            }
        )
    return draft, reads


def _vision_read(service, docs, sources, selected, case, arm):
    settings = service.settings
    caption = next(
        (
            sources[_key(item.source)]
            for item in selected
            if case["caption_phrase"] in sources[_key(item.source)].text
        ),
        None,
    )
    if caption is None:
        return {"status": "caption_not_selected", "actual_calls": 0}
    doc = docs[caption.document_id]
    neighbors = [
        source
        for source in sources.values()
        if source.document_id == caption.document_id and source.parent_id == caption.source_id
    ]
    intent = object_intent(case["question"])
    choices = object_choices(intent, neighbors) if intent else []
    if len(choices) != 1:
        return {"status": "caption_has_no_unique_child", "actual_calls": 0}
    source = choices[0]
    if (doc.file_name, source.source_id) != case["visual_target"]:
        return {
            "status": "incorrect_child_blocked",
            "actual_calls": 0,
            "candidate": _alias(source).model_dump(),
        }
    key = _alias(source)
    request = EvidenceRequest(
        source=key,
        expected_content_sha256=doc.content_sha256,
        expected_publication_epoch=doc.publication_epoch,
        rendition="object_crop",
    )
    rendered = service.vision.evidence.render_current(request)
    if (
        not rendered.display
        or rendered.display.actual_source != key
        or rendered.display.fallback_reason
    ):
        return {"status": "rendition_unavailable", "actual_calls": 0}
    display_sha = rendered.display.byte_sha256
    with invoking(Invocation("caption", arm, "vision", case["question"], image_sha256=display_sha)):
        started = perf_counter()
        result = service.vision.process_current(
            request.model_copy(
                update={"read_intent": "directed_question", "question": case["question"]}
            ),
            expected_input_sha256=display_sha,
        )
    parent = service.vision.evidence.render_current(
        request.model_copy(update={"rendition": "original_page"})
    )
    return {
        "status": result.read_status,
        "capability": result.capability_reason,
        "actual_calls": result.external_call_count,
        "elapsed_ms": round((perf_counter() - started) * 1000, 3),
        "provider_configured": bool(settings.dashscope_api_key.strip()),
        "caption": _alias(caption).model_dump(),
        "source": key.model_dump(),
        "locator": source.locator,
        "crop_sha256": display_sha,
        "crop": rendered.display.rect_visible_norm,
        "parent_open_status": parent.status,
        "parent_source": parent.display.actual_source.model_dump() if parent.display else None,
        "records": [item.model_dump(mode="json") for item in result.reads],
    }


def _assess(case, reads, generation, vision):
    actual = {(item["file_name"], item["source"]["source_id"]) for item in reads}
    by_id = {item["id"]: item for item in reads}
    cited = {
        (by_id[key]["file_name"], by_id[key]["source"]["source_id"])
        for line in generation.statements
        for key in line.citations
    }
    text = "\n".join(line.text for line in generation.statements) + "\n" + generation.missing
    record = {"answer": text.strip(), "cited_sources": list(sorted(cited))}
    if vision and any(
        item["origin_execution"] == "real_provider"
        and item["status"] in {"partial", "succeeded"}
        and item["actual_input_sha256"] == vision.get("crop_sha256")
        and item["actual_source"] == vision.get("source")
        for item in vision.get("records", [])
    ):
        actual.add(case["visual_target"])
        record["exact_actual_image"] = True
    gold = set(case["gold"])
    record["missing_sources"] = list(sorted(gold - actual))
    answer_enabled = generation.status in {"succeeded", "partial", "insufficient"}
    required, unallowed = case.get("required", []), case["unallowed"]
    if vision:
        read_text = "\n".join(item["text"] for item in vision.get("records", []))
        record["visual_read_text"] = read_text
        record["passed"] = (
            record.get("exact_actual_image", False)
            and not record["missing_sources"]
            and bool(re.search(case["required_vision"], read_text, re.IGNORECASE))
            and not any(re.search(rule, read_text, re.IGNORECASE) for rule in unallowed)
        )
    else:
        record["passed"] = (
            answer_enabled
            and not record["missing_sources"]
            and (gold <= cited or generation.missing and not generation.statements)
            and all(re.search(rule, text, re.IGNORECASE) for rule in required)
            and not any(re.search(rule, text, re.IGNORECASE) for rule in unallowed)
        )
    return record


def _candidate_name(item, docs):
    return [docs[item.source.document_id].file_name, item.source.source_id]


def evaluate(settings, out, *, run_retry, subject):
    files = FIXTURES["make_fixture"](settings.course_root)
    service = make_workspace(settings)
    imported = service.library.import_paths(COURSE, list(files))
    if len(imported) != len(files) or any(item.status != "published" for item in imported):
        raise RuntimeError("expected a fresh, isolated public import")
    scope = SnapshotRequest((COURSE,))
    snapshot = service.snapshots.read(scope)
    docs = {doc.document_id: doc for doc in snapshot.documents}
    sources = {_key(source): source for source in snapshot.sources}
    tables = [item for item in sources.values() if item.locator.get("object_type") == "table"]
    if len(tables) != 1 or docs[tables[0].document_id].file_name != FIXTURES["RESULTS"]:
        raise RuntimeError("could not identify the native public table")
    table_id = tables[0].source_id
    guard = Egress(settings, snapshot)
    scorer = JevClient(settings)
    rewrite = BilingualRewriter(settings)
    searcher = StructureSearcher(service.snapshots)
    retry = None
    results = {}
    with guard.gate():
        for name, original in CASES.items():
            case = original | {
                "gold": [
                    (file, table_id if key == "table:azure" else key)
                    for file, key in original["gold"]
                ]
            }
            question = case["question"]
            request = StructureSearchRequest(
                query=question,
                scope=StructureScope(course_ids=[COURSE]),
                include_page_fallback=False,
                include_adjacency=True,
            )
            results[name] = {"question": question, "expected_sources": case["gold"], "arms": {}}
            for strategy in STRATEGIES:
                started = perf_counter()
                with invoking(Invocation(name, strategy, "plan", question)):
                    plan = plan_retrieval(
                        settings,
                        searcher,
                        snapshot,
                        request,
                        frozenset({COURSE}),
                        strategy=strategy,
                        budget=BUDGET,
                        library=service.library.store,
                        rewrite=rewrite,
                        scorer=scorer,
                    )
                plan_ms = round((perf_counter() - started) * 1000, 3)
                assert len(plan.selected) <= BUDGET
                draft, reads = _actual_text(
                    service,
                    docs,
                    sources,
                    plan.selected,
                    BUDGET - 1 if name == "caption" else BUDGET,
                )
                with invoking(
                    Invocation(name, strategy, "answer", question, evidence=draft.sources)
                ):
                    generation = generate_response(
                        settings,
                        question,
                        draft.sources,
                        mode="ask",
                        coverage={
                            "full_text_scope": False,
                            "source_count": len(reads),
                            "mode": "retrieval",
                        },
                    )
                vision = (
                    _vision_read(service, docs, sources, plan.selected, case, strategy)
                    if name == "caption"
                    else None
                )
                assert len(reads) + bool(vision and vision.get("records")) <= BUDGET
                results[name]["arms"][strategy] = {
                    "initial": plan.initial.status,
                    "candidate_pool": plan.trace.get("candidate_pool", []),
                    "selected": [_candidate_name(item, docs) for item in plan.selected],
                    "reads": reads,
                    "plan_ms": plan_ms,
                    "plan": plan.trace,
                    "generation": generation.trace,
                    "vision": vision,
                    "statements": [
                        {"text": item.text, "citations": item.citations}
                        for item in generation.statements
                    ],
                    "missing": generation.missing,
                    "assessment": _assess(case, reads, generation, vision),
                }
                if (
                    run_retry
                    and retry is None
                    and strategy == "jev"
                    and plan.selected
                    and settings.jev_api_key.strip()
                ):
                    with invoking(
                        Invocation(name, "retry_probe", "retry_probe", question, inject_503=True)
                    ):
                        retry_result = JevClient(settings).score(
                            question, [_jev_candidate(item) for item in plan.selected]
                        )
                    retry = {"trace": retry_result.trace, "scored": len(retry_result.scores)}
            arms = results[name]["arms"]
            assert arms["bilingual"]["candidate_pool"] == arms["jev"]["candidate_pool"], (
                name,
                "changed pool",
            )
            assert service.snapshots.read(scope) == snapshot, "published course snapshot changed"
    audit = {
        "subject": subject,
        "budget": BUDGET,
        "strategy_order": STRATEGIES,
        "calls": guard.events,
        "retry_probe": retry,
        "provider_configured": {
            name: bool(getattr(settings, f"{name}_api_key").strip())
            for name in ("deepseek", "jev", "dashscope")
        },
        "model_names": {
            "rewrite": settings.deepseek_query_model,
            "answer": settings.deepseek_chat_model,
            "jev": settings.jev_model,
            "vision": settings.dashscope_vision_model,
        },
        "documents": [
            {
                "file_name": item.file_name,
                "document_id": item.document_id,
                "version": item.version,
                "content_sha256": item.content_sha256,
                "publication_epoch": item.publication_epoch,
            }
            for item in snapshot.documents
        ],
        "cases": results,
    }
    (out / "G17-live.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--retry-probe", action="store_true")
    parser.add_argument("--subject", required=True)
    args = parser.parse_args()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    expected = subprocess.check_output(["git", "rev-parse", args.subject], text=True).strip()
    if expected != actual:
        parser.error("HEAD doesn't match the requested experiment subject")
    out = args.out.resolve()
    out.mkdir(mode=0o700)
    cutoff = (
        {name + "_api_key": "" for name in ("deepseek", "jev", "dashscope")}
        if not args.env_file
        else {}
    )
    settings = Settings(
        _env_file=args.env_file,
        course_root=out / "public",
        data_dir=out / "data",
        workspace_course_ids=COURSE,
        sqlite_path=out / "library.db",
        workspace_answer_enabled=True,
        source_vision_enabled=bool(args.env_file),
        qdrant_api_key="",
        qdrant_url="http://127.0.0.1:1",
        **cutoff,
    )
    audit = evaluate(settings, out, run_retry=args.retry_probe, subject=actual)
    print(
        json.dumps(
            {
                "subject": actual,
                "out": str(out),
                "configured": audit["provider_configured"],
                "blocked": sum(bool(item.get("blocked")) for item in audit["calls"]),
            },
            ensure_ascii=False,
        )
    )
    return int(any(item.get("blocked") for item in audit["calls"]))


if __name__ == "__main__":
    raise SystemExit(main())
