"""Bounded bilingual retrieval and Jev ranking over one authorized publication snapshot."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import unicodedata
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from time import perf_counter
from typing import Any, Literal

import httpx

from app.clients.deepseek import DeepSeekClient
from app.clients.jev import JevCandidate, JevClient
from app.config import Settings
from app.rag.structure_search import (
    StructureCandidate,
    StructureScope,
    StructureSearcher,
    StructureSearchRequest,
    StructureSearchResponse,
    _structural_candidate,
)
from app.rag.workspace_evidence import readable_source_text
from app.schemas.local import SourceKey
from app.schemas.workspace import WorkspaceError
from app.stores.local_library import LocalLibraryStore
from app.stores.structure_snapshot import (
    SnapshotRequest,
    SnapshotSource,
    StructureSnapshot,
)

RetrievalStrategy = Literal[
    "fts", "bilingual", "jev", "expanded", "s0", "s2a", "s2b", "s2a_jev", "vector"
]
_VALID_STRATEGIES = frozenset(
    {"fts", "bilingual", "jev", "expanded", "s0", "s2a", "s2b", "s2a_jev", "vector"}
)
_Identity = tuple[str, int, str]
_SCOPED_FAILURES = {"ambiguous_scope", "scope_not_found", "location_not_found"}
_HAN = re.compile(r"[\u3400-\u9fff]")
_NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[+-]?\d+)?"
_NUMBERS = re.compile(rf"{_NUMBER}(?:\s*%)?", re.IGNORECASE)
_OPERATORS = re.compile(r"<=|>=|==|!=|<|>|=")
_NEGATION = re.compile(
    r"排除|不(?!(?:同|仅|但|只|过|论|管|止|妨|均(?:衡)?|少于|低于|高于|大于|小于|超过))|无(?:需|法)|未|没有|非(?!常)|"
    r"\b(?:no|not|never|without|excluding|exclude|neither|none|cannot|don't|doesn't|"
    r"didn't|isn't|mustn't|non-)\b",
    re.IGNORECASE,
)
_LITERAL = re.compile(
    r"`([^`\n]{1,160})`|[\"“]([^\"”\n]{1,160})[\"”]|"
    r"(?<![\w.])(?:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+|[A-Za-z]\w*_\w+|"
    r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]*\d[A-Za-z0-9]*)+|[A-Z][A-Z0-9]*-\w+|"
    r"[vV]\d+(?:\.\d+)*)(?![\w.])"
)
_ANCHOR = re.compile(
    r"(?<![\w.-])(?:[A-Z]{2,}(?:[-_.][A-Za-z0-9]+)+|[A-Z]+\d[A-Za-z0-9_-]*|"
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)(?![\w-]|\.[A-Za-z0-9_])"
)
_SENTENCE_SPLIT = re.compile(r"[。！？；\n.!?;]+")
_S2A_BACKTICK = re.compile(r"`([A-Za-z0-9_.-]{3,64})`")
_S2A_SNAKE = re.compile(r"(?<![\w.-])([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)(?![\w.-])")
_S2A_DOTTED = re.compile(r"(?<![\w.-])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)(?![\w.-])")
_S2A_IDENT = re.compile(
    r"(?<![\w.-])([A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]*\d[A-Za-z0-9]*)+"
    r"|[A-Z]{2,}[A-Za-z0-9]*\d+[A-Za-z0-9_-]*)(?![\w.-])"
)
_S2A_TERM_STOPS = frozenset(
    {
        "table-1", "table-2", "table-3", "table-4", "figure-1", "figure-2", "figure-3",
        "page-1", "page-2", "section-1", "section-2", "e-mail", "co-author", "pre-trained",
        "fine-tuning", "fine-tuned", "open-source", "high-level", "low-rank", "real-time",
        "end-to-end", "self-attention", "cross-attention", "zero-shot", "few-shot",
    }
)
_UNIT_FORMS = {
    "pct": r"percent(?:age)?|%",
    "ms": r"milliseconds?|毫秒|ms",
    "us": r"microseconds?|微秒|[μµu]s",
    "ns": r"nanoseconds?|纳秒|ns",
    "s": r"seconds?|秒|s",
    "hz": r"hertz|赫兹|hz",
    "khz": r"kilohertz|千赫(?:兹)?|khz",
    "mhz": r"megahertz|兆赫(?:兹)?|mhz",
    "mm": r"millimeters?|毫米|mm",
    "cm": r"centimeters?|厘米|cm",
    "mg": r"milligrams?|毫克|mg",
    "kg": r"kilograms?|千克|公斤|kg",
    "g": r"grams?|克|(?<![\w])g(?![A-Za-z])",
    "gb": r"gigabytes?|gb|(?<=\d)G(?![A-Za-z])",
    "mb": r"megabytes?|mb",
    "kb": r"kilobytes?|kb",
    "tb": r"terabytes?|tb",
    "db": r"decibels?|分贝|db",
    "deg": r"degrees?|度|deg|°",
    "samples": r"[个条]?\s*采样点|[个条]?\s*样本|samples?|examples?",
    "bit": r"bits?|比特|位",
}
_UNIT_BODY = "|".join(f"(?P<{unit}>{forms})" for unit, forms in _UNIT_FORMS.items())
_UNIT_AFTER = re.compile(rf"^(?:[\s-]*)(?:{_UNIT_BODY})(?![A-Za-z])", re.IGNORECASE)
_VAGUE_NUMERAL = re.compile(
    r"几(?:[十百千万亿兆]|(?:[多余来])?(?:个|次|步|条|倍|轮|层|张|页|项))|"
    r"(?:好|大)几[十百千万亿]|"
    r"数[十百千万亿兆]+|"
    r"(?:成[百千]上[千万]|上[百千万亿])|"
    r"[零一二两三四五六七八九十百千万亿]+\s*[来多余]\s*[个次步条倍轮层张页万亿]"
)
_VAGUE_ENGLISH = re.compile(
    r"\b(?:several\s+(?:hundred|thousand|million|billion|trillion)|"
    r"(?:hundreds|thousands|millions|billions)\s+of)\b",
    re.IGNORECASE,
)
_ZH_IDIOMS = re.compile(
    r"十分(?!之)|一般|一致|一直|一起|一些|一样|一定|一再|一切|一并|一共|一律|"
    r"统一|唯一|单一|万一|万分|千万(?:别|不要|不)|"
    r"(?:另|同|这|那|每|某)一(?=[个款项种份类步轮次])|"
    r"第(?![零一二两三四五六七八九十百千万\d\s]*(?:讲|章|节|课))[零一二两三四五六七八九十百千万\d]+(?=[阶段页章节点维阶位层轮步])?|"
    r"[零一二两三四]阶|[一二三]维"
)
_ZH_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_ZH_UNITS = {"十": 10, "百": 100, "千": 1_000}
_ZH_SCALES = {"万": 10**4, "亿": 10**8, "兆": 10**12}
_EN_SCALE_WORDS = {
    "hundred": Decimal("100"),
    "thousand": Decimal("1000"),
    "million": Decimal("1000000"),
    "billion": Decimal("1000000000"),
    "trillion": Decimal("1000000000000"),
}
_EN_SUFFIX_SCALES = {
    "K": Decimal("1000"),
    "M": Decimal("1000000"),
    "B": Decimal("1000000000"),
    "T": Decimal("1000000000000"),
}
_EN_SMALL_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


@dataclass(frozen=True)
class RewriteResult:
    query: str | None
    trace: dict[str, Any]


class BilingualRewriter:
    """One alternate-language query with an exact original-input cache."""

    def __init__(self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.transport = transport
        self._cache: OrderedDict[bytes, str] = OrderedDict()

    def __call__(self, query: str, *, verify: Callable[[], None] = lambda: None) -> RewriteResult:
        target = "English" if _HAN.search(query) else "Chinese"
        started = perf_counter()
        attempts: list[dict[str, Any]] = []
        trace: dict[str, Any] = {
            "provider": "deepseek",
            "model": self.settings.deepseek_query_model,
            "target": target,
            "attempts": attempts,
        }

        def finish(status, alternate=None, *, cache=False):
            trace.update(
                status=status,
                cache_hit=cache,
                external_call_count=len(attempts),
                elapsed_ms=round((perf_counter() - started) * 1_000, 3),
            )
            return RewriteResult(alternate, trace)

        verify()
        if not self.settings.deepseek_api_key.strip():
            return finish("not_configured")
        signing_input = "\0".join((self.settings.deepseek_base_url, trace["model"], target, query))
        cache_key = hashlib.sha256(signing_input.encode()).digest()
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return finish("succeeded", self._cache[cache_key], cache=True)
        client = DeepSeekClient(
            self.settings,
            transport=self.transport,
            before_attempt=verify,
            on_attempt=attempts.append,
        )
        try:
            reply = asyncio.run(client.bilingual_query(query, target))
        except (ValueError, TypeError, RecursionError):
            verify()
            return finish("invalid_response")
        except (RuntimeError, httpx.HTTPError, httpx.InvalidURL):
            verify()
            return finish("provider_error")
        verify()
        if isinstance(reply.get("usage"), dict):
            tokens = reply["usage"].get("total_tokens")
            if type(tokens) is int and tokens >= 0:
                trace["total_tokens"] = tokens
        alternate = reply["query"]
        if reason := _validate_rewrite_constraints(query, alternate, target):
            trace["constraint_reason"] = reason
            return finish("constraints_unverified")
        self._cache[cache_key] = alternate
        if len(self._cache) > 128:
            self._cache.popitem(last=False)
        return finish("succeeded", alternate)


@dataclass(frozen=True)
class PlannerResult:
    action: Literal["search", "finish"]
    missing_aspect: str
    queries: tuple[str, ...]
    reason: str
    trace: dict[str, Any]


class SearchPlanner:
    """Single-step LLM planner deciding whether a bounded second query is needed."""

    def __init__(self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.transport = transport

    def __call__(
        self, payload: dict[str, Any], *, verify: Callable[[], None] = lambda: None
    ) -> PlannerResult:
        started = perf_counter()
        attempts: list[dict[str, Any]] = []
        trace: dict[str, Any] = {
            "provider": "deepseek",
            "model": self.settings.deepseek_query_model,
            "attempts": attempts,
        }

        def finish(
            status: str,
            action: Literal["search", "finish"] = "finish",
            missing: str = "",
            queries: Sequence[str] = (),
            reason: str = "",
        ) -> PlannerResult:
            trace.update(
                status=status,
                action=action,
                missing_aspect=missing,
                queries=list(queries),
                reason=reason,
                external_call_count=len(attempts),
                elapsed_ms=round((perf_counter() - started) * 1_000, 3),
            )
            return PlannerResult(action, missing, tuple(queries), reason, trace)

        verify()
        if not self.settings.deepseek_api_key.strip():
            return finish("not_configured")
        client = DeepSeekClient(
            self.settings,
            transport=self.transport,
            before_attempt=verify,
            on_attempt=attempts.append,
        )
        try:
            reply = _run_coro_sync(client.plan_next_search(payload))
        except (ValueError, TypeError, RecursionError, json.JSONDecodeError):
            verify()
            return finish("invalid_response")
        except (RuntimeError, httpx.HTTPError, httpx.InvalidURL):
            verify()
            return finish("provider_error")
        verify()
        if isinstance(reply.get("usage"), dict):
            tokens = reply["usage"].get("total_tokens")
            if type(tokens) is int and tokens >= 0:
                trace["total_tokens"] = tokens
        return finish(
            "succeeded",
            action=reply["action"],
            missing=reply["missing_aspect"],
            queries=reply["queries"],
            reason=reply["reason"],
        )


def _run_coro_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


@dataclass(frozen=True)
class RetrievalPlan:
    initial: StructureSearchResponse
    selected: tuple[StructureCandidate, ...]
    trace: dict[str, Any]
    visual_candidates: tuple[StructureCandidate, ...] = ()
    visual_read_targets: tuple[StructureCandidate, ...] = ()


def plan_retrieval(
    settings: Settings,
    searcher: StructureSearcher,
    snapshot: StructureSnapshot,
    request: StructureSearchRequest,
    allowed_courses: frozenset[str],
    *,
    strategy: RetrievalStrategy,
    budget: int,
    library: LocalLibraryStore | None = None,
    active: Callable[[], bool] = lambda: True,
    rewrite: Callable[..., RewriteResult] | None = None,
    scorer: JevClient | None = None,
    planner: Callable[..., PlannerResult] | None = None,
    vector_bridge: Any = None,
    evidence_service: Any = None,
    original_question: str | None = None,
    supplement_saved_outputs: bool = False,
    allow_bilingual: bool = True,
    kg_enabled: bool = False,
    kg_store: Any = None,
) -> RetrievalPlan:
    """Reuse a shared pool and enforce one final source budget across all strategies."""

    if strategy not in _VALID_STRATEGIES or not 1 <= budget <= request.top_k:
        raise ValueError("invalid workspace retrieval strategy or source budget")
    if not active():
        raise WorkspaceError("interrupted")
    question = request.query if original_question is None else original_question
    scope = _PinnedScope(searcher, snapshot, request, allowed_courses, active)
    initial = searcher.search(request, allowed_course_ids=allowed_courses)
    scope.accept_initial(initial)
    text_initial = [candidate for candidate in initial.candidates if _has_readable_text(candidate)]
    visual_candidates = [
        candidate for candidate in initial.candidates if not _has_readable_text(candidate)
    ]
    visual_read_targets: tuple[StructureCandidate, ...] = ()
    pool = list(text_initial)
    trace: dict[str, Any] = {
        "strategy": strategy,
        "effective_strategy": "fts",
        "budget": budget,
        "pool_cap": request.top_k,
        "include_adjacency": request.include_adjacency,
        "original_query": question,
        "search_query": request.query,
        "alternate_query": None,
        "original_search_status": initial.status,
        "pins": [_pin_trace(doc) for doc in initial.snapshot],
        "original_pool": [_candidate_trace(candidate) for candidate in text_initial],
        "visual_candidates": [_candidate_trace(candidate) for candidate in visual_candidates],
        "rewrite": {"status": "not_requested", "external_call_count": 0},
        "jev": {"status": "not_requested", "external_call_count": 0},
        "expansion": {"status": "not_requested", "external_call_count": 0},
        "planner": {"status": "not_requested", "external_call_count": 0},
        "vector": {"status": "not_requested", "external_call_count": 0},
        "kg": {"status": "not_requested", "external_call_count": 0},
    }

    def finish(selected: Sequence[StructureCandidate]) -> RetrievalPlan:
        nonlocal visual_candidates
        selected_list = list(selected)
        if strategy in {"fts", "s0", "bilingual", "vector"}:
            selected_list = prioritize_matching_tables(
                selected_list,
                question=question,
                alternate_query=trace["alternate_query"],
                snapshot_sources=scope.sources,
                docs=scope.docs,
                budget=budget,
            )
            combined_tbl = {_identity(item.source): item for item in (*selected_list, *pool)}
            pool[:] = list(combined_tbl.values())[:request.top_k]
        if (
            kg_enabled
            and kg_store is not None
            and request.location is None
            and initial.status not in _SCOPED_FAILURES
            and initial.snapshot
        ):
            from app.rag.workspace_kg import expand_kg_one_hop

            kg_res = expand_kg_one_hop(
                kg_store,
                course_ids=scope.snapshot_request.course_ids,
                docs=scope.docs,
                sources=scope.sources,
                question=question,
                alternate_query=trace["alternate_query"],
                pool=selected_list,
                budget=budget,
                verify=scope.verify,
            )
            trace["kg"] = kg_res.trace
            if kg_res.visual_candidates:
                visual_candidates = _merge_visual(
                    visual_candidates,
                    kg_res.visual_candidates,
                    max(request.top_k, len(visual_candidates) + len(kg_res.visual_candidates)),
                )
            if kg_res.trace.get("status") == "succeeded":
                selected_list = kg_res.pool
                combined_pool = {_identity(item.source): item for item in (*selected_list, *pool)}
                pool[:] = list(combined_pool.values())[:request.top_k]
                trace["effective_strategy"] = f"{trace['effective_strategy']}+kg"
        if supplement_saved_outputs and trace["rewrite"]["status"] == "constraints_unverified":
            selected_list, trace["saved_output_fallback"] = _reserve_saved_outputs(
                scope, selected_list, budget
            )
            if trace["saved_output_fallback"]["status"] == "reserved":
                combined = {_identity(item.source): item for item in (*selected_list, *pool)}
                pool[:] = list(combined.values())[:request.top_k]
        scope.check_active()
        trace["candidate_pool"] = [_candidate_trace(candidate) for candidate in pool]
        trace["visual_candidates"] = [
            _candidate_trace(candidate) for candidate in visual_candidates
        ]
        trace["visual_read_targets"] = [
            _candidate_trace(candidate) for candidate in visual_read_targets
        ]
        trace["selected"] = [candidate.source.model_dump() for candidate in selected_list[:budget]]
        trace["external_call_count"] = sum(
            trace[name].get("external_call_count", 0)
            for name in ("rewrite", "jev", "expansion", "planner", "vector", "kg")
        )
        return RetrievalPlan(
            initial,
            tuple(selected_list[:budget]),
            trace,
            tuple(visual_candidates),
            visual_read_targets,
        )

    if (
        (strategy == "fts" and not kg_enabled)
        or request.location is not None
        or initial.status in _SCOPED_FAILURES
    ):
        return finish(pool)
    if not initial.snapshot:
        return finish(pool)

    # S0 / shared bilingual first-hop enrichment when query needs cross-lingual search.
    if allow_bilingual and strategy != "fts":
        scope.verify()
        rewritten = (rewrite or BilingualRewriter(settings))(request.query, verify=scope.verify)
        scope.verify()
        trace["rewrite"] = rewritten.trace
        if rewritten.query:
            trace["alternate_query"] = rewritten.query
            alternate = scope.search(rewritten.query)
            trace["alternate_search_status"] = alternate.status
            text_alternate = [
                candidate for candidate in alternate.candidates if _has_readable_text(candidate)
            ]
            visual_candidates = _merge_visual(
                visual_candidates,
                [
                    candidate
                    for candidate in alternate.candidates
                    if not _has_readable_text(candidate)
                ],
                request.top_k,
            )
            trace["alternate_pool"] = [
                _candidate_trace(candidate) for candidate in text_alternate
            ]
            pool = _fuse(text_initial, text_alternate, request.top_k)
            pool = _prioritize_continuations(
                pool, snapshot_sources=scope.sources, docs=scope.docs
            )
            trace["effective_strategy"] = "bilingual"

    if strategy in {"fts", "s0", "bilingual"}:
        pool = _prioritize_continuations(
            pool, snapshot_sources=scope.sources, docs=scope.docs
        )
        return finish(pool)

    # S3: Optional vector retrieval (text semantic supplement + visual page recall).
    if strategy == "vector" or (
        settings.workspace_vector_retrieval_enabled and vector_bridge is not None
    ):
        if vector_bridge is not None:
            vec_result = vector_bridge.search(
                snapshot=snapshot,
                docs=scope.docs,
                sources=scope.sources,
                question=question,
                alternate_query=trace["alternate_query"],
                pool_cap=request.top_k,
                s0_status=initial.status,
                s0_pool=pool,
                library_store=library,
                evidence_service=evidence_service,
                verify=scope.verify,
                max_visual_reads=settings.workspace_s3_max_visual_reads,
            )
            trace["vector"] = vec_result.trace
            if vec_result.status == "succeeded":
                if vec_result.text_candidates:
                    pool = _fuse_with_anchor_protection(
                        pool, vec_result.text_candidates, request.top_k
                    )
                    pool = _prioritize_continuations(
                        pool, snapshot_sources=scope.sources, docs=scope.docs
                    )
                if vec_result.visual_candidates:
                    visual_candidates = _merge_visual(
                        vec_result.visual_candidates,
                        visual_candidates,
                        max(request.top_k, len(vec_result.visual_candidates)),
                    )
                    visual_read_targets = vec_result.visual_read_targets
                trace["effective_strategy"] = "vector"
        else:
            trace["vector"] = {
                "status": "unavailable",
                "reason": "bridge_not_configured",
                "external_call_count": 0,
            }
        if strategy == "vector":
            return finish(pool)

    # S2a: Bounded deterministic 2-step literal-anchor expansion (decoupled from Jev).
    if strategy == "s2a":
        expanded_pool, trace["expansion"] = _expand_s2a(
            library, scope, question, pool, budget, trace
        )
        if trace["expansion"]["status"] == "succeeded":
            pool = expanded_pool
            trace["effective_strategy"] = "s2a"
        return finish(pool)

    # S2b: Bounded 2-step LLM planner (1 planner call + 1 scoped search).
    if strategy == "s2b":
        planned_pool, trace["planner"] = _expand_s2b(
            library,
            scope,
            planner or SearchPlanner(settings),
            question,
            pool,
            budget,
            trace,
        )
        if trace["planner"].get("added_count", 0) > 0:
            pool = planned_pool
            trace["effective_strategy"] = "s2b"
        return finish(pool)

    # S1 / legacy expanded / s2a_jev: Jev scoring with 48 KB batching & strict tie-break.
    scorer = scorer or JevClient(settings)
    scored, trace["jev"] = _score(scorer, question, pool, scope.verify)
    if trace["jev"]["status"] not in {"succeeded", "partial"}:
        if strategy in {"expanded", "s2a_jev"}:
            trace["expansion"]["status"] = (
                "unavailable_without_jev"
                if strategy == "expanded"
                else trace["jev"]["status"]
            )
        if strategy == "s2a_jev":
            expanded_pool, trace["expansion"] = _expand_s2a(
                library, scope, question, pool, budget, trace
            )
            if trace["expansion"]["status"] == "succeeded":
                pool = expanded_pool
                trace["effective_strategy"] = "s2a"
            return finish(pool)
        return finish(pool)
    if trace["jev"]["status"] == "succeeded" or trace["jev"].get("scored_count", 0) > 0:
        trace["effective_strategy"] = "jev"
    if strategy == "expanded":
        scored, trace["expansion"] = _expand(
            library, scope, scorer, question, scored, budget, trace
        )
        if trace["expansion"]["status"] == "succeeded":
            trace["effective_strategy"] = "expanded"
    elif strategy == "s2a_jev":
        scored, trace["expansion"] = _expand_s2a(
            library, scope, question, scored, budget, trace
        )
        if trace["expansion"]["status"] == "succeeded":
            trace["effective_strategy"] = "s2a_jev"
    return finish(scored)


class _PinnedScope:
    def __init__(self, searcher, snapshot, request, allowed, active):
        self.searcher = searcher
        self.request = request
        self.allowed = allowed
        self.active = active
        courses = set(request.scope.course_ids) & allowed
        ids = set(request.scope.document_ids)
        self.docs = {
            doc.document_id: doc
            for doc in snapshot.documents
            if doc.course_id in courses and (not ids or doc.document_id in ids)
        }
        self.sources = {_identity(source): source for source in snapshot.sources}
        self.snapshot_request: SnapshotRequest | None = None

    def check_active(self) -> None:
        if not self.active():
            raise WorkspaceError("interrupted")

    def accept_initial(self, initial: StructureSearchResponse) -> None:
        expected = self.docs
        if self.request.scope.file_names or initial.status in _SCOPED_FAILURES:
            expected = {doc.document_id: self.docs.get(doc.document_id) for doc in initial.snapshot}
        self._check(initial, expected)
        self.docs = {doc.document_id: doc for doc in initial.snapshot}
        courses = tuple(sorted({doc.course_id for doc in initial.snapshot}))
        self.snapshot_request = SnapshotRequest(courses, tuple(self.docs))

    def verify(self) -> None:
        self.check_active()
        current = self.searcher.store.read(self.snapshot_request)
        self.check_active()
        if current.resolution_status != "resolved" or not self._matches(
            current.documents, self.docs
        ):
            raise WorkspaceError("stale_reference")
        if any(self.sources.get(_identity(source)) != source for source in current.sources) or len(
            current.sources
        ) != sum(1 for source in self.sources.values() if source.document_id in self.docs):
            raise WorkspaceError("stale_reference")

    def search(self, query: str) -> StructureSearchResponse:
        self.check_active()
        request = self.request.model_copy(
            update={
                "query": query,
                "scope": StructureScope(
                    course_ids=list(self.snapshot_request.course_ids),
                    document_ids=list(self.docs),
                ),
            }
        )
        response = self.searcher.search(request, allowed_course_ids=self.allowed)
        self._check(response, self.docs)
        return response

    @staticmethod
    def _matches(actual, expected) -> bool:
        return len(actual) == len(expected) and all(
            (doc := expected.get(item.document_id)) is not None and _pin(doc) == _pin(item)
            for item in actual
        )

    def _check(self, response, expected) -> None:
        self.check_active()
        if not self._matches(response.snapshot, expected):
            raise WorkspaceError("stale_reference")
        for candidate in response.candidates:
            key = candidate.source
            source = self.sources.get(_identity(key))
            if (
                key.document_id not in expected
                or source is None
                or source.course_id != candidate.course_id
                or source.text != candidate.text
            ):
                raise WorkspaceError("stale_reference")


def _reserve_saved_outputs(
    scope: _PinnedScope, candidates: Sequence[StructureCandidate], budget: int
) -> tuple[list[StructureCandidate], dict[str, Any]]:
    """Reserve existing candidate slots for saved results missed by text search."""
    if budget < 2 or len(scope.docs) < 2:
        return list(candidates), {"status": "single_document_or_slot", "added": []}
    if any(item.kind == "notebook_output" for item in candidates[:budget]):
        return list(candidates), {"status": "output_already_selected", "added": []}
    outputs = [
        source for source in scope.sources.values()
        if source.document_id in scope.docs
        and source.kind == "notebook_output"
        and source.text.strip()
    ]
    if not outputs:
        return list(candidates), {"status": "no_saved_text_output", "added": []}
    scope.verify()
    terms = set(re.findall(r"[a-z][a-z0-9_-]{2,}", scope.request.query.casefold()))
    selected = sorted(
        outputs,
        key=lambda source: (
            -len(terms.intersection(re.findall(
                r"[a-z][a-z0-9_-]{2,}", f"{source.title} {source.text}".casefold()
            ))),
            scope.docs[source.document_id].file_name.casefold(),
            source.ordinal,
        ),
    )
    # Keep the first lexical anchor, then make space for exact outputs before the tail.
    reserved = [
        _structural_candidate(source, source, "saved_output_fallback", scope.docs.values())
        for source in selected[:min(2, budget - 1)]
    ]
    ranked = [*candidates[:1], *reserved, *candidates[1:]][:budget]
    return ranked, {
        "status": "reserved",
        "added": [item.source.model_dump() for item in reserved],
        "displaced": [
            item.source.model_dump() for item in candidates[:budget] if item not in ranked
        ],
    }


def _has_readable_text(candidate: StructureCandidate) -> bool:
    return bool(candidate.text.strip())


def _merge_visual(
    existing: Sequence[StructureCandidate],
    incoming: Sequence[StructureCandidate],
    cap: int,
) -> list[StructureCandidate]:
    merged: dict[_Identity, StructureCandidate] = {}
    for item in (*existing, *incoming):
        key = _identity(item.source)
        if key not in merged:
            merged[key] = item
    return list(merged.values())[:cap]


def _fuse(original, alternate, pool_cap: int) -> list[StructureCandidate]:
    sources: dict[_Identity, StructureCandidate] = {}
    ranks: dict[_Identity, float] = {}
    has_text_anchor = any(item.match_kind != "browse" for item in (*original, *alternate))
    for candidates in (original, alternate):
        for rank, item in enumerate(candidates, start=1):
            if has_text_anchor and item.match_kind == "browse":
                continue
            if not _has_readable_text(item):
                continue
            key = _identity(item.source)
            sources.setdefault(key, item)
            ranks[key] = ranks.get(key, 0) + 1 / (60 + rank)
    return [sources[key] for key in sorted(ranks, key=ranks.get, reverse=True)[:pool_cap]]


_TABLE_QUERY_CUE = re.compile(
    r"多表|跨表|各表|两张表|两个表|表格|参数表|限值表|限制表|阈值表|对照表|数据表|结果表|性能表|"
    r"(?<![图列行])表\s*#?\s*(?:[0-9]{1,4}|[零〇一二两三四五六七八九十]{1,3})|"
    r"(?<!图)表(?:中|内|里)|\btable(?:s|\s*#?\s*[0-9]{1,4})?\b",
    re.IGNORECASE,
)
_TABLE_COMPARE_CUE = re.compile(
    r"多表|跨表|各表|两张表|两个表|对比|比较|差异|不同|\b(?:compar(?:e|ison)|versus|vs\.?)\b",
    re.IGNORECASE,
)
_TABLE_NEGATION = re.compile(
    r"(?:不要|不用|不必|无需|跳过|忽略|别)\s*(?:查|看|读|引|用)?\s*(?:表格|表\s*[0-9]+|表)|"
    r"\b(?:skip|ignore|without|do\s+not|don['’]t)\s+(?:the\s+)?tables?\b",
    re.IGNORECASE,
)
_TABLE_ORDINAL_CUE = re.compile(
    r"(?:\btable\s*#?\s*(?P<en>[0-9]{1,4})\b|"
    r"第\s*(?P<zh_ord>[0-9]{1,4}|[零〇一二两三四五六七八九十]{1,3})\s*(?:张|个)\s*(?:表格|表)|"
    r"(?<![第图])表\s*#?\s*(?P<zh>[0-9]{1,4})(?![0-9.]))",
    re.IGNORECASE,
)
_TABLE_TERM_STOPS = frozenset(
    {
        "table", "tables", "compare", "comparison", "metric", "metrics", "value", "values",
        "result", "results", "data", "what", "which", "with", "from", "that", "this",
    }
)
_TABLE_ZH_STOPS = frozenset(
    {
        "表格", "参数", "指标", "对比", "比较", "数值", "数据", "限制", "限值", "实验",
        "代码", "输出", "变化", "多少", "什么", "请问", "结合", "核对", "分别", "各自",
        "哪些", "哪个", "对应", "是否", "达标", "结果", "实测", "讲义", "说明", "关系",
    }
)


def _is_table_candidate(cand: StructureCandidate) -> bool:
    return (
        cand.kind in {"table", "pptx_table"}
        or (cand.kind == "page_object" and cand.locator.get("object_type") == "table")
    ) and _has_readable_text(cand)


def _table_query_terms(question: str, alternate_query: str | None) -> tuple[list[str], set[int]]:
    ordinals: set[int] = set()
    for match in _TABLE_ORDINAL_CUE.finditer(question):
        raw = match.group("en") or match.group("zh") or match.group("zh_ord") or ""
        if raw.isdigit():
            ordinals.add(int(raw))
        elif raw in _ZH_DIGITS:
            ordinals.add(_ZH_DIGITS[raw])
    blob = "\n".join(filter(None, (question, alternate_query)))
    terms: list[str] = []
    seen: set[str] = set()
    for tok in re.findall(r"[A-Za-z0-9_.-]{2,}", blob):
        low = tok.casefold()
        if low in _TABLE_TERM_STOPS or low in seen or low.isdigit():
            continue
        seen.add(low)
        terms.append(low)
    for run in re.findall(r"[\u3400-\u9fff]{2,}", blob):
        if len(run) <= 4 and run not in _TABLE_ZH_STOPS and run not in seen:
            seen.add(run)
            terms.append(run)
        for idx in range(len(run) - 1):
            bg = run[idx : idx + 2]
            if bg not in _TABLE_ZH_STOPS and bg not in seen:
                seen.add(bg)
                terms.append(bg)
    return terms, ordinals


def _table_page_key(cand: StructureCandidate) -> tuple[str, int | str]:
    page = cand.locator.get("page")
    return (cand.source.document_id, page if page is not None else cand.source.source_id)


def prioritize_matching_tables(
    pool: Sequence[StructureCandidate],
    *,
    question: str,
    alternate_query: str | None = None,
    snapshot_sources: Mapping[_Identity, SnapshotSource] | None = None,
    docs: Mapping[str, Any] | None = None,
    budget: int = 6,
) -> list[StructureCandidate]:
    """Keep 1-2 query-matched table nodes inside budget=6 without displacing primary anchors."""
    if not pool or budget < 2:
        return list(pool)
    if _TABLE_NEGATION.search(question):
        return list(pool)
    has_table_cue = bool(_TABLE_QUERY_CUE.search(question))
    has_compare_cue = bool(_TABLE_COMPARE_CUE.search(question))
    if not has_table_cue and not has_compare_cue:
        return list(pool)

    terms, req_ordinals = _table_query_terms(question, alternate_query)
    hit_docs = {
        item.source.document_id
        for item in pool[: min(budget, len(pool))]
        if item.match_kind != "browse"
    }
    first_seed_by_doc: dict[str, SnapshotSource] = {}
    for item in pool:
        if item.match_kind != "browse" and item.source.document_id not in first_seed_by_doc:
            if snapshot_sources is not None:
                seed_src = snapshot_sources.get(_identity(item.source))
                if seed_src is not None:
                    first_seed_by_doc[item.source.document_id] = seed_src

    candidates: dict[_Identity, tuple[StructureCandidate, bool, int]] = {}
    for rank, item in enumerate(pool):
        if _is_table_candidate(item):
            candidates[_identity(item.source)] = (item, True, rank)

    if snapshot_sources is not None and docs is not None:
        for rank_off, src in enumerate(snapshot_sources.values(), start=len(pool)):
            is_tbl = src.kind in {"table", "pptx_table"} or (
                src.kind == "page_object" and src.locator.get("object_type") == "table"
            )
            if not is_tbl or not readable_source_text(src).strip():
                continue
            key = (src.document_id, src.version, src.source_id)
            if key in candidates:
                continue
            seed_src = first_seed_by_doc.get(src.document_id, src)
            cand = _structural_candidate(src, seed_src, "child_object", docs.values())
            candidates[key] = (cand, False, rank_off)

    if not candidates:
        return list(pool)

    ordinal_patterns = [
        (num, re.compile(rf"(?:\btable|表)\s*#?\s*{num}(?![0-9.])", re.IGNORECASE))
        for num in sorted(req_ordinals)
    ]

    page_captions: dict[tuple[str, int, int], list[str]] = {}
    if snapshot_sources is not None:
        for src in snapshot_sources.values():
            is_cap = src.kind == "caption" or (
                src.kind == "page_object" and src.locator.get("object_type") == "caption"
            )
            if is_cap and isinstance(src.locator.get("page"), int):
                cap_text = readable_source_text(src).strip()
                if cap_text:
                    page_captions.setdefault(
                        (src.document_id, src.version, src.locator["page"]), []
                    ).append(cap_text)

    scored: list[
        tuple[
            tuple[int, int, int, int, int, int],
            frozenset[str],
            frozenset[str],
            frozenset[int],
            bool,
            StructureCandidate,
        ]
    ] = []
    for cand, in_pool, orig_rank in candidates.values():
        parent_text = ""
        cap_texts: list[str] = []
        if snapshot_sources is not None and isinstance(cand.locator.get("page"), int):
            page_num = cand.locator["page"]
            p_src = snapshot_sources.get(
                (cand.source.document_id, cand.source.version, f"page:{page_num}")
            )
            if p_src is not None:
                parent_text = f"{p_src.title}\n{p_src.text}"
            cap_texts = page_captions.get(
                (cand.source.document_id, cand.source.version, page_num), []
            )
        blob = f"{cand.title}\n{cand.text}".casefold()
        aux_blob = "\n".join(part for part in (parent_text, *cap_texts) if part).casefold()
        matched_ords: set[int] = set()
        if req_ordinals:
            tbl_idx = cand.locator.get("table_index")
            if isinstance(tbl_idx, int) and tbl_idx in req_ordinals:
                matched_ords.add(tbl_idx)
            for num, pat in ordinal_patterns:
                if pat.search(f"{cand.title}\n{cand.text}") or (
                    parent_text and pat.search(parent_text)
                ):
                    matched_ords.add(num)
        ord_hit = 1 if matched_ords else 0
        direct_terms = frozenset(term for term in terms if term in blob)
        aux_terms = (
            frozenset(term for term in terms if term not in direct_terms and term in aux_blob)
            if aux_blob
            else frozenset()
        )
        direct_hits = len(direct_terms)
        aux_hits = len(aux_terms)
        total_hits = direct_hits + aux_hits
        same_doc = 1 if cand.source.document_id in hit_docs else 0
        if in_pool:
            if ord_hit == 0 and total_hits == 0 and not same_doc:
                continue
        elif same_doc:
            if ord_hit == 0 and total_hits == 0:
                continue
        elif ord_hit == 0 and total_hits < 2:
            continue
        has_match = bool(ord_hit or total_hits)
        scored.append(
            (
                (
                    ord_hit,
                    direct_hits * 2 + aux_hits,
                    direct_hits,
                    same_doc,
                    1 if in_pool else 0,
                    -orig_rank,
                ),
                direct_terms,
                aux_terms,
                frozenset(matched_ords),
                has_match,
                cand,
            )
        )

    if not scored:
        return list(pool)
    if not has_table_cue:
        matched_tables = [entry for entry in scored if (len(entry[1]) + len(entry[2])) > 0]
        if (
            len(matched_tables) < 2
            or len({_table_page_key(entry[5]) for entry in matched_tables}) < 2
            or not any(entry[1] for entry in matched_tables)
        ):
            return list(pool)
    scored.sort(key=lambda item: item[0], reverse=True)
    distinct_docs = {entry[5].source.document_id for entry in scored}
    distinct_pages = {_table_page_key(entry[5]) for entry in scored}
    want_multi = bool(
        len(req_ordinals) >= 2
        or _TABLE_COMPARE_CUE.search(question)
        or len(distinct_docs) >= 2
    )
    max_slots = 1
    if want_multi:
        if budget >= 6 and len(distinct_pages) >= 3:
            max_slots = min(3, max(1, budget // 2))
        else:
            max_slots = min(2, max(1, budget // 3))
    chosen_tables: list[StructureCandidate] = []
    _, first_direct, first_aux, covered_ords_ro, _, first_cand = scored[0]
    chosen_tables.append(first_cand)
    seen_pages: set[tuple[str, int | str]] = {_table_page_key(first_cand)}
    chosen_docs: set[str] = {first_cand.source.document_id}
    covered_terms = set(first_direct | first_aux)
    covered_ords = set(covered_ords_ro)
    remaining = scored[1:]
    while len(chosen_tables) < max_slots and remaining:
        best_idx = max(
            range(len(remaining)),
            key=lambda i: (
                1
                if (
                    _table_page_key(remaining[i][5]) not in seen_pages
                    and (
                        bool((remaining[i][1] | remaining[i][2]) - covered_terms)
                        or bool(remaining[i][3] - covered_ords)
                    )
                )
                else 0,
                1
                if (
                    remaining[i][5].source.document_id not in chosen_docs
                    and remaining[i][4]
                )
                else 0,
                1
                if (
                    _table_page_key(remaining[i][5]) not in seen_pages
                    and remaining[i][4]
                )
                else 0,
                len(remaining[i][3] - covered_ords),
                len(remaining[i][1] - covered_terms) * 2
                + len(remaining[i][2] - covered_terms),
                len(remaining[i][1] - covered_terms),
                *remaining[i][0],
            ),
        )
        _, m_direct, m_aux, m_ords, _, next_cand = remaining.pop(best_idx)
        if len(chosen_tables) >= 2 and not (
            _table_page_key(next_cand) not in seen_pages
            and (bool((m_direct | m_aux) - covered_terms) or bool(m_ords - covered_ords))
        ):
            break
        seen_pages.add(_table_page_key(next_cand))
        chosen_docs.add(next_cand.source.document_id)
        covered_terms.update(m_direct | m_aux)
        covered_ords.update(m_ords)
        chosen_tables.append(next_cand)

    if not chosen_tables:
        return list(pool)

    chosen_ids = {_identity(item.source) for item in chosen_tables}
    anchor_head: list[StructureCandidate] = []
    if pool:
        anchor_head.append(pool[0])
        if len(pool) > 1 and pool[1].kind in {"notebook_output", "cell"}:
            anchor_head.append(pool[1])
    head_ids = {_identity(item.source) for item in anchor_head}
    promoted = [item for item in chosen_tables if _identity(item.source) not in head_ids]
    tail = [
        item
        for item in pool
        if _identity(item.source) not in head_ids and _identity(item.source) not in chosen_ids
    ]
    return [*anchor_head, *promoted, *tail]


def _prioritize_continuations(
    pool: Sequence[StructureCandidate],
    *,
    anchors: int = 3,
    snapshot_sources: Mapping[_Identity, SnapshotSource] | None = None,
    docs: Mapping[str, Any] | None = None,
) -> list[StructureCandidate]:
    """Keep structural continuations and paired Notebook code <-> stdout together."""
    if not pool:
        return list(pool)
    anchor_items = list(pool[:anchors])
    anchor_ids = {_identity(item.source) for item in anchor_items}
    followers: dict[_Identity, list[StructureCandidate]] = {}
    moved: set[_Identity] = set()
    preferred_codes = (
        "child_object",
        "next_position",
        "previous_position",
        "parent_context",
        "caption_target",
    )
    for anchor in anchor_items:
        anchor_id = _identity(anchor.source)
        # J03/K02 co-ranking: keep readable child_output paired directly after its parent code cell.
        if anchor.kind == "cell":
            cell_children: list[tuple[int, StructureCandidate]] = []
            for rank, item in enumerate(pool):
                item_id = _identity(item.source)
                if item_id == anchor_id or item_id in anchor_ids or item_id in moved:
                    continue
                if (
                    item.kind == "notebook_output"
                    and item.source.document_id == anchor.source.document_id
                    and item.source.version == anchor.source.version
                    and item.locator.get("cell_index") == anchor.locator.get("cell_index")
                    and _has_readable_text(item)
                ):
                    out_idx = int(item.locator.get("output_index", rank))
                    cell_children.append((out_idx, item))
            if cell_children:
                cell_children.sort(key=lambda pair: pair[0])
                chosen = cell_children[0][1]
                moved.add(_identity(chosen.source))
                followers[anchor_id] = [chosen]
                continue
            if snapshot_sources is not None and docs is not None:
                anchor_src = snapshot_sources.get(anchor_id)
                if anchor_src is not None:
                    snap_children = sorted(
                        (
                            src
                            for src in snapshot_sources.values()
                            if src.kind == "notebook_output"
                            and src.document_id == anchor_src.document_id
                            and src.version == anchor_src.version
                            and src.parent_id == anchor_src.source_id
                            and src.text.strip()
                        ),
                        key=lambda s: (int(s.locator.get("output_index", s.ordinal)), s.ordinal),
                    )
                    if snap_children:
                        child_src = snap_children[0]
                        child_id = (child_src.document_id, child_src.version, child_src.source_id)
                        if child_id not in anchor_ids and child_id not in moved:
                            moved.add(child_id)
                            followers[anchor_id] = [
                                _structural_candidate(
                                    child_src, anchor_src, "child_object", docs.values()
                                )
                            ]
                            continue

        # Bidirectional Notebook pairing: readable stdout pulls its parent code cell.
        if (
            anchor.kind == "notebook_output"
            and snapshot_sources is not None
            and docs is not None
        ):
            out_src = snapshot_sources.get(anchor_id)
            if out_src is not None and out_src.parent_id:
                parent_id = (out_src.document_id, out_src.version, out_src.parent_id)
                parent_in_pool = next(
                    (
                        item
                        for item in pool
                        if _identity(item.source) == parent_id
                        and _identity(item.source) not in anchor_ids
                        and _identity(item.source) not in moved
                    ),
                    None,
                )
                if parent_in_pool is not None and _has_readable_text(parent_in_pool):
                    moved.add(parent_id)
                    followers[anchor_id] = [parent_in_pool]
                    continue
                if parent_id not in anchor_ids and parent_id not in moved:
                    parent_src = snapshot_sources.get(parent_id)
                    if (
                        parent_src is not None
                        and parent_src.kind == "cell"
                        and parent_src.text.strip()
                    ):
                        moved.add(parent_id)
                        followers[anchor_id] = [
                            _structural_candidate(
                                parent_src, out_src, "parent_context", docs.values()
                            )
                        ]
                        continue

        if anchor_items.index(anchor) >= 2:
            continue

        linked: list[tuple[int, int, StructureCandidate]] = []
        for rank, item in enumerate(pool):
            item_id = _identity(item.source)
            if item_id == anchor_id or item_id in anchor_ids or item_id in moved:
                continue
            codes = [
                reason.code
                for reason in item.reasons
                if reason.seed is not None and _identity(reason.seed) == anchor_id
            ]
            best = min(
                (preferred_codes.index(code) for code in codes if code in preferred_codes),
                default=None,
            )
            if best is not None:
                linked.append((best, rank, item))
        linked.sort(key=lambda pair: (pair[0], pair[1]))
        if linked:
            chosen = linked[0][2]
            moved.add(_identity(chosen.source))
            followers[anchor_id] = [chosen]
    ordered: list[StructureCandidate] = []
    for item in pool:
        item_id = _identity(item.source)
        if item_id in moved:
            continue
        ordered.append(item)
        ordered.extend(followers.get(item_id, ()))
    return ordered


def _fuse_with_anchor_protection(
    s0_pool: Sequence[StructureCandidate],
    vector_pool: Sequence[StructureCandidate],
    pool_cap: int,
) -> list[StructureCandidate]:
    """Fuse S0 lexical hits with S3 vector hits while protecting top S0 lexical anchors."""
    anchors = [item for item in s0_pool if item.match_kind != "browse"][:2]
    anchor_ids = {_identity(item.source) for item in anchors}
    sources: dict[_Identity, StructureCandidate] = {}
    ranks: dict[_Identity, float] = {}
    for rank, item in enumerate(s0_pool, start=1):
        if item.match_kind == "browse":
            continue
        key = _identity(item.source)
        sources.setdefault(key, item)
        ranks[key] = ranks.get(key, 0.0) + 1.0 / (60 + rank)
    for rank, item in enumerate(vector_pool, start=1):
        key = _identity(item.source)
        sources.setdefault(key, item)
        ranks[key] = ranks.get(key, 0.0) + 1.0 / (60 + rank)
    rest = [
        sources[key]
        for key in sorted(ranks, key=ranks.get, reverse=True)
        if key not in anchor_ids
    ]
    return [*anchors, *rest][:pool_cap]


def _interleave_candidates(
    lists: Sequence[Sequence[StructureCandidate]],
) -> list[StructureCandidate]:
    """Round-robin interleave multiple candidate lists without duplicates (SAG-Wiki style)."""
    result: list[StructureCandidate] = []
    seen: set[_Identity] = set()
    max_len = max((len(items) for items in lists), default=0)
    for idx in range(max_len):
        for items in lists:
            if idx >= len(items):
                continue
            cand = items[idx]
            key = _identity(cand.source)
            if key not in seen:
                seen.add(key)
                result.append(cand)
    return result


def _score(scorer, question, pool, verify, *, evidence=()):
    if not pool:
        return list(pool), {"status": "empty", "external_call_count": 0, "input_ids": []}
    inputs = [_jev_candidate(item) for item in pool]
    result = scorer.score(question, inputs, evidence=evidence, verify=verify)
    verify()
    wanted = [item.candidate_id for item in inputs]
    trace = {**result.trace, "input_ids": wanted}
    status = result.trace.get("status")
    if status not in {"succeeded", "partial"} or not result.scores:
        return list(pool), trace
    if any(
        key not in set(wanted)
        or type(value) not in (int, float)
        or isinstance(value, bool)
        or not 0 <= value <= 1
        for key, value in result.scores.items()
    ):
        return list(pool), {**trace, "status": "invalid_response"}
    scores_map = {cid: float(val) for cid, val in result.scores.items()}
    trace["scores"] = {cid: scores_map[cid] for cid in wanted if cid in scores_map}
    trace["scored_count"] = len(scores_map)

    vals = list(scores_map.values())
    if max(vals) < 0.15 or (len(vals) >= 3 and max(vals) - min(vals) < 1e-3 and min(vals) > 0.8):
        return list(pool), {**trace, "status": "low_confidence"}

    total = len(pool)
    ranked_pairs = [
        (
            item,
            scores_map.get(cid := _candidate_id(item.source)) is not None,
            round(scores_map[cid], 4) if cid in scores_map else -1.0,
            (total - idx) / total,
        )
        for idx, item in enumerate(pool)
    ]
    ranked_pairs.sort(key=lambda pair: (pair[1], pair[2], pair[3]), reverse=True)
    return [pair[0] for pair in ranked_pairs], trace


def _prefer_parent_pages_over_child_shapes(
    items: Sequence[StructureCandidate],
) -> list[StructureCandidate]:
    """Promote parent page above redundant child text shapes when both appear."""
    parent_by_key = {
        (item.source.document_id, item.source.version, item.source.source_id): item
        for item in items
        if item.kind == "page"
    }
    ordered: list[StructureCandidate] = []
    seen: set[_Identity] = set()
    for item in items:
        src = item.source
        page_no = item.locator.get("page")
        if (
            item.kind == "page_object"
            and item.locator.get("format") == "pptx"
            and item.locator.get("object_type") in {"title", "text"}
            and isinstance(page_no, int)
        ):
            parent = parent_by_key.get((src.document_id, src.version, f"page:{page_no}"))
            if parent is not None:
                pkey = _identity(parent.source)
                if pkey not in seen:
                    seen.add(pkey)
                    ordered.append(parent)
                continue
        ikey = _identity(src)
        if ikey not in seen:
            seen.add(ikey)
            ordered.append(item)
    return ordered


def _rank_step2_candidates(candidates: Sequence[StructureCandidate]) -> list[StructureCandidate]:
    """Place concrete measurements/outputs/pages ahead of short markdown heading cells."""
    def _tier(cand: StructureCandidate) -> int:
        if cand.kind == "notebook_output":
            return 0
        if cand.kind == "page":
            return 1
        stripped = cand.text.strip()
        if (
            cand.kind == "cell"
            and stripped.startswith("#")
            and len(stripped) < 100
            and "\n" not in stripped
        ):
            return 3
        return 2

    return sorted(candidates, key=lambda cand: (_tier(cand),))


def _extract_s2a_terms(
    text: str, prior_queries: str, covered_doc_ids: set[str], scope: _PinnedScope
) -> list[tuple[str, float, StructureSearchResponse]]:
    candidates: dict[str, float] = {}
    q_tokens = {
        tok.casefold()
        for tok in re.findall(r"[A-Za-z0-9_.-]{2,}|[\u3400-\u9fff]{2,}", prior_queries)
    }
    patterns = (
        (_S2A_BACKTICK, 2.0),
        (_S2A_SNAKE, 2.0),
        (_S2A_DOTTED, 1.5),
        (_S2A_IDENT, 1.4),
    )
    for sentence in _SENTENCE_SPLIT.split(text):
        sent = sentence.strip()
        if not sent:
            continue
        sent_low = sent.casefold()
        ctx_bonus = 1.2 if any(tok in sent_low for tok in q_tokens) else 0.0
        for pattern, base_weight in patterns:
            min_len = 3 if pattern in (_S2A_BACKTICK, _S2A_SNAKE) else 4
            for match in pattern.finditer(sent):
                term = match.group(1).strip()
                if (
                    len(term) < min_len
                    or term.casefold() in _S2A_TERM_STOPS
                    or _literal_term(term, prior_queries)
                ):
                    continue
                weight = base_weight + ctx_bonus
                if candidates.get(term, 0.0) < weight:
                    candidates[term] = weight
    ordered = sorted(candidates.items(), key=lambda pair: (-pair[1], pair[0]))[:8]
    if not ordered:
        return []
    # Specificity-weighted local probe: targeted 1-2 doc links beat generic multi-PDF terms.
    scored: list[tuple[str, float, StructureSearchResponse]] = []
    for term, weight in ordered:
        resp = scope.search(term)
        cross_docs = {
            cand.source.document_id
            for cand in resp.candidates
            if cand.match_kind != "browse"
            and cand.source.document_id not in covered_doc_ids
            and _literal_term(term, f"{cand.title}\n{cand.text}")
        }
        new_hits = [
            cand
            for cand in resp.candidates
            if cand.match_kind != "browse" and _literal_term(term, f"{cand.title}\n{cand.text}")
        ]
        if new_hits:
            if 1 <= len(cross_docs) <= 2:
                cross_bonus = 2.2
            elif len(cross_docs) == 3:
                cross_bonus = 1.0
            else:
                cross_bonus = 0.1
            scored.append((term, weight + cross_bonus, resp))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [(term, score, resp) for term, score, resp in scored[:3]]


def _expand_s2a(library, scope, question, pool, budget, prior_trace):
    trace: dict[str, Any] = {
        "status": "no_read_anchor",
        "external_call_count": 0,
        "terms": [],
    }
    if budget < 2:
        return list(pool), {
            **trace,
            "status": "insufficient_budget",
            "stop_reason": "insufficient_budget",
        }
    if library is None or not pool:
        return list(pool), {**trace, "stop_reason": "no_read_anchor"}

    pool_ordered = _prefer_parent_pages_over_child_shapes(pool)
    protected_count = min(3, max(2, budget // 2))
    protected = list(pool_ordered[:protected_count])
    covered_docs = {item.source.document_id for item in protected}
    prior_queries = "\n".join(
        filter(None, (question, scope.request.query, prior_trace.get("alternate_query")))
    )
    anchor_texts: list[tuple[StructureCandidate, str]] = []
    for item in pool_ordered[: min(6, len(pool_ordered))]:
        if item.match_kind == "browse" or not item.text.strip():
            continue
        scope.verify()
        raw = library.read_source(item.source, course_ids=(item.course_id,))
        doc = scope.docs[item.source.document_id]
        if (
            not raw.is_current
            or raw.publication_epoch != doc.publication_epoch
            or raw.content_sha256 != doc.content_sha256
        ):
            raise WorkspaceError("stale_reference")
        if raw.status == "available" and raw.source is not None:
            raw_text = readable_source_text(raw.source)
            if raw_text.strip():
                anchor_texts.append((item, raw_text))

    if not anchor_texts:
        return list(pool), {**trace, "stop_reason": "no_read_anchor"}

    merged_candidates: dict[
        str, tuple[float, StructureSearchResponse, dict[str, Any], StructureCandidate]
    ] = {}
    for item, raw_text in anchor_texts:
        for term, score, resp in _extract_s2a_terms(raw_text, prior_queries, covered_docs, scope):
            if term not in merged_candidates or score > merged_candidates[term][0]:
                merged_candidates[term] = (
                    score,
                    resp,
                    {
                        "term": term,
                        "score": round(score, 3),
                        "anchor_source": item.source.model_dump(),
                    },
                    item,
                )

    if not merged_candidates:
        return list(pool), {**trace, "status": "no_new_terms", "stop_reason": "no_new_terms"}

    chosen_entries = sorted(
        merged_candidates.values(), key=lambda entry: (-entry[0], entry[2]["term"])
    )[:3]
    trace["terms"] = [entry[2] for entry in chosen_entries]
    known = {_identity(item.source) for item in pool_ordered}
    per_term_lists: list[list[StructureCandidate]] = []
    for entry in chosen_entries:
        term = entry[2]["term"]
        resp = entry[1]
        filtered = _rank_step2_candidates(
            [
                cand
                for cand in resp.candidates
                if _identity(cand.source) not in known
                and cand.match_kind != "browse"
                and _literal_term(term, f"{cand.title}\n{cand.text}")
            ]
        )
        if filtered:
            per_term_lists.append(filtered)

    additions = _interleave_candidates(per_term_lists)
    trace["new_candidates"] = [_candidate_trace(candidate) for candidate in additions]
    if not additions:
        return list(pool), {
            **trace,
            "status": "no_new_candidates",
            "stop_reason": "no_new_candidates",
        }

    # Keep Step-1 anchor item(s) alongside top base slots, leaving room for Step-2 additions.
    max_protected = max(2, budget - 2)
    anchor_items = [entry[3] for entry in chosen_entries]
    protected_with_anchors: list[StructureCandidate] = []
    p_keys: set[_Identity] = set()
    for item in (*anchor_items, *protected):
        key = _identity(item.source)
        if key in p_keys:
            continue
        p_keys.add(key)
        protected_with_anchors.append(item)
        if len(protected_with_anchors) >= max_protected:
            break

    selected: list[StructureCandidate] = list(protected_with_anchors)
    selected_keys = {_identity(item.source) for item in selected}
    for cand in (*additions, *pool_ordered):
        key = _identity(cand.source)
        if key in selected_keys:
            continue
        selected_keys.add(key)
        selected.append(cand)
        if len(selected) >= budget:
            break
    trace.update(
        status="succeeded",
        stop_reason="step_limit_2",
        protected_count=len(protected_with_anchors),
        added_sources=[
            cand.source.model_dump()
            for cand in additions[: max(0, budget - len(protected_with_anchors))]
        ],
    )
    return selected[:budget], trace


def _expand_s2b(library, scope, planner, question, pool, budget, prior_trace):
    trace: dict[str, Any] = {
        "status": "not_requested",
        "external_call_count": 0,
        "added_count": 0,
        "steps_executed": 1,
    }
    if budget < 2:
        return list(pool), {
            **trace,
            "status": "insufficient_budget",
            "stop_reason": "insufficient_budget",
        }

    pool_ordered = _prefer_parent_pages_over_child_shapes(pool)
    scope.verify()
    previews: list[dict[str, Any]] = []
    for item in pool_ordered[: min(5, len(pool_ordered))]:
        raw_text = item.text
        if library is not None:
            raw = library.read_source(item.source, course_ids=(item.course_id,))
            doc = scope.docs[item.source.document_id]
            if (
                not raw.is_current
                or raw.publication_epoch != doc.publication_epoch
                or raw.content_sha256 != doc.content_sha256
            ):
                raise WorkspaceError("stale_reference")
            if raw.status == "available" and raw.source is not None:
                raw_text = readable_source_text(raw.source) or item.text
        previews.append(
            {
                "file_name": item.file_name,
                "source_id": item.source.source_id,
                "kind": item.kind,
                "title": item.title,
                "excerpt": raw_text.strip()[:500],
            }
        )

    per_doc_items: dict[str, list[dict[str, Any]]] = {doc_id: [] for doc_id in scope.docs}
    seen_outline: set[tuple[str, str]] = set()
    for src in scope.sources.values():
        if src.document_id not in scope.docs or src.kind not in {"page", "cell"}:
            continue
        title = src.title.strip() or (src.text.strip().splitlines()[0] if src.text.strip() else "")
        if not title:
            continue
        doc_name = scope.docs[src.document_id].file_name
        pair = (doc_name, title)
        if pair in seen_outline:
            continue
        seen_outline.add(pair)
        per_doc_items[src.document_id].append(
            {"file_name": doc_name, "source_id": src.source_id, "title": title[:100]}
        )

    outline: list[dict[str, Any]] = []
    doc_queues = [entries for entries in per_doc_items.values() if entries]
    max_rounds = max((len(entries) for entries in doc_queues), default=0)
    for idx in range(max_rounds):
        for entries in doc_queues:
            if idx < len(entries):
                outline.append(entries[idx])
                if len(outline) >= 60:
                    break
        if len(outline) >= 60:
            break

    if not previews and not outline:
        return list(pool), {**trace, "status": "no_step1_pool", "stop_reason": "no_step1_pool"}

    already_tried = [
        item
        for item in dict.fromkeys(
            (question, scope.request.query, prior_trace.get("alternate_query") or "")
        )
        if item
    ]
    payload = {
        "original_question": question,
        "step_1_evidence_previews": previews,
        "authorized_catalog_outline": outline,
        "already_tried_queries": already_tried,
    }
    plan_res = planner(payload, verify=scope.verify)
    trace.update(plan_res.trace)
    if plan_res.trace.get("status") != "succeeded":
        trace["stop_reason"] = f"planner_{plan_res.trace.get('status', 'error')}"
        return list(pool), trace
    if plan_res.action == "finish":
        trace["stop_reason"] = "planner_finished"
        return list(pool), trace

    tried_norm = {" ".join(q.split()).casefold() for q in already_tried}
    new_queries = [
        q
        for q in plan_res.queries
        if " ".join(q.split()).casefold() not in tried_norm
    ][:2]
    trace["filtered_queries"] = new_queries
    if not new_queries:
        trace["stop_reason"] = "no_new_terms"
        return list(pool), trace

    known = {_identity(item.source) for item in pool_ordered}
    per_query_lists: list[list[StructureCandidate]] = []
    for query in new_queries:
        scope.verify()
        resp = scope.search(query)
        filtered = _rank_step2_candidates(
            _prefer_parent_pages_over_child_shapes(
                [
                    cand
                    for cand in resp.candidates
                    if _identity(cand.source) not in known
                    and cand.match_kind != "browse"
                    and _has_readable_text(cand)
                ]
            )
        )
        if filtered:
            per_query_lists.append(filtered)

    additions = _interleave_candidates(per_query_lists)
    trace["steps_executed"] = 2
    trace["new_candidates"] = [_candidate_trace(cand) for cand in additions]
    if not additions:
        trace["stop_reason"] = "no_new_candidates"
        return list(pool), trace

    max_protected = min(4, max(2, budget - 2)) if pool_ordered else 0
    protected: list[StructureCandidate] = []
    seen_docs: set[str] = set()
    p_keys: set[_Identity] = set()
    preview_window = pool_ordered[: min(5, len(pool_ordered))]
    # Preserve at least one top hit per distinct document seen in planner previews before dupes.
    for cand in preview_window:
        if len(protected) >= max_protected:
            break
        if cand.source.document_id not in seen_docs:
            seen_docs.add(cand.source.document_id)
            p_keys.add(_identity(cand.source))
            protected.append(cand)
    for cand in preview_window:
        if len(protected) >= max_protected:
            break
        key = _identity(cand.source)
        if key not in p_keys:
            p_keys.add(key)
            protected.append(cand)
    selected: list[StructureCandidate] = list(protected)
    selected_keys = set(p_keys)
    added_sources: list[dict[str, Any]] = []
    addition_keys = {_identity(cand.source) for cand in additions}
    for cand in (*additions, *pool_ordered):
        key = _identity(cand.source)
        if key in selected_keys:
            continue
        selected_keys.add(key)
        selected.append(cand)
        if key in addition_keys:
            added_sources.append(cand.source.model_dump())
        if len(selected) >= budget:
            break

    trace.update(
        status="succeeded",
        stop_reason="step_limit_2",
        protected_count=len(protected),
        added_count=len(added_sources),
        added_sources=added_sources,
    )
    return selected[:budget], trace


def _expand(library, scope, scorer, question, pool, budget, prior_trace):
    trace: dict[str, Any] = {"status": "no_read_anchor", "external_call_count": 0}
    if budget < 2:
        return pool, {**trace, "status": "insufficient_budget"}
    if library is None:
        return pool, trace
    prior_queries = "\n".join(
        filter(None, (question, scope.request.query, prior_trace["alternate_query"]))
    )
    anchor = _find_anchor(library, scope, pool[:budget], prior_queries)
    if anchor is None:
        return pool, trace
    item, raw_text, term, start = anchor
    trace.update(
        term=term,
        anchor_source=item.source.model_dump(),
        anchor_span=[start, start + len(term)],
        anchor_text_sha256=hashlib.sha256(raw_text.encode()).hexdigest(),
    )
    known = {_identity(item.source) for item in pool}
    found = scope.search(term)
    additions = [
        candidate
        for candidate in found.candidates
        if _identity(candidate.source) not in known
        and candidate.match_kind != "browse"
        and _literal_term(term, f"{candidate.title}\n{candidate.text}")
    ]
    trace["new_candidates"] = [_candidate_trace(candidate) for candidate in additions]
    if not additions:
        return pool, {**trace, "status": "no_new_candidates"}
    ranked, scoring = _score(
        scorer,
        question,
        additions,
        scope.verify,
        evidence=(JevCandidate(_candidate_id(item.source), item.title, raw_text),),
    )
    trace.update(scoring=scoring, external_call_count=scoring["external_call_count"])
    if scoring["status"] != "succeeded":
        return pool, {**trace, "status": "scoring_unavailable"}
    chosen = ranked[0]
    if scoring["scores"][_candidate_id(chosen.source)] < 0.5:
        return pool, {**trace, "status": "unsupported_new_candidate"}
    trace.update(status="succeeded", added_source=chosen.source.model_dump())
    selected = pool[:budget]
    if len(selected) == budget:
        weakest = next(
            index for index in range(budget - 1, -1, -1) if selected[index].source != item.source
        )
        trace["displaced_source"] = selected.pop(weakest).source.model_dump()
    return [*selected, chosen], trace


def _find_anchor(library, scope, pool, prior_queries):
    for item in pool:
        if item.match_kind == "browse" or not item.text:
            continue
        match = next(
            (
                match
                for match in _ANCHOR.finditer(item.text)
                if not _literal_term(match[0], prior_queries)
            ),
            None,
        )
        if match is None:
            continue
        scope.verify()
        raw = library.read_source(item.source, course_ids=(item.course_id,))
        doc = scope.docs[item.source.document_id]
        if (
            not raw.is_current
            or raw.publication_epoch != doc.publication_epoch
            or (raw.content_sha256 != doc.content_sha256)
        ):
            raise WorkspaceError("stale_reference")
        if raw.status == "available" and raw.source and raw.source.text == item.text:
            return item, raw.source.text, match[0], match.start()
    return None


def _literal_term(term: str, value: str) -> bool:
    pattern = rf"(?<![\w.-]){re.escape(term)}(?![\w-]|\.[A-Za-z0-9_])"
    return re.search(pattern, value, re.IGNORECASE) is not None


_ASSIGN_PAIR = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*([+-]?\d+(?:\.\d+)?)\b")
_CLAUSE_SPLIT = re.compile(
    r"[,，、；;。?!？！]|\s+\b(?:and|while|whereas|versus|vs\.?)\b\s+|与"
)
_CLAUSE_ANCHOR = re.compile(
    r"(?<![A-Za-z0-9.])(?:[vV]\d+(?:\.\d+)*|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+|"
    r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]*\d[A-Za-z0-9]*)+|[A-Za-z]\w*_\w+|"
    r"\d+(?:\.\d+)?[BMKT]\b)(?![\w.])"
)
_MASK_VERBATIM = re.compile(
    r"(?<![A-Za-z0-9.])(?:[vV]\d+(?:\.\d+)*|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+|"
    r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]*\d[A-Za-z0-9]*)+)(?![\w.])"
)
_ZH_PERCENT = re.compile(
    r"百分之\s*([+-]?(?:\d+(?:\.\d+)?|[零一二两三四五六七八九十百千万点]+))"
)
_ZH_FRACTION = re.compile(
    r"([零一二两三四五六七八九十百千万亿]+)\s*分之\s*"
    r"([+-]?(?:\d+(?:\.\d+)?|[零一二两三四五六七八九十百千万点]+))"
)
_ARABIC_QTY = re.compile(
    rf"(?<![\w.])({_NUMBER})(?:\s*-\s*|\s+)?"
    r"(thousand|million|billion|trillion|[KMGTB](?![A-Za-z])|"
    r"[万亿兆]|(?:milligrams?|microseconds?|milliseconds?|nanoseconds?|kilograms?|"
    r"megabytes?|gigabytes?|kilobytes?|terabytes?|percent(?:age)?|decibels?|"
    r"kilohertz|megahertz|millimeters?|centimeters?|degrees?|samples?|examples?|seconds?|grams?|"
    r"hertz|bits?|毫克|千克|公斤|微秒|毫秒|纳秒|千赫兹?|兆赫兹?|毫米|厘米|分贝|秒|"
    r"赫兹|克|比特|位|个采样点|条采样点|个样本|条样本|采样点|样本|%|[μµu]s|ms|ns|mg|kg|"
    r"mb|gb|kb|tb|db|hz|khz|mhz|mm|cm|deg|°|s|g)\b|"
    r"(?<=[\d.])%|[μµu]s|[毫秒微纳克位%°])?",
    re.IGNORECASE,
)
_EN_WORD_PART = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|"
    r"trillion|point)"
)
_EN_WORD_QTY = re.compile(
    rf"\b({_EN_WORD_PART}(?:[\s-]+{_EN_WORD_PART})*)\b"
    r"(?!-(?:stage|step|shot|level|tier|pass|hop|way|sided|fold|part|phase|branch|head|layer|stream)\b)",
    re.IGNORECASE,
)
_EN_ORDINALS = {
    "first": Decimal("1"),
    "second": Decimal("2"),
    "third": Decimal("3"),
    "fourth": Decimal("4"),
    "fifth": Decimal("5"),
    "sixth": Decimal("6"),
    "seventh": Decimal("7"),
    "eighth": Decimal("8"),
    "ninth": Decimal("9"),
    "tenth": Decimal("10"),
}
_EN_ORDINAL_SECTION = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+"
    r"(?=lecture|chapter|module|unit|section)\b",
    re.IGNORECASE,
)
_EN_FRACTION_DENOMS = {
    "half": Decimal("2"),
    "halves": Decimal("2"),
    "quarter": Decimal("4"),
    "quarters": Decimal("4"),
    "fifth": Decimal("5"),
    "fifths": Decimal("5"),
    "eighth": Decimal("8"),
    "eighths": Decimal("8"),
    "tenth": Decimal("10"),
    "tenths": Decimal("10"),
    "hundredth": Decimal("100"),
    "hundredths": Decimal("100"),
    "thousandth": Decimal("1000"),
    "thousandths": Decimal("1000"),
    "millionth": Decimal("1000000"),
    "millionths": Decimal("1000000"),
}
_EN_FRACTION = re.compile(
    rf"\b(a|an|{_EN_WORD_PART}(?:[\s-]+{_EN_WORD_PART})*|\d+(?:\.\d+)?)"
    r"[\s-]+(half|halves|quarter|quarters|fifths?|eighths?|tenths?|hundredths?|"
    r"thousandths?|millionths?)\b",
    re.IGNORECASE,
)
_ZH_COMPOUND_OR_UNIT_QTY = re.compile(
    r"(?:(?:[零一二两三四五六七八九十百千万亿兆点]{2,})|"
    r"(?<=[取为=于及或和与到比在])\s*[零一二两三四五六七八九十百千万亿]|"
    r"[零一二两三四五六七八九十百千](?=\s*(?:个|项|种|份|次|轮|层|步|张|条|位|讲|章|节|课|"
    r"比特|毫克|千克|公斤|毫秒|微秒|纳秒|秒|赫兹|千赫|兆赫|毫米|厘米|分贝|度|倍|%|percent)))"
)
_COMP_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "<=",
        re.compile(
            r"<=|不超过|至多|最多|不高于|不大于|小于等于|"
            r"(?<=[\d零一二两三四五六七八九十百千万亿万])\s*(?:及)?以下|"
            r"\b(?:at\s+most|no\s+more\s+than|not\s+more\s+than|"
            r"less\s+than\s+or\s+equal\s+to|up\s+to|maximum\s+of)\b",
            re.IGNORECASE,
        ),
    ),
    (
        ">=",
        re.compile(
            r">=|至少|不少于|不低于|不小于|大于等于|"
            r"(?<=[\d零一二两三四五六七八九十百千万亿万])\s*(?:及)?以上|"
            r"\b(?:at\s+least|no\s+less\s+than|not\s+less\s+than|"
            r"greater\s+than\s+or\s+equal\s+to|minimum\s+of)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "!=",
        re.compile(r"!=|不等于", re.IGNORECASE),
    ),
    (
        "<",
        re.compile(
            r"<|小于|少于|低于|\b(?:less\s+than|fewer\s+than|lower\s+than|smaller\s+than)\b|"
            r"\bbelow\b(?=\s*(?:\d|zero|one|two|three|four|five|six|seven|eight|nine|ten))",
            re.IGNORECASE,
        ),
    ),
    (
        ">",
        re.compile(
            r">|大于|多于|高于|超过|超出|"
            r"\b(?:greater\s+than|more\s+than|higher\s+than|larger\s+than)\b|"
            r"\b(?:above|over|exceeds?|exceeding)\b"
            r"(?=\s*(?:\d|zero|one|two|three|four|five|six|seven|eight|nine|ten))",
            re.IGNORECASE,
        ),
    ),
)
_NEGATION_COMPARISON_STRIP = re.compile(
    r"不超过|不少于|不低于|不高于|不小于|不大于|"
    r"\b(?:no\s+more\s+than|no\s+less\s+than|not\s+more\s+than|not\s+less\s+than)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _QtySpan:
    start: int
    end: int
    value: Decimal
    unit: str | None


def _norm_dec(value: Decimal) -> Decimal:
    normalized = value.normalize()
    return Decimal("0") if normalized == 0 else normalized


def _parse_zh_number(token: str, *, allow_bare_unit: bool = False) -> Decimal | None:
    raw = token.strip()
    if not raw:
        return None
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?)", raw):
        try:
            return _norm_dec(Decimal(raw))
        except InvalidOperation:
            return None
    if "点" in raw:
        parts = raw.split("点")
        if len(parts) != 2 or not parts[1]:
            return None
        int_part = Decimal("0") if not parts[0] else _parse_zh_number(parts[0])
        if int_part is None:
            return None
        frac_digits: list[str] = []
        for ch in parts[1]:
            if ch.isdigit():
                frac_digits.append(ch)
            elif ch in _ZH_DIGITS:
                frac_digits.append(str(_ZH_DIGITS[ch]))
            else:
                # Handle decimal with trailing scale like 一点五亿 or 两点五万亿
                break
        tail = parts[1][len(frac_digits) :]
        if not frac_digits:
            return None
        base = int_part + Decimal(f"0.{''.join(frac_digits)}")
        scale_val = Decimal("1")
        if tail:
            if all(ch in _ZH_SCALES for ch in tail):
                for ch in tail:
                    scale_val *= Decimal(_ZH_SCALES[ch])
            else:
                return None
        return _norm_dec(base * scale_val)
    # Mixed Arabic + trailing Chinese scale like 540亿 / 1.5万
    mixed = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)([万亿兆]+)", raw)
    if mixed:
        scale_val = Decimal("1")
        for ch in mixed[2]:
            scale_val *= Decimal(_ZH_SCALES[ch])
        return _norm_dec(Decimal(mixed[1]) * scale_val)
    if raw == "两":
        return Decimal("2")
    total = Decimal("0")
    section = Decimal("0")
    current = Decimal("0")
    has_digit = False
    for index, ch in enumerate(raw):
        if ch in _ZH_DIGITS:
            current = Decimal(_ZH_DIGITS[ch])
            has_digit = True
        elif ch in _ZH_UNITS:
            unit = Decimal(_ZH_UNITS[ch])
            if not has_digit:
                if index == 0 and (ch == "十" or allow_bare_unit):
                    current = Decimal("1")
                else:
                    return None
            section += current * unit
            current = Decimal("0")
            has_digit = False
        elif ch in _ZH_SCALES:
            scale = Decimal(_ZH_SCALES[ch])
            section += current
            if section == 0 and total == 0:
                if index == 0 and allow_bare_unit:
                    section = Decimal("1")
                else:
                    return None
            # Support trailing 万亿 (`万` then `亿`) -> multiply total+section
            if scale > Decimal("10000") and total > 0 and section == 0:
                total *= scale
            else:
                if ch == "亿" and "万" in raw[index + 1 :]:
                    total = (total + section) * scale
                else:
                    total = total + section * scale
            section = Decimal("0")
            current = Decimal("0")
            has_digit = False
        else:
            return None
    return _norm_dec(total + section + current)


def _parse_en_words(phrase: str) -> Decimal | None:
    words = [w.casefold() for w in re.split(r"[\s-]+", phrase.strip()) if w.strip()]
    if not words:
        return None
    if len(words) == 1 and words[0] == "one":
        return None
    if "point" in words:
        idx = words.index("point")
        left, right = words[:idx], words[idx + 1 :]
        if not left or not right:
            return None
        int_val = _parse_en_int_words(left)
        if int_val is None:
            return None
        frac_str: list[str] = []
        scale_idx = len(right)
        for pos, word in enumerate(right):
            if word in _EN_SMALL_WORDS and _EN_SMALL_WORDS[word] < 10:
                frac_str.append(str(_EN_SMALL_WORDS[word]))
            elif word in _EN_SCALE_WORDS and pos > 0:
                scale_idx = pos
                break
            else:
                return None
        if not frac_str:
            return None
        val = int_val + Decimal(f"0.{''.join(frac_str)}")
        for s_word in right[scale_idx:]:
            if s_word not in _EN_SCALE_WORDS:
                return None
            val *= _EN_SCALE_WORDS[s_word]
        return _norm_dec(val)
    return _parse_en_int_words(words)


def _parse_en_int_words(words: list[str]) -> Decimal | None:
    total = Decimal("0")
    current = Decimal("0")
    for word in words:
        if word in _EN_SMALL_WORDS:
            current += Decimal(_EN_SMALL_WORDS[word])
        elif word == "hundred":
            current = (current or Decimal("1")) * Decimal("100")
        elif word in _EN_SCALE_WORDS:
            scale = _EN_SCALE_WORDS[word]
            total += (current or Decimal("1")) * scale
            current = Decimal("0")
        else:
            return None
    return _norm_dec(total + current)


def _match_unit(tail: str) -> tuple[str | None, int]:
    match = _UNIT_AFTER.match(tail)
    if not match or not match.lastgroup:
        return None, 0
    return match.lastgroup, match.end()


def _extract_quantities(text: str) -> list[_QtySpan]:
    masked = _MASK_VERBATIM.sub(lambda m: " " * len(m[0]), text)
    spans: list[_QtySpan] = []

    def add_if_free(start: int, end: int, value: Decimal, unit: str | None) -> None:
        if any(not (end <= s.start or start >= s.end) for s in spans):
            return
        spans.append(_QtySpan(start, end, _norm_dec(value), unit))

    # 1. Chinese percentages & fractions
    for match in _ZH_PERCENT.finditer(masked):
        val = _parse_zh_number(match[1])
        if val is not None:
            add_if_free(match.start(), match.end(), val, "pct")
    for match in _ZH_FRACTION.finditer(masked):
        denom = _parse_zh_number(match[1], allow_bare_unit=True)
        numer = _parse_zh_number(match[2])
        if denom and numer is not None and denom != 0:
            # Verify terminating base-10 fraction
            d_check = denom
            for factor in (2, 5):
                while d_check % factor == 0:
                    d_check //= factor
            if d_check == 1:
                add_if_free(match.start(), match.end(), numer / denom, None)
    for match in _EN_FRACTION.finditer(masked):
        denom = _EN_FRACTION_DENOMS.get(match[2].casefold())
        raw_num = match[1].strip().casefold()
        if raw_num in {"a", "an", "one"}:
            numer = Decimal("1")
        elif re.fullmatch(r"\d+(?:\.\d+)?", raw_num):
            numer = Decimal(raw_num)
        else:
            numer = _parse_en_words(raw_num)
        if denom and numer is not None:
            add_if_free(match.start(), match.end(), numer / denom, None)

    # 2. Arabic quantities with optional scale / unit
    for match in _ARABIC_QTY.finditer(masked):
        num_str = match[1].replace(",", "")
        try:
            base = Decimal(num_str)
        except InvalidOperation:
            continue
        suffix = (match[2] or "").strip()
        end_pos = match.end()
        unit: str | None = None
        if suffix.casefold() in _EN_SCALE_WORDS:
            base *= _EN_SCALE_WORDS[suffix.casefold()]
        elif suffix in _EN_SUFFIX_SCALES and (
            match.group(0)[len(match[1]) : len(match[1]) + 1] == suffix
        ):
            base *= _EN_SUFFIX_SCALES[suffix]
        elif suffix and all(ch in _ZH_SCALES for ch in suffix):
            for ch in suffix:
                base *= Decimal(_ZH_SCALES[ch])
        elif suffix:
            u_name, u_len = _match_unit(suffix)
            if u_name:
                unit = u_name
                end_pos = match.start(2) + u_len
            else:
                end_pos = match.end(1)
        if unit is None:
            u_name, u_len = _match_unit(masked[end_pos:])
            if u_name:
                unit = u_name
                end_pos += u_len
        add_if_free(match.start(1), end_pos, base, unit)

    # 3. Spelled-out English number words
    for match in _EN_WORD_QTY.finditer(masked):
        val = _parse_en_words(match[1])
        end_pos = match.end()
        unit = None
        if val is None and match[1].casefold() == "one":
            u_name, u_len = _match_unit(masked[end_pos:])
            if u_name:
                val = Decimal("1")
                unit = u_name
                end_pos += u_len
        if val is None:
            continue
        if unit is None:
            u_name, u_len = _match_unit(masked[end_pos:])
            if u_name:
                unit = u_name
                end_pos += u_len
        add_if_free(match.start(), end_pos, val, unit)

    for match in _EN_ORDINAL_SECTION.finditer(masked):
        ord_val = _EN_ORDINALS.get(match[1].casefold())
        if ord_val is not None:
            add_if_free(match.start(1), match.end(1), ord_val, None)

    # 4. Pure Chinese numerals (with idioms neutralized)
    cleaned = _ZH_IDIOMS.sub(lambda m: " " * len(m[0]), masked)
    for match in _ZH_COMPOUND_OR_UNIT_QTY.finditer(cleaned):
        if any(not (match.end() <= s.start or match.start() >= s.end) for s in spans):
            continue
        val = _parse_zh_number(match[0])
        if val is None:
            continue
        end_pos = match.end()
        u_name, u_len = _match_unit(cleaned[end_pos:])
        if u_name:
            end_pos += u_len
        add_if_free(match.start(), end_pos, val, u_name)

    spans.sort(key=lambda item: item.start)
    return spans


def _extract_comparisons(
    text: str, quantities: Sequence[_QtySpan]
) -> tuple[Counter[tuple[str, Decimal | None, str | None]], int] | None:
    masked = _MASK_VERBATIM.sub(lambda m: " " * len(m[0]), text)
    found: list[tuple[int, int, str, str]] = []
    for canon_op, pattern in _COMP_SPECS:
        for match in pattern.finditer(masked):
            if any(not (match.end() <= s or match.start() >= e) for s, e, _, _ in found):
                continue
            found.append((match.start(), match.end(), canon_op, match[0]))
    bindings: Counter[tuple[str, Decimal | None, str | None]] = Counter()
    for start, end, canon_op, raw_op in found:
        after = next((q for q in quantities if 0 <= q.start - end <= 32), None)
        before = (
            next((q for q in reversed(quantities) if 0 <= start - q.end <= 12), None)
            if after is None and raw_op.endswith(("以上", "以下"))
            else None
        )
        target = after or before
        if target is None:
            if raw_op in {"<=", ">=", "<", ">", "!="}:
                bindings[(canon_op, None, None)] += 1
        else:
            bindings[(canon_op, target.value, target.unit)] += 1
    non_assign_text = _ASSIGN_PAIR.sub(" ", text)
    eq_count = len(re.findall(r"==|(?<![<>!=])=(?!=)", non_assign_text))
    return bindings, eq_count


def _extract_clause_bindings(
    text: str,
) -> tuple[dict[str, Counter[tuple[Decimal, str | None]]], Counter[tuple[str, Decimal]]]:
    assigns: Counter[tuple[str, Decimal]] = Counter(
        (m[1].casefold(), _norm_dec(Decimal(m[2]))) for m in _ASSIGN_PAIR.finditer(text)
    )
    clauses = [part.strip() for part in _CLAUSE_SPLIT.split(text) if part.strip()]
    anchor_map: dict[str, Counter[tuple[Decimal, str | None]]] = {}
    if len(clauses) < 2:
        return anchor_map, assigns
    for clause in clauses:
        anchors = [m[0].casefold() for m in _CLAUSE_ANCHOR.finditer(clause)]
        if not anchors:
            continue
        clause_clean = _ASSIGN_PAIR.sub(" ", clause)
        qtys = _extract_quantities(clause_clean)
        companion: Counter[tuple[Decimal, str | None]] = Counter()
        for q in qtys:
            q_slice = clause_clean[q.start : q.end].strip().casefold()
            if q_slice in anchors:
                continue
            companion[(q.value, q.unit)] += 1
        for anchor in anchors:
            anchor_map.setdefault(anchor, Counter()).update(companion)
    return anchor_map, assigns


def _validate_rewrite_constraints(original: str, candidate: str, target: str) -> str | None:
    if not candidate or len(candidate) > min(4_000, max(160, len(original) * 8)):
        return "length_or_empty"
    if candidate.casefold() == original.casefold() or (
        target == "Chinese" and not _HAN.search(candidate)
    ):
        return "language_mismatch"
    if target == "English" and not re.search(r"[A-Za-z]{3}", candidate):
        return "language_mismatch"

    translation = str.maketrans({"≤": "<=", "≥": ">=", "≠": "!=", "−": "-"})
    old, new = (
        unicodedata.normalize("NFKC", value).translate(translation)
        for value in (original, candidate)
    )

    for match in _LITERAL.finditer(original):
        literal = match[1] or match[2] or match[0]
        if target == "English" and _HAN.search(literal):
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_./-]*", literal):
                if token.casefold() not in new.casefold():
                    return "literal_missing"
            continue
        if match[1] or match[2]:
            if literal not in candidate:
                return "literal_missing"
        elif not _literal_term(literal, new):
            return "literal_missing"

    if _VAGUE_NUMERAL.search(old):
        return "vague_numeral"
    old_neg = bool(_NEGATION.search(_NEGATION_COMPARISON_STRIP.sub(" ", old)))
    new_neg = bool(_NEGATION.search(_NEGATION_COMPARISON_STRIP.sub(" ", new)))
    if old_neg != new_neg:
        return "negation_mismatch"

    old_qty = _extract_quantities(old)
    new_qty = _extract_quantities(new)
    if _VAGUE_ENGLISH.search(old) != _VAGUE_ENGLISH.search(new) or (
        (_VAGUE_ENGLISH.search(old) or _VAGUE_ENGLISH.search(new))
        and (old_qty or new_qty)
    ):
        return "vague_numeral"

    if Counter((q.value, q.unit) for q in old_qty) != Counter((q.value, q.unit) for q in new_qty):
        return "numeric_or_unit_mismatch"

    if _extract_comparisons(old, old_qty) != _extract_comparisons(new, new_qty):
        return "comparison_mismatch"

    old_bind, old_assign = _extract_clause_bindings(old)
    new_bind, new_assign = _extract_clause_bindings(new)
    if old_assign != new_assign:
        return "binding_mismatch"
    shared_anchors = set(old_bind) & set(new_bind)
    if len(shared_anchors) >= 2 and any(old_bind[a] != new_bind[a] for a in shared_anchors):
        return "binding_mismatch"
    return None


def _valid_rewrite(original: str, candidate: str, target: str) -> bool:
    return _validate_rewrite_constraints(original, candidate, target) is None


def _identity(source) -> _Identity:
    return source.document_id, source.version, source.source_id


def _candidate_id(source: SourceKey) -> str:
    encoded = json.dumps(_identity(source), ensure_ascii=False, separators=(",", ":")).encode()
    return f"candidate_{hashlib.sha256(encoded).hexdigest()[:24]}"


def _jev_candidate(item: StructureCandidate) -> JevCandidate:
    return JevCandidate(_candidate_id(item.source), item.title, item.text)


def _candidate_trace(item: StructureCandidate) -> dict[str, Any]:
    return {
        "candidate_id": _candidate_id(item.source),
        "source": item.source.model_dump(),
        "match_kind": item.match_kind,
        "reasons": [reason.model_dump() for reason in item.reasons],
    }


def _pin(doc) -> tuple[str, int, int, str]:
    return doc.course_id, doc.version, doc.publication_epoch, doc.content_sha256


def _pin_trace(doc) -> dict[str, Any]:
    return doc.model_dump(
        include={"course_id", "document_id", "version", "publication_epoch", "content_sha256"}
    )
