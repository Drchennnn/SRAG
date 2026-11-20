"""Bounded bilingual retrieval and Jev ranking over one authorized publication snapshot."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import unicodedata
from collections import Counter, OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
)
from app.schemas.local import SourceKey
from app.schemas.workspace import WorkspaceError
from app.stores.local_library import LocalLibraryStore
from app.stores.structure_snapshot import SnapshotRequest, StructureSnapshot

RetrievalStrategy = Literal["fts", "bilingual", "jev", "expanded"]
_Identity = tuple[str, int, str]
_SCOPED_FAILURES = {"ambiguous_scope", "scope_not_found", "location_not_found"}
_HAN = re.compile(r"[\u3400-\u9fff]")
_NUMBER = r"[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?"
_NUMBERS = re.compile(rf"{_NUMBER}(?:\s*%)?", re.IGNORECASE)
_OPERATORS = re.compile(r"<=|>=|==|!=|<|>|=")
_NEGATION = re.compile(
    r"排除|不(?:要|含|包括|使用|能|可|是|同|考虑|比较|属于)|无(?:需|法)|未|没有|"
    r"\b(?:no|not|never|without|excluding|exclude|neither|none|cannot|don't|doesn't|"
    r"didn't|isn't|mustn't|non-)\b",
    re.IGNORECASE,
)
_LITERAL = re.compile(
    r"`([^`\n]{1,160})`|[\"“]([^\"”\n]{1,160})[\"”]|"
    r"(?<![\w.])(?:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+|[A-Za-z]\w*_\w+|"
    r"[A-Z][A-Z0-9]*-\w+|[vV]\d+(?:\.\d+)*)(?![\w.])"
)
_ANCHOR = re.compile(
    r"(?<![\w.-])(?:[A-Z]{2,}(?:[-_.][A-Za-z0-9]+)+|[A-Z]+\d[A-Za-z0-9_-]*|"
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)(?![\w-]|\.[A-Za-z0-9_])"
)
_UNIT_FORMS = {
    "ms": r"milliseconds?|毫秒|ms",
    "us": r"microseconds?|微秒|[μµu]s",
    "s": r"seconds?|秒|s",
    "hz": r"hertz|赫兹|hz",
    "khz": r"kilohertz|千赫(?:兹)?|khz",
    "mhz": r"megahertz|兆赫(?:兹)?|mhz",
    "mm": r"millimeters?|毫米|mm",
    "cm": r"centimeters?|厘米|cm",
    "db": r"decibels?|分贝|db",
    "deg": r"degrees?|度|deg|°",
    "samples": r"个采样点|个样本|采样点|样本|samples?",
    "pct": r"percent(?:age)?|%",
}
_UNIT = re.compile(
    rf"({_NUMBER})\s*(?:"
    + "|".join(f"(?P<{unit}>{forms})" for unit, forms in _UNIT_FORMS.items())
    + r")(?![A-Za-z])",
    re.IGNORECASE,
)


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
        if not _valid_rewrite(query, alternate, target):
            return finish("constraints_unverified")
        self._cache[cache_key] = alternate
        if len(self._cache) > 128:
            self._cache.popitem(last=False)
        return finish("succeeded", alternate)


@dataclass(frozen=True)
class RetrievalPlan:
    initial: StructureSearchResponse
    selected: tuple[StructureCandidate, ...]
    trace: dict[str, Any]


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
    original_question: str | None = None,
) -> RetrievalPlan:
    """Reuse a single bilingual pool and enforce one final source budget in all arms."""

    if strategy not in {"fts", "bilingual", "jev", "expanded"} or not 1 <= budget <= request.top_k:
        raise ValueError("invalid workspace retrieval strategy or source budget")
    if not active():
        raise WorkspaceError("interrupted")
    question = request.query if original_question is None else original_question
    scope = _PinnedScope(searcher, snapshot, request, allowed_courses, active)
    initial = searcher.search(request, allowed_course_ids=allowed_courses)
    scope.accept_initial(initial)
    pool = list(initial.candidates)
    trace: dict[str, Any] = {
        "strategy": strategy,
        "effective_strategy": "fts",
        "budget": budget,
        "original_query": question,
        "search_query": request.query,
        "original_search_status": initial.status,
        "alternate_query": None,
        "pins": [_pin_trace(doc) for doc in initial.snapshot],
        "original_pool": [_candidate_trace(candidate) for candidate in pool],
        "rewrite": {"status": "not_requested", "external_call_count": 0},
        "jev": {"status": "not_requested", "external_call_count": 0},
        "expansion": {"status": "not_requested", "external_call_count": 0},
    }

    def finish(selected: Sequence[StructureCandidate]) -> RetrievalPlan:
        scope.check_active()
        trace["candidate_pool"] = [_candidate_trace(candidate) for candidate in pool]
        trace["selected"] = [candidate.source.model_dump() for candidate in selected[:budget]]
        trace["external_call_count"] = sum(
            trace[name]["external_call_count"] for name in ("rewrite", "jev", "expansion")
        )
        return RetrievalPlan(initial, tuple(selected[:budget]), trace)

    if strategy == "fts" or request.location is not None or initial.status in _SCOPED_FAILURES:
        return finish(pool)
    if not initial.snapshot:
        return finish(pool)
    scope.verify()
    rewritten = (rewrite or BilingualRewriter(settings))(request.query, verify=scope.verify)
    scope.verify()
    trace["rewrite"] = rewritten.trace
    if rewritten.query:
        trace["alternate_query"] = rewritten.query
        alternate = scope.search(rewritten.query)
        trace["alternate_search_status"] = alternate.status
        trace["alternate_pool"] = [
            _candidate_trace(candidate) for candidate in alternate.candidates
        ]
        pool = _fuse(initial.candidates, alternate.candidates, request.top_k)
        trace["effective_strategy"] = "bilingual"
    if strategy == "bilingual":
        return finish(pool)
    scorer = scorer or JevClient(settings)
    scored, trace["jev"] = _score(scorer, question, pool, scope.verify)
    if trace["jev"]["status"] != "succeeded":
        if strategy == "expanded":
            trace["expansion"]["status"] = "unavailable_without_jev"
        return finish(pool)
    trace["effective_strategy"] = "jev"
    if strategy == "expanded":
        scored, trace["expansion"] = _expand(
            library, scope, scorer, question, scored, budget, trace
        )
        if trace["expansion"]["status"] == "succeeded":
            trace["effective_strategy"] = "expanded"
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


def _fuse(original, alternate, pool_cap: int) -> list[StructureCandidate]:
    sources: dict[_Identity, StructureCandidate] = {}
    ranks: dict[_Identity, float] = {}
    has_text_anchor = any(item.match_kind != "browse" for item in (*original, *alternate))
    for candidates in (original, alternate):
        for rank, item in enumerate(candidates, start=1):
            if has_text_anchor and item.match_kind == "browse":
                continue
            key = _identity(item.source)
            sources.setdefault(key, item)
            ranks[key] = ranks.get(key, 0) + 1 / (60 + rank)
    return [sources[key] for key in sorted(ranks, key=ranks.get, reverse=True)[:pool_cap]]


def _score(scorer, question, pool, verify, *, evidence=()):
    inputs = [_jev_candidate(item) for item in pool]
    result = scorer.score(question, inputs, evidence=evidence, verify=verify)
    verify()
    wanted = [item.candidate_id for item in inputs]
    trace = {**result.trace, "input_ids": wanted}
    if result.trace["status"] != "succeeded":
        return list(pool), trace
    if set(result.scores) != set(wanted) or any(
        type(value) not in (int, float) or not 0 <= value <= 1 for value in result.scores.values()
    ):
        return list(pool), {**trace, "status": "invalid_response"}
    trace["scores"] = {candidate_id: result.scores[candidate_id] for candidate_id in wanted}
    return sorted(
        pool, key=lambda item: result.scores[_candidate_id(item.source)], reverse=True
    ), trace


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


def _valid_rewrite(original: str, candidate: str, target: str) -> bool:
    if not candidate or len(candidate) > min(4_000, max(160, len(original) * 8)):
        return False
    if candidate.casefold() == original.casefold() or (
        target == "Chinese" and not _HAN.search(candidate)
    ):
        return False
    if target == "English" and not re.search(r"[A-Za-z]{3}", candidate):
        return False
    for match in _LITERAL.finditer(original):
        if (match[1] or match[2] or match[0]) not in candidate:
            return False
    translation = str.maketrans({"≤": "<=", "≥": ">=", "≠": "!=", "−": "-"})
    old, new = (
        unicodedata.normalize("NFKC", value).translate(translation)
        for value in (original, candidate)
    )
    if Counter(_OPERATORS.findall(old)) != Counter(_OPERATORS.findall(new)):
        return False
    if bool(_NEGATION.search(old)) != bool(_NEGATION.search(new)):
        return False
    return _numeric_constraints(old) == _numeric_constraints(new)


def _numeric_constraints(query: str):
    units = [(match[1].casefold(), match.lastgroup) for match in _UNIT.finditer(query)]
    # Percent may be written as a word, but its normalized unit must still agree.
    numbers = Counter(value.rstrip("% ").casefold() for value in _NUMBERS.findall(query))
    return numbers, Counter(units)


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
