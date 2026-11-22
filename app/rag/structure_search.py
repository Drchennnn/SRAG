"""Scope resolution and lexical structure search over a fixed local snapshot."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.local import SourceKey
from app.stores.structure_snapshot import (
    SnapshotDocument,
    SnapshotRequest,
    SnapshotSource,
    StructureSnapshotStore,
)

SearchStatus = Literal[
    "ok",
    "no_text_anchor",
    "ambiguous_scope",
    "scope_not_found",
    "location_not_found",
    "empty",
]

_GLOSSARY_VERSION = "c02-public-v1"
_GLOSSARY = (
    ("岭回归", "ridge regression", "ridge"),
    ("快速傅里叶变换", "fast fourier transform", "fft"),
)
_DOTTED_IDENTIFIER = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
_NEGATION = re.compile(r"(?:不是|不要|没有|\bnot\b|\bwithout\b)", re.IGNORECASE)
_EN_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "between", "both", "by",
        "can", "could", "did", "do", "does", "doing", "done", "during", "each", "for", "from",
        "had", "has", "have", "having", "how", "if", "in", "into", "is", "it", "its", "may",
        "might", "more", "most", "no", "not", "of", "on", "or", "our", "out", "respectively",
        "should", "so", "such", "than", "that", "the", "their", "them", "then", "there",
        "these", "they", "this", "those", "to", "two", "under", "used", "using", "was", "we",
        "were", "what", "when", "where", "which", "who", "why", "will", "with", "without",
        "would",
    }
)
_FTS_REASON_CODES = {
    "native_text": "fts_native",
    "legacy_text": "fts_legacy",
    "enhancement": "fts_enhancement",
    "ocr": "fts_ocr",
}


class StructureScope(BaseModel):
    course_ids: list[str] = Field(min_length=1)
    document_ids: list[str] = Field(default_factory=list)
    file_names: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def clean_scope(self) -> StructureScope:
        self.course_ids = _clean(self.course_ids)
        self.document_ids = _clean(self.document_ids)
        self.file_names = _clean(self.file_names)
        if not self.course_ids:
            raise ValueError("course_ids must contain a non-empty course")
        return self


class StructureLocation(BaseModel):
    page: int | None = Field(default=None, ge=1)
    page_range: tuple[int, int] | None = None
    cell_index: int | None = Field(default=None, ge=0)
    cell_range: tuple[int, int] | None = None
    cell_id: str | None = None

    @model_validator(mode="after")
    def one_location(self) -> StructureLocation:
        selected = [
            self.page is not None,
            self.page_range is not None,
            self.cell_index is not None,
            self.cell_range is not None,
            bool(self.cell_id and self.cell_id.strip()),
        ]
        if sum(selected) != 1:
            raise ValueError("exactly one page or cell location is required")
        if self.page_range is not None:
            _validate_range(self.page_range, maximum_width=20, name="page_range", base=1)
        if self.cell_range is not None:
            _validate_range(self.cell_range, maximum_width=50, name="cell_range", base=0)
        if self.cell_id is not None:
            self.cell_id = self.cell_id.strip()
        return self


class StructureSearchRequest(BaseModel):
    query: str = ""
    scope: StructureScope
    location: StructureLocation | None = None
    top_k: int = Field(default=8, ge=1, le=50)
    include_adjacency: bool = True
    include_page_fallback: bool = True

    @model_validator(mode="after")
    def query_or_location(self) -> StructureSearchRequest:
        self.query = self.query.strip()
        if not self.query and self.location is None:
            raise ValueError("query is required when location is absent")
        return self


class PublicDocument(BaseModel):
    course_id: str
    document_id: str
    version: int
    publication_epoch: int
    content_sha256: str
    file_name: str
    title: str
    document_format: str


class ScopeOption(BaseModel):
    course_id: str
    document_id: str
    version: int
    file_name: str


class QueryVariant(BaseModel):
    text: str
    reason: Literal["raw", "glossary", "code_component"]


class QueryTrace(BaseModel):
    raw_query: str
    variants: list[QueryVariant]
    glossary_version: str
    negation_detected: bool


class ScopeTrace(BaseModel):
    requested_course_ids: list[str]
    effective_course_ids: list[str]
    requested_document_ids: list[str]
    requested_file_names: list[str]
    resolved_documents: list[PublicDocument]
    checked_location: dict[str, Any] | None = None


class CandidateReason(BaseModel):
    code: str
    seed: SourceKey


class StructureCandidate(BaseModel):
    source: SourceKey
    course_id: str
    file_name: str
    kind: str
    title: str
    text: str
    locator: dict[str, Any]
    state: str
    text_available: bool
    score: float | None = None
    fts_origins: list[str] = Field(default_factory=list)
    match_kind: Literal["direct", "fts", "exact_literal", "token_component", "structural", "browse"]
    location_text_matched: bool | None = None
    reasons: list[CandidateReason]


class CoverageTrace(BaseModel):
    snapshot_document_count: int
    enumerated_source_count: int
    searchable_source_count: int
    fts_candidate_cap: int
    fts_seed_count: int
    evidence_inspected_count: int
    checked_adjacency_edges: int = 0
    reserved_adjacency_slots: int = 0
    checked_positions: list[str] = Field(default_factory=list)
    returned_count: int
    truncated: bool
    truncated_by_reason: dict[str, int] = Field(default_factory=dict)
    fts_scan_truncated: bool
    full_file_read: bool = False


class StructureSearchResponse(BaseModel):
    status: SearchStatus
    snapshot: list[PublicDocument]
    scope_trace: ScopeTrace
    query_trace: QueryTrace
    options: list[ScopeOption] = Field(default_factory=list)
    candidates: list[StructureCandidate] = Field(default_factory=list)
    coverage: CoverageTrace
    misses: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _QueryPlan:
    trace: QueryTrace
    fts_match: str | None
    dotted_literals: tuple[str, ...]
    terms: tuple[str, ...]


class StructureSearcher:
    def __init__(self, store: StructureSnapshotStore) -> None:
        self.store = store

    def search(
        self, request: StructureSearchRequest, *, allowed_course_ids: frozenset[str]
    ) -> StructureSearchResponse:
        effective = tuple(
            course for course in request.scope.course_ids if course in allowed_course_ids
        )
        plan = _query_plan(request.query)
        cap = min(request.top_k * 3, 150)
        snapshot = self.store.read(
            SnapshotRequest(
                course_ids=effective,
                document_ids=tuple(request.scope.document_ids),
                file_names=tuple(request.scope.file_names),
                fts_match=plan.fts_match if request.location is None else None,
                fts_limit=cap if request.location is None else 0,
                require_single_document=request.location is not None,
            )
        )
        trace = _scope_trace(request, effective, snapshot.documents)
        base = _base_response(request, plan.trace, snapshot, trace, cap)
        if snapshot.resolution_status == "scope_not_found":
            return base.model_copy(
                update={"status": "scope_not_found", "misses": list(snapshot.resolution_misses)}
            )
        if snapshot.resolution_status == "ambiguous_scope":
            options = _options(snapshot.documents, {d.document_id for d in snapshot.documents})
            return base.model_copy(update={"status": "ambiguous_scope", "options": options})

        if request.location is not None:
            sources, checked, missing_positions = _location_sources(
                snapshot.sources, request.location
            )
            trace.checked_location = request.location.model_dump(exclude_none=True)
            if not sources:
                misses = [_location_miss(request.location, snapshot.sources)]
                coverage = base.coverage.model_copy(update={"checked_positions": checked})
                return base.model_copy(
                    update={"status": "location_not_found", "coverage": coverage, "misses": misses}
                )
            direct_candidates = [
                _direct_candidate(source, snapshot.documents, plan) for source in sources
            ]
            inspected = len(sources)
            checked_edges = 0
            if request.include_adjacency:
                edges = _adjacency_edges(snapshot.sources, sources)
                same_position = [
                    edge
                    for edge in edges
                    if edge[0] in {"caption_target", "child_object", "same_position_object"}
                ]
                checked_edges = len(same_position)
                direct_candidates = _merge_edge_candidates(
                    direct_candidates,
                    same_position,
                    snapshot.documents,
                )
                inspected = len(direct_candidates)
            candidates = direct_candidates[: request.top_k]
            truncated_reasons = _truncated_reasons(direct_candidates[request.top_k :])
            full_file = _full_file_read(candidates, snapshot.sources, request.location)
            coverage = base.coverage.model_copy(
                update={
                    "checked_positions": checked,
                    "evidence_inspected_count": inspected,
                    "checked_adjacency_edges": checked_edges,
                    "returned_count": len(candidates),
                    "truncated": len(direct_candidates) > request.top_k,
                    "truncated_by_reason": truncated_reasons,
                    "full_file_read": full_file,
                }
            )
            return base.model_copy(
                update={
                    "status": "ok",
                    "candidates": candidates,
                    "coverage": coverage,
                    "misses": [f"location_missing:{item}" for item in missing_positions],
                }
            )

        source_map = {
            (source.document_id, source.version, source.source_id): source
            for source in snapshot.sources
        }
        seed_candidates = []
        seed_sources = []
        for seed in snapshot.fts_seeds:
            source = source_map.get((seed.document_id, seed.version, seed.source_id))
            if source is not None:
                seed_sources.append(source)
                seed_candidates.append(_fts_candidate(source, seed, snapshot.documents, plan))
        if not seed_candidates:
            return _no_text_result(request, snapshot, base)

        candidates = seed_candidates[: request.top_k]
        checked_edges = 0
        reserved_slots = 0
        inspected = len(seed_candidates)
        misses: list[str] = []
        if request.include_adjacency:
            edges = _adjacency_edges(snapshot.sources, seed_sources)
            checked_edges = len(edges)
            candidates, reserved_slots, candidate_pool = _select_with_reserved_edges(
                seed_candidates,
                edges,
                documents=snapshot.documents,
                top_k=request.top_k,
            )
            if not edges:
                misses.append("no_structural_neighbor")
        else:
            candidate_pool = seed_candidates
        inspected = len(candidate_pool)
        truncated_reasons = _truncated_reasons(
            [
                candidate
                for candidate in candidate_pool
                if _candidate_tuple(candidate)
                not in {_candidate_tuple(item) for item in candidates}
            ]
        )
        coverage = base.coverage.model_copy(
            update={
                "fts_seed_count": len(snapshot.fts_seeds),
                "evidence_inspected_count": inspected,
                "checked_adjacency_edges": checked_edges,
                "reserved_adjacency_slots": reserved_slots,
                "returned_count": len(candidates),
                "truncated": len(candidate_pool) > len(candidates),
                "truncated_by_reason": truncated_reasons,
                "checked_positions": _candidate_positions(candidate_pool),
            }
        )
        return base.model_copy(
            update={
                "status": "ok" if candidates else "empty",
                "candidates": candidates,
                "coverage": coverage,
                "misses": misses,
            }
        )


def _query_plan(raw_query: str) -> _QueryPlan:
    normalized = unicodedata.normalize("NFKC", raw_query).strip()
    variants: list[QueryVariant] = []
    if normalized:
        variants.append(QueryVariant(text=normalized, reason="raw"))
    folded = normalized.casefold()
    for group in _GLOSSARY:
        if any(_alias_present(folded, alias.casefold()) for alias in group):
            variants.extend(QueryVariant(text=alias, reason="glossary") for alias in group)
    dotted = tuple(dict.fromkeys(_DOTTED_IDENTIFIER.findall(normalized)))
    for literal in dotted:
        variants.extend(
            QueryVariant(text=part, reason="code_component") for part in literal.split(".")
        )
    deduped: list[QueryVariant] = []
    seen: set[str] = set()
    for variant in variants:
        key = variant.text.casefold()
        if key not in seen and len(deduped) < 16:
            seen.add(key)
            deduped.append(variant)
    terms: list[str] = []
    for variant in deduped:
        terms.extend(re.findall(r"[^\W_]+", variant.text, flags=re.UNICODE))
    unique_terms = tuple(dict.fromkeys(term.casefold() for term in terms if term.strip()))
    content_terms = tuple(term for term in unique_terms if term not in _EN_STOPWORDS)
    bm25_terms = content_terms or unique_terms
    match = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in bm25_terms)
    trace = QueryTrace(
        raw_query=raw_query,
        variants=deduped,
        glossary_version=_GLOSSARY_VERSION,
        negation_detected=bool(_NEGATION.search(normalized)),
    )
    return _QueryPlan(trace, match or None, dotted, unique_terms)


def _alias_present(query: str, alias: str) -> bool:
    if re.search(r"[\u3400-\u9fff]", alias):
        return alias in query
    return re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", query) is not None


def _base_response(request, query_trace, snapshot, scope_trace, cap):
    public = [_public_document(document) for document in snapshot.documents]
    return StructureSearchResponse(
        status="empty",
        snapshot=public,
        scope_trace=scope_trace,
        query_trace=query_trace,
        coverage=CoverageTrace(
            snapshot_document_count=len(snapshot.documents),
            enumerated_source_count=len(snapshot.sources),
            searchable_source_count=snapshot.searchable_source_count,
            fts_candidate_cap=cap,
            fts_seed_count=len(snapshot.fts_seeds),
            evidence_inspected_count=0,
            returned_count=0,
            truncated=False,
            fts_scan_truncated=snapshot.fts_scan_truncated,
        ),
    )


def _scope_trace(request, effective, documents):
    return ScopeTrace(
        requested_course_ids=request.scope.course_ids,
        effective_course_ids=list(effective),
        requested_document_ids=request.scope.document_ids,
        requested_file_names=[Path(name).name for name in request.scope.file_names],
        resolved_documents=[_public_document(document) for document in documents],
    )


def _public_document(document: SnapshotDocument) -> PublicDocument:
    return PublicDocument(
        course_id=document.course_id,
        document_id=document.document_id,
        version=document.version,
        publication_epoch=document.publication_epoch,
        content_sha256=document.content_sha256,
        file_name=document.file_name,
        title=document.title,
        document_format=document.document_format,
    )


def _options(documents, selected_ids):
    return [
        ScopeOption(
            course_id=document.course_id,
            document_id=document.document_id,
            version=document.version,
            file_name=document.file_name,
        )
        for document in documents
        if document.document_id in selected_ids
    ]


def _location_sources(sources, location):
    selected: list[SnapshotSource] = []
    checked: list[str] = []
    for source in sources:
        if source.kind not in {"page", "cell"}:
            continue
        locator = source.locator
        page = _integer(locator.get("page"))
        cell_index = _integer(locator.get("cell_index"))
        cell_id = locator.get("cell_id")
        matched = False
        if location.page is not None:
            matched = page == location.page
        elif location.page_range is not None and page is not None:
            matched = location.page_range[0] <= page <= location.page_range[1]
        elif location.cell_index is not None:
            matched = cell_index == location.cell_index
        elif location.cell_range is not None and cell_index is not None:
            matched = location.cell_range[0] <= cell_index <= location.cell_range[1]
        elif location.cell_id is not None:
            matched = isinstance(cell_id, str) and cell_id == location.cell_id
        if matched:
            selected.append(source)
            position = f"page:{page}" if page is not None else f"cell:{cell_index}"
            if position not in checked:
                checked.append(position)
    if location.cell_id is not None:
        missing = [] if selected else [f"cell_id:{location.cell_id}"]
    else:
        missing = [item for item in _requested_positions(location) if item not in checked]
    return selected, checked, missing


def _direct_candidate(source, documents, plan):
    document = _document_for(source.document_id, documents)
    key = SourceKey(
        document_id=source.document_id, version=source.version, source_id=source.source_id
    )
    return StructureCandidate(
        source=key,
        course_id=source.course_id,
        file_name=document.file_name,
        kind=source.kind,
        title=source.title,
        text=source.text,
        locator=source.locator,
        state=source.state,
        text_available=bool(source.text.strip()),
        match_kind="direct",
        location_text_matched=(
            None
            if not plan.trace.raw_query.strip()
            else any(term in f"{source.title}\n{source.text}".casefold() for term in plan.terms)
        ),
        reasons=[CandidateReason(code="direct_location", seed=key)],
    )


def _fts_candidate(source, seed, documents, plan):
    document = _document_for(source.document_id, documents)
    key = SourceKey(
        document_id=source.document_id, version=source.version, source_id=source.source_id
    )
    origins = list(dict.fromkeys(match.origin for match in seed.matches))
    literal_exact = any(
        _contains_literal(f"{match.title}\n{match.body}", literal)
        for literal in plan.dotted_literals
        for match in seed.matches
    )
    match_kind = (
        "exact_literal" if literal_exact else ("token_component" if plan.dotted_literals else "fts")
    )
    return StructureCandidate(
        source=key,
        course_id=source.course_id,
        file_name=document.file_name,
        kind=source.kind,
        title=source.title,
        text=source.text,
        locator=source.locator,
        state=source.state,
        text_available=any(match.body.strip() for match in seed.matches),
        score=-seed.rank,
        fts_origins=origins,
        match_kind=match_kind,
        reasons=[CandidateReason(code=_fts_reason(origin), seed=key) for origin in origins],
    )


def _structural_candidate(source, seed, reason, documents):
    document = _document_for(source.document_id, documents)
    return StructureCandidate(
        source=_source_key(source),
        course_id=source.course_id,
        file_name=document.file_name,
        kind=source.kind,
        title=source.title,
        text=source.text,
        locator=source.locator,
        state=source.state,
        text_available=bool(source.text.strip()),
        match_kind="structural",
        reasons=[CandidateReason(code=reason, seed=_source_key(seed))],
    )


def _browse_candidate(source, documents):
    document = _document_for(source.document_id, documents)
    key = _source_key(source)
    return StructureCandidate(
        source=key,
        course_id=source.course_id,
        file_name=document.file_name,
        kind=source.kind,
        title=source.title,
        text=source.text,
        locator=source.locator,
        state=source.state,
        text_available=bool(source.text.strip()),
        match_kind="browse",
        reasons=[CandidateReason(code="page_browse_fallback", seed=key)],
    )


def _adjacency_edges(sources, seeds):
    source_map = {_source_tuple(source): source for source in sources}
    children: dict[tuple[str, int, str], list[SnapshotSource]] = {}
    positions: dict[tuple[str, int, str, int], list[SnapshotSource]] = {}
    bases: dict[tuple[str, int, str], list[SnapshotSource]] = {}
    for source in sources:
        if source.parent_id:
            children.setdefault((source.document_id, source.version, source.parent_id), []).append(
                source
            )
        position = _position_tuple(source)
        if position is not None:
            positions.setdefault(position, []).append(source)
        if source.kind in {"page", "cell"}:
            bases.setdefault((source.document_id, source.version, source.kind), []).append(source)
    for values in (*children.values(), *positions.values(), *bases.values()):
        values.sort(key=lambda item: (item.ordinal, item.source_id))

    def targets(reason, seed):
        if reason == "parent_context" and seed.parent_id:
            parent = source_map.get((seed.document_id, seed.version, seed.parent_id))
            return [parent] if parent is not None else []
        if reason == "caption_target":
            target_id = seed.locator.get("caption_target_source_id")
            if not isinstance(target_id, str) or not target_id:
                return []
            target = source_map.get((seed.document_id, seed.version, target_id))
            seed_page = _page_number(seed)
            if (
                target is None
                or target.kind != "page_object"
                or seed_page is None
                or _page_number(target) != seed_page
            ):
                return []
            return [target]
        if reason == "child_object":
            return children.get((seed.document_id, seed.version, seed.source_id), [])
        if reason == "same_position_object":
            position = _position_tuple(seed)
            return positions.get(position, []) if position is not None else []
        if reason in {"previous_position", "next_position"}:
            anchor = _base_anchor(seed, source_map, positions)
            if anchor is None:
                return []
            sequence = bases.get((anchor.document_id, anchor.version, anchor.kind), [])
            index = next(
                (
                    position
                    for position, item in enumerate(sequence)
                    if item.source_id == anchor.source_id
                ),
                None,
            )
            if index is None:
                return []
            target_index = index - 1 if reason == "previous_position" else index + 1
            return [sequence[target_index]] if 0 <= target_index < len(sequence) else []
        return []

    edges = []
    seen: set[tuple[str, tuple[str, int, str], tuple[str, int, str]]] = set()
    reasons = (
        "parent_context",
        "caption_target",
        "child_object",
        "same_position_object",
        "previous_position",
        "next_position",
    )
    for reason in reasons:
        for seed in seeds:
            for target in targets(reason, seed):
                if target is None or _source_tuple(target) == _source_tuple(seed):
                    continue
                identity = (reason, _source_tuple(target), _source_tuple(seed))
                if identity not in seen:
                    seen.add(identity)
                    edges.append((reason, target, seed))
    return edges


def _merge_edge_candidates(candidates, edges, documents):
    merged = [candidate.model_copy(deep=True) for candidate in candidates]
    by_key = {_candidate_tuple(candidate): candidate for candidate in merged}
    for reason, target, seed in edges:
        key = _source_tuple(target)
        candidate = by_key.get(key)
        edge_reason = CandidateReason(code=reason, seed=_source_key(seed))
        if candidate is None:
            candidate = _structural_candidate(target, seed, reason, documents)
            merged.append(candidate)
            by_key[key] = candidate
        elif not any(
            item.code == edge_reason.code and item.seed == edge_reason.seed
            for item in candidate.reasons
        ):
            candidate.reasons.append(edge_reason)
    return merged


def _select_with_reserved_edges(seed_candidates, edges, *, documents, top_k):
    if not edges:
        return seed_candidates[:top_k], 0, seed_candidates
    pool = _merge_edge_candidates(seed_candidates, edges, documents)
    by_key = {_candidate_tuple(candidate): candidate for candidate in pool}
    if top_k == 1:
        return [by_key[_candidate_tuple(seed_candidates[0])]], 0, pool
    reserved = min(max(1, top_k // 3), 4, top_k - 1)
    selected = [
        by_key[_candidate_tuple(candidate)] for candidate in seed_candidates[: top_k - reserved]
    ]
    selected_keys = {_candidate_tuple(candidate) for candidate in selected}
    seed_ranks = {
        _source_tuple(candidate.source): index
        for index, candidate in enumerate(seed_candidates)
    }
    reason_order = {
        "parent_context": 0,
        "caption_target": 1,
        "same_position_object": 2,
        "previous_position": 3,
        "next_position": 4,
        "child_object": 5,
    }
    ordered_edges = sorted(
        edges,
        key=lambda edge: (
            edge[0] == "child_object" and not bool(edge[1].text.strip()),
            seed_ranks.get(_source_tuple(edge[2]), len(seed_candidates)),
            reason_order.get(edge[0], len(reason_order)),
        ),
    )
    edge_keys = list(dict.fromkeys(_source_tuple(target) for _, target, _ in ordered_edges))
    used_edges = 0
    for key in edge_keys:
        if key in selected_keys:
            continue
        selected.append(by_key[key])
        selected_keys.add(key)
        used_edges += 1
        if used_edges == reserved:
            break
    if used_edges < reserved:
        for candidate in seed_candidates[top_k - reserved :]:
            key = _candidate_tuple(candidate)
            if key not in selected_keys:
                selected.append(by_key[key])
                selected_keys.add(key)
            if len(selected) == top_k:
                break
    return selected[:top_k], reserved, pool


def _no_text_result(request, snapshot, base):
    if (
        not request.include_adjacency
        or len(snapshot.documents) != 1
        or not request.include_page_fallback
    ):
        return base.model_copy(update={"status": "empty", "misses": ["no_fts_match"]})
    browsable = [source for source in snapshot.sources if source.kind in {"page", "cell"}]
    if not browsable:
        return base.model_copy(update={"status": "empty", "misses": ["no_browsable_position"]})
    inspected = [_browse_candidate(source, snapshot.documents) for source in browsable]
    candidates = inspected[: request.top_k]
    coverage = base.coverage.model_copy(
        update={
            "evidence_inspected_count": len(inspected),
            "returned_count": len(candidates),
            "truncated": len(inspected) > request.top_k,
            "truncated_by_reason": _truncated_reasons(inspected[request.top_k :]),
            "checked_positions": _candidate_positions(inspected),
        }
    )
    return base.model_copy(
        update={
            "status": "no_text_anchor",
            "candidates": candidates,
            "coverage": coverage,
            "misses": ["no_text_anchor", "visual_not_read"],
        }
    )


def _truncated_reasons(candidates):
    counts: dict[str, int] = {}
    for candidate in candidates:
        for reason in candidate.reasons:
            counts[reason.code] = counts.get(reason.code, 0) + 1
    return counts


def _full_file_read(candidates, sources, location):
    if location is None:
        return False
    return {_candidate_tuple(candidate) for candidate in candidates} == {
        _source_tuple(source) for source in sources
    }


def _base_anchor(seed, source_map, positions):
    if seed.kind in {"page", "cell"}:
        return seed
    if seed.parent_id:
        parent = source_map.get((seed.document_id, seed.version, seed.parent_id))
        if parent is not None and parent.kind in {"page", "cell"}:
            return parent
    position = _position_tuple(seed)
    if position is None:
        return None
    return next(
        (source for source in positions.get(position, []) if source.kind in {"page", "cell"}),
        None,
    )


def _position_tuple(source):
    page = _page_number(source)
    if page is not None:
        return (source.document_id, source.version, "page", page)
    cell = _integer(source.locator.get("cell_index"))
    if cell is not None:
        return (source.document_id, source.version, "cell", cell)
    return None


def _page_number(source):
    return _integer(source.locator.get("page"))


def _source_key(source):
    return SourceKey(
        document_id=source.document_id,
        version=source.version,
        source_id=source.source_id,
    )


def _source_tuple(source):
    return (source.document_id, source.version, source.source_id)


def _candidate_tuple(candidate):
    return (
        candidate.source.document_id,
        candidate.source.version,
        candidate.source.source_id,
    )


def _contains_literal(text: str, literal: str) -> bool:
    return re.search(rf"(?<![\w.]){re.escape(literal)}(?![\w.])", text, re.IGNORECASE) is not None


def _fts_reason(origin: str) -> str:
    return _FTS_REASON_CODES.get(origin, "fts_legacy")


def _candidate_positions(candidates):
    positions: list[str] = []
    for candidate in candidates:
        page = _integer(candidate.locator.get("page"))
        cell = _integer(candidate.locator.get("cell_index"))
        position = (
            f"page:{page}"
            if page is not None
            else (f"cell:{cell}" if cell is not None else candidate.source.source_id)
        )
        if position not in positions:
            positions.append(position)
    return positions


def _location_miss(location, sources):
    pages = sorted({_integer(source.locator.get("page")) for source in sources} - {None})
    cells = sorted({_integer(source.locator.get("cell_index")) for source in sources} - {None})
    requested = location.model_dump(exclude_none=True)
    return (
        f"location_not_found:{requested};available_pages={_bounds(pages)};"
        f"available_cells={_bounds(cells)}"
    )


def _requested_positions(location):
    if location.page is not None:
        return [f"page:{location.page}"]
    if location.page_range is not None:
        return [
            f"page:{page}" for page in range(location.page_range[0], location.page_range[1] + 1)
        ]
    if location.cell_index is not None:
        return [f"cell:{location.cell_index}"]
    if location.cell_range is not None:
        return [
            f"cell:{index}" for index in range(location.cell_range[0], location.cell_range[1] + 1)
        ]
    return [f"cell_id:{location.cell_id}"]


def _bounds(values):
    return None if not values else [values[0], values[-1]]


def _document_for(document_id, documents):
    return next(document for document in documents if document.document_id == document_id)


def _integer(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _clean(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _validate_range(value, *, maximum_width, name, base):
    start, end = value
    if start < base or end < start or end - start + 1 > maximum_width:
        raise ValueError(f"invalid {name}")
