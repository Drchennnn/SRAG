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
    include_adjacency: bool = False
    include_page_fallback: bool = True

    @model_validator(mode="after")
    def query_or_location(self) -> StructureSearchRequest:
        self.query = self.query.strip()
        if not self.query and self.location is None:
            raise ValueError("query is required when location is absent")
        if self.include_adjacency:
            raise ValueError("structure adjacency is not available until C03")
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
    match_kind: Literal["direct", "fts", "exact_literal", "token_component"]
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
    checked_positions: list[str] = Field(default_factory=list)
    returned_count: int
    truncated: bool
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
            sources, checked = _location_sources(snapshot.sources, request.location)
            trace.checked_location = request.location.model_dump(exclude_none=True)
            if not sources:
                misses = [_location_miss(request.location, snapshot.sources)]
                coverage = base.coverage.model_copy(update={"checked_positions": checked})
                return base.model_copy(
                    update={"status": "location_not_found", "coverage": coverage, "misses": misses}
                )
            candidates = [
                _direct_candidate(source, snapshot.documents, plan)
                for source in sources[: request.top_k]
            ]
            coverage = base.coverage.model_copy(
                update={
                    "checked_positions": checked,
                    "evidence_inspected_count": len(sources),
                    "returned_count": len(candidates),
                    "truncated": len(sources) > request.top_k,
                }
            )
            return base.model_copy(
                update={"status": "ok", "candidates": candidates, "coverage": coverage}
            )

        source_map = {
            (source.document_id, source.version, source.source_id): source
            for source in snapshot.sources
        }
        candidates = []
        for seed in snapshot.fts_seeds[: request.top_k]:
            source = source_map.get((seed.document_id, seed.version, seed.source_id))
            if source is not None:
                candidates.append(_fts_candidate(source, seed, snapshot.documents, plan))
        coverage = base.coverage.model_copy(
            update={
                "fts_seed_count": len(snapshot.fts_seeds),
                "evidence_inspected_count": len(snapshot.fts_seeds),
                "returned_count": len(candidates),
                "truncated": len(snapshot.fts_seeds) > request.top_k,
                "checked_positions": _candidate_positions(candidates),
            }
        )
        return base.model_copy(
            update={
                "status": "ok" if candidates else "empty",
                "candidates": candidates,
                "coverage": coverage,
                "misses": [] if candidates else ["no_fts_match"],
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
    match = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in unique_terms)
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
    checked = _requested_positions(location)
    for source in sources:
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
            if (page is not None and source.kind == "page") or (
                cell_index is not None and source.kind == "cell"
            ):
                selected.append(source)
            position = f"page:{page}" if page is not None else f"cell:{cell_index}"
            if position not in checked:
                checked.append(position)
    return selected, checked


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
        reasons=[CandidateReason(code=f"fts_{origin}", seed=key) for origin in origins],
    )


def _contains_literal(text: str, literal: str) -> bool:
    return re.search(rf"(?<![\w.]){re.escape(literal)}(?![\w.])", text, re.IGNORECASE) is not None


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
