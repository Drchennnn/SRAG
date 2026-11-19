"""One literal follow-up retrieval over the authorized, pinned first-hop publications."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from sqlite3 import Error as SQLiteError
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.rag.structure_search import (
    PublicDocument,
    ScopeOption,
    StructureSearcher,
    StructureSearchRequest,
    _query_plan,
)
from app.schemas.local import SourceKey
from app.schemas.turn_evidence import ReadStatus, SourceRevision, TurnEvidence, TurnRead, TurnScope
from app.stores.local_library import LocalLibraryStore
from app.stores.structure_snapshot import SnapshotDocument, SnapshotRequest, StructureSnapshotStore

Strategy = Literal["single", "supplement"]
Requirement = Literal["definition", "experiment"]
SourceIdentity = tuple[str, int, str]
Access = frozenset[str] | Callable[[], frozenset[str]]

_DEFINITION = re.compile(
    r"^[ \t]*(?:#+[ \t]+)?(?:Definition|定义)[ \t]*[:：][ \t]*(\S[^\r\n]*)",
    re.IGNORECASE | re.MULTILINE,
)
_EXPERIMENT = re.compile(
    r"(?:^|(?<=[.;\n]))[ \t]*(?:Experiment[- ]ID|实验(?:编号|代号))\s*[:：]\s*"
    r"([A-Za-z][A-Za-z0-9]{4,39})(?![\w-])",
    re.IGNORECASE,
)
_FIELDS = re.compile(
    r"(?:^|(?<=[.;\n]))[ \t]*(Split|Metric|Unit|Baseline|Trial|Samples)[ \t]*:[ \t]*",
    re.IGNORECASE,
)
_BLOCK_BREAK = re.compile(r"\r?\n[ \t]*\r?\n|(?m:^[ \t]*(?=#{1,6}[ \t]+\S))")
_FIELD_DELIMITER = re.compile(r"[;\r\n]|\.(?=\s|$)")
_UNOBSERVED = re.compile(
    r"\b(?:experiment(?:al)? plan|planned|expected|proposal|proposed|hypothetical|anticipated|"
    r"unobserved|unmeasured|"
    r"not\s+(?:observed|measured|performed|run)|never\s+(?:performed|measured|run)|"
    r"no\s+(?:values?|results?|measurements?)\s+(?:were|was|have been)\s+"
    r"(?:observed|recorded|measured))\b|"
    r"(?:实验计划|实验提案|实验设想|拟开展|假设(?:的)?(?:结果|数值|数据)|"
    r"预期|预计|预估|尚?未(?:运行|实测|测量|观测|执行)|未有实际结果)",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)")
_WORD = re.compile(r"[^\W_]+")
_SPLITS = {
    "held-out": re.compile(r"留出(?:集)?|\b(?:held|hold)[- ]?out\b", re.IGNORECASE),
    "training": re.compile(r"训练(?:集|数据)|\btrain(?:ing)?(?:[- ]set)?\b", re.IGNORECASE),
    "validation": re.compile(r"验证集|\bvalid(?:ation)?[- ](?:set|split|data)\b", re.IGNORECASE),
    "test": re.compile(r"测试集|\btest[- ](?:set|split|data)\b", re.IGNORECASE),
}
_METRICS = {
    "mae": re.compile(r"\b(?:MAE|mean absolute error)\b|平均绝对误差", re.IGNORECASE),
    "rmse": re.compile(r"\b(?:RMSE|root mean square error)\b|均方根误差", re.IGNORECASE),
    "mse": re.compile(r"\b(?:MSE|mean square error)\b|均方误差", re.IGNORECASE),
    "mape": re.compile(r"\bMAPE\b|平均绝对百分比误差", re.IGNORECASE),
    "accuracy": re.compile(r"\baccuracy\b|准确率", re.IGNORECASE),
    "precision": re.compile(r"\bprecision\b|精确率", re.IGNORECASE),
    "recall": re.compile(r"\brecall\b|召回率", re.IGNORECASE),
    "f1": re.compile(r"\bF1\b", re.IGNORECASE),
}
_NEGATES_CONDITION = re.compile(
    r"(?:\b(?:not|without|exclude|except)|不要|不(?:用|是|采用|使用)|排除)"
    r"[ \t]*(?:(?:use|using|the|only)[ \t]+)?$",
    re.IGNORECASE,
)
_TOPIC_PARTS = re.compile(r"[a-z0-9]+(?:[ \t._-]+[a-z0-9]+)*|[\u3400-\u9fff]{2,40}")
_ZH_TOPIC_PREFIX = re.compile(r"^(?:请问|请给出|请说明|解释|什么是)[ \t]*")
_EXPLICIT_TOPIC = re.compile(
    r"\b(?:definition\s+of|define)\s+(.+?)(?=[?？.!。;；,，\n]|"
    r"\b(?:and|then|while|versus|vs|compared (?:with|to)|rather than|in contrast to)\b|$)",
    re.IGNORECASE,
)
_ZH_EXPLICIT_TOPIC = re.compile(r"(?:^|[?？.!。;；,，\n])\s*([^?？.!。;；,，\n]+?)\s*(?:的)?定义")
_UNRESOLVED_TOPIC = re.compile(
    r"[()\[\]{}【】「」『』、]|"
    r"相对于|相较于|相比|比起|对比|对照|比较|区别|不同于|而非|并非|不是|排除|"
    r"以及|或者|和|与|或|"
    r"\b(?:or|not|excluding|as opposed to|instead of)\b",
    re.IGNORECASE,
)
_QUESTION_WORDS = frozenset(
    {
        "the",
        "of",
        "is",
        "with",
        "in",
        "on",
        "for",
        "are",
        "to",
        "a",
        "an",
        "can",
        "does",
        "what",
        "which",
        "how",
        "and",
        "definition",
        "define",
        "validate",
        "validation",
        "experiment",
        "held",
        "hold",
        "out",
        "measure",
        "compare",
    }
)
_MAX_EXCERPT = 16_000
_LOCATOR_FIELDS = {
    "page",
    "cell_index",
    "cell_id",
    "output_index",
    "json_pointer",
    "bbox",
    "shape_path",
    "shape_id",
    "parent_source_id",
}


class PinnedDocument(BaseModel):
    course_id: str
    document_id: str
    version: int
    content_sha256: str
    publication_epoch: int
    file_name: str


class CandidateTrace(BaseModel):
    source: SourceKey
    course_id: str
    file_name: str
    locator: dict[str, Any] = Field(default_factory=dict)
    rank: int
    score: float | None = None
    origins: list[str] = Field(default_factory=list)
    read_status: ReadStatus = "not_attempted"
    evidence_id: str | None = None
    disposition: str = "candidate"
    reason: str | None = None


class HopTrace(BaseModel):
    kind: Literal["first", "literal_supplement"]
    query: str
    scope: TurnScope
    elapsed_ms: float = 0.0
    candidates: list[CandidateTrace] = Field(default_factory=list)


class TermAnchor(BaseModel):
    word: str
    source: SourceKey
    course_id: str
    evidence_id: str
    excerpt_sha256: str
    start: int
    end: int


class ExperimentValues(BaseModel):
    experiment_id: str
    split: str
    metric: str
    unit: str
    baseline: str
    trial: str
    samples: int


class EvidenceSupport(BaseModel):
    requirement: Requirement
    evidence_id: str
    actual_source: SourceKey
    quote: str
    start: int
    end: int
    experiment: ExperimentValues | None = None


class Replacement(BaseModel):
    removed: SourceRevision
    added: SourceRevision
    removed_locator: dict[str, Any]
    original_role: Literal["auxiliary"] = "auxiliary"
    reason: Literal["required_experiment_priority"] = "required_experiment_priority"


class SupplementReport(BaseModel):
    strategy: Strategy
    question: str
    budget: int
    scope: TurnScope = Field(default_factory=TurnScope)
    snapshot: list[PinnedDocument] = Field(default_factory=list)
    hops: list[HopTrace] = Field(default_factory=list)
    reads: list[TurnEvidence] = Field(default_factory=list)
    final_evidence: list[TurnEvidence] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    first_supported: list[Requirement] = Field(default_factory=list)
    first_missing: list[Requirement] = Field(default_factory=list)
    supports: list[EvidenceSupport] = Field(default_factory=list)
    missing: list[Requirement] = Field(default_factory=list)
    term: TermAnchor | None = None
    replacements: list[Replacement] = Field(default_factory=list)
    first_search_status: str | None = None
    misses: list[str] = Field(default_factory=list)
    scope_options: list[ScopeOption] = Field(default_factory=list)
    status: Literal["complete", "partial", "no_evidence", "unavailable"] = "no_evidence"
    stop_reason: str = "single_hop"
    supplement_count: int = 0
    snapshot_recheck_count: int = 0
    source_read_count: int = 0
    elapsed_ms: float = 0.0


@dataclass(frozen=True)
class _Definition:
    evidence: TurnEvidence
    support: EvidenceSupport
    term: TermAnchor | None


class _InvalidTurn(Exception):
    def __init__(self, reason: Literal["cancelled", "not_accessible", "stale_snapshot"]):
        self.reason = reason


class SupplementRetriever:
    """Read actual current native evidence and optionally add one same-course lookup."""

    def __init__(self, library: LocalLibraryStore, snapshots: StructureSnapshotStore):
        self.library = library
        self.snapshots = snapshots
        self.first_search = StructureSearcher(snapshots)

    def retrieve(
        self,
        request: StructureSearchRequest,
        *,
        allowed_course_ids: Access,
        strategy: Strategy = "supplement",
        cancelled: Callable[[], bool] | None = None,
    ) -> SupplementReport:
        if strategy not in ("single", "supplement"):
            raise ValueError("unknown supplement strategy")
        return _Turn(self, request, allowed_course_ids, strategy, cancelled).run()


class _Turn:
    def __init__(
        self,
        retriever: SupplementRetriever,
        request: StructureSearchRequest,
        authorized: Access,
        strategy: Strategy,
        cancelled: Callable[[], bool] | None,
    ):
        self.retriever = retriever
        self.request = request
        self.authorized = authorized
        self.cancelled = cancelled
        requirements = _requirements(request.query)
        self.report = SupplementReport(
            strategy=strategy,
            question=request.query,
            budget=request.top_k,
            requirements=requirements,
            missing=requirements.copy(),
        )
        self.pins: dict[str, PinnedDocument] = {}
        self.seen: set[SourceIdentity] = set()
        self.first_traces: dict[SourceIdentity, CandidateTrace] = {}

    def run(self) -> SupplementReport:
        started = perf_counter()
        try:
            self._retrieve()
        except _InvalidTurn as exc:
            self._redact(exc.reason)
        except SQLiteError:
            self._redact("search_failed")
        self.report.elapsed_ms = _elapsed(started)
        return self.report

    def _retrieve(self) -> None:
        self._check_access()
        allowed = self._allowed()
        if not allowed.intersection(self.request.scope.course_ids):
            raise _InvalidTurn("not_accessible")
        started = perf_counter()
        first = self.retriever.first_search.search(self.request, allowed_course_ids=allowed)
        first_ms = _elapsed(started)
        self._check_access()
        if not set(first.scope_trace.effective_course_ids).issubset(self._allowed()):
            raise _InvalidTurn("not_accessible")
        self.report.first_search_status = first.status
        self.report.misses = first.misses
        self.report.scope_options = first.options
        if first.status in {"scope_not_found", "ambiguous_scope", "location_not_found"}:
            self.report.stop_reason = first.status
            return

        self.report.snapshot = [_pin(document) for document in first.snapshot]
        self.pins = {pin.document_id: pin for pin in self.report.snapshot}
        self.report.scope = _scope(self.report.snapshot)
        self._recheck()
        hop = HopTrace(
            kind="first", query=self.request.query, scope=self.report.scope, elapsed_ms=first_ms
        )
        self.report.hops.append(hop)
        for rank, candidate in enumerate(first.candidates[: self.report.budget], 1):
            pin = self._member(candidate.source, self.pins, candidate.course_id)
            trace = CandidateTrace(
                source=candidate.source,
                course_id=pin.course_id,
                file_name=pin.file_name,
                locator=_locator(candidate.locator),
                rank=rank,
                score=candidate.score,
                origins=candidate.fts_origins,
            )
            hop.candidates.append(trace)
            self.first_traces[_identity(trace.source)] = trace
            if evidence := self._read(trace, pin):
                self.report.final_evidence.append(evidence)

        variants = [variant.text for variant in first.query_trace.variants]
        definition = self._definition()
        experiments = self._initial_experiments(definition)
        self.report.first_supported = [
            requirement
            for requirement in self.report.requirements
            if (definition if requirement == "definition" else experiments)
        ]
        self.report.first_missing = [
            requirement
            for requirement in self.report.requirements
            if requirement not in self.report.first_supported
        ]
        stop = self._next_step(definition, experiments, variants)
        if stop is None:
            assert definition is not None and definition.term is not None
            stop = self._supplement(definition, experiments)
        self._finish(definition, experiments, stop)

    def _read(self, trace: CandidateTrace, pin: PinnedDocument) -> TurnEvidence | None:
        key = _identity(trace.source)
        if key in self.seen:
            trace.disposition = "duplicate"
            return None
        self.seen.add(key)
        self._recheck()
        self.report.source_read_count += 1
        read = self.retriever.library.read_source(trace.source, course_ids=(pin.course_id,))
        self._check_access()
        if read.status != "not_accessible" and (
            read.content_sha256 != pin.content_sha256
            or read.publication_epoch != pin.publication_epoch
            or not read.is_current
        ):
            raise _InvalidTurn("stale_snapshot")

        evidence = TurnEvidence(
            evidence_id=f"ev-{self.report.source_read_count}",
            revision=SourceRevision(
                source=trace.source,
                course_id=pin.course_id,
                publication_epoch=pin.publication_epoch,
                content_sha256=pin.content_sha256,
            ),
            file_name=pin.file_name,
            locator=trace.locator,
        )
        actual = read.source
        if actual is not None and actual.source_id != trace.source.source_id:
            raise _InvalidTurn("stale_snapshot")
        if read.status != "available" or actual is None:
            evidence.read_status = "failed"
            trace.reason = read.status
        elif not actual.text.strip():
            trace.reason = "native_text_unavailable"
        else:
            evidence.locator = _locator(actual.locator)
            evidence.read_status = "partial" if len(actual.text) > _MAX_EXCERPT else "succeeded"
            evidence.reads = [
                TurnRead(
                    actual_source=trace.source,
                    method="native_text",
                    status=evidence.read_status,
                    text=actual.text[:_MAX_EXCERPT],
                )
            ]
        trace.evidence_id = evidence.evidence_id
        trace.read_status = evidence.read_status
        trace.disposition = "read" if evidence.reads else "unreadable"
        self.report.reads.append(evidence)
        return evidence if evidence.reads else None

    def _definition(self) -> _Definition | None:
        if "definition" not in self.report.requirements:
            return None
        definitions = [
            result
            for evidence in self.report.final_evidence
            if (result := _definition_in(evidence, self.request.query)) is not None
        ]
        if len({candidate.evidence.revision.course_id for candidate in definitions}) > 1:
            self.report.stop_reason = "ambiguous_definition_course"
            return None
        return definitions[0] if definitions else None

    def _initial_experiments(self, definition: _Definition | None) -> list[EvidenceSupport]:
        supports: list[EvidenceSupport] = []
        if definition is None or "experiment" not in self.report.requirements:
            return supports
        for evidence in self.report.final_evidence:
            if evidence.evidence_id == definition.evidence.evidence_id:
                continue
            support, reason = _experiment_in(evidence, definition, self.request.query)
            if support:
                _add_unique_support(supports, support)
            elif reason != "no_experiment_marker":
                self.first_traces[_identity(evidence.revision.source)].reason = reason
        return supports

    def _next_step(
        self,
        definition: _Definition | None,
        experiments: list[EvidenceSupport],
        first_variants: list[str],
    ) -> str | None:
        if self.report.stop_reason == "ambiguous_definition_course":
            return self.report.stop_reason
        if definition and (experiments or self.report.requirements == ["definition"]):
            return "already_supported"
        if self.report.strategy == "single":
            return "single_hop"
        if self.report.requirements != ["definition", "experiment"] or definition is None:
            return "no_definition_anchor"
        scope = self._anchor_scope(definition)
        if self.request.location is not None or len(scope.document_ids) < 2:
            return "scope_limited"
        if self.report.budget < 2:
            return "budget_exhausted"
        if definition.term is None or _already_used(
            definition.term.word, [self.request.query, *first_variants]
        ):
            return "no_new_terms"
        return None

    def _supplement(self, definition: _Definition, experiments: list[EvidenceSupport]) -> str:
        term = definition.term
        assert term is not None
        scope = self._anchor_scope(definition)
        pins = {document_id: self.pins[document_id] for document_id in scope.document_ids}
        self._recheck()
        self.report.term = term
        self.report.supplement_count = 1
        hop = HopTrace(kind="literal_supplement", query=term.word, scope=scope)
        self.report.hops.append(hop)
        started = perf_counter()
        try:
            snapshot = self.retriever.snapshots.read(
                SnapshotRequest(
                    course_ids=tuple(scope.course_ids),
                    document_ids=tuple(scope.document_ids),
                    fts_match=f'"{term.word}"',
                    fts_limit=self.report.budget,
                ),
            )
        except SQLiteError:
            hop.elapsed_ms = _elapsed(started)
            return "search_failed"
        hop.elapsed_ms = _elapsed(started)
        self._check_access()
        if snapshot.resolution_status != "resolved" or not _same_snapshot(snapshot.documents, pins):
            raise _InvalidTurn("stale_snapshot")
        self._recheck()

        locations = {
            (source.document_id, source.version, source.source_id): _locator(source.locator)
            for source in snapshot.sources
        }
        for rank, seed in enumerate(snapshot.fts_seeds[: self.report.budget], 1):
            key = SourceKey(
                document_id=seed.document_id, version=seed.version, source_id=seed.source_id
            )
            pin = self._member(key, pins)
            trace = CandidateTrace(
                source=key,
                course_id=pin.course_id,
                file_name=pin.file_name,
                locator=locations.get(_identity(key), {}),
                rank=rank,
                score=seed.rank,
                origins=list(dict.fromkeys(match.origin for match in seed.matches)),
            )
            hop.candidates.append(trace)
            evidence = self._read(trace, pin)
            if evidence is None:
                continue
            support, reason = _experiment_in(evidence, definition, self.request.query)
            if support is None:
                trace.disposition = (
                    "mismatched" if reason.endswith("_mismatch") else "still_missing"
                )
                trace.reason = reason
            elif any(existing.quote == support.quote for existing in experiments):
                trace.disposition = "duplicate"
                trace.reason = "identical_experiment_excerpt"
            elif self._keep(evidence, definition, experiments):
                trace.disposition = "accepted"
                experiments.append(support)
            else:
                trace.disposition = "budget_exhausted"

        dispositions = {candidate.disposition for candidate in hop.candidates}
        if experiments:
            return "supplemented"
        if dispositions == {"duplicate"}:
            return "duplicate_only"
        if "mismatched" in dispositions:
            return "mismatched"
        if "still_missing" in dispositions:
            return "still_missing"
        return "no_additional_evidence"

    def _keep(
        self,
        evidence: TurnEvidence,
        definition: _Definition,
        experiments: list[EvidenceSupport],
    ) -> bool:
        final = self.report.final_evidence
        if len(final) >= self.report.budget:
            required = {definition.evidence.evidence_id, *(s.evidence_id for s in experiments)}
            for index in reversed(range(len(final))):
                if final[index].evidence_id not in required:
                    removed = final.pop(index)
                    self.report.replacements.append(
                        Replacement(
                            removed=removed.revision,
                            added=evidence.revision,
                            removed_locator=removed.locator,
                        )
                    )
                    break
            else:
                return False
        final.append(evidence)
        return True

    def _finish(
        self, definition: _Definition | None, experiments: list[EvidenceSupport], stop: str
    ) -> None:
        self._recheck()
        final_ids = {item.evidence_id for item in self.report.final_evidence}
        supports = ([definition.support] if definition else []) + experiments
        self.report.supports = [support for support in supports if support.evidence_id in final_ids]
        self.report.missing = [
            item
            for item in self.report.requirements
            if not any(support.requirement == item for support in self.report.supports)
        ]
        self.report.stop_reason = stop
        if self.report.final_evidence:
            self.report.status = (
                "complete" if self.report.requirements and not self.report.missing else "partial"
            )

    def _anchor_scope(self, definition: _Definition) -> TurnScope:
        course = definition.evidence.revision.course_id
        return _scope([pin for pin in self.report.snapshot if pin.course_id == course])

    def _recheck(self) -> None:
        self._check_access()
        if not self.pins:
            return
        self.report.snapshot_recheck_count += 1
        snapshot = self.retriever.snapshots.read(
            SnapshotRequest(
                course_ids=tuple(self.report.scope.course_ids),
                document_ids=tuple(self.report.scope.document_ids),
            ),
        )
        self._check_access()
        if snapshot.resolution_status != "resolved" or not _same_snapshot(
            snapshot.documents, self.pins
        ):
            raise _InvalidTurn("stale_snapshot")

    def _check_access(self) -> None:
        if self.cancelled is not None and self.cancelled():
            raise _InvalidTurn("cancelled")
        if not set(self.report.scope.course_ids).issubset(self._allowed()):
            raise _InvalidTurn("not_accessible")

    def _allowed(self) -> frozenset[str]:
        return self.authorized() if callable(self.authorized) else self.authorized

    @staticmethod
    def _member(
        source: SourceKey, pins: dict[str, PinnedDocument], course: str | None = None
    ) -> PinnedDocument:
        pin = pins.get(source.document_id)
        if pin is None or pin.version != source.version or (course and pin.course_id != course):
            raise _InvalidTurn("stale_snapshot")
        return pin

    def _redact(self, reason: str) -> None:
        self.report = SupplementReport(
            strategy=self.report.strategy,
            question=self.report.question,
            budget=self.report.budget,
            requirements=self.report.requirements,
            missing=self.report.requirements.copy(),
            status="unavailable",
            stop_reason=reason,
            supplement_count=self.report.supplement_count,
            snapshot_recheck_count=self.report.snapshot_recheck_count,
            source_read_count=self.report.source_read_count,
        )


def _requirements(query: str) -> list[Requirement]:
    result: list[Requirement] = []
    if re.search(r"定义|\bdefin(?:e|ition)\b", query, re.IGNORECASE):
        result.append("definition")
    if re.search(r"实验|验证|\bexperiment\b|\bvalidat(?:e|ion)\b", query, re.IGNORECASE):
        result.append("experiment")
    return result


def asks_for_definition_and_experiment(query: str) -> bool:
    omitted = re.search(
        r"(?:不要|不必|无需|不用|不需要|(?<!\w)(?:请)?别"
        r"|(?:也|但|并|可|千万|切记|务必|先|还|定义|实验|验证)别)"
        r"[^，。！？；,.;!?\n]{0,14}(?:定义|实验|验证)"
        r"|\b(?:do not|don['’]t|without|omit|exclude|skip|no)\b"
        r"[^,.!?;\n]{0,30}\b(?:definition|experiments?|validation)\b",
        query,
        re.IGNORECASE,
    )
    return not omitted and len(_requirements(query)) == 2


def _definition_in(evidence: TurnEvidence, question: str) -> _Definition | None:
    read = evidence.reads[0]
    match = _DEFINITION.search(read.text)
    if match is None or (read.status == "partial" and match.end() == len(read.text)):
        return None
    _, block_end = _block_bounds(read.text, match.start())
    anchor = _adjacent_experiment_marker(read.text, match, block_end)
    definition_end = min(match.end(1), anchor.start()) if anchor else match.end(1)
    quote = read.text[match.start(1) : definition_end].rstrip(" \t;")
    if not _mentions_definition_topic(quote, question):
        return None
    support = EvidenceSupport(
        requirement="definition",
        evidence_id=evidence.evidence_id,
        actual_source=read.actual_source,
        start=match.start(1),
        end=match.start(1) + len(quote),
        quote=quote,
    )
    term = None
    if read.status == "succeeded" and anchor is not None:
        term = TermAnchor(
            word=anchor.group(1),
            source=read.actual_source,
            course_id=evidence.revision.course_id,
            evidence_id=evidence.evidence_id,
            excerpt_sha256=hashlib.sha256(read.text.encode()).hexdigest(),
            start=anchor.start(1),
            end=anchor.end(1),
        )
    return _Definition(evidence=evidence, support=support, term=term)


def _adjacent_experiment_marker(
    text: str, definition: re.Match[str], block_end: int
) -> re.Match[str] | None:
    markers = list(_EXPERIMENT.finditer(text, definition.start(1), block_end))
    if len(markers) != 1:
        return None
    marker = markers[0]
    if marker.start() > definition.end() and text[definition.end() : marker.start()].strip():
        return None
    return marker


def _experiment_in(
    evidence: TurnEvidence, definition: _Definition, question: str
) -> tuple[EvidenceSupport | None, str]:
    read = evidence.reads[0]
    markers = list(_EXPERIMENT.finditer(read.text))
    if not markers:
        return None, "no_experiment_marker"
    base = definition.evidence.revision
    if evidence.revision.course_id != base.course_id:
        return None, "experiment_course_mismatch"
    if evidence.revision.source.document_id == base.source.document_id:
        return None, "experiment_document_mismatch"
    if read.status != "succeeded":
        return None, "experiment_read_incomplete"
    anchor = definition.term
    if anchor is None:
        return None, "no_definition_term"
    matching = [(idx, marker) for idx, marker in enumerate(markers) if marker[1] == anchor.word]
    if not matching:
        return None, "experiment_id_mismatch"
    if len(matching) != 1:
        return None, "experiment_record_ambiguous"

    index, marker = matching[0]
    block_start, block_end = _block_bounds(read.text, marker.start())
    next_marker = markers[index + 1].start() if index + 1 < len(markers) else len(read.text)
    segment = read.text[marker.end() : min(block_end, next_marker)]
    context = read.text if len(markers) == 1 else read.text[block_start:block_end]
    if _UNOBSERVED.search(context):
        return None, "experiment_not_observed"

    fields, field_end, problem = _experiment_fields(segment)
    if problem:
        return None, problem
    if not all(
        fields.get(name) for name in ("split", "metric", "unit", "baseline", "trial", "samples")
    ):
        return None, "experiment_fields_missing"
    if problem := _condition_problem(question, fields):
        return None, problem
    if not all(_NUMBER.fullmatch(fields[name]) for name in ("baseline", "trial")):
        return None, "experiment_values_missing"
    samples = fields["samples"]
    if not re.fullmatch(r"[1-9][0-9]{0,8}", samples):
        return None, "experiment_samples_missing"

    quote_start = marker.start() + max(0, len(marker[0]) - len(marker[0].lstrip()))
    quote_end = marker.end() + field_end
    if read.text[quote_end : quote_end + 1] in (".", ";"):
        quote_end += 1
    return (
        EvidenceSupport(
            requirement="experiment",
            evidence_id=evidence.evidence_id,
            actual_source=read.actual_source,
            quote=read.text[quote_start:quote_end],
            start=quote_start,
            end=quote_end,
            experiment=ExperimentValues(experiment_id=anchor.word, **fields),
        ),
        "",
    )


def _block_bounds(text: str, offset: int) -> tuple[int, int]:
    start = 0
    for boundary in _BLOCK_BREAK.finditer(text):
        if boundary.start() > offset:
            return start, boundary.start()
        start = boundary.end()
    return start, len(text)


def _experiment_fields(segment: str) -> tuple[dict[str, str], int, str | None]:
    fields: dict[str, str] = {}
    field_end = 0
    for field in _FIELDS.finditer(segment):
        name = field[1].lower()
        if name in fields:
            return {}, 0, "experiment_record_ambiguous"
        if segment[field_end : field.start()].strip(" \t\r\n.;"):
            return {}, 0, "experiment_fields_disconnected"
        suffix = segment[field.end() :]
        delimiter = _FIELD_DELIMITER.search(suffix)
        value = suffix if delimiter is None else suffix[: delimiter.start()]
        fields[name] = value.strip()
        field_end = field.end() + len(value.rstrip())
    return fields, field_end, None


def _condition_problem(question: str, fields: dict[str, str]) -> str | None:
    for field, aliases in (("split", _SPLITS), ("metric", _METRICS)):
        named = {
            label: list(pattern.finditer(question))
            for label, pattern in aliases.items()
            if pattern.search(question)
        }
        if len(named) > 1 or any(
            _NEGATES_CONDITION.search(question[: match.start()])
            for matches in named.values()
            for match in matches
        ):
            return "experiment_condition_ambiguous"
        if named:
            requested = next(iter(named))
            actual = fields[field]
            if actual.casefold() != requested and not aliases[requested].fullmatch(actual):
                return f"experiment_{field}_mismatch"
    return None


def _add_unique_support(items: list[EvidenceSupport], new: EvidenceSupport) -> None:
    if not any(item.quote == new.quote for item in items):
        items.append(new)


def _already_used(word: str, queries: list[str]) -> bool:
    return any(
        word.casefold() in _WORD.findall(unicodedata.normalize("NFKC", query).casefold())
        for query in queries
    )


def _mentions_definition_topic(quote: str, question: str) -> bool:
    normalized = unicodedata.normalize("NFKC", question)
    topics = [
        match
        for pattern in (_EXPLICIT_TOPIC, _ZH_EXPLICIT_TOPIC)
        if (match := pattern.search(normalized)) is not None
    ]
    if not topics:
        return False
    topic = min(topics, key=lambda match: match.start(1))[1].strip()
    topic = _ZH_TOPIC_PREFIX.sub("", topic).casefold()
    if _UNRESOLVED_TOPIC.search(topic):
        return False
    phrases = set(_TOPIC_PARTS.findall(topic))
    phrases.update(
        variant.text.casefold()
        for variant in _query_plan(topic).trace.variants
        if variant.reason == "glossary"
    )
    reading = unicodedata.normalize("NFKC", quote).casefold()
    reading = re.sub(r"^[\s“\"'`*]*(?:the\s+|an?\s+)?", "", reading)
    reading_words = " ".join(_WORD.findall(reading)) + " "
    for phrase in phrases:
        if not phrase.isascii():
            if reading.startswith(phrase):
                return True
            continue
        words = " ".join(word for word in _WORD.findall(phrase) if word not in _QUESTION_WORDS)
        if words and reading_words.startswith(words + " "):
            return True
    return False


def _pin(document: PublicDocument | SnapshotDocument) -> PinnedDocument:
    return PinnedDocument(
        course_id=document.course_id,
        document_id=document.document_id,
        version=document.version,
        content_sha256=document.content_sha256,
        publication_epoch=document.publication_epoch,
        file_name=document.file_name,
    )


def _same_snapshot(documents: Sequence[SnapshotDocument], pins: dict[str, PinnedDocument]) -> bool:
    def identity(pin: PinnedDocument) -> tuple[str, int, str, int]:
        return (pin.course_id, pin.version, pin.content_sha256, pin.publication_epoch)

    return len(documents) == len(pins) and {
        doc.document_id: identity(_pin(doc)) for doc in documents
    } == {key: identity(pin) for key, pin in pins.items()}


def _scope(pins: list[PinnedDocument]) -> TurnScope:
    return TurnScope(
        course_ids=list(dict.fromkeys(pin.course_id for pin in pins)),
        document_ids=[pin.document_id for pin in pins],
    )


def _locator(locator: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in locator.items() if key in _LOCATOR_FIELDS}


def _identity(source: SourceKey) -> SourceIdentity:
    return (source.document_id, source.version, source.source_id)


def _elapsed(started: float) -> float:
    return (perf_counter() - started) * 1000
