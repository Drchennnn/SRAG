"""Deterministic answers grounded in the retained, current evidence of an F02 turn."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from decimal import Decimal
from re import Pattern
from sqlite3 import Error as SQLiteError
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.rag.evidence_supplement import (
    _METRICS,
    _SPLITS,
    Access,
    EvidenceSupport,
    ExperimentValues,
    PinnedDocument,
    Requirement,
    Strategy,
    SupplementReport,
    SupplementRetriever,
    _Definition,
    _definition_in,
    _identity,
    _locator,
    _requested_experiment_in,
    _requirements,
    _same_snapshot,
)
from app.rag.structure_search import StructureSearchRequest
from app.schemas.local import SourceKey
from app.schemas.turn_evidence import ReadStatus, SourceRevision, TurnEvidence
from app.stores.structure_snapshot import SnapshotRequest

AnswerStatus = Literal["complete", "partial", "conflicting", "no_evidence", "unavailable"]
SectionStatus = Literal["supported", "missing", "conflicting"]

_MISSING = {
    "definition": "本轮可核对的原文不足以确认定义。",
    "experiment": "本轮可核对的原文不足以确认符合所问对象、条件和指标的实测比较。",
}


class AnswerCitation(BaseModel):
    evidence_id: str
    revision: SourceRevision
    actual_source: SourceKey
    file_name: str
    locator: dict[str, Any]
    method: Literal["native_text"] = "native_text"
    read_status: ReadStatus
    start: int
    end: int
    quote: str


class AnswerStatement(BaseModel):
    text: str
    citation: AnswerCitation
    experiment: ExperimentValues | None = None


class AnswerSection(BaseModel):
    requirement: Requirement
    status: SectionStatus
    statements: list[AnswerStatement] = Field(default_factory=list)
    message: str = ""


class EvidenceAnswer(BaseModel):
    status: AnswerStatus
    sections: list[AnswerSection]
    stop_reason: str
    budget: int
    verified_source_count: int = 0
    rejected_support_count: int = 0
    message: str = ""


class AnswerRun(BaseModel):
    answer: EvidenceAnswer
    retrieval: SupplementReport | None = None


class _UnverifiedTurn(Exception):
    def __init__(self, reason: str):
        self.reason = reason


class EvidenceAnswerer:
    """Read from B and use F02's own support rules before returning factual statements."""

    def __init__(self, retriever: SupplementRetriever) -> None:
        self.retriever = retriever

    def answer(
        self,
        request: StructureSearchRequest,
        *,
        allowed_course_ids: Access,
        strategy: Strategy = "supplement",
        cancelled: Callable[[], bool] | None = None,
    ) -> AnswerRun:
        report = self.retriever.retrieve(
            request,
            allowed_course_ids=allowed_course_ids,
            strategy=strategy,
            cancelled=cancelled,
        )
        answer = self.compose(report, allowed_course_ids=allowed_course_ids, cancelled=cancelled)
        return AnswerRun(
            answer=answer,
            retrieval=None if answer.status == "unavailable" else report,
        )

    def compose(
        self,
        report: SupplementReport,
        *,
        allowed_course_ids: Access,
        cancelled: Callable[[], bool] | None = None,
    ) -> EvidenceAnswer:
        """Use an internal F02 report. Public callers start from a scoped search request."""
        requirements = _requirements(report.question)
        if report.status == "unavailable":
            return _unavailable(report, requirements, report.stop_reason)

        try:
            pins = _turn_pins(report, requirements)
            self._check_snapshot(report, pins, allowed_course_ids, cancelled)
            final = _final_reads(report)
            grounded: dict[str, TurnEvidence] = {}
            asserted_ids = {support.evidence_id for support in report.supports}
            for item in final:
                if item.evidence_id not in asserted_ids:
                    continue
                _check_access(report, allowed_course_ids, cancelled)
                if self._verify_source(item, pins):
                    grounded[item.evidence_id] = item
            sections, accepted = _sections(report, requirements, grounded)
            self._check_snapshot(report, pins, allowed_course_ids, cancelled)
        except _UnverifiedTurn as error:
            return _unavailable(report, requirements, error.reason)
        except SQLiteError:
            return _unavailable(report, requirements, "answer_verification_failed")

        return EvidenceAnswer(
            status=_status(sections),
            sections=sections,
            stop_reason=report.stop_reason,
            budget=report.budget,
            verified_source_count=len(grounded),
            rejected_support_count=len(report.supports) - accepted,
            message=(
                "本轮只呈现已核对的原文。"
                if requirements
                else "本轮按定义与实验比较提取证据，当前问题没有明确识别出这些要求。"
            ),
        )

    def _check_snapshot(
        self,
        report: SupplementReport,
        pins: dict[str, PinnedDocument],
        allowed: Access,
        cancelled: Callable[[], bool] | None,
    ) -> None:
        _check_access(report, allowed, cancelled)
        if not pins:
            return
        snapshot = self.retriever.snapshots.read(
            SnapshotRequest(
                course_ids=tuple(report.scope.course_ids),
                document_ids=tuple(report.scope.document_ids),
            )
        )
        _check_access(report, allowed, cancelled)
        if snapshot.resolution_status != "resolved" or not _same_snapshot(snapshot.documents, pins):
            raise _UnverifiedTurn("stale_snapshot")

    def _verify_source(self, evidence: TurnEvidence, pins: dict[str, PinnedDocument]) -> bool:
        revision = evidence.revision
        key = revision.source
        pin = pins.get(key.document_id)
        if pin is None or (
            pin.course_id,
            pin.version,
            pin.publication_epoch,
            pin.content_sha256,
            pin.file_name,
        ) != (
            revision.course_id,
            key.version,
            revision.publication_epoch,
            revision.content_sha256,
            evidence.file_name,
        ):
            return False

        if len(evidence.reads) != 1:
            return False
        read = evidence.reads[0]
        if (
            read.actual_source != key
            or read.method != "native_text"
            or read.execution is not None
            or read.origin_execution is not None
            or read.status not in ("succeeded", "partial")
            or evidence.read_status != read.status
            or not read.text.strip()
        ):
            return False

        current = self.retriever.library.read_source(key, course_ids=(pin.course_id,))
        if current.status == "not_accessible":
            return False
        if (
            not current.is_current
            or current.content_sha256 != pin.content_sha256
            or current.publication_epoch != pin.publication_epoch
        ):
            raise _UnverifiedTurn("stale_snapshot")
        source = current.source
        if (
            current.status != "available"
            or source is None
            or source.source_id != key.source_id
            or _locator(source.locator) != evidence.locator
        ):
            return False
        if read.status == "succeeded":
            return source.text == read.text
        return len(read.text) < len(source.text) and source.text.startswith(read.text)


def _turn_pins(
    report: SupplementReport, requirements: list[Requirement]
) -> dict[str, PinnedDocument]:
    pins = {document.document_id: document for document in report.snapshot}
    identities = {_identity(item.revision.source) for item in report.final_evidence}
    if (
        report.budget < 1
        or report.budget > 50
        or report.supplement_count not in (0, 1)
        or (report.strategy == "single" and report.supplement_count != 0)
        or len(report.snapshot) != len(pins)
        or set(report.scope.document_ids) != pins.keys()
        or set(report.scope.course_ids) != {pin.course_id for pin in pins.values()}
        or not len(report.final_evidence) == len(identities) <= report.budget
        or report.requirements != requirements
    ):
        raise _UnverifiedTurn("invalid_evidence_report")
    return pins


def _final_reads(report: SupplementReport) -> list[TurnEvidence]:
    read_ids = Counter(item.evidence_id for item in report.reads)
    final_ids = Counter(item.evidence_id for item in report.final_evidence)
    reads = {item.evidence_id: item for item in report.reads}
    return [
        item
        for item in report.final_evidence
        if (
            read_ids[item.evidence_id] == final_ids[item.evidence_id] == 1
            and reads[item.evidence_id] == item
        )
    ]


def _check_access(
    report: SupplementReport,
    allowed_course_ids: Access,
    cancelled: Callable[[], bool] | None,
) -> None:
    if cancelled is not None and cancelled():
        raise _UnverifiedTurn("cancelled")
    allowed = allowed_course_ids() if callable(allowed_course_ids) else allowed_course_ids
    if not set(report.scope.course_ids).issubset(allowed):
        raise _UnverifiedTurn("not_accessible")


def _sections(
    report: SupplementReport,
    requirements: list[Requirement],
    evidence: dict[str, TurnEvidence],
) -> tuple[list[AnswerSection], int]:
    support_rows = [
        row
        for row in report.supports
        if row.requirement in requirements
        and row.evidence_id in evidence
        and _valid_span(row, evidence[row.evidence_id])
    ]
    definitions = [
        result
        for support in support_rows
        if support.requirement == "definition"
        and (result := _definition_in(evidence[support.evidence_id], report.question)) is not None
        and result.support == support
    ]
    anchors = {_identity(match.evidence.revision.source): match for match in definitions}
    definition = next(iter(anchors.values())) if len(anchors) == 1 else None
    experiments = _confirmed_experiments(support_rows, definition, report.question, evidence)
    sections = []
    for requirement in requirements:
        if requirement == "definition" and definition:
            sections.append(
                AnswerSection(
                    requirement="definition",
                    status="supported",
                    statements=[_statement(definition.support, definition.evidence)],
                )
            )
        elif requirement == "experiment" and experiments:
            sections.append(_experiments_section(experiments, evidence))
        else:
            sections.append(_missing(requirement))
    return sections, sum(len(section.statements) for section in sections)


def _confirmed_experiments(
    rows: list[EvidenceSupport],
    definition: _Definition | None,
    question: str,
    evidence: dict[str, TurnEvidence],
) -> list[EvidenceSupport]:
    if definition is None:
        return []
    confirmed = []
    seen_quotes = set()
    for row in rows:
        if row.requirement != "experiment" or row.quote in seen_quotes:
            continue
        validated, _ = _requested_experiment_in(evidence[row.evidence_id], definition, question)
        if validated == row:
            confirmed.append(row)
            seen_quotes.add(row.quote)
    return confirmed


def _valid_span(support: EvidenceSupport, evidence: TurnEvidence) -> bool:
    read = evidence.reads[0]
    return (
        read.actual_source == support.actual_source
        and 0 <= support.start < support.end <= len(read.text)
        and read.text[support.start : support.end] == support.quote
    )


def _experiments_section(
    rows: list[EvidenceSupport], evidence: dict[str, TurnEvidence]
) -> AnswerSection:
    outcomes: dict[tuple[str, str, str, str, int], set[tuple[Decimal, Decimal]]] = {}
    for row in rows:
        values = row.experiment
        assert values is not None
        condition = (
            values.experiment_id,
            _canonical(values.split, _SPLITS),
            _canonical(values.metric, _METRICS),
            values.unit.strip(),
            values.samples,
        )
        outcomes.setdefault(condition, set()).add((Decimal(values.baseline), Decimal(values.trial)))
    conflict = any(len(values) > 1 for values in outcomes.values())
    return AnswerSection(
        requirement="experiment",
        status="conflicting" if conflict else "supported",
        statements=[_statement(row, evidence[row.evidence_id]) for row in rows],
        message=_condition_message(conflict, len(outcomes)),
    )


def _canonical(value: str, aliases: dict[str, Pattern[str]]) -> str:
    return next(
        (label for label, match in aliases.items() if match.fullmatch(value)),
        value.strip().casefold(),
    )


def _condition_message(conflict: bool, count: int) -> str:
    if conflict:
        return "相同记录条件的数值存在冲突，以下保留各自来源和值。"
    if count > 1:
        return "原文记录了不同实验条件，以下分开呈现。"
    return ""


def _statement(support: EvidenceSupport, evidence: TurnEvidence) -> AnswerStatement:
    values = support.experiment
    text = support.quote
    if values is not None:
        text = (
            f"实验 {values.experiment_id}，切分 {values.split}，样本数 {values.samples}。"
            f"{values.metric}（{values.unit}）基线为 {values.baseline}，试验为 {values.trial}。"
        )
    return AnswerStatement(
        text=text,
        experiment=values,
        citation=AnswerCitation(
            evidence_id=evidence.evidence_id,
            revision=evidence.revision,
            actual_source=support.actual_source,
            file_name=evidence.file_name,
            locator=evidence.locator,
            read_status=evidence.read_status,
            start=support.start,
            end=support.end,
            quote=support.quote,
        ),
    )


def _missing(requirement: Requirement) -> AnswerSection:
    return AnswerSection(requirement=requirement, status="missing", message=_MISSING[requirement])


def _status(sections: list[AnswerSection]) -> AnswerStatus:
    if not any(section.statements for section in sections):
        return "no_evidence"
    if any(section.status == "conflicting" for section in sections):
        return "conflicting"
    if any(section.status == "missing" for section in sections):
        return "partial"
    return "complete"


def _unavailable(
    report: SupplementReport, requirements: list[Requirement], reason: str
) -> EvidenceAnswer:
    return EvidenceAnswer(
        status="unavailable",
        sections=[_missing(requirement) for requirement in requirements],
        stop_reason=reason,
        budget=report.budget,
        message="本轮无法重新核对证据。",
    )
