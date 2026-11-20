"""Compose verified native principle, table and saved-output evidence in one turn."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from time import perf_counter
from typing import Any, Literal

from app.rag.evidence_supplement import (
    _EXPERIMENT,
    _EXPLICIT_TOPIC,
    _METRICS,
    _NUMBER,
    _SPLITS,
    _UNOBSERVED,
    _ZH_EXPLICIT_TOPIC,
    EvidenceSupport,
    TermAnchor,
    _already_used,
    _block_bounds,
    _Definition,
    _definition_in,
    _experiment_in,
)
from app.rag.structure_search import (
    PublicDocument,
    StructureScope,
    StructureSearcher,
    StructureSearchRequest,
)
from app.schemas.local import SourceDraft, SourceKey
from app.schemas.turn_evidence import SourceRevision, TurnEvidence, TurnRead
from app.schemas.workspace import WorkspaceError, inaccessible
from app.stores.local_library import LocalLibraryStore
from app.stores.structure_snapshot import (
    SnapshotDocument,
    SnapshotRequest,
    SnapshotSource,
    StructureSnapshot,
)

Role = Literal["principle", "table", "saved_output"]
_ROLES: tuple[Role, ...] = ("principle", "table", "saved_output")
_LABELS = {"principle": "定义或适用条件", "table": "原生表格", "saved_output": "保存的实测输出"}
_CANDIDATE_BUDGET = 24
_READ_LIMIT = 16_000
_POSITION_KEYS = {"page", "cell_index", "cell_id", "output_index", "shape_id_path", "bbox"}
_PRINCIPLE = r"定义|公式|原理|适用条件|\b(?:defin(?:e|ition)|formula|principle|assumptions?)\b"
_TABLE = r"表格|参数表|限制表|限值表|阈值表|数据表|\b(?:table|tabulated)\b"
_OUTPUT = (
    r"保存(?:的|下来的)?(?:实验|运行|Notebook)?\s*输出|实验输出|运行输出|实测(?:结果|输出|数据)|"
    r"\b(?:(?:saved|recorded|experiment(?:al)?|notebook)\s+"
    r"(?:test\s+|run\s+|experiment\s+)?(?:outputs?|results?|measurements?))\b"
)
_NEGATION = (
    r"(?:不要|不用|不必|无需|不需要|排除|(?<!\w)(?:请)?别|"
    r"(?:也|但|并|可|千万|切记|务必|先|还|定义|公式|原理|实验|验证|表格|输出)别|"
    r"\b(?:without|ignore|skip|exclude|do\s+not|don['’]t)\b)"
    r"[^，。！？；,.;!?\n]{0,18}"
)
_PIVOT_TOPIC = re.compile(
    r"\b(formula|principle|assumptions?)\s+of\b|(?:的)?(?:公式|原理|适用条件)",
    re.IGNORECASE,
)
_ZH_DIRECTIVE = re.compile(r"^(?:请\s*)?(?:分别\s*)?(?:给出|解释|说明|介绍|列出|概述)\s*")
_PRINCIPLE_PREFIX = re.compile(r"^(?:公式|原理|Formula|Principle)\s*[:：]", re.IGNORECASE)
_FOLLOWING_CONDITION = re.compile(
    r"[ \t]*\r?\n[ \t]*(?:Formula|Condition|Assumption|公式|条件|假设|适用条件)"
    r"[ \t]*[:：][ \t]*\S[^\r\n]*",
    re.IGNORECASE,
)
_SPLIT_FIELD = re.compile(r"(?:split|data(?:set)?|subset|切分|数据集|阶段)", re.IGNORECASE)
_ID_FIELD = re.compile(r"(?:experiment[- _]?id|实验(?:编号|代号))", re.IGNORECASE)
_SUBJECT_FIELD = re.compile(
    r"series|subject|model|algorithm|method|name|系列|对象|主题|模型|算法|方法|名称", re.IGNORECASE
)
_UPPER_LIMIT = re.compile(
    r"(?:(?:trial|overall)\s+)?"
    r"(?:(?:max(?:imum)?|upper (?:limit|bound))\s+METRIC"
    r"|METRIC\s+(?:max(?:imum)?|upper (?:limit|bound)))"
    r"|(?:试验|实验|实测)?(?:最大\s*METRIC|METRIC\s*上限)",
    re.IGNORECASE,
)
_LIMIT_WITH_UNIT = re.compile(r"([^（）()]+?)[ \t]*[（(]\s*([^（）()]+?)\s*[）)]")
_CJK = re.compile(r"[\u3400-\u9fff]")
_NON_MEASUREMENT_VALUE = re.compile(r"\b(?:exception|traceback)\b", re.IGNORECASE)


@dataclass(frozen=True)
class CompositionStatement:
    text: str
    citations: tuple[str, ...]


@dataclass
class Composition:
    statements: list[CompositionStatement] = field(default_factory=list)
    evidence: dict[str, TurnEvidence] = field(default_factory=dict)
    read_evidence: dict[str, TurnEvidence] = field(default_factory=dict)
    notices: list[tuple[str, str]] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _Table:
    evidence: TurnEvidence
    headers: tuple[str, ...]
    cells: tuple[str, ...]
    row_index: int
    split: str | None


@dataclass
class _Selection:
    principle: _Definition | None = None
    table: _Table | None = None
    output: tuple[TurnEvidence, EvidenceSupport] | None = None
    conflicts: set[Role] = field(default_factory=set)

    def missing(self) -> list[Role]:
        return [
            role
            for role in _ROLES
            if getattr(self, "output" if role == "saved_output" else role) is None
        ]


def asks_for_composition(text: str) -> bool:
    return all(
        re.search(pattern, text, re.IGNORECASE)
        and not re.search(_NEGATION + f"(?:{pattern})", text, re.IGNORECASE)
        for pattern in (_PRINCIPLE, _TABLE, _OUTPUT)
    )


def compose_evidence(
    text: str,
    snapshot: StructureSnapshot,
    searcher: StructureSearcher,
    library_store: LocalLibraryStore,
    *,
    allowed_courses: frozenset[str],
    active: Callable[[], bool],
) -> Composition:
    """Preserve first-hop pins while reading and validating independently typed roles."""
    result = Composition(
        trace={
            "route": "evidence_composition",
            "requirements": list(_ROLES),
            "supplement_count": 0,
            "external_call_count": 0,
            "hops": [],
            "role_checks": [],
        }
    )
    pins = {doc.document_id: doc for doc in snapshot.documents}
    if snapshot.resolution_status != "resolved" or not pins:
        _complete(result, _Selection())
        return result

    def check_active() -> None:
        if not active():
            raise WorkspaceError("interrupted")
        if not all(doc.course_id in allowed_courses for doc in pins.values()):
            raise inaccessible()

    check_active()
    scope = _scope(pins.values())
    result.trace["effective_scope"] = scope.model_dump(mode="json")
    topic = _topic(_topic_question(text))
    initial_query = f"{text}\n{topic}" if topic else text
    started = perf_counter()
    initial = searcher.search(
        StructureSearchRequest(query=initial_query, scope=scope, top_k=_CANDIDATE_BUDGET),
        allowed_course_ids=allowed_courses,
    )
    check_active()
    _check_pins(initial.snapshot, pins)
    keys = [hit.source for hit in initial.candidates]
    result.trace["hops"].append(_hop("first", initial_query, scope, keys, started))
    sources = {(src.document_id, src.version, src.source_id): src for src in snapshot.sources}
    seen: set[tuple[str, int, str]] = set()
    reads: list[tuple[Role, TurnEvidence, SourceDraft]] = []

    def read_candidates(keys: Sequence[SourceKey]) -> None:
        for key in keys:
            identity = (key.document_id, key.version, key.source_id)
            if identity in seen:
                continue
            seen.add(identity)
            candidate = sources.get(identity)
            pin = pins.get(key.document_id)
            if pin is None or pin.version != key.version or candidate is None:
                raise WorkspaceError("stale_reference")
            role = _role(candidate)
            if role is None:
                continue
            check_active()
            read = library_store.read_source(key, course_ids=(pin.course_id,))
            if read.status == "not_accessible":
                raise WorkspaceError("stale_reference")
            if (
                not read.is_current
                or read.publication_epoch != pin.publication_epoch
                or read.content_sha256 != pin.content_sha256
                or read.source is None
                or read.source.source_id != key.source_id
            ):
                raise WorkspaceError("stale_reference")
            item = {
                "role": role,
                "source": key.model_dump(mode="json"),
                "read_status": read.status,
            }
            result.trace["role_checks"].append(item)
            if read.status != "available" or not read.source.text.strip():
                continue
            actual = read.source
            if _role(actual) != role:
                raise WorkspaceError("stale_reference")
            clipped = len(actual.text) > _READ_LIMIT
            evidence = TurnEvidence(
                evidence_id=f"ev-{len(reads) + 1}",
                revision=SourceRevision(
                    source=key,
                    course_id=pin.course_id,
                    publication_epoch=pin.publication_epoch,
                    content_sha256=pin.content_sha256,
                ),
                file_name=pin.file_name,
                locator={
                    key: actual.locator[key] for key in _POSITION_KEYS if key in actual.locator
                },
                read_status="partial" if clipped else "succeeded",
                reads=[
                    TurnRead(
                        actual_source=key,
                        method="native_table" if role == "table" else "native_text",
                        status="partial" if clipped else "succeeded",
                        text=actual.text[:_READ_LIMIT],
                        warnings=["excerpt_truncated"] if clipped else [],
                    )
                ],
            )
            item.update(evidence_id=evidence.evidence_id, read_status=evidence.read_status)
            reads.append((role, evidence, actual))
            result.read_evidence[evidence.evidence_id] = evidence.model_copy(
                update={"locator": {**evidence.locator, "evidence_role": role}}
            )

    read_candidates(keys)
    selected = _select(reads, text)
    result.trace["first_missing"] = selected.missing()
    anchor = selected.principle.term if selected.principle else None
    if selected.missing() and anchor and not _already_used(anchor.word, [initial_query]):
        anchor_pins = {
            doc_id: doc for doc_id, doc in pins.items() if doc.course_id == anchor.course_id
        }
        extra_scope = _scope(anchor_pins.values())
        started = perf_counter()
        check_active()
        extra = searcher.store.read(
            SnapshotRequest(
                course_ids=tuple(extra_scope.course_ids),
                document_ids=tuple(extra_scope.document_ids),
                fts_match=f'"{anchor.word}"',
                fts_limit=_CANDIDATE_BUDGET,
            )
        )
        _check_pins(extra.documents, anchor_pins)
        if extra.resolution_status != "resolved":
            raise WorkspaceError("stale_reference")
        keys = [
            SourceKey(document_id=seed.document_id, version=seed.version, source_id=seed.source_id)
            for seed in extra.fts_seeds
        ]
        result.trace["supplement_count"] = 1
        result.trace["term"] = anchor.model_dump(mode="json")
        result.trace["hops"].append(
            _hop("literal_supplement", anchor.word, extra_scope, keys, started)
        )
        read_candidates(keys)
        selected = _select(reads, text)
    check_active()
    current = searcher.store.read(
        SnapshotRequest(
            course_ids=tuple(scope.course_ids),
            document_ids=tuple(scope.document_ids),
        )
    )
    if current.resolution_status != "resolved":
        raise WorkspaceError("stale_reference")
    _check_pins(current.documents, pins)
    _complete(result, selected)
    return result


def _scope(documents: Iterable[SnapshotDocument]) -> StructureScope:
    documents = tuple(documents)
    return StructureScope(
        course_ids=sorted({doc.course_id for doc in documents}),
        document_ids=sorted(doc.document_id for doc in documents),
    )


def _check_pins(
    documents: Sequence[PublicDocument | SnapshotDocument], pins: dict[str, SnapshotDocument]
) -> None:
    def identity(doc: PublicDocument | SnapshotDocument) -> tuple[str, int, str, int, str]:
        return (
            doc.course_id,
            doc.version,
            doc.content_sha256,
            doc.publication_epoch,
            doc.file_name,
        )

    if {doc.document_id: identity(doc) for doc in documents} != {
        doc_id: identity(pin) for doc_id, pin in pins.items()
    }:
        raise WorkspaceError("stale_reference")


def _hop(
    kind: str, text: str, scope: StructureScope, keys: Sequence[SourceKey], started: float
) -> dict[str, Any]:
    return {
        "kind": kind,
        "query": text,
        "scope": scope.model_dump(mode="json"),
        "candidates": [key.model_dump(mode="json") for key in keys],
        "elapsed_ms": (perf_counter() - started) * 1_000,
    }


def _role(source: SnapshotSource | SourceDraft) -> Role | None:
    locator = source.locator
    if source.kind == "page_object" and locator.get("object_type") == "table":
        return "table"
    if (
        source.kind == "notebook_output"
        and locator.get("output_type") in {"stream", "execute_result", "display_data"}
        and locator.get("stream_name") != "stderr"
    ):
        return "saved_output"
    if source.kind == "page" or (source.kind == "cell" and locator.get("cell_type") == "markdown"):
        return "principle"
    return None


def _topic_question(question: str) -> str:
    return _PIVOT_TOPIC.sub(
        lambda match: "definition of" if match.group(1) else "的定义",
        _ZH_DIRECTIVE.sub("", question),
    )


def _topic(question: str) -> str:
    matches = [
        match
        for pattern in (_EXPLICIT_TOPIC, _ZH_EXPLICIT_TOPIC)
        if (match := pattern.search(question))
    ]
    return min(matches, key=lambda match: match.start(1))[1].strip() if matches else ""


def _select(reads: list[tuple[Role, TurnEvidence, SourceDraft]], question: str) -> _Selection:
    selection = _Selection()
    topic_question = _topic_question(question)
    principles = [
        definition
        for role, evidence, _ in reads
        if role == "principle" and (definition := _principle(evidence, topic_question))
    ]
    distinct = {
        (
            item.evidence.revision.course_id,
            item.support.quote,
            item.term.word if item.term else "",
        )
        for item in principles
    }
    if len(distinct) > 1:
        selection.conflicts.add("principle")
        return selection
    if not principles:
        return selection
    selection.principle = definition = principles[0]

    tables = []
    outputs = []
    for role, evidence, source in reads:
        if evidence.revision.course_id != definition.evidence.revision.course_id:
            continue
        if role == "table":
            tables.extend(_table_rows(evidence, source, definition, topic_question))
        elif role == "saved_output" and not _NON_MEASUREMENT_VALUE.search(source.text):
            support, _ = _experiment_in(evidence, definition, question)
            if support:
                outputs.append((evidence, support))
    output_values = {
        item.experiment.model_dump_json() for _, item in outputs if item.experiment is not None
    }
    if len(output_values) > 1:
        selection.conflicts.add("saved_output")
    elif outputs:
        selection.output = outputs[0]

    condition = (
        _split(selection.output[1].experiment.split, exact=True) if selection.output else None
    )
    matching = [item for item in tables if not condition or item.split in {condition, None}]
    if tables and not matching:
        selection.conflicts.add("table")
    distinct_rows = {(item.headers, item.cells) for item in matching}
    if len(distinct_rows) > 1:
        selection.conflicts.add("table")
    elif matching:
        selection.table = matching[0]
    return selection


def _principle(evidence: TurnEvidence, question: str) -> _Definition | None:
    read = evidence.reads[0]
    text = read.text
    definition = _definition_in(evidence, question)
    if definition is None and (prefix := _PRINCIPLE_PREFIX.match(text)):
        alias = "定义：".ljust(prefix.end())
        shifted = evidence.model_copy(
            update={"reads": [read.model_copy(update={"text": alias + text[prefix.end() :]})]}
        )
        definition = _definition_in(shifted, question)
    if not definition:
        return None
    support = definition.support
    end = support.end
    _, block_end = _block_bounds(text, support.start)
    while match := _FOLLOWING_CONDITION.match(text, end, block_end):
        end = match.end()
    if end != support.end:
        if read.status == "partial" and end == len(text):
            end = support.end
        support = support.model_copy(update={"end": end, "quote": text[support.start : end]})
    term = definition.term
    excerpt_hash = hashlib.sha256(text.encode()).hexdigest()
    if term:
        term = term.model_copy(update={"excerpt_sha256": excerpt_hash})
    elif read.status == "succeeded" and end > definition.support.end:
        markers = list(_EXPERIMENT.finditer(text, support.start, block_end))
        if len(markers) == 1:
            marker = markers[0]
            if marker.start() >= end and not text[end : marker.start()].strip():
                term = TermAnchor(
                    word=marker[1],
                    source=read.actual_source,
                    course_id=evidence.revision.course_id,
                    evidence_id=evidence.evidence_id,
                    excerpt_sha256=excerpt_hash,
                    start=marker.start(1),
                    end=marker.end(1),
                )
    return replace(definition, evidence=evidence, support=support, term=term)


def _table_rows(
    evidence: TurnEvidence, source: SourceDraft, definition: _Definition, question: str
) -> list[_Table]:
    if evidence.read_status != "succeeded" or _UNOBSERVED.search(source.text):
        return []
    rows = source.locator.get("rows")
    if not isinstance(rows, list) or len(rows) < 2:
        return []
    headers = _cells(rows[0])
    if not headers:
        return []
    split_columns = [
        index for index, header in enumerate(headers) if _SPLIT_FIELD.fullmatch(header)
    ]
    id_columns = [index for index, header in enumerate(headers) if _ID_FIELD.fullmatch(header)]
    subject_columns = [
        index for index, header in enumerate(headers) if _SUBJECT_FIELD.fullmatch(header)
    ]
    if len(split_columns) > 1 or len(id_columns) > 1:
        return []
    requested_split = _split(question)
    result = []
    for row_index, raw in enumerate(rows[1:], start=1):
        cells = _cells(raw)
        if len(cells) != len(headers):
            continue
        record_matches = bool(
            id_columns and definition.term and cells[id_columns[0]] == definition.term.word
        )
        if id_columns and not record_matches:
            continue
        subjects = subject_columns or ([] if record_matches else [0])
        if not all(_matches_subject(cells[index], definition, question) for index in subjects):
            continue
        split = _split(cells[split_columns[0]], exact=True) if split_columns else None
        if split_columns and (split is None or (requested_split and split != requested_split)):
            continue
        result.append(_Table(evidence, headers, cells, row_index, split))
    return result


def _normalize_subject(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).strip().casefold()
    return " ".join(re.sub(r"^(?:the|an?)\s+", "", value).split())


def _matches_subject(cell: str, definition: _Definition, question: str) -> bool:
    subject = _normalize_subject(_topic(question))
    if not subject:
        return False
    value = _normalize_subject(cell)
    if subject == value:
        return True
    if not _CJK.search(subject) or not re.fullmatch(r"[a-z][a-z0-9 -]*", value):
        return False
    quote = _normalize_subject(definition.support.quote)
    if not quote.startswith(subject):
        return False
    # An alias must be written directly after the named topic in the same original quote.
    alias = quote[len(subject) :]
    return (
        re.match(
            rf"(?:\s+|\s*\()({re.escape(value)})(?=\)|[：:,，。]|\s*[\u3400-\u9fff]"
            rf"|\s+(?:is|passes|uses|means|refers)\b|$)",
            alias,
        )
        is not None
    )


def _cells(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(cell, str) for cell in value):
        return ()
    return tuple(cell.strip() for cell in value)


def _split(text: str, *, exact: bool = False) -> str | None:
    matches = [
        name
        for name, pattern in _SPLITS.items()
        if (pattern.fullmatch(text.strip()) if exact else pattern.search(text))
    ]
    return matches[0] if len(matches) == 1 else None


def _complete(result: Composition, selection: _Selection) -> None:
    supports = {}
    if definition := selection.principle:
        evidence = definition.evidence
        _add_statement(result, f"讲义原文：{definition.support.quote}", evidence)
        supports["principle"] = {
            "evidence_id": evidence.evidence_id,
            "span": [definition.support.start, definition.support.end],
            "quote": definition.support.quote,
        }
    if table := selection.table:
        pairs = " / ".join(
            f"{header} = {value}" for header, value in zip(table.headers, table.cells, strict=True)
        )
        _add_statement(result, f"原生表格第 {table.row_index + 1} 行：{pairs}", table.evidence)
        supports["table"] = {
            "evidence_id": table.evidence.evidence_id,
            "row_index": table.row_index,
            "cells": dict(zip(table.headers, table.cells, strict=True)),
        }
    if output := selection.output:
        evidence, support = output
        _add_statement(result, f"保存的实测输出：{support.quote}", evidence)
        supports["saved_output"] = {
            "evidence_id": evidence.evidence_id,
            "span": [support.start, support.end],
            "quote": support.quote,
        }
    missing = selection.missing()
    for role in missing:
        conflicting = role in selection.conflicts
        message = (
            f"{_LABELS[role]}存在冲突，无法确认适用项。"
            if conflicting
            else f"本轮缺少可核对的{_LABELS[role]}。"
        )
        result.notices.append(
            (f"composition_{'conflict' if conflicting else 'missing'}_{role}", message)
        )
    comparison = "missing_evidence"
    if selection.table and selection.output and selection.principle:
        comparison = _compare(result, selection.table, selection.output)
    result.trace.update(
        missing=missing,
        conflicts=sorted(selection.conflicts),
        supports=supports,
        comparison=comparison,
        final_evidence_ids=list(result.evidence),
        answer_status=(
            "supported"
            if not missing and comparison == "verified"
            else "partial"
            if result.statements
            else "insufficient"
        ),
    )


def _compare(
    result: Composition, table: _Table, output: tuple[TurnEvidence, EvidenceSupport]
) -> str:
    evidence, support = output
    values = support.experiment
    assert values is not None
    metric_pattern = next(
        (
            pattern
            for name, pattern in _METRICS.items()
            if name == values.metric.casefold() or pattern.fullmatch(values.metric)
        ),
        None,
    )
    split_pattern = _SPLITS.get(table.split or "")
    limits = []
    aligned = bool(split_pattern and split_pattern.fullmatch(values.split))
    for header, value in zip(table.headers, table.cells, strict=True):
        if _SPLIT_FIELD.fullmatch(header):
            aligned = bool(aligned and split_pattern and split_pattern.fullmatch(value))
            continue
        if _SUBJECT_FIELD.fullmatch(header) or _ID_FIELD.fullmatch(header):
            continue  # The selected row already verified every identity column.
        label = _LIMIT_WITH_UNIT.fullmatch(header)
        if (
            metric_pattern
            and label
            and label[2].strip() == values.unit
            and _UPPER_LIMIT.fullmatch(metric_pattern.sub("METRIC", label[1]).strip())
            and _NUMBER.fullmatch(value)
        ):
            limits.append(value)
        else:
            aligned = False  # An unverified contextual column may qualify the entire row.
    if aligned and len(limits) == 1:
        allowed = Decimal(values.trial) <= Decimal(limits[0])
        message = "不高于" if allowed else "高于"
        _add_statement(
            result,
            f"在 {values.split} 条件下，实测 {values.metric} {values.trial} {values.unit}"
            f" {message}同条件表格上限 {limits[0]} {values.unit}。",
            table.evidence,
            evidence,
        )
        return "verified"
    result.notices.append(
        (
            "composition_comparison_unverified",
            "表格与输出缺少相同指标、单位和条件的明确上限，无法据此作数值比较。",
        )
    )
    return "unverified_alignment"


def _add_statement(result: Composition, text: str, *evidence: TurnEvidence) -> None:
    for item in evidence:
        result.evidence[item.evidence_id] = item
    result.statements.append(
        CompositionStatement(text, tuple(item.evidence_id for item in evidence))
    )
