"""F03 claim validation and terminal replay over real imported B/C/F02 sources."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from sqlite3 import OperationalError

import pytest
from test_evidence_supplement import (
    COURSE,
    FILES,
    MATCH,
    QUESTION,
    RecordingSnapshots,
    first_pair,
    publish,
    request,
)

from app.config import Settings
from app.rag.evidence_answer import EvidenceAnswerer
from app.rag.evidence_supplement import SupplementRetriever
from app.rag.local_library import LocalLibrary
from tools import evidence_answer as cli

ALLOWED = frozenset({COURSE})


@pytest.fixture
def local(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("the F03 offline answer must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    settings = Settings(
        _env_file=None,
        course_root=tmp_path / "course",
        data_dir=tmp_path / "data",
        sqlite_path=tmp_path / "evidence.sqlite",
        deepseek_api_key="",
        dashscope_api_key="",
    )
    settings.course_root.mkdir()
    library = LocalLibrary(settings)
    snapshots = RecordingSnapshots(settings)
    retriever = SupplementRetriever(library.store, snapshots)
    return settings, library, snapshots, EvidenceAnswerer(retriever)


def _facts(answer):
    return [statement for section in answer.sections for statement in section.statements]


def _flat_states(answer):
    return {section.requirement: section.status for section in answer.sections}


def _assert_citations(run):
    trace = run.retrieval
    assert trace is not None
    retained = {item.evidence_id: item for item in trace.final_evidence}
    assert len(retained) <= trace.budget
    for fact in _facts(run.answer):
        ref = fact.citation
        evidence = retained[ref.evidence_id]
        assert ref.revision == evidence.revision
        assert ref.locator == evidence.locator
        read = next(read for read in evidence.reads if read.actual_source == ref.actual_source)
        assert read.status == ref.read_status == "succeeded"
        assert read.text[ref.start : ref.end] == ref.quote


def test_exact_native_definition_and_experiment_at_the_same_final_budget(local):
    _, library, snapshots, answerer = local
    lecture, notes = first_pair(library)
    lab = publish(library, "lab")
    single = answerer.answer(request(), allowed_course_ids=ALLOWED, strategy="single")
    snapshots.calls.clear()
    deep = answerer.answer(request(), allowed_course_ids=ALLOWED)

    assert _flat_states(single.answer) == {"definition": "supported", "experiment": "missing"}
    assert single.answer.status == "partial"
    assert _flat_states(deep.answer) == {"definition": "supported", "experiment": "supported"}
    assert deep.answer.status == "complete" and deep.answer.rejected_support_count == 0
    assert deep.answer.verified_source_count == 2
    assert sum(call.fts_match == MATCH for call in snapshots.lookups) == 1
    assert deep.retrieval.hops[0].model_dump(exclude={"elapsed_ms"}) == (
        single.retrieval.hops[0].model_dump(exclude={"elapsed_ms"})
    )
    assert deep.retrieval.replacements[0].removed.source.document_id == notes.document_id

    definition, comparison = _facts(deep.answer)
    assert definition.text == definition.citation.quote
    assert [fact.citation.actual_source.document_id for fact in (definition, comparison)] == [
        lecture.document_id,
        lab.document_id,
    ]
    assert all(
        fact.citation.actual_source.source_id == "cell:0" for fact in (definition, comparison)
    )
    assert comparison.experiment.samples == 4
    assert "held-out" in comparison.text and "MAE（amplitude）" in comparison.text
    assert "基线为 0.24" in comparison.text and "试验为 0.18" in comparison.text
    assert "因为" not in comparison.text
    assert str(library.course_root) not in deep.model_dump_json()
    _assert_citations(single)
    _assert_citations(deep)


@pytest.mark.parametrize(
    ("record", "reason"),
    [
        (None, "duplicate_only"),
        (FILES["lab"]["text"].replace("held-out", "training"), "mismatched"),
        (
            FILES["lab"]["text"].replace(" Trial: 0.18.", ""),
            "still_missing",
        ),
    ],
)
def test_duplicate_mismatch_or_missing_field_never_creates_experiment_claim(local, record, reason):
    _, library, _, answerer = local
    lecture, _ = first_pair(library)
    if record is not None:
        publish(library, "lab", text=record)
    run = answerer.answer(request(), allowed_course_ids=ALLOWED)
    assert run.answer.stop_reason == reason
    assert run.answer.status == "partial"
    assert _flat_states(run.answer) == {"definition": "supported", "experiment": "missing"}
    assert [fact.citation.actual_source.document_id for fact in _facts(run.answer)] == [
        lecture.document_id
    ]
    assert "0.18" not in run.answer.model_dump_json()
    _assert_citations(run)


@pytest.mark.parametrize(
    ("question", "experiment_state", "rejection"),
    [
        (QUESTION, "supported", None),
        (QUESTION.replace("留出集", "盐雾"), "missing", "experiment_condition_unverified"),
        (
            QUESTION.replace("留出集", "低温循环后的留出集"),
            "missing",
            "experiment_condition_unverified",
        ),
        (
            QUESTION.replace("留出集", "均值滤波的留出集"),
            "missing",
            "experiment_condition_unverified",
        ),
        (QUESTION.replace("留出集", "测试集"), "missing", "experiment_split_mismatch"),
        (QUESTION + "请给出 RMSE。", "missing", "experiment_metric_mismatch"),
        (QUESTION + "请给出尾部延迟。", "missing", "experiment_condition_unverified"),
        (
            "Define a median filter and explain its underwater held-out experiment using MAE.",
            "missing",
            "experiment_condition_unverified",
        ),
        (QUESTION.replace("留出集", "中值滤波 median filter 的留出集"), "supported", None),
    ],
)
def test_experiment_only_supports_the_requested_conditions(
    local, question, experiment_state, rejection
):
    _, library, _, answerer = local
    lecture, _ = first_pair(library)
    lab = publish(library, "lab")
    run = answerer.answer(request(question=question), allowed_course_ids=ALLOWED)

    assert _flat_states(run.answer) == {"definition": "supported", "experiment": experiment_state}
    _assert_citations(run)
    facts = _facts(run.answer)
    assert facts[0].citation.actual_source.document_id == lecture.document_id
    if experiment_state == "supported":
        assert facts[1].citation.actual_source.document_id == lab.document_id
        assert "0.18" in facts[1].text
    else:
        assert len(facts) == 1
        assert "所问对象、条件和指标" in run.answer.sections[1].message
        assert rejection in [
            candidate.reason
            for hop in run.retrieval.hops
            for candidate in hop.candidates
            if candidate.source.document_id == lab.document_id
        ]


def test_experiment_accepts_recorded_split_and_metric_literals(local):
    _, library, _, answerer = local
    first_pair(library)
    lab = publish(
        library,
        "lab",
        text=FILES["lab"]["text"].replace("held-out", "CinderBatch").replace("MAE", "tail_ratio"),
    )
    question = QUESTION.replace("留出集", "CinderBatch") + "请给出 tail_ratio。"
    run = answerer.answer(request(question=question), allowed_course_ids=ALLOWED)
    assert _flat_states(run.answer) == {"definition": "supported", "experiment": "supported"}
    fact = _facts(run.answer)[1]
    assert fact.citation.actual_source.document_id == lab.document_id
    assert "CinderBatch" in fact.text and "tail_ratio" in fact.text
    _assert_citations(run)


@pytest.mark.parametrize(
    ("change", "expected", "question"),
    [
        ("alias", "conflicting", QUESTION),
        ("split", "supported", QUESTION.replace("留出集", "")),
        ("unit", "supported", QUESTION),
    ],
)
def test_conflict_and_different_conditions_keep_each_exact_record(
    local, change, expected, question
):
    _, library, _, answerer = local
    first_pair(library)
    text = FILES["lab"]["text"]
    if change == "unit":
        text = text.replace("Unit: amplitude", "Unit: mW")
    lab = publish(library, "lab", text=text)
    modified = text.replace("Trial: 0.18", "Trial: 0.21")
    if change == "alias":
        modified = modified.replace("Metric: MAE", "Metric: mean absolute error")
        modified = modified.replace("held-out", "留出集")
    elif change == "split":
        modified = modified.replace("held-out", "training")
    else:
        modified = modified.replace("Unit: mW", "Unit: MW")
    rival = publish(library, "lab", text=modified, file_name="A7-alternative.ipynb")
    run = answerer.answer(request(question=question, budget=3), allowed_course_ids=ALLOWED)
    compare = run.answer.sections[1]
    assert compare.requirement == "experiment" and compare.status == expected
    assert run.answer.status == ("conflicting" if expected == "conflicting" else "complete")
    assert "冲突" in compare.message if expected == "conflicting" else "分开" in compare.message
    assert {fact.experiment.trial for fact in compare.statements} == {"0.18", "0.21"}
    assert {fact.citation.actual_source.document_id for fact in compare.statements} == {
        lab.document_id,
        rival.document_id,
    }
    assert len(compare.statements) == 2 and not any("平均为" in f.text for f in compare.statements)
    _assert_citations(run)


@pytest.mark.parametrize(
    "tamper",
    ["span", "quote", "structured_value", "actual_source", "withdrawn", "revision", "staged_only"],
)
def test_rejects_tampered_support_and_keeps_independently_verified_definition(local, tamper):
    _, library, _, answerer = local
    _, notes = first_pair(library)
    publish(library, "lab")
    report = answerer.retriever.retrieve(request(), allowed_course_ids=ALLOWED).model_copy(
        deep=True
    )
    row = report.supports[1]
    withdrawn = next(
        evidence
        for evidence in report.reads
        if evidence.revision.source.document_id == notes.document_id
    )
    if tamper == "span":
        row.end += 1
    elif tamper == "quote":
        row.quote = "Fabricated accuracy."
    elif tamper == "structured_value":
        row.experiment.trial = "0.99"
    elif tamper == "actual_source":
        row.actual_source = withdrawn.revision.source
    elif tamper == "withdrawn":
        row.evidence_id = withdrawn.evidence_id
    elif tamper == "revision":
        for evidence in [*report.final_evidence, *report.reads]:
            if evidence.evidence_id == row.evidence_id:
                evidence.revision.publication_epoch += 4
    else:
        report.reads = [item for item in report.reads if item.evidence_id != row.evidence_id]
    answer = answerer.compose(report, allowed_course_ids=ALLOWED)
    assert answer.status == "partial" and answer.rejected_support_count == 1
    assert _flat_states(answer) == {"definition": "supported", "experiment": "missing"}
    assert len(_facts(answer)) == 1 and "0.99" not in answer.model_dump_json()


@pytest.mark.parametrize("forgery", ["native_text", "mock_read", "unread", "duplicate_evidence_id"])
def test_b_recheck_rejects_forges_matching_read_and_final_shapes(local, forgery):
    _, library, _, answerer = local
    first_pair(library)
    publish(library, "lab")
    report = answerer.retriever.retrieve(request(), allowed_course_ids=ALLOWED)
    row = report.supports[1]
    evidence = next(item for item in report.final_evidence if item.evidence_id == row.evidence_id)
    if forgery == "native_text":
        source = evidence.reads[0].text
        old = source[row.start : row.end]
        new = old.replace("Trial: 0.18", "Trial: 0.99")
        assert len(old) == len(new) == len(row.quote)
        evidence.reads[0].text = source[: row.start] + new + source[row.end :]
        row.quote = new
        row.experiment.trial = "0.99"
    elif forgery == "mock_read":
        evidence.reads[0].method = "vision"
        evidence.reads[0].execution = "mock_protocol"
    elif forgery == "unread":
        evidence.read_status = "not_attempted"
        evidence.reads = []
    else:
        report.reads.append(evidence)

    answer = answerer.compose(report, allowed_course_ids=ALLOWED)
    assert answer.status == "partial"
    assert answer.rejected_support_count == 1
    assert _flat_states(answer) == {"definition": "supported", "experiment": "missing"}
    assert "0.99" not in answer.model_dump_json()


@pytest.mark.parametrize("failure", ["version", "epoch", "revoke", "cancel", "db"])
def test_scope_or_snapshot_invalidation_suppresses_all_facts(local, monkeypatch, failure):
    _, library, snapshots, answerer = local
    first_pair(library)
    publish(library, "lab")
    untouched = publish(library, "notes", file_name="Off-topic.ipynb", text="Other.")
    report = answerer.retriever.retrieve(request(), allowed_course_ids=ALLOWED)
    allowed = ALLOWED
    stop = None
    if failure == "version":
        publish(
            library,
            "notes",
            file_name="Off-topic.ipynb",
            text="A newer document.",
            document_id=untouched.document_id,
        )
    elif failure == "epoch":
        assert library.delete(untouched.document_id)
        repeat = publish(
            library,
            "notes",
            file_name="Off-topic.ipynb",
            text="Other.",
            document_id=untouched.document_id,
        )
        assert repeat.version == untouched.version
        assert repeat.publication_epoch > untouched.publication_epoch
    elif failure == "revoke":
        allowed = frozenset()
    elif failure == "cancel":

        def stopped():
            return True

        stop = stopped
    else:

        def offline(_query):
            raise OperationalError("simulated verification failure")

        monkeypatch.setattr(snapshots, "read", offline)
    monkeypatch.setattr(answerer.retriever, "retrieve", lambda *args, **kwargs: report)
    run = answerer.answer(request(), allowed_course_ids=allowed, cancelled=stop)
    assert run.retrieval is None and run.answer.status == "unavailable"
    assert _facts(run.answer) == []
    assert "SignalTrialA7" not in run.answer.model_dump_json()
    assert (
        run.answer.stop_reason
        == {
            "version": "stale_snapshot",
            "epoch": "stale_snapshot",
            "revoke": "not_accessible",
            "cancel": "cancelled",
            "db": "answer_verification_failed",
        }[failure]
    )


def test_late_change_after_citation_read_is_checked_before_delivery(local, monkeypatch):
    _, library, _, answerer = local
    first_pair(library)
    lab = publish(library, "lab")
    report = answerer.retriever.retrieve(request(), allowed_course_ids=ALLOWED)
    read_source = library.store.read_source

    def revise_after_read(key, *, course_ids):
        read = read_source(key, course_ids=course_ids)
        if key.document_id == lab.document_id and key.version == lab.version:
            publish(
                library,
                "lab",
                text=FILES["lab"]["text"].replace("Trial: 0.18", "Trial: 0.33"),
                document_id=lab.document_id,
            )
        return read

    monkeypatch.setattr(library.store, "read_source", revise_after_read)
    answer = answerer.compose(report, allowed_course_ids=ALLOWED)
    assert answer.status == "unavailable" and answer.stop_reason == "stale_snapshot"
    assert _facts(answer) == []


def test_failed_literal_lookup_preserves_current_definition(local):
    _, library, snapshots, answerer = local
    lecture, _ = first_pair(library)
    publish(library, "lab")

    def fail_query(query, _snapshot):
        if query.fts_match == MATCH:
            raise OperationalError("controlled supplemental failure")

    snapshots.after_read = fail_query
    run = answerer.answer(request(), allowed_course_ids=ALLOWED)
    assert run.answer.status == "partial"
    assert run.answer.stop_reason == "search_failed"
    assert [fact.citation.actual_source.document_id for fact in _facts(run.answer)] == [
        lecture.document_id
    ]
    assert run.retrieval.supplement_count == 1
    _assert_citations(run)


def test_cli_uses_trusted_scope_and_can_emit_exact_traces(local, monkeypatch, capsys):
    settings, library, _, _ = local
    first_pair(library)
    publish(library, "lab")
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setenv("SRAG_STRUCTURE_COURSES", COURSE)
    args = [QUESTION, "--course", COURSE, "--strategy", "supplement"]
    assert cli.main([*args, "--trace"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["answer"]["status"] == "complete"
    assert payload["retrieval"]["term"]["word"] == "SignalTrialA7"
    assert payload["retrieval"]["supplement_count"] == 1
    assert str(Path(settings.course_root)) not in json.dumps(payload)
    with pytest.raises(SystemExit) as denied:
        cli.main([QUESTION, "--course", "unallowed"])
    assert denied.value.code == 2
    assert capsys.readouterr().out == ""


def test_cli_uses_validated_json_on_text_format_failure(local, monkeypatch, capsys):
    settings, library, _, _ = local
    first_pair(library)
    publish(library, "lab")

    def format_failure(_answer):
        raise ValueError("controlled formatter failure")

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "format_text", format_failure)
    monkeypatch.setenv("SRAG_STRUCTURE_COURSES", COURSE)
    assert cli.main([QUESTION, "--course", COURSE]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["answer"]["status"] == "complete"
    assert output["answer"]["sections"][1]["statements"][0]["experiment"]["trial"] == "0.18"
    assert "retrieval" not in output


def test_text_formatter_prints_quoted_source_as_text_without_controls(local):
    _, library, _, answerer = local
    publish(
        library,
        "lecture",
        text=FILES["lecture"]["text"].replace(" odd window.", " odd window.\x1b[31m"),
    )
    publish(library, "notes")
    publish(library, "lab")
    run = answerer.answer(request(), allowed_course_ids=ALLOWED)
    assert "\x1b" in _facts(run.answer)[0].citation.quote
    assert "\x1b" not in cli.format_text(run.answer)
    _assert_citations(run)


def test_current_publication_without_lab_original_keeps_only_definition(local):
    _, library, _, answerer = local
    lecture, _ = first_pair(library)
    lab = publish(library, "lab")
    report = answerer.retriever.retrieve(request(), allowed_course_ids=ALLOWED)
    with library.store._connect() as db:
        db.execute(
            "UPDATE local_versions SET original = NULL WHERE document_id = ? AND version = ?",
            (lab.document_id, lab.version),
        )
    answer = answerer.compose(report, allowed_course_ids=ALLOWED)
    assert answer.status == "partial" and answer.rejected_support_count == 1
    assert _flat_states(answer) == {"definition": "supported", "experiment": "missing"}
    assert [fact.citation.actual_source.document_id for fact in _facts(answer)] == [
        lecture.document_id
    ]
