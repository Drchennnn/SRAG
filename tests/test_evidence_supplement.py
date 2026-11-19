"""The F02 policy against real B publication, C lookup, and SQLite, with no providers."""

from __future__ import annotations

import hashlib
import json
import socket
from collections.abc import Callable
from pathlib import Path
from statistics import mean

import nbformat
import pytest

from app.config import Settings
from app.rag.evidence_supplement import SupplementRetriever
from app.rag.local_library import LocalLibrary
from app.rag.structure_search import StructureLocation, StructureScope, StructureSearchRequest
from app.schemas.local import ImportResult, SourceKey
from app.stores.structure_snapshot import SnapshotRequest, StructureSnapshot, StructureSnapshotStore

SAMPLE = json.loads(
    (Path(__file__).resolve().parents[1] / "docs/F/样例/SignalTrialA7.json").read_text()
)
COURSE = SAMPLE["course_id"]
QUESTION = SAMPLE["question"]
TERM = "SignalTrialA7"
MATCH = f'"{TERM}"'
FILES = {document["role"]: document for document in SAMPLE["documents"]}


class RecordingSnapshots(StructureSnapshotStore):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.calls: list[SnapshotRequest] = []
        self.after_read: Callable[[SnapshotRequest, StructureSnapshot], None] | None = None

    def read(self, request: SnapshotRequest) -> StructureSnapshot:
        self.calls.append(request)
        result = super().read(request)
        if self.after_read:
            self.after_read(request, result)
        return result

    @property
    def lookups(self) -> list[SnapshotRequest]:
        return [call for call in self.calls if call.fts_match]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbid(*args, **kwargs):
        pytest.fail("a local F02 probe attempted an external connection")

    monkeypatch.setattr(socket.socket, "connect", forbid)
    monkeypatch.setattr(socket.socket, "connect_ex", forbid)


@pytest.fixture
def catalog(tmp_path):
    root = tmp_path / "course"
    root.mkdir()
    settings = Settings(
        _env_file=None,
        course_root=root,
        sqlite_path=tmp_path / "local.sqlite3",
        data_dir=tmp_path / "data",
        dashscope_api_key="",
        deepseek_api_key="",
    )
    library = LocalLibrary(settings)
    snapshots = RecordingSnapshots(settings)
    return library, snapshots, SupplementRetriever(library.store, snapshots)


def publish(
    library: LocalLibrary,
    role: str,
    *,
    course: str = COURSE,
    text: str | None = None,
    file_name: str | None = None,
    document_id: str | None = None,
) -> ImportResult:
    definition = FILES[role]
    cell = nbformat.v4.new_markdown_cell(
        definition["text"] if text is None else text,
        id="evidence",
    )
    file = library.course_root / course / (file_name or definition["file_name"])
    file.parent.mkdir(exist_ok=True)
    nbformat.write(nbformat.v4.new_notebook(cells=[cell]), file)
    imported = library.import_paths(course, [file], document_id=document_id)[0]
    assert imported.status == "published", imported
    return imported


def request(
    *,
    question: str = QUESTION,
    courses=(COURSE,),
    documents=(),
    budget=2,
    location=None,
):
    return StructureSearchRequest(
        query=question,
        scope=StructureScope(course_ids=list(courses), document_ids=list(documents)),
        location=location,
        top_k=budget,
        include_adjacency=False,
        include_page_fallback=False,
    )


def first_pair(library: LocalLibrary) -> tuple[ImportResult, ImportResult]:
    return publish(library, "lecture"), publish(library, "notes")


def run(retriever, query=None, *, strategy="supplement", allowed=None, **kwargs):
    return retriever.retrieve(
        request() if query is None else query,
        strategy=strategy,
        allowed_course_ids=frozenset({COURSE}) if allowed is None else allowed,
        **kwargs,
    )


def test_paired_same_budget_repairs_missing_experiment_and_records_eviction(catalog):
    library, snapshots, retriever = catalog
    lecture, notes = first_pair(library)
    lab = publish(library, "lab")

    single = run(retriever, strategy="single")
    assert len(snapshots.lookups) == 1
    snapshots.calls.clear()
    deep = run(retriever)

    assert single.scope == deep.scope
    assert single.snapshot == deep.snapshot
    assert {pin.document_id for pin in deep.snapshot} == {
        lecture.document_id,
        notes.document_id,
        lab.document_id,
    }
    assert single.hops[0].model_dump(exclude={"elapsed_ms"}) == deep.hops[0].model_dump(
        exclude={"elapsed_ms"}
    )
    assert [candidate.file_name for candidate in deep.hops[0].candidates] == [
        FILES["lecture"]["file_name"],
        FILES["notes"]["file_name"],
    ]
    assert single.budget == deep.budget == 2
    assert len(single.final_evidence) == len(deep.final_evidence) == 2
    assert single.missing == ["experiment"] and single.stop_reason == "single_hop"
    assert deep.status == "complete" and deep.stop_reason == "supplemented"
    assert deep.missing == []
    assert deep.supplement_count == 1
    assert deep.source_read_count == 3 == len(deep.reads)
    assert len(snapshots.lookups) == 2
    assert snapshots.lookups[1].fts_match == MATCH
    assert snapshots.lookups[1].fts_limit == deep.budget
    rechecks = [call for call in snapshots.calls if call.fts_match is None]
    assert deep.snapshot_recheck_count == len(rechecks)
    assert all(set(check.document_ids) == set(deep.scope.document_ids) for check in rechecks)
    assert all(check.fts_limit == 0 for check in rechecks)

    anchor = deep.term
    assert anchor is not None and anchor.word == TERM
    actual = next(e for e in deep.reads if e.evidence_id == anchor.evidence_id).reads[0]
    assert actual.actual_source == anchor.source
    assert actual.text[anchor.start : anchor.end] == TERM
    assert hashlib.sha256(actual.text.encode()).hexdigest() == anchor.excerpt_sha256
    assert TERM not in QUESTION
    assert _ids(deep.final_evidence) == {lecture.document_id, lab.document_id}
    assert deep.replacements[0].removed.source.document_id == notes.document_id
    assert deep.replacements[0].removed_locator["cell_index"] == 0
    assert deep.replacements[0].added.source.document_id == lab.document_id
    assert deep.replacements[0].reason == "required_experiment_priority"

    result = next(support for support in deep.supports if support.requirement == "experiment")
    assert result.actual_source.document_id == lab.document_id
    assert result.actual_source.source_id == "cell:0"
    assert result.experiment is not None
    assert result.experiment.baseline == str(mean([0.20, 0.24, 0.28, 0.24]))
    assert result.experiment.trial == str(mean([0.16, 0.20, 0.18, 0.18]))
    for support in deep.supports:
        retained = next(
            evidence
            for evidence in deep.final_evidence
            if evidence.evidence_id == support.evidence_id
        )
        assert retained.reads[0].text[support.start : support.end] == support.quote

    dispositions = {
        candidate.source.document_id: candidate.disposition for candidate in deep.hops[1].candidates
    }
    assert dispositions[lab.document_id] == "accepted"
    assert dispositions[lecture.document_id] == "duplicate"
    assert str(library.course_root) not in deep.model_dump_json()


def _ids(evidence):
    return {item.revision.source.document_id for item in evidence}


@pytest.mark.parametrize(
    ("lab_text", "stop", "reason"),
    [
        (None, "duplicate_only", None),
        (
            FILES["lab"]["text"].replace("held-out", "training"),
            "mismatched",
            "experiment_split_mismatch",
        ),
        (
            f"Experiment-ID: {TERM}. Split: held-out. Metric: MAE. Baseline: 0.24.",
            "still_missing",
            "experiment_fields_missing",
        ),
        (
            f"Experiment-ID: AnOtherRun. Split: held-out. Mention of earlier {TERM}.",
            "mismatched",
            "experiment_id_mismatch",
        ),
    ],
)
def test_one_returned_batch_preserves_duplicate_mismatch_and_incomplete_evidence(
    catalog, lab_text, stop, reason
):
    library, snapshots, retriever = catalog
    lecture, notes = first_pair(library)
    lab = None if lab_text is None else publish(library, "lab", text=lab_text)
    single = run(retriever, strategy="single")
    snapshots.calls.clear()
    deep = run(retriever)
    assert (
        _ids(single.final_evidence)
        == _ids(deep.final_evidence)
        == {
            lecture.document_id,
            notes.document_id,
        }
    )
    assert deep.stop_reason == stop and deep.supplement_count == 1
    assert [support.requirement for support in deep.supports] == ["definition"]
    assert deep.missing == ["experiment"] and deep.replacements == []
    assert sum(call.fts_match == MATCH for call in snapshots.lookups) == 1
    if lab is not None:
        candidate = next(
            item for item in deep.hops[1].candidates if item.source.document_id == lab.document_id
        )
        assert candidate.reason == reason and candidate.read_status == "succeeded"


def test_scope_is_limited_to_selected_documents_and_anchor_course(catalog):
    library, snapshots, retriever = catalog
    lecture, notes = first_pair(library)
    excluded = publish(library, "lab")
    other = publish(library, "lab", course="other-course")
    query = request(
        courses=(COURSE, "other-course"),
        documents=(lecture.document_id, notes.document_id, other.document_id),
    )
    result = run(retriever, query, allowed=frozenset({COURSE, "other-course"}))
    assert result.stop_reason == "duplicate_only"
    assert set(result.scope.course_ids) == {COURSE, "other-course"}
    assert set(result.scope.document_ids) == set(query.scope.document_ids)
    assert other.document_id not in _ids(result.reads)
    assert excluded.document_id not in _ids(result.reads)
    supplement = snapshots.lookups[1]
    assert supplement.course_ids == (COURSE,)
    assert set(supplement.document_ids) == {lecture.document_id, notes.document_id}


def test_same_id_experiment_other_authorized_course_is_not_first_hop_support(catalog):
    library, _, retriever = catalog
    first_pair(library)
    other = publish(
        library, "lab", course="other-course", text="Median filter report.\n" + FILES["lab"]["text"]
    )
    query = request(courses=(COURSE, "other-course"), budget=3)
    result = run(retriever, query, allowed=frozenset({COURSE, "other-course"}))
    assert result.missing == ["experiment"]
    candidate = next(
        c for c in result.hops[0].candidates if c.source.document_id == other.document_id
    )
    assert candidate.reason == "experiment_course_mismatch"
    assert other.document_id not in result.hops[1].scope.document_ids
    assert result.stop_reason == "duplicate_only"


def test_missing_requested_document_prevents_reusing_partial_c_snapshot(catalog, monkeypatch):
    library, snapshots, retriever = catalog
    lecture, _ = first_pair(library)
    publish(library, "lab")
    monkeypatch.setattr(
        library.store,
        "read_source",
        lambda *_args, **_kwargs: pytest.fail("a partial unresolved scope was reused"),
    )
    result = run(retriever, request(documents=(lecture.document_id, "unavailable-document")))
    assert result.stop_reason == result.first_search_status == "scope_not_found"
    assert result.scope.document_ids == [] and result.snapshot == []
    assert result.reads == [] and result.final_evidence == []
    assert result.supplement_count == result.source_read_count == 0
    assert sum(call.fts_match == MATCH for call in snapshots.lookups) == 0
    assert any("unavailable-document" in miss for miss in result.misses)


@pytest.mark.parametrize(
    "special", ["single_file", "explicit_cell", "budget", "no_term", "used_term"]
)
def test_zero_follow_up_when_scope_budget_or_anchor_cannot_support_one(catalog, special):
    library, snapshots, retriever = catalog
    if special == "no_term":
        lecture = publish(library, "lecture", text=FILES["lecture"]["text"].splitlines()[0])
    else:
        lecture = publish(library, "lecture")
    publish(library, "notes")
    if special != "used_term":
        publish(library, "lab")
    conditions = {
        "single_file": (request(documents=(lecture.document_id,)), "scope_limited"),
        "explicit_cell": (
            request(documents=(lecture.document_id,), location=StructureLocation(cell_index=0)),
            "scope_limited",
        ),
        "budget": (request(budget=1), "budget_exhausted"),
        "no_term": (request(), "no_new_terms"),
        "used_term": (request(question=f"{QUESTION} {TERM}."), "no_new_terms"),
    }
    query, expected = conditions[special]
    result = run(retriever, query)
    assert result.stop_reason == expected and result.supplement_count == 0
    assert result.term is None and result.missing == ["experiment"]
    assert sum(call.fts_match == MATCH for call in snapshots.lookups) == 0


def test_derived_index_match_is_not_a_readable_experiment(catalog):
    library, _, retriever = catalog
    first_pair(library)
    fake = publish(library, "lab", text="Recorded plans only.")
    assert library.store.write_enhancement(
        SourceKey(document_id=fake.document_id, version=fake.version, source_id="cell:0"),
        course_ids=(COURSE,),
        content_sha256=fake.content_sha256,
        publication_epoch=fake.publication_epoch,
        text=FILES["lab"]["text"],
    )
    result = run(retriever)
    candidate = next(
        c for c in result.hops[1].candidates if c.source.document_id == fake.document_id
    )
    assert candidate.origins == ["enhancement"]
    assert candidate.reason == "no_experiment_marker"
    assert result.missing == ["experiment"]
    assert fake.document_id not in _ids(result.final_evidence)


def test_cancellation_discards_already_returned_batch_and_stops_work(catalog):
    library, snapshots, retriever = catalog
    first_pair(library)
    publish(library, "lab")
    stopped = False

    def cancel_after_lookup(query, _snapshot):
        nonlocal stopped
        if query.fts_match == MATCH:
            stopped = True

    snapshots.after_read = cancel_after_lookup
    result = run(retriever, cancelled=lambda: stopped)
    assert result.stop_reason == "cancelled"
    assert result.supplement_count == 1
    assert result.source_read_count == 2
    _assert_redacted(result)


@pytest.mark.parametrize("change", ["different_version", "same_version_new_epoch", "revoke"])
def test_recheck_covers_course_outside_anchor_and_conceals_stale_data(catalog, change):
    library, snapshots, retriever = catalog
    first_pair(library)
    publish(library, "lab")
    extra = publish(library, "notes", course="other-course", text="Off-topic document.")
    args = {"course": "other-course", "document_id": extra.document_id}
    courses = {COURSE, "other-course"}

    def update_on_lookup(query, _snapshot):
        if query.fts_match != MATCH:
            return
        if change == "revoke":
            courses.remove("other-course")
        elif change == "different_version":
            updated = publish(library, "notes", text="Changed byte text.", **args)
            assert updated.version > extra.version
        else:
            assert library.delete(extra.document_id)
            updated = publish(library, "notes", text="Off-topic document.", **args)
            assert updated.version == extra.version
            assert updated.publication_epoch > extra.publication_epoch

    snapshots.after_read = update_on_lookup
    result = run(
        retriever, request(courses=tuple(sorted(courses))), allowed=lambda: frozenset(courses)
    )
    assert result.stop_reason == ("not_accessible" if change == "revoke" else "stale_snapshot")
    assert result.supplement_count == 1 and len(snapshots.lookups) == 2
    assert result.source_read_count == 2
    _assert_redacted(result)


def _assert_redacted(result):
    assert result.status == "unavailable"
    assert result.snapshot == [] and result.scope.document_ids == []
    assert result.reads == result.final_evidence == result.supports == result.hops == []
    assert result.term is None
    assert TERM not in result.model_dump_json()


def test_distinct_courses_with_two_valid_definitions_require_clarification(catalog):
    library, snapshots, retriever = catalog
    first_pair(library)
    publish(library, "lecture", course="other-course")
    query = request(courses=(COURSE, "other-course"), budget=3)
    result = run(retriever, query, allowed=frozenset({COURSE, "other-course"}))
    assert result.stop_reason == "ambiguous_definition_course"
    assert result.missing == ["definition", "experiment"]
    assert result.supplement_count == 0 and len(snapshots.lookups) == 1


def test_initial_support_ends_without_a_literal_lookup(catalog):
    library, snapshots, retriever = catalog
    lecture, _ = first_pair(library)
    lab = publish(library, "lab", text=FILES["lab"]["text"].replace("held-out", "留出集"))
    result = run(retriever, request(question=f"{QUESTION} {TERM}", budget=3))
    assert result.stop_reason == "already_supported"
    assert result.missing == [] and result.supplement_count == 0
    assert len(snapshots.lookups) == 1
    assert [support.actual_source.document_id for support in result.supports] == [
        lecture.document_id,
        lab.document_id,
    ]
    assert result.supports[-1].experiment.split == "留出集"


def test_budget_retains_two_conflicting_experiments_for_later_answer(catalog):
    library, _, retriever = catalog
    lecture, notes = first_pair(library)
    lab = publish(library, "lab")
    conflicting = publish(
        library,
        "lab",
        file_name="A7-reconsidered.ipynb",
        text=FILES["lab"]["text"].replace("Trial: 0.18", "Trial: 0.21"),
    )
    result = run(retriever, request(budget=3))
    assert result.supplement_count == 1 and result.missing == []
    assert len(result.final_evidence) == result.budget == 3
    assert _ids(result.final_evidence) == {
        lecture.document_id,
        lab.document_id,
        conflicting.document_id,
    }
    assert result.replacements[0].removed.source.document_id == notes.document_id
    assert {support.experiment.trial for support in result.supports if support.experiment} == {
        "0.18",
        "0.21",
    }


def test_duplicate_experiment_excerpt_does_not_take_second_slot(catalog):
    library, _, retriever = catalog
    first_pair(library)
    lab = publish(library, "lab")
    repeat = publish(library, "lab", file_name="A7-copy.ipynb")
    result = run(retriever, request(budget=3))
    experiment_sources = {lab.document_id, repeat.document_id}
    assert len(_ids(result.final_evidence) & experiment_sources) == 1
    assert len([support for support in result.supports if support.experiment]) == 1
    assert any(c.reason == "identical_experiment_excerpt" for c in result.hops[1].candidates)


def test_cancellation_after_first_lookup_prevents_native_reading(catalog):
    library, snapshots, retriever = catalog
    first_pair(library)
    publish(library, "lab")
    stopped = False

    def cancel_after_first(query, _snapshot):
        nonlocal stopped
        if query.fts_match:
            stopped = True

    snapshots.after_read = cancel_after_first
    result = run(retriever, cancelled=lambda: stopped)
    assert result.stop_reason == "cancelled"
    assert result.supplement_count == result.source_read_count == 0
    assert len(snapshots.lookups) == 1
    _assert_redacted(result)


def test_off_topic_definition_matching_boilerplate_is_only_auxiliary(catalog):
    library, snapshots, retriever = catalog
    publish(library, "lecture", text="Definition: A diode passes current.\nWhat experiment?")
    publish(library, "notes", text="Documentation notes.")
    result = run(retriever, request(question="What is the definition of median filter experiment?"))
    assert result.hops[0].candidates
    assert result.supports == []
    assert result.missing == ["definition", "experiment"]
    assert result.stop_reason == "no_definition_anchor"
    assert result.supplement_count == 0 and len(snapshots.lookups) == 1


@pytest.mark.parametrize(
    ("question", "lab_text", "problem"),
    [
        (
            QUESTION,
            "This is an experimental plan. The numbers below are expected, not observed.\n\n"
            + FILES["lab"]["text"],
            "experiment_not_observed",
        ),
        (
            QUESTION,
            FILES["lab"]["text"]
            + "\n\nWarning: this run was never performed. No values were measured.",
            "experiment_not_observed",
        ),
        (
            QUESTION + " 本实验用 MAE 评价。",
            FILES["lab"]["text"].replace("Metric: MAE", "Metric: RMSE"),
            "experiment_metric_mismatch",
        ),
        (
            "中值滤波 median filter 的定义是什么？指定仅使用训练集实验结果。",
            FILES["lab"]["text"],
            "experiment_split_mismatch",
        ),
        (
            QUESTION,
            FILES["lab"]["text"].replace(
                " Trial:",
                "\n\n# Different dataset with no shared baseline\n\nTrial:",
            ),
            "experiment_fields_missing",
        ),
        (
            QUESTION,
            FILES["lab"]["text"].replace(" Trial:", " Different dataset. Trial:"),
            "experiment_fields_disconnected",
        ),
        (
            QUESTION + " 不使用留出集。",
            FILES["lab"]["text"],
            "experiment_condition_ambiguous",
        ),
        (
            QUESTION + " 训练集或留出集的 MAE 或 RMSE。",
            FILES["lab"]["text"],
            "experiment_condition_ambiguous",
        ),
    ],
)
def test_unobserved_or_unbound_comparison_cannot_replace_initial_evidence(
    catalog, question, lab_text, problem
):
    library, snapshots, retriever = catalog
    lecture, _ = first_pair(library)
    lab = publish(library, "lab", text=lab_text)
    initial = run(retriever, request(question=question), strategy="single")
    snapshots.calls.clear()
    result = run(retriever, request(question=question))
    assert result.snapshot == initial.snapshot
    assert result.final_evidence == initial.final_evidence
    assert lecture.document_id in _ids(result.final_evidence)
    assert result.replacements == [] and result.missing == ["experiment"]
    assert [item.requirement for item in result.supports] == ["definition"]
    lab_trace = next(
        item
        for hop in result.hops
        for item in hop.candidates
        if item.source.document_id == lab.document_id
    )
    assert lab_trace.read_status == "succeeded" and lab_trace.reason == problem
    assert result.supplement_count == 1
    assert sum(call.fts_match == MATCH for call in snapshots.lookups) == 1


@pytest.mark.parametrize(
    "separation",
    [
        "\n\nSeparate future assignment unrelated to this definition.\n",
        "\nSeparate future assignment unrelated to this definition.\n",
        "\n# Separate dataset\n",
    ],
)
def test_experiment_tag_must_be_attached_to_actual_definition(catalog, separation):
    library, snapshots, retriever = catalog
    lecture = publish(library, "lecture", text=FILES["lecture"]["text"].replace("\n", separation))
    publish(library, "notes")
    publish(library, "lab")
    result = run(retriever)
    assert result.stop_reason == "no_new_terms"
    assert result.missing == ["experiment"] and result.term is None
    assert result.supports[0].actual_source.document_id == lecture.document_id
    assert result.supplement_count == 0 and len(snapshots.lookups) == 1


@pytest.mark.parametrize(
    "foreign_definition",
    ["A diode is a type of semiconductor.", "A mean filter replaces samples by their mean."],
)
def test_explicit_english_definition_topic_ignores_boilerplate_and_other_methods(
    catalog, foreign_definition
):
    library, snapshots, retriever = catalog
    invalid = publish(
        library,
        "lecture",
        text=f"Definition: {foreign_definition}\nExperiment-ID: {TERM}.",
    )
    publish(library, "notes")
    publish(library, "lab")
    question = "What is the definition of median filter and how did the experiment validate it?"
    result = run(retriever, request(question=question))
    assert any(c.source.document_id == invalid.document_id for c in result.hops[0].candidates)
    assert result.missing == ["definition", "experiment"] and result.supports == []
    assert result.term is None and result.supplement_count == 0
    assert len(snapshots.lookups) == 1


def test_explicit_english_topic_and_requested_metric_match_native_data(catalog):
    library, _, retriever = catalog
    lecture, _ = first_pair(library)
    lab = publish(
        library,
        "lab",
        text=FILES["lab"]["text"].replace("Metric: MAE", "Metric: mean absolute error"),
    )
    question = "Define a median filter and explain its held-out experiment using MAE."
    result = run(retriever, request(question=question))
    assert not result.missing
    assert result.supports[0].actual_source.document_id == lecture.document_id
    assert result.supports[1].actual_source.document_id == lab.document_id
    assert result.supports[1].experiment.metric == "mean absolute error"


@pytest.mark.parametrize(
    ("question", "other_definition"),
    [
        (
            "What is the definition of median filter and how did the experiment compare it "
            "with ridge regression?",
            "Ridge regression minimizes a quadratic penalty.",
        ),
        (
            "What is the definition of median filter compared with ridge regression "
            "in the experiment?",
            "Ridge regression minimizes a quadratic penalty.",
        ),
        (
            "What is the definition of median filter and how did the experiment validate it?",
            "Ridge regression differs from a median filter.",
        ),
    ],
)
def test_whole_question_glossary_does_not_change_explicit_definition_subject(
    catalog, question, other_definition
):
    library, snapshots, retriever = catalog
    unrelated = publish(
        library,
        "lecture",
        text=f"Definition: {other_definition}\nExperiment-ID: {TERM}.",
    )
    publish(library, "notes")
    publish(library, "lab")
    result = run(retriever, request(question=question))
    assert any(c.source.document_id == unrelated.document_id for c in result.hops[0].candidates)
    assert result.missing == ["definition", "experiment"]
    assert result.supports == [] and result.term is None
    assert result.supplement_count == 0 and len(snapshots.lookups) == 1


@pytest.mark.parametrize(
    "preface",
    [
        "Experiment proposal. The numbers below are hypothetical and illustrate "
        "the anticipated outcome.",
        "这是实验提案，下面仅是假设数据，用于演示预估结果。",
    ],
)
def test_explicit_proposal_numbers_do_not_supply_observed_comparison(catalog, preface):
    library, _, retriever = catalog
    lecture, _ = first_pair(library)
    lab = publish(library, "lab", text=f"{preface}\n\n{FILES['lab']['text']}")
    question = "Define a median filter and explain its held-out experiment using MAE."
    result = run(retriever, request(question=question))
    trace = next(c for c in result.hops[0].candidates if c.source.document_id == lab.document_id)
    assert trace.read_status == "succeeded" and trace.reason == "experiment_not_observed"
    assert result.replacements == [] and result.missing == ["experiment"]
    assert [item.actual_source.document_id for item in result.supports] == [lecture.document_id]
    assert result.supplement_count <= 1


@pytest.mark.parametrize(
    ("question", "original"),
    [
        (
            "Define ridge regression and explain its held-out experiment using MAE.",
            "岭回归 使用二次惩罚约束模型系数。",
        ),
        (
            "请给出岭回归的定义，如何验证其留出集实验？",
            "Ridge regression applies a quadratic penalty to coefficients.",
        ),
    ],
)
def test_glossary_alias_of_explicit_topic_is_retained_in_both_languages(
    catalog, question, original
):
    library, _, retriever = catalog
    lecture = publish(library, "lecture", text=f"Definition: {original}\nExperiment-ID: {TERM}.")
    publish(library, "notes")
    lab = publish(library, "lab")
    result = run(retriever, request(question=question, budget=3))
    assert result.missing == []
    assert result.supports[0].actual_source.document_id == lecture.document_id
    assert result.supports[0].quote == original
    assert result.supports[1].actual_source.document_id == lab.document_id
    assert result.supplement_count <= 1


@pytest.mark.parametrize(
    "topic",
    [
        "中值滤波（不是岭回归）",
        "中值滤波相对于岭回归",
        "中值滤波而非岭回归",
        "中值滤波或岭回归",
    ],
)
def test_unresolved_chinese_topic_cannot_borrow_comparison_alias(catalog, topic):
    library, snapshots, retriever = catalog
    unrelated = publish(
        library,
        "lecture",
        text=(
            f"Definition: 岭回归使用二次惩罚约束参数。\nExperiment-ID: {TERM}.\n\n"
            "Related topic for comparison: median filter. median filter."
        ),
    )
    publish(library, "notes")
    publish(library, "lab")
    question = f"请给出{topic}的定义，并说明留出集实验。关键词 median filter。"
    result = run(retriever, request(question=question))
    assert any(c.source.document_id == unrelated.document_id for c in result.hops[0].candidates)
    assert result.missing == ["definition", "experiment"]
    assert result.supports == [] and result.term is None
    assert result.stop_reason == "no_definition_anchor"
    assert result.supplement_count == 0 and len(snapshots.lookups) == 1
