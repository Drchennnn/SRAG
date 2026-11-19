"""Three-role composition against published native sources and real SQLite FTS."""

from __future__ import annotations

import runpy
import socket
from pathlib import Path

import nbformat
import pytest
from pptx import Presentation
from pptx.util import Inches

from app.config import Settings
from app.rag.local_library import LocalLibrary
from app.rag.structure_search import StructureSearcher
from app.rag.workspace_composition import asks_for_composition, compose_evidence
from app.schemas.workspace import WorkspaceError
from app.stores.structure_snapshot import SnapshotRequest, StructureSnapshotStore

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "docs/G/G05-probe.py"
COURSE = "pulse-evidence"
QUESTION = (
    "What is the definition of a pulse gate, and how did the held-out experiment validate it? "
    "Use the threshold from the separate limit table and compare it with the saved output."
)
PRINCIPLE = "Definition: A pulse gate passes only pulses above its operating threshold."
RECORDED = (
    "Experiment-ID: TidalRunQ9. Split: held-out. Metric: MAE. Unit: amplitude. "
    "Baseline: 0.42. Trial: 0.29. Samples: 7."
)


class AuditedStore(StructureSnapshotStore):
    def __init__(self, settings):
        super().__init__(settings)
        self.matches = []

    def read(self, request):
        result = super().read(request)
        if request.fts_match:
            self.matches.append(request)
        return result


@pytest.fixture
def published(tmp_path, monkeypatch):
    def forbid(*args, **kwargs):
        pytest.fail("a local composition made an external request")

    monkeypatch.setattr(socket.socket, "connect", forbid)
    monkeypatch.setattr(socket.socket, "connect_ex", forbid)
    settings = Settings(
        _env_file=None,
        course_root=tmp_path,
        sqlite_path=tmp_path / "catalog.sqlite3",
        data_dir=tmp_path / "runtime",
        dashscope_api_key="",
        deepseek_api_key="",
    )
    library = LocalLibrary(settings)
    snapshots = AuditedStore(settings)
    fixture = runpy.run_path(str(PUBLIC))
    fixture["build"](tmp_path)
    paths = tmp_path / "fixtures"
    return library, snapshots, paths


def publish(published, path, *, course=COURSE, document_id=None):
    library, _, _ = published
    result = library.import_paths(course, [path], document_id=document_id)[0]
    assert result.status == "published", result
    return result.document_id


def table(
    path,
    header="MAE maximum (amplitude)",
    split="held-out",
    *,
    scalar="0.30",
    subject="Pulse gate",
    context=(),
):
    pptx = Presentation()
    slide = pptx.slides.add_slide(pptx.slide_layouts[6])
    grid = slide.shapes.add_table(
        2, 3 + len(context), Inches(1), Inches(1), Inches(8), Inches(2)
    ).table
    rows = (
        ("Series", header, "Dataset", *(key for key, _ in context)),
        (subject, scalar, split, *(value for _, value in context)),
    )
    for row, values in enumerate(rows):
        for column, value in enumerate(values):
            grid.cell(row, column).text = value
    pptx.save(path)


def notebook(path, cell):
    nbformat.write(nbformat.v4.new_notebook(cells=[cell]), path)


def solve(published, *, courses=(COURSE,), files=(), question=QUESTION, active=lambda: True):
    library, snapshots, _ = published
    snapshot = snapshots.read(SnapshotRequest(courses, files))
    return compose_evidence(
        question,
        snapshot,
        StructureSearcher(snapshots),
        library.store,
        allowed_courses=frozenset(courses),
        active=active,
    )


@pytest.mark.parametrize(
    "question,expected",
    [
        (QUESTION, True),
        ("解释脉冲门的定义，结合参数表及保存的实验输出。", True),
        ("请分别解释脉冲门的定义，结合参数表及保存的实验输出。", True),
        ("分别解释脉冲门的定义和实验输出，并别引用参数表。", False),
        ("给出脉冲门的定义，结合限值表及保存的运行输出。", True),
        (
            "Explain the pulse gate definition with the saved output but do not consult the table.",
            False,
        ),
        ("What is the formula of pulse gate? Use the table and saved output.", True),
        (QUESTION + " Skip the table.", False),
        ("Explain the definition and experiment. Use the limit table.", False),
        ("Read the table and the saved output. 不用定义。", False),
        ("What is the definition of a pulse gate and how did the experiment validate it?", False),
    ],
)
def test_requires_all_three_roles_without_negation(question, expected):
    assert asks_for_composition(question) is expected


def test_public_fixture_preserves_real_units_and_exact_three_roles(published):
    _, snapshots, paths = published
    for name in ("PulseGate-lecture.ipynb", "PulseGate-limits.pptx", "PulseGate-run.ipynb"):
        publish(published, paths / name)
    result = solve(published)
    readbacks = {evidence.file_name: evidence for evidence in result.evidence.values()}
    assert set(readbacks) == {
        "PulseGate-lecture.ipynb",
        "PulseGate-limits.pptx",
        "PulseGate-run.ipynb",
    }
    assert result.trace["missing"] == []
    assert result.trace["comparison"] == "unverified_alignment"
    assert result.trace["external_call_count"] == 0
    for evidence in readbacks.values():
        assert evidence.read_status == "succeeded"
        assert evidence.reads[0].actual_source == evidence.revision.source
    table_ref = readbacks["PulseGate-limits.pptx"]
    assert table_ref.revision.source.source_id == "page:1:shape:2"
    assert table_ref.reads[0].method == "native_table"
    assert "17.5" in table_ref.reads[0].text
    assert readbacks["PulseGate-run.ipynb"].revision.source.source_id == "cell:0:output:0"
    assert all(s.citations and set(s.citations) <= set(result.evidence) for s in result.statements)
    assert len(snapshots.matches) <= 2


@pytest.mark.parametrize("metric_limit, relation", [("0.30", "不高于"), ("0.27", "高于")])
def test_numeric_comparison_needs_real_matched_limit_and_output(published, metric_limit, relation):
    _, _, paths = published
    limit = paths / "MAE-limit.pptx"
    table(limit, scalar=metric_limit)
    for path in (paths / "PulseGate-lecture.ipynb", limit, paths / "PulseGate-run.ipynb"):
        publish(published, path)
    result = solve(published)
    assert result.trace["comparison"] == "verified"
    assert result.trace["answer_status"] == "supported"
    last = result.statements[-1]
    assert f" {relation}同条件表格上限" in last.text
    assert {result.evidence[ref].file_name for ref in last.citations} == {
        "MAE-limit.pptx",
        "PulseGate-run.ipynb",
    }


@pytest.mark.parametrize(
    "header,context",
    [
        ("Baseline MAE maximum (amplitude)", ()),
        ("MAE (amplitude), not an upper limit", ()),
        ("MAE maximum (amplitude)", (("Temperature", "5 C"),)),
    ],
)
def test_non_applicable_comparison_keeps_originals_without_derived_claim(
    published, header, context
):
    _, _, paths = published
    limit = paths / "Qualified-limit.pptx"
    table(limit, header=header, context=context)
    for path in (paths / "PulseGate-lecture.ipynb", limit, paths / "PulseGate-run.ipynb"):
        publish(published, path)
    result = solve(published)
    assert result.trace["missing"] == []
    assert result.trace["comparison"] == "unverified_alignment"
    assert all(len(item.citations) == 1 for item in result.statements)


@pytest.mark.parametrize(
    "subject,split,context",
    [
        ("Adaptive smoothing", "Pulse gate held-out group", ()),
        ("Adaptive smoothing", "held-out", (("Experiment ID", "TidalRunQ9"),)),
        ("Pulse gate", "not held-out", ()),
    ],
)
def test_table_other_columns_cannot_substitute_for_subject_or_split(
    published, subject, split, context
):
    _, _, paths = published
    limit = paths / "Wrong-context-limit.pptx"
    table(limit, split=split, subject=subject, context=context)
    for path in (paths / "PulseGate-lecture.ipynb", limit, paths / "PulseGate-run.ipynb"):
        publish(published, path)
    result = solve(published)
    assert result.trace["missing"] == ["table"]
    assert result.trace["comparison"] == "missing_evidence"
    assert all("0.30" not in item.text for item in result.statements)


def test_bilingual_principle_and_literal_topic_bind_native_table_and_output(published):
    _, snapshots, paths = published
    lecture = paths / "Bilingual-lecture.ipynb"
    native = "定义：脉冲门 pulse gate 只放行超过阈值的脉冲。\n实验编号：TidalRunQ9。"
    notebook(lecture, nbformat.v4.new_markdown_cell(native))
    limit = paths / "Bilingual-limit.pptx"
    table(limit, context=(("Experiment ID", "TidalRunQ9"),))
    for path in (lecture, limit, paths / "PulseGate-run.ipynb"):
        publish(published, path)
    question = "给出脉冲门的定义，结合限值表及保存的运行输出。"
    result = solve(published, question=question)
    assert result.trace["missing"] == []
    assert set(result.trace["supports"]) == {"principle", "table", "saved_output"}
    assert "脉冲门" in snapshots.matches[0].fts_match
    assert len(snapshots.matches) <= 2
    assert "脉冲门 pulse gate" in result.trace["supports"]["principle"]["quote"]


@pytest.mark.parametrize("gap", ["wrong_split", "missing_table", "adjacent_code", "foreign_output"])
def test_missing_roles_and_scope_do_not_invent_support(published, gap):
    _, snapshots, paths = published
    publish(published, paths / "PulseGate-lecture.ipynb")
    if gap == "wrong_split":
        training = paths / "Training-limit.pptx"
        table(training, split="training", scalar="0.10")
        publish(published, training)
    elif gap != "missing_table":
        publish(published, paths / "PulseGate-limits.pptx")
    if gap == "adjacent_code":
        code = paths / "Code-only.ipynb"
        notebook(code, nbformat.v4.new_code_cell(RECORDED))
        publish(published, code)
    else:
        output_course = "other-evidence" if gap == "foreign_output" else COURSE
        publish(published, paths / "PulseGate-run.ipynb", course=output_course)
    courses = (COURSE, "other-evidence") if gap == "foreign_output" else (COURSE,)
    result = solve(published, courses=courses)
    missing = "table" if gap in {"wrong_split", "missing_table"} else "saved_output"
    assert missing in result.trace["missing"]
    assert any(code.endswith(missing) for code, _ in result.notices)
    assert result.trace["comparison"] == "missing_evidence"
    text = " ".join(statement.text for statement in result.statements)
    assert ("0.29" in text) is (gap in {"missing_table", "wrong_split"})
    assert "0.10" not in text
    assert {item.revision.course_id for item in result.evidence.values()} <= {COURSE}
    assert len(snapshots.matches) <= 2
    if result.trace["supplement_count"]:
        extra = snapshots.matches[-1]
        assert extra.fts_match == '"TidalRunQ9"'
        assert extra.course_ids == (COURSE,)
        assert extra.document_ids


def test_conflicting_table_limits_are_not_returned_as_fact(published):
    _, _, paths = published
    for index, limit in enumerate(("0.30", "0.25")):
        path = paths / f"MAE-limit-{index}.pptx"
        table(path, scalar=limit)
        publish(published, path)
    for name in ("PulseGate-lecture.ipynb", "PulseGate-run.ipynb"):
        publish(published, paths / name)
    result = solve(published)
    assert "table" in result.trace["missing"]
    assert "table" in result.trace["conflicts"]
    assert "verified" != result.trace["comparison"]
    assert all(item.reads[0].method != "native_table" for item in result.evidence.values())


@pytest.mark.parametrize("separator", ["\n", "\n\n", "\n## Other experiment\n"])
def test_conditions_link_only_to_an_anchor_in_the_same_block(published, separator):
    _, _, paths = published
    lecture = paths / "Conditioned-lecture.ipynb"
    principle = (
        PRINCIPLE
        + "\nFormula: output = pulse if pulse > threshold."
        + "\nCondition: Only the held-out run is considered."
        + separator
        + "Experiment-ID: TidalRunQ9."
    )
    notebook(lecture, nbformat.v4.new_markdown_cell(principle))
    limit = paths / "MAE-limit.pptx"
    table(limit)
    for path in (lecture, limit, paths / "PulseGate-run.ipynb"):
        publish(published, path)
    result = solve(published)
    support = result.trace["supports"]["principle"]
    quote = support["quote"]
    assert "Formula: output =" in quote
    assert "Condition: Only the held-out run" in quote
    assert "Experiment-ID" not in quote
    assert principle[slice(*support["span"])] == quote
    assert ("saved_output" not in result.trace["missing"]) is (separator == "\n")
    if separator != "\n":
        assert result.trace["supplement_count"] == 0
        assert "0.29" not in " ".join(item.text for item in result.statements)


def test_cancelled_turn_checks_before_retrieval(published):
    _, snapshots, paths = published
    publish(published, paths / "PulseGate-lecture.ipynb")
    with pytest.raises(WorkspaceError, match="interrupted"):
        solve(published, active=lambda: False)
    assert not snapshots.matches
