"""Replay F04 in a public SRAG archive with isolated SQLite and outbound sockets blocked."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import nbformat

import app
from app.config import Settings
from app.rag.evidence_answer import AnswerRun, EvidenceAnswerer
from app.rag.evidence_supplement import SupplementRetriever
from app.rag.local_library import LocalLibrary
from app.rag.structure_search import StructureScope, StructureSearchRequest
from app.stores.structure_snapshot import SnapshotRequest, StructureSnapshotStore

BASELINE = "6651933fa6a15aa90bee0751e33b72d15afa1fcf"
SAMPLE = "docs/F/样例/SignalTrialA7.json"
BUDGET = 2
TERM = "SignalTrialA7"
OFF_COURSE = "signal-demo-other"
HASHES = {
    "app/rag/evidence_answer.py": (
        "082eeb397eaa33f9a94fb1c78a9d8288a87ec819f6e45b5b8d1675426d73026b"
    ),
    "app/rag/evidence_supplement.py": (
        "696d0a32a56355f47bb84b9f9ca49293fd0d09d7d6e9a988da08c7ca81ae9b55"
    ),
    "app/rag/structure_search.py": (
        "2a23752195f25605eff484556519d8dce7b14bb0270cda6a373998904d9153c8"
    ),
    "app/stores/local_library.py": (
        "42acdcbbb107cfb23228041f1ddd6c629aac381253f2d917bae257825c2460bb"
    ),
    "app/stores/structure_snapshot.py": (
        "0662a65cf7ddd9939cea74246a0b9ce2acad43399b45cf96fb9dedfde992d965"
    ),
    "tools/evidence_answer.py": "cdbe80fa467c09b068bbae7a081547740175bffb8359f2747c5bafe6a6299960",
    "app/schemas/turn_evidence.py": (
        "44572c6fccb5fc3bbcf0648132e3db2f4ff625a162cad66b0ec7cb578367780f"
    ),
    SAMPLE: "1806d8bf248ac1d0c897f1efd75298ed6d7fe4e7e8558fd56762f2ee71116119",
}


@dataclass(frozen=True)
class Case:
    name: str
    lab: str
    expected_stop: str
    expected_reason: str | None = None
    exclude_lab: bool = False


CASES = (
    Case("completion_at_full_budget", "observed", "supplemented"),
    Case("duplicate_only", "absent", "duplicate_only"),
    Case("mismatched_train_split", "training", "mismatched", "experiment_split_mismatch"),
    Case("still_missing_trial_value", "incomplete", "still_missing", "experiment_fields_missing"),
    Case("two_authorized_courses", "foreign", "duplicate_only"),
    Case("explicit_file_exclusion", "observed", "duplicate_only", exclude_lab=True),
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _blocked(*args, **kwargs):
    raise AssertionError("unexpected outbound socket during F04")


def _elapsed(start: float) -> float:
    return round((perf_counter() - start) * 1000, 3)


class _SnapshotLedger(StructureSnapshotStore):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.calls: list[dict] = []

    def read(self, request: SnapshotRequest):
        started = perf_counter()
        result = super().read(request)
        self.calls.append(
            {
                "course_ids": request.course_ids,
                "document_ids": request.document_ids,
                "fts_match": request.fts_match,
                "fts_limit": request.fts_limit,
                "elapsed_ms": _elapsed(started),
            }
        )
        return result


def _prepare(case: Case, root: Path, sample: dict):
    courses = root / "courses"
    courses.mkdir()
    settings = Settings(
        _env_file=None,
        course_root=courses,
        data_dir=root / "data",
        sqlite_path=root / "library.sqlite3",
        dashscope_api_key="",
        deepseek_api_key="",
    )
    library = LocalLibrary(settings)
    labels = {}
    user_course = sample["course_id"]
    for item in sample["documents"]:
        role = item["role"]
        is_lab = role == "lab"
        if is_lab and case.lab == "absent":
            continue
        text = item["text"]
        if is_lab and case.lab == "training":
            text = text.replace("Split: held-out", "Split: training")
        if is_lab and case.lab == "incomplete":
            text = text.replace(" Trial: 0.18.", "")
        course = OFF_COURSE if is_lab and case.lab == "foreign" else user_course
        path = courses / item["file_name"]
        cell = nbformat.v4.new_markdown_cell(text, id="evidence")
        nbformat.write(nbformat.v4.new_notebook(cells=[cell]), path)
        result = library.import_paths(course, [path])[0]
        assert result.status == "published", (case.name, result)
        labels[role] = (item["file_name"], result.document_id)
    return settings, library, labels


def _statements(run: AnswerRun):
    return [item for section in run.answer.sections for item in section.statements]


def _check_citations(run: AnswerRun) -> None:
    report = run.retrieval
    final = {item.evidence_id: item for item in report.final_evidence}
    actual = {item.evidence_id: item for item in report.reads}
    assert len(final) == len(report.final_evidence) <= BUDGET
    for statement in _statements(run):
        cite = statement.citation
        kept = final[cite.evidence_id]
        read = actual[cite.evidence_id].reads[0]
        assert cite.actual_source == kept.revision.source == read.actual_source
        assert kept == actual[cite.evidence_id]
        assert cite.revision == kept.revision and cite.locator == kept.locator
        assert cite.method == read.method == "native_text"
        assert cite.read_status == read.status == "succeeded"
        assert 0 <= cite.start < cite.end <= len(read.text)
        assert read.text[cite.start : cite.end] == cite.quote
    used = [statement.citation.actual_source for statement in _statements(run)]
    assert all(row.removed.source not in used for row in report.replacements)


def _serialize(run: AnswerRun, calls: list[dict], wall_ms: float, excerpts: dict) -> dict:
    report = run.retrieval
    matches = [call for call in calls if call["fts_match"] is not None]
    reads = {}
    for evidence in report.reads:
        parts = evidence.model_dump(mode="json")
        for part in parts["reads"]:
            literal = part.pop("text")
            digest = _sha(literal.encode("utf-8"))
            excerpts[digest] = literal
            part["text_sha256"] = digest
        reads[evidence.evidence_id] = parts
    return {
        "wall_ms": wall_ms,
        "retrieval_ms": report.elapsed_ms,
        "first_hop_ms": report.hops[0].elapsed_ms,
        "match_calls": matches,
        "non_match_snapshot_calls": len(calls) - len(matches),
        "supplement_count": report.supplement_count,
        "supplement_hop": (
            report.hops[1].model_dump(mode="json") if report.supplement_count else None
        ),
        "term": report.term.model_dump(mode="json") if report.term else None,
        "stop_reason": report.stop_reason,
        "missing": report.missing,
        "reads": reads,
        "final_evidence_ids": [item.evidence_id for item in report.final_evidence],
        "replacements": [item.model_dump(mode="json") for item in report.replacements],
        "answer": run.answer.model_dump(mode="json"),
    }


def _cli(
    repo: Path, root: Path, settings: Settings, course: str, question: str, runs
) -> list[dict]:
    marker = root / "network-guard-loaded"
    (root / "sitecustomize.py").write_text(
        "import os, socket\n"
        "from pathlib import Path\n"
        "def deny(*a, **k): raise AssertionError('unexpected outbound socket during F04 CLI')\n"
        "socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny\n"
        "with Path(os.environ['SRAG_F04_GUARD']).open('a') as f: f.write('loaded\\n')\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(root), str(repo))),
        "PYTHONDONTWRITEBYTECODE": "1",
        "COURSE_ROOT": str(settings.course_root),
        "DATA_DIR": str(settings.data_dir),
        "SQLITE_PATH": str(settings.sqlite_path),
        "DASHSCOPE_API_KEY": "",
        "DEEPSEEK_API_KEY": "",
        "SRAG_STRUCTURE_COURSES": course,
        "SRAG_F04_GUARD": str(marker),
    }
    evidence = []
    for strategy, expected in runs.items():
        started = perf_counter()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.evidence_answer",
                question,
                "--course",
                course,
                "--top-k",
                str(BUDGET),
                "--strategy",
                strategy,
                "--trace",
            ],
            cwd=repo,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        elapsed = _elapsed(started)
        assert not result.stderr, result.stderr
        payload = AnswerRun.model_validate_json(result.stdout)
        _check_citations(payload)
        assert payload.answer == expected.answer
        assert payload.retrieval.snapshot == expected.retrieval.snapshot
        evidence.append(
            {
                "strategy": strategy,
                "exit_code": result.returncode,
                "wall_ms": elapsed,
                "status": payload.answer.status,
                "supplement_count": payload.retrieval.supplement_count,
                "citations": [row.citation.model_dump(mode="json") for row in _statements(payload)],
            }
        )
    assert marker.read_text().splitlines() == ["loaded", "loaded"]
    return evidence


def _check_pair(case: Case, single: AnswerRun, added: AnswerRun, labels, sample: dict) -> None:
    before, after = single.retrieval, added.retrieval
    assert before.budget == after.budget == BUDGET
    assert before.question == after.question == sample["question"]
    assert before.snapshot == after.snapshot and before.scope == after.scope
    assert before.hops[0].model_dump(exclude={"elapsed_ms"}) == after.hops[0].model_dump(
        exclude={"elapsed_ms"}
    )
    assert [row.source.document_id for row in before.hops[0].candidates] == [
        labels[role][1] for role in ("lecture", "notes")
    ]
    assert before.missing == ["experiment"] and single.answer.status == "partial"
    assert before.supplement_count == 0 and after.supplement_count == 1
    assert len(before.final_evidence) == len(after.final_evidence) == BUDGET
    assert after.stop_reason == added.answer.stop_reason == case.expected_stop
    assert after.hops[1].query == TERM
    assert after.hops[1].scope.course_ids == [sample["course_id"]]
    expected_ids = {labels[role][1] for role in ("lecture", "notes")}
    if case.lab not in ("foreign", "absent") and not case.exclude_lab:
        expected_ids.add(labels["lab"][1])
    assert set(after.hops[1].scope.document_ids) == expected_ids
    if case.lab == "foreign":
        assert labels["lab"][1] in {doc.document_id for doc in after.snapshot}
    if case.exclude_lab:
        assert labels["lab"][1] not in {doc.document_id for doc in after.snapshot}
    if case.expected_reason:
        assert any(row.reason == case.expected_reason for row in after.hops[1].candidates)
    term = after.term
    read = next(item for item in after.reads if item.evidence_id == term.evidence_id).reads[0]
    assert term.source == read.actual_source
    assert any(item.source == term.source for item in after.hops[0].candidates)
    assert read.text[term.start : term.end] == term.word == TERM
    assert _sha(read.text.encode("utf-8")) == term.excerpt_sha256
    if case.name == "completion_at_full_budget":
        assert after.missing == [] and added.answer.status == "complete"
        experiment = added.answer.sections[1].statements[0].experiment
        assert (experiment.split, experiment.metric, experiment.unit, experiment.samples) == (
            "held-out",
            "MAE",
            "amplitude",
            4,
        )
        baseline = sum(map(Decimal, ("0.20", "0.24", "0.28", "0.24"))) / 4
        trial = sum(map(Decimal, ("0.16", "0.20", "0.18", "0.18"))) / 4
        assert Decimal(experiment.baseline) == baseline and Decimal(experiment.trial) == trial
        assert len(after.replacements) == 1
        replacement = after.replacements[0]
        assert replacement.removed.source.document_id == labels["notes"][1]
        assert replacement.added.source.document_id == labels["lab"][1]
    else:
        assert after.missing == ["experiment"] and added.answer.status == "partial"
        assert not after.replacements and not added.answer.sections[1].statements


def _run_case(case: Case, repo: Path, root: Path, sample: dict, excerpts: dict) -> dict:
    settings, library, labels = _prepare(case, root, sample)
    ledger = _SnapshotLedger(settings)
    answerer = EvidenceAnswerer(SupplementRetriever(library.store, ledger))
    course = sample["course_id"]
    courses = [course, OFF_COURSE] if case.lab == "foreign" else [course]
    files = [labels[role][0] for role in ("lecture", "notes")] if case.exclude_lab else []
    question = sample["question"]
    request = StructureSearchRequest(
        query=question,
        scope=StructureScope(course_ids=courses, file_names=files),
        top_k=BUDGET,
        include_adjacency=False,
        include_page_fallback=False,
    )
    runs, traces = {}, {}
    for strategy in ("single", "supplement"):
        ledger.calls.clear()
        started = perf_counter()
        result = answerer.answer(request, allowed_course_ids=frozenset(courses), strategy=strategy)
        wall_ms = _elapsed(started)
        _check_citations(result)
        logical_matches = [call for call in ledger.calls if call["fts_match"] is not None]
        if strategy == "single":
            assert len(logical_matches) == 1
        else:
            assert len(logical_matches) == 2
            assert logical_matches[1]["fts_match"] == f'"{TERM}"'
            assert logical_matches[1]["fts_limit"] == BUDGET
        runs[strategy] = result
        traces[strategy] = _serialize(result, ledger.calls, wall_ms, excerpts)
    single, added = runs["single"], runs["supplement"]
    _check_pair(case, single, added, labels, sample)
    assert (
        traces["single"]["match_calls"][0]["fts_match"]
        == traces["supplement"]["match_calls"][0]["fts_match"]
    )
    before = single.retrieval
    return {
        "case": case.name,
        "input_scope": request.scope.model_dump(mode="json"),
        "published_documents": labels,
        "effective_scope": before.scope.model_dump(mode="json"),
        "snapshot": [row.model_dump(mode="json") for row in before.snapshot],
        "first_hop_candidates": [row.model_dump(mode="json") for row in before.hops[0].candidates],
        "first_hop_supports": [row.model_dump(mode="json") for row in before.supports],
        "first_hop_missing": before.missing,
        "strategies": traces,
        "cli": (
            _cli(repo, root, settings, course, question, runs)
            if case.name == "completion_at_full_budget"
            else []
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    output = parser.parse_args().out.resolve()
    repo = Path.cwd()
    assert Path(app.__file__).resolve().parent.parent == repo, "run inside the exact archive"
    assert {name: _sha((repo / name).read_bytes()) for name in HASHES} == HASHES
    sample = json.loads((repo / SAMPLE).read_text(encoding="utf-8"))
    excerpts, evidence = {}, []
    with (
        patch.object(socket.socket, "connect", _blocked),
        patch.object(socket.socket, "connect_ex", _blocked),
        patch.object(socket, "create_connection", _blocked),
    ):
        for case in CASES:
            with TemporaryDirectory(prefix="srag-f04-case.") as temporary:
                evidence.append(_run_case(case, repo, Path(temporary), sample, excerpts))
    payload = {
        "baseline": BASELINE,
        "verified_sha256": HASHES,
        "question": sample["question"],
        "budget": BUDGET,
        "read_excerpts_by_sha256": excerpts,
        "cases": evidence,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(evidence)} paired cases passed, socket guard enabled, trace {output}")


if __name__ == "__main__":
    main()
