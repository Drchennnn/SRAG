"""B03 raw Notebook import contract tests."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from app.rag.parsers.local_notebook import parse_local_notebook

SAMPLE_V1 = Path("docs/A/样例/Signal-lab.ipynb")
SAMPLE_V2 = Path("docs/A/样例/revisions/v2/Signal-lab.ipynb")


def _write_notebook(path: Path, cells: list[object]) -> None:
    path.write_text(
        json.dumps(
            {
                "cells": cells,
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_parses_raw_order_sections_code_and_every_saved_output(tmp_path: Path) -> None:
    image = base64.b64encode(b"same-image").decode("ascii")
    path = tmp_path / "ordered.ipynb"
    _write_notebook(
        path,
        [
            {
                "cell_type": "markdown",
                "id": "intro",
                "metadata": {},
                "source": ["# Course\n", "## Topic\n", "Read only."],
            },
            {
                "cell_type": "code",
                "id": "plots",
                "metadata": {},
                "source": ["print('do not execute')"],
                "execution_count": 1,
                "outputs": [
                    {"output_type": "stream", "name": "stdout", "text": ["saved\n"]},
                    {
                        "output_type": "display_data",
                        "metadata": {},
                        "data": {"image/png": image},
                    },
                    {"output_type": "display_data", "metadata": {}, "data": {}},
                    {
                        "output_type": "display_data",
                        "metadata": {},
                        "data": {
                            "text/plain": ["chart caption"],
                            "image/png": image,
                            "image/svg+xml": "<svg><text>chart</text></svg>",
                        },
                    },
                ],
            },
        ],
    )

    parsed = parse_local_notebook(path)

    assert parsed.title == "Course"
    assert [source.source_id for source in parsed.sources] == [
        "cell:0",
        "cell:1",
        "cell:1:output:0",
        "cell:1:output:1",
        "cell:1:output:2",
        "cell:1:output:3",
    ]
    assert [source.ordinal for source in parsed.sources] == list(range(6))
    markdown, code, stream, first_image, blank, bundle = parsed.sources
    assert markdown.title == "Topic"
    assert markdown.locator["section_path"] == ["Course", "Topic"]
    assert code.text == "print('do not execute')"
    assert code.locator["cell_id"] == "plots"
    assert code.locator["outputs_state"] == "available"
    assert stream.text == "saved\n"
    assert first_image.parent_id == "cell:1"
    assert blank.state == "empty"
    assert bundle.text == "chart caption"
    assert [item["mime_type"] for item in bundle.locator["representations"]] == [
        "text/plain",
        "image/png",
        "image/svg+xml",
    ]
    first_png = first_image.locator["representations"][0]
    second_png = bundle.locator["representations"][1]
    assert first_png["sha256"] == second_png["sha256"]
    assert first_png["json_pointer"] == "/cells/1/outputs/1/data/image~1png"
    assert second_png["json_pointer"] == "/cells/1/outputs/3/data/image~1png"
    assert second_png["encoding"] == "base64"
    assert bundle.locator["representations"][2]["encoding"] == "utf-8"


def test_missing_invalid_and_duplicate_cell_ids_fall_back_to_indices(tmp_path: Path) -> None:
    path = tmp_path / "old-style.ipynb"
    _write_notebook(
        path,
        [
            {"cell_type": "markdown", "metadata": {}, "source": "# Old"},
            {"cell_type": "code", "id": "same", "metadata": {}, "source": "a = 1"},
            {"cell_type": "code", "id": "same", "metadata": {}, "source": "b = 2"},
            {"cell_type": "raw", "id": "bad id", "metadata": {}, "source": "raw"},
            {"cell_type": "raw", "id": "unique_ok", "metadata": {}, "source": "raw"},
        ],
    )

    parsed = parse_local_notebook(path)

    assert [source.source_id for source in parsed.sources] == [f"cell:{i}" for i in range(5)]
    assert [source.locator["cell_id"] for source in parsed.sources] == [
        None,
        None,
        None,
        None,
        "unique_ok",
    ]


def test_empty_code_missing_outputs_and_malformed_media_are_explicit(tmp_path: Path) -> None:
    path = tmp_path / "states.ipynb"
    _write_notebook(
        path,
        [
            {"cell_type": "code", "metadata": {}, "source": ""},
            {"cell_type": "code", "metadata": {}, "source": "pass", "outputs": []},
            {
                "cell_type": "code",
                "metadata": {},
                "source": "display(x)",
                "outputs": [
                    {
                        "output_type": "display_data",
                        "metadata": {},
                        "data": {"text/plain": "still readable", "image/png": "not base64!"},
                    },
                    None,
                    {"output_type": [], "text": "bad type"},
                    {
                        "output_type": "display_data",
                        "metadata": {},
                        "data": {"image/png": ""},
                    },
                ],
            },
        ],
    )

    parsed = parse_local_notebook(path)

    (
        first,
        second,
        third,
        partial_output,
        malformed_output,
        unknown_output,
        empty_image,
    ) = parsed.sources
    assert first.state == "empty"
    assert first.locator["content_state"] == "empty"
    assert first.locator["outputs_state"] == "missing"
    assert second.locator["outputs_state"] == "empty"
    assert third.locator["outputs_state"] == "available"
    assert partial_output.state == "partial"
    assert partial_output.text == "still readable"
    assert partial_output.locator["representations"][1] == {
        "mime_type": "image/png",
        "json_pointer": "/cells/2/outputs/0/data/image~1png",
        "status": "malformed",
        "encoding": "base64",
        "error": "invalid_image_data",
    }
    assert malformed_output.state == "malformed"
    assert malformed_output.locator["parse_error"] == "output_not_object"
    assert unknown_output.state == "available"
    assert unknown_output.locator["output_type"] == "unknown"
    assert empty_image.state == "empty"
    assert empty_image.locator["representations"][0]["status"] == "empty"


def test_non_code_outputs_are_reported_without_creating_invalid_parents(tmp_path: Path) -> None:
    path = tmp_path / "non-code-outputs.ipynb"
    _write_notebook(
        path,
        [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "# Valid heading\n```python\n# Not a heading\n```\nText",
                "outputs": [{"output_type": "stream", "text": "invalid parent"}],
            }
        ],
    )

    parsed = parse_local_notebook(path)

    assert [source.source_id for source in parsed.sources] == ["cell:0"]
    assert parsed.sources[0].locator["outputs_state"] == "unexpected"
    assert parsed.sources[0].locator["headings"] == [
        {"level": 1, "title": "Valid heading", "section_path": ["Valid heading"]}
    ]


def test_import_never_executes_code_or_reads_markdown_resource(tmp_path: Path) -> None:
    marker = tmp_path / "executed.txt"
    external = tmp_path / "secret.png"
    external.write_bytes(b"must-not-be-read")
    path = tmp_path / "read-only.ipynb"
    _write_notebook(
        path,
        [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "# Safe\n![external](secret.png)\n![remote](https://example.invalid/x.png)",
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')",
                "outputs": [],
            },
        ],
    )
    original_external_hash = hashlib.sha256(external.read_bytes()).hexdigest()

    parsed = parse_local_notebook(path)

    assert not marker.exists()
    assert hashlib.sha256(external.read_bytes()).hexdigest() == original_external_hash
    assert parsed.sources[0].locator.keys().isdisjoint({"assets", "external_resources"})


@pytest.mark.parametrize(
    ("path", "expected_second_hash"),
    [
        (SAMPLE_V1, "ce741007281a939680e7a21020fc80c34abfcb10d6d76d13e7984bc9d5743aa5"),
        (SAMPLE_V2, "7ec65dab6628a30590421e4c1a150c52a6ec22933d67efce47eda279097fbd68"),
    ],
)
def test_a03_samples_remain_unchanged_and_second_image_is_addressable(
    path: Path, expected_second_hash: str
) -> None:
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    parsed = parse_local_notebook(path)

    assert [source.source_id for source in parsed.sources] == [
        "cell:0",
        "cell:1",
        "cell:1:output:0",
        "cell:1:output:1",
        "cell:1:output:2",
    ]
    output = parsed.sources[-1]
    assert output.locator["cell_index"] == 1
    assert output.locator["cell_id"] == "signal-plots"
    assert output.locator["output_index"] == 2
    assert output.locator["representations"][0]["sha256"] == expected_second_hash
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_rejects_invalid_json_shape(tmp_path: Path) -> None:
    wrong_suffix = tmp_path / "notebook.json"
    wrong_suffix.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="只支持"):
        parse_local_notebook(wrong_suffix)

    malformed = tmp_path / "malformed.ipynb"
    malformed.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="有效"):
        parse_local_notebook(malformed)

    missing_cells = tmp_path / "missing-cells.ipynb"
    missing_cells.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="cells"):
        parse_local_notebook(missing_cells)
