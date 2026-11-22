"""Run a continuous cross-capability Chromium session spanning G28 and M2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from textwrap import wrap

DEFAULT_RUNNER = Path("/Users/Link/code/SRAG-g28-independent/docs/G/G28-browser-replay.py")
DEFAULT_OBSERVER_BASE = Path("/Users/Link/code/SRAG-g28/docs/G/G28-definition/observe.py")
DEFAULT_OBSERVER_WIRE = Path(
    "/Users/Link/code/SRAG-g28-independent/docs/G/G28-p04-diagnostic/observe_boundaries.py"
)


def write_pdf(path: Path, pages: list[tuple[str, list[str]]]) -> None:
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    children = []
    for title, paragraphs in pages:
        lines = [title, *[line for paragraph in paragraphs for line in wrap(paragraph, 90)]]
        commands = ["BT /F1 12 Tf 48 752 Td 16 TL"]
        for line in lines:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            commands.append(f"({escaped}) Tj T*")
        stream = "\n".join([*commands, "ET"])
        page_id = len(objects) + 1
        children.append(f"{page_id} 0 R")
        objects.extend(
            [
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>",
                f"<< /Length {len(stream.encode('ascii'))} >>\nstream\n{stream}\nendstream",
            ]
        )
    objects[1] = f"<< /Type /Pages /Count {len(children)} /Kids [{' '.join(children)}] >>"
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, value in enumerate(objects, 1):
        offsets.append(len(content))
        content.extend(f"{idx} 0 obj\n{value}\nendobj\n".encode("ascii"))
    xref = len(content)
    content.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        content.extend(f"{offset:010} 00000 n \n".encode())
    content.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    path.write_bytes(content)


def write_notebook(path: Path, programs: list[tuple[str, str]]) -> None:
    cells = []
    for index, (cell_id, program) in enumerate(programs, 1):
        out = StringIO()
        with redirect_stdout(out):
            exec(program, {})
        cells.append(
            {
                "cell_type": "code",
                "id": cell_id,
                "metadata": {},
                "execution_count": index,
                "source": program.splitlines(keepends=True),
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": out.getvalue().splitlines(keepends=True),
                    }
                ],
            }
        )
    path.write_text(
        json.dumps(
            {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": cells},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


def build_manifest(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_name = "Joint-Protocol-Guide.pdf"
    nb_name = "Joint-Latency-Log.ipynb"
    write_pdf(
        out_dir / pdf_name,
        [
            ("Overview", ["Joint protocol guide covering Quorum Gate and verification metrics."]),
            (
                "Quorum Gate",
                [
                    "Quorum Gate: a batch is released only when three independent verifiers sign "
                    "the same epoch digest within 5 ticks.",
                    "If a signature is delayed past 5 ticks or duplicated, "
                    "the tally resets to zero.",
                ],
            ),
            (
                "Experiment Notes",
                ["Mean verification latency measurements live in the companion notebook."],
            ),
        ],
    )
    write_notebook(
        out_dir / nb_name,
        [
            (
                "quorum-bench",
                'latencies = [14.0, 15.0, 14.2, 14.8]\n'
                'print("Quorum Gate benchmark: 4 verified batches.")\n'
                'print(f"Mean verification latency: {sum(latencies) / len(latencies):.1f} ms.")\n',
            )
        ],
    )
    manifest = {
        "schema": 1,
        "config": {
            "workspace_input_token_limit": 32768,
            "workspace_retrieval_strategy": "fts",
            "workspace_answer_enabled": True,
            "source_vision_enabled": True,
        },
        "files": {
            pdf_name: {
                "version": 1,
                "sha256": hashlib.sha256((out_dir / pdf_name).read_bytes()).hexdigest(),
            },
            nb_name: {
                "version": 1,
                "sha256": hashlib.sha256((out_dir / nb_name).read_bytes()).hexdigest(),
            },
        },
        "turns": [
            {
                "id": "J01_course_prime",
                "new": False,
                "scope": "course",
                "question": (
                    "讲义怎么定义 Quorum Gate？"
                    "配套实验测得的 mean verification latency 是多少毫秒？"
                ),
            },
            {
                "id": "J02_narrow_mixed_comma",
                "new": False,
                "scope": pdf_name,
                "question": "只看我选中的讲义，把刚才问过的部分重新梳理一遍，配套实验还能核对吗？",
            },
            {
                "id": "J03_narrow_guess_num",
                "new": False,
                "scope": pdf_name,
                "question": "只看我选中的讲义，刚才那个要求的测试集实验平均延迟测得是 14.5 ms 吗？",
            },
            {
                "id": "J04_switch_back_course",
                "new": False,
                "scope": "course",
                "question": (
                    "切回课程全部资料，"
                    "配套实验测得的 mean verification latency 是多少毫秒？"
                ),
            },
        ],
    }
    manifest_path = out_dir / "joint-session.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path("/Users/Link/code/SRAG/.env"))
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = build_manifest(args.fixture_dir)
    import runpy

    sys.argv = [
        str(DEFAULT_OBSERVER_WIRE),
        str(DEFAULT_OBSERVER_BASE),
        "--runner",
        str(DEFAULT_RUNNER),
        "--subject",
        str(args.subject),
        "--manifest",
        str(manifest_path),
        "--env-file",
        str(args.env_file),
        "--out",
        str(args.out),
    ]
    runpy.run_path(str(DEFAULT_OBSERVER_WIRE), run_name="__main__")


if __name__ == "__main__":
    main()
