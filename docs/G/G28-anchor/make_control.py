"""Generate a new public multi-turn control for subject-omitted follow-ups."""

import argparse
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
from textwrap import wrap


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
        json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": cells}, ensure_ascii=False, indent=2)
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = "Relay-Protocol-Notes.pdf"
    nb_name = "Relay-Latency-Log.ipynb"

    pages = [
        ("Overview", ["Relay protocol lecture notes covering release gates, backoff timers, and batch logging."]),
        (
            "Quorum Gate",
            [
                "Quorum Gate: a batch is released only when three independent verifiers sign the same epoch digest within 5 ticks.",
                "If a signature is delayed past 5 ticks or duplicated by the same verifier, the tally resets to zero.",
                "Advisory pings do not count toward the three-verifier quorum.",
            ],
        ),
        ("Verifier Roles", ["Verifiers operate independently and never share transient signature buffers."]),
        ("Epoch Boundaries", ["Each epoch has a monotonic sequence number and a fixed 5-tick verification window."]),
        (
            "Lease Backoff",
            [
                "Lease Backoff: after a worker lease timeout, the worker waits 8 ticks before requesting renewal.",
                "Lease Backoff only throttles renewal attempts; it does not authorize batch release.",
            ],
        ),
        ("Retry Accounting", ["Failed renewals increment the backoff counter without altering queued batches."]),
        ("Queue Isolation", ["Quarantined batches remain isolated from active dispatch queues until cleared."]),
        (
            "Batch Pipeline",
            [
                "Batch Pipeline processing order: incoming batches first enter staging, then undergo digest check, and finally record an audit entry.",
                "Each stage logs its timestamp before handing off to the next stage.",
            ],
        ),
        ("Audit Trail", ["Audit entries append monotonically to the local verification log."]),
        ("Experiment Notes", ["Latency benchmarks are stored separately in the verification notebook, not in these lecture notes."]),
    ]
    write_pdf(args.out_dir / pdf_name, pages)
    write_notebook(
        args.out_dir / nb_name,
        [
            (
                "quorum-bench",
                'latencies = [14.0, 15.0, 14.2, 14.8]\n'
                'print("Quorum Gate benchmark: 4 verified batches.")\n'
                'print(f"Mean verification latency: {sum(latencies) / len(latencies):.1f} ms.")\n',
            )
        ],
    )

    prime_q = "讲义怎么定义 Quorum Gate？配套实验测得的 mean verification latency 是多少毫秒？"
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
                "sha256": hashlib.sha256((args.out_dir / pdf_name).read_bytes()).hexdigest(),
            },
            nb_name: {
                "version": 1,
                "sha256": hashlib.sha256((args.out_dir / nb_name).read_bytes()).hexdigest(),
            },
        },
        "turns": [
            {"id": "C01_prime_a", "new": False, "scope": "course", "question": prime_q},
            {
                "id": "C02_omitted_subject",
                "new": False,
                "scope": pdf_name,
                "question": "只看我刚选的这份讲义，上面到底是怎么界定刚才那个要求的？配套实验现在还能核对吗？",
            },
            {"id": "C01_prime_b", "new": True, "scope": "course", "question": prime_q},
            {
                "id": "C03_explicit_switch",
                "new": False,
                "scope": pdf_name,
                "question": "Lease Backoff 怎么定义？",
            },
            {"id": "C01_prime_c", "new": True, "scope": "course", "question": prime_q},
            {
                "id": "C04_unrelated",
                "new": False,
                "scope": pdf_name,
                "question": "这份讲义里，批次在流水线阶段按什么顺序处理？",
            },
        ],
    }
    (args.out_dir / "control.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
