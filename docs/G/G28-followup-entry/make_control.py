"""Generate a public control where followup=True, previous=None, and _definition_reference is None."""

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

    pdf_name = "Beacon-Sync-Manual.pdf"
    nb_name = "Beacon-Sync-Log.ipynb"

    pages = [
        ("Overview", ["Beacon sync manual covering barrier admission, backoff windows, and slot logs."]),
        (
            "Barrier Latch",
            [
                "Barrier Latch: admission succeeds only when two distinct peers confirm the same slot hash before tick 4.",
                "A confirmation from the same peer twice, or arriving at tick 4 or later, resets the latch.",
            ],
        ),
        ("Peer Isolation", ["Peers maintain separate hash buffers and never forward unverified slot hashes."]),
        (
            "Cooldown Guard",
            [
                "Cooldown Guard: after a reset, the sender pauses for 6 ticks before sending a fresh slot hash.",
                "Cooldown Guard only paces retries; it never opens barrier admission.",
            ],
        ),
        (
            "Slot Dispatch",
            [
                "Slot Dispatch order: incoming slots first enter staging, then pass hash validation, and finally append to the slot log.",
            ],
        ),
        ("Experiment Notes", ["Measured sync delays are stored only in the benchmark notebook, not in this manual."]),
    ]
    write_pdf(args.out_dir / pdf_name, pages)
    write_notebook(
        args.out_dir / nb_name,
        [
            (
                "barrier-bench",
                'delays = [11.0, 13.0, 12.0]\n'
                'print("Barrier Latch benchmark: 3 admitted slots.")\n'
                'print(f"Mean sync delay: {sum(delays) / len(delays):.1f} ms.")\n',
            )
        ],
    )

    prime_q = "讲义怎么定义 Barrier Latch？配套实验测得的 mean sync delay 是多少毫秒？"
    ambig_q = "讲义怎么定义 Barrier Latch 和 Cooldown Guard？配套实验测得的 mean sync delay 是多少毫秒？"
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
            {"id": "F01_prime", "new": False, "scope": "course", "question": prime_q},
            {
                "id": "F02_elided_followup",
                "new": False,
                "scope": pdf_name,
                "question": "只看我选中的讲义，把刚才问过的部分重新梳理一遍；配套实验还能核对吗？",
            },
            {"id": "F01_prime_b", "new": True, "scope": "course", "question": prime_q},
            {
                "id": "F03_explicit_switch",
                "new": False,
                "scope": pdf_name,
                "question": "Cooldown Guard 怎么定义？",
            },
            {"id": "F01_prime_c", "new": True, "scope": "course", "question": prime_q},
            {
                "id": "F04_unrelated",
                "new": False,
                "scope": pdf_name,
                "question": "这份讲义里，slot 在分发阶段按什么顺序处理？",
            },
            {"id": "F01_prime_d", "new": True, "scope": "course", "question": prime_q},
            {
                "id": "F05_measurement_only",
                "new": False,
                "scope": pdf_name,
                "question": "接着前一问说明其测量结果。配套实验现在还能核对吗？",
            },
            {"id": "F06_ambig_prime", "new": True, "scope": "course", "question": ambig_q},
            {
                "id": "F07_ambig_followup",
                "new": False,
                "scope": pdf_name,
                "question": "只看我选中的讲义，把刚才问过的部分重新梳理一遍。",
            },
        ],
    }
    (args.out_dir / "control.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
