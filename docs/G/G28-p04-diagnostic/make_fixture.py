"""Build an independent synthetic lecture and lab for the public diagnosis."""

import argparse
import hashlib
import json
from pathlib import Path
from textwrap import wrap


def write_pdf(path, pages):
    # A deterministic, text-only PDF with one manual and no external assets.
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


def chapters():
    prior = [
        (
            "Operator vocabulary",
            "Qualified epoch: an epoch with both sensors healthy and a valid tick.",
        ),
        (
            "Dispatch threshold",
            "A dispatch threshold selects a queue, based on all qualifying reads.",
        ),
        (
            "Quiet Window",
            "Quiet Window: a mute interval of fourteen ticks after an alert. "
            "It is neither an acceptance rule nor a requirement before admitting a sample.",
        ),
        (
            "Receipt counter",
            "Receipt counter counts accepted bundles, not the readings in a bundle.",
        ),
        ("Fallback range", "Fallback range excludes missing sensors in diagnostic preview only."),
        (
            "Replay timebase",
            "Latency replay reports mean release lag after dequeue, never the fleet average.",
        ),
        ("Pending lane", "The pending lane stores samples while the alarm gate is muted."),
        ("Sensor review", "Use raw sensor readings when deciding whether an epoch qualifies."),
        (
            "Acceptance Grid",
            "Acceptance Grid: in the same epoch, both sensors must each be "
            "strictly above 23 units. Equality on either sensor fails. "
            "Never average the two readings.",
        ),
        (
            "Operator interval",
            "An interval ends at the timestamp of the first disqualified reading.",
        ),
        (
            "Archive bundle",
            "An archive bundle is a group of accepted samples with a shared device label.",
        ),
        (
            "Range review",
            "Archive range is an interval for locating samples, not an acceptance criterion.",
        ),
    ]
    sections = [
        ("Maximum scan work", "Worst case is linear only when every batch entry is visited."),
        (
            "Cached counts",
            "Cached summary reads are constant time while the archive index is valid.",
        ),
        (
            "Replay coverage",
            "The guide defines procedure, and the separate lab records its measurements.",
        ),
        (
            "Accepted samples",
            "A single validated receipt does not imply an accepted archive bundle.",
        ),
        (
            "Jitter notes",
            "Jitter from timing replay is a diagnostic property, not a sensor threshold.",
        ),
        (
            "Draft mode",
            "Data from draft mode must never substitute for results of a completed replay.",
        ),
        (
            "Supervisor queue",
            "A supervisor queue may hold pending samples after a delayed dispatch.",
        ),
        ("Deferred lane", "Deferred samples are rechecked at the next available healthy epoch."),
        ("Partial logs", "A partial replay log records no measurements for epochs that never ran."),
        (
            "Operator review",
            "The acceptance rule and the post-alert mute interval describe different events.",
        ),
        ("Maintenance window", "During maintenance, monitoring continues but dispatch is held."),
        ("Release policy", "Replay data and quality rules are separate evidence for audit."),
    ]
    return [
        (
            name,
            [
                f"Field Operations Guide | section {idx:02d}",
                body,
                "Operator review uses the defined condition for its own step. "
                "Do not transfer a rule between acceptance, dispatch and logging. "
                "Each section is an independent reference for a separate review.",
                "Review order: identify the subject of the rule, then keep its event and boundary.",
            ],
        )
        for idx, (name, body) in enumerate(prior + sections, 1)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    lecture = "Field-Operations-Guide.pdf"
    notebook = "Release-Lag-Record.ipynb"
    write_pdf(out / lecture, chapters())
    code = (
        "lags = [17, 21, 19]\n"
        "print('Latency replay for Acceptance Grid: three accepted bundles after dequeue.')\n"
        "print(f'Mean release lag: {sum(lags)//len(lags)} ms.')\n"
    )
    (out / notebook).write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [
                    {
                        "cell_type": "code",
                        "id": "release-lag",
                        "metadata": {},
                        "execution_count": 1,
                        "source": code.splitlines(keepends=True),
                        "outputs": [
                            {
                                "output_type": "stream",
                                "name": "stdout",
                                "text": [
                                    "Latency replay for Acceptance Grid: three accepted "
                                    "bundles after dequeue.\n",
                                    "Mean release lag: 19 ms.\n",
                                ],
                            }
                        ],
                    }
                ],
            },
            indent=2,
        )
        + "\n"
    )
    opened = (
        "讲义怎么定义 Acceptance Grid？配套 Latency replay 实验的 mean release lag 为多少毫秒？"
    )
    implicit = "按刚才的问题，那条规则怎么定义？刚才那次回放实验的平均释放延迟是多少毫秒？"
    explicit = implicit.replace("那条规则", "Acceptance Grid")
    turns = [
        ("I01", False, "course", opened),
        ("I02", False, lecture, implicit),
        ("E01", True, "course", opened),
        ("E02", False, lecture, explicit),
        ("C01", True, "course", opened),
        ("C02", False, "course", implicit),
    ]
    config = dict(
        workspace_input_token_limit=32768,
        workspace_retrieval_strategy="fts",
        workspace_answer_enabled=True,
        source_vision_enabled=True,
    )
    (out / "fixture.json").write_text(
        json.dumps(
            {
                "config": config,
                "files": {
                    name: {"sha256": hashlib.sha256((out / name).read_bytes()).hexdigest()}
                    for name in (lecture, notebook)
                },
                "turns": [
                    dict(id=i, new=new, scope=scope, question=text) for i, new, scope, text in turns
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
