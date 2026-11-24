"""Generate public punctuation-variant control from G28-followup-entry fixtures."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    src_dir = Path(__file__).resolve().parents[1] / "G28-followup-entry/fixture"
    pdf_name = "Beacon-Sync-Manual.pdf"
    nb_name = "Beacon-Sync-Log.ipynb"
    for name in (pdf_name, nb_name):
        shutil.copy2(src_dir / name, args.out_dir / name)

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
            {"id": "P_prime_1", "new": False, "scope": "course", "question": prime_q},
            {
                "id": "F02_comma",
                "new": False,
                "scope": pdf_name,
                "question": "只看我选中的讲义，把刚才问过的部分重新梳理一遍，配套实验还能核对吗？",
            },
            {"id": "P_prime_2", "new": True, "scope": "course", "question": prime_q},
            {
                "id": "F02_conjunction",
                "new": False,
                "scope": pdf_name,
                "question": "只看我选中的讲义，把刚才问过的部分重新梳理一遍，顺便看下配套实验还能核对吗？",
            },
            {"id": "P_prime_3", "new": True, "scope": "course", "question": prime_q},
            {
                "id": "M01_pure_measure_comma",
                "new": False,
                "scope": pdf_name,
                "question": "只看我选中的讲义，刚才那次实验的延迟结果还能核对吗？",
            },
            {"id": "P_prime_4", "new": True, "scope": "course", "question": prime_q},
            {
                "id": "F02_semicolon",
                "new": False,
                "scope": pdf_name,
                "question": "只看我选中的讲义，把刚才问过的部分重新梳理一遍；配套实验还能核对吗？",
            },
            {"id": "P_prime_5", "new": True, "scope": "course", "question": prime_q},
            {
                "id": "F05_measurement_only",
                "new": False,
                "scope": pdf_name,
                "question": "接着前一问说明其测量结果。配套实验现在还能核对吗？",
            },
            {"id": "P_prime_6", "new": True, "scope": "course", "question": prime_q},
            {
                "id": "E01_explicit_switch",
                "new": False,
                "scope": pdf_name,
                "question": "Cooldown Guard 怎么定义？",
            },
            {"id": "A01_ambig_prime", "new": True, "scope": "course", "question": ambig_q},
            {
                "id": "A02_ambig_comma",
                "new": False,
                "scope": pdf_name,
                "question": "只看我选中的讲义，把刚才问过的部分重新梳理一遍，配套实验还能核对吗？",
            },
        ],
    }
    (args.out_dir / "control.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
