"""Compare public experiment qualifiers through the formal-style ASGI entry point."""

import argparse
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent / "M2-public-diagnosis" / "diagnose.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("subject", "prep", "env-file", "out"):
        parser.add_argument("--" + argument, type=Path, required=True)
    parser.add_argument("--fixtures-from", type=Path)
    args = parser.parse_args()
    subject, out = args.subject.resolve(), args.out.resolve()
    git = ["git", "-C", str(subject)]
    if subprocess.check_output([*git, "status", "--porcelain"]).strip():
        raise RuntimeError("Expected a clean product checkout")
    prep = runpy.run_path(str(args.prep.resolve()))
    diag = runpy.run_path(str(DIAG))
    sys.path.insert(0, str(subject))
    out.mkdir()
    files = out / "fixtures"
    if args.fixtures_from:
        source = args.fixtures_from.resolve()
        pinned = json.loads((source / "result.json").read_text())["fixtures"]
        for name, digest in pinned.items():
            if prep["sha"]((source / "fixtures" / name).read_bytes()) != digest:
                raise RuntimeError(f"Fixture changed: {name}")
        shutil.copytree(source / "fixtures", files)
    else:
        prep["fixtures"](files)
    settings = prep["settings_for"](args.env_file.resolve(), out / "runtime", web=False)
    app, service = prep["app_for"](settings)
    events = []
    prep["observe_providers"](settings, events, web=False, image_hashes={})
    cases = (
        ("held_out", prep["QUESTION"]),
        ("salt_fog", prep["MISSING"]),
        ("test_split", diag["DEFINITION"]),
        ("unfamiliar_condition", prep["QUESTION"].replace("留出集", "低温循环后的留出集")),
        ("different_metric", prep["QUESTION"] + "请给出 RMSE。"),
        ("different_object", prep["QUESTION"].replace("留出集", "Quartz latch 的留出集")),
    )
    try:
        with TestClient(app, base_url=diag["HOST"], headers={"Origin": diag["HOST"]}) as client:
            session = diag["ApiSession"](client, service, prep["COURSE"])
            assert session.empty
            for name in (prep["GUIDE"], prep["DECK"], prep["LAB"]):
                assert session.upload(files / name)["version"] == 1
            answers = {}
            for name, question in cases:
                response = session.ask(name, question)
                answers[name] = response | {"opened_text": session.evidence(response)}
    finally:
        prep["restore_providers"]()
    prep["dump"](
        out / "result.json",
        {
            "subject": subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip(),
            "fixtures": {
                str(p.relative_to(files)): prep["sha"](p.read_bytes())
                for p in files.rglob("*")
                if p.is_file()
            },
            "effective_settings": prep["effective"](settings),
            "empty": session.empty,
            "calls": events,
            "cases": answers,
        },
    )


if __name__ == "__main__":
    main()
