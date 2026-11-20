"""Build versioned public inputs for the G16 live browser check."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

BASE = runpy.run_path(str(Path(__file__).with_name("G14-fixtures.py")))
COURSE = BASE["COURSE"]
FILENAME = BASE["FILENAME"]
QUESTIONS = {
    "first": BASE["QUESTIONS"][0],
    "follow": BASE["QUESTIONS"][1],
    "current": "本页夏季批次的项目签字截止和正式结业登记分别是哪天？",
    "compare": "和旧版相比呢？",
    "specific_compare": f"请比较 {FILENAME} 第 5 页旧版和当前版的夏季截止和登记时间。",
}
DATES = {
    3: ("May 31", "June 18"),
    5: ("August 23", "September 12"),
    7: ("December 8", "January 14"),
    8: ("July 15",),
}
CURRENT_DATES = ("August 21", "September 10")
AWARD_ROUNDS = tuple(page[0] for page in BASE["PAGES"] if page[0].endswith("award round"))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(out: Path) -> dict:
    v1 = out / "fixtures/v1" / FILENAME
    v2 = out / "fixtures/v2" / FILENAME
    BASE["make_pdf"](v1)
    old_bytes = v1.read_bytes()
    new_bytes = old_bytes
    for old_date, new_date in zip(DATES[5], CURRENT_DATES, strict=True):
        old, new = old_date.encode(), new_date.encode()
        assert len(old) == len(new) and new_bytes.count(old) >= 1
        new_bytes = new_bytes.replace(old, new)
    v2.parent.mkdir(parents=True, exist_ok=True)
    v2.write_bytes(new_bytes)
    manifest = {
        "filename": FILENAME,
        "course_id": COURSE,
        "fixture_hashes": {"v1": sha256(old_bytes), "v2": sha256(new_bytes)},
        "questions": QUESTIONS,
        "v1_evidence_pages": BASE["EXPECTED"],
        "v2_page_5": CURRENT_DATES,
        "award_rounds": AWARD_ROUNDS,
    }
    (out / "fixtures.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    build(parser.parse_args().out)
