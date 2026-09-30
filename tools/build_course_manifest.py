"""Create a reviewable snapshot of the local course corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import subprocess
from pathlib import Path
from zipfile import ZipFile

SUPPORTED = {".ipynb", ".pdf", ".pptx", ".ppt"}
FIELDS = (
    "course_id",
    "relative_path",
    "format",
    "pages_or_slides",
    "size_bytes",
    "sha256",
    "decision",
    "reason",
    "academic_term",
    "release_at",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _page_count(path: Path) -> int | str:
    if path.suffix.lower() == ".pdf":
        result = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True)
        match = re.search(r"^Pages:\s+(\d+)", result.stdout, re.MULTILINE)
        return int(match.group(1)) if result.returncode == 0 and match else ""
    if path.suffix.lower() == ".pptx":
        with ZipFile(path) as archive:
            return sum(
                bool(re.fullmatch(r"ppt/slides/slide\d+\.xml", name))
                for name in archive.namelist()
            )
    return ""


def _classification(relative: Path) -> tuple[str, str]:
    parts = [part.casefold() for part in relative.parts]
    name = relative.name.casefold()
    if ".ipynb_checkpoints" in parts:
        return "skip", "Jupyter 自动检查点"
    if relative.suffix.lower() == ".ppt":
        return "convert", "旧版 PPT；需转换为 PPTX 后复核"
    if "previousdocument" in parts:
        return "skip", "旧资料目录；优先使用主目录版本"
    if "referencepapers" in parts or "paper" in parts:
        return "skip", "参考论文；不属于首批课程讲义"
    if any("assignment" in part or "assisgnment" in part for part in parts):
        return "skip", "作业资料；暂不进入讲义语料"
    if "group project" in parts:
        return "skip", "项目说明；暂不进入讲义语料"
    if any(word in name for word in ("answer", "solution", "completed")):
        return "review", "答案或已完成教程；先核对课程用途与发布日期"
    review_names = ("revision", "syllabus", "matrixcookbook", "lecture_notes", "课程笔记")
    if any(word in name for word in review_names):
        return "review", "复习或参考资料；先核对版本与用途"
    if relative.parts[0].casefold().startswith("msai orientation"):
        return "skip", "入学介绍；不属于课程讲义"
    if any(
        part.startswith(("lecture", "week", "tutorial")) for part in parts[1:]
    ) or name.startswith(("cs5494-week", "cs5494-intro")):
        cs5495_pilot = {"lecture2a.pdf", "lecture2b.pdf", "lecture3a.pdf", "lecture3b.pdf"}
        if (
            relative.parts[0].startswith("CS5495")
            and relative.suffix.lower() == ".pdf"
            and name in cs5495_pilot
        ):
            return "pilot", "跨课程首批验证；与同主题 Notebook 分别检查"
        return "candidate", "课程讲义或教程候选；学期与版本待确认"
    return "review", "无法仅凭路径判断；需要人工复核"


def build_manifest(root: Path) -> list[dict[str, str | int]]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    rows: list[dict[str, str | int]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        relative = path.relative_to(root)
        digest = _sha256(path)
        decision, reason = _classification(relative)
        rows.append(
            {
                "course_id": relative.parts[0].split(" ", 1)[0],
                "relative_path": relative.as_posix(),
                "format": path.suffix.lower().lstrip("."),
                "pages_or_slides": _page_count(path),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
                "decision": decision,
                "reason": reason,
                "academic_term": "",
                "release_at": "",
            }
        )
    preferred_by_hash: dict[str, dict[str, str | int]] = {}
    priority = {"pilot": 0, "candidate": 1, "review": 2, "convert": 3, "skip": 4}
    for row in rows:
        digest = str(row["sha256"])
        preferred = preferred_by_hash.get(digest)
        if preferred is None or (
            priority[str(row["decision"])] < priority[str(preferred["decision"])]
        ):
            preferred_by_hash[digest] = row
    for row in rows:
        preferred = preferred_by_hash[str(row["sha256"])]
        if row is not preferred:
            row["decision"] = "skip"
            row["reason"] = f"与 {preferred['relative_path']} 完全相同"
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="扫描课程目录，生成待复核入库清单")
    parser.add_argument("root", type=Path, help="本地课程目录")
    parser.add_argument("--output", type=Path, default=Path("data/ingest/course_manifest.csv"))
    args = parser.parse_args()
    rows = build_manifest(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"已生成 {args.output}：{len(rows)} 个文件")


if __name__ == "__main__":
    main()
