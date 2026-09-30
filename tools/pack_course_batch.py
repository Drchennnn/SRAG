"""Pack a small, verified PDF/PPTX batch for Colab preprocessing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="打包待 Colab 预处理的幻灯片")
    parser.add_argument("root", type=Path, help="与清单相同的本地课程根目录")
    parser.add_argument("--manifest", type=Path, default=Path("data/ingest/course_manifest.csv"))
    parser.add_argument("--decision", choices=("pilot", "candidate"), default="pilot")
    parser.add_argument("--course-id", help="可选，只打包一门课程")
    parser.add_argument("--limit", type=int, default=4, help="一次最多处理的文件数")
    parser.add_argument("--output", type=Path, default=Path("data/ingest/colab_batch.zip"))
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit 必须大于零")
    root = args.root.expanduser().resolve()
    with args.manifest.open(encoding="utf-8-sig", newline="") as source:
        rows = [
            row for row in csv.DictReader(source)
            if row["decision"] == args.decision
            and row["format"] in {"pdf", "pptx"}
            and (not args.course_id or row["course_id"] == args.course_id)
        ][: args.limit]
    if not rows:
        parser.error("清单中没有符合条件的 PDF/PPTX")
    files: list[tuple[dict[str, str], Path]] = []
    for row in rows:
        path = (root / row["relative_path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"文件不存在或路径越界：{row['relative_path']}")
        if _sha256(path) != row["sha256"]:
            raise ValueError(f"文件内容已改变，请重新生成清单：{row['relative_path']}")
        files.append((row, path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parent.parent
    with ZipFile(args.output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "batch.json",
            json.dumps(
                {
                    "files": [
                        {
                            "course_id": row["course_id"],
                            "relative_path": row["relative_path"],
                            "sha256": row["sha256"],
                            "pages_or_slides": row["pages_or_slides"],
                        }
                        for row, _ in files
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        for row, path in files:
            archive.write(path, f"inputs/{row['relative_path']}")
        for source in [
            project_root / "pyproject.toml",
            project_root / "README.md",
            *sorted((project_root / "app").rglob("*.py")),
        ]:
            archive.write(source, f"project/{source.relative_to(project_root).as_posix()}")
    print(f"已打包 {len(files)} 个文件：{args.output}")
    for row, _ in files:
        print(f"- {row['relative_path']}")


if __name__ == "__main__":
    main()
