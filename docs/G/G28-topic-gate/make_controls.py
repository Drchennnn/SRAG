"""Derive independent topic controls from the existing public dense fixture."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dense_fixture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.dense_fixture.read_text())
    first, implicit = data["turns"][:2]
    lecture = implicit["scope"]
    lab = "刚才那次回放实验的平均释放延迟是多少毫秒？"
    bare = "只在选中的资料里，接着前一问说明其定义。" + lab
    multi = (
        "讲义怎么定义 Acceptance Grid 和 Quiet Window？"
        "配套 Latency replay 实验的 mean release lag 为多少毫秒？"
    )
    data["turns"] = [
        {**first, "id": "D01"},
        {**implicit, "id": "D02"},
        {"id": "D03", "new": False, "scope": lecture, "question": "Quiet Window 怎么定义？"},
        {**first, "id": "D04", "new": True},
        {**implicit, "id": "D05", "question": bare},
        {**first, "id": "D06", "new": True, "question": multi},
        {**implicit, "id": "D07"},
        {**implicit, "id": "D08", "new": True},
    ]
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
