"""Build dual-version Notebook fixtures with real PNG plots."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

import nbformat
from PIL import Image, ImageDraw


def _render_chart(title: str, subtitle: str, value_text: str, accent: tuple[int, int, int]) -> str:
    img = Image.new("RGB", (640, 320), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(20, 20), (620, 300)], outline=(40, 40, 40), width=2)
    draw.text((40, 38), title, fill=(15, 15, 15))
    draw.text((40, 72), subtitle, fill=(50, 50, 50))
    draw.rectangle([(40, 120), (420, 210)], fill=accent, outline=(20, 20, 20), width=2)
    draw.text((56, 155), value_text, fill=(255, 255, 255))
    draw.text((40, 248), "X-axis: batch_size (items) | Y-axis: metric", fill=(30, 30, 30))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def write_bench_v1(path: Path) -> None:
    img_a = _render_chart(
        "Curve A: Throughput", "Unit: ops/s (v1)", "Peak throughput: 120 ops/s", (30, 90, 180)
    )
    img_b = _render_chart(
        "Curve B: Queue Latency", "Unit: ms (v1)", "P95 queue latency: 18 ms", (160, 70, 50)
    )
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Bench Drift v1\nDefinition: Bench Gate passes when throughput >= 100 ops/s.",
            id="overview-cell",
        ),
        nbformat.v4.new_code_cell(
            "plot_bench_curves(version=1)",
            id="curve-cell",
            execution_count=1,
            outputs=[
                nbformat.v4.new_output("display_data", data={"image/png": img_a}),
                nbformat.v4.new_output("display_data", data={"image/png": img_b}),
                nbformat.v4.new_output(
                    "stream",
                    name="stdout",
                    text="v1 summary: Curve A peak=120 ops/s; Curve B p95=18 ms.\n",
                ),
            ],
        ),
    ]
    nbformat.write(nbformat.v4.new_notebook(cells=cells), path)


def write_bench_v2_same_order(path: Path) -> None:
    img_a = _render_chart(
        "Curve A: Throughput", "Unit: ops/s (v2)", "Peak throughput: 240 ops/s", (25, 130, 80)
    )
    img_b = _render_chart(
        "Curve B: Queue Latency", "Unit: ms (v2)", "P95 queue latency: 9 ms", (140, 50, 140)
    )
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Bench Drift v2\nDefinition: Bench Gate passes when throughput >= 100 ops/s.",
            id="overview-cell",
        ),
        nbformat.v4.new_code_cell(
            "plot_bench_curves(version=2)",
            id="curve-cell",
            execution_count=1,
            outputs=[
                nbformat.v4.new_output("display_data", data={"image/png": img_a}),
                nbformat.v4.new_output("display_data", data={"image/png": img_b}),
                nbformat.v4.new_output(
                    "stream",
                    name="stdout",
                    text="v2 summary: Curve A peak=240 ops/s; Curve B p95=9 ms.\n",
                ),
            ],
        ),
    ]
    nbformat.write(nbformat.v4.new_notebook(cells=cells), path)


def write_bench_v2_reordered(path: Path) -> None:
    # Genuine structural output reorder: v1 Curve B moves to output:0, Curve A moves to output:1.
    img_a = _render_chart(
        "Curve A: Throughput", "Unit: ops/s (v1)", "Peak throughput: 120 ops/s", (30, 90, 180)
    )
    img_b = _render_chart(
        "Curve B: Queue Latency", "Unit: ms (v1)", "P95 queue latency: 18 ms", (160, 70, 50)
    )
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Bench Drift v2 Reordered\n"
            "Definition: Bench Gate passes when throughput >= 100 ops/s.",
            id="overview-cell",
        ),
        nbformat.v4.new_code_cell(
            "plot_bench_curves(version=2, swapped=True)",
            id="curve-cell",
            execution_count=1,
            outputs=[
                nbformat.v4.new_output("display_data", data={"image/png": img_b}),
                nbformat.v4.new_output("display_data", data={"image/png": img_a}),
                nbformat.v4.new_output(
                    "stream",
                    name="stdout",
                    text="v2 reordered: Curve B p95=18 ms; Curve A peak=120 ops/s.\n",
                ),
            ],
        ),
    ]
    nbformat.write(nbformat.v4.new_notebook(cells=cells), path)


def write_bench_v2_deleted(path: Path) -> None:
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Bench Drift v2 Deleted\nDefinition: Bench Gate passes when throughput >= 100 ops/s.",
            id="overview-cell",
        ),
        nbformat.v4.new_code_cell(
            "plot_bench_curves(version=2, removed=True)",
            id="curve-cell",
            execution_count=1,
            outputs=[
                nbformat.v4.new_output(
                    "stream", name="stdout", text="v2 deleted: plots removed.\n"
                ),
            ],
        ),
    ]
    nbformat.write(nbformat.v4.new_notebook(cells=cells), path)


def generate_all(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "v1": out_dir / "Bench-Drift-v1.ipynb",
        "v2_same": out_dir / "Bench-Drift-v2-same.ipynb",
        "v2_reordered": out_dir / "Bench-Drift-v2-reordered.ipynb",
        "v2_deleted": out_dir / "Bench-Drift-v2-deleted.ipynb",
    }
    write_bench_v1(paths["v1"])
    write_bench_v2_same_order(paths["v2_same"])
    write_bench_v2_reordered(paths["v2_reordered"])
    write_bench_v2_deleted(paths["v2_deleted"])
    return paths


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "fixtures"
    res = generate_all(out)
    print(json.dumps({k: str(v) for k, v in res.items()}, indent=2))
