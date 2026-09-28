"""真实 ipynb 文件的只读解析测试。"""

import base64
from pathlib import Path

import nbformat

from app.rag.parsers.notebook import NotebookParser
from app.schemas.document import BlockKind


def test_notebook_parser_reads_cells_outputs_and_assets(tmp_path: Path) -> None:
    notebook = nbformat.v4.new_notebook()
    local_image = tmp_path / "local-curve.png"
    local_image.write_bytes(b"local-png")
    markdown = nbformat.v4.new_markdown_cell(
        "# CS5489 Machine Learning\n"
        "# Lecture 3: Optimization\n\n"
        "## 梯度下降\n学习率控制更新步长。\n\n"
        "![学习曲线](local-curve.png)"
    )
    markdown["id"] = "intro"
    code = nbformat.v4.new_code_cell("print('loss=0.1')")
    code["id"] = "train"
    code["outputs"] = [
        nbformat.v4.new_output("stream", name="stdout", text="loss=0.1\n"),
        nbformat.v4.new_output(
            "display_data",
            data={
                "text/html": (
                    "<style>.dataframe { color: red; }</style>"
                    "<table><tr><th>epoch</th><th>loss</th></tr>"
                    "<tr><td>1</td><td>0.1</td></tr></table>"
                ),
                "image/png": base64.b64encode(b"fake-png").decode("ascii"),
            },
            metadata={},
        ),
        nbformat.v4.new_output(
            "error",
            ename="ValueError",
            evalue="bad value",
            traceback=["ValueError: bad value"],
        ),
    ]
    notebook.cells = [markdown, code]
    path = tmp_path / "course.ipynb"
    nbformat.write(notebook, path)

    parser = NotebookParser(tmp_path / "assets")
    parsed = parser.parse(path, document_id="doc-course")

    assert parsed.title == "Lecture 3: Optimization"
    assert parsed.source_path == path.as_posix()
    assert any(block.kind is BlockKind.MARKDOWN for block in parsed.blocks)
    assert any(block.kind is BlockKind.CODE for block in parsed.blocks)
    assert any(block.kind is BlockKind.TABLE for block in parsed.blocks)
    assert any("ValueError" in block.text for block in parsed.blocks)
    code_block = next(block for block in parsed.blocks if block.kind is BlockKind.CODE)
    output_blocks = [block for block in parsed.blocks if block.related_block_id]
    assert output_blocks
    assert all(block.related_block_id == code_block.block_id for block in output_blocks)
    table_block = next(block for block in parsed.blocks if block.kind is BlockKind.TABLE)
    assert "epoch" in table_block.text
    assert "color: red" not in table_block.text
    assert table_block.assets
    assert Path(table_block.assets[0].path).read_bytes() == b"fake-png"
    assert table_block.section_path == ["Lecture 3: Optimization", "梯度下降"]
    local_asset = next(
        asset
        for block in parsed.blocks
        for asset in block.assets
        if asset.caption == "学习曲线"
    )
    assert Path(local_asset.path) == local_image
    heading_block = next(block for block in parsed.blocks if "梯度下降" in block.text)
    assert heading_block.heading_level == 2


def test_notebook_parser_recognizes_setext_heading(tmp_path: Path) -> None:
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_markdown_cell("Running Python\n==============\nUse IPython.")]
    )
    path = tmp_path / "setext.ipynb"
    nbformat.write(notebook, path)

    parsed = NotebookParser(tmp_path / "assets").parse(path)

    assert parsed.title == "Running Python"
    assert parsed.blocks[0].section_path == ["Running Python"]
    assert parsed.blocks[0].heading_level == 1
