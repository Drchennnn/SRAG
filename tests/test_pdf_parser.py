"""PDF 原生快路径的轻量烟雾测试。"""

from pathlib import Path

import pytest

from app.rag.parsers.pdf import PDFParser

pytest.importorskip("docling")
PIL = pytest.importorskip("PIL.Image")


def test_pdf_parser_keeps_page_boundary_and_screenshot(tmp_path: Path) -> None:
    pdf_path = tmp_path / "one-page.pdf"
    PIL.new("RGB", (320, 180), "white").save(pdf_path, "PDF")

    deck = PDFParser(tmp_path / "output").parse(pdf_path)

    assert len(deck.slides) == 1
    assert deck.slides[0].slide_no == 1
    assert deck.slides[0].page_image_path is not None
    assert Path(deck.slides[0].page_image_path).exists()
