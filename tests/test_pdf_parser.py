"""PDF 原生快路径的轻量烟雾测试。"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.rag.parsers.pdf import PDFParser
from app.schemas.document import Slide, SlideAsset, SlideElement, SlideElementKind

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


def test_sparse_illustrated_page_keeps_native_content_without_ocr(tmp_path: Path) -> None:
    parser = PDFParser(tmp_path, enable_ocr_fallback=True)
    parser._ocr_converter = SimpleNamespace(
        convert=lambda *_args, **_kwargs: pytest.fail("unexpected OCR")
    )
    slide = Slide(
        slide_id="doc-slide-0001",
        document_id="doc",
        slide_no=1,
        title="Face Recognition",
        elements=[
            SlideElement(
                element_id="title",
                kind=SlideElementKind.TITLE,
                text="Face Recognition",
            )
        ],
        assets=[
            SlideAsset(asset_id="photo", media_type="image/png", path="photo.png", sha256="hash")
        ],
        page_image_path="slide.png",
    )

    assert parser._supplement_sparse_pages_with_ocr([slide], file_path=tmp_path / "a.pdf") == [
        slide
    ]


def test_ocr_supplements_blank_page_without_replacing_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from docling_core.types import doc as docling_doc

    class FakeTextItem:
        def __init__(self, text: str) -> None:
            self.text = text
            self.prov = []

    monkeypatch.setattr(docling_doc, "TextItem", FakeTextItem)
    document = SimpleNamespace(
        pages={1: SimpleNamespace(size=SimpleNamespace())},
        iterate_items=lambda **_kwargs: iter(
            [(FakeTextItem("Stereo vision estimates depth"), None)]
        ),
    )
    parser = PDFParser(tmp_path, enable_ocr_fallback=True)
    parser._ocr_converter = SimpleNamespace(
        convert=lambda *_args, **_kwargs: SimpleNamespace(document=document)
    )
    slide = Slide(
        slide_id="doc-slide-0001",
        document_id="doc",
        slide_no=1,
        assets=[
            SlideAsset(asset_id="scan", media_type="image/png", path="scan.png", sha256="hash")
        ],
        page_image_path="slide.png",
    )

    updated = parser._supplement_sparse_pages_with_ocr([slide], file_path=tmp_path / "a.pdf")[0]

    assert updated.page_image_path == slide.page_image_path
    assert updated.assets == slide.assets
    assert [element.text for element in updated.elements] == ["Stereo vision estimates depth"]
