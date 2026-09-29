"""将幻灯片式 PDF 按页解析为统一 SlideDeck。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from app.config import Settings
from app.schemas.document import (
    NormalizedBoundingBox,
    Slide,
    SlideAsset,
    SlideAssetRole,
    SlideDeck,
    SlideElement,
    SlideElementKind,
    SlideSourceFormat,
)


class PDFParser:
    """优先读取 PDF 原生文本，可对空文本页单独启用 OCR。"""

    def __init__(
        self,
        output_root: Path = Path("data/preprocessed"),
        *,
        enable_ocr_fallback: bool = False,
        ocr_min_text_tokens: int = 8,
    ) -> None:
        self.output_root = output_root
        self.enable_ocr_fallback = enable_ocr_fallback
        self.ocr_min_text_tokens = ocr_min_text_tokens
        self._native_converter: Any | None = None
        self._ocr_converter: Any | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> PDFParser:
        return cls(
            settings.data_dir / "preprocessed",
            enable_ocr_fallback=settings.slide_pdf_enable_ocr_fallback,
            ocr_min_text_tokens=settings.slide_pdf_ocr_min_text_tokens,
        )

    def parse(self, file_path: Path, document_id: str | None = None) -> SlideDeck:
        file_path = file_path.expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在：{file_path}")
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"不是 PDF 文件：{file_path}")

        document_id = document_id or f"document_{_file_hash(file_path)[:24]}"
        bundle_dir = self.output_root / document_id
        asset_dir = bundle_dir / "assets"
        slide_dir = bundle_dir / "slides"
        asset_dir.mkdir(parents=True, exist_ok=True)
        slide_dir.mkdir(parents=True, exist_ok=True)

        result = self._get_native_converter().convert(file_path)
        document = result.document
        slides = [
            self._parse_page(
                document,
                page_no,
                document_id=document_id,
                asset_dir=asset_dir,
                slide_dir=slide_dir,
            )
            for page_no in sorted(document.pages)
        ]
        if self.enable_ocr_fallback:
            slides = self._supplement_sparse_pages_with_ocr(
                slides,
                file_path=file_path,
            )
        return SlideDeck(
            document_id=document_id,
            source_path=str(file_path),
            source_format=SlideSourceFormat.PDF,
            title=file_path.stem,
            slides=_link_slides(slides),
        )

    def _parse_page(
        self,
        document: Any,
        page_no: int,
        *,
        document_id: str,
        asset_dir: Path,
        slide_dir: Path,
        logical_slide_no: int | None = None,
    ) -> Slide:
        from docling_core.types.doc import PictureItem, TableItem, TextItem

        slide_no = logical_slide_no or page_no
        slide_id = f"{document_id}-slide-{slide_no:04d}"
        page = document.pages[page_no]
        page_image_path = _save_page_image(page, slide_dir / f"slide-{slide_no:04d}.png")
        elements: list[SlideElement] = []
        assets: list[SlideAsset] = []
        title: str | None = None

        for index, (item, _) in enumerate(
            document.iterate_items(page_no=page_no, traverse_pictures=True), start=1
        ):
            bbox = _docling_bbox(item, page.size)
            kind = _docling_kind(item)
            text = ""
            asset_id: str | None = None
            if isinstance(item, TableItem):
                text = item.export_to_markdown(document)
            elif isinstance(item, TextItem):
                text = item.text.strip()
            if kind is SlideElementKind.TITLE and text and title is None:
                title = text
            if isinstance(item, PictureItem):
                asset = _save_docling_picture(
                    item, document, slide_id, index, bbox, asset_dir, kind
                )
                if asset:
                    assets.append(asset)
                    asset_id = asset.asset_id
            if text or asset_id or kind in {SlideElementKind.TABLE, SlideElementKind.CHART}:
                elements.append(
                    SlideElement(
                        element_id=f"{slide_id}-element-{index:04d}",
                        kind=kind,
                        text=text,
                        bbox=bbox,
                        asset_id=asset_id,
                    )
                )
        if title is None:
            title, elements = _promote_title(elements)
        return Slide(
            slide_id=slide_id,
            document_id=document_id,
            slide_no=slide_no,
            title=title,
            elements=elements,
            assets=assets,
            page_image_path=page_image_path,
        )

    def _supplement_sparse_pages_with_ocr(
        self,
        slides: list[Slide],
        *,
        file_path: Path,
    ) -> list[Slide]:
        from docling_core.types.doc import TextItem

        from app.rag.chunker import estimate_token_count

        supplemented = list(slides)
        for index, slide in enumerate(slides):
            text = "\n".join(element.text for element in slide.elements)
            if estimate_token_count(text) >= self.ocr_min_text_tokens:
                continue
            # 有原生标题和图像的幻灯片本来就可能文字很少，画面不应被 OCR 噪声替代。
            if slide.title or (slide.assets and text.strip()):
                continue
            result = self._get_ocr_converter().convert(
                file_path, page_range=(slide.slide_no, slide.slide_no)
            )
            ocr_document = result.document
            if not ocr_document.pages:
                continue
            page_no = next(iter(ocr_document.pages))
            page = ocr_document.pages[page_no]
            native_texts = {
                element.text.strip().casefold() for element in slide.elements if element.text
            }
            additions: list[SlideElement] = []
            for item, _ in ocr_document.iterate_items(page_no=page_no):
                if not isinstance(item, TextItem):
                    continue
                value = item.text.strip()
                normalized = value.casefold()
                if len(value) < 4 or normalized in native_texts:
                    continue
                native_texts.add(normalized)
                additions.append(
                    SlideElement(
                        element_id=f"{slide.slide_id}-ocr-{len(additions) + 1:04d}",
                        kind=SlideElementKind.TEXT,
                        text=value,
                        bbox=_docling_bbox(item, page.size),
                    )
                )
            if estimate_token_count("\n".join(item.text for item in additions)) >= 4:
                supplemented[index] = slide.model_copy(
                    update={"elements": [*slide.elements, *additions]}
                )
        return supplemented

    def _get_native_converter(self) -> Any:
        if self._native_converter is None:
            try:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import NativePdfPipelineOptions
                from docling.document_converter import (
                    DocumentConverter,
                    NativePdfFormatOption,
                )
            except ImportError as exc:  # pragma: no cover - 取决于可选依赖
                raise RuntimeError("解析 PDF 需要安装 documents 可选依赖") from exc
            options = NativePdfPipelineOptions(
                generate_page_images=True,
                generate_picture_images=True,
                images_scale=2.0,
            )
            self._native_converter = DocumentConverter(
                format_options={InputFormat.PDF: NativePdfFormatOption(pipeline_options=options)}
            )
        return self._native_converter

    def _get_ocr_converter(self) -> Any:
        if self._ocr_converter is None:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            options = PdfPipelineOptions(
                do_ocr=True,
                generate_page_images=True,
                generate_picture_images=True,
                images_scale=2.0,
            )
            self._ocr_converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
        return self._ocr_converter


def _docling_kind(item: Any) -> SlideElementKind:
    label = str(getattr(item, "label", "")).lower()
    if "title" in label or "section_header" in label:
        return SlideElementKind.TITLE
    if "table" in label:
        return SlideElementKind.TABLE
    if "chart" in label:
        return SlideElementKind.CHART
    if "formula" in label:
        return SlideElementKind.FORMULA
    if "picture" in label:
        return SlideElementKind.IMAGE
    return SlideElementKind.TEXT


def _promote_title(
    elements: list[SlideElement],
) -> tuple[str | None, list[SlideElement]]:
    """在 Docling 未标注标题时，使用页面上方的第一个短文本。"""

    for index, element in enumerate(elements):
        if element.kind is not SlideElementKind.TEXT or not element.text.strip():
            continue
        if element.bbox and element.bbox.top > 0.35:
            continue
        if len(element.text) > 200:
            continue
        promoted = list(elements)
        promoted[index] = element.model_copy(update={"kind": SlideElementKind.TITLE})
        return element.text.strip(), promoted
    return None, elements


def _docling_bbox(item: Any, page_size: Any) -> NormalizedBoundingBox | None:
    if not getattr(item, "prov", None):
        return None
    bbox = item.prov[0].bbox.to_top_left_origin(page_size.height).normalized(page_size)
    return NormalizedBoundingBox(
        left=max(0.0, min(1.0, bbox.l)),
        top=max(0.0, min(1.0, bbox.t)),
        right=max(0.0, min(1.0, bbox.r)),
        bottom=max(0.0, min(1.0, bbox.b)),
    )


def _save_page_image(page: Any, path: Path) -> str | None:
    image_ref = getattr(page, "image", None)
    image = image_ref.pil_image if image_ref else None
    if image is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return str(path)


def _save_docling_picture(
    item: Any,
    document: Any,
    slide_id: str,
    index: int,
    bbox: NormalizedBoundingBox | None,
    asset_dir: Path,
    kind: SlideElementKind,
) -> SlideAsset | None:
    image = item.get_image(document)
    if image is None:
        return None
    path = asset_dir / f"{slide_id}-asset-{index:04d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    informative = kind in {SlideElementKind.CHART, SlideElementKind.FORMULA}
    captions = getattr(item, "captions", [])
    return SlideAsset(
        asset_id=f"{slide_id}-asset-{index:04d}",
        media_type="image/png",
        path=str(path),
        sha256=digest,
        bbox=bbox,
        caption=" ".join(str(caption) for caption in captions) or None,
        role=SlideAssetRole.INFORMATIVE if informative else SlideAssetRole.UNKNOWN,
        role_reasons=["图表或公式资源"] if informative else [],
    )


def _link_slides(slides: list[Slide]) -> list[Slide]:
    return [
        slide.model_copy(
            update={
                "previous_slide_id": slides[index - 1].slide_id if index else None,
                "next_slide_id": (slides[index + 1].slide_id if index + 1 < len(slides) else None),
            }
        )
        for index, slide in enumerate(slides)
    ]


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
