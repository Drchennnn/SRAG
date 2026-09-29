"""将 PPTX 文本、表格、图表和图片解析为统一 SlideDeck。"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from app.config import Settings
from app.schemas.document import (
    NormalizedBoundingBox,
    Slide,
    SlideAsset,
    SlideDeck,
    SlideElement,
    SlideElementKind,
    SlideSourceFormat,
)


class PPTXParser:
    """使用 python-pptx 保留页边界、位置和内嵌资源。"""

    def __init__(
        self,
        output_root: Path = Path("data/preprocessed"),
        *,
        render_slides: bool = True,
    ) -> None:
        self.output_root = output_root
        self.render_slides = render_slides

    @classmethod
    def from_settings(cls, settings: Settings) -> PPTXParser:
        return cls(
            settings.data_dir / "preprocessed",
            render_slides=settings.slide_pptx_render_slides,
        )

    def parse(self, file_path: Path, document_id: str | None = None) -> SlideDeck:
        try:
            from pptx import Presentation
        except ImportError as exc:  # pragma: no cover - 取决于可选依赖
            raise RuntimeError("解析 PPTX 需要安装 documents 可选依赖") from exc

        file_path = file_path.expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"PPTX 文件不存在：{file_path}")
        if file_path.suffix.lower() != ".pptx":
            raise ValueError(f"只支持 .pptx，旧版 .ppt 请先转换：{file_path}")

        document_id = document_id or f"document_{_file_hash(file_path)[:24]}"
        bundle_dir = self.output_root / document_id
        asset_dir = bundle_dir / "assets"
        slide_image_dir = bundle_dir / "slides"
        asset_dir.mkdir(parents=True, exist_ok=True)
        slide_image_dir.mkdir(parents=True, exist_ok=True)

        presentation = Presentation(file_path)
        if self.render_slides:
            _render_slide_images(file_path, slide_image_dir)
        slide_width = float(presentation.slide_width)
        slide_height = float(presentation.slide_height)
        slides = [
            self._parse_slide(
                slide,
                slide_no=index,
                document_id=document_id,
                slide_width=slide_width,
                slide_height=slide_height,
                asset_dir=asset_dir,
                slide_image_dir=slide_image_dir,
            )
            for index, slide in enumerate(presentation.slides, start=1)
        ]
        return SlideDeck(
            document_id=document_id,
            source_path=str(file_path),
            source_format=SlideSourceFormat.PPTX,
            title=file_path.stem,
            slides=_link_slides(slides),
        )

    def _parse_slide(
        self,
        slide: Any,
        *,
        slide_no: int,
        document_id: str,
        slide_width: float,
        slide_height: float,
        asset_dir: Path,
        slide_image_dir: Path,
    ) -> Slide:
        slide_id = f"{document_id}-slide-{slide_no:04d}"
        title = _extract_slide_title(slide, slide_height)
        elements: list[SlideElement] = []
        assets: list[SlideAsset] = []

        for shape_index, shape in enumerate(_iter_shapes(slide.shapes), start=1):
            element_id = f"{slide_id}-element-{shape_index:04d}"
            bbox = _shape_bbox(shape, slide_width, slide_height)
            kind = _shape_kind(shape, title)
            asset_id: str | None = None
            if kind is SlideElementKind.IMAGE:
                asset = _save_picture(shape, slide_id, shape_index, bbox, asset_dir)
                if asset:
                    assets.append(asset)
                    asset_id = asset.asset_id
            text = _shape_text(shape, kind)
            if text or asset_id or kind in {SlideElementKind.CHART, SlideElementKind.TABLE}:
                elements.append(
                    SlideElement(
                        element_id=element_id,
                        kind=kind,
                        text=text,
                        bbox=bbox,
                        asset_id=asset_id,
                    )
                )

        rendered = slide_image_dir / f"slide-{slide_no:04d}.png"
        return Slide(
            slide_id=slide_id,
            document_id=document_id,
            slide_no=slide_no,
            title=title or None,
            elements=elements,
            assets=assets,
            page_image_path=str(rendered) if rendered.exists() else None,
        )


def _iter_shapes(shapes: Any) -> Iterable[Any]:
    for shape in shapes:
        yield shape
        if int(getattr(shape, "shape_type", 0)) == 6:  # MSO_SHAPE_TYPE.GROUP
            yield from _iter_shapes(shape.shapes)


def _extract_slide_title(slide: Any, slide_height: float) -> str:
    title_shape = slide.shapes.title
    if title_shape and title_shape.text.strip():
        return title_shape.text.strip()
    candidates = [
        shape
        for shape in _iter_shapes(slide.shapes)
        if (text := getattr(shape, "text", "").strip())
        and len(text) <= 120
        and float(shape.top) / slide_height <= 0.18
    ]
    return min(candidates, key=lambda shape: shape.top).text.strip() if candidates else ""


def _shape_kind(shape: Any, slide_title: str) -> SlideElementKind:
    shape_type = int(getattr(shape, "shape_type", 0))
    text = getattr(shape, "text", "").strip()
    if shape_type == 13:  # MSO_SHAPE_TYPE.PICTURE
        return SlideElementKind.IMAGE
    if getattr(shape, "has_table", False):
        return SlideElementKind.TABLE
    if getattr(shape, "has_chart", False):
        return SlideElementKind.CHART
    element_xml = getattr(getattr(shape, "element", None), "xml", "")
    if "oMath" in element_xml:
        return SlideElementKind.FORMULA
    if slide_title and text == slide_title:
        return SlideElementKind.TITLE
    return SlideElementKind.TEXT if text else SlideElementKind.OTHER


def _shape_text(shape: Any, kind: SlideElementKind) -> str:
    if kind is SlideElementKind.TABLE:
        rows = [[cell.text.strip() for cell in row.cells] for row in shape.table.rows]
        return _table_to_markdown(rows)
    if kind is SlideElementKind.CHART:
        parts: list[str] = []
        chart = shape.chart
        if chart.has_title and chart.chart_title.has_text_frame:
            parts.append(chart.chart_title.text_frame.text.strip())
        parts.extend(
            series.name.strip()
            for series in chart.series
            if isinstance(series.name, str) and series.name.strip()
        )
        return "\n".join(dict.fromkeys(parts))
    return getattr(shape, "text", "").strip()


def _table_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    lines = [
        "| " + " | ".join(normalized[0]) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in normalized[1:])
    return "\n".join(lines)


def _save_picture(
    shape: Any,
    slide_id: str,
    shape_index: int,
    bbox: NormalizedBoundingBox | None,
    asset_dir: Path,
) -> SlideAsset | None:
    image = getattr(shape, "image", None)
    if image is None:
        return None
    blob = image.blob
    digest = hashlib.sha256(blob).hexdigest()
    path = asset_dir / (f"{slide_id}-asset-{shape_index:04d}-{digest[:12]}.{image.ext or 'bin'}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return SlideAsset(
        asset_id=f"{slide_id}-asset-{shape_index:04d}",
        media_type=image.content_type or "application/octet-stream",
        path=str(path),
        sha256=digest,
        bbox=bbox,
    )


def _shape_bbox(
    shape: Any, slide_width: float, slide_height: float
) -> NormalizedBoundingBox | None:
    if not slide_width or not slide_height:
        return None
    left = max(0.0, min(1.0, float(shape.left) / slide_width))
    top = max(0.0, min(1.0, float(shape.top) / slide_height))
    right = max(left, min(1.0, float(shape.left + shape.width) / slide_width))
    bottom = max(top, min(1.0, float(shape.top + shape.height) / slide_height))
    return NormalizedBoundingBox(left=left, top=top, right=right, bottom=bottom)


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


def _render_slide_images(file_path: Path, output_dir: Path) -> bool:
    """当 LibreOffice 可用时，将 PPTX 转 PDF 并渲染整页图片。"""

    executable = shutil.which("libreoffice") or shutil.which("soffice")
    if executable is None:
        return False
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        conversion = subprocess.run(
            [
                executable,
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                temporary,
                str(file_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        pdf_path = Path(temporary) / f"{file_path.stem}.pdf"
        if conversion.returncode != 0 or not pdf_path.exists():
            return False
        document = pdfium.PdfDocument(pdf_path)
        try:
            for index, page in enumerate(document, start=1):
                image = page.render(scale=2).to_pil()
                image.save(output_dir / f"slide-{index:04d}.png", format="PNG")
        finally:
            document.close()
    return True


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
