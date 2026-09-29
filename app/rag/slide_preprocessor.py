"""幻灯片式 PDF/PPTX 的统一预处理入口。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.rag.parsers.pdf import PDFParser
from app.rag.parsers.pptx import PPTXParser
from app.rag.slide_assets import classify_slide_assets
from app.rag.slide_chunker import SlideChunker
from app.schemas.document import SlideChunkingResult, SlideDeck


@dataclass(frozen=True, slots=True)
class SlidePreprocessResult:
    """一次预处理产生的内存结果与导出路径。"""

    deck: SlideDeck
    chunking: SlideChunkingResult
    bundle_dir: Path
    deck_path: Path
    chunks_path: Path


class SlideDeckPreprocessor:
    """解析、资源标注、页级切块并导出可携带的 JSON 包。"""

    def __init__(
        self,
        *,
        output_root: Path,
        chunker: SlideChunker,
        enable_pdf_ocr_fallback: bool = False,
        pdf_ocr_min_text_tokens: int = 8,
        render_pptx_slides: bool = True,
    ) -> None:
        self.output_root = output_root
        self.chunker = chunker
        self.pdf_parser = PDFParser(
            output_root,
            enable_ocr_fallback=enable_pdf_ocr_fallback,
            ocr_min_text_tokens=pdf_ocr_min_text_tokens,
        )
        self.pptx_parser = PPTXParser(output_root, render_slides=render_pptx_slides)

    @classmethod
    def from_settings(cls, settings: Settings) -> SlideDeckPreprocessor:
        return cls(
            output_root=settings.data_dir / "preprocessed",
            chunker=SlideChunker.from_settings(settings),
            enable_pdf_ocr_fallback=settings.slide_pdf_enable_ocr_fallback,
            pdf_ocr_min_text_tokens=settings.slide_pdf_ocr_min_text_tokens,
            render_pptx_slides=settings.slide_pptx_render_slides,
        )

    def preprocess(
        self, file_path: Path, document_id: str | None = None
    ) -> SlidePreprocessResult:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            deck = self.pdf_parser.parse(file_path, document_id)
        elif suffix == ".pptx":
            deck = self.pptx_parser.parse(file_path, document_id)
        else:
            raise ValueError(f"幻灯片预处理不支持该格式：{file_path.suffix}")

        deck = deck.model_copy(update={"slides": classify_slide_assets(deck.slides)})
        chunking = self.chunker.chunk(deck)
        bundle_dir = self.output_root / deck.document_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        deck_path = bundle_dir / "deck.json"
        chunks_path = bundle_dir / "chunks.json"
        _write_json(deck_path, _portable_deck(deck, bundle_dir).model_dump(mode="json"))
        _write_json(chunks_path, chunking.model_dump(mode="json"))
        return SlidePreprocessResult(
            deck=deck,
            chunking=chunking,
            bundle_dir=bundle_dir,
            deck_path=deck_path,
            chunks_path=chunks_path,
        )


def _portable_deck(deck: SlideDeck, bundle_dir: Path) -> SlideDeck:
    slides = []
    for slide in deck.slides:
        assets = [
            asset.model_copy(update={"path": _relative_path(asset.path, bundle_dir)})
            for asset in slide.assets
        ]
        slides.append(
            slide.model_copy(
                update={
                    "assets": assets,
                    "page_image_path": (
                        _relative_path(slide.page_image_path, bundle_dir)
                        if slide.page_image_path
                        else None
                    ),
                }
            )
        )
    return deck.model_copy(update={"slides": slides})


def _relative_path(path: str, bundle_dir: Path) -> str:
    candidate = Path(path).resolve()
    try:
        return candidate.relative_to(bundle_dir.resolve()).as_posix()
    except ValueError:
        return str(candidate)


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)
