"""PPTX 文档解析器，使用 python-pptx 提取文本、图片和表格。"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation

from app.config import Settings
from app.schemas.document import ContentBlock, ParsedDocument


class PPTXParser:
    """使用 python-pptx 解析 PPTX 为结构化文档。"""

    def __init__(self) -> None:
        pass

    @classmethod
    def from_settings(cls, settings: Settings) -> PPTXParser:
        return cls()

    def parse(self, file_path: Path) -> ParsedDocument:
        """解析 PPTX 文件为 ParsedDocument。

        每张幻灯片作为一个独立的 ContentBlock。
        """
        if not file_path.exists():
            raise FileNotFoundError(f"PPTX 文件不存在：{file_path}")
        if not file_path.suffix.lower() in [".ppt", ".pptx"]:
            raise ValueError(f"不是 PPTX 文件：{file_path}")

        prs = Presentation(file_path)
        blocks: list[ContentBlock] = []

        # 文档元信息
        document_title = file_path.stem
        document_id = f"document_{self._compute_hash(file_path)}"

        for slide_index, slide in enumerate(prs.slides):
            # 提取幻灯片标题
            title = self._extract_slide_title(slide)

            # 提取幻灯片内容
            content_lines = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    # 跳过标题（已单独提取）
                    if shape.text.strip() != title:
                        content_lines.append(shape.text.strip())

            # 合并内容
            content = "\n\n".join(content_lines)

            # 创建 block
            section_path = [title] if title else [f"幻灯片 {slide_index + 1}"]
            blocks.append(
                ContentBlock(
                    block_id=f"slide_{slide_index + 1}",
                    document_id=document_id,
                    cell_id=None,
                    kind="text",  # 改为 text（slide 不是有效的 BlockKind）
                    text=content,  # 使用 text 字段
                    section_path=section_path,
                    assets=[],
                )
            )

        return ParsedDocument(
            document_id=document_id,
            source_path=str(file_path),
            title=document_title,
            file_type="pptx",
            blocks=blocks,
            assets=[],
        )

    def _extract_slide_title(self, slide) -> str:
        """提取幻灯片标题。"""
        if slide.shapes.title:
            return slide.shapes.title.text.strip()
        return ""

    def _compute_hash(self, file_path: Path) -> str:
        """计算文件哈希。"""
        import hashlib

        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:24]
