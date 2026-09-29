"""PDF 文档解析器，使用 Docling 提取文本、标题、表格和图片。"""

from __future__ import annotations

from pathlib import Path

from docling.document_converter import DocumentConverter

from app.config import Settings
from app.schemas.document import ContentBlock, ParsedDocument


class PDFParser:
    """使用 Docling 解析 PDF 为结构化文档。"""

    def __init__(self) -> None:
        self.converter = DocumentConverter()

    @classmethod
    def from_settings(cls, settings: Settings) -> PDFParser:
        return cls()

    def parse(self, file_path: Path) -> ParsedDocument:
        """解析 PDF 文件为 ParsedDocument。

        Docling 输出 Markdown 格式，包含：
        - 标题层级（## H1, ### H2）
        - 段落文本
        - 表格（Markdown 表格）
        - 图片引用
        """
        if not file_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在：{file_path}")
        if not file_path.suffix.lower() == ".pdf":
            raise ValueError(f"不是 PDF 文件：{file_path}")

        # 使用 Docling 转换
        result = self.converter.convert(str(file_path))
        markdown_text = result.document.export_to_markdown()

        # 提取文档元信息
        document_title = file_path.stem  # Docling 的 DoclingDocument 没有 title 属性
        document_id = f"document_{self._compute_hash(file_path)}"

        # 解析 Markdown 为结构化 blocks
        blocks = self._parse_markdown_to_blocks(markdown_text, file_path, document_id)

        return ParsedDocument(
            document_id=document_id,
            source_path=str(file_path),
            title=document_title,
            blocks=blocks,
        )

    def _parse_markdown_to_blocks(
        self,
        markdown: str,
        file_path: Path,
        document_id: str,
    ) -> list[ContentBlock]:
        """将 Markdown 文本解析为 ContentBlock 列表。"""
        blocks: list[ContentBlock] = []
        current_section: list[str] = []
        current_content: list[str] = []
        block_index = 0

        for line in markdown.split("\n"):
            line = line.rstrip()

            # 检测标题
            if line.startswith("#"):
                # 保存之前的内容
                if current_content:
                    blocks.append(
                        self._create_block(
                            block_index,
                            current_section.copy(),
                            "\n".join(current_content),
                            file_path,
                            document_id,
                        )
                    )
                    block_index += 1
                    current_content = []

                # 更新章节路径
                level = len(line) - len(line.lstrip("#"))
                title = line.lstrip("#").strip()
                current_section = current_section[: level - 1] + [title]

            else:
                # 普通内容行
                if line.strip():
                    current_content.append(line)

        # 保存最后一段内容
        if current_content:
            blocks.append(
                self._create_block(
                    block_index,
                    current_section.copy(),
                    "\n".join(current_content),
                    file_path,
                    document_id,
                )
            )

        return blocks

    def _create_block(
        self,
        index: int,
        section_path: list[str],
        content: str,
        file_path: Path,
        document_id: str,
    ) -> ContentBlock:
        """创建一个 ContentBlock。"""
        return ContentBlock(
            block_id=f"block_{index}",
            document_id=document_id,
            cell_id=None,  # PDF 没有 cell 概念
            kind="text",  # PDF 内容统一为 text
            text=content,  # 使用 text 字段
            section_path=section_path,
            assets=[],
        )

    def _compute_hash(self, file_path: Path) -> str:
        """计算文件哈希（简化版，实际应用中应使用完整的哈希）。"""
        import hashlib

        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:24]
