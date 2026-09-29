"""PDF 文档切块器，基于页面或语义段落切分。"""

from __future__ import annotations

from app.config import Settings
from app.rag.chunker import estimate_token_count
from app.schemas.document import NotebookChunk, ParsedDocument


class PDFChunker:
    """PDF 文档切块器。

    策略：基于 Docling 解析的标题段进行语义切分，
    类似 NotebookChunker 的逻辑。
    """

    def __init__(
        self,
        *,
        target_tokens: int = 280,
        max_tokens: int = 450,
        soft_min_tokens: int = 120,
    ) -> None:
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.soft_min_tokens = soft_min_tokens

    @classmethod
    def from_settings(cls, settings: Settings) -> PDFChunker:
        return cls(
            target_tokens=settings.notebook_chunk_target_tokens,
            max_tokens=settings.notebook_chunk_max_tokens,
            soft_min_tokens=settings.notebook_chunk_soft_min_tokens,
        )

    def chunk(self, document: ParsedDocument) -> list[NotebookChunk]:
        """将 PDF 文档切分为 chunks。

        基于标题段进行切分，类似 Notebook。
        """
        chunks: list[NotebookChunk] = []
        chunk_index = 0

        # 按标题段分组
        current_section: list[str] = []
        current_content: list[str] = []

        for block in document.blocks:
            # 更新当前章节路径
            if block.section_path and block.section_path != current_section:
                # 保存之前的 chunk
                if current_content:
                    chunks.append(
                        self._create_chunk(
                            chunk_index,
                            document,
                            current_section.copy(),
                            "\n\n".join(current_content),
                        )
                    )
                    chunk_index += 1
                    current_content = []

                current_section = block.section_path.copy()

            # 累积内容
            if block.text.strip():
                current_content.append(block.text.strip())

                # 检查是否需要切分（超过 max_tokens）
                combined_text = "\n\n".join(current_content)
                if estimate_token_count(combined_text) > self.max_tokens:
                    # 保存当前 chunk（不包含最后一段）
                    if len(current_content) > 1:
                        chunks.append(
                            self._create_chunk(
                                chunk_index,
                                document,
                                current_section.copy(),
                                "\n\n".join(current_content[:-1]),
                            )
                        )
                        chunk_index += 1
                        # 保留最后一段继续
                        current_content = current_content[-1:]
                    else:
                        # 单段过长，直接保存
                        chunks.append(
                            self._create_chunk(
                                chunk_index,
                                document,
                                current_section.copy(),
                                current_content[0],
                            )
                        )
                        chunk_index += 1
                        current_content = []

        # 保存最后一个 chunk
        if current_content:
            chunks.append(
                self._create_chunk(
                    chunk_index,
                    document,
                    current_section.copy(),
                    "\n\n".join(current_content),
                )
            )

        return chunks

    def _create_chunk(
        self,
        index: int,
        document: ParsedDocument,
        section_path: list[str],
        content: str,
    ) -> NotebookChunk:
        """创建一个 chunk。"""
        chunk_id = f"pdf_chunk_{index}"
        section_id = f"section_{hash(tuple(section_path)) & 0xFFFFFFFF}"

        # 添加 PDF 前缀
        embedding_text = f"""文件：{document.title}
类型：PDF

{"章节：" + " / ".join(section_path) if section_path else ""}

{content}
"""

        return NotebookChunk(
            chunk_id=chunk_id,
            section_id=section_id,
            document_id=document.document_id,
            kind="mixed",
            section_path=section_path,
            embedding_text=embedding_text.strip(),
            display_text=content,
            group_ids=[],
            block_ids=[],
            cell_ids=[],
            asset_ids=[],
            token_count=estimate_token_count(embedding_text),
            previous_chunk_id=None,
            next_chunk_id=None,
        )
