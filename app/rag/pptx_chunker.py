"""PPTX 文档切块器，一张幻灯片 = 一个 chunk。"""

from __future__ import annotations

from app.config import Settings
from app.rag.chunker import estimate_token_count
from app.schemas.document import NotebookChunk, ParsedDocument


class PPTXChunker:
    """PPTX 文档切块器。

    策略：一张幻灯片 = 一个 chunk。
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def from_settings(cls, settings: Settings) -> PPTXChunker:
        return cls()

    def chunk(self, document: ParsedDocument) -> list[NotebookChunk]:
        """将 PPTX 文档切分为 chunks。

        每张幻灯片作为一个独立的 chunk。
        """
        if document.file_type != "pptx":
            raise ValueError(f"PPTXChunker 只支持 PPTX 文档，当前类型：{document.file_type}")

        chunks: list[NotebookChunk] = []

        for block in document.blocks:
            # 每个 block 就是一张幻灯片
            chunk_id = block.block_id  # 如 "slide_1"
            section_path = block.section_path or ["幻灯片"]

            # 添加 PPTX 前缀
            embedding_text = f"""文件：{document.title}
类型：PPTX
幻灯片：{section_path[0] if section_path else chunk_id}

{block.display_text}
"""

            chunks.append(
                NotebookChunk(
                    chunk_id=chunk_id,
                    section_id=f"section_{chunk_id}",
                    kind="pptx_slide",
                    section_path=section_path,
                    embedding_text=embedding_text.strip(),
                    display_text=block.display_text,
                    group_ids=[],
                    block_ids=[block.block_id],
                    cell_ids=[],
                    asset_ids=[],
                    token_count=estimate_token_count(embedding_text),
                    previous_chunk_id=None,
                    next_chunk_id=None,
                )
            )

        return chunks
