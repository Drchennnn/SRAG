"""Notebook 文本与视觉表示准备及最小索引流程。"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from app.clients.dashscope import DashScopeClient
from app.config import Settings
from app.rag.chunker import NotebookChunker
from app.rag.parsers.notebook import NotebookParser
from app.schemas.document import ParsedDocument
from app.schemas.embedding import EmbeddingBatch, MultimodalEmbeddingBatch
from app.schemas.indexing import (
    EmbeddedTextRepresentation,
    EmbeddedVisualRepresentation,
    PreparedDocumentIndex,
    RepresentationType,
    TextRepresentation,
    VisualRepresentation,
)
from app.schemas.rag import IndexRequest, IndexStatus, JobState
from app.stores.qdrant import QdrantStore

_VISUAL_MEDIA_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/bmp",
    "image/tiff",
}


class TextEmbeddingProvider(Protocol):
    embedding_model: str
    embedding_dimension: int

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch: ...


class VisualEmbeddingProvider(Protocol):
    visual_embedding_model: str
    visual_embedding_dimension: int

    async def embed_visual_fusion(
        self,
        *,
        text: str,
        image_path: Path,
        media_type: str,
    ) -> MultimodalEmbeddingBatch: ...


class DocumentParser(Protocol):
    def parse(self, path: Path, document_id: str | None = None) -> ParsedDocument: ...


class TextRepresentationStore(Protocol):
    async def upsert_text_representations(
        self,
        prepared: PreparedDocumentIndex,
        *,
        course_id: str,
        version: int = 1,
        release_at: datetime | None = None,
    ) -> int: ...


class DocumentIndexer:
    """编排 Notebook 切块、文本表示和图片融合表示。"""

    def __init__(
        self,
        chunker: NotebookChunker,
        text_embedding_provider: TextEmbeddingProvider,
        visual_embedding_provider: VisualEmbeddingProvider | None = None,
        *,
        parser: DocumentParser | None = None,
        text_store: TextRepresentationStore | None = None,
    ) -> None:
        self.chunker = chunker
        self.text_embedding_provider = text_embedding_provider
        self.visual_embedding_provider = visual_embedding_provider
        self.parser = parser
        self.text_store = text_store

    @classmethod
    def from_settings(cls, settings: Settings) -> DocumentIndexer:
        provider = DashScopeClient(settings)
        return cls(
            chunker=NotebookChunker.from_settings(settings),
            text_embedding_provider=provider,
            visual_embedding_provider=provider,
            parser=NotebookParser.from_settings(settings),
            text_store=QdrantStore(settings),
        )

    async def prepare_document(
        self,
        document: ParsedDocument,
        *,
        include_visual: bool = True,
    ) -> PreparedDocumentIndex:
        """生成 document outline、内容 chunk 与图片融合向量。"""

        structure = self.chunker.chunk(document)
        text_inputs = [
            TextRepresentation(
                representation_id=_stable_id("representation", document.document_id, "outline"),
                representation_type=RepresentationType.DOCUMENT_OUTLINE,
                source_id=document.document_id,
                document_id=document.document_id,
                text=structure.outline_text,
            ),
            *[
                TextRepresentation(
                    representation_id=_stable_id(
                        "representation", chunk.chunk_id, RepresentationType.TEXT_CONTENT
                    ),
                    representation_type=RepresentationType.TEXT_CONTENT,
                    source_id=chunk.chunk_id,
                    document_id=document.document_id,
                    text=chunk.embedding_text,
                )
                for chunk in structure.chunks
            ],
        ]
        text_batch = await self.text_embedding_provider.embed_documents(
            [representation.text for representation in text_inputs]
        )
        vectors_by_index = {vector.index: vector for vector in text_batch.vectors}
        if set(vectors_by_index) != set(range(len(text_inputs))):
            raise RuntimeError("文本 Embedding 结果无法与全部表示一一对应")

        embedded_text: list[EmbeddedTextRepresentation] = []
        for index, representation in enumerate(text_inputs):
            vector = vectors_by_index[index]
            if len(vector.dense) != self.text_embedding_provider.embedding_dimension:
                raise RuntimeError(
                    "文本向量维度与配置不一致："
                    f"期望 {self.text_embedding_provider.embedding_dimension}，"
                    f"实际 {len(vector.dense)}"
                )
            embedded_text.append(
                EmbeddedTextRepresentation(
                    representation=representation,
                    dense=vector.dense,
                    sparse=vector.sparse,
                )
            )

        assets = {
            asset.asset_id: asset
            for block in document.blocks
            for asset in block.assets
        }
        visual_inputs: list[VisualRepresentation] = []
        skipped_assets: list[str] = []
        seen_assets: set[str] = set()
        for chunk in structure.chunks:
            for asset_id in chunk.asset_ids:
                if asset_id in seen_assets:
                    continue
                seen_assets.add(asset_id)
                asset = assets.get(asset_id)
                if asset is None or asset.media_type not in _VISUAL_MEDIA_TYPES:
                    skipped_assets.append(asset_id)
                    continue
                context = (
                    f"{chunk.embedding_text}\n\n"
                    f"图像：{asset.caption or Path(asset.path).name}"
                )
                visual_inputs.append(
                    VisualRepresentation(
                        representation_id=_stable_id(
                            "representation", asset_id, RepresentationType.VISUAL_FUSION
                        ),
                        source_id=asset_id,
                        document_id=document.document_id,
                        chunk_id=chunk.chunk_id,
                        image_path=asset.path,
                        media_type=asset.media_type,
                        context_text=context,
                    )
                )

        embedded_visual: list[EmbeddedVisualRepresentation] = []
        visual_total_tokens = 0
        request_ids = list(text_batch.request_ids)
        visual_provider = self.visual_embedding_provider if include_visual else None
        if visual_provider is None:
            skipped_assets.extend(item.source_id for item in visual_inputs)
        else:
            for representation in visual_inputs:
                batch = await visual_provider.embed_visual_fusion(
                    text=representation.context_text,
                    image_path=Path(representation.image_path),
                    media_type=representation.media_type,
                )
                if len(batch.vectors) != 1:
                    raise RuntimeError("图片融合 Embedding 必须返回一个向量")
                vector = batch.vectors[0]
                if len(vector.dense) != visual_provider.visual_embedding_dimension:
                    raise RuntimeError(
                        "视觉向量维度与配置不一致："
                        f"期望 {visual_provider.visual_embedding_dimension}，"
                        f"实际 {len(vector.dense)}"
                    )
                embedded_visual.append(
                    EmbeddedVisualRepresentation(
                        representation=representation,
                        dense=vector.dense,
                    )
                )
                visual_total_tokens += batch.total_tokens
                request_ids.extend(batch.request_ids)

        visual_model = (
            visual_provider.visual_embedding_model if visual_provider else ""
        )
        visual_dimension = (
            visual_provider.visual_embedding_dimension if visual_provider else 0
        )
        return PreparedDocumentIndex(
            document_id=document.document_id,
            source_path=document.source_path,
            title=document.title,
            groups=structure.groups,
            sections=structure.sections,
            chunks=structure.chunks,
            text_representations=embedded_text,
            visual_representations=embedded_visual,
            skipped_visual_asset_ids=list(dict.fromkeys(skipped_assets)),
            text_embedding_model=text_batch.model,
            text_embedding_dimension=self.text_embedding_provider.embedding_dimension,
            visual_embedding_model=visual_model,
            visual_embedding_dimension=visual_dimension,
            text_total_tokens=text_batch.total_tokens,
            visual_total_tokens=visual_total_tokens,
            request_ids=request_ids,
        )

    async def index(self, request: IndexRequest) -> IndexStatus:
        """解析选定 Notebook，并将文本表示写入 Qdrant。"""

        if self.parser is None or self.text_store is None:
            raise RuntimeError("DocumentIndexer.index() 尚未配置 parser 或 Qdrant store")

        errors: dict[str, str] = {}
        indexed_files = 0
        for path in request.paths:
            resolved = path.expanduser().resolve()
            try:
                document = self.parser.parse(resolved)
                prepared = await self.prepare_document(document, include_visual=False)
                await self.text_store.upsert_text_representations(
                    prepared,
                    course_id=request.course_id,
                    version=1,
                    release_at=request.release_at,
                )
            except Exception as exc:
                errors[resolved.as_posix()] = str(exc)
            else:
                indexed_files += 1

        return IndexStatus(
            job_id=f"index_{uuid4().hex}",
            state=JobState.FAILED if errors else JobState.COMPLETED,
            indexed_files=indexed_files,
            failed_files=len(errors),
            errors=errors,
        )


def _stable_id(kind: str, *parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"{kind}_{digest}"
