"""索引器的切块与向量对应关系测试。"""

from collections.abc import Sequence

import pytest

from app.rag.chunker import ParentChildChunker
from app.rag.indexer import DocumentIndexer
from app.schemas.document import BlockKind, ContentBlock, ParsedDocument
from app.schemas.embedding import EmbeddingBatch, EmbeddingVector, SparseValue


class FakeEmbeddingProvider:
    embedding_model = "fake-embedding"
    embedding_dimension = 3

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        return EmbeddingBatch(
            model=self.embedding_model,
            vectors=[
                EmbeddingVector(
                    index=index,
                    dense=[float(index), 0.5, 1.0],
                    sparse=[SparseValue(index=100 + index, value=0.8, token="测试")],
                )
                for index, _ in enumerate(texts)
            ],
            total_tokens=sum(len(text) for text in texts),
            request_ids=["fake-request"],
        )


@pytest.mark.asyncio
async def test_prepare_document_aligns_chunks_and_embeddings() -> None:
    document = ParsedDocument(
        document_id="doc-index",
        source_path="course/index.ipynb",
        title="索引测试",
        blocks=[
            ContentBlock(
                block_id="cell-1",
                document_id="doc-index",
                kind=BlockKind.MARKDOWN,
                text="\n\n".join(["向量化后的子块必须和来源块保持稳定对应。"] * 8),
                section_path=["索引"],
            )
        ],
    )
    chunker = ParentChildChunker(
        parent_min_tokens=20,
        parent_max_tokens=80,
        child_min_tokens=10,
        child_max_tokens=35,
        child_overlap_tokens=5,
    )
    indexer = DocumentIndexer(chunker, FakeEmbeddingProvider())

    prepared = await indexer.prepare_document(document)

    assert prepared.document_id == document.document_id
    assert prepared.parents
    assert prepared.children
    assert prepared.embedding_model == "fake-embedding"
    assert prepared.embedding_dimension == 3
    assert prepared.request_ids == ["fake-request"]
    assert all(len(item.dense) == 3 for item in prepared.children)
    assert all(item.chunk.block_ids == ["cell-1"] for item in prepared.children)
