"""增量索引编排。

当前已实现统一文档的切块与向量化准备；文件登记、解析、持久化和版本发布
将在 Notebook 解析器与存储层完成后接入。
"""

from collections.abc import Sequence
from typing import Protocol

from app.clients.dashscope import DashScopeClient
from app.config import Settings
from app.rag.chunker import ParentChildChunker
from app.schemas.document import ParsedDocument
from app.schemas.embedding import EmbeddingBatch
from app.schemas.indexing import EmbeddedChild, PreparedDocumentIndex
from app.schemas.rag import IndexRequest, IndexStatus


class EmbeddingProvider(Protocol):
    """Indexer 使用的最小向量化接口。"""

    embedding_model: str
    embedding_dimension: int

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch: ...


class DocumentIndexer:
    """在不依赖全局实例的前提下编排一次增量索引任务。"""

    def __init__(
        self,
        chunker: ParentChildChunker,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.chunker = chunker
        self.embedding_provider = embedding_provider

    @classmethod
    def from_settings(cls, settings: Settings) -> "DocumentIndexer":
        """使用项目配置创建索引器及其依赖。"""

        return cls(
            chunker=ParentChildChunker.from_settings(settings),
            embedding_provider=DashScopeClient(settings),
        )

    async def prepare_document(self, document: ParsedDocument) -> PreparedDocumentIndex:
        """完成切块和向量化，生成待持久化的单文档索引数据包。"""

        parents, children = self.chunker.chunk(document)
        if not children:
            return PreparedDocumentIndex(
                document_id=document.document_id,
                parents=parents,
                children=[],
                embedding_model=self.embedding_provider.embedding_model,
                embedding_dimension=self.embedding_provider.embedding_dimension,
            )

        batch = await self.embedding_provider.embed_documents([child.text for child in children])
        vectors_by_index = {vector.index: vector for vector in batch.vectors}
        expected_indices = set(range(len(children)))
        if set(vectors_by_index) != expected_indices:
            raise RuntimeError("Embedding 结果无法与全部子块一一对应")

        embedded_children: list[EmbeddedChild] = []
        for index, child in enumerate(children):
            vector = vectors_by_index[index]
            if len(vector.dense) != self.embedding_provider.embedding_dimension:
                raise RuntimeError(
                    "Embedding 向量维度与配置不一致："
                    f"期望 {self.embedding_provider.embedding_dimension}，"
                    f"实际 {len(vector.dense)}"
                )
            embedded_children.append(
                EmbeddedChild(
                    chunk=child,
                    dense=vector.dense,
                    sparse=vector.sparse,
                )
            )

        return PreparedDocumentIndex(
            document_id=document.document_id,
            parents=parents,
            children=embedded_children,
            embedding_model=batch.model,
            embedding_dimension=self.embedding_provider.embedding_dimension,
            total_tokens=batch.total_tokens,
            request_ids=batch.request_ids,
        )

    async def index(self, request: IndexRequest) -> IndexStatus:
        """执行完整增量索引；等待解析器与存储层接入。"""

        raise NotImplementedError("完整增量索引将在 Notebook 解析器与存储层完成后接入")
