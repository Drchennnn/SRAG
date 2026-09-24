"""切块和向量化完成后、写入存储前的数据结构。"""

from pydantic import BaseModel, Field

from app.schemas.document import ChildChunk, ParentChunk
from app.schemas.embedding import SparseValue


class EmbeddedChild(BaseModel):
    """一个子块及其 dense 与 sparse 向量。"""

    chunk: ChildChunk
    dense: list[float]
    sparse: list[SparseValue] = Field(default_factory=list)


class PreparedDocumentIndex(BaseModel):
    """已经可以写入 SQLite 和 Qdrant 的单文档索引数据包。"""

    document_id: str
    parents: list[ParentChunk]
    children: list[EmbeddedChild]
    embedding_model: str
    embedding_dimension: int
    total_tokens: int = 0
    request_ids: list[str] = Field(default_factory=list)
