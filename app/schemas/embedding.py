"""文本向量化请求结果的数据结构。"""

from pydantic import BaseModel, Field


class SparseValue(BaseModel):
    """稀疏向量中的单个 Token 权重。"""

    index: int
    value: float
    token: str | None = None


class EmbeddingVector(BaseModel):
    """一个输入文本对应的 dense 与 sparse 向量。"""

    index: int
    dense: list[float] = Field(default_factory=list)
    sparse: list[SparseValue] = Field(default_factory=list)


class EmbeddingBatch(BaseModel):
    """一次或多次批量请求合并后的向量结果。"""

    model: str
    vectors: list[EmbeddingVector]
    total_tokens: int = 0
    request_ids: list[str] = Field(default_factory=list)
