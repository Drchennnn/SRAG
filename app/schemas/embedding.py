"""文本与多模态向量化请求结果的数据结构。"""

from pydantic import BaseModel, Field


class SparseValue(BaseModel):
    """稀疏向量中的单个 Token 权重。"""

    index: int
    value: float
    token: str | None = None


class EmbeddingVector(BaseModel):
    """一个文本输入对应的 dense 与 sparse 向量。"""

    index: int
    dense: list[float] = Field(default_factory=list)
    sparse: list[SparseValue] = Field(default_factory=list)


class EmbeddingBatch(BaseModel):
    """一次或多次文本向量请求合并后的结果。"""

    model: str
    vectors: list[EmbeddingVector]
    total_tokens: int = 0
    request_ids: list[str] = Field(default_factory=list)


class MultimodalEmbeddingVector(BaseModel):
    """一个多模态输入对应的 dense 向量。"""

    index: int
    dense: list[float]
    input_type: str


class MultimodalEmbeddingBatch(BaseModel):
    """一次多模态向量请求的结果。"""

    model: str
    vectors: list[MultimodalEmbeddingVector]
    total_tokens: int = 0
    request_ids: list[str] = Field(default_factory=list)
