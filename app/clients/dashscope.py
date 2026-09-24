"""封装 Qwen 向量化、重排、文本生成、OCR 与视觉理解 API 的 DashScope 客户端。"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Literal

import httpx

from app.config import Settings
from app.schemas.embedding import EmbeddingBatch, EmbeddingVector, SparseValue

TextType = Literal["document", "query"]
OutputType = Literal["dense", "sparse", "dense&sparse"]

_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
_EMBEDDING_PATH = "/services/embeddings/text-embedding/text-embedding"
_RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class DashScopeClient:
    """保存服务配置，模块导入阶段不发起网络请求。"""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = settings.dashscope_api_key
        self.base_url = (settings.dashscope_base_url or _DEFAULT_BASE_URL).rstrip("/")
        self.embedding_model = settings.dashscope_embedding_model
        self.embedding_dimension = settings.dashscope_embedding_dimension
        self.embedding_batch_size = settings.dashscope_embedding_batch_size
        self.rerank_model = settings.dashscope_rerank_model
        self.query_model = settings.dashscope_query_model
        self.timeout_seconds = settings.dashscope_timeout_seconds
        self.max_retries = settings.dashscope_max_retries
        self.transport = transport

        if not 1 <= self.embedding_batch_size <= 20:
            raise ValueError("qwen3.7-text-embedding 的批大小必须在 1 到 20 之间")

    def require_api_key(self) -> str:
        """仅在实际调用 DashScope 时检查 API Key。"""

        if not self.api_key:
            raise RuntimeError("尚未在 .env 中配置 DASHSCOPE_API_KEY")
        return self.api_key

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        """为待入库文本生成 dense+sparse 向量。"""

        return await self.embed(texts, text_type="document")

    async def embed_query(
        self,
        text: str,
        *,
        instruct: str | None = None,
    ) -> EmbeddingVector:
        """为用户查询生成 dense+sparse 向量。"""

        batch = await self.embed([text], text_type="query", instruct=instruct)
        return batch.vectors[0]

    async def embed(
        self,
        texts: Sequence[str],
        *,
        text_type: TextType,
        output_type: OutputType = "dense&sparse",
        instruct: str | None = None,
    ) -> EmbeddingBatch:
        """分批调用 DashScope 原生文本向量接口并保持输入顺序。"""

        clean_texts = [text.strip() for text in texts]
        if not clean_texts:
            raise ValueError("向量化输入不能为空")
        if any(not text for text in clean_texts):
            raise ValueError("向量化输入不能包含空文本")
        if instruct and text_type != "query":
            raise ValueError("instruct 仅可用于 query 类型的向量化")

        headers = {
            "Authorization": f"Bearer {self.require_api_key()}",
            "Content-Type": "application/json",
        }
        vectors: list[EmbeddingVector] = []
        request_ids: list[str] = []
        total_tokens = 0

        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            for start in range(0, len(clean_texts), self.embedding_batch_size):
                batch = clean_texts[start : start + self.embedding_batch_size]
                parameters: dict[str, object] = {
                    "text_type": text_type,
                    "dimension": self.embedding_dimension,
                    "output_type": output_type,
                }
                if instruct:
                    parameters["instruct"] = instruct

                payload = {
                    "model": self.embedding_model,
                    "input": {"texts": batch},
                    "parameters": parameters,
                }
                data = await self._post_with_retry(client, headers, payload)
                parsed, used_tokens, request_id = self._parse_embedding_response(
                    data,
                    index_offset=start,
                )
                vectors.extend(parsed)
                total_tokens += used_tokens
                if request_id:
                    request_ids.append(request_id)

        vectors.sort(key=lambda vector: vector.index)
        if [vector.index for vector in vectors] != list(range(len(clean_texts))):
            raise RuntimeError("DashScope 返回的向量索引与输入顺序不一致")
        return EmbeddingBatch(
            model=self.embedding_model,
            vectors=vectors,
            total_tokens=total_tokens,
            request_ids=request_ids,
        )

    async def _post_with_retry(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        payload: dict[str, object],
    ) -> dict[str, object]:
        url = f"{self.base_url}{_EMBEDDING_PATH}"
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await client.post(url, headers=headers, json=payload)
            except httpx.TransportError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(2**attempt)
                continue

            if response.status_code in _RETRYABLE_STATUS_CODES:
                last_error = RuntimeError(f"DashScope 暂时不可用：HTTP {response.status_code}")
                if attempt < self.max_retries:
                    await asyncio.sleep(2**attempt)
                    continue
                break

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"DashScope 向量化请求失败：HTTP {response.status_code}"
                ) from exc

            try:
                data = response.json()
            except ValueError as exc:
                raise RuntimeError("DashScope 返回了无法解析的 JSON") from exc
            if not isinstance(data, dict):
                raise RuntimeError("DashScope 返回了无法识别的响应结构")
            if data.get("code"):
                raise RuntimeError(
                    f"DashScope 调用失败：{data.get('code')} - {data.get('message', '')}"
                )
            return data

        raise RuntimeError(f"DashScope 向量化请求失败：{last_error}") from last_error

    @staticmethod
    def _parse_embedding_response(
        data: dict[str, object],
        *,
        index_offset: int,
    ) -> tuple[list[EmbeddingVector], int, str | None]:
        output = data.get("output")
        if not isinstance(output, dict):
            raise RuntimeError("DashScope 响应缺少 output")
        raw_embeddings = output.get("embeddings")
        if not isinstance(raw_embeddings, list):
            raise RuntimeError("DashScope 响应缺少 embeddings")

        vectors: list[EmbeddingVector] = []
        for position, raw in enumerate(raw_embeddings):
            if not isinstance(raw, dict):
                raise RuntimeError("DashScope 返回了无效的向量条目")
            local_index = raw.get("text_index", position)
            if not isinstance(local_index, int):
                raise RuntimeError("DashScope 返回了无效的 text_index")
            dense = raw.get("embedding", [])
            sparse = raw.get("sparse_embedding", [])
            if not isinstance(dense, list) or not isinstance(sparse, list):
                raise RuntimeError("DashScope 返回了无效的 dense 或 sparse 向量")
            vectors.append(
                EmbeddingVector(
                    index=index_offset + local_index,
                    dense=[float(value) for value in dense],
                    sparse=[SparseValue.model_validate(item) for item in sparse],
                )
            )

        usage = data.get("usage")
        total_tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0
        request_id = data.get("request_id")
        return vectors, int(total_tokens), request_id if isinstance(request_id, str) else None
