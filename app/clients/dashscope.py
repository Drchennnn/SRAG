"""封装 Qwen 向量化、重排、文本生成、OCR 与视觉理解 API 的 DashScope 客户端。"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import httpx

from app.config import Settings
from app.schemas.embedding import (
    EmbeddingBatch,
    EmbeddingVector,
    MultimodalEmbeddingBatch,
    MultimodalEmbeddingVector,
    SparseValue,
)

TextType = Literal["document", "query"]
OutputType = Literal["dense", "sparse", "dense&sparse"]

_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
_EMBEDDING_PATH = "/services/embeddings/text-embedding/text-embedding"
_MULTIMODAL_EMBEDDING_PATH = (
    "/services/embeddings/multimodal-embedding/multimodal-embedding"
)
_RERANK_PATH = "/services/rerank/text-rerank/text-rerank"
_RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
_SUPPORTED_VISUAL_MEDIA_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/bmp",
    "image/tiff",
}


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
        self.visual_embedding_model = settings.dashscope_visual_embedding_model
        self.visual_embedding_dimension = settings.dashscope_visual_embedding_dimension
        self.visual_max_image_bytes = settings.dashscope_visual_max_image_bytes
        self.rerank_model = settings.dashscope_rerank_model
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

    async def embed_visual_fusion(
        self,
        *,
        text: str,
        image_path: Path,
        media_type: str,
    ) -> MultimodalEmbeddingBatch:
        """将一张图片与其课程上下文融合为同一个视觉向量。"""

        clean_text = text.strip()
        if not clean_text:
            raise ValueError("多模态向量化的文本上下文不能为空")
        if media_type not in _SUPPORTED_VISUAL_MEDIA_TYPES:
            raise ValueError(f"qwen3-vl-embedding 不支持图片类型：{media_type}")
        resolved_path = image_path.expanduser().resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"多模态向量化图片不存在：{resolved_path}")
        image_bytes = resolved_path.read_bytes()
        if len(image_bytes) > self.visual_max_image_bytes:
            raise ValueError(
                "多模态向量化图片超过大小限制："
                f"{len(image_bytes)} > {self.visual_max_image_bytes} bytes"
            )

        encoded = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": self.visual_embedding_model,
            "input": {
                "contents": [
                    {"text": clean_text},
                    {"image": f"data:{media_type};base64,{encoded}"},
                ]
            },
            "parameters": {
                "enable_fusion": True,
                "dimension": self.visual_embedding_dimension,
                "output_type": "dense",
            },
        }
        return await self._request_multimodal_embedding(payload)

    async def embed_visual_query(self, text: str) -> MultimodalEmbeddingBatch:
        """用视觉模型为文本问题生成同空间向量，供后续整页视觉召回。"""

        clean_text = text.strip()
        if not clean_text:
            raise ValueError("视觉检索问题不能为空")
        payload = {
            "model": self.visual_embedding_model,
            "input": {"contents": [{"text": clean_text}]},
            "parameters": {
                "dimension": self.visual_embedding_dimension,
                "output_type": "dense",
            },
        }
        return await self._request_multimodal_embedding(payload)

    async def _request_multimodal_embedding(
        self, payload: dict[str, object]
    ) -> MultimodalEmbeddingBatch:
        headers = {
            "Authorization": f"Bearer {self.require_api_key()}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            data = await self._post_with_retry(
                client,
                headers,
                payload,
                endpoint=_MULTIMODAL_EMBEDDING_PATH,
            )
        return self._parse_multimodal_embedding_response(data)

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
        *,
        endpoint: str = _EMBEDDING_PATH,
    ) -> dict[str, object]:
        url = f"{self.base_url}{endpoint}"
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

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int | None = None,
    ) -> list[tuple[int, float]]:
        """使用 Qwen rerank API 对候选文档重排序。

        Args:
            query: 用户查询
            documents: 候选文档列表
            top_n: 返回前 N 个结果，None 表示返回全部

        Returns:
            [(原始索引, 相关性得分)] 列表，按得分降序排列
        """
        if not query.strip():
            raise ValueError("rerank 查询不能为空")
        if not documents:
            return []
        if top_n is not None and top_n < 1:
            raise ValueError("top_n 必须大于等于 1")

        payload = {
            "model": self.rerank_model,
            "input": {
                "query": query.strip(),
                "documents": [doc.strip() for doc in documents],
            },
        }
        if top_n is not None:
            payload["parameters"] = {"top_n": min(top_n, len(documents))}

        headers = {
            "Authorization": f"Bearer {self.require_api_key()}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            data = await self._post_with_retry(
                client,
                headers,
                payload,
                endpoint=_RERANK_PATH,
            )

        return self._parse_rerank_response(data)

    def _parse_multimodal_embedding_response(
        self,
        data: dict[str, object],
    ) -> MultimodalEmbeddingBatch:
        output = data.get("output")
        if not isinstance(output, dict):
            raise RuntimeError("DashScope 多模态响应缺少 output")
        raw_embeddings = output.get("embeddings")
        if not isinstance(raw_embeddings, list) or len(raw_embeddings) != 1:
            raise RuntimeError("DashScope 多模态融合响应必须包含一个向量")
        raw = raw_embeddings[0]
        if not isinstance(raw, dict) or not isinstance(raw.get("embedding"), list):
            raise RuntimeError("DashScope 返回了无效的多模态向量")
        dense = [float(value) for value in raw["embedding"]]
        if len(dense) != self.visual_embedding_dimension:
            raise RuntimeError(
                "多模态向量维度与配置不一致："
                f"期望 {self.visual_embedding_dimension}，实际 {len(dense)}"
            )
        usage = data.get("usage")
        total_tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0
        request_id = data.get("request_id")
        return MultimodalEmbeddingBatch(
            model=self.visual_embedding_model,
            vectors=[
                MultimodalEmbeddingVector(
                    index=int(raw.get("index", 0)),
                    dense=dense,
                    input_type=str(raw.get("type", "fusion")),
                )
            ],
            total_tokens=int(total_tokens),
            request_ids=[request_id] if isinstance(request_id, str) else [],
        )

    @staticmethod
    def _parse_rerank_response(data: dict[str, object]) -> list[tuple[int, float]]:
        """解析 rerank API 响应。"""
        output = data.get("output")
        if not isinstance(output, dict):
            raise RuntimeError("DashScope rerank 响应缺少 output")
        results = output.get("results")
        if not isinstance(results, list):
            raise RuntimeError("DashScope rerank 响应缺少 results")

        ranked: list[tuple[int, float]] = []
        for item in results:
            if not isinstance(item, dict):
                raise RuntimeError("DashScope rerank 返回了无效的结果项")
            index = item.get("index")
            score = item.get("relevance_score")
            if not isinstance(index, int) or not isinstance(score, (int, float)):
                raise RuntimeError("DashScope rerank 结果缺少有效的 index 或 relevance_score")
            ranked.append((index, float(score)))

        return ranked
