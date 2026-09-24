"""DashScope 向量客户端的批处理与响应解析测试。"""

import json

import httpx
import pytest

from app.clients.dashscope import DashScopeClient
from app.config import Settings


@pytest.mark.asyncio
async def test_embedding_client_batches_and_preserves_input_order() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        texts = payload["input"]["texts"]
        embeddings = [
            {
                "text_index": index,
                "embedding": [float(index), 0.5, 1.0],
                "sparse_embedding": [{"index": 100 + index, "value": 0.8, "token": text[:1]}],
            }
            for index, text in enumerate(texts)
        ]
        return httpx.Response(
            200,
            json={
                "request_id": f"request-{len(requests)}",
                "output": {"embeddings": embeddings},
                "usage": {"total_tokens": len(texts) * 3},
            },
        )

    settings = Settings(
        dashscope_api_key="test-key",
        dashscope_embedding_batch_size=2,
        dashscope_embedding_dimension=1024,
        dashscope_max_retries=0,
    )
    client = DashScopeClient(settings, transport=httpx.MockTransport(handler))

    result = await client.embed_documents(["第一段", "第二段", "第三段"])

    assert len(requests) == 2
    assert [vector.index for vector in result.vectors] == [0, 1, 2]
    assert result.vectors[2].dense == [0.0, 0.5, 1.0]
    assert result.vectors[2].sparse[0].token == "第"
    assert result.total_tokens == 9
    assert requests[0]["parameters"] == {
        "text_type": "document",
        "dimension": 1024,
        "output_type": "dense&sparse",
    }


@pytest.mark.asyncio
async def test_query_embedding_passes_query_type_and_instruction() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "request_id": "query-request",
                "output": {
                    "embeddings": [{"text_index": 0, "embedding": [0.1], "sparse_embedding": []}]
                },
                "usage": {"total_tokens": 4},
            },
        )

    settings = Settings(dashscope_api_key="test-key", dashscope_max_retries=0)
    client = DashScopeClient(settings, transport=httpx.MockTransport(handler))

    vector = await client.embed_query("什么是学习率？", instruct="Retrieve course concepts")

    assert vector.index == 0
    assert captured["parameters"] == {
        "text_type": "query",
        "dimension": 1024,
        "output_type": "dense&sparse",
        "instruct": "Retrieve course concepts",
    }
