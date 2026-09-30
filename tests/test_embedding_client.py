"""DashScope 向量客户端的批处理与响应解析测试。"""

import json
from pathlib import Path

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


@pytest.mark.asyncio
async def test_visual_embedding_uses_image_text_fusion(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith(
            "/services/embeddings/multimodal-embedding/multimodal-embedding"
        )
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "request_id": "visual-request",
                "output": {
                    "embeddings": [
                        {"index": 0, "embedding": [0.1, 0.2, 0.3], "type": "fusion"}
                    ]
                },
                "usage": {"input_tokens": 4, "image_tokens": 8, "total_tokens": 12},
            },
        )

    image_path = tmp_path / "plot.png"
    image_path.write_bytes(b"fake-png")
    settings = Settings(
        dashscope_api_key="test-key",
        dashscope_visual_embedding_dimension=3,
        dashscope_max_retries=0,
    )
    client = DashScopeClient(settings, transport=httpx.MockTransport(handler))

    result = await client.embed_visual_fusion(
        text="Lecture 3 > Decision Boundary",
        image_path=image_path,
        media_type="image/png",
    )

    contents = captured["input"]["contents"]
    assert contents[0] == {"text": "Lecture 3 > Decision Boundary"}
    assert contents[1]["image"].startswith("data:image/png;base64,")
    assert captured["parameters"] == {
        "enable_fusion": True,
        "dimension": 3,
        "output_type": "dense",
    }
    assert result.vectors[0].input_type == "fusion"
    assert result.vectors[0].dense == [0.1, 0.2, 0.3]
    assert result.total_tokens == 12


@pytest.mark.asyncio
async def test_visual_query_uses_same_model_without_image_fusion() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "request_id": "visual-query",
                "output": {
                    "embeddings": [{"index": 0, "embedding": [0.1, 0.2], "type": "text"}]
                },
                "usage": {"total_tokens": 6},
            },
        )

    client = DashScopeClient(
        Settings(
            dashscope_api_key="test-key",
            dashscope_visual_embedding_dimension=2,
            dashscope_max_retries=0,
        ),
        transport=httpx.MockTransport(handler),
    )
    result = await client.embed_visual_query("  医学影像流程图  ")

    assert captured["input"] == {"contents": [{"text": "医学影像流程图"}]}
    assert captured["parameters"] == {"dimension": 2, "output_type": "dense"}
    assert result.vectors[0].input_type == "text"
    assert result.vectors[0].dense == [0.1, 0.2]
