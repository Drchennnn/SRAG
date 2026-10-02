"""测试 DeepSeek 客户端基本功能。"""

from unittest.mock import AsyncMock

import pytest

from app.clients.deepseek import DeepSeekClient
from app.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(deepseek_api_key="test_key")


def test_require_api_key_missing():
    """未配置 API Key 时抛出异常。"""
    settings = Settings(deepseek_api_key="")
    client = DeepSeekClient(settings)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        client.require_api_key()


def test_require_api_key_present(settings: Settings):
    """配置了 API Key 时返回。"""
    client = DeepSeekClient(settings)
    assert client.require_api_key() == "test_key"


@pytest.mark.asyncio
async def test_understand_query_disabled(settings: Settings):
    """查询理解功能关闭时返回原始查询。"""
    client = DeepSeekClient(settings)
    result = await client.understand_query("test query", enable_rewrite=False)

    assert result["original_query"] == "test query"
    assert result["intent"] is None
    assert result["rewritten_queries"] == []


async def test_classify_context_need_parses_strict_boolean(settings: Settings, monkeypatch) -> None:
    client = DeepSeekClient(settings)
    completion = AsyncMock(
        return_value={
            "content": '{"needs_context": true, "reason": "缺少主题"}',
            "usage": {"total_tokens": 40},
        }
    )
    monkeypatch.setattr(client, "_chat_completion", completion)

    result = await client.classify_context_need("什么是 LIME？", "这个机制在哪段？")

    assert result["needs_context"] is True
    assert result["reason"] == "缺少主题"
    assert result["attempts"] == 1
    assert completion.await_args.kwargs["response_format"] == {"type": "json_object"}


async def test_classify_context_need_retries_empty_content(settings: Settings, monkeypatch) -> None:
    client = DeepSeekClient(settings)
    completion = AsyncMock(
        side_effect=[
            {"content": "", "finish_reason": "length", "usage": {"total_tokens": 1024}},
            {"content": '{"needs_context": false}', "usage": {"total_tokens": 30}},
        ]
    )
    monkeypatch.setattr(client, "_chat_completion", completion)

    result = await client.classify_context_need("什么是 LIME？", "SHAP 是什么？")

    assert result["needs_context"] is False
    assert result["attempts"] == 2
    assert result["usage"]["total_tokens"] == 1054
    assert completion.await_args.kwargs["max_tokens"] == 2048


async def test_classify_context_need_rejects_two_empty_responses(
    settings: Settings, monkeypatch
) -> None:
    client = DeepSeekClient(settings)
    completion = AsyncMock(
        return_value={"content": "", "finish_reason": "length", "usage": {"total_tokens": 1024}}
    )
    monkeypatch.setattr(client, "_chat_completion", completion)

    with pytest.raises(RuntimeError, match="连续返回空内容"):
        await client.classify_context_need("问题一", "这个在哪段？")
    assert completion.await_count == 2


async def test_classify_context_need_rejects_non_boolean(settings: Settings, monkeypatch) -> None:
    client = DeepSeekClient(settings)
    monkeypatch.setattr(
        client,
        "_chat_completion",
        AsyncMock(return_value={"content": '{"needs_context": "true"}'}),
    )

    with pytest.raises(RuntimeError, match="布尔 needs_context"):
        await client.classify_context_need("问题一", "这个在哪段？")


async def test_generate_answer_uses_chat_model_and_retries_empty(
    settings: Settings, monkeypatch
) -> None:
    client = DeepSeekClient(settings)
    completion = AsyncMock(
        side_effect=[
            {"content": "", "usage": {"total_tokens": 4096}},
            {"content": "  根据课件……[1]  ", "usage": {"total_tokens": 80}},
        ]
    )
    monkeypatch.setattr(client, "_chat_completion", completion)

    result = await client.generate_answer([{"role": "user", "content": "问题"}])

    assert result == {"content": "根据课件……[1]", "usage": {"total_tokens": 4176}}
    assert completion.await_args.kwargs["model"] == settings.deepseek_chat_model
    assert completion.await_args.kwargs["thinking"] == {"type": "disabled"}
    assert completion.await_args.kwargs["max_tokens"] == 8192


async def test_generate_answer_does_not_accept_truncated_nonempty_answer(settings, monkeypatch):
    client = DeepSeekClient(settings)
    completion = AsyncMock(
        side_effect=[
            {"content": "半截公式 G(x)=", "finish_reason": "length"},
            {"content": "完整定义。[1]", "finish_reason": "stop"},
        ]
    )
    monkeypatch.setattr(client, "_chat_completion", completion)
    result = await client.generate_answer([{"role": "user", "content": "公式"}])
    assert result["content"] == "完整定义。[1]"
    assert completion.await_count == 2


async def test_generate_answer_rejects_repeated_truncation(settings, monkeypatch):
    client = DeepSeekClient(settings)
    completion = AsyncMock(return_value={"content": "半截总结", "finish_reason": "length"})
    monkeypatch.setattr(client, "_chat_completion", completion)
    with pytest.raises(RuntimeError, match="被截断"):
        await client.generate_answer([{"role": "user", "content": "总结"}])
