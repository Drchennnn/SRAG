"""测试 DeepSeek 客户端基本功能。"""

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
