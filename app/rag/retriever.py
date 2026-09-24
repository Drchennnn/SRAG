"""检索流程的唯一受控入口。"""

from app.schemas.rag import SearchRequest, SearchResponse


class HybridRetriever:
    """执行范围过滤、按需查询转换、混合召回和重排。"""

    async def search(self, request: SearchRequest) -> SearchResponse:
        raise NotImplementedError("混合检索将在索引基线完成后实现")
