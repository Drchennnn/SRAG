"""保存 dense 与 sparse 子块向量的 Qdrant 存储边界。"""

from app.config import Settings


class QdrantStore:
    """保存连接配置，模块导入阶段不连接 Qdrant。"""

    def __init__(self, settings: Settings) -> None:
        self.url = settings.qdrant_url
        self.api_key = settings.qdrant_api_key or None
        self.collection = settings.qdrant_collection
