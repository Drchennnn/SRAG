"""保存文档版本、父块、资源和任务状态的 SQLite 存储边界。"""

from pathlib import Path

from app.config import Settings


class MetadataStore:
    """保存 SQLite 路径，模块导入阶段不打开连接。"""

    def __init__(self, settings: Settings) -> None:
        self.path: Path = settings.sqlite_path
