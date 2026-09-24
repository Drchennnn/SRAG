"""从本地 .env 文件加载带类型的应用配置。"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """运行时配置；模块导入阶段不初始化外部服务。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    course_root: Path = Path("/Users/Link/Course")
    data_dir: Path = Path("./data")

    dashscope_api_key: str = ""
    dashscope_base_url: str = ""
    dashscope_embedding_model: str = "qwen3.7-text-embedding"
    dashscope_embedding_dimension: int = 1024
    dashscope_embedding_batch_size: int = 20
    dashscope_rerank_model: str = "qwen3.7-text-rerank"
    dashscope_query_model: str = ""
    dashscope_ocr_model: str = ""
    dashscope_vision_model: str = ""
    dashscope_timeout_seconds: float = 60.0
    dashscope_max_retries: int = 3

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = ""

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "course_chunks"
    sqlite_path: Path = Path("./data/srag.sqlite3")

    child_chunk_min_tokens: int = 300
    child_chunk_max_tokens: int = 600
    child_chunk_overlap_tokens: int = 60
    parent_chunk_min_tokens: int = 1000
    parent_chunk_max_tokens: int = 1800
    retrieval_top_k: int = 8
    notebook_output_max_chars: int = 12000


@lru_cache
def get_settings() -> Settings:
    """返回进程内共享的配置对象。"""

    return Settings()
