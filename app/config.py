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
    course_root: Path = Path("./courses")
    data_dir: Path = Path("./data")

    dashscope_api_key: str = ""
    dashscope_base_url: str = ""
    dashscope_embedding_model: str = "qwen3.7-text-embedding"
    dashscope_embedding_dimension: int = 1024
    dashscope_embedding_batch_size: int = 20
    dashscope_visual_embedding_model: str = "qwen3-vl-embedding"
    dashscope_visual_embedding_dimension: int = 1024
    dashscope_visual_max_image_bytes: int = 10 * 1024 * 1024
    dashscope_rerank_model: str = "qwen3.7-text-rerank"
    dashscope_ocr_model: str = ""
    dashscope_vision_model: str = ""
    dashscope_timeout_seconds: float = 60.0
    dashscope_max_retries: int = 3

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_query_model: str = "deepseek-flash"
    deepseek_chat_model: str = "deepseek-flash"
    deepseek_timeout_seconds: float = 60.0
    deepseek_max_retries: int = 3

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "course_chunks"
    sqlite_path: Path = Path("./data/srag.sqlite3")

    notebook_chunk_soft_min_tokens: int = 120
    notebook_chunk_target_tokens: int = 280
    notebook_chunk_max_tokens: int = 450
    notebook_chunk_overlap_tokens: int = 40
    slide_chunk_max_tokens: int = 650
    slide_chunk_overlap_tokens: int = 50
    slide_pdf_enable_ocr_fallback: bool = False
    slide_pdf_ocr_min_text_tokens: int = 8
    slide_pptx_render_slides: bool = True
    retrieval_top_k: int = 8
    retrieval_enable_visual: bool = True
    retrieval_context_budget_tokens: int = 8000
    retrieval_max_neighbor_chunks: int = 2
    notebook_output_max_chars: int = 12000


@lru_cache
def get_settings() -> Settings:
    """返回进程内共享的配置对象。"""

    return Settings()
