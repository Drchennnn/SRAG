"""FastAPI 应用入口。"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.health import router as health_router
from app.api.local_library import router as local_library_router
from app.api.rag import router as rag_router
from app.api.source_evidence import router as source_evidence_router
from app.api.structure_search import router as structure_search_router
from app.api.workspace import router as workspace_router
from app.config import get_settings


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """准备本地目录，避免在模块导入阶段连接外部服务。"""

    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    yield


def create_app() -> FastAPI:
    """创建 HTTP 应用。"""

    app = FastAPI(
        title="SRAG",
        version=__version__,
        description="面向课程 Notebook 与 PDF 资料的本地优先 RAG 服务",
        lifespan=lifespan,
    )
    app.include_router(health_router, prefix="/api", tags=["健康检查"])
    app.include_router(rag_router, prefix="/api/rag", tags=["知识检索"])
    app.include_router(source_evidence_router, prefix="/api/rag/local/evidence", tags=["原件"])
    app.include_router(structure_search_router, prefix="/api/rag/structure", tags=["结构检索"])
    app.include_router(local_library_router, prefix="/api/rag/local", tags=["本地资料"])
    app.include_router(workspace_router, prefix="/api/workspace", tags=["课程工作区"])
    workspace_assets = Path(__file__).with_name("web") / "workspace"
    app.mount("/workspace", StaticFiles(directory=workspace_assets, html=True), name="workspace")
    return app


app = create_app()
