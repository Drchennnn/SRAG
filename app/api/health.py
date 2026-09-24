"""服务健康检查接口。"""

from fastapi import APIRouter

from app import __version__

router = APIRouter()


@router.get(
    "/health",
    summary="检查服务状态",
    response_description="SRAG 服务的当前状态",
)
async def health() -> dict[str, str]:
    """返回进程状态；该检查不依赖模型 API 或 Qdrant。"""

    return {"status": "ok", "service": "srag", "version": __version__}
