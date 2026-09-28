"""RAG 索引、文档管理与检索接口契约。"""

from collections.abc import AsyncIterator
from typing import Annotated, NoReturn

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import get_settings
from app.rag.indexer import DocumentIndexer
from app.rag.retriever import HybridRetriever
from app.schemas.rag import (
    DeleteDocumentResponse,
    DocumentSummary,
    IndexRequest,
    IndexStatus,
    SearchRequest,
    SearchResponse,
)
from app.stores.qdrant import QdrantStore

router = APIRouter()


async def get_document_indexer() -> AsyncIterator[DocumentIndexer]:
    """按请求创建索引器，并在请求结束后关闭 Qdrant 连接。"""

    indexer = DocumentIndexer.from_settings(get_settings())
    try:
        yield indexer
    finally:
        if isinstance(indexer.text_store, QdrantStore):
            await indexer.text_store.close()


async def get_retriever() -> AsyncIterator[HybridRetriever]:
    """按请求创建检索器，并在请求结束后关闭 Qdrant 连接。"""

    retriever = HybridRetriever.from_settings(get_settings())
    try:
        yield retriever
    finally:
        if isinstance(retriever.vector_store, QdrantStore):
            await retriever.vector_store.close()


def _pending() -> NoReturn:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="RAG 功能正在实现，当前仅提供 HTTP 接口契约。",
    )


@router.post(
    "/index",
    response_model=IndexStatus,
    summary="创建索引任务",
    response_description="已创建的索引任务状态",
)
async def create_index(
    request: IndexRequest,
    indexer: Annotated[DocumentIndexer, Depends(get_document_indexer)],
) -> IndexStatus:
    """提交明确选择的本地文件进行增量索引。"""

    return await indexer.index(request)


@router.get(
    "/index/{job_id}",
    response_model=IndexStatus,
    summary="查询索引任务",
    response_description="指定索引任务的当前状态",
)
async def get_index_status(job_id: str) -> IndexStatus:
    """返回指定索引任务的状态。"""

    del job_id
    _pending()


@router.get(
    "/documents",
    response_model=list[DocumentSummary],
    summary="查询已登记文档",
    response_description="当前可被检索的文档版本",
)
async def list_documents() -> list[DocumentSummary]:
    """列出已登记且可被检索的文档版本。"""

    _pending()


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteDocumentResponse,
    summary="删除文档",
    response_description="文档及派生数据的删除结果",
)
async def delete_document(document_id: str) -> DeleteDocumentResponse:
    """删除指定文档及其派生块和资源。"""

    del document_id
    _pending()


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="检索课程资料",
    response_description="经过 dense 召回的课程证据",
)
async def search(
    request: SearchRequest,
    retriever: Annotated[HybridRetriever, Depends(get_retriever)],
) -> SearchResponse:
    """通过统一检索入口返回 dense baseline 证据。"""

    return await retriever.search(request)
