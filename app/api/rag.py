"""RAG 索引、文档管理与检索接口契约。"""

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import get_settings
from app.rag.indexer import DocumentIndexer
from app.rag.local_library import LocalLibrary
from app.rag.query import construct_query
from app.rag.retriever import HybridRetriever
from app.rag.slide_indexer import SlideIndexer
from app.schemas.local import LocalHit
from app.schemas.rag import (
    DeleteDocumentResponse,
    DocumentSummary,
    IndexRequest,
    IndexStatus,
    SearchHit,
    SearchRequest,
    SearchResponse,
    SlideIndexRequest,
    SourceReference,
)
from app.stores.qdrant import QdrantStore
from app.stores.sqlite import MetadataStore

router = APIRouter()


async def get_document_indexer() -> AsyncIterator[DocumentIndexer]:
    """按请求创建索引器，并在请求结束后关闭 Qdrant 连接。"""

    indexer = DocumentIndexer.from_settings(get_settings())
    try:
        yield indexer
    finally:
        if isinstance(indexer.text_store, QdrantStore):
            await indexer.text_store.close()


async def get_retriever(request: SearchRequest) -> AsyncIterator[object]:
    """按配置和显式视觉请求选择检索器，保留既有依赖覆盖入口。"""

    settings = get_settings()
    if settings.rag_search_backend == "local" and request.retrieval_mode != "visual":
        yield LocalRetrieverAdapter(LocalLibrary(settings))
        return
    retriever = HybridRetriever.from_settings(settings)
    try:
        yield retriever
    finally:
        if isinstance(retriever.vector_store, QdrantStore):
            await retriever.vector_store.close()


class LocalRetrieverAdapter:
    """将本地 FTS 命中适配到旧 `/search` 响应，不触发模型或向量服务。"""

    def __init__(self, library: LocalLibrary) -> None:
        self.library = library

    async def search(self, request: SearchRequest) -> SearchResponse:
        constructed = construct_query(request)
        local_request = request.model_copy(update={"query": constructed.text})
        local_hits = await asyncio.to_thread(self.library.search, local_request)
        return SearchResponse(
            query=request.query,
            hits=[_local_hit_to_legacy(hit) for hit in local_hits],
            retrieval_query=constructed.text,
            query_method=constructed.method,
        )


def _local_hit_to_legacy(hit: LocalHit) -> SearchHit:
    locator = hit.locator
    source_id = hit.source_id
    raw_cell_id = locator.get("cell_id")
    cell_id = raw_cell_id if isinstance(raw_cell_id, str) and raw_cell_id else None
    raw_cell_index = locator.get("cell_index")
    cell_index = (
        raw_cell_index
        if isinstance(raw_cell_index, int) and not isinstance(raw_cell_index, bool)
        else None
    )
    raw_section_path = locator.get("section_path")
    section_path = (
        [item.strip() for item in raw_section_path if isinstance(item, str) and item.strip()]
        if isinstance(raw_section_path, list)
        else []
    )
    page = locator.get("page")
    if not section_path and isinstance(page, int) and hit.title.strip():
        section_path = [hit.title.strip()]
    return SearchHit(
        chunk_id=f"local:{hit.document_id}:{hit.version}:{source_id}",
        section_id=f"local:{hit.document_id}:{hit.version}",
        representation_type=f"local_{hit.origin}",
        score=hit.score,
        content=hit.text,
        source=SourceReference(
            course_id=hit.course_id,
            document_id=hit.document_id,
            file=hit.source_path,
            section_path=section_path,
            cell_ids=[cell_id] if cell_id else [],
            cell_id=cell_id,
            cell_index=cell_index,
            page=page if isinstance(page, int) else None,
            version=hit.version,
        ),
    )


def get_metadata_store() -> MetadataStore:
    """创建并初始化当前项目的 SQLite 元数据存储。"""

    store = MetadataStore(get_settings())
    store.initialize()
    return store


async def get_qdrant_store() -> AsyncIterator[QdrantStore]:
    """为文档管理接口创建 Qdrant 客户端。"""

    store = QdrantStore(get_settings())
    try:
        yield store
    finally:
        await store.close()


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


@router.post(
    "/slides/index",
    response_model=IndexStatus,
    summary="导入幻灯片向量包",
    response_description="PDF/PPTX 幻灯片索引任务状态",
)
async def create_slide_index(request: SlideIndexRequest) -> IndexStatus:
    """将已预处理和向量化的 PDF/PPTX 增量导入。"""

    indexer = SlideIndexer.from_settings(get_settings())
    try:
        return await indexer.index(request)
    finally:
        await indexer.vectors.close()


@router.get(
    "/index/{job_id}",
    response_model=IndexStatus,
    summary="查询索引任务",
    response_description="指定索引任务的当前状态",
)
async def get_index_status(
    job_id: str,
    metadata_store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> IndexStatus:
    """返回指定索引任务的状态。"""

    result = metadata_store.get_job(job_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="索引任务不存在")
    return result


@router.get(
    "/documents",
    response_model=list[DocumentSummary],
    summary="查询已登记文档",
    response_description="当前可被检索的文档版本",
)
async def list_documents(
    metadata_store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> list[DocumentSummary]:
    """列出已登记且可被检索的文档版本。"""

    return metadata_store.list_documents()


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteDocumentResponse,
    summary="删除文档",
    response_description="文档及派生数据的删除结果",
)
async def delete_document(
    document_id: str,
    metadata_store: Annotated[MetadataStore, Depends(get_metadata_store)],
    vector_store: Annotated[QdrantStore, Depends(get_qdrant_store)],
) -> DeleteDocumentResponse:
    """删除指定文档及其派生块和资源。"""

    if not metadata_store.has_document(document_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")
    asset_paths = metadata_store.asset_paths(document_id)
    await vector_store.delete_document_versions(document_id)
    deleted = metadata_store.delete_document(document_id)
    metadata_store.delete_managed_asset_files(asset_paths)
    return DeleteDocumentResponse(document_id=document_id, deleted=deleted)


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="检索课程资料",
    response_description="包含实际检索文本和来源的课程证据",
)
async def search(
    request: SearchRequest,
    retriever: Annotated[object, Depends(get_retriever)],
) -> SearchResponse:
    """通过统一检索入口返回课程证据。"""

    return await retriever.search(request)
