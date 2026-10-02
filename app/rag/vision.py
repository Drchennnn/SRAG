"""读取已入库页面截图，缓存视觉文字并追加文本索引。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from uuid import uuid4

from qdrant_client.http.exceptions import ApiException

from app.clients.dashscope import DashScopeClient
from app.schemas.embedding import EmbeddingVector
from app.schemas.rag import DocumentContent, DocumentSummary, SearchHit, SearchScope
from app.schemas.vision import VisualExtraction
from app.stores.qdrant import QdrantStore
from app.stores.sqlite import MetadataStore

_PROMPT_VERSION = 2
_VISUAL_QUESTION = re.compile(
    r"公式|高斯|核|矩阵|图表|图像|图片|图中|曲线|(?:页|的|这张|那张)\s*图|formula|matrix", re.I
)


class VisualTextReader:
    """最多按需读取两页；不把未注册的路径发给模型。"""

    def __init__(
        self, catalog: MetadataStore, client: DashScopeClient, store: QdrantStore, cache_dir: Path
    ) -> None:
        self.catalog, self.client, self.store, self.cache_dir = catalog, client, store, cache_dir

    def _registered(self, hit: SearchHit, scope: SearchScope) -> tuple[DocumentSummary, SearchHit]:
        documents = [
            item
            for item in self.catalog.list_documents()
            if item.document_id == hit.source.document_id
        ]
        if len(documents) != 1:
            raise ValueError("视觉来源已更新或不可见")
        document = documents[0]
        if hit.source.version is not None and hit.source.version != document.version:
            raise ValueError("召回来源已不是当前可见版本")
        read = self.catalog.read_document(
            document.document_id,
            document.version,
            course_ids=scope.course_ids,
            document_ids=scope.document_ids,
        )
        matches = [
            item
            for item in read.hits
            if (
                item.source.page == hit.source.page
                if hit.source.page is not None
                else item.chunk_id == hit.chunk_id
            )
            and item.source.image_path
        ]
        if not matches:
            raise ValueError("此来源没有已注册的可读截图")
        return document, matches[0]

    def _cache_path(self, document: DocumentSummary, hit: SearchHit) -> tuple[Path, str]:
        image = Path(hit.source.image_path)
        image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
        signature = [
            document.document_id,
            document.version,
            hit.source.page,
            hit.chunk_id,
            image_hash,
            self.client.vision_model,
            _PROMPT_VERSION,
            self.client.embedding_model,
            self.client.embedding_dimension,
        ]
        key = hashlib.sha256(json.dumps(signature).encode()).hexdigest()
        return self.cache_dir / (key + ".json"), key

    async def read(
        self, hit: SearchHit, scope: SearchScope, *, cached_only: bool = False
    ) -> SearchHit:
        document, registered = self._registered(hit, scope)
        path, key = self._cache_path(document, registered)
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            extraction = VisualExtraction.model_validate(record["extraction"])
            vector = EmbeddingVector.model_validate(record["vector"])
        else:
            if cached_only:
                return hit
            extraction = await self.client.extract_visual_text(Path(registered.source.image_path))
            text = (
                registered.content
                + "\n视觉理解提取（未经人工核对）：\n"
                + extraction.readable_text()
            )
            result = await self.client.embed_documents([text])
            if len(result.vectors) != 1:
                raise ValueError("视觉文字向量化没有返回单个向量")
            vector = result.vectors[0]
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "document_id": document.document_id,
                "version": document.version,
                "page": registered.source.page,
                "image_path": registered.source.image_path,
                "vision_model": self.client.vision_model,
                "prompt_version": _PROMPT_VERSION,
                "extraction": extraction.model_dump(),
                "vector": vector.model_dump(),
            }
            temporary = path.with_suffix(f".{uuid4().hex}.tmp")
            temporary.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
        # 发布/回答前再验证同一个可见版本，更新或删除不复用旧缓存。
        current, _ = self._registered(registered, scope)
        if current.version != document.version:
            raise ValueError("视觉理解期间文档已更新，请重新提问")
        enriched = registered.model_copy(
            update={
                "content": registered.content
                + "\n视觉理解提取（未经人工核对）：\n"
                + extraction.readable_text(),
                "evidence_origin": "visual_extract",
                "origin_chunk_id": registered.chunk_id,
                "score": hit.score,
            }
        )
        if not cached_only:
            await self.store.upsert_visual_text(
                document,
                enriched,
                vector,
                cache_key=key,
                embedding_model=self.client.embedding_model,
            )
        return enriched

    async def enrich(
        self, query: str, hits: list[SearchHit], scope: SearchScope
    ) -> tuple[list[SearchHit], list[str]]:
        output, warnings, pages = [], [], set()
        for hit in hits:
            key = (hit.source.document_id, hit.source.page)
            if (
                not _VISUAL_QUESTION.search(query)
                or len(pages) >= 2
                or key in pages
                or hit.evidence_origin == "visual_extract"
            ):
                output.append(hit)
                continue
            pages.add(key)
            try:
                output.append(await self.read(hit, scope))
            except (ValueError, RuntimeError, OSError, KeyError, ApiException) as error:
                warnings.append(f"第 {hit.source.page} 页视觉补充失败：{type(error).__name__}")
                output.append(hit)
        return output, warnings

    async def cached_document(self, read: DocumentContent, scope: SearchScope) -> DocumentContent:
        hits = []
        for hit in read.hits:
            try:
                hits.append(await self.read(hit, scope, cached_only=True))
            except (ValueError, RuntimeError, OSError, KeyError):
                hits.append(hit)
        extracted = {hit.source.page for hit in hits if hit.evidence_origin == "visual_extract"}
        return read.model_copy(
            update={
                "hits": hits,
                "image_only_pages": [p for p in read.image_only_pages if p not in extracted],
            }
        )
