"""Qdrant collection 初始化与向量存储边界。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import AsyncQdrantClient, models

from app.config import Settings
from app.schemas.indexing import PreparedDocumentIndex, RepresentationType

TEXT_DENSE_VECTOR = "text_dense"
TEXT_SPARSE_VECTOR = "text_sparse"
VISUAL_DENSE_VECTOR = "visual_dense"
COLLECTION_SCHEMA_VERSION = 1
_UPSERT_BATCH_SIZE = 64

_PAYLOAD_INDEXES = {
    "course_id": models.PayloadSchemaType.KEYWORD,
    "document_id": models.PayloadSchemaType.KEYWORD,
    "representation_type": models.PayloadSchemaType.KEYWORD,
    "version": models.PayloadSchemaType.INTEGER,
    "release_at": models.PayloadSchemaType.DATETIME,
}


class QdrantSchemaError(RuntimeError):
    """现有 collection 与当前索引配置不兼容。"""


class QdrantStore:
    """管理 SRAG 使用的 Qdrant collection；初始化过程不会删除已有数据。"""

    def __init__(
        self,
        settings: Settings,
        *,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        self.url = settings.qdrant_url
        self.api_key = settings.qdrant_api_key or None
        self.collection = settings.qdrant_collection
        self.text_dimension = settings.dashscope_embedding_dimension
        self.visual_dimension = settings.dashscope_visual_embedding_dimension
        self.text_embedding_model = settings.dashscope_embedding_model
        self.visual_embedding_model = settings.dashscope_visual_embedding_model
        self.client = client or AsyncQdrantClient(
            url=self.url,
            api_key=self.api_key,
        )
        self._owns_client = client is None

    async def ensure_collection(self) -> models.CollectionInfo:
        """创建或校验 collection，并补齐检索需要的 payload 索引。"""

        exists = await self.client.collection_exists(self.collection)
        if not exists:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    TEXT_DENSE_VECTOR: models.VectorParams(
                        size=self.text_dimension,
                        distance=models.Distance.COSINE,
                    ),
                    VISUAL_DENSE_VECTOR: models.VectorParams(
                        size=self.visual_dimension,
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    TEXT_SPARSE_VECTOR: models.SparseVectorParams(
                        index=models.SparseIndexParams(),
                    )
                },
                metadata={
                    "schema_version": COLLECTION_SCHEMA_VERSION,
                    "text_embedding_model": self.text_embedding_model,
                    "visual_embedding_model": self.visual_embedding_model,
                },
            )

        info = await self.client.get_collection(self.collection)
        self._validate_vector_schema(info)
        await self._ensure_payload_indexes(info)
        return await self.client.get_collection(self.collection)

    async def upsert_text_representations(
        self,
        prepared: PreparedDocumentIndex,
        *,
        course_id: str,
        version: int = 1,
        release_at: datetime | None = None,
    ) -> int:
        """将一个文档的文本 dense+sparse 表示及溯源 payload 幂等写入 Qdrant。"""

        clean_course_id = course_id.strip()
        if not clean_course_id:
            raise ValueError("course_id 不能为空")
        if version < 1:
            raise ValueError("version 必须大于等于 1")

        await self.ensure_collection()
        chunks = {chunk.chunk_id: chunk for chunk in prepared.chunks}
        release_value = _serialize_datetime(release_at) if release_at else None
        points: list[models.PointStruct] = []

        for embedded in prepared.text_representations:
            representation = embedded.representation
            if len(embedded.dense) != self.text_dimension:
                raise ValueError(
                    f"文本向量 {representation.representation_id!r} 维度应为 "
                    f"{self.text_dimension}，实际为 {len(embedded.dense)}"
                )

            vector: dict[str, list[float] | models.SparseVector] = {
                TEXT_DENSE_VECTOR: embedded.dense,
            }
            if embedded.sparse:
                sparse = sorted(embedded.sparse, key=lambda item: item.index)
                vector[TEXT_SPARSE_VECTOR] = models.SparseVector(
                    indices=[item.index for item in sparse],
                    values=[item.value for item in sparse],
                )

            payload: dict[str, object] = {
                "schema_version": COLLECTION_SCHEMA_VERSION,
                "course_id": clean_course_id,
                "document_id": prepared.document_id,
                "source_path": prepared.source_path,
                "title": prepared.title,
                "version": version,
                "representation_id": representation.representation_id,
                "representation_type": representation.representation_type.value,
                "source_id": representation.source_id,
                "content": representation.text,
                "embedding_model": prepared.text_embedding_model,
            }
            if release_value:
                payload["release_at"] = release_value

            chunk = chunks.get(representation.source_id)
            if chunk is not None:
                payload.update(
                    {
                        "chunk_id": chunk.chunk_id,
                        "section_id": chunk.section_id,
                        "section_path": chunk.section_path,
                        "group_ids": chunk.group_ids,
                        "block_ids": chunk.block_ids,
                        "cell_ids": chunk.cell_ids,
                        "asset_ids": chunk.asset_ids,
                        "chunk_kind": chunk.kind.value,
                        "token_count": chunk.token_count,
                    }
                )
                if chunk.previous_chunk_id:
                    payload["previous_chunk_id"] = chunk.previous_chunk_id
                if chunk.next_chunk_id:
                    payload["next_chunk_id"] = chunk.next_chunk_id

            points.append(
                models.PointStruct(
                    id=_point_id(representation.representation_id, version),
                    vector=vector,
                    payload=payload,
                )
            )

        for start in range(0, len(points), _UPSERT_BATCH_SIZE):
            await self.client.upsert(
                collection_name=self.collection,
                points=points[start : start + _UPSERT_BATCH_SIZE],
                wait=True,
            )
        return len(points)

    async def upsert_visual_representations(
        self,
        prepared: PreparedDocumentIndex,
        *,
        course_id: str,
        version: int = 1,
        release_at: datetime | None = None,
    ) -> int:
        """将一个文档的视觉融合表示写入 Qdrant。"""

        clean_course_id = course_id.strip()
        if not clean_course_id:
            raise ValueError("course_id 不能为空")
        if version < 1:
            raise ValueError("version 必须大于等于 1")

        await self.ensure_collection()
        chunks = {chunk.chunk_id: chunk for chunk in prepared.chunks}
        release_value = _serialize_datetime(release_at) if release_at else None
        points: list[models.PointStruct] = []

        for embedded in prepared.visual_representations:
            representation = embedded.representation
            if len(embedded.dense) != self.visual_dimension:
                raise ValueError(
                    f"视觉向量 {representation.representation_id!r} 维度应为 "
                    f"{self.visual_dimension}，实际为 {len(embedded.dense)}"
                )

            vector: dict[str, list[float]] = {
                VISUAL_DENSE_VECTOR: embedded.dense,
            }

            payload: dict[str, object] = {
                "schema_version": COLLECTION_SCHEMA_VERSION,
                "course_id": clean_course_id,
                "document_id": prepared.document_id,
                "source_path": prepared.source_path,
                "title": prepared.title,
                "version": version,
                "representation_id": representation.representation_id,
                "representation_type": representation.representation_type.value,
                "source_id": representation.source_id,
                "content": representation.context_text,
                "image_path": representation.image_path,
                "media_type": representation.media_type,
                "embedding_model": prepared.visual_embedding_model,
            }
            if release_value:
                payload["release_at"] = release_value

            # 关联到对应的 chunk
            chunk = chunks.get(representation.chunk_id)
            if chunk is not None:
                payload.update(
                    {
                        "chunk_id": chunk.chunk_id,
                        "section_id": chunk.section_id,
                        "section_path": chunk.section_path,
                        "group_ids": chunk.group_ids,
                        "block_ids": chunk.block_ids,
                        "cell_ids": chunk.cell_ids,
                        "chunk_kind": chunk.kind.value,
                    }
                )

            points.append(
                models.PointStruct(
                    id=_point_id(representation.representation_id, version),
                    vector=vector,
                    payload=payload,
                )
            )

        for start in range(0, len(points), _UPSERT_BATCH_SIZE):
            await self.client.upsert(
                collection_name=self.collection,
                points=points[start : start + _UPSERT_BATCH_SIZE],
                wait=True,
            )
        return len(points)

    async def search_text_dense(
        self,
        query_vector: list[float],
        *,
        limit: int,
        course_ids: Sequence[str] = (),
        document_ids: Sequence[str] = (),
        visible_versions: Sequence[tuple[str, int]] | None = None,
    ) -> list[models.ScoredPoint]:
        """在 text_dense 空间召回正文 chunk，不包含 document outline。"""

        if len(query_vector) != self.text_dimension:
            raise ValueError(
                f"查询向量维度应为 {self.text_dimension}，实际为 {len(query_vector)}"
            )
        if not 1 <= limit <= 50:
            raise ValueError("dense 召回数量必须在 1 到 50 之间")

        await self.ensure_collection()
        conditions: list[models.Condition] = [
            models.FieldCondition(
                key="representation_type",
                match=models.MatchValue(value=RepresentationType.TEXT_CONTENT.value),
            )
        ]
        clean_course_ids = _clean_filter_values(course_ids)
        clean_document_ids = _clean_filter_values(document_ids)
        if clean_course_ids:
            conditions.append(
                models.FieldCondition(
                    key="course_id",
                    match=models.MatchAny(any=clean_course_ids),
                )
            )
        if clean_document_ids:
            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=clean_document_ids),
                )
            )
        if visible_versions is not None:
            clean_versions = list(
                dict.fromkeys(
                    (document_id.strip(), version)
                    for document_id, version in visible_versions
                    if document_id.strip() and version >= 1
                )
            )
            if not clean_versions:
                return []
            conditions.append(
                models.Filter(
                    should=[
                        models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="document_id",
                                    match=models.MatchValue(value=document_id),
                                ),
                                models.FieldCondition(
                                    key="version",
                                    match=models.MatchValue(value=version),
                                ),
                            ]
                        )
                        for document_id, version in clean_versions
                    ]
                )
            )

        response = await self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            using=TEXT_DENSE_VECTOR,
            query_filter=models.Filter(must=conditions),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return response.points

    async def search_text_sparse(
        self,
        query_sparse: models.SparseVector,
        *,
        limit: int,
        course_ids: Sequence[str] = (),
        document_ids: Sequence[str] = (),
        visible_versions: Sequence[tuple[str, int]] | None = None,
    ) -> list[models.ScoredPoint]:
        """在 text_sparse 空间召回正文 chunk，不包含 document outline。"""

        if not 1 <= limit <= 50:
            raise ValueError("sparse 召回数量必须在 1 到 50 之间")

        await self.ensure_collection()
        conditions: list[models.Condition] = [
            models.FieldCondition(
                key="representation_type",
                match=models.MatchValue(value=RepresentationType.TEXT_CONTENT.value),
            )
        ]
        clean_course_ids = _clean_filter_values(course_ids)
        clean_document_ids = _clean_filter_values(document_ids)
        if clean_course_ids:
            conditions.append(
                models.FieldCondition(
                    key="course_id",
                    match=models.MatchAny(any=clean_course_ids),
                )
            )
        if clean_document_ids:
            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=clean_document_ids),
                )
            )
        if visible_versions is not None:
            clean_versions = list(
                dict.fromkeys(
                    (document_id.strip(), version)
                    for document_id, version in visible_versions
                    if document_id.strip() and version >= 1
                )
            )
            if not clean_versions:
                return []
            conditions.append(
                models.Filter(
                    should=[
                        models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="document_id",
                                    match=models.MatchValue(value=document_id),
                                ),
                                models.FieldCondition(
                                    key="version",
                                    match=models.MatchValue(value=version),
                                ),
                            ]
                        )
                        for document_id, version in clean_versions
                    ]
                )
            )

        response = await self.client.query_points(
            collection_name=self.collection,
            query=query_sparse,
            using=TEXT_SPARSE_VECTOR,
            query_filter=models.Filter(must=conditions),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return response.points

    async def search_document_outline(
        self,
        query_vector: list[float],
        *,
        limit: int,
        course_ids: Sequence[str] = (),
        document_ids: Sequence[str] = (),
        visible_versions: Sequence[tuple[str, int]] | None = None,
    ) -> list[models.ScoredPoint]:
        """在 text_dense 空间召回 document outline 表示。"""

        if len(query_vector) != self.text_dimension:
            raise ValueError(
                f"查询向量维度应为 {self.text_dimension}，实际为 {len(query_vector)}"
            )
        if not 1 <= limit <= 50:
            raise ValueError("outline 召回数量必须在 1 到 50 之间")

        await self.ensure_collection()
        conditions: list[models.Condition] = [
            models.FieldCondition(
                key="representation_type",
                match=models.MatchValue(value=RepresentationType.DOCUMENT_OUTLINE.value),
            )
        ]
        clean_course_ids = _clean_filter_values(course_ids)
        clean_document_ids = _clean_filter_values(document_ids)
        if clean_course_ids:
            conditions.append(
                models.FieldCondition(
                    key="course_id",
                    match=models.MatchAny(any=clean_course_ids),
                )
            )
        if clean_document_ids:
            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=clean_document_ids),
                )
            )
        if visible_versions is not None:
            clean_versions = list(
                dict.fromkeys(
                    (document_id.strip(), version)
                    for document_id, version in visible_versions
                    if document_id.strip() and version >= 1
                )
            )
            if not clean_versions:
                return []
            conditions.append(
                models.Filter(
                    should=[
                        models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="document_id",
                                    match=models.MatchValue(value=document_id),
                                ),
                                models.FieldCondition(
                                    key="version",
                                    match=models.MatchValue(value=version),
                                ),
                            ]
                        )
                        for document_id, version in clean_versions
                    ]
                )
            )

        response = await self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            using=TEXT_DENSE_VECTOR,
            query_filter=models.Filter(must=conditions),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return response.points

    async def search_visual_fusion(
        self,
        query_vector: list[float],
        *,
        limit: int,
        course_ids: Sequence[str] = (),
        document_ids: Sequence[str] = (),
        visible_versions: Sequence[tuple[str, int]] | None = None,
    ) -> list[models.ScoredPoint]:
        """在 visual_dense 空间召回视觉融合表示。"""

        if len(query_vector) != self.visual_dimension:
            raise ValueError(
                f"视觉查询向量维度应为 {self.visual_dimension}，实际为 {len(query_vector)}"
            )
        if not 1 <= limit <= 50:
            raise ValueError("visual 召回数量必须在 1 到 50 之间")

        await self.ensure_collection()
        conditions: list[models.Condition] = [
            models.FieldCondition(
                key="representation_type",
                match=models.MatchValue(value=RepresentationType.VISUAL_FUSION.value),
            )
        ]
        clean_course_ids = _clean_filter_values(course_ids)
        clean_document_ids = _clean_filter_values(document_ids)
        if clean_course_ids:
            conditions.append(
                models.FieldCondition(
                    key="course_id",
                    match=models.MatchAny(any=clean_course_ids),
                )
            )
        if clean_document_ids:
            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=clean_document_ids),
                )
            )
        if visible_versions is not None:
            clean_versions = list(
                dict.fromkeys(
                    (document_id.strip(), version)
                    for document_id, version in visible_versions
                    if document_id.strip() and version >= 1
                )
            )
            if not clean_versions:
                return []
            conditions.append(
                models.Filter(
                    should=[
                        models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="document_id",
                                    match=models.MatchValue(value=document_id),
                                ),
                                models.FieldCondition(
                                    key="version",
                                    match=models.MatchValue(value=version),
                                ),
                            ]
                        )
                        for document_id, version in clean_versions
                    ]
                )
            )

        response = await self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            using=VISUAL_DENSE_VECTOR,
            query_filter=models.Filter(must=conditions),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return response.points

    async def get_chunks_by_ids(
        self,
        chunk_ids: Sequence[str],
        *,
        document_id: str,
        version: int,
    ) -> list[models.ScoredPoint]:
        """根据 chunk_id 批量获取 chunk 的完整 payload（用于上下文回取）。"""

        if not chunk_ids:
            return []

        await self.ensure_collection()
        conditions: list[models.Condition] = [
            models.FieldCondition(
                key="representation_type",
                match=models.MatchValue(value=RepresentationType.TEXT_CONTENT.value),
            ),
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id),
            ),
            models.FieldCondition(
                key="version",
                match=models.MatchValue(value=version),
            ),
            models.FieldCondition(
                key="chunk_id",
                match=models.MatchAny(any=list(chunk_ids)),
            ),
        ]

        records, offset = await self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(must=conditions),
            limit=len(chunk_ids),
            with_payload=True,
            with_vectors=False,
        )

        # 转换为 ScoredPoint（scroll 返回的是 Record）
        return [
            models.ScoredPoint(
                id=record.id,
                version=record.version or 0,
                score=1.0,  # scroll 没有得分，设置为 1.0
                payload=record.payload,
            )
            for record in records
        ]

    async def delete_stale_text_points(
        self,
        prepared: PreparedDocumentIndex,
        *,
        version: int,
    ) -> int:
        """删除同一文档版本中已不属于本次索引结果的文本 points。"""

        await self.ensure_collection()
        desired_ids = {
            _point_id(item.representation.representation_id, version)
            for item in prepared.text_representations
        }
        records, offset = await self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=prepared.document_id),
                    ),
                    models.FieldCondition(
                        key="version",
                        match=models.MatchValue(value=version),
                    ),
                ]
            ),
            limit=256,
            with_payload=False,
            with_vectors=False,
        )
        while offset is not None:
            page, offset = await self.client.scroll(
                collection_name=self.collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=prepared.document_id),
                        ),
                        models.FieldCondition(
                            key="version",
                            match=models.MatchValue(value=version),
                        ),
                    ]
                ),
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            records.extend(page)

        stale_ids = [record.id for record in records if str(record.id) not in desired_ids]
        if stale_ids:
            await self.client.delete(
                collection_name=self.collection,
                points_selector=models.PointIdsList(points=stale_ids),
                wait=True,
            )
        return len(stale_ids)

    async def delete_document_versions(
        self,
        document_id: str,
        *,
        keep_version: int | None = None,
    ) -> None:
        """删除文档全部 points，或保留指定版本。"""

        await self.ensure_collection()
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            ],
            must_not=(
                [
                    models.FieldCondition(
                        key="version",
                        match=models.MatchValue(value=keep_version),
                    )
                ]
                if keep_version is not None
                else None
            ),
        )
        await self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=query_filter),
            wait=True,
        )

    async def close(self) -> None:
        """关闭由存储对象创建的客户端连接。"""

        if self._owns_client:
            await self.client.close()

    def _validate_vector_schema(self, info: models.CollectionInfo) -> None:
        dense = info.config.params.vectors
        sparse = info.config.params.sparse_vectors or {}
        if not isinstance(dense, dict):
            raise QdrantSchemaError("Qdrant collection 必须使用命名向量")

        expected_dense = {
            TEXT_DENSE_VECTOR: self.text_dimension,
            VISUAL_DENSE_VECTOR: self.visual_dimension,
        }
        errors: list[str] = []
        for name, expected_size in expected_dense.items():
            params = dense.get(name)
            if params is None:
                errors.append(f"缺少 dense 向量 {name}")
                continue
            if params.size != expected_size:
                errors.append(f"{name} 维度应为 {expected_size}，实际为 {params.size}")
            if params.distance != models.Distance.COSINE:
                errors.append(f"{name} 距离度量应为 Cosine，实际为 {params.distance.value}")

        if TEXT_SPARSE_VECTOR not in sparse:
            errors.append(f"缺少 sparse 向量 {TEXT_SPARSE_VECTOR}")

        if errors:
            joined = "；".join(errors)
            raise QdrantSchemaError(
                f"Qdrant collection {self.collection!r} 与当前配置不兼容：{joined}。"
                "为避免破坏已有数据，程序不会自动重建 collection。"
            )

    async def _ensure_payload_indexes(self, info: models.CollectionInfo) -> None:
        for field_name, expected_type in _PAYLOAD_INDEXES.items():
            current = info.payload_schema.get(field_name)
            if current is not None:
                if current.data_type != expected_type:
                    raise QdrantSchemaError(
                        f"Qdrant payload 字段 {field_name!r} 的索引类型应为 "
                        f"{expected_type.value}，实际为 {current.data_type.value}"
                    )
                continue
            await self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field_name,
                field_schema=expected_type,
                wait=True,
            )


def _point_id(representation_id: str, version: int) -> str:
    """把项目内部稳定 ID 映射为 Qdrant 支持的稳定 UUID。"""

    return str(uuid5(NAMESPACE_URL, f"srag:{representation_id}:v{version}"))


def _serialize_datetime(value: datetime) -> str:
    """转换为 Qdrant datetime 索引接受的 UTC RFC 3339 字符串。"""

    aware = value if value.tzinfo else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _clean_filter_values(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))
