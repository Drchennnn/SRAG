"""Optional isolated local/remote Qdrant vector retrieval bridge for S3."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal
from uuid import NAMESPACE_URL, uuid5

import httpx
from qdrant_client import QdrantClient, models

from app.clients.dashscope import DashScopeClient
from app.config import Settings
from app.rag.fusion import visual_query_intent
from app.rag.structure_search import StructureCandidate, _structural_candidate
from app.rag.workspace_evidence import readable_source_text
from app.schemas.indexing import RepresentationType
from app.schemas.local import SourceKey
from app.schemas.workspace import WorkspaceError
from app.stores.local_library import LocalLibraryStore
from app.stores.qdrant import (
    COLLECTION_SCHEMA_VERSION,
    TEXT_DENSE_VECTOR,
    TEXT_SPARSE_VECTOR,
    VISUAL_DENSE_VECTOR,
    QdrantSchemaError,
    build_search_filter,
    validate_collection_schema,
)
from app.stores.structure_snapshot import SnapshotDocument, SnapshotSource, StructureSnapshot

if TYPE_CHECKING:
    from app.rag.source_evidence import SourceEvidenceService

logger = logging.getLogger(__name__)

_SOURCE_ORIGIN = "local_library"
_MAX_INDEX_CHARS = 8_000
_UPSERT_BATCH = 64
_EMBED_TEXT_BATCH = 16
_LOW_TEXT_VISUAL_LIMIT = 40
_VISUAL_SLIDE_QUERY = re.compile(
    r"(?:实验|架构|框|曲线|示意|流程|结构|对比|比较|分布|结果|热力|散点|柱状|折线|直方|系统|原理|网络|训练|损失|性能)图"
)


@dataclass(frozen=True)
class VectorRetrievalResult:
    status: Literal["not_requested", "succeeded", "unavailable", "provider_error"]
    text_candidates: tuple[StructureCandidate, ...] = ()
    visual_candidates: tuple[StructureCandidate, ...] = ()
    visual_read_targets: tuple[StructureCandidate, ...] = ()
    trace: dict[str, Any] = field(default_factory=lambda: {"status": "not_requested"})


def _run_coro_sync(coro: Any) -> Any:
    """Run an async coroutine safely whether or not the caller thread has an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


def _point_id(
    course_id: str,
    document_id: str,
    version: int,
    source_id: str,
    representation_type: str,
) -> str:
    rep_id = f"{_SOURCE_ORIGIN}:{course_id}:{document_id}:{source_id}"
    return str(uuid5(NAMESPACE_URL, f"{rep_id}:v{version}:{representation_type}"))


def _needs_text_embedding(
    source: SnapshotSource, by_key: dict[tuple[str, int, str], SnapshotSource]
) -> str:
    text = readable_source_text(source).strip()
    if not text:
        return ""
    object_type = source.locator.get("object_type")
    if (
        source.locator.get("format") == "pptx"
        and object_type in {"title", "text"}
        and source.parent_id
    ):
        parent = by_key.get((source.document_id, source.version, source.parent_id))
        if parent is not None and text in parent.text:
            return ""
    return text[:_MAX_INDEX_CHARS]


def _is_sparse_visual_source(source: SnapshotSource) -> bool:
    text = readable_source_text(source).strip()
    if not text:
        return True
    if source.locator.get("format") == "pptx" and len(text) < _LOW_TEXT_VISUAL_LIMIT:
        return True
    return False


def _visual_context_text(doc: SnapshotDocument, source: SnapshotSource) -> str:
    text = readable_source_text(source).strip()
    if text:
        return text[:1_200]
    page = source.locator.get("page")
    cell = source.locator.get("cell_index")
    if isinstance(page, int):
        loc = f"第 {page} 页"
    elif isinstance(cell, int):
        loc = f"Cell {cell + 1}"
    else:
        loc = source.source_id
    title = source.title.strip() or "幻灯片整页视觉内容"
    return f"{doc.file_name} · {loc} · {title}"


class LocalVectorBridge:
    """Single-instance synchronous Qdrant bridge with explicit lock/close lifecycle for S3."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: QdrantClient | None = None,
        dashscope_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.collection = settings.qdrant_collection
        self.text_dimension = settings.dashscope_embedding_dimension
        self.visual_dimension = settings.dashscope_visual_embedding_dimension
        self.text_model = settings.dashscope_embedding_model
        self.visual_model = settings.dashscope_visual_embedding_model
        self._local_path = settings.qdrant_local_path
        self._qdrant_url = settings.qdrant_url
        self._qdrant_api_key = settings.qdrant_api_key or None
        self._external_client = client
        self._client: QdrantClient | None = client
        self._owns_client = client is None
        self._closed = False
        self._lock = Lock()
        self._dashscope_transport = dashscope_transport

    def _dashscope(self) -> DashScopeClient:
        return DashScopeClient(self.settings, transport=self._dashscope_transport)

    def _get_client(self) -> QdrantClient:
        if self._closed:
            raise RuntimeError("LocalVectorBridge has been closed")
        if self._client is not None:
            return self._client
        if self._local_path is not None:
            path = Path(self._local_path)
            path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(path))
        else:
            self._client = QdrantClient(
                url=self._qdrant_url,
                api_key=self._qdrant_api_key,
                timeout=max(2.0, min(10.0, self.settings.dashscope_timeout_seconds)),
            )
        return self._client

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._owns_client and self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

    def _ensure_collection(self, client: QdrantClient) -> None:
        if not client.collection_exists(self.collection):
            client.create_collection(
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
                    "text_embedding_model": self.text_model,
                    "visual_embedding_model": self.visual_model,
                },
            )
        info = client.get_collection(self.collection)
        validate_collection_schema(
            self.collection,
            info,
            text_dimension=self.text_dimension,
            visual_dimension=self.visual_dimension,
        )

    def ensure_synced(
        self,
        snapshot: StructureSnapshot,
        library_store: LocalLibraryStore,
        evidence_service: SourceEvidenceService,
        *,
        active: Callable[[], bool] = lambda: True,
        include_visual: bool = True,
    ) -> dict[str, Any]:
        """Idempotently embed and upsert current snapshot sources missing from Qdrant."""
        started = perf_counter()
        if not self.settings.dashscope_api_key.strip():
            return {"status": "unavailable", "reason": "not_configured", "external_call_count": 0}
        if not active():
            raise WorkspaceError("interrupted")

        docs = {doc.document_id: doc for doc in snapshot.documents}
        by_key = {(src.document_id, src.version, src.source_id): src for src in snapshot.sources}
        text_specs: list[tuple[str, SnapshotDocument, SnapshotSource, str]] = []
        visual_specs: list[tuple[str, SnapshotDocument, SnapshotSource, str]] = []
        for src in snapshot.sources:
            doc = docs.get(src.document_id)
            if doc is None:
                continue
            text = _needs_text_embedding(src, by_key)
            if text:
                pid = _point_id(
                    doc.course_id,
                    doc.document_id,
                    doc.version,
                    src.source_id,
                    RepresentationType.TEXT_CONTENT.value,
                )
                text_specs.append((pid, doc, src, text))
            if include_visual and _is_visual_index_target(doc, src):
                pid = _point_id(
                    doc.course_id,
                    doc.document_id,
                    doc.version,
                    src.source_id,
                    RepresentationType.VISUAL_FUSION.value,
                )
                visual_specs.append((pid, doc, src, _visual_context_text(doc, src)))

        all_ids = [item[0] for item in (*text_specs, *visual_specs)]
        if not all_ids:
            return {
                "status": "empty",
                "indexed_text": 0,
                "indexed_visual": 0,
                "external_call_count": 0,
            }

        try:
            with self._lock:
                client = self._get_client()
                self._ensure_collection(client)
                existing_records = client.retrieve(
                    collection_name=self.collection,
                    ids=all_ids,
                    with_payload=True,
                    with_vectors=False,
                )
        except WorkspaceError:
            raise
        except Exception as exc:
            return {
                "status": "unavailable",
                "reason": type(exc).__name__,
                "external_call_count": 0,
                "elapsed_ms": round((perf_counter() - started) * 1000, 3),
            }

        existing_map = {str(rec.id): rec.payload or {} for rec in existing_records}
        missing_text = [
            spec
            for spec in text_specs
            if not _payload_matches_doc(existing_map.get(spec[0]), spec[1], spec[2])
        ]
        missing_visual = [
            spec
            for spec in visual_specs
            if not _payload_matches_doc(existing_map.get(spec[0]), spec[1], spec[2])
        ]
        if not missing_text and not missing_visual:
            return {
                "status": "cached",
                "indexed_text": 0,
                "indexed_visual": 0,
                "external_call_count": 0,
                "elapsed_ms": round((perf_counter() - started) * 1000, 3),
            }

        dashscope = self._dashscope()
        points_to_upsert: list[models.PointStruct] = []
        external_calls = 0
        total_tokens = 0
        skipped_visual = 0

        try:
            for idx in range(0, len(missing_text), _EMBED_TEXT_BATCH):
                if not active():
                    raise WorkspaceError("interrupted")
                batch = missing_text[idx : idx + _EMBED_TEXT_BATCH]
                batch_result = _run_coro_sync(
                    dashscope.embed([item[3] for item in batch], text_type="document")
                )
                external_calls += 1
                total_tokens += batch_result.total_tokens
                for (pid, doc, src, content), vec in zip(batch, batch_result.vectors, strict=True):
                    vector: dict[str, Any] = {TEXT_DENSE_VECTOR: vec.dense}
                    if vec.sparse:
                        sorted_sparse = sorted(vec.sparse, key=lambda item: item.index)
                        vector[TEXT_SPARSE_VECTOR] = models.SparseVector(
                            indices=[item.index for item in sorted_sparse],
                            values=[item.value for item in sorted_sparse],
                        )
                    points_to_upsert.append(
                        models.PointStruct(
                            id=pid,
                            vector=vector,
                            payload=_point_payload(
                                doc,
                                src,
                                representation_id=f"{_SOURCE_ORIGIN}:{doc.course_id}:{doc.document_id}:{src.source_id}",
                                representation_type=RepresentationType.TEXT_CONTENT.value,
                                model_name=self.text_model,
                                content=content[:800],
                            ),
                        )
                    )

            from app.rag.source_evidence import EvidenceRequest

            with TemporaryDirectory(prefix="srag-j-vec-vis-") as tmp_dir:
                tmp_root = Path(tmp_dir)
                for idx, (pid, doc, src, context_text) in enumerate(missing_visual):
                    if not active():
                        raise WorkspaceError("interrupted")
                    rendition = (
                        "notebook_output" if src.kind == "notebook_output" else "original_page"
                    )
                    req = EvidenceRequest(
                        source=SourceKey(
                            document_id=src.document_id,
                            version=src.version,
                            source_id=src.source_id,
                        ),
                        expected_content_sha256=doc.content_sha256,
                        expected_publication_epoch=doc.publication_epoch,
                        rendition=rendition,
                    )
                    rendered = evidence_service.render_current(req)
                    if rendered.status in {"stale_reference", "not_accessible"}:
                        raise WorkspaceError(rendered.status)
                    display = rendered.display
                    if rendered.status != "available" or display is None:
                        skipped_visual += 1
                        continue
                    img_bytes = base64.b64decode(display.data_base64, validate=True)
                    if hashlib.sha256(img_bytes).hexdigest() != display.byte_sha256:
                        skipped_visual += 1
                        continue
                    supported_img_types = {
                        "image/png",
                        "image/jpeg",
                        "image/webp",
                        "image/bmp",
                        "image/tiff",
                    }
                    if (
                        len(img_bytes) > self.settings.dashscope_visual_max_image_bytes
                        or display.media_type not in supported_img_types
                    ):
                        skipped_visual += 1
                        continue
                    ext = display.media_type.split("/")[-1].replace("jpeg", "jpg")
                    img_file = tmp_root / f"vis_{idx}.{ext}"
                    img_file.write_bytes(img_bytes)
                    vis_batch = _run_coro_sync(
                        dashscope.embed_visual_fusion(
                            text=context_text,
                            image_path=img_file,
                            media_type=display.media_type,
                        )
                    )
                    external_calls += 1
                    total_tokens += vis_batch.total_tokens
                    points_to_upsert.append(
                        models.PointStruct(
                            id=pid,
                            vector={VISUAL_DENSE_VECTOR: vis_batch.vectors[0].dense},
                            payload=_point_payload(
                                doc,
                                src,
                                representation_id=f"{_SOURCE_ORIGIN}:{doc.course_id}:{doc.document_id}:{src.source_id}",
                                representation_type=RepresentationType.VISUAL_FUSION.value,
                                model_name=self.visual_model,
                                content=context_text[:800],
                            ),
                        )
                    )

            if points_to_upsert:
                with self._lock:
                    client = self._get_client()
                    for start in range(0, len(points_to_upsert), _UPSERT_BATCH):
                        client.upsert(
                            collection_name=self.collection,
                            points=points_to_upsert[start : start + _UPSERT_BATCH],
                            wait=True,
                        )
        except WorkspaceError:
            raise
        except Exception as exc:
            return {
                "status": "provider_error",
                "reason": type(exc).__name__,
                "error": str(exc)[:240],
                "external_call_count": external_calls,
                "elapsed_ms": round((perf_counter() - started) * 1000, 3),
            }

        return {
            "status": "succeeded",
            "indexed_text": len(missing_text),
            "indexed_visual": len(missing_visual) - skipped_visual,
            "skipped_visual": skipped_visual,
            "external_call_count": external_calls,
            "total_tokens": total_tokens,
            "elapsed_ms": round((perf_counter() - started) * 1000, 3),
        }

    def search(
        self,
        *,
        snapshot: StructureSnapshot,
        docs: dict[str, SnapshotDocument],
        sources: dict[tuple[str, int, str], SnapshotSource],
        question: str,
        alternate_query: str | None,
        pool_cap: int,
        s0_status: str,
        s0_pool: Sequence[StructureCandidate],
        library_store: LocalLibraryStore | None = None,
        evidence_service: SourceEvidenceService | None = None,
        verify: Callable[[], None] = lambda: None,
        max_visual_reads: int = 2,
    ) -> VectorRetrievalResult:
        """Query text dense+sparse and conditional visual fusion over current visible versions."""
        started = perf_counter()
        verify()
        if not self.settings.dashscope_api_key.strip():
            return VectorRetrievalResult(
                status="unavailable",
                trace={
                    "status": "unavailable",
                    "reason": "not_configured",
                    "external_call_count": 0,
                    "stale_vector_filtered": 0,
                    "elapsed_ms": round((perf_counter() - started) * 1000, 3),
                },
            )
        if not docs:
            return VectorRetrievalResult(
                status="succeeded",
                trace={
                    "status": "empty_scope",
                    "external_call_count": 0,
                    "stale_vector_filtered": 0,
                    "elapsed_ms": round((perf_counter() - started) * 1000, 3),
                },
            )

        sync_trace: dict[str, Any] = {"status": "skipped", "external_call_count": 0}
        if library_store is not None and evidence_service is not None:
            sync_trace = self.ensure_synced(
                snapshot,
                library_store,
                evidence_service,
                active=lambda: _check_active(verify),
            )
            if sync_trace.get("status") in {"unavailable", "provider_error"}:
                return VectorRetrievalResult(
                    status=sync_trace["status"],
                    trace={
                        **sync_trace,
                        "stage": "sync",
                        "stale_vector_filtered": 0,
                        "fallback_to": "s0",
                    },
                )

        verify()
        course_ids = list(dict.fromkeys(doc.course_id for doc in docs.values()))
        doc_ids = list(docs)
        visible_versions = [(doc.document_id, doc.version) for doc in docs.values()]
        text_filter = build_search_filter(
            RepresentationType.TEXT_CONTENT.value,
            course_ids=course_ids,
            document_ids=doc_ids,
            visible_versions=visible_versions,
            source_origin=_SOURCE_ORIGIN,
        )
        visual_filter = build_search_filter(
            RepresentationType.VISUAL_FUSION.value,
            course_ids=course_ids,
            document_ids=doc_ids,
            visible_versions=visible_versions,
            source_origin=_SOURCE_ORIGIN,
        )
        if text_filter is None or visual_filter is None:
            return VectorRetrievalResult(
                status="succeeded",
                trace={
                    "status": "empty_versions",
                    "external_call_count": 0,
                    "stale_vector_filtered": 0,
                },
            )

        dashscope = self._dashscope()
        query_text = alternate_query or question
        external_calls = 0
        query_tokens = 0
        stale_filtered = 0
        visual_triggered = False

        try:
            qvec = _run_coro_sync(dashscope.embed_query(query_text))
            external_calls += 1
            sparse_vec = (
                models.SparseVector(
                    indices=[item.index for item in sorted(qvec.sparse, key=lambda x: x.index)],
                    values=[item.value for item in sorted(qvec.sparse, key=lambda x: x.index)],
                )
                if qvec.sparse
                else None
            )

            verify()
            with self._lock:
                client = self._get_client()
                self._ensure_collection(client)
                dense_points = client.query_points(
                    collection_name=self.collection,
                    query=qvec.dense,
                    using=TEXT_DENSE_VECTOR,
                    query_filter=text_filter,
                    limit=min(50, max(pool_cap, 12)),
                    with_payload=True,
                    with_vectors=False,
                ).points
                sparse_points = (
                    client.query_points(
                        collection_name=self.collection,
                        query=sparse_vec,
                        using=TEXT_SPARSE_VECTOR,
                        query_filter=text_filter,
                        limit=min(50, max(pool_cap, 12)),
                        with_payload=True,
                        with_vectors=False,
                    ).points
                    if sparse_vec is not None
                    else []
                )

            verify()
            verified_dense, f1 = _post_verify_points(dense_points, docs, sources)
            verified_sparse, f2 = _post_verify_points(sparse_points, docs, sources)
            stale_filtered += f1 + f2

            # RRF fuse dense + sparse text vector hits before deciding visual fallback.
            rrf_scores: dict[tuple[str, int, str], float] = {}
            src_by_key: dict[tuple[str, int, str], tuple[SnapshotSource, float]] = {}
            for ranked_list in (verified_dense, verified_sparse):
                for rank, (src, raw_score) in enumerate(ranked_list, start=1):
                    key = (src.document_id, src.version, src.source_id)
                    rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank)
                    if key not in src_by_key or raw_score > src_by_key[key][1]:
                        src_by_key[key] = (src, raw_score)

            text_candidates = tuple(
                _vector_candidate(src_by_key[key][0], docs, "vector_text", score=rrf_scores[key])
                for key in sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:pool_cap]
            )

            visual_triggered = _should_trigger_visual_search(
                question,
                s0_status,
                s0_pool,
                docs,
                sources,
                vector_text_count=len(text_candidates),
            )
            visual_points: list[models.ScoredPoint] = []
            if visual_triggered:
                vis_query = (
                    f"{question} ({alternate_query})"
                    if alternate_query and alternate_query.strip()
                    else question
                )
                vis_batch = _run_coro_sync(dashscope.embed_visual_query(vis_query))
                external_calls += 1
                query_tokens += vis_batch.total_tokens
                vis_qvec = vis_batch.vectors[0].dense
                verify()
                with self._lock:
                    client = self._get_client()
                    visual_points = client.query_points(
                        collection_name=self.collection,
                        query=vis_qvec,
                        using=VISUAL_DENSE_VECTOR,
                        query_filter=visual_filter,
                        limit=16,
                        with_payload=True,
                        with_vectors=False,
                    ).points
        except WorkspaceError:
            raise
        except (QdrantSchemaError, httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
            status_val: Literal["unavailable", "provider_error"] = (
                "unavailable"
                if isinstance(exc, (QdrantSchemaError, OSError)) or "lock" in str(exc).casefold()
                else "provider_error"
            )
            return VectorRetrievalResult(
                status=status_val,
                trace={
                    "status": status_val,
                    "error": f"{type(exc).__name__}: {str(exc)[:180]}",
                    "external_call_count": (
                        external_calls + sync_trace.get("external_call_count", 0)
                    ),
                    "stale_vector_filtered": stale_filtered,
                    "fallback_to": "s0",
                    "sync": sync_trace,
                    "elapsed_ms": round((perf_counter() - started) * 1000, 3),
                },
            )

        verify()
        verified_visual, f3 = _post_verify_points(visual_points, docs, sources)
        stale_filtered += f3

        # Rank pure/sparse-text visual pages first when visual_triggered.
        visual_candidates_list: list[StructureCandidate] = []
        visual_read_targets_list: list[StructureCandidate] = []
        seen_vis: set[tuple[str, int, str]] = set()
        prioritized_visual = sorted(
            verified_visual,
            key=lambda pair: (
                not _is_sparse_visual_source(pair[0]),
                -float(pair[1]),
            ),
        )
        for src, raw_score in prioritized_visual:
            if raw_score <= 0.0:
                continue
            key = (src.document_id, src.version, src.source_id)
            if key in seen_vis:
                continue
            seen_vis.add(key)
            cand = _vector_candidate(src, docs, "vector_visual", score=raw_score)
            visual_candidates_list.append(cand)
            if (
                _is_sparse_visual_source(src)
                and len(visual_read_targets_list) < max_visual_reads
            ):
                visual_read_targets_list.append(cand)
            if len(visual_candidates_list) >= 3:
                break

        return VectorRetrievalResult(
            status="succeeded",
            text_candidates=text_candidates,
            visual_candidates=tuple(visual_candidates_list),
            visual_read_targets=tuple(visual_read_targets_list),
            trace={
                "status": "succeeded",
                "query_used": query_text,
                "visual_triggered": visual_triggered,
                "stale_vector_filtered": stale_filtered,
                "text_hit_count": len(text_candidates),
                "visual_hit_count": len(visual_candidates_list),
                "visual_read_target_count": len(visual_read_targets_list),
                "external_call_count": external_calls,
                "sync": sync_trace,
                "elapsed_ms": round((perf_counter() - started) * 1000, 3),
            },
        )


def _check_active(verify: Callable[[], None]) -> bool:
    verify()
    return True


def _is_visual_index_target(doc: SnapshotDocument, source: SnapshotSource) -> bool:
    if source.kind == "page" and doc.document_format in {"pdf", "pptx"}:
        return True
    if source.kind == "notebook_output":
        return any(
            isinstance(rep, dict)
            and rep.get("status") == "available"
            and str(rep.get("mime_type", "")).startswith("image/")
            for rep in source.locator.get("representations", ())
        )
    return False


def _payload_matches_doc(
    payload: dict[str, Any] | None, doc: SnapshotDocument, src: SnapshotSource
) -> bool:
    if not payload:
        return False
    return (
        payload.get("source_origin") == _SOURCE_ORIGIN
        and payload.get("course_id") == doc.course_id
        and payload.get("document_id") == doc.document_id
        and payload.get("version") == doc.version
        and payload.get("publication_epoch") == doc.publication_epoch
        and payload.get("content_sha256") == doc.content_sha256
        and payload.get("source_id") == src.source_id
    )


def _point_payload(
    doc: SnapshotDocument,
    src: SnapshotSource,
    *,
    representation_id: str,
    representation_type: str,
    model_name: str,
    content: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "source_origin": _SOURCE_ORIGIN,
        "source_kind": src.kind,
        "course_id": doc.course_id,
        "document_id": doc.document_id,
        "title": doc.title,
        "version": doc.version,
        "publication_epoch": doc.publication_epoch,
        "content_sha256": doc.content_sha256,
        "representation_id": representation_id,
        "representation_type": representation_type,
        "source_id": src.source_id,
        "content": content,
        "embedding_model": model_name,
    }
    if isinstance(page := src.locator.get("page"), int):
        payload["page"] = page
    return payload


def _post_verify_points(
    points: Sequence[models.ScoredPoint],
    docs: dict[str, SnapshotDocument],
    sources: dict[tuple[str, int, str], SnapshotSource],
) -> tuple[list[tuple[SnapshotSource, float]], int]:
    verified: list[tuple[SnapshotSource, float]] = []
    filtered = 0
    for point in points:
        payload = point.payload or {}
        doc_id = payload.get("document_id")
        version = payload.get("version")
        source_id = payload.get("source_id")
        doc = docs.get(doc_id) if isinstance(doc_id, str) else None
        src = (
            sources.get((doc_id, version, source_id))
            if isinstance(doc_id, str) and isinstance(version, int) and isinstance(source_id, str)
            else None
        )
        if (
            doc is None
            or src is None
            or payload.get("source_origin") != _SOURCE_ORIGIN
            or payload.get("course_id") != doc.course_id
            or payload.get("version") != doc.version
            or payload.get("publication_epoch") != doc.publication_epoch
            or payload.get("content_sha256") != doc.content_sha256
        ):
            filtered += 1
            continue
        verified.append((src, float(point.score)))
    return verified, filtered


def _should_trigger_visual_search(
    question: str,
    s0_status: str,
    s0_pool: Sequence[StructureCandidate],
    docs: dict[str, SnapshotDocument],
    sources: dict[tuple[str, int, str], SnapshotSource],
    *,
    vector_text_count: int = 0,
) -> bool:
    if (
        visual_query_intent(question)
        or _VISUAL_SLIDE_QUERY.search(question)
        or re.search(r"幻灯片|\bslide\b", question, re.IGNORECASE)
    ):
        return True
    has_sparse_visual = any(
        src.document_id in docs
        and _is_visual_index_target(docs[src.document_id], src)
        and len(readable_source_text(src).strip()) < _LOW_TEXT_VISUAL_LIMIT
        for src in sources.values()
    )
    if not has_sparse_visual:
        return False
    lexical_low = s0_status in {"empty", "no_text_anchor"} or (
        len([item for item in s0_pool if item.match_kind != "browse"]) < 2
    )
    return lexical_low and vector_text_count < 2


def _vector_candidate(
    source: SnapshotSource,
    docs: dict[str, SnapshotDocument],
    reason_code: str,
    *,
    score: float,
) -> StructureCandidate:
    candidate = _structural_candidate(source, source, reason_code, docs.values())
    return candidate.model_copy(update={"score": round(score, 6)})
