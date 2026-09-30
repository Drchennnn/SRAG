"""将 Colab 预处理包和向量包登记到 SQLite 与 Qdrant。"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from uuid import uuid4

from app.config import Settings
from app.schemas.document import ParsedDocument, SlideChunkingResult, SlideDeck
from app.schemas.embedding import SlideEmbeddingResult
from app.schemas.rag import IndexStatus, JobState, SlideIndexRequest
from app.stores.qdrant import QdrantStore
from app.stores.sqlite import MetadataStore, VersionPlan


class SlideIndexer:
    def __init__(self, settings: Settings, metadata: MetadataStore, vectors: QdrantStore) -> None:
        self.settings = settings
        self.metadata = metadata
        self.vectors = vectors

    @classmethod
    def from_settings(cls, settings: Settings) -> SlideIndexer:
        metadata = MetadataStore(settings)
        metadata.initialize()
        return cls(settings, metadata, QdrantStore(settings))

    async def index(self, request: SlideIndexRequest) -> IndexStatus:
        job_id = f"index_{uuid4().hex}"
        self.metadata.create_job(job_id)
        plan: VersionPlan | None = None
        managed_dir: Path | None = None
        source = request.source_path.expanduser().resolve()
        try:
            if source.suffix.lower() not in {".pdf", ".pptx"} or not source.is_file():
                raise ValueError(f"源文件不存在或格式不支持：{source}")
            bundle = request.bundle_dir.expanduser().resolve()
            deck = SlideDeck.model_validate_json((bundle / "deck.json").read_text(encoding="utf-8"))
            chunking = SlideChunkingResult.model_validate_json(
                (bundle / "chunks.json").read_text(encoding="utf-8")
            )
            embeddings = SlideEmbeddingResult.model_validate_json(
                request.embeddings_path.expanduser().resolve().read_text(encoding="utf-8")
            )
            content_hash = _file_hash(source)
            if (
                deck.document_id != f"document_{content_hash[:24]}"
                or chunking.document_id != deck.document_id
                or embeddings.document_id != deck.document_id
                or deck.source_format.value != source.suffix.lower().lstrip(".")
                or Path(deck.source_path).name != source.name
                or Path(embeddings.source_path).name != source.name
            ):
                raise ValueError("源文件、预处理包和向量包不属于同一文档")
            chunks = {chunk.chunk_id: chunk for chunk in chunking.chunks}
            slides = {slide.slide_id: slide for slide in deck.slides}
            embedded_chunk_ids = [item.chunk_id for item in embeddings.text_representations]
            skipped_chunk_ids = embeddings.skipped_text_chunk_ids
            embedded_slide_ids = [item.slide_id for item in embeddings.visual_representations]
            if (
                not chunks
                or len(chunks) != len(chunking.chunks)
                or len(slides) != len(deck.slides)
                or len(embedded_chunk_ids) != len(set(embedded_chunk_ids))
                or len(skipped_chunk_ids) != len(set(skipped_chunk_ids))
                or set(embedded_chunk_ids) & set(skipped_chunk_ids)
                or set(embedded_chunk_ids) | set(skipped_chunk_ids) != set(chunks)
                or len(embedded_slide_ids) != len(set(embedded_slide_ids))
                or set(embedded_slide_ids) != set(slides)
            ):
                raise ValueError("向量包必须覆盖全部 Chunk 和截图且 ID 唯一")
            for item in embeddings.visual_representations:
                if item.slide_no != slides[item.slide_id].slide_no or not item.chunk_ids:
                    raise ValueError("视觉向量页码或 Chunk 映射无效")
            deck = deck.model_copy(update={"source_path": source.as_posix()})
            plan = self.metadata.reserve_version(
                ParsedDocument(
                    document_id=deck.document_id,
                    source_path=deck.source_path,
                    title=deck.title,
                    blocks=[],
                ),
                course_id=request.course_id,
                content_hash=content_hash,
                release_at=request.release_at,
                document_type=deck.source_format.value,
                force=request.force,
            )
            if not plan.should_index:
                status = IndexStatus(job_id=job_id, state=JobState.COMPLETED, skipped_files=1)
                self.metadata.finish_job(status)
                return status
            managed_dir = (
                self.settings.data_dir / "assets" / "slides" / deck.document_id / str(plan.version)
            ).resolve()
            if managed_dir.exists():
                shutil.rmtree(managed_dir)
            paths = {asset.path for slide in deck.slides for asset in slide.assets}
            paths.update(slide.page_image_path for slide in deck.slides if slide.page_image_path)
            asset_paths: dict[str, str] = {}
            for relative in paths:
                source_asset = (bundle / relative).resolve()
                if not source_asset.is_relative_to(bundle) or not source_asset.is_file():
                    raise ValueError(f"预处理包资源缺失或越界：{relative}")
                target = (managed_dir / relative).resolve()
                if not target.is_relative_to(managed_dir):
                    raise ValueError(f"受管资源路径越界：{relative}")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_asset, target)
                asset_paths[relative] = target.as_posix()
            self.metadata.save_slide_structure(
                deck, chunking, version=plan.version, asset_paths=asset_paths
            )
            await self.vectors.upsert_slide_embeddings(
                deck,
                chunking,
                embeddings,
                course_id=request.course_id,
                source_path=deck.source_path,
                asset_paths=asset_paths,
                version=plan.version,
                release_at=request.release_at,
            )
            self.metadata.activate_version(deck.document_id, plan.version)
            status = IndexStatus(job_id=job_id, state=JobState.COMPLETED, indexed_files=1)
        except Exception as exc:
            if plan is not None and plan.should_index:
                self.metadata.fail_version(plan.document_id, plan.version, str(exc))
            if managed_dir is not None and managed_dir.exists():
                shutil.rmtree(managed_dir)
            status = IndexStatus(
                job_id=job_id,
                state=JobState.FAILED,
                failed_files=1,
                errors={source.as_posix(): str(exc)},
            )
        if status.state == JobState.COMPLETED and status.indexed_files:
            try:
                await self.vectors.delete_document_versions(
                    deck.document_id, keep_version=plan.version
                )
                self.metadata.prune_superseded_structure(
                    deck.document_id, keep_version=plan.version
                )
            except Exception as exc:
                status.errors[f"{source.as_posix()}#cleanup"] = str(exc)
        self.metadata.finish_job(status)
        return status


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
