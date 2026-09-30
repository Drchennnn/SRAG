from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from app.config import Settings
from app.evaluation.slide_retrieval import rank_pages
from app.rag.slide_indexer import SlideIndexer
from app.schemas.document import (
    Slide,
    SlideChunk,
    SlideChunkingResult,
    SlideDeck,
    SlideSection,
    SlideSourceFormat,
)
from app.schemas.embedding import EmbeddedSlideImage, EmbeddedSlideText, SlideEmbeddingResult
from app.schemas.rag import JobState, SlideIndexRequest
from app.stores.qdrant import QdrantStore
from app.stores.sqlite import MetadataStore


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")
async def test_slide_import_keeps_page_provenance_and_versions(tmp_path: Path) -> None:
    source = tmp_path / "lecture.pdf"
    source.write_bytes(b"small test slide")
    document_id = f"document_{hashlib.sha256(source.read_bytes()).hexdigest()[:24]}"
    bundle = tmp_path / "bundle"
    (bundle / "slides").mkdir(parents=True)
    (bundle / "slides" / "slide-0001.png").write_bytes(b"image")
    (bundle / "slides" / "slide-0002.png").write_bytes(b"image-only")
    slide = Slide(
        slide_id="slide-1",
        document_id=document_id,
        slide_no=1,
        title="Example",
        page_image_path="slides/slide-0001.png",
    )
    deck = SlideDeck(
        document_id=document_id,
        source_path=source.as_posix(),
        source_format=SlideSourceFormat.PDF,
        title="Lecture",
        slides=[
            slide,
            Slide(
                slide_id="slide-2",
                document_id=document_id,
                slide_no=2,
                page_image_path="slides/slide-0002.png",
            ),
        ],
    )
    chunk = SlideChunk(
        chunk_id="chunk-1",
        section_id="section-1",
        document_id=document_id,
        slide_id="slide-1",
        slide_no=1,
        title="Example",
        display_text="An example slide",
        embedding_text="An example slide",
        token_count=3,
    )
    chunking = SlideChunkingResult(
        document_id=document_id,
        outline_text="Example",
        sections=[
            SlideSection(
                section_id="section-1",
                document_id=document_id,
                title="Example",
                slide_ids=["slide-1", "slide-2"],
                chunk_ids=["chunk-1", "chunk-2"],
            )
        ],
        chunks=[
            chunk,
            SlideChunk(
                chunk_id="chunk-2",
                section_id="section-1",
                document_id=document_id,
                slide_id="slide-2",
                slide_no=2,
                display_text="",
                embedding_text="",
                token_count=0,
            ),
        ],
    )
    embeddings = SlideEmbeddingResult(
        document_id=document_id,
        source_path=source.as_posix(),
        text_model="test-text",
        text_dimension=3,
        visual_model="test-visual",
        visual_dimension=2,
        skipped_text_chunk_ids=["chunk-2"],
        text_representations=[
            EmbeddedSlideText(
                chunk_id="chunk-1",
                slide_id="slide-1",
                slide_no=1,
                text="An example slide",
                dense=[1, 0, 0],
            )
        ],
        visual_representations=[
            EmbeddedSlideImage(
                slide_id="slide-1",
                slide_no=1,
                chunk_ids=["chunk-1"],
                image_path="slides/slide-0001.png",
                context_text="Example",
                dense=[1, 0],
            ),
            EmbeddedSlideImage(
                slide_id="slide-2",
                slide_no=2,
                chunk_ids=["chunk-2"],
                image_path="slides/slide-0002.png",
                context_text="Lecture",
                dense=[0, 1],
            ),
        ],
    )
    (bundle / "deck.json").write_text(deck.model_dump_json())
    (bundle / "chunks.json").write_text(chunking.model_dump_json())
    embeddings_path = tmp_path / "vectors.json"
    embeddings_path.write_text(embeddings.model_dump_json())
    settings = Settings(
        data_dir=tmp_path / "data",
        sqlite_path=tmp_path / "metadata.sqlite3",
        qdrant_collection="test_slides",
        dashscope_embedding_dimension=3,
        dashscope_visual_embedding_dimension=2,
        dashscope_embedding_model="test-text",
        dashscope_visual_embedding_model="test-visual",
    )
    metadata = MetadataStore(settings)
    metadata.initialize()
    client = AsyncQdrantClient(":memory:")
    vectors = QdrantStore(settings, client=client)
    indexer = SlideIndexer(settings, metadata, vectors)
    request = SlideIndexRequest(
        course_id="CS5187", source_path=source, bundle_dir=bundle, embeddings_path=embeddings_path
    )
    try:
        first = await indexer.index(request)
        assert first.state == JobState.COMPLETED and first.indexed_files == 1
        assert (await indexer.index(request)).skipped_files == 1
        visible = metadata.visible_versions(course_ids=["CS5187"])
        assert len(visible) == 1 and visible[0].version == 1
        points = await vectors.search_text_dense(
            [1, 0, 0], limit=5, visible_versions=[(document_id, 1)]
        )
        assert rank_pages(points) == [(document_id, 1)]
        assert rank_pages(
            await vectors.search_visual_fusion([0, 1], limit=5, visible_versions=[(document_id, 1)])
        )[0] == (document_id, 2)
        assert Path(
            (
                await vectors.search_visual_fusion(
                    [1, 0], limit=5, visible_versions=[(document_id, 1)]
                )
            )[0].payload["image_path"]
        ).is_file()
        second = await indexer.index(request.model_copy(update={"force": True}))
        assert second.state == JobState.COMPLETED
        assert metadata.visible_versions(course_ids=["CS5187"])[0].version == 2
        assert not await vectors.search_text_dense(
            [1, 0, 0], limit=5, visible_versions=[(document_id, 1)]
        )
    finally:
        await client.close()
