"""SlideDeck 页边界、长页切分与追溯测试。"""

from app.rag.slide_chunker import SlideChunker
from app.schemas.document import (
    Slide,
    SlideDeck,
    SlideElement,
    SlideElementKind,
    SlideSourceFormat,
)


def _slide(no: int, title: str, text: str) -> Slide:
    slide_id = f"doc-slide-{no:04d}"
    return Slide(
        slide_id=slide_id,
        document_id="doc",
        slide_no=no,
        title=title,
        elements=[
            SlideElement(
                element_id=f"{slide_id}-element-1",
                kind=SlideElementKind.TITLE,
                text=title,
            ),
            SlideElement(
                element_id=f"{slide_id}-element-2",
                kind=SlideElementKind.TEXT,
                text=text,
            ),
        ],
    )


def test_slide_chunker_never_merges_pages_and_links_chunks() -> None:
    deck = SlideDeck(
        document_id="doc",
        source_path="course/week1.pdf",
        source_format=SlideSourceFormat.PDF,
        title="Week 1",
        slides=[
            _slide(1, "Introduction", "Short overview."),
            _slide(2, "Introduction", "Another short slide."),
            _slide(3, "Model", " ".join(f"term{i}" for i in range(180))),
        ],
    )
    chunker = SlideChunker(max_tokens=100, overlap_tokens=10)

    result = chunker.chunk(deck)
    repeated = chunker.chunk(deck)

    assert [chunk.chunk_id for chunk in result.chunks] == [
        chunk.chunk_id for chunk in repeated.chunks
    ]
    assert len(result.sections) == 2
    assert result.sections[0].slide_ids == ["doc-slide-0001", "doc-slide-0002"]
    assert len([chunk for chunk in result.chunks if chunk.slide_no == 1]) == 1
    assert len([chunk for chunk in result.chunks if chunk.slide_no == 2]) == 1
    assert len([chunk for chunk in result.chunks if chunk.slide_no == 3]) > 1
    assert all(chunk.token_count <= 100 for chunk in result.chunks)
    assert result.chunks[0].next_chunk_id == result.chunks[1].chunk_id
    assert result.chunks[-1].previous_chunk_id == result.chunks[-2].chunk_id

