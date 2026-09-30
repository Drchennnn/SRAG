"""便携 SlideDeck 包的页级文本与整页截图向量化测试。"""

import json
from pathlib import Path

import httpx
import pytest

from app.clients.dashscope import DashScopeClient
from app.config import Settings
from app.rag.slide_chunker import SlideChunker
from app.rag.slide_embedder import SlideEmbedder, save_slide_embeddings
from app.schemas.document import (
    Slide,
    SlideDeck,
    SlideElement,
    SlideElementKind,
    SlideSourceFormat,
)


def _bundle(root: Path) -> Path:
    slides_dir = root / "slides"
    slides_dir.mkdir(parents=True)
    slides = [
        Slide(
            slide_id=f"doc-slide-{number:04d}",
            document_id="doc",
            slide_no=number,
            title=title,
            page_image_path=f"slides/slide-{number:04d}.png",
            visual_embedding_candidate=False,
            elements=[
                SlideElement(
                    element_id=f"element-{number}",
                    kind=SlideElementKind.TEXT,
                    text=body,
                )
            ],
        )
        for number, title, body in (
            (1, "Face Recognition", ""),
            (2, "OCR", "Read characters from images."),
        )
    ]
    for number in (1, 2):
        (slides_dir / f"slide-{number:04d}.png").write_bytes(b"fake-png")
    deck = SlideDeck(
        document_id="doc",
        source_path="/content/Lecture1-1.pdf",
        source_format=SlideSourceFormat.PDF,
        title="Lecture1-1",
        slides=slides,
    )
    chunks = SlideChunker().chunk(deck)
    # 模拟已导出的旧 Colab 包，其 embedding_text 仍带文件名和页码。
    old_chunks = [
        chunk.model_copy(update={"embedding_text": f"文件：Lecture1-1.pdf\n页码：{chunk.slide_no}"})
        for chunk in chunks.chunks
    ]
    (root / "deck.json").write_text(deck.model_dump_json(), encoding="utf-8")
    (root / "chunks.json").write_text(
        chunks.model_copy(update={"chunks": old_chunks}).model_dump_json(), encoding="utf-8"
    )
    return root


@pytest.mark.asyncio
async def test_embedder_uses_semantic_text_and_all_page_images(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if "multimodal-embedding" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "request_id": "visual-request",
                    "output": {
                        "embeddings": [
                            {"index": 0, "embedding": [0.2, 0.3], "type": "fusion"}
                        ]
                    },
                    "usage": {"total_tokens": 10},
                },
            )
        return httpx.Response(
            200,
            json={
                "request_id": "text-request",
                "output": {
                    "embeddings": [
                        {
                            "text_index": i,
                            "embedding": [0.1, 0.2, 0.3],
                            "sparse_embedding": [{"index": i, "value": 0.8}],
                        }
                        for i in range(len(payload["input"]["texts"]))
                    ]
                },
                "usage": {"total_tokens": 7},
            },
        )

    client = DashScopeClient(
        Settings(
            dashscope_api_key="test-key",
            dashscope_embedding_dimension=3,
            dashscope_visual_embedding_dimension=2,
            dashscope_max_retries=0,
        ),
        transport=httpx.MockTransport(handler),
    )
    result = await SlideEmbedder(client).embed_bundle(bundle)

    assert len(result.text_representations) == 2
    assert len(result.visual_representations) == 2
    assert [item.slide_no for item in result.visual_representations] == [1, 2]
    assert requests[0]["input"]["texts"] == [
        "Face Recognition",
        "OCR\n\n[正文]\nRead characters from images.",
    ]
    assert result.text_representations[0].sparse[0].value == 0.8
    assert result.visual_representations[0].chunk_ids == [result.text_representations[0].chunk_id]
    assert result.text_total_tokens == 7
    assert result.visual_total_tokens == 20
    assert result.request_ids == ["text-request", "visual-request", "visual-request"]
    path = save_slide_embeddings(result, tmp_path / "vectors.json")
    assert json.loads(path.read_text())["visual_representations"][1]["slide_no"] == 2


@pytest.mark.asyncio
async def test_embedder_checks_all_screenshots_before_api_calls(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    (bundle / "slides" / "slide-0002.png").unlink()
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    client = DashScopeClient(
        Settings(dashscope_api_key="test-key"), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ValueError, match="截图路径无效"):
        await SlideEmbedder(client).embed_bundle(bundle)
    assert not requests
