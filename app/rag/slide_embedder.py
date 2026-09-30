"""从 SlideDeck 预处理包生成页级文本与整页视觉向量。"""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from app.clients.dashscope import DashScopeClient
from app.config import Settings, get_settings
from app.rag.slide_chunker import slide_embedding_text
from app.schemas.document import Slide, SlideChunk, SlideChunkingResult, SlideDeck
from app.schemas.embedding import EmbeddedSlideImage, EmbeddedSlideText, SlideEmbeddingResult


class SlideEmbedder:
    """复用现有 Qwen 客户端，保持文本 Chunk 和整页截图的独立表示。"""

    def __init__(self, client: DashScopeClient) -> None:
        self.client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> SlideEmbedder:
        return cls(DashScopeClient(settings))

    async def embed_bundle(
        self,
        bundle_dir: Path,
        *,
        slide_numbers: Sequence[int] | None = None,
        include_visual: bool = True,
    ) -> SlideEmbeddingResult:
        """读取便携包；默认给每页截图向量化，不使用视觉候选标记筛页。"""

        root = bundle_dir.expanduser().resolve()
        deck = SlideDeck.model_validate_json((root / "deck.json").read_text(encoding="utf-8"))
        chunking = SlideChunkingResult.model_validate_json(
            (root / "chunks.json").read_text(encoding="utf-8")
        )
        if deck.document_id != chunking.document_id:
            raise ValueError("deck.json 与 chunks.json 的 document_id 不一致")

        slides_by_id = {slide.slide_id: slide for slide in deck.slides}
        if len(slides_by_id) != len(deck.slides):
            raise ValueError("SlideDeck 存在重复 slide_id")
        available_numbers = {slide.slide_no for slide in deck.slides}
        selected_numbers = set(slide_numbers) if slide_numbers is not None else available_numbers
        if not selected_numbers or not selected_numbers <= available_numbers:
            raise ValueError("所选页码为空或不在 SlideDeck 中")

        chunks_by_slide: dict[str, list[SlideChunk]] = defaultdict(list)
        selected_text: list[tuple[SlideChunk, str]] = []
        skipped_text: list[str] = []
        for chunk in chunking.chunks:
            slide = slides_by_id.get(chunk.slide_id)
            if slide is None or chunk.slide_no != slide.slide_no:
                raise ValueError(f"Chunk 的幻灯片来源无效：{chunk.chunk_id}")
            chunks_by_slide[chunk.slide_id].append(chunk)
            if chunk.slide_no not in selected_numbers:
                continue
            # 旧 Colab 包的 embedding_text 带文件名、格式、页码；在此重建语义输入。
            semantic_text = slide_embedding_text(chunk.title, chunk.display_text)
            if semantic_text:
                selected_text.append((chunk, semantic_text))
            else:
                skipped_text.append(chunk.chunk_id)

        selected_slides = [
            slide for slide in deck.slides if slide.slide_no in selected_numbers
        ]
        if any(not chunks_by_slide[slide.slide_id] for slide in selected_slides):
            raise ValueError("所选幻灯片缺少文本 Chunk")

        visual_inputs: list[tuple[Slide, Path, str, str]] = []
        if include_visual:
            for slide in selected_slides:
                if not slide.page_image_path:
                    raise ValueError(f"第 {slide.slide_no} 页没有整页截图")
                image_path = _bundle_file(root, slide.page_image_path)
                media_type = mimetypes.guess_type(image_path.name)[0]
                if media_type is None:
                    raise ValueError(f"无法识别截图类型：{slide.page_image_path}")
                first_chunk = chunks_by_slide[slide.slide_id][0]
                context = (
                    slide.title
                    or first_chunk.display_text.strip()[:500]
                    or deck.title
                )
                visual_inputs.append((slide, image_path, media_type, context))

        text_representations: list[EmbeddedSlideText] = []
        request_ids: list[str] = []
        text_total_tokens = 0
        if selected_text:
            batch = await self.client.embed_documents([text for _, text in selected_text])
            vectors = {vector.index: vector for vector in batch.vectors}
            if set(vectors) != set(range(len(selected_text))):
                raise RuntimeError("文本 Embedding 返回数量或索引与输入不一致")
            for index, (chunk, semantic_text) in enumerate(selected_text):
                vector = vectors[index]
                if len(vector.dense) != self.client.embedding_dimension:
                    raise RuntimeError(f"第 {chunk.slide_no} 页文本向量维度不符")
                text_representations.append(
                    EmbeddedSlideText(
                        chunk_id=chunk.chunk_id,
                        slide_id=chunk.slide_id,
                        slide_no=chunk.slide_no,
                        text=semantic_text,
                        dense=vector.dense,
                        sparse=vector.sparse,
                    )
                )
            text_total_tokens = batch.total_tokens
            request_ids.extend(batch.request_ids)

        visual_representations: list[EmbeddedSlideImage] = []
        visual_total_tokens = 0
        for slide, image_path, media_type, context in visual_inputs:
            batch = await self.client.embed_visual_fusion(
                text=context,
                image_path=image_path,
                media_type=media_type,
            )
            if (
                len(batch.vectors) != 1
                or len(batch.vectors[0].dense) != self.client.visual_embedding_dimension
            ):
                raise RuntimeError(f"第 {slide.slide_no} 页视觉向量数量或维度不符")
            visual_representations.append(
                EmbeddedSlideImage(
                    slide_id=slide.slide_id,
                    slide_no=slide.slide_no,
                    chunk_ids=[chunk.chunk_id for chunk in chunks_by_slide[slide.slide_id]],
                    image_path=slide.page_image_path,
                    context_text=context,
                    dense=batch.vectors[0].dense,
                )
            )
            visual_total_tokens += batch.total_tokens
            request_ids.extend(batch.request_ids)

        return SlideEmbeddingResult(
            document_id=deck.document_id,
            source_path=deck.source_path,
            text_model=self.client.embedding_model,
            text_dimension=self.client.embedding_dimension,
            visual_model=self.client.visual_embedding_model if include_visual else None,
            visual_dimension=self.client.visual_embedding_dimension if include_visual else None,
            text_total_tokens=text_total_tokens,
            visual_total_tokens=visual_total_tokens,
            request_ids=request_ids,
            skipped_text_chunk_ids=skipped_text,
            text_representations=text_representations,
            visual_representations=visual_representations,
        )


def _bundle_file(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError(f"预处理包中的截图路径无效：{relative_path}")
    return candidate


def save_slide_embeddings(result: SlideEmbeddingResult, output_path: Path) -> Path:
    """原子写入向量包，供后续 Qdrant 索引和检索实验复用。"""

    output = output_path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="向量化一个 SlideDeck 预处理包")
    parser.add_argument("bundle", type=Path, help="包含 deck.json 和 chunks.json 的目录")
    parser.add_argument("--output", type=Path, help="输出 JSON；默认写入 data/embeddings")
    parser.add_argument("--pages", help="可选的逗号分隔页码，例如 1,14,22")
    parser.add_argument("--text-only", action="store_true", help="只向量化页级文本")
    parser.add_argument("--force", action="store_true", help="覆盖已有结果并重新调用 API")
    args = parser.parse_args()
    settings = get_settings()
    pages = [int(value) for value in args.pages.split(",")] if args.pages else None
    suffix = "all" if pages is None else "pages-" + "-".join(str(n) for n in pages)
    output = args.output or settings.data_dir / "embeddings" / f"{args.bundle.name}-{suffix}.json"
    if output.exists() and not args.force:
        parser.error(f"输出已存在，避免重复调用 API：{output}；如需重算请加 --force")
    result = asyncio.run(
        SlideEmbedder.from_settings(settings).embed_bundle(
            args.bundle,
            slide_numbers=pages,
            include_visual=not args.text_only,
        )
    )
    saved = save_slide_embeddings(result, output)
    print(
        f"文档 {result.document_id}：文本 {len(result.text_representations)} 条，"
        f"整页视觉 {len(result.visual_representations)} 条，"
        f"用量 text={result.text_total_tokens}, visual={result.visual_total_tokens} tokens"
    )
    print(f"向量包：{saved}")


if __name__ == "__main__":
    main()
