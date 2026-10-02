"""为指定已入库页面补充可缓存的公式/图表文字，不重新预处理课件。"""

import argparse
import asyncio
from pathlib import Path

from app.clients.dashscope import DashScopeClient
from app.config import get_settings
from app.rag.vision import VisualTextReader
from app.schemas.rag import SearchScope
from app.stores.qdrant import QdrantStore
from app.stores.sqlite import MetadataStore
from tools.chat import _resolve_document


async def run(course: str, selector: str, pages: list[int]) -> None:
    settings = get_settings()
    catalog = MetadataStore(settings)
    document = _resolve_document(catalog, course, selector)
    scope = SearchScope(course_ids=[course], document_ids=[document.document_id])
    read = catalog.read_document(
        document.document_id,
        document.version,
        course_ids=scope.course_ids,
        document_ids=scope.document_ids,
    )
    selected = [hit for hit in read.hits if hit.source.page in pages]
    if set(pages) != {hit.source.page for hit in selected}:
        raise ValueError("指定页码不在当前文件的完整索引中")
    store = QdrantStore(settings)
    reader = VisualTextReader(
        catalog, DashScopeClient(settings), store, settings.data_dir / "visual_text"
    )
    try:
        seen = set()
        for hit in selected:
            if hit.source.page in seen:
                continue
            seen.add(hit.source.page)
            result = await reader.read(hit, scope)
            print(
                f"\n{Path(document.source_path).name} v{document.version} 第 {hit.source.page} 页"
            )
            print(result.content)
    finally:
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="按页补充公式/图表文字并写入 Qdrant")
    parser.add_argument("--course", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--pages", required=True, nargs="+", type=int)
    args = parser.parse_args()
    if len(args.pages) > 8 or min(args.pages) < 1:
        parser.error("每次指定 1–8 个正整数页码，避免意外大量 API 调用")
    asyncio.run(run(args.course, args.file, args.pages))


if __name__ == "__main__":
    main()
