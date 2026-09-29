"""测试 PDF/PPTX 解析和切块质量。"""

import sys
from pathlib import Path

from app.rag.parsers.pdf import PDFParser
from app.rag.parsers.pptx import PPTXParser
from app.rag.pdf_chunker import PDFChunker
from app.rag.pptx_chunker import PPTXChunker


def test_pdf_parsing(pdf_path: Path):
    """测试 PDF 解析和切块。"""
    print(f"\n{'='*80}")
    print(f"测试 PDF: {pdf_path.name}")
    print(f"{'='*80}")

    # 解析
    parser = PDFParser()
    try:
        document = parser.parse(pdf_path)
        print(f"\n✅ 解析成功")
        print(f"   - 文档ID: {document.document_id}")
        print(f"   - 标题: {document.title}")
        print(f"   - 文件类型: {document.file_type}")
        print(f"   - Blocks数量: {len(document.blocks)}")

        # 显示前3个 blocks
        print(f"\n📄 前3个 Blocks:")
        for i, block in enumerate(document.blocks[:3]):
            print(f"\n   Block {i+1}:")
            print(f"     - ID: {block.block_id}")
            print(f"     - 章节: {' > '.join(block.section_path) if block.section_path else '无'}")
            print(f"     - 内容预览: {block.display_text[:100]}...")

        # 切块
        chunker = PDFChunker()
        chunks = chunker.chunk(document)
        print(f"\n✅ 切块成功")
        print(f"   - Chunks数量: {len(chunks)}")

        # 统计
        token_counts = [chunk.token_count for chunk in chunks]
        print(f"\n📊 切块统计:")
        print(f"   - 平均tokens: {sum(token_counts) / len(token_counts):.0f}")
        print(f"   - 最小tokens: {min(token_counts)}")
        print(f"   - 最大tokens: {max(token_counts)}")
        print(f"   - 超过450 tokens的chunk: {sum(1 for t in token_counts if t > 450)}")

        # 显示前3个 chunks
        print(f"\n📦 前3个 Chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n   Chunk {i+1}:")
            print(f"     - ID: {chunk.chunk_id}")
            print(f"     - 章节: {' > '.join(chunk.section_path)}")
            print(f"     - Tokens: {chunk.token_count}")
            print(f"     - 内容预览:")
            print(f"       {chunk.embedding_text[:200]}...")

        return True

    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pptx_parsing(pptx_path: Path):
    """测试 PPTX 解析和切块。"""
    print(f"\n{'='*80}")
    print(f"测试 PPTX: {pptx_path.name}")
    print(f"{'='*80}")

    # 解析
    parser = PPTXParser()
    try:
        document = parser.parse(pptx_path)
        print(f"\n✅ 解析成功")
        print(f"   - 文档ID: {document.document_id}")
        print(f"   - 标题: {document.title}")
        print(f"   - 文件类型: {document.file_type}")
        print(f"   - 幻灯片数量: {len(document.blocks)}")

        # 显示前3张幻灯片
        print(f"\n📄 前3张幻灯片:")
        for i, block in enumerate(document.blocks[:3]):
            print(f"\n   幻灯片 {i+1}:")
            print(f"     - ID: {block.block_id}")
            print(f"     - 标题: {' > '.join(block.section_path) if block.section_path else '无'}")
            print(f"     - 内容预览: {block.display_text[:100]}...")

        # 切块
        chunker = PPTXChunker()
        chunks = chunker.chunk(document)
        print(f"\n✅ 切块成功")
        print(f"   - Chunks数量: {len(chunks)}")

        # 统计
        token_counts = [chunk.token_count for chunk in chunks]
        print(f"\n📊 切块统计:")
        print(f"   - 平均tokens: {sum(token_counts) / len(token_counts):.0f}")
        print(f"   - 最小tokens: {min(token_counts)}")
        print(f"   - 最大tokens: {max(token_counts)}")
        print(f"   - 超过450 tokens的chunk: {sum(1 for t in token_counts if t > 450)}")

        # 显示前3个 chunks
        print(f"\n📦 前3个 Chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n   Chunk {i+1}:")
            print(f"     - ID: {chunk.chunk_id}")
            print(f"     - 标题: {' > '.join(chunk.section_path)}")
            print(f"     - Tokens: {chunk.token_count}")
            print(f"     - 内容预览:")
            print(f"       {chunk.embedding_text[:200]}...")

        return True

    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """测试多个 PDF/PPTX 文件的切分质量。"""
    # 测试文件列表
    test_files = [
        # PDF 文件
        "/Users/Link/Course/CS5494 Topics in Generative AI /CS5494-week1.pdf",
        "/Users/Link/Course/CS5495 X AI/Lecture2/Lecture2a.pdf",
        "/Users/Link/Course/CS5489 Machine learning/Lecture1/Lecture1 Intro.pdf",
        # PPTX 文件
        "/Users/Link/Course/CS5489 Machine learning/CS5489_Machine_Learning/课程笔记.pptx",
    ]

    results = []
    for file_path_str in test_files:
        file_path = Path(file_path_str)
        if not file_path.exists():
            print(f"\n⚠️  文件不存在，跳过: {file_path}")
            continue

        if file_path.suffix.lower() == ".pdf":
            success = test_pdf_parsing(file_path)
        elif file_path.suffix.lower() in [".ppt", ".pptx"]:
            success = test_pptx_parsing(file_path)
        else:
            print(f"\n⚠️  不支持的文件类型: {file_path}")
            continue

        results.append((file_path.name, success))

    # 总结
    print(f"\n{'='*80}")
    print("📊 测试总结")
    print(f"{'='*80}")
    for name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"   {status} - {name}")
    print()


if __name__ == "__main__":
    main()
