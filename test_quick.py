"""快速测试 PDF/PPTX 解析（使用小文件）。"""

from pathlib import Path

from app.rag.parsers.pdf import PDFParser
from app.rag.parsers.pptx import PPTXParser
from app.rag.pdf_chunker import PDFChunker
from app.rag.pptx_chunker import PPTXChunker


def test_small_pdf():
    """测试一个较小的 PDF。"""
    pdf_path = Path("/Users/Link/Course/CS5489 Machine learning/Lecture1/Lecture1 Intro.pdf")

    if not pdf_path.exists():
        print(f"❌ 文件不存在: {pdf_path}")
        return

    print(f"\n{'='*60}")
    print(f"测试 PDF: {pdf_path.name}")
    print(f"文件大小: {pdf_path.stat().st_size / 1024:.1f} KB")
    print(f"{'='*60}")

    try:
        # 解析
        parser = PDFParser()
        print("\n⏳ 正在解析 PDF...")
        document = parser.parse(pdf_path)

        print(f"✅ 解析成功")
        print(f"   - Blocks: {len(document.blocks)}")
        print(f"   - 标题: {document.title}")

        # 切块
        chunker = PDFChunker()
        print("\n⏳ 正在切块...")
        chunks = chunker.chunk(document)

        print(f"✅ 切块成功")
        print(f"   - Chunks: {len(chunks)}")

        # 统计
        tokens = [c.token_count for c in chunks]
        print(f"\n📊 Token 统计:")
        print(f"   - 平均: {sum(tokens)/len(tokens):.0f}")
        print(f"   - 范围: {min(tokens)} - {max(tokens)}")
        print(f"   - 超450: {sum(1 for t in tokens if t > 450)}/{len(tokens)}")

        # 显示第一个 chunk
        if chunks:
            print(f"\n📦 第一个 Chunk:")
            print(f"   章节: {' > '.join(chunks[0].section_path)}")
            print(f"   内容:\n{chunks[0].embedding_text[:300]}...\n")

    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()


def test_pptx():
    """测试 PPTX。"""
    pptx_path = Path("/Users/Link/Course/CS5489 Machine learning/CS5489_Machine_Learning/课程笔记.pptx")

    if not pptx_path.exists():
        print(f"❌ 文件不存在: {pptx_path}")
        return

    print(f"\n{'='*60}")
    print(f"测试 PPTX: {pptx_path.name}")
    print(f"文件大小: {pptx_path.stat().st_size / 1024:.1f} KB")
    print(f"{'='*60}")

    try:
        # 解析
        parser = PPTXParser()
        print("\n⏳ 正在解析 PPTX...")
        document = parser.parse(pptx_path)

        print(f"✅ 解析成功")
        print(f"   - 幻灯片: {len(document.blocks)}")
        print(f"   - 标题: {document.title}")

        # 切块
        chunker = PPTXChunker()
        print("\n⏳ 正在切块...")
        chunks = chunker.chunk(document)

        print(f"✅ 切块成功")
        print(f"   - Chunks: {len(chunks)}")

        # 统计
        tokens = [c.token_count for c in chunks]
        print(f"\n📊 Token 统计:")
        print(f"   - 平均: {sum(tokens)/len(tokens):.0f}")
        print(f"   - 范围: {min(tokens)} - {max(tokens)}")
        print(f"   - 超450: {sum(1 for t in tokens if t > 450)}/{len(tokens)}")

        # 显示前3张幻灯片
        print(f"\n📦 前3张幻灯片:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n   {i+1}. {' > '.join(chunk.section_path)}")
            print(f"      Tokens: {chunk.token_count}")
            print(f"      内容: {chunk.display_text[:100]}...")

    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n🧪 PDF/PPTX 切分质量测试\n")
    test_small_pdf()
    test_pptx()
    print("\n✅ 测试完成\n")
