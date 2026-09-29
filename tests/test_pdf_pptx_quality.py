"""测试 PDF/PPTX 解析和切块质量。"""

from pathlib import Path

from app.rag.parsers.pdf import PDFParser
from app.rag.parsers.pptx import PPTXParser
from app.rag.pdf_chunker import PDFChunker
from app.rag.pptx_chunker import PPTXChunker


def analyze_chunks(chunks, file_type):
    """分析 chunks 的质量。"""
    if not chunks:
        print("   ⚠️  没有生成 chunks")
        return

    tokens = [c.token_count for c in chunks]

    print(f"\n📊 切分质量统计:")
    print(f"   总chunks数: {len(chunks)}")
    print(f"   Token统计:")
    print(f"     - 平均: {sum(tokens)/len(tokens):.0f}")
    print(f"     - 最小: {min(tokens)}")
    print(f"     - 最大: {max(tokens)}")
    print(f"     - 中位数: {sorted(tokens)[len(tokens)//2]}")

    # 分布
    small = sum(1 for t in tokens if t < 120)
    good = sum(1 for t in tokens if 120 <= t <= 450)
    large = sum(1 for t in tokens if t > 450)

    print(f"   Token分布:")
    print(f"     - <120 (太小): {small} ({small/len(tokens)*100:.1f}%)")
    print(f"     - 120-450 (正常): {good} ({good/len(tokens)*100:.1f}%)")
    print(f"     - >450 (太大): {large} ({large/len(tokens)*100:.1f}%)")

    # 显示示例
    print(f"\n📦 示例 Chunk (第1个):")
    chunk = chunks[0]
    print(f"   ID: {chunk.chunk_id}")
    print(f"   章节: {' > '.join(chunk.section_path) if chunk.section_path else '无'}")
    print(f"   Tokens: {chunk.token_count}")
    print(f"   内容预览:")
    lines = chunk.embedding_text.split('\n')[:10]
    for line in lines:
        print(f"     {line}")
    if len(chunk.embedding_text.split('\n')) > 10:
        print(f"     ... (共 {len(chunk.embedding_text.split('\n'))} 行)")


def test_pdf_quality():
    """测试 PDF 切分质量。"""
    pdf_path = Path("/Users/Link/Course/CS5495 X AI/Lecture2/Lecture2a.pdf")

    if not pdf_path.exists():
        print(f"❌ PDF文件不存在")
        return False

    print(f"\n{'='*70}")
    print(f"测试 PDF: {pdf_path.name}")
    print(f"文件大小: {pdf_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"{'='*70}")

    try:
        # 解析
        print("\n⏳ 解析中...")
        parser = PDFParser()
        document = parser.parse(pdf_path)

        print(f"✅ 解析成功")
        print(f"   - 文档标题: {document.title}")
        print(f"   - Blocks数: {len(document.blocks)}")

        # 切块
        print("\n⏳ 切块中...")
        chunker = PDFChunker()
        chunks = chunker.chunk(document)

        print(f"✅ 切块成功")
        analyze_chunks(chunks, "PDF")

        return True

    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pptx_quality():
    """测试 PPTX 切分质量。"""
    pptx_path = Path("/Users/Link/Course/CS5489 Machine learning/CS5489_Machine_Learning/课程笔记.pptx")

    if not pptx_path.exists():
        print(f"❌ PPTX文件不存在")
        return False

    print(f"\n{'='*70}")
    print(f"测试 PPTX: {pptx_path.name}")
    print(f"文件大小: {pptx_path.stat().st_size / 1024:.0f} KB")
    print(f"{'='*70}")

    try:
        # 解析
        print("\n⏳ 解析中...")
        parser = PPTXParser()
        document = parser.parse(pptx_path)

        print(f"✅ 解析成功")
        print(f"   - 文档标题: {document.title}")
        print(f"   - 幻灯片数: {len(document.blocks)}")

        # 切块
        print("\n⏳ 切块中...")
        chunker = PPTXChunker()
        chunks = chunker.chunk(document)

        print(f"✅ 切块成功")
        analyze_chunks(chunks, "PPTX")

        return True

    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🧪 PDF/PPTX 切分质量测试\n")

    pdf_ok = test_pdf_quality()
    pptx_ok = test_pptx_quality()

    print(f"\n{'='*70}")
    print("📊 测试总结")
    print(f"{'='*70}")
    print(f"   PDF:  {'✅ 通过' if pdf_ok else '❌ 失败'}")
    print(f"   PPTX: {'✅ 通过' if pptx_ok else '❌ 失败'}")
    print()
