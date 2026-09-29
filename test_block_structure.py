"""验证 ContentBlock 结构是否正确。"""

from app.schemas.document import ContentBlock, BlockKind

# 测试 PDF ContentBlock
pdf_block = ContentBlock(
    block_id="block_0",
    document_id="doc_123",
    kind=BlockKind.TEXT,
    text="这是测试内容",
    section_path=["Chapter 1", "Section 1.1"],
    cell_id=None,
)

print("✅ PDF ContentBlock 创建成功")
print(f"   block_id: {pdf_block.block_id}")
print(f"   kind: {pdf_block.kind}")

# 测试 PPTX ContentBlock
pptx_block = ContentBlock(
    block_id="slide_1",
    document_id="doc_456",
    kind=BlockKind.TEXT,
    text="幻灯片内容",
    section_path=["Introduction"],
    cell_id=None,
)

print("\n✅ PPTX ContentBlock 创建成功")
print(f"   block_id: {pptx_block.block_id}")
print(f"   kind: {pptx_block.kind}")

print("\n✅ ContentBlock 结构验证通过")
