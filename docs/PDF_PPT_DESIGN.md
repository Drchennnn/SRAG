# PDF/PPT 解析与切块设计方案

## 一、现状分析

### 课程文件统计
- **总数**：152 个 PDF/PPTX 文件
- **大小范围**：1-10MB
- **主要类型**：
  - 课程讲义：CS5494-week1.pdf (10MB)
  - Lecture 系列：Lecture1a.pdf, SVM.pdf
  - 作业文档：Assignment1.pdf
  - 课程笔记：课程笔记.pptx

### 与 Notebook 的差异

| 维度 | Notebook (.ipynb) | PDF | PPTX |
|------|-------------------|-----|------|
| **结构** | Cell 为单位，有明确边界 | 页面为单位，内容连续 | 幻灯片为单位，独立性强 |
| **文本** | Markdown + 代码 | 连续段落、标题层级模糊 | 标题+正文，层级明确 |
| **图表** | 嵌入式图片+表格 | 图文混排，位置关系复杂 | 图文独立，位置固定 |
| **代码** | 有执行结果 | 纯文本或截图 | 通常为截图 |
| **导航** | 章节标题（Markdown heading） | 页码、目录（可能缺失） | 幻灯片标题 |

---

## 二、解析工具选择

### PDF 解析

**推荐工具**：[Docling](https://github.com/DS4SD/docling)
- ✅ 完全本地，无模型加载（符合项目要求）
- ✅ 原生支持表格、图片、公式识别
- ✅ 保留文档结构（标题、段落、列表）
- ✅ 输出 Markdown（易于处理）

**备选**：
- PyPDF2 / pdfplumber：简单，但表格识别弱
- PyMuPDF (fitz)：功能强，但需要手动处理布局

### PPTX 解析

**推荐工具**：python-pptx
- ✅ 官方库，稳定
- ✅ 可提取文本、形状、图片
- ✅ 保留幻灯片顺序和层级

---

## 三、Document Outline 设计

### Notebook Outline 的问题

**当前实现**（chunker.py）：
```python
outline_text = f"""
文件：{document.title}
主题：{document.title}

# 章节结构
- Chapter 1
- Chapter 2
- List Comprehension  ← 包含关键词
"""
```

**问题**：
- 包含所有章节标题（关键词密度高）
- 与查询语义相关但无具体答案
- Dense + Sparse 两路都得分高 → RRF 融合后排在精确 chunk 前面

### PDF/PPT Outline 的设计原则

**目标**：用于文档发现，不参与精确答案排序

**方案 1：最小化 Outline（推荐）**
```python
outline_text = f"""
文件类型：{file_type}  # PDF / PPTX
文档名称：{document_title}
页数/幻灯片数：{page_count}
课程：{course_id}
"""
```

**特点**：
- 只包含元信息，不包含内容关键词
- 用于过滤文档类型、课程范围
- 不会与内容查询产生语义匹配

**方案 2：结构化 Outline（待验证）**
```python
outline_text = f"""
文件：{document_title}
类型：{file_type}
页数：{page_count}

# 章节结构（仅一级标题）
1. Introduction (p.1-5)
2. Methods (p.6-15)
3. Results (p.16-25)
"""
```

**特点**：
- 包含一级标题和页码范围
- 帮助用户快速定位章节
- 仍可能匹配关键词，需降权或分离 pipeline

**推荐**：暂时采用方案 1，待 PDF 索引完成后评估是否需要方案 2

---

## 四、PDF Chunking 策略

### 挑战

1. **页面边界问题**：
   - 一个概念可能跨多页
   - 简单按页切分会打断完整内容

2. **标题识别困难**：
   - PDF 没有语义标记（不像 Markdown `#`）
   - 需要通过字体大小、位置推断

3. **图表处理**：
   - 图表与说明文字的关联
   - 表格跨页问题

### 策略设计

#### 策略 A：基于页面的固定切分（Baseline）

**逻辑**：
```python
for page in pdf_pages:
    chunk = {
        "content": page.text,
        "page_number": page.number,
        "images": page.images,
        "tables": page.tables,
    }
```

**优点**：
- 简单、稳定
- 页码溯源清晰

**缺点**：
- 可能打断完整概念
- 上下文不足

**适用**：作为 baseline，快速验证效果

---

#### 策略 B：语义段落切分（推荐）

**逻辑**：
1. 使用 Docling 解析 PDF 为 Markdown
2. 识别标题层级（H1/H2/H3）
3. 按标题段切分，类似 Notebook chunker
4. 自适应合并小段落（target_tokens=280）

**Docling 输出示例**：
```markdown
## Introduction

Deep learning has revolutionized...

### Background

Neural networks are...

## Methods

### Model Architecture

We propose a new...
```

**切块逻辑**（复用 NotebookChunker 思路）：
```python
class PDFChunker:
    def chunk(self, pdf_document: ParsedPDFDocument):
        # 1. 按标题段分组
        sections = self._group_by_heading(pdf_document.markdown)
        
        # 2. 每个 section 内自适应切块
        chunks = []
        for section in sections:
            section_chunks = self._adaptive_chunk(
                section,
                target_tokens=280,
                max_tokens=450,
            )
            chunks.extend(section_chunks)
        
        return chunks
```

**优点**：
- 保持语义完整性
- 标题上下文清晰
- 可复用 Notebook chunker 逻辑

**挑战**：
- Docling 标题识别准确性
- 需要验证效果

---

#### 策略 C：滑动窗口 + 重叠（备选）

**逻辑**：
```python
chunks = []
for i in range(0, len(pages), stride):
    chunk = pages[i : i + window_size]  # 如 3 页为 1 个 chunk
    chunks.append(chunk)
```

**优点**：
- 保证上下文连续性
- 减少边界问题

**缺点**：
- 重复内容多（存储和检索开销大）
- 语义边界仍然模糊

**适用**：如果策略 B 效果不好，可作为后备

---

## 五、PPTX Chunking 策略

### 特点

- **独立性强**：每张幻灯片是独立的信息单元
- **结构清晰**：标题 + 正文 + 图片
- **篇幅短**：单张幻灯片文本通常 <200 tokens

### 策略：一张幻灯片 = 一个 Chunk（推荐）

**逻辑**：
```python
class PPTXChunker:
    def chunk(self, pptx_document: ParsedPPTXDocument):
        chunks = []
        for slide in pptx_document.slides:
            chunk = NotebookChunk(
                chunk_id=f"slide_{slide.number}",
                content=slide.text,  # 标题 + 正文
                page=slide.number,
                images=slide.images,
                tables=slide.tables,
            )
            chunks.append(chunk)
        return chunks
```

**优点**：
- 语义单元完整
- 溯源清晰（幻灯片编号）
- 实现简单

**特殊情况**：
- **单张幻灯片文本过长**（>450 tokens）：拆分为多个 chunk
- **多张幻灯片属于同一主题**：通过 section 标题关联

---

## 六、多表示索引策略

### 表示类型

与 Notebook 保持一致：

| 表示类型 | 内容 | 向量空间 | 用途 |
|----------|------|----------|------|
| **document_outline** | 文档元信息（方案 1） | text_dense | 文档发现（暂时禁用） |
| **text_content** | 页面/幻灯片正文 | text_dense + text_sparse | 精确答案检索 |
| **visual_fusion** | 图表+上下文说明 | visual_dense | 图表理解 |

### Embedding 策略

**文本 Embedding**（text_content）：
```python
# PDF 页面示例
embedding_text = f"""
文件：CS5494-week1.pdf
页码：12-13

## Deep Learning Fundamentals

Neural networks consist of layers...

[图片: neural_network_diagram.png]

Key concepts:
- Forward propagation
- Backpropagation
"""

# 调用 Qwen API
vector = dashscope.embed_document(embedding_text, output="dense&sparse")
```

**图表 Embedding**（visual_fusion）：
```python
# 图表 + 上下文
context = f"""
文件：CS5494-week1.pdf (p.12)
章节：Deep Learning Fundamentals

图表说明：Neural network architecture diagram
上下文：Neural networks consist of layers. The diagram shows...
"""

# 调用 Qwen Multimodal API
vector = dashscope.embed_visual_fusion(
    image=image_bytes,
    text=context,
)
```

---

## 七、溯源信息设计

### PDF Chunk Payload

```python
{
    "chunk_id": "chunk_abc123",
    "document_id": "document_xyz",
    "source_path": "CS5494-week1.pdf",
    "file_type": "pdf",
    "page_numbers": [12, 13],  # 可能跨页
    "section_path": ["Deep Learning", "Fundamentals"],
    "content": "...",
    "images": ["img_001.png"],
    "tables": ["table_001"],
}
```

### PPTX Chunk Payload

```python
{
    "chunk_id": "slide_012",
    "document_id": "document_xyz",
    "source_path": "Lecture1.pptx",
    "file_type": "pptx",
    "slide_number": 12,
    "slide_title": "Neural Networks",
    "content": "...",
    "images": ["img_012.png"],
}
```

---

## 八、实现优先级

### Phase 1：PDF Baseline（1 周）

1. 接入 Docling 解析 PDF
2. 实现基于页面的固定切分（策略 A）
3. 生成 text_content 表示并索引
4. 在 5-10 个 PDF 上验证检索效果

### Phase 2：语义切分优化（1 周）

1. 实现基于标题段的自适应切分（策略 B）
2. 对比策略 A 和 B 的检索效果
3. 调整 chunk 大小参数

### Phase 3：PPTX 支持（3 天）

1. 使用 python-pptx 解析
2. 一张幻灯片 = 一个 chunk
3. 验证效果

### Phase 4：图表支持（1 周）

1. 提取 PDF/PPTX 中的图片
2. 生成 visual_fusion 表示
3. 验证图表理解查询效果

### Phase 5：Document Outline 重新评估（待定）

1. 实现方案 1（最小化 outline）
2. 评估是否需要方案 2（结构化 outline）
3. 验证 outline 在 PDF 场景的价值

---

## 九、评估方案

### 评估集设计

**新增 PDF/PPTX 问题**（20-30 题）：
- 跨页概念查询："什么是反向传播？"（可能跨 2-3 页）
- 图表理解："神经网络架构图中各层的作用是什么？"
- 公式查询："梯度下降的更新公式是什么？"
- 表格查询："论文中各模型的准确率对比是多少？"

### 对比指标

| 指标 | Notebook | PDF | PPTX |
|------|----------|-----|------|
| Recall@8 | 1.00 | ? | ? |
| MRR@8 | 0.893 | ? | ? |
| 页码/幻灯片命中率 | (cell 命中率 1.00) | ? | ? |

**目标**：
- PDF Recall@8 ≥ 0.90
- PPTX Recall@8 ≥ 0.95（幻灯片独立性强，应更容易）

---

## 十、风险与挑战

### 已知问题

1. **Docling 标题识别准确性**
   - 部分 PDF 扫描版，OCR 错误
   - 标题样式不统一

2. **跨页内容切分**
   - 公式、表格跨页
   - 需要人工验证切分效果

3. **图表与文字关联**
   - 图表说明可能在下一页
   - 需要设计上下文窗口

### 缓解措施

1. **人工抽查**：选 5-10 个 PDF 手动验证切分质量
2. **多策略对比**：Baseline vs 语义切分 vs 滑动窗口
3. **用户反馈**：实际使用中收集失败案例

---

## 十一、与 Notebook 的统一接口

### Parser Protocol

```python
class DocumentParser(Protocol):
    def parse(self, file_path: Path) -> ParsedDocument: ...

class NotebookParser(DocumentParser): ...
class PDFParser(DocumentParser): ...
class PPTXParser(DocumentParser): ...
```

### Chunker Protocol

```python
class DocumentChunker(Protocol):
    def chunk(self, document: ParsedDocument) -> ChunkingResult: ...

class NotebookChunker(DocumentChunker): ...
class PDFChunker(DocumentChunker): ...
class PPTXChunker(DocumentChunker): ...
```

### Indexer 自动选择

```python
def index_document(file_path: Path):
    if file_path.suffix == ".ipynb":
        parser = NotebookParser()
        chunker = NotebookChunker()
    elif file_path.suffix == ".pdf":
        parser = PDFParser()
        chunker = PDFChunker()
    elif file_path.suffix in [".ppt", ".pptx"]:
        parser = PPTXParser()
        chunker = PPTXChunker()
    
    document = parser.parse(file_path)
    chunks = chunker.chunk(document)
    # ... 后续流程统一
```

---

## 十二、总结

### 核心设计决策

1. **PDF Chunking**：语义段落切分（基于 Docling + 标题层级）
2. **PPTX Chunking**：一张幻灯片 = 一个 chunk
3. **Document Outline**：最小化设计，仅元信息，避免关键词污染
4. **统一接口**：Parser + Chunker Protocol，复用索引流程

### 下一步

1. 实现 PDFParser（Docling）
2. 实现 PDFChunker（语义切分）
3. 在 5-10 个 PDF 上验证效果
4. 准备 PDF 评估集（20 题）

---

**文档版本**：v1.0  
**最后更新**：2026-09-29  
**状态**：设计完成，待实现
