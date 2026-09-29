# SRAG：课程资料检索系统

SRAG 是一个面向课程资料的本地优先 RAG 项目。当前 Notebook 已形成可评估、可追溯的混合检索链路；PDF/PPTX 因主要是幻灯片资料，正在统一为按页处理的 SlideDeck 预处理链路。Embedding、查询、重排和回答使用外部 API；文档解析和按需 OCR 可在 Colab GPU 临时运行，本地保存 Qdrant、元数据和应用服务。

详细设计见 [RAG.md](./RAG.md)，开发进度见 [CHECKLIST.md](./CHECKLIST.md)。

## 当前状态

Notebook 混合检索闭环已经完成：

- 只读解析 `.ipynb`，保留 Markdown、代码、输出、HTML 表格、图片和原始 cell 顺序。
- 按标题段和 Cell Group 自适应切块，代码与对应输出保持关联。
- 通过 DashScope API 生成 Qwen dense+sparse 文本向量和图文融合向量。
- 使用 Qdrant 命名向量保存表示和课程、文件、章节、cell 等溯源信息。
- 支持多文件幂等索引、原始文本 query 的 Dense+Sparse RRF Top K 召回、课程/文件过滤。
- `/api/rag/index` 与 `/api/rag/search` 已接入真实索引和检索流程。
- SQLite 保存文档结构、内容哈希、版本、发布日期和索引任务；查询只使用已发布的 active 版本。
- 支持未变化文件跳过 Embedding、新版本原子切换、旧向量清理和文档删除。
- 真实课件验证了中文问题检索英文内容，以及文件、章节和 cell 来源回溯。
- 已建立 20 题 Notebook 检索基线，并记录原始 query 的 dense Top 8 结果。
- PDF/PPTX 已收敛为统一 SlideDeck 预处理链路，已有页级解析、资源标注、超长页切分、整页图片和便携预处理包；尚未进行幻灯片 Embedding 或接入索引 API。

当前检索支持：
- **Dense + Sparse RRF 融合**：已接入 `/api/rag/search`，当前 20 题基线达到 Recall@8=1.00。
- **Qwen Rerank**：客户端和消融实验已完成，尚未接入正式检索入口。
- **Visual fusion**：索引、存储和召回方法已实现，正式检索入口尚未调用视觉召回。
- **上下文回取**：预算与排序工具已有实现，相邻 chunk ID 回取和正式集成未完成。
- **DeepSeek 查询理解**：客户端已实现，正式检索入口尚未调用。

Document outline 因融合问题暂时禁用，待 PDF/PPT 接入后重新设计。

PDF/PPTX 当前统一按幻灯片处理：一页/一张作为基础 Chunk，保留页码、布局、相邻关系、整页截图和资源，只对超长页做页内切分。数字 PDF 优先读取原生文本，可在 Colab 对文本稀少页开启 OCR。资源会按重复率、位置和面积标注，视觉表示仍以整页截图为主。预处理输出 `deck.json + chunks.json + slides/ + assets/`；幻灯片 Embedding 与索引接入是下一阶段。旧版 `.ppt` 需先转换为 `.pptx`。

## 流程

```text
.ipynb（当前正式链路）
  ↓
NotebookParser（不执行课件代码）
  ↓
标题段 + Cell Group + 自适应 Chunk
  ↓
Qwen Embedding API
  ↓
Qdrant
  ↓
query embedding → dense + sparse → RRF → 课程/文件过滤
  ↓
chunk 内容 + 文件/章节/cell 来源
```

## 技术栈

| 模块 | 选型 |
| --- | --- |
| API | FastAPI |
| Notebook 解析 | nbformat |
| 幻灯片预处理 | Docling、RapidOCR、python-pptx（Colab GPU） |
| 文本 Embedding | `qwen3.7-text-embedding` API |
| 视觉 Embedding | `qwen3-vl-embedding` API |
| 查询理解（组件已实现，待集成） | `deepseek-flash` API |
| Rerank（组件及消融已完成，待集成） | `qwen3.7-text-rerank` API |
| 向量库 | Qdrant，本地 Docker |
| 元数据 | SQLite |
| 回答模型（规划） | `deepseek-flash` API |

项目不在本地电脑或云服务器部署长期模型服务；Docling/RapidOCR 仅在 Colab 临时运行时下载并执行解析模型。

## 快速开始

### 1. 安装依赖

需要 Python 3.11+、[uv](https://docs.astral.sh/uv/) 和 Docker。

```bash
uv sync --extra dev
cp .env.example .env
```

需要在本地解析 PDF/PPTX 时安装文档可选依赖：

```bash
uv sync --extra dev --extra documents
```

在 `.env` 中至少填写：

```dotenv
COURSE_ROOT=/path/to/your/courses
DASHSCOPE_API_KEY=your_api_key
```

其他模型、切块和存储配置见 [.env.example](./.env.example)。

### 2. 启动 Qdrant

```bash
docker compose up -d
```

- REST API：<http://localhost:6333>
- Web UI：<http://localhost:6333/dashboard>
- gRPC：`localhost:6334`

数据默认写入 `data/qdrant`，不会提交到 Git。

### 3. 启动 API

```bash
uv run uvicorn app.main:app --reload
```

健康检查：

```bash
curl http://127.0.0.1:8000/api/health
```

### 4. 索引 Notebook

```bash
curl -X POST http://127.0.0.1:8000/api/rag/index \
  -H 'Content-Type: application/json' \
  -d '{
    "course_id": "CS5489",
    "paths": ["/absolute/path/to/Lecture1.ipynb"]
  }'
```

接口接受一个或多个明确的文件路径。目录扫描和自动同步尚未进入服务层。

### 5. 检索

```bash
curl -X POST http://127.0.0.1:8000/api/rag/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "贝叶斯公式是什么？",
    "scope": {
      "course_ids": ["CS5489"],
      "document_ids": []
    },
    "top_k": 5
  }'
```

响应包含分数、chunk 内容、文件路径、章节路径和 cell IDs。Cosine 分数用于同一检索配置下的相对排序，不代表答案正确率。

## 实验 Notebook

### 端到端检索体验

[notebooks/retrieval_end_to_end.ipynb](./notebooks/retrieval_end_to_end.ipynb)

覆盖：

```text
目录发现 → 解析 → 切块预览 → HTTP 索引 → Qdrant 统计
→ 自由 query → Top K 内容与来源
```

运行：

```bash
uv run jupyter lab notebooks/retrieval_end_to_end.ipynb
```

配置 `COURSE_DIR` 和 `MAX_FILES` 后即可处理单个课程目录。将 `MAX_FILES=None` 会索引全部符合条件的 Notebook，因此首次运行建议先限制文件数量。

### 切块与多模态表示实验

[notebooks/chunk_embedding_experiment.ipynb](./notebooks/chunk_embedding_experiment.ipynb)

用于检查 Cell Group、Section、Chunk、文本表示和图片融合表示。视觉表示当前只在准备阶段验证，尚未写入正式检索闭环。

### Colab 幻灯片预处理实验

[notebooks/cloud_document_preprocess.ipynb](./notebooks/cloud_document_preprocess.ipynb)

用于在 Colab GPU 上运行统一 SlideDeck 预处理：PDF 原生文本与按需 OCR、PPTX 结构提取与整页渲染、资源标注、页级 Chunk，以及 ZIP 预处理包下载。该 Notebook 不调用 Embedding API，也不连接本地 Qdrant。

## 评估结果

**20 题 Notebook 基线**（7 份 ML/X AI 课件）：

| 策略 | Recall@1 | Recall@8 | MRR@8 | 延迟(ms) |
|------|----------|----------|-------|----------|
| Dense | 0.80 | 0.95 | 0.875 | 1,609 |
| **RRF** | **0.85** | **1.00** | **0.893** | **1,317** |
| RRF+Outline+Rerank（实验） | **0.90** | **1.00** | **0.950** | 1,988 |

**当前默认**：RRF。Rerank 数据来自独立消融脚本，正式检索接口尚未启用该路径。延迟是单次 API 实验结果，会受网络影响。

评估详情见 [evals/results/](./evals/results/)。

## API 状态

| 方法 | 路径 | 状态 |
| --- | --- | --- |
| `GET` | `/api/health` | 已实现 |
| `POST` | `/api/rag/index` | 已实现 |
| `POST` | `/api/rag/search` | 已实现 |
| `GET` | `/api/rag/index/{job_id}` | 已实现 |
| `GET` | `/api/rag/documents` | 已实现 |
| `DELETE` | `/api/rag/documents/{document_id}` | 已实现 |
| `POST` | `/api/chat` | 后续阶段 |

## 项目结构

```text
app/
├── api/                 # HTTP 接口
├── clients/
│   ├── dashscope.py     # Qwen embedding、rerank API
│   └── deepseek.py      # DeepSeek 查询理解、改写、回答 API
├── evaluation/          # 检索评估逻辑
├── rag/
│   ├── parsers/         # 文件解析
│   ├── chunker.py       # Notebook 结构化切块
│   ├── slide_chunker.py # PDF/PPTX 统一页级切块
│   ├── slide_assets.py  # 幻灯片资源信息价值标注
│   ├── slide_preprocessor.py # 预处理包导出
│   ├── indexer.py       # 索引编排
│   ├── retriever.py     # 混合检索入口
│   ├── fusion.py        # RRF 融合与去重
│   └── context.py       # 上下文回取与预算管理
├── schemas/             # 请求、文档、向量和索引结构
├── stores/
│   ├── qdrant.py        # 向量存储（text/visual）
│   └── sqlite.py        # 元数据、版本和索引任务
├── config.py
└── main.py

notebooks/              # 可运行实验
evals/                  # 检索题集、运行入口和基线结果
tests/                  # 单元与接口测试
CHECKLIST.md            # 开发任务
RAG.md                  # 架构和方法设计
agent.md                # 当前状态与最近变更
PROGRESS_HISTORY.md     # 完整里程碑历史
```

## 下一步

根据 [CHECKLIST.md](./CHECKLIST.md)：

1. 用代表性 PDF/PPTX 验收 SlideDeck 的文本连续性、资源溯源与处理时间。
2. 对幻灯片文字和整页截图分别生成文本与多模态向量，并接入本地索引。
3. 用固定题集验证融合召回，再接通 Rerank、查询理解、上下文和回答生成。
4. 检索质量稳定后再开发上传、来源展示和可选云端部署。

## 测试

```bash
uv run pytest -q
uv run ruff check app tests
```

核心测试包含无个人路径的 SlideDeck 切块、资源筛选、PPTX 预处理包和 PDF 页级烟雾测试。
