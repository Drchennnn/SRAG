# SRAG：课程资料检索系统

SRAG 是一个面向课程资料的本地优先 RAG 项目。模型通过外部 API 调用，本地运行应用服务、Qdrant 和 SQLite；文档解析可在 Colab GPU 临时执行。

方法与设计决策见 [RAG.md](./RAG.md)，任务状态见 [CHECKLIST.md](./CHECKLIST.md)。

## 当前状态

- **Notebook**：已支持 `.ipynb` 只读解析、结构化切块、Qwen Embedding、Qdrant/SQLite 版本索引，以及 `/api/rag/index` 和 `/api/rag/search` 的 Dense+Sparse RRF 检索。20 题基线 Recall@8 为 1.00。
- **PDF/PPTX**：四份 CS5187 课件共 109 页已入库，108 条文本与 109 条整页视觉向量；纯图示页可以只保留视觉向量。`/api/rag/search` 可自动路由明确描述页面图像的问题，返回页码和截图路径。两套各 28 题的评估均已完成。
- **待集成组件**：Qwen Rerank、DeepSeek 查询理解和相邻上下文已有组件或实验结果，尚未进入正式检索入口。Document outline 暂时禁用。

## 流程

```text
.ipynb → NotebookParser；PDF/PPTX → Colab SlideDeck 预处理包
  ↓
Notebook 不执行代码；幻灯片保留页码与截图
  ↓
Notebook 标题段/Cell Group；幻灯片页级 Chunk
  ↓
Qwen Embedding API
  ↓
Qdrant
  ↓
课程/文件与版本过滤 → 文本 dense+sparse RRF；明确的图像问题优先视觉召回
  ↓
Chunk 内容 + 文件/章节/cell 或页码来源
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

项目不部署长期模型服务；Docling/RapidOCR 可在 Colab 临时运行。

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

`retrieval_mode` 默认为 `auto`：照片、图示、曲线等明确的视觉问题优先返回视觉页；也可传 `text` 或 `visual` 显式选择。响应的 `mode_used` 标明实际模式，来源包含 Notebook cell ID 或幻灯片页码，视觉命中另含 `image_path`。分数只用于同一检索模式内排序，不代表答案正确率。

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

### 分批扩充课程语料

对 `/Users/Link/Course` 的一次扫描得到 218 个文件：66 个 IPYNB、116 个 PDF、34 个 PPTX、2 个旧版 PPT。本地清单 `data/ingest/course_manifest.csv`（`data/` 不提交 Git）把 4 份 CS5495 PDF 设为 `pilot`，122 份课程讲义/教程设为 `candidate`，20 份复习资料或答案设为 `review`，其余为 `skip` 或 `convert`。`candidate` 是候选，不表示已核实学期、解封时间或内容质量；清单中的 `academic_term` 和 `release_at` 留空待确认。重新扫描会覆盖手动填写的清单字段，因此应先备份已复核版本。

首批四份 PDF 共 91 页；[批处理 Colab Notebook](./notebooks/cloud_batch_preprocess.ipynb) 可以逐份预处理并把独立结果 ZIP 保存到 Drive：

```bash
uv run python tools/build_course_manifest.py /Users/Link/Course
uv run python tools/pack_course_batch.py /Users/Link/Course
```

把生成的 `data/ingest/colab_batch.zip` 上传到 Google Drive 的 `MyDrive/Colab Notebooks/SRAG/`，运行批处理 Notebook。Colab 路径前缀为 `/content/drive/MyDrive/Colab Notebooks/SRAG/`。批次包同时携带当前项目解析代码，不依赖 GitHub 上可能较旧的 `main` 分支。处理报告及每份 `.slidedeck.zip` 会写入其 `processed/` 子目录。下载并解压后，先核对页数、截图和切块，再用下一节的命令在**本地**调用 Qwen Embedding API，最后导入 Qdrant。Colab 负责 PDF/PPTX 的计算密集型解析与切块；IPYNB 沿用本地 Notebook 索引入口。后续新文件先加入/重新生成清单，按小批次重复流程；文件上传服务尚未实现。

CS5495 首批四份 PDF 已完成预处理和本地向量化：91 页、133 条文本向量和 91 条整页视觉向量。源文件、预处理包、Chunk 和向量 ID 已核对；本地 `data/ingest/embedding_report.json` 记录每份文件的向量包路径与 Token 用量，尚未导入 Qdrant。

需要扩大批次时可选课程和数量，例如 `--decision candidate --course-id CS5494 --limit 4`。批次包会重新核对源文件 SHA-256，避免把修改后的资料误用旧清单处理。

下载并解压预处理包后，在本地用 `.env` 中的 Qwen API 生成独立向量包：

```bash
uv run python -m app.rag.slide_embedder /path/to/slidedeck-bundle --pages 1,14
uv run python -m app.rag.slide_embedder /path/to/slidedeck-bundle
```

不指定 `--pages` 时会处理全部页面及截图；`--text-only` 只生成文本向量。结果默认写入 Git 忽略的 `data/embeddings/`，保留页码、Chunk ID、模型和 Token 用量。已有结果需加 `--force` 才会重新调用 API。此步骤不写入 Qdrant；用下述接口导入已有向量包。

### 导入已向量化的幻灯片

本地需有原始 PDF/PPTX、Colab 下载并解压的 `deck.json + chunks.json + slides/ + assets/` 包，以及对应的完整向量 JSON。原始文件 SHA-256 必须与包内文档 ID 一致。

```bash
curl -X POST http://127.0.0.1:8000/api/rag/slides/index \
  -H 'Content-Type: application/json' \
  -d '{"course_id":"CS5187","source_path":"/absolute/path/to/Lecture1-1.pdf","bundle_dir":"/absolute/path/to/bundle","embeddings_path":"/absolute/path/to/embeddings.json"}'
```

重复内容会跳过；需重建同一文件时传 `"force": true`。导入后可用 `/api/rag/search` 以 `course_ids` 或 `document_ids` 限定检索。自动视觉路由由 `RETRIEVAL_ENABLE_VISUAL=true` 控制；设为 `false` 时仍可显式请求 `retrieval_mode=visual`。

```bash
uv run python -m app.evaluation.slide_retrieval
```

题集在 [evals/slide_cases.jsonl](./evals/slide_cases.jsonl)，逐题排名和指标写入 `evals/results/slide_retrieval_baseline.json`。

新课件题集运行方式：

```bash
uv run python -m app.evaluation.slide_retrieval \
  --cases evals/slide_holdout_cases.jsonl \
  --output evals/results/slide_holdout_baseline.json
```

## 评估结果

**20 题 Notebook 基线**（7 份 ML/X AI 课件）：

| 策略 | Recall@1 | Recall@8 | MRR@8 | 延迟(ms) |
|------|----------|----------|-------|----------|
| Dense | 0.80 | 0.95 | 0.875 | 1,609 |
| **RRF** | **0.85** | **1.00** | **0.893** | **1,317** |
| RRF+Outline+Rerank（实验） | **0.90** | **1.00** | **0.950** | 1,988 |

**当前默认**：RRF。Rerank 数据来自独立消融脚本，正式检索接口尚未启用该路径。延迟是单次 API 实验结果，会受网络影响。

**28 题 CS5187 幻灯片页码基线**（32 页 PDF + 26 页 PPTX，含 4 道只描述图像细节的题）：

| 方法 | Recall@1 | Recall@5 | MRR@8 | 图像细节 Recall@5 |
| --- | ---: | ---: | ---: | ---: |
| 文本 Dense | 0.929 | 0.964 | 0.946 | 0.75 |
| 文本 Dense+Sparse RRF | 0.893 | 0.964 | 0.929 | 0.75 |
| 视觉向量 | 0.857 | 1.000 | 0.923 | 1.00 |
| 三路 RRF | 0.929 | 0.964 | 0.946 | 0.75 |

一题只描述河流、玫瑰和雪山的图片时，文本 Dense 排第 38、视觉排第 1、三路 RRF 排第 23。可见视觉表示有补充价值，简单等权融合却不能保证保留该结果。题集仅来自两份课件，多数题较易且没有独立保留集，数字不能外推到全课程。

**新课件 28 题测试集**（Week02 边缘检测 32 页、Week08 相机成像 19 页；检索候选包含上述四份课件）：

| 方法 | Recall@1 | Recall@5 | MRR@8 | 图像细节 Recall@5 |
| --- | ---: | ---: | ---: | ---: |
| 文本 Dense | 0.857 | 0.893 | 0.875 | 0.40 |
| 文本 Dense+Sparse RRF | 0.893 | 0.893 | 0.893 | 0.40 |
| 视觉向量 | 0.786 | 0.929 | 0.841 | 0.80 |
| 三路 RRF | 0.893 | 0.893 | 0.893 | 0.40 |

三道仅靠图示辨别的题，视觉召回排第 3、7、1，文本 Top 50 均未命中，等权融合排第 50、52、49。测试集已用于观察失败模式，后续调参需保留另一个未参与设计的最终测试集。指标只衡量来源页码命中，不衡量回答正确性；报告的耗时包含两次 Query Embedding 与三路检索，不是各方法的独立延迟。

**视觉路由复测**：在同一份四文档语料上，开发集的自动路由 Recall@5/8 为 **1.00/1.00**（文本 RRF 为 0.964/0.964）；新课件题集为 **0.964/1.00**（文本 RRF 为 0.893/0.893）。自动路由分别触发 8/28 和 4/28 题。细微图像差异题仍只在 Top 8 命中。逐题数据见 [开发集复测](./evals/results/slide_development_visual_route.json)与[新课件复测](./evals/results/slide_holdout_visual_route.json)。新课件题集已用于设计判断，后续仍需不同课程的最终检验。

**CS5495 新课程验证**：四份 PDF 共 91 页、133 条文本 Chunk 与 91 条视觉表示已入库；另有一份原有 Notebook。固定 [24 题页码题集](./evals/cs5495_slide_cases.jsonl)的结果如下。`正式入口`直接调用 `HybridRetriever.search`，其 Top K 可能包含同页多个 Chunk 或 Notebook 结果；离线页级基线先按页去重，两者指标不可直接等同。

| 检索范围与方法 | Recall@1 | Recall@5/8 | MRR@8 | 图表细节 Recall@5 |
| --- | ---: | ---: | ---: | ---: |
| CS5495 正式入口 | 0.667 | 0.875 | 0.748 | 0.25 |
| 全课程正式入口 | 0.583 | 0.875 | 0.696 | 0.25 |
| CS5495 离线页级自动路由 | 0.708 | 0.917 | 0.781 | 0.50 |

三道正式入口漏题都是图表定位题；现有词面路由都选择了文本，其中一题的视觉单路可排第 1，另两题的视觉排名为第 15 和 Top 50 之外。课程概念、公式和相近页题在 Top 5 均命中。题集只测来源页码，不测最终回答；这份新课程题集已经用于诊断，后续调整路由时应另留新资料验证。逐题结果见 [CS5495 页级消融](./evals/results/cs5495_slide_baseline.json)、[限定课程的正式入口](./evals/results/cs5495_formal_retrieval.json)和[全课程正式入口](./evals/results/cs5495_global_retrieval.json)。

评估详情见 [evals/results/](./evals/results/)。

## API 状态

| 方法 | 路径 | 状态 |
| --- | --- | --- |
| `GET` | `/api/health` | 已实现 |
| `POST` | `/api/rag/index` | 已实现，Notebook |
| `POST` | `/api/rag/slides/index` | 已实现，PDF/PPTX 向量包 |
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
│   ├── slide_embedder.py # 幻灯片离线向量化
│   ├── indexer.py       # Notebook 索引编排
│   ├── slide_indexer.py # 幻灯片向量包导入
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

## 测试

```bash
uv run pytest -q
uv run ruff check app tests
```

核心测试包含无个人路径的 SlideDeck 切块、资源筛选、PPTX 预处理包和 PDF 页级烟雾测试。
