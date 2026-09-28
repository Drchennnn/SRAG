# SRAG：课程资料检索系统

SRAG 是一个面向课程资料的本地优先 RAG 项目。当前聚焦 Jupyter Notebook 的解析、结构化切块、向量索引与可追溯检索；模型推理统一使用 API，本地只运行 Qdrant、元数据存储和应用服务。

详细设计见 [RAG.md](./RAG.md)，开发进度见 [CHECKLIST.md](./CHECKLIST.md)。

## 当前状态

Notebook 最小检索闭环已经完成：

- 只读解析 `.ipynb`，保留 Markdown、代码、输出、HTML 表格、图片和原始 cell 顺序。
- 按标题段和 Cell Group 自适应切块，代码与对应输出保持关联。
- 通过 DashScope API 生成 Qwen dense+sparse 文本向量和图文融合向量。
- 使用 Qdrant 命名向量保存表示和课程、文件、章节、cell 等溯源信息。
- 支持多文件幂等索引、原始文本 query 的 dense Top K 召回、课程/文件过滤。
- `/api/rag/index` 与 `/api/rag/search` 已接入真实索引和检索流程。
- 真实课件验证了中文问题检索英文内容，以及文件、章节和 cell 来源回溯。

当前正式检索链路只使用文本 dense 向量。sparse/RRF、rerank、视觉召回、版本管理、PDF 和回答生成仍在开发计划中。

## 流程

```text
.ipynb
  ↓
NotebookParser（不执行课件代码）
  ↓
标题段 + Cell Group + 自适应 Chunk
  ↓
Qwen Embedding API
  ↓
Qdrant
  ↓
query embedding → dense Top K → 课程/文件过滤
  ↓
chunk 内容 + 文件/章节/cell 来源
```

## 技术栈

| 模块 | 选型 |
| --- | --- |
| API | FastAPI |
| Notebook 解析 | nbformat |
| 文本 Embedding | `qwen3.7-text-embedding` API |
| 视觉 Embedding | `qwen3-vl-embedding` API |
| 查询理解（规划） | `deepseek-flash` API |
| Rerank（规划） | `qwen3.7-text-rerank` API |
| 向量库 | Qdrant，本地 Docker |
| 元数据 | SQLite，下一阶段 |
| 回答模型（规划） | GPT API |

项目不下载或运行本地模型权重。

## 快速开始

### 1. 安装依赖

需要 Python 3.11+、[uv](https://docs.astral.sh/uv/) 和 Docker。

```bash
uv sync --extra dev
cp .env.example .env
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

## API 状态

| 方法 | 路径 | 状态 |
| --- | --- | --- |
| `GET` | `/api/health` | 已实现 |
| `POST` | `/api/rag/index` | 已实现 |
| `POST` | `/api/rag/search` | 已实现 |
| `GET` | `/api/rag/index/{job_id}` | 待实现 |
| `GET` | `/api/rag/documents` | 待实现 |
| `DELETE` | `/api/rag/documents/{document_id}` | 待实现 |
| `POST` | `/api/chat` | 后续阶段 |

## 项目结构

```text
app/
├── api/                 # HTTP 接口
├── clients/             # 外部模型 API 客户端
├── rag/
│   ├── parsers/         # 文件解析
│   ├── chunker.py       # Notebook 结构化切块
│   ├── indexer.py       # 索引编排
│   └── retriever.py     # 检索入口
├── schemas/             # 请求、文档、向量和索引结构
├── stores/
│   ├── qdrant.py        # 向量存储
│   └── sqlite.py        # 元数据存储（开发中）
├── config.py
└── main.py

notebooks/              # 可运行实验
tests/                  # 单元与接口测试
CHECKLIST.md            # 开发任务
RAG.md                  # 架构和方法设计
agent.md                # 当前进度与重大变更
```

## 下一阶段

1. SQLite 文档登记、版本、发布日期和结构关系持久化。
2. 幂等增量更新、旧版本清理和文档删除。
3. 建立小型 Retrieval Evaluation 基线。
4. sparse/dense RRF、rerank 和按需 query transformation。
5. 视觉召回与上下文回取。
6. PDF 解析、GPT 回答层和网页交互。

以 [CHECKLIST.md](./CHECKLIST.md) 为准逐项推进。

## 测试

```bash
uv run pytest -q
uv run ruff check app tests
```

当前测试覆盖 Notebook 解析、切块、Embedding 客户端、Qdrant 存储、索引器、检索器和 RAG HTTP 接口。
