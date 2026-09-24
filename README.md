# 课程资料 RAG

面向课程资料（Notebook、PDF 课件等）的检索增强问答系统。当前阶段只开发 **RAG（检索）部分**：文件解析、切块、向量化、混合检索与重排；**对话部分**（基于检索证据生成回答）留待后续接入，网页端交互也在 RAG 基线稳定后再设计。

详细设计依据见 [`RAG.md`](./RAG.md)。

## 整体流程

```text
┌───────────────────────────────────┐
│  RAG・离线索引（当前开发）          │
└───────────────────────────────────┘
课程文件（ipynb / PDF）
        │
        ▼
文件登记与版本检查
        │
        ▼
格式解析（Notebook / PDF）
        │
        ▼
父子切块
        │
        ▼
Qwen Embedding API（dense + sparse）
        │
        ▼
Qdrant 向量库
        │
        ▼
┌───────────────────────────────────┐
│  RAG・在线检索（当前开发）          │
└───────────────────────────────────┘
用户问题 → 范围过滤（课程 / 文件 / 版本）
        │
        ▼
混合召回（dense + sparse，RRF，含上方 Qdrant 向量库）
        │
        ▼
Qwen Rerank API
        │
        ▼
回取父块与原始资源
        │
        ▼
┌───────────────────────────────────┐
│  对话层（后续规划，当前不实现）      │
└───────────────────────────────────┘
GPT 生成回答（附文件 / 页码 / cell 引用）
        │
        ▼
网页端展示
```

- **RAG・离线索引**（当前开发）：课程文件解析、切块、调用 Qwen Embedding API 写入 Qdrant。
- **RAG・在线检索**（当前开发）：用户问题经范围过滤、混合召回、Qwen Rerank API 重排后回取证据。
- **对话层**（后续规划，当前不实现）：只表示与 RAG 部分的衔接点，暂不开发。

## RAG 部分：离线索引流程

对应 `RAG.md` 各阶段方法表第 1–7 步。

### 1. 文件登记

只导入选定课程文件，不对 `/Users/Link/Course` 全量扫描。对每个文件计算内容哈希，记录课程、文件路径、资料类型、版本、发布日期和导入状态；重复内容跳过，内容变更则生成新版本。发布日期由可信输入提供，不从目录名推断；未到发布日期的资料不进入可查询范围。

### 2. 格式解析

- **Notebook**：用 `nbformat` 按原始 cell 顺序读取 Markdown、代码、文本/HTML 输出、图片和附件，不执行 notebook；由 Markdown 标题建立章节路径，保留代码与其输出的关联。需处理异常/缺失的 cell ID，长输出、报错堆栈和大数组只保留可检索摘要。已确认部分课程存在中文重写版 notebook（中英夹杂、含 LaTeX 公式与 HTML 样式），双语支持按硬需求设计。
- **PDF**（第二批接入）：本地只做无模型的原生文字提取、页面渲染和质量检测；若页面缺少可靠文字层，或主要信息位于表格、公式和图表中，则调用 Qwen OCR / VL API 做结构化解析和图像说明。首期不启用 Docling 默认的本地版面、OCR 或表格模型；若使用 Docling，仅采用其 model-free native pipeline。课程资料中既有可直接提取文字的 PDF，也有前几页几乎提取不到文字的文件，因此不能假设所有 PDF 都无需 OCR。
- 两种解析器统一产出 `document → section → block → asset` 结构；图片二进制不写入向量 payload。

### 3. 父子切块

父块是一个有边界的教学单元（小节、短课件页、相关 Markdown＋代码＋输出），约 1,000–1,800 tokens；子块是可独立匹配的定义、解释、代码逻辑或图表说明，约 300–600 tokens。标题、公式、函数和图表说明优先保持完整，不把整本 PDF 或整个 notebook 当父块，具体大小以检索题集效果调整。

### 4. 检索表示

每个正文子块附上简短的课程/文件/章节前缀；图表、复杂表格可有独立文字说明入口，均指回原始资源和父块。仅在评估显示收益时才为代码用途或章节摘要增加独立入口，暂不使用逐 token 的 ColBERT 多向量。

### 5. 向量化（Qwen API）与存储

调用 `qwen3.7-text-embedding` **DashScope 原生接口**（非 OpenAI 兼容模式，当前 sparse 输出需要原生接口），设置 `output_type=dense&sparse`，索引侧固定 `text_type=document`，一次调用同时获得：

- **dense 向量**：首期使用 1,024 维；更高或更低维度只通过同一 API 做对照实验
- **sparse 向量**：词项权重（index/value/token），用于关键词匹配

批大小、token 上限、限流和价格以实际开通地域及账户的当前 API 返回和官方文档为准，不把这些易变化的数值写成系统不变量。导入任务按批次调用，并实现限流退避、失败重试、断点续传和用量记录。

向量与可过滤元数据（课程、文件、版本、发布日期、章节路径、cell/页码等）写入本机 Qdrant；父块正文与文件清单存 SQLite；原始文件与图片资源留在文件存储中。记录模型版本、维度和索引版本，便于后续更换配置时重建索引。

## RAG 部分：在线检索流程

对应 `RAG.md` 各阶段方法表第 8–9 步。

### 6. 查询与召回

先解析用户明确选择的课程/文件范围和当前可见版本；原问题并行走 dense、sparse 检索，查询侧调用 embedding API 时使用 `text_type=query`，用 RRF 融合候选。保留课程编号、代码符号、公式和否定词，避免被过度改写。指代补全、中文到英文的辅助查询和后续可能的 query transformation 均调用 Qwen 文本生成 API；原问题始终保留，模型生成内容只作为检索线索。HyDE、step-back、多查询默认关闭，仅在评测显示收益时启用。

### 7. 重排与回取

对融合后的有限候选调用 `qwen3.7-text-rerank` API。按 `parent_id` 去重后回取必要的父块、相邻 cell、原图或原表，并限制总上下文长度，记录每条证据的具体来源（文件、页码或 cell）。候选数量、请求预算、超时和重试策略通过实测确定。

## 对话部分（后续规划，当前不实现）

后续问答层优先接 GPT（以账户和服务地区实际可用性为前提），只依据检索证据回答，给出文件名、页码或 cell 引用；资料不足时明确说明缺口。课程/考试总结类问题走"按文件和章节枚举证据 → 分节汇总"的流程，不依赖一次 top-k 查询。这部分不在当前开发范围内，架构图中仅标出与 RAG 部分的衔接点（检索证据 → GPT → 网页展示）。

## 增量更新与评估

- **增量更新**：新文件生成父块、子块和资源后再切换为可见版本；替换或删除文件时同步清理旧子块、派生说明和失效资源；每次导入记录失败与重试状态。检索时同时校验 `document_id`、版本和发布日期，避免旧版或未解封内容泄露。
- **评估**：人工标注约 60–100 个课程问题及原文件页码/cell，覆盖概念、代码、图表、指定文件、跨章节和无答案题；测 `Recall@k`、`nDCG@k`、引用准确性、回答依据、延迟与资源占用。固定同一题集，对比单层块、父子块、混合检索、多表示与重排的效果差异；课程总结另测章节覆盖率，增量导入另测旧版本清理是否彻底。

## 技术选型摘要

| 组件             | 选型                                                | 说明                                                        |
| ---------------- | --------------------------------------------------- | ----------------------------------------------------------- |
| 向量库           | Qdrant（本机 Docker）                               | dense + sparse 命名向量，支持课程/版本/发布日期等元数据过滤 |
| 元数据与父块存储 | SQLite                                              | 文件清单、版本、父块正文、资源关系                          |
| Embedding        | `qwen3.7-text-embedding`（DashScope 原生 API）       | 一次调用同时获取 dense + sparse，不本地部署                 |
| Query transformation | Qwen 文本生成 API                              | 仅在指代、跨语言或复杂问题中按需调用                        |
| Reranker         | `qwen3.7-text-rerank` API                            | 对融合候选做语义重排                                        |
| OCR / 图表理解   | Qwen OCR / VL API                                    | PDF 回退解析与图片说明，不本地部署视觉模型                  |
| 对话模型（后续） | GPT API                                              | 仅在检索证据基础上生成回答，当前不开发                      |

## 关键约束

- **Notebook 是首要数据模型**：cell 输出可能包含图片、HTML 表格或多种 MIME 表示，不能整份转纯文本后固定字符数切割。
- **图表说明是检索线索，不是精确事实来源**：回答图中数值或曲线关系时需回取原图、表格或原始输出核对。
- **父子块与多表示可以组合**：父子块解决"找到细节后补足上下文"，多表示解决"正文之外怎样找到代码、图和表"；不为每个子块复制同一份标题或章节摘要向量。
- **存储与模型推理分离**：Qdrant、SQLite、原始文件和无模型解析工具保存在本地（后续可迁移到云端）；embedding、查询改写、重排、OCR、图像理解及最终问答全部调用外部 API，不下载或运行模型权重。
- **全部模型推理依赖外部 API**：需处理限流退避、失败重试、网络异常、API Key 保密、费用观测，以及课件内容发送到第三方服务的取舍。API 不可用时暂停相关推理任务，不静默切换成本地模型。
- **先测后调**：token 范围、候选数量、模型配置均是初始值，最终以真实课程问题的效果、延迟证据决定。

## 项目结构

项目采用按功能域组织的精简结构。`api` 只负责 HTTP 协议，`rag` 负责索引与检索流程，`clients` 封装外部模型 API，`stores` 封装本地存储。Python 包目录中的 `__init__.py` 在下图中省略。

```text
SRAG/
├── app/
│   ├── main.py                     # FastAPI 入口与应用生命周期
│   ├── config.py                   # .env 的唯一读取入口
│   ├── api/
│   │   ├── health.py               # 健康检查
│   │   ├── rag.py                  # 索引、文档管理与检索接口
│   │   └── chat.py                 # 对话阶段添加
│   ├── schemas/
│   │   ├── document.py             # 文档、内容块、资源与父子块结构
│   │   ├── embedding.py            # dense/sparse 向量结果结构
│   │   ├── indexing.py             # 待写入存储的索引数据包
│   │   ├── rag.py                  # 索引与检索请求/响应
│   │   └── chat.py                 # 对话阶段添加
│   ├── rag/
│   │   ├── indexer.py              # 增量索引编排
│   │   ├── chunker.py              # 父子切块与检索表示
│   │   ├── retriever.py            # 混合检索、重排与父块回取
│   │   └── parsers/
│   │       ├── notebook.py         # Notebook 只读解析
│   │       └── pdf.py              # PDF 阶段添加
│   ├── clients/
│   │   ├── dashscope.py            # Qwen API 客户端
│   │   └── openai.py               # GPT API 客户端，对话阶段添加
│   ├── stores/
│   │   ├── qdrant.py               # dense/sparse 向量与过滤检索
│   │   └── sqlite.py               # 文档版本、父块和导入状态
│   └── chat/
│       └── answerer.py             # 证据组织与回答生成，对话阶段添加
├── tests/
│   ├── fixtures/
│   ├── test_notebook_parser.py
│   ├── test_chunker.py
│   ├── test_indexer.py
│   └── test_retriever.py
├── evals/
│   ├── retrieval_cases.jsonl
│   └── run_retrieval.py
├── notebooks/
│   └── chunk_embedding_experiment.ipynb
├── data/                            # 本地运行数据，不提交 Git
│   ├── assets/
│   ├── qdrant/
│   └── srag.sqlite3
├── .env                             # 本机密钥与服务配置，不提交 Git
├── .env.example                     # 可提交的配置模板
├── compose.yml                      # 本地 Qdrant
├── pyproject.toml
├── README.md
├── RAG.md
└── agent.md
```

首期只创建 Notebook 检索闭环需要的文件。PDF、GPT 对话及对应文件在进入相关里程碑时添加，不创建无实现的空模块。

### 依赖方向

```text
api
 ├── rag.indexer ─── parsers + chunker + dashscope + stores
 ├── rag.retriever ─ dashscope + stores
 └── chat.answerer ─ rag.retriever + openai（后续）
```

- API 路由不直接操作 Qdrant、SQLite 或模型 SDK。
- `clients` 只处理外部 API 请求、超时、重试和错误转换，不包含检索策略。
- Query translation、dense+sparse 召回、RRF、rerank 和父块回取只有 `rag/retriever.py` 一个入口。
- 外部资源在 FastAPI lifespan 中初始化，禁止在模块导入阶段连接数据库或验证 API Key。

### 命名规范

- 包和文件统一使用 `snake_case`；集合型目录使用复数，如 `clients`、`schemas`、`stores`、`parsers`。
- 目录已表达上下文时不重复前缀，例如使用 `rag/indexer.py`，不使用 `rag_vector_index_service.py`。
- 核心类使用职责名称，例如 `DocumentIndexer`、`ParentChildChunker`、`HybridRetriever`、`QdrantStore` 和 `MetadataStore`。
- 暂不创建职责模糊的 `utils.py`、`common.py`、`manager.py` 或无实际复用需求的基类。

## 配置接口

所有外部模型 API 的密钥、地址和模型名统一从 `.env` 读取。仓库只提交 `.env.example`；`.env`、Qdrant 数据、SQLite 和解析出的资源目录均由 `.gitignore` 排除。切块和检索实验参数使用 `config.py` 中的版本化默认值，并将实际值记录到索引元数据中。

```dotenv
# 应用配置
APP_ENV=development
APP_HOST=127.0.0.1
APP_PORT=8000
COURSE_ROOT=/Users/Link/Course
DATA_DIR=./data

# DashScope / Qwen：RAG 模型配置
DASHSCOPE_API_KEY=
DASHSCOPE_BASE_URL=
DASHSCOPE_EMBEDDING_MODEL=qwen3.7-text-embedding
DASHSCOPE_EMBEDDING_DIMENSION=1024
DASHSCOPE_EMBEDDING_BATCH_SIZE=20
DASHSCOPE_RERANK_MODEL=qwen3.7-text-rerank
DASHSCOPE_QUERY_MODEL=
DASHSCOPE_OCR_MODEL=
DASHSCOPE_VISION_MODEL=
DASHSCOPE_TIMEOUT_SECONDS=60
DASHSCOPE_MAX_RETRIES=3

# OpenAI：对话阶段配置
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_CHAT_MODEL=

# 本地存储
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=course_chunks
SQLITE_PATH=./data/srag.sqlite3
```

## 已实现：Notebook 解析、父子切块与 Embedding

- `NotebookParser` 只读解析真实 `.ipynb`，不执行代码；支持 Markdown 标题路径、代码、stdout、报错、纯文本/HTML 表格输出以及 PNG/JPEG/SVG/GIF 资源提取。
- `ParentChildChunker` 按章节和源内容块构建父块，再生成带少量重叠的子块；ID 由内容稳定生成，子块保留实际来源 `block_id`，用于后续 cell 引用。
- 当前 Token 数是不加载本地模型的中英文混合估算值，负责控制切块边界；实际用量以 DashScope 返回的 `usage.total_tokens` 为准。
- `DashScopeClient` 使用原生 DashScope 接口，文档固定传入 `text_type=document`，查询固定传入 `text_type=query`，默认一次返回 1,024 维 dense 和 sparse 权重。
- `DocumentIndexer.prepare_document()` 已串联 `ParsedDocument → 父子切块 → 批量 Embedding`，并校验每个子块与向量的顺序、数量和维度，输出可直接交给 SQLite/Qdrant 存储层的数据包。
- 客户端支持最多 20 条一批、保持输入顺序、超时、有限重试和请求 ID 记录；API Key 不写日志。
- 实验 Notebook：[`notebooks/chunk_embedding_experiment.ipynb`](./notebooks/chunk_embedding_experiment.ipynb)。在 `NOTEBOOK_PATH` 填写一个 `.ipynb` 地址后，它会通过项目的 Parser、Indexer、Chunker 和 Embedding 客户端展示内容块、父子块、向量维度、sparse 词项和查询相似度排序。

运行实验：

```bash
uv run jupyter lab notebooks/chunk_embedding_experiment.ipynb
```

## 应用接口契约

### RAG

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `POST` | `/api/rag/index` | 提交一组明确选择的本地课程文件进行增量索引 |
| `GET` | `/api/rag/index/{job_id}` | 查询索引任务状态 |
| `GET` | `/api/rag/documents` | 查询已登记文档及版本 |
| `DELETE` | `/api/rag/documents/{document_id}` | 删除文档及其派生索引 |
| `POST` | `/api/rag/search` | 按课程、文件和可见版本过滤后检索证据 |

检索响应必须包含 `chunk_id`、`parent_id`、融合或重排分数，以及文件、页码或 cell 定位。API 层不把 Qdrant 原始响应直接暴露给调用方。

### 对话（后续）

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `POST` | `/api/chat` | 基于检索证据返回完整回答 |
| `POST` | `/api/chat/stream` | 以 SSE 返回流式回答和最终引用 |

对话请求统一包含 `session_id`、`message`、`mode` 和课程/文件范围；`mode` 预留 `qa` 与 `summary`。GPT 只接收 `HybridRetriever` 产出的证据，不直接访问向量库。

## 首个可验收里程碑

选取少量具有 Markdown 说明、代码、图片输出的 notebook，完成导入、父子块检索、按文件过滤及 cell 引用；建立一批人工问题，对比纯 dense、dense+sparse、加重排、加父块回取。该基线稳定后，再接入 PDF 和对话层。

## 参考资料

- [Jupyter Notebook 文件格式](https://nbformat.readthedocs.io/en/latest/format_description.html)
- [通用文本向量 API（qwen3.7-text-embedding，dense+sparse 混合检索）](https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api)
- [通用文本排序模型 API（qwen3.7-text-rerank）](https://help.aliyun.com/zh/model-studio/text-rerank-api)
- [Qwen OCR API](https://help.aliyun.com/en/model-studio/qwen-vl-ocr-api-reference)
- [阿里云百炼限流说明](https://help.aliyun.com/zh/model-studio/rate-limit)
- [Qwen3 Embedding 与 Reranker 技术背景](https://github.com/QwenLM/Qwen3-Embedding)（仅供架构原理参考，本项目通过 API 调用而非本地部署）
- [Qdrant 本地部署](https://qdrant.tech/documentation/quick-start/)与[混合查询](https://qdrant.tech/documentation/search/hybrid-queries/)
- [Docling model-free native PDF pipeline](https://docling-project.github.io/docling/reference/pipeline_options/)
