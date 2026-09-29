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
- SQLite 保存文档结构、内容哈希、版本、发布日期和索引任务；查询只使用已发布的 active 版本。
- 支持未变化文件跳过 Embedding、新版本原子切换、旧向量清理和文档删除。
- 真实课件验证了中文问题检索英文内容，以及文件、章节和 cell 来源回溯。
- 已建立 20 题 Notebook 检索基线，并记录原始 query 的 dense Top 8 结果。

当前检索链路支持：
- **Dense + Sparse RRF 融合**：并行召回 dense 和 sparse 向量，使用 Reciprocal Rank Fusion 融合
- **Document outline 召回**：已实现但暂时禁用（包含关键词但无答案内容，导致 RRF 融合后挤掉精确 chunks）
- **Visual fusion 召回**：支持图文融合向量的索引与召回（可选，默认关闭）
- **上下文回取**：按需回取相邻 chunk、原图和原表，限制总上下文预算
- **Qwen Rerank**：使用 `qwen3.7-text-rerank` API 对候选重排
- **DeepSeek 查询理解**：可选查询改写（当前默认关闭，保留原始查询）

**Outline 功能说明**：Document outline 包含文档所有章节标题，设计用于文档发现场景。但在精确问答场景中，outline 在 Dense 和 Sparse 两路都匹配到查询关键词，经 RRF 融合后得分系统性高于精确的 text_content chunks，导致用户查询无法命中正确答案。当前已禁用，待接入 PDF/PPT 后重新评估是否需要分离 pipeline 实现文档发现功能。

PDF 和回答生成仍在开发计划中。

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
| 元数据 | SQLite |
| 回答模型（规划） | `deepseek-flash` API |

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

## 检索评估

[evals/retrieval_cases.jsonl](./evals/retrieval_cases.jsonl) 包含 15 条开发集和 5 条保留集问题，覆盖 7 份 ML/X AI Notebook。每题标注参考答案、预期文件、章节和 cell；评估按 cell 严格匹配，并统计 Recall@K、MRR、nDCG、文件/章节/cell 命中率和查询延迟。

### 运行评估

首次运行可幂等索引评估语料：

```bash
uv run python evals/run_retrieval.py --index
```

语料已索引时直接运行基线评估：

```bash
uv run python evals/run_retrieval.py
```

### 消融实验

运行检索策略消融实验，对比 dense、RRF、RRF+outline、RRF+outline+rerank：

```bash
uv run python evals/run_retrieval_ablation.py --index  # 首次运行需索引
uv run python evals/run_retrieval_ablation.py
```

可选参数：
- `--strategies dense rrf rrf+outline rrf+outline+rerank`：指定评估策略
- `--top-k 8`：返回结果数量
- `--split dev|holdout|all`：评估数据集切分

### 评估结果总结

完整消融实验报告见 [evals/EVALUATION_REPORT.md](./evals/EVALUATION_REPORT.md)。

#### 策略对比（20 题评估集）

| 策略 | Recall@1 | Recall@8 | MRR@8 | nDCG@8 | Cell命中@8 | 延迟(ms) |
|------|----------|----------|-------|--------|-----------|----------|
| Dense | 0.80 | 0.95 | 0.875 | 0.895 | 0.95 | 1,008 |
| **RRF** ⭐ | **0.85** | **1.00** | **0.893** | **0.919** | **1.00** | **829** |
| RRF+Outline | 0.00 ⚠️ | 0.85 | 0.175 | 0.328 | 0.85 | 1,186 |
| RRF+Outline+Rerank | **0.90** | **1.00** | **0.950** | **0.963** | **1.00** | 2,140 |

**关键发现**：
1. ✅ **RRF 最优性价比**：提升 6% Recall@1，达到 100% Recall@8，延迟降低 18%
2. ⚠️ **Outline 召回问题**：Document outline 包含章节标题和关键词，在 Dense/Sparse 两路都得分高，经 RRF 融合后系统性排在精确 text_content 前面，但 outline 只有标题无答案内容，导致 Recall@1 降至 0。根本原因：设计用于文档发现，不适合精确问答。当前已禁用。
3. ✅ **Rerank 最高精度**：Recall@1 达到 0.90，但延迟翻倍（适合高精度场景）

**推荐配置**：
- **日常使用**：RRF（Dense + Sparse 融合，默认策略）
- **高精度场景**：RRF + Rerank（考试辅助、论文写作等）
- **暂不启用**：Outline 召回（需修复后再评估）

### 评估方法

#### 已完成 ✅

**离线评估**：
- 20 题人工标注数据集（开发集 15 + 保留集 5）
- 自动计算指标：Recall@K、MRR、nDCG、命中率、延迟
- 4 种策略消融对比（Dense、RRF、RRF+Outline、RRF+Outline+Rerank）
- 分阶段延迟分析（Embedding、召回、Rerank）

**单元测试**：
- 42 个单元测试覆盖核心组件
- 边界情况和错误处理验证

**性能监控**：
- 每个查询的延迟记录（P50/P95/Max）
- 失败案例追踪

#### 待完成 🔲

**扩展评估集**：
- 从 20 题扩展到 60-100 题
- 添加难度分层（简单/中等/困难）
- 包含多模态查询（图表理解题）
- 跨文档推理场景
- 无答案场景（明确回答"不知道"）

**LLM-as-Judge**：
- Context Precision：检索 chunks 的相关性自动评分
- Context Recall：标准答案需要的证据是否都被检索到
- 未来回答生成的 Faithfulness 检测（幻觉识别）

**在线监控**：
- 查询日志记录与分析
- 用户满意度反馈收集
- A/B 测试框架

**压力测试**：
- 并发 QPS 测试
- 大规模数据索引性能测试（100/1000/10000 文档）
- 长查询和边界情况测试

**人工评估**：
- 定期抽查（每月 50-100 条）
- 多标注者一致性验证
- 发现新的失败模式

### Dense Baseline

原始 `raw query → text_dense cosine Top 8` 基线结果记录在 [evals/results/dense_baseline.json](./evals/results/dense_baseline.json)：

| 指标 | 结果 |
| --- | ---: |
| Recall@1 | 0.80 |
| Recall@3 / Recall@8 | 0.95 / 0.95 |
| MRR@8 | 0.875 |
| nDCG@8 | 0.895 |
| 文件 / 章节 / cell 命中率@8 | 1.00 / 1.00 / 0.95 |

该结果只代表当前 20 题 Notebook 小型基线。延迟受模型 API 和网络影响，以结果文件中的单次运行数据为参考。消融实验结果将记录在 `evals/results/` 下各策略对应的 JSON 文件中。

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
agent.md                # 当前进度与重大变更
```

## 下一阶段

根据 [消融实验结果](./evals/EVALUATION_REPORT.md)，以下是优先级排序的后续工作：

### 短期优化（1-2 周）

1. **扩展评估集到 50-60 题**
   - 增加难度分层（简单/中等/困难）
   - 包含多模态查询（图表理解题）
   - 跨文档推理场景
   - 目标：更准确评估策略效果

2. **优化 Rerank 延迟**
   - 只 rerank top-N 候选（如 top 15→8）
   - 实现结果缓存减少重复 API 调用
   - 考虑批量 rerank 降低延迟
   - 目标：延迟从 2140ms 降至 <1500ms

3. **实现查询日志和基础监控**
   - 记录每次查询的延迟、结果数量、命中情况
   - 失败查询分析
   - 定期生成统计报告

### 中期扩展（1 个月）

1. **接入 PDF/PPT 解析**
   - 实现 PDF 页面、表格、图表解析
   - 设计 PDF 专用切块策略
   - 评估 Outline 在 PDF 场景的文档发现价值
   - 考虑分离 pipeline：Outline 用于文档发现，text_content 用于精确答案

2. **实现 LLM-as-Judge 评估**
   - Context Precision：检索 chunks 相关性自动评分
   - Context Recall：验证标准答案的证据是否完整
   - 为未来回答生成做准备

3. **工程化优化**
   - 查询日志记录与失败分析
   - 实现查询缓存（Redis 或本地）
   - 添加限流和降级策略
   - 完善监控和告警

### 长期规划（2-3 个月）

1. **视觉召回验证**：准备包含图表的测试集，验证 visual_fusion 效果
2. **上下文回取集成**：将相邻 chunk 回取应用到实际检索
3. **PDF 解析**：接入 PDF 文档解析与专用切块策略
4. **回答生成**：接入 DeepSeek 回答层和网页交互
5. **云端迁移评估**：评估 Qdrant Cloud、对象存储、服务部署方案

以 [CHECKLIST.md](./CHECKLIST.md) 为准逐项推进。

## 测试

```bash
uv run pytest -q
uv run ruff check app tests
```

当前测试覆盖 Notebook 解析、切块、Embedding 客户端、Qdrant 存储、索引器、检索器和 RAG HTTP 接口。
