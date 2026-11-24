# SRAG：课程复习助手

面向课程 **PDF**、**PPTX** 与 **IPYNB** 资料的本地优先检索、多模态核对与问答工具。学生在本地浏览器工作区导入课程讲义、幻灯片与实验 Notebook，用自然语言提问并点开原页/裁图/保存输出逐项核对，沿单课程可溯源知识图谱在概念定义与实验结果间跳转，并在资料换版后区分当前版本与历史引用。

- **当前交付状态**：A–G（M2 双轨放行）、J（多文件策略定型）、H（单课程知识图谱）、R（冻结人工复核）均已通过并合入 `main`，M4 最终交付（K 部）与验收见 [产品目标与交付路线](./docs/产品目标与交付路线.md)、[K01 发布验收设计](./docs/K/K01-发布验收设计.md) 与 [任务清单](./CHECKLIST.md)。
- **默认运行定位**：以浏览器工作区 `/workspace/` 为第一产品入口；默认仅依赖本机 Python 与 SQLite FTS5，**无需 Docker、无需外部 Qdrant 服务**即可启动与完整离线浏览。

## 5 分钟快速上手（零 Docker 冷启动）

### 1. 安装环境

需要 Python `>=3.11`（实测 `3.14.7`）、`uv`，以及可选的 LibreOffice（用于 PPTX 整页与表格/插图渲染）：

```bash
uv sync --extra dev --extra documents
test -f .env || cp .env.example .env
```

> 仅执行离线文字/PDF/Notebook 流程可不装 LibreOffice（PPTX 整页会平滑回退原生文本）；安装 LibreOffice 后可通过 `SRAG_SOFFICE=$(command -v soffice)` 启用真实幻灯片整页与裁图渲染。

### 2. 离线冷启动（零外部密钥立即可用）

使用仓库自带的可分享样例课程 [examples/demo-course](./examples/demo-course) 从空数据目录启动：

```bash
rm -rf ./data/clean-demo && mkdir -p ./data/clean-demo
COURSE_ROOT="$PWD/examples/demo-course" \
DATA_DIR="$PWD/data/clean-demo" \
SQLITE_PATH="$PWD/data/clean-demo/srag.sqlite3" \
WORKSPACE_COURSE_IDS=demo-course \
SRAG_SOFFICE="$(command -v soffice || true)" \
WORKSPACE_ANSWER_ENABLED=false \
SOURCE_VISION_ENABLED=false \
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

在浏览器打开 `http://127.0.0.1:8000/workspace/`（健康检查：`http://127.0.0.1:8000/api/health`）：

1. 在左侧「导入资料」选择课程 `demo-course`，上传 `examples/demo-course/` 下的 PDF（*Axis-LoRA*、*FlashAttention-3*）、PPTX 与 `v1/Lab-01-Retrieval-Bench.ipynb`；
2. 展开左侧「资料目录」逐层浏览页、表格、公式、局部图与 Notebook 保存输出，或打开「原件阅读」预览；
3. 展开左侧「概念导航」（H 单课程知识图谱，零外部模型构建），点击概念卡片分别跳转「讲义定义原页」与「实验 Cell 及保存输出图」；
4. 在输入框搜索关键词或提问，离线模式下自动给出可核对的原文摘录（`excerpt_fallback`）与图文来源卡。

### 3. 开启完整在线回答与多模态读图

在 `.env` 配置以下后端环境变量后重启服务即可启用真实中文综合回答与定向读图：

```dotenv
WORKSPACE_COURSE_IDS=demo-course
WORKSPACE_ANSWER_ENABLED=true
DEEPSEEK_API_KEY=你的DeepSeek密钥
DEEPSEEK_CHAT_MODEL=deepseek-flash
DEEPSEEK_QUERY_MODEL=deepseek-flash
WORKSPACE_INPUT_TOKEN_LIMIT=32768
WORKSPACE_RETRIEVAL_STRATEGY=s0
WORKSPACE_RETRIEVAL_BUDGET=6
WORKSPACE_RETRIEVAL_INCLUDE_ADJACENCY=true

SOURCE_VISION_ENABLED=true
DASHSCOPE_API_KEY=你的DashScope密钥
DASHSCOPE_VISION_MODEL=qwen3-vl-plus
```

典型体验流程：提问论文公式/三线表（点「依据」与「回原件」看归一化高亮）→ 聚焦 `Lab-01` 多图输出 Cell 提问并从澄清按钮选中目标输出 → 勾选「更新当前资料的版本」上传 `v2/Lab-01-Retrieval-Bench.ipynb` → 对比最新版与旧版、回看旧回答 `v1` 原件与失效概念芯片优雅降级。

---

## 工作区核心能力与离线/在线边界

| 功能模块 | 无外部密钥（离线模式） | 开启 DeepSeek + `qwen3-vl-plus`（在线模式） |
| :--- | :--- | :--- |
| **三格式导入与版本发布** | 完整可用（SQLite 事务发布、FTS5、同内容复用、失败保留旧版） | 完整可用 |
| **目录树与原件阅读器** | 完整可用（PDF 原页/局部裁图、PPTX 需 `SRAG_SOFFICE`、Notebook 保存输出） | 完整可用 |
| **单课程概念导航（`#kg-panel` 与消息芯片）** | **完整可用**（离线确定性原文构边，无可靠边平滑回退常规检索） | 完整可用（并在回答消息下方挂载 `.kg-concept-chip`） |
| **检索与问答** | 关键词全文查找、单文件摘录与原生表格/定位展示（标记 `excerpt_fallback`） | 中文综合回答、按需双语改写、定义与实验对照、多轮指代与跨版对比 |
| **图表/公式视觉读取** | 展示原图并明确标注「图像未读取」，不猜测图内数值 | 点击「问这张图」或定向问图触发真实 `qwen3-vl-plus` 实读并缓存绑定原图 SHA-256 |

## 检索策略与图谱开关说明（J03 / H03 定型结论）

`WORKSPACE_RETRIEVAL_STRATEGY` 合法取值为以下 9 项：`fts`、`s0`、`bilingual`、`expanded`、`s2a`、`s2b`、`jev`、`s2a_jev`、`vector`（完整配置见 [.env.example](./.env.example)）：

| 策略 / 开关 | 默认状态 | 适用场景与实测定型建议 |
| :--- | :--- | :--- |
| **`s0`**（推荐默认底座） | 建议显式设为 `s0`（代码默认 `fts`，`BUDGET=6`） | FTS5 + 结构邻接（`WORKSPACE_RETRIEVAL_INCLUDE_ADJACENCY=true`）+ 按需双语改写；零 Qdrant / 零 Jev 强依赖 |
| **`vector`（`S3` 纯图与语义增强）** | 默认关闭 | **必须同时设置** `WORKSPACE_RETRIEVAL_STRATEGY=vector` 与 `WORKSPACE_VECTOR_RETRIEVAL_ENABLED=true`，并指定 `QDRANT_LOCAL_PATH`、`DASHSCOPE_API_KEY`、`SOURCE_VISION_ENABLED=true`、`WORKSPACE_S3_MAX_VISUAL_READS=2`；专治纯图/少字 PPTX 召回与前移 D03 实读（切勿在 `s0`/`fts` 下仅开布尔开关） |
| **`s2a` / `s2b`（有界 2 步检索）** | 显式可选 | 无共享概念词的跨文件 2-hop：`s2a` 为零外呼字面锚扩展，`s2b` 为有界 1 次 LLM Planner |
| **`jev` / `expanded` / `s2a_jev`（`S1`）** | **本版保持关闭** | 48 KB 分批修复已消除 `input_limit`，但远端 `Zev-1` RTT 导致 `p95` 增量远超 `+8s` 阈值且低置信回退 `>20%` |
| **`WORKSPACE_KG_RETRIEVAL_ENABLED`** | **默认 `false`** | 课程概念导航界面默认启用；将图谱一跳作为问答候选增强时可设为 `true`（可提升零重叠跨文件召回，但在 6 槽预算下对纯背景提及题存在槽位权衡） |

## 数据安全与模型外传边界

1. **本机存储**：原始课程文件字节、SQLite 版本库（`SQLITE_PATH`）、会话与消息记录、原件渲染缓存（`DATA_DIR/source_evidence_cache`）、视觉读图缓存（`DATA_DIR/source_vision_cache`）及可选本地向量库（`QDRANT_LOCAL_PATH`）全部保留在本机，不上传任何第三方存储；前端网页仅通过同源 Cookie 与来源 ID 访问已登记资源。
2. **模型出站数据**：
   - 当 `WORKSPACE_ANSWER_ENABLED=false` 且 `SOURCE_VISION_ENABLED=false` 时，**零文本与零图片出站**；
   - 开启 `WORKSPACE_ANSWER_ENABLED=true` 后，仅将当前问题、有限上下文及本轮实际读到的必要文字片段送往配置的 DeepSeek 端点；
   - 开启 `SOURCE_VISION_ENABLED=true` 后，仅在用户触发定向读图（或 `S3` 命中纯图页）时将选定单张图片字节与当前问题送往配置的 DashScope 多模态端点；
   - 所有模型 API Key 仅由后端从 `.env` 读取，绝不下发至浏览器。

## 已知限制（如实披露）

- **扫描 PDF 默认不自动后台 OCR**：基础导入零阻塞优先发布原生文字与页级定位，纯扫描页需显式运行 `uv run python -m tools.workspace_reading ocr --course <COURSE_ID> --document <DOCUMENT_ID>` 增强；
- **复杂无框图形与合并单元格三线表**：无法确认精确坐标时保守回退整页（`parent_page_fallback`）与原生页面读取，避免错框误导；
- **离线无模型模式下的中文词法与严格对照边界**：离线零密钥模式底层使用 SQLite FTS5 `unicode61` 分词，连续未加空格的中文长短语需按概念词或空格分隔检索（在线 `s0` 模式由按需双语改写自动补全）；同时「定义+实验」跨文件结构验证契约要求实验记录包含可核对的编号与测试条件字段，未满足结构契约的补充陈述会被保守过滤并提示缺项；
- **需求池未包含能力**：桌面安装包、本地文件夹自动监听、MCP / Skill 接入及完整检索轨迹可视化界面均留在需求池，本交付版本不包含。

## 兼容终端、CLI 与旧向量入口

除默认网页工作区 `/workspace/` 外，仓库仍保留以下 CLI 与兼容 HTTP 接口：

- **工作区共享终端会话**：`uv run python -m tools.workspace_chat --help`（复用同一持久会话与权限校验）
- **版本化顺序阅读与扫描 OCR**：`uv run python -m tools.workspace_reading read --course <COURSE_ID> --document <DOCUMENT_ID>`（支持 `ocr` 子命令）
- **离线定义与实验比较 CLI**：`uv run python -m tools.evidence_answer --course <COURSE_ID> "问题" --trace`（需设置 `SRAG_STRUCTURE_COURSES`）
- **本地导入与结构检索 API**：`POST /api/rag/local/import`、`POST /api/rag/local/search`、`POST /api/rag/structure/search`、`POST /api/rag/local/evidence/current`、`POST /api/rag/local/migrate`、`POST /api/rag/local/migrate/bundle`
- **旧版向量索引与 Qdrant Docker（可选）**：仅使用旧 `/api/rag/index`、`/api/rag/slides/index` 与外部 Qdrant 面板（`http://localhost:6333/dashboard`）时启动 `docker compose up -d`，终端交互入口为 `uv run python -m tools.chat`（详细见 [RAG.md](./RAG.md) 与 [notebooks/](./notebooks/)）。
- **可重放验收与质量验证**：
  ```bash
  uv run ruff check app evals tools tests
  SRAG_SOFFICE="$(command -v soffice || true)" uv run pytest -q
  SRAG_SOFFICE="$(command -v soffice || true)" uv run python docs/K/scripts/run_k02_public_acceptance.py
  ```

## 文档导航

| 文档 | 内容 |
| :--- | :--- |
| [产品目标与交付路线](./docs/产品目标与交付路线.md) | 产品目标、学习场景、M0–M4 里程碑与第 5 节验收清单 |
| [K01 发布验收设计](./docs/K/K01-发布验收设计.md) | C1–C7 整机验收方案、公开样例与隔离评测协议 |
| [K02 交付包与整机质量报告](./docs/K/K02-交付包与整机质量报告.md) | 公开样例实测、固定题集与未用课程脱敏统计、已知限制 |
| [当前执行交接](./docs/执行交接.md) | 当前任务状态、工作区基线与交接记录 |
| [RAG.md](./RAG.md) | 目标架构、行内证据契约与模块边界 |
| [evals/README.md](./evals/README.md) | 评估工具、指标口径与历史基线 |
| [任务与审核清单](./CHECKLIST.md) | 分部依赖、小任务交付与独立门审记录 |
| [agent.md](./agent.md) · [PROGRESS_HISTORY.md](./PROGRESS_HISTORY.md) | 协作约定与完整历史过程记录 |
