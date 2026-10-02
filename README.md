# SRAG：课程复习助手

面向课程 IPYNB、PDF 和 PPTX 的检索与问答项目，帮助理解知识点、公式和步骤，并给出课件页码或 Notebook cell 来源。服务与数据本地运行，模型全部通过 API 调用。

## 已实现

- 结构化切块、文本/视觉向量化、Qdrant 检索与 SQLite 文档版本管理。
- 终端全库或指定课程/文件问答、连续追问、页码定位和单文件总结。
- 按需识别图内公式/图表，输出文字解释与来源引用。

终端已完成跨格式回归验收，结果见 [评估说明](./evals/README.md)。当前 HTTP 只提供检索与索引，网页对话尚未实现。

## 启动

需要 Python 3.11+、uv 和 Docker。

```bash
uv sync --extra dev
cp .env.example .env  # 首次配置；已有 .env 时不要覆盖
```

在 `.env` 填入 `COURSE_ROOT`、`DASHSCOPE_API_KEY` 和 `DEEPSEEK_API_KEY`；其他配置见 [.env.example](./.env.example)。

```bash
docker compose up -d
uv run python -m tools.chat                         # 全部可见课程
uv run python -m tools.chat --course CS5187          # 指定课程
```

常用命令：`/files` 列文件，`/file 编号` 选文件，`/all` 回到当前课程，`/global` 回到全库，`/scope` 查看范围，`/quit` 退出。

启动 API：

```bash
uv run uvicorn app.main:app --reload
```

接口文档：http://127.0.0.1:8000/docs；向量库界面：http://localhost:6333/dashboard。

## 导入资料

- **IPYNB**：在接口文档调用 `/api/rag/index`，传课程 ID 和文件路径；[体验 Notebook](./notebooks/retrieval_end_to_end.ipynb) 可展示整个流程。
- **PDF/PPTX**：用 [单文件](./notebooks/cloud_document_preprocess.ipynb) 或 [批量 Colab Notebook](./notebooks/cloud_batch_preprocess.ipynb) 预处理，下载输出包后运行：

```bash
uv run python -m app.rag.slide_embedder /path/to/slidedeck-bundle
```

然后通过 `/api/rag/slides/index` 导入向量包。Colab 默认 Drive 前缀为 `/content/drive/MyDrive/Colab Notebooks/SRAG/`；本地解析需额外安装 `uv sync --extra dev --extra documents`。

当前可见文件以 `/api/rag/documents` 或终端 `/files` 为准。模型配置与资料不提交仓库；本地日志、缓存和报告保存在 `data/`。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [RAG.md](./RAG.md) | 架构、模块边界、处理方法与关键参数 |
| [evals/README.md](./evals/README.md) | 评估工具、命令、指标、标准和历史结果 |
| [CHECKLIST.md](./CHECKLIST.md) | 当前与后续任务 |
| [agent.md](./agent.md) | 开发状态和文档维护规则 |
| [PROGRESS_HISTORY.md](./PROGRESS_HISTORY.md) | 完整开发过程与决策记录 |

## 维护约定

- 只保留项目成果、运行/导入入口和文档导航，命令需与代码一致。
- 架构参数、评估细节及开发历史引用对应文档，不在此展开；功能变化更新原段落。
