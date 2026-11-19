# SRAG：课程复习助手

面向课程 IPYNB、PDF 和 PPTX 的检索与问答项目，帮助理解知识点、公式和步骤，并给出课件页码或 Notebook cell 来源。服务与数据本地运行，模型全部通过 API 调用。

## 已实现

- PDF、PPTX、IPYNB 可在本地发布版本、保留来源结构并通过 SQLite FTS5 查找原生文字；这条路径可以独立启动。
- 结构化切块、文本/视觉向量化、Qdrant 检索与 SQLite 文档版本管理。
- 终端全库或指定课程/文件问答、连续追问、页码定位和单文件总结。
- 按需识别图内公式/图表，输出文字解释与来源引用。

终端已完成跨格式回归测试；10 月 17 日的候选版本未达到答案质量验收门槛，待人工复核。当前通过终端对话，HTTP 提供检索与索引。11 月 19 日更新的[目标架构](./RAG.md)优先设计本地导入即搜、按需读图和回答正文中的原件预览；范围及验收见[设计清单](./CHECKLIST.md)。

## 启动

需要 Python 3.11+、uv 和 Docker。

```bash
uv sync --extra dev
test -f .env || cp .env.example .env  # 已有配置时保留
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

## 本地导入与文字检索

使用 Python 3.11+ 和带 FTS5 的 SQLite。PDF/PPTX 解析沿用 `documents` 可选依赖。

```bash
uv sync --extra documents
COURSE_ROOT="$PWD/docs/A/样例" DATA_DIR="$PWD/data/local-demo" \
  SQLITE_PATH="$PWD/data/local-demo/library.sqlite3" uv run uvicorn app.main:app
```

```bash
curl -sS http://127.0.0.1:8000/api/rag/local/import \
  -H 'Content-Type: application/json' \
  -d '{"course_id":"signal-demo","paths":["Signal-reference.pdf","Signal-slides.pptx","Signal-lab.ipynb"]}'
curl -sS http://127.0.0.1:8000/api/rag/local/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"lookahead","scope":{"course_ids":["signal-demo"]}}'
```

`paths` 接受 `COURSE_ROOT` 内的文件或目录。结果逐文件记录版号、发布代次、结构及可搜行数。扫描页保留页号并标待识别。相同内容重导复用既有版号和原件；换路径关联旧资料时传单个文件和 `document_id`。新路径使用本机 SQLite 与解析器，Qdrant 和模型凭据只供旧向量/问答路径使用。

在接口文档可用 `POST /api/rag/local/migrate` 显式回填旧库的当前版；`POST /api/rag/local/migrate/bundle` 接受旧 deck/chunks 包、课程、原路径和已核对的完整原件 SHA-256。 `RAG_SEARCH_BACKEND=local` 可让旧 `/api/rag/search` 的文本请求查本地索引，设置为 `legacy` 即切回。缺等价当前版的旧向量会被 SQLite 隐藏。[发布设计与回退](./docs/B/B01-发布设计.md)

## 范围与结构检索

启动服务时将 `SRAG_STRUCTURE_COURSES` 设为获准检索的课程 ID（逗号分隔）。请求同时提供课程范围，可用 `document_ids`、`file_names`、一基页码或零基 Cell 位置缩小范围。默认启用一跳结构邻接，响应记录实际版本、候选的引入原因与覆盖范围。

```bash
curl -sS http://127.0.0.1:8000/api/rag/structure/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"lookahead","scope":{"course_ids":["signal-demo"]},"top_k":8}'
```

设置 `include_adjacency:false` 可用同一预算执行纯 FTS 对照。[结构检索与复现](./docs/C/C03-结构邻接与覆盖.md)

## 原件预览

服务端设置 `SOURCE_EVIDENCE_COURSE_IDS` 为获准预览的课程 ID（逗号分隔）。`POST /api/rag/local/evidence/current` 接收来源三元组、原件 SHA-256、发布代次和展示规格。可预览 PDF/PPTX 整页、可靠对象裁区和 Notebook 的指定已保存输出。未配置课程时入口关闭。PPTX 整页依赖 LibreOffice，可用 `SRAG_SOFFICE` 指定可执行文件。没有转换器时返回原生定位和可用正文。[原件预览与复现](./docs/D/D02-版本化原件与裁图.md)

## 旧向量导入

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
| [当前执行交接](./docs/执行交接.md) | 当前可领取任务、接手工作区、基线命令与最近审核证据 |
| [产品目标与交付路线](./docs/产品目标与交付路线.md) | 最终使用体验、必交功能、阶段交付物与验收条件 |
| [RAG.md](./RAG.md) | 11 月 19 日目标架构、行内证据契约及现有实现 |
| [evals/README.md](./evals/README.md) | 评估工具、命令、指标、标准和历史结果 |
| [任务与审核清单](./CHECKLIST.md) | 分部依赖、承接角色、小任务交付物、审核条件与状态 |
| [agent.md](./agent.md) | 开发状态和文档维护规则 |
| [PROGRESS_HISTORY.md](./PROGRESS_HISTORY.md) | 完整开发过程与决策记录 |

## 维护约定

- 只保留项目成果、运行/导入入口和文档导航，命令需与代码一致。
- 架构参数、评估细节及开发历史引用对应文档，不在此展开；功能变化更新原段落。
