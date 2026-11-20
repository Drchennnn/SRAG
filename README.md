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

请求同一入口时可用 `read_intent=directed_question` 并提供 `question`，或用 `read_intent=generic_transcription` 转写眼前的图。实际模型读取需服务端同时配置 `SOURCE_VISION_ENABLED=true` 和 DashScope 凭据。响应分别记录 `display`、`reads` 和 `external_call_count`。通用转写经版本校验可写入 FTS，定向读图保存在对应问句的缓存。[模型、缓存和复现](./docs/D/D03-定向读图与回写.md)

## 浏览器课程工作区

```sh
WORKSPACE_COURSE_IDS=signals .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1
```

在本机打开 `http://127.0.0.1:8000/workspace`，选择已授权的课程，导入 PDF、PPTX 或 Notebook。
资料库可展开到页、Cell 和对象。会话在刷新后恢复，卡片动作“问这张图”和文字“刚才第二张图”均在所属消息内解析。
普通首问可在问题中指定文件和页、Cell 或图序，例如“实验.ipynb Cell 2 第 2 张图的横轴单位是什么？”。能唯一定位时，系统读取指定图或原生表格。位置不清或读取不可用时，回答会提示缺少哪一步。
PDF 的独立图片只在能核对原件边界时出现在页下。可以直接用“右侧图”这样的相对位置提问，从裁图回到同版原页。此前导入的 PDF 可用同一个原件重导以补登独立图片，已有页级引用沿用原版号。扫描页与无法区分的图片会提示缺少定位。
核对原表或具体列时，回答可同时附上保留表头和单位的原表裁图。表格原生内容、裁图和引用都来自同一版的同一对象。PPTX 裁图需要 LibreOffice。预览上的视觉状态只记录图片是否实际读取。
“当前第 2 张图”按当前原件里的图序，“刚才第 2 张图”按上一条展示的顺序。比较已有引用的上一版与当前版时，可从同一条回答分别打开两份原件。[图序与版本验收](./docs/G/G07-图序与版本.md)
设置 `WORKSPACE_COURSE_IDS` 可用逗号分隔多个课程。模型关闭时仍可搜索原文和查看原件，图片会标明读取状态。
要在网页生成答案，在 `.env` 设置 `WORKSPACE_ANSWER_ENABLED=true` 和现有的 `DEEPSEEK_API_KEY`，重启后提问。生成会把当前问题和实际读取的必要文字发送给所配置的 DeepSeek 服务。点击引用可核对本轮来源。配置不可用时页面标记摘录回退。 [真实网页复验](./docs/G/G16-网页回答与续问.md)

默认检索策略为 `fts`。在 `.env` 可用 `WORKSPACE_RETRIEVAL_STRATEGY` 显式选择 `bilingual`、`jev` 或 `expanded` 并重启。评分策略使用 `JEV_API_KEY`，默认服务 `https://api.zleap.com/v1`、模型 `Zev-1`。短文件全文及显式页可直接读取本地资料。[Jev 实测与适用范围](./docs/G/G17-双语与证据评分对照.md)

输入框使用「发送」提问。「总结此文件」与「查找原文」只执行当次操作，并保留聊天草稿。点目录、回答引用或图卡可开关原件侧栏，窄窗口以抽屉显示，关闭后保留聊天位置。侧栏显示引用版本、原页或裁区，并标记图像读取状态。PPTX 整页不可用时仍可查看原生文字。[浏览器回放](./docs/G/G18-网页回放.md)

`python -m tools.workspace_chat --help` 提供使用同一会话服务的终端入口。[会话及原件权限](./docs/E/E02-会话服务.md)

从空库导入到消息里显示原件的浏览器演示可复跑。[M1 与两种网页追问](./docs/E/E04-M1.md)

用单独工作树和空数据目录试用选中文件后的中文多页问答、图序追问和换版回看。[G19 独立实测与启动命令](./docs/G/G19-独立整审与试用.md)

## 跨文件补证据

已发布的 PDF、PPTX、Notebook 可用 `python -m tools.workspace_reading read --course COURSE_ID --document DOCUMENT_ID` 查看有版本和原文偏移的顺序阅读。扫描 PDF 用同模块的 `ocr` 子命令在基础发布后按需增强。 [使用与复现](./docs/G/G15-顺序阅读与扫描增强.md)

在课程工作区选择资料范围，用普通“问答”同时提问定义和对应的实验验证。系统先读取本轮命中原文，缺实验时最多补查一次。点击回答的“依据”可查看当时引用的文件、版本和页或 Cell。资料换版后仍可从历史回答打开原引用。证据不足时，回答会说明哪一项缺少可核对原文。

同时要求讲义、表格与保存的实验输出时，答案会分别列出实际采用的原件。表格的指标、单位和实验条件与输出一致时才给数值比较。[三类原件演示](./docs/G/G08-三类原件.md)

`python -m tools.evidence_answer` 从本地 SQLite 生成带原文引用的定义与实验比较。单跳与补查使用同一个 `--top-k`，补查从已读定义取字面词，最多查一次。配置 `SRAG_STRUCTURE_COURSES` 后可用 `--course`、`--file` 缩小范围，用 `--trace` 检查每跳、缺项和候选取舍。[公开样例与命令](./docs/F/F03-答案依据与降级.md)

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
