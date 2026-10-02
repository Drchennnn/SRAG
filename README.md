# SRAG：课程资料检索

SRAG 为课程 Notebook、PDF 和 PPTX 建立可增量更新的本地检索库。FastAPI、Qdrant 与 SQLite 在本地运行；Embedding 等模型能力通过 API 调用。计算较重的幻灯片预处理可在 Colab GPU 运行。当前提供检索接口和终端问答试验入口，尚无网页对话。

- [RAG.md](./RAG.md)：当前方法与设计取舍
- [CHECKLIST.md](./CHECKLIST.md)：未完成任务
- [agent.md](./agent.md)：供后续开发使用的简短状态
- [PROGRESS_HISTORY.md](./PROGRESS_HISTORY.md)：历史里程碑

## 当前链路

| 输入 | 处理 | 入库 |
| --- | --- | --- |
| IPYNB | 只读解析 cell，按标题段和 Cell Group 切块；代码与输出保持关联 | Qwen 文本/视觉 Embedding → Qdrant；SQLite 管理文件版本 |
| PDF/PPTX | 按幻灯片页预处理，保留文本、页码、整页截图与资源；可在 Colab 执行 | 本地使用 Qwen API 生成向量包，再导入 Qdrant/SQLite |

正式检索入口接受课程或文件范围，先过滤可见版本，再用文本 Dense+Sparse RRF 召回。明确询问图像内容时，auto 模式优先检索整页视觉向量，并用文本候选补足。返回来源文件、Notebook cell 或幻灯片页码及可用截图路径。查询理解、Rerank、相邻上下文与回答生成尚未接入正式 HTTP 入口；终端试验入口可用 DeepSeek 基于检索结果回答。

历史检索基线包括 Notebook 20 题及 CS5187/CS5495 页码题集，报告与 query 实验见 [评估资产](./evals/README.md)。Notebook RRF Recall@8=1.00；CS5495 正式入口 24 题 Recall@5=0.875，图表细节 1/4。它们仅描述当时索引的来源召回，不代表新增视觉文字后的当前指标或答案质量；后续使用固定跨文件题集验收。

## 当前已入库资料

2026-10-01 本地可见版本快照：15 份文件（7 个 IPYNB、7 个 PDF、1 个 PPTX），Qdrant 中对应 751 个点。下表路径相对于 `COURSE_ROOT`；三门课的目录分别为 `CS5187 Vison and Image/`、`CS5489 Machine learning/`、`CS5495 X AI/`。

| 课程 | 课程目录下的文件 | 版本 |
| --- | --- | --- |
| CS5187 | `CS5187_Vision_and_Image/Week01/Lecture1-1.pdf` | v2 |
| CS5187 | `CS5187_Vision_and_Image/Week02/Lecture2-1.pdf` | v1 |
| CS5187 | `CS5187_Vision_and_Image/Week08/Lecture8-2.pdf` | v1 |
| CS5187 | `Lecture/Week1/Lecture1-1.pptx` | v2 |
| CS5489 | `Lecture1/Lecture1a.ipynb`、`Lecture1/Lecture1b.ipynb` | v1、v1 |
| CS5489 | `Lecture2/Lecture2a.ipynb`、`Lecture2/Lecture2b.ipynb` | v1、v1 |
| CS5489 | `Lecture3/Lecture3a.ipynb`、`Lecture4/Lecture4a.ipynb` | v1、v2 |
| CS5495 | `Lecture2/Lecture2a.pdf`、`Lecture2/Lecture2b.pdf` | v1、v1 |
| CS5495 | `Lecture3/Lecture3a.ipynb`、`Lecture3/Lecture3a.pdf`、`Lecture3/Lecture3b.pdf` | v1、v1、v1 |

这只是本地索引快照；新增、更新或删除后，以 `GET /api/rag/documents` 返回的当前可见版本为准。入库范围和学期仍待人工复核，见 [CHECKLIST.md](./CHECKLIST.md)。

2026-10-02 按需识别试验额外写入 8 个视觉文字派生点（文件数不变），本地总点数为 759。目录接口列文件，不能据此推算向量点数。

## 本地运行

需要 Python 3.11+、uv 和 Docker。模型 Key 只填入本地 .env，不要提交。

~~~bash
uv sync --extra dev
cp .env.example .env
docker compose up -d
uv run uvicorn app.main:app --reload
~~~

至少设置 COURSE_ROOT 和 DASHSCOPE_API_KEY；其他选项见 [.env.example](./.env.example)。Qdrant Dashboard 位于 http://localhost:6333/dashboard，API 健康检查为 http://127.0.0.1:8000/api/health。需要在本地解析 PDF/PPTX 时安装文档可选依赖：uv sync --extra dev --extra documents。

### 索引 Notebook

~~~bash
curl -X POST http://127.0.0.1:8000/api/rag/index \
  -H 'Content-Type: application/json' \
  -d '{"course_id":"CS5489","paths":["/absolute/path/to/Lecture1.ipynb"]}'
~~~

接口接受明确文件路径；目录自动同步和网页上传尚未实现。[检索体验 Notebook](./notebooks/retrieval_end_to_end.ipynb) 可展示发现、切块、索引和 Top K 结果。[切块实验 Notebook](./notebooks/chunk_embedding_experiment.ipynb) 用于检查表示与 API Embedding。

### 预处理并导入 PDF/PPTX

先用 [单文件](./notebooks/cloud_document_preprocess.ipynb) 或 [批处理 Colab Notebook](./notebooks/cloud_batch_preprocess.ipynb) 生成 deck.json、chunks.json、slides/ 和 assets/。批处理工具：

~~~bash
uv run python tools/build_course_manifest.py /path/to/courses
uv run python tools/pack_course_batch.py /path/to/courses
~~~

批次 ZIP 可上传到 Google Drive；Notebook 默认目录前缀为 /content/drive/MyDrive/Colab Notebooks/SRAG/。Colab 只做解析与切块。下载并核对预处理包后，在本地向量化：

~~~bash
uv run python -m app.rag.slide_embedder /path/to/slidedeck-bundle
~~~

命令默认生成文本和整页视觉向量，结果放在 Git 忽略的 data/embeddings/；--text-only 可仅处理文本。向量包与原始文件哈希核对后入库：

~~~bash
curl -X POST http://127.0.0.1:8000/api/rag/slides/index \
  -H 'Content-Type: application/json' \
  -d '{"course_id":"CS5187","source_path":"/absolute/path/to/Lecture1.pdf","bundle_dir":"/absolute/path/to/bundle","embeddings_path":"/absolute/path/to/embeddings.json"}'
~~~

已有相同内容会跳过；需要重建同一文件时传 "force": true。旧版 .ppt 需先转换为 .pptx。

### 检索

~~~bash
curl -X POST http://127.0.0.1:8000/api/rag/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"贝叶斯公式是什么？","scope":{"course_ids":["CS5489"],"document_ids":[]},"top_k":5}'
~~~

retrieval_mode 可选 auto（默认）、text、visual；响应的 mode_used 指出实际模式。query_construction 默认为 raw，保持原问题检索；显式设为 contextual 并传入最多 6 条 history 后，含指代词且缺少具体主题的追问会补入最近一条用户问题。响应的 retrieval_query 和 query_method 可用于核对实际查询，原始 query 保持不变。历史中的助手回答和文件范围不会被用于推断检索权限。此阶段不调用 DeepSeek 改写，长对话摘要也未实现。得分只适合同次检索排序，不是正确率。API 还提供索引任务查询、文档列表及删除；具体请求结构见 [app/schemas/rag.py](./app/schemas/rag.py) 和 [app/api/rag.py](./app/api/rag.py)。

### 终端对话试验

先启动本地 Qdrant，并在 `.env` 配置 Qwen 和 DeepSeek Key。示例把问答限定在已入库的 CS5187 课件；省略 `--file` 则检索整门课：

~~~bash
uv run python -m tools.chat --course CS5187 --file Lecture2-1.pdf --contextual
~~~

输入 `/files` 查看文件编号，`/file 编号` 改为单文件，`/all` 回到整门课，`/course CS5489` 切换课程；`/scope` 查看当前范围，`/context on|off` 切换追问补全，`/quit` 退出。范围切换清空历史和主题状态。终端默认开启规则追问补全并使用文本检索；`--raw` 可关闭以作对比。正式 HTTP 检索仍默认 raw；`/mode auto` 可实验视觉召回。

知识问答使用本轮最多 6 条文字证据并显示页码/cell。追问保留最近的知识主题，支持“这两个方向的核”等泛称和上轮引入的 Gx/Gy 等短符号；明确命名的新主题保留原问题。每轮重新检索可见版本，历史答案和旧来源记录不直接作为证据。DeepSeek 仍只看到最近两轮问答，无历史摘要。回答的引用编号会校验，事实与引用是否匹配仍需人工核对。

问题里明确写出文件名/讲义编号时，本轮只在该文件检索；歧义要求指定。当前仅支持阿拉伯数字页码直读，可沿用用户前一轮明确指定的文件，不从混合命中猜文件。文件焦点受原范围和版本约束，不永久修改用户范围；日志的 `effective_scope` 记录实际范围。

历史问题直接读完整会话；目录/数量问题直接读 SQLite 可见文件清单，不调用模型或向量检索。`总结 lecture2-1.pdf` 在当前范围定位文件，读取该可见版本全部文字块；同名或 PDF/PPTX 歧义要求指定文件。长文件按 14000 字符分批归纳，保留原来源编号，输出页/块覆盖及文字缺失页。当前是单文件总结，不自动生成全课程复习安排。

公式/图表问题按需用 `qwen3-vl-plus` 读取已注册截图（每轮最多两页），转写公式、符号和图示；DeepSeek 仍接收文字。复用 `.env` 的 Qwen Key，模型名为 `DASHSCOPE_VISION_MODEL`。缓存按文件版本、图片哈希、模型和提示词版本区分，保存在忽略的 `data/visual_text/`；派生文字另做 dense+sparse 向量入库，不覆盖原始表示，同来源的新转写替换旧派生点。总结只复用已有视觉缓存，不自动识别全部页面。指定页面可提前补充：

~~~bash
uv run python -m tools.visual_text --course CS5187 --file Lecture2-1.pdf --pages 16
~~~

视觉识别标记为“未经人工核对”；失败时显示提示并使用原文字。回答区分课件与通用补充，当前无联网搜索。DeepSeek 回答关闭思考模式，空输出/截断仅重试一次，仍失败则报错。逐轮日志位于忽略的 `data/chat_sessions/`。

目前“文件讲了什么”、中文页码与连续讨论对象定位仍不稳定。下一步优先完善通用 query 定位，再固定跨文件题集验证；暂不全量扩库、重切或更换 Embedding。任务见 [CHECKLIST.md](./CHECKLIST.md)，最新实测见 [PROGRESS_HISTORY.md](./PROGRESS_HISTORY.md)。

## 评估与测试

固定题集、运行脚本和当前逐题结果保存在 [评估资产](./evals/README.md)。例如：

~~~bash
uv run python -m app.evaluation.slide_retrieval
uv run python -m evals.run_query_followup
uv run python -m evals.run_query_followup --cases evals/query_notebook_holdout_cases.jsonl --output evals/results/query_notebook_holdout_current_rule.json --oracle-missed
uv run python -m evals.run_query_intent
uv run pytest -q
uv run ruff check app evals tools tests
~~~

评估脚本可能调用付费模型 API；测试命令不要求运行中的 Qdrant。原始课程资料、预处理包、向量包和本地数据库均位于 Git 忽略的 data/ 或用户指定目录。
