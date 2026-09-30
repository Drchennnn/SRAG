# SRAG：课程资料检索

SRAG 为课程 Notebook、PDF 和 PPTX 建立可增量更新的本地检索库。FastAPI、Qdrant 与 SQLite 在本地运行；Embedding 等模型能力通过 API 调用。计算较重的幻灯片预处理可在 Colab GPU 运行。当前提供检索接口，尚无自动回答或网页对话。

- [RAG.md](./RAG.md)：当前方法与设计取舍
- [CHECKLIST.md](./CHECKLIST.md)：未完成任务
- [agent.md](./agent.md)：供后续开发使用的简短状态
- [PROGRESS_HISTORY.md](./PROGRESS_HISTORY.md)：历史里程碑

## 当前链路

| 输入 | 处理 | 入库 |
| --- | --- | --- |
| IPYNB | 只读解析 cell，按标题段和 Cell Group 切块；代码与输出保持关联 | Qwen 文本/视觉 Embedding → Qdrant；SQLite 管理文件版本 |
| PDF/PPTX | 按幻灯片页预处理，保留文本、页码、整页截图与资源；可在 Colab 执行 | 本地使用 Qwen API 生成向量包，再导入 Qdrant/SQLite |

正式检索入口接受课程或文件范围，先过滤可见版本，再用文本 Dense+Sparse RRF 召回。明确询问图像内容时，auto 模式优先检索整页视觉向量，并用文本候选补足。返回来源文件、Notebook cell 或幻灯片页码及可用截图路径。查询理解、Rerank、相邻上下文与回答生成尚未接入正式入口。

当前已有 20 题 Notebook 检索基线，以及 CS5187/CS5495 的页码召回题集和逐题报告，见 [evals](./evals/)。Notebook 的 RRF Recall@8 为 1.00；CS5495 跨课程正式入口的 24 题 Recall@5 为 0.875，图表细节仅命中 1/4。这些指标只衡量来源召回，不代表回答质量；题集已用于开发分析，后续调整需用新资料验收。

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

retrieval_mode 可选 auto（默认）、text、visual；响应的 mode_used 指出实际模式。得分只适合同次检索排序，不是正确率。API 还提供索引任务查询、文档列表及删除；具体请求结构见 [app/schemas/rag.py](./app/schemas/rag.py) 和 [app/api/rag.py](./app/api/rag.py)。

## 评估与测试

固定题集、运行脚本和历史逐题结果保存在 [evals](./evals/)。例如：

~~~bash
uv run python -m app.evaluation.slide_retrieval
uv run pytest -q
uv run ruff check app evals tools tests
~~~

评估脚本可能调用付费模型 API；测试命令不要求运行中的 Qdrant。原始课程资料、预处理包、向量包和本地数据库均位于 Git 忽略的 data/ 或用户指定目录。
