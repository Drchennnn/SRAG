# 课程资料 RAG：本地存储 + 外部 API 推理的架构草案

> 状态：Notebook 数据生命周期、Dense+Sparse RRF 检索和 20 题消融基线已实现；Rerank、查询理解、视觉召回与上下文扩展仍待接入正式查询入口。PDF/PPTX 已有解析和切块原型，尚未接入索引链路。

## 目标与边界

- 首期以课程 `ipynb` 为主要资料，支持指定课程、文件和章节的可溯源检索；PDF/PPTX 原型需通过质量验证并接入统一索引后才算正式支持。
- 支持资料的新版本、发布日期和下架。未到发布日期的资料不进入可查询范围。
- 所有模型推理（embedding、查询改写、重排、OCR、图像理解和最终问答）都调用外部 API，不在本地或自建服务器下载、加载或服务模型权重。本地/云端只承载文件存储、SQLite、Qdrant、无模型解析工具和编排代码。
- 先在本地电脑上完成解析、检索链路搭建和评估；成熟后再评估是否把 Qdrant 和编排服务迁往云端。当前云服务器为 4 核、4 GB 内存、40 GB SSD，不满足 [Milvus Standalone 的 8 GB 内存要求](https://milvus.io/docs/prerequisite-docker.md)。即便模型均走 API，仍需评估 Qdrant 内存、磁盘、API 延迟、限流、费用和网络稳定性。
- `COURSE_ROOT` 指向候选资料来源，不代表所有文件都应入库。导入采用明确的文件白名单，排除虚拟环境、缓存、训练数据集和无关图片。
- 全部模型推理走 API 意味着课件内容会发送给阿里云，查询理解和后续回答生成还会把问题及检索证据发送给 DeepSeek；这是已知取舍。需要妥善保管 API Key（不在客户端或仓库中明文存储），记录发送范围，并处理限流、重试与 API 不可用时暂停任务的策略。

## 数据流

```text
选定资料（当前正式支持 ipynb；PDF/PPTX 为原型）
  → 文件登记与版本检查
  → 按格式解析为统一的章节 / 内容块 / 资源
  → Notebook 标题段、Cell Group 与自适应 Chunk
  → Qwen 文本 dense+sparse 与图片融合向量 → Qdrant 索引
  → 查询范围过滤 → 混合召回 → 调用 Qwen 重排 API → 回取相邻 cell 和原始资源
  → 基于证据回答并引用文件、页码或 cell
```

原始文件和图片资源保留在文件存储中；SQLite 保存文件清单、版本、SectionRecord、Cell Group 与资源关系；Qdrant 保存 document outline、文本 chunk 和图片融合表示。Notebook 不预先复制大型父块正文，后续根据 `section_id`、相邻 chunk 和 cell 关系重组上下文。

## 各阶段方法

| 阶段                            | 首期采用的方法                                                                                                                                                                                                                                                                                                                                  | 后续扩展或验证点                                                                                                                                                                                                                                                                                           |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. 文件登记                     | 只导入选定课程文件；计算内容哈希，记录课程、文件路径、资料类型、版本、发布日期和导入状态。重复内容跳过，变更文件形成新版本。                                                                                                                                                                                                                    | Web 上传、定时扫描和权限管理。发布日期由可信输入提供，不从目录名推断。                                                                                                                                                                                                                                     |
| 2. Notebook 解析                | 用 `nbformat` 按原始 cell 顺序读取 Markdown、代码、文本/HTML 输出、图片和附件；不执行 notebook。由 Markdown 标题建立章节路径，保留代码与其输出的关联。                                                                                                                                                                                          | 处理异常或缺失的 cell ID；长输出、报错堆栈和大数组只保留可检索摘要，完整内容仍可回取。抽查确认部分课程存在中文重写版 notebook（中英夹杂、含 LaTeX 公式与 HTML 样式），双语支持是硬需求，不是可选扩展。                                                                                                     |
| 3. PDF/PPTX 解析               | 已有 Docling PDF Markdown 导出、python-pptx 文本提取及对应 Chunker 原型；当前只形成文本 block，尚未可靠保留 PDF 页码、表格/图片资源和 PPTX 图片/表格。                                                                                                                                                                                        | 声明并锁定依赖，建立小样本自动化质量测试；按页判断原生文字质量，必要时调用 Qwen OCR/VL API。只支持 `.pptx`，旧版 `.ppt` 需先转换。                                                                                                                                                                             |
| 4. 统一内容结构                 | Notebook、PDF 和 PPTX Parser 均产出 `ParsedDocument`/`ContentBlock`，但 `DocumentIndexer` 仍固定使用 Notebook Parser/Chunker。                                                                                                                                                                                                                  | 统一 Parser/Chunker Protocol、按文件类型分派，并补齐页码、幻灯片号、asset、稳定 section/chunk ID 后接入索引 API。                                                                                                                                                                                           |
| 5. Notebook 结构化切块          | Markdown 形成叙述 group，代码与输出形成执行 group；连续相同标题路径形成 SectionRecord。标题段内按 120 soft-min、280 target、450 hard-max tokens 自适应装箱，短而完整的小节允许保留，只有拆分超长 cell 时使用 40 tokens overlap。                                                                                | PDF 后续使用独立切块策略；用课程题集调整参数，不强求 chunk 等长。                                                                                                                                                                                                                                          |
| 6. 多粒度与多模态表示           | 每个 Notebook 生成一份 `document_outline`，每个 chunk 生成 `text_content`，受支持图片生成图片加章节上下文的 `visual_fusion`。Cell Group 只负责结构与溯源，不和 chunk 重复向量化；HTML/DataFrame 表格转为清理后的结构化文本。                                                                                                           | 只有图片表格或布局本身重要时增加视觉表示；代码自然语言摘要、假设问题和额外标题向量等候评测证明收益后再启用。                                                                                                                                                                                               |
| 7. 向量化（Qwen API）与本地存储 | 文本调用 `qwen3.7-text-embedding` 获取 1,024 维 dense+sparse；图片调用 `qwen3-vl-embedding` 图文融合模式获取独立的 1,024 维视觉向量。不同模型空间使用不同命名向量。按批次调用并记录限流、重试、用量、模型和索引版本。                                                                                                                        | 用课程题集对比 document outline、visual fusion 的增量收益。更换模型、维度或切块配置时新建索引并重算向量；API 数量限制与价格按实际地域和账户验证。                                                                                                                                                            |
| 8. 查询与召回                   | 正式入口保留原问题，并行执行 dense、sparse 后使用 RRF 融合；查询 embedding 使用 `text_type=query`，支持课程/文件和可见版本过滤。                                                                                                                                                                                                               | DeepSeek 查询理解客户端已实现但尚未被入口调用。指代补全、英文辅助查询、HyDE、step-back 和多查询仅在固定评测证明有效后启用。                                                                                                                                                                                  |
| 9. 重排与回取                   | Qwen Rerank 已在独立消融脚本验证；视觉召回、相邻上下文与预算管理已有存储/工具代码。当前 `/api/rag/search` 只返回 RRF 文本候选，尚未接入这些步骤。                                                                                                                                                                                                | 修正正式入口的候选保留与配置开关，补齐相邻 chunk/version 溯源，再逐项评估 Rerank、视觉召回和上下文扩展的收益与延迟。                                                                                                                                                                                         |
| 10. 回答                        | 后续问答层使用 `deepseek-flash` API；只依据检索证据回答，给出文件名、页码或 cell 引用。资料不足时说明缺口。                                                                                                                                                                                                                                    | 课程/考试总结走按文件和章节枚举的覆盖流程，再分节汇总；不依赖一次 top-k 查询。                                                                                                                                                                                                                             |
| 11. 增量更新                    | 新文件生成 SectionRecord、Chunk、多表示和资源后再切换为可见版本；替换或删除文件时同步清理旧表示和失效资源。每次导入记录失败与重试状态。                                                                                                                                                                                                         | Web 上传与自动导入；检索时同时校验 `document_id`、版本和发布日期，避免旧版或未解封内容泄露。                                                                                                                                                                                                               |
| 12. 评估                        | 已建立 20 题 Notebook 小型基线：15 条开发集、5 条保留集，标注参考答案、文件、章节和 cell；测 `Recall@k`、MRR、`nDCG@k`、来源命中率和延迟。后续扩展到 60–100 题，加入图表、指定文件、跨章节和无答案题。                                                                                                                               | 固定相同题集及证据预算，对比 text content、document outline、visual fusion、混合检索与重排；总结另测章节覆盖率，增量导入另测旧版本清理。                                                                                                                                                                  |

## 关键约束

- **Notebook 是首要数据模型。** 一个代码 cell 的输出可能包含图片、HTML 表格或多种 MIME 表示；不能把整份 notebook 转成纯文本后按固定字符数切割。
- **图表说明是检索线索，不是精确事实来源。** 回答图中数值或曲线关系时回取原图、表格或原始输出核对。
- **Notebook 不预生成重复父块。** Cell Group、SectionRecord 和相邻 chunk 保存结构关系；多表示只用于 document outline、正文和视觉资源，不为 Cell Group 与 chunk 重复生成同一内容的向量。
- **存储与模型推理分离。** Qdrant、SQLite、原始文件和无模型解析工具保存在本地（后续可迁移到云端）；所有模型推理通过 API 完成。API 不可用时暂停相关推理任务，不静默加载本地模型。
- **全部模型推理依赖外部 API。** 需要处理限流退避、失败重试、网络异常、API Key 保密、费用观测，以及课件内容发送到第三方服务的取舍。
- **区分组件完成与链路完成。** Parser、客户端、存储方法或实验脚本存在，不代表已接入 API；只有正式入口、配置、测试和真实样本验证全部通过后才标记为已支持。
- **先测后调。** 上述 token 范围、候选数量和模型均是初始配置；最终以真实课程问题的效果、延迟和内存证据决定。

## 首个可验收里程碑

选取少量具有 Markdown 说明、代码、表格和图片输出的 notebook，完成结构化导入、按文件过滤及 cell 引用；建立人工问题，对比 text content、document outline、visual fusion、混合召回和重排。该基线稳定后，再接入 PDF 和网页问答。

**当前进度**：✅ 已完成并评估。
- 20 题评估集（15 开发 + 5 保留）覆盖 7 份 Notebook
- Dense baseline: Recall@8=0.95, MRR=0.875
- RRF 策略: Recall@8=1.00, MRR=0.893（推荐默认）
- RRF+Outline+Rerank 实验: Recall@1=0.90, MRR=0.950（尚未接入正式入口）
- Document outline 识别融合问题，暂不启用

机器可读结果见 [evals/results](./evals/results/)；当前仓库没有单独的人工评估报告文件。

## 参考资料

- [Jupyter Notebook 文件格式](https://nbformat.readthedocs.io/en/latest/format_description.html)
- [通用文本向量 API（qwen3.7-text-embedding，dense+sparse 混合检索）](https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api)
- [多模态向量 API（qwen3-vl-embedding，图文融合向量）](https://help.aliyun.com/zh/model-studio/multimodal-embedding-api-reference)
- [通用文本排序模型 API（qwen3.7-text-rerank）](https://help.aliyun.com/zh/model-studio/text-rerank-api)
- [DeepSeek Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/)
- [DeepSeek 图像理解 API（deepseek-flash）](https://api-docs.deepseek.com/zh-cn/guides/vision/)
- [Qwen OCR API](https://help.aliyun.com/en/model-studio/qwen-vl-ocr-api-reference)
- [阿里云百炼限流说明](https://help.aliyun.com/zh/model-studio/rate-limit)
- [Qwen3 Embedding 与 Reranker 技术背景（开源模型仓库，仅供架构原理参考，本项目通过 API 调用而非本地部署）](https://github.com/QwenLM/Qwen3-Embedding)
- [Qdrant 本地部署](https://qdrant.tech/documentation/quick-start/)与[混合查询](https://qdrant.tech/documentation/search/hybrid-queries/)
- [Docling model-free native PDF pipeline](https://docling-project.github.io/docling/reference/pipeline_options/)
