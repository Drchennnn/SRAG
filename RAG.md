# 课程资料 RAG：方法与设计决策

实现进度以 [CHECKLIST.md](./CHECKLIST.md) 为准；运行方式和接口见 [README.md](./README.md)。

## 目标与边界

- Notebook 已形成正式检索基线；PDF/PPTX 已完成 SlideDeck 预处理、向量化和 Qdrant/SQLite 导入；CS5187 两套各 28 题及 CS5495 新课程 24 题已完成页码召回评估。
- 支持资料的新版本、发布日期和下架。未到发布日期的资料不进入可查询范围。
- Embedding、查询改写、重排、图像理解和最终问答调用外部 API。PDF 解析与按页 OCR 可在 Colab GPU 临时运行 Docling/RapidOCR；本地电脑和自建服务器不部署长期模型服务。
- 先在本地电脑上完成解析、检索链路搭建和评估；成熟后再评估是否把 Qdrant 和编排服务迁往云端。当前云服务器为 4 核、4 GB 内存、40 GB SSD，不满足 [Milvus Standalone 的 8 GB 内存要求](https://milvus.io/docs/prerequisite-docker.md)。即便模型均走 API，仍需评估 Qdrant 内存、磁盘、API 延迟、限流、费用和网络稳定性。
- `COURSE_ROOT` 指向候选资料来源，不代表所有文件都应入库。导入采用明确的文件白名单，排除虚拟环境、缓存、训练数据集和无关图片。
- 向量化会把课件文本或图片发送给阿里云，查询理解和后续回答生成还会把问题及检索证据发送给 DeepSeek；这是已知取舍。需要妥善保管 API Key（不在客户端或仓库中明文存储），记录发送范围，并处理限流、重试与 API 不可用时暂停任务的策略。

## 数据流

```text
选定资料（ipynb 直接索引；PDF/PPTX 从 Colab 预处理包和向量包导入）
  → 文件登记与版本检查
  → Notebook 解析为标题段 / Cell Group，PDF/PPTX 解析为 SlideDeck / Slide / SlideAsset
  → Notebook 自适应 Chunk，幻灯片以一页/一张为基础 Chunk
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
| 3. SlideDeck 解析              | PDF/PPTX 均视为幻灯片。PDF 原生文本优先，稀疏页 OCR 仅补充原生内容；PPTX 读取文本框、表格、图片和布局。两者导出整页截图、页码/幻灯片号、相邻关系和资源。真实 PDF/PPTX 已通过页级文本、资源与整页截图核对。                                                                                                                                                | 锁定 Docling、python-pptx 和渲染依赖；只支持 `.pptx`，旧版 `.ppt` 需先转换。用 Colab 输出可回传本地的 `deck.json + chunks.json + slides/ + assets/`。                                                                                                                                    |
| 4. 统一内容结构                 | Notebook 保持 `ParsedDocument`/`ContentBlock`；幻灯片新增 `SlideDeck`/`Slide`/`SlideAsset`，以稳定 document/slide/asset ID 连接来源和资源。`DocumentIndexer` 当前仍固定使用 Notebook Parser/Chunker。                                                                                                                                           | 当前用独立的 SlideIndexer 导入预处理包；以后若增加服务端直接处理原文件，再统一文件类型分派。                                                                                                                                                                                                               |
| 5. Notebook 结构化切块          | Markdown 形成叙述 group，代码与输出形成执行 group；连续相同标题路径形成 SectionRecord。标题段内按 120 soft-min、280 target、450 hard-max tokens 自适应装箱，短而完整的小节允许保留，只有拆分超长 cell 时使用 40 tokens overlap。                                                                                | SlideDeck 采用一页/一张为基础 Chunk，只拆分超长页；用课程题集调整参数，不强求 chunk 等长。                                                                                                                                                                                                                  |
| 6. 多粒度与多模态表示           | Notebook 沿用正文和视觉表示。幻灯片文字、表格生成 `text_dense/text_sparse`；幻灯片全部整页截图先生成 `visual_dense` 基线，再用检索评估决定是否筛选。先按哈希、重复位置、面积和母版来源过滤 Logo、背景和小型角落装饰，不逐张图片生成向量。                                                                                                                                    | 原课件和新课件题集均显示视觉单路可召回纯图像细节，简单等权 RRF 会埋没该结果。正式入口现用轻量词面路由处理明确视觉问题；新课件题集 Recall@8 达 1.00。CS5495 新课程验证中，正式入口图表细节 Top 5 仅 1/4；需改进路由与排序并用新资料复验，暂不筛掉整页截图。                                                                                                                                                                                                  |
| 7. 向量化（Qwen API）与本地存储 | 文本调用 `qwen3.7-text-embedding` 获取 1,024 维 dense+sparse；整页截图调用 `qwen3-vl-embedding` 图文融合模式获取独立的 1,024 维视觉向量。SlideDeck 离线向量包已实现，PDF/PPTX 的全部页面文本与整页截图均已实测，已通过独立导入接口写入 Qdrant/SQLite。不同模型空间使用不同命名向量。文本按批次调用，视觉逐页调用；离线包记录模型、用量和页级关联，索引版本在入库阶段记录。                                                                                                                        | 用课程题集对比 document outline、visual fusion 的增量收益。更换模型、维度或切块配置时新建索引并重算向量；API 数量限制与价格按实际地域和账户验证。                                                                                                                                                            |
| 8. 查询与召回                   | 正式入口保留原问题；普通问题用 Qwen 文本 query embedding 与 Dense+Sparse RRF，明确描述照片、图示、曲线的问题用同空间视觉 query embedding 优先召回整页截图，文本候选补足。支持 `auto/text/visual` 模式、课程/文件及可见版本过滤，响应标出实际模式与截图路径。                                                                                                                                                                                                               | DeepSeek 查询理解客户端已实现但尚未被入口调用。指代补全、英文辅助查询、HyDE、step-back 和多查询仅在固定评测证明有效后启用。                                                                                                                                                                                  |
| 9. 重排与回取                   | Qwen Rerank 已在独立消融脚本验证；视觉召回、相邻上下文与预算管理已有存储/工具代码。当前 `/api/rag/search` 只返回 RRF 文本候选，尚未接入这些步骤。                                                                                                                                                                                                | 修正正式入口的候选保留与配置开关，补齐相邻 chunk/version 溯源，再逐项评估 Rerank、视觉召回和上下文扩展的收益与延迟。                                                                                                                                                                                         |
| 10. 回答                        | 后续问答层使用 `deepseek-flash` API；只依据检索证据回答，给出文件名、页码或 cell 引用。资料不足时说明缺口。                                                                                                                                                                                                                                    | 课程/考试总结走按文件和章节枚举的覆盖流程，再分节汇总；不依赖一次 top-k 查询。                                                                                                                                                                                                                             |
| 11. 增量更新                    | 新文件生成 SectionRecord、Chunk、多表示和资源后再切换为可见版本；替换或删除文件时同步清理旧表示和失效资源。每次导入记录失败与重试状态。                                                                                                                                                                                                         | Web 上传与自动导入；检索时同时校验 `document_id`、版本和发布日期，避免旧版或未解封内容泄露。                                                                                                                                                                                                               |
| 12. 评估                        | Notebook 已有 20 题基线；CS5187 原两份课件与新两份 PDF 各有 28 题页码题集，测文本 Dense、Dense+Sparse、视觉与三路 RRF 的 Recall@k、MRR 和逐题排名。新课件测试集三道纯图示题均被等权融合压到 Top 8 之外；视觉路由使新课件 Recall@5/8 达 0.964/1.00。CS5495 四份 PDF 的 24 题新课程验证已完成，正式入口全课程 Recall@5=0.875，图表细节仅 1/4；两批题集已用于分析，后续修改需另留新资料验收。                                                                                                                               | 固定相同题集及证据预算，对比 text content、document outline、visual fusion、混合检索与重排；总结另测章节覆盖率，增量导入另测旧版本清理。                                                                                                                                                                  |

## 关键约束

- **按资料形态建模。** Notebook 保留 cell、输出和章节关系；PDF/PPTX 作为 SlideDeck 保留页码、布局和整页视觉信息，不能统一降级成 Markdown 后再按标题切割。
- **幻灯片边界优先。** 一页/一张是基础 Chunk；短页不因 token 少而与无关页面合并，只有超长页继续拆分。稀疏页可只有标题和图片，OCR 只能补充原生内容，不应覆盖原有标题与资源。明显装饰资源用规则过滤，模糊图片保留在整页视觉表示中。
- **图表说明是检索线索，不是精确事实来源。** 回答图中数值或曲线关系时回取原图、表格或原始输出核对。
- **Notebook 不预生成重复父块。** Cell Group、SectionRecord 和相邻 chunk 保存结构关系；多表示只用于 document outline、正文和视觉资源，不为 Cell Group 与 chunk 重复生成同一内容的向量。
- **存储与推理分离。** Qdrant、SQLite 和原始文件保存在本地（后续可迁移到云端）；Colab 只输出预处理包，不直接访问本地 Qdrant。
- **业务模型依赖外部 API。** Embedding、查询、重排和回答需要处理限流退避、失败重试、API Key 保密、费用观测，以及课件内容发送到第三方服务的取舍。Docling/RapidOCR 只用于临时文档预处理。
- **区分组件完成与链路完成。** Parser、客户端、存储方法或实验脚本存在，不代表已接入 API；只有正式入口、配置、测试和真实样本验证全部通过后才标记为已支持。
- **先测后调。** 上述 token 范围、候选数量和模型均是初始配置；最终以真实课程问题的效果、延迟和内存证据决定。

## 验证原则

- Notebook 检索使用固定题集比较 Dense、RRF、Rerank 等策略，机器可读结果保存在 [evals/results](./evals/results/)。
- SlideDeck 先验证页级文本、资源和原图的溯源；CS5187 的第 14、22、30 页分别覆盖图像示例、流程图和图片对比。检索评估需同时检查文字、视觉内容与正确页码。
- 先对代表性课件的全部页面建立视觉向量基线，再用相同题集比较视觉候选页筛选的效果和调用成本；CS5187 PDF 第 12 页是现有候选标记的漏判样本。

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
- [Docling Native PDF pipeline](https://docling-project.github.io/docling/usage/advanced_options/)
