# SRAG：方法与设计决策

运行方式与接口见 [README.md](./README.md)；未完成任务见 [CHECKLIST.md](./CHECKLIST.md)。本文件只记录当前链路和对后续开发仍有效的取舍。

## 边界

- 本地运行 FastAPI、Qdrant、SQLite；课程文件和资源保留本地。Qwen 负责 Embedding，DeepSeek 预留给查询理解与回答；模型均通过 API 调用。
- PDF/PPTX 可在 Colab GPU 临时预处理；Colab 输出便携包，不直接连接本地 Qdrant。
- 仅索引明确选定的课程文件。SQLite 管理文档版本、发布日期、可见性和删除；未发布资料不进入正式检索。
- 调用外部模型会发送课件文本或图片；API Key 保存在本地 .env，不能进入前端或版本库。

## 资料表示

| 资料 | 切块与来源 | 向量 |
| --- | --- | --- |
| IPYNB | 只读解析 Markdown、代码、输出、附件；标题形成章节，连续 cell 形成 Cell Group。章节内按约 120/280/450 token 的软下限/目标/硬上限装箱，超长 cell 拆分时 overlap 40 token。 | Qwen 文本 dense+sparse；必要的视觉表示。Section/Cell Group 保留结构关系，不重复生成父块正文向量。 |
| PDF/PPTX | 两者均视为 SlideDeck，一页/一张为基础 Chunk；超长页才继续拆分。PDF 原生文字优先，稀疏页 OCR 只补充；保留页码、整页截图和资源关系。 | 文本向量只使用标题和正文；全部整页截图建立视觉基线。装饰图片用规则过滤，但不按候选标记删掉整页向量。 |

文本使用 qwen3.7-text-embedding 的 dense+sparse 表示；整页图像使用 qwen3-vl-embedding，放在独立的 Qdrant 命名向量空间。换模型、维度或切块策略时需要重算对应向量并建立新索引版本。原始图片与文件不塞入向量 payload；payload 保留回溯来源所需的 ID、路径、页码和版本。

## 当前检索

1. 根据课程/文件范围和 SQLite 可见版本过滤。
2. 原始问题生成 Qwen 文本 query 向量，分别召回 Dense 与 Sparse 候选，以 RRF 合并。
3. auto 模式对明确询问照片、图表、示意图等视觉内容的问题生成 Qwen 视觉查询向量，优先返回整页视觉命中，文本候选补足；可显式选 text 或 visual。
4. 返回原始内容和文件、章节/cell 或页码、可用截图路径。当前尚不生成自然语言答案。

视觉查询失败时，auto 模式记录异常并返回文本结果；显式 visual 模式抛出错误。这是已测试的降级边界。曾尝试将文本与视觉结果等权 RRF：纯图示页可能被文本候选埋没，因此正式入口使用可解释的视觉意图路由。CS5495 的图表细节题仍是主要短板。

Qwen Rerank、DeepSeek 查询改写和 document outline 已在独立实验中评估，未接入正式检索。Outline 仅含标题时会压过正文，当前禁用；HyDE、step-back、多查询也不作为默认路径。相邻 cell/页上下文回取要等来源版本与预算契约明确后实现，不能把占位工具当作已完成能力。

## 回答与评估方向

下一阶段以**文字解答**为首要输出，逐条引用课程文件的 cell 或页码。课程图片只在匹配答案时作为辅助；如果课程资料中没有合适的图片，可单独检索外部图片，明确区分外部来源与课程证据。总结类问题需要按文件或章节枚举覆盖，不能只依赖一次 Top K。

现有 Notebook 20 题、CS5187 两套各 28 题、CS5495 24 题检索集及逐题报告位于 [evals](./evals/)。它们用于比较来源 Recall@k、MRR、视觉细节命中与延迟，尚未评估生成答案的事实性、引用准确率或章节覆盖率。CS5187/CS5495 题集已参与分析；路线调整后用新资料做独立验收。

## 参考

- [Notebook 格式](https://nbformat.readthedocs.io/en/latest/format_description.html)
- [Qwen 文本向量 API](https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api)
- [Qwen 多模态向量 API](https://help.aliyun.com/zh/model-studio/multimodal-embedding-api-reference)
- [Qdrant 文档](https://qdrant.tech/documentation/)
