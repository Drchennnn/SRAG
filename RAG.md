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
2. 默认保留原始问题；显式 contextual 模式仅在含指代词、且当前问题缺少具体主题时补入最近一条用户问题。历史最多 6 条，助手回答不参与，课程/文件范围仍只来自本次请求。构造后的文本生成 Qwen query 向量，分别召回 Dense 与 Sparse 候选，以 RRF 合并。
3. auto 模式对明确询问照片、图表、示意图等视觉内容的问题生成 Qwen 视觉查询向量，优先返回整页视觉命中，文本候选补足；可显式选 text 或 visual。
4. 返回原始问题、实际检索文本与方法，以及命中内容、来源文件、章节/cell 或页码、可用截图路径。当前尚不生成自然语言答案。

视觉查询失败时，auto 模式记录异常并返回文本结果；显式 visual 模式抛出错误。这是已测试的降级边界。曾尝试将文本与视觉结果等权 RRF：纯图示页可能被文本候选埋没，因此正式入口使用可解释的视觉意图路由。CS5495 的图表细节题仍是主要短板。

Qwen Rerank 和 document outline 已在独立实验中评估，未接入正式检索；DeepSeek 查询改写客户端已有初版，仍需建立评估后再接入。Outline 仅含标题时会压过正文，当前禁用；HyDE、step-back、多查询也不作为默认路径。相邻 cell/页上下文回取要等来源版本与预算契约明确后实现，不能把占位工具当作已完成能力。

首批幻灯片诊断显示词面指代会误扩展已换话题的问题。新增具体主题判断后，新页码集的误扩展降为 0/8；两份 PDF 的模拟多轮题最初误扩展 2/8，进一步识别“主点”“K 矩阵”等短术语后降为 0/8，追问目标页保持 8/8 进入 Top 5。复用 7 份 Notebook 的 [16 道模拟多轮题](./evals/query_notebook_dialogue_cases.jsonl) 时，发现“在哪段”未被识别为位置问法；修正后追问目标 cell 的 Top 5 命中从 1/8 升至 8/8。旧/新报告位于 [evals/results](./evals/results/)。

另留 [8 道 Notebook cell 题](./evals/query_notebook_holdout_cases.jsonl) 检查迁移：4 道独立问题均保持 raw；4 道追问只补全 1 道，目标 cell 命中 2/4。两道漏检强制补入上一条用户问题后均排第 1，说明主要是“这个机制／总结”被当作具体主题，而非向量库没有对应内容。独立的 DeepSeek 二分类实验在这 4 道追问上命中 4/4、独立题误扩展 0/4；但复查 PDF 模拟对话时独立题误扩展 3/8，其中 1 题由命中变为漏检。Notebook 诊断集也没有来源命中增益。因此分类器仅保留离线评估，contextual 仍显式选择，默认 raw；生成式改写不进入正式检索。见 [评估资产](./evals/README.md)。题集均为人工构造，不能推断真实用户流量效果。

## 回答与评估方向

下一阶段以**文字解答**为首要输出，逐条引用课程文件的 cell 或页码。课程图片只在匹配答案时作为辅助；如果课程资料中没有合适的图片，可单独检索外部图片，明确区分外部来源与课程证据。总结类问题需要按文件或章节枚举覆盖，不能只依赖一次 Top K。

现有 Notebook 20 题、CS5187 两套各 28 题、CS5495 24 题检索集及逐题报告见 [评估资产](./evals/README.md)。它们用于比较来源 Recall@k、MRR、视觉细节命中与延迟，尚未评估生成答案的事实性、引用准确率或章节覆盖率。CS5187/CS5495 题集已参与分析；路线调整后用新资料做独立验收。

## 参考

- [Notebook 格式](https://nbformat.readthedocs.io/en/latest/format_description.html)
- [Qwen 文本向量 API](https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api)
- [Qwen 多模态向量 API](https://help.aliyun.com/zh/model-studio/multimodal-embedding-api-reference)
- [Qdrant 文档](https://qdrant.tech/documentation/)
