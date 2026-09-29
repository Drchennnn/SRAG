# SRAG 完整进度记录

本文件保存项目的完整里程碑历史。[agent.md](./agent.md) 只保留当前状态和最近几次关键变化；详细架构与方法以 [README.md](./README.md) 和 [RAG.md](./RAG.md) 为准。

## 里程碑

| 日期 | 变更与结果 |
| --- | --- |
| 2026-09-23 | 完成架构文档；确认所有模型推理走 API；进入 Notebook 导入实现阶段。 |
| 2026-09-23 | 采用精简功能域结构；统一 `.env` 配置；完成首期 RAG 项目骨架。 |
| 2026-09-24 | 完成项目说明中文化；确认 FastAPI 与本地 Qdrant 可用；下一目标为 Notebook 只读解析。 |
| 2026-09-24 | 完成父子切块与 Qwen dense+sparse Embedding；真实 API 和实验 Notebook 验证通过。 |
| 2026-09-24 | 完成真实 ipynb 只读解析；Parser → Indexer → Chunker → Embedding 链路验证通过。 |
| 2026-09-24 | Notebook 改为标题段与 Cell Group 自适应切块；增加 document outline、text content、qwen3-vl visual fusion 三类表示，真实课件 API 验证通过。 |
| 2026-09-25 | 查询理解与按需改写改用 `deepseek-flash`；Qwen 继续负责 query embedding 和 rerank。 |
| 2026-09-25 | 建立最小检索闭环 checklist；完成 Qdrant 三类命名向量与 payload 索引的安全初始化。 |
| 2026-09-25 | 完成文本 dense+sparse 向量及溯源 payload 幂等写入；真实 Lecture1a 入库 42 个 Qdrant points。 |
| 2026-09-28 | 接通 `DocumentIndexer.index()` 正式入口；真实 Lecture1a 重复索引成功且 Qdrant 保持 42 个稳定 points。 |
| 2026-09-28 | 完成原始文本 query 的 dense Top 8 召回；真实 pandas 问题首条命中对应课程章节。 |
| 2026-09-28 | 完成课程/文件范围过滤与完整来源返回；真实测试中正确课程和 document 各返回 8 条，错误课程返回 0 条。 |
| 2026-09-28 | 接通 `/api/rag/index` 与 `/api/rag/search`；真实 Lecture1a 经 HTTP 完成解析、幂等入库、查询和 cell 来源回溯。Notebook 最小检索闭环完成。 |
| 2026-09-28 | 新增端到端体验 Notebook，支持递归发现、批量索引、自由 query 和来源展示；Lecture1a/Lecture1b 共入库 86 个 points。 |
| 2026-09-28 | 精简 README；清理仓库内个人绝对路径，Notebook 配置改为通过 `.env` 的 `COURSE_ROOT` 指定。 |
| 2026-09-29 | 重排完整检索链路 Checklist；补充版本可见性、评估基线、outline 与视觉入库验收点；回答模型改为 `deepseek-flash`。 |
| 2026-09-29 | 完成 SQLite 数据生命周期：结构与任务持久化、内容哈希跳过、版本原子切换、发布日期过滤、旧向量清理和文档删除。 |
| 2026-09-29 | 完成 20 题 Notebook 检索基线：15 条开发集、5 条保留集，覆盖 7 份课件；dense Top 8 的 Recall@1=0.80、Recall@8=0.95、MRR@8=0.875。 |
| 2026-09-29 | 实现 sparse 召回、RRF、document outline、Qwen rerank、DeepSeek 查询理解及消融实验脚本。 |
| 2026-09-29 | 实现 visual fusion 索引/召回方法和上下文预算工具；相关步骤尚未全部接入正式查询入口。 |
| 2026-09-29 | 完成四种策略消融：RRF 达到 Recall@1=0.85、Recall@8=1.00；RRF+Outline+Rerank 实验达到 Recall@1=0.90、MRR@8=0.950。 |
| 2026-09-29 | 确认 Outline 会因只有标题而在融合中压过正文 chunk；正式入口禁用 Outline，默认采用 Dense+Sparse RRF。 |
| 2026-09-29 | 实现 PDF/PPTX Parser 和 Chunker 原型；手工质量脚本尚未纳入标准测试门禁。 |
| 2026-09-29 | 复核实现边界：Rerank、查询理解、视觉召回和上下文扩展仍是组件或实验路径；PDF/PPTX 尚未声明依赖、统一协议或接入索引 API。 |
| 2026-09-30 | Colab T4 完成 32 页幻灯片 PDF 预处理实测；解析耗时 80.2 秒，输出 33 Blocks/28 Chunks，并暴露页内断层、图片占位和页码缺失问题。 |
| 2026-09-30 | PDF/PPTX 统一采用 SlideDeck：一页/一张为基础 Chunk，原生文本优先、按页 OCR 回退，整页视觉表示为主，明显装饰资源用规则过滤。 |

## 记录规则

- 每次重大变更在此追加一条可验证结果，不记录逐文件实现细节。
- `agent.md` 只保留最近 3–5 条关键变化；旧记录迁移到本文件。
- 评估数字必须能对应 `evals/results/` 中的结果文件。
