# SRAG Agent 状态

> 你是一名 RAG 和 Agent 工程师，负责以可评估、可追溯、可增量更新的方式推进本项目。开始工作前阅读 [README.md](./README.md) 和 [RAG.md](./RAG.md)；架构、目标与技术选型以这两份文档为准，不在本文件重复记录。

## 当前进度

更新时间：2026-09-28

- 阶段：Notebook 最小检索闭环已完成，进入完整检索链路开发。
- 已完成：真实 ipynb 解析、结构化切块、Qwen Embedding、Qdrant 幂等索引、dense Top K 检索、范围过滤、来源回溯和 HTTP 接口。
- 当前目标：实现 SQLite 文档登记、版本、发布日期和结构关系持久化。
- 后续：实现增量索引、DeepSeek 查询理解、混合检索、rerank API 和首批评测问题。
- 阻塞：无。API 凭据在接入 API 阶段再配置，禁止写入仓库。

## 更新规则

只在以下情况更新本文件：里程碑状态变化、重要架构决策调整、出现或解除阻塞、获得关键评测结果、下一目标改变。详细设计同步写入 README/RAG，本文件只记录结果与进度。

## 重大变更

| 日期 | 变更与结果 |
| --- | --- |
| 2026-09-23 | 完成架构文档；确认所有模型推理走 API；进入 Notebook 导入实现阶段。 |
| 2026-09-23 | 采用精简功能域结构；统一 `.env` 配置；完成首期 RAG 项目骨架。 |
| 2026-09-24 | 完成项目说明中文化；确认 FastAPI 与本地 Qdrant 可用；下一目标为 Notebook 只读解析。 |
| 2026-09-24 | 完成父子切块与 Qwen dense+sparse Embedding；真实 API 和实验 Notebook 验证通过。 |
| 2026-09-24 | 完成真实 ipynb 只读解析；Parser → Indexer → Chunker → Embedding 链路验证通过。 |
| 2026-09-24 | Notebook 改为标题段与 Cell Group 自适应切块；增加 document outline、text content、qwen3-vl visual fusion 三类表示，真实课件 API 验证通过。 |
| 2026-09-25 | 查询理解与按需改写改用支持图片输入的 `deepseek-flash`；Qwen 继续负责 query embedding 和 rerank。 |
| 2026-09-25 | 建立最小检索闭环 checklist；完成 Qdrant 三类命名向量与 payload 索引的安全初始化。 |
| 2026-09-25 | 完成文本 dense+sparse 向量及溯源 payload 幂等写入；真实 Lecture1a 入库 42 个 Qdrant points。 |
| 2026-09-28 | 接通 `DocumentIndexer.index()` 正式入口；真实 Lecture1a 重复索引成功且 Qdrant 保持 42 个稳定 points。 |
| 2026-09-28 | 完成原始文本 query 的 dense Top 8 召回；真实 pandas 问题首条命中对应课程章节。 |
| 2026-09-28 | 完成课程/文件范围过滤与完整来源返回；真实测试中正确课程和 document 各返回 8 条，错误课程返回 0 条。 |
| 2026-09-28 | 接通 `/api/rag/index` 与 `/api/rag/search`；真实 Lecture1a 经 HTTP 完成解析、幂等入库、查询和 cell 来源回溯，Qdrant 保持 42 个 points。Notebook 最小检索闭环完成。 |
| 2026-09-28 | 新增端到端体验 Notebook，支持递归发现课程目录中的 ipynb、批量索引、自由 query 和逐条来源展示；真实索引 Lecture1a/Lecture1b 共 86 个 points，pandas 与 NumPy 查询均命中对应文件和章节。 |
| 2026-09-28 | 精简 README 为当前能力、快速开始、API 状态和路线图；清理仓库内个人绝对路径，Notebook 配置改为通过 `.env` 的 `COURSE_ROOT` 指定。 |
