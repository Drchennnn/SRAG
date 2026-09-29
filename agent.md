# SRAG Agent 状态

> 你是一名 RAG 和 Agent 工程师，负责以可评估、可追溯、可增量更新的方式推进本项目。开始工作前阅读 [README.md](./README.md) 和 [RAG.md](./RAG.md)；架构、目标与技术选型以这两份文档为准，不在本文件重复记录。

## 当前进度

更新时间：2026-09-29

- 阶段：检索增强和多模态上下文已完成，消融实验已完成并识别最优策略。
- 已完成：
  - 数据生命周期（版本管理、增量索引、发布日期控制）
  - 20 题评估集（15 开发 + 5 保留，覆盖 7 份 ML/X AI Notebook）
  - Dense baseline（Recall@8=0.95, MRR=0.875）
  - 检索增强：Sparse/RRF、Outline 融合、Rerank、查询理解
  - 多模态与上下文：Visual fusion 索引与召回、上下文回取与预算管理
  - 消融实验：对比 4 种策略，识别 RRF 为最优性价比，Rerank 为最高精度
- 当前目标：
  - 修复 Outline 融合问题（当前会导致 Recall@1 降至 0.00）
  - 将 RRF 设为默认策略，暂时禁用 Outline
  - 优化 Rerank 延迟（考虑缓存、批量调用）
- 后续：
  - 扩展评估集到 60-100 题（难度分层、多模态、跨文档）
  - 实现 LLM-as-Judge 自动评估
  - 工程化优化（查询日志、监控、限流、缓存）
  - 验证视觉召回效果
  - 接入 PDF 解析
- 阻塞：无。所有核心功能已实现，Outline 融合问题已识别，等待修复。
- 评估结论：
  - **推荐默认策略**：RRF（Recall@1=0.85, Recall@8=1.00, 延迟=829ms）
  - **高精度场景**：RRF+Rerank（Recall@1=0.90, MRR=0.950, 延迟=2140ms）
  - **暂不启用**：Outline 召回（需修复融合逻辑）

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
| 2026-09-29 | 重排完整检索链路 Checklist，补充版本可见性、早期评估基线、outline 与视觉入库验收点；后续回答模型由 GPT 改为 `deepseek-flash`。 |
| 2026-09-29 | 完成 SQLite 数据生命周期：结构与任务持久化、内容哈希跳过、版本原子切换、发布日期过滤、旧向量清理和文档删除；真实测试验证 version 2 替换、未来资料隐藏及删除后不可检索。 |
| 2026-09-29 | 完成 20 题 Notebook 检索基线：15 条开发集、5 条保留集，覆盖 7 份 ML/X AI 课件；dense Top 8 的 Recall@1=0.80、Recall@8=0.95、MRR@8=0.875、cell 命中率@8=0.95。 |
| 2026-09-29 | 完成检索增强：实现 sparse 召回、RRF 融合、document outline 召回与跨表示融合去重、Qwen rerank、DeepSeek 查询理解；接通消融实验脚本。 |
| 2026-09-29 | 完成多模态与上下文：实现 visual fusion 表示索引与召回、上下文回取（相邻 chunk、原图原表）、上下文预算管理；视觉召回默认关闭，可通过配置启用。 |
| 2026-09-29 | 完成消融实验：对比 Dense、RRF、RRF+Outline、RRF+Outline+Rerank 四种策略；RRF 提升 Recall@1 至 0.85 并达到 100% Recall@8；Rerank 进一步提升至 Recall@1=0.90、MRR=0.950；识别 Outline 融合问题导致严重退化（Recall@1 降至 0.00），需修复后再启用。推荐默认采用 RRF 策略。 |
