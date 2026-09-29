# SRAG 开发 Checklist

## 最小检索闭环

- [X] 定义并安全初始化 Qdrant collection、命名向量与 payload 索引。
- [X] 将单个 Notebook 的文本表示写入 Qdrant，并保留文件、章节和 cell 溯源信息。
- [X] 接通 `DocumentIndexer.index()` 的单文件索引流程。
- [X] 实现原始文本 query 的 dense 向量召回。
- [X] 支持按课程和文件过滤，并返回 Top K chunk 与来源。
- [X] 接通 `/api/rag/index` 与 `/api/rag/search`。
- [X] 使用真实 Notebook 验证“解析 → 入库 → 查询 → 来源回溯”。

## 完整检索链路

### 数据生命周期

- [X] 建立 SQLite schema 与迁移，持久化文档、版本、发布日期、Section、Cell Group、Chunk、Asset 和索引任务。
- [X] 基于内容哈希实现幂等增量索引，以及 `building → active → superseded/failed` 版本状态切换。
- [X] 查询仅使用 active 且已到发布日期的版本，并完成旧版本、孤立向量、文档和资源删除。
- [X] 接通索引任务查询、文档列表和文档删除 HTTP 接口。

### 检索评估基线

- [X] 审阅代表性 Notebook，冻结首批评估语料与开发集/保留集划分。
- [X] 建立 20 题 Notebook 检索基线集，标注参考答案、预期文件、章节和 cell。
- [X] 实现 Recall@K、MRR/nDCG、文件/章节/cell 命中率和延迟统计。
- [X] 在真实 Qdrant 上运行并记录当前 dense baseline。

### 检索增强

- [X] 实现 sparse 召回与 dense/sparse RRF 融合。
- [X] 评估 document outline 独立召回，并完成跨表示候选融合与去重。
- [X] 识别 outline 问题：包含关键词但无答案内容，RRF 融合后得分过高。
- [X] 暂时禁用 outline 召回，待 PDF/PPT 接入后重新评估文档发现需求。
- [X] 接入 `qwen3.7-text-rerank` 候选重排。
- [X] 接入 `deepseek-flash` 查询理解和按需改写；原始 query 始终保留，HyDE、step-back 和多查询仅在评估证明有效后启用。

### 多模态与上下文

- [X] 将 `visual_fusion` 表示及其资源关系正式写入 Qdrant 和 SQLite。
- [X] 实现 visual query embedding、视觉召回及文本/视觉候选融合。
- [X] 按需回取相邻 chunk、cell、原图和原表，并限制最终上下文预算。

### 工程化与正式评估

- [X] 完成 20 题 Notebook 检索基线，标注参考答案、文件、章节和 cell。
- [X] 实现 Recall@K、MRR/nDCG、文件/章节/cell 命中率和延迟统计。
- [X] 在真实 Qdrant 上运行并记录当前 dense baseline。
- [X] 完成 4 种策略消融实验：Dense、RRF、RRF+Outline、RRF+Outline+Rerank。
- [X] 识别 RRF 为最佳性价比策略，Rerank 为最高精度策略。
- [X] 发现并记录 Outline 融合问题（Recall@1 降至 0.00）。
- [X] 分析 Outline 问题根本原因：包含关键词但无答案，RRF 中得分过高。
- [X] 暂时禁用 Outline 召回，避免影响检索质量。
- [ ] 将评估集扩展到 60–100 题，增加难度分层和多模态查询。
- [ ] 实现 LLM-as-Judge 自动评估 Context Precision 和 Relevance。
- [ ] 完善超时、限流、重试、部分失败、任务状态、查询日志和可观测性。
- [ ] 添加在线监控和用户满意度反馈收集。
- [ ] 实现压力测试：并发 QPS、大规模索引性能（100/1000/10000 文档）。

## 后续阶段

- [ ] 重新设计 Outline 功能：分离 pipeline，用于文档发现而非精确答案排序。
- [ ] 实现 Web 文件上传接口与存储管理。
- [ ] 接入 PDF 解析与 PDF 专用切块策略。
- [ ] 接入 DeepSeek 回答层和网页交互。
- [ ] 评估云端迁移方案（Qdrant Cloud、对象存储、服务部署）。
