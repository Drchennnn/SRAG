# SRAG 开发 Checklist

## 最小检索闭环

- [x] 定义并安全初始化 Qdrant collection、命名向量与 payload 索引。
- [x] 将单个 Notebook 的文本表示写入 Qdrant，并保留文件、章节和 cell 溯源信息。
- [x] 接通 `DocumentIndexer.index()` 的单文件索引流程。
- [x] 实现原始文本 query 的 dense 向量召回。
- [x] 支持按课程和文件过滤，并返回 Top K chunk 与来源。
- [x] 接通 `/api/rag/index` 与 `/api/rag/search`。
- [x] 使用真实 Notebook 验证“解析 → 入库 → 查询 → 来源回溯”。

## 完整检索链路（最小闭环后）

- [ ] SQLite 文档登记、版本、发布日期和结构关系持久化。
- [ ] 幂等增量更新、旧版本清理和文档删除。
- [ ] sparse 召回与 dense/sparse RRF 融合。
- [ ] `qwen3.7-text-rerank` 候选重排。
- [ ] `deepseek-flash` 查询理解、按需改写与多模态问题输入。
- [ ] visual query embedding、视觉召回及跨表示融合。
- [ ] 相邻 chunk、cell、原图和原表的上下文回取。
- [ ] 超时、限流、部分失败、任务状态和可观测性完善。

## 后续阶段

- [ ] 构建 Retrieval Evaluation Set 与评估脚本。
- [ ] 接入 PDF 解析与 PDF 专用切块策略。
- [ ] 接入 GPT 回答层和网页交互。
