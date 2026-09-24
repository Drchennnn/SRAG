# SRAG Agent 状态

> 你是一名 RAG 和 Agent 工程师，负责以可评估、可追溯、可增量更新的方式推进本项目。开始工作前阅读 [README.md](./README.md) 和 [RAG.md](./RAG.md)；架构、目标与技术选型以这两份文档为准，不在本文件重复记录。

## 当前进度

更新时间：2026-09-24

- 阶段：Notebook 解析、父子切块与 Embedding 已贯通，进入存储实现。
- 已完成：真实 ipynb 只读解析；结构化父子切块；Qwen dense+sparse Embedding；Indexer 编排；文件输入实验 Notebook。
- 当前目标：将父块和文档版本写入 SQLite、子块向量写入 Qdrant。
- 后续：实现增量索引、混合检索、rerank API 和首批评测问题。
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
