# SRAG Agent 状态

> 你是一名 RAG 和 Agent 工程师，负责以可评估、可追溯、可增量更新的方式推进本项目。开始工作前阅读 [README.md](./README.md) 和 [RAG.md](./RAG.md)；任务状态以 [CHECKLIST.md](./CHECKLIST.md) 为准，完整历史见 [PROGRESS_HISTORY.md](./PROGRESS_HISTORY.md)。

## 当前状态

更新时间：2026-09-29

- 阶段：Notebook 的 Dense+Sparse RRF 正式检索链路已完成；PDF/PPTX 仍是组件原型。
- 基线：20 题中 RRF 的 Recall@1=0.85、Recall@8=1.00、MRR@8=0.893；Outline 已禁用。
- 当前目标：先把 Rerank、查询理解、视觉召回和上下文扩展接入正式查询入口，再稳定并接入 PDF/PPTX。
- PDF/PPTX 缺口：依赖锁定、可重复测试、稳定 ID、页码/幻灯片与资源溯源、统一协议和索引 API 集成。
- 阻塞：无。

## 最近变更

| 日期 | 结果 |
| --- | --- |
| 2026-09-29 | 正式查询入口采用 Dense+Sparse RRF；20 题基线达到 Recall@8=1.00。 |
| 2026-09-29 | Outline 因标题候选压过正文 chunk 而停用；Rerank 高精度结果保留在独立消融路径。 |
| 2026-09-29 | 新增 PDF/PPTX Parser 和 Chunker 原型，尚未进入正式索引链路。 |
| 2026-09-29 | 复核组件边界并同步 README、RAG、CHECKLIST；未集成能力不再标记为正式完成。 |

## 更新规则

- 仅在阶段、目标、阻塞或关键评估结果变化时更新本文件。
- 本文件最多保留最近 3–5 条变更；更早记录移入 [PROGRESS_HISTORY.md](./PROGRESS_HISTORY.md)。
- 架构和方法写入 README/RAG，具体任务勾选写入 CHECKLIST。
