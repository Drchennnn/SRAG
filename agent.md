# SRAG Agent 状态

> 你是一名 RAG 和 Agent 工程师，负责以可评估、可追溯、可增量更新的方式推进本项目。开始工作前阅读 [README.md](./README.md) 和 [RAG.md](./RAG.md)；任务状态以 [CHECKLIST.md](./CHECKLIST.md) 为准，完整历史见 [PROGRESS_HISTORY.md](./PROGRESS_HISTORY.md)。

## 当前状态

更新时间：2026-09-30

- 阶段：Notebook 检索基线已完成；统一 SlideDeck 预处理已实现，正在真实课件验收。
- 基线：20 题中 RRF 的 Recall@1=0.85、Recall@8=1.00、MRR@8=0.893；Outline 已禁用。
- 当前目标：在 Colab 用代表性 PDF 验收页内连续性、资源溯源和处理时间；通过后开始幻灯片 Embedding。
- 幻灯片策略：一页/一张为基础 Chunk；原生文本优先、按页 OCR 回退；整页视觉表示为主，只过滤明显装饰资源。
- 阻塞：无。

## 最近变更

| 日期 | 结果 |
| --- | --- |
| 2026-09-30 | 统一 SlideDeck Schema、PDF/PPTX Parser、SlideChunker、资源标注和预处理包已实现，已通过合成文件测试。 |
| 2026-09-30 | Colab T4 完成 32 页 PDF 预处理实测；确认当前 Markdown 标题切分存在页内断层，图片只保留占位符。 |
| 2026-09-30 | PDF/PPTX 统一改为 SlideDeck 设计；当前先完成解析和 Chunk，尚未开始幻灯片 Embedding。 |
| 2026-09-29 | 正式查询入口采用 Dense+Sparse RRF；20 题基线达到 Recall@8=1.00。 |
| 2026-09-29 | Outline 因标题候选压过正文 chunk 而停用；Rerank 高精度结果保留在独立消融路径。 |

## 更新规则

- 仅在阶段、目标、阻塞或关键评估结果变化时更新本文件。
- 本文件最多保留最近 3–5 条变更；更早记录移入 [PROGRESS_HISTORY.md](./PROGRESS_HISTORY.md)。
- 架构和方法写入 README/RAG，具体任务勾选写入 CHECKLIST。
