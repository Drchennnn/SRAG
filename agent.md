# SRAG Agent 状态

> 你是一名 RAG 和 Agent 工程师，以可评估、可追溯、可增量更新的方式推进项目。开始前阅读 [README.md](./README.md) 与 [RAG.md](./RAG.md)；任务以 [CHECKLIST.md](./CHECKLIST.md) 为准。

- **当前阶段**：IPYNB、PDF/PPTX 已形成入库与检索闭环；正式入口为文本 RRF 加按需视觉召回。自动回答尚未实现。
- **当前目标**：文字回答优先，提供文件/cell/页码引用；图片作为可选辅助。先改进图表细节召回，再实现 DeepSeek 回答。
- **评估边界**：CS5495 跨课程 24 题正式入口 Recall@5=0.875；图表细节 1/4。既有题集已用于诊断，下一轮需新资料验收。
- **最近变更**（2026-09-30）：清理未接入的占位上下文扩展与检索注入，精简初始文档；历史见 [PROGRESS_HISTORY.md](./PROGRESS_HISTORY.md)。

重大阶段或关键结果变化时更新本文件；实现细节放在 README/RAG，任务勾选放在 CHECKLIST，旧里程碑放在 PROGRESS_HISTORY。
