# SRAG Agent 状态

> 你是一名 RAG 和 Agent 工程师，以可评估、可追溯、可增量更新的方式推进项目。开始前阅读 [README.md](./README.md) 与 [RAG.md](./RAG.md)；任务以 [CHECKLIST.md](./CHECKLIST.md) 为准。

- **当前阶段**：IPYNB、PDF/PPTX 已形成入库与检索闭环；正式入口为文本 RRF 加按需视觉召回。自动回答尚未实现。
- **当前目标**：采集真实连续对话验证追问误扩展与漏扩展；随后构建以文字和课程引用为主的回答链路。
- **评估边界**：CS5495 跨课程 24 题正式入口 Recall@5=0.875；图表细节 1/4。既有题集已用于诊断，下一轮需新资料验收。
- **最近变更**（2026-10-01）：独立评估 DeepSeek 追问分类器：新 cell 题命中 2/4→4/4，但 PDF 独立题误扩展 3/8、伤害 1/8，暂不接入默认检索。逐题报告见 [评估资产](./evals/README.md)，历史见 [PROGRESS_HISTORY.md](./PROGRESS_HISTORY.md)。

重大阶段或关键结果变化时更新本文件；实现细节放在 README/RAG，任务勾选放在 CHECKLIST，旧里程碑放在 PROGRESS_HISTORY。
