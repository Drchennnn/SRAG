# 逐次评估记录

每次效果评估结束后新增`YYYYMMDD-HHMMSS_任务.md`，失败/中断也保留；时间用Asia/Hong_Kong。普通单测随对应修改记录，不单独建档。

## 简短模板

```markdown
# 评估任务
时间：…；范围：题集/规模、模型与版本。

| 指标 | 结果 |
| --- | --- |
| 关键指标 | 数值及分母 |
| 延迟/用量 | … |

- 评价：收益、失败类型与局限；条件不同不直接比较。
- 下一步：保留/修复/补测及原因。
- 原始报告：唯一文件路径及SHA256。

## 维护约定
只修正转录错误并注明原因，新运行另建档。
```

## 维护约定

- 用中文标题、少量表格和短结论，通常不超过一页；不贴JSON、长日志或重复讲指标原理。
- 原始回答/逐题数据用唯一文件名留在`data/evals/`；无数据注明未测，不补猜测分数。
- 指标口径引用 [评估说明](../README.md)，重要决策再汇总；完整模型配置和版本指纹放原报告。

## 历史记录

均依据已有报告补记，未重跑。无准确时间的记录用`undated`，不由文件修改时间推断。原始报告保留本地，以下摘要可提交仓库。

| 评估内容 | 记录 |
| --- | --- |
| Notebook：RRF检索基线 | [查看](./20260929-180141_rrf_baseline.md) |
| Notebook：追问补全诊断 | [查看](./20261001-143433_query_notebook_dialogue_current_rule.md) |
| PDF：追问补全诊断 | [查看](./20261001-143524_query_dialogue_file_current_rule.md) |
| Notebook：新增cell题检查 | [查看](./20261001-143651_query_notebook_holdout_current_rule.md) |
| Notebook新题：规则与模型分类对比 | [查看](./20261001-145714_query_intent_holdout.md) |
| PDF：规则与模型分类对比 | [查看](./20261001-150048_query_intent_pdf_diagnostic.md) |
| Notebook：规则与模型分类对比 | [查看](./20261001-150131_query_intent_notebook_diagnostic.md) |
| 自然用户提问：主题切换诊断 | [查看](./20261001-155748_real_dialogue_report.md) |
| 对话验收：早期重跑 | [查看](./20261002-170505_chat_acceptance_final.md) |
| 对话验收：环境复核 | [查看](./20261002-170922_chat_acceptance_verified.md) |
| 迁移检查：首次失败 | [查看](./20261002-171108_chat_fresh_probe.md) |
| 迁移检查：引用修复回归 | [查看](./20261002-171549_chat_fresh_probe_current.md) |
| 对话验收：开发回归 | [查看](./20261002-171612_chat_acceptance_current.md) |
| 相机公式：两轮冒烟检查 | [查看](./20261002-173148_formula_smoke_report.md) |
| 迁移检查：阶段最终结果 | [查看](./20261002-173556_chat_fresh_probe_report.md) |
| 对话验收：阶段最终结果 | [查看](./20261002-173709_chat_acceptance_report.md) |
| CS5495：课程范围检索 | [查看](./undated_cs5495_formal_retrieval.md) |
| CS5495：全库范围检索 | [查看](./undated_cs5495_global_retrieval.md) |
| CS5187：开发集视觉路由对比 | [查看](./undated_slide_development_visual_route.md) |
| CS5187：后建题集视觉路由对比 | [查看](./undated_slide_holdout_visual_route.md) |
| 幻灯片：早期文本与视觉检索对比 | [查看](./undated_slide_retrieval_baseline.md) |
