# 评估资产

只保留可重跑的题集、评估脚本和仍用于判断当前方案的逐题报告。历史消融的汇总指标记在本页；数值只衡量来源召回，不代表回答质量。题集均为人工构造，已参与开发分析。

| 任务 | 题集与入口 | 保留的逐题结果 |
| --- | --- | --- |
| Notebook 检索 | `retrieval_cases.jsonl`、`retrieval_corpus.json`（7 份原文件哈希）；`run_retrieval_ablation.py` | [RRF 基线](./results/rrf_baseline.json) |
| 幻灯片检索 | `slide_cases.jsonl`、`slide_holdout_cases.jsonl`、`cs5495_slide_cases.jsonl`；`app.evaluation.slide_retrieval` 和 `run_formal_slide_eval.py` | [CS5187 开发集](./results/slide_development_visual_route.json)、[CS5187 保留集](./results/slide_holdout_visual_route.json)、[CS5495 课程范围](./results/cs5495_formal_retrieval.json)、[CS5495 全库范围](./results/cs5495_global_retrieval.json) |
| 追问构造 | `query_*cases.jsonl`；`run_query_followup.py` 支持页码或 Notebook cell | [PDF 模拟对话](./results/query_dialogue_file_current_rule.json)、[Notebook 模拟对话](./results/query_notebook_dialogue_current_rule.json)、[Notebook 新 cell 题](./results/query_notebook_holdout_current_rule.json) |
| 追问意图分类 | 同上三组题集；`run_query_intent.py` 比较规则与 DeepSeek 决策，并对新增补全重新检索 | [Notebook 新 cell 题](./results/query_intent_holdout.json)、[PDF 诊断题](./results/query_intent_pdf_diagnostic.json)、[Notebook 诊断题](./results/query_intent_notebook_diagnostic.json) |

Notebook 的 20 题早期消融：dense Recall@8=0.95、RRF=1.00、RRF+outline=0.85（Recall@5=0.15）、RRF+outline+rerank=1.00。Outline 单独合并会压过正文，故未进入正式入口；rerank 结果只属于实验路径。

CS5187 的 28 题开发集：早期文本 Dense、等权融合 Recall@5 均为 0.964，视觉为 1.000；当前视觉路由为 1.000。另一份 28 题页码集：文本 RRF、等权融合均为 0.893，视觉为 0.929，当前路由为 0.964。CS5495 的 24 题离线视觉路由 Recall@5=0.917；正式入口按课程及全库检索均为 0.875，图表细节题仅 1/4 命中。

追问规则的历史诊断：首批 8 道独立问题误扩展 4 道、伤害 2 道；另一份 8 道独立问题在旧规则下误扩展 7 道、伤害 2 道，加入具体主题判断后均降为 0。两份 PDF 的模拟对话题从误扩展 2/8、伤害 1/8 降为 0；Notebook 诊断题修正“在哪段”后，追问 Top 5 命中从 1/8 升为 8/8。新留出的 Notebook cell 题中，追问目标命中 2/4，两道漏检强制补全后都排第 1。后两项保留逐题报告。

DeepSeek 意图分类仅用上一条用户问题和当前问题判断是否补全，不生成改写，不决定文件范围。与当前规则比较：新 Notebook cell 题的追问目标 Top 5 从 2/4 升至 4/4，独立题误扩展 0/4；PDF 诊断题追问仍为 8/8，但独立题误扩展 3/8，其中 1 道掉出 Top 5；Notebook 诊断题追问仍为 8/8，独立题 Top 5 仍为 6/8。三组共 40 题，模型追问命中 20/20、独立题误扩展 3/20，1 题受伤害；规则追问命中 18/20、独立题误扩展 0/20。PDF 的“这个主点”虽有具体主题，模型仍要求上文，导致其原本第 1 的目标页漏检。40 次分类累计 18086 tokens，平均单次约 1.9–2.1 秒。分类提示词参考过新 cell 题的漏检现象，另外两组也曾用于规则开发，因此三组都是诊断集，并非对分类器的盲测；此结果不支持默认接入，仍需真实对话与全新题集检验。报告记录题集哈希、可见索引版本、模型与逐题排名。

复跑分类实验需先有相应基线报告和本地索引，再配置 DeepSeek 与 Qwen API；脚本按题集哈希及索引可见版本拒绝不匹配的基线，调用付费 API。默认运行新 Notebook cell 题；其余两组分别传入 `--cases`、`--baseline`、`--output`。分类 JSON 空输出时仅重试一次，并在报告中记录重试次数和 token 用量。

`run_retrieval_ablation.py --index` 可按题集文件幂等索引 Notebook；默认评估不写入索引。运行评估会调用模型 API，重跑前应固定题集、索引版本和模型。部分报告记录 `case_file_sha256` 与 `corpus_versions`，可用来核对可比性；缺少版本字段的旧报告还需另核对入库快照。
