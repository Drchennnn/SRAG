# 评估工具与验收

本页负责工具、启动命令、指标、验收标准和重要历史结果。架构见 [RAG](../RAG.md)，服务启动见 [项目 README](../README.md)。`tests/` 验证代码行为，效果评估使用下列题集。

历次简短评价见 [runs/记录目录](./runs/README.md)；目前补记已有原始报告，未重新运行评估。

## 工具

| 入口 | 输入 | 方法/适用范围 |
| --- | --- | --- |
| [run_chat_acceptance.py](./run_chat_acceptance.py) | `chat_acceptance_cases.json` 17轮；`chat_fresh_probe_cases.json` 6轮 | 当前ChatSession端到端；任务、范围、证据、引用、总结、延迟和指纹，默认Top6文本检索 |
| [run_retrieval_ablation.py](./run_retrieval_ablation.py) | `retrieval_cases.jsonl` 20题；`retrieval_corpus.json` 7份文件哈希 | Notebook Dense/Sparse/RRF/Outline/Rerank消融 |
| [slide_retrieval.py](../app/evaluation/slide_retrieval.py) | `slide_cases.jsonl`、`slide_holdout_cases.jsonl`各28题；`cs5495_slide_cases.jsonl`24题 | 页级文本/视觉/融合/路由离线消融 |
| [run_formal_slide_eval.py](./run_formal_slide_eval.py) | 上述幻灯片题集 | 正式HybridRetriever课程/全库检索，非HTTP传输测试 |
| [run_query_followup.py](./run_query_followup.py) | `query_dialogue_file_cases.jsonl`、`query_notebook_dialogue_cases.jsonl`各16题；`query_notebook_holdout_cases.jsonl`8题 | 检索API的raw/contextual对照，不含完整对话逻辑 |
| [run_query_intent.py](./run_query_intent.py) | 同query题集及对应规则基线 | 规则/DeepSeek分类的收益与伤害，离线实验 |
| [run_real_dialogues.py](./run_real_dialogues.py) | 本地匿名`real_dialogues.jsonl` | 底层追问分类诊断，不测回答质量；助手回答只统计是否存在 |

## 启动方法

在项目根目录执行。API评估需要`.env`、Qdrant及对应可见资料，会产生模型费用；报告默认写入忽略的`data/evals/`。

```bash
# 对话：先校验版本，再运行；迁移集用 --cases/--output 指定
uv run python -m evals.run_chat_acceptance --validate-only
uv run python -m evals.run_chat_acceptance
uv run python -m evals.run_chat_acceptance --cases evals/chat_fresh_probe_cases.json --output data/evals/chat_fresh_probe_report.json
# 来源检索
uv run python -m evals.run_retrieval_ablation --strategies dense rrf
uv run python -m app.evaluation.slide_retrieval
uv run python -m evals.run_formal_slide_eval --cases evals/cs5495_slide_cases.jsonl --output data/evals/cs5495_global_retrieval.json
# 默认Notebook8题：先生成同题集基线，再调用分类模型
uv run python -m evals.run_query_followup --oracle-missed
uv run python -m evals.run_query_intent
# 代码验证，不调用付费模型
uv run pytest -q
uv run ruff check app evals tools tests
```

query其他题集传`--cases/--output`，分类还需同题集`--baseline`；哈希/版本不匹配报错。`--oracle-missed`只诊断强制补全的潜在收益；Notebook消融`--index`会写索引，冻结评估时不用。文档依赖缺失可能跳过PDF/PPTX测试，应查看pytest摘要。

真实样本按一行JSON一段对话保存`data/evals/real_dialogues.jsonl`，首轮无`kind`，后续标`followup/independent/unclear`；运行`uv run python -m evals.run_real_dialogues`，可加`--compare-model`：

```json
{"session_id":"example","turns":[{"query":"什么是LIME？","assistant_reply":"实际回答"},{"query":"这个机制在哪段？","kind":"followup"}]}
```

## 指标

| 指标 | 定义 |
| --- | --- |
| Recall@k | Notebook逐题计算相关来源召回比例再平均；幻灯片/query按Top k是否命中可接受页/cell计题命中率 |
| MRR@k / nDCG@k | 首个相关排名倒数的题均值 / 按标注相关程度归一化的排序增益；nDCG目前用于Notebook |
| 追问误扩展/伤害 | 错补历史的独立题占比 / 原文Top5命中、补全后漏检的独立题数 |
| 来源组召回率 | 已完成机械评估轮中命中组数/标注组数，每组any_of命中一处即可；异常/未尝试轮不进分母 |
| 完成率/机械通过率 | 完成轮数/题集总轮数；全部检查通过轮数/尝试轮数，异常计未通过 |
| 分项通过率 | 已机械评估且适用该检查的轮次中，通过数/适用数 |
| 引用/读取覆盖 | 编号合法及必要引用存在；总结已读文字块/总块数。均不等于事实或主题覆盖正确率 |
| 延迟/tokens | nearest-rank p50/p95；对话用量含回答、摘要、校对及reasoning，不含Embedding/视觉API |

无适用分母记null。来源组/分项率须结合完成率阅读。比较时固定题集、原文件、可见版本、模型/提示词、范围、Top K及视觉缓存/派生点；不能只看文档版本。

## 对话验收标准

**自动门**：无异常、任务/query/调用符合预期、范围不越界、必要引用存在且编号合法、总结读取全部文字块。来源组一般目标≥90%；现有回归集要求逐题全部标注组命中。`overall_passed`表示逐轮机械通过，`frozen_environment_passed`要求原文件/SQLite/Qdrant/缓存未变，二者同时满足才是`stage_automatic_passed`。

评测禁用视觉派生点发布；首次生成缓存仍会改变指纹，需要确认缓存后重跑。延迟和成本单列，当前无统一性能通过阈值。

**质量核对**：对照原文字/截图，按以下五项逐轮标0（错误/缺失）、1（部分）、2（充分），无适用项N/A：

| 维度 | 核对重点 |
| --- | --- |
| 事实/公式 | 定义、符号、维度、归一化、单位和统计含义 |
| 引用支持 | 页/cell支持对应结论，图表对象正确 |
| 答题覆盖 | 所需定义/公式/步骤；总结覆盖预标主题 |
| 对话一致性 | 正确承接指代、换题和指定范围 |
| 可学习性 | 解释清楚，补充与课件区分，未知不猜 |

质量评分尚未自动化，也未将旧23轮补打上述分数。记录`turn_id/分值/支持来源/错误类型/理由`；关键事实与引用项不得为0，其他缺项记录处理结论。模型校对不是评分器，不合成“总准确率”。错误按定位、召回、视觉转写、生成、引用、性能分类。

## 重要验收与实验结果

均为历史快照与回归/诊断结果；`holdout/fresh`题集已参与修复，不再是盲测。

| 日期/实验 | 结果与决策 |
| --- | --- |
| 09-29 Notebook20题 | Dense Recall@8=.95；RRF @1=.85、@8=1、MRR=.893、nDCG=.919；Outline融合@8=.85、@5=.15；再加Rerank @8=1、MRR=.950。保留RRF，其他仅实验。 |
| 09-30 CS5187两组各28题 | 开发集文本/等权融合Recall@5=.964、视觉/路由=1；后建集文本/融合=.893、视觉=.929、路由=.964。采用视觉意图路由。 |
| 09-30 CS5495正式24题 | 课程/全库Recall@5均.875，MRR=.748/.696，均延迟973/1161ms，图表1/4；离线路由Recall@5=.917。路径与范围分开比较。 |
| 10-01 规则诊断 | 早期两组独立题误扩展4/8、7/8，各伤害2题；具体主题判断后后一组均0。后续PDF/Notebook各16题追问Top5均1/8→8/8，误扩展/伤害0；Notebook独立题6/8。 |
| 10-01 分类对照40题 | Notebook新cell8题规则追问2/4，强制补全救回两题；合计规则/模型追问18/20、20/20，独立题误扩展0/20、3/20，模型伤害1题。18086tokens、均1.9–2.1s；不默认接分类器。 |
| 10-01 自然提问3段10问 | 7次相邻提问均经用户标为独立主题，规则/模型误扩展0/7、4/7；无实际AI回答和目标页，不用于答案/召回评分。 |
| 10-02 对话23轮 | 三课程、三格式：机械23/23，来源组22/22，合法引用23/23；五次总结读取19/19、26/26、54/54、29/29、37/37块，环境未变。达到终端试用标准。 |
| 10-03 清理验证 | 189测试、Ruff/编译/链接/题集版本校验通过；原23轮离线重聚合通过，未重调API。 |

对话来源抽检覆盖相机投影维度/归一化、高斯MLE的N与无偏N−1、Logistic的C、NB平滑、Canny及权重表/绘图代码；课程成绩30%/20%/50%只作为原课件事实。普通问答1.7–4.7s、公式19–31s、总结15–80s，共228712个对话模型tokens。迁移集首跑5/6，修复代码下标误认引用后6/6；不是答案准确率100%。

完整输出及实现/题集哈希留在`data/evals/chat_acceptance_report.json`、`chat_fresh_probe_report.json`；离线聚合在`chat_metrics_reaggregated.json`。旧11报告/2早期题集归档`data/evals/archive_20261002/`，阶段记录见 [PROGRESS_HISTORY](../PROGRESS_HISTORY.md)。

## 维护约定

- 每次效果评估（含失败/中断）在 [runs/](./runs/README.md) 新增简短结果评价，按模板记录指标、对比、局限和下一步；原始报告使用唯一文件名。

- 只维护工具入口、运行条件、指标分母、验收规则和影响决策的历史结果；架构和开发任务用链接引用。
- 新结果写明题集/版本、规模、方法和局限；修复样例算回归，无新实测不更新旧分数。
- 原始回答、哈希和逐题报告留`data/evals/`；本页不贴长日志，不重复每轮细节；废弃工具同步清理命令与引用。
