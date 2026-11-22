# M2 测量数值与可点击引用直接核对修复交接报告

- **起点提交**：`8c4a9e234f3fb35d3fbd7944df47245165a516f9`（`feat/m2-condition-gen-fix`，保留原 14 问与 [M2-condition-answer-fix](../M2-condition-answer-fix/README.md) 原始首跑不动）
- **产品修复提交**：`8d530e4887330f422364375f897b29da5e97a224`（仅含 `app/rag/workspace_generation.py` 与 `tests/test_m2_condition_generation.py`）
- **最终分支**：`feat/m2-condition-gen-fix`（最终 `HEAD` SHA 见远端推送回读；`A03` 保持封存，不合入 `main`）
- **实测服务加载 SHA**：`8d530e4887330f422364375f897b29da5e97a224`（`worktree_status: "clean"`，由 [post_fix_evidence.json](./post_fix_evidence.json) 各套件 `process_meta` 实证）

---

## 1. 修前反例复现与可达层定位（基线 `8c4a9e2`）

以 `ev-1`（无测量值的定义页 `Guide.pdf`）和 `ev-2`（`TrialA` 留出集 `0.26`）为依据，在 `8c4a9e2` 上复现：
1. **函数级 `_supported_technical_statements`**：
   - ① 问 `"测试集测得多少？"`，声明 `"测试集测得 0.87。"`（仅引 `ev-1`，所有原件均无 `0.87`）：`8c4a9e2` 放行返回该声明（因 `stmt_nums & all_source_nums` 为空集）。
   - ② 问 `"测试集是否测得 0.26？"`，声明 `"测试集测得 0.26。"`（仅引 `ev-1`，`0.26` 来自用户猜测及另一切分 `ev-2`）：`8c4a9e2` 放行返回该声明（因 `question_nums` 被并入 `cited_nums`）。
2. **完整生成链路 `generate_response`**：当问题同时包含定义对象（如 `"Basalt latch 的定义是什么？讲义所对应的测试集实验..."`）使 `ev-1` 作为定义页不被 `classify_experiment_evidence` 拒绝时，上述两条错误声明在 `8c4a9e2` 均能穿透 `generate_response`，直接进入最终 `statements` 与网页前端 `text` 展示块。

---

## 2. 产品修复（`8d530e4887330f422364375f897b29da5e97a224`）

修改 [app/rag/workspace_generation.py](../../../app/rag/workspace_generation.py)：
1. **数字归一化与定位符豁免（`_norm_number_tokens`）**：忽略页码/版本号/Cell 编号（`_LOCATOR_OR_VERSION_NUM`），对数字做千分位与小数尾零归一化（`0.260` == `0.26`、`8.0` == `8`）。
2. **每条结论自身引用核对（`_supported_technical_statements`）**：
   - 取消 `all_source_nums` 过滤，任何出现在 `statement` 中的测量数字必须存在于该条 `statement.citations` 自身原文中；
   - 取消把 `question_nums` 无条件并入肯定声明的 `cited_nums`：仅当该条语句带有明确不确定/否定表述（`_UNCERTAINTY_CUE`，如 `"无法确认测试集是否为 0.26"`）时允许回显用户问题中的待核数字；肯定测量声明必须由自身引用原文直接支持。
3. **实验测量声明必须引用受支持实验（`_EXPERIMENT_CLAIM_CUE`）**：含测量数值的肯定实验声明（如 `"测试集测得 8"`）不能仅凭定义页 `ev-1` 放行，必须至少引用一条 `supported_exp_ids`（即已通过对象、切分、指标校验的实验记录）。
4. **两条展示路径协同**：正文 `statements` 与缺证提示 `missing_evidence`（`_sanitize_missing_numbers`）均执行数字与引用校验；在 `missing_evidence` 中仅保留 `kept_cids` 原文或用户问题待核条件中的数字，剔除凭空数字（`0.87`）与被拒实验残留数字。

---

## 3. 固定输出回放 vs 真实网页首跑（分列报告）

### 3.1 固定输出回放（`tests/test_m2_condition_generation.py::test_statement_measurement_numbers_require_direct_citation_support`）
- **凭空出现的数（`0.87`）**：函数级 `_supported_technical_statements` 与端到端 `generate_response` 均拦截，正文与 `missing` 均不含 `0.87`。
- **照抄用户猜测（`0.26` 仅引 `ev-1`）**：函数级与端到端均拦截，正文与 `missing` 均不含未确认的肯定结论 `0.26`。
- **数字相同但实验或指标不同（`TrialB` `0.26` / `RMSE` `0.26`）**：由 `classify_experiment_evidence` 与声明过滤拦截，返回 `partial` 且不含 `0.26`。
- **正确实验直接支持（含格式差异 `8.0` / `0.260`、`第 1 页`、`v1`）**：保留定义、页码、版本号与测量结论，状态 `succeeded`。
- **诚实回复缺证（`"无法确认测试集是否为 0.26。"`）**：正文与 `missing_evidence` 均保留不确定表述，状态 `partial`。
- **全量静检与单测**（干净提交 `8d530e4`）：`ruff check app tests` 通过；`PYTHONPATH=. pytest -q` 为 **639 passed, 3 skipped**。

### 3.2 真实网页与生成关闭回归（干净提交 `8d530e4`，[post_fix_evidence.json](./post_fix_evidence.json)）

- **分母由逐题记录自动生成**：`original_14 = 14`，`total_with_citation_counterexamples = 16`。
- **真实模型波动**：`deepseek-flash` 在 `3_same_cell_fwd.ask_a` 与 `5_original_six.held_out` 中因表述为“未提供具体验证步骤/如何直接验证定义”（未命中固定短语模板）而额外输出 `missing_evidence` 导致 `gen_status=partial`，但其全部正例数值（`0.26`/`0.19`）与可点击引用均完全正确；其余 14 项状态与预期完全一致。32 次 DeepSeek API 调用全部 HTTP 200，0 供应商异常。

| 配置模式 | 原 14 问归属得分 | 含 2 项引用数值反例总分 | 测量数值可点击核对 |
| --- | --- | --- | --- |
| **生成关闭**（`workspace_answer_enabled=False`） | **14 / 14 PASS** | **16 / 16 PASS** | **16 / 16 可核对**（`ask_test_guess_026` 返 `no_evidence`，无数值泄漏） |
| **生成开启**（`workspace_answer_enabled=True`，Chromium + `deepseek-flash`） | **14 / 14 PASS**（按原归属判据） | **16 / 16 PASS** | **16 / 16 可核对**（`ask_test_value`、`ask_test_guess_026` 均 `partial` 且仅引 PDF 定义页，无 `0.26`/`0.19`/`0.87` 泄漏） |

---

## 4. 复跑命令

```bash
cd /Users/Link/code/SRAG-m2-cond-gen-fix

# 1. 静态检查与全量单测
../SRAG/.venv/bin/ruff check app tests
PYTHONPATH=. ../SRAG/.venv/bin/pytest -q

# 2. 干净提交上的双模式 16 问端到端验收
../SRAG/.venv/bin/python docs/G/M2-condition-citation-verify/verify.py \
  --commit 8d530e4887330f422364375f897b29da5e97a224 \
  --out docs/G/M2-condition-citation-verify/post_fix_evidence.json
```
