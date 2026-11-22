# M2 最终网页回答两项缺陷修复交接报告

- **起点提交**：`6b2ef56dfdc05c30385ae1c1ed2f94ee4eb652a2`（`feat/m2-condition-gen-fix`）
- **独审基准**：`85b5bb59acf4c336c0badc6387d4bb11b37f97ce`（`review/m2-condition-fix-gate`）
- **产品修复提交**：`1fb70230d79583109dea44ccfab398de8fb2609c`（仅含 `app/rag/workspace_generation.py` 与 `tests/test_m2_condition_generation.py`）
- **最终分支**：`feat/m2-condition-gen-fix`（最终分支 `HEAD` SHA 见远端推送回读；`A03` 保持封存，不合入 `main`）
- **实测服务加载 SHA**：`1fb70230d79583109dea44ccfab398de8fb2609c`（`worktree_status: "clean"`，由 [post_fix_evidence.json](./post_fix_evidence.json) 各套件 `process_meta` 实证）

---

## 1. 两项缺陷的修前复现与产品修复

### 缺陷 1（优先）：缺 RMSE 时补述 MAE `0.26`/`0.19`，但只剩定义页引用
- **修前行为**：在 `6b2ef56` 的开发证据 `docs/G/M2-condition-attribution/post_fix_evidence.json`（`5_original_six.different_metric`）中，`ev-5`（`Latch-Trial.ipynb`）因 `experiment_metric_mismatch` 被拒绝、可点击引用仅剩定义页 `ev-1`（`Basalt-Latch-Guide.pdf`），但模型输出的 `missing` 字段仍含 `缺少留出集实验的 RMSE 数值；现有证据只给出 MAE（基线 0.26、试验 0.19，单位 amplitude）。`，导致无引用支撑的 `0.26`/`0.19` 展示在 `missing_evidence` 提示中。同时 `_supported_technical_statements` 仅校验英文技术词，未校验来自未引用/被拒证据的数字。
- **修复方案**（[app/rag/workspace_generation.py](../../../app/rag/workspace_generation.py)）：
  1. 在 `_supported_technical_statements` 中增加数字集合校验（`_NUMBER_TOKEN`）：若 `statement` 出现来源于 `evidence` 的数字但未出现在该条 `statement.citations` 或 `question` 中，予以剔除；
  2. 新增 `_sanitize_missing_numbers`：从 `missing` 中剔除仅来自被拒/未引用证据、且未出现在已保留 `statements` 所引用原文或 `question` 中的实验数值子句，保留准确的缺证说明（如 `缺少 RMSE 的数值。`），不把被拒实验标为支持 RMSE。

### 缺陷 2：问题已经答全，网页仍显示 `partial`（假性 `partial`）
- **修前行为**：在独审 `85b5bb5` 的 `3_same_cell_rev.ask_a` 与 `5_original_six.held_out` 中，讲义与实验已通过 `Experiment-ID` 完整闭合且给出 `0.26`/`0.19`，`deepseek-flash` 却额外要求“针对定义阈值（8 V / 连续三次读数）的独立验证流程”或“触发条件之间的关系”，输出 `missing_evidence` 并将 `generation.status` 降为 `partial`。
- **修复方案**（[app/rag/workspace_generation.py](../../../app/rag/workspace_generation.py)）：
  - 在 `mode == "definition_and_experiment"` 且 `supported_exp_ids` 非空、已保留结论实际引用了受支持实验、`rejected_exp_reasons` 为空且无未支持结论时，通过 `_EXTRA_VERIFICATION_DEMAND` 与 `_QUESTION_ASKS_EXTRA_VERIFICATION` 区分“模型自行追加的验证要求”与“用户明确追问”：仅当用户未追问独立验证流程/触发条件/机理时，清除这类多余 `missing` 子句；当用户明确追问（如反例 `请给出针对定义阈值的独立验证流程`）时，完整保留 `missing_evidence` 与 `partial`。

---

## 2. 固定输出回放 vs 真实网页结果（分列报告）

### 2.1 固定输出回放（确定性单测，`tests/test_m2_condition_generation.py`）
- `test_rejected_experiment_numbers_stripped_from_missing_and_uncited_statements`：直接回放开发侧公开 `different_metric` 输出（含引用 `ev-1` 的 `0.26`/`0.19` statement 与含 `0.26`/`0.19` 的 `missing`）。结果：`status == "partial"`、`rejected_experiment_reasons == {"ev-5": "experiment_metric_mismatch"}`、`unsupported_literal_statements == 1`，`statements` 与 `missing` 均不含 `0.26`/`0.19`，保留 `"RMSE"` 缺证说明。
- `test_extra_verification_demand_cleared_when_answered_but_kept_when_asked`：回放 `85b5bb5` 中 `5_original_six.held_out` 与 `3_same_cell_rev.ask_a` 的模型输出均恢复 `status == "succeeded"`（`missing == ""`）；加入用户明确追问 `请给出针对定义阈值的独立验证流程` 反例时保持 `status == "partial"` 且保留 `missing`。
- 全量静检与单测（干净提交 `1fb7023`）：`ruff check app tests` 通过；`PYTHONPATH=. pytest -q` 为 **638 passed, 3 skipped**。

### 2.2 真实网页首跑（干净提交 `1fb7023`，`Chromium + deepseek-flash`，分母由逐题记录生成）

- **真实模型波动记录**：`deepseek-flash` 在不同运行中对正例可能给出简洁或带逐样本误差的叙述（如 `held_out` 给出逐样本误差 tuple），对缺证组可能附带定义页中的 `dry bench` 观测说明（如 `salt_fog`、`test_split`），但在当前后处理约束下均未再出现无引用数值泄漏或假性 `partial`；`salt_fog` 中模型曾生成 1 条无原文直证的附带句（`unsupported_literal_statements=1`），已被自动过滤。
- **供应商调用**：共 28 次 `https://api.deepseek.com/v1/chat/completions` 请求全部 HTTP 200，**0 次供应商异常**。

| 配置模式 | 原 13 问归属得分 | 含反例总分（分母由记录生成） | 状态准确性 | 数值可点击核对 |
| --- | --- | --- | --- | --- |
| **生成关闭**（`workspace_answer_enabled=False`） | **13 / 13 PASS** | **14 / 14 PASS** | **14 / 14 准确** | **14 / 14 可核对** |
| **生成开启**（`workspace_answer_enabled=True`） | **13 / 13 PASS** | **14 / 14 PASS** | **14 / 14 准确** | **14 / 14 可核对** |

#### 逐项核对明细（14 问，来自 [post_fix_evidence.json](./post_fix_evidence.json)）

| 序号 | 套件 / 问题 | 生成关闭状态与引用 | 生成开启状态（`gen_status`） | 状态准确？ | 数值能否点击核对？ |
| --- | --- | --- | --- | --- | --- |
| 1 | `1_only_b_declares_equiv` / `ask_a` | `partial`（`ev-1`） | `partial`（`rej={'ev-1':'experiment_split_mismatch'}`，`ev-2`） | 是 | 是（仅定义 `8 V`，无 `0.26`/`0.19`） |
| 2 | `2_a_declares_equiv` / `ask_a` | `complete`（`ev-1`,`ev-2`） | `succeeded`（`ev-1`,`ev-2`） | 是 | 是（`Guide-A` + `Lab-ASelf` 核对 `8 V`、`0.26`/`0.19`） |
| 3 | `3_same_cell_fwd` / `ask_a` | `complete`（`ev-1`,`ev-3`） | `succeeded`（`ev-2`,`ev-1`） | 是 | 是（`Guide-AB Cell 1` + `Lab-AB-Fwd` 核对 `TrialA` `0.26`/`0.19`） |
| 4 | `3_same_cell_fwd` / `ask_b` | `partial`（`ev-1`） | `partial`（`rej={'ev-1':'experiment_split_mismatch'}`，`ev-3`） | 是 | 是（仅定义 `5 V`，无实验数值） |
| 5 | `3_same_cell_rev` / `ask_a` | `complete`（`ev-1`,`ev-3`） | `succeeded`（`ev-1`,`ev-3`） | 是 | 是（`Guide-AB Cell 1` + `Lab-AB-Rev` 核对 `TrialA` `0.26`/`0.19`） |
| 6 | `3_same_cell_rev` / `ask_b` | `partial`（`ev-1`） | `partial`（`rej={'ev-3':'experiment_split_mismatch'}`，`ev-2`） | 是 | 是（仅定义 `5 V`，无实验数值） |
| 7 | `4_explicit_diff` / `ask_a` | `partial`（`ev-1`） | `partial`（`rej={'ev-2':'experiment_split_mismatch'}`，`ev-1`） | 是 | 是（仅定义 `8 V`，无实验数值） |
| 8 | `5_original_six` / `held_out` | `complete`（`ev-1`,`ev-3`） | `succeeded`（`ev-5`,`ev-4`） | 是 | 是（`Basalt-Latch-Guide.pdf` + `Latch-Trial.ipynb` 核对 `0.26`/`0.19` 及误差） |
| 9 | `5_original_six` / `salt_fog` | `partial`（`ev-1`） | `partial`（`ev-4`） | 是 | 是（仅定义 `8 V`，无实验数值） |
| 10 | `5_original_six` / `unfamiliar_condition` | `partial`（`ev-1`） | `partial`（`rej={'ev-4':'experiment_condition_unverified'}`，`ev-5`） | 是 | 是（仅定义 `8 V`，无实验数值） |
| 11 | `5_original_six` / `different_object` | `partial`（`ev-1`） | `partial`（`rej={'ev-4':'experiment_condition_unverified'}`，`ev-5`） | 是 | 是（仅定义 `8 V`，无实验数值） |
| 12 | `5_original_six` / `different_metric` | `partial`（`ev-1`） | `partial`（`rej={'ev-4':'experiment_metric_mismatch'}`，`ev-5`，提示 `缺少 RMSE 的数值。`） | 是 | 是（仅定义 `8 V`，无 `0.26`/`0.19` 泄漏） |
| 13 | `5_original_six` / `test_split` | `partial`（`ev-1`） | `partial`（`rej={'ev-4':'experiment_split_mismatch'}`，`ev-5`） | 是 | 是（仅定义 `8 V`，无实验数值） |
| 14 | `6_counter_example` / `independent_verification_proc` | `partial`（`ev-1`） | `partial`（`rej={'ev-4':'experiment_condition_unverified'}`，`ev-5`，提示缺少独立验证流程） | 是 | 是（反例正确保留缺证提示与 `partial`） |

---

## 3. 复跑命令

```bash
cd /Users/Link/code/SRAG-m2-cond-gen-fix

# 1. 静态检查与全量单测
../SRAG/.venv/bin/ruff check app tests
PYTHONPATH=. ../SRAG/.venv/bin/pytest -q

# 2. 基于干净产品修复提交的双模式 14 问端到端验收
../SRAG/.venv/bin/python docs/G/M2-condition-answer-fix/verify.py \
  --commit 1fb70230d79583109dea44ccfab398de8fb2609c \
  --out docs/G/M2-condition-answer-fix/post_fix_evidence.json
```
