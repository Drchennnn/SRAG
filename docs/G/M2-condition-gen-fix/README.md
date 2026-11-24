# M2 开启文字生成路径的实验条件约束修复

- **起点提交**：`7e96dfad20c1a5892ba143201ae1c6fa1c699dc4`（接续独审报告 `a70dad31c93e378c9513f6db5798f25c422903c6`）
- **独立开发分支**：`feat/m2-condition-gen-fix`（worktree `/Users/Link/code/SRAG-m2-cond-gen-fix`，未改动原工作区或 `main`）
- **核心改动**：[evidence_supplement.py](../../../app/rag/evidence_supplement.py) · [workspace_generation.py](../../../app/rag/workspace_generation.py) · [test_m2_condition_generation.py](../../../tests/test_m2_condition_generation.py)
- **首跑复现与双模式实测证据**：[baseline_test_split_repro.json](./baseline_test_split_repro.json) · [verify.py](./verify.py) · [证据.json](./证据.json)

---

## 1. 复现与根因

1. **在 `7e96dfad` 复现 `test_split`**：
   - 提问 `"Basalt latch 的定义是什么？讲义所对应的测试集实验如何验证？"`，唯一实验记录 `Latch-Trial.ipynb` 仅载 `Split: held-out. Metric: MAE. Baseline: 0.26. Trial: 0.19. Samples: 4.`。
   - 当 `workspace_answer_enabled=False` 时，走 `evidence_answer` 正确拒收（`partial`，仅引讲义）；但当 `workspace_answer_enabled=True` 时，[workspace.py#L1007](../../../app/rag/workspace.py#L1007) 绕过 `_answer_with_supplement`，`deepseek-flash` 返回 `generation.status="succeeded"` 并直接输出 `held-out` 数值 `0.26 → 0.19`（详见 [baseline_test_split_repro.json](./baseline_test_split_repro.json)）。

---

## 2. 修复方案

1. **复用条件校验（`evidence_supplement.classify_experiment_evidence`）**：
   - 在 [evidence_supplement.py](../../../app/rag/evidence_supplement.py) 导出 `classify_experiment_evidence(evidence, question)`，复用 `_definition_in`、`_requested_experiment_in` 与 `_condition_problem` 判断每份结构化实验依据对当前问题的支持/拒收状态（不引入场景词表或特判）。
   - 在 [workspace_generation.py](../../../app/rag/workspace_generation.py) 的 `mode == "definition_and_experiment"` 校验阶段：过滤引用了被拒收实验依据的声明，记录 `rejected_experiment_reasons` 与 `unverified_experiment_statements`；保留同轮有依据的定义与检查表声明；若无任何受支持的实验依据，强制保留缺证提示使 `status` 不得为 `succeeded`。
2. **显式 `held-out` ↔ `test split` 等价识别（`_explicit_split_equivalence`）**：
   - 仅当原件（实验或定义文本）明确写明 `held-out` 用作/即 `test split` / `测试集` 且无否定前缀时允许匹配；仅凭词面相近不认定等价。

---

## 3. 实测与记分（生成关闭与生成开启分开记分）

### 3.1 原六组公开替身（`Basalt-Latch-Guide.pdf` + `Latch-Check.pptx` + `Latch-Trial.ipynb`）

- **文字生成关闭 (`workspace_answer_enabled=False`)**：**6/6 通过**（`held_out` 保留 `0.26 → 0.19` 与 `ev-1`/`ev-3`；`salt_fog`、`unfamiliar_condition`、`different_object`、`different_metric`、`test_split` 全部 `partial` 且零泄露）。
- **文字生成开启 (`workspace_answer_enabled=True`，真实 `HeadlessChrome/157.0.0.0` + `deepseek-flash` 从独立空库起跑并逐枚点击引用)**：**6/6 通过**（每轮 2 次 HTTP `200` 调用）：
  - `held_out`：保留定义（`ev-5`）与 `held-out` `0.26 → 0.19`（`ev-1`）。
  - `salt_fog`：`partial`，指出仅干台架观测、无盐雾实验。
  - `unfamiliar_condition`（低温）：`partial`，`rej={'ev-1': 'experiment_condition_unverified'}`，`unv=2`，仅留 `ev-5` 讲义引用，零泄露 `0.26 → 0.19`。
  - `different_object`（Quartz latch）：`partial`，`rej={'ev-1': 'experiment_condition_unverified'}`，`unv=2`，仅留 `ev-5`。
  - `different_metric`（RMSE）：`partial`，`rej={'ev-1': 'experiment_metric_mismatch'}`，`unv=2`，仅留 `ev-5`。
  - `test_split`（测试集）：`partial`，`rej={'ev-1': 'experiment_split_mismatch'}`，`unv=2`，仅留 `ev-5`，零泄露 `0.26 → 0.19`（修前为 `succeeded` 并错引 `held-out`）。

### 3.2 新增正反例（生成关闭与开启均各自通过）

1. **明确 `test` 正例（`Latch-Test-Positive.ipynb`，`Split: test. Baseline: 0.31. Trial: 0.22. Samples: 6.`）**：
   - 生成关闭：`complete`，`0.31 → 0.22`，`ev-1` + `ev-3`。
   - 生成开启：`succeeded`，`0.31 → 0.22`（6 样本），点击 `ev-1`/`ev-2` 均打开对应原件 `v1`。
2. **双切分对照反例（`Latch-Dual-Splits.ipynb`，Cell 1 `held-out 0.26 → 0.19` + Cell 2 `test 0.31 → 0.22`）**：
   - 问 `test`：生成关闭 `complete`（`0.31 → 0.22`，Cell 2）；生成开启 `rej={'ev-4': 'experiment_split_mismatch'}`、`unv=1`，仅保留 `0.31 → 0.22`（`ev-5` Cell 2）+ 定义（`ev-6`），未混用 `0.26 → 0.19`。
   - 问 `held-out`：生成关闭 `complete`（`0.26 → 0.19`，Cell 1）；生成开启 `rej={'ev-5': 'experiment_split_mismatch'}`，仅保留 `0.26 → 0.19`（`ev-4` Cell 1）+ 定义（`ev-6`）。

### 3.3 确定性测试、供应商失败与静态检查

- [test_m2_condition_generation.py](../../../tests/test_m2_condition_generation.py) 锁定：条件不符过滤、`held-out` ↔ `test split` 显式等价与否定/无说明区分、双切分防混用，以及供应商失败（`httpx.ConnectError` → `generation.status="provider_error"` + `excerpt_fallback`）。
- 全量离线 `pytest` 与 `ruff check` 通过（见交付输出）；v2 焦点重绑问题按要求单列，本轮未改动。
