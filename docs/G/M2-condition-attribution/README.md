# M2 实验切分等价归属修复公开报告

- **起点提交**：`cfcfe5e79bde6dbd8b5346da752e4d527f4e18cd`（`feat/m2-condition-gen-fix`）
- **产品修复提交**：`ef44e00e7b3d074e6200d51535d57352dd14b027`（仅含 `app/rag/evidence_supplement.py` 与 `tests/test_m2_condition_generation.py`）
- **最终分支**：`feat/m2-condition-gen-fix`（最终分支 `HEAD` SHA 见远端推送回读）
- **独立工作树**：`/Users/Link/code/SRAG-m2-cond-gen-fix`（不改动原工作区、`A03` 及 `main`）
- **剩余限制**：
  1. 真实网页开启生成（`workspace_answer_enabled=True`）使用 `deepseek-flash` 进行在线问答，回答措辞随模型输出存在波动，条件拦截与状态裁决由确定性逻辑及 `tests/test_m2_condition_generation.py` 锁住。
  2. 当同一段落/Cell 中出现多个 `Experiment-ID` 标记且存在无实验主语的泛化等价句时，当前策略按“归属不明”保守拒绝（`experiment_split_mismatch`），要求等价声明显式归属到对应实验。
  3. 仅在问 `TrialA` 或 `TrialB` 的实验条件下验证同页/同 Cell 双实验；更复杂的多段跨页交叉引用未在本轮公开替身中展开。

---

## 1. 修前基线复现（`cfcfe5e79bde6dbd8b5346da752e4d527f4e18cd`）

### 1.1 函数级首败复现（[baseline_repro.json](./baseline_repro.json)）

- 问题：`实验 A 的测试集 MAE 是多少？`
- 实验记录字段：`{"split": "held-out", "metric": "MAE"}`
- 仅含 `Experiment-ID: TrialA. Basalt latch TrialA. Split: held-out. Metric: MAE. Baseline: 0.26. Trial: 0.19.` 时：`_condition_problem` 返回 **`experiment_split_mismatch`**。
- 在同一段后追加 `For Trial B only, held-out is used as the test split.` 后：`_condition_problem` 返回 **`None`**，`classify_experiment_evidence` 返回 `supported: ["ev-2"]`, `rejected: {}`。

### 1.2 真实网页修前首跑（[pre_fix_web.json](./pre_fix_web.json)）

导入 `Guide-A.ipynb`（`TrialA` 定义）与 `Lab-BOnly.ipynb`（`TrialA` 使用 `held-out` 得 `0.26 / 0.19`，后附 `For Trial B only, held-out is used as the test split.`），询问 `Basalt latch 的定义是什么？讲义所对应的测试集实验如何验证？`：

- **生成关闭（`workspace_answer_enabled=False`）**：`answer_status="complete"`，直接输出 `实验 TrialA，切分 held-out，样本数 4。MAE（amplitude）基线为 0.26，试验为 0.19。`（`ev-2`）。
- **生成开启（`workspace_answer_enabled=True`，真实 Chromium + `deepseek-flash`）**：`generation.status="succeeded"`，`external_call_count=1`，模型输出将 `TrialA` 的 `held-out` 结果 `0.26` / `0.19` 当作测试集答案且未被条件校验拦截。

---

## 2. 产品修复方案（`ef44e00e7b3d074e6200d51535d57352dd14b027`）

修改 [app/rag/evidence_supplement.py](../../../app/rag/evidence_supplement.py)：

1. **句级与实验 ID 归属解析**：新增 `_norm_exp_id`、`_sentence_bounds`、`_sentence_exp_refs`、`_ref_matches_target`、`_question_exp_id` 及 `_SENTENCE_BREAK` / `_EXP_REF_LABELED` / `_EXP_REF_CODE`。
2. **等价声明严格归属**：`_explicit_split_equivalence` 按句检查否定词（含 `different from`、`distinct from`、`不同于`、`并非`、`区别于`）与实验引用：
   - 若等价句明确提到实验标识（如 `Trial B` / `TrialB`），则必须与当前实验记录的 `target_experiment_id` 归一化匹配；
   - 若上下文存在多个 `Experiment-ID`（`ambiguous_context=True`），无明确实验归属的泛化等价句不再放行。
3. **同一指南多实验定义页兼容**：`classify_experiment_evidence` 中当其他定义片段不含实验指标字段（`not _FIELDS.search(item.reads[0].text)`）时忽略 `experiment_document_mismatch`，并在 `definition is None` 时回退提取片段自身的 `Experiment-ID`。

---

## 3. 修后验收结果（[post_fix_evidence.json](./post_fix_evidence.json)）

### 3.1 独立记分总览

| 配置模式 | 覆盖范围 | 得分 | 结论 |
| --- | --- | --- | --- |
| **生成关闭**（`workspace_answer_enabled=False`） | 4 组归属新对照（6 问）+ 原 6 组回归 = **12/12** | **12/12** | **PASS** |
| **生成开启**（`workspace_answer_enabled=True`，真实 Chromium + `deepseek-flash` + 点击引用） | 4 组归属新对照（6 问）+ 原 6 组回归 = **12/12** | **12/12** | **PASS** |

### 3.2 逐组详细结果

| 对照组 | 提问 | 生成关闭（`mode_off`） | 生成开启（`mode_on_browser`，含点击引用核对） |
| --- | --- | --- | --- |
| **1. 仅 B 声明等价** | 问 A 测试集 | `partial`，仅保留 A 定义（`ev-1`），未输出 `0.26/0.19` | `gen_status=partial`，`rejected_experiment_reasons={'ev-1': 'experiment_split_mismatch'}`，点击引用 `ev-2`（`Guide-A.ipynb · Cell 1`），无 `0.26/0.19` |
| **2. A 自己声明等价** | 问 A 测试集 | `complete`，输出 A 的 `0.26` / `0.19`，引用 `ev-1`, `ev-2` | `gen_status=succeeded`，`rejected=None`，输出 `0.26` / `0.19`，点击引用 `ev-1`（`Guide-A.ipynb`）与 `ev-2`（`Lab-ASelf.ipynb`） |
| **3. 同 Cell 双实验正序（A 在前、B 在后）** | 问 A 测试集 | `complete`，输出 `TrialA` `0.26` / `0.19`，引用 `ev-1`, `ev-3` | `gen_status=partial`（未出现 `TrialB` 数据），输出 `0.26` / `0.19`，点击引用 `ev-2`（`Guide-AB.ipynb · Cell 1`）与 `ev-1`（`Lab-AB-Fwd.ipynb · Cell 1`） |
| **3. 同 Cell 双实验正序（A 在前、B 在后）** | 问 B 测试集 | `partial`，仅保留 B 定义（`ev-1`），拒绝混用 `TrialA` | `gen_status=partial`，`rejected={'ev-1': 'experiment_split_mismatch'}`，点击引用 `ev-3`（`Guide-AB.ipynb · Cell 2`），无 `0.26/0.19` |
| **3. 同 Cell 双实验反序（B 在前、A 在后）** | 问 A 测试集 | `complete`，输出 `TrialA` `0.26` / `0.19`，引用 `ev-1`, `ev-3` | `gen_status=succeeded`，输出 `0.26` / `0.19`，点击引用 `ev-1`（`Guide-AB.ipynb · Cell 1`）与 `ev-3`（`Lab-AB-Rev.ipynb · Cell 1`） |
| **3. 同 Cell 双实验反序（B 在前、A 在后）** | 问 B 测试集 | `partial`，仅保留 B 定义（`ev-1`），拒绝混用 `TrialA` | `gen_status=partial`，`rejected={'ev-3': 'experiment_split_mismatch'}`，点击引用 `ev-2`（`Guide-AB.ipynb · Cell 2`），无 `0.26/0.19` |
| **4. 明确声明 held-out ≠ test** | 问 A 测试集 | `partial`，仅保留 A 定义（`ev-1`） | `gen_status=partial`，`rejected={'ev-2': 'experiment_split_mismatch'}`，点击引用 `ev-1`（`Guide-A.ipynb · Cell 1`），无 `0.26/0.19` |
| **5. 原 6 组回归：`held_out`** | 留出集 | `complete`，输出 `SlateTrialB6` `0.26` / `0.19` | `gen_status=partial`，输出 `0.26` / `0.19`，点击引用 `ev-1`（`Basalt-Latch-Guide.pdf`）与 `ev-5`（`Latch-Trial.ipynb`） |
| **5. 原 6 组回归：`salt_fog`** | 盐雾 | `partial`，仅引用 `ev-1` | `gen_status=partial`，指出缺少盐雾实验，点击引用 `ev-1`, `ev-2`, `ev-4` |
| **5. 原 6 组回归：`unfamiliar_condition`** | 低温循环 | `partial`，仅引用 `ev-1` | `gen_status=partial`，`rejected={'ev-5': 'experiment_condition_unverified'}`，点击引用 `ev-1` |
| **5. 原 6 组回归：`different_object`** | Quartz latch | `partial`，仅引用 `ev-1` | `gen_status=partial`，`rejected={'ev-5': 'experiment_condition_unverified'}`，点击引用 `ev-1` |
| **5. 原 6 组回归：`different_metric`** | RMSE | `partial`，仅引用 `ev-1` | `gen_status=partial`，`rejected={'ev-5': 'experiment_metric_mismatch'}`，页面明确说明“缺少留出集实验的 RMSE 数值；现有证据只给出 MAE（基线 0.26、试验 0.19）”，点击引用 `ev-1` |
| **5. 原 6 组回归：`test_split`** | 测试集 | `partial`，仅引用 `ev-1` | `gen_status=partial`，`rejected={'ev-5': 'experiment_split_mismatch'}`，点击引用 `ev-1` |

### 3.3 供应商失败记录（确定性测试覆盖）

- 由 `tests/test_m2_condition_generation.py::test_dual_split_selects_only_matching_split_and_handles_provider_failure` 覆盖：当外部生成供应商返回失败时，`trace.generation.status == "provider_error"` 且回答块安全降级为 `code == "excerpt_fallback"`，不编造或越权放行未验证实验结论。

---

## 4. 测试与复跑命令

```bash
cd /Users/Link/code/SRAG-m2-cond-gen-fix

# 1. 全量静态检查与单测（修后实测：ruff All checks passed!；pytest 636 passed, 3 skipped）
../SRAG/.venv/bin/ruff check .
../SRAG/.venv/bin/pytest -q

# 2. 归属定向回归单测（含新增 test_split_equivalence_attribution_rejects_other_experiment_and_preserves_order_invariance）
../SRAG/.venv/bin/pytest -q tests/test_m2_condition_generation.py tests/test_evidence_answer.py tests/test_evidence_supplement.py

# 3. 真实网页（Chromium + deepseek-flash）与生成关闭双模式端到端复验脚本
../SRAG/.venv/bin/python docs/G/M2-condition-attribution/verify.py
```
