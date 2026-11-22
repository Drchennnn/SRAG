# M2 实验切分等价归属修复独立门禁审核报告

- **审核分支**：`review/m2-condition-fix-gate`
- **交付候选**：`6b2ef56dfdc05c30385ae1c1ed2f94ee4eb652a2`（`feat/m2-condition-gen-fix`）
- **产品修复提交**：`ef44e00e7b3d074e6200d51535d57352dd14b027`（经 `git diff ef44e00 6b2ef56 -- app/ tests/` 核对产品与测试代码完全一致）
- **修前基线**：`cfcfe5e79bde6dbd8b5346da752e4d527f4e18cd`
- **环境隔离**：全程使用临时 detached worktree、独立空库、独立数据目录与随机端口；不修改产品代码；`A03` 继续封存；保留 `docs/G/M2-condition-gate/` 旧档案不动。

---

## 1. 版本、分母与旧档案差异核对

1. **开发侧 `post_fix_evidence.json` 的 `subject_commit` 失真**：其记录为 `cfcfe5e79bde6dbd8b5346da752e4d527f4e18cd`（修前基线 SHA），原因是开发侧在提交产品修复 `ef44e00` 之前运行了 `verify.py`。
2. **开发侧分母少计 1 问**：开发侧 `README.md` 声称每种模式 `12/12`（“4 组归属新对照（6 问）+ 原 6 组回归 = 12/12”），但 `post_fix_evidence.json` 实际包含 `1 + 1 + 2 + 2 + 1 + 6 = 13` 问（归属新对照实际为 7 问）。本轮独审按 **13 问** 逐项独立评分。
3. **实际运行进程核验**（完整记录于 [gate_evidence.json](./gate_evidence.json) 的 `process_meta`）：
   - 基线与候选的 `Mode OFF` 及 `Mode ON`（HTTP `/__audit_meta`）均实测返回 detached worktree 的真实 `git_sha`、`worktree_status: "clean"`、`app.rag.evidence_supplement.__file__` 路径及生效 `Settings`（`workspace_answer_enabled`、`workspace_retrieval_strategy="fts"`、`workspace_input_token_limit=32768`、`source_vision_enabled=True`、独立临时 `sqlite_path`/`data_dir` 与端口）。

---

## 2. 基线（`cfcfe5e`）失败确认

| 检查维度 | 实测结果 | 结论 |
| --- | --- | --- |
| **函数级 `_condition_problem`** | 无 B 句返回 `experiment_split_mismatch`；追加 `For Trial B only, held-out is used as the test split.` 后返回 `None` | **确认失败** |
| **生成关闭（`only_b` 问 A）** | `answer_status="complete"`，直接输出 `实验 TrialA，切分 held-out，样本数 4。MAE（amplitude）基线为 0.26，试验为 0.19。` | **确认失败** |
| **生成开启（Chromium + `deepseek-flash`，`only_b` 问 A）** | `generation.status="succeeded"`（2 次 HTTP 200），同样将 `TrialA` 的 `held-out` `0.26`/`0.19` 当作测试集答案输出并引用 `Lab-OnlyB.ipynb` | **确认失败** |

---

## 3. 候选（`6b2ef56`）双模式独立首跑汇总（13 问逐项评分）

| 配置模式 | 首跑得分（分子/分母） | 失败项 | 供应商异常 | 按原判据结论 |
| --- | --- | --- | --- | --- |
| **生成关闭（`workspace_answer_enabled=False`）** | **13 / 13** | 无 | 不适用（`external_call_count=0`） | **PASS** |
| **真实网页生成开启（`workspace_answer_enabled=True`，Chromium + `deepseek-flash` + 点击引用）** | **13 / 13** | 无（另列 2 处假性 `partial` 缺陷与开发侧 1 处无引用数值泄漏缺陷） | **0 异常**（26/26 次 `https://api.deepseek.com/v1/chat/completions` HTTP 200） | **PASS（附缺陷清单）** |

### 3.1 逐项实测表（13 问）

| 序号 | 套件与问题 | 生成关闭（`mode_off`） | 生成开启（`mode_on_browser`，含点击引用核对） |
| --- | --- | --- | --- |
| 1 | `1_only_b_declares_equiv` / `ask_a` | `partial`，仅引用 `ev-1`（`Guide-A.ipynb`），无 `0.26`/`0.19` | `gen=partial`，`rej={'ev-2': 'experiment_split_mismatch'}`，点击 `ev-1`（`Guide-A.ipynb · Cell 1`），无 `0.26`/`0.19` |
| 2 | `2_a_declares_equiv` / `ask_a` | `complete`，输出 `TrialA` `0.26`/`0.19`，引用 `ev-1`, `ev-2` | `gen=succeeded`，`rej=None`，输出 `TrialA` `0.26`/`0.19`，点击 `ev-2`（`Guide-A.ipynb`）与 `ev-1`（`Lab-ASelf.ipynb`） |
| 3 | `3_same_cell_fwd` / `ask_a` | `complete`，输出 `TrialA` `0.26`/`0.19`，引用 `ev-1`, `ev-3` | `gen=succeeded`，`rej=None`，输出 `TrialA` `0.26`/`0.19`，点击 `ev-1`（`Guide-AB.ipynb · Cell 1`）与 `ev-3`（`Lab-AB-Fwd.ipynb · Cell 1`） |
| 4 | `3_same_cell_fwd` / `ask_b` | `partial`，仅引用 `ev-1`（`Guide-AB.ipynb · Cell 2`），无 `0.26`/`0.19`/`0.44`/`0.31` | `gen=partial`，`rej={'ev-3': 'experiment_split_mismatch'}`，点击 `ev-2`（`Guide-AB.ipynb · Cell 2`），无实验数值 |
| 5 | `3_same_cell_rev` / `ask_a` | `complete`，输出 `TrialA` `0.26`/`0.19`，引用 `ev-1`, `ev-3` | `gen=partial`（※假性 `partial`），`rej=None`，输出 `TrialA` `0.26`/`0.19`，点击 `ev-2`（`Guide-AB.ipynb · Cell 1`）与 `ev-1`（`Lab-AB-Rev.ipynb · Cell 1`） |
| 6 | `3_same_cell_rev` / `ask_b` | `partial`，仅引用 `ev-1`（`Guide-AB.ipynb · Cell 2`），无实验数值 | `gen=partial`，`rej={'ev-1': 'experiment_split_mismatch'}`，点击 `ev-3`（`Guide-AB.ipynb · Cell 2`），无实验数值 |
| 7 | `4_explicit_diff` / `ask_a` | `partial`，仅引用 `ev-1`（`Guide-A.ipynb`），无实验数值 | `gen=partial`，`rej={'ev-1': 'experiment_split_mismatch'}`，点击 `ev-2`（`Guide-A.ipynb · Cell 1`），无实验数值 |
| 8 | `5_original_six` / `held_out` | `complete`，输出 `SlateTrialB6` `0.26`/`0.19` | `gen=partial`（※假性 `partial`），`rej=None`，输出 `SlateTrialB6` `0.26`/`0.19` 及逐样本误差，点击 `ev-5`（`Basalt-Latch-Guide.pdf`）与 `ev-4`（`Latch-Trial.ipynb`） |
| 9 | `5_original_six` / `salt_fog` | `partial`，仅引用 `ev-1` | `gen=partial`，指出缺少盐雾实验，点击 `ev-4`（PDF）、`ev-1..ev-3`（PPTX） |
| 10 | `5_original_six` / `unfamiliar_condition` | `partial`，仅引用 `ev-1` | `gen=partial`，`rej={'ev-4': 'experiment_condition_unverified'}`，点击 `ev-5`（PDF） |
| 11 | `5_original_six` / `different_object` | `partial`，仅引用 `ev-1` | `gen=partial`，`rej={'ev-4': 'experiment_condition_unverified'}`，点击 `ev-5`（PDF） |
| 12 | `5_original_six` / `different_metric` | `partial`，仅引用 `ev-1` | `gen=partial`，`rej={'ev-4': 'experiment_metric_mismatch'}`，指出缺少 RMSE（本轮未附带 MAE 数值），点击 `ev-5`（PDF） |
| 13 | `5_original_six` / `test_split` | `partial`，仅引用 `ev-1` | `gen=partial`，`rej={'ev-4': 'experiment_split_mismatch'}`，点击 `ev-5`（PDF） |

---

## 4. 专项核对与观察到的缺陷

### 4.1 “数值正确但状态 `partial`”样例核对
- **涉及样例**：开发侧 `post_fix_evidence.json` 中的 `3_same_cell_fwd.ask_a` 与 `5_original_six.held_out`；本轮独审首跑中的 `3_same_cell_rev.ask_a` 与 `5_original_six.held_out`。
- **缺项是否真实**：**不真实**。讲义与实验记录已通过 `Experiment-ID`（`TrialA` / `SlateTrialB6`）建立完整对应，且包含 `Split`、`Metric`、`Baseline`、`Trial`、`Samples` 及等价说明；生成关闭模式对这几项均判 `complete`。生成开启时，`deepseek-flash` 自行额外要求“针对定义阈值（8 V / 连续三次读数）的独立验证流程说明”，从而输出 `missing_evidence` 并将 `generation.status` 降为 `partial`。
- **裁决与缺陷定性**：按原判据（定义、实验身份、数值、直接引用及无虚构）仍可通过；单列为 **缺陷 1：生成模型过度保守导致的假性 `partial`**。

### 4.2 “回答中附带其他指标数值”核对
- **涉及样例**：开发侧 `post_fix_evidence.json` 中的 `5_original_six.different_metric`（回答文本：`缺少留出集实验的 RMSE 数值；现有证据只给出 MAE（基线 0.26、试验 0.19，单位 amplitude）。`）。
- **文字是否区分指标**：**是**，文本清晰区分了缺失的 `RMSE` 与现有记录的 `MAE`，未将 `MAE` 当作 `RMSE`。
- **可点击引用是否支持这些数值**：**不支持**。该条记录中 `Latch-Trial.ipynb`（`ev-5`）已被 `experiment_metric_mismatch` 拒绝，前端可点击引用仅剩 `ev-1`（`Basalt-Latch-Guide.pdf` 第 1 页，仅含定义与 `Experiment-ID: SlateTrialB6`，不含 `0.26`/`0.19`）。因此用户点击引用无法在原件中核对到模型说出的 `0.26`/`0.19`。
- **裁决与缺陷定性**：本轮独审首跑的 `different_metric` 未输出 `0.26`/`0.19`；开发侧记录的现象单列为 **缺陷 2：被拒实验的指标数值偶发残留于模型提示文本且无对应可点击引用支撑**。

---

## 5. 复跑命令

```bash
# 1. 静态检查与定向单测（detached worktree 6b2ef56 实测：ruff All checks passed!；pytest 92 passed）
git -C /Users/Link/code/SRAG worktree add --detach /tmp/srag-cand-check 6b2ef56dfdc05c30385ae1c1ed2f94ee4eb652a2
/Users/Link/code/SRAG/.venv/bin/ruff check app tests
PYTHONPATH=/tmp/srag-cand-check /Users/Link/code/SRAG/.venv/bin/pytest -q \
  /tmp/srag-cand-check/tests/test_m2_condition_generation.py \
  /tmp/srag-cand-check/tests/test_evidence_answer.py \
  /tmp/srag-cand-check/tests/test_evidence_supplement.py

# 2. 独立门禁双模式全量复验（基线确认 + 候选 13 问生成关闭/真实网页生成开启）
cd /Users/Link/code/SRAG-m2-cond-gate
/Users/Link/code/SRAG/.venv/bin/python docs/G/M2-condition-attribution-gate/verify.py \
  --out docs/G/M2-condition-attribution-gate/gate_evidence.json
```
