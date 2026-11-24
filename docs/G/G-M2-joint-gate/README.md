# SRAG G28 + M2 联合候选独立审核报告

- **受测联合候选**：`codex/g-m2-joint-candidate` 远端完整 SHA `d4c8597d92db20a8803259f47c59faf16ca14f6a`
- **已核对祖先与整合**：包含 `18fb8929e6aaaa41de40dd34d933d34741c661e6`（`codex/g28-dev`）与 `e55a9b4a9281d127c78f6fe28f7511c064312884`（`feat/m2-condition-gen-fix`），且整合交接（`docs/执行交接.md`、`CHECKLIST.md`、`PROGRESS_HISTORY.md`、`README.md`）已发布。
- **审核分支**：`codex/g-m2-joint-gate`（仅含可公开报告、聚合证据与复跑入口；未改动产品代码，未运行 `A03`）
- **联合独审裁决**：**建议开放 A03**

---

## 一、服务进程实载核对

所有验证均在 `d4c8597d92db20a8803259f47c59faf16ca14f6a` 的干净 `detached worktree`、独立空 SQLite 库、独立临时 `data_dir` 与随机本地端口启动（详情见 [joint_aggregate.json](./joint_aggregate.json) 的 `process_verification`）：
- **实测进程 SHA 与状态**：`git_sha = d4c8597d92db20a8803259f47c59faf16ca14f6a`，`worktree_status = "clean"`。
- **实际代码加载路径**：`app/rag/workspace.py`、`app/rag/workspace_generation.py`、`app/rag/evidence_supplement.py` 均从临时 detached `candidate/` 目录加载，与主仓库工作区完全隔离。
- **生效配置**：`workspace_retrieval_strategy = "fts"`、`workspace_input_token_limit = 32768`、`source_vision_enabled = True`、`deepseek_chat_model = "deepseek-flash"`；M2 分别运行 `workspace_answer_enabled = False`（关闭生成）与 `True`（开启生成）。

---

## 二、固定模型输出回放、真实网页与异常汇总（分列）

| 类别 | 范围与命令 | 首跑实测结果 | 异常 / 回退 / 补跑 |
| --- | --- | --- | --- |
| **静态检查与离线全量** | `ruff check --no-cache app evals tools tests`；`pytest -q -p no:cacheprovider` | Ruff **全通过**；`pytest` **640 passed, 3 skipped**（缺 `SRAG_SOFFICE`） | 无 |
| **固定模型输出回放（M2）** | `pytest -v -p no:cacheprovider tests/test_m2_condition_generation.py` | **7 passed**（含被拒数值剥离、假性 `partial` 清理与反例保留、直接引用数值核对） | 无 |
| **真实网页 G28 冻结集（13 问）** | Chromium `/workspace/` + `deepseek-flash` | 原五题 **5/5**、前批四题 **4/4**、双文件四题 **4/4**；21 次 HTTP 200 | 0 异常 / 0 回退 / 0 补跑 |
| **真实网页 G28 公开逗号对照（14 轮）** | `docs/G/G28-comma-fix/fixture/control.json` | 15 次 HTTP 200，多候选澄清 0 外调 | 0 异常 / 0 回退 / 0 补跑 |
| **真实网页 M2 16 问（开启生成）** | [run_m2_16q.py](./run_m2_16q.py) | 32 次 HTTP 200，点击引用全打开 | 0 异常 / 0 回退 / 0 补跑 |
| **真实网页跨能力连续会话（4 轮）** | [run_joint_session.py](./run_joint_session.py) | 6 次 HTTP 200，点击引用全打开 | 0 异常 / 0 回退 / 0 补跑 |

总计 **74 / 74** 次真实网页模型 HTTP 调用返回 `200`，**0 次供应商异常、0 次摘录回退、0 次补跑**。

---

## 三、分项验收实测

### 3.1 G28 原冻结题与公开逗号变体复跑

- **冻结题三组首跑（合计 13/13）**：原五题 **5/5**、前批四题 **4/4**、双文件四题 **4/4**。私有题面、答案与逐题轨迹保留在独审侧本地。
- **P04 三项独立核对**：
  1. **实际出站目标**：出站 `question` 保留原问句并显式绑定唯一目标（`definition_topic=Max_heapify`），`previous_context={}`，**通过**。
  2. **最终定义与直接引用**：最终回答给出所问定义及必要条件，`ev-7` 直接点击打开对应 `v1` 讲义定义页，**通过**。
  3. **收窄后旧实验隔离**：排除旧实验在候选、实读、出站、引用与旧数值复述五项均为 **0**，保留 `experiment_outside_scope` 提示与重选说明；`S03` 切回恢复、`S04` 新会话不继承均通过。
- **公开逗号混合追问与纯测量反例（`g28_comma_first.json`）**：
  - `F02_comma` 与 `F02_conjunction`：均绑定 `Barrier Latch` 写入出站 `question`，`previous_context={}`，旧实验五项零泄漏（`experiment_outside_scope`），首击打开 `v1` 第 2 页。
  - `M01_pure_measure_comma`（“只看我选中的讲义，刚才那次实验的延迟结果还能核对吗？”）与 `F05_measurement_only`：均未误继承 `Barrier Latch`（`definition_topic=None`），打开第 6 页。
  - `E01_explicit_switch` 显式改问 `Cooldown Guard` 优先；`A02_ambig_comma` 多候选零模型调用澄清。

### 3.2 M2 实验条件与数值引用 16 问（双模式独立计分）

使用 [run_m2_16q.py](./run_m2_16q.py) 分别以 `workspace_answer_enabled=False`（生成关闭）与 `True`（真实 Chromium 开启生成）逐项复跑 16 问（含等价归属 7 问 + 原 6 问 + 验证流程反例 1 问 + 引用数值反例 2 问）：

| 模式 | 实验归属（分子/分母） | 数值由可点击引用支持 / 无测量声明 / 错数或漏证 | 回答状态准确性 | 已登记“答案完整但额外提示缺证”发生次数 |
| --- | --- | --- | --- | --- |
| **生成关闭** | **16 / 16** | **4** 条正例点击引用核对 `0.26`/`0.19`；**12** 条标记 `无测量声明`；**0** 错数/无引泄漏 | **16 / 16**（正例 `complete`，缺证/反例 `partial` 或 `no_evidence`） | **0 次** |
| **生成开启（真实网页）** | **16 / 16** | **4** 条正例点击引用直接核对 `0.26`/`0.19`；**12** 条标记 `无测量声明`；**0** 错数/无引泄漏（`ask_test_guess_026` 明确“无法确认”，正文未写入肯定 `0.26`） | **16 / 16**（4 条正例全 `succeeded`，12 条负例/反例全 `partial`） | **0 次**（注：整合交接曾披露 `deepseek-flash` 在 `3_same_cell_fwd.ask_a` 与 `5_original_six.held_out` 偶发因表述未命中模板而输出额外缺证导致假性 `partial`；本轮联合首跑未触发，发生次数为 `0`；即便在真实模型波动下发生，也不影响核心数值 `0.26`/`0.19` 与点击引用，仅会让界面多显示一条保守核验提示） |

### 3.3 贯穿两项能力的真实网页 4 轮连续会话（`run_joint_session.py`）

在同一 Chromium 会话依次执行 4 轮（`Joint-Protocol-Guide.pdf` + `Joint-Latency-Log.ipynb`，6 次 HTTP 200，0 异常）：
1. **`J01` 课程范围首问**：同时引用 `Joint-Protocol-Guide.pdf · v1 · 第 2 页`（`Quorum Gate` 定义）与 `Joint-Latency-Log.ipynb · v1 · Cell 1 · 输出 0`（`14.5 ms`）。
2. **`J02` 收窄至讲义后逗号混合追问**：出站 `question` 保留原问句并锚定 `Quorum Gate`（`definition_topic=Quorum Gate`，`previous_context={}`），旧实验 `Joint-Latency-Log.ipynb` 出站为 `0`，触发 `experiment_outside_scope` 且未复述 `14.5 ms`，点击打开讲义 `v1` 第 2、3 页。
3. **`J03` 收窄后带猜测数值 `14.5 ms` 追问**：`outbound_files` 仅含 `Joint-Protocol-Guide.pdf`（旧实验 0 候选/实读/出站/引用）；`previous_context` 仅含 `J02` 已剔除实验的回答（不含 `14.5 ms`）；最终展示为“不能确认测试集实验的平均延迟是 14.5 ms。讲义只说明平均验证延迟（mean verification latency）的测量数据存放在配套 notebook 中，并未给出具体数值。”——用户猜测未成为肯定结论，点击引用仅打开讲义第 3 页。
4. **`J04` 切回课程全部资料**：恢复检索 `Joint-Latency-Log.ipynb`（2 次 HTTP 200），回答明确输出 `14.5 ms`，直接点击打开 `Joint-Latency-Log.ipynb · v1 · Cell 1 · 输出 0`。

---

## 四、复跑入口

```bash
# 1. 静态检查与 M2 固定模型输出回放 / 离线全量
git -C /Users/Link/code/SRAG worktree add --detach /tmp/srag-joint-check d4c8597d92db20a8803259f47c59faf16ca14f6a
cd /tmp/srag-joint-check
/Users/Link/code/SRAG/.venv/bin/ruff check --no-cache app evals tools tests
PYTHONDONTWRITEBYTECODE=1 /Users/Link/code/SRAG/.venv/bin/python -m pytest -v -p no:cacheprovider tests/test_m2_condition_generation.py

# 2. M2 16 问双模式独立验收
/Users/Link/code/SRAG/.venv/bin/python docs/G/G-M2-joint-gate/run_m2_16q.py \
  --repo /Users/Link/code/SRAG --commit d4c8597d92db20a8803259f47c59faf16ca14f6a \
  --out /tmp/m2_16q_first.json

# 3. 跨能力真实网页 4 轮连续会话
/Users/Link/code/SRAG/.venv/bin/python docs/G/G-M2-joint-gate/run_joint_session.py \
  --subject /tmp/srag-joint-check --fixture-dir /tmp/joint_fix --out /tmp/joint_session.json
```

## 五、交回结论：**建议开放 A03**

在固定联合候选 `d4c8597d92db20a8803259f47c59faf16ca14f6a` 上，G28 原冻结 `5/4/4`（含 `P04` 出站锚定、最终定义及直引、旧实验五项零泄漏）与逗号混合追问/纯测量反例全部通过；M2 16 问在生成关闭与真实网页开启双模式下归属、可点击引用支撑（4 正例直接核对、12 负例/反例 `无测量声明`）与状态准确性均达到 `16/16`；跨能力真实网页连续 4 轮会话无越界、无猜测数值泄漏且切回后可恢复直接实验引用。未触发任何阻断项，建议主审开放 `A03`。本轮未执行 `A03`，停止在 `codex/g-m2-joint-gate`。
