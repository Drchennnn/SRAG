# SRAG G/M2 联合验收候选交接报告

- **当前状态划分**：
  - **联合候选已生成**（`codex/g-m2-joint-candidate`）
  - **独审待完成**（待独立审核者对联合候选开展独立门审）
  - **A03 未启动**（M2 保持返工，不合入 `main`）
- **基线与整合来源（已验证均为主祖先）**：
  - `origin/main`：`54fbd9e9daf97d6aa9f28d6ef0bdc0431f41c8fd`
  - `origin/codex/g28-dev`：`18fb8929e6aaaa41de40dd34d933d34741c661e6`（产品修复提交 `d63447a99178bfc5c55a8edadbc62a9604951420`）
  - `origin/feat/m2-condition-gen-fix`：`e55a9b4a9281d127c78f6fe28f7511c064312884`（产品修复提交 `8d530e4887330f422364375f897b29da5e97a224`）
  - **联合受测产品树提交**：`90cd99b8ec925c58cb7f291fae646d1eec69b94f`（仅合入上述两路交付，后续文档/治理提交未改动 `app/` 与 `tests/`）

---

## 1. 联合树离线回归与静态检查

在 `/Users/Link/code/SRAG-g-m2-joint` 实测（`git merge` 无冲突完成）：

- **Ruff 静态检查**（`../SRAG/.venv/bin/ruff check --no-cache app evals tools tests`）：**All checks passed!**
- **全量离线单测**（`PYTHONPATH=. ../SRAG/.venv/bin/pytest -q -rs -p no:cacheprovider`）：**640 passed, 3 skipped**（1 warning：`StarletteDeprecationWarning`）。
- **跳过项与原因（如实保留，本机未安装 LibreOffice）**：
  - `tests/test_source_evidence.py:747`: `SRAG_SOFFICE not configured`
  - `tests/test_workspace.py:1019`: `SRAG_SOFFICE not configured`
  - `tests/test_workspace_preview.py:118`: `Set SRAG_SOFFICE to exercise genuine PPTX page rendering`

---

## 2. 真实网页公开冒烟（[smoke.py](./smoke.py) · [smoke.json](./smoke.json)）

配置：独立空库与独立端口、`workspace_answer_enabled=True`、`workspace_retrieval_strategy="fts"`、`source_vision_enabled=True`、`workspace_input_token_limit=32768`，模型 `deepseek-flash`，实测进程 `git_sha=90cd99b8ec925c58cb7f291fae646d1eec69b94f`。导入公开 `Beacon-Sync-Manual.pdf`（v1）与 `Beacon-Sync-Log.ipynb`（v1）：

1. **第 1 轮（课程范围）**：问 `"讲义怎么定义 Barrier Latch？配套实验测得的 mean sync delay 是多少毫秒？"`
   - 结果：`turn_status="completed"`、`generation.status="succeeded"`（2 次 DeepSeek HTTP 200）。
   - 回答给出 Barrier Latch 定义及 `Mean sync delay 为 12.0 ms`（`3 个 admitted slots`），点击引用打开 `Beacon-Sync-Manual.pdf · v1 · 第 2 页` 与 `Beacon-Sync-Log.ipynb · v1 · Cell 1 · 输出 0`。
2. **收窄范围并逗号追问**：目录勾选收窄至 `Beacon-Sync-Manual.pdf`，发送逗号连接自然追问 `"只看我选中的讲义，把刚才问过的部分重新梳理一遍，配套实验还能核对吗？"`
   - **话题继承**：`trace.definition_topic == "Barrier Latch"`，成功继承先前的 Barrier Latch 话题。
   - **定义引用**：点击 `ev-2` 打开 `Beacon-Sync-Manual.pdf · v1 · 第 2 页`（`open=true`，原文含 Barrier Latch 定义），同时 `ev-4`/`ev-5`/`ev-6` 分别打开第 4、5、6 页。
   - **范围提示**：触发 `experiment_outside_scope` 提示（`"本轮选定范围已改变，上一轮引用的实验依据已超出当前所选文件..."`）、`experiment_missing=true`、`generation.status="partial"`。
   - **旧实验测量值隔离**：第 2 轮正文、`missing_evidence` 与 `visible_text` 均**零泄漏**第 1 轮实验测量数值（无 `12.0`、`11.0`、`13.0` 或 `3 个 admitted slots`）。
   - 全程 3/3 次 DeepSeek HTTP 200，0 供应商异常。

---

## 3. 已知限制与两处假性 `partial` 披露

- 在 `M2` 路真实网页回归（`8d530e4887330f422364375f897b29da5e97a224` / [M2-condition-citation-verify](../M2-condition-citation-verify/README.md)）中，已知存在两处由 `deepseek-flash` 表述波动引起的**假性 `partial`**：
  1. `3_same_cell_fwd.ask_a`：模型在 `missing` 中写出“未提供 TrialA 中 basalt latch 的具体验证步骤或判定流程，仅有实验编号、划分、指标、基线、结果和样本数”，未命中后处理固定短语而降为 `partial`；
  2. `5_original_six.held_out`：模型在 `missing` 中写出“证据未说明该留出集实验如何直接验证 basalt latch 的定义…现有证据仅给出 MAE 结果”，同样因措辞变体未命中后处理模板而降为 `partial`。
  - 两处样例的正文核心数值（`0.26`/`0.19`）与可点击原件引用均准确，固定输出回放均为 `succeeded`；暂按已知模型波动披露，交独立门审评估。

---

## 4. 复跑命令

```bash
cd /Users/Link/code/SRAG-g-m2-joint

# 1. 验证两个整合提交与 main 均是当前候选祖先
git merge-base --is-ancestor 54fbd9e9daf97d6aa9f28d6ef0bdc0431f41c8fd HEAD
git merge-base --is-ancestor 18fb8929e6aaaa41de40dd34d933d34741c661e6 HEAD
git merge-base --is-ancestor e55a9b4a9281d127c78f6fe28f7511c064312884 HEAD

# 2. 静态检查与全量离线测试
../SRAG/.venv/bin/ruff check --no-cache app evals tools tests
PYTHONPATH=. ../SRAG/.venv/bin/pytest -q -rs -p no:cacheprovider

# 3. 真实网页入口公开冒烟（课程首问 -> 收窄到讲义 -> 逗号追问）
../SRAG/.venv/bin/python docs/G/G-M2-joint-smoke/smoke.py --out docs/G/G-M2-joint-smoke/smoke.json
```
