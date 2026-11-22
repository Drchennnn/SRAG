# M2 实验条件错配独立复验报告

- **受测候选提交**：`7e96dfad20c1a5892ba143201ae1c6fa1c699dc4` (`codex/m2-condition-fix`)
- **对照基线提交**：`37a8b811f3a9b5be58aa6ea1c3e57e59df95a6cc` (`G28` 线证提交)
- **公开替身生成提交**：`128bf5b646352f12018873687faaad10b31d1e06` (`codex/m2-retest-prep`)
- **独立审核工作树与分支**：`/Users/Link/code/SRAG-m2-cond-gate` · `review/m2-condition-fix-gate`（产品代码保持 `7e96dfad20c1a5892ba143201ae1c6fa1c699dc4` 未改，未写入原工作区 `/Users/Link/code/SRAG` 或 `main`）
- **复验脚本与原始证据**：[verify.py](./verify.py) · [证据.json](./证据.json)

---

## 1. 两种配置各自的总结裁决

| 配置模式 | 路由与生效机制 | 基线 (`37a8b811`) | 候选 (`7e96dfad`) | 裁决 |
| --- | --- | --- | --- | --- |
| **文字生成关闭** (`workspace_answer_enabled=False`) | `evidence_answer`（`_answer_with_supplement` → `_requested_experiment_in`），0 次外部模型调用 | **3/6**（`salt_fog`、`unfamiliar_condition`、`different_object` 均错将留出集 `0.26 → 0.19` 标为 `complete`） | **6/6**（`held_out` 保留 `0.26 → 0.19` 与双引；5 组缺证全部 `partial`、实验项未支持、零错用数值） | **PASS** |
| **文字生成开启** (`workspace_answer_enabled=True`) | `generated_answer`（[workspace.py#L1007](../../../app/rag/workspace.py#L1007) 因 `not self.settings.workspace_answer_enabled` 绕过 `_answer_with_supplement`，每轮 2 次 `deepseek-flash` HTTP `200`） | **4/6**（`salt_fog`/`unfamiliar_condition`/`different_object`/`different_metric` 指出缺证，但 `held_out` 误报 `partial` 且 `test_split` 未指出缺 `test` 切分） | **5/6**（`test_split` 返工：`generation.status="succeeded"`，直接输出 `held-out` `0.26 → 0.19` 且未指出缺少 `test` 切分；且全 6 组均无结构化实验支持状态） | **返工** |

---

## 2. 实际生效配置与固定公开替身

- **替身原件与固定 SHA-256**（两套配置、基线与候选完全同字节）：
  - `Basalt-Latch-Guide.pdf` (`a0e1eb8b9f45aa615325a4a8d677d4e8e621ceef17a3240203c2f5b00a58a8c9`)
  - `Latch-Check.pptx` (`63789149533e78f5badc2d4645f16e21a50e3b41d8cf62c7c060573429a52676`)
  - `Latch-Trial.ipynb` (`218a7456a079c0e6a8fef79c2a9148ecb601d9441ff4a466277f6f61086d39c2`)
- **Mode A（文字生成关闭）配置**：`workspace_answer_enabled=False`、`workspace_retrieval_strategy="fts"`、`source_vision_enabled=True`、`workspace_input_token_limit=32768`、`deepseek_configured=False`、`jev_configured=False`、`qdrant_configured=False`。
- **Mode B（文字生成开启，真实 Chromium）配置**：`workspace_answer_enabled=True`、`workspace_retrieval_strategy="fts"`、`source_vision_enabled=True`、`workspace_input_token_limit=32768`、`deepseek_chat_model="deepseek-flash"`、`deepseek_query_model="deepseek-flash"`、`jev_configured=False`、`qdrant_configured=False`。

---

## 3. 文字生成关闭（`workspace_answer_enabled=False`）逐组对照

### 3.1 基线（`37a8b811`）首败复现
- **`salt_fog`**（`Basalt latch 的定义是什么？讲义所对应的盐雾实验如何验证？`）：完全复现原首败——`route="evidence_answer"`、`answer_status="complete"`、外部调用 `0`，正文直接输出 `"实验 SlateTrialB6，切分 held-out，样本数 4。MAE（amplitude）基线为 0.26，试验为 0.19。"`，引用 `['ev-1', 'ev-3']`（`Basalt-Latch-Guide.pdf` v1 第 1 页 + `Latch-Trial.ipynb` v1 Cell 1）。
- **`unfamiliar_condition`（低温）** 与 **`different_object`（Quartz latch）**：同样错标 `answer_status="complete"` 并直接输出 `0.26 → 0.19`。
- **`different_metric`（RMSE）** 与 **`test_split`（测试集）**：`answer_status="partial"`，仅引用 `['ev-1']` 并返回旧缺证提示。

### 3.2 候选（`7e96dfad`）逐组复测（6/6 PASS）

| 组别 | 状态 (`answer_status`) | 外部调用 | 引用 | 用户可见正文与逐句核查 |
| --- | --- | ---: | --- | --- |
| `held_out`（留出集正例） | `complete` | 0 | `ev-1` (`Basalt-Latch-Guide.pdf` p.1)、`ev-3` (`Latch-Trial.ipynb` Cell 1) | 准确保留 `Basalt latch` 定义及 `SlateTrialB6` / `held-out` / `4 samples` / `MAE (amplitude)` / `0.26 → 0.19`。 |
| `salt_fog`（盐雾） | `partial` | 0 | `ev-1` (`Basalt-Latch-Guide.pdf` p.1) | 实验项未输出任何数值，正文提示 `"本轮可核对的原文不足以确认符合所问对象、条件和指标的实测比较。"`（未泄露留出集数值）。 |
| `unfamiliar_condition`（低温） | `partial` | 0 | `ev-1` | 同上，未提及或误用 `0.26 → 0.19`。 |
| `different_object`（Quartz latch） | `partial` | 0 | `ev-1` | 同上，未提及或误用 `0.26 → 0.19`。 |
| `different_metric`（RMSE） | `partial` | 0 | `ev-1` | 同上，未提及或误用 `0.26 → 0.19`。 |
| `test_split`（测试集） | `partial` | 0 | `ev-1` | 同上，未提及或误用 `0.26 → 0.19`。 |

注：在关闭生成路径下，5 个缺证组均采用同一通用缺证说明（`"本轮可核对的原文不足以确认符合所问对象、条件和指标的实测比较。"`），未在文案中逐个点名具体所缺的条件/对象/指标词，但内部 `answer_status="partial"`、引用裁剪与不输出错误条件数值均严格一致。

---

## 4. 文字生成开启（`workspace_answer_enabled=True`）真实网页对照

两版均从独立空库启动真实服务与 `HeadlessChrome/157.0.0.0`，通过 `/workspace/` 上传 3 份替身文件、逐题发送并点击全部引用按钮打开 `#reader` 核验。每题均产生 `2` 次 `deepseek-flash` HTTP `200` 调用（1 次查询改写 + 1 次 `mode="definition_and_experiment"` 生成）。

| 组别 | 基线 (`37a8b811`) 首跑 | 候选 (`7e96dfad`) 首跑 | 候选点击引用与事实核查 |
| --- | --- | --- | --- |
| `held_out` | `gen="partial"`（误报未说明如何验证定义）；`ext_calls=2`；点击 `ev-5`/`ev-1` 打开 PDF p.1 与 Notebook Cell 1 | `gen="succeeded"`；`ext_calls=2`；点击 `ev-1`/`ev-2` 打开 PDF v1 p.1 与 Notebook v1 Cell 1 | 正确给出 `0.26 → 0.19` 与双引支持。**通过** |
| `salt_fog` | `gen="partial"`；仅引用 `ev-4` (PDF p.1)，明确指出缺少盐雾实验 | `gen="partial"`；点击 `ev-1`/`ev-2`/`ev-4` 打开 PDF p.1 与 PPTX p.1 | 明确指出仅干台架观测、无盐雾实验，零泄露留出集数值。**通过** |
| `unfamiliar_condition` | `gen="partial"`；附带留出集 `0.26 → 0.19` 作背景并明确未说明低温循环 | `gen="partial"`；点击 `ev-1`/`ev-2` 打开 PDF p.1 与 Notebook Cell 1 | 给出留出集背景并明确指出未提供低温循环条件与验证描述，未冒充低温结果。**通过** |
| `different_object` | `gen="partial"`；明确指出无 Quartz latch 留出集实验 | `gen="partial"`；点击 `ev-1`/`ev-2` 打开 PDF p.1 与 Notebook Cell 1 | 明确指出仅含 Basalt latch 与 `SlateTrialB6`，无 Quartz latch 实验。**通过** |
| `different_metric` | `gen="partial"`；明确指出只报告 MAE、无 RMSE | `gen="partial"`；点击 `ev-1`/`ev-2` 打开 PDF p.1 与 Notebook Cell 1 | 明确指出只给出 MAE (`0.26 → 0.19`)，未报告 RMSE。**通过** |
| `test_split` | `gen="partial"`（误因：称未说明阈值对应关系，未指出缺少 `test` 切分） | **`gen="succeeded"`**（未触发缺证提示）；点击 `ev-1`/`ev-2`/`ev-3` | **缺陷**：直接输出 `held-out` (`0.26 → 0.19`) 回答“测试集实验如何验证”，只字未提缺少 `test` 切分！**不通过** |

**根因定位**：[app/rag/workspace.py#L1007](../../../app/rag/workspace.py#L1007) 限定 `not self.settings.workspace_answer_enabled` 才进入 `_answer_with_supplement`，因此开启生成时 `7e96dfad` 的 `_requested_experiment_in` 条件校验完全未执行，也未向前端/trace 产出结构化的条件支持状态。

---

## 5. 发布 v2 后的焦点重绑（`PUT /focus`）协议核对

仅依据有权读取的公开文档与冻结交接（不开启 A03）：
1. **网页重绑能证明什么**：在 `M2-public-diagnosis` 与 `G19` 公开回放中，Chromium 上传替换 v2 后实际发出 `POST /import(v2) → PUT /focus(v2_root) → POST /turns`，证明**网页前端在同会话上传换版后会自动重绑焦点到 v2 根节点**，下一轮可直接读取 v2 并保留对 v1 历史引用的只读回看。
2. **裸 API 未重绑能证明什么**：在 `M2-public-diagnosis` ASGI 序列中，直接 `POST /import(v2)` 而不调用 `PUT /focus` 会使数据库保留已退役的 v1 焦点，导致直接 `POST /turns` 报 `failed / stale_reference`，证明**后端 API 故意将发布动作与焦点更新分离，并通过护栏阻止陈旧焦点被当作当前证据**。
3. **正式锁定步骤核对结论**：**待核**（与 [G28-双项冻结交接.md](../G28-双项冻结交接.md) 一致；`G01`/`G03` 公开文本仅记载两轮间发布一次，不足以确认私有运行器在第二轮前是否包含 `PUT /focus` 重绑，不开启 A03）。

---

## 6. 复现命令与离线测试

```sh
# 1. 候选定向单元测试与静态检查（/Users/Link/code/SRAG-m2-cond-gate）
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/Link/code/SRAG-m2-cond-gate /Users/Link/code/SRAG/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_evidence_answer.py tests/test_evidence_supplement.py
/Users/Link/code/SRAG/.venv/bin/ruff check --no-cache app/rag/evidence_answer.py app/rag/evidence_supplement.py tests/test_evidence_answer.py

# 2. 双模式（生成关闭 + 生成开启真实网页）独立复跑
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/Link/code/SRAG-m2-cond-gate /Users/Link/code/SRAG/.venv/bin/python docs/G/M2-condition-gate/verify.py \
  --repo /Users/Link/code/SRAG \
  --candidate-dir /Users/Link/code/SRAG-m2-cond-gate \
  --out /Users/Link/code/SRAG-m2-cond-gate/docs/G/M2-condition-gate/证据.json
```

- 候选定向 `pytest`：`88 passed`；`ruff check`：`All checks passed!`。
