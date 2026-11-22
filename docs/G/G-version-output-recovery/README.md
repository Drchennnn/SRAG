# G 版本发布后跨版对象恢复与多输出 Cell 澄清修复交接

- **分支**：`codex/g-version-output-recovery`
- **基准基线 SHA**：`d4c8597d92db20a8803259f47c59faf16ca14f6a`（Task 1 基线冻结 `acedf8fb87e2d4fb3ff4e83624c81d027919f822`）
- **产品修复 SHA**：`0f5c93ca7aabc5631c728f49fccee986f70c878f`（包含 `f6d2c238` 主修复及独立盲验回归修复 `0f5c93ca`）
- **公开材料边界**：仅使用公开 `.ipynb` 夹具及新构建的 `PIL` 双版 Notebook 夹具（`Bench-Drift-*.ipynb` 与独立审核员自建声学水听器双版夹具）；A03 私有材料全程保持封存，未打开或读取任何 A03 私有原件；`main` 未改动。

---

## 1. 修复范围

1. **问题一：发布 v2（不补 `PUT /focus`）后继承上轮单次实读对象并区分三种意图**
   - 当上一轮展示了多个对象（`read_status="not_attempted"`）但仅实读其中一个（`method="vision"`、`status` 为 `succeeded`/`partial`），随后通过正式协议 `/import(document_id=...)` 发布 v2 且**未调用 `PUT /focus`** 时：
     - **读取新版对应对象**（[workspace.py](file:///Users/Link/code/SRAG-version-output/app/rag/workspace.py) `_resolve_reference` / `resolve_current_counterpart`）：在同文件、可靠输出顺序未破坏时自动映射并实读 v2 对应输出。
     - **回看刚才读过的旧版对象**（`_follow_display` + `_historical_verified_visual_read`）：保留 `historical_only` 语义，同时附带该历史版本经过 `real_provider` 验证的原读内容与 v1 来源，不触碰新版。
     - **比较两版**（[workspace_versions.py](file:///Users/Link/code/SRAG-version-output/app/rag/workspace_versions.py) `compare_versions` + `_old_candidates`）：恢复陈旧焦点下的单文档历史追踪，优先命中单次实际视觉读取而非未读多图展示，同时读取 v1 与 v2 双边依据对比。
   - **安全护栏与对照保留**：
     - 输出重排（插入/删除/类型序列变化或相同图片 SHA 换位）或对应不唯一时，明确进入 `clarification` 给出全部候选选项（修复前曾因继承 `ordinal=1` 导致候选项被误过滤为单个）；对应对象已删时返回带说明的澄清。
     - `"只看旧版"` 仍返回 `historical_only`；`"普通新问题遇到陈旧焦点"`（如 `"看当前 Cell"`）仍返回 `stale_reference`。
     - `_IMAGE_REFERENCE` 现已兼容 `"刚才读过的旧版图"` / `"最新版里刚才读过的图"` 等带中文定语修饰的指代表达。

2. **问题二：聚焦多输出 Cell 未指明具体输出时给出可操作澄清并实读所选输出**
   - 聚焦含多个保存输出的 `cell` 且未指明具体输出序号时（无论进入 `_answer_object` 或 `_retrieve`），不再返回死路 `object_ambiguous`（`options=[]`）或直接贴出未读多图占位 `excerpt_fallback`，而是调用 `_clarify_focused_cell_outputs` 返回 `clarification` 与可点击的 `ClarificationAnchor` 选项（区分图像输出与文本输出）。
   - 用户选择并提交后，图像输出进入 `answer_current_image` 实读，文本输出（`forced.source.kind == "notebook_output"`）跳过整 Cell 检索歧义直读选中输出。

---

## 2. 修复前后对比（G01 真实 `qwen3-vl-plus` 配置，不补 `PUT /focus`）

配置：`workspace_retrieval_strategy="fts"`、`workspace_input_token_limit=32768`、`workspace_answer_enabled=False`、`source_vision_enabled=True`、`deepseek_api_key=""`，真实 DashScope `qwen3-vl-plus`。

| 用例 / 场景 | 修复前基线 (`baseline_traces.json` @ `d4c8597d`) | 修复后结果 (`post_fix_traces.json` @ `0f5c93ca`) |
|---|---|---|
| 问题二：多输出 Cell 问图 (`img_question`) | `completed` · `object_ambiguous`（`opts=0`, `reads=[]`） | `clarification`（2 个可选图像输出）→ 选 `输出 1` 后 `vision` 实读 `cell:1:output:1` (`batch_size (item)`) |
| 问题二：多输出 Cell 问输出 (`out_question`) | `completed` · `excerpt_fallback`（2 张未读占位图） | `clarification`（3 个可选输出：2 图 + 1 stdout）→ 选 `输出 2` 后 `native_text` 实读 `v1 summary` |
| 问题一：发布 v2 后读取新版对应图 | `clarification`（错误绑定到 Turn 1 的 2 张展示图） | `completed` · `vision` 实读 `v2 cell:1:output:0` (`240 ops/s`) |
| 问题一：发布 v2 后回看旧版已读图 | `clarification`（错误绑定到 2 张展示图） | `completed` · `historical_only` + 已验证 v1 读值 (`120 ops/s`)，不读 v2 |
| 问题一：发布 v2 后比较两版 | `completed` · `comparison_scope` 失败 | `completed` · `version_comparison`，同时引用 `v1` (`120`) 与 `v2` (`240`) |
| 对照：只看旧版 | `completed` · `historical_only` | `completed` · `historical_only`（v1 `cell:1:output:0`） |
| 对照：普通新问题遇陈旧焦点 | `failed` · `stale_reference` | `failed` · `stale_reference` |
| 重排 v2 / 已删 v2 / 零实读多图对比 | — | 重排 → `clarification`（2 选项，选后实读 `v2 output:1`）；已删 → `clarification` 提示已删除 |

---

## 3. 真实 Chromium (`/workspace/`) 与网页自然换版兼容回归 (`browser_verify.json` @ `0f5c93ca`)

使用 `agent-browser`（HeadlessChrome）访问本地 `uvicorn /workspace/`（共发起 `3` 次真实 `DashScope qwen3-vl-plus` 调用，均 HTTP 200）：
1. **多输出 Cell 澄清闭环（无脚本代补 `PUT /focus`）**：聚焦 `Browser-P2.ipynb` Cell 2 提问 → `.clarification-actions` 出现 2 个按钮 → 点击第 2 项并发送 → 实读 `cell:1:output:1`（`batch_size (item)`），点击依据按钮在原件阅读器打开 `v1 · 输出 1`（`图像已读取`）。
2. **发布 v2 无 `PUT /focus` 跨版读取**：Turn 1 展示两张图（`0` 次视觉调用）→ Turn 2 实读 `output:0`（`120 ops/s`）→ 后端 `/import(document_id)` 发布 v2（未调用 `PUT /focus`）→ 直接提问新版峰值 → 实读 `v2 cell:1:output:0`（`240 ops/s`），点击依据按钮打开 `v2 · 输出 0`（`图像已读取`）。
3. **兼容回归：网页端自然替换文档后自动重绑**：新会话选中文档并通过界面勾选「替换已有文件」导入 v2 后，前端自然刷新并自动重绑 `focus` 到 `v2`，后续提问直接读取并引用 `v2`（点击依据打开 `当前版本` v2）。

---

## 4. 独立盲验复核汇总

由未参与开发的独立审核子代理在 `0f5c93ca` 上使用全新的深海声学水听器标定双版 Notebook 夹具（`hydro_telemetry_lab`、`hydro_reorder`、`hydro_multi`）、新建 `/tmp` 空库、G01 真实配置（`12` 次真实 `qwen3-vl-plus` 调用，`8,160` input tokens，`775` output tokens）复跑：
- **裁决**：**通过（11 / 11 核心门禁场景与两项前次中级缺陷回归探针全部 100% PASS，无阻塞项）**。
- 详见 [独立审核报告.md](file:///Users/Link/code/SRAG-version-output/docs/G/G-version-output-recovery/独立审核报告.md) 与 [independent_review_traces.json](file:///Users/Link/code/SRAG-version-output/docs/G/G-version-output-recovery/independent_review_traces.json)。
- **保留非阻塞低优先级备注**：重排比较澄清选定某输出后当前仅路由 `answer_current_image` 实读新版单边（模型在对照问句下返回 `status="partial"`）；此外 `answer_current_image` / `compare_versions` 图像通过 `/media/{display_id}` 返回 HTTP 200，而 `/evidence/{eid}/preview` 返回 404（仅 `_follow_display` 路径提供 `/preview` HTTP 200）。

---

## 5. 离线测试与可复现命令

- 代码静态检查与全量离线单测（在 `0f5c93ca` 修改后已全部执行通过：`All checks passed!`，`643 passed, 3 skipped`）：

```bash
cd /Users/Link/code/SRAG-version-output
/Users/Link/code/SRAG/.venv/bin/ruff check --no-cache app evals tools tests docs/G/G-version-output-recovery
/Users/Link/code/SRAG/.venv/bin/python -m pytest -q -p no:cacheprovider tests/
```

- 真实 G01 配置修复前后与真实 Chromium `/workspace/` 复跑：

```bash
cd /Users/Link/code/SRAG-version-output
/Users/Link/code/SRAG/.venv/bin/python docs/G/G-version-output-recovery/repro_baseline.py --out docs/G/G-version-output-recovery/baseline_traces.json
/Users/Link/code/SRAG/.venv/bin/python docs/G/G-version-output-recovery/verify_post_fix.py
```
