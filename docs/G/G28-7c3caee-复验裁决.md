# G28 7c3caee 独立复验裁决

- **受测产品 SHA**：`7c3caee6ef3c99eb489c01c26146ec31b75e8417`（`codex/g28-dev`，产品修复 `0a153569731702450c43d5585f9113527cddf98d`）
- **接续独审 SHA**：`536a397c1d129f532df820c41ecf4cf0086aa1ab`（`codex/g28-gate`）
- **服务进程实载 SHA**：`7c3caee6ef3c99eb489c01c26146ec31b75e8417`（6 组实测全部核对一致）
- **各组首跑分数**：原五题 **5/5**、前批四题 **4/4**、双文件四题 **4/4**（合计 **13/13**）
- **P04 三项裁决**：
  1. **出站 HTTP 请求锚定**：**通过**（保留原问句前缀并显式附加受限目标说明，明确将省略式追问与前轮唯一目标关联，`previous_context={}`）
  2. **最终答案与直接引用**：**通过**（给出所问定义与必要条件，直接引用首击打开正确 `v1` 讲义定义页）
  3. **收窄后范围隔离**：**通过**（旧实验候选/实读/出站/引用/旧数值复述均为 `0`，`experiment_outside_scope` 提示与重选办法完整；`S03` 切回课程恢复实验依据，`S04` 新会话不继承旧话题）
- **总裁决**：**通过**（`F02` 逗号对照单独记录于第四节，不影响原冻结题通过裁决）

---

## 一、冻结题首跑汇总

沿用原独审工作树、独立空 SQLite 库、独立数据目录及原冻结配置（`fts`、开启生成与视觉、输入上限 `32768`、`deepseek-flash`），从真实 Chromium `/workspace/` 首跑。全部 6 组均采用首次运行结果：**供应商异常 0 次、摘录回退（`excerpt_fallback`）0 次、补跑 0 次**。私有材料留在本机独审侧，公开聚合见 [公开聚合 JSON](./G28-7c3caee-公开聚合.json)。

| 组别 | 首跑得分 | 模型调用（HTTP 200） | 引用首击 | 说明 |
| --- | --- | --- | --- | --- |
| 原五题（`P01`–`P05`） | **5/5** | 9 / 9 | 19 / 19 | 全部通过；`P04` 出站请求、最终定义、直接引用与范围隔离均达标。 |
| 前批四题（`N01`–`N04`） | **4/4** | 7 / 7 | 10 / 10 | 全部通过；`N04` 范围隔离与重选说明完整。 |
| 双文件四题（`S01`–`S04`） | **4/4** | 5 / 5 | 8 / 8 | `S02` 隔离、`S03` 切回恢复实验、`S04` 新会话不继承均通过。 |

## 二、P04 三项独立核对

1. **实际发往供应商的 HTTP 请求（通过）**：出站 `question` 以原用户问句为前缀并显式附加唯一受限目标说明（非仅在证据正文中出现），`definition_topic` 为唯一确认目标，`previous_context` 为 `{}`。
2. **最终答案与直接引用（通过）**：最终回答给出所问定义及必要条件；实际点击直接引用，首击打开对应文件 `v1` 的定义所在页。
3. **收窄后范围隔离（通过）**：旧实验在候选、实读、出站、引用与旧数值复述五项均为 `0`，保留 `experiment_outside_scope` 范围变化说明与重选/切回步骤。

## 三、公开入口对照复核（`F01`–`F07`）

共 11 次真实 HTTP 200 调用、20 枚引用全部首击打开（失败 0、回退 0）：
- `F02_elided_followup`（`_definition_reference` 仍为 `None`）：出站 `question` 保留原问句并显式绑定 `Barrier Latch`，`previous_context={}`，触发 `experiment_outside_scope`，首击打开 `v1` 第 2 页。
- `F03_explicit_switch`：显式换题 `Cooldown Guard` 优先，未继承 `Barrier Latch`，首击打开 `v1` 第 4 页。
- `F04_unrelated`：无关流程问题未继承 `Barrier Latch`，首击打开 `v1` 第 5 页。
- `F05_measurement_only`：纯测量追问未继承 `Barrier Latch`，触发 `experiment_outside_scope`，首击打开 `v1` 第 6 页。
- `F07_ambig_followup`：多候选无法区分时零模型调用返回 `clarification`。

## 四、单列新增观察：`F02` 分号改逗号对照

将公开 `F02` 连接两个子问的分号改为逗号（`只看我选中的讲义，把刚才问过的部分重新梳理一遍，配套实验还能核对吗？`），原意不变，独立首跑观察如下（共 3 次 HTTP 200、8 枚首击打开）：
- **实际路由**：`generated_answer`
- **出站结果**：`definition_topic` 为 `null`，出站 `question` 原样发出（未绑定 `Barrier Latch`），`previous_context={}`，`notices` 为 `['missing_evidence', 'experiment_outside_scope']`。
- **原因与边界记录**：`app/rag/workspace.py` 中 `_is_measurement_only_followup()` 仅按 `[。！？!?；;\n]` 分句，未按逗号分句，因此逗号连接的“梳理刚才问过的部分，配套实验还能核对吗”被当作单句并命中 `_EXPERIMENT_TEXT`，跳过话题绑定。该观察单列供后续稳健性参考，不推翻本轮冻结题 **13/13** 与 `P04` 三项通过裁决。

## 五、复跑入口与交回

```sh
STATE_DIR=/Users/Link/.local/state/srag-g28-independent.ESYpChUb
"$PYTHON" docs/G/G28-p04-diagnostic/observe_boundaries.py \
  "$STATE_DIR/p04-public-diagnostic/observer.py" \
  --runner docs/G/G28-browser-replay.py \
  --subject "$STATE_DIR/candidate-7c3caee" \
  --manifest "$STATE_DIR/freeze.json" \
  --env-file /Users/Link/code/SRAG/.env \
  --out "$STATE_DIR/7c3caee-runs/regression-first.json"
```

本轮停在 `codex/g28-gate`，不修改产品代码、不合入 `main`；私有题面、答案与轨迹保留在本机独审侧，`A03` 继续封存。
