# H · 单课程可溯源知识图谱盲测套件与冻结判据（开发前独立冻结）

- **冻结角色**：`h_gate`（独立门禁审核者，零修改 `app/` 与 `app/web/` 生产代码，零触碰 A03 私有封存）
- **基线修订**：`c04149e`（分支 `review/h-gate`）
- **冻结时间**：`2026-11-24`
- **主测课程 ID**：`cs-ctrl-h`（*CS-Ctrl-H: 离散状态反馈控制、状态观测与数字滤波*）
- **隔离对照课程 ID**：`cs-other-h`（仅含 `OtherCourse-Distractor.ipynb`）
- **生成与校验脚本**：[make_h_corpus.py](./make_h_corpus.py)
- **结构化判据 JSON**：[H-blind-criteria.json](./H-blind-criteria.json)

> **保密提示**：在 `H01` 设计审核期间，严禁向执行侧泄露本套件中任何具体文件名、概念名、自定义实验函数名或具体考题题面。

---

## 1. 冻结语料清单（6 份原件，含换版件与跨课隔离件）

| 文件名 | 所属课程 | 格式 | 字节数 | SHA-256（前 16 位） | 页/Cell 与核心角色 |
| --- | --- | --- | ---: | --- | --- |
| `Lecture-01-Core-Theory.pdf` | `cs-ctrl-h` | pdf | 499,495 | `08ccb9f976e97377` | 5 页；定义 C1–C3 及条件/实验函数锚点；p5 为通用词汇反例区 |
| `Slides-02-Method-Walkthrough.pptx` | `cs-ctrl-h` | pptx | 37,750 | `fed40bcf6c0c504c` | 7 页（经 `soffice 26.8.3.1` 实测渲染）；定义 C4–C5 及带单位表；p6 为参考文献仅提及陷阱，p7 为通用词汇反例 |
| `Lab-01-Experiments.ipynb` | `cs-ctrl-h` | ipynb | 145,654 | `a5e8e60225942008` | 9 Cells（v1）；演示 C1–C4（零概念词重合）+ `cell:7` 否定注释陷阱 + `cell:8` 未运行计划 Cell |
| `Lab-01-Experiments-v2.ipynb` | `cs-ctrl-h` | ipynb | 110,211 | `0163d3541d72760d` | 7 Cells（v2 同 `document_id` 换版）；修改 C1 数值、移除 C2、C3/C4 Cell 索引前移 |
| `NoEdge-Notes.pdf` | `cs-ctrl-h` | pdf | 357,287 | `3342045579f4478d` | 2 页；定义 C6（全课零实验演示），用于验证无可靠边显式提示与回退 `S0` |
| `OtherCourse-Distractor.ipynb` | `cs-other-h` | ipynb | 37,558 | `e7c335f8f173f2e7` | 4 Cells；同名概念但不同数值及 `cs-other-h` 独有实验，验证单课程隔离与 HTTP 404 |

---

## 2. 真边清单（Expected Accepted Edges）与换版演进

### 2.1 v1 快照真边（共 10 条：`6 DEFINED_IN` + `4 DEMONSTRATED_BY`）

| 边 ID | 关系 | 概念 | 目标原件与源锚点 | 依据要点 |
| --- | --- | --- | --- | --- |
| `E-DEF-01` | `DEFINED_IN` | `极点配置滞后补偿` (C1) | `Lecture-01-Core-Theory.pdf` `page:1` | 正式定义 + 3 项数学成立条件 + 锚定 `run_loop_trim_p1()` |
| `E-DEF-02` | `DEFINED_IN` | `扩展卡尔曼协方差膨胀` (C2) | `Lecture-01-Core-Theory.pdf` `page:3` | 正式定义 + $R_k>0$ 与连续 3 步 $\chi^2$ 条件 + 锚定 `eval_res_scale_k7()` |
| `E-DEF-03` | `DEFINED_IN` | `预测时域滑模抗扰` (C3) | `Lecture-01-Core-Theory.pdf` `page:4` | 正式定义 + 边界层 $\phi \ge 2\epsilon T_s$ + 锚定 `probe_band_law_m4()` |
| `E-DEF-04` | `DEFINED_IN` | `双二阶自适应陷波滤波` (C4) | `Slides-02-Method-Walkthrough.pptx` `page:2` | 正式定义 + $r$ 公式（p3 带单位表）+ 锚定 `verify_iir_dip_b2()` |
| `E-DEF-05` | `DEFINED_IN` | `条件积分抗饱和钳位` (C5) | `Slides-02-Method-Walkthrough.pptx` `page:4` | 正式定义 + p5 带单位对比表（无 Notebook 实验） |
| `E-DEF-06` | `DEFINED_IN` | `H∞ 混合灵敏度环路成形` (C6) | `NoEdge-Notes.pdf` `page:1` | 正式定义 + 权函数约束（明确全课无实验） |
| `E-DEMO-01` | `DEMONSTRATED_BY` | C1 | `Lab-01` v1 `cell:1` (`output:0`, `output:1`) | `run_loop_trim_p1()`: `ts=0.020 s`, `zeta_ratio=0.72`, `overshoot=4.15%`, `settling=185.0` |
| `E-DEMO-02` | `DEMONSTRATED_BY` | C2 | `Lab-01` v1 `cell:3` (`output:0`, `output:1`) | `eval_res_scale_k7()`: `rmse_unscaled=0.418`, `rmse_scaled=0.064`, `reject=2.30%` |
| `E-DEMO-03` | `DEMONSTRATED_BY` | C3 | `Lab-01` v1 `cell:5` (`output:0`, `output:1`) | `probe_band_law_m4()`: `osc_amp=0.018 Nm`, `steady_band=0.024 rad`, `14.2 Hz` |
| `E-DEMO-04` | `DEMONSTRATED_BY` | C4 | `Lab-01` v1 `cell:6` (`output:0`, `output:1`) | `verify_iir_dip_b2()`: `q_factor=8.00`, `dip=-31.4 dB`, `ripple=0.18 dB` |

### 2.2 v2 换版后真边（共 9 条：`6 DEFINED_IN` + `3 DEMONSTRATED_BY`）

- `E-DEMO-01-V2`：C1 → `Lab-01` v2 `cell:1`（`ts=0.010 s`, `zeta_ratio=0.84`, `overshoot=1.42%`, `settling=132.0`）。
- C2 的 `DEMONSTRATED_BY` 边：**已随 `eval_res_scale_k7()` 移除而失效（0 条当前演示边）**。
- `E-DEMO-03-V2`：C3 → `Lab-01` v2 `cell:3`（由 v1 `cell:5` 前移，须跟随新位置）。
- `E-DEMO-04-V2`：C4 → `Lab-01` v2 `cell:4`（由 v1 `cell:6` 前移，须跟随新位置）。

---

## 3. 必拒反例清单（10 项，要求 0 误接受）

| 反例 ID | 陷阱类型 | 涉及概念/术语 | 禁止出现的目标/关系 |
| --- | --- | --- | --- |
| `REJ-01` | 文献仅提及 | `扩展卡尔曼协方差膨胀` | `Slides-02` `page:6` (`DEFINED_IN` / `DEMONSTRATED_BY`) |
| `REJ-02` | 文献仅提及 | `极点配置滞后补偿` | `Slides-02` `page:6` (`DEFINED_IN` / `DEMONSTRATED_BY`) |
| `REJ-03` | 文献仅提及 | `H∞ 混合灵敏度环路成形` | `Slides-02` `page:6` (`DEFINED_IN` / `DEMONSTRATED_BY`) |
| `REJ-04` | 未定义背景词 | `Smith 预估器` / `零相位前馈预整形` | `Slides-02` `page:6` (`DEFINED_IN` / `DEMONSTRATED_BY`) |
| `REJ-05` | 代码否定注释提及 | `扩展卡尔曼协方差膨胀` | `Lab-01` `cell:7` `measure_fft_wallclock_overhead` (`DEMONSTRATED_BY`) |
| `REJ-06` | 代码否定注释提及 | `双二阶自适应陷波滤波` | `Lab-01` `cell:7` `measure_fft_wallclock_overhead` (`DEMONSTRATED_BY`) |
| `REJ-07` | 未运行未来计划 Cell | `极点配置滞后补偿` | `Lab-01` `cell:8` `planned_lag_comp_ablation` (`DEMONSTRATED_BY`) |
| `REJ-08` | 通用词汇共现 | `采样周期 / 单位圆 / 稳态误差 / 协方差` | `Lecture-01` `page:5` & `Slides-02` `page:7` |
| `REJ-09` | 跨课程同名干扰 | C1 / C2 / C6 | `OtherCourse-Distractor.ipynb` (`cs-other-h`) 不得进入 `cs-ctrl-h` |
| `REJ-10` | 换版/删档失效边 | C2 (`v1` 旧边) / 已删文件定义边 | `v1 -> v2` 后新查询严禁沿用 `Lab-01 v1 cell:3`；删档后对应边立即退出 |

---

## 4. 完整 H03 门禁测试矩阵与 10 题同预算三路对照集

### 4.1 必测门禁项（`M1`–`M6`）

1. **`M1_DUAL_ENTRY`（双入口实走）**：针对 `H_Q01`，分别从**课程目录概念入口**与**网页回答内概念入口**进入 `极点配置滞后补偿`，逐一核对 `DEFINED_IN` 与 `DEMONSTRATED_BY` 的原文依据、重读原页/Cell，并点击打开 `Lecture-01 page:1` 与 `Lab-01 cell:1` 代码、stdout（`0.72 / 4.15% / 185.0`）及 PNG 图。
2. **`M2_COUNTEREXAMPLE_REJECTION`（反例拒收率）**：10 项反例 `REJ-01`–`REJ-10` 全部拒收（0 误接受）。
3. **`M3_NO_EDGE_FALLBACK`（无可靠边显式提示与回退）**：在 `cs-ctrl-h` 查询 `H∞ 混合灵敏度环路成形`（C6），导航必须显式提示无可靠实验边并平滑回退常规检索命中 `NoEdge-Notes.pdf page:1–2`，0 跨课泄漏。
4. **`M4_VERSION_REPLACE_AND_DELETE`（换版与删档失效）**：
   - `Lab-01 v1 → v2`：新查询立即切换为 `zeta_ratio=0.84, overshoot=1.42%, settling=132.0`（C1）、C2 旧边退出（0 旧数值泄漏）、C3/C4 自动指向 `cell:3` / `cell:4`；而 v1 历史回答里的旧引用仍可打开 v1 原件。
   - 删除 `NoEdge-Notes.pdf`（或 `Slides-02`）后，新图谱查询与问答 0 旧边残留。
5. **`M5_CROSS_COURSE_404`（单课程隔离与越权 404）**：在 `cs-ctrl-h` 会话下访问 `cs-other-h` 的概念、图谱、目录或证据/媒体端点（及反向），严格返回 HTTP 404。
6. **`M6_SAME_BUDGET_THREE_WAY_EVAL`（同预算三路问答对照）**：固定 `cs-ctrl-h` 同快照、同一 `deepseek-flash` 模型与同一最终证据预算，跑完 `H_Q01`–`H_Q08`（plus `H_Q09` v2 核验与 `H_Q10` 隔离核验），对比：
   - **Arm 1**：`常规检索 (S0)`
   - **Arm 2**：`S0 + 结构邻接`
   - **Arm 3**：`S0 + 图谱一跳候选`
   分别报告**导航独立价值**（概念→讲义→实验代码/输出完整率与依据准确率）与**问答候选净收益**（跨文件零词面重合题召回/整题得分增量、控制组无退化、额外耗时与 0 严重错误）。

### 4.2 10 道冻结考题分布

| 题号 | 快照 | 题型类别 | 预期来源（`expected_sources`） | 核心判据要点 |
| --- | --- | --- | --- | --- |
| `H_Q01` | v1 | 跨文件零词面重合 + 双入口核心题 | `Lecture-01 page:1`, `Lab-01 cell:1` + `output:0` | 3 项条件 + `run_loop_trim_p1` (`0.020 s`, `0.72`, `4.15%`, `185.0`) |
| `H_Q02` | v1 | 跨文件零词面重合（v2 将移除） | `Lecture-01 page:3`, `Lab-01 cell:3` + `output:0` | `R_k>0`, `chi2=7.81` 连续 3 步 + `0.418 rad -> 0.064 rad`, `2.30%` |
| `H_Q03` | v1 | 跨文件零词面重合（v2 Cell 前移） | `Lecture-01 page:4`, `Lab-01 cell:5` + `output:0` | `phi >= 2*eps*T_s` + `0.018 Nm`, `0.024 rad`, `14.2 Hz` |
| `H_Q04` | v1 | PPTX 定义 + 单位表 + 跨文件零重合 | `Slides-02 page:2,3`, `Lab-01 cell:6` + `output:0` | $r$ 公式 + `0.9529`, `Q=8.00`, `-31.4 dB`, `0.18 dB` |
| `H_Q05` | v1 | 控制组：PPTX 定义 + 带单位对比表 | `Slides-02 page:4,5` | `4.8 % / 190.0 ms` vs `9.4 % / 265.0 ms` |
| `H_Q06` | v1 | 反例：代码否定注释仅提及陷阱 | `Lab-01 cell:7` + `output:0` | 明确否定其为 EKF 实验，仅 `numpy.fft` 耗时 `68.4 us` |
| `H_Q07` | v1 | 反例：未运行计划 Cell + 背景引用 | `Slides-02 page:6`, `Lab-01 cell:8` | 明确未运行/无输出，Smith 仅背景提及 |
| `H_Q08` | v1 | 无可靠边概念：显式提示 + 回退 `S0` | `NoEdge-Notes.pdf page:1,2` | `||M||_inf < gamma`, `-26 dB`, `20 dB/dec`，无实验；0 泄漏 `cs-other-h` |
| `H_Q09` | v2 | 换版 `v1 → v2` 新值准确 | `Lecture-01 page:1`, `Lab-01 v2 cell:1` + `output:0` | `0.010 s`, `1.95`, `0.84`, `1.42%`, `132.0`；0 泄漏 v1 (`0.72/4.15%`) |
| `H_Q10` | v1 | 单课程隔离负例 | `[]`（`cs-ctrl-h` 范围无） | 严禁泄漏 `cs-other-h` 的 `run_hinf_other_x9` / `0.51` / `16.80%` |
