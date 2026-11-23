# G29 · 公共编排收口与契约统一 · 开发前独立冻结盲测套件与判据（Phase 1）

**状态：已由独立门禁审核者 `g29_gate` 在任何 G29 生产代码修改前，于隔离工作区 `/Users/Link/code/SRAG-g29-gate`（分支 `review/g29-gate`，起点 `b1b8c60`）完成未见真实复杂排版三格式原件冻结、CC BY 4.0 许可核验、原件真值盘点与全部门禁判据冻结。**

- **严格独立性声明**：
  - 未修改任何 `app/` 生产代码，未向 `/Users/Link/code/SRAG-g29` 或 `/Users/Link/code/SRAG` 写入任何文件。
  - 未读取或触碰任何 A03 私有封存文件（`docs/A/A03-question-lock.json`、`docs/A/A03-answer-key.json`、`data/g-private/`）。
  - 经全仓 `rg` 零命中核查，本套件所有 2 份真实论文、1 份全新课程 PPTX、1 套双版/重排 Notebook 均完全未在 SRAG 任何既有开发、测试或评测中出现（严格排除 Mixtral、DPO、PagedAttention、Mamba、QLoRA、ReAct、GaLore、FOCCUS、SensorLab 等全部既有材料）。
- **结构化判据 JSON**：[G29-blind-criteria.json](./G29-blind-criteria.json)
- **PPTX / Notebook 确定性生成脚本**：[make_blind_fixtures.py](./make_blind_fixtures.py)

---

## 1. 冻结原件清单与哈希校验

| 标识 | 文件路径 | 类型 | 字节数 | SHA-256 | 页/幻灯片/Cell 规模 | 许可与来源 |
| --- | --- | --- | ---: | --- | --- | --- |
| `pdf_1` | `fixtures/Speculative-Decoding-2211.17192v2.pdf` | 真实论文 PDF（ICML 双栏） | `574,982` | `8967e2daab74178b8a73a6f0b0780c5cbb40f9f6dcbd5c456b24a89c128c52ed` | `13` 页（原生 `48,238` 字符） | **CC BY 4.0** · arXiv `2211.17192v2` |
| `pdf_2` | `fixtures/YaRN-2309.00071v3.pdf` | 真实论文 PDF（ICLR 单栏高密数学） | `838,521` | `e7c0268a796138460c6ba2f67a7cba5bd922c401aea94c1fcd09c4b76b883c85` | `20` 页（原生 `53,951` 字符） | **CC BY 4.0** · arXiv `2309.00071v3` |
| `pptx_1` | `fixtures/Week06-Thermal-Cache-Lab.pptx` | 全新课程 PPTX（ECE-418 第 6 周） | `135,453` | `4c4fc25a94b7e845219780723eb69de9c3259f767b4bec36517d6ca23a582688` | `4` 页（封面/原生表/实验图/图文机制） | 自制公开夹具（`make_blind_fixtures.py`） |
| `nb_v1` | `fixtures/notebook/v1/Lab06-Thermal-Sweep.ipynb` | Notebook `v1`（Rev-A） | `127,612` | `074d0993f0ff339242ff1379d550b5c5943f18057d026a7ac81e56de52dccdd1` | `3` Cell（`code-sweep` 含 `1` stdout + `2` PNG） | 自制公开夹具（`make_blind_fixtures.py`） |
| `nb_v2` | `fixtures/notebook/v2/Lab06-Thermal-Sweep.ipynb` | Notebook `v2`（Rev-B，同序换单位/值） | `127,938` | `a2628fcb95c52faec50ae6aa7f89837fa66faa3daaac53f437dc715c17d0bf3f` | `3` Cell（`code-sweep` 含 `1` stdout + `2` PNG） | 自制公开夹具（`make_blind_fixtures.py`） |
| `nb_v2r` | `fixtures/notebook/v2_reordered/Lab06-Thermal-Sweep.ipynb` | Notebook `v2_reordered`（插输出 + 换图序） | `128,128` | `c57d220072f7ae3f40a4190b081283aa2a82383571414ea5584adb8e1311dbfa` | `3` Cell（`code-sweep` 含 `2` stdout + `2` 换序 PNG） | 自制公开夹具（`make_blind_fixtures.py`） |

### 内嵌图像字节哈希（用于预览对称性与视觉输入字节核对）

| 图像标识 | 所属原件与对象 | SHA-256 |
| --- | --- | --- |
| `slide3_chart` | `Week06-Thermal-Cache-Lab.pptx` `page:3:shape:3` | `fa641df6dfbc82fcdff95d3a3b6e2db1cd78067f6ff4d5c5ee57a615a54e28df` |
| `slide4_diagram` | `Week06-Thermal-Cache-Lab.pptx` `page:4:shape:4` | `ae85e6b6a8f0784aae110c37e7372384192d1f13e935933903d1874de4d99683` |
| `v1_temp` | `nb_v1` `cell:1:output:1`（`Elapsed Time (s)`，`Peak: 91.8 C at 42 s`） | `ac3717423bfcec594d94e255fb97683e076284a222f0cf4e61fef5824b623deb` |
| `v1_power` | `nb_v1` `cell:1:output:2`（`Clock Frequency (GHz)`，`Max: 64.5 W at 3.6 GHz`） | `27b8d274336c71e0eaba614128614f5c65a63cc2bb0b2b95bbed21dddf1029ed` |
| `v2_temp` | `nb_v2` `cell:1:output:1` & `nb_v2r` `cell:1:output:3`（`Elapsed Time (min)`，`Peak: 83.4 C at 7.0 min`） | `8a6513042f2adc3dbd923dc0023de3d829d37625f46292919044d2420a6a8e84` |
| `v2_power` | `nb_v2` `cell:1:output:2` & `nb_v2r` `cell:1:output:2`（`Clock Frequency (GHz)`，`Max: 53.2 W at 3.6 GHz`） | `5c2451304a2b890d6256dba21c73183e634f435796c0a8ababe96a8a5aa18460` |

> **防文本泄漏设计**：Notebook 所有图像输出均采用真实 Matplotlib 风格通用 `text/plain = "<Figure size 1200x720 with 1 Axes>"`，不含坐标轴单位、曲线名或峰值数值；PPTX Slide 4 的 DVFS 电压轨（`1.15 V / 0.98 V / 0.85 V`）与传感器采样周期（`dt = 2 ms`）仅存在于内嵌 PNG 框图内，强制验证真实定向视觉读图。

---

## 2. 真实复杂排版 PDF 选材与 CC BY 4.0 许可核验原文

### 2.1 `Speculative-Decoding-2211.17192v2.pdf`
- **题名**：*Fast Inference from Transformers via Speculative Decoding*（ICML 2023 Oral）
- **作者**：Yaniv Leviathan, Matan Kalman, Yossi Matias
- **来源页面**：`https://arxiv.org/abs/2211.17192v2`（`Submitted on 30 Nov 2022 (v1), last revised 18 May 2023 (this version, v2)`）
- **CC BY 4.0 许可核验原文（arXiv `<div class="abs-license">` 实抓）**：
  ```html
  <a href="http://creativecommons.org/licenses/by/4.0/" title="Rights to this article" class="has_license">
    <img alt="license icon" role="presentation" src="https://arxiv.org/icons/licenses/by-4.0.png"/>
    <span>view license</span>
  </a>
  ```
- **`pypdfium2` 逐页实查排版特征**：
  - 标准 ICML 双栏排版，每页顶部含页眉横线，p3/p4/p5 含脚注横线，p3 左栏含带横线的 `Algorithm 1` 伪代码框，p11 正文段首出现 `"Table 2."` 文本引用（无实际表格）。
  - **纯矢量 Form XObject 图**：p3 `Figure 2`（`[0.502, 0.352, 0.885, 0.574]`）、**p5 左右并列双矢量图** `Figure 3`（左 `[0.091, 0.103, 0.473, 0.324]`）与 `Figure 4`（右 `[0.502, 0.103, 0.885, 0.324]`）、p6 宽幅矢量图 `Figure 5`（`[0.091, 0.085, 0.885, 0.197]`）；另 p2 为位图 `Figure 1`（`[0.091, 0.085, 0.885, 0.238]`）。基线 `b1b8c60` 仅登记了 p2 位图，**漏登全部 4 张矢量图**。
  - **带条件独立公式**：p3 `Eq. (1)` $E(\#\ \text{generated tokens}) = \frac{1-\alpha^{\gamma+1}}{1-\alpha}$，紧邻上文限定 `βs are i.i.d., α = E(β)`。
  - **无竖线 LaTeX 三线表**：`Table 1`（p5，限定 `c = \hat{c} = 0`）、`Table 2`（p6，6 列 12 行）、`Table 3`（p7，4 列 32 行含双横线分组）、`Table 4` & `Table 5`（p12）。

### 2.2 `YaRN-2309.00071v3.pdf`
- **题名**：*YaRN: Efficient Context Window Extension of Large Language Models*
- **作者**：Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole
- **来源页面**：`https://arxiv.org/abs/2309.00071v3`（`Submitted on 31 Aug 2023 (v1), last revised 6 Feb 2026 (this version, v3)`）
- **CC BY 4.0 许可核验原文（arXiv `<div class="abs-license">` 实抓）**：
  ```html
  <a href="http://creativecommons.org/licenses/by/4.0/" title="Rights to this article" class="has_license">
    <img alt="license icon" role="presentation" src="https://arxiv.org/icons/licenses/by-4.0.png"/>
    <span>view license</span>
  </a>
  ```
- **`pypdfium2` 逐页实查排版特征**：
  - 高密度数学单栏学术论文，含 **24 条带编号独立公式**（p2 `(1)-(4)`、p3 `(5)-(9)`、p5 `(10)-(13)` 含分段函数 `(11)`、p6 `(14)-(15)` 含 LLaMA/Llama 2 推荐公式 `(15)`、p13 `(16)-(22)`、p14 `(23)-(24)`）。
  - **左右并列双子图**：p7 `Figure 2`（左全量 Loss `2.0–6.0`，右局部放大 `1.8–2.6`）、p8 `Figure 3`（左 `PPL vs Context Size`，右 `Passkey Retrieval Accuracy (%) vs Context Size`）；另含上下双图（p15 `Figure 5` + `Figure 6`）与纯矢量 Form XObject 图（p18 `Figure 7`，基线漏登）。
  - **10 张无竖线 LaTeX 三线表**：p8 `Table 1`（多级跨列表头 `Evaluation Context Window Size: 8192..131072`）、**p9 同页三表**（`Table 2 / 3 / 4`）、p16 `Table 5`、p17 `Table 6 / 7`、p18 `Table 8`、p19 `Table 9 / 10`（基线漏登 p19 两表）。

---

## 3. 真实排版 PDF 对象登记准确率 / 误报率审计表（含 `< 70%` 路线复议红线）

终审时必须在 G29 候选实现上运行真实解析器导入上述 2 份未见 PDF（共 `33` 页），对照下表真值逐类统计 `TP / FP / FN`、召回率（`Recall = TP / GT`）、精确率（`Precision = TP / (TP + FP)`）与误报率（`FPR = FP / (TP + FP)`）：

| 对象类别 | `Speculative-Decoding` 真值数 | `YaRN` 真值数 | 合计真值 `GT` | 匹配判定口径 | 基线 `b1b8c60` 现状 |
| --- | ---: | ---: | ---: | --- | --- |
| **视觉图面板**（位图 + 矢量 Form） | `5`（1 位图 + 4 矢量） | `10`（9 位图 + 1 矢量 p18） | **`15`** | 同页、同类，归一化可见页 bbox `IoU >= 0.45`（或无跨栏/邻物串扰的 `>= 0.65` 包容）；同一真对象重复登记计 `1 TP + (k-1) FP` | 仅登位图 `10/15`（全部 `5` 张矢量图漏登） |
| **无竖线三线表**（`table`） | `5`（Table 1–5） | `10`（Table 1–10） | **`15`** | 同页、同类，bbox `IoU >= 0.45` 且主体列归属可核对 | `13/15`（漏登 `YaRN` p19 Table 9–10） |
| **图注对象**（`caption`） | `5`（Figure 1–5） | `8`（Figure 1–8） | **`13`** | 同页、图号及图注正文锚点重合 `>= 0.60`；准确匹配真表注的 `table caption` 单列统计不计为 FP | `0/13`（未登记） |
| **带编号独立公式**（`formula`） | `1`（p3 Eq. 1） | `24`（p2/3/5/6/13/14 Eq. 1–24） | **`25`** | 同页、公式编号与表达式锚点匹配；`Speculative-Decoding` p4/p12 无编号展示公式记为 `allow_optional`（不惩罚 FP、不计漏检） | `0/25`（未登记） |
| **总计（Micro）** | **`16`** | **`52`** | **`68`** | 另设明确误报陷阱核查：`Algorithm 1` 框（p3）、p11 行内 `"Table 2."`、p14 行内 `"Figure 4."`、页眉/脚注横线与普通正文段 | `23/68`（`33.8%`） |

> [!CAUTION]
> **路线复议红线（`< 70%`）**：在 2 份未见真实复杂排版 PDF 上，若 G29 候选实现的**总体对象登记微平均 Recall `< 70%` 或微平均 Precision `< 70%`（即误报率 `FPR > 30%`）**，或**视觉图面板 Recall `< 70%`、三线表 Recall `< 70%`、图注/公式 Precision `< 70%`** 任一触发，则**强制触发路线复议红线**——不得放行纯启发式 PDFium 规则，必须升级架构负责人复议是否引入更强布局解析器（PyMuPDF / Docling 混合路径）。

---

## 4. 冻结盲测考题与硬门槛覆盖矩阵

| 题号 | 覆盖 G29 硬门槛 | 目标原件与真值锚点 | 核心通过判据 |
| --- | --- | --- | --- |
| **G29-PDF-01** | 真实双栏 PDF 矢量子图定向读图 | `Speculative-Decoding` p5 右图 `Figure 4`（矢量 Form `[0.502, 0.103, 0.885, 0.324]`） | 无需预选卡片定向选读右图；答出 `Speed`（实线）与 `Ops`（虚线）两类曲线及 `α=0.90, γ=10` 处约 `6.8–6.9X / 1.6X`；实读凭证与预览对称 200 |
| **G29-PDF-02** | **聚焦双图页无页码歧义反例（`Q04` 缺陷类）** | `Speculative-Decoding` `PUT /focus` 到 `page:5`（含左右两张矢量图） | 不给页码、不给左右发问 → **严禁**被 `_IMAGE_REFERENCE` 抢跑落入无候选泛化澄清；必须返回 `object_ambiguous` + 左右两个可点击候选（`Figure 3` / `Figure 4`）；选左图后正确读出 `Optimal γ`（`0–24`，参数 `c ∈ {0.01, 0.02, 0.05, 0.1}`） |
| **G29-PDF-03** | 真实单栏并列嵌入双图定向读图 + 聚焦双图歧义 | `YaRN` p8 `Figure 3` 右图（`Passkey Retrieval Accuracy (%)`） & p7 `PUT /focus` 双图 `Figure 2` | `03a` 读出 `NTK-aware` 在 `>25k` 骤降至 `~0%`；`03b` 聚焦 p7 无页码问纵轴范围必须返回左右两候选，选右图答出放大范围 `1.8–2.6` |
| **G29-PDF-04** | **真实 PDF 图注对象定位 + `GET /evidence/{eid}` HTTP 200** | `Speculative-Decoding` p6 `Figure 5` 带条件图注 | 准确答出顶行 `γ=7` / 中行 `γ=3` / 底行标准解码及紫/蓝/黄/橙块对应模型；**全部引用 `GET /evidence/{eid}` 必须 HTTP 200（0 次 409）** |
| **G29-PDF-05** | **真实 PDF 带条件独立公式定位 + `GET /evidence/{eid}` HTTP 200** | `Speculative-Decoding` p3 `Eq. (1)` | 答出 $E(\#\ \text{tokens}) = \frac{1-\alpha^{\gamma+1}}{1-\alpha}$ 及 `β` 独立同分布 `α=E(β)` 前提；**全部引用 `GET /evidence/{eid}` 必须 HTTP 200（0 次 409）** |
| **G29-PDF-06** | 真实 PDF 分段/条件公式 + `GET /evidence/{eid}` HTTP 200 | `YaRN` p5 `Eq. (11)` 分段函数 & p6 `Eq. (15)` 温度缩放公式 | `06a` 答出三段定义及 Llama 推荐 `α=1, β=32`；`06b` 答出 $\sqrt{1/t}=0.1\ln(s)+1$ 及适用/拟合模型范围；**全部 `GET /evidence/{eid}` HTTP 200（0 次 409）** |
| **G29-PDF-07** | 真实双栏无竖线三线表定位与列归属 | `Speculative-Decoding` p5 `Table 1` & p6 `Table 2` | 准确给出 `c=\hat{c}=0` 限定、`1.63X / 3.69X`、`1.60X / 6.86X` 及 `Table 2` 中 `T5-small (0.75, 3.4X)` vs `T5-large (0.82, 1.7X)`，无错列 |
| **G29-PDF-08** | 真实跨列多级表头 & 同页三表三线表 | `YaRN` p8 `Table 1` & p9 `Table 4` | `08a` 准确归属 `65536 / 131072` 列（`2.29 / >10^1` vs `2.31 / 2.24`）；`08b` 准确区分同页第 3 张表（`256+128`、`?`、`64000`） |
| **G29-PPTX-01** | 全新课程 PPTX 多列带单位原生表 | `Week06-Thermal-Cache-Lab.pptx` Slide 2 `page:2:shape:3` | 准确读出最高 Miss Rate `GraphBFS-CSR (11.4%)`、最高 Peak Tj `SparseFFT-16K (91.8 °C, 2.95 GHz, 21.6 mJ·s)` 及 `T_amb=25 °C` |
| **G29-PPTX-02** | 全新课程 PPTX 内嵌高对比度实验图视觉实读 | Slide 3 `page:3:shape:3`（`slide3_chart`） | 视觉读出横纵轴与 8 entries 拐点 `LRU 9.2%` vs `FIFO 10.8%`；预览对称 200 且字节 SHA-256 一致 |
| **G29-PPTX-03** | 全新课程 PPTX 图文混排 + 仅图内信息 | Slide 4 `page:4:shape:4`（`slide4_diagram`） | 视觉读出仅存在于框图内的电压轨 `P0 1.15 V / P1 0.98 V / P2 0.85 V` 与 `dt = 2 ms`；预览对称 200 |
| **G29-NB-01** | 双版 Notebook 无 `PUT /focus` 跨版读新/回旧/双边对比 | `v1` → `v2`（同 `document_id`，**不调用 `PUT /focus`**） | `01a` 读新版 `min` / `83.4 °C`；`01b` `只看旧版` 严格只引 `v1`（`s` / `91.8 °C`，0 串版）；`01c` 双边对比 `v1+v2` 完整非 `partial` |
| **G29-NB-02** | **重排后比较澄清 → 选定候选 → 恢复双边 `b2` 对比** | `v1` → `v2_reordered`（插输出 + 换图序，无 `PUT /focus`） | Turn 1 触发可恢复澄清（保留 `parent_action` + `paired_historical_ref`）；Turn 2 选 `cell:1:output:3` 后**必须恢复 `v1+v2` 双边完整对比**（关闭单边 `partial` 缺陷！），双侧预览均 HTTP 200 且哈希匹配 |
| **G29-NB-03** | 聚焦多输出 Cell 澄清选文本 vs 选图 | `v2` `PUT /focus` 到 `cell:1`（`code-sweep`） | Turn 1 提供文本输出与图像输出可选澄清；选文本答出 `83.4 C / 41.6% / throttle_events=0` 且 `/evidence` 200；选第 2 张图视觉读出 `53.2 W at 3.6 GHz` |
| **G29-CTR-01** | **图像预览端点对称性全局门禁** | 全部含图回合（含 `answer_current_image` 与 `compare_versions`） | 每个 `(display_id, evidence_id)` 的 `/media/{display_id}` 与 `/evidence/{eid}/preview` **均返回 HTTP 200 (`image/png`) 且字节 SHA-256 完全一致** |
| **G29-CTR-02** | 跨会话权限隔离（HTTP 404） | Session B 越权访问 Session A 的媒体/证据/预览/锚点 | 全部 4 类越权请求均返回 **HTTP 404 (`not_accessible`)**，零内容泄漏 |
| **G29-CTR-03** | `stale_reference` 失效引用对照 | 过期/已删/无效锚点或失效澄清状态提交 | 显式返回 `stale_reference` 或受控拒绝/重澄清，严禁静默替换错误对象或错误版本 |
