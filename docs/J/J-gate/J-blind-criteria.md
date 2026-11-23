# J · 独立盲测课程套件与冻结判据（开发前独立冻结）

- **冻结角色**：`j_gate`（独立门禁审核者，零修改 `app/` 生产代码、零触碰 A03 私有封存）
- **基线修订**：`d126b0e`（分支 `review/j-gate`）
- **冻结时间**：`2026-11-23`
- **课程 ID**：`cs-mlsys-j`（*CS-MLSys-J: 大模型推理系统与压缩优化*）
- **生成与校验脚本**：[make_course_corpus.py](./make_course_corpus.py)
- **结构化判据 JSON**：[J-blind-criteria.json](./J-blind-criteria.json)

> **保密提示**：在 `J01` 设计审核期间，严禁向执行侧泄露本套件中任何具体文件名、论文标题、自定义函数名或具体题面。

---

## 1. 冻结课程语料清单（9 份多格式原件）

所有原件存放于 `docs/J/J-gate/course_corpus/`，总解析规模 **219 个来源对象 / 194,238 原生字符**（远超单窗 32K 上限），其中 3 份学术论文 PDF 均已核验 arXiv `<div class="abs-license">` 为 `CC-BY-4.0`（`http://creativecommons.org/licenses/by/4.0/`），且全库 `rg` 确认此前从未出现在 SRAG 任何测试或文档中（严格避开 `Mixtral`、`DPO`、`PagedAttention`、`Mamba`、`QLoRA`、`ReAct`、`GaLore`、`FOCCUS`、`SensorLab`、`Speculative-Decoding`、`YaRN`）。

| 文件名 | 格式 | 页/Cell 数 | 字节数 | SHA-256 | 解析来源数 / 字符数 | 用途与许可 |
| --- | --- | ---: | ---: | --- | ---: | --- |
| `Paper-01-GPTQ-2210.17323v2.pdf` | PDF | 16 页 | 521,818 | `35e339171cd48bbf8dda246bd018b11aa33a5cda0c4a1587cdae574208f56396` | 50 / 67,103 | arXiv `2210.17323v2`（CC BY 4.0） |
| `Paper-02-KIVI-2402.02750v2.pdf` | PDF | 15 页 | 2,194,402 | `df31ef32d71bfb280c533c5db8220cadf5ef42076bf45d82ba4c8da8e50ea5f4` | 53 / 62,539 | arXiv `2402.02750v2`（CC BY 4.0） |
| `Paper-03-PositionalInterpolation-2306.15595v2.pdf` | PDF | 18 页 | 751,217 | `890702e3170e5b8e2fb55552bd86cd99b0aa4ddbb3670ae8c351faa575cfaf09` | 51 / 55,863 | arXiv `2306.15595v2`（CC BY 4.0） |
| `Deck-01-Core-Concepts.pptx` | PPTX | 5 页 | 34,903 | `a08ecdf1e5663163de11bce76e474f84e3def8f340412ea502f82643bd43623f` | 16 / 3,052 | 理论概念定义 + 4 个自定义实验入口（`soffice` 已实测） |
| `Deck-02-Pure-Visual-Diagrams.pptx` | PPTX | 4 页 | 282,351 | `47ac8347fd3ea0f6866c57e55765ca61083ba1c14904e3fd61f52c841bc9dfcd` | 12 / 195 | 第 2–4 页仅含 `"Slide 2/3/4"` 文字锚点，信息全在高对比 PNG 中（`soffice` 已实测） |
| `Deck-03-Benchmarks-And-Tradeoffs.pptx` | PPTX | 4 页 | 33,111 | `29e665c36df1447dc0eeaad521f2a472e1f55be09ad64b8a1b7a2e4263d36e9c` | 12 / 2,302 | 原生带单位 5×5 表 + 共享高频干扰词页（`soffice` 已实测） |
| `Lab-01-ZeroOverlap-Experiment.ipynb` | IPYNB | 8 Cells | 54,962 | `d2c08b14673a48c9065c3c7a2c9aa09dfa3310c98161df2ac5fd4994615d6014` | 14 / 1,630 | 无共享概念词实验本（FTS5 零词面重叠已验证） |
| `Lab-02-CrossDoc-Verification.ipynb` | IPYNB | 4 Cells | 35,262 | `61838fb98a6b8cc2f9255b1e28ebee72006e88b180054bd647737c603cadf2c7` | 7 / 989 | 配套 KIVI/PI 多输出验证实验本（stdout + PNG） |
| `Lab-03-Distractor-Baseline.ipynb` | IPYNB | 2 Cells | 31,938 | `e819d978f822af9af0015f29808fa3d2ebab4d964d99d5707c1aea3f9b6d65d1` | 4 / 565 | 异条件干扰实验本（CPU-offload / PCIe / INT8 / SeqLen=512） |

### 3 份 PDF CC BY 4.0 许可核验原文

- `2210.17323`：`<div class="abs-license"><a href="http://creativecommons.org/licenses/by/4.0/" title="Rights to this article" class="has_license"> <img alt="license icon" role="presentation" src="https://arxiv.org/icons/licenses/by-4.0.png"/> <span>view license</span> </a></div>`
- `2402.02750`：`<div class="abs-license"><a href="http://creativecommons.org/licenses/by/4.0/" title="Rights to this article" class="has_license"> <img alt="license icon" role="presentation" src="https://arxiv.org/icons/licenses/by-4.0.png"/> <span>view license</span> </a></div>`
- `2306.15595`：`<div class="abs-license"><a href="http://creativecommons.org/licenses/by/4.0/" title="Rights to this article" class="has_license"> <img alt="license icon" role="presentation" src="https://arxiv.org/icons/licenses/by-4.0.png"/> <span>view license</span> </a></div>`

---

## 2. 18 道冻结考题分布与真值判据

- **Cat A（4 题，`J_Q01`–`J_Q04`）**：全课 9 文件范围内长文后段 + 多文件同名术语干扰题（检验候选池截断与 Jev 重排价值）。
- **Cat B（4 题，`J_Q05`–`J_Q08`）**：跨文件无共享概念词 2-hop 题（讲义定义 → 自定义函数标识 → `Lab-01` 代码/输出，检验单跳双语 vs `S2a` 字面锚补查 vs `S2b` 有界两步检索）。
- **Cat C（4 题，`J_Q09`–`J_Q12`）**：纯图无文字锚点 PPTX 召回题（不给页码与文件名，用中文问 `Deck-02` 纯图页结构/曲线/数值，检验纯 FTS+邻接 vs 可选整页视觉向量召回）。
- **Cat D（3 题，`J_Q13`–`J_Q15`）**：控制组（显式页码/单文件定位，检验增强策略不退化、不引入无谓高延迟）。
- **Cat E（3 题，`J_Q16`–`J_Q18`）**：缺证/条件不匹配/跨课隔离反例（验证 0 幻觉、防 `Lab-03` 张冠李戴、严格跨课隔离）。

| 题号 | 类别 | 预期来源（`expected_sources`） | 必需证据单元 | 原件核验真值摘要 |
| --- | --- | --- | --- | --- |
| `J_Q01` | Cat A | `Paper-01-GPTQ` (`page:13`, `page:14`) | `E01`, `E02` | p13 Table 8: OPT-125M 3-bit WikiText2 `OBQ=69.32, GPTQ=53.85` (FP16 `27.66`)；p14 Table 11: OPT-66B C4 `RTN 4-bit=309., GPTQ 4-bit=11.23` (FP16 `10.99`)，校准集采自 C4 训练集故非完全 zero-shot |
| `J_Q02` | Cat A | `Paper-02-KIVI` (`page:13`, `page:14`) | `E03`, `E04` | p13 Table 6: Llama-2-13B GSM8K `KIVI-2 R128=20.77 -> R32=20.62`；Appendix B NIAH: `7-digit passkey` + `Paul Graham Essays`；p14 Table 7: LongBench Avg `44.69 -> 44.75` |
| `J_Q03` | Cat A | `Paper-03-PI` (`page:10`, `page:15`) | `E05`, `E06` | p10 Table 6: LLaMA-7B Extended 16K GovReport `ROUGE-1=60.0, ROUGE-2=28.0, ROUGE-L=29.5`，`temperature=0.5, topp=0.95`；p15 Appendix B: `d=128, c=10000` |
| `J_Q04` | Cat A | `Paper-01-GPTQ` (`page:8`), `Paper-02-KIVI` (`page:8`) | `E07`, `E08` | GPTQ p8 Table 6: OPT-175B 3-bit 1xA100 `71 ms` vs FP16 5xA100 `230 ms` (`3.24x`)；KIVI p8 Sec 4.2.4: Llama-2-7B 2-bit 1xA100 最大 batch 扩 `4x`、吞吐 `2.35x ~ 3.47x` |
| `J_Q05` | Cat B | `Deck-01` (`page:2`) → `Lab-01` (`cell:1`) | `E09`, `E10` | `热冷锚点分级策略` → `run_bucket_evict_probe()` → `b_cap=64, hot_k=8: t99_ms=19.40, slot_eff=91.2%`; `b_cap=128, hot_k=16: t99_ms=14.70, slot_eff=95.8%` |
| `J_Q06` | Cat B | `Deck-01` (`page:2`) → `Lab-01` (`cell:3`) | `E11`, `E12` | `动态块哈希折叠` → `eval_prefix_reuse_v3()` → `stride=32: gen_tps=1840, v_gb=11.60, red_pct=38.4%`; `stride=64: gen_tps=1695, v_gb=12.85, red_pct=31.7%` |
| `J_Q07` | Cat B | `Deck-01` (`page:3`) → `Lab-01` (`cell:5`) | `E13`, `E14` | `双阶段离群通道抑制` → `calib_permute_omega_k9()` → `tile=128, alpha=0.65: w2_score=8.92, pk_mb=4120`; `tile=256, alpha=0.80: w2_score=9.41, pk_mb=3980` |
| `J_Q08` | Cat B | `Deck-01` (`page:4`) → `Lab-01` (`cell:7`) | `E15`, `E16` | `流水线微批次错峰调度` → `trace_async_gate_w4()` → `depth=4, q_lim=24: wait_pct=3.80%, cyc_ms=6.15`; `depth=2, q_lim=12: wait_pct=9.40%, cyc_ms=7.82` |
| `J_Q09` | Cat C | `Deck-02` (`page:2`) | `E17` | 纯图 PNG：Stage 2A `FP16, cap=256 tokens, sink_tokens=4`；Stage 2B `2-bit asymmetric, group_size=32, residual_buf=128`；Stage 4 `kernel_overhead=4.2%` |
| `J_Q10` | Cat C | `Deck-02` (`page:3`) | `E18` | 纯图 PNG：bs=16 时 `KIVI-Tiered-2bit=21.3, 2bit-Symmetric=30.7, FP16-Static=48.2 ms/tok`；`FP16-Static` 在 `bs=32` 出现 `OOM` |
| `J_Q11` | Cat C | `Deck-02` (`page:4`) | `E19` | 纯图 PNG：Weights `38.4 GB`、KV Pool `67.2 GB`、Runtime Scratch `6.8 GB`、Total `112.4 GB`、Headroom `47.6 GB`、Max Safe Batch `44` |
| `J_Q12` | Cat C | `Deck-02` (`page:3`, `page:2`) | `E20`, `E21` | 纯图 PNG：p3 bs=16 拐点加速比 `2.26x`；p2 Stage 1 `max_seq=16384`，Stage 3 `warp_tile=128` |
| `J_Q13` | Cat D | `Paper-01-GPTQ` (`page:8`) | `E22` | 显式页码直读：Table 5 OPT-175B Wiki2 `Baseline 16=8.34, GPTQ 4=8.37, GPTQ 3/g128=8.45` |
| `J_Q14` | Cat D | `Deck-03` (`page:2`) | `E23` | 显式页码直读：`Mistral-7B | KIVI-2 (GQA-8) | 8192 | 1.95 GB | 1,760 tok/s` |
| `J_Q15` | Cat D | `Lab-02` (`cell:1`) | `E24` | 单文件直读：`R=128: 1420.5 tok/s, 26.4 GB`; `R=32: 1685.0 tok/s, 22.8 GB` |
| `J_Q16` | Cat E | `[]`（全库均无） | `E25` | 诚实声明均未实测：无 1-bit GPTQ OPT-175B，无 KIVI H200-141GB 128K 数据 |
| `J_Q17` | Cat E | `Lab-03` (`cell:0`, `cell:1`) | `E26` | 指出条件不匹配：`47.3 ms/tok, 312 tok/s` 属 `CPU-offload / PCIe Gen4 x16 / INT8-symmetric / SeqLen=512`，非 KIVI A100 2-bit 条件 |
| `J_Q18` | Cat E | `[]`（`scope=other-course-isolated`） | `E27` | 跨课隔离：严禁从 `cs-mlsys-j` 泄漏 `run_bucket_evict_probe` 或任何数值 |
