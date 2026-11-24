# SRAG PDF 子页对象恢复（矢量子图 / 图注 / 独立公式）

- **工作分支**: `codex/g-pdf-object-recovery`
- **基线提交**: `d4c8597d92db20a8803259f47c59faf16ca14f6a`
- **产品修复提交**: `ed616f2606fe5dd331fea77993e8b05c5157201f`
- **正式 G01 运行配置**:
  - `workspace_answer_enabled = False`
  - `source_vision_enabled = True`（`vision_model = "qwen3-vl-plus"`，`provider = "dashscope-native"`）
  - `workspace_retrieval_strategy = "fts"`（`qdrant_url = "http://127.0.0.1:1"`）
  - `workspace_input_token_limit = 32768`
- **范围与约束**: 全程仅使用独立审核者实现前冻结的公开合成夹具（[frozen_criteria.json](./frozen_criteria.json)），未访问任何 `data/g-private/` 或 `docs/A/A03-*` 材料；未合入 `main`。

---

## 一、独立冻结的公开夹具与评判标准

由独立审核者（子代理 `6899b0fe-c8f0-4187-80c5-034fb98dee8b`）在产品实现前生成并冻结：
1. [Vector-Subfigures-Guide.pdf](./fixtures/Vector-Subfigures-Guide.pdf)（`sha256: 0bcc3f74e42982eb308ae710cfff93a25a0b14a4ac6e100da45d3dbc37976d35`）
   - 第 1 页：全宽分隔线 + `Table 1: Channel Calibration Parameters`（`page:1:table:1`）。
   - 第 2 页：左右双矢量子图 `(a) Left Panel: Step Response` 与 `(b) Right Panel: Damped Decay`，以及共享图注 `Figure 1: Qualitative bench waveforms. Only a schematic illustration; does not support causal delay claims.`。
2. [Formula-And-Notes.pdf](./fixtures/Formula-And-Notes.pdf)（`sha256: 7415c8607af4f9b1ce5b724856a2f4596d53bbb2042f48c652d7376f38d1e1a5`）
   - 第 1 页：背景与干扰公式 `E_loss = 0.5 * k * x^2`、干扰图注 `Figure 2`。
   - 第 2 页：独立公式 `tau_eff = (W - 1) / 2`（`Condition: for odd W >= 3 only`）、邻近干扰公式 `alpha_max = 1 / (N + 1)`、矢量图与 `Figure 3` 图注。

6 道冻结考题：
- `Q01_vector_subfigure`：右子图 `(b)` 坐标轴物理量、尖峰时间（ms）、峰值增益（dB）
- `Q02_figure_caption`：`Figure 1` 图注对因果延迟结论的限制
- `Q03_standalone_formula`：`tau_eff` 公式及其适用条件（排除邻近 `alpha_max` 干扰）
- `Q04_ambiguity_counterexample`：仅问“第 2 页那张图”，不指定左/右子图（歧义反例）
- `Q05_table_regression`：`Table 1` 表格回归（验证全宽分隔线不阻断表格识别）
- `Q06_page_citation_regression`：第 1 页全文/页级引用回归

---

## 二、修前基线逐层复现（`d4c8597`）

复现脚本：[repro.py](./repro.py)，产物：[baseline_repro.json](./baseline_repro.json)。

- **对象登记层（Parser / Catalog）**：
  - `Vector-Subfigures-Guide.pdf` 仅登记 `['page:1', 'page:2']`（`0` 个子图、`0` 个图注、`0` 个表格）。
  - `Formula-And-Notes.pdf` 仅登记 `['page:1', 'page:2']`（`0` 个公式、`0` 个图注、`0` 个矢量图）。
- **逐题表现（修前）**：
  - `Q01_vector_subfigure`：`route=None`，退化选中整页 `page:2`（`native_text`），`vision=None`，展示 `rendition_kind="original_page"`，`rect_visible_norm=None`。
  - `Q02_figure_caption`：`route="target_object"`，返回 `object_not_found`（`找不到能核对位置的指定对象`），无证据、无展示。
  - `Q03_standalone_formula`：`route=None`，退化选中整页 `page:2`（`native_text`），`rendition_kind="original_page"`，`rect_visible_norm=None`。
  - `Q04_ambiguity_counterexample`：`turn_status="clarification"`，但未列出第 2 页具体子图候选项。
  - `Q05_table_regression`：`route="target_object"`，因全宽分隔线与表格横线同组导致 `_parse_rule_table` 失败，未登记 `page:1:table:1`，返回 `object_not_found`。
  - `Q06_page_citation_regression`：`page:1` 正常返回，但缺 `page:1:table:1`。

---

## 三、产品修复（`ed616f2606fe5dd331fea77993e8b05c5157201f`）

1. **横线表格分隔线容错**（[pdf_tables.py](../../../app/rag/parsers/pdf_tables.py)）：在 `extract_horizontal_rule_tables` 中，当 4 条及以上横线组整体解析失败时，尝试尾部子序列，避免表格上方的全宽章节分隔线阻断 `Table 1` 提取。
2. **PDF 矢量图对象与图注/独立公式对象登记**（[local_slides.py](../../../app/rag/parsers/local_slides.py)、[local_library.py](../../../app/stores/local_library.py)）：
   - `_pdf_vector_image_sources`：聚合页面 `FPDF_PAGEOBJ_PATH` 矢量路径簇（`path_count >= 3`，尺寸 `>= 48x48`，排除与表格重叠或整页边框），登记为 `page:{p}:image:{i}`（`position_basis="pdfium_native_vector_extents_v1"`）。
   - `_pdf_text_object_sources`：基于 PDFium 文本块提取图注（`caption`）与独立公式（`formula`）`page_object`，将文本写入 `locator["object_text"]`（保持 `SourceDraft.text=""` 以不破坏既有 `searchable_count` 与 FTS 页级排序，`position_basis="pdfium_native_text_block_v1"`）。
3. **对象意图与消歧路由**（[workspace_objects.py](../../../app/rag/workspace_objects.py)、[workspace.py](../../../app/rag/workspace.py)）：
   - 将 `subfigure` 归入图片对象、`caption`/`formula` 纳入 `ObjectKind` 与 `_NATIVE_POSITION_BASES`。
   - 当快照中不存在匹配的 `caption`/`formula` `page_object`（如 Notebook 场景）时回退原有通用流程；存在时按对象自身文本/位置直接打分消歧并支持 `_` 连写术语（如 `tau_eff`）。
4. **确定性回归测试**：更新 [test_local_slides.py](../../../tests/test_local_slides.py)，新增 [test_pdf_object_recovery.py](../../../tests/test_pdf_object_recovery.py)。

---

## 四、修后同题对照（ASGI + 真实浏览器点击）

- **修后 ASGI 产物**：[post_fix_asgi.json](./post_fix_asgi.json)
- **修后真实浏览器产物**：[post_fix_browser.json](./post_fix_browser.json)（脚本：[browser_verify.py](./browser_verify.py)）

### 1. 对象登记结果（修后）
- `Vector-Subfigures-Guide.pdf`：`['page:1', 'page:1:table:1', 'page:2', 'page:2:image:1', 'page:2:image:2', 'page:2:caption:1']`
- `Formula-And-Notes.pdf`：`['page:1', 'page:2', 'page:2:image:1', 'page:1:caption:1', 'page:1:formula:1', 'page:2:caption:1', 'page:2:formula:1', 'page:2:formula:2']`

### 2. 逐题四维指标（修后）

| 题号 | 正确入选对象 | 实际读取通道与内容 | 引用定位（文件 / 版本 / 页码 / 裁区） | 真实浏览器点击 `#reader` 核验 |
| --- | --- | --- | --- | --- |
| `Q01_vector_subfigure` | `page:2:image:2` (`route=target_object`) | 真实 `qwen3-vl-plus`（`dashscope-native`, `external_call=true`, `416 image_tokens`, `1853.4 ms`）读出 `Gain [dB]`、`Time [ms]`、`8 ms`、`+6 dB` | `Vector-Subfigures-Guide.pdf` · `v1` · 第 2 页 · `object_crop` · `[0.498366, 0.178737, 0.961634, 0.641263]` | `ev-1` 点击打开 `open=true`，`has_img=true`（`569x733`），标题含 `(b) Right Panel: Damped Decay` |
| `Q02_figure_caption` | `page:2:caption:1` (`route=target_object`) | `native_text:succeeded` 读出 `Figure 1: Qualitative bench waveforms. Only a schematic illustration; does not support causal delay claims.` | `Vector-Subfigures-Guide.pdf` · `v1` · 第 2 页 · `object_crop` · `[0.063971, 0.626763, 0.671554, 0.704438]` | `ev-1` 点击打开 `open=true`，`has_img=true`（`744x124`），标题含 `Figure 1` 图注 |
| `Q03_standalone_formula` | `page:2:formula:1` (`route=target_object`) | `native_text:succeeded` 读出 `tau_eff = (W - 1) / 2` 与 `Condition: for odd W >= 3 only`（未混入 `alpha_max`） | `Formula-And-Notes.pdf` · `v1` · 第 2 页 · `object_crop` · `[0.098039, 0.138485, 0.901961, 0.301515]` | `ev-1` 点击打开 `open=true`，`has_img=true`（`986x259`），标题含 `Effective Group Delay (Standalone Formula):` |
| `Q04_ambiguity_counterexample` | 无强选（触发澄清） | `turn_status="clarification"`，提示 `请明确要读取的当前文件或图片。` | `citations=[]`，`displays=[]` | 网页显示澄清提示，未误选左或右子图 |
| `Q05_table_regression` | `page:1:table:1` (`route=target_object`) | `native_table:succeeded` 读出 `CH-A (Step)` 与 `CH-B (Decay)` 行 | `Vector-Subfigures-Guide.pdf` · `v1` · 第 1 页 · `object_crop` · `[0.077712, 0.312828, 0.922288, 0.487374]` | `ev-1` 点击打开 `open=true`，`has_img=true`（`1034x278`），标题含 `Channel | Nominal Range` |
| `Q06_page_citation_regression` | `page:1` + `page:1:table:1` | `native_text:succeeded` + `native_table:succeeded` | `ev-3` `original_page` (`rect_visible_norm=null`)；`ev-4` `object_crop` | `ev-1` 点击打开 `open=true`，`has_img=true`（`1224x1584`），标题 `Vector-Subfigures-Guide.pdf · v1 · 第 1 页` |

> 注：在 ASGI 原始接口巡检中，`Q01` 的 `GET /evidence/ev-1` 返回 `404`（图片类块走 `source_image` 媒体预览接口而非文本证据接口），`Q02`/`Q03` 的 `GET /evidence/ev-1` 返回 `409`（`caption`/`formula` 文本存放在 `locator["object_text"]`）；而在前端真实浏览器点击 `button[data-evidence-id]` 时，所有 5 题有引用的回答均正常打开 `#reader` 并渲染对应版本的 `object_crop` / `original_page` 图像。

---

## 五、离线测试与复跑命令

在 `ed616f2606fe5dd331fea77993e8b05c5157201f` 上执行：
- `ruff check --no-cache app tests` -> `All checks passed!`
- `PYTHONPATH=. ../SRAG/.venv/bin/pytest -q -rs -p no:cacheprovider` -> `642 passed, 3 skipped, 1 warning in 27.95s`（3 个跳过均为本机未配置 `SRAG_SOFFICE`）

复跑命令：
```bash
PYTHONPATH=. ../SRAG/.venv/bin/python docs/G/G-pdf-object-recovery/repro.py --out docs/G/G-pdf-object-recovery/post_fix_asgi.json
PYTHONPATH=. ../SRAG/.venv/bin/python docs/G/G-pdf-object-recovery/browser_verify.py --out docs/G/G-pdf-object-recovery/post_fix_browser.json
```
