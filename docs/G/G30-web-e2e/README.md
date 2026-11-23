# G30 轨 2：真实产品配置三格式网页终验（`/workspace/` Chromium E2E）

- **验证角色**：`g30_web_gate`（独立验证者，零修改生产代码、零触碰 `main` 与 A03 私有封存文件）
- **受测归档**：`33f0f14a0333a715c0e8e3a63ac0bb8110aa35f0`（由 `git -C /Users/Link/code/SRAG-g29 archive` 导出至 `/tmp/srag-g30-web-subject-*`，实测确认 `app.rag.workspace` 严格加载自 `/private/tmp/srag-g30-web-subject-6y_qft5e/app/rag/workspace.py`）
- **复现脚本**：[run_g30_web_e2e.py](./run_g30_web_e2e.py)
- **脱敏实测结果**：[G30-web-e2e-results.json](./G30-web-e2e-results.json)

---

## 1. 产品配置与真实公开材料

- **服务配置**：随机本地端口 `http://127.0.0.1:63787/workspace/`、全新空 SQLite 与空资料目录、`WORKSPACE_COURSE_IDS=g30-course`、`WORKSPACE_ANSWER_ENABLED=true`（真实 `deepseek-flash`）、`WORKSPACE_RETRIEVAL_STRATEGY=fts`、`SOURCE_VISION_ENABLED=true`（真实 `qwen3-vl-plus`）、`QDRANT_URL=http://127.0.0.1:1`、`SRAG_SOFFICE=/opt/homebrew/bin/soffice`（`LibreOffice 26.8.3.1`），凭据仅从 `/Users/Link/code/SRAG/.env` 只读加载进内存。
- **公开三格式材料**（只读复用 `/Users/Link/code/SRAG-g29-gate/docs/G/G29-gate/fixtures/`）：
  - `Speculative-Decoding-2211.17192v2.pdf`（13 页 CC BY 4.0 论文）
  - `Week06-Thermal-Cache-Lab.pptx`（4 页课程 PPTX，含 5×5 原生带单位表与嵌入曲线图）
  - `notebook/v1/Lab06-Thermal-Sweep.ipynb` 与 `notebook/v2_reordered/Lab06-Thermal-Sweep.ipynb`

---

## 2. 8 步真实 Chromium 学生流程验收结果（`8/8` 通过）

| 步骤 | 验收场景 | 核心核验结果 | 状态 |
| --- | --- | --- | --- |
| **Step 1** | 三格式空库网页导入 | 通过网页 `#import-files` + `#import-button` 导入 PDF / PPTX / Notebook v1，`#catalog` 均显示 `v1` | **PASS** |
| **Step 2** | 长英文 PDF 中文综合问答 + `#reader` | `deepseek-flash` 生成中文综合回答（`5.51s`），点击引用打开 `#reader`（`v1 · 第 2 页`，`1224×1584`），全部绑定 `/evidence` 返回 HTTP 200 | **PASS** |
| **Step 3** | PDF 复杂对象（三线表 + Q04 双图澄清 + 公式/图注 `/evidence` 200） | `3a` 准确输出 `Table 1`（`1.60X / 6.86X`），点击引用打开裁图（`343×224`），点击「回原件」打开整页（`1224×1584`）且 `.reader-highlight` 归一化框（`14.02%, 56.54%, 27.94%, 14.11%`）无错位；`3b` 目录聚焦 p5 问无页码图题渲染 2 个 `.clarification-actions` 按钮，点击左图后真实 `qwen3-vl-plus` 实读并打开裁图（`518×415`），`/media` 与 `/preview` 均为 HTTP 200 且 SHA-256 一致；`3c` `Eq. (1)`（`447×117`）与 `Figure 5` 图注（`1031×88`）`/evidence/{eid}` 均为 HTTP 200 且 `#reader` 可打开 | **PASS** |
| **Step 4** | 真实 LibreOffice PPTX 全流程 | `4a` 第 2 页原生表准确给出 `GraphBFS-CSR (11.4%)` 与 `SparseFFT-16K (91.8 °C, 2.95 GHz, 21.6 mJ·s)`，`#reader` 渲染 LibreOffice 表格图（`1791×663`，HTTP 200）；`4b` 第 3 页嵌入曲线图经真实 LibreOffice + `qwen3-vl-plus` 读出 `Victim Cache Entries` / `L2 Miss Rate (%)` 与 `LRU` / `FIFO` 趋势，`#reader` 打开原图（`1734×937`，HTTP 200） | **PASS** |
| **Step 5** | Notebook 多输出澄清 → 换版 `v2_reordered` → 只看旧版 → 对比两版 | `5a` 聚焦 `Cell 2` 问图触发 2 个多输出按钮，选 `输出 1` 完成 `v1` 实读（`91.8 C at 42 s`）；`5b` 勾选 `#replace-document` 网页导入 `v2_reordered`，目录升为 `v2`；`5c` `"只看旧版..."` 严格仅返 `v1`（`91.8 C`，零 `v2` 串版，`/preview` SHA-256 与 `v1` 原字节完全一致）；`5d` `"对比最新版和旧版..."` 触发重排澄清，选 `输出 3` 后恢复双边 `v1+v2` 对比（2 张 `.source-card`，`91.8 C` vs `83.4 C`，双侧 `/media` 与 `/preview` 均 HTTP 200） | **PASS** |
| **Step 6** | 浏览器刷新恢复 | `reload` 后 26 条会话消息完整恢复，点击历史 `v1` 引用正常打开 `#reader`（`1200×720`），媒体字节 SHA-256 与刷新前一致（`ac371742...623deb`） | **PASS** |
| **Step 7** | 缺证诚实边界 | 询问三份资料均未记载的 `GPU L3` 容量与 `WMT14 EnFr BLEU`，真实 `deepseek-flash` 明确声明未在已读证据中出现（零幻觉） | **PASS** |
| **Step 8** | 跨会话隔离 | 独立 Chromium 会话 Session B 越权访问 Session A 的 `/media`、PPTX `/evidence`、PDF `/evidence` 与 `/preview`，全部返回 **HTTP 404** | **PASS** |

---

## 3. 真实调用账与耗时分位数

| 指标 | 实测值 |
| --- | --- |
| **总通过率** | **`8/8`（100%）** |
| **严重错误数** | **`0`** |
| **模拟协议调用（`mock_protocol`）** | **`0`** |
| **DeepSeek（`deepseek-flash`）** | **`4` 次真实调用**（`2` 次生成回答 `20,430` tokens + `2` 次按需双语改写 `491` tokens，合计 `20,921` tokens） |
| **Qwen3-VL（`qwen3-vl-plus`）** | **`4` 次真实定向读图调用**（`4,136` tokens，`0` 次缓存复用） |
| **14 轮交互耗时分位数（秒）** | **`p50 = 1.317s` / `p95 = 5.935s` / `max = 6.725s`** |

---

## 4. 复跑命令

```bash
/Users/Link/code/SRAG/.venv/bin/python /Users/Link/code/SRAG-g30/docs/G/G30-web-e2e/run_g30_web_e2e.py
```
