# SRAG 项目交付完成总览与原版演进对照（交接与面试记忆补全文档）

> **文档用途**：供下游主代理（面试复盘、简历梳理、源码导学或下一阶段迭代代理）一次性补全 SRAG 项目的完整上下文。本文所有代码结构、模块职责、演进对比与实测数据均严格核自仓库 `main` 最新交付基线，**零虚构指标、零个人课程原文泄漏**，可直接用于简历撰写与技术面试深挖。
>
> **权威基线**：`origin/main` · `f16b771d84dbc6b5a07b8be81e824c4c52091219`（2026-11-24，M0–M4 全部 11 个分部 `A–G / J / H / R / K` 经独立门禁审核通过并按 `git merge --no-ff` 完整合入 `main`）。

---

## 一、项目定位与当前交付规模一览

### 1.1 一句话定位
**SRAG** 是一个面向高校课程 **PDF（含双栏学术论文）、PPTX 与 Jupyter Notebook（`.ipynb`）** 的**本地优先（Local-First）多模态课程问答、原件核对与单课程可溯源知识图谱工作区**。学生可在无 Docker、无外部向量服务的条件下启动本地浏览器工作区 `/workspace/`，导入课程资料后即时检索、点开回答中的原页/表格裁图/论文局部图/Notebook 输出图核对并继续追问，沿有原文依据的概念图谱在讲义定义与实验结果之间双向跳转，且在课件换版（`v1 → v2`）后严格区分当前版新答与历史版旧引。

### 1.2 交付体量与质量基线（实测核验）

| 指标维度 | 原版起点（2026-10 `33fad8e`） | 当前 M4 交付版（2026-11 `f16b771`） | 净变化 / 说明 |
| --- | --- | --- | --- |
| **应用代码（`app/`）** | 40 个文件（约 8,978 行） | 84 个文件（**39,218 行**） | `+30,502 / -262` 行（保留全部旧入口兼容） |
| **自动化测试（`tests/`）** | 27 个测试文件 | **62 个测试文件（26,033 行）** | `+20,030` 行；`pytest` **`697 passed, 0 failed, 0 skipped`**（真 `LibreOffice 26.8.3.1`） |
| **静态与语法检查** | 基础检查 | `ruff check` **0 错**；`node --check workspace.js` **退出码 0** | 全量通过 |
| **提交与治理轨迹** | 原型阶段 | 216 笔完整保留历史的 commits（`--no-ff` 合入） | 覆盖 `A–G, J, H, R, K` 全套设计/实现/独立盲审报告 |
| **严重错误红线** | 历史 `candidate2` 原 30 轮 0 严重错误 | **全链路所有测试轨道（G30 / J03 / H03 / K02 / K03）均为 `0` 严重错误** | 严守“引用必属本轮授权实读、缺证不编造”底线 |

### 1.3 下游代理使用口径红线（务必遵守）
1. **绝不混并口径**：原版 `ChatSession`（`candidate2` 历史 53 轮）、G30 轨 1（A03 原锁无文字生成内核）与当前完整产品 `/workspace/`（开启 DeepSeek + `qwen3-vl-plus`）是三套不同运行边界，陈述时必须分列分母与配置。
2. **绝不夸大未交付项**：桌面端安装包、本地文件夹自动监听、MCP / Skill 接入、完整检索轨迹可视化 UI 及 `S1`（Jev）线上默认开启均在需求池或关闭态，不得宣称已上线。
3. **严格区分四层证据**：自动化单测/端到端脚本、AI 初评、人工核对签认（`R01–R03`）与未覆盖项必须分开表述。

---

## 二、核心演进：原版 SRAG vs 当前 M4 完整版

### 2.1 全景对比表（12 个核心维度）

| 维度 | 原版 SRAG（2026-09 ~ 2026-10，基线 `33fad8e`） | 当前 M4 完整版（2026-11，`f16b771`） | 关键工程价值 |
| --- | --- | --- | --- |
| **1. 产品形态与第一入口** | CLI 终端（`python -m tools.chat`）+ 仅含索引/检索的 REST API（`app/api/rag.py`），无网页问答与原件浏览 | 同源浏览器工作区 `/workspace/`（`app/web/workspace/` 1,661 行原生 HTML/CSS/JS）+ 完整工作区 API（`app/api/workspace.py`），同时保留旧 CLI/API 兼容 | 从“命令行调试工具”升级为“边看原件边追问的完整课程工作台” |
| **2. 默认部署与外部依赖** | 强依赖 Docker Qdrant + 云端 Embedding 才可入库与检索，PPTX/PDF 常需 Colab GPU 预处理包 | **零 Docker、零默认 Qdrant**；仅需本机 Python + SQLite FTS5 即可导入三格式、全文快搜、浏览原件与使用离线概念图谱；Qdrant 降级为按需嵌入式本地目录（`QDRANT_LOCAL_PATH`） | 首用门槛由“搭向量库+跑向量化”降至“秒级本地发布可搜” |
| **3. 存储与检索架构** | 向量中心：Qdrant Dense+Sparse RRF + 整页视觉向量为主，SQLite 仅做辅助元数据 | **SQLite 版本快照中心**：`local_library.py` + `structure_snapshot.py` 同事务发布正文/结构/FTS5；默认 `S0`（FTS5 + 结构邻接 + 按需双语改写 + G23 `Decimal` 数量等价校验）；可选 `S2a/S2b` 有界两步检索与 `S3` 嵌入式本地向量+视觉闭环 | 首查快（离线零调用）、可解释、无向量库宕机阻塞 |
| **4. 身份与版本控制** | Notebook 用绝对路径派生 ID（挪目录即变 ID）；PDF/PPTX 用内容哈希前 24 位（改内容即变 ID）；缺乏历史引用原件回看契约 | 统一跨层身份三元组 **`(document_id, version, source_id)`** + `content_sha256` + 单调 `publication_epoch`；事务切换版本、失败保留上一版、预约到期提升（`promote_due`）、迟到增强写回 CAS 校验；换版后新答只见 `v2`，旧回答引用精确回看 `v1` 原件字节 | 彻底解决课件改版、重导、删档下的串版与旧引用错位 |
| **5. 三格式解析深度** | Notebook 按 Cell Group 切块；PDF/PPTX 偏重页级文字/整页截图与 Colab 包导入 | `local_slides.py`（1,851 行）+ `pdf_tables.py`（548 行）+ `local_notebook.py`（355 行）：支持 PDFium 文本与可信图像框、**无竖线学术三线表**（横线+字符坐标列对齐）、公式/图注对象、旋转/裁边统一投影 `visible_page_norm_v1`；PPTX 原生表格与编组 shape 坐标 + LibreOffice 渲染；Notebook 不执行代码只读解析并保留 0 基 `cell_index` + `output_index` 多图归属 | 从“只到页/块级”升级为“精确到页内左右子图、三线表单元格、公式与同 Cell 第 N 张输出图” |
| **6. 多模态展示与视觉读取** | `VisualTextReader` 按页/块读图后写回文本并顺手做 Embedding；“展示过图片”与“模型实际读过图片”未严格拆账 | **展示（`display_source`）与实读（`actual_source` / `actual_input_sha256` / `origin_execution`）严格双账本**；消息内直接内嵌整页 PPT、表格裁图、双栏论文局部图与 Notebook 输出图；点击“回原件”打开同版阅读器并按统一坐标系高亮；框不可靠时显式降级 `parent_page_fallback`（绝不伪造错位框）；通用转写与定向读图分缓存隔离 | 杜绝“展示图 A 却读图 B”或“未读图却宣称读出数值”的多模态幻觉 |
| **7. 会话编排与多轮追问** | `dialogue.py` / `chat.py` 维护简单主题词与最近两轮历史，无消息内图序绑定与可恢复澄清动作 | **四阶段显式管道**（`workspace_context.py` Scope/Focus → `workspace_router.py` 统一意图 → `workspace_clarification.py` 可恢复 `ClarificationAnchor` → `workspace_evidence.py` 统一证据与预览）；区分原件内图序与消息可见展示顺序；支持双独立浏览器会话严格 HTTP `404` 隔离、刷新持久恢复、同 `client_turn_id` 幂等去重与故障原位重试 | 复杂多轮追问、歧义澄清与跨版对比全流程状态可控可恢复 |
| **8. 跨文件补证据与条件归属** | 单次 Top-K 召回或单文件分批总结，缺乏有界跨文件“定义+实验”缺项补查 | `evidence_supplement.py`（1,166 行）+ `workspace_composition.py`（672 行）：拆分“定义/条件/实验结果”，仅从已读原文提取字面锚在授权范围内最多追加一跳；验证实验条件与直接引用支持，收窄文件范围后识别被排除的旧实验并输出 `experiment_outside_scope` | 覆盖跨文件“讲义定义 → 实验验证”真实学习链路且零越界 |
| **9. 知识图谱** | 无 | `local_kg.py`（2,245 行）+ `workspace_kg.py`（689 行）：**离线零外部模型**确定性构建单课程可溯源知识图谱，仅接受带 12 字段字符精确原文 span 的 `defined_in` / `demonstrated_by` 边，硬拒收 5 类伪边反例；默认启用 `#kg-panel` 与消息内 `.kg-concept-chip` 双入口导航，问答一跳候选（`workspace_kg_retrieval_enabled`）默认关闭按需启用 | 不依赖昂贵/易幻觉的 LLM 抽图，保证每条概念边 100% 可回溯原文且随换版/删档原子失效 |
| **10. 预算与防幻觉护栏** | 字符上限估算（14,000/10,000 字符）与实际发送 JSON 序列化长度不一致，长课件易触发 `budget_exceeded` 假失败 | **G21 统一预算模型**（`workspace_reading.py` + `workspace_generation.py`）：阅读窗口与生成预检共用最终 `messages` 二次 JSON 序列化字节估算（0 字节误差，32,768 上限含 `8,192` 输出预留）；引用后校验强制每条结论绑定本轮已授权实读证据 | 彻底消除“已读证据却因估算偏差 0 调用失败”，守住 0 严重错误底线 |
| **11. 外部模型/策略实证取舍** | 经验性关闭 Outline/Rerank/HyDE；未对接 Jev 决策评分或同预算对比 | 基于 `SAG-Wiki` (`92cec19`) 协议接入真实 Jev (`Zev-1`)、有界两步 `S2a/S2b` 与本地嵌入式向量 `S3`，在未见多文件课程上做**同快照同预算实测淘汰**：量化证明 `S1` 延迟超红线（`p95=39.3s`）而果断关闭，定型 `S0` 默认 + `S3` 纯图/语义按需增强 | 面试可讲清“用实验数据决定删掉/关闭某个流行方案”的工程克制 |
| **12. 评测与门禁体系** | 单次固定 53 轮离线脚本 + 人工抽查缺失（AI 初评存在误差） | 建立**预冻结盲测语料 + 执行/审核物理隔离 + `git archive` 干净复现 + 四层结论分列**门禁流程；覆盖 A03 双轨、G29 论文 68 对象审计、J03/H03/K03 三套独立未见盲测、`R01–R03` 人工复核与 `C1–C7` 故障注入 | 既能客观量化收益，也能诚实暴露边界与架构权衡 |

### 2.2 为什么这样演进？（可直接在面试中讲清的架构判断）
1. **借鉴 SAG / Carbon 但不照搬**：吸收了“本地资料优先、原生文字先可搜、图表按需读取”的思路，但针对课程学习独有的**多课程权限隔离、课件换版追溯、双栏论文/Notebook 多图精准定位与跨轮一致性**，补建了版本化 SQLite 契约与四阶段状态机。
2. **保留原版兼容而非推倒重来**：`tools.chat`、`/api/rag/*`、旧 `MetadataStore` 与 Qdrant 向量链路完整保留并持续通过回归；旧数据可通过 `app/rag/local_migration.py`（622 行）显式回填迁移至本地 FTS5 快照。
3. **从“堆正则打补丁”转向“契约收口 + 盲测驱动”**：在 G28 末期 `workspace.py` 积累过多平行正则导致路由互相抢跑；G29 主动停下来先拆清四阶段边界与公共契约再合并分支，随后在 K03 经 8 轮严格盲审把最后几类通用路由/跨版/多表边界逐一闭环。

---

## 三、已完成模块全景与代码地图（按 A–K 分部）

### 3.1 核心代码目录与模块职责

| 层次 | 核心文件（行数） | 核心职责与关键机制 |
| --- | --- | --- |
| **HTTP & Web 前端** | `app/main.py` (50)<br>`app/api/workspace.py` (307)<br>`app/web/workspace/{index.html,workspace.css,workspace.js}` (1,661) | FastAPI 统一入口；同源守卫（`_same_origin` + `HttpOnly SameSite=Strict` Cookie，跨会话/未授权课程 fail-closed 返回 `404`）；单页课程工作区（目录树、导入/换版、行内图卡、原件阅读器归一化高亮、`#kg-panel` 与 `.kg-concept-chip`、澄清按钮与失败重试） |
| **本地解析层 (B/G11/G27)** | `app/rag/parsers/local_slides.py` (1,851)<br>`app/rag/parsers/pdf_tables.py` (548)<br>`app/rag/parsers/local_notebook.py` (355)<br>`app/rag/pdf_coordinates.py` (33) | PDFium 文本与独立图像对象抽取、水平矢量线无竖线三线表列对齐、公式/图注识别及 T1–T3/V1–V5/X1–X4 复杂排版护栏；PPTX 原生表格与编组 shape 坐标；Notebook 不执行代码只读解析；统一投影到 `visible_page_norm_v1` |
| **本地存储与版本 (B/E/H)** | `app/stores/local_library.py` (879)<br>`app/stores/structure_snapshot.py` (532)<br>`app/stores/local_visibility.py` (110)<br>`app/stores/workspace.py` (631)<br>`app/stores/local_kg.py` (2,245)<br>`app/rag/local_migration.py` (622) | SQLite 同事务发布正文/对象/FTS5；版本与 `publication_epoch` 原子切换；预约发布 `promote_due`；会话/轮次/有序消息块/`client_turn_id` 幂等持久化；单课程可溯源知识图谱存储与 5 类反例拒收构边；旧库/旧包迁移回填 |
| **检索与多策略层 (C/F/J)** | `app/rag/structure_search.py` (917)<br>`app/rag/workspace_retrieval.py` (2,421)<br>`app/rag/workspace_vector.py` (848)<br>`app/rag/evidence_supplement.py` (1,166)<br>`app/clients/jev.py` (573) | 显式位置直读、FTS5 + 无空格中文/汉字数字切片、按需双语改写 + G23 `Decimal` 精确数量等价校验；跨页多表 `budget=6` 3 槽互补排序；有界 2 步 `S2a/S2b`；`LocalVectorBridge` 嵌入式本地 Qdrant（同步单实例+锁+损坏 0 阻塞退回 `S0`）；Jev 48 KB 三段摘取+分批+二分重试客户端 |
| **原件裁图与视觉 (D)** | `app/rag/source_evidence.py` (1,186)<br>`app/rag/source_vision.py` (718)<br>`app/rag/workspace_preview.py` (108) | 版本化原件渲染（LibreOffice PPTX / PDFium / Notebook 原图）+ 同键单航班并发防抖 + 裁图安全检查与 `parent_page_fallback`；定向读图与通用转写分离缓存、输入图字节 `SHA-256` 绑定、迟到写回代次拦截 |
| **工作区四阶段编排 (E/G29/H/K)** | `app/rag/workspace.py` (4,489)<br>`app/rag/workspace_context.py` (383)<br>`app/rag/workspace_router.py` (450)<br>`app/rag/workspace_clarification.py` (394)<br>`app/rag/workspace_evidence.py` (212)<br>`app/rag/workspace_objects.py` (697)<br>`app/rag/workspace_versions.py` (1,953)<br>`app/rag/workspace_reading.py` (435)<br>`app/rag/workspace_generation.py` (674)<br>`app/rag/workspace_composition.py` (672)<br>`app/rag/workspace_kg.py` (689) | 四阶段回合编排（Scope/Focus → 统一意图路由 → 可恢复澄清锚点 → 统一证据/预览）；页内左右/上下多图方位消歧（G25）；自然跨版对比与 nbformat 4.5 随机 `cell_id` 可靠配对（含 `_cell_id_moved` 移位防护与重排澄清）；G21 0 字节误差统一阅读/生成预算；概念图谱双入口 API 与受控一跳重读 |
| **交付样例与验收工具 (K)** | `examples/demo-course/`<br>`docs/K/scripts/run_k02_public_acceptance.py`<br>`docs/K/K-gate/run_k03_gate.py` | 可公开再分发样例课程（CC BY 4.0 *Axis-LoRA* + *FlashAttention-3* + 配套 PPTX + `v1/v2` IPYNB + 干扰 PDF）与独立盲测课程（CC BY 4.0 *Graph-of-Thought* + PPTX + `v1/v2` IPYNB）；一键干净环境 Chromium 端到端验收脚本 |

### 3.2 全里程碑完成时间线与放行结论（M0–M4 全部完成）

| 里程碑 / 分部 | 核心目标与交付物 | 最终状态与关键验证 |
| --- | --- | --- |
| **A · 基线与公共契约** | 固定 `(document_id, version, source_id)`、显示 vs 实读、坐标系与兼容性基线 | **通过**（[A01](./A/A01-公共契约.md) · [A04](./A/A04-整部审核.md)） |
| **B · 本地导入与版本发布** | PDF/PPTX/IPYNB 零向量服务本地解析、事务发布、换版/回退、旧库迁移 | **通过**（[B05](./B/B05-独立整审.md)） |
| **C · 范围与结构检索** | 显式位置直读、双语词法、同预算一跳结构邻接、跨课隔离 | **通过**（[C04](./C/C04-检索独立整审.md)） |
| **D · 原件、裁图与视觉读取** | 版本化渲染、并发单航班、无框回退、定向读图双账本与缓存校验 | **通过**（[D04](./D/D04-原件独立整审.md)） |
| **E · 网页、回答与持久会话** | 同源 `/workspace/` 网页、有序块与行内原图、目录树、多轮刷新与会话隔离（含 M1） | **通过**（[E04](./E/E04-M1.md) · [E05](./E/E05-网页独立整审.md)） |
| **F · 跨文件补充证据** | 定义+实验缺项拆解、原文锚点最多一跳补查、冲突分列与降级 | **通过**（[F04](./F/F04-跨文件独立整审.md)） |
| **G（G01–G30）· 日常版整体验收（M2）** | 经历多轮诊断（G20 证据断点、G21 预算统一、G22/G23 双语数量校验、G24–G28 专项）、G29 四阶段编排重构与两路整合、G30 双轨 M2 裁决 | **通过（双轨 M2 放行）**（[G29 终审](./G/G29-gate/G29-独立终审报告.md) · [G30 A03](./G/G30-a03-retest/README.md) · [G30 网页 E2E](./G/G30-web-e2e/README.md)） |
| **J（J01–J03）· 真实多文件精准检索定型（M3 前半）** | 同预算对比 `S0 / S1(Jev) / S2a / S2b / S3(本地向量+视觉)`，定型各策略默认开关与降级条件 | **通过**（[J02](./J/J02-对照实验报告.md) · [J03 盲验定型](./J/J-gate/J03-独立审核与策略定型报告.md)） |
| **H（H01–H03）· 单课程可溯源知识图谱（M3 后半）** | 离线零模型原文支持构边、5 类反例拒收、双入口导航（默认开）与受控一跳问答（可选） | **通过**（[H02](./H/H02-实现与自测.md) · [H03 盲验](./H/H-gate/H03-独立审核报告.md)） |
| **R（R01–R03）· 历史冻结答案人工复核（M0 闭环）** | 只读对照历史 `candidate2` 53 轮，重点 8 题人评签认、45 题显式未看，历史结论隔离留档 | **通过**（[R01](./R/R01-复核口径与覆盖表.md) · [R02](./R/R02-人工复核汇总.md) · [R03](./R/R03-独立审核.md)） |
| **K（K01–K03）· 课程产品最终交付（M4）** | `README`/`.env.example` 重写、公开可分享样例课程、干净环境 `C1–C7`、固定 53 轮 + 未用真实课件 + 独立未见盲测（8 轮打磨终过） | **通过（M4 附 C4 架构豁免放行）**（[K02 整机报告](./K/K02-交付包与整机质量报告.md) · [K03 终审](./K/K-gate/K03-独立终审报告.md)） |

---

## 四、核心实测数据与证据账本（面试可直接引用的真实数据）

> 以下全部为已封存/已入库实测值，每组均标清测试集、配置与分母，**禁止张冠李戴**。

### 4.1 解析、检索、图谱与双轨门禁关键实测

1. **G21 预算一致性修复（真实 Chromium）**：
   - 修复前因读后二次 JSON 转义膨胀导致本地预检超 `32,768` 拦截（`33,939 > 32,768`，0 次模型调用）；统一最终 `messages` 序列化口径后，阅读窗与生成预检误差 **`0` 字节**（`32,753 <= 32,768`、陌生 37 页 *Mamba* `32,768 == 32,768` 满额），1 次真实 DeepSeek 调用完成中文综合。
2. **G23 双语改写数量等价校验（`Decimal`）**：
   - 冻结 40 组中英文等价/不等价正反例（含“五千四百亿” vs `540-billion`、百万/十亿量级错译、单位与比较方向颠倒）：**`40 / 40`（0 误放行、0 误拦）**；陌生 33 页 *ReAct* 与 23 页 *GaLore* 真实 Chromium 救回目标页至 #1。
3. **G29 复杂排版 PDF 审计与四契约闭环**：
   - 未见两篇真实 ICML/arXiv 论文（*Speculative Decoding* + *YaRN*，共 33 页、68 个对象）审计：**Recall `77.9% → 94.1%`、Precision `44.2% → 92.8%`（FPR 降至 `7.2%`）**（视觉图 `15/15`、三线表 `15/15`、图注 `13/13` 均 100%，编号公式 `21/25`）。
4. **G30 双轨 M2 验收（固定 SHA `33f0f14`）**：
   - **轨 1（A03 原锁无文字生成内核，10 题 11 轮）**：较 G19/联合候选，完成度 `9/10 → 10/10`、对话一致 `0/2 → 2/2`、关键题 `2/5 → 4/5`、整题 **`4/10 → 6/10`**、`读对答错` `1 → 0`、0 回退、0 严重错误。
   - **轨 2（真实产品配置三格式 Chromium E2E）**：开启 DeepSeek + `qwen3-vl-plus` + 真 LibreOffice，8 步流程 **`8/8` PASS**，14 轮 `p50 = 1.317s` / `p95 = 5.935s`，0 严重错误。
5. **J03 多文件精准检索同预算盲测（未见 9 文件 18 题 `cs-mlsys-j`，`budget=6, pool_cap=12`，全部 0 严重错误）**：
   - `S0_pool12`（默认 FTS+按需双语）：`7/18 (38.9%)`（`p50=3.43s, p95=4.62s`；`Cat D/E` 6/6 全通）→ **保留为默认零依赖底座**；
   - `S1`（Jev `Zev-1` 重排，48 KB 确定性修复后 `0/27` 批超限）：`10/18 (55.6%)`（长文多文件 `Cat A` `1/4 → 3/4`），但远端 `p95 = 39.33s`（增量 `+34.71s >> +8s` 门槛）且低置信回退 `21.4% > 20%` → **果断关闭**（留重测门槛 `p95 <= +8s`）；
   - `S2a / S2b`（有界 2 步）：`S2a` `7/18`（`Cat B 2/4`，非 2-hop 有 2 题退化）；`S2b` `9/18 (50.0%)`（`Cat B 2/4`，0 退化、`p95=6.23s`）→ **显式可选**；
   - `S3`（本地嵌入式 Qdrant + 文本/整页视觉向量 + 前移有界 D03 读图）：**`14/18 (77.8%)`**、**纯图无正文 PPTX（`Cat C`）`4/4 (100%)`**（`p50=3.88s, p95=12.02s`），损坏 Qdrant 0 阻塞退回 `S0`，`v1→v2` 换版零旧版泄漏 → **默认关闭，定型为显式/按需视觉与语义增强模式**。
6. **H03 单课程知识图谱盲测（未见课程 `cs-ctrl-h` + 干扰课 `cs-other-h`）**：
   - 离线零模型构边：`10/10` 真边命中且 100% 字符精确原文 span、`10/10` 反例 **0 误接受**、0 垃圾概念；
   - 同预算三臂对照（`budget=6, pool_cap=12`，0 严重错误）：`S0_base 8/10`（零重叠跨文件 `2/4`）vs `S0_adj 8/10`（`2/4`）vs **`S0_kg 9/10`（零重叠跨文件 `4/4`，`p50=3.020s`）**；
   - 结论分列：双入口导航（`M1–M5`）**默认启用**；问答一跳（`workspace_kg_retrieval_enabled`）因 6 槽挤占权衡保持**默认关闭、按需启用**。
7. **R01–R03 历史冻结答案人工复核（`candidate2` 53 轮）**：
   - 冻结目录 `3,090/3,090` 文件 SHA-256 只读零改动；重点 8 题完成 `8/8`（**同意 AI `7/8`**、`旧 22` **分歧-应通过 `1/8`**、待决 `0`），其余 `45/53` 显式标为 `未看-范围外`（绝不混作同意 AI）；原 30 轮维持 `24/30`、旧 23 轮人评口径 `22/23`。

### 4.2 K02–K03 M4 整机终审实测全景（最终交付 `a8de427` / `aa3b19b` / `7459e9d`）

| 评测维度 | 测试载体与配置 | 最终实测结果 | 严重错误 |
| --- | --- | --- | :---: |
| **代码回归与静态** | 干净 `git archive aa3b19b` + 真 `LibreOffice 26.8.3.1` | `ruff` 0 错、`node --check` 退出码 0、`pytest` **`697 passed, 0 failed, 0 skipped`** | **0** |
| **`C1` 干净环境冷启动** | 空目录零密钥离线 + 在线首问 | **`5/5` PASS**（离线 `external_call_count=0`，在线首问 `10.27s` 开 `Lecture-01 p2 1920x1080`） | **0** |
| **`C2` 三种来源核对** | 整页 PPT、双栏表裁图、双栏公式、Notebook 多图、无框回退 | **`5/5` PASS**（`/media == /preview` 同 SHA-256、`parent_page_fallback` 无错位框） | **0** |
| **`C3` 多轮/刷新/隔离/换版** | 澄清、切文件、刷新 `28==28`、双会话 `404`、`v1→v2` + `v1` 旧图 | **`5/5` PASS** | **0** |
| **`C5-Public` 公开样例课程** | CC BY 4.0 *Axis-LoRA* + *FlashAttention-3* + PPTX + `v1/v2` IPYNB | **`10/10` 题（`12/12` 轮）PASS**（主跑 `p50=2.58s`；纯图页 `S0` 漏答 → `S3` 读全） | **0** |
| **`C5-Blind` K03 预冻结盲测** | 独立未见论文 *Graph-of-Thought* (`2305.16582v2`) + PPTX + `v1/v2` IPYNB（8 题） | 经 R1(`2/8`) → R8 通用打磨达 **`7/8 (87.5%)` 覆盖 + `4/4 (100%)` 视觉选读匹配**（`K_Q02 9/9`、`K_Q03 S0+KG 10/10`、`K_Q04 3/3`、`K_Q05 S3 8/9`、`K_Q06 3/3`、`K_Q07 5/5` 含 nbformat 4.5 随机 `cell_id` 跨版实读、`K_Q08` 诚实拒答；`K_Q01` 召回全 3 表得 `5/6` 并诚实报缺） | **0** |
| **`C6` 图谱与策略联动** | 双入口导航 + `S0 vs S0+KG` + `S0 vs S3` | **`3/3` PASS**（`K_Q03` `S0 8/10 → S0+KG 10/10, 0 外呼`；`K_Q05` `S0 0/9 → S3 8/9`） | **0** |
| **`C7` 四类故障注入与恢复** | 坏文件导入、坏 DeepSeek 密钥+重复提交、坏视觉密钥重试、删档过期焦点 | **`4/4` PASS**（`0` 半发布激活版本、SQLite 严格 `1/1/1` 无重复轮次、`retry` 原位恢复） | **0** |
| **`C4` 历史固定 53 轮** *(本地私有)* | `/workspace/ S0 (budget=6)` 对照历史 15 份原件（R2 `10f053b`） | `30/30` 零异常、来源组 **`30/32 (93.8%)` 过 `≥90%` 门槛**、原 30 轮 `18/30`、多轮 `1/4`、旧 23 轮 `17/23`；**按架构口径豁免单列留档** | **0** |
| **`C5-Local-Unused`** *(本地私有)* | 6 份未用真实三格式课件（12 题 + 消融，R2 `10f053b`） | `12/12` 零异常、事实与引用 **`10/10`**、全维满分 `5/12`；`S3` 救回手写公式 PPTX、`S2b` 2 步补齐代码诊断输出 | **0** |

---

## 五、坦诚边界与已知待改进项（面试中展现工程判断力的关键素材）

1. **首次开箱配置体验偏硬（真实试用暴露）**：工作区 API 为保证 fail-closed 安全，未设置 `WORKSPACE_COURSE_IDS` 时 `/api/workspace/sessions` 直接返回 `404`；若新手直接启动 `uvicorn` 打开 `/workspace/` 且未在 `.env` 配置授权课程，页面会无友好引导直接报错（后续应增加面向本机的配置向导或默认可写本地工作空间）。
2. **`C4` 历史 30 轮（`18/30`）与原 `ChatSession`（`24/30`）的架构口径权衡**：原 30 轮假定“常开向量检索 + 多批全文总结 + 遇歧义不暂停自动猜”，而新版 `/workspace/` 默认坚持“零 Qdrant `S0` + `budget=6` 低延迟 + 遇歧义先澄清不硬猜”，因此做到 0 严重错误与 `93.8%` 来源命中，但整题覆盖偏保守；若默认常开 `S3` 或大预算会显著拉高日常延迟（实测 `S3` 首次冷建库需 `60–140s`）。
3. **复杂折行无框 PDF 表格单元格抽取边界**：如盲测 `K_Q01` 中第 7 页无框多栏表把 `87.59` 折行拆到上方空行，导致同行仅剩 `"7.5 (±0.20)"`；当前系统的优点是**触发 `missing_evidence` 诚实报缺而不编造**，但底层 PDF 表格行归并算法仍有提升空间。
4. **偶发结构化 JSON 生成回退与极简拒答文案**：长文档总结时 DeepSeek 偶发结构化校验 `invalid_response` 后安全退回 `excerpt_fallback`（不编造但可读性降级）；个别定向视觉拒答（如 `LU-06`）将具体原因记在内部 `missing_or_error` 而正文过于简略。
5. **纯扫描 PDF 与需求池边界**：纯扫描 PDF 默认不自动后台 OCR（需按需触发）；桌面端安装包、文件夹自动监听、MCP/Skill 接入与完整检索轨迹 UI 明确留在需求池，尚未开发。

---

## 六、高频面试深挖题库与 STAR 故事素材（供面试/简历主代理直接调用）

### 故事 1：如何通过同预算对照实验做检索架构取舍（甚至主动“砍掉”看起来很酷的外部重排方案）
- **考察点**：AI 应用架构判断力、成本/延迟/效果权衡、反盲目堆组件。
- **核心事实**：
  - 面对“该用 FTS、双语改写、Jev (`Zev-1`) 决策评分、2-hop 还是本地向量+视觉”的争论，设计 **J01–J03** 同快照同预算（`pool_cap=12, budget=6`）四策略对照，在未见 9 文件 18 题真实课程上跑全链路盲测。
  - 先用三段摘取 + 42 KB 分批 + HTTP 400/413/422 递归二分彻底修掉了 Jev 的 `input_limit`（`0/27` 批超限），证明 `S1` 确实把长文竞争题从 `1/4` 提到 `3/4`；但实测 `p95` 飙升至 `39.33s`（`+34.71s >> +8s` 门槛）且低置信回退 `21.4%`，因此**基于数据果断决定本版保持关闭**并写死重开条件。
  - 相反，`S3`（零 Docker 嵌入式本地 `QdrantClient(path=...)` + 整页视觉向量 + 前移有界 D03 读图）将总分拉到 **`14/18 (77.8%)`**、纯图无文字 PPTX 拉到 **`4/4 (100%)`**，且在损坏 Qdrant 时 0 阻塞退回 `S0`，定型为按需视觉/语义增强模式；`S0` 保留零依赖默认底座。
- **源码/证据**：`app/rag/workspace_retrieval.py`、`app/rag/workspace_vector.py`、`app/clients/jev.py`、[J03 盲验定型报告](./J/J-gate/J03-独立审核与策略定型报告.md)。

### 故事 2：如何解决多模态 RAG 中“展示了图但没真读 / 读了图 A 引用了图 B / 框画歪了”的可信度难题
- **考察点**：多模态工程落地、防幻觉、可核验引用契约设计。
- **核心事实**：
  - 制定跨层来源三元组 `(document_id, version, source_id)` 与“展示 vs 实读”双账本：消息块存 `display_id` + `display_source`，证据对象独立存 `read_status` 与 `reads[*]`（`actual_source`、`actual_input_sha256`、`origin_execution`）。
  - 所有 PDF/PPTX 裁区统一投影到旋转/裁边后的归一化坐标系 `visible_page_norm_v1`，确保消息裁图与阅读器“回原件”高亮严格对齐；框不可靠时降级 `parent_page_fallback`（`unreliable_bbox`、`reader_has_highlight=false`），宁给整页绝不画假框；同键渲染加进程内单航班锁防并发重复渲染。
  - 结果：D04 / G29 / K03 全程验证 `/media` 与 `/evidence/{eid}/preview` 字节 `SHA-256` 一致，K03 盲测 4 道多模态选图题（含同 Cell 双输出图与换版后 `v2` 图）达成 **`4/4 (100%)` 精准命中且 0 严重错误**。
- **源码/证据**：`app/rag/source_evidence.py`、`app/rag/source_vision.py`、`app/rag/workspace_preview.py`、`app/schemas/turn_evidence.py`。

### 故事 3：当巨型路由补丁互相打架时，如何做小范围强约束重构 + 盲审闭环真实工程深坑
- **考察点**：复杂状态机治理、重构节奏控制、深度 Debug 能力。
- **核心事实**：
  - **G29 重构**：早期 `workspace.py` 积累超过百条正则后，出现 `Q04` 焦点被泛化澄清抢跑、`caption/formula` `/evidence` 返回 409、重排比较 `b2` 单边 `partial`、部分 `/preview` 404 四处契约裂缝。主动停下补丁，拆出显式四阶段模块（`workspace_context.py`、`workspace_router.py`、`workspace_clarification.py`、`workspace_evidence.py`）并保持 HTTP/SQLite/前端零破坏，G29 盲审后 68 对象真实论文审计由 `77.9%/44.2%` 跃升至 **`94.1% Recall / 92.8% Precision`**。
  - **K03 8 轮盲审深坑案例（极佳面试细节！）**：在未见 *Graph-of-Thought* 盲测中，`K_Q07`（无文件名跨版对比 Notebook 文本输出与第 2 张图）连续暴露 4 层真实工程 Bug——① 候选未过滤只有 `v1` 的单版本文件；② 跨轮重复引用同一 `v1` 源未按 `(old_key, new_key)` 去重致误报 `comparison_ambiguous`；③ **nbformat 4.5 重存 Notebook 会随机生成新的 8 位 `cell_id`**，仅靠 `cell_id` 精确匹配直接失败（最终用 `cell_index` + `_cell_outputs_reliable` + `_cell_id_moved` 移位保护通用闭环）；④ 混合“函数调用/数值变化 + 第 2 张输出图”问法需自动挂载成对 `v1/v2` 原生 `stdout` 与视觉实读。全程零题面特判，盲测覆盖由 `2/8` 提升至 **`7/8` + `4/4` 视觉**，回归测试增至 **`697/0/0`**。
- **源码/证据**：`app/rag/workspace_versions.py`、`app/rag/workspace_objects.py`、`app/rag/workspace_router.py`、`tests/test_k02_r2_regressions.py`。

### 故事 4：不做“LLM 抽三元组黑盒”，如何落地可溯源、零误边的课程知识图谱
- **考察点**：GraphRAG / 知识图谱理性落地、防垃圾边、数据生命周期一致性。
- **核心事实**：
  - 明确单课程图谱只解决“概念 → 讲义定义 → 实验验证”两条高价值关系（`defined_in`、`demonstrated_by`），放弃易幻觉且难维护的 LLM 全量抽图，改在 `app/stores/local_kg.py` 实现**离线零模型确定性抽取**，硬约束每条边携带 12 字段证据且满足 `readable_source_text(source)[start:end] == evidence_excerpt`。
  - 针对真实课程设计 5 类反例硬拒收（仅共享术语、参考文献提及、未运行计划 Cell、代码注释/句中隔词否定、跨课同名），盲测达 **`10/10` 真边、`10/10` 反例 0 误收**。
  - 解决版本生命周期一致性：换版、`delete_local`（`BEGIN IMMEDIATE`）与 `promote_due` 到期提升通过 `ensure_course_kg_synced()` 保证旧边即时失效；并基于同预算实测把**导航功能（默认开）**与**问答一跳增强（默认关、显式可选，因 6 槽挤占权衡）**拆开决策（`H03` 零重叠跨文件 `2/4 → 4/4`、`K03 K_Q03` `8/10 → 10/10` 且 0 额外外呼）。
- **源码/证据**：`app/stores/local_kg.py`、`app/rag/workspace_kg.py`、[H03 独立审核报告](./H/H-gate/H03-独立审核报告.md)。

### 故事 5：如何搭建高可信 AI 评测与独立盲审体系（特别适配 AI 评测工程师 / Agent 评测岗位）
- **考察点**：评测集防泄漏、评测口径设计、AI 评分 vs 人工复核分歧治理、自动化回归。
- **核心事实**：
  - **执行与审核物理隔离**：每个大阶段（G29/G30/J/H/K）均由独立 gate 子代理在实现/复测**开始前**预冻结未见语料、题目与 JSON 判据，实现完成后从干净 `git archive` 在全新空库复跑，彻底杜绝“拿测试集调参”。
  - **多维拆解而非只看端到端通过率**：把失败拆到首次断点（范围/指代 → 候选召回 → 对象选中 → 视觉/文本实读 → 统一预算 → 生成引用），四维分评（事实与公式、引用支持、问题覆盖、对话一致性）叠加独立“严重错误”一票否决红线。
  - **AI 评分校准与人工复核（`R01–R03`）**：针对历史 53 轮建立逐题原件映射与预览包，明确 `8` 道重点争议题人工逐题核原件、`45` 道未看题显式标为 `未看-范围外`（绝不混作同意 AI），实测发现 `7/8` 同意 AI 扣分、1 例分歧（`旧 22`：AI 复审因后台日志与末尾免责口径冲突改扣 1 分，而真实学生视角答卷完整诚实应满分），沉淀出清晰的人评/自动评分边界。
  - **不强行刷分，双轨处理口径冲突**：G30 保留 A03 原锁标尺（`4/10 → 6/10`）配对真实网页 E2E（`8/8`），K03 诚实暴露 `C4` 默认 `S0` 固定 30 轮 `18/30`（来源 `93.8%`、0 严重错误）并以架构豁免单列，不靠放宽门槛或暗改规则制造假繁荣。
- **源码/证据**：[G20 证据断点诊断](./G/G20-D-诊断汇总与独立裁决.md)、[R01–R03 人工复核](./R/R02-人工复核汇总.md)、[K02 整机质量报告](./K/K02-交付包与整机质量报告.md)、[K03 独立终审报告](./K/K-gate/K03-独立终审报告.md)。

### 故事 6：预算一致性与双语数值守卫——两个“微小但致命”的生产级细节
- **考察点**：线上稳定性、边界条件思维、确定性护栏设计。
- **核心事实**：
  - **G21 0 字节预算误差**：排查出 `Q01` 证据全读完却 `0` 次调用 DeepSeek 的真因是读后二次 JSON 转义膨胀超 `32,768` 预检线；重构为阅读窗与生成预检共享同一最终 JSON 序列化字节核算并动态收缩末段，误差降至 `0` 字节。
  - **G23 `Decimal` 数量等价校验**：双语检索改写中，中文“五千四百亿参数”被译为 `540-billion-parameter` 时旧正则按纯字面数字比对误拦；改用 `Decimal` 解析中英文数词/量级（`万/亿/K/M/B/T`）、单位、百分数、比较方向与分句绑定，同时显式拦截模糊数词与量级错译（如把 540B 错译为 540M），冻结正反例 **`40/40`** 零误判。
- **源码/证据**：`app/rag/workspace_reading.py`、`app/rag/workspace_generation.py`、`app/rag/workspace_retrieval.py`、[G21 报告](./G/G21-阅读与生成预算一致性.md)、[G23 独立裁决](./G/G23-独立裁决.md)。

---

## 七、快速本地验证与关键文档导航

### 7.1 本地启动与回归命令（已实测可用）
```bash
# 1. 全量 0-skip 自动化回归（697 passed, 0 failed, 0 skipped）
SRAG_SOFFICE=/opt/homebrew/bin/soffice .venv/bin/python -m pytest -q

# 2. 启动本地完整工作区（零 Docker，复用 .env 中 DeepSeek 与 DashScope 密钥）
mkdir -p ./data/local-try
DATA_DIR="$PWD/data/local-try" \
SQLITE_PATH="$PWD/data/local-try/srag.sqlite3" \
WORKSPACE_COURSE_IDS=demo-course,my-course \
WORKSPACE_ANSWER_ENABLED=true \
SOURCE_VISION_ENABLED=true \
WORKSPACE_RETRIEVAL_STRATEGY=s0 \
WORKSPACE_RETRIEVAL_BUDGET=6 \
SRAG_SOFFICE=/opt/homebrew/bin/soffice \
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
# 浏览器打开 http://127.0.0.1:8000/workspace/
```

### 7.2 权威参考文档索引
- 项目入口与运行手册：[README.md](../README.md) · [.env.example](../.env.example)
- 完整架构与演进设计：[RAG.md](../RAG.md) · [产品目标与交付路线.md](./产品目标与交付路线.md)
- 任务放行与逐项日志：[CHECKLIST.md](../CHECKLIST.md) · [PROGRESS_HISTORY.md](../PROGRESS_HISTORY.md) · [执行交接.md](./执行交接.md)
- 关键阶段终审报告：[G29 终审](./G/G29-gate/G29-独立终审报告.md) · [G30 双轨](./G/G30-web-e2e/README.md) · [J03 策略定型](./J/J-gate/J03-独立审核与策略定型报告.md) · [H03 图谱盲验](./H/H-gate/H03-独立审核报告.md) · [R02/R03 人工复核](./R/R03-独立审核.md) · [K02 整机报告](./K/K02-交付包与整机质量报告.md) · [K03 终审](./K/K-gate/K03-独立终审报告.md)
