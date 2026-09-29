# SRAG 开发任务清单

## 已完成 ✅

### 核心功能
- [x] Notebook 解析与结构化切块（Markdown 标题段 + Cell Group）
- [x] 文本 embedding（Qwen dense + sparse）
- [x] 视觉 embedding 与入库（Qwen multimodal fusion）
- [x] Qdrant 文本向量索引与召回（text_dense、text_sparse）
- [x] Qdrant visual_dense 存取接口
- [x] SQLite 元数据管理（版本、发布日期、资源）
- [x] 索引 API（`POST /api/rag/index`）
- [x] 检索 API（`POST /api/rag/search`）

### 正式检索链路
- [x] Dense + Sparse RRF 融合并接入 `/api/rag/search`
- [x] 课程、文件、版本和发布日期过滤
- [x] Document outline 召回实验与根因分析；正式入口暂时禁用

### 已完成的实验组件
- [x] Qwen Rerank 客户端与独立消融实验
- [x] DeepSeek 查询理解客户端
- [x] Visual fusion 召回方法
- [x] 上下文预算与 chunk 排序工具

### 评估
- [x] 20 题 Notebook 评估集（15 开发 + 5 保留）
- [x] Dense baseline（Recall@8=0.95, MRR=0.875）
- [x] 4 种策略消融实验（Dense、RRF、RRF+Outline、RRF+Outline+Rerank）
- [x] 确认 RRF 为当前正式入口默认策略
- [x] 验证 RRF+Outline+Rerank 实验路径的精度上限
- [x] 分析并禁用 Outline 召回

---

## 进行中 🚧

### 完整 Notebook 检索链路
- [ ] 修复正式入口的候选保留逻辑，接入可配置的 Qwen Rerank
- [ ] 接入可配置的 DeepSeek 查询理解；始终保留原始 query
- [ ] 将 visual_dense 召回与文本候选融合，并建立图表题评估
- [ ] 补齐相邻 chunk ID 与版本溯源，接入上下文扩展和预算控制
- [ ] 通过 `/api/rag/search` 重新运行固定评估集，记录各步骤增量收益

### PDF/PPTX 支持
- [x] 实现 PDFParser/PDFChunker 原型（Docling Markdown 文本路径）
- [x] 实现 PPTXParser/PPTXChunker 原型（一张幻灯片一个 chunk）
- [ ] 在 `pyproject.toml`/lockfile 声明并锁定 Docling、python-pptx
- [ ] 建立无个人绝对路径、可快速重复的解析与切块测试
- [ ] 补齐 PDF 页码、表格/图片资源和 PPTX 幻灯片号、表格/图片溯源
- [ ] 修复跨文档稳定 section/chunk/representation ID
- [ ] 统一 Parser/Chunker Protocol，并按文件类型分派
- [ ] 集成到 SQLite、Qdrant 和索引 API

### 评估扩展
- [ ] 准备 PDF/PPT 评估集（20-30 题）
- [ ] 扩展 Notebook 评估集到 60 题

---

## 待完成 📋

### 短期

**PDF/PPTX 完善**
- [ ] 优化 PDF 跨页内容处理
- [ ] 实现图表与文字关联
- [ ] 支持 PDF 表格解析
- [ ] 评估 PDF 检索效果（目标 Recall@8 ≥ 0.90）

**检索优化**
- [ ] 实现查询结果缓存（本地文件/内存）
- [ ] 优化 Rerank 延迟（批量调用、top-N 过滤）
- [ ] 实现基础查询日志（延迟、命中情况）

**评估完善**
- [ ] 实现 LLM-as-Judge 自动评估（Context Precision/Recall）
- [ ] 完成 80-100 题评估集（Notebook + PDF/PPT）
- [ ] 增加难度分层和多模态查询

### 中期

**高级功能**
- [ ] 重新设计 Document Outline（分离 pipeline，仅用于文档发现）

**回答生成**
- [ ] 扩展 DeepSeek 客户端并接入回答生成
- [ ] 实现基于检索证据的回答生成
- [ ] 评估回答质量（Faithfulness、Relevance）

### 未来规划

**Web 界面**
- [ ] 文件上传接口
- [ ] 检索结果展示
- [ ] 来源追溯可视化

**云端部署（可选）**
- [ ] 评估 Qdrant Cloud 方案
- [ ] 对象存储集成
- [ ] 服务部署与运维

---

## 已放弃 ❌

- ~~在线监控和用户满意度反馈~~（过度工程化）
- ~~压力测试（并发 QPS、大规模索引）~~（过度工程化）
- ~~A/B 测试框架~~（过度工程化）
- ~~复杂的限流、降级策略~~（过度工程化）

---

**最后更新**：2026-09-29  
**当前重点**：先完成 Notebook 正式检索链路，再稳定并接入 PDF/PPTX
