# SRAG 开发任务清单

## 已完成 ✅

### 核心功能
- [x] Notebook 解析与结构化切块（Markdown 标题段 + Cell Group）
- [x] 文本 embedding（Qwen dense + sparse）
- [x] 视觉 embedding（Qwen multimodal fusion）
- [x] Qdrant 向量索引与召回（text_dense、text_sparse、visual_dense）
- [x] SQLite 元数据管理（版本、发布日期、资源）
- [x] 索引 API（`POST /api/rag/index`）
- [x] 检索 API（`POST /api/rag/search`）

### 检索增强
- [x] Dense + Sparse RRF 融合
- [x] Qwen Rerank 重排
- [x] DeepSeek 查询理解（可选）
- [x] Document outline 召回（已实现但暂时禁用）
- [x] 上下文回取与预算管理

### 评估
- [x] 20 题 Notebook 评估集（15 开发 + 5 保留）
- [x] Dense baseline（Recall@8=0.95, MRR=0.875）
- [x] 4 种策略消融实验（Dense、RRF、RRF+Outline、RRF+Rerank）
- [x] 识别最优策略：RRF（默认）、RRF+Rerank（高精度）
- [x] 分析并禁用 Outline 召回

---

## 进行中 🚧

### PDF/PPT 支持
- [x] 设计 PDF/PPT 解析与切块方案
- [x] 实现 PDFParser（Docling）
- [x] 实现 PDFChunker（语义段落切分）
- [x] 实现 PPTXParser（python-pptx）
- [x] 实现 PPTXChunker（一张幻灯片 = 一个 chunk）
- [ ] 验证 PDF/PPTX 切分质量（测试转移到 Kaggle）
- [ ] 统一 Parser/Chunker Protocol
- [ ] 集成到索引 API

### 评估扩展
- [ ] 准备 PDF/PPT 评估集（20-30 题）
- [ ] 扩展 Notebook 评估集到 60 题

---

## 待完成 📋

### 短期（1-2 个月）

**PDF/PPT 完善**
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

### 中期（2-3 个月）

**高级功能**
- [ ] 验证视觉召回效果（图表理解题）
- [ ] 集成上下文回取到实际检索流程
- [ ] 重新设计 Document Outline（分离 pipeline，仅用于文档发现）

**回答生成**
- [ ] 接入 DeepSeek Chat API
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
**当前重点**：PDF/PPT 解析与切块
