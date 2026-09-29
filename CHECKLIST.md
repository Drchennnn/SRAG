# SRAG 开发任务清单

## 已完成 ✅

### Notebook 正式链路
- [x] 只读解析、标题段与 Cell Group 自适应切块
- [x] Qwen 文本 dense+sparse 与视觉表示生成
- [x] Qdrant 命名向量、SQLite 版本生命周期和增量索引
- [x] `/api/rag/index`、`/api/rag/search` 与范围过滤
- [x] Dense+Sparse RRF 正式召回
- [x] 20 题评估集与消融实验；RRF Recall@8=1.00、MRR@8=0.893
- [x] 定位 Outline 融合问题并从正式入口禁用

### 已验证组件与原型
- [x] Qwen Rerank、DeepSeek 查询理解、visual fusion 和上下文预算组件
- [x] PDF/PPTX Parser 与 Chunker 初版原型
- [x] Colab T4 文档预处理 Notebook
- [x] 32 页幻灯片 PDF 实测；确认 Markdown 标题切分会破坏页内连续性，图片只保留占位符
- [x] 确定 PDF/PPTX 统一采用 SlideDeck 数据模型

---

## 当前阶段：统一幻灯片预处理 🚧

- [x] 定义 `SlideDeck`、`Slide`、`SlideAsset` 及稳定 ID、页码/幻灯片号、位置和相邻关系
- [x] PDF 改为原生文本优先、稀疏页 OCR 回退，并保存整页截图和图片/表格资源
- [x] PPTX 保留文本框、图片、表格和布局；可通过 LibreOffice 生成整页截图
- [x] 实现统一 `SlideChunker`：一页/一张为基础 Chunk，仅拆分超长页
- [x] 标注重复 Logo 和小型角落装饰；暂不逐图生成向量
- [x] 输出可回传本地的 `deck.json + chunks.json + slides/ + assets/` 预处理包
- [x] 锁定文档依赖，并建立无个人路径的 PDF/PPTX 自动化测试
- [ ] 用代表性 PDF 与 PPTX 复测文本连续性、资源溯源和处理耗时

## 下一阶段：Embedding 与索引

- [ ] 幻灯片文字调用 Qwen text dense+sparse Embedding
- [ ] 有视觉信息的整页截图调用 Qwen 多模态 Embedding
- [ ] 将 SlideDeck 接入 SQLite、Qdrant、文件类型分派和索引 API
- [ ] 建立 20–30 题幻灯片检索集，验证文本、视觉与融合召回

## 后续：完整检索与回答

- [ ] 接通可配置的 Rerank、DeepSeek 查询理解、视觉召回和相邻上下文
- [ ] 使用固定评估集记录每一步的精度与延迟收益
- [ ] 接入基于检索证据的 DeepSeek 回答与来源引用
- [ ] 增加文件上传、检索结果和来源追溯界面

## 延后事项

- 查询缓存、复杂监控、压力测试和云端迁移在本地链路稳定后再评估。
- Document Outline 仅保留为文档发现候选，不进入当前召回主线。

---

**最后更新**：2026-09-30

**当前重点**：先完成统一 SlideDeck 预处理并验证质量，再开始幻灯片 Embedding。
