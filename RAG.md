# SRAG 架构与方法

运行方式见 [README](./README.md)，评估见 [evals/README](./evals/README.md)，开发任务见 [CHECKLIST](./CHECKLIST.md)。

## 架构

```text
课程文件 → 解析/切块 → Qwen Embedding → Qdrant
                       文档/版本/正文 → SQLite
用户问题 → 任务与范围定位 → 召回或全文读取 → 视觉文字补充 → DeepSeek → 答案/来源
```

FastAPI、Qdrant、SQLite 本地运行；PDF/PPTX 可在 Colab GPU 预处理。模型均走 API，Colab 不直接写本地向量库。

| 目录/模块 | 职责 |
| --- | --- |
| `app/api/`、`app/schemas/` | HTTP 入口和数据契约；当前仅索引/检索，无对话 API |
| `app/rag/parsers/`、`chunker.py` | Notebook 解析与 Cell Group 切块 |
| `slide_preprocessor.py`、`slide_chunker.py`、`slide_assets.py` | PDF/PPTX 页级预处理、切块和资源分类 |
| `indexer.py`、`slide_embedder.py`、`slide_indexer.py` | 向量生成、便携向量包与入库 |
| `query.py`、`retriever.py`、`fusion.py` | 检索 API query 构造、文本/视觉召回与融合 |
| `dialogue.py`、`chat.py`、`summary.py`、`vision.py` | 终端任务定位、会话、全文总结和视觉转写 |
| `app/clients/`、`app/stores/` | 模型 API、Qdrant 与 SQLite 访问 |
| `tools/`、`notebooks/` | 命令行工具与交互实验 |
| `app/evaluation/`、`evals/`、`tests/` | 指标实现、效果评估和代码回归测试 |

表中未带目录的文件均位于 `app/rag/`。

## 资料与存储

| 输入 | 切块策略 | 表示 |
| --- | --- | --- |
| IPYNB | 标题划章节，连续 cell 形成 Cell Group，代码关联输出；约120/280/450 token软下限/目标/硬上限，超长cell重叠40 token | 文本dense+sparse，必要的视觉表示；不重复嵌入父块正文 |
| PDF/PPTX | 一页/一张为基础，超长页再拆；原生文字优先，稀疏页OCR补充；保留页码、整页截图和资源 | 标题/正文文本向量 + 全部整页视觉向量；装饰资源过滤不删除整页表示 |

- 文本：`qwen3.7-text-embedding`；视觉：`qwen3-vl-embedding`，使用独立命名向量。
- SQLite 保存正文、资源、内容哈希和文档版本；检索只读取当前可见、已发布且未删除的版本。
- Qdrant payload 保存来源标识，原文件/图片留本地；同内容导入跳过，更新时切换版本并清理旧向量。
- 换切块策略、模型或维度需重算对应表示，不混用旧向量。

## 检索与对话边界

| 环节 | 当前方法 |
| --- | --- |
| HTTP query | 默认raw；显式contextual只用最多6条历史中的最近用户问题补全缺主题指代，助手回答不参与 |
| 文本检索 | 课程/文件和可见版本过滤 → Dense+Sparse候选 → RRF |
| 视觉检索 | auto对明确视觉问题优先视觉候选、文本补足；失败记录警告后降级文本，显式visual失败报错 |
| 终端定位 | 默认全库；课程ID/文件名形成本轮硬范围，中文/数字页码直读，歧义澄清；不扩大用户范围 |
| 追问 | 保留近期知识主题、回答短术语与确认文件；显式换题解除旧偏好，每轮重新读取证据 |
| 生成上下文 | 最多6条本轮证据、最近两轮问答；历史不是事实证据，范围切换清空历史，无长期记忆/历史压缩 |
| 特殊任务 | 历史/目录直读；单文件总结读取全部文字块，约14000字符分批归纳，保留来源编号和覆盖统计 |

正式路径不默认使用 Outline、Rerank、生成式改写、HyDE、step-back或多查询；取舍依据保存在评估文档。

## 视觉文字与回答

Embedding 不生成文字。`VisualTextReader` 用 `qwen3-vl-plus` 读取注册截图，转写公式、符号、图表及不确定项；普通问答最多两页，总结只复用已有缓存。

缓存键包含文档版本、图片哈希、模型和提示词版本（v3）。转写另做dense+sparse派生点，新转写替换同来源旧派生点，原表示保留。回答预算优先视觉转写，失败显示警告并使用原生文字。

DeepSeek只接收文字，区分课件与通用补充；普通回答关闭思考，公式/矩阵问题及最终摘要增加一次low思考校对。校对证据/草稿上限14000/10000字符，预算8192 token，空输出/截断最多重试至16384；不展示推理。摘要校对只看分批笔记，校对不保证正确。引用编号/范围校验不等于事实核验，终端不渲染LaTeX。

尚不支持课程别名、相对页码、章节子范围、全课程复习规划、联网和网页上传；当前目标以课件范围内的知识讲解和来源定位为主。

## 资料快照

2026-10-01已入库15份文件，三门课的目录相对于`COURSE_ROOT`。这是历史清单，当前状态以目录接口为准。

| 课程目录 | 文件与版本 |
| --- | --- |
| `CS5187 Vison and Image/` | `CS5187_Vision_and_Image/Week01/Lecture1-1.pdf` v2；Week02 `Lecture2-1.pdf`、Week08 `Lecture8-2.pdf` v1；`Lecture/Week1/Lecture1-1.pptx` v2 |
| `CS5489 Machine learning/` | `Lecture1/Lecture1a.ipynb`、`Lecture1b.ipynb`；Lecture2 `Lecture2a.ipynb`、`Lecture2b.ipynb`；Lecture3 `Lecture3a.ipynb`均v1；Lecture4 `Lecture4a.ipynb` v2 |
| `CS5495 X AI/` | Lecture2 `Lecture2a.pdf`、`Lecture2b.pdf`；Lecture3 `Lecture3a.ipynb`、`Lecture3a.pdf`、`Lecture3b.pdf`均v1 |

## 维护约定

- 只记录当前架构、模块边界、方法/参数及取舍；已实现与实验能力明确区分。
- 实现变化更新对应章节，移除失效设计；运行命令、评估分数和任务状态引用对应文档。
- 资料清单注明快照日期，不冒充实时索引；接口和模块名与代码保持一致。
