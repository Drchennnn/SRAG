# M2 公开替身的实验条件支撑修复

独立开发分支 `codex/m2-condition-fix`，基线 `695dd9c67c1f3847f380d53831e0c57e125449ac` 包含公开故障诊断。两个交付分支的共同产品基线为 `37a8b811f3a9b5be58aa6ea1c3e57e59df95a6cc`。只用 `128bf5b646352f12018873687faaad10b31d1e06` 的公开替身生成器和正式公开配置。未启动 M2/A03，v2 裸 API 与网页的差异沿用既有诊断记录。

## 条件校验

原 F 本地证据路径能核对论文定义与同实验 ID、原生读记录、已知数据切分和指标。遇到实验请求里无法识别的其他对象或条件时，旧路径没有把它们当成未解决项，即使当前读到的实验属于另一条件也会记为 supported。

修复在 F 首读、补查及最终事实复核共用同一校验。现有原文、对象 ID、切分和指标校验先运行。随后以明确的定义话题、实际记录的字段及受支持的切分或指标别名核对实验问法。仍有未支持限定或对象时，实验支撑记为缺项，答复说明当前原文不足以确认所问对象、条件和指标。该记录不作为实验事实输出。校验没有盐雾或低温等场景词表。原有 F 证据预算 2 保留，组合证据流程继续使用它自己的结构化判定。

校验是保守的确定性规则。当前 F 原生记录只提供有限字段，陌生但能与原文记录的切分或指标精确匹配的名称仍可通过。其他语义等价的生僻表述、未结构化的条件等可能按缺证处理。

## 首跑与修后实测

两版分别从空 SQLite、原件与缓存走生产 ASGI API。在两个空库使用同一批经 SHA-256 固定的替身文件、同一课程范围、FTS，关闭外部文字生成、Jev 和向量出站，视觉启用但用例均为文字。6 个用例在两版均 HTTP 200，零模型出站。 [首次和修后正文、状态、原文引用](./证据.json)

| 目标 | 修前实验项 | 修后实验项和引用 |
| --- | --- | --- |
| 留出集 MAE | supported | supported。定义保留讲义 v1 第 1 页，实验保留 Notebook v1 Cell 1、4 个样本、0.26 → 0.19。 |
| 盐雾 | supported，沿用留出集数值 | missing，明确缺证，只有讲义引用。 |
| 测试集 | missing | missing，切分不匹配；只有讲义引用。 |
| 低温循环后的留出集 | supported，误用无该条件记录 | missing，只有讲义引用。 |
| RMSE | missing | missing，指标不匹配；只有讲义引用。 |
| 改问 Quartz latch 实验 | supported，误用 Basalt latch 实验 | missing，只有原问题中 Basalt latch 定义的讲义引用。 |

定向 pytest 125 通过、1 跳过。全量 pytest 632 通过、3 跳过。Ruff 通过。与 G28 的 `b142661ac6b64c3ee1eba74d4da511444ea08184` 在共同基线做 Git 合成检查无冲突，合成临时归档上 12 项相关回归通过。合成只用于验证。

## 公开复跑

```sh
PYTHON=/Users/Link/code/SRAG/.venv/bin/python
ENV_FILE=/Users/Link/code/SRAG/.env
trial="$(mktemp -d)"
git worktree add --detach "$trial/prep" 128bf5b646352f12018873687faaad10b31d1e06
git worktree add --detach "$trial/base" 695dd9c67c1f3847f380d53831e0c57e125449ac
runner=docs/G/M2-condition-fix/replay.py
prep="$trial/prep/docs/G/M2-retest-prep/rehearse.py"
"$PYTHON" "$runner" --subject "$trial/base" --prep "$prep" --env-file "$ENV_FILE" \
  --out "$trial/first"
"$PYTHON" "$runner" --subject "$PWD" --prep "$prep" --env-file "$ENV_FILE" \
  --fixtures-from "$trial/first" --out "$trial/fixed"
"$PYTHON" -m pytest -q
```
