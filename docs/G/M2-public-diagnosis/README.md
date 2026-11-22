# M2 公开演练的条件错配与换版焦点

执行者 Codex，独立工作树 `SRAG-m2-diag`，分支 `codex/m2-public-diagnosis`。用准备分支固定提交 `128bf5b646352f12018873687faaad10b31d1e06` 的新编 Basalt Latch 替身和设置工厂，对照 `7da7d059b998e67b59d27655b64c85778a6f2472` 与 `37a8b811f3a9b5be58aa6ea1c3e57e59df95a6cc` 两个干净产品树。第二个树复用第一轮生成并校验的同一批文件字节。只使用公开材料。诊断未改产品、门槛或正式流程。

## 盐雾条件

两版均从空库通过生产 ASGI TestClient 调用 `/api/workspace/sessions/*`。正式口径的对应设置为外部文字生成关、FTS、Jev 和向量出站关闭、定向视觉启用。每版三个问题在同一公开资料快照下执行，三题均没有模型出站，证据预算为 2。

| 问法 | 两版实际结果 |
| --- | --- |
| 原演练的留出集正例 | 讲义定义和留出集实测均获支持。打开 Notebook Cell 1 可核对样本数 4、MAE 0.26 → 0.19。 |
| 原演练的盐雾问题 | 同样返回 `experiment=supported`、`answer_status=complete`，实验正文和正例逐字相同。虽然正文写了 `held-out` 标签，却未指出盐雾未获证实，也未把数值作为明确分开的对照。按目标条件判，这是错用。 |
| 已识别的条件反例，改问测试集 | 保留定义与原文引用，实验 `missing`，不返回留出集数值。 |

**第一处分歧在实验条件判定。** 两版均从已读讲义的 `SlateTrialB6` 获得字面锚，补查并实际读取到正确的留出集 Notebook 行。诊断观察器仅记录真实 `_condition_problem` 的入参与返回值。补查记录的字段为 `split=held-out`、`metric=MAE`。留出集和盐雾问法都返回无拒收，测试集问法返回 `experiment_split_mismatch`。现有校验只对识别出来的切分和指标做比较，锚点相同且未知实验条件未核对时，记录仍能作为事实通过。问题位于产品的本地证据支撑判定，早于答复拼装。测值和引用原件是真实的公开替身记录，错在对应了用户所问的条件。 [条件校验](/Users/Link/code/SRAG-m2-diag/app/rag/evidence_supplement.py:803)

**正式验收影响与建议。** 如果正式题出现类似条件，按适用维度核事实、引用支持与缺证据边界。两者只有在明确说明“盐雾缺证”，并把留出集的数值独立标为对照时，才构成对照答复。公开诊断不足以预判正式样本的分数或严重错误裁决。建议把此三问作为产品最小回归，在条件支撑阶段识别未核对的请求条件。未有对应观测时保留定义、说明缺证，已有结果继续归属其真实条件。

## v2 发布与焦点

用同一 Notebook v1/v2 分别做裸 API 对照和 Chromium 真实目录操作。两版行为一致。ASGI 每版有两次定向 Qwen 实际出站，均为 `qwen3-vl-plus` HTTP 200，20 秒设置。网页只展示原件，用记录到的真实 HTTP 变更验证是否重绑。

| 操作 | 两版实际结果与合同 |
| --- | --- |
| 裸 API 发布 v2 | `POST /import` 正常发布 v2，文件 ID 与已选范围保持稳定。数据库焦点仍为 v1 根节点，`GET session` 投影 `focus=null`，目录已有新根节点。这是裸 API 序列上第一个可见分歧。 |
| 发布后直接正常提问 | `POST /turns` HTTP 200，内层 `failed / stale_reference`、可重试，候选和模型实读均未开始。网页若遇此错误，将提示资料版本变化并请重新选择。显式 `PUT /focus` 指向退役节点返回 404。 |
| 发布后图序 | 明确说“刚才”保留历史对象，返回 `historical_only` 提示，无新增旧图实读。旧引用仍为 v1。裸“第二张图”遇到当前焦点与旧展示冲突，走澄清。 |
| API 显式重绑后继续 | 按 `GET catalog` 的当前根节点 `PUT /focus` 返回 200、范围修订加一。下一问展示 v2，再问刚才第二张图读到 `mm`。v1 实读 `Voltage [mV]`，换版后仍能以原引用预览相同字节。新旧实际发送的图片各自与对应预览 SHA 一致。 |
| Chromium 正常上传替换 | 实际发出 `POST /import(v2) → PUT /focus(新根节点) → POST /turns`。无须用户重新点击目录，首问完成，两个可见图片对象的版号均为 2。 |

G19 明确验证的是网页换版后自动重绑和已退役节点拒绝。G07 区分当前、历史和冲突指代。生产 API 的发布动作与焦点更新分开，护栏阻止陈旧焦点被当作当前证据。原公开演练用裸 API 发布后直接提问，省略了网页实际执行的重绑。随后用“刚才”追问，还在沿用上一条成功展示的旧图。脚本因此未演练到 v2 读图。API 本身也确实保留了裸发布后的退役焦点，这是必须明确的行为边界。 [G19](../G19-独立整审与试用.md) · [G07](../G07-图序与版本.md)

**正式验收影响与建议。** 这条裸 API 失败不能用来判网页换版失败，也不能笼统记为成功的 v2 轮。主审应先让获授权执行员核对锁定换版步骤是否包含目录刷新与焦点重绑，另设网页等价性的公开回归。若正式步骤要求裸 API 发布后继续，保留该差异交主审裁定，避免临场为个别题添加步骤。后续把裸发布拒绝、历史回看、冲突澄清与重绑后读 v2 分项记录。未启动正式复测。

## 公开复跑

```sh
PYTHON=/Users/Link/code/SRAG/.venv/bin/python
ENV_FILE=/Users/Link/code/SRAG/.env
trial="$(mktemp -d)"
git worktree add --detach "$trial/prep" 128bf5b646352f12018873687faaad10b31d1e06
git worktree add --detach "$trial/old" 7da7d059b998e67b59d27655b64c85778a6f2472
git worktree add --detach "$trial/new" 37a8b811f3a9b5be58aa6ea1c3e57e59df95a6cc
prep="$trial/prep/docs/G/M2-retest-prep/rehearse.py"
probe=docs/G/M2-public-diagnosis/diagnose.py
"$PYTHON" "$probe" --subject "$trial/old" --prep "$prep" --env-file "$ENV_FILE" \
  --out "$trial/old-run"
"$PYTHON" "$probe" --subject "$trial/new" --prep "$prep" --env-file "$ENV_FILE" \
  --fixtures-from "$trial/old-run" --out "$trial/new-run"
```

仅复跑条件三问时加 `--conditions-only`，无模型调用，也不启动 Chromium。浏览器序列只展示两张已保存的图，实际视觉模型读取由 ASGI 回放单独验证。真实响应、版本及行为摘要见[证据](./证据.json)。
