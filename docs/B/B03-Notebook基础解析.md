# B03 Notebook 基础解析

## 结论

实现已完成，当前状态为**待独立审核**。本项只新增原始 JSON 只读解析器、专用测试和本证据；未修改旧 `NotebookParser`、共享发布模型、A03 原件或治理台账。

工作区基线 HEAD 为 `33fad8ea339222574780d56ab401d0ed779369a8`。待审实现文件 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `app/rag/parsers/local_notebook.py` | `07f58d1372840a1850cb1d6cf5fc6bc4f2e4fb4e7888612a3d0bfe3461d1df41` |
| `tests/test_local_notebook.py` | `b761a1eab84d8b8c63b91f82ba86ab8cc3934b6218155e1b2c28281300137661` |

## 解析契约

- `parse_local_notebook(Path)` 直接读取 UTF-8 JSON，不调用内核、不执行代码、不读取 Markdown 链接或附件，也不发出网络请求。
- Cell 来源 ID 固定为 `cell:<原始零基索引>`；输出来源 ID 固定为 `cell:<i>:output:<原始零基索引>`。输出紧跟父 Cell 分配全局 `ordinal`，保留原始 Cell 与 outputs 数组顺序，包括 stream、空输出和相同字节的重复图。
- 原件 Cell ID 只在匹配 `[A-Za-z0-9_-]{1,64}` 且全文件唯一时写入 locator。缺失、无效或重复 ID 均记为 `null`，来源身份继续按原始索引稳定定位；重复 ID 的两个出现都不会被信任。
- Markdown 的 ATX/Setext 标题更新 `section_path`；围栏代码内的伪标题不参与章节。Cell 原文完整保留，代码只作为文本。
- 代码 Cell 的 `content_state` 与 `outputs_state` 分开记录。后者区分 `missing`、`empty`、`available` 和 `malformed`；非代码 Cell 非法携带 outputs 时记 `unexpected`，不生成违反父键契约的输出来源。
- 每个输出数组项都生成来源。stream、error、Markdown/HTML/plain/JSON MIME 形成可读文字；空项为 `empty`，不可解析项为 `malformed`，有效文字与损坏媒体并存时为 `partial`。
- MIME bundle 的每种表示记录 MIME、RFC 6901 JSON Pointer、编码、字节长度与 SHA-256。位图只接受可严格校验的 base64，SVG 按 UTF-8；空媒体为 `empty`，错误媒体为 `malformed`。不复制媒体资产，后续读取按同版原件及 pointer 解码。

## 实测结果

专用用例覆盖：原始顺序、Markdown 标题、代码文本、stream、空输出、同一 Cell 多个独立输出、一个 bundle 多种图像表示、重复图字节、旧式无 Cell ID、重复/非法 ID、缺失 outputs、空代码、损坏 base64、非字符串 output type、非代码 outputs、代码不执行、Markdown 本地/远程资源不解析，以及无效 JSON。

A03 锁定原件保持不变：

| 原件 | 文件 SHA-256 | 第二张输出图 SHA-256 | 定位 |
| --- | --- | --- | --- |
| `docs/A/样例/Signal-lab.ipynb` | `8c4808a38493583f6637b80cd0b30ab0d1a4dba933bb9b4306da7e5933f0f2d6` | `ce741007281a939680e7a21020fc80c34abfcb10d6d76d13e7984bc9d5743aa5` | `cell:1:output:2`，`cell_id=signal-plots` |
| `docs/A/样例/revisions/v2/Signal-lab.ipynb` | `ec7470dfd0ef0e1231cb88bb218e8d4f90dc5e5cf11917074af8314168dc3b12` | `7ec65dab6628a30590421e4c1a150c52a6ec22933d67efce47eda279097fbd68` | `cell:1:output:2`，`cell_id=signal-plots` |

上述文件哈希分别与 `docs/A/A03-lock-manifest.json`、`docs/A/A03-question-lock.json` 和 `docs/B/接手版本.json` 一致。

## 复现命令

```bash
.venv/bin/pytest -q tests/test_local_notebook.py
# 8 passed

.venv/bin/pytest -q tests/test_notebook_parser.py tests/test_local_notebook.py
# 10 passed

.venv/bin/ruff check app/rag/parsers/local_notebook.py tests/test_local_notebook.py
# All checks passed!

.venv/bin/ruff format --check app/rag/parsers/local_notebook.py tests/test_local_notebook.py
# 2 files already formatted

git diff --check -- app/rag/parsers/local_notebook.py tests/test_local_notebook.py
sha256sum docs/A/样例/Signal-lab.ipynb docs/A/样例/revisions/v2/Signal-lab.ipynb
```

## 边界与交接

- 本项只产生 `ParsedSource`，不写 SQLite、FTS5 或文件资产；发布、版本可见性和原件回读由 B04 接线验证。
- Notebook attachment 和 Markdown 外部资源不属于 B03 基础已保存输出，不会在导入时打开。若后续需要纳入来源，须另立来源类型与安全边界。
- 独立审核应在上述两个实现哈希上复跑专用测试，抽查 A03 v1/v2 的 `cell:1:output:2` pointer 与原始 JSON 字节，并确认解析前后原件哈希不变。
