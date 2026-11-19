#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

e03_python="${SRAG_PYTHON:-$PWD/.venv/bin/python}"
e03_port="${E03_PORT:-18803}"
e03_out="${E03_OUT:-$(mktemp -d /tmp/srag-e03-replay.XXXXXX)}"
mkdir -p "$e03_out"
e03_session="$(agent-browser session id --scope worktree --prefix "${E03_BROWSER_PREFIX:-srag-e03-replay}")"
e03_url="http://127.0.0.1:$e03_port/workspace/"

ab() { agent-browser --session "$e03_session" "$@"; }
click() { ab find role button click --name "$1" --exact; }
ask() {
  ab find label "输入问题" fill "$1"
  click "发送"
}
located() {
  export E03_LOCATION="$1"
  "$e03_python" -c 'import os,json; print("document.getElementById(\"location-title\").textContent === " + json.dumps(os.environ["E03_LOCATION"]))' > "$e03_out/location.js"
  ab wait --fn "$(cat "$e03_out/location.js")"
}

"$e03_python" docs/E/E03-fixture.py --port "$e03_port" --trace "$e03_out/trace.json"   > "$e03_out/server.log" 2>&1 &
e03_pid=$!
trap 'ab close >/dev/null 2>&1 || true; kill "$e03_pid" 2>/dev/null || true; wait "$e03_pid" 2>/dev/null || true' EXIT
curl --silent --show-error --fail --retry 8 --retry-connrefused --connect-timeout 1 "$e03_url" >/dev/null

ab open "$e03_url"
ab wait --text "暂无资料"
ab snapshot -i > "$e03_out/empty.txt"
ab find text "导入资料" click --exact
ab upload "#import-files" "$PWD/docs/A/样例/Signal-lab.ipynb"
click "开始导入"
ab wait --text "导入完成"

click "Signal-lab.ipynb v1"
ab wait --text "Cell 0 · lab-intro"
click "Cell 0 · lab-intro"
located "Cell 0 · lab-intro"
click "下一项"
located "Cell 1 · signal-plots"
click "返回上级"
located "Signal-lab.ipynb"
click "Cell 1 · signal-plots"
ab wait --text "输出 1"
click "输出 1"
located "输出 1"
click "打开原件"
ab wait --fn 'document.querySelector("#reader img")?.naturalWidth > 0'
click "放大"
click "适应窗口"
click "关闭原件"
click "返回上级"
located "Cell 1 · signal-plots"

ask "展示该 Cell 的两张图"
ab wait --fn 'Array.from(document.querySelectorAll(".source-card img")).length === 2 && Array.from(document.querySelectorAll(".source-card img")).every(img=>img.naturalWidth > 0)'
ab snapshot > "$e03_out/ordered-images.txt"
ab click '[data-image-ordinal="2"] .source-actions button:first-child'
ab wait --text "引用图 2"
ask "横轴单位？"
ab wait --text "已定位卡片 d2"
ask "刚才第二张图"
ab wait --text "已按消息显示顺序定位 d2"

ask "那张图"
ab wait --text "需要澄清"
ab reload
ab wait --text "需要澄清"
click "选择图 1"
click "发送"
ab wait --text "已确认澄清选项 option-1"
ask "失败演练"
ab wait --text "处理失败"
click "重试本轮"
ab wait --text "重试已完成"
ask "等待演练"
ab wait --text "正在处理，可取消本轮"
click "取消本轮"
ab wait --text "已取消"

ab find text "搜索" click --exact
ask "信号"
ab wait --text "检索结果"
ab find text "单文件总结" click --exact
click "发送"
ab wait --text "当前单文件总结"
click "新建会话"
ab wait --text "从资料库选择位置"
ab select "#sessions" "fixture-session-1"
ab wait --text "当前单文件总结"
ab snapshot > "$e03_out/restored.txt"

"$e03_python" - "$e03_out/trace.json" <<'PY'
import json
import sys
from pathlib import Path

events = json.loads(Path(sys.argv[1]).read_text())
turns = [event["payload"] for event in events if event["event"] == "turn"]
card = next(turn for turn in turns if turn["text"] == "横轴单位？")
plain = next(turn for turn in turns if turn["text"] == "刚才第二张图")
assert set(card["anchor"]) == {"type", "message_id", "display_id"}
assert card["anchor"]["type"] == "display" and card["anchor"]["display_id"] == "d2"
assert set(plain) == {"client_turn_id", "kind", "text", "scope_revision"}
assert any(turn.get("anchor", {}).get("option_id") == "option-1" for turn in turns)
assert {turn["kind"] for turn in turns} == {"ask", "search", "summary"}
upload = next(event for event in events if event["event"] == "import")
assert upload["content_type"] == "application/octet-stream"
sample = Path("docs/A/样例/Signal-lab.ipynb")
assert upload["filename"] == sample.name and upload["bytes"] == sample.stat().st_size
assert {"retry", "cancel", "preview", "focus", "scope"} <= {event["event"] for event in events}
print("E03 browser contract verified")
PY
printf '%s\n' "$e03_out"
