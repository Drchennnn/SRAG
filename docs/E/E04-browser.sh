#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

e04_python="${SRAG_PYTHON:-$PWD/.venv/bin/python}"
e04_port="${E04_PORT:-18814}"
e04_out="${E04_OUT:-$(mktemp -d /tmp/srag-e04-browser.XXXXXX)}"
e04_soffice="${SRAG_SOFFICE:-$(command -v soffice || true)}"
test -x "$e04_soffice"
mkdir -p "$e04_out"
test ! -e "$e04_out/library.sqlite3"
e04_session="$(agent-browser session id --scope worktree --prefix "${E04_BROWSER_PREFIX:-srag-e04-replay}")"
e04_url="http://127.0.0.1:$e04_port/workspace/"

ab() { agent-browser --session "$e04_session" "$@"; }
click() { ab find role button click --name "$1" --exact; }
pause_recording() {
  if [ "${E04_RECORD:-0}" = 1 ]; then sleep 1; fi
}
ask() {
  ab find label "输入问题" fill "$1"
  click "发送"
}
upload() {
  ab select "#import-course" signals
  ab upload "#import-files" "$@"
  click "开始导入"
}

args=(docs/E/E04-server.py --data-dir "$e04_out" --trace "$e04_out/browser-trace.jsonl"
  --port "$e04_port" --soffice "$e04_soffice")
if [ "${E04_REAL_VISION:-0}" = 1 ]; then args+=(--real-vision); fi
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 "$e04_python" "${args[@]}" > "$e04_out/server.log" 2>&1 &
e04_pid=$!
trap 'ab close >/dev/null 2>&1 || true; kill "$e04_pid" 2>/dev/null || true; wait "$e04_pid" 2>/dev/null || true' EXIT
curl --silent --show-error --fail --retry 7 --retry-connrefused --connect-timeout 1 "$e04_url" >/dev/null

ab open "$e04_url"
ab set viewport 1320 860
ab wait --text "暂无资料"
if [ "${E04_RECORD:-0}" = 1 ]; then ab record start "$e04_out/m1.mp4" --cursor --fps 15; fi
pause_recording
ab find text "导入资料" click --exact
upload "$PWD/docs/A/样例/Signal-lab.ipynb"
ab wait --text "导入完成"
pause_recording
click "Signal-lab.ipynb v1"
ab wait --fn 'document.querySelector("#location-title").textContent === "Signal-lab.ipynb"'
click "Median filter lab"
ab wait --fn 'document.querySelector("#location-title").textContent === "Median filter lab"'
click "Cell 1 · Median filter lab"
ab wait --fn 'document.querySelector("#location-title").textContent.startsWith("Cell 1")'
click "下一项"
ab wait --fn 'document.querySelector("#location-title").textContent.startsWith("Cell 2")'
pause_recording
click "返回上级"
ab wait --fn 'document.querySelector("#location-title").textContent === "Median filter lab"'
click "Cell 2 · Median filter lab"
ab wait --fn 'document.querySelector("#location-title").textContent.startsWith("Cell 2")'
ask "展示这个 Cell 保存的两张图"
ab wait --fn 'document.querySelectorAll(".source-card img").length === 2 && [...document.querySelectorAll(".source-card img")].every(i=>i.naturalWidth===440 && i.naturalHeight===260)'
ab wait --fn 'document.querySelector(".source-card .source-details").textContent.includes("Cell 2")'
ab scrollintoview '[data-image-ordinal="1"]'
ab screenshot "$e04_out/m1.png"
pause_recording
if [ "${E04_RECORD:-0}" = 1 ]; then ab record stop --json > "$e04_out/m1-record.json"; fi

if [ "${E04_RECORD:-0}" = 1 ]; then ab record start "$e04_out/followups.mp4" --cursor --fps 15; fi
ab click '[data-image-ordinal="2"] .source-actions button:nth-child(2)'
ab wait --fn 'document.querySelector("#reader img")?.naturalWidth===440 && document.querySelector("#reader img").src.includes("view=inline")'
click "关闭原件"
ab click '[data-image-ordinal="2"] .source-actions button:first-child'
ab wait --text "引用图 2"
pause_recording
ask "横轴单位是什么？"
ab wait --fn 'document.querySelectorAll("#messages .message:not(.message-user)").length===2'
pause_recording
ask "刚才第二张图"
ab wait --fn 'document.querySelectorAll("#messages .message:not(.message-user)").length===3'
ask "那张图"
ab wait --text "请选择你指的图片"
ab reload
ab wait --text "请选择你指的图片"
ab snapshot > "$e04_out/followups-restored.txt"
pause_recording
if [ "${E04_RECORD:-0}" = 1 ]; then ab record stop --json > "$e04_out/followups-record.json"; fi

ab find text "导入资料" click --exact
upload "$PWD/docs/A/样例/Signal-slides.pptx" "$PWD/docs/A/样例/Signal-reference.pdf"
ab wait --fn '[...document.querySelectorAll("#catalog button")].some(b=>b.textContent.includes("Signal-reference.pdf"))'
click "Signal-slides.pptx v1"
ab wait --fn 'document.querySelector("#location-title").textContent === "Signal-slides.pptx"'
click "第 1 页 · Median filter experiment"
ab wait --fn 'document.querySelector("#location-title").textContent.startsWith("第 1 页")'
ask "展示课件第 1 页原件"
ab wait --fn 'document.querySelectorAll("#messages .message:not(.message-user)").length===5 && [...document.querySelectorAll("#messages article:last-child img")].some(i=>i.naturalWidth===1440 && i.naturalHeight===720)'
click "表格"
ab wait --fn 'document.querySelector("#location-title").textContent === "表格"'
ask "展示这张表格"
ab wait --fn 'document.querySelectorAll("#messages .message:not(.message-user)").length===6 && [...document.querySelectorAll("#messages article:last-child img")].some(i=>i.naturalWidth===1210 && i.naturalHeight===462)'
ab scrollintoview '#messages article:last-child figure'
ab screenshot "$e04_out/pptx-table.png"

click "Signal-reference.pdf v1"
ab wait --fn 'document.querySelector("#location-title").textContent === "Signal-reference.pdf"'
click "第 2 页 · Impulse-spike illustration"
ab wait --fn 'document.querySelector("#location-title").textContent.startsWith("第 2 页")'
ask "展示论文右侧图"
ab wait --fn 'document.querySelectorAll("#messages .message:not(.message-user)").length===7 && [...document.querySelectorAll("#messages article:last-child img")].some(i=>i.naturalWidth===527 && i.naturalHeight===596)'
ab click '#messages article:last-child .source-actions button:nth-child(2)'
ab wait --fn 'document.querySelector("#reader img")?.naturalWidth===527'
click "放大"
click "适应窗口"
click "关闭原件"
ab click '#messages article:last-child .source-actions button:nth-child(3)'
ab wait --fn 'document.querySelector("#reader img")?.naturalWidth===1224 && !!document.querySelector("#reader .reader-highlight")'
ab screenshot "$e04_out/pdf-highlight.png"
click "关闭原件"
ab find text "单文件总结" click --exact
click "发送"
ab wait --fn '[...document.querySelectorAll("#messages .message:not(.message-user)")].length===8'
ab reload
ab wait --fn '[...document.querySelectorAll("#messages .message:not(.message-user)")].length===8'
ab snapshot > "$e04_out/restored.txt"

"$e04_python" - "$e04_out/browser-trace.jsonl" "${E04_REAL_VISION:-0}" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
by_text = {row["request"]["text"]: row for row in rows}
m1 = by_text["展示这个 Cell 保存的两张图"]["response"]["message"]
assert [block["display_id"] for block in m1["blocks"] if block["type"] == "source_image"] == m1["visible_display_ids"]
first, second = m1["visible_display_ids"]
assert first != second
card = by_text["横轴单位是什么？"]
plain = by_text["刚才第二张图"]
assert card["request"]["anchor"] == {"type": "display", "message_id": m1["message_id"], "display_id": second}
assert set(plain["request"]) == {"client_turn_id", "kind", "text", "scope_revision"}
for row, method in [(card, "display_anchor"), (plain, "visible_ordinal")]:
    turn = row["response"]["turn"]
    resolved = turn["resolution"]
    assert resolved["method"] == method and resolved["display_id"] == second
    assert resolved["ordinal"] == 2 and resolved["actual_source"]["source_id"] == "cell:1:output:2"
    assert turn["trace"]["external_call_count"] == int(sys.argv[2])
assert by_text["那张图"]["response"]["turn"]["status"] == "clarification"
for question, source_id, rendition in [
    ("展示课件第 1 页原件", "page:1", "original_page"),
    ("展示这张表格", "page:1:shape:3", "object_crop"),
    ("展示论文右侧图", "page:2", "object_crop"),
]:
    images = by_text[question]["response"]["message"]["blocks"]
    assert any(block.get("display_source", {}).get("source_id") == source_id
               and block["rendition_kind"] == rendition for block in images)
summary = by_text["总结当前资料"]["response"]["turn"]["trace"]
assert len(summary["effective_scope"]["document_ids"]) == 1
print("E04 browser verified")
PY
printf '%s\n' "$e04_out"
