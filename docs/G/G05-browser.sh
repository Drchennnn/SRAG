#!/usr/bin/env bash
# Use Chrome controls to send every question without a selected attachment.
set -euo pipefail
cd "$(dirname "$0")/../.."
python="${SRAG_PYTHON:-$PWD/.venv/bin/python}"
port="${G05_PORT:-18841}"
out="${G05_OUT:-$(mktemp -d /tmp/srag-g05-browser.XXXXXX)}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
test ! -e "$out/library.sqlite3"
session="$(agent-browser session id --scope worktree --prefix "${G05_BROWSER_PREFIX:-g05-diagnosis}")"
url="http://127.0.0.1:$port/workspace/"
ab() { agent-browser --session "$session" "$@"; }
click() { ab find role button click --name "$1" --exact; }
wait_js() { ab wait --fn "$1"; }
save() { ab eval "$2" > "$out/$1.json"; }

courses() {
  click "$1"
  wait_js "document.querySelector('#scope-summary').textContent.includes('$1') && document.querySelector('#scope-summary').textContent.includes('全部资料')"
}
ask() {
  local before
  before="$(ab get count '#messages article:not(.message-user)')"
  ab find label '输入问题' fill "$2"
  click '发送'
  wait_js "document.querySelectorAll('#messages article:not(.message-user)').length === ($before + 1)"
  save "$1" '(() => { const m=document.querySelector("#messages article:last-child"); return {message_id:m.dataset.messageId, text:m.textContent, displays:[...m.querySelectorAll("figure[data-image-ordinal]")].map(x=>({display_id:x.dataset.displayId, ordinal:Number(x.dataset.imageOrdinal), url:x.querySelector("img")?.getAttribute("src"), caption:x.querySelector("figcaption")?.textContent})), citations:[...m.querySelectorAll("[data-evidence-id]")].map(x=>x.textContent)} })()'
}
upload() {
  local course="$1" mode="$2" title="$3"
  shift 3
  wait_js '!document.querySelector("#import-button").disabled'
  if [ "$(ab eval 'document.querySelector("#import-panel").open')" = false ]; then
    ab find text '导入资料' click --exact
  fi
  ab select '#import-course' "$course"
  ab upload '#import-files' "$@"
  if [ "$mode" = replace ]; then ab check '#replace-document'; fi
  click '开始导入'
  ab wait --text "$title"
  wait_js '!document.querySelector("#import-button").disabled && document.querySelector("#import-files").files.length===0'
}
value() {
  "$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$out/$1.json" "$2"
}
view_image() {
  local name="$1" version="$2" from="$3" message_id
  message_id="$(value "$from" message_id)"
  ab click "[data-message-id='$message_id'] [data-image-ordinal='2'] .source-actions button:nth-child(2)"
  wait_js "document.querySelector('#reader img')?.naturalWidth===640 && document.querySelector('#reader-title').textContent.includes('v$version')"
  save "$name" '({title:document.querySelector("#reader-title").textContent, location:document.querySelector("#reader-location").textContent, media:document.querySelector("#reader img").getAttribute("src"), width:document.querySelector("#reader img").naturalWidth, height:document.querySelector("#reader img").naturalHeight})'
  click '关闭原件'
}

args=(docs/G/G05-probe.py --data-dir "$out" --port "$port")
soffice="${SRAG_SOFFICE:-$(command -v soffice || true)}"
if [ -n "$soffice" ]; then args+=(--soffice "$soffice"); fi
if [ "${G05_REAL_VISION:-0}" = 1 ]; then
  args+=(--real-vision --env-file "${G05_ENV_FILE:-$PWD/.env}")
fi
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 "$python" "${args[@]}" > "$out/server.log" 2>&1 &
pid=$!
trap 'ab close >/dev/null 2>&1 || true; kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
printf 'G05 output %s\n' "$out"
curl --silent --fail --retry 7 --retry-connrefused --connect-timeout 1 "$url" >/dev/null

ab open "$url"
ab set viewport 1440 900
ab wait --text '暂无资料'
save browser 'navigator.userAgent'
upload harbor-panels new 'HarborTrace.ipynb · 导入完成 · v1' "$out/fixtures/v1/HarborTrace.ipynb"
courses harbor-panels
first='HarborTrace.ipynb 第 2 个 Cell 右侧琥珀子图的横轴刻度单位是什么？'
ask first "$first"
wait_js '[...document.querySelectorAll("#messages article:last-child figure img")].filter(x=>x.naturalWidth===640).length === 2'
ask follow '刚才第二张图'
if [ "${G05_REAL_VISION:-0}" = 1 ]; then
  ask follow-cache '刚才第二张图'
fi
click 'HarborTrace.ipynb v1'
wait_js 'document.querySelector("#location-title").textContent === "HarborTrace.ipynb"'
upload harbor-panels replace 'HarborTrace.ipynb · 导入完成 · v2' "$out/fixtures/v2/HarborTrace.ipynb"
courses harbor-panels
ask compare '请同时比较上一版和当前版 HarborTrace.ipynb 的第 2 个 Cell，概述两版琥珀色子图上的刻度单位。'
wait_js '[...document.querySelectorAll("#messages article:last-child figure img")].filter(x=>x.naturalWidth===640).length === 2'
view_image open-old 1 first
view_image open-current 2 compare

upload pulse-evidence new 'PulseGate-run.ipynb · 导入完成 · v1' \
  "$out/fixtures/PulseGate-lecture.ipynb" "$out/fixtures/PulseGate-limits.pptx" "$out/fixtures/PulseGate-run.ipynb"
courses pulse-evidence
two='What is the definition of a pulse gate, and how did the held-out experiment validate it?'
ask two "$two"
ask three "$two Use the threshold from the separate limit table and compare it with the saved output."
ask table 'PulseGate-limits.pptx 第 1 页表格里的 Pulse gate 设定值是多少？'

PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 "$python" docs/G/G05-probe.py --data-dir "$out" --verify
