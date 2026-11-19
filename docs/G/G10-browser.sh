#!/usr/bin/env bash
# G10 synthetic development set. Product questions enter through Chromium only.
set -euo pipefail
cd "$(dirname "$0")/../.."
python="${SRAG_PYTHON:-$PWD/.venv/bin/python}"
subject="${G10_SUBJECT:-e921fa87f48a4e67c58010181a187c6e0e12b833}"
port="${G10_PORT:-18860}"
out="${G10_OUT:-$(mktemp -d /tmp/srag-g10-browser.XXXXXX)}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
test ! -e "$out/library.sqlite3"
session="$(agent-browser session id --scope worktree --prefix "g10-development-$port")"
url="http://127.0.0.1:$port/workspace/"
ab() { agent-browser --session "$session" "$@"; }
click() { ab find role button click --name "$1" --exact; }
wait_js() { ab wait --fn "$1"; }
save() { ab eval "$2" > "$out/$1.json"; }
question() {
  "$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["questions"][sys.argv[2]])' "${G10_QUESTIONS:-$out/fixtures.json}" "$1"
}
upload() {
  local mode="$1" title="$2"
  shift 2
  wait_js '!document.querySelector("#import-button").disabled'
  if [ "$(ab eval 'document.querySelector("#import-panel").open')" = false ]; then
    ab find text '导入资料' click --exact
  fi
  ab select '#import-course' drum-lab
  ab upload '#import-files' "$@"
  if [ "$mode" = replace ]; then ab check '#replace-document'; fi
  click '开始导入'
  ab wait --text "$title"
  wait_js '!document.querySelector("#import-button").disabled && document.querySelector("#import-files").files.length===0'
}
new_session() {
  local count
  count="$(ab get count '#sessions option')"
  click '新建会话'
  wait_js "document.querySelectorAll('#sessions option').length === ($count + 1) && document.querySelectorAll('#messages article').length === 0"
}
ask() {
  local count
  count="$(ab get count '#messages article:not(.message-user)')"
  ab find label '输入问题' fill "$2"
  click '发送'
  wait_js "document.querySelectorAll('#messages article:not(.message-user)').length === ($count+1)"
  wait_js '!document.querySelector("#messages article:last-child figure img[src]:not([src=\"\"])") || Array.from(document.querySelectorAll("#messages article:last-child figure img")).every(img => img.complete)'
  save "$1" '(() => {
    const m=document.querySelector("#messages article:last-child");
    const focus=document.querySelector("#location");
    return {message_id:m.dataset.messageId, text:m.textContent, focus:focus.hidden?null:document.querySelector("#location-title").textContent,
      citations:[...m.querySelectorAll("button[data-evidence-id]")].map(x=>({id:x.dataset.evidenceId,label:x.textContent})),
      images:[...m.querySelectorAll("figure img")].map(x=>({width:x.naturalWidth,height:x.naturalHeight,url:x.getAttribute("src")}))};
  })()'
}
open_citation() {
  local evidence="$1" expected="$2" selector
  selector="#messages article:last-child button[data-evidence-id='ev-$evidence']"
  if [ "$(ab get count "$selector")" = 1 ]; then
    ab click "$selector"
    wait_js "document.querySelector('#reader').open && document.querySelector('#reader-title').textContent.includes('$expected')"
    save "citation-$evidence" '({title:document.querySelector("#reader-title").textContent,location:document.querySelector("#reader-location").textContent,body:document.querySelector("#reader-body").textContent})'
    click '关闭原件'
  fi
}

PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 "$python" docs/G/G10-probe.py --data-dir "$out" --port "$port" --subject "$subject" --soffice "${G10_SOFFICE:-}" > "$out/server.log" 2>&1 &
pid=$!
trap 'ab close >/dev/null 2>&1 || true; kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
printf 'G10 output %s\n' "$out"
curl --silent --fail --retry 6 --retry-connrefused --connect-timeout 1 "$url" >/dev/null

ab open "$url"
ab set viewport 1440 1140
ab wait --text '暂无资料'
save browser 'navigator.userAgent'
upload new 'Drum-Lab.pdf · 导入完成 · v1' "$out/fixtures/Drum-Lab.pdf"
ask normal "$(question normal)"
new_session
click 'Drum-Lab.pdf v1'
wait_js 'document.querySelector("#location-title").textContent === "Drum-Lab.pdf"'
click '第 1 页 · Drum modulation lab'
wait_js 'document.querySelector("#location-title").textContent.startsWith("第 1 页")'
ask page "$(question normal)"
new_session
ask informal "$(question informal)"
new_session
upload new 'Drum-Rotated.pdf · 导入完成 · v1' "$out/fixtures/Drum-Rotated.pdf"
ask rotated "$(question rotated)"
upload new 'Drum-Readings.pptx · 导入完成 · v1' "$out/fixtures/Drum-Readings.pptx"
ask table "$(question table)"
open_citation 1 v1
new_session
click 'Drum-Readings.pptx v1'
wait_js 'document.querySelector("#location-title").textContent === "Drum-Readings.pptx"'
ab find role button click --name '第 1 页'
wait_js 'document.querySelector("#location-title").textContent.startsWith("第 1 页")'
ask table_page "$(question table)"
new_session
ask table_informal "$(question table_informal)"
upload new 'Drum-Scan.pdf · 导入完成 · v1' "$out/fixtures/Drum-Scan.pdf"
ask missing "$(question missing)"
ask ambiguous "$(question ambiguous)"
new_session
upload new 'Drum-Trend.ipynb · 导入完成 · v1' "$out/fixtures/v1/Drum-Trend.ipynb"
ask one_figure "$(question one_figure)"
ab find text '搜索' click --exact
ask old 'Resonance drift measured'
click 'Drum-Trend.ipynb v1'
wait_js 'document.querySelector("#location-title").textContent === "Drum-Trend.ipynb"'
upload replace 'Drum-Trend.ipynb · 导入完成 · v2' "$out/fixtures/v2/Drum-Trend.ipynb"
ab find text '问答' click --exact
ask compare "$(question compare)"
open_citation 1 v1
open_citation 2 v2

PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 "$python" docs/G/G10-probe.py --data-dir "$out" --verify --subject "$subject"
