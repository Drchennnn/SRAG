#!/usr/bin/env bash
# Browser-only questions and original-citation opens against a fresh offline workspace.
set -euo pipefail
cd "$(dirname "$0")/../.."
python="${SRAG_PYTHON:-$PWD/.venv/bin/python}"
port="${G07_PORT:-18859}"
out="${G07_OUT:-$(mktemp -d /tmp/srag-g07-browser.XXXXXX)}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
test ! -e "$out/library.sqlite3"
session="$(agent-browser session id --scope worktree --prefix "${G07_BROWSER_PREFIX:-g07-public-history}")"
url="http://127.0.0.1:$port/workspace/"
export G07_SUBJECT_BASE="${G07_SUBJECT_BASE:-$(git rev-parse HEAD)}"
ab() { agent-browser --session "$session" "$@"; }
click() { ab find role button click --name "$1" --exact; }
wait_js() { ab wait --fn "$1"; }
save() { ab eval "$2" > "$out/$1.json"; }
courses() {
  click "$1"
  wait_js "document.querySelector('#scope-summary').textContent.includes('$1') && document.querySelector('#scope-summary').textContent.includes('全部资料')"
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
new_session() {
  local before
  before="$(ab get count '#sessions option')"
  click '新建会话'
  wait_js "document.querySelectorAll('#sessions option').length === ($before + 1) && document.querySelectorAll('#messages article').length === 0"
}
ask() {
  local before
  before="$(ab get count '#messages article:not(.message-user)')"
  if [ "$#" -gt 1 ]; then ab find label '输入问题' fill "$2"; fi
  click '发送'
  wait_js "document.querySelectorAll('#messages article:not(.message-user)').length === ($before + 1)"
  if [ "$1" = current ] || [ "$1" = choice ]; then
    wait_js 'document.querySelector("#messages article:last-child figure img")?.naturalWidth===640'
  fi
  save "$1" '(() => {
    const m=document.querySelector("#messages article:last-child");
    return {message_id:m.dataset.messageId, text:m.textContent,
            citations:[...m.querySelectorAll("button[data-evidence-id]")].map(x=>({id:x.dataset.evidenceId,label:x.textContent})),
            images:[...m.querySelectorAll("figure img")].map(x=>({width:x.naturalWidth,url:x.getAttribute("src")}))};
  })()'
}
open_citation() {
  local version="$1" value="$2" message_id selector
  message_id="$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["message_id"])' "$out/compare.json")"
  selector="[data-message-id='$message_id'] button[data-evidence-id='ev-$version']"
  ab eval "document.querySelector(\`$selector\`).scrollIntoView({block:'center'})"
  ab focus "$selector"
  ab press Enter
  wait_js "document.querySelector('#reader').open && document.querySelector('#reader-title').textContent.includes('v$version') && document.querySelector('#reader-body').textContent.includes('$value')"
  save "citation-v$version" '({title:document.querySelector("#reader-title").textContent,
       location:document.querySelector("#reader-location").textContent,body:document.querySelector("#reader-body").textContent})'
  click '关闭原件'
}

PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 "$python" docs/G/G07-probe.py --data-dir "$out" --port "$port" > "$out/server.log" 2>&1 &
pid=$!
trap 'ab close >/dev/null 2>&1 || true; kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
printf 'G07 output %s\n' "$out"
curl --silent --fail --retry 7 --retry-connrefused --connect-timeout 1 "$url" >/dev/null

ab open "$url"
ab set viewport 1440 1140
ab wait --text '暂无资料'
save browser 'navigator.userAgent'
upload harbor-panels new 'HarborTrace.ipynb · 导入完成 · v1' "$out/fixtures/v1/HarborTrace.ipynb"
courses harbor-panels
click 'HarborTrace.ipynb v1'
wait_js 'document.querySelector("#location-title").textContent === "HarborTrace.ipynb"'
click 'Harbor pump study'
wait_js 'document.querySelector("#location-title").textContent === "Harbor pump study"'
click 'Cell 2 · Harbor pump study'
wait_js 'document.querySelector("#location-title").textContent.startsWith("Cell 2")'
ask history-empty '刚才第二张图'
save focus-setup '({focus:document.querySelector("#location-title").textContent,history_images:document.querySelectorAll("#messages figure img").length})'
ask current '第二张图'

new_session
courses harbor-panels
ask ambiguous '那张图是哪一张？'
wait_js 'document.querySelectorAll(".clarification-actions button").length === 2'
options='[...document.querySelectorAll(".clarification-actions button")].map(x=>x.textContent)'
save options-before-refresh "$options"
ab reload
wait_js 'document.querySelectorAll(".clarification-actions button").length === 2'
save options-after-refresh "$options"
ab click '.clarification-actions button:nth-child(2)'
wait_js '!document.querySelector("#anchor-bar").hidden'
ask choice

new_session
upload pulse-evidence new 'VersionDeck.ipynb · 导入完成 · v1' "$out/fixtures/v1/VersionDeck.ipynb"
courses pulse-evidence
ab find text '搜索' click --exact
ask version-history 'Pilot amplitude frequency'
click 'VersionDeck.ipynb v1'
wait_js 'document.querySelector("#location-title").textContent === "VersionDeck.ipynb"'
upload pulse-evidence replace 'VersionDeck.ipynb · 导入完成 · v2' "$out/fixtures/v2/VersionDeck.ipynb"
courses pulse-evidence
ab find text '问答' click --exact
ask compare '比较 VersionDeck.ipynb 上一版和当前版 Cell 1 保存输出。'
open_citation 1 '2.5 kHz'
open_citation 2 '4.5 kHz'

PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 "$python" docs/G/G07-probe.py --data-dir "$out" --verify
