#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

g02_python="${SRAG_PYTHON:-$PWD/.venv/bin/python}"
g02_port="${G02_PORT:-18831}"
g02_out="${G02_OUT:-$(mktemp -d /tmp/srag-g02-browser.XXXXXX)}"
g02_soffice="${SRAG_SOFFICE:-$(command -v soffice || true)}"
g02_url="http://127.0.0.1:$g02_port/workspace/"
g02_session="$(agent-browser session id --scope worktree --prefix "${G02_BROWSER_PREFIX:-srag-g02}")"
g02_a="$PWD/docs/A/样例"
g02_question='中值滤波 median filter 的定义是什么？课件所对应的留出集实验如何验证？'
mkdir -p "$g02_out"
g02_out="$(cd "$g02_out" && pwd)"
test ! -e "$g02_out/library.sqlite3"

ab() { agent-browser --session "$g02_session" "$@"; }
click() { ab find role button click --name "$1" --exact; }
wait_text() { ab wait --text "$1"; }
wait_js() { ab wait --fn "$1"; }
save_js() { ab eval "$2" > "$g02_out/$1.json"; }
ask() {
  local before
  before="$(ab get count '#messages article:not(.message-user)')"
  ab find label '输入问题' fill "$1"
  click '发送'
  wait_js "document.querySelectorAll('#messages article:not(.message-user)').length === ($before + 1)"
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
  wait_text "$title"
  wait_js '!document.querySelector("#import-button").disabled && document.querySelector("#import-files").files.length===0'
}
file_focus() {
  click "$1 v$2"
  wait_js "document.querySelector('#location-title').textContent === '$1'"
}
book_cell() {
  file_focus 'Signal-lab.ipynb' "$1"
  click 'Median filter lab'
  wait_js 'document.querySelector("#location-title").textContent === "Median filter lab"'
  click 'Cell 2 · Median filter lab'
  wait_js 'document.querySelector("#location-title").textContent.startsWith("Cell 2")'
}
last_contains() {
  local expected
  expected="$("$g02_python" -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1")"
  wait_js "document.querySelector('#messages article:last-child')?.textContent.includes($expected)"
}
book_figures='[...document.querySelectorAll("#messages article:last-child figure img")].filter(i=>i.naturalWidth===440 && i.naturalHeight===260).length===2'
record_reader='({title:document.querySelector("#reader-title").textContent, location:document.querySelector("#reader-location").textContent, text:document.querySelector("#reader-body").textContent.trim()})'
close_reader() { click '关闭原件'; }
view_citation() {
  local key="$1" label="$2" content="$3"
  click "$label"
  local expected
  expected="$("$g02_python" -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$content")"
  wait_js "document.querySelector('#reader').open && document.querySelector('#reader-body').textContent.includes($expected)"
  save_js "$key" "$record_reader"
  close_reader
}

args=(docs/G/G02-server.py --data-dir "$g02_out" --port "$g02_port")
if [ -n "$g02_soffice" ]; then args+=(--soffice "$g02_soffice"); fi
if [ "${G02_REAL_VISION:-0}" = 1 ]; then
  args+=(--real-vision --env-file "${G02_ENV_FILE:-$PWD/.env}")
fi
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 "$g02_python" "${args[@]}" > "$g02_out/server.log" 2>&1 &
g02_pid=$!
trap 'ab close >/dev/null 2>&1 || true; kill "$g02_pid" 2>/dev/null || true; wait "$g02_pid" 2>/dev/null || true' EXIT
printf 'G02 output %s\n' "$g02_out"
curl --silent --show-error --fail --retry 7 --retry-connrefused --connect-timeout 1 "$g02_url" >/dev/null

ab open "$g02_url"
ab set viewport 1440 900
wait_text '暂无资料'
if [ "${G02_RECORD:-0}" = 1 ]; then ab record start "$g02_out/replay.mp4" --cursor --fps 15; fi
ab eval "$(cat <<'JS'
(() => {
  const messages = document.querySelector("#messages");
  const assistants = () => messages.querySelectorAll("article:not(.message-user)");
  const laps = window.g02Laps = {agent: navigator.userAgent, turns: [], imports: []};
  let pending;
  document.querySelector("#import-form").addEventListener("submit", () => {
    laps.imports.push({
      start: performance.now(),
      files: [...document.querySelector("#import-files").files].map(x => x.name),
      course: document.querySelector("#import-course").value,
      replacement: document.querySelector("#replace-document").checked
    });
  }, {capture: true});
  document.querySelector("#composer").addEventListener("submit", () => {
    pending = {start: performance.now(), messages: assistants().length};
  }, {capture: true});
  new MutationObserver(() => {
    const done = assistants();
    if (pending && done.length > pending.messages) {
      laps.turns.push({
        message_id: done[done.length - 1].dataset.messageId,
        end: performance.now(),
        visible_ms: Math.round((performance.now() - pending.start) * 100) / 100
      });
      pending = null;
    }
  }).observe(messages, {childList: true});
})()
JS
)"

upload signals new 'Signal-reference.pdf · 导入完成 · v1' "$g02_a/Signal-reference.pdf"
file_focus 'Signal-reference.pdf' 1
ab find text '搜索' click --exact
ask 'lookahead'
last_contains 'lookahead'
ask 'lookahead'
last_contains 'lookahead'

upload signals new 'Signal-slides.pptx · 导入完成 · v1' "$g02_a/Signal-slides.pptx"
file_focus 'Signal-slides.pptx' 1
ask 'Median filter experiment'
last_contains 'Median filter experiment'
ask 'Median filter experiment'
last_contains 'Median filter experiment'

upload signals new 'Signal-lab.ipynb · 导入完成 · v1' "$g02_a/Signal-lab.ipynb"
book_cell 1
ask 'Median filter'
last_contains 'Median filter'
ask 'Median filter'
last_contains 'Median filter'

file_focus 'Signal-reference.pdf' 1
ab find text '问答' click --exact
ask '中值滤波 median filter 的定义是什么？'
last_contains 'median'
upload signals replace 'Signal-reference.pdf · 导入完成 · v1' "$g02_a/Signal-reference.pdf"
book_cell 1
ask '展示这个 Cell 保存的两张图'
wait_js "$book_figures"
save_js notebook-v1 '({message_id:document.querySelector("#messages article:last-child").dataset.messageId, displays:[...document.querySelectorAll("#messages article:last-child figure")].map(x=>({display_id:x.dataset.displayId, ordinal:Number(x.dataset.imageOrdinal), url:x.querySelector("img").getAttribute("src"), img_width:x.querySelector("img").naturalWidth}))})'
g02_image_message="$("$g02_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["message_id"])' "$g02_out/notebook-v1.json")"
g02_second='[data-message-id="'"$g02_image_message"'"] [data-image-ordinal="2"]'
ab click "$g02_second .source-actions button:nth-child(2)"
wait_js 'document.querySelector("#reader img")?.naturalWidth===440 && document.querySelector("#reader-title").textContent.includes("v1")'
save_js image-original '({title:document.querySelector("#reader-title").textContent, location:document.querySelector("#reader-location").textContent, media:document.querySelector("#reader img").getAttribute("src"), width:document.querySelector("#reader img").naturalWidth, height:document.querySelector("#reader img").naturalHeight})'
close_reader
ask '刚才第二张图'
ask '刚才第二张图'

upload signal-demo new 'A7-observations.ipynb · 导入完成 · v1' \
  "$g02_out/fixtures/complete/Median-lecture.ipynb" \
  "$g02_out/fixtures/complete/Median-drills.ipynb" \
  "$g02_out/fixtures/complete/A7-observations.ipynb"
click signal-demo
wait_js 'document.querySelector("#scope-summary").textContent.includes("signal-demo") && document.querySelector("#scope-summary").textContent.includes("全部资料")'
ask "$g02_question"
last_contains '0.18'
view_citation definition-citation 'Median-lecture.ipynb · v1 · Cell 1' 'A median filter replaces'
view_citation experiment-citation 'A7-observations.ipynb · v1 · Cell 1' 'Trial: 0.18.'

file_focus 'A7-observations.ipynb' 1
upload signal-demo replace 'A7-observations.ipynb · 导入完成 · v2' \
  "$g02_out/fixtures/v2/A7-observations.ipynb"
click '查看所选课程'
wait_js 'document.querySelector("#scope-summary").textContent.includes("全部资料")'
ask "$g02_question"
last_contains '0.16'
ab find role button click --name 'A7-observations.ipynb · v2 · Cell 1' --exact
wait_js 'document.querySelector("#reader").open && document.querySelector("#reader-body").textContent.includes("Trial: 0.16.")'
save_js current-experiment "$record_reader"
close_reader

click signals
wait_js 'document.querySelector("#scope-summary").textContent.includes("signals")'
file_focus 'Signal-lab.ipynb' 1
upload signals replace 'Signal-lab.ipynb · 导入完成 · v2' "$g02_a/revisions/v2/Signal-lab.ipynb"

upload signal-incomplete new 'A7-observations.ipynb · 导入完成 · v1' \
  "$g02_out/fixtures/incomplete/Median-lecture.ipynb" \
  "$g02_out/fixtures/incomplete/Median-drills.ipynb" \
  "$g02_out/fixtures/incomplete/A7-observations.ipynb"
click signal-incomplete
wait_js 'document.querySelector("#scope-summary").textContent.includes("signal-incomplete") && document.querySelector("#scope-summary").textContent.includes("全部资料")'
ask "$g02_question"
last_contains '原文不足以确认与该定义相符的实测比较'
save_js incomplete-visible '({text:document.querySelector("#messages article:last-child").textContent, buttons:[...document.querySelectorAll("#messages article:last-child button")].map(x=>x.textContent)})'
save_js visual-times 'window.g02Laps'

ab reload
wait_js 'document.querySelectorAll("#messages article:not(.message-user)").length === 13'
view_citation historical-experiment 'A7-observations.ipynb · v1 · Cell 1' 'Trial: 0.18.'
ab click "$g02_second .source-actions button:nth-child(2)"
wait_js 'document.querySelector("#reader img")?.naturalWidth===440 && document.querySelector("#reader-title").textContent.includes("v1")'
save_js historical-image '({title:document.querySelector("#reader-title").textContent, location:document.querySelector("#reader-location").textContent, media:document.querySelector("#reader img").getAttribute("src"), width:document.querySelector("#reader img").naturalWidth, height:document.querySelector("#reader img").naturalHeight})'
close_reader

if [ "${G02_RECORD:-0}" = 1 ]; then ab record stop --json > "$g02_out/replay-meta.json"; fi

"$g02_python" docs/G/G02-verify.py "$g02_out"
printf '%s\n' "$g02_out"
