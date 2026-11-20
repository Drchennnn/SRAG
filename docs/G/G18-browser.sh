#!/usr/bin/env bash
# Replay G18 using a cold data directory and the selected product commit.
set -euo pipefail
suite="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$suite" rev-parse --show-toplevel)"
python="${SRAG_PYTHON:-$repo/.venv/bin/python}"
target="$(git -C "$repo" rev-parse "${G18_SUBJECT:-HEAD}")"
port="${G18_PORT:-19148}"
out="${G18_OUT:-$(mktemp -d /tmp/srag-g18-browser.XXXXXX)}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
mkdir "$out/checkout"
git -C "$repo" archive "$target" | tar -x -C "$out/checkout"
cd "$out/checkout"
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1
args=("$out" --port "$port")
if [[ -n "${G18_SOFFICE:-}" ]]; then args+=(--soffice "$G18_SOFFICE"); fi
if [[ "${G18_NO_RENDERER:-false}" == true ]]; then args+=(--without-soffice); fi
"$python" "$suite/G18-browser-probe.py" "${args[@]}" > "$out/server.log" 2>&1 &
pid=$!
session="$(agent-browser session id --scope worktree --prefix "g18-$port")"
url="http://127.0.0.1:$port/workspace/"
ab() { agent-browser --session "$session" "$@"; }
cleanup() {
  local rc="$?"
  if [[ "$rc" != 0 ]]; then ab snapshot > "$out/last-snapshot.txt" 2>/dev/null || true; fi
  ab close >/dev/null 2>&1 || true
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}
trap cleanup EXIT
printf 'G18 output %s\n' "$out"
curl --silent --fail --retry 6 --retry-connrefused --connect-timeout 1 "$url" >/dev/null
ab --allowed-domains 127.0.0.1 open "$url"
ab set viewport 1440 900
ab wait --text '暂无资料'
ab eval 'navigator.userAgent' > "$out/browser.json"
click() { ab find role button click --name "$1" --exact; }
wait_js() { ab wait --timeout 15000 --fn "$1"; }
reader_loaded='document.querySelector("#reader").open && document.querySelector("#reader img")?.naturalWidth > 0 && !document.querySelector(".reader-picture").hidden'
pdf='Northpoint-Completion-Briefing.pdf'
pptx='Drum-Readings.pptx'
notebook='Drum-Trend.ipynb'

upload() {
  local fixture="$1" version="$2"
  if [[ "$(ab eval 'document.querySelector("#import-panel").open')" == false ]]; then
    ab find text '导入资料' click --exact
  fi
  ab select '#import-course' srag-g18-public
  ab upload '#import-files' "$out/fixtures/$fixture"
  if [[ "$version" == 2 ]]; then ab check '#replace-document'; fi
  click '开始导入'
  ab wait --text "${fixture##*/} · 导入完成 · v$version"
  wait_js '!document.querySelector("#import-button").disabled && !document.querySelector("#import-files").files.length'
}

catalog_node() {
  local file="$1" child="$2" id
  wait_js "(() => {
    const doc = [...document.querySelectorAll('#catalog button')].find(button => button.querySelector('.catalog-label')?.textContent === '$file');
    return doc && [...doc.parentElement.querySelectorAll('button')].some(button => button.querySelector('.catalog-label')?.textContent.startsWith('$child'));
  })()"
  id="$(ab eval "(() => {
    const tree = [...document.querySelectorAll('#catalog button')].find(button => button.querySelector('.catalog-label')?.textContent === '$file');
    return [...tree.parentElement.querySelectorAll('button')].find(button => button.querySelector('.catalog-label')?.textContent.startsWith('$child')).dataset.nodeId;
  })()" | "$python" -c 'import json,sys; print(json.load(sys.stdin))')"
  ab click "button[data-node-id='$id']"
}

message() {
  wait_js "document.querySelectorAll('#messages article:not(.message-user)').length === $1"
  ab eval '(() => {
    const entry = document.querySelector("#messages article:last-child");
    return {id:entry.dataset.messageId, text:entry.textContent, citations:[...entry.querySelectorAll("[data-evidence-id]")].map(x=>x.dataset.evidenceId), draft:document.querySelector("#question").value};
  })()' > "$out/message-$1.json"
}

ask() {
  ab find label '输入问题' fill "$2"
  click '发送'
  message "$1"
}

reader_state() {
  ab eval '(() => {
    const reader=document.querySelector("#reader"), image=reader.querySelector("img");
    return {title:document.querySelector("#reader-title").textContent, details:document.querySelector("#reader-location").textContent,
      status:document.querySelector("#reader-status").textContent, body:document.querySelector("#reader-body").innerText,
      chosen:image?.getAttribute("src"), image:[image?.naturalWidth??0,image?.naturalHeight??0],
      rect:reader.querySelector(".reader-highlight")?.getAttribute("style"), open:reader.open,
      layout:getComputedStyle(reader).position, viewport:[innerWidth,innerHeight], width:reader.getBoundingClientRect().width,
      horizontalOverflow:document.documentElement.scrollWidth>innerWidth, scroll:scrollY};
  })()' > "$out/reader-$1.json"
}

open_citation() {
  local index="$1" selector
  selector="$("$python" -c '
import json
import sys
data=json.load(open(sys.argv[1]))
quote = chr(39)
print("article[data-message-id={q}{id}{q}] button[data-evidence-id={q}{ev}{q}]".format(
    q=quote, id=data["id"], ev=data["citations"][0]))
' "$out/message-$index.json")"
  ab eval "document.querySelector(\"$selector\").scrollIntoView({block:'center'})"
  ab eval 'window.scrollY' > "$out/scroll-before-$index.json"
  ab click "$selector"
}

upload "v1/$pdf" 1
click "$pdf v1"
catalog_node "$pdf" '第 5 页'
wait_js "$reader_loaded"
reader_state pdf-tree
click '关闭原件'
click '原件预览'
wait_js "$reader_loaded"
click '关闭原件'
ask 1 '第 5 页夏季档的项目签字截止是什么？'
ab set viewport 680 700
open_citation 1
wait_js "$reader_loaded"
reader_state pdf-citation
cp "$out/scroll-before-1.json" "$out/scroll-before-narrow.json"
ab press Escape
wait_js '!document.querySelector("#reader").open && document.querySelector("#toggle-reader").getAttribute("aria-expanded")==="false"'
ask 2 '用一句话概括刚才的回答。'
ab set viewport 1440 900
ab find label '输入问题' fill '等一下继续问这份文件。'
click '总结此文件'
message 3
ab find text '查找原文' click --exact
ab find label '原文关键词' fill 'August 23'
click '查找'
message 4
ask 5 '第 5 页的提交时间是什么？'
ab eval '({buttons:document.querySelectorAll("#composer button[type=submit]").length, modes:document.querySelectorAll("#composer input[type=radio]").length})' > "$out/composer.json"
upload "v2/$pdf" 2
ab reload
message 5
open_citation 1
wait_js "$reader_loaded"
reader_state pdf-historical
click '关闭原件'
upload "$pptx" 1
click "$pptx v1"
catalog_node "$pptx" '第 1 页'
if [[ "${G18_NO_RENDERER:-false}" == true ]]; then
  wait_js 'document.querySelector("#reader").open && document.querySelector("#reader-body").innerText.includes("暂时无法渲染")'
else
  wait_js "$reader_loaded"
fi
reader_state pptx-tree
click '关闭原件'
catalog_node "$pptx" '表格'
if [[ "${G18_NO_RENDERER:-false}" == true ]]; then
  wait_js 'document.querySelector("#reader").open && document.querySelector("#reader-body").innerText.includes("暂时无法渲染")'
else
  wait_js "$reader_loaded"
fi
reader_state pptx-crop
click '关闭原件'
if [[ "$(ab eval 'document.querySelector("#search-panel").open')" == false ]]; then
  ab find text '查找原文' click --exact
fi
ab find label '原文关键词' fill 'Cedar'
click '查找'
message 6
open_citation 6
if [[ "${G18_NO_RENDERER:-false}" == true ]]; then
  wait_js 'document.querySelector("#reader").open && document.querySelector("#reader-body").innerText.includes("暂时无法渲染")'
else
  wait_js "$reader_loaded"
fi
reader_state pptx-citation
click '关闭原件'
upload "v1/$notebook" 1
click "$notebook v1"
catalog_node "$notebook" 'Cell 1'
catalog_node "$notebook" '输出 1'
wait_js "$reader_loaded"
reader_state notebook-output
click '关闭原件'
ask 7 '显示本 Cell 的原件图片'
wait_js 'Array.from(document.querySelectorAll("#messages article:last-child .source-actions button")).some(b=>b.textContent==="问这张图"&&!b.disabled)'
ab eval 'document.querySelector("#messages article:last-child .source-actions button").scrollIntoView({block:"center"})'
ab click '#messages article:last-child .source-actions button:first-child'
ask 8 '这张图横轴的单位是什么？'
open_citation 8
wait_js "$reader_loaded"
reader_state notebook-follow-citation
ab eval 'document.querySelector("#sessions").value' > "$out/session-a.json"
click '新建会话'
wait_js '!document.querySelector("#reader").open && !document.querySelector("#reader-body").children.length && !document.querySelector("#messages article")'
ab eval '({session:document.querySelector("#sessions").value, empty:!document.querySelector("#messages article"), preview:document.querySelector("#reader").open})' > "$out/session-b.json"
ab select '#sessions' "$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1])))' "$out/session-a.json")"
message 8
"$python" "$suite/G18-browser-verify.py" "$out" "$target" "${G18_NO_RENDERER:-false}"
