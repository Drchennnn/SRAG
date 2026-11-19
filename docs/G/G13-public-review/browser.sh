#!/usr/bin/env bash
# Use the normal web UI against an export of a single commit. No injected turn or anchor.
set -euo pipefail
sealed="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$sealed" rev-parse --show-toplevel)"
revision="$(git -C "$repo" rev-parse "${1:?pass one fixed revision}^{commit}")"
out="${2:?pass an empty output directory}"
port="${3:?pass an unused local port}"
python="${SRAG_PYTHON:-$repo/.venv/bin/python}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
test ! -e "$out/config.json"
test ! -e "$out/library.sqlite3"
test ! -e "$out/tree"
mkdir "$out/tree"
git -C "$repo" archive "$revision" | tar -x -C "$out/tree"
cd "$out/tree"
session="$(agent-browser session id --scope worktree --prefix "g13-withheld-${port}")"
url="http://127.0.0.1:$port/workspace/"
ab() { agent-browser --session "$session" "$@"; }
click() { ab find role button click --name "$1" --exact; }
wait_js() { ab wait --fn "$1"; }
save() { ab eval "$2" > "$out/$1.json"; }

upload() {
  local course="$1" mode="$2" status="$3" path="$4"
  wait_js '!document.querySelector("#import-button").disabled'
  if [ "$(ab eval 'document.querySelector("#import-panel").open')" = false ]; then
    ab find text '导入资料' click --exact
  fi
  ab select '#import-course' "$course"
  ab upload '#import-files' "$path"
  if [ "$mode" = replace ]; then ab check '#replace-document'; fi
  click '开始导入'
  ab wait --text "$status"
  wait_js '!document.querySelector("#import-button").disabled && document.querySelector("#import-files").files.length===0'
}

course_only() {
  click "$1"
  wait_js "document.querySelector('#scope-summary').textContent.includes('$1') && document.querySelector('#scope-summary').textContent.includes('全部资料')"
}

new_session() {
  local before
  before="$(ab get count '#sessions option')"
  click '新建会话'
  wait_js "document.querySelectorAll('#sessions option').length === ($before + 1) && document.querySelectorAll('#messages article').length === 0"
}

open_reader() {
  local selector="$1" name="$2"
  ab eval "document.querySelector(\`$selector\`).scrollIntoView({block:'center'})" >/dev/null
  ab focus "$selector"
  ab press Enter
  wait_js "document.querySelector('#reader').open"
  if [ "$(ab get count '#reader img')" -gt 0 ]; then
    wait_js "document.querySelector('#reader img')?.complete"
  fi
  save "$name" '(() => {
    const img=document.querySelector("#reader img"), highlight=document.querySelector(".reader-highlight");
    return {title:document.querySelector("#reader-title").textContent,
      location:document.querySelector("#reader-location").textContent,body:document.querySelector("#reader-body").textContent,
      image:img?.getAttribute("src"),width:img?.naturalWidth,height:img?.naturalHeight,
      rect:highlight?[highlight.style.left,highlight.style.top,highlight.style.width,highlight.style.height]:null};
  })()'
  click '关闭原件'
}

ask() {
  local name="$1" question before images citations idx selector
  question="$("$python" -c 'import json,sys; print(next(x["text"] for x in json.load(open(sys.argv[1]))["cases"] if x["id"]==sys.argv[2]))' "$sealed/protocol.json" "$name")"
  before="$(ab get count '#messages article:not(.message-user)')"
  ab find label '输入问题' fill "$question"
  click '发送'
  wait_js "document.querySelectorAll('#messages article:not(.message-user)').length === ($before + 1)"
  wait_js 'document.querySelector("#messages article:last-child:not(.message-user)") != null'
  wait_js '[...document.querySelectorAll("#messages article:last-child figure img")].every(i=>i.complete)'
  save "$name" '(() => {
    const m=document.querySelector("#messages article:last-child");
    return {message_id:m.dataset.messageId,text:m.textContent,
      answer:[...m.querySelectorAll(".text-block")].map(x=>x.textContent),
      notices:[...m.querySelectorAll(".notice-block")].map(x=>x.textContent),
      citations:[...m.querySelectorAll("button[data-evidence-id]")].map(x=>({id:x.dataset.evidenceId,label:x.textContent})),
      images:[...m.querySelectorAll("figure")].map(x=>({id:x.dataset.displayId,label:x.textContent,
        width:x.querySelector("img")?.naturalWidth,height:x.querySelector("img")?.naturalHeight,
        url:x.querySelector("img")?.getAttribute("src")})),
      clarification:[...document.querySelectorAll(".clarification-actions button")].map(x=>x.textContent)};
  })()'
  images="$(ab get count '#messages article:last-child figure')"
  citations="$(ab get count '#messages article:last-child button[data-evidence-id]')"
  for ((idx=1;idx<=images;idx++)); do
    selector="#messages article:last-child figure:nth-of-type($idx) .source-actions button:nth-child(2)"
    if [ "$(ab eval "!!document.querySelector(\`$selector\`) && !document.querySelector(\`$selector\`).disabled")" = true ]; then
      open_reader "$selector" "$name-image-$idx"
    fi
    selector="#messages article:last-child figure:nth-of-type($idx) .source-actions button:nth-child(3)"
    if [ "$(ab eval "!!document.querySelector(\`$selector\`) && !document.querySelector(\`$selector\`).disabled")" = true ]; then
      open_reader "$selector" "$name-page-$idx"
    fi
  done
  for ((idx=0;idx<citations;idx++)); do
    selector="#messages article:last-child button[data-evidence-id]"
    # Mark the exact located DOM node temporarily, and click the same browser control a student would.
    ab eval "document.querySelectorAll(\`$selector\`)[$idx].setAttribute('data-g13-open', 'true')" >/dev/null
    open_reader "$selector[data-g13-open]" "$name-citation-$((idx + 1))"
    ab eval "document.querySelector(\`$selector[data-g13-open]\`).removeAttribute('data-g13-open')" >/dev/null
  done
}

args=("$sealed/probe.py" --out "$out" --port "$port" --revision "$revision")
if [ -n "${G13_ENV_FILE:-}" ]; then args+=(--env-file "$G13_ENV_FILE"); fi
if [ -n "${SRAG_SOFFICE:-}" ]; then args+=(--soffice "$SRAG_SOFFICE"); fi
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$out/tree" "$python" "${args[@]}" > "$out/server.log" 2>&1 &
pid=$!
trap 'ab close >/dev/null 2>&1 || true; kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
curl --silent --fail --retry 9 --retry-connrefused --connect-timeout 1 "$url" >/dev/null
ab open "$url"
ab set viewport 1536 1060
ab wait --text '暂无资料'
save browser '({agent:navigator.userAgent, origin:window.location.origin})'

upload azimuth-workshop new 'Azimuth-Workshop.pdf · 导入完成 · v1' "$sealed/materials/Azimuth-Workshop.pdf"
upload beacon-inspection new 'Beacon-Inspection.pptx · 导入完成 · v1' "$sealed/materials/Beacon-Inspection.pptx"
upload bench-retest new 'BenchRetest.ipynb · 导入完成 · v1' "$sealed/materials/v1/BenchRetest.ipynb"
for name in pdf-right-formal pdf-right-warm pdf-left-colloquial pdf-contrast-negation pdf-ambiguous; do
  new_session
  course_only azimuth-workshop
  ask "$name"
done
for name in native-table not-in-table; do
  new_session
  course_only beacon-inspection
  ask "$name"
done
new_session
course_only bench-retest
ask notebook-original
click 'BenchRetest.ipynb v1'
wait_js 'document.querySelector("#location-title").textContent === "BenchRetest.ipynb"'
upload bench-retest replace 'BenchRetest.ipynb · 导入完成 · v2' "$sealed/materials/v2/BenchRetest.ipynb"
course_only bench-retest
ask notebook-both
