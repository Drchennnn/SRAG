#!/usr/bin/env bash
# Replay the predeclared independent fixture without using a developer or private student corpus.
set -euo pipefail
fixture_suite="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$fixture_suite" rev-parse --show-toplevel)"
python="${SRAG_PYTHON:-$repo/.venv/bin/python}"
out="${G19_OUT:-$(mktemp -d /tmp/srag-g19-independent.XXXXXX)}"
port="${G19_PORT:-19329}"
sha="$(git -C "$repo" rev-parse "${G19_SUBJECT:?Specify the published integrated commit SHA}")"
mkdir -p "$out/checkout"
out="$(cd "$out" && pwd)"
printf '%s\n' "$sha" > "$out/subject.txt"
git -C "$repo" archive "$sha" | tar -x -C "$out/checkout"
cd "$out/checkout"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD"
"$python" "$fixture_suite/serve.py" "$out" --env-file "${G19_ENV_FILE:-$repo/.env}" \
  --port "$port" --soffice "${G19_SOFFICE:?Specify a LibreOffice executable}" > "$out/server.log" 2>&1 &
pid=$!
session="$(agent-browser session id --scope worktree --prefix "g19-independent-$port")"
ab() { agent-browser --session "$session" "$@"; }
cleanup() {
  ab close >/dev/null 2>&1 || true
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}
trap cleanup EXIT
printf 'G19 independent artifacts %s\n' "$out"
url="http://127.0.0.1:$port/workspace/"
curl --silent --fail --retry 6 --retry-connrefused --connect-timeout 1 "$url" >/dev/null
ab --allowed-domains 127.0.0.1 open "$url"
ab set viewport 1310 790
ab wait --text '暂无资料'
ab eval 'navigator.userAgent' > "$out/chromium.json"
click() { ab find role button click --name "$1" --exact; }
wait_js() { ab wait --timeout "${G19_WAIT_MS:-70000}" --fn "$1"; }
ask() {
  local case="$1" text
  text="$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["questions"][sys.argv[2]])' "$out/manifest.json" "$case")"
  ab find label '输入问题' fill "$text"
  click '发送'
  if ! wait_js "document.querySelectorAll('#messages article.message-user').length === $2 &&
    (document.querySelectorAll('#messages article.message-user').item($2-1).nextElementSibling?.matches('article:not(.message-user)') ||
    document.querySelector('#turn-status')?.textContent.includes('处理失败') ||
    document.querySelector('#turn-status')?.textContent.includes('需要澄清'))"; then
    ab snapshot -i > "$out/$2-$case-stalled.txt"
  fi
  ab eval "(() => {
    const user=document.querySelectorAll('#messages article.message-user').item($2-1);
    const m=user?.nextElementSibling;
    if (!m?.matches('article:not(.message-user)')) return {missing:true, notices:document.querySelector('#turn-status')?.textContent};
    return {id:m.dataset.messageId, text:m.textContent,
      citations:[...new Set([...m.querySelectorAll('button[data-evidence-id]')].map(b=>b.dataset.evidenceId))],
      cards:[...m.querySelectorAll('figure')].map(f=>({id:f.dataset.displayId,ordinal:f.dataset.imageOrdinal,
        location:f.querySelector('figcaption')?.innerText, url:f.querySelector('img')?.getAttribute('src'),
        visible:f.querySelector('img')?.naturalWidth>0 && !f.querySelector('img').hidden}))};
  })()" > "$out/$2-$case.json"
}
upload() {
  local fixture="$1" version="$2"
  if [[ "$(ab eval 'document.querySelector("#import-panel").open')" == false ]]; then
    ab find text '导入资料' click --exact
  fi
  ab select '#import-course' g19-independent-canopy
  ab upload '#import-files' "$out/fixtures/$fixture"
  if [[ "$version" == 2 ]]; then ab check '#replace-document'; fi
  click '开始导入'
  ab wait --text "${fixture##*/} · 导入完成 · v$version"
  wait_js '!document.querySelector("#import-button").disabled && !document.querySelector("#import-files").files.length'
}
reader() {
  ab eval '(() => {const r=document.querySelector("#reader");const img=r.querySelector("img");
    return {open:r.open,location:document.querySelector("#reader-location").innerText,
      title:document.querySelector("#reader-title").innerText,status:document.querySelector("#reader-status").innerText,
      body:document.querySelector("#reader-body").innerText,url:img?.getAttribute("src"),
      displayed:img?.naturalWidth>0 && !img.closest(".reader-picture")?.hidden,
      highlight:r.querySelector(".reader-highlight")?.getAttribute("style")};})()'
}
open_citations() {
  local case="$1" label="${2:-$1}" choice selector
  while IFS= read -r choice; do
    [[ -n "$choice" ]] || continue
    selector="article[data-message-id='${choice%%:*}'] button[data-evidence-id='${choice#*:}']"
    ab eval "document.querySelector(\"$selector\").scrollIntoView({block:'center'})"
    ab click "$selector"
    wait_js 'document.querySelector("#reader").open && (!document.querySelector("#reader img") ||
      document.querySelector("#reader img").complete) &&
      !document.querySelector("#reader-body").innerText.includes("正在加载原件")'
    reader > "$out/$label-citation-${choice#*:}.json"
    click '关闭原件'
  done < <("$python" -c 'import json,sys;d=json.load(open(sys.argv[1])); [print(d["id"]+":"+i) for i in d.get("citations", [])]' "$out/$case.json")
}
confirmed_cards() {
  local record="$1"
  if wait_js 'Array.from(document.querySelectorAll("#messages article:last-child figure img")).filter(i=>!i.hidden&&i.naturalWidth>0).length===2'; then
    ab eval 'Array.from(document.querySelectorAll("#messages article:last-child figure")).map(f=>({id:f.dataset.displayId,
      ordinal:f.dataset.imageOrdinal,status:f.querySelector("figcaption").innerText,url:f.querySelector("img").getAttribute("src")}))' > "$out/$record-actual-visible.json"
  else
    ab snapshot -i > "$out/$record-picture-count-failure.txt"
  fi
}

guide='Canopy-Coupler-Field-Notes.pdf'
deck='Sparrow-Coupler-Release-Log.pptx'
ipynb='Sparrow-Repeat-Run.ipynb'
upload "v1/$guide" 1
click "$guide v1"
ask first 1
open_citations 1-first
ask summary 2
open_citations 2-summary
ask pdf_visual 3
open_citations 3-pdf_visual
upload "v1/$ipynb" 1
click "$ipynb v1"
ask show_saved 4
confirmed_cards 4-show_saved
ask second 5
open_citations 5-second
upload "v2/$ipynb" 2
ask show_saved 6
confirmed_cards 6-show_saved
ask second 7
open_citations 7-second
open_citations 5-second 5-second-after-v2
click "$guide v1"
upload "v2/$guide" 2
ask pdf_v2 8
open_citations 8-pdf_v2
ask version 9
open_citations 9-version
ab reload
wait_js 'document.querySelectorAll("#messages article.message-user").length === 9'
open_citations 9-version 9-version-after-refresh
upload "$deck" 1
click '查看所选课程'
ask cross_file 10
open_citations 10-cross_file
ask missing 11
open_citations 11-missing
ask composition_missing 12
open_citations 12-composition_missing
printf 'Saved HTTP, provider and DOM artifacts in %s\n' "$out"
