#!/usr/bin/env bash
# Run the public fixed-evidence trial before the normal G16 browser questions.
set -euo pipefail
suite="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$suite" rev-parse --show-toplevel)"
python="${SRAG_PYTHON:-$repo/.venv/bin/python}"
target="$(git -C "$repo" rev-parse "${G16_SUBJECT:-HEAD}")"
port="${G16_PORT:-19146}"
out="${G16_OUT:-$(mktemp -d /tmp/srag-g16-browser.XXXXXX)}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
test ! -e "$out/checkout"
mkdir "$out/checkout"
git -C "$repo" archive "$target" | tar -x -C "$out/checkout"
cd "$out/checkout"
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1
fixture="$suite/G16-browser-fixtures.py"
probe="$suite/G16-browser-probe.py"
"$python" "$fixture" "$out"
probe_args=("$out")
if [[ -n "${G16_ENV_FILE:-}" ]]; then
  probe_args+=(--env-file "$G16_ENV_FILE")
  "$python" "$probe" "${probe_args[@]}" --gold
fi
session="$(agent-browser session id --scope worktree --prefix "g16-workspace-$port")"
url="http://127.0.0.1:$port/workspace/"
ab() { agent-browser --session "$session" "$@"; }
click() { ab find role button click --name "$1" --exact; }
wait_js() { ab wait --timeout 90000 --fn "$1"; }

"$python" "$probe" "${probe_args[@]}" --port "$port" > "$out/server.log" 2>&1 &
pid=$!
cleanup() {
  ab close >/dev/null 2>&1 || true
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}
trap cleanup EXIT
printf 'G16 output %s\n' "$out"
curl --silent --fail --retry 6 --retry-connrefused --connect-timeout 1 "$url" >/dev/null

ab --allowed-domains 127.0.0.1 open "$url"
ab set viewport 1440 1080
ab wait --text '暂无资料'
ab eval 'navigator.userAgent' > "$out/browser.json"
file='Northpoint-Completion-Briefing.pdf'

upload() {
  local version="$1"
  if [ "$(ab eval 'document.querySelector("#import-panel").open')" = false ]; then
    ab find text '导入资料' click --exact
  fi
  ab select '#import-course' northpoint-completion
  ab upload '#import-files' "$out/fixtures/v$version/$file"
  if [ "$version" = 2 ]; then ab check '#replace-document'; fi
  click '开始导入'
  ab wait --text "$file · 导入完成 · v$version"
  wait_js '!document.querySelector("#import-button").disabled && !document.querySelector("#import-files").files.length'
}

ask() {
  local step="$1" case="$2" question
  question="$("$python" -c 'import runpy,sys; print(runpy.run_path(sys.argv[1])["QUESTIONS"][sys.argv[2]])' "$fixture" "$case")"
  ab find label '输入问题' fill "$question"
  click '发送'
  wait_js "document.querySelectorAll('#messages article:not(.message-user)').length === $step"
  ab eval '(() => {
    const message = document.querySelector("#messages article:last-child");
    return {message_id: message.dataset.messageId, focus:document.querySelector("#location-title").textContent,
      html_text:message.textContent, citations:[...message.querySelectorAll("button[data-evidence-id]")].map(x=>({label:x.textContent, id:x.dataset.evidenceId}))};
  })()' > "$out/turn-$step.json"
}

open_citations() {
  local step="$1" id selector
  while IFS= read -r id; do
    [ -n "$id" ] || continue
    selector="#messages article:last-child button[data-evidence-id='$id']"
    ab eval "document.querySelector(\"$selector\").scrollIntoView({block:'center'})"
    ab find first "$selector" click
    wait_js 'document.querySelector("#reader").open && /当前版本|历史版本/.test(document.querySelector("#reader-location").textContent)'
    ab eval '({title:document.querySelector("#reader-title").textContent, location:document.querySelector("#reader-location").textContent, body:document.querySelector("#reader-body").textContent})' > "$out/reader-$step-$id.json"
    click '关闭原件'
  done < <("$python" -c 'import json,sys; print("\n".join(dict.fromkeys(x["id"] for x in json.load(open(sys.argv[1]))["citations"])))' "$out/turn-$step.json")
}

upload 1
click "$file v1"
wait_js 'document.querySelector("#location-title").textContent === "Northpoint-Completion-Briefing.pdf"'
ask 1 first
open_citations 1
ask 2 follow
ab find role button click --name '第 2 页 · Directory and administrative contacts' --exact
wait_js 'document.querySelector("#location-title").textContent.startsWith("第 2 页")'
ask 3 first
click "$file v1"
upload 2
click "$file v2"
ab find role button click --name '第 5 页 · Summer award round' --exact
wait_js 'document.querySelector("#location-title").textContent.startsWith("第 5 页")'
ask 4 current
open_citations 4
ask 5 compare
open_citations 5
"$python" "$probe" "${probe_args[@]}" --verify "$target"
printf '%s\n' "$out/G16-browser-result.json"
