#!/usr/bin/env bash
# Submit an ordinary first question on a fresh workspace with one selected public source.
set -euo pipefail
cd "$(dirname "$0")/../.."
python="${SRAG_PYTHON:-$PWD/.venv/bin/python}"
port="${G06_PORT:-18854}"
out="${G06_OUT:-$(mktemp -d /tmp/srag-g06-browser.XXXXXX)}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
test ! -e "$out/library.sqlite3"
session="$(agent-browser session id --scope worktree --prefix "${G06_BROWSER_PREFIX:-g06-public-first}")"
url="http://127.0.0.1:$port/workspace/"
export G06_SUBJECT_BASE="${G06_SUBJECT_BASE:-$(git rev-parse HEAD)}"
ab() { agent-browser --session "$session" "$@"; }
click() { ab find role button click --name "$1" --exact; }
wait_js() { ab wait --fn "$1"; }

upload() {
  local course="$1" title="$2"
  shift 2
  wait_js '!document.querySelector("#import-button").disabled'
  if [ "$(ab eval 'document.querySelector("#import-panel").open')" = false ]; then
    ab find text '导入资料' click --exact
  fi
  ab select '#import-course' "$course"
  ab upload '#import-files' "$@"
  click '开始导入'
  ab wait --text "$title"
  wait_js '!document.querySelector("#import-button").disabled && document.querySelector("#import-files").files.length===0'
}
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
  if [ "$1" != table ]; then
    wait_js 'document.querySelector("#messages article:last-child figure img")?.naturalWidth === 640'
  fi
  ab eval '(() => {
    const message=document.querySelector("#messages article:last-child");
    const img=message.querySelector("figure img");
    return {message_id:message.dataset.messageId,text:message.textContent,media_url:img?.getAttribute("src"),
            width:img?.naturalWidth,height:img?.naturalHeight,
            displays:message.querySelectorAll("figure").length};
  })()' > "$out/$1.json"
}

args=(docs/G/G06-probe.py --data-dir "$out" --port "$port")
soffice="${SRAG_SOFFICE:-$(command -v soffice || true)}"
if [ -n "$soffice" ]; then args+=(--soffice "$soffice"); fi
if [ "${G06_REAL_VISION:-0}" = 1 ]; then
  args+=(--real-vision --env-file "${G06_ENV_FILE:?set G06_ENV_FILE for authorized vision}")
fi
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 "$python" "${args[@]}" > "$out/server.log" 2>&1 &
pid=$!
trap 'ab close >/dev/null 2>&1 || true; kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
printf 'G06 output %s\n' "$out"
curl --silent --fail --retry 7 --retry-connrefused --connect-timeout 1 "$url" >/dev/null
ab open "$url"
ab set viewport 1440 900
ab wait --text '暂无资料'
ab eval 'navigator.userAgent' > "$out/browser.json"
upload harbor-panels 'HarborTrace.ipynb · 导入完成 · v1' "$out/fixtures/v1/HarborTrace.ipynb"
courses harbor-panels
question='HarborTrace.ipynb Cell 2 第 2 张图的横轴刻度单位是什么？'
ask first "$question"
if [ "${G06_REAL_VISION:-0}" = 1 ]; then ask repeat "$question"; fi

upload pulse-evidence 'PulseGate-limits.pptx · 导入完成 · v1' "$out/fixtures/PulseGate-limits.pptx"
courses pulse-evidence
ask table 'PulseGate-limits.pptx 第 1 页表格里的 Pulse gate 设定值是多少？'
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 "$python" docs/G/G06-probe.py --data-dir "$out" --verify
