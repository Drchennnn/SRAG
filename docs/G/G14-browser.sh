#!/usr/bin/env bash
# Reproduce both G14 failures in Chromium on the exact G13 publication.
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
suite="$repo/docs/G"
python="${SRAG_PYTHON:-$repo/.venv/bin/python}"
subject=82c5a40d3de6224c238ab461bc5d670c5005ba39
port="${G14_PORT:-19144}"
out="${G14_OUT:-$(mktemp -d /tmp/srag-g14-browser.XXXXXX)}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
test ! -e "$out/checkout"
mkdir "$out/checkout"
git -C "$repo" archive "$subject" | tar -x -C "$out/checkout"
cd "$out/checkout"
"$python" "$suite/G14-fixtures.py" "$out"
session="$(agent-browser session id --scope worktree --prefix "g14-baseline-$port")"
url="http://127.0.0.1:$port/workspace/"
ab() { agent-browser --session "$session" "$@"; }
click() { ab find role button click --name "$1" --exact; }
wait_js() { ab wait --fn "$1"; }

PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 "$python" "$suite/G14-browser-probe.py" \
  "$out" --port "$port" > "$out/server.log" 2>&1 &
pid=$!
cleanup() {
  ab close >/dev/null 2>&1 || true
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}
trap cleanup EXIT
printf 'G14 output %s\n' "$out"
curl --silent --fail --retry 6 --retry-connrefused --connect-timeout 1 "$url" >/dev/null

ab --allowed-domains 127.0.0.1 open "$url"
ab set viewport 1440 1080
ab wait --text '暂无资料'
ab eval 'navigator.userAgent' > "$out/browser.json"
ab find text '导入资料' click --exact
ab select '#import-course' northpoint-completion
ab upload '#import-files' "$out/Northpoint-Completion-Briefing.pdf"
click '开始导入'
ab wait --text 'Northpoint-Completion-Briefing.pdf · 导入完成 · v1'
click 'Northpoint-Completion-Briefing.pdf v1'
wait_js 'document.querySelector("#location-title").textContent === "Northpoint-Completion-Briefing.pdf"'

ask() {
  local number="$1" question
  question="$("$python" -c 'import runpy,sys; print(runpy.run_path(sys.argv[1])["QUESTIONS"][int(sys.argv[2])-1])' "$suite/G14-fixtures.py" "$number")"
  ab find label '输入问题' fill "$question"
  click '发送'
  wait_js "document.querySelectorAll('#messages article:not(.message-user)').length === $number"
  ab eval '(() => {
    const message = document.querySelector("#messages article:last-child");
    return {message_id: message.dataset.messageId, focus:document.querySelector("#location-title").textContent,
      text:message.textContent, citations:[...message.querySelectorAll("button[data-evidence-id]")].map(x=>x.textContent)};
  })()' > "$out/turn-$number.json"
  ab click '#messages article:last-child button[data-evidence-id="ev-1"]'
  wait_js 'document.querySelector("#reader").open && document.querySelector("#reader-title").textContent.includes("v1")'
  click '关闭原件'
}

ask 1
ask 2
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 "$python" "$suite/G14-browser-probe.py" "$out" --verify
printf '%s\n' "$out/G14-browser-baseline.json"
