#!/usr/bin/env bash
# Real ordinary browser turns for a missing table and a compatible three-source comparison.
set -euo pipefail
cd "$(dirname "$0")/../.."
py="${SRAG_PYTHON:-$PWD/.venv/bin/python}"
port="${G08_PORT:-18851}"
out="${G08_OUT:-$(mktemp -d /tmp/srag-g08-browser.XXXXXX)}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
test ! -e "$out/library.sqlite3"
export G08_SUBJECT="${G08_SUBJECT:-$(git rev-parse HEAD)}"
session="$(agent-browser session id --scope worktree --prefix "${G08_BROWSER_PREFIX:-srag-g08}")"
url="http://127.0.0.1:$port/workspace/"
ab() { agent-browser --session "$session" "$@"; }
click() { ab find role button click --name "$1" --exact; }
wait_js() { ab wait --fn "$1"; }
upload() {
  local expected="$1"
  shift
  if [ "$(ab eval 'document.querySelector("#import-panel").open')" = false ]; then
    ab find text '导入资料' click --exact
  fi
  ab select '#import-course' 'pulse-evidence'
  ab upload '#import-files' "$@"
  click '开始导入'
  ab wait --text "$expected"
  wait_js '!document.querySelector("#import-button").disabled && document.querySelector("#import-files").files.length===0'
}
question='What is the definition of a pulse gate, and how did the held-out experiment validate it? Use the threshold from the separate limit table and compare it with the saved output.'
ask() {
  ab find label '输入问题' fill "$question"
  click '发送'
  wait_js "document.querySelectorAll('#messages article:not(.message-user)').length === $1"
}
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 "$py" docs/G/G05-probe.py --data-dir "$out" --port "$port" > "$out/server.log" 2>&1 &
pid=$!
trap 'ab close >/dev/null 2>&1 || true; kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
curl --silent --fail --retry 7 --retry-connrefused --connect-timeout 1 "$url" >/dev/null
ab open "$url"
ab wait --text '暂无资料'
upload 'PulseGate-run.ipynb · 导入完成 · v1' "$out/fixtures/PulseGate-lecture.ipynb" "$out/fixtures/PulseGate-run.ipynb"
click 'pulse-evidence'
wait_js 'document.querySelector("#scope-summary").textContent.includes("pulse-evidence")'
ask 1
ab wait --text '本轮缺少可核对的原生表格。'
"$py" - "$out" <<'PY'
from pathlib import Path
from sys import argv
from pptx import Presentation
path = Path(argv[1]) / "fixtures/PulseGate-limits.pptx"
deck = Presentation(path)
table = deck.slides[0].shapes[0].table
table.cell(0, 1).text = "MAE maximum (amplitude)"
table.cell(1, 1).text = "0.30"
deck.save(path)
PY
upload 'PulseGate-limits.pptx · 导入完成 · v1' "$out/fixtures/PulseGate-limits.pptx"
click 'pulse-evidence'
wait_js 'document.querySelector("#scope-summary").textContent.includes("pulse-evidence")'
ask 2
wait_js 'document.querySelector("#messages article:last-child").textContent.includes("0.30")'
ab eval '[...document.querySelectorAll("#messages article:last-child button[data-evidence-id]")].find(b=>b.textContent.includes("PulseGate-limits.pptx")).click(); true'
wait_js 'document.querySelector("#reader").open && document.querySelector("#reader-body").textContent.includes("0.30")'
ab eval '({title:document.querySelector("#reader-title").textContent, text:document.querySelector("#reader-body").textContent})' > "$out/table-reader.json"
"$py" - "$out" <<'PY'
import hashlib
import json
import os
from pathlib import Path
from sys import argv

out = Path(argv[1])
http = [json.loads(line) for line in (out / "http.jsonl").read_text().splitlines()]
turns = [v for v in http if v["category"] == "turn"]
assert len(turns) == 2
for v in turns:
    assert v["status"] == 200 and v["request"]["kind"] == "ask" and not v["request"].get("anchor")
    trace = v["response"]["turn"]["trace"]
    assert trace["route"] == "evidence_composition"
    assert trace["supplement_count"] <= 1 and trace["external_call_count"] == 0
a, b = (v["response"] for v in turns)
assert a["turn"]["trace"]["missing"] == ["table"]
assert set(b["turn"]["trace"]["supports"]) == {"principle", "table", "saved_output"}
assert b["turn"]["trace"]["comparison"] == "verified"
assert len(b["message"]["sources"]) == 3
assert any(len(block.get("citations", [])) == 2 for block in b["message"]["blocks"])
citation = next(v for v in http if v["category"] == "citation")
assert citation["status"] == 200
assert citation["response"]["source"]["source_id"] == "page:1:shape:2"
assert not (out / "provider.jsonl").exists()
payload = {
    "subject_base_or_exact": os.environ["G08_SUBJECT"],
    "product_sha256": {
        f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
        for f in ("app/rag/workspace.py", "app/rag/workspace_composition.py")
    },
    "turns": turns,
    "opened_table": citation,
    "visible_reader": json.loads((out / "table-reader.json").read_text()),
}
(out / "G08-实测.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
print(f"G08 browser verified: {out}/G08-实测.json")
PY
