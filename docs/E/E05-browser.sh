#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

e05_python="${SRAG_PYTHON:-$PWD/.venv/bin/python}"
e05_port="${E05_PORT:-18821}"
e05_out="${E05_OUT:-$(mktemp -d /tmp/srag-e05-browser.XXXXXX)}"
e05_prefix="${E05_BROWSER_PREFIX:-srag-e05-audit}"
e05_base="http://127.0.0.1:$e05_port/workspace/"
e05_a="$(agent-browser session id --scope worktree --prefix "$e05_prefix-a")"
e05_b="$(agent-browser session id --scope worktree --prefix "$e05_prefix-b")"
test "$e05_a" != "$e05_b"
mkdir -p "$e05_out"
test ! -e "$e05_out/library.sqlite3"

ab() { agent-browser --session "$@"; }
click() { ab "$1" find role button click --name "$2" --exact; }
ask() {
  ab "$1" find label "输入问题" fill "$2"
  click "$1" "发送"
}
upload() {
  ab "$1" find text "导入资料" click --exact
  ab "$1" select '#import-course' "$2"
  ab "$1" upload '#import-files' "$3"
  click "$1" "开始导入"
}
book_cell() {
  click "$1" "Signal-lab.ipynb v$2"
  ab "$1" wait --fn 'document.querySelector("#location-title").textContent === "Signal-lab.ipynb"'
  click "$1" "Median filter lab"
  ab "$1" wait --fn 'document.querySelector("#location-title").textContent === "Median filter lab"'
  click "$1" "Cell 2 · Median filter lab"
  ab "$1" wait --fn 'document.querySelector("#location-title").textContent.startsWith("Cell 2")'
}
loaded_notebook='[...document.querySelectorAll("#messages article:last-child figure img")].filter(i=>i.naturalWidth===440 && i.naturalHeight===260).length===2'

PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 "$e05_python" docs/E/E05-server.py \
  --data-dir "$e05_out" --port "$e05_port" > "$e05_out/server.log" 2>&1 &
e05_pid=$!
trap 'ab "$e05_a" close >/dev/null 2>&1 || true; ab "$e05_b" close >/dev/null 2>&1 || true; kill "$e05_pid" 2>/dev/null || true; wait "$e05_pid" 2>/dev/null || true' EXIT
curl --silent --show-error --fail --retry 7 --retry-connrefused --connect-timeout 1 "$e05_base" >/dev/null

ab "$e05_a" open "$e05_base"
ab "$e05_a" set viewport 1320 860
ab "$e05_a" wait --text "暂无资料"
ab "$e05_b" open "$e05_base"
ab "$e05_b" set viewport 1320 860
ab "$e05_b" wait --text "暂无资料"

upload "$e05_a" signals "$PWD/docs/A/样例/Signal-lab.ipynb"
ab "$e05_a" wait --text "导入完成"
book_cell "$e05_a" 1
ask "$e05_a" "展示这个 Cell 保存的两张图"
ab "$e05_a" wait --fn "$loaded_notebook"
ab "$e05_a" eval 'JSON.stringify({
  session:document.querySelector("#sessions").value,
  message:document.querySelector("#messages article:last-child").dataset.messageId,
  displays:[...document.querySelectorAll("#messages article:last-child figure")].map(x=>({
    display:x.dataset.displayId,
    ordinal:Number(x.dataset.imageOrdinal),
    path:x.querySelector("img").getAttribute("src")
  }))
})' > "$e05_out/browser-a-v1.json"

ask "$e05_a" "刚才第二张图"
ab "$e05_a" wait --text "已定位到第 2 张图"
ab "$e05_a" click '[data-image-ordinal="2"] .source-actions button:first-child'
ab "$e05_a" wait --text "引用图 2"
ask "$e05_a" "请问这个图的横轴单位"
ab "$e05_a" wait --fn 'document.querySelectorAll("#messages .message:not(.message-user)").length===3'
ask "$e05_a" "那张图"
ab "$e05_a" wait --fn 'document.querySelectorAll(".clarification-actions button").length===2'
ab "$e05_a" reload
ab "$e05_a" wait --fn 'document.querySelectorAll(".clarification-actions button").length===2'
ab "$e05_a" click '.clarification-actions button:nth-child(2)'
ab "$e05_a" wait --fn 'document.querySelector("#anchor-label").textContent.includes("第 2 张图")'
click "$e05_a" "发送"
ab "$e05_a" wait --fn 'document.querySelectorAll("#messages .message:not(.message-user)").length===5'

upload "$e05_a" signals "$PWD/docs/A/样例/Signal-reference.pdf"
ab "$e05_a" wait --fn '[...document.querySelectorAll("#catalog button")].some(b=>b.textContent.includes("Signal-reference.pdf"))'
click "$e05_a" "Signal-reference.pdf v1"
ab "$e05_a" wait --fn 'document.querySelector("#location-title").textContent === "Signal-reference.pdf"'
ab "$e05_a" find text "单文件总结" click --exact
click "$e05_a" "发送"
ab "$e05_a" wait --text "当前文件摘要"
ab "$e05_a" reload
ab "$e05_a" wait --fn 'document.querySelector("#scope-summary").textContent.includes("Signal-reference.pdf") && document.querySelector("#messages")?.textContent.includes("当前文件摘要")'
ab "$e05_a" snapshot > "$e05_out/a-switch-refresh.txt"

upload "$e05_b" isolation "$PWD/docs/A/样例/Signal-reference.pdf"
ab "$e05_b" wait --text "导入完成"
click "$e05_b" isolation
ab "$e05_b" wait --fn 'document.querySelector("#scope-summary").textContent.includes("isolation")'
ab "$e05_b" click '#catalog > ul > li:has(> button[aria-current="location"]) > ul > li > button'
ab "$e05_b" wait --fn 'document.querySelector("#location-title").textContent==="Signal-reference.pdf"'
click "$e05_b" "第 1 页 · Median Filter Reference"
ab "$e05_b" wait --fn 'document.querySelector("#location-title").textContent.startsWith("第 1 页")'
ask "$e05_b" "展示当前讲义第 1 页"
ab "$e05_b" wait --fn 'document.querySelector("#messages article:last-child img")?.naturalWidth===1224'
ab "$e05_b" eval 'JSON.stringify({
  session:document.querySelector("#sessions").value,
  message:document.querySelector("#messages article:last-child").dataset.messageId,
  displays:[...document.querySelectorAll("#messages article:last-child figure")].map(x=>({
    display:x.dataset.displayId,
    ordinal:Number(x.dataset.imageOrdinal),
    path:x.querySelector("img").getAttribute("src")
  }))
})' > "$e05_out/browser-b.json"

ask "$e05_a" "E05 暂时故障后复查 median filter"
ab "$e05_a" wait --text "重试本轮"
click "$e05_a" "重试本轮"
ab "$e05_a" wait --fn 'document.querySelectorAll("#messages .message:not(.message-user)").length===7 && !document.querySelector("#turn-actions button")'
ask "$e05_a" "E05 等待取消 median filter"
ab "$e05_a" wait --text "取消本轮"
ask "$e05_b" "另一会话并行查看第 1 页"
ab "$e05_b" wait --fn 'document.querySelectorAll("#messages .message:not(.message-user)").length===2'
click "$e05_a" "取消本轮"
ab "$e05_a" wait --text "已取消"

click "$e05_a" "Signal-lab.ipynb v1"
ab "$e05_a" wait --fn 'document.querySelector("#location-title").textContent==="Signal-lab.ipynb"'
ab "$e05_a" find text "导入资料" click --exact
ab "$e05_a" select '#import-course' signals
ab "$e05_a" upload '#import-files' "$PWD/docs/A/样例/revisions/v2/Signal-lab.ipynb"
ab "$e05_a" check '#replace-document'
click "$e05_a" "开始导入"
ab "$e05_a" wait --text "Signal-lab.ipynb · 导入完成 · v2"
book_cell "$e05_a" 2
ask "$e05_a" "查看换版后这个 Cell 的两张图"
ab "$e05_a" wait --fn "$loaded_notebook"
ab "$e05_a" reload
ab "$e05_a" wait --fn "$loaded_notebook"
ab "$e05_a" eval 'JSON.stringify({
  session:document.querySelector("#sessions").value,
  message:document.querySelector("#messages article:last-child").dataset.messageId,
  displays:[...document.querySelectorAll("#messages article:last-child figure")].map(x=>({
    display:x.dataset.displayId,
    ordinal:Number(x.dataset.imageOrdinal),
    path:x.querySelector("img").getAttribute("src")
  }))
})' > "$e05_out/browser-a-v2.json"
ab "$e05_a" click '#messages article:first-child + article [data-image-ordinal="2"] .source-actions button:nth-child(2)'
ab "$e05_a" wait --fn 'document.querySelector("#reader img")?.naturalWidth===440 && document.querySelector("#reader-title").textContent.includes("v1")'
ab "$e05_a" snapshot -s '#reader' > "$e05_out/a-old-reader.txt"
click "$e05_a" "关闭原件"

compare() {
  "$e05_python" - "$1" "$2" "${3:-}" <<'PY'
import json
import sys
from pathlib import Path

def decode(path):
    value = json.loads(Path(path).read_text())
    return json.loads(value) if isinstance(value, str) else value

own, other = (decode(path) for path in sys.argv[1:3])
old = decode(sys.argv[3]) if sys.argv[3] else own
args = json.dumps([
    f'/api/workspace/sessions/{own["session"]}', own["displays"][-1]["path"],
    f'/api/workspace/sessions/{other["session"]}', other["displays"][-1]["path"],
    old["displays"][-1]["path"],
])
print("""(async () => {
  const [own, current, foreign, foreignMedia, old] = %s;
  const records = await Promise.all([own, current, foreign, foreignMedia, old].map(
    path => fetch(path, {cache:"no-store"})));
  async function content(response) {
    if (!response.ok) return null;
    const array = new Uint8Array(await crypto.subtle.digest("SHA-256", await response.arrayBuffer()));
    return [...array].map(value=>value.toString(16).padStart(2,"0")).join("");
  }
  return JSON.stringify({
    own:records[0].status, current:records[1].status, foreign:records[2].status,
    foreign_media:records[3].status, old:records[4].status,
    current_sha256:await content(records[1]), old_sha256:await content(records[4]),
    snapshot:await records[0].json()
  });
})()""" % args)
PY
}
ab "$e05_a" eval "$(compare "$e05_out/browser-a-v2.json" "$e05_out/browser-b.json" "$e05_out/browser-a-v1.json")" > "$e05_out/check-a.json"
ab "$e05_b" reload
ab "$e05_b" wait --fn 'document.querySelectorAll("#messages .message:not(.message-user)").length===2 && document.querySelector("#scope-summary").textContent.includes("Signal-reference.pdf")'
ab "$e05_b" eval "$(compare "$e05_out/browser-b.json" "$e05_out/browser-a-v1.json")" > "$e05_out/check-b.json"

WORKSPACE_COURSE_IDS=signals,isolation SOURCE_VISION_ENABLED=false DASHSCOPE_API_KEY= \
DEEPSEEK_API_KEY= DATA_DIR="$e05_out/data" SQLITE_PATH="$e05_out/library.sqlite3" \
COURSE_ROOT="$e05_out/courses" PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 "$e05_python" \
  -m tools.workspace_chat --course isolation --kind search --json "lookahead" \
  > "$e05_out/terminal.json"
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 "$e05_python" -m tools.chat --help > "$e05_out/legacy-help.txt"

"$e05_python" - "$e05_out" <<'PY'
import base64
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

out = Path(sys.argv[1])

def read(name):
    value = json.loads((out / name).read_text())
    return json.loads(value) if isinstance(value, str) else value

def lines(name):
    return [json.loads(row) for row in (out / name).read_text().splitlines()]

def sha(data):
    return hashlib.sha256(data).hexdigest()

def original_png(relative_path):
    source = json.loads((Path("docs/A/样例") / relative_path).read_text())
    return sha(base64.b64decode("".join(source["cells"][1]["outputs"][2]["data"]["image/png"])))

def row_for(text, anchor=None):
    return next(
        row for row in turns
        if row["request"]["text"] == text
        and (anchor is None or row["request"].get("anchor", {}).get("type") == anchor)
    )

a1, a2, b = (read(file) for file in ("browser-a-v1.json", "browser-a-v2.json", "browser-b.json"))
a_check, b_check = read("check-a.json"), read("check-b.json")
a, b_state = a_check.pop("snapshot"), b_check.pop("snapshot")
assert a1["session"] == a2["session"] == a["session_id"] != b["session"] == b_state["session_id"]
with sqlite3.connect(out / "library.sqlite3") as db:
    owners = db.execute(
        "SELECT COUNT(DISTINCT owner_digest) FROM workspace_sessions WHERE session_id IN (?,?)",
        (a["session_id"], b_state["session_id"]),
    ).fetchone()[0]
assert owners == 2
for check in (a_check, b_check):
    assert (check["own"], check["current"], check["foreign"], check["foreign_media"]) == (200, 200, 404, 404)
assert a_check["old"] == 200
assert a_check["old_sha256"] == original_png("Signal-lab.ipynb")
assert a_check["current_sha256"] == original_png("revisions/v2/Signal-lab.ipynb")
assert a_check["old_sha256"] != a_check["current_sha256"]

turns, actions = lines("turns.jsonl"), lines("actions.jsonl")
assert len(turns) == 11 and len(actions) == 6
assert all(row["status"] == 200 for row in [*turns, *actions])
imports = [row["response"] for row in actions if row["path"].endswith("/import")]
assert [row["version"] for row in imports] == [1, 1, 1, 2]
assert imports[0]["document_id"] == imports[3]["document_id"]
notebook_id, pdf_a, pdf_b = [row["document_id"] for row in imports[:3]]
assert pdf_a != pdf_b
assert a["scope"] == {"course_ids": ["signals"], "document_ids": [notebook_id]}
assert b_state["scope"] == {"course_ids": ["isolation"], "document_ids": [pdf_b]}
for number, (browser, question) in enumerate(
    [(a1, "展示这个 Cell 保存的两张图"), (a2, "查看换版后这个 Cell 的两张图")], 1
):
    message = row_for(question)["response"]["message"]
    figures = [block for block in message["blocks"] if block["type"] == "source_image"]
    assert message["message_id"] == browser["message"]
    assert message["visible_display_ids"] == [entry["display"] for entry in browser["displays"]]
    assert [entry["ordinal"] for entry in browser["displays"]] == [1, 2]
    assert [figure["display_id"] for figure in figures] == message["visible_display_ids"]
    assert [figure["display_source"] for figure in figures] == [
        {"document_id": notebook_id, "version": number, "source_id": f"cell:1:output:{index}"}
        for index in (1, 2)
    ]
    assert all(figure["read_status"] == "not_attempted" for figure in figures)

plain = row_for("刚才第二张图")
card = row_for("请问这个图的横轴单位")
unclear = next(row for row in turns if row["response"]["turn"]["status"] == "clarification")
chosen = row_for("那张图", anchor="clarification")
assert set(plain["request"]) == {"client_turn_id", "kind", "text", "scope_revision"}
second = a1["displays"][1]["display"]
assert card["request"]["anchor"] == {"type": "display", "message_id": a1["message"], "display_id": second}
options = unclear["response"]["options"]
assert len(options) == 2
assert chosen["request"]["anchor"]["option_id"] == options[1]["option_id"]
for row, method in ((plain, "visible_ordinal"), (card, "display_anchor"), (chosen, "clarification_choice")):
    turn = row["response"]["turn"]
    resolution = turn["resolution"]
    assert (resolution["method"], resolution["message_id"], resolution["display_id"], resolution["ordinal"]) == (
        method, a1["message"], second, 2
    )
    assert resolution["actual_source"] == {"document_id": notebook_id, "version": 1, "source_id": "cell:1:output:2"}
    assert turn["trace"]["external_call_count"] == 0
    assert [item["read_status"] for item in row["response"]["message"]["sources"].values()] == ["not_attempted"]

summary = row_for("总结当前资料")["response"]["turn"]["trace"]
assert summary["effective_scope"] == {"course_ids": ["signals"], "document_ids": [pdf_a]}
assert "当前文件摘要" in (out/"a-switch-refresh.txt").read_text()
assert "Signal-lab.ipynb · v1 · 输出 2" in (out/"a-old-reader.txt").read_text()

failed = row_for("E05 暂时故障后复查 median filter")["response"]["turn"]
retried = next(row["response"]["turn"] for row in actions if row["path"].endswith("/retry"))
cancelled = next(row["response"]["turn"] for row in actions if row["path"].endswith("/cancel"))
late = row_for("E05 等待取消 median filter")["response"]
assert (failed["status"], failed["error"], failed["retryable"]) == ("failed", "temporarily_unavailable", True)
assert (retried["status"], retried["turn_id"], retried["client_turn_id"]) == (
    "completed", failed["turn_id"], failed["client_turn_id"]
)
assert cancelled["status"] == late["turn"]["status"] == "cancelled"
assert cancelled["turn_id"] == late["turn"]["turn_id"] and late["message"] is None
for turn, expected in ((retried, ["user", "assistant"]), (cancelled, ["user"])):
    assert [m["role"] for m in a["messages"] if m["turn_id"] == turn["turn_id"]] == expected
parallel = row_for("另一会话并行查看第 1 页")["response"]["turn"]
assert parallel["status"] == "completed"
assert parallel["trace"]["effective_scope"] == b_state["scope"]
terminal = read("terminal.json")["turn"]
assert terminal["status"] == "completed" and terminal["trace"]["effective_scope"] == b_state["scope"]
assert "usage:" in (out/"legacy-help.txt").read_text().lower()

def followup(row):
    turn = row["response"]["turn"]
    return {
        "request": row["request"], "resolution": turn["resolution"],
        "external_call_count": turn["trace"]["external_call_count"],
        "read_status": "not_attempted",
    }

evidence = {
    "runtime": "真实 Chromium 双会话，真实 localhost FastAPI，视觉服务关闭",
    "owners": owners,
    "request_counts": {"turns_200": len(turns), "other_actions_200": len(actions)},
    "visible_order_v1": a1,
    "visible_order_v2": a2,
    "plain": followup(plain),
    "card": followup(card),
    "clarification": {"options_after_refresh": len(options), **followup(chosen)},
    "owner_a": {"access": a_check, "restored_scope": a["scope"]},
    "owner_b": {"access": b_check, "restored_scope": b_state["scope"], "turns": len(b_state["turns"])},
    "summary_scope": summary["effective_scope"],
    "version_switch": {"document_id": notebook_id, "versions": [1, 2], "history_preserved": True},
    "controlled_faults": {
        "first_attempt": failed, "retry": retried["status"], "same_ids": True,
        "retry_user_messages": 1, "cancellation": cancelled["status"], "cancelled_assistant_messages": 0,
        "owner_b_completed_during_owner_a_inflight": True,
    },
    "terminal": {"status": terminal["status"], "scope": b_state["scope"], "legacy_help": True},
}
(out/"public-evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+"\n")
print("E05 browser verified")
PY
printf 'E05 replay artifacts %s\n' "$e05_out"
