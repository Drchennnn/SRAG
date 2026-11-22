"""Play the public G26/G27 two-turn scenario against an empty real workspace.

Use --env-file for the local model configuration and --out for the JSON trace.
"""

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BROWSER = runpy.run_path(str(HERE / "G25-verify.py"))
PDF = HERE / "G20-fixtures/PagedAttention-2309.06180v1.pdf"
TABLE = "page:9:table:1"
QUESTIONS = [
    "PagedAttention 里 13B、66B、175B 三种配置分别用什么型号、多少张 GPU，"
    "最大 KV cache slots 各是多少？",
    "刚才那张表里，66B 留给 KV 缓存的显存是多少 GB？",
]


def run_browser(url, session):
    def ab(*args):
        return BROWSER["_ab"](session, *args)

    def js(expr):
        return json.loads(ab("eval", expr))

    def wait(expr):
        BROWSER["_wait_js"](session, expr)

    ab("--allowed-domains", "127.0.0.1", "open", url)
    ab("wait", "--text", "暂无资料")
    if not js("document.querySelector('#import-panel').open"):
        ab("find", "text", "导入资料", "click", "--exact")
    ab("select", "#import-course", "public")
    ab("upload", "#import-files", str(PDF))
    ab("click", "#import-button")
    ab("wait", "--text", f"{PDF.name} · 导入完成 · v1")
    js(f"""[...document.querySelectorAll('#catalog button')].find(
        b=>b.textContent.includes({json.dumps(PDF.name)})).click() || true""")
    wait(f"document.querySelector('#scope-summary').textContent.includes({json.dumps(PDF.name)})")

    rows = []
    for idx, question in enumerate(QUESTIONS, 1):
        ab("find", "label", "输入问题", "fill", question)
        ab("click", "#send")
        wait(
            f'document.querySelectorAll("article.message-user").length === {idx} && '
            f'document.querySelectorAll("article.message-user")[{idx - 1}]'
            ".nextElementSibling?.dataset.messageId"
        )
        row = js("""(async()=>{
            const last=[...document.querySelectorAll("article.message-user")].at(-1);
            const msg=last.nextElementSibling;
            const sid=document.querySelector("#sessions").value;
            const data=await (await fetch("/api/workspace/sessions/"+sid)).json();
            const message=data.messages.find(m=>m.message_id===msg.dataset.messageId);
            const t=data.turns.at(-1), buttons=msg.querySelectorAll("button[data-evidence-id]");
            const sources=Object.entries(message.sources)
              .filter(([,s])=>s.revision.source.source_id.includes(":table:"))
              .map(([id,s])=>({
              id, key:s.revision.source, reads:s.reads.map(r=>({
                method:r.method, status:r.status, source:r.actual_source,
                text:r.method==="native_table" &&
                  s.revision.source.source_id==="page:9:table:1"?r.text:undefined
              }))
            }));
            return {status:t.status,route:t.trace?.route || null, message_id:msg.dataset.messageId,
              visible:msg.innerText,
              buttons:[...new Set([...buttons].map(b=>b.dataset.evidenceId))],
              answer:message.blocks.filter(b=>["text","notice"].includes(b.type)).map(b=>b.text),
              displayed:message.blocks.filter(b=>b.type==="source_image").map(b=>b.display_source),
              sources, generation:t.trace?.generation?.status || null,
              model:t.trace?.generation?.model || null,
              attempts:t.trace?.generation?.attempts || []};
        })()""")
        reader = []
        msg_id = row.pop("message_id")
        for source in row["sources"]:
            eid = source["id"]
            if source["key"]["source_id"] != TABLE or eid not in row["buttons"]:
                continue
            selector = f'article[data-message-id="{msg_id}"] button[data-evidence-id="{eid}"]'
            js(f"document.querySelector({json.dumps(selector)}).click() || true")
            wait(
                "document.querySelector('#reader').open && "
                "document.querySelector('#reader img')?.naturalWidth > 0"
            )
            reader.append(
                js(f"""(()=>({{
              evidence_id:{json.dumps(eid)},title:document.querySelector('#reader-title').innerText,
              location:document.querySelector('#reader-location').innerText,
              size:[document.querySelector('#reader img').naturalWidth,
                document.querySelector('#reader img').naturalHeight],
              highlighted:!!document.querySelector('#reader .reader-highlight')
            }}))()""")
            )
            ab("press", "Escape")
            wait("!document.querySelector('#reader').open")
        rows.append({**row, "question": question, "reader": reader})
    return rows


def verify(row, followup):
    native = [
        read["text"]
        for source in row["sources"]
        if source["key"]["source_id"] == TABLE and source["key"]["version"] == 1
        for read in source["reads"]
        if read["method"] == "native_table"
        and read["status"] == "succeeded"
        and read["source"] == source["key"]
    ]
    expected_rows = [
        "| GPUs | A100 | 4×A100 | 8×A100-80GB |",
        "| Memory for KV cache | 12 GB | 21 GB | 264 GB |",
        "| Max. # KV cache slots | 15.7K | 9.7K | 60.1K |",
    ]
    terms = (
        ["66B", "21"] if followup else ["13B", "66B", "175B", "A100", "80GB", "15.7", "9.7", "60.1"]
    )
    answer = "".join(row["answer"]).replace(" ", "")
    return {
        "expected_route": row["route"] == ("followup_table" if followup else "generated_answer"),
        "native_rows": any(all(expected in text for expected in expected_rows) for text in native),
        "answer_terms": all(term in answer for term in terms),
        "clicked_table": any(
            PDF.name in r["title"] and "v1 · 第 9 页" in (r["title"] + r["location"])
            for r in row["reader"]
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    port = BROWSER["_free_port"]()
    session = f"g26g27-{port}"
    with tempfile.TemporaryDirectory(prefix="srag-table-gate-") as tmp:
        root = Path(tmp)
        server = root / "server.py"
        server.write_text(BROWSER["_server_script"](root, args.env_file.resolve(), "public", port))
        env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
        with subprocess.Popen([sys.executable, str(server)], cwd=ROOT, env=env) as proc:
            try:
                url = f"http://127.0.0.1:{port}/workspace/"
                for _ in range(100):
                    try:
                        with urllib.request.urlopen(url, timeout=1):
                            break
                    except OSError:
                        time.sleep(0.2)
                rows = run_browser(url, session)
                checks = [verify(row, bool(idx)) for idx, row in enumerate(rows)]
                read_keys = [
                    {
                        tuple(sorted(s["key"].items()))
                        for s in row["sources"]
                        if s["key"]["source_id"] == TABLE
                        and any(
                            r["method"] == "native_table" and r["status"] == "succeeded"
                            for r in s["reads"]
                        )
                    }
                    for row in rows
                ]
                same_table = len(read_keys[0] & read_keys[1]) == 1
                args.out.write_text(
                    json.dumps(
                        {
                            "pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
                            "turns": rows,
                            "checks": checks,
                            "same_table_across_turns": same_table,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n"
                )
                if not same_table or not all(all(items.values()) for items in checks):
                    raise SystemExit(f"Joint check failed: {checks}")
            finally:
                proc.terminate()
                subprocess.run(
                    ["agent-browser", "--session", session, "close"], capture_output=True
                )


if __name__ == "__main__":
    main()
