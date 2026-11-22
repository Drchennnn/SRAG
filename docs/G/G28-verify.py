"""Replay definition-and-experiment questions in isolated Chromium workspaces."""

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
FIXTURES = HERE / "G20-fixtures"
MIXTRAL = FIXTURES / "Mixtral-of-Experts-2401.04088v1.pdf"
DPO = FIXTURES / "Direct-Preference-Optimization-2305.18290v3.pdf"
NOTEBOOK = FIXTURES / "v1/Mixtral-Router-Verification.ipynb"
PAGED = FIXTURES / "PagedAttention-2309.06180v1.pdf"
Q07 = (
    "在 Mixtral 课件和配套实验里，门控网络 G(x) 的计算公式是怎么定义的？"
    "实验在 ArXiv 验证集上测出的第 15 层连续词元首选专家重复率和平均路由分发延迟分别是多少？"
)
H04 = (
    "论文对逻辑 KV 块到物理 KV 块的块表映射是怎么定义的？"
    "在翻译和聊天机器人负载实验中，vLLM 相比现有系统取得了多少倍的吞吐量提升？"
)
CASES = {
    "Q07": ([MIXTRAL, DPO, NOTEBOOK], Q07, None),
    "H04": ([PAGED], H04, None),
    "missing": ([MIXTRAL], Q07, None),
    "mismatch": ([MIXTRAL, NOTEBOOK], Q07.replace("ArXiv", "WikiText-103"), None),
    "scoped": ([MIXTRAL, NOTEBOOK], Q07, MIXTRAL),
}


def server_code(root, env_file, port, answer_enabled):
    return f"""
import uvicorn
from pathlib import Path
from app.api.workspace import get_workspace_service
from app.config import Settings, get_settings
from app.main import create_app
from app.rag.workspace import make_workspace

root = Path({str(root)!r})
overrides = {{"workspace_answer_enabled": True}} if {answer_enabled!r} else {{}}
settings = Settings(
    _env_file={str(env_file)!r},
    course_root=root / "courses",
    data_dir=root / "data",
    sqlite_path=root / "library.sqlite3",
    workspace_course_ids="public",
    **overrides,
)
print(f"workspace_answer_enabled={{settings.workspace_answer_enabled}} "
      f"retrieval={{settings.workspace_retrieval_strategy}} model={{settings.deepseek_chat_model}}",
      flush=True)
app = create_app()
service = make_workspace(settings)
app.dependency_overrides[get_settings] = lambda: settings
app.dependency_overrides[get_workspace_service] = lambda: service
uvicorn.run(app, host="127.0.0.1", port={port}, log_level="warning")
"""


def browser_run(url, session, files, question, scoped):
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
    ab("upload", "#import-files", *map(str, files))
    ab("click", "#import-button")
    for path in files:
        ab("wait", "--text", f"{path.name} · 导入完成 · v1")
    wait("!document.querySelector('#import-files').value")
    if scoped:
        ab("find", "text", scoped.name, "click", "--exact")
        wait(f"document.querySelector('#scope-summary').textContent.includes({json.dumps(scoped.name)})")
    scope = ab("get", "text", "#scope-summary")
    ab("find", "label", "输入问题", "fill", question)
    ab("click", "#send")
    wait('document.querySelector("article.message-user")?.nextElementSibling?.dataset.messageId')
    row = js("""(async()=>{
        const msg=document.querySelector("article.message-user").nextElementSibling;
        const sid=document.querySelector("#sessions").value;
        const data=await (await fetch("/api/workspace/sessions/"+sid)).json();
        const message=data.messages.find(m=>m.message_id===msg.dataset.messageId);
        const turn=data.turns.at(-1);
        const buttons=[...new Set([...msg.querySelectorAll("button[data-evidence-id]")]
          .map(b=>b.dataset.evidenceId))];
        return {message_id:msg.dataset.messageId, visible:msg.innerText, buttons,
          blocks:message.blocks, trace:turn.trace, status:turn.status,
          sources:Object.fromEntries(Object.entries(message.sources).map(([id,s])=>[
            id,{source:s.revision.source, file:s.file_name, locator:s.locator,
              reads:s.reads.map(r=>({method:r.method,status:r.status,source:r.actual_source,
                text_length:r.text?.length}))}
          ]))};
    })()""")
    row["scope"] = scope
    row["question"] = question
    row["reader"] = {}
    msg_id = row.pop("message_id")
    for evidence_id in row["buttons"]:
        selector = f'article[data-message-id="{msg_id}"] button[data-evidence-id="{evidence_id}"]'
        js(f"document.querySelector({json.dumps(selector)}).click() || true")
        wait("""document.querySelector('#reader').open &&
            (!document.querySelector('#reader img') ||
             document.querySelector('#reader img').complete)""")
        row["reader"][evidence_id] = js("""({
            title:document.querySelector('#reader-title').innerText,
            location:document.querySelector('#reader-location').innerText,
            status:document.querySelector('#reader-status').innerText,
            image_width:document.querySelector('#reader img')?.naturalWidth ?? null,
            text:document.querySelector('#reader-body').innerText
        })""")
        ab("click", "#close-reader")
        wait("!document.querySelector('#reader').open")
    return row


def run_case(name, env_file, answer_enabled):
    files, question, scoped = CASES[name]
    port = BROWSER["_free_port"]()
    session = f"g28-{port}"
    with tempfile.TemporaryDirectory(prefix="srag-g28-") as tmp:
        root = Path(tmp)
        server = root / "server.py"
        server.write_text(server_code(root, env_file, port, answer_enabled))
        env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
        with subprocess.Popen([sys.executable, str(server)], cwd=ROOT, env=env) as proc:
            try:
                url = f"http://127.0.0.1:{port}/workspace/"
                for _ in range(150):
                    if proc.poll() is not None:
                        raise RuntimeError(f"Workspace exited with {proc.returncode}")
                    try:
                        with urllib.request.urlopen(url, timeout=1):
                            break
                    except OSError:
                        time.sleep(0.2)
                else:
                    raise RuntimeError("Workspace did not start")
                row = browser_run(url, session, files, question, scoped)
                row["fixtures"] = {
                    str(path.relative_to(FIXTURES)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in files
                }
                return row
            finally:
                proc.terminate()
                subprocess.run(
                    ["agent-browser", "--session", session, "close"],
                    check=False,
                    capture_output=True,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--enable-answer", action="store_true")
    parser.add_argument("cases", nargs="*", metavar="CASE")
    args = parser.parse_args()
    cases = args.cases or list(CASES)
    if unknown := set(cases) - CASES.keys():
        parser.error(f"unknown cases: {sorted(unknown)}")
    results = {"answer_flag_override": args.enable_answer, "cases": {}}
    for name in cases:
        results["cases"][name] = run_case(name, args.env_file.resolve(), args.enable_answer)
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
        row = results["cases"][name]
        print(name, row["trace"].get("route"), row["trace"].get("generation", {}).get("status"),
              flush=True)


if __name__ == "__main__":
    main()
