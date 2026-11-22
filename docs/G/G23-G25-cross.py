"""Replay a public two-version PDF through the real workspace and vision provider."""

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

import pypdfium2 as pdfium

HERE = Path(__file__).resolve().parent
browser = runpy.run_path(str(HERE / "G25-verify.py"))
charts = runpy.run_path(str(HERE / "G25-fixtures.py"))
FILENAME = "Directional-Spectrum.pdf"


def fixture(root, version, unit):
    path = root / str(version) / FILENAME
    path.parent.mkdir()
    doc = pdfium.PdfDocument.new()
    page = doc.new_page(charts["PAGE_WIDTH_PT"], charts["PAGE_HEIGHT_PT"])
    charts["_add_pdf_text"](doc, page, "Directional spectrum, CC BY 4.0", 24, 700)
    for label, box in [("s", (0.06, 0.28, 0.40, 0.64)), (unit, (0.43, 0.28, 0.88, 0.64))]:
        plot = charts["_draw_chart"](
            title="Spectrum",
            x_label=f"Frequency [{label}]",
            y_label="Magnitude [dB]",
            x_ticks=("10", "20", "30"),
            y_ticks=("1", "2", "3"),
            points=[(0.1, 0.2), (0.4, 0.8), (0.9, 0.5)],
            color="#19619a",
        )
        charts["_add_pdf_image"](doc, page, plot, box)
    page.gen_content()
    doc.save(path)
    doc.close()
    return path


def run(root, env_file, output, brief):
    port = browser["_free_port"]()
    session = f"g2325-{port}"

    def ab(*args):
        return browser["_ab"](session, *args)

    def js(expr):
        return json.loads(ab("eval", expr))

    def wait(expr):
        browser["_wait_js"](session, expr)

    v1, v2 = fixture(root, 1, "Hz"), fixture(root, 2, "kHz")
    server = root / "server.py"
    server.write_text(browser["_server_script"](root, env_file, "public", port))
    base = f"http://127.0.0.1:{port}/workspace/"
    env = {**os.environ, "PYTHONPATH": str(HERE.parents[1]), "PYTHONDONTWRITEBYTECODE": "1"}
    with subprocess.Popen([sys.executable, str(server)], cwd=HERE.parents[1], env=env) as proc:
        try:
            for _ in range(100):
                try:
                    with urllib.request.urlopen(base, timeout=1):
                        break
                except OSError:
                    time.sleep(0.2)
            ab("--allowed-domains", "127.0.0.1", "open", base)
            ab("wait", "--text", "暂无资料")
            rows = []
            for idx, (path, question) in enumerate(
                [
                    (v1, "第1页右图的横轴单位是什么？"),
                    (
                        v2,
                        "跟上一版比呢？"
                        if brief
                        else "跟上一版相比，第1页右图的横轴单位有什么变化？",
                    ),
                ],
                1,
            ):
                if not js('document.querySelector("#import-panel").open'):
                    ab("find", "text", "导入资料", "click", "--exact")
                ab("select", "#import-course", "public")
                if idx == 2:
                    ab("click", "#replace-document")
                ab("upload", "#import-files", str(path))
                ab("click", "#import-button")
                ab("wait", "--text", f"{FILENAME} · 导入完成 · v{idx}")
                js(
                    '[...document.querySelectorAll("#catalog button")]'
                    f'.find(b=>b.textContent.includes("{FILENAME}")).click() || true'
                )
                ab("find", "label", "输入问题", "fill", question)
                ab("click", "#send")
                wait(
                    f'document.querySelectorAll("article.message-user").length === {idx} && '
                    f'document.querySelectorAll("article.message-user")[{idx - 1}]'
                    ".nextElementSibling?.dataset.messageId && "
                    '!document.querySelector("#turn-actions button")'
                )
                row = js("""(async()=>{
                  const users=document.querySelectorAll("article.message-user");
                  const m=[...users].at(-1).nextElementSibling;
                  const sid=document.querySelector("#sessions").value;
                  const data=await (await fetch("/api/workspace/sessions/"+sid)).json();
                  const t=data.turns.at(-1);
                  const msg=data.messages.find(x=>x.message_id===m.dataset.messageId);
                  return {route:t.trace.route,text:m.innerText,id:m.dataset.messageId,
                    sources:msg.sources, images:msg.blocks.filter(b=>b.type==="source_image")};
                })()""")
                row["question"] = question
                row["citations"] = []
                for image in row["images"]:
                    eid = image["evidence_id"]
                    ab(
                        "click",
                        f'article[data-message-id="{row["id"]}"] button[data-evidence-id="{eid}"]',
                    )
                    wait(
                        'document.querySelector("#reader").open && '
                        'document.querySelector("#reader img")?.complete && '
                        'document.querySelector("#reader img").naturalWidth>0'
                    )
                    reader = js("""(async()=>{
                        const r=document.querySelector("#reader");
                        const image=await fetch(r.querySelector("img").src);
                        if (!image.ok) throw Error("media fetch failed");
                        const b=await image.arrayBuffer();
                        const digest=await crypto.subtle.digest("SHA-256",b);
                        const hex=[...new Uint8Array(digest)]
                            .map(x=>x.toString(16).padStart(2,"0")).join("");
                        return {title:document.querySelector("#reader-title").innerText,
                            location:document.querySelector("#reader-location").innerText,
                            status:document.querySelector("#reader-status").innerText,digest:hex};
                    })()""")
                    row["citations"].append(reader)
                    ab("press", "Escape")
                rows.append(row)
                output.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
            wire = [
                json.loads(s)["wire_image_sha256"]
                for s in (root / "provider.jsonl").read_text().splitlines()
                if "wire_image_sha256" in s
            ]
            if brief:
                print(
                    json.dumps(
                        {
                            "second_turn": rows[-1]["text"],
                            "route": rows[-1]["route"],
                            "wire_count": len(wire),
                        },
                        ensure_ascii=False,
                    )
                )
            assert len(wire) == 2 and wire[0] != wire[1]
            assert [r["route"] for r in rows] == ["target_object", "version_comparison"]
            assert "kHz" in rows[1]["text"] and "Hz" in rows[0]["text"]
            for row, versions in zip(rows, [[1], [1, 2]], strict=True):
                assert len(row["images"]) == len(versions)
                for image, version, reader in zip(
                    row["images"], versions, row["citations"], strict=True
                ):
                    ev = row["sources"][image["evidence_id"]]
                    key = ev["revision"]["source"]
                    assert key["version"] == version and key["source_id"] == "page:1:image:2"
                    assert any(r["actual_input_sha256"] == wire[version - 1] for r in ev["reads"])
                    assert reader["digest"] == wire[version - 1]
                    assert f"v{version} · 第 1 页" in reader["location"]
            print(
                json.dumps(
                    {
                        "wire_sha256": wire,
                        "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    }
                )
            )
        finally:
            proc.terminate()
            ab("close")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--brief", action="store_true", help="Record an elliptical follow-up")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        run(Path(tmp), args.env_file.resolve(), args.out.resolve(), args.brief)
