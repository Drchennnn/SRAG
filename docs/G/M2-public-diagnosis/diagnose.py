"""Diagnose the documented public rehearsal failures against a clean product checkout."""

from __future__ import annotations

import argparse
import base64
import json
import os
import runpy
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

BASE = "/api/workspace/sessions"
HOST = "http://127.0.0.1:18886"
DEFINITION = "Basalt latch 的定义是什么？讲义所对应的测试集实验如何验证？"


def request_summary(response):
    if response.headers.get("content-type", "").startswith("application/json"):
        return {"http": response.status_code, "data": response.json()}
    from hashlib import sha256

    return {
        "http": response.status_code,
        "sha256": sha256(response.content).hexdigest(),
        "bytes": len(response.content),
    }


def exercise_api(prep, output, env_file, files, image_hashes, *, conditions_only):
    results = {}
    cases = ("conditions",) if conditions_only else ("conditions", "replacement")
    for case in cases:
        settings = prep["settings_for"](env_file, output / case, web=False)
        app, service = prep["app_for"](settings)
        slips = []
        prep["observe_providers"](settings, slips, web=False, image_hashes=image_hashes)
        try:
            with TestClient(app, base_url=HOST, headers={"Origin": HOST}) as client:
                session = ApiSession(client, service, prep["COURSE"])
                if case == "conditions":
                    result = conditions(session, prep, files)
                else:
                    result = replacement(session, prep, files)
                results[case] = result | {
                    "effective_settings": prep["effective"](settings),
                    "provider": slips,
                    "empty": session.empty,
                }
                prep["dump"](output / "api.json", results)
        finally:
            prep["restore_providers"]()


class ApiSession:
    def __init__(self, client, service, course):
        self.client, self.service, self.course = client, service, course
        started = client.post(BASE, json={}).json()
        self.id = started["session_id"]
        self.path = f"{BASE}/{self.id}"
        self.empty = not any(c["documents"] for c in self.catalog()["courses"])

    def catalog(self):
        return self.client.get(self.path + "/catalog").json()

    def session(self):
        return self.client.get(self.path).json()

    def document(self, document_id):
        return next(
            d
            for c in self.catalog()["courses"]
            for d in c["documents"]
            if d["document_id"] == document_id
        )

    def focus_state(self, document_id):
        owner = self.client.cookies.get("srag_workspace_owner")
        saved = self.service.store.session(owner, self.id)
        visible = self.session()["focus"]
        doc = self.document(document_id)
        return {
            "persisted_focus": saved.focus_id,
            "scope_revision": saved.scope_revision,
            "selected_documents": saved.document_ids,
            "visible_focus": visible["node_id"] if visible else None,
            "catalog_node": doc["node_id"],
            "catalog_version": doc["version"],
        }

    def upload(self, path, document_id=None):
        fields = {"course_id": self.course, "filename": path.name}
        if document_id:
            fields["document_id"] = document_id
        response = self.client.post(
            self.path + "/import",
            params=fields,
            content=path.read_bytes(),
            headers={"Content-Type": "application/octet-stream"},
        )
        response.raise_for_status()
        return response.json()

    def choose(self, document_id):
        response = self.client.patch(
            self.path + "/scope",
            json={
                "course_ids": [self.course],
                "document_ids": [document_id],
                "scope_revision": self.session()["scope_revision"],
            },
        )
        response.raise_for_status()
        return self.focus(self.document(document_id)["node_id"])

    def focus(self, node):
        return request_summary(
            self.client.put(
                self.path + "/focus",
                json={"node_id": node, "scope_revision": self.session()["scope_revision"]},
            )
        )

    def ask(self, key, question):
        payload = {
            "client_turn_id": key,
            "kind": "ask",
            "text": question,
            "scope_revision": self.session()["scope_revision"],
        }
        response = self.client.post(self.path + "/turns", json=payload)
        return {"request": payload} | request_summary(response)

    def evidence(self, reply, *, preview=False):
        message = reply["data"].get("message") or {}
        ids = sorted(
            {
                token
                for block in message.get("blocks", [])
                if block["type"] == "text"
                for token in block.get("citations", [])
            }
        )
        result = {}
        for token in ids:
            endpoint = f"{self.path}/messages/{message['message_id']}/evidence/{token}"
            result[token] = request_summary(
                self.client.get(endpoint + ("/preview" if preview else ""))
            )
        return result


def conditions(api, prep, files):
    imports = [api.upload(files / prep[key]) for key in ("GUIDE", "DECK", "LAB")]
    from app.rag import evidence_supplement

    original = evidence_supplement._condition_problem
    decisions = []

    def observe(question, fields):
        result = original(question, fields)
        decisions.append({"question": question, "fields": fields.copy(), "rejection": result})
        return result

    answers = {}
    for case, question in (
        ("matching", prep["QUESTION"]),
        ("salt_fog", prep["MISSING"]),
        ("known_split_mismatch", DEFINITION),
    ):
        with patch.object(evidence_supplement, "_condition_problem", observe):
            reply = api.ask(case, question)
        answers[case] = reply | {"opened_text": api.evidence(reply)}
    return {"imports": imports, "answers": answers, "condition_decisions": decisions}


def replacement(api, prep, files):
    path = prep["PLOTS"]
    v1 = api.upload(files / "v1" / path)
    doc = v1["document_id"]
    chosen = api.choose(doc)
    first = api.focus_state(doc)
    show = api.ask("show-v1", prep["SHOW"])
    old = api.ask("vision-v1", prep["VISION"])
    old_preview = api.evidence(old, preview=True)
    v2 = api.upload(files / "v2" / path, doc)
    stale = api.focus_state(doc)
    asgi_stale = api.ask("show-stale", prep["SHOW"])
    historical = api.ask("historic-explicit", prep["VISION"])
    bare = api.ask("bare-ordinal", "第二张图，横轴的单位是什么？")
    expired = api.focus(first["catalog_node"])
    repaired = api.focus(stale["catalog_node"])
    after_rebind = api.focus_state(doc)
    after = api.ask("show-v2", prep["SHOW"])
    new_vision = api.ask("vision-v2", prep["VISION"])
    return {
        "import_v1": v1,
        "choose_v1_http": chosen["http"],
        "v1_focus": first,
        "show_v1": show,
        "vision_v1": old,
        "old_preview_before": old_preview,
        "import_v2": v2,
        "after_raw_import": stale,
        "raw_ask": asgi_stale,
        "historic_ask": historical,
        "bare_ordinal_ask": bare,
        "set_retired_focus": expired["http"],
        "retired_focus_error": expired["data"],
        "rebind_current_focus": repaired["http"],
        "after_rebind": after_rebind,
        "show_v2": after,
        "vision_v2": new_vision,
        "new_preview": api.evidence(new_vision, preview=True),
        "old_preview_after": api.evidence(old, preview=True),
    }


def serve(prep, env_file, output, port):
    import uvicorn

    settings = prep["settings_for"](env_file, output / "browser-storage", web=False)
    app, _ = prep["app_for"](settings)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def exercise_browser(prep, args, output, files):
    browser = prep["BROWSER"]
    port = browser["_free_port"]()
    name = f"m2-diag-{port}"
    url = f"http://127.0.0.1:{port}/workspace/"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--subject",
            str(args.subject),
            "--prep",
            str(args.prep),
            "--env-file",
            str(args.env_file),
            "--out",
            str(output),
            "--serve",
            str(port),
        ],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        cwd=args.subject,
    )

    def ab(*values):
        return browser["_ab"](name, *values)

    def js(expression):
        return json.loads(ab("eval", expression))

    def wait(expression):
        browser["_wait_js"](name, expression, 35_000)

    def upload(path, version):
        if not js("document.querySelector('#import-panel').open"):
            ab("find", "text", "导入资料", "click", "--exact")
        ab("select", "#import-course", prep["COURSE"])
        ab("upload", "#import-files", str(path))
        if version == 2:
            ab("check", "#replace-document")
        ab("click", "#import-button")
        ab("wait", "--text", f"{path.name} · 导入完成 · v{version}")
        wait("!document.querySelector('#import-files').files.length")

    def question(idx):
        ab("find", "label", "输入问题", "fill", prep["SHOW"])
        ab("click", "#send")
        wait(f"window.diag.filter(x=>x.url.endsWith('/turns')).length === {idx}")

    state = """(async()=>{
      const id=document.querySelector('#sessions').value;
      const session=await (await fetch('/api/workspace/sessions/'+id)).json();
      return {focus:session.focus, scope_revision:session.scope_revision, scope:session.scope};
    })()"""
    result = {}
    try:
        for _ in range(100):
            if proc.poll() is not None:
                raise RuntimeError("Workspace exited")
            try:
                with urllib.request.urlopen(url, timeout=1):
                    break
            except OSError:
                time.sleep(0.2)
        else:
            raise RuntimeError("Workspace failed to start")
        ab("--allowed-domains", "127.0.0.1", "open", url)
        ab("wait", "--text", "暂无资料")
        ab("set", "viewport", "1310", "850")
        result["empty"] = True
        js("""(()=>{
          const real=window.fetch; window.diag=[];
          window.fetch=async function(input, options) {
            const response=await real.call(this,input,options);
            const method=options?.method;
            const url=String(input);
            if (['POST','PATCH','PUT'].includes(method) &&
                url.startsWith('/api/workspace/sessions/')) {
              window.diag.push({method,url,status:response.status,
                request:typeof options.body==='string'?JSON.parse(options.body):null,
                response:await response.clone().json()});
            }
            return response;
          };
          return true;
        })()""")
        plot = prep["PLOTS"]
        upload(files / "v1" / plot, 1)
        ab("find", "role", "button", "click", "--name", f"{plot} v1", "--exact")
        wait(f"document.querySelector('#scope-summary').textContent.includes({json.dumps(plot)})")
        result["v1"] = js(state)
        question(1)
        upload(files / "v2" / plot, 2)
        result["v2_after_upload"] = js(state)
        question(2)
        result["http"] = js("window.diag")
    finally:
        proc.terminate()
        proc.wait(timeout=15)
        subprocess.run(
            ["agent-browser", "--session", name, "close"], check=False, capture_output=True
        )
        prep["dump"](output / "browser.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("subject", "prep", "env-file", "out"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--serve", type=int)
    parser.add_argument("--fixtures-from", type=Path)
    parser.add_argument("--conditions-only", action="store_true")
    args = parser.parse_args()
    args.subject, args.prep = args.subject.resolve(), args.prep.resolve()
    args.env_file, args.out = args.env_file.resolve(), args.out.resolve()
    prep = runpy.run_path(str(args.prep))
    sys.path.insert(0, str(args.subject))
    if args.serve:
        serve(prep, args.env_file, args.out, args.serve)
        return
    git = ["git", "-C", str(args.subject)]
    if subprocess.check_output([*git, "status", "--porcelain"]).strip():
        raise RuntimeError("Expected a clean product checkout")
    args.out.mkdir()
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    files = args.out / "fixtures"
    if args.fixtures_from:
        checkpoint = args.fixtures_from.resolve()
        metadata = json.loads((checkpoint / "meta.json").read_text())
        if metadata["prep_runner_sha256"] != prep["sha"](args.prep.read_bytes()):
            raise RuntimeError("Different fixture builder")
        for name, digest in metadata["fixture_sha256"].items():
            if prep["sha"]((checkpoint / "fixtures" / name).read_bytes()) != digest:
                raise RuntimeError(f"Fixture changed: {name}")
        shutil.copytree(checkpoint / "fixtures", files)
        images = {
            prep["sha"](base64.b64decode(item["data"]["image/png"]))
            for path in files.glob(f"v*/{prep['PLOTS']}")
            for cell in json.loads(path.read_text())["cells"]
            for item in cell["outputs"]
        }
    else:
        images = prep["fixtures"](files)
    prep["dump"](
        args.out / "meta.json",
        {
            "product": revision,
            "prep_runner_sha256": prep["sha"](args.prep.read_bytes()),
            "fixture_sha256": {
                str(p.relative_to(files)): prep["sha"](p.read_bytes())
                for p in files.rglob("*")
                if p.is_file()
            },
        },
    )
    exercise_api(prep, args.out, args.env_file, files, images, conditions_only=args.conditions_only)
    if not args.conditions_only:
        exercise_browser(prep, args, args.out, files)


if __name__ == "__main__":
    main()
