"""Serve the browser workspace against the signed E01 wire shape, without SQLite or providers."""

import argparse
import base64
import json
from pathlib import Path
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "docs" / "A" / "样例" / "Signal-lab.ipynb"
COURSE = "signal"
DOCUMENT = "fixture-signal-notebook"


def catalog_document(filename):
    def base(number, kind, label, **fields):
        return {
            "node_id": number,
            "label": label,
            "kind": kind,
            "document_id": DOCUMENT,
            "version": 1,
            "has_media": False,
            "children": [],
        } | fields

    def output(number, image):
        return base(
            f"output-{number}",
            "notebook_output",
            f"输出 {number}",
            parent_id="cell-1",
            has_media=True,
            locator={"cell_index": 1, "output_index": number},
            previous_id="output-1" if number == 2 else None,
            next_id="output-2" if image == 1 else None,
        )

    return base(
        "document-1",
        "document",
        filename,
        file_name=filename,
        nodes=[
            base(
                "cell-0",
                "cell",
                "Cell 0 · lab-intro",
                parent_id="document-1",
                next_id="cell-1",
                locator={"cell_index": 0},
                text="Signal laboratory introduction",
            ),
            base(
                "cell-1",
                "cell",
                "Cell 1 · signal-plots",
                parent_id="document-1",
                previous_id="cell-0",
                locator={"cell_index": 1},
                children=[output(1, 1), output(2, 2)],
            ),
        ],
    )


def source(item, number):
    key = {"document_id": DOCUMENT, "version": 1, "source_id": f"cell:1:output:{number}"}
    return {
        "evidence_id": item,
        "revision": {"source": key, "course_id": COURSE},
        "file_name": "Signal-lab.ipynb",
        "read_status": "not_attempted",
        "locator": {"cell_index": 1, "output_index": number},
        "reads": [],
    }


def message(role, blocks, **extra):
    return {
        "message_id": uuid4().hex,
        "role": role,
        "blocks": blocks,
        "sources": {},
        "visible_display_ids": [],
    } | extra


def text_block(text):
    return {"type": "text", "text": text, "citations": []}


def make_app(trace_file=None):
    app = FastAPI()
    state = {"document": None, "sessions": {}, "events": []}
    notebook = json.loads(NOTEBOOK.read_text())
    images = {
        str(number): base64.b64decode(notebook["cells"][1]["outputs"][number]["data"]["image/png"])
        for number in (1, 2)
    }

    def record(event, **fields):
        state["events"].append({"event": event} | fields)
        if trace_file:
            trace_file.write_text(
                json.dumps(state["events"], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    def session_for(sid, request):
        entry = state["sessions"].get(sid)
        if not entry or entry["owner"] != request.cookies.get("fixture-owner"):
            raise HTTPException(404, "not_accessible")
        return entry["session"]

    def find_node(node_id):
        document = state["document"]
        if not document:
            raise HTTPException(404, "not_accessible")

        def visit(node):
            if node["node_id"] == node_id:
                return node
            for child in node.get("nodes", node["children"]):
                found = visit(child)
                if found:
                    return found
            return None

        result = visit(document)
        if not result:
            raise HTTPException(404, "not_accessible")
        return result

    def publish(session, turn, result):
        result["turn_id"] = turn["turn_id"]
        turn["message_id"] = result["message_id"]
        session["messages"].append(result)
        return {"turn": turn, "message": result, "options": turn.get("options", [])}

    def render_images(session, turn):
        result = message("assistant", [text_block("找到两张保存的图。模型离线，图像仅供浏览。")])
        for number in (1, 2):
            evidence_id = f"ev-{number}"
            evidence = source(evidence_id, number)
            display_id = f"d{number}"
            base = f"/api/workspace/sessions/{session['session_id']}/messages"
            result["blocks"].append(
                {
                    "type": "source_image",
                    "display_id": display_id,
                    "evidence_id": evidence_id,
                    "display_source": evidence["revision"]["source"],
                    "rendition_kind": "notebook_output",
                    "caption": f"保存输出 {number}",
                    "media_url": f"{base}/{result['message_id']}/media/{display_id}?view=inline",
                    "read_status": "not_attempted",
                    "rect_visible_norm": None,
                }
            )
            result["visible_display_ids"].append(display_id)
            result["sources"][evidence_id] = evidence
            if number == 1:
                result["blocks"].append({"type": "notice", "text": "图像占位失败，未计入图序。"})
        return publish(session, turn, result)

    @app.get("/api/workspace/sessions")
    def list_sessions(request: Request):
        owner = request.cookies.get("fixture-owner")
        if not owner:
            raise HTTPException(404, "not_accessible")
        return {
            "sessions": [
                entry["session"] for entry in state["sessions"].values() if entry["owner"] == owner
            ]
        }

    @app.post("/api/workspace/sessions")
    def create_session(request: Request, response: Response):
        owner = request.cookies.get("fixture-owner") or uuid4().hex
        number = len(state["sessions"]) + 1
        sid = f"fixture-session-{number}"
        session = {
            "session_id": sid,
            "title": f"复习会话 {number}",
            "scope": {"course_ids": [COURSE], "document_ids": []},
            "scope_revision": 1,
            "focus": None,
            "turns": [],
            "messages": [],
        }
        state["sessions"][sid] = {"owner": owner, "session": session}
        response.set_cookie("fixture-owner", owner, httponly=True, samesite="strict")
        return session

    @app.get("/api/workspace/sessions/{sid}")
    def get_session(sid: str, request: Request):
        return session_for(sid, request)

    @app.get("/api/workspace/sessions/{sid}/catalog")
    def get_catalog(sid: str, request: Request):
        session = session_for(sid, request)
        documents = [state["document"]] if state["document"] else []
        return {
            "courses": [{"course_id": COURSE, "label": "信号处理", "documents": documents}],
            "scope_revision": session["scope_revision"],
        }

    @app.patch("/api/workspace/sessions/{sid}/scope")
    def update_scope(sid: str, request: Request, payload: dict):
        session = session_for(sid, request)
        if payload["scope_revision"] != session["scope_revision"]:
            raise HTTPException(409, "scope_changed")
        session["scope"] = {
            "course_ids": payload["course_ids"],
            "document_ids": payload["document_ids"],
        }
        session["scope_revision"] += 1
        session["focus"] = None
        record("scope", payload=payload)
        return session

    @app.put("/api/workspace/sessions/{sid}/focus")
    def update_focus(sid: str, request: Request, payload: dict):
        session = session_for(sid, request)
        session["focus"] = find_node(payload["node_id"]) if payload["node_id"] else None
        record("focus", payload=payload)
        return session

    @app.post("/api/workspace/sessions/{sid}/import")
    async def upload(sid: str, request: Request, course_id: str, filename: str):
        session_for(sid, request)
        raw = await request.body()
        content_type = request.headers.get("content-type")
        if course_id != COURSE or content_type != "application/octet-stream" or not raw:
            raise HTTPException(422, "invalid_request")
        state["document"] = catalog_document(filename)
        record("import", filename=filename, bytes=len(raw), content_type=content_type)
        return {"status": "published", "document_id": DOCUMENT, "version": 1}

    @app.post("/api/workspace/sessions/{sid}/turns")
    def create_turn(sid: str, request: Request, response: Response, payload: dict):
        session = session_for(sid, request)
        turn = {"turn_id": uuid4().hex, "status": "completed", "retryable": False} | payload
        session["turns"].append(turn)
        session["messages"].append(message("user", [text_block(payload["text"])]))
        record("turn", payload=payload)
        text = payload["text"]
        anchor = payload.get("anchor")
        if text == "等待演练":
            turn["status"] = "running"
            response.status_code = 202
            return {"turn": turn, "message": None, "options": []}
        if text == "失败演练":
            turn.update(status="failed", retryable=True, error="interrupted")
            return {"turn": turn, "message": None, "options": []}
        if text == "那张图" and not anchor:
            turn["status"] = "clarification"
            turn["options"] = [
                {"option_id": "option-1", "label": "选择图 1", "kind": "display"},
                {"option_id": "option-2", "label": "选择图 2", "kind": "display"},
            ]
            return publish(
                session, turn, message("assistant", [text_block("请选择要追问的图像。")])
            )
        if anchor and anchor["type"] == "clarification":
            reply = f"已确认澄清选项 {anchor['option_id']}。"
        elif anchor and anchor["type"] == "display":
            reply = f"已定位卡片 {anchor['display_id']}，模型未读取。"
        elif text == "刚才第二张图":
            previous = next(
                item for item in reversed(session["messages"]) if item["visible_display_ids"]
            )
            reply = f"已按消息显示顺序定位 {previous['visible_display_ids'][1]}，模型未读取。"
        elif payload["kind"] == "search":
            reply = "检索结果 · Signal-lab.ipynb"
        elif payload["kind"] == "summary":
            reply = "当前单文件总结 · 信号实验资料"
        else:
            return render_images(session, turn)
        return publish(session, turn, message("assistant", [text_block(reply)]))

    @app.post("/api/workspace/sessions/{sid}/turns/{client_id}/{action}")
    def change_turn(sid: str, client_id: str, action: str, request: Request):
        session = session_for(sid, request)
        turn = next(item for item in session["turns"] if item["client_turn_id"] == client_id)
        record(action, client_turn_id=client_id)
        if action == "retry":
            turn.update(status="completed", retryable=False, error=None)
            return publish(session, turn, message("assistant", [text_block("重试已完成。")]))
        turn["status"] = "cancelled"
        return {"turn": turn, "message": None, "options": []}

    @app.get("/api/workspace/sessions/{sid}/messages/{message_id}/media/{display_id}")
    def message_media(sid: str, message_id: str, display_id: str, request: Request):
        session = session_for(sid, request)
        entry = next(
            (item for item in session["messages"] if item["message_id"] == message_id), None
        )
        if not entry or display_id not in entry["visible_display_ids"]:
            raise HTTPException(404, "not_accessible")
        return Response(images[display_id[-1]], media_type="image/png")

    @app.get("/api/workspace/sessions/{sid}/catalog/{node_id}/media")
    def catalog_media(sid: str, node_id: str, request: Request):
        session_for(sid, request)
        if not find_node(node_id)["has_media"]:
            raise HTTPException(404, "not_accessible")
        record("preview", node_id=node_id)
        return Response(images[node_id[-1]], media_type="image/png")

    app.mount("/workspace", StaticFiles(directory=ROOT / "app" / "web" / "workspace", html=True))
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8803)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    uvicorn.run(make_app(args.trace), host="127.0.0.1", port=args.port)
