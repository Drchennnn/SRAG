"""Serve the public workspace with two opt-in E05 lifecycle test faults."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path
from threading import Event, Lock
from time import monotonic

import uvicorn

from app.rag.workspace import WorkspaceService

FAIL_ONCE = "E05 暂时故障后复查 median filter"
WAIT_FOR_CANCEL = "E05 等待取消 median filter"


def app_for(directory: Path):
    setup = runpy.run_path(str(Path(__file__).with_name("E04-server.py")))
    service = setup["run_workspace"](directory, directory / "turns.jsonl", None, False)
    real_retrieve = WorkspaceService._retrieve
    failed: set[str] = set()
    lock, wait = Lock(), Event()

    def retrieve(self, turn, session, text, **kwargs):
        if text == FAIL_ONCE:
            with lock:
                unmatched = turn.turn_id not in failed
                failed.add(turn.turn_id)
            if unmatched:
                raise RuntimeError("E05 controlled first-attempt failure")
        if text == WAIT_FOR_CANCEL:
            deadline = monotonic() + 25
            while monotonic() < deadline and self.store.active(turn):
                wait.wait(0.03)
            raise RuntimeError("E05 controlled response after cancellation")
        return real_retrieve(self, turn, session, text, **kwargs)

    WorkspaceService._retrieve = retrieve

    async def traced(scope, receive, send):
        path = scope.get("path", "")
        routes = ("/retry", "/cancel", "/import")
        if scope["type"] != "http" or scope["method"] != "POST" or not path.endswith(routes):
            return await service(scope, receive, send)

        raw = bytearray()
        status = 0

        async def observe(event):
            nonlocal status
            if event["type"] == "http.response.start":
                status = event["status"]
            if event["type"] == "http.response.body":
                raw.extend(event.get("body", b""))
                if not event.get("more_body"):
                    response = json.loads(raw)
                    with (directory / "actions.jsonl").open("a", encoding="utf-8") as stream:
                        record = {"path": path, "status": status, "response": response}
                        stream.write(json.dumps(record) + "\n")
            await send(event)

        return await service(scope, receive, observe)

    return traced


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    uvicorn.run(app_for(args.data_dir), host="127.0.0.1", port=args.port, lifespan="off")
