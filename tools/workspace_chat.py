"""Local terminal entry for the same persistent workspace service as HTTP."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

from app.config import get_settings
from app.rag.workspace import make_workspace
from app.schemas.workspace import CreateSession, TurnRequest, WorkspaceError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", help="existing local CLI session ID")
    parser.add_argument("--course", action="append", default=[])
    parser.add_argument("--file", action="append", default=[], metavar="DOCUMENT_ID")
    parser.add_argument("--kind", choices=["ask", "search", "summary"], default="ask")
    parser.add_argument("--json", action="store_true", help="emit the workspace JSON snapshot")
    parser.add_argument("question", nargs="?", default="")
    args = parser.parse_args()
    settings = get_settings()
    capability = settings.data_dir / "workspace_cli_owner"
    owner = capability.read_text(encoding="ascii").strip() if capability.exists() else None
    service = make_workspace(settings)
    try:
        if args.session and owner is not None:
            session = service.session(owner, args.session)
        elif args.session:
            raise WorkspaceError("not_accessible", 404)
        else:
            session, token = service.create(
                owner, CreateSession(course_ids=args.course, document_ids=args.file)
            )
            if owner is None:
                _save_owner(capability, token)
            owner = token
        if not args.question and args.kind == "ask":
            result = session
        else:
            result = service.submit(
                owner,
                session["session_id"],
                TurnRequest(
                    client_turn_id=uuid4().hex,
                    kind=args.kind,
                    text=args.question,
                    scope_revision=session["scope_revision"],
                ),
            )
    except (WorkspaceError, OSError, ValueError) as exc:
        summary = exc.code if isinstance(exc, WorkspaceError) else type(exc).__name__
        print(summary, file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif result.get("message"):
        for block in result["message"]["blocks"]:
            print(block.get("text") or block.get("caption") or "")
    else:
        print(session["session_id"])
    return int(result.get("turn", {}).get("status") == "failed")


def _save_owner(path: Path, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
        handle.write(token + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
