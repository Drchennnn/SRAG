"""Durable local workspace state sharing B's publication transaction."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from sqlite3 import Connection, Row
from time import time
from typing import Any
from uuid import uuid4

from app.schemas.workspace import TurnRequest, WorkspaceError, inaccessible
from app.stores.local_visibility import promote_due
from app.stores.sqlite import MetadataStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS workspace_sessions (
    session_id TEXT PRIMARY KEY,
    owner_digest TEXT NOT NULL,
    title TEXT NOT NULL,
    course_ids TEXT NOT NULL,
    document_ids TEXT NOT NULL,
    scope_revision INTEGER NOT NULL DEFAULT 1,
    focus_id TEXT,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS workspace_session_owner
ON workspace_sessions(owner_digest, updated_at);
CREATE TABLE IF NOT EXISTS workspace_turns (
    turn_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES workspace_sessions(session_id),
    client_turn_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN (
        'running', 'completed', 'clarification', 'failed', 'cancelled'
    )),
    attempt TEXT NOT NULL,
    lease_until REAL NOT NULL,
    outcome TEXT NOT NULL DEFAULT '{}',
    UNIQUE(session_id, client_turn_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS workspace_running_session
ON workspace_turns(session_id) WHERE status = 'running';
CREATE TABLE IF NOT EXISTS workspace_messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES workspace_sessions(session_id),
    turn_id TEXT NOT NULL REFERENCES workspace_turns(turn_id),
    ordinal INTEGER NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    payload TEXT NOT NULL,
    UNIQUE(session_id, ordinal),
    UNIQUE(turn_id, role)
);
CREATE TABLE IF NOT EXISTS workspace_media (
    message_id TEXT NOT NULL REFERENCES workspace_messages(message_id),
    display_id TEXT NOT NULL,
    description TEXT NOT NULL,
    image BLOB NOT NULL,
    PRIMARY KEY(message_id, display_id)
);
"""

_OWNER = re.compile(r"[A-Za-z0-9_-]{43}")
_LEASE_SECONDS = 300


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def owner_digest(owner: str | None) -> str:
    if not owner or _OWNER.fullmatch(owner) is None:
        raise inaccessible()
    return hashlib.sha256(owner.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class Session:
    session_id: str
    title: str
    course_ids: list[str]
    document_ids: list[str]
    scope_revision: int
    focus_id: str | None

    @classmethod
    def from_row(cls, row: Row) -> Session:
        return cls(
            row["session_id"],
            row["title"],
            json.loads(row["course_ids"]),
            json.loads(row["document_ids"]),
            row["scope_revision"],
            row["focus_id"],
        )


@dataclass(frozen=True)
class Turn:
    turn_id: str
    session_id: str
    request: TurnRequest
    status: str
    attempt: str
    outcome: dict[str, Any]

    @classmethod
    def from_row(cls, row: Row) -> Turn:
        return cls(
            row["turn_id"],
            row["session_id"],
            TurnRequest.model_validate_json(row["payload"]),
            row["status"],
            row["attempt"],
            json.loads(row["outcome"]),
        )


@dataclass(frozen=True)
class Attachment:
    display_id: str
    description: dict[str, Any]
    image: bytes


class WorkspaceStore(MetadataStore):
    def initialize(self) -> None:
        with self._connect() as db:
            db.executescript(_SCHEMA)

    def create(
        self,
        owner: str,
        *,
        fresh: bool,
        title: str,
        course_ids: list[str],
        document_ids: list[str],
    ) -> Session:
        digest = owner_digest(owner)
        session_id = uuid4().hex
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            exists = db.execute(
                "SELECT 1 FROM workspace_sessions WHERE owner_digest=? LIMIT 1", (digest,)
            ).fetchone()
            if bool(exists) == fresh:
                raise inaccessible()
            db.execute(
                """
                INSERT INTO workspace_sessions
                (session_id, owner_digest, title, course_ids, document_ids, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, digest, title, json_text(course_ids), json_text(document_ids), time()),
            )
            return _session(db, session_id, digest)

    def sessions(self, owner: str) -> list[Session]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT * FROM workspace_sessions WHERE owner_digest=?
                ORDER BY updated_at DESC, session_id
                """,
                (owner_digest(owner),),
            ).fetchall()
            if not rows:
                raise inaccessible()
            return [Session.from_row(row) for row in rows]

    def session(self, owner: str, session_id: str) -> Session:
        with self._connect() as db:
            return _session(db, session_id, owner_digest(owner))

    def update_context(
        self,
        owner: str,
        session: Session,
        course_ids: list[str],
        document_ids: list[str],
        focus_id: str | None,
    ) -> Session:
        digest = owner_digest(owner)
        with self._connect() as db:
            updated = db.execute(
                """
                UPDATE workspace_sessions
                SET course_ids=?, document_ids=?, focus_id=?, scope_revision=scope_revision+1,
                    updated_at=?
                WHERE session_id=? AND owner_digest=? AND scope_revision=?
                """,
                (
                    json_text(course_ids),
                    json_text(document_ids),
                    focus_id,
                    time(),
                    session.session_id,
                    digest,
                    session.scope_revision,
                ),
            )
            if updated.rowcount != 1:
                raise WorkspaceError("scope_changed")
            return _session(db, session.session_id, digest)

    def start(self, owner: str, session_id: str, request: TurnRequest) -> tuple[Turn, bool]:
        payload = json_text(request.model_dump(mode="json"))
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            session = _session(db, session_id, owner_digest(owner))
            _expire(db, session_id)
            existing = _turn_by_client(db, session_id, request.client_turn_id)
            if existing is not None:
                if existing["payload"] != payload:
                    raise WorkspaceError("idempotency_conflict")
                return Turn.from_row(existing), False
            if request.scope_revision != session.scope_revision:
                raise WorkspaceError("scope_changed")
            _require_idle(db, session_id)
            turn_id, attempt = uuid4().hex, uuid4().hex
            db.execute(
                """
                INSERT INTO workspace_turns
                (turn_id, session_id, client_turn_id, payload, status, attempt, lease_until)
                VALUES (?, ?, ?, ?, 'running', ?, ?)
                """,
                (
                    turn_id,
                    session_id,
                    request.client_turn_id,
                    payload,
                    attempt,
                    time() + _LEASE_SECONDS,
                ),
            )
            _insert_message(
                db,
                session_id,
                turn_id,
                "user",
                [{"type": "text", "text": request.text, "citations": []}],
            )
            db.execute(
                "UPDATE workspace_sessions SET updated_at=? WHERE session_id=?",
                (time(), session_id),
            )
            return _get_turn(db, turn_id), True

    def retry(self, owner: str, session_id: str, client_id: str) -> tuple[Turn, bool]:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            session = _session(db, session_id, owner_digest(owner))
            _expire(db, session_id)
            row = _existing_turn(db, session_id, client_id)
            turn = Turn.from_row(row)
            if turn.status != "failed" or not turn.outcome.get("retryable"):
                return turn, False
            if turn.request.scope_revision != session.scope_revision:
                raise WorkspaceError("scope_changed")
            _require_idle(db, session_id)
            db.execute(
                """
                DELETE FROM workspace_media WHERE message_id IN (
                    SELECT message_id FROM workspace_messages WHERE turn_id=? AND role='assistant'
                )
                """,
                (turn.turn_id,),
            )
            db.execute(
                "DELETE FROM workspace_messages WHERE turn_id=? AND role='assistant'",
                (turn.turn_id,),
            )
            db.execute(
                """
                UPDATE workspace_turns
                SET status='running', attempt=?, lease_until=?, outcome='{}' WHERE turn_id=?
                """,
                (uuid4().hex, time() + _LEASE_SECONDS, turn.turn_id),
            )
            return _get_turn(db, turn.turn_id), True

    def cancel(self, owner: str, session_id: str, client_id: str) -> Turn:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _session(db, session_id, owner_digest(owner))
            _expire(db, session_id)
            row = _existing_turn(db, session_id, client_id)
            if row["status"] == "running":
                db.execute(
                    "UPDATE workspace_turns SET status='cancelled', outcome='{}' WHERE turn_id=?",
                    (row["turn_id"],),
                )
            return _get_turn(db, row["turn_id"])

    def turn(self, owner: str, session_id: str, client_id: str) -> Turn:
        with self._connect() as db:
            _session(db, session_id, owner_digest(owner))
            _expire(db, session_id)
            return Turn.from_row(_existing_turn(db, session_id, client_id))

    def clarification(self, session_id: str, turn_id: str) -> Turn | None:
        with self._connect() as db:
            row = db.execute(
                """
                SELECT * FROM workspace_turns
                WHERE session_id=? AND turn_id=? AND status='clarification'
                """,
                (session_id, turn_id),
            ).fetchone()
            return Turn.from_row(row) if row else None

    def active(self, turn: Turn) -> bool:
        with self._connect() as db:
            row = db.execute(
                """SELECT t.status, t.attempt, t.lease_until, s.scope_revision
                   FROM workspace_turns t JOIN workspace_sessions s USING (session_id)
                   WHERE t.turn_id=?""",
                (turn.turn_id,),
            ).fetchone()
            return (
                row is not None
                and row["status"] == "running"
                and row["attempt"] == turn.attempt
                and row["lease_until"] > time()
                and row["scope_revision"] == turn.request.scope_revision
            )

    def finish(
        self,
        turn: Turn,
        status: str,
        outcome: dict[str, Any],
        answer: dict[str, Any] | None,
        images: Sequence[Attachment],
        publication_refs: Sequence[dict[str, Any]],
        allowed_courses: frozenset[str],
        historical_refs: Sequence[dict[str, Any]] = (),
    ) -> Turn:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _expire(db, turn.session_id)
            current = _get_turn(db, turn.turn_id)
            if current.status != "running" or current.attempt != turn.attempt:
                return current
            revision = db.execute(
                "SELECT scope_revision FROM workspace_sessions WHERE session_id=?",
                (turn.session_id,),
            ).fetchone()["scope_revision"]
            promote_due(db, time())
            if revision != turn.request.scope_revision:
                status, outcome, answer, images = "failed", {"error": "scope_changed"}, None, ()
            elif not _all_accessible(
                db, publication_refs, allowed_courses, current=True
            ) or not _all_accessible(db, historical_refs, allowed_courses, current=False):
                status, outcome, answer, images = (
                    "failed",
                    {"error": "stale_reference", "retryable": True},
                    None,
                    (),
                )
            if answer is not None:
                message_id = answer["message_id"]
                _insert_message(
                    db,
                    turn.session_id,
                    turn.turn_id,
                    "assistant",
                    answer["blocks"],
                    message_id=message_id,
                    extra=answer,
                )
                db.executemany(
                    """
                    INSERT INTO workspace_media
                    (message_id, display_id, description, image) VALUES (?, ?, ?, ?)
                    """,
                    [
                        (message_id, image.display_id, json_text(image.description), image.image)
                        for image in images
                    ],
                )
            db.execute(
                "UPDATE workspace_turns SET status=?, outcome=? WHERE turn_id=?",
                (status, json_text(outcome), turn.turn_id),
            )
            return _get_turn(db, turn.turn_id)

    def timeline(
        self, owner: str, session_id: str
    ) -> tuple[Session, list[Turn], list[dict[str, Any]]]:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            session = _session(db, session_id, owner_digest(owner))
            _expire(db, session_id)
            turns = db.execute(
                "SELECT * FROM workspace_turns WHERE session_id=? ORDER BY rowid",
                (session_id,),
            ).fetchall()
            messages = db.execute(
                "SELECT payload FROM workspace_messages WHERE session_id=? ORDER BY ordinal",
                (session_id,),
            ).fetchall()
            return (
                session,
                [Turn.from_row(row) for row in turns],
                [json.loads(row["payload"]) for row in messages],
            )

    def answer(self, turn: Turn) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT payload FROM workspace_messages WHERE turn_id=? AND role='assistant'",
                (turn.turn_id,),
            ).fetchone()
            return json.loads(row["payload"]) if row else None

    def completed_answer(
        self, owner: str, session_id: str, message_id: str
    ) -> tuple[Turn, dict[str, Any]] | None:
        with self._connect() as db:
            _session(db, session_id, owner_digest(owner))
            row = db.execute(
                """
                SELECT t.*, a.payload AS message_payload FROM workspace_messages a
                JOIN workspace_turns t ON t.turn_id=a.turn_id
                WHERE a.session_id=? AND a.message_id=? AND a.role='assistant'
                    AND t.status='completed'
                """,
                (session_id, message_id),
            ).fetchone()
            if row is None:
                return None
            return Turn.from_row(row), json.loads(row["message_payload"])

    def attachment(self, session_id: str, message_id: str, display_id: str) -> Attachment | None:
        with self._connect() as db:
            row = db.execute(
                """
                SELECT m.description, m.image FROM workspace_media m
                JOIN workspace_messages a ON a.message_id=m.message_id
                JOIN workspace_turns t ON t.turn_id=a.turn_id
                WHERE a.session_id=? AND a.message_id=? AND m.display_id=?
                    AND t.status='completed'
                """,
                (session_id, message_id, display_id),
            ).fetchone()
            if row is None:
                return None
            return Attachment(display_id, json.loads(row["description"]), row["image"])

    def display_ordinal(self, session_id: str, message_id: str, display_id: str) -> int:
        with self._connect() as db:
            row = db.execute(
                """
                SELECT a.payload FROM workspace_messages a
                JOIN workspace_turns t ON t.turn_id=a.turn_id
                WHERE a.session_id=? AND a.message_id=? AND t.status='completed'
                """,
                (session_id, message_id),
            ).fetchone()
            ids = json.loads(row["payload"])["visible_display_ids"] if row else []
            if display_id not in ids:
                raise inaccessible()
            return ids.index(display_id) + 1

    def filter_accessible(
        self,
        refs: Sequence[dict[str, Any]],
        allowed_courses: frozenset[str],
        *,
        current: bool = False,
    ) -> list[bool]:
        with self._connect() as db:
            if current:
                promote_due(db, time())
            db.execute("BEGIN")
            return [_accessible(db, ref, allowed_courses, current=current) for ref in refs]


def _session(db: Connection, session_id: str, digest: str) -> Session:
    row = db.execute(
        "SELECT * FROM workspace_sessions WHERE session_id=? AND owner_digest=?",
        (session_id, digest),
    ).fetchone()
    if row is None:
        raise inaccessible()
    return Session.from_row(row)


def _turn_by_client(db: Connection, session_id: str, client_id: str) -> Row | None:
    return db.execute(
        "SELECT * FROM workspace_turns WHERE session_id=? AND client_turn_id=?",
        (session_id, client_id),
    ).fetchone()


def _existing_turn(db: Connection, session_id: str, client_id: str) -> Row:
    row = _turn_by_client(db, session_id, client_id)
    if row is None:
        raise inaccessible()
    return row


def _get_turn(db: Connection, turn_id: str) -> Turn:
    row = db.execute("SELECT * FROM workspace_turns WHERE turn_id=?", (turn_id,)).fetchone()
    assert row is not None
    return Turn.from_row(row)


def _require_idle(db: Connection, session_id: str) -> None:
    if db.execute(
        "SELECT 1 FROM workspace_turns WHERE session_id=? AND status='running'",
        (session_id,),
    ).fetchone():
        raise WorkspaceError("turn_in_progress")


def _expire(db: Connection, session_id: str) -> None:
    db.execute(
        """
        UPDATE workspace_turns SET status='failed', outcome=?
        WHERE session_id=? AND status='running' AND lease_until<=?
        """,
        (json_text({"error": "interrupted", "retryable": True}), session_id, time()),
    )


def _insert_message(
    db: Connection,
    session_id: str,
    turn_id: str,
    role: str,
    blocks: list[dict[str, Any]],
    *,
    message_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    if role == "user":
        ordinal = db.execute(
            """
            SELECT COALESCE(MAX(ordinal), -1)+2 FROM workspace_messages
            WHERE session_id=? AND role='user'
            """,
            (session_id,),
        ).fetchone()[0]
    else:
        ordinal = db.execute(
            "SELECT ordinal+1 FROM workspace_messages WHERE turn_id=? AND role='user'",
            (turn_id,),
        ).fetchone()[0]
    message_id = message_id or uuid4().hex
    message = {
        **(extra or {}),
        "message_id": message_id,
        "turn_id": turn_id,
        "role": role,
        "ordinal": ordinal,
        "blocks": blocks,
        "sources": (extra or {}).get("sources", {}),
        "visible_display_ids": (extra or {}).get("visible_display_ids", []),
    }
    db.execute(
        """
        INSERT INTO workspace_messages(message_id, session_id, turn_id, ordinal, role, payload)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (message_id, session_id, turn_id, ordinal, role, json_text(message)),
    )


def _all_accessible(
    db: Connection,
    refs: Sequence[dict[str, Any]],
    allowed_courses: frozenset[str],
    *,
    current: bool,
) -> bool:
    return all(_accessible(db, ref, allowed_courses, current=current) for ref in refs)


def _accessible(
    db: Connection, ref: dict[str, Any], allowed_courses: frozenset[str], *, current: bool
) -> bool:
    if ref["course_id"] not in allowed_courses:
        return False
    row = db.execute(
        """
        SELECT d.current_version, d.publication_epoch, v.content_sha256 FROM local_documents d
        JOIN local_versions v ON d.document_id=v.document_id
        WHERE d.document_id=? AND d.course_id=? AND d.deleted=0 AND d.current_version IS NOT NULL
            AND v.version=? AND v.published_at IS NOT NULL
        """,
        (ref["document_id"], ref["course_id"], ref["version"]),
    ).fetchone()
    return (
        row is not None
        and row["content_sha256"] == ref["content_sha256"]
        and (
            not current
            or (
                row["current_version"] == ref["version"]
                and row["publication_epoch"] == ref["publication_epoch"]
            )
        )
    )
