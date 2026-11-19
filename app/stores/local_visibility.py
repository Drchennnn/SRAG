"""Shared SQLite visibility for local publication and legacy readers."""

from sqlite3 import Connection


def promote_due(connection: Connection, cutoff: float) -> None:
    """Promote ready local publications with the current writer's transaction."""

    due = "deleted = 0 AND pending_version IS NOT NULL AND pending_at <= ?"
    if not connection.execute(
        f"SELECT 1 FROM local_documents WHERE {due} LIMIT 1", (cutoff,)
    ).fetchone():
        return
    owns_transaction = not connection.in_transaction
    if owns_transaction:
        connection.execute("BEGIN IMMEDIATE")
    connection.execute(
        f"""
        UPDATE local_versions
        SET published_at = COALESCE(published_at, ?)
        WHERE (document_id, version) IN (
            SELECT document_id, pending_version FROM local_documents WHERE {due}
        )
        """,
        (cutoff, cutoff),
    )
    connection.execute(
        f"""
        UPDATE local_documents
        SET current_version = pending_version, pending_version = NULL, pending_at = NULL,
            publication_epoch = publication_epoch + 1
        WHERE {due}
        """,
        (cutoff,),
    )
    if owns_transaction:
        connection.commit()


def has_local_schema(connection: Connection) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'local_legacy_versions'"
        ).fetchone()
        is not None
    )


def legacy_visibility_sql(connection: Connection, cutoff: float) -> str:
    """Build a fixed predicate for old document/version aliases d/v."""

    if not has_local_schema(connection):
        return "1"
    promote_due(connection, cutoff)
    return """
        (NOT EXISTS (
            SELECT 1 FROM local_legacy_versions a WHERE a.legacy_document_id = d.document_id
        ) OR EXISTS (
            SELECT 1
            FROM local_legacy_versions a
            JOIN local_documents ld ON ld.document_id = a.document_id
            WHERE a.legacy_document_id = d.document_id AND a.legacy_version = v.version
              AND a.course_id = d.course_id AND a.content_sha256 = v.content_hash
              AND ld.course_id = d.course_id AND ld.deleted = 0 AND ld.current_version = a.version
        ))
    """


def revoke_legacy_document(connection: Connection, legacy_document_id: str) -> None:
    if has_local_schema(connection):
        connection.execute(
            """
            UPDATE local_documents
            SET deleted = 1, current_version = NULL, pending_version = NULL, pending_at = NULL,
                publication_epoch = publication_epoch + 1
            WHERE deleted = 0 AND document_id IN (
                SELECT document_id FROM local_legacy_versions WHERE legacy_document_id = ?
            )
            """,
            (legacy_document_id,),
        )
