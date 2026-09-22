import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.models import (
    IncidentActionPlan,
    IncidentRequest,
)


# =========================================================
# DATABASE PATH
# =========================================================

DB_PATH = Path("data") / "incidents.db"


# =========================================================
# INITIALIZE DATABASE
# =========================================================

def initialize_database():

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(
        DB_PATH
    ) as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                incident TEXT NOT NULL,
                downtime_minutes REAL,
                loss_rate_usd_per_hour REAL,
                severity TEXT NOT NULL,
                estimated_loss_usd REAL,
                escalation_required INTEGER NOT NULL,
                confidence REAL NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                role TEXT NOT NULL,
                message_type TEXT NOT NULL,
                content TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_chat_messages_session_id
            ON chat_messages (session_id, id)
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_meta (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL
            )
            """
        )

        connection.commit()


# =========================================================
# CHAT CONVERSATION PERSISTENCE
# =========================================================

def save_chat_message(
    session_id: str,
    role: str,
    message_type: str,
    content: dict | str,
) -> None:
    """Record one chat turn so past conversations survive a
    frontend refresh or a new browser session."""

    created_at = datetime.now(timezone.utc).isoformat()

    serialized_content = (
        content if isinstance(content, str) else json.dumps(content)
    )

    with sqlite3.connect(DB_PATH) as connection:

        connection.execute(
            """
            INSERT INTO chat_messages (
                session_id, created_at, role, message_type, content
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, created_at, role, message_type, serialized_content),
        )

        connection.commit()


def replace_conversation(session_id: str, messages: list[dict]) -> None:
    """Replace a conversation's entire stored message list.

    Used when the user edits an earlier message and regenerates the
    reply: the frontend recomputes the full remaining conversation
    and this makes the stored history match it exactly, so a later
    refresh doesn't resurrect the discarded branch."""

    created_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as connection:

        connection.execute(
            "DELETE FROM chat_messages WHERE session_id = ?",
            (session_id,),
        )

        for message in messages:
            content = message.get("content", "")
            serialized_content = (
                content if isinstance(content, str) else json.dumps(content)
            )
            connection.execute(
                """
                INSERT INTO chat_messages (
                    session_id, created_at, role, message_type, content
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    created_at,
                    message.get("role", "user"),
                    message.get("type", "text"),
                    serialized_content,
                ),
            )

        connection.commit()


def get_conversation(session_id: str) -> list[dict]:
    """Return every stored message for one conversation, oldest first."""

    with sqlite3.connect(DB_PATH) as connection:

        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT created_at, role, message_type, content
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()

    messages = []
    for row in rows:
        try:
            content = json.loads(row["content"])
        except (json.JSONDecodeError, TypeError):
            content = row["content"]
        messages.append({
            "created_at": row["created_at"],
            "role": row["role"],
            "type": row["message_type"],
            "content": content,
        })
    return messages


def list_conversations(limit: int = 30) -> list[dict]:
    """Return the most recently active conversations, newest first,
    with a short preview of the first user message in each (or a
    custom title, if one was set)."""

    with sqlite3.connect(DB_PATH) as connection:

        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT
                chat_messages.session_id AS session_id,
                MIN(chat_messages.created_at) AS started_at,
                MAX(chat_messages.created_at) AS last_active_at,
                (
                    SELECT content FROM chat_messages AS first_message
                    WHERE first_message.session_id = chat_messages.session_id
                    AND first_message.role = 'user'
                    ORDER BY first_message.id ASC
                    LIMIT 1
                ) AS preview,
                conversation_meta.title AS title
            FROM chat_messages
            LEFT JOIN conversation_meta
                ON conversation_meta.session_id = chat_messages.session_id
            GROUP BY chat_messages.session_id
            ORDER BY last_active_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    conversations = []
    for row in rows:
        preview = row["preview"] or ""
        try:
            preview = json.loads(preview)
            if isinstance(preview, dict):
                preview = preview.get("reply") or str(preview)
        except (json.JSONDecodeError, TypeError):
            pass
        conversations.append({
            "session_id": row["session_id"],
            "started_at": row["started_at"],
            "last_active_at": row["last_active_at"],
            "preview": str(preview)[:80],
            "title": row["title"],
        })
    return conversations


def set_conversation_title(session_id: str, title: str) -> None:
    """Set or replace the display title for a conversation."""

    with sqlite3.connect(DB_PATH) as connection:

        connection.execute(
            """
            INSERT INTO conversation_meta (session_id, title)
            VALUES (?, ?)
            ON CONFLICT(session_id) DO UPDATE SET title = excluded.title
            """,
            (session_id, title.strip()[:80]),
        )

        connection.commit()


def delete_conversation(session_id: str) -> int:
    """Delete every stored message for one conversation.
    Returns how many rows were removed."""

    with sqlite3.connect(DB_PATH) as connection:

        cursor = connection.execute(
            "DELETE FROM chat_messages WHERE session_id = ?",
            (session_id,),
        )

        connection.execute(
            "DELETE FROM conversation_meta WHERE session_id = ?",
            (session_id,),
        )

        connection.commit()

        return cursor.rowcount


def delete_conversations(session_ids: list[str]) -> int:
    """Delete several conversations at once.
    Returns how many rows were removed."""

    if not session_ids:
        return 0

    with sqlite3.connect(DB_PATH) as connection:

        placeholders = ", ".join("?" for _ in session_ids)

        cursor = connection.execute(
            f"DELETE FROM chat_messages WHERE session_id IN ({placeholders})",
            session_ids,
        )

        connection.execute(
            f"DELETE FROM conversation_meta WHERE session_id IN ({placeholders})",
            session_ids,
        )

        connection.commit()

        return cursor.rowcount


def delete_all_conversations() -> int:
    """Delete every stored conversation. Returns how many
    rows were removed."""

    with sqlite3.connect(DB_PATH) as connection:

        cursor = connection.execute("DELETE FROM chat_messages")

        connection.execute("DELETE FROM conversation_meta")

        connection.commit()

        return cursor.rowcount


# =========================================================
# SAVE INCIDENT
# =========================================================

def save_incident(
    request: IncidentRequest,
    plan: IncidentActionPlan,
) -> int:

    created_at = datetime.now(
        timezone.utc
    ).isoformat()

    with sqlite3.connect(
        DB_PATH
    ) as connection:

        cursor = connection.execute(
            """
            INSERT INTO incidents (
                created_at,
                incident,
                downtime_minutes,
                loss_rate_usd_per_hour,
                severity,
                estimated_loss_usd,
                escalation_required,
                confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                request.incident,
                request.downtime_minutes,
                request.loss_rate_usd_per_hour,
                plan.severity,
                plan.estimated_loss_usd,
                int(
                    plan.escalation_required
                ),
                plan.confidence,
            ),
        )

        connection.commit()

        return cursor.lastrowid


# =========================================================
# GET INCIDENT LIST
# =========================================================

def get_incidents(
    limit: int = 50,
) -> list[dict]:

    with sqlite3.connect(
        DB_PATH
    ) as connection:

        connection.row_factory = (
            sqlite3.Row
        )

        rows = connection.execute(
            """
            SELECT
                id,
                created_at,
                incident,
                downtime_minutes,
                loss_rate_usd_per_hour,
                severity,
                estimated_loss_usd,
                escalation_required,
                confidence
            FROM incidents
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        {
            "id": row["id"],
            "created_at": (
                row["created_at"]
            ),
            "incident": (
                row["incident"]
            ),
            "downtime_minutes": (
                row["downtime_minutes"]
            ),
            "loss_rate_usd_per_hour": (
                row[
                    "loss_rate_usd_per_hour"
                ]
            ),
            "severity": (
                row["severity"]
            ),
            "estimated_loss_usd": (
                row["estimated_loss_usd"]
            ),
            "escalation_required": bool(
                row[
                    "escalation_required"
                ]
            ),
            "confidence": (
                row["confidence"]
            ),
        }
        for row in rows
    ]


# =========================================================
# GET INCIDENT BY ID
# =========================================================

def get_incident_by_id(
    incident_id: int,
) -> dict | None:

    with sqlite3.connect(
        DB_PATH
    ) as connection:

        connection.row_factory = (
            sqlite3.Row
        )

        row = connection.execute(
            """
            SELECT
                id,
                created_at,
                incident,
                downtime_minutes,
                loss_rate_usd_per_hour,
                severity,
                estimated_loss_usd,
                escalation_required,
                confidence
            FROM incidents
            WHERE id = ?
            """,
            (
                incident_id,
            ),
        ).fetchone()

    if row is None:
        return None

    return {
        "id": row["id"],
        "created_at": (
            row["created_at"]
        ),
        "incident": (
            row["incident"]
        ),
        "downtime_minutes": (
            row["downtime_minutes"]
        ),
        "loss_rate_usd_per_hour": (
            row[
                "loss_rate_usd_per_hour"
            ]
        ),
        "severity": (
            row["severity"]
        ),
        "estimated_loss_usd": (
            row["estimated_loss_usd"]
        ),
        "escalation_required": bool(
            row[
                "escalation_required"
            ]
        ),
        "confidence": (
            row["confidence"]
        ),
    }