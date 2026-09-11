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

        connection.commit()


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