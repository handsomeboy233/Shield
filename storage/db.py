from __future__ import annotations

import sqlite3
from pathlib import Path


def _ensure_column(cur: sqlite3.Cursor, table: str, column: str, decl: str) -> None:
    cur.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cur.fetchall()}
    if column not in existing:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def init_db(db_path: str) -> None:
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            input_type TEXT,
            input_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            record_id TEXT,
            source_type TEXT,
            rule_hit INTEGER,
            rule_name TEXT,
            rule_severity TEXT,
            anomaly_score REAL,
            is_anomaly INTEGER,
            is_unknown INTEGER,
            final_label TEXT,
            stage TEXT,
            confidence REAL,
            risk_score REAL,
            raw_text TEXT,
            reason_rule TEXT,
            reason_anomaly TEXT,
            reason_osr TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    _ensure_column(cur, "detections", "anomaly_threshold", "REAL")
    _ensure_column(cur, "detections", "anomaly_model", "TEXT")
    _ensure_column(cur, "detections", "anomaly_version", "TEXT")
    _ensure_column(cur, "detections", "osr_method", "TEXT")
    _ensure_column(cur, "detections", "osr_confidence", "REAL")

    conn.commit()
    conn.close()
