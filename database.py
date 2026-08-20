import sqlite3
from datetime import datetime
from config import DB_PATH


def utc_now_iso():
    return datetime.now().isoformat(timespec="seconds")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            member_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            confidence REAL,
            FOREIGN KEY(member_id) REFERENCES members(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS app_state (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()
    conn.close()


def add_member(name):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO members (name, created_at, active) VALUES (?, ?, 1)",
        (name, utc_now_iso()),
    )
    member_id = cur.lastrowid
    conn.commit()
    conn.close()
    return member_id


def deactivate_member(member_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE members SET active = 0 WHERE id = ?", (member_id,))
    conn.commit()
    conn.close()


def reactivate_member(member_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE members SET active = 1 WHERE id = ?", (member_id,))
    conn.commit()
    conn.close()


def get_all_active_members():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM members WHERE active = 1")
    rows = cur.fetchall()
    conn.close()
    return {row["id"]: row["name"] for row in rows}


def get_member_name(member_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM members WHERE id = ? AND active = 1", (member_id,))
    row = cur.fetchone()
    conn.close()
    return row["name"] if row else None


def get_members():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, created_at, active FROM members ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    return rows


def get_recent_attendance(limit=500):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT attendance.id, members.name, attendance.timestamp, attendance.confidence
        FROM attendance
        JOIN members ON attendance.member_id = members.id
        ORDER BY attendance.timestamp DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def log_attendance(member_id, confidence):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO attendance (member_id, timestamp, confidence) VALUES (?, ?, ?)",
        (member_id, utc_now_iso(), confidence),
    )
    conn.commit()
    conn.close()


def _set_app_state(key, value):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO app_state (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )
    conn.commit()
    conn.close()


def _get_app_state(key):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT value FROM app_state WHERE key = ?", (key,))
    row = cur.fetchone()
    conn.close()
    return row["value"] if row else None


def mark_dataset_changed():
    _set_app_state("dataset_changed_at", utc_now_iso())


def mark_model_trained():
    _set_app_state("model_trained_at", utc_now_iso())


def get_training_status():
    model_trained_at = _get_app_state("model_trained_at")
    dataset_changed_at = _get_app_state("dataset_changed_at")

    if not model_trained_at:
        return {
            "ready": False,
            "reason": "No trained model is recorded yet.",
            "model_trained_at": None,
            "dataset_changed_at": dataset_changed_at,
        }

    if dataset_changed_at and dataset_changed_at > model_trained_at:
        return {
            "ready": False,
            "reason": "Enrollment or member status changed since the last training run.",
            "model_trained_at": model_trained_at,
            "dataset_changed_at": dataset_changed_at,
        }

    return {
        "ready": True,
        "reason": "",
        "model_trained_at": model_trained_at,
        "dataset_changed_at": dataset_changed_at,
    }
