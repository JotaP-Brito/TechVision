import sqlite3
from datetime import datetime
from config import DB_PATH


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
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
    conn.commit()
    conn.close()


def add_member(name):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO members (name, created_at, active) VALUES (?, ?, 1)",
        (name, datetime.now().isoformat()),
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
    cur.execute("SELECT name FROM members WHERE id = ?", (member_id,))
    row = cur.fetchone()
    conn.close()
    return row["name"] if row else None


def log_attendance(member_id, confidence):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO attendance (member_id, timestamp, confidence) VALUES (?, ?, ?)",
        (member_id, datetime.now().isoformat(), confidence),
    )
    conn.commit()
    conn.close()
