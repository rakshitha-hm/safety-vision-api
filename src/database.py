import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("DB_PATH", ROOT / "data" / "safety.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    location TEXT NOT NULL,
    filename TEXT,
    num_faces INTEGER NOT NULL,
    num_violations INTEGER NOT NULL,
    latency_ms REAL
);
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL REFERENCES predictions(id),
    class TEXT NOT NULL,
    confidence REAL NOT NULL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    location TEXT NOT NULL,
    rule TEXT NOT NULL,
    message TEXT NOT NULL,
    prediction_id INTEGER REFERENCES predictions(id)
);
CREATE INDEX IF NOT EXISTS idx_pred_loc_time ON predictions(location, created_at);
CREATE INDEX IF NOT EXISTS idx_alert_loc_rule_time ON alerts(location, rule, created_at);
"""


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def minutes_ago(minutes):
    t = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    return t.isoformat(timespec="seconds")


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with db() as conn:
        conn.executescript(SCHEMA)

def save_prediction(conn, location, filename, detections, num_violations, latency_ms):
    cur = conn.execute(
        "INSERT INTO predictions (created_at, location, filename, num_faces, num_violations, latency_ms) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (now_utc(), location, filename, len(detections), num_violations, latency_ms),
    )
    prediction_id = cur.lastrowid
    conn.executemany(
        "INSERT INTO detections (prediction_id, class, confidence, x1, y1, x2, y2) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(prediction_id, d["class"], d["confidence"],
          d["box"]["x1"], d["box"]["y1"], d["box"]["x2"], d["box"]["y2"]) for d in detections],
    )
    return prediction_id


def window_stats(conn, location, minutes):
    row = conn.execute(
        "SELECT COUNT(*) AS images, "
        "COALESCE(SUM(num_faces), 0) AS faces, "
        "COALESCE(SUM(num_violations), 0) AS violations "
        "FROM predictions WHERE location = ? AND created_at >= ?",
        (location, minutes_ago(minutes)),
    ).fetchone()
    return dict(row)


def recent_alert_exists(conn, location, rule, minutes):
    row = conn.execute(
        "SELECT 1 FROM alerts WHERE location = ? AND rule = ? AND created_at >= ? LIMIT 1",
        (location, rule, minutes_ago(minutes)),
    ).fetchone()
    return row is not None


def save_alert(conn, location, rule, message, prediction_id):
    conn.execute(
        "INSERT INTO alerts (created_at, location, rule, message, prediction_id) VALUES (?, ?, ?, ?, ?)",
        (now_utc(), location, rule, message, prediction_id),
    )


def list_alerts(conn, limit):
    rows = conn.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]