"""SQLite history storage.

Android note: the app's install directory is read-only, so the database lives
in the app's private files dir (ANDROID_PRIVATE, set by python-for-android).
On desktop it stays next to this file.
"""
import os
import sqlite3
from contextlib import closing
from datetime import datetime


def _data_dir():
    base = os.environ.get("ANDROID_PRIVATE")
    if base and os.path.isdir(base):
        return base
    return os.path.dirname(os.path.abspath(__file__))


DB_PATH = os.environ.get("OCR_DB_PATH") or os.path.join(_data_dir(), "ocr_history.db")

_COLUMNS = ("id, timestamp, image_path, raw_text, cleaned_num, "
            "engine_used, confidence, notes")


def _connect():
    return sqlite3.connect(DB_PATH)


def _row_to_dict(r):
    return {
        "id": r[0], "timestamp": r[1], "image_path": r[2], "raw_text": r[3],
        "cleaned_num": r[4], "engine": r[5], "confidence": r[6], "notes": r[7],
    }


def init_database():
    """Create the schema if it does not exist (never fatal for the app)."""
    try:
        _init_database()
    except Exception as e:  # a read-only or full storage must not kill the app
        print(f"[DB] Could not prepare the database: {e}")


def _init_database():
    with closing(_connect()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ocr_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                image_path  TEXT    NOT NULL,
                raw_text    TEXT,
                cleaned_num TEXT,
                engine_used TEXT,
                confidence  REAL,
                notes       TEXT,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    print(f"[DB] Database ready: {DB_PATH}")


def save_result(image_path, raw_text, cleaned_num, engine_used, confidence, notes=""):
    """Save one OCR result. Returns the new record id."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with closing(_connect()) as conn:
        cur = conn.execute(
            "INSERT INTO ocr_history "
            "(timestamp, image_path, raw_text, cleaned_num, engine_used, confidence, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (timestamp, image_path or "", raw_text, cleaned_num, engine_used, confidence, notes),
        )
        conn.commit()
        record_id = cur.lastrowid
    print(f"[DB] Saved record #{record_id}: {cleaned_num}")
    return record_id


def get_all_history(limit=100):
    """Newest first."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM ocr_history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def delete_record(record_id):
    with closing(_connect()) as conn:
        conn.execute("DELETE FROM ocr_history WHERE id = ?", (record_id,))
        conn.commit()
    print(f"[DB] Deleted record #{record_id}")


def search_history(query):
    """Match against cleaned_num or notes."""
    like = f"%{query}%"
    with closing(_connect()) as conn:
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM ocr_history "
            "WHERE cleaned_num LIKE ? OR notes LIKE ? ORDER BY id DESC",
            (like, like),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_stats():
    with closing(_connect()) as conn:
        total = conn.execute("SELECT COUNT(*) FROM ocr_history").fetchone()[0]
        avg_conf = conn.execute(
            "SELECT AVG(confidence) FROM ocr_history WHERE confidence > 0"
        ).fetchone()[0] or 0
    return {"total": total, "avg_confidence": round(avg_conf, 1)}


init_database()
