import json
import sqlite3
from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS files(
  file_id TEXT PRIMARY KEY, sha256 TEXT, source_path TEXT, organised_path TEXT,
  type TEXT, mime TEXT,
  taken_at TEXT, taken_at_source TEXT, taken_at_confidence REAL, timezone_source TEXT,
  place TEXT, people TEXT, event TEXT, caption TEXT, tags TEXT, ocr_text TEXT,
  metadata_status TEXT, flags TEXT, rating INTEGER, camera TEXT, raw_metadata TEXT,
  img_vec BLOB, txt_vec BLOB, indexed_at TEXT
);
CREATE TABLE IF NOT EXISTS duplicates(file_id TEXT, source_path TEXT, UNIQUE(file_id, source_path));
"""

JSON_COLS = ["place", "people", "event", "caption", "tags", "flags", "camera", "raw_metadata"]


def connect():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def row_to_dict(row, with_raw=False):
    d = dict(row)
    for c in JSON_COLS:
        if d.get(c):
            d[c] = json.loads(d[c])
    d.pop("img_vec", None)
    d.pop("txt_vec", None)
    if not with_raw:
        d.pop("raw_metadata", None)
    return d
