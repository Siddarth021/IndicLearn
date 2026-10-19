"""SQLite persistence (WAL mode, one short-lived connection per call).

The storage layer is intentionally tiny and isolated here so that moving to PostgreSQL later only
means re-implementing this module.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, display_name TEXT, pw_hash TEXT, salt TEXT,
  is_guest INTEGER DEFAULT 0, lang TEXT DEFAULT 'en', created_at REAL
);
CREATE TABLE IF NOT EXISTS tokens (
  token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at REAL, expires_at REAL,
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT, filename TEXT, ext TEXT, size_bytes INTEGER,
  lang TEXT, script TEXT, unit_label TEXT, n_pages INTEGER, status TEXT, error TEXT, created_at REAL,
  stats_json TEXT, summary TEXT, is_sample INTEGER DEFAULT 0,
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id, created_at);
CREATE TABLE IF NOT EXISTS doc_blocks (
  doc_id TEXT NOT NULL, idx INTEGER NOT NULL, page INTEGER, kind TEXT, level INTEGER, text TEXT, meta_json TEXT,
  PRIMARY KEY(doc_id, idx), FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS doc_graph (
  doc_id TEXT PRIMARY KEY, nodes_json TEXT, edges_json TEXT, path_json TEXT,
  FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS chunks (
  doc_id TEXT NOT NULL, id TEXT NOT NULL, section_id TEXT, page INTEGER, text TEXT, idx INTEGER, vec BLOB,
  PRIMARY KEY(doc_id, id), FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS evidence (
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, doc_id TEXT NOT NULL, node_id TEXT NOT NULL,
  kind TEXT NOT NULL, score REAL, weight REAL, ts REAL, meta_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_evidence_ud ON evidence(user_id, doc_id, node_id);
CREATE TABLE IF NOT EXISTS doubts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, doc_id TEXT NOT NULL, label TEXT, question TEXT,
  parent_node_id TEXT, lang TEXT, covered INTEGER DEFAULT 1, created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_doubts_ud ON doubts(user_id, doc_id);
CREATE TABLE IF NOT EXISTS chat_messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, doc_id TEXT NOT NULL, role TEXT, content TEXT,
  meta_json TEXT, ts REAL
);
CREATE INDEX IF NOT EXISTS idx_chat_ud ON chat_messages(user_id, doc_id, id);
CREATE TABLE IF NOT EXISTS quizzes (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, doc_id TEXT NOT NULL, created_at REAL, scope_json TEXT
);
CREATE TABLE IF NOT EXISTS quiz_questions (
  id TEXT PRIMARY KEY, quiz_id TEXT NOT NULL, idx INTEGER, node_id TEXT, qtype TEXT, payload_json TEXT,
  answered INTEGER DEFAULT NULL, correct INTEGER DEFAULT NULL, answered_at REAL,
  FOREIGN KEY(quiz_id) REFERENCES quizzes(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS quiz_bank (
  id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, node_id TEXT NOT NULL, lang TEXT NOT NULL, src_id TEXT,
  payload_json TEXT, engine TEXT, created_at REAL,
  FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_bank_doc ON quiz_bank(doc_id, lang, node_id);
"""


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        self._init_lock = threading.Lock()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._init_lock:
            c = sqlite3.connect(self.path, timeout=30)
            try:
                c.execute("PRAGMA journal_mode=WAL")
                c.executescript(SCHEMA)
            finally:
                c.close()

    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        c.row_factory = sqlite3.Row
        try:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA foreign_keys=ON")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("BEGIN")
            yield c
            c.execute("COMMIT")
        except Exception:
            try:
                c.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            c.close()

    # small helpers ---------------------------------------------------------
    def one(self, sql: str, params: tuple | list = ()) -> sqlite3.Row | None:
        with self.conn() as c:
            return c.execute(sql, params).fetchone()

    def all(self, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        with self.conn() as c:
            return c.execute(sql, params).fetchall()

    def run(self, sql: str, params: tuple | list = ()) -> int:
        with self.conn() as c:
            cur = c.execute(sql, params)
            return cur.lastrowid or 0


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def jload(s: str | None, default: Any = None) -> Any:
    if not s:
        return default
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return default


def now() -> float:
    return time.time()
