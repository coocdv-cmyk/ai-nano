"""Durable state: SQLite (WAL) with checksummed checkpoints and an append-only audit log.

Each task's full state is one JSON document. Every save writes it to `tasks`
and appends a numbered checkpoint carrying a SHA-256 of the document. Loading
verifies the checksum; a corrupted current state falls back to the newest
valid checkpoint, and the engine then re-verifies any step after it.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from .model import Task, TaskStatus

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    idem_key TEXT UNIQUE,
    status TEXT NOT NULL,
    state_json TEXT NOT NULL,
    state_sha256 TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoints (
    task_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    state_json TEXT NOT NULL,
    state_sha256 TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (task_id, seq)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    data_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_task ON events(task_id, id);
CREATE TABLE IF NOT EXISTS plan_cache (
    signature TEXT PRIMARY KEY,
    plan_json TEXT NOT NULL,
    hits INTEGER NOT NULL DEFAULT 0,
    successes INTEGER NOT NULL DEFAULT 1,
    updated_at REAL NOT NULL
);
"""

KEEP_CHECKPOINTS = 50


class CorruptState(RuntimeError):
    pass


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), timeout=30, isolation_level=None, check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # -- tasks -----------------------------------------------------------------

    def save(self, task: Task, event: str | None = None, data: dict | None = None) -> int:
        """Atomically persist the task, append a checkpoint, and optionally log an event."""
        state = _dump(task.to_state())
        digest = _sha(state)
        with self._lock:
            c = self.conn
            c.execute("BEGIN IMMEDIATE")
            try:
                c.execute(
                    "INSERT INTO tasks(id, idem_key, status, state_json, state_sha256, updated_at) "
                    "VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status, "
                    "state_json=excluded.state_json, state_sha256=excluded.state_sha256, updated_at=excluded.updated_at",
                    (task.id, task.idem_key, task.status.value, state, digest, time.time()),
                )
                seq = c.execute(
                    "SELECT COALESCE(MAX(seq), 0) + 1 FROM checkpoints WHERE task_id=?", (task.id,)
                ).fetchone()[0]
                c.execute(
                    "INSERT INTO checkpoints(task_id, seq, state_json, state_sha256, created_at) VALUES(?,?,?,?,?)",
                    (task.id, seq, state, digest, time.time()),
                )
                c.execute(
                    "DELETE FROM checkpoints WHERE task_id=? AND seq <= ?", (task.id, seq - KEEP_CHECKPOINTS)
                )
                if event:
                    c.execute(
                        "INSERT INTO events(task_id, ts, kind, data_json) VALUES(?,?,?,?)",
                        (task.id, time.time(), event, _dump(data or {})),
                    )
                c.execute("COMMIT")
            except BaseException:
                c.execute("ROLLBACK")
                raise
        return seq

    def load(self, task_id: str) -> tuple[Task, bool]:
        """Return (task, recovered_from_checkpoint). Raises CorruptState if nothing valid remains."""
        with self._lock:
            row = self.conn.execute(
                "SELECT state_json, state_sha256 FROM tasks WHERE id=? OR id LIKE ?", (task_id, task_id + "%")
            ).fetchone()
            if row is None:
                raise KeyError(task_id)
            state, digest = row
            if _sha(state) == digest:
                return Task.from_state(json.loads(state)), False
            full_id = json.loads(state)["id"] if _looks_json(state) else task_id
            self.log(full_id, "corruption_detected", {"where": "tasks.state_json"})
            for st, dg, seq in self.conn.execute(
                "SELECT state_json, state_sha256, seq FROM checkpoints WHERE task_id=? ORDER BY seq DESC",
                (full_id,),
            ):
                if _sha(st) == dg:
                    self.log(full_id, "restored_checkpoint", {"seq": seq})
                    return Task.from_state(json.loads(st)), True
        raise CorruptState(f"no valid checkpoint for task {task_id}")

    def find_by_key(self, idem_key: str) -> str | None:
        row = self.conn.execute("SELECT id FROM tasks WHERE idem_key=?", (idem_key,)).fetchone()
        return row[0] if row else None

    def list_tasks(self, statuses: set[TaskStatus] | None = None) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT id, status, state_json, updated_at FROM tasks ORDER BY updated_at DESC")
        out = []
        for tid, status, state, upd in rows:
            if statuses and TaskStatus(status) not in statuses:
                continue
            goal = json.loads(state).get("goal", "") if _looks_json(state) else "<corrupt>"
            out.append({"id": tid, "status": status, "goal": goal, "updated_at": upd})
        return out

    # -- audit log -------------------------------------------------------------

    def log(self, task_id: str | None, kind: str, data: dict | None = None) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO events(task_id, ts, kind, data_json) VALUES(?,?,?,?)",
                (task_id, time.time(), kind, _dump(data or {})),
            )

    def events(self, task_id: str) -> Iterator[dict[str, Any]]:
        for ts, kind, data in self.conn.execute(
            "SELECT ts, kind, data_json FROM events WHERE task_id=? ORDER BY id", (task_id,)
        ):
            yield {"ts": ts, "kind": kind, "data": json.loads(data)}

    # -- procedural plan cache (adaptive compute tier 0) --------------------------

    def cache_get(self, signature: str) -> list[dict] | None:
        row = self.conn.execute("SELECT plan_json FROM plan_cache WHERE signature=?", (signature,)).fetchone()
        if not row:
            return None
        self.conn.execute("UPDATE plan_cache SET hits = hits + 1 WHERE signature=?", (signature,))
        return json.loads(row[0])

    def cache_put(self, signature: str, plan: list[dict]) -> None:
        self.conn.execute(
            "INSERT INTO plan_cache(signature, plan_json, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(signature) DO UPDATE SET plan_json=excluded.plan_json, "
            "successes=successes+1, updated_at=excluded.updated_at",
            (signature, _dump(plan), time.time()),
        )

    def cache_drop(self, signature: str) -> None:
        self.conn.execute("DELETE FROM plan_cache WHERE signature=?", (signature,))


def _looks_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except ValueError:
        return False
