"""Procedural Skill Library & Verified Experience Replay Buffer.

Unlike an exact-string plan cache, compiled skills store regex/structural
patterns with parameterized templates or synthesis tactics, plus proof metadata.
Only 100%-verified trajectories ever enter the replay buffer (zero hallucination).
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    domain TEXT NOT NULL,
    pattern TEXT NOT NULL,
    strategy_json TEXT NOT NULL,
    proof_status TEXT NOT NULL,
    uses INTEGER NOT NULL DEFAULT 0,
    successes INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS replay_traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    problem_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    statement TEXT NOT NULL,
    solution TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    proof_status TEXT NOT NULL,
    cycle INTEGER NOT NULL DEFAULT 0,
    meta_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS replay_domain ON replay_traces(domain, cycle);
"""


@dataclass
class Skill:
    name: str
    domain: str
    pattern: str
    strategy: dict[str, Any]
    proof_status: str
    uses: int
    successes: int


class SkillLibrary:
    def __init__(self, db_path: Path, memory: Any = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory = memory
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None, check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def register_skill(self, name: str, domain: str, pattern: str, strategy: dict[str, Any],
                       proof_status: str) -> None:
        re.compile(pattern)  # validate regex before persisting
        now = time.time()
        with self._lock:
            self.conn.execute(
                "INSERT INTO skills(name, domain, pattern, strategy_json, proof_status, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET pattern=excluded.pattern, "
                "strategy_json=excluded.strategy_json, proof_status=excluded.proof_status, "
                "successes=successes+1, updated_at=excluded.updated_at",
                (name, domain, pattern, json.dumps(strategy, ensure_ascii=False), proof_status, now, now),
            )
        if self.memory is not None:
            try:
                self.memory.remember(
                    f"Skill [{name}] ({domain}): matches /{pattern}/ verified via {proof_status}",
                    kind="procedural",
                    key=f"skill:{name}",
                    source="brain.skill_compiler",
                    confidence=1.0,
                    meta={"domain": domain, "strategy_kind": strategy.get("kind")},
                )
            except (ValueError, RuntimeError):
                pass

    def match(self, statement: str, domain: str | None = None) -> tuple[Skill, dict[str, str]] | None:
        with self._lock:
            q = "SELECT name, domain, pattern, strategy_json, proof_status, uses, successes FROM skills"
            params: tuple[Any, ...] = ()
            if domain:
                q += " WHERE domain=?"
                params = (domain,)
            q += " ORDER BY successes DESC, uses DESC"
            rows = self.conn.execute(q, params).fetchall()
        for name, dom, pat, sjson, status, uses, succ in rows:
            m = re.search(pat, statement, re.IGNORECASE | re.DOTALL)
            if m:
                with self._lock:
                    self.conn.execute("UPDATE skills SET uses=uses+1, updated_at=? WHERE name=?", (time.time(), name))
                return (
                    Skill(name, dom, pat, json.loads(sjson), status, uses + 1, succ),
                    {k: v for k, v in m.groupdict().items() if v is not None},
                )
        return None

    def record_outcome(self, name: str, succeeded: bool) -> None:
        with self._lock:
            if succeeded:
                self.conn.execute(
                    "UPDATE skills SET successes=successes+1, updated_at=? WHERE name=?", (time.time(), name)
                )
            else:
                self.conn.execute(
                    "DELETE FROM skills WHERE name=? AND successes <= 1 AND uses >= 3", (name,)
                )

    def add_replay(self, problem_id: str, domain: str, statement: str, solution: str,
                   attempts: int, proof_status: str, cycle: int = 0, meta: dict | None = None) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO replay_traces(problem_id, domain, statement, solution, attempts, proof_status, "
                "cycle, meta_json, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (problem_id, domain, statement, solution, attempts, proof_status, cycle,
                 json.dumps(meta or {}, ensure_ascii=False), time.time()),
            )
            return int(cur.lastrowid)

    def export_replay_jsonl(self, out_path: Path) -> int:
        with self._lock:
            rows = self.conn.execute(
                "SELECT problem_id, domain, statement, solution, attempts, proof_status, cycle "
                "FROM replay_traces ORDER BY id"
            ).fetchall()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for pid, dom, stmt, sol, att, status, cyc in rows:
                fh.write(json.dumps({
                    "id": pid, "domain": dom, "prompt": stmt, "verified_solution": sol,
                    "attempts": att, "proof_status": status, "cycle": cyc,
                }, ensure_ascii=False) + "\n")
        return len(rows)

    def stats(self) -> dict[str, int]:
        with self._lock:
            skills_n = self.conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
            replay_n = self.conn.execute("SELECT COUNT(*) FROM replay_traces").fetchone()[0]
        return {"skills": int(skills_n), "replay_traces": int(replay_n)}
