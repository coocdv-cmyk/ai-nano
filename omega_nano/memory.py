"""Memory Fabric: persistent, versioned, provenance-tracked memory with hybrid retrieval.

Every record answers the constitution's memory questions: what (text), why
(kind), where from (source), when valid (valid_from / valid_to), how confident
(confidence), how to correct (re-remember with the same key supersedes the old
version but keeps it as history), how to delete (forget = hard delete).

Retrieval fuses two independent signals with reciprocal-rank fusion:
  * lexical  — SQLite FTS5 BM25 over normalized text
  * semantic — signed feature-hashing embeddings (word uni/bi-grams + char
    trigrams), stored int8-quantized (1 byte per dimension), scored by brute
    force dot product over an in-RAM matrix.
No model download, no network, CPU-only. The embedder is deliberately a
replaceable component: a learned encoder can be dropped in behind `embed()`.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import time
import zlib
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import kernels

try:
    import numpy as np
except ImportError:  # pure-Python fallback keeps the fabric dependency-free
    np = None

DIM = 512
KINDS = {"semantic", "episodic", "procedural", "project", "preference", "evidence"}
NEAR_DUP = 0.97
RRF_K = 60

_AR_DIACRITICS = re.compile(r"[ً-ْٰـ]")
_AR_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه", "ى": "ي", "ؤ": "و", "ئ": "ي",
                         "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5", "٦": "6",
                         "٧": "7", "٨": "8", "٩": "9"})
_WORD = re.compile(r"\w+", re.UNICODE)
STOP = set(
    "the a an of to in on for and or is are was were be been it this that with as at by from what who "
    "do you know about tell me my your our where when which how why who whom whose many much does did "
    "kind sort type there their they them can could would should will".split()
    + "في من على الى إلى عن ما ماذا هل هو هي ان أن و او أو كان تعرف عن لي اين أين متى كيف كم لماذا اي أي "
      "هذا هذه ذلك التي الذي".split()
)


def normalize(text: str) -> str:
    text = _AR_DIACRITICS.sub("", text).translate(_AR_MAP).casefold()
    return " ".join(_WORD.findall(text))


STOP = {normalize(w) for w in STOP}  # compare in normalized form (أين -> اين)


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def _feat(h: str) -> tuple[int, float]:
    x = zlib.crc32(h.encode("utf-8"))
    return x % DIM, (1.0 if (x >> 31) & 1 else -1.0)


def embed(text: str) -> list[float]:
    toks = tokens(text)
    content = [t for t in toks if t not in STOP] or toks
    feats: dict[str, float] = {}

    def add(f: str, w: float) -> None:
        feats[f] = feats.get(f, 0.0) + w

    for t in content:
        add("w:" + t, 1.0)
        # Arabic definite article and English plural folding for recall.
        if t.startswith("ال") and len(t) > 3:
            add("w:" + t[2:], 0.6)
        if t.endswith("s") and len(t) > 3:
            add("w:" + t[:-1], 0.6)
        padded = f"#{t}#"
        for i in range(len(padded) - 2):
            add("c:" + padded[i:i + 3], 0.25)
    for a, b in zip(content, content[1:]):
        add(f"b:{a}_{b}", 0.7)
    vec = [0.0] * DIM
    for f, w in feats.items():
        i, sign = _feat(f)
        vec[i] += sign * (1.0 + math.log(w)) if w >= 1 else sign * w
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def quantize(vec: list[float]) -> tuple[bytes, float]:
    m = max((abs(v) for v in vec), default=0.0) or 1.0
    scale = m / 127.0
    q = array("b", (max(-127, min(127, round(v / scale))) for v in vec))
    return q.tobytes(), scale


@dataclass
class Hit:
    id: int
    text: str
    kind: str
    key: str | None
    source: str
    confidence: float
    valid_from: float
    valid_to: float | None
    score: float
    lexical_rank: int | None
    vector_rank: int | None
    cosine: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class MemoryFabric:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), timeout=30, isolation_level=None, check_same_thread=False)
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL, key TEXT, text TEXT NOT NULL, norm TEXT NOT NULL,
                source TEXT NOT NULL, confidence REAL NOT NULL,
                created_at REAL NOT NULL, valid_from REAL NOT NULL, valid_to REAL,
                meta_json TEXT NOT NULL, vec BLOB NOT NULL, scale REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS mem_key ON memories(key, valid_to);
            CREATE INDEX IF NOT EXISTS mem_norm ON memories(norm);
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(norm, tokenize='unicode61 remove_diacritics 2');
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_tri USING fts5(norm, tokenize='trigram');
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_vocab USING fts5vocab(memories_fts, 'row');
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_tri_vocab USING fts5vocab(memories_tri, 'row');
            """
        )
        # Backfill the trigram index for databases created before it existed.
        if self.conn.execute("SELECT COUNT(*) FROM memories_tri").fetchone()[0] == 0:
            self.conn.execute("INSERT INTO memories_tri(rowid, norm) SELECT id, norm FROM memories")
        self._ids: list[int] = []
        self._mat = None  # int8 matrix [n, DIM] of active memories, or list of arrays
        self._scales = None
        self._dirty = True

    def close(self) -> None:
        self.conn.close()

    # -- write path --------------------------------------------------------------

    def remember(self, text: str, kind: str = "semantic", source: str = "user", key: str | None = None,
                 confidence: float = 0.9, meta: dict | None = None) -> dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("cannot remember empty text")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {sorted(KINDS)}")
        confidence = max(0.0, min(1.0, float(confidence)))
        norm = normalize(text)
        vec = embed(text)
        with self._lock:
            dup = self.conn.execute(
                "SELECT id FROM memories WHERE norm=? AND valid_to IS NULL", (norm,)
            ).fetchone()
            if dup:
                return {"id": dup[0], "status": "duplicate"}
            if key is None:
                near = self._nearest_active(vec)
                if near and near[1] >= NEAR_DUP:
                    return {"id": near[0], "status": "near_duplicate", "cosine": round(near[1], 4)}
            now = time.time()
            superseded = []
            if key:
                key = normalize(key)
                superseded = [r[0] for r in self.conn.execute(
                    "SELECT id FROM memories WHERE key=? AND valid_to IS NULL", (key,))]
            blob, scale = quantize(vec)
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                if superseded:
                    self.conn.executemany("UPDATE memories SET valid_to=? WHERE id=?", [(now, i) for i in superseded])
                cur = self.conn.execute(
                    "INSERT INTO memories(kind,key,text,norm,source,confidence,created_at,valid_from,valid_to,"
                    "meta_json,vec,scale) VALUES(?,?,?,?,?,?,?,?,NULL,?,?,?)",
                    (kind, key, text, norm, source, confidence, now, now, json.dumps(meta or {}, ensure_ascii=False),
                     blob, scale),
                )
                mid = cur.lastrowid
                self.conn.execute("INSERT INTO memories_fts(rowid, norm) VALUES(?,?)", (mid, norm))
                self.conn.execute("INSERT INTO memories_tri(rowid, norm) VALUES(?,?)", (mid, norm))
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            if superseded:
                self._dirty = True
            else:
                self._append(mid, blob, scale)
        return {"id": mid, "status": "stored", "superseded": superseded}

    def remember_many(self, items: list[dict[str, Any]], batch: int = 5000) -> dict[str, int]:
        """Bulk ingestion: one transaction per batch, exact-duplicate removal, no near-dup scan.

        Each item is {"text", optional "kind", "source", "key", "confidence", "meta"}.
        Keys are stored but do not supersede here; use remember() for versioned updates.
        """
        stored = 0
        now = time.time()
        with self._lock:
            for start in range(0, len(items), batch):
                chunk = items[start:start + batch]
                rows, seen = [], set()
                for it in chunk:
                    text = str(it["text"]).strip()
                    norm = normalize(text)
                    if not text or norm in seen:
                        continue
                    seen.add(norm)
                    blob, scale = quantize(embed(text))
                    rows.append((it.get("kind", "semantic"), normalize(it["key"]) if it.get("key") else None, text,
                                 norm, it.get("source", "bulk"), float(it.get("confidence", 0.9)), now, now,
                                 json.dumps(it.get("meta") or {}, ensure_ascii=False), blob, scale))
                existing = set()
                for i in range(0, len(rows), 500):
                    part = [r[3] for r in rows[i:i + 500]]
                    existing.update(r[0] for r in self.conn.execute(
                        f"SELECT norm FROM memories WHERE valid_to IS NULL AND norm IN ({','.join('?' * len(part))})",
                        part))
                rows = [r for r in rows if r[3] not in existing]
                self.conn.execute("BEGIN IMMEDIATE")
                try:
                    for r in rows:
                        if r[0] not in KINDS:
                            raise ValueError(f"kind must be one of {sorted(KINDS)}")
                        mid = self.conn.execute(
                            "INSERT INTO memories(kind,key,text,norm,source,confidence,created_at,valid_from,"
                            "valid_to,meta_json,vec,scale) VALUES(?,?,?,?,?,?,?,?,NULL,?,?,?)", r).lastrowid
                        self.conn.execute("INSERT INTO memories_fts(rowid, norm) VALUES(?,?)", (mid, r[3]))
                        self.conn.execute("INSERT INTO memories_tri(rowid, norm) VALUES(?,?)", (mid, r[3]))
                        self._append(mid, r[9], r[10])
                    self.conn.execute("COMMIT")
                except BaseException:
                    self.conn.execute("ROLLBACK")
                    self._dirty = True
                    raise
                stored += len(rows)
        return {"stored": stored, "skipped_duplicates": len(items) - stored}

    def forget(self, mid: int) -> bool:
        """Hard delete (privacy): removes the record, its index entry and its vector."""
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            n = self.conn.execute("DELETE FROM memories WHERE id=?", (mid,)).rowcount
            self.conn.execute("DELETE FROM memories_fts WHERE rowid=?", (mid,))
            self.conn.execute("DELETE FROM memories_tri WHERE rowid=?", (mid,))
            self.conn.execute("COMMIT")
            self._dirty = True
        return n > 0

    # -- read path ---------------------------------------------------------------

    def _load_matrix(self) -> None:
        rows = self.conn.execute("SELECT id, vec, scale FROM memories WHERE valid_to IS NULL ORDER BY id").fetchall()
        self._ids = [r[0] for r in rows]
        if np is not None:
            cap = max(1024, 2 * len(rows))
            self._mat = np.zeros((cap, DIM), dtype=np.int8)
            self._scales = np.zeros((cap,), dtype=np.float32)
            if rows:
                self._mat[:len(rows)] = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.int8).reshape(-1, DIM)
                self._scales[:len(rows)] = [r[2] for r in rows]
        else:
            self._mat = [array("b", r[1]) for r in rows]
            self._scales = [r[2] for r in rows]
        self._dirty = False

    def _append(self, mid: int, blob: bytes, scale: float) -> None:
        """Grow the in-RAM index in place (amortized O(1)) instead of reloading it."""
        if self._dirty:
            return  # the next search reloads everything anyway
        n = len(self._ids)
        if np is not None:
            if n == len(self._mat):
                grow = max(1024, n)
                self._mat = np.concatenate([self._mat, np.zeros((grow, DIM), dtype=np.int8)])
                self._scales = np.concatenate([self._scales, np.zeros((grow,), dtype=np.float32)])
            self._mat[n] = np.frombuffer(blob, dtype=np.int8)
            self._scales[n] = scale
        else:
            self._mat.append(array("b", blob))
            self._scales.append(scale)
        self._ids.append(mid)

    def _cosines(self, vec: list[float]):
        if self._dirty:
            self._load_matrix()
        if np is not None:
            n = len(self._ids)
            q = np.asarray(vec, dtype=np.float32)
            return kernels.scaled_dot(self._mat[:n], q, self._scales[:n]) if n else np.zeros(0)
        return [sum(a * b for a, b in zip(row, vec)) * s for row, s in zip(self._mat, self._scales)]

    def _nearest_active(self, vec: list[float]) -> tuple[int, float] | None:
        cos = self._cosines(vec)
        if not len(self._ids):
            return None
        if np is not None:
            i = int(np.argmax(cos))
            return self._ids[i], float(cos[i])
        i = max(range(len(cos)), key=cos.__getitem__)
        return self._ids[i], cos[i]

    def recall(self, query: str, k: int = 5, kind: str | None = None, include_history: bool = False,
               pool: int = 50, exclude: tuple[str, ...] = (),
               signals: tuple[str, ...] = ("lexical", "vector", "fuzzy")) -> list[Hit]:
        qn = normalize(query)
        qt = [t for t in qn.split() if t not in STOP] or qn.split()
        if not qt:
            return []
        with self._lock:
            lex: list[int] = []
            vrank: list[tuple[int, float]] = []
            prefixes, typos = self._unknown(qt) if "fuzzy" in signals else ([], [])
            fts_q = " OR ".join(['"' + t.replace('"', "") + '"' for t in qt] +
                                ['"' + p.replace('"', "") + '"*' for p in prefixes])
            try:
                lex = [] if "lexical" not in signals else [r[0] for r in self.conn.execute(
                    "SELECT rowid FROM memories_fts WHERE memories_fts MATCH ? ORDER BY bm25(memories_fts) LIMIT ?",
                    (fts_q, pool))]
            except sqlite3.OperationalError:
                lex = []
            if "vector" in signals:
                cos = self._cosines(embed(query))
                if np is not None and len(self._ids):
                    top = np.argpartition(-cos, min(pool, len(cos) - 1))[:pool]
                    top = top[np.argsort(-cos[top])]
                    vrank = [(self._ids[i], float(cos[i])) for i in top if cos[i] > 0.05]
                else:
                    order = sorted(range(len(self._ids)), key=lambda i: -cos[i])[:pool]
                    vrank = [(self._ids[i], cos[i]) for i in order if cos[i] > 0.05]
            fuzzy = self._fuzzy(typos, pool) if typos else []
            cos_by_id = dict(vrank)
            fused: dict[int, float] = {}
            lex_rank = {mid: r for r, mid in enumerate(lex)}
            vec_rank = {mid: r for r, (mid, _) in enumerate(vrank)}
            for mid, r in lex_rank.items():
                fused[mid] = fused.get(mid, 0.0) + 1.0 / (RRF_K + r + 1)
            for mid, r in vec_rank.items():
                fused[mid] = fused.get(mid, 0.0) + 1.0 / (RRF_K + r + 1)
            for r, mid in enumerate(fuzzy):
                fused[mid] = fused.get(mid, 0.0) + 1.0 / (RRF_K + r + 1)
            if not fused:
                return []
            ids = sorted(fused, key=fused.get, reverse=True)
            marks = ",".join("?" * len(ids))
            rows = {r[0]: r for r in self.conn.execute(
                f"SELECT id,text,kind,key,source,confidence,valid_from,valid_to FROM memories WHERE id IN ({marks})",
                ids)}
        hits: list[Hit] = []
        for mid in ids:
            r = rows.get(mid)
            if r is None or (r[7] is not None and not include_history) or (kind and r[2] != kind) \
                    or r[2] in exclude:
                continue
            hits.append(Hit(id=mid, text=r[1], kind=r[2], key=r[3], source=r[4], confidence=r[5],
                            valid_from=r[6], valid_to=r[7], score=round(fused[mid] * (0.5 + 0.5 * r[5]), 6),
                            lexical_rank=lex_rank.get(mid), vector_rank=vec_rank.get(mid),
                            cosine=round(cos_by_id.get(mid, 0.0), 4)))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def _unknown(self, qt: list[str]) -> tuple[list[str], list[str]]:
        """Split query words absent from the vocabulary into (inflected prefixes, likely misspellings).

        A word whose stem-like prefix matches indexed terms ("manages" -> "manag")
        is an inflection; one whose prefix matches nothing is a likely misspelling.
        """
        cand = [t for t in qt if len(t) >= 4 and not t.isdigit()]
        if not cand:
            return [], []
        known = {r[0] for r in self.conn.execute(
            f"SELECT term FROM memories_vocab WHERE term IN ({','.join('?' * len(cand))})", cand)}
        prefixes, typos = [], []
        for t in cand:
            if t in known:
                continue
            pre = t[:max(4, len(t) - 3)]
            hit = self.conn.execute(
                "SELECT 1 FROM memories_vocab WHERE term >= ? AND term < ? LIMIT 1", (pre, pre + "￿")
            ).fetchone()
            (prefixes if hit else typos).append(pre if hit else t)
        return prefixes, typos

    def _fuzzy(self, typos: list[str], pool: int, keep: int = 8) -> list[int]:
        """Typo-tolerant signal: match the rarest character trigrams of misspelled words.

        Rare trigrams carry identity (entity names); common ones carry noise and
        dominate the cost, so only the `keep` rarest present trigrams are queried.
        """
        grams = sorted({t[i:i + 3] for t in typos if len(t) >= 5 for i in range(len(t) - 2)})
        if not grams:
            return []
        df = self.conn.execute(
            f"SELECT term, doc FROM memories_tri_vocab WHERE term IN ({','.join('?' * len(grams))})", grams).fetchall()
        rare = [g for g, _ in sorted(df, key=lambda r: r[1])[:keep]]
        if not rare:
            return []
        q = " OR ".join('"' + g.replace('"', "") + '"' for g in rare)
        try:
            return [r[0] for r in self.conn.execute(
                "SELECT rowid FROM memories_tri WHERE memories_tri MATCH ? ORDER BY bm25(memories_tri) LIMIT ?",
                (q, pool))]
        except sqlite3.OperationalError:
            return []

    def history(self, key: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id,text,valid_from,valid_to,source FROM memories WHERE key=? ORDER BY valid_from, id",
            (normalize(key),))
        return [{"id": r[0], "text": r[1], "valid_from": r[2], "valid_to": r[3], "source": r[4]} for r in rows]

    def stats(self) -> dict[str, Any]:
        total, active = self.conn.execute(
            "SELECT COUNT(*), SUM(valid_to IS NULL) FROM memories").fetchone()
        size = sum(p.stat().st_size for p in self.path.parent.glob(self.path.name + "*") if p.is_file())
        return {"records": total, "active": active or 0, "bytes_on_disk": size,
                "bytes_per_record": round(size / total, 1) if total else None, "dim": DIM,
                "vector_bytes_per_record": DIM, "backend": kernels.backend()}
