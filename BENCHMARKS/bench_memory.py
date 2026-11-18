"""Memory Fabric benchmark: retrieval quality (with ablation), latency, ingest rate, storage per record.

Corpus: deterministic synthetic facts about invented entities (no contamination
from any pretrained knowledge, and the ground truth is known exactly). Each
query targets one fact and comes in three styles:
  * paraphrase — different wording, shares the entity and some content words
  * typo       — paraphrase with a misspelled entity name
  * arabic     — Arabic facts queried in Arabic with diacritics/alef variants

Usage: python BENCHMARKS/bench_memory.py --sizes 1000 10000 50000 --queries 300
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omega_nano.hw import detect  # noqa: E402
from omega_nano.memory import MemoryFabric  # noqa: E402

SYL = ["ka", "lo", "vi", "ren", "tor", "mi", "sa", "dun", "pel", "zor", "qua", "bri", "nex", "ol", "tam", "ves"]
REL = [
    ("is headquartered in {v}", "where is {e} headquartered", "{v}"),
    ("was founded in the year {n}", "what year was {e} founded", "{n}"),
    ("produces industrial {v} sensors", "what kind of sensors does {e} produce", "{v}"),
    ("is managed by director {v}", "who manages {e}", "{v}"),
    ("employs {n} engineers", "how many engineers work at {e}", "{n}"),
]
CITY = ["Aldmere", "Brovik", "Castelune", "Dorran", "Esterfall", "Fenwick", "Galdor", "Hollin", "Istra", "Jorvale"]
AR_REL = [("مقرها في مدينة {v}", "أين يقع مقر {e}"), ("أسسها المهندس {v}", "مَن أسَّس {e}")]
AR_V = ["القاهرة", "الإسكندرية", "أسوان", "طنطا", "المنصورة", "أحمد", "إبراهيم", "آمنة", "يوسف", "مريم"]
AR_SYL = ["سا", "مي", "رو", "كا", "لي", "نو", "دا", "في"]


def name(rng: random.Random, syl=SYL, n=3) -> str:
    return "".join(rng.choice(syl) for _ in range(n)).capitalize()


def typo(word: str, rng: random.Random) -> str:
    i = rng.randrange(1, len(word) - 1)
    return word[:i] + word[i + 1:]  # drop one interior character


def build(n: int, seed: int = 7):
    rng = random.Random(seed)
    facts, queries, used = [], [], set()
    n_ar = max(1, n // 10)
    while len(facts) < n - n_ar:
        e = name(rng) + name(rng, n=2)
        if e in used:
            continue
        used.add(e)
        tpl, q, _ = rng.choice(REL)
        v, num = rng.choice(CITY) + str(rng.randrange(100)), str(rng.randrange(1800, 2100))
        facts.append({"text": f"{e} {tpl.format(v=v, n=num)}", "key": None, "_q": q.format(e=e), "_e": e})
    while len(facts) < n:
        e = name(rng, AR_SYL) + " " + name(rng, AR_SYL, 2)
        if e in used:
            continue
        used.add(e)
        tpl, q = rng.choice(AR_REL)
        v = rng.choice(AR_V)
        facts.append({"text": f"شركة {e} {tpl.format(v=v)}", "_q": q.format(e=e), "_e": e, "_ar": True})
    for i, f in enumerate(facts):
        f["_id"] = i
    return facts, rng


def run(size: int, n_queries: int, out: dict) -> None:
    facts, rng = build(size)
    tmp = Path(tempfile.mkdtemp(prefix="omega_bench_"))
    mem = MemoryFabric(tmp / "m.db")
    t0 = time.perf_counter()
    mem.remember_many([{"text": f["text"], "source": "bench"} for f in facts])
    ingest_s = time.perf_counter() - t0
    # map ground-truth index -> memory id
    id_of = {r[0]: r[1] for r in mem.conn.execute("SELECT text, id FROM memories")}
    en = [f for f in facts if not f.get("_ar")]
    ar = [f for f in facts if f.get("_ar")]
    sample = {
        "paraphrase": [(f["_q"], id_of[f["text"]]) for f in rng.sample(en, min(n_queries, len(en)))],
        "typo": [(f["_q"].replace(f["_e"], typo(f["_e"], rng)), id_of[f["text"]])
                 for f in rng.sample(en, min(n_queries, len(en)))],
        "arabic": [(f["_q"], id_of[f["text"]]) for f in rng.sample(ar, min(n_queries, len(ar)))],
    }
    mem.recall("warmup query")
    res = {"records": size, "ingest_s": round(ingest_s, 3), "ingest_per_s": round(size / ingest_s),
           "storage": mem.stats(), "by_signals": {}}
    for label, signals in (("hybrid", ("lexical", "vector", "fuzzy")), ("lexical+vector", ("lexical", "vector")),
                           ("lexical_only", ("lexical",)), ("vector_only", ("vector",)), ("fuzzy_only", ("fuzzy",))):
        block = {}
        for style, qs in sample.items():
            lat, r1, r5, mrr = [], 0, 0, 0.0
            for q, want in qs:
                t = time.perf_counter()
                hits = mem.recall(q, k=5, signals=signals)
                lat.append((time.perf_counter() - t) * 1000)
                ids = [h.id for h in hits]
                if ids[:1] == [want]:
                    r1 += 1
                if want in ids:
                    r5 += 1
                    mrr += 1 / (ids.index(want) + 1)
            lat.sort()
            block[style] = {"n": len(qs), "recall@1": round(r1 / len(qs), 3), "recall@5": round(r5 / len(qs), 3),
                            "mrr@5": round(mrr / len(qs), 3), "p50_ms": round(statistics.median(lat), 2),
                            "p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 2)}
        res["by_signals"][label] = block
    mem.close()
    out["runs"].append(res)
    h = res["by_signals"]["hybrid"]
    print(f"N={size:>6}  ingest {res['ingest_per_s']}/s  {res['storage']['bytes_per_record']} B/rec  | hybrid "
          + "  ".join(f"{k}: R@1={v['recall@1']} R@5={v['recall@5']} p50={v['p50_ms']}ms p95={v['p95_ms']}ms"
                      for k, v in h.items()), flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 50000])
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--out", default=str(ROOT / "BENCHMARKS" / "results" / "memory.json"))
    a = ap.parse_args()
    out = {"benchmark": "memory-fabric-v1", "date": time.strftime("%Y-%m-%d %H:%M"), "hardware": detect(),
           "method": "synthetic invented-entity facts, 10% Arabic; single-threaded queries; warm cache", "runs": []}
    for n in a.sizes:
        run(n, a.queries, out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
