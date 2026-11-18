"""EXP-H experiment orchestration.

Stages:
  A  smoke tests (``--stage A``) — generators/Ast safety/tiny fits
  B  pilot       (``--stage B``) — small seeds/grids, every experiment runs
  C  campaign    (``--stage C``) — the full bounded campaign (default)

Usage:
  python -m constructive.exp_h.runner              # stage C
  python -m constructive.exp_h.runner --stage A
  python -m constructive.exp_h.runner --only H1 H3
  python -m constructive.exp_h.runner --quick
  python -m constructive.exp_h.runner --merge-only # rebuild latest.json
  python -m constructive.exp_h.runner --no-resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ExpHConfig, probe_hardware, fingerprint_of
from .analysis import summary_table, failure_taxonomy
from ..model import ConstructiveModel  # noqa: F401  (public-interface check)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
LATEST = RESULTS_DIR / "latest.json"
HISTORY_DIR = RESULTS_DIR / "experiment-history"

H1 = "H1"; H2 = "H2"; H3 = "H3"; H4 = "H4"; H5 = "H5"; H6 = "H6"; H7 = "H7"; H8 = "H8"
ALL = (H1, H2, H3, H4, H5, H6, H7, H8)


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _source_revision() -> str:
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        rev = (out.stdout or "").strip()
        if rev:
            return rev
        return "no-commits"
    except Exception:
        return "unknown"


def _fingerprint(exp_id: str, rows: list[dict[str, Any]]) -> str:
    names = sorted({r.get("task", "") for r in rows} or {r.get("pool_item", "") for r in rows})
    return fingerprint_of({"experiment": exp_id, "tasks": names})


def _conform(exp: dict[str, Any], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the JSON schema 4.2 / 9 requirements in one place."""
    exp = dict(exp)
    exp.setdefault("experiment_id", "H?")
    exp.setdefault("experiment_name", exp["experiment_id"])
    exp.setdefault("status", "completed")
    exp["episodes"] = episodes
    exp["software"] = probe_hardware().software
    exp["dataset_fingerprint"] = _fingerprint(exp["experiment_id"], episodes)
    exp["source_revision"] = _source_revision()
    exp["hardware"] = probe_hardware().hardware
    exp.setdefault("hardware_caveat",
                   "peak bytes are traced Python heap (tracemalloc), not OS RSS")
    exp.setdefault("configuration", {})
    exp.setdefault("training",
                   {"examples": sum(r.get("k", 0) for r in episodes),
                    "concepts": sum(1 for r in episodes if r.get("accepted"))})
    exp.setdefault("limitations", [])
    exp.setdefault("artifacts", [])
    return exp


def _save_experiment(exp_id: str, result: dict[str, Any], *, stamp: str) -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{exp_id}-{stamp}.json"
    (HISTORY_DIR / fname).write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8")
    (EXPERIMENTS_DIR / f"{exp_id}.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8")


def _load_experiment(exp_id: str) -> dict[str, Any] | None:
    path = EXPERIMENTS_DIR / f"{exp_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def run(only=None, stage: str = "C", quick: bool = False, merge_only: bool = False,
        resume: bool = True) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    seeds = tuple(range(10))
    if quick:
        seeds = (0,)
    cfg = ExpHConfig(seeds=seeds, quick=quick, h4_max_concepts=32 if quick else 100,
                     h2_max_depth=8 if quick else 32)
    stamp = _stamp()
    targets = tuple(only or ALL)

    from . import h1_discovery, h2_scaling, h3_ood, h4_continual, h5_ast, h6_abstraction, h7_adversarial, h8_efficiency

    dispatch = {
        H1: lambda: h1_discovery.run_h1(cfg, seeds=cfg.seeds if not quick else seeds[:1]),
        H2: lambda: h2_scaling.run_h2(cfg, seeds=cfg.seeds[:3] if not quick else seeds[:1],
                                      depths=(2, 3, 5, 8, 12, 16) if not quick else (2, 4),
                                      time_budget_s=2.0 if quick else 3.0),
        H3: lambda: h3_ood.run_h3(cfg, seeds=cfg.seeds if not quick else seeds[:2]),
        H4: lambda: h4_continual.run_h4(cfg, seeds=cfg.seeds[:1] if quick else cfg.seeds[:2],
                                        n_concepts=cfg.h4_max_concepts,
                                        time_budget_s=2.0 if quick else 2.5,
                                        max_nodes=150_000),
        H5: lambda: h5_ast.run_h5(cfg, seeds=cfg.seeds[:4] if quick else cfg.seeds[:8],
                                  max_candidates=8000 if quick else 50_000),
        H6: lambda: h6_abstraction.run_h6(cfg, seeds=cfg.seeds[:1] if quick else cfg.seeds[:3],
                                          k=4),
        H7: lambda: h7_adversarial.run_h7(cfg, seeds=cfg.seeds[:3] if quick else cfg.seeds[:6]),
        H8: lambda: h8_efficiency.run_h8(cfg, seeds=cfg.seeds[:2] if quick else cfg.seeds[:3]),
    }

    if merge_only:
        merged = _merge_latest(files=True)
        print(json.dumps(summary_table(merged), indent=2))
        return merged

    results: dict[str, Any] = {}
    failures: dict[str, str] = {}
    if stage == "A":
        from .smoke import run_stage_a
        block = run_stage_a()
        print(json.dumps(block, indent=2))
        (RESULTS_DIR / f"stage-a-{stamp}.json").write_text(
            json.dumps(block, indent=2, default=str), encoding="utf-8")
        (RESULTS_DIR / "stage-a.json").write_text(
            json.dumps(block, indent=2, default=str), encoding="utf-8")
        return block
    for exp_id in targets:
        cached = _load_experiment(exp_id) if resume else None
        if cached is not None and stage != "A":
            print(f"[{stamp}] {exp_id}: resumed from {EXPERIMENTS_DIR / (exp_id + '.json')}", flush=True)
            results[exp_id] = cached
            continue
        try:
            print(f"[{stamp}] {exp_id}: running ...", flush=True)
            t0 = time.perf_counter()
            t0 = time.perf_counter()
            res = dispatch[exp_id]()
            res = _conform(res, res.get("episodes", []))
            res["wall_s"] = res.get("wall_s", time.perf_counter() - t0)
            _save_experiment(exp_id, res, stamp=stamp)
            results[exp_id] = res
            print(f"[{stamp}] {exp_id}: done in {res['wall_s']:.1f}s "
                  f"({len(res.get('episodes', []))} episodes, status={res.get('status')})",
                  flush=True)
        except Exception as exc:  # record failure, keep going; it is data
            failures[exp_id] = f"{type(exc).__name__}: {exc}"
            import traceback
            tb = traceback.format_exc()
            fail_block = {
                "experiment_id": exp_id, "experiment_name": exp_id,
                "status": "error", "error": failures[exp_id], "traceback": tb,
                "episodes": [], "wall_s": 0.0,
                "software": probe_hardware().software,
                "hardware": probe_hardware().hardware,
                "dataset_fingerprint": fingerprint_of({"error": tb, "exp": exp_id}),
                "source_revision": _source_revision(),
                "limitations": ["experiment crashed; analyze traceback"],
                "artifacts": [],
            }
            _save_experiment(exp_id, fail_block, stamp=stamp)
            results[exp_id] = fail_block
            print(f"[{stamp}] {exp_id}: FAILED -> {failures[exp_id]}", flush=True)

    merged = {"schema": "exp_h-latest/1", "stamp": stamp, "stage": stage,
              "quick": quick, "source_revision": _source_revision(),
              "hardware": probe_hardware().hardware, "software": probe_hardware().software,
              "experiments": {eid: results[eid] for eid in results},
              "failures": failures}
    # merge re-run experiments with the cached ones on disk
    merged_exp = dict(results)
    for eid in ALL:
        if eid not in merged_exp:
            cached = _load_experiment(eid)
            if cached is not None:
                merged_exp[eid] = cached
    merged["experiments"] = {eid: merged_exp[eid] for eid in ALL if eid in merged_exp}
    merged["analysis"] = {
        "summary_table": summary_table({e: r for e, r in merged_exp.items()
                                        if e in ALL}),
        "failure_taxonomy": failure_taxonomy({e: r for e, r in merged_exp.items()
                                              if e in ALL}),
    }
    LATEST.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
    print(f"[{stamp}] latest.json written to {LATEST}", flush=True)
    if failures:
        print(f"[{stamp}] FAILED experiments: {', '.join(failures)}", flush=True)
    return merged


def _merge_latest(files: bool = False) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for eid in ALL:
        block = _load_experiment(eid)
        if block is not None:
            results[eid] = block
    stamp = _stamp()
    merged = {"schema": "exp_h-latest/1", "stamp": stamp, "stage": "merge",
              "quick": False, "source_revision": _source_revision(),
              "hardware": probe_hardware().hardware, "software": probe_hardware().software,
              "experiments": results,
              "failures": {k: v for k, v in ((e, results[e].get("error")) for e in results)
                           if v}}
    merged["analysis"] = {
        "summary_table": summary_table({e: r for e, r in results.items()}),
        "failure_taxonomy": failure_taxonomy({e: r for e, r in results.items()}),
    }
    LATEST.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
    return merged


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="EXP-H runner")
    ap.add_argument("--stage", default="C", choices=("A", "B", "C"))
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)
    run(only=args.only, stage=args.stage, quick=args.quick,
        merge_only=args.merge_only, resume=not args.no_resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
