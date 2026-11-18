"""EXP-J orchestration, persistence, integrity verification and CLI.

Usage:
  python -m constructive.exp_j.runner --stage A
  python -m constructive.exp_j.runner --quick
  python -m constructive.exp_j.runner
  python -m constructive.exp_j.runner --only J1 J3
  python -m constructive.exp_j.runner --no-resume
  python -m constructive.exp_j.runner --merge-only
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from . import analysis
from . import smoke
from .config import (ExpJConfig, PROTECTED, RESULTS_DIR, fingerprint_of,
                     probe_hardware, quick_config, reproducibility_metadata,
                     sha256_file, source_revision)
from .resources import ResourceGovernor, ResourceLimit, run_isolated
from .types import CompletionState

EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
HISTORY_DIR = RESULTS_DIR / "experiment-history"
LATEST = RESULTS_DIR / "latest.json"
BASELINE_HASHES = RESULTS_DIR / "baseline_hashes.json"
INTEGRITY = RESULTS_DIR / "integrity.json"
STAGE_A = RESULTS_DIR / "stage-a-smoke.json"

_ROOT = Path(__file__).resolve().parents[2]
ALL = analysis.ALL


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _paths() -> None:
    for p in (RESULTS_DIR, EXPERIMENTS_DIR, HISTORY_DIR):
        p.mkdir(parents=True, exist_ok=True)


def _record_baseline_hashes() -> dict[str, str]:
    _paths()
    hashes = {f: sha256_file(_ROOT / f) for f in PROTECTED}
    BASELINE_HASHES.write_text(
        json.dumps({
            "recorded_at": _stamp(), "algorithm": "SHA-256",
            "note": "EXP-J STAGE 0: recorded before any EXP-J execution against protected files",
            "n_files": len(hashes), "files": hashes,
        }, indent=2), encoding="utf-8")
    return hashes


def verify_integrity() -> dict[str, Any]:
    if not BASELINE_HASHES.exists():
        _record_baseline_hashes()
    data = json.loads(BASELINE_HASHES.read_text(encoding="utf-8"))
    baseline = data["files"]
    changed = []
    for f, expected in baseline.items():
        p = _ROOT / f
        if not p.exists():
            changed.append({"file": f, "issue": "missing"})
        elif sha256_file(p) != expected:
            changed.append({"file": f, "issue": "hash mismatch"})
    report = {
        "verified_at": _stamp(), "n_files": len(baseline),
        "protected_files_unchanged": not changed,
        "changed": changed, "source_revision": source_revision(),
        "scope": "core + omega_nano + exp_h + exp_i + pre-existing tests",
    }
    INTEGRITY.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _dispatch(eid: str, cfg: ExpJConfig) -> Callable[[], dict[str, Any]]:
    from . import abstraction_economics as J4
    from . import deep_composition as J3
    from . import interactions as J5
    from . import robustness as J2
    from . import transfer as J1

    return {
        "J1": lambda: J1.run_transfer(cfg),
        "J2": lambda: J2.run_noise_drift(cfg),
        "J3": lambda: J3.run_deep(cfg),
        "J4": lambda: J4.run_abstraction_economics(cfg),
        "J5": lambda: J5.run_interactions(cfg),
    }[eid]


def _conform(eid: str, block: dict[str, Any], cfg: ExpJConfig,
             governor: ResourceGovernor) -> dict[str, Any]:
    block = dict(block)
    block.setdefault("experiment_id", eid)
    block.setdefault("status", "completed")
    block["reproducibility"] = reproducibility_metadata(cfg)
    block["resource_report"] = governor.report()
    timed = bool(block.get("resources", {}).get("timed_out")) or \
        any(x.get("status") == CompletionState.TIMED_OUT.value
            for x in block.get("checks", []) if isinstance(x, dict))
    if timed:
        completion = CompletionState.TIMED_OUT.value
    elif block.get("status") == "completed":
        completion = CompletionState.COMPLETED.value
    else:
        completion = CompletionState.PARTIAL.value
    block["completion_status"] = completion
    episodes = block.get("episodes", [])
    block["dataset_fingerprint"] = fingerprint_of(
        {"experiment": eid,
         "tasks": [e.get("task", e.get("hidden", e.get("name", "")))
                   for e in episodes]})
    block.setdefault("hardware", probe_hardware()["hardware"])
    block.setdefault("limitations", ["Synthetic typed-DSL tasks; no natural-input claim."])
    return block


def _save(eid: str, block: dict[str, Any], stamp: str) -> None:
    _paths()
    payload = json.dumps(block, indent=2, default=str)
    (HISTORY_DIR / f"{eid}-{stamp}.json").write_text(payload, encoding="utf-8")
    (EXPERIMENTS_DIR / f"{eid}.json").write_text(payload, encoding="utf-8")


def _load(eid: str) -> dict[str, Any] | None:
    p = EXPERIMENTS_DIR / f"{eid}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def merge_only(cfg: ExpJConfig) -> dict[str, Any]:
    _paths()
    blocks = {eid: b for eid in ALL if (b := _load(eid)) is not None}
    integrity = verify_integrity()
    # J6 is a recomputed efficiency block, never a stored mutable artifact.
    blocks["J6"] = {
        "experiment_id": "J6", "experiment_name": "Computational Efficiency",
        "status": "completed", "completion_status": CompletionState.COMPLETED.value,
        **analysis.efficiency(blocks),
    }
    body = {
        "experiment": "EXP-J", "stamp": _stamp(), "status": "merged",
        "experiments": blocks,
        "experiment_status": analysis.experiment_status(blocks),
        "failure_taxonomy": analysis.failure_taxonomy(blocks),
        "headline_metrics": analysis.headline_metrics(blocks),
        "independence_evidence": analysis.independence_evidence(blocks),
        "integration_readiness": analysis.integration_readiness(blocks, integrity),
        "integrity": integrity,
        "reproducibility": reproducibility_metadata(cfg),
    }
    LATEST.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    return body


def run(only: tuple[str, ...] | None = None, stage: str = "C",
        quick: bool = False, no_resume: bool = False,
        merge_only_flag: bool = False) -> dict[str, Any]:
    _paths()
    if not BASELINE_HASHES.exists():
        _record_baseline_hashes()
    if stage == "A":
        result = smoke.run_stage_a()
        STAGE_A.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    if merge_only_flag:
        cfg = quick_config() if quick else ExpJConfig()
        return merge_only(cfg)

    cfg = quick_config() if quick else ExpJConfig()
    stamp = _stamp()
    global_gov = ResourceGovernor(global_deadline_s=cfg.campaign_deadline_s)
    errors: list[dict[str, Any]] = []
    targets = tuple(only or ALL[:-1])  # J6 computed at merge
    for eid in targets:
        if not no_resume and _load(eid) is not None:
            continue
        gov = ResourceGovernor(
            global_deadline_s=max(1.0, cfg.campaign_deadline_s - global_gov.elapsed_s),
            node_budget=cfg.search_node_budget * 400,
        )
        block: dict[str, Any] | None = None
        try:
            with gov.scope(eid, deadline_s=cfg.experiment_deadline_s):
                gov.checkpoint(eid)
                block = _dispatch(eid, cfg)()
            block = _conform(eid, block, cfg, gov)
        except ResourceLimit as exc:
            block = {
                "experiment_id": eid, "status": "partial",
                "reason": f"{exc.kind}: {exc.detail}",
                "completion_status": CompletionState.TIMED_OUT.value,
                "resource_report": gov.report(),
                "partial_diagnostics": gov.diagnostics,
            }
        except BaseException as exc:  # noqa: BLE001
            errors.append({"experiment": eid, "error": f"{type(exc).__name__}: {exc}"})
            block = {
                "experiment_id": eid, "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "completion_status": CompletionState.ERROR.value,
                "resource_report": gov.report(),
            }
        _save(eid, block, stamp)
    body = merge_only(cfg)
    if errors:
        body.setdefault("runner_errors", []).extend(errors)
    return body


def main() -> None:
    ap = argparse.ArgumentParser(description="EXP-J research runner")
    ap.add_argument("--stage", default="C", choices=["A", "C"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()
    body = run(only=tuple(args.only) if args.only else None, stage=args.stage,
               quick=args.quick, no_resume=args.no_resume,
               merge_only_flag=args.merge_only)
    print(json.dumps({
        "stage": body.get("stage"), "experiment": body.get("experiment"),
        "n_checks": body.get("n_checks"), "passed": body.get("passed"),
        "failures": body.get("failures", len(body.get("runner_errors", []))),
        "experiment_status": body.get("experiment_status"),
        "integrity": body.get("integrity", {}).get("protected_files_unchanged"),
        "runner_errors": body.get("runner_errors", []),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
