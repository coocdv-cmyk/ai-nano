"""EXP-I orchestration, persistence and integrity verification.

Usage:
  python -m constructive.exp_i.runner --stage A     # smoke checks
  python -m constructive.exp_i.runner --quick        # fast pilot
  python -m constructive.exp_i.runner                # full campaign
  python -m constructive.exp_i.runner --only I-A I-C
  python -m constructive.exp_i.runner --merge-only   # rebuild latest.json
  python -m constructive.exp_i.runner --no-resume

Results are immutable: each run is preserved under results/experiment-history/.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from . import experiments as EX
from . import smoke
from .config import (ExpIConfig, RESULTS_DIR, fingerprint_of, probe_hardware,
                     reproducibility_metadata, sha256_file, source_revision)
from .resources import ResourceLimit, ResourceGovernor, run_isolated
from .types import CompletionStatus, ValidationState

EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
HISTORY_DIR = RESULTS_DIR / "experiment-history"
LATEST = RESULTS_DIR / "latest.json"
BASELINE_HASHES = RESULTS_DIR / "baseline_hashes.json"
INTEGRITY = RESULTS_DIR / "integrity.json"

ALL = ("I-A", "I-B", "I-C", "I-D", "I-E", "I-F", "I-G", "I-H")

# Protected files recorded at STAGE 0. Never modified by EXP-I.
PROTECTED = (
    "constructive/core.py", "constructive/model.py", "constructive/worlds.py",
    "constructive/baselines.py", "constructive/bench.py", "constructive/__init__.py",
    "constructive/exp_h/runner.py", "constructive/exp_h/smoke.py",
    "constructive/exp_h/engine.py", "constructive/exp_h/generators.py",
    "constructive/exp_h/episode.py", "constructive/exp_h/config.py",
    "constructive/exp_h/metrics.py", "constructive/exp_h/h4_pool.py",
    "tests/test_exp_h.py", "tests/test_constructive.py", "tests/conftest.py",
    "omega_nano/engine.py", "omega_nano/memory.py", "omega_nano/model.py",
    "omega_nano/reasoner.py", "omega_nano/core/engine.py",
)

_ROOT = Path(__file__).resolve().parents[2]


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _paths() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _record_baseline_hashes() -> dict[str, str]:
    _paths()
    hashes = {f: sha256_file(_ROOT / f) for f in PROTECTED}
    BASELINE_HASHES.write_text(
        json.dumps({"recorded_at": _stamp(), "algorithm": "SHA-256",
                    "note": "recorded at STAGE 0 before any EXP-I implementation",
                    "files": hashes}, indent=2), encoding="utf-8")
    return hashes


def verify_integrity() -> dict[str, Any]:
    """Re-verify protected files against the STAGE-0 baseline manifest."""
    if not BASELINE_HASHES.exists():
        _record_baseline_hashes()
    data = json.loads(BASELINE_HASHES.read_text(encoding="utf-8"))
    baseline = data["files"]
    changed = []
    for f, expected in baseline.items():
        path = _ROOT / f
        if not path.exists():
            changed.append({"file": f, "issue": "missing"})
        elif sha256_file(path) != expected:
            changed.append({"file": f, "issue": "hash mismatch"})
    report = {
        "verified_at": _stamp(),
        "n_files": len(baseline),
        "protected_files_unchanged": not changed,
        "changed": changed,
        "source_revision": source_revision(),
    }
    INTEGRITY.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _dispatch(exp_id: str, cfg: ExpIConfig) -> Callable[[], dict[str, Any]]:
    return {
        "I-A": lambda: EX.run_ia(cfg),
        "I-B": lambda: EX.run_ib(cfg),
        "I-C": lambda: EX.run_ic(cfg),
        "I-D": lambda: EX.run_id(cfg),
        "I-E": lambda: EX.run_ie(cfg),
        "I-F": lambda: EX.run_if(cfg),
        "I-G": lambda: EX.run_ig(cfg),
        "I-H": lambda: EX.run_ih(cfg),
    }[exp_id]


def _conform(exp_id: str, block: dict[str, Any], cfg: ExpIConfig,
             governor: ResourceGovernor) -> dict[str, Any]:
    block = dict(block)
    block.setdefault("experiment_id", exp_id)
    block.setdefault("status", "completed")
    block["reproducibility"] = reproducibility_metadata(cfg)
    block["resource_report"] = governor.report()
    block["completion_status"] = (
        CompletionStatus.TIMED_OUT.value if result_timed_out(block)
        else CompletionStatus.COMPLETED.value if block["status"] == "completed"
        else CompletionStatus.PARTIAL.value)
    block["dataset_fingerprint"] = fingerprint_of(
        {"experiment": exp_id,
         "episodes": [r.get("task", r.get("name", r.get("rule", "")))
                      for r in block.get("episodes", [])]})
    block.setdefault("hardware", probe_hardware()["hardware"])
    block.setdefault("software", probe_hardware()["software"])
    block.setdefault("limitations", [
        "Synthetic tasks in a small typed DSL; no natural-language validity claimed.",
        "Results are CPU-only and local to these generator families.",
    ])
    return block


def result_timed_out(block: dict[str, Any]) -> bool:
    return any(r.get("timed_out") for r in block.get("episodes", [])) or \
        any(c.get("status") == CompletionStatus.TIMED_OUT.value
            for c in block.get("checks", []) if isinstance(c, dict) and "status" in c) and \
        block.get("status") != "completed"


def _save(exp_id: str, block: dict[str, Any], stamp: str) -> None:
    _paths()
    payload = json.dumps(block, indent=2, default=str)
    (HISTORY_DIR / f"{exp_id}-{stamp}.json").write_text(payload, encoding="utf-8")
    (EXPERIMENTS_DIR / f"{exp_id}.json").write_text(payload, encoding="utf-8")


def _load(exp_id: str) -> dict[str, Any] | None:
    path = EXPERIMENTS_DIR / f"{exp_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def merge_only(cfg: ExpIConfig) -> dict[str, Any]:
    from .analysis import experiment_status, failure_taxonomy, integration_readiness
    from . import analysis

    _paths()
    blocks = {eid: b for eid in ALL if (b := _load(eid)) is not None}
    integrity = verify_integrity()
    body = {
        "experiment": "EXP-I",
        "stamp": _stamp(),
        "status": "merged",
        "experiments": blocks,
        "experiment_status": experiment_status(blocks),
        "failure_taxonomy": failure_taxonomy(blocks),
        "headline_metrics": analysis.headline_metrics(blocks),
        "integration_readiness": integration_readiness(blocks, integrity),
        "integrity": integrity,
        "reproducibility": reproducibility_metadata(cfg),
    }
    LATEST.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    return body


def run(only: tuple[str, ...] | None = None, stage: str = "C", quick: bool = False,
        no_resume: bool = False, merge_only: bool = False) -> dict[str, Any]:
    _paths()
    if not BASELINE_HASHES.exists():
        _record_baseline_hashes()
    if stage == "A":
        result = smoke.run_stage_a()
        (RESULTS_DIR / "stage-a-smoke.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        return result
    if merge_only:
        return merge_only_cfg(quick)

    cfg = ExpIConfig(
        seeds=(0, 1, 2, 3, 4) if not quick else (0,),
        n_episodes=30 if not quick else 8,
        quick=quick,
    )
    targets = tuple(only or ALL)
    stamp = _stamp()
    global_gov = ResourceGovernor(global_deadline_s=cfg.campaign_deadline_s)
    errors: list[dict[str, Any]] = []
    for exp_id in targets:
        if not no_resume and _load(exp_id) is not None:
            continue
        block: dict[str, Any] | None = None
        gov = ResourceGovernor(
            global_deadline_s=max(1.0, cfg.campaign_deadline_s - global_gov.elapsed_s),
            node_budget=cfg.search_node_budget * 40,
        )

        def execute(eid: str = exp_id) -> dict[str, Any]:
            with gov.scope(eid, deadline_s=cfg.experiment_deadline_s):
                return _dispatch(eid, cfg)()

        try:
            block = execute()
            block = _conform(exp_id, block, cfg, gov)
        except ResourceLimit as exc:
            block = {
                "experiment_id": exp_id, "status": "partial",
                "reason": f"{exc.kind}: {exc.detail}",
                "completion_status": CompletionStatus.TIMED_OUT.value,
                "resource_report": gov.report(),
                "partial_diagnostics": gov.diagnostics,
            }
        except BaseException as exc:  # noqa: BLE001
            errors.append({"experiment": exp_id, "error": f"{type(exc).__name__}: {exc}"})
            block = {
                "experiment_id": exp_id, "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "completion_status": CompletionStatus.ERROR.value,
                "resource_report": gov.report(),
            }
        _save(exp_id, block, stamp)

    return merge_only_cfg(quick, extra_errors=errors)


def merge_only_cfg(quick: bool = False,
                   extra_errors: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    cfg = ExpIConfig(seeds=(0,) if quick else (0, 1, 2, 3, 4), quick=quick)
    body = merge_only(cfg)
    if extra_errors:
        body.setdefault("runner_errors", []).extend(extra_errors)
    return body


def main() -> None:
    ap = argparse.ArgumentParser(description="EXP-I research runner")
    ap.add_argument("--stage", default="C", choices=["A", "C"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()
    body = run(only=tuple(args.only) if args.only else None, stage=args.stage,
               quick=args.quick, no_resume=args.no_resume,
               merge_only=args.merge_only)
    print(json.dumps({
        "stage": body.get("stage"),
        "experiment": body.get("experiment"),
        "n_checks": body.get("n_checks"),
        "passed": body.get("passed"),
        "failures": body.get("failures", len(body.get("runner_errors", []))),
        "experiment_status": body.get("experiment_status"),
        "integrity": body.get("integrity", {}).get("protected_files_unchanged"),
        "runner_errors": body.get("runner_errors", []),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
