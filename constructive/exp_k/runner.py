"""EXP-K orchestration, persistence and integrity verification.

Usage:
  python -m constructive.exp_k.runner --stage A
  python -m constructive.exp_k.runner --stage K0
  python -m constructive.exp_k.runner --stage K0 --quick
  python -m constructive.exp_k.runner --merge-only
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from . import baseline, smoke
from .config import (RESULTS_DIR, ExpKConfig, HISTORY_DIR, MANIFESTS_DIR,
                     quick_config, reproducibility)
from .resources import ResourceGovernor, ResourceLimit
from .types import Completion

EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
LATEST = RESULTS_DIR / "latest.json"
INTEGRITY = RESULTS_DIR / "integrity.json"
STAGE_A = RESULTS_DIR / "stage-a-smoke.json"

STAGES = ("K0", "K1", "K2", "K3", "K4", "K5", "K6", "K7")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _paths() -> None:
    for p in (RESULTS_DIR, EXPERIMENTS_DIR, HISTORY_DIR, MANIFESTS_DIR):
        p.mkdir(parents=True, exist_ok=True)


def _dispatch(stage: str, cfg: ExpKConfig) -> Callable[[], dict[str, Any]]:
    if stage == "K0":
        return lambda: baseline.run_k0(cfg)
    if stage == "K1":
        from . import deep_composition
        return lambda: deep_composition.run_k1(cfg)
    if stage == "K2":
        from . import retrieval_ood
        return lambda: retrieval_ood.run_k2(cfg)
    if stage == "K3":
        from . import drift_detection
        return lambda: drift_detection.run_k3(cfg)
    if stage == "K4":
        from . import calibration
        return lambda: calibration.run_k4(cfg)
    if stage == "K5":
        from . import interactions
        return lambda: interactions.run_k5(cfg)
    if stage == "K6":
        from . import adversarial
        return lambda: adversarial.run_k6(cfg)
    if stage == "K7":
        from . import finalize
        return lambda: finalize.run_k7(cfg)
    raise KeyError(f"stage {stage} is not implemented in this revision")


def _save(stage: str, block: dict[str, Any], stamp: str, *, quick: bool = False) -> None:
    """Persist a stage block. Quick runs never overwrite principal results."""
    _paths()
    payload = json.dumps(block, indent=2, default=str)
    suffix = "-quick" if quick else ""
    (HISTORY_DIR / f"{stage}{suffix}-{stamp}.json").write_text(payload, encoding="utf-8")
    (EXPERIMENTS_DIR / f"{stage}{suffix}.json").write_text(payload, encoding="utf-8")


def verify_integrity() -> dict[str, Any]:
    report = baseline.integrity_snapshot()
    _paths()
    INTEGRITY.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def merge_only(cfg: ExpKConfig) -> dict[str, Any]:
    _paths()
    blocks = {}
    for stage in STAGES:
        p = EXPERIMENTS_DIR / f"{stage}.json"
        if p.exists():
            try:
                blocks[stage] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    integ = verify_integrity()
    body = {
        "experiment": "EXP-K", "stamp": _stamp(), "status": "merged",
        "stages": list(blocks), "experiments": blocks,
        "integrity": integ,
        "protected_files_unchanged": all(
            v.get("all_unchanged", False) for v in integ["manifests"].values()),
        "reproducibility": reproducibility(cfg),
    }
    LATEST.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    return body


def run(stage: str = "K0", quick: bool = False, no_resume: bool = False,
        merge_only_flag: bool = False) -> dict[str, Any]:
    _paths()
    cfg = quick_config() if quick else ExpKConfig()
    if stage == "A":
        result = smoke.run_stage_a()
        STAGE_A.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    if merge_only_flag:
        return merge_only(cfg)
    gov = ResourceGovernor(global_deadline_s=cfg.campaign_deadline_s)
    stamp = _stamp()
    block: dict[str, Any]
    try:
        with gov.scope(stage, deadline_s=cfg.experiment_deadline_s):
            gov.checkpoint(stage)
            block = _dispatch(stage, cfg)()
        block["completion_status"] = (
            Completion.TIMED_OUT.value if block.get("resources", {}).get("timed_out")
            else Completion.COMPLETED.value)
        block["resource_report"] = gov.report()
        block["reproducibility"] = reproducibility(cfg)
    except ResourceLimit as exc:
        block = {"stage": stage, "status": "partial",
                 "reason": f"{exc.kind}: {exc.detail}",
                 "completion_status": Completion.TIMED_OUT.value,
                 "resource_report": gov.report()}
    except BaseException as exc:  # noqa: BLE001
        block = {"stage": stage, "status": "error",
                 "error": f"{type(exc).__name__}: {exc}",
                 "completion_status": Completion.ERROR.value,
                 "resource_report": gov.report()}
    _save(stage, block, stamp, quick=cfg.quick)
    return merge_only(cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="EXP-K research runner")
    ap.add_argument("--stage", default="K0", choices=["A", "K0", "K1", "K2", "K3",
                                                      "K4", "K5", "K6", "K7"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()
    body = run(stage=args.stage, quick=args.quick, no_resume=args.no_resume,
               merge_only_flag=args.merge_only)
    if args.stage == "A":
        print(json.dumps({k: body.get(k) for k in
                          ("stage", "status", "n_checks", "passed", "failures")}, indent=2))
    else:
        print(json.dumps({
            "experiment": body.get("experiment"),
            "stages": body.get("stages"),
            "protected_files_unchanged": body.get("protected_files_unchanged"),
        }, indent=2))


if __name__ == "__main__":
    main()
