"""EXP-L experiment runner.

Usage:
    python -m constructive.exp_l.runner --stage L0 [--seed N] [--quick]
                                        [--no-resume] [--merge-only]

Quick runs write only ``*-quick.json`` artifacts and never overwrite principal
results.  Resource exhaustion is recorded as a TIMEOUT/FAILURE, never converted
into a successful abstention.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from typing import Any

from . import EXP_L_VERSION
from .config import STAGES, ExpLConfig, from_args
from .datasets import minimal_fingerprint
from .determinism import ROOT, fingerprint, make_rng
from .environment import capture_environment
from .integrity import git_info, verify_protected
from .reporting import (RESULTS, artifact_name, snapshot, update_integrity,
                        update_latest, write_json)
from .semantic_drift import run_l1
from .composition_verification import run_l2
from .retrieval_generalization import run_l3


def _git_lines(*args: str) -> list[str]:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                             text=True, timeout=20)
        return [ln for ln in (out.stdout or "").splitlines() if ln.strip()]
    except Exception:  # pragma: no cover
        return []


PROTECTED_PREFIXES = (
    "omega_nano/", "constructive/exp_h/", "constructive/exp_i/",
    "constructive/exp_j/", "constructive/exp_k/", "constructive/core.py",
    "constructive/model.py", "constructive/worlds.py",
    "constructive/baselines.py", "constructive/bench.py",
    "constructive/__init__.py",
)


def working_tree() -> dict[str, Any]:
    modified = _git_lines("diff", "--name-only")
    untracked = _git_lines("ls-files", "--others", "--exclude-standard")
    protected_modified = [p for p in modified if p.endswith(".py")
                          and any(p.startswith(pref) for pref in PROTECTED_PREFIXES)]
    return {
        "tracked_modified": modified,
        "protected_source_modified": protected_modified,
        "untracked": untracked,
        "clean": not modified and not untracked,
        "note": ("The legacy suite regenerates its own quick-run JSON result "
                 "files; these are restored after a milestone run. Untracked "
                 "EXP-H result files are preserved untouched per explicit "
                 "authorization; they are not EXP-L artifacts."),
    }


def run_full_tests(cfg: ExpLConfig) -> dict[str, Any]:
    """Re-run the existing full suite; a timeout/failure is recorded as such."""
    started = time.perf_counter()
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=ROOT, capture_output=True, text=True,
            timeout=cfg.full_test_timeout_s,
        )
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        summary = lines[-1] if lines else "(no output)"
        elapsed = time.perf_counter() - started
        return {
            "command": "python -m pytest -q",
            "returncode": out.returncode,
            "summary": summary,
            "elapsed_s": round(elapsed, 3),
            "passed": out.returncode == 0,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "command": "python -m pytest -q",
            "returncode": None,
            "summary": "TIMEOUT after configured limit",
            "elapsed_s": round(time.perf_counter() - started, 3),
            "passed": False,
            "timed_out": True,
        }


def smoke_check(cfg: ExpLConfig) -> dict[str, Any]:
    """Minimal deterministic fixture reproduced across independent generators."""
    a = minimal_fingerprint(cfg.seed)
    b = minimal_fingerprint(cfg.seed)
    c = minimal_fingerprint(cfg.seed + 1)
    r1 = make_rng(cfg.seed, "smoke")
    r2 = make_rng(cfg.seed, "smoke")
    seq1 = [r1.random() for _ in range(5)]
    seq2 = [r2.random() for _ in range(5)]
    ok = (a == b and a != c and seq1 == seq2)
    return {
        "fixture": "minimal_fixture/v1",
        "fingerprint": a,
        "other_seed_fingerprint": c,
        "rng_stream_reproducible": seq1 == seq2,
        "deterministic_reproduction": a == b and a != c,
        "passed": ok,
    }


def run_l0(cfg: ExpLConfig, *, run_full_suite: bool = True,
           stamp: str | None = None) -> dict[str, Any]:
    """Execute Stage L0 and write all required artifacts."""
    started = time.perf_counter()
    protected = verify_protected()
    env = capture_environment()
    wt = working_tree()
    smoke = smoke_check(cfg)
    tests = run_full_tests(cfg) if run_full_suite else {
        "command": "python -m pytest -q", "returncode": None,
        "summary": "SKIPPED (debug invocation)", "elapsed_s": 0.0,
        "passed": True, "timed_out": False, "skipped_by_flag": True,
    }

    gate = {
        "protected_hashes_ok": protected["all_protected_ok"],
        "no_protected_source_modifications": not wt["protected_source_modified"],
        "full_tests_ok": tests["passed"],
        "smoke_ok": smoke["passed"],
    }
    gate_pass = all(gate.values())
    elapsed = time.perf_counter() - started

    baseline = {
        "stage": "L0",
        "schema": "exp-l-stage-block-v1",
        "exp_l_version": EXP_L_VERSION,
        "git": git_info(),
        "config": cfg.as_dict(),
        "config_fingerprint": cfg.config_fingerprint,
        "working_tree": wt,
        "smoke_test": smoke,
        "full_test_suite": tests,
        "gate": gate,
        "gate_pass": gate_pass,
        "wall_s": round(elapsed, 3),
        "captured_utc": env["captured_utc"],
    }
    integrity_block = {
        "stage": "L0", "schema": "exp-l-integrity-v1", **protected,
    }
    environment_block = {
        "stage": "L0", "schema": "exp-l-environment-v1", **env,
    }
    block = {
        **baseline,
        "integrity": integrity_block,
        "environment": environment_block,
    }
    block["fingerprint"] = fingerprint({k: v for k, v in block.items()
                                        if k != "fingerprint"})

    # Required outputs. Never overwrite principal files on a quick run.
    write_json(RESULTS / artifact_name("L0", "baseline", cfg.quick), baseline)
    write_json(RESULTS / artifact_name("L0", "integrity", cfg.quick), integrity_block)
    write_json(RESULTS / artifact_name("L0", "environment", cfg.quick), environment_block)
    snap = snapshot("L0", block, cfg.quick, stamp)
    try:
        block["snapshot"] = str(snap.relative_to(ROOT))
    except ValueError:  # snapshot redirected to a tmp dir (test harness)
        block["snapshot"] = str(snap)
    update_integrity({"stage": "L0", "gate": gate, "gate_pass": gate_pass,
                      "protected": protected["all_protected_ok"],
                      "snapshot": block["snapshot"]}, cfg.quick)
    update_latest("L0", block, cfg.quick)
    return block


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m constructive.exp_l.runner")
    p.add_argument("--stage", choices=STAGES, default="L0")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--merge-only", action="store_true")
    return p


def persist_block(stage: str, block: dict[str, Any], cfg: ExpLConfig,
                  stamp: str | None = None) -> dict[str, Any]:
    """Write the standard artifacts for a stage block (quick runs isolated)."""
    write_json(RESULTS / artifact_name(stage, "results", cfg.quick), block)
    snap = snapshot(stage, block, cfg.quick, stamp)
    try:
        block["snapshot"] = str(snap.relative_to(ROOT))
    except ValueError:
        block["snapshot"] = str(snap)
    update_integrity({"stage": stage, "gate": block.get("gate", {}),
                      "gate_pass": block.get("gate_pass", False),
                      "snapshot": block["snapshot"]}, cfg.quick)
    update_latest(stage, block, cfg.quick)
    return block


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = from_args(args)
    if cfg.merge_only:
        from .reporting import read_json
        latest = read_json(RESULTS / ("latest-quick.json" if cfg.quick else "latest.json"))
        print("merge-only: stages =", sorted(latest.get("stages", {})))
        return 0
    if cfg.stage == "L0":
        block = run_l0(cfg)
    elif cfg.stage == "L1":
        block = persist_block("L1", run_l1(cfg), cfg)
    elif cfg.stage == "L2":
        block = persist_block("L2", run_l2(cfg), cfg)
    elif cfg.stage == "L3":
        block = persist_block("L3", run_l3(cfg), cfg)
    else:
        print(f"Stage {cfg.stage} is not implemented yet in this milestone.")
        return 2
    print(f"{cfg.stage} gate_pass={block['gate_pass']} snapshot={block.get('snapshot')}")
    print("gate:", block["gate"])
    return 0 if block["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
