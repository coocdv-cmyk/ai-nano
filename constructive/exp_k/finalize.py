"""K7 — final reproduction and integrity verification."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from . import analysis
from . import calibration, drift_detection, retrieval_ood
from .config import (EXPJ_BASELINE, EXPI_BASELINE, RESULTS_DIR, ExpKConfig,
                     git_branch, git_commit, reproducibility, sha256_file)
from .resources import Monitor


def verify_manifests() -> dict[str, Any]:
    out = {}
    for name, path in (("exp_i", EXPI_BASELINE), ("exp_j", EXPJ_BASELINE)):
        root = path.parents[3]
        data = __import__("json").loads(path.read_text(encoding="utf-8"))
        changed = [f for f, h in data["files"].items()
                   if not (root / f).exists() or sha256_file(root / f) != h]
        out[name] = {"n_files": len(data["files"]), "changed": changed,
                     "all_unchanged": not changed}
    return out


def git_history() -> dict[str, Any]:
    try:
        out = subprocess.run(["git", "log", "--format=%H%x09%s", "-40"],
                             capture_output=True, text=True, timeout=30)
        lines = [line for line in (out.stdout or "").splitlines() if line.strip()]
        commits = [{"hash": line.split("\t")[0], "message": line.split("\t", 1)[1]}
                   for line in lines if "\t" in line]
    except Exception as exc:  # pragma: no cover
        commits, _ = [], str(exc)
    return {"branch": git_branch(), "head": git_commit(),
            "exp_k_commits": [c for c in commits if c["message"].startswith("EXP-K:")]}


def snapshot_inventory() -> dict[str, Any]:
    hist = RESULTS_DIR / "history"
    files = sorted(hist.glob("K*-*.json")) if hist.exists() else []
    per_stage = {}
    for f in files:
        stage = f.name.split("-")[0]
        per_stage[stage] = per_stage.get(stage, 0) + 1
    return {"total": len(files), "per_stage": per_stage}


def rerun_key_experiments(cfg: ExpKConfig) -> dict[str, Any]:
    """Full-config reruns of two key experiments; exact equality is required for determinism."""
    stored = analysis.load_stages()
    out: dict[str, Any] = {}
    k2 = retrieval_ood.run_k2(cfg)
    old2 = stored.get("K2", {})
    out["K2"] = {
        "stored_silent_substitution": old2.get("by_arm", {}).get("baseline", {}).get(
            "silent_substitution_ood", {}).get("mean"),
        "rerun_silent_substitution": k2["by_arm"]["baseline"]["silent_substitution_ood"]["mean"],
        "stored_runs": old2.get("n_runs"), "rerun_runs": k2["n_runs"],
    }
    out["K2"]["deterministic"] = (
        out["K2"]["stored_silent_substitution"] == out["K2"]["rerun_silent_substitution"]
        and out["K2"]["stored_runs"] == out["K2"]["rerun_runs"])
    k3 = drift_detection.run_k3(cfg)
    old3 = stored.get("K3", {})
    out["K3"] = {
        "stored_miss_rate": old3.get("by_arm", {}).get("acceptance_baseline", {}).get("miss_rate"),
        "rerun_miss_rate": k3["by_arm"]["acceptance_baseline"]["miss_rate"],
        "stored_runs": old3.get("n_runs"), "rerun_runs": k3["n_runs"],
    }
    out["K3"]["deterministic"] = (
        out["K3"]["stored_miss_rate"] == out["K3"]["rerun_miss_rate"]
        and out["K3"]["stored_runs"] == out["K3"]["rerun_runs"])
    k4 = calibration.run_k4(cfg)
    old4 = stored.get("K4", {})
    out["K4"] = {
        "stored_raw_ece": old4.get("by_method", {}).get("raw", {}).get("ece"),
        "rerun_raw_ece": k4["by_method"]["raw"]["ece"],
        "deterministic": (old4.get("by_method", {}).get("raw", {}).get("ece")
                          == k4["by_method"]["raw"]["ece"]),
    }
    return out


def run_k7(cfg: ExpKConfig) -> dict[str, Any]:
    mon = Monitor()
    blocks = analysis.load_stages()
    manifests = verify_manifests()
    history = git_history()
    snaps = snapshot_inventory()
    reruns = rerun_key_experiments(cfg)
    final = analysis.final_report_data(blocks, manifests)
    out = {
        "stage": "K7", "experiment_id": "K7",
        "experiment_name": "Final Reproduction and Integrity",
        "status": "completed",
        "config": cfg.as_dict(), "config_fingerprint": reproducibility(cfg)["config_fingerprint"],
        "git": history, "protected_manifests": manifests,
        "snapshots": snaps, "reruns": reruns,
        "final_summary": final,
        "all_deterministic": all(v.get("deterministic") for v in reruns.values()),
        "protected_files_unchanged": all(v["all_unchanged"] for v in manifests.values()),
        "limitations": [
            "Reruns compare aggregate metrics, not byte-for-byte result files (timestamps differ).",
            "No natural-input or external-data validation was performed.",
        ],
    }
    out.update(mon.finish())
    return out
