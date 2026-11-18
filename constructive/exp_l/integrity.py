"""Read-only integrity verification of protected systems and prior artifacts.

EXP-L never alters an old manifest.  This module only reads, recomputes, and
reports.  A new additive EXP-L manifest is written by ``reporting.py``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .determinism import ROOT, fingerprint, full_hash, sha256_file, stable_json

EXPI_MANIFEST = "constructive/exp_i/results/baseline_hashes.json"
EXPJ_MANIFEST = "constructive/exp_j/results/baseline_hashes.json"
EXPK_INTEGRITY = "constructive/exp_k/results/integrity.json"
EXPK_DATASET_MANIFEST = "constructive/exp_k/results/manifests/k0_dataset_manifest.json"

# Read-only systems under test, hashed but never modified.
SOURCE_SYSTEMS = [
    "constructive/core.py",
    "constructive/model.py",
    "constructive/worlds.py",
    "constructive/baselines.py",
    "constructive/bench.py",
    "omega_nano/engine.py",
    "omega_nano/model.py",
    "omega_nano/reasoner.py",
    "omega_nano/core/engine.py",
]


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                             text=True, timeout=20)
        return (out.stdout or "").strip()
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def git_info() -> dict[str, str]:
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
    }


def verify_hash_manifest(rel_path: str) -> dict[str, Any]:
    """Recompute every SHA-256 in a recorded ``files`` manifest."""
    path = ROOT / rel_path
    data = json.loads(path.read_text(encoding="utf-8"))
    files: dict[str, str] = data.get("files", {})
    changed: list[dict[str, str]] = []
    missing: list[str] = []
    for rel, recorded in files.items():
        fp = ROOT / rel
        if not fp.exists():
            missing.append(rel)
            continue
        actual = sha256_file(fp)
        if actual.lower() != str(recorded).lower():
            changed.append({"path": rel, "recorded": recorded, "actual": actual})
    return {
        "manifest": rel_path,
        "algorithm": data.get("algorithm", "SHA-256"),
        "n_files": len(files),
        "changed": changed,
        "missing": missing,
        "all_unchanged": not changed and not missing,
    }


def verify_expk_dataset_manifest() -> dict[str, Any]:
    """Verify the recorded K0 dataset fingerprint and split integrity.

    The fingerprint is ``fingerprint_of({seeds, rules, shifts, config})`` where
    rules/shifts are taken in first-appearance order in the manifest rows.
    """
    path = ROOT / EXPK_DATASET_MANIFEST
    m = json.loads(path.read_text(encoding="utf-8"))
    rows = m["rows"]
    rules = list(dict.fromkeys(r["rule"] for r in rows))
    shifts = list(dict.fromkeys(r["shift"] for r in rows))
    recomputed = fingerprint({
        "seeds": m["config"]["seeds"], "rules": rules,
        "shifts": shifts, "config": m["config"],
    })
    recorded = m["fingerprint"]
    clean = bool(m.get("all_splits_disjoint")) and not m.get("dirty")
    return {
        "manifest": EXPK_DATASET_MANIFEST,
        "recorded_fingerprint": recorded,
        "recomputed_fingerprint": recomputed,
        "fingerprint_match": recorded == recomputed,
        "n_datasets": len(rows),
        "all_splits_disjoint": clean,
        "dirty": m.get("dirty", []),
        "all_ok": recorded == recomputed and clean,
    }


def verify_expk_integrity() -> dict[str, Any]:
    """Re-verify the manifests that EXP-K's own integrity.json attests to."""
    path = ROOT / EXPK_INTEGRITY
    ik = json.loads(path.read_text(encoding="utf-8"))
    checks = []
    for name, entry in ik.get("manifests", {}).items():
        rel = str(entry["path"]).replace("\\", "/")
        checks.append({"name": name, **verify_hash_manifest(rel)})
    return {
        "manifest": EXPK_INTEGRITY,
        "recorded_git_head": ik.get("git_head"),
        "recorded_git_branch": ik.get("git_branch"),
        "checks": checks,
        "all_ok": all(c["all_unchanged"] for c in checks),
    }


def source_hashes() -> dict[str, str]:
    return {rel: sha256_file(ROOT / rel) for rel in SOURCE_SYSTEMS
            if (ROOT / rel).exists()}


def verify_protected() -> dict[str, Any]:
    gi = git_info()
    expi = verify_hash_manifest(EXPI_MANIFEST)
    expj = verify_hash_manifest(EXPJ_MANIFEST)
    expk_int = verify_expk_integrity()
    expk_ds = verify_expk_dataset_manifest()
    ok = (expi["all_unchanged"] and expj["all_unchanged"]
          and expk_int["all_ok"] and expk_ds["all_ok"])
    return {
        "git": gi,
        "exp_i": expi,
        "exp_j": expj,
        "exp_k_integrity": expk_int,
        "exp_k_dataset": expk_ds,
        "source_hashes": source_hashes(),
        "source_hash_fingerprint": fingerprint(source_hashes()),
        "all_protected_ok": ok,
    }


def unchanged_snapshot(paths: list[str]) -> dict[str, Any]:
    """Capture current hashes of arbitrary read-only paths for later diffing."""
    snap = {}
    for rel in paths:
        fp = ROOT / rel
        if fp.exists():
            snap[rel] = sha256_file(fp)
    return {"algorithm": "SHA-256", "files": snap,
            "fingerprint": fingerprint(snap)}
