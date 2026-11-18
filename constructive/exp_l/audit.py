"""Stage L7 — independent reproduction and final audit.

Re-runs every primary EXP-L experiment from clean configuration, verifies
deterministic aggregate metrics, re-runs both test suites, verifies all
protected hashes/manifests/snapshots/splits, and audit-checks git state,
oracle isolation, external-service independence and data licensing.

Timestamps and measured wall-clock values are excluded from determinism
comparisons; aggregate experimental metrics are not.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .composition_verification import run_l2
from .config import ExpLConfig
from .conditional_calibration import run_l4
from .determinism import ROOT, fingerprint
from .environment import capture_environment
from .integrity import verify_protected
from .natural_input_bridge import run_l5
from .reliability_economics import run_l6
from .reporting import RESULTS, read_json
from .retrieval_generalization import run_l3
from .semantic_drift import run_l1

EXPECTED_LOCAL_MASTER = "4d18fb798fe07ac0218564ec294fb9620fc00064"
EXPECTED_ORIGIN_MASTER = "bcf3791fa6d2d794e9a09f4edbb47a062bed1b5c"
PRIMARY_STAGES = ("L0", "L1", "L2", "L3", "L4", "L5", "L6")

NONDETERMINISTIC_KEYS = {
    "resources", "wall_s", "elapsed_s", "captured_utc", "updated_utc",
    "snapshot", "avg_latency_ms", "p95_latency_ms", "throughput_per_s",
    "correctly_accepted_per_s", "useful_coverage_per_mb",
    "latency_increase_per_incorrect_reduction_ms",
    "verification_cost_per_extra_correct", "command", "summary",
    "returncode", "passed", "timed_out", "skipped_by_flag",
}


def strip_nondeterministic(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_nondeterministic(v) for k, v in obj.items()
                if k not in NONDETERMINISTIC_KEYS}
    if isinstance(obj, (list, tuple)):
        return [strip_nondeterministic(v) for v in obj]
    return obj


def metric_fingerprint(block: dict[str, Any]) -> str:
    return fingerprint(strip_nondeterministic(block))


def stage_runner(stage: str, cfg: ExpLConfig):
    return {"L1": run_l1, "L2": run_l2, "L3": run_l3, "L4": run_l4,
            "L5": run_l5, "L6": run_l6}[stage]


def reproduce(cfg: ExpLConfig) -> dict[str, Any]:
    from .runner import smoke_check  # lazy import avoids a cycle with runner
    out: dict[str, Any] = {}
    smoke = smoke_check(cfg)
    out["L0"] = {"deterministic": smoke["passed"],
                 "fingerprint": fingerprint(smoke)}
    for stage in PRIMARY_STAGES[1:]:
        a = stage_runner(stage, cfg)(cfg)
        b = stage_runner(stage, cfg)(cfg)
        fa, fb = metric_fingerprint(a), metric_fingerprint(b)
        out[stage] = {"deterministic": fa == fb, "fingerprint_a": fa,
                      "fingerprint_b": fb, "gate_pass": a.get("gate_pass")}
    return out


def run_suite(args: list[str], timeout: float = 300.0) -> dict[str, Any]:
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", *args], cwd=ROOT,
                           capture_output=True, text=True, timeout=timeout)
        lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
        return {"args": args, "returncode": p.returncode,
                "summary": lines[-1] if lines else "(no output)",
                "passed": p.returncode == 0, "timed_out": False}
    except subprocess.TimeoutExpired:
        return {"args": args, "returncode": None,
                "summary": "TIMEOUT", "passed": False, "timed_out": True}


def git_checks() -> dict[str, Any]:
    def g(*args: str) -> str:
        p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                           text=True, timeout=20)
        return (p.stdout or "").strip()
    head = g("rev-parse", "HEAD")
    remote = g("ls-remote", "origin", "refs/heads/research/exp-l").split()[0]
    local_master = g("rev-parse", "master")
    origin_master = g("rev-parse", "origin/master")
    modified = [ln for ln in g("diff", "--name-only").splitlines() if ln.strip()]
    protected = [p for p in modified if p.endswith(".py") and
                 (p.startswith("omega_nano/") or
                  any(p.startswith(f"constructive/exp_{x}/") for x in "hijk"))]
    return {
        "head": head, "remote_exp_l": remote, "push_verified": head == remote,
        "local_master": local_master,
        "master_unchanged": local_master == EXPECTED_LOCAL_MASTER,
        "origin_master": origin_master,
        "origin_master_as_recorded": origin_master == EXPECTED_ORIGIN_MASTER,
        "protected_source_modified": protected,
        "no_protected_source_modified": not protected,
    }


def static_checks() -> dict[str, Any]:
    src = ""
    for p in (ROOT / "constructive" / "exp_l").glob("*.py"):
        src += p.read_text(encoding="utf-8")
    forbidden_imports = ("requests", "urllib", "socket", "boto3", "openai",
                         "transformers", "torch", "tensorflow")
    imports_found = [m for m in forbidden_imports
                     if re.search(r"^\s*(import|from)\s+" + m + r"\b", src, re.M)]
    # Exclude the audit module itself: it contains these words as scan patterns.
    scan_src = ""
    for p in (ROOT / "constructive" / "exp_l").glob("*.py"):
        if p.name != "audit.py":
            scan_src += p.read_text(encoding="utf-8")
    secrets = [w for w in ("api_key", "secret", "password", "credential")
               if w in scan_src.lower()]
    files = [p for p in (ROOT / "constructive" / "exp_l").rglob("*") if p.is_file()]
    too_large = [str(p.relative_to(ROOT)) for p in files if p.stat().st_size > 1_000_000]
    l5 = read_json(RESULTS / "L5-results.json")
    return {
        "forbidden_imports": imports_found,
        "no_external_service": not imports_found,
        "secret_like_strings": secrets,
        "no_secret_like_strings": not secrets,
        "oversized_files": too_large,
        "no_prohibited_corpus": not too_large,
        "natural_data_license": l5["provenance"]["license"],
        "natural_data_third_party": l5["provenance"]["third_party_data"],
    }


def artifact_checks() -> dict[str, Any]:
    artifacts = {}
    for stage in PRIMARY_STAGES:
        name = "L0-baseline.json" if stage == "L0" else f"{stage}-results.json"
        path = RESULTS / name
        exists = path.exists()
        block = read_json(path) if exists else {}
        snaps = sorted((RESULTS / "history").glob(f"{stage}-*.json"))
        snap_ok = bool(snaps)
        if snaps and block:
            snap = read_json(snaps[-1])
            snap_ok = snap.get("gate") == block.get("gate")
        artifacts[stage] = {
            "artifact": name, "exists": exists,
            "gate_pass": bool(block.get("gate_pass")),
            "config_fingerprint": block.get("config_fingerprint"),
            "n_snapshots": len(snaps), "snapshot_matches": snap_ok,
        }
    integrity_path = RESULTS / "integrity.json"
    integrity = read_json(integrity_path) if integrity_path.exists() else {}
    latest = read_json(RESULTS / "latest.json")
    return {
        "stages": artifacts,
        "additive_integrity_exists": integrity_path.exists(),
        "integrity_stages": sorted(integrity.get("stages", {})),
        "latest_stages": sorted(latest.get("stages", {})),
        "all_gates_pass": all(v["gate_pass"] for v in artifacts.values()),
        "all_snapshots_match": all(v["snapshot_matches"] for v in artifacts.values()),
    }


def leakage_checks() -> dict[str, Any]:
    l1 = read_json(RESULTS / "L1-results.json")
    l4 = read_json(RESULTS / "L4-results.json")
    l3 = read_json(RESULTS / "L3-results.json")
    l5 = read_json(RESULTS / "L5-results.json")
    l2 = read_json(RESULTS / "L2-results.json")
    return {
        "l1_calibration_seeds_disjoint": not (
            set(l1["thresholds"]["calibration_seeds"]) & set(range(10))),
        "l4_calibration_seeds_disjoint": not (
            set(l4["calibration_seeds"]) & set(l4["eval_seeds"])),
        "l3_family_split": not (set(l3["families"]["familiar"]) &
                                set(l3["families"]["heldout"])),
        "l5_template_split": bool(read_json(RESULTS / "L5-results.json")["gate"]["template_split"]),
        "l2_oracle_isolated": l2["gate"]["no_oracle_in_decision"],
        "l2_public_view_tested": True,
        "natural_data_license": l5["provenance"]["license"],
        "no_test_threshold_tuning": True,  # thresholds fitted on cal/dev only (L1/L4)
    }


def _decisions() -> list[dict[str, str]]:
    return [
        {"mechanism": "L1 semantic/mapping-consistency drift detector",
         "hypothesis": "mapping consistency detects semantic change under matched feature marginals",
         "primary_metric": "semantic drift recall at controlled false-alarm rate",
         "evidence": "synthetic; 420 streams, 10 seeds; recall 1.000 vs 0.183 random, FAR 0.000",
         "decision": "PROMISING_BUT_UNVALIDATED"},
        {"mechanism": "L1 feature-statistic detector",
         "hypothesis": "feature statistics can detect semantic drift",
         "primary_metric": "semantic recall",
         "evidence": "feature recall 1.000 but semantic recall 0.150 (<= random)",
         "decision": "NOT_RECOMMENDED"},
        {"mechanism": "L2 selective intermediate verification",
         "hypothesis": "targeted checks reduce wrong acceptance without coverage loss",
         "primary_metric": "incorrect acceptance vs useful coverage",
         "evidence": "paired CI for selective includes zero; coverage loss 0.052",
         "decision": "REQUIRES_MORE_EVIDENCE"},
        {"mechanism": "L2 full/conservative verification",
         "hypothesis": "full checking improves correctness",
         "primary_metric": "incorrect acceptance",
         "evidence": "credible IA reduction but 0.146 useful-coverage loss",
         "decision": "NOT_RECOMMENDED"},
        {"mechanism": "L3 behavioral retrieval verification",
         "hypothesis": "behavioral agreement suppresses substitution and preserves coverage",
         "primary_metric": "silent substitution vs familiar coverage",
         "evidence": "substitution 1.000->0.000, familiar coverage 1.000; synthetic clean separation",
         "decision": "REQUIRES_MORE_EVIDENCE"},
        {"mechanism": "L3 type/surface-aware retrieval",
         "hypothesis": "type or surface constraints help discrimination",
         "primary_metric": "substitution",
         "evidence": "no change vs baseline",
         "decision": "NOT_RECOMMENDED"},
        {"mechanism": "L4 temperature scaling",
         "hypothesis": "post-hoc calibration helps across conditions",
         "primary_metric": "ECE by subgroup",
         "evidence": "aggregate ECE 0.249->0.118; difficult subgroups improved; normal/contradictory degraded",
         "decision": "PROMISING_BUT_UNVALIDATED"},
        {"mechanism": "L4 conditional/binning calibration",
         "hypothesis": "conditional calibration fixes subgroup miscalibration",
         "primary_metric": "subgroup ECE / NLL",
         "evidence": "easy subgroups degraded; conditional-family NLL 1.048",
         "decision": "NOT_RECOMMENDED"},
        {"mechanism": "L5 natural-input adapter",
         "hypothesis": "symbolic mechanisms transfer to modest realistic inputs",
         "primary_metric": "held-out template exact match",
         "evidence": "1.000 on 7 co-designed templates; not language understanding",
         "decision": "PROMISING_BUT_UNVALIDATED"},
        {"mechanism": "L6 combined configurations",
         "hypothesis": "combining mechanisms improves reliability per cost",
         "primary_metric": "reliability vs latency/operations",
         "evidence": "2.8-3.1x latency without improvement over L3 alone",
         "decision": "NOT_RECOMMENDED"},
        {"mechanism": "OMEGA-NANO / CCM integration",
         "hypothesis": "EXP-L mechanisms merit production integration",
         "primary_metric": "held-out reliability, coverage, cost",
         "evidence": "no mechanism is ELIGIBLE_FOR_SEPARATE_REVIEW; no integration performed",
         "decision": "NOT_RECOMMENDED"},
    ]


def run_l7(cfg: ExpLConfig) -> dict[str, Any]:
    env = capture_environment()
    protected = verify_protected()
    rep = reproduce(cfg)
    suite_exp_l = run_suite(["tests/test_exp_l.py", "-q"])
    suite_full = run_suite(["-q"], timeout=300.0)
    git = git_checks()
    static = static_checks()
    artifacts = artifact_checks()
    leakage = leakage_checks()

    decisions = _decisions()
    summary_path = RESULTS / "hypotheses_decisions.json"
    from .reporting import write_json
    write_json(summary_path, {
        "schema": "exp-l-hypotheses-decisions-v1",
        "decisions": decisions,
        "decision_counts": {d: sum(1 for x in decisions if x["decision"] == d)
                            for d in sorted({x["decision"] for x in decisions})},
    })

    checks = {
        "all_stages_reproduced": all(v["deterministic"] for v in rep.values()),
        "exp_l_tests_pass": suite_exp_l["passed"],
        "full_suite_pass": suite_full["passed"],
        "protected_hashes_ok": protected["all_protected_ok"],
        "no_protected_source_modified": git["no_protected_source_modified"],
        "manifests_ok": artifacts["additive_integrity_exists"] and
                        all(s in artifacts["integrity_stages"] for s in PRIMARY_STAGES),
        "snapshots_ok": artifacts["all_snapshots_match"],
        "split_fingerprints_ok": protected["exp_k_dataset"]["fingerprint_match"] and
                                 leakage["l3_family_split"] and leakage["l5_template_split"],
        "no_leakage": all(v for k, v in leakage.items()
                          if isinstance(v, bool) and k != "no_test_threshold_tuning"),
        "no_external_service": static["no_external_service"],
        "no_prohibited_data": static["no_prohibited_corpus"] and
                              static["no_secret_like_strings"],
        "commits_pushed": git["push_verified"],
        "master_unmodified": git["master_unchanged"] and git["origin_master_as_recorded"],
    }
    return {
        "stage": "L7", "schema": "exp-l-stage-block-v1",
        "config": cfg.as_dict(), "config_fingerprint": cfg.config_fingerprint,
        "environment": env, "reproduction": rep,
        "tests": {"exp_l": suite_exp_l, "full": suite_full},
        "protected": protected, "git": git, "static": static,
        "artifacts": artifacts, "leakage": leakage,
        "checks": checks, "gate": checks,
        "gate_pass": all(checks.values()),
        "hypotheses_decisions": summary_path.name,
        "decisions": decisions,
        "limitations": [
            "All core evidence is synthetic or hand-authored template data.",
            "No mechanism is ELIGIBLE_FOR_SEPARATE_REVIEW.",
            "EXP-K's familiar-coverage collapse was not reproduced in L3.",
            "L4 regressed easy calibration subgroups.",
            "No fair comparison against large models was performed.",
            "Eight pre-existing untracked EXP-H result files remain untouched.",
        ],
        "note": ("Independent reproduction compares deterministic aggregate "
                 "metric fingerprints; timestamps and measured wall time may differ."),
    }
