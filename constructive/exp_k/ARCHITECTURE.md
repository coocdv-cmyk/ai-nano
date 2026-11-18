# OMEGA EXP-K — Architecture

> Isolated research package. No integration into CCM or `omega_nano`.
> EXP-I and EXP-J are read-only systems under test. All generators, safeguards,
> diagnostics, evaluation and persistence are EXP-K originals.

## 1. Design contract

EXP-K isolates the four unresolved reliability failures reported by EXP-J:
deep-composition incorrect acceptance, OOD retrieval silent substitution, drift
detection failures and confidence-calibration failure. Every treatment is
compared with a **matched baseline** built from the same constructors, seeds,
splits and configuration. Test labels are used only by the evaluation side,
after a decision has been made.

## 2. Module map

```
config.py           policy, hardware probe, git/source fingerprints, manifests
types.py            stage/outcome enums, Episode record
metrics.py          mean/median/std/CI, ECE, Brier, NLL, accuracy/coverage
resources.py        Monitor + read-only reuse of EXP-I ResourceGovernor
generators.py       16 NEW rule families with independent reference labels
distributions.py    explicit shift matrix
datasets.py         split-integrity checks, dataset manifests, seed registry
constructors.py     shared deep-chain / drift / retrieval-pool builders
engine.py           adapter that drives EXP-I validation read-only
baseline.py         K0 matched baseline reproduction of all four failures
deep_composition.py K1 tracing and matched safeguard arms
retrieval_ood.py    K2 OOD/silent-substitution arms
drift_detection.py  K3 detector arms with calibration-fixed thresholds
calibration.py      K4 post-hoc calibration methods on a separate split
interactions.py     K5 factorial 2x2 interaction analysis
adversarial.py      K6 twelve-attack adversarial campaign + abstain control
analysis.py         K7 aggregation, resource accounting, integration decisions
finalize.py         K7 independent reruns, manifests, git/snapshot verification
smoke.py            stage-A invariants
runner.py           stage dispatch, snapshots, integrity, CLI
```

Dependency order: `config/types/metrics/resources` → `generators/distributions` →
`datasets/constructors` → `engine` → stage modules → `runner`.

## 3. Independence

No EXP-H/EXP-I/EXP-J generator or label table is imported. EXP-K's 16 rule
families are disjoint by name from EXP-I's and EXP-J's rules. EXP-I's typed DSL,
search, retrieval and validation modules are used only as the mechanisms under
test; their source hashes are recorded in every result block.

## 4. Persistence and integrity

- `results/experiments/K*.json` — current stage blocks.
- `results/history/K*-<stamp>.json` — immutable snapshots.
- `results/manifests/` — seed registry, baseline dataset manifest, SHA-256 records.
- `results/latest.json` — merged view.
- `results/integrity.json` — re-verification of the EXP-I (22) and EXP-J (84)
  protected-file manifests, total 106 files.

A timeout is recorded as `TIMED_OUT`; an exception as `ERROR`; neither can be
merged as `COMPLETED`. No stage commits or pushes automatically — the runner is
invoked by the maintainer, who verifies and publishes each stage separately.
