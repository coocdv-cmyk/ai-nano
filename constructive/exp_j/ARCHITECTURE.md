# OMEGA EXP-J — Architecture

> Isolated research package. No integration into CCM, `omega_nano/`, EXP-H or
> EXP-I. EXP-I is the *system under test* and is imported read-only; EXP-J
> supplies all generators, noise models, drift models, cost accounting and
> resource governance.

## 1. Design contract

EXP-J answers one question: **do EXP-I's improvements generalize beyond the
synthetic distributions on which they were developed?** To make that test
honest, four properties are enforced:

1. **Independent data.** Generators, rule families, RNGs and reference label
   functions are written from scratch. No EXP-H/EXP-I generator or label table
   is imported.
2. **Separated information.** Train / validation / test are disjoint by
   construction. Test labels are used only by the evaluation side *after* a
   decision is made — never for candidate selection, support fitting or
   calibration.
3. **Explicit uncertainty.** Unknown, contradictory, insufficient,
   non-identifiable and resource-limited remain distinct outcomes.
4. **Honest accounting.** Every experiment carries a completion state, resource
   measurements, source fingerprints and a raw episode log. Timeouts are never
   reported as successes.

## 2. Module map and dependency order

```
config.py                policy, hardware probe, STAGE-0 protected manifest
types.py                 enums (NoiseKind, DriftKind, Outcome, Arm, CompletionState),
                         ShiftSpec, EpisodeRecord
generators.py            12 NEW rule families, independent reference labels,
                         candidate programs, disjoint splits, shift domains
distributions.py         explicit train/test shift descriptors and matrix
noise.py                 random, structured, contradiction, missing, duplicate,
                         misleading-support noise (test labels untouched)
drift.py                 gradual / abrupt drift streams with base/drift/recover phases
metrics.py               mean/median/std/CI, accuracy-coverage, ECE (split fit),
                         retrieval P/R, confusion, comparison, detection delay
resources.py             Monitor + reuse of the audited EXP-I ResourceGovernor
engine.py                adapter that drives EXP-I validation with EXP-J data
transfer.py              J1 — independent distribution transfer + baselines
robustness.py            J2 — noise/corruption + drift
deep_composition.py      J3 — depths 2..16, six-way outcome classification,
                         plain/factorized/cegis/cached strategies
abstraction_economics.py J4 — five arms, explicit lifecycle cost model
interactions.py          J5 — six controlled mechanism combinations
analysis.py              taxonomy, efficiency (J6), independence, readiness
smoke.py                 stage-A invariants
runner.py                orchestration, snapshots, integrity, latest.json, CLI
```

Dependency order is strictly bottom-up: `types/config` → `generators` →
`noise/drift/distributions` → `engine/metrics/resources` → experiment modules →
`analysis/runner`. Each layer is independently testable.

## 3. Independent generators (`generators.py`)

Twelve rule families, none present in EXP-H or EXP-I:

| Family | Type | Novel operator combination |
|---|---|---|
| `affine_mod` | int→int | `mul → add → mod` |
| `digit_sum_parity` | int→bool | `sum_digits → even` |
| `threshold_parity` | int→bool | `gt → and → odd` |
| `prime_and_odd` | int→bool | `is_prime → not → and` |
| `fib_mod` | int→int | `fib → mod` |
| `square_digit_sum` | int→int | `square → sum_digits` |
| `range_of_evens` | list→int | `evens → range_list` |
| `sum_top_k` | list→int | `sort_desc → take → sum_list` |
| `count_distinct_gt` | list→int | `unique → count_gt` |
| `prefix_sum_gt` | list→bool | `take → sum_list → gt` |
| `evens_of_even` | list→int | `evens → count_even` |
| `weighted_pair` | pair→int | `first → mul → add` |

Each family has one semantically correct program and two confusable
distractors. Reference labels are independent pure-Python functions; the
generator asserts the correct program agrees with them on every sampled point.
Shifts cover magnitude, sign, width, list length and combined shifts.

## 4. Evidence adapter (`engine.py`)

`run_validation` builds an EXP-I `ValidationEngine` with EXP-J support /
validation data and unlabelled probes drawn outside all labelled splits. It
returns the raw EXP-I `ValidationResult` plus derived fields. Test accuracy of
an accepted candidate is computed only *after* the decision, by the evaluation
side.

## 5. Experiments

- **J1 (`transfer.py`)** — 12 rules × 4 shifts × 10 seeds = 480 episodes.
  Baselines: support-only fit (always accept) and the evidence engine.
  Retrieval is tested with a concept store built from independent source tasks.
- **J2 (`robustness.py`)** — 7 noise kinds × 6 levels × 10 seeds × 5 rules,
  plus gradual/abrupt drift streams. Measures accuracy–coverage, calibration,
  evidence quality, detection delay and recovery.
- **J3 (`deep_composition.py`)** — chains at depths 2,3,4,5,8,12,16 × 10 seeds
  × 4 strategies. Six-way outcome classification: correct discovered, incorrect
  accepted, correct not discovered, non-identifiable, representation limitation,
  budget exhaustion.
- **J4 (`abstraction_economics.py`)** — arms A–E on shared-core and unrelated
  streams. Explicit cost model: discovery + storage + retrieval + instantiation
  − search savings. Coefficients are disclosed constants.
- **J5 (`interactions.py`)** — six mechanism pairs, each with enabled/disabled
  arms; regressions are recorded explicitly.
- **J6 (`analysis.py`)** — aggregate wall time, peak RSS, expansions, nodes,
  cache hits, verification calls, memory per concept and depth scaling.

## 6. Persistence and integrity

- `results/latest.json` — merged campaign result.
- `results/experiments/J*.json` — current per-experiment blocks.
- `results/experiment-history/J*-<stamp>.json` — immutable snapshots.
- `results/baseline_hashes.json` — SHA-256 of 84 protected files recorded at
  STAGE 0 (core, all of `omega_nano`, EXP-H, EXP-I and pre-existing tests).
- `results/integrity.json` — close-out verification against the manifest.

The runner recomputes J6 at merge time rather than trusting a mutable stored
block. A `ResourceLimit` maps to `TIMED_OUT`; a caught exception maps to
`ERROR`; neither can be merged as `COMPLETED`.

## 7. Reproducibility

All generators are seeded with `random.Random(seed * constant + …)`; no
`hash()` of strings is used in data generation. `python -m
constructive.exp_j.runner --no-resume` reproduces the full campaign. Config,
source hashes (including all EXP-I files), hardware and dataset fingerprints are
written into every block.
