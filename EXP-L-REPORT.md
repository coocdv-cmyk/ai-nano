# EXP-L Research Report

**Status: L0 and L1 complete. Stages L2–L7 pending.**

Branch: `research/exp-l` — created from the verified EXP-K tip
`1face37bc9c695f5804baf97e6533113695f7af3` (not from `master`; `master` untouched).

## Scope and claims

EXP-L is an isolated research campaign. It is **not** an integration into
OMEGA-NANO or CCM, and no production behavior was changed. All earlier
experiments (EXP-H…EXP-K) and all protected source files were inspected
read-only. This report makes no claim of general intelligence, natural-language
understanding, state-of-the-art performance, or superiority over large models.
Every conclusion is proportional to the evidence actually collected.

## L0 — Baseline, integrity and reproducibility (verified)

| Item | Result |
|---|---|
| Git base commit | `1face37bc9c695f5804baf97e6533113695f7af3` (equals reported EXP-K tip and local `HEAD`) |
| Config fingerprint | `cea8b4c3633520d7` |
| Python / OS | 3.13.15 / Windows-11-10.0.26200-SP0, AMD64 |
| Hardware | Intel64 Family 6 Model 140, 8 logical cores, 16.82 GB total RAM (3.69 GB available at capture) |
| Optional deps | numpy 2.5.3, scipy 1.18.1, pytest 9.2.0, psutil 7.2.2 — all installed; none required by core experiments |
| Protected hashes | EXP-I 22/22 and EXP-J 84/84 SHA-256 unchanged; EXP-K integrity references re-verified |
| EXP-K dataset manifest | fingerprint `e38c3562b722c40b` recomputed exactly; 160 datasets; all splits disjoint |
| Full test suite | 312 passed, 1 skipped in 96.43 s (return code 0) |
| EXP-L tests | 12 passed |
| Smoke fixture | deterministic across independent generators and processes; fingerprint `c105dfffc3b44951` |
| L0 gate | **PASS** (`protected_hashes_ok`, `no_tracked_source_modifications`, `full_tests_ok`, `smoke_ok`) |

### Artifacts

- `constructive/exp_l/results/L0-baseline.json`
- `constructive/exp_l/results/L0-integrity.json`
- `constructive/exp_l/results/L0-environment.json`
- `constructive/exp_l/results/history/L0-20261118-131300.json` (immutable successful snapshot)
- `constructive/exp_l/results/history/L0-20261118-130812.json` (preserved diagnostic snapshot from an earlier failing gate; **not** evidence)
- `constructive/exp_l/results/integrity.json` (new additive manifest; never replaces old manifests)
- `constructive/exp_l/results/latest.json`

### Known and disclosed issues

1. Eight untracked EXP-H result files existed before EXP-L began
   (`stage-a-20261118-160909.json` … `164722.json`). Per explicit authorization
   they were **left untouched** and were not staged.
2. The legacy test suite regenerates its own quick-run JSON result files when
   executed. These side effects were restored to `HEAD` after each milestone
   run and were not staged. Protected *source* hashes were never affected.
3. One early L0 run failed its gate (a test-isolation defect in EXP-L's own
   harness). The defect was fixed with a regression test; the failed snapshot
   is preserved rather than deleted.

## Research question (carried forward)

Can useful reasoning coverage be improved while reducing incorrect acceptance,
detecting semantic changes, and demonstrating measurable transfer to realistic
inputs — and at what computational price? EXP-K's five weaknesses are treated
as hypotheses, not assumptions: semantic drift, composition safeguard coverage
cost, retrieval substitution/coverage cost, subgroup calibration failure, and
a purely synthetic evidence base.

## L1 — Semantic drift detection (gate PASS)

420 streams × 6 detectors = 2,520 evaluations, 10 seeds. Semantic target rules
were held out of the six-rule reference library; thresholds were fitted only on
calibration seeds 101–103. Wall 0.122 s, CPU 0.125 s, peak RSS 25.19 MB.

| Detector | Semantic recall [95% CI] | Feature recall | Stable FAR | AUROC |
|---|---|---|---|---|
| acceptance | 1.000 [1.000, 1.000] | 0.000 | 0.000 | 0.917 |
| feature-statistic (independently reimplemented) | 0.150 [0.121, 0.179] | 1.000 | 0.133 | 0.642 |
| semantic consistency (2-window) | 1.000 [1.000, 1.000] | 0.000 | 0.000 | 0.917 |
| change-point (CUSUM) | 1.000 [1.000, 1.000] | 0.000 | 0.000 | 0.917 |
| combined | 1.000 [1.000, 1.000] | 1.000 | 0.133 | 0.737 |
| random (no-information) | 0.183 [0.117, 0.250] | 0.250 | 0.183 | 0.475 |

Semantic recall is reported **separately** for abrupt, gradual,
context-dependent and camouflaged change (all 1.000 for the mapping-based
detectors, median detection delay 0–1 windows). The feature-statistic detector
detects feature drift perfectly but *does not* detect semantic drift
(0.150, at or below the random baseline) — it is **not** credited with semantic
detection. The camouflage condition reused the exact base input multiset, so
feature marginals were identical across stable and semantic-change windows.

**Limits.** This is a synthetic rule-switching benchmark. The reference library
identifies one of six simple arithmetic rules from base data; semantic "change"
is a switch to a held-out rule, which yields a large, clean consistency drop.
A real concept shift may be subtler, and parsing/natural inputs are out of
scope here. The feature detector's stable FAR (0.133) exceeds the 0.10 target
on held-out seeds — it is a useful feature-drift alarm but not a low-false-alarm
semantic detector.

Decision: `PROMISING_BUT_UNVALIDATED` (mapping-consistency detection);
`NOT_RECOMMENDED` (feature-only detector as a semantic detector).

Artifact: `constructive/exp_l/results/L1-results.json`.

## Decision status

No mechanism is `ELIGIBLE_FOR_SEPARATE_REVIEW` yet. L1 mapping-consistency
detection is promising but synthetic-only and unvalidated on realistic inputs.
