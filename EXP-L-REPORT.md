# EXP-L Research Report

**Status: L0 complete (baseline, integrity, reproducibility). Stages L1–L7 pending.**

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

## Decision status

No mechanism has been evaluated or approved for integration at this stage.
Nothing is `ELIGIBLE_FOR_SEPARATE_REVIEW` yet.
