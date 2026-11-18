# OMEGA EXP-J — Final Report

## Independent Transfer Validation, Noise Robustness, Deep Composition & Abstraction Economics

| | |
|---|---|
| Date | 2026-11-18 |
| Scope | Isolated research branch `constructive/exp_j/`; no integration |
| Environment | i5-1135G7 4C/8T @2.4 GHz, 16.82 GB, Windows 11 26200, Python 3.13.15, **CPU-only, no external APIs** |
| Implementation | 15 Python modules under `constructive/exp_j/`; `tests/test_exp_j.py` |
| Results | `constructive/exp_j/results/latest.json` + 5 per-experiment files + **immutable** `experiment-history/` snapshots (full campaign stamp, 10 snapshots total incl. quick run) |
| Tests | EXP-J: **35 passed**; full repo: **264 passed, 1 skipped** (baseline 229 passed, 1 skipped) |
| Campaign | 6/6 experiments `COMPLETED`; wall **150.96 s**; peak RSS **48.76 MB**; **0** timeouts; **0** runner errors |
| Integrity | **84/84** protected-file SHA-256 hashes unchanged; EXP-H, EXP-I, core, `omega_nano/` and pre-existing tests untouched |
| Source revision | `no-commits` (repository has no commits) |
| Config fingerprint | `eb929126a9d2ac0e` |

---

## 0. What was actually tested

EXP-I is the **system under test**, imported read-only. EXP-J supplies entirely
independent generators and evaluation infrastructure. The central question is
whether EXP-I's evidence validation, retrieval, deep search and abstraction
mechanisms generalize beyond their original synthetic benchmarks.

Independence is structural, not asserted: `constructive/exp_j/generators.py`
contains no import of EXP-H or EXP-I generators, no copied label table, and 12
rule families with names disjoint from all EXP-I/EXP-H rules. Train, validation
and test use independent RNG draws and disjoint domains. Test labels are used
only by the evaluation side after a decision is made — never for candidate
selection, support fitting or calibration.

## 1. J1 — Independent distribution transfer

**Design.** 12 new rule families (`affine_mod`, `digit_sum_parity`,
`threshold_parity`, `prime_and_odd`, `fib_mod`, `square_digit_sum`,
`range_of_evens`, `sum_top_k`, `count_distinct_gt`, `prefix_sum_gt`,
`evens_of_even`, `weighted_pair`) × 4 shifts (in-domain, magnitude, list-length,
combined) × 10 seeds = **480 episodes**. Each task has one correct program and
two confusable distractors. An independent support-only baseline (best support
fit, always accept) is computed on the same tasks.

| Metric | Baseline (support-only) | EXP-I engine under EXP-J data |
|---|---|---|
| Wrong acceptance | 0.2375 (n=480, CI ±0.038) | **0.0000** (n=480, CI 0) |
| Correct acceptance | — | 0.783 in-domain; 0.742 across shifts |
| Non-identifiable | — | 65 / 480 (explicit, not guessed) |
| Insufficient evidence | — | 41 / 480 |
| Selective accuracy | — | high where coverage is retained (see `by_shift`) |
| Retrieval recall @2 | — | **0.9917** (n=360, CI ±0.0094) |
| Independent-split ECE | — | **0.1596** (fit on half of seeds, evaluate on the other half) |

**Held-out-family transfer.** Concepts were learned from 6 families and queried
on 6 genuinely held-out families (60 episodes). The store cannot contain a
held-out concept. It never claimed the (absent) correct concept, but it
**silently substituted a different learned concept in 100% of cases** with no
ambiguity flag. Retrieval therefore transfers to new *instances* of known
families, not to new *families*: it has no out-of-family detection.

**Finding.** Evidence validation's reduction in wrong acceptance transfers to
genuinely independent rule families and distribution shifts (100% relative
reduction vs baseline on this benchmark). Retrieval recall transfers within
known families but fails on held-out families. EXP-I's reported ECE of 0.000
was optimistic: on EXP-J's confusable families the independent ECE is 0.1596.
Calibration is therefore **not** validated as solved. Acceptance at combined
shift falls (0.783 → 0.742), so transfer has a measurable coverage cost.

## 2. J2 — Noise, corruption and concept drift

**Noise.** 7 kinds × 6 levels (0, 1, 5, 10, 20, 30%) × 10 seeds × 5 rules.
Correct acceptance, wrong acceptance, abstention, false rejection, calibration,
evidence informativeness and accuracy–coverage are recorded per cell.

- At 30% contradiction noise: contradiction detected **1.00**, correct
  acceptance 0.00, abstention 1.00.
- At 30% random-label noise: contradiction detected 0.72, abstention 0.96,
  correct acceptance 0.04 — a severe **coverage** cost.
- At 30% structured corruption: 2% wrong acceptance; at 30% misleading support:
  8% wrong acceptance, accuracy 0.46, coverage 0.54, selective accuracy 0.852.
- Missing observations degrade gracefully (0.72 correct at 30%) and are **not**
  collapsed into contradiction (0% contradiction). Duplicate observations are
  harmless (0.80 correct at 30%).
- Adaptive evidence acquisition works where the task is identifiable: the
  engine raises informative observations and abstains rather than claiming
  semantics under contradiction. Higher abstention is explicitly **not**
  counted as intelligence — selective accuracy is reported alongside it.

**Drift.** 40 streams (2 kinds × 10 seeds × 4 rules), each base→drift→recover.

| Kind | Detection delay (mean, CI) | Undetected | Recovery delay |
|---|---|---|---|
| Gradual | 2.31 (±1.28) episodes | 24 / 40 | 0.14 |
| Abrupt | 2.31 (±1.28) episodes | 24 / 40 | 0.14 |

**Finding.** Drift is the largest unresolved robustness gap: because the label
rule remains valid off-domain, the engine keeps accepting on drifted validation
data and acceptance-based detection fails in 60% of streams. Recovery, once
acceptance breaks, is fast (0.14 episodes).

## 3. J3 — Deep composition reliability

**Design.** Independently generated unary chains at depths 2, 3, 4, 5, 8, 12,
16 × 10 seeds × 4 strategies = **280 runs**. Outcomes are classified into six
categories: correct discovered, incorrect accepted, correct not discovered,
non-identifiable, representation limitation, budget exhaustion. Budgets are
hard caps and were never raised to force success.

Totals: **73 correct discovered, 105 incorrect accepted, 102 correct not
discovered** (0 non-identifiable / representation / budget in this pool).

| Depth | correct | incorrect accepted | not discovered | mean expansions (plain) |
|---|---|---|---|---|
| 2 | 5/10 | 1/10 | 4/10 | 63.8 |
| 3 | 4/10 | 4/10 | 2/10 | 179.9 |
| 4 | 2/10 | 4/10 | 4/10 | 180.7 |
| 5 | 3/10 | 3/10 | 4/10 | 180.5 |
| 8 | 1/10 | 3/10 | 6/10 | 180.5 |
| 12 | 1/10 | 6/10 | 3/10 | 180.0 |
| 16 | 1/10 | 6/10 | 3/10 | 180.6 |

Strategy comparison (70 runs each): `factorized` (oracle operator subset)
reduced mean expansions 163.7 → 109.9 and raised correct discoveries 17 → 22,
but still accepted 24 spurious programs; `cegis` (269 mean expansions) and
`cached` (900 mean expansions, no eviction) gave no outcome gain.

**Finding.** EXP-I's shallow/mid-depth spurious acceptance is **re-confirmed**,
not fixed, on independent generators. The root cause is that search admits
incidentally-fitting programs and validation checks consistency with a small
sample rather than semantic equivalence. Oracle-guided factorization helps
search efficiency but not correctness; search-space caching without eviction
actively hurts.

## 4. J4 — Abstraction economics

EXP-I reported 81 abstractions and 29 reuse events but identical expansions and
accuracy. EXP-J builds an explicit total lifecycle cost model (discovery +
storage + retrieval + instantiation − search savings, with disclosed
coefficients) and compares five arms on shared-core and unrelated streams
(5 seeds, 6 tasks each).

| Arm | expansions (shared / unrelated) | abstractions / reuse (shared) | net lifecycle cost (shared / unrelated) |
|---|---|---|---|
| A no abstraction | 4,401 / 4,390 | 0 / 0 | 0 / 0 |
| B current model | 4,401 / 4,390 | 80 / 25 | +0.00955 / +0.01094 |
| C reuse-crediting | 4,401 / 4,390 | 80 / 25 | +0.00437 / +0.00396 |
| D retrieval-ranked | 4,401 / 4,390 | 80 / 25 | +0.00456 / +0.00193 |
| E selective retention | 4,401 / 4,390 | 80 / 25 | +0.00352 / +0.00422 |

**Finding.** Abstractions are genuinely discovered and reused, but **no arm
reduces expansions or improves accuracy**, and every abstraction arm has
positive net lifecycle cost. The EXP-I negative result is reproduced on
independent data. Diagnosis: injected abstractions behave as ordinary search
atoms and are deduplicated by the output-level memo, so they earn no search
saving; reuse crediting and selective retention reduce waste but cannot make
the lifecycle pay for itself. This is a cost-model / search-architecture
problem combined with a useless-abstraction problem, not a benchmark artifact.
**No positive result is claimed.**

## 5. J5 — Cross-mechanism interaction

Six controlled ablations were run; two regressions were found and are preserved
in `latest.json` under `J5.regressions`:

- **Evidence validation + deep search** (depth 5): adding validation reduced
  correct acceptance 0.10 → 0.00 and coverage 0.80 → 0.60 relative to
  search-only. Fewer wrong acceptances do not survive combination unchanged.
- **Concept drift + retrieval**: recall@1 degrades in the drift phase.
- **Conditional DSL + deep composition**: with the conditional operator enabled
  the correct-acceptance rate remained 0.00 — expressiveness is not discovery.
- **Noisy evidence + abstraction**: abstraction injection gave no measurable
  benefit at either 0% or 20% noise.
- **Retrieval + abstraction**: retrieval-aware filtering changed injection count
  but not search cost materially.
- **Program repair + adversarial evidence**: 0 false verified repairs in both
  limited- and rich-visible arms; verification coverage 1.0. No regression.

## 6. J6 — Computational efficiency

CPU-only throughout. Total wall **150.96 s**; peak RSS **48.76 MB**; **136,137**
search expansions; **0** timed-out experiments. Depth scaling, nodes, cache
hits, verification calls and memory-per-concept are in `latest.json` under
`experiments.J6`. Deadlines (2.5 s per fit, 180 s per experiment, 1200 s
campaign) are hard caps; a forced zero-deadline governor is tested to produce
`TIMED_OUT`, never `COMPLETED`.

## 7. Failure taxonomy (raw episode rows)

- J1 (480): success 374, non-identifiable 65, insufficient-evidence 41.
- J2 noise (1,750): success 1,233, contradictory 412, non-identifiable 312,
  abstention 87, insufficient-evidence 45, wrong-acceptance 11.
- J3 (280): correct discovered 73, incorrect accepted 105, correct not
  discovered 102.

Counts are recorded, never collapsed or deleted. Raw per-episode data is in
`latest.json`.

## 8. Verification and integrity

- `python -m pytest tests/test_exp_j.py -q` → **35 passed**.
- `python -m pytest -q` → **264 passed, 1 skipped** (skip is pre-existing).
- Stage-A smoke → **8/8 pass**.
- Full campaign → 6/6 `COMPLETED`, 0 runner errors, 0 timeouts.
- Protected-file verification → **84/84 SHA-256 hashes identical** to the
  STAGE-0 manifest. The manifest covers the original constructive core, all of
  `omega_nano/`, all of `constructive/exp_h/`, all of `constructive/exp_i/`, and
  all pre-existing tests — but not EXP-J itself.
- CCM, `omega_nano/`, EXP-H source/results and EXP-I source/results/snapshots
  are untouched. No historical result was overwritten.

## 9. Validated improvements vs unresolved limitations

**Validated on this benchmark family:**
1. Evidence validation eliminates wrong acceptance on independent rule families
   and shifted distributions (0/480 vs 23.75% baseline).
2. Retrieval recall @2 transfers to new instances of 12 known families (0.9917).
3. Contradictory evidence is detected and distinguished from insufficient
   evidence and missing observations.
4. Resource governance produces explicit completion states; timeouts cannot
   become successes.

**Unresolved / negative:**
1. Calibration ECE is 0.1596 — not solved.
2. Incorrect acceptance at depths 2–16 persists (105/280); factorization helps
   efficiency but not correctness.
3. Drift detection by acceptance fails in 60% of streams.
4. Abstraction lifecycle cost is positive in every arm; no expansion or
   accuracy benefit.
5. Retrieval silently substitutes a learned concept on 100% of held-out-family
   queries — no out-of-family detection.
6. Misleading support still yields 8% wrong acceptance at 30%.
7. Conditional DSL does not yield correct acceptance in the tested deep setting.

**Fundamentally limited by evidence scope:** all results come from a small
typed-DSL synthetic universe. Nothing here validates natural-input behavior.

## 10. Integration-readiness assessment

**Not ready. No integration authorized.** `analysis.integration_readiness()`
records `integration_recommended = false`. The decisive reasons are: (1) all
evidence is synthetic; (2) J3's deep-search failure and J4's abstraction
negative result are not cleared; (3) J2 drift detection is unresolved; (4) two
cross-mechanism regressions are documented. Nothing in EXP-J supports a claim
of general intelligence, natural-language understanding, universal
generalization, or any OMEGA-NANO capability. No efficiency figure here should
be read as a system-level speed-up.

## 11. Reproduction

```powershell
cd "D:\ai model"
python -m constructive.exp_j.runner --stage A
python -m constructive.exp_j.runner --quick
python -m constructive.exp_j.runner --no-resume
python -m constructive.exp_j.runner --merge-only
python -m pytest tests/test_exp_j.py -q
python -m pytest -q
python -c "from constructive.exp_j.runner import verify_integrity; print(verify_integrity())"
```

Every run writes immutable snapshots to
`constructive/exp_j/results/experiment-history/`; `latest.json` is rebuilt from
per-experiment files with `--merge-only`. All generators are seed-deterministic
(no `hash()` of strings in data generation). Config, source hashes (including
every EXP-I file), hardware and dataset fingerprints are recorded in each block.

## 12. Exact next recommended research step

Do **not** promote any mechanism. The highest-value next experiment is a
semantic-equivalence-based validator (not sample-consistency) for deep
composition, paired with an explicit drift change-point detector, tested on the
same independent EXP-J generators before any further architectural change.
