# EXP-J — Root-Cause Updates and Before/After Comparisons

All numbers are from `results/latest.json` (full 10-seed campaign). "Before" is
EXP-I's own reported measurement or the independent baseline computed inside
EXP-J; the source of each baseline is named.

## 1. Evidence validation on independent distributions — VALIDATED

| Metric | Before (EXP-I / baseline) | After (EXP-J J1) | Verdict |
|---|---|---|---|
| Wrong acceptance | support-only baseline 0.2375 (n=480) | **0.0000** (n=480), 95% CI 0 | Confirmed on independent families |
| Acceptance under correct rule | — | 0.783 in-domain, 0.742 combined-shift | Coverage retained, not "reject all" |
| Retrieval recall @2 | EXP-I 1.0 on H4 families | **0.9917** (n=360) on 12 new families | Transfers |
| Independent-split ECE | EXP-I ECE 0.000 (optimistic, same-domain) | **0.1596** (fit/eval split) | Honest, non-zero; calibration is *not* solved |

Root cause: EXP-I's abstention was real, but its ECE was optimistic because
correctness was nearly deterministic on the original twin benchmark. EXP-J's
new families produce genuinely confusable hypotheses and a non-zero ECE.
**Held-out-family transfer:** retrieval silently substituted a different
learned concept on 60/60 queries from unseen families (0 ambiguity flags), so
cross-family transfer is unresolved despite within-family recall of 0.9917.

## 2. Noise and corruption — PARTIALLY VALIDATED

| Condition | 0% correct acceptance | 30% correct acceptance | 30% abstention | 30% contradiction detected |
|---|---|---|---|---|
| clean | 0.80 | 0.80 | 0.20 | — |
| random label | 0.80 | 0.04 | 0.96 | 0.72 |
| structured corruption | 0.80 | 0.16 | 0.82 | 0.38 |
| contradiction | 0.80 | 0.00 | 1.00 | **1.00** |
| missing | 0.80 | 0.72 | 0.28 | — |
| duplicate | 0.80 | 0.80 | 0.20 | — |
| misleading support | 0.80 | 0.46 | 0.46 | 0.34 |

- Correct acceptance falls and abstention rises with noise — appropriate, but
  not free: at 30% random-label noise the engine correctly accepts only 4% of
  tasks. This is a **coverage limitation**, not evidence of intelligence.
- Contradictory evidence is detected in 100% of injected cases at 30%, and
  "missing" is *not* collapsed into "contradictory" (0% contradiction for
  missing). State discrimination works.
- Misleading support still yields 8% wrong acceptance at 30% — **unresolved**.
- Accuracy–coverage at 30% misleading support: accuracy 0.46, coverage 0.54,
  selective accuracy 0.852. Reporting selective accuracy prevents the mistake
  of equating abstention with correctness.

## 3. Drift — UNRESOLVED

| Drift kind | Detection delay (mean, episodes) | Undetected streams | Recovery delay |
|---|---|---|---|
| gradual | 2.31 (CI ±1.28) | 24 / 40 | 0.14 |
| abrupt | 2.31 (CI ±1.28) | 24 / 40 | 0.14 |

Root cause: when the label rule stays valid off-domain (as in these generated
families), the engine continues to accept correctly on drifted validation data;
acceptance-based "detection" therefore does not fire in 60% of streams. The
engine has no explicit change-point detector. This is a genuine architectural
gap, not a measurement artifact.

## 4. Deep composition — UNRESOLVED (re-confirmed)

EXP-I reported spurious acceptance 0.33–0.67 at depths 2–5 and safe abstention
with zero solves at depths ≥8. EXP-J with 10 seeds per depth finds:

| Depth | correct discovered | incorrect accepted | correct not discovered | mean expansions (plain) |
|---|---|---|---|---|
| 2 | 5/10 | 1/10 | 4/10 | 63.8 |
| 3 | 4/10 | 4/10 | 2/10 | 179.9 |
| 4 | 2/10 | 4/10 | 4/10 | 180.7 |
| 5 | 3/10 | 3/10 | 4/10 | 180.5 |
| 8 | 1/10 | 3/10 | 6/10 | 180.5 |
| 12 | 1/10 | 6/10 | 3/10 | 180.0 |
| 16 | 1/10 | 6/10 | 3/10 | 180.6 |

**Root cause:** the search is bounded by a cost/beam frontier that admits
incidentally-fitting shallow programs; validation judges consistency with a
small support/validation pool, not semantic equivalence. Expanding the budget
does not fix it: `cached` (shared memo, no eviction) raised mean expansions to
900 and produced no improvement in correct discovery. `factorized` (oracle op
subset) reduced mean expansions to 110 and gave 22 correct discoveries vs 17
for `plain`, but still accepted 24 spurious programs — factorization narrows
search space but does not solve spurious acceptance. `cegis` added counterexample
points (mean 269 expansions) with no outcome gain. Six failure categories are
recorded explicitly; no budget was raised to force a success.

## 5. Abstraction economics — NEGATIVE RESULT REPRODUCED

EXP-I: 81 abstractions, 29 reuse events, identical expansions (9,842) and
accuracy (0.80625). EXP-J (5 seeds, 6-task streams):

| Arm | shared-core expansions | unrelated expansions | net lifecycle cost (shared / unrelated) |
|---|---|---|---|
| A no abstraction | 4,401 | 4,390 | 0 / 0 |
| B current | 4,401 | 4,390 | +0.00955 / +0.01094 |
| C reuse-credit | 4,401 | 4,390 | +0.00437 / +0.00396 |
| D retrieval-ranked | 4,401 | 4,390 | +0.00456 / +0.00193 |
| E selective | 4,401 | 4,390 | +0.00352 / +0.00422 |

Abstractions are genuinely discovered and reused (B: 80 abstractions / 25 reuse
events shared-core), but **expansions and accuracy are identical to baseline in
every arm**. Root cause is therefore a combination of a *cost-model /
search-architecture* problem and a *useless-abstraction* problem: injected
abstractions behave like ordinary atoms and are deduplicated by the search's
output-level memo, so they neither shrink the search nor earn a reuse credit
that changes cost. Selective retention and retrieval-aware ranking reduce the
*waste* (lower net cost than B) but never produce a net saving. No positive
result is claimed.

## 6. Cross-mechanism interaction — TWO REGRESSIONS DOCUMENTED

- `evidence_validation + deep_search`: relative to search-only, adding evidence
  validation reduced correct acceptance 0.10 → 0.00 and coverage 0.80 → 0.60
  at depth 5. Individual success (fewer wrong acceptances) does not survive the
  combination unchanged. Recorded in `latest.json` under `J5.regressions`.
- `concept_drift + retrieval`: recall@1 falls in the drift phase relative to
  base in the controlled stream; see `J5.combinations`.
- `conditional DSL + deep composition`: even with the conditional operator
  enabled, correct acceptance remained 0.00 — representation availability is
  not discovery.
- `program repair + adversarial evidence`: 0 false verified repairs in both
  limited- and rich-visible arms; verification coverage 1.0. No regression
  detected here.

## 7. Resource governance — VALIDATED

6/6 principal experiments completed with explicit `COMPLETED` state; 0 runner
errors; 0 timeouts; total wall 150.96 s; peak RSS 48.76 MB; 136,137 search
expansions. A forced zero-deadline governor raised `ResourceLimit` and produced
a `TIMED_OUT` completion state in the apparatus test — a timeout can never be
merged as success.
