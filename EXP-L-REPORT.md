# EXP-L Research Report

**Status: L0–L5 complete. Stages L6–L7 pending.**

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

## L2 — Selective composition verification (gate PASS, hypothesis weakly supported)

490 cases per seed (7 depths × 7 case types), 10 seeds; 7 methods. Wall 0.35 s,
CPU 0.34 s, peak RSS 26.6 MB. `decide()` receives a stripped `public_view`; the
hidden program, held-out test labels and first-error diagnostics never reach an
operational decision.

| Method | Incorrect accept | Useful coverage | Abstention | Selective accuracy | Checks/task |
|---|---|---|---|---|---|
| baseline | 0.021 | 0.332 | 0.646 | 0.939 | 0.0 |
| always-abstain | 0.000 | 0.000 | 1.000 | **undefined** | 0.0 |
| conservative | 0.000 | 0.186 | 0.814 | 1.000 | 5.0 |
| local (first 2) | 0.018 | 0.263 | 0.720 | 0.936 | 1.4 |
| selective (uncertain steps) | 0.016 | 0.280 | 0.704 | 0.946 | 1.2 |
| full (all steps) | 0.014 | 0.186 | 0.800 | 0.929 | 3.2 |
| budget (=depth) | 0.014 | 0.186 | 0.800 | 0.929 | 3.2 |

Paired per-seed bootstrap deltas vs baseline (10 seeds):

| Method | Δ incorrect acceptance [95% CI] | Δ useful coverage [95% CI] |
|---|---|---|
| selective | −0.0054 [−0.0107, 0.000] | −0.0518 [−0.0625, −0.0411] |
| full | −0.0071 [−0.0143, 0.000] | −0.1464 [−0.1589, −0.1339] |
| conservative | −0.0214 [−0.0304, −0.0107] | −0.1464 [−0.1589, −0.1339] |

The risk-coverage curve falls from 0.021 to 0.000 incorrect acceptance as the
checkpoint budget grows, but is **not strictly monotone** (small bumps at
budgets 8–12) and useful coverage falls from 0.332 to 0.200. Selective checking
preserves more useful coverage than full checking (0.280 vs 0.186) but its
incorrect-acceptance reduction is not statistically distinguishable from zero.
Only the conservative policy shows a credible reduction, and it sacrifices
roughly 44% of useful coverage. Compensating-error cases (correct final answer
via an invalid intermediate path) are correctly identified as intermediate
errors and are rejected by full/conservative verification — at a coverage cost.

**Conclusion.** The hypothesis that targeted intermediate verification gives a
credible correctness gain at low coverage cost is **not supported** by this
synthetic benchmark. The structural gate passed (oracle isolation, undefined
selective accuracy for all-abstain, a real risk-coverage trade-off), but the
effect size is too small to be practically meaningful.

Decision: `REQUIRES_MORE_EVIDENCE` (selective verification);
`NOT_RECOMMENDED` (full/conservative verification as coverage-preserving
improvements). Two failed L2 diagnostic snapshots are preserved in history.

Artifact: `constructive/exp_l/results/L2-results.json`.

## L3 — Retrieval generalization (gate PASS)

270 queries from 10 seeds, 9 query kinds, 12 concepts split by **family
identity** (6 familiar families in the retrieval pool, 6 held-out families
absent). Wall 0.034 s, CPU 0.031 s, peak RSS 24.88 MB.

| Method | Recall@1 | Silent substitution (held-out) | Familiar correct acceptance | Useful coverage |
|---|---|---|---|---|
| baseline | 1.000 | 1.000 | 1.000 | 0.333 |
| type-aware | 1.000 | 1.000 | 1.000 | 0.333 |
| similarity-only (surface) | 1.000 | 1.000 | 1.000 | 0.333 |
| evidence-aware | 1.000 | 0.000 | 1.000 | 0.333 |
| independent verification | 1.000 | 0.250 | 1.000 | 0.333 |
| abstention-based | 1.000 | 0.000 | 1.000 | 0.333 |
| rank + verify | 1.000 | 0.000 | 1.000 | 0.333 |

Paired per-seed deltas vs baseline: evidence-aware, abstention and rank+verify
each reduce substitution by −1.000 [−1.000, −1.000] with familiar correct
acceptance change 0.000 [0.000, 0.000]. Recall@1 is 1.000 for methods that
accept everything because the correct concept is always top-ranked — recall
alone would have hidden the 100% substitution on held-out queries.

**Important limitation.** In this benchmark familiar concepts have perfect
behavioral agreement and held-out concepts have near-zero agreement, so the
separation is clean and safeguards cost **no** familiar coverage. This does
**not** reproduce EXP-K's familiar top-1 collapse (0.867 → 0.333) and must not
be read as evidence that real retrieval safeguards are free. Type-aware and
surface-similarity ranking changed nothing, consistent with EXP-K's finding
that shared input/output types defeat type constraints. Held-out recall is
undefined (no correct concept in the pool) and is not reported as a success.
The independent-verification variant still substituted on 25% of held-out
queries, showing that a single high-agreement candidate is not sufficient.

Decision: `PROMISING_BUT_UNVALIDATED` (behavioral-agreement safeguards);
`NOT_RECOMMENDED` (type-aware / surface-similarity ranking).

Artifact: `constructive/exp_l/results/L3-results.json`.

## L4 — Conditional calibration (gate PASS with disclosed subgroup regression)

800 evaluation rows (10 seeds × 8 conditions × 10), 480 calibration rows from
disjoint seeds 201–204. Wall/CPU/memory recorded in the artifact. All
parameters fitted on calibration seeds only; accuracy is unchanged by every
post-hoc transform (0.417).

| Method | ECE | Adaptive ECE | Brier | NLL | Accuracy |
|---|---|---|---|---|---|
| raw | 0.2491 | 0.2437 | 0.2959 | 0.8111 | 0.417 |
| temperature (T=5.0) | 0.1177 | 0.1193 | 0.2462 | 0.6854 | 0.417 |
| binning (10 bins) | 0.0407 | 0.0527 | 0.2265 | 0.7412 | 0.417 |
| conditional by family | 0.0585 | 0.0716 | 0.2361 | 1.0476 | 0.417 |
| conditional by depth+evidence | 0.0989 | 0.0889 | 0.2346 | 0.8463 | 0.417 |

Difficult subgroups improved by temperature: deep 0.405→0.147,
retrieval-uncertain 0.225→0.001, semantic-drift 0.386→0.229,
adversarial 0.361→0.168, OOD 0.448→0.290. **Hidden regression disclosed:**
temperature degraded easy subgroups — normal 0.055→0.158 and contradictory
0.081→0.317; binning degraded normal 0.055→0.203 and low-evidence
0.030→0.193. Conditional-by-family overfit calibration and produced the worst
NLL (1.048). Every reported subgroup has n=150 (above the n=30 reliability
floor). The fitted T=5.0 reflects the deliberately large confidence biases in
this synthetic fixture and is not transferable to a real model.

Confidence remains a heuristic score: calibration never changed `accuracy` and
is not evidence of correctness. The gate checks the stated criterion (no
degradation in *difficult* subgroups), while the easy-subgroup regressions are
recorded in `subgroup_analysis` and must not be ignored.

Decision: `PROMISING_BUT_UNVALIDATED` (temperature scaling);
`REQUIRES_MORE_EVIDENCE` (conditional depth+evidence);
`NOT_RECOMMENDED` (binning and conditional-by-family as general solutions).

Artifact: `constructive/exp_l/results/L4-results.json`.

## L5 — Natural-input bridge (gate PASS, narrowly scoped)

840 self-authored CC0 examples, 14 templates (7 dev, 7 held-out, split by
template ID so no phrasing crosses splits). No external model, no third-party
data, no OMEGA-NANO invocation. Wall 0.021 s, CPU 0.031 s, peak RSS 25.45 MB.

| Method | Exact match | Held-out exact match | Correct abstention | Incorrect acceptance | Parse failure |
|---|---|---|---|---|---|
| constant (0) | 0.030 | 0.033 | 0.000 | 0.286 | n/a (no parser) |
| first number | 0.007 | 0.005 | 0.000 | 0.286 | n/a (no parser) |
| adapter, no ambiguity guard | 0.600 | 0.600 | 0.000 | 0.286 | 0.000 |
| adapter, with guard | **1.000** | **1.000** | 0.286 | 0.000 | 0.000 |

Parsing, language-to-representation mapping, reasoning and ambiguity detection
were scored separately. The unguarded adapter parses every example but accepts
all 28.6% ambiguous/incomplete cases; the guarded adapter abstains on all of
them. Failed cases are preserved in the result artifact.

**Critical limitation.** The adapter and the ground-truth generator were
co-designed around a small shared operation vocabulary and regex normalization.
The perfect score therefore measures template-parsing transfer, **not** natural
language understanding, broad paraphrase robustness, or real-world capability.
There are only 14 hand-written templates; no held-out *lexical* vocabulary and
no genuinely natural corpus. Nothing here generalizes beyond these families.

Decision: `PROMISING_BUT_UNVALIDATED` (the bridge as a measured capability);
`NOT_RECOMMENDED` (the adapter as evidence of language understanding or for
integration).

Artifact: `constructive/exp_l/results/L5-results.json`.

## Decision status

No mechanism is `ELIGIBLE_FOR_SEPARATE_REVIEW`. L1 mapping consistency is
promising but synthetic-only; L2 selective verification lacks a credible effect
size; L3 behavioral safeguards are unvalidated on overlapping realistic families;
L4 calibration regresses easy subgroups; L5 shows template-level transfer only.
