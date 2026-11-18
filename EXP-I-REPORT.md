# OMEGA EXP-I — Final Report

| | |
|---|---|
| Date | 2026-11-18 |
| Scope | Isolated research branch `constructive/exp_i/`; no integration |
| Environment | i5-1135G7 4C/8T @2.4 GHz, 16.82 GB, Windows 11 26200, Python 3.13.15, NumPy 2.5.3, **CPU-only, no external APIs** |
| Implementation | 18 Python modules under `constructive/exp_i/`; `tests/test_exp_i.py` |
| Results | `constructive/exp_i/results/latest.json` + 8 per-experiment files + **immutable** `experiment-history/` snapshots (6 campaign stamps, 48 snapshots) |
| Tests | EXP-I: **64 passed**; full repo: **229 passed, 1 skipped** (baseline was 165 passed, 1 skipped) |
| Campaign | 8/8 experiments `completed`; wall ~63 s; peak RSS ~207 MB; 0 experiment-level timeouts |
| Integrity | 22/22 protected-file SHA-256 hashes unchanged from STAGE 0 |
| Source revision | `no-commits` (repository has no commits) |

---

## 1. What was implemented

An independent, modular experimental system (no CCM dependency): strict typed
DSL with safe bounded interpreter; multi-sample adaptive evidence engine with six
validation states, explicit non-identifiability, calibration and abstention;
collision-aware versioned concept retrieval; memoized cost-aware typed search;
independent generators with disjoint splits; numeric/structural OOD tests;
separated repair generation and bounded verification; validated abstraction
harvest with provenance/retirement; and central resource governance with global,
per-experiment and per-fit deadlines, node budgets, cancellation and structured
timeout reporting. See `ARCHITECTURE.md` and `ROOT-CAUSE-MATRIX.md`.

## 2. Root causes addressed — before / after

All numbers come from `results/latest.json`; baselines from
`CONSTRUCTIVE-EXP-H-REPORT.md`. Different but matched benchmark families were
used where EXP-H's raw probes were not reproducible; this is stated per row.

| Area | EXP-H baseline | EXP-I measured | Verdict |
|---|---|---|---|
| **A. Wrong-twin acceptance** (H1) | 15/60 wrong twins accepted; abstain 0.15 | **0 wrong acceptances / 60** (n=30 discriminating + 30 ambiguous); coverage 0.933; false rejection 0.067; ambiguous-condition correct abstention 1.0; ECE 0.0 | **Fixed on this benchmark** |
| **B. Deep composition** (H2) | depth-2 acc 1.0; spurious 0.33–0.67 at depth ≥3; 2 timeouts | depth 2: correct 0.33, spurious 0.33; depth 3/5: spurious 0.67; depth 8/12/16: spurious 0 but correct 0; ~185–206 expansions, 0.15–0.17 s, 0 timeouts (n=3/depth) | **Partially improved / unresolved** |
| **C. Retrieval interference** (H4) | 13/157 (8.3%) interference, parametric families | silent substitution **1/60 (1.67%)**; correct-top 0.95; ambiguity 0.067; precision/recall@2 = 1.0; latency 0.104 ms; store 1147 B (n=60, 2 seeds, exact H4 families) | **Improved** (different probe set, same families; not the same 157 probes) |
| **D. Conditionals / exclusion** (H1/H2) | exclusion 0.007, conditional 0.048 — unreachable | bitwise and (nested) conditionals translated, well-typed and evaluated; all four expressiveness cases classified `EXPRESSIBLE_DISCOVERED`; all DSL safety checks pass | **Representation fixed; discovery not measured** |
| **E. Numeric OOD** (H3) | numeric 0.728; `sum_digits` 0.185 spuriously accepted | `sum_digits`: **0/5 accepted** (correct abstention); square/fib/is_prime/sum_of_evens/distinct_count: 5/5 accepted, **1.0 OOD accuracy**, 0 false acceptance; scene rules 0/5 accepted (explicit representation limitation) | **Fixed for numeric; representation limit for scenes** |
| **F. False repairs** (H5) | 5/144 false repairs | **0/36 false verified**; 3/36 detected false repairs; verification coverage 0.997; states: 12 VERIFIED, 21 AMBIGUOUS, 3 NOT_EQUIVALENT; generator never saw the oracle | **Acceptance target met** (36 episodes, 6 templates incl. the original 5 cases) |
| **G. Contradiction / corruption** (H7) | contradiction 1/6 salvage; blind-positive 6/6 spurious | contradictory support → `CONTRADICTORY`; corrupted validation → never ACCEPTED; blind-positive → abstains; one-example misleading support → abstains | **Fixed on these cases** |
| **H. Abstraction transfer** (H6) | 2.66× fewer expansions, 6.1× faster, acc 0.786→0.944, 1 false abstraction | 81 abstractions created, 29 reuse events, but **no measured benefit**: expansions 9,842 = 9,842; accuracy 0.80625 = 0.80625; fit median 0.164 s vs 0.169 s (harvest slightly slower) | **Negative result — not replicated** |
| **I. Resource governance** (EXP-H §I.8) | no global timeout | 6/6 governance checks pass: deadline, node budget, cancellation, timeout≠success, deterministic rerun, failure isolation | **Fixed** |

## 3. Detailed evidence

### I-A — Evidence engine (n=60 twin episodes)
- Discriminating validation: coverage mean 0.933 (CI ±0.089), selective accuracy
  1.000, wrong acceptance **0.000**, false rejection 0.067.
- Ambiguous validation: coverage 0.000, correct abstention **1.000**,
  32 `NON_IDENTIFIABLE` states recorded explicitly.
- Calibration: ECE 0.000 on an independent half-split. **Caveat:** correctness is
  nearly deterministic in this synthetic setup, so this ECE is optimistic and
  is not evidence of calibration on natural or noisy distributions.
- Accuracy/coverage trade-off is reported rather than "reject everything":
  coverage stays 0.933 while wrong acceptance falls to zero.

### I-C — Retrieval (n=60, 2 seeds, families `count_gt`, `take`, `filter_gt`, `map_add`, `add`)
- Silent substitution 1/60; correct top-1 57/60; ambiguity raised in 4/60;
  top-2 precision and recall both 1.0.
- Meaning: near-identical parameterizations are ranked with behavioral
  fingerprints; when two are plausible but behaviorally distinct the engine
  returns both and requests evidence instead of silently substituting.
- Memory/latency are measured, not assumed. The EXP-H 250-concept claim is
  **not** repeated: pool expansion is limited to DSL-supported concepts.

### I-D — Deep search (n=18, 3 seeds, depths 2/3/5/8/12/16)
- Small search budget (hard caps): ~185–206 expansions, 0.15–0.17 s, no timeouts.
- At depths ≥8 the system **abstains safely** (spurious 0) but also solves none
  of these chains. At depths 2–5 it still accepts incidentally-fitting programs
  (spurious 0.33–0.67). This is the principal unresolved engineering problem:
  the typed search + validation combination has not beaten EXP-H's
  spurious-acceptance failure at shallow-to-moderate depth.
- Sample size is 3 per depth — far below the preferred 30. No generalization
  claim is made from it.

### I-E — Generalization (n=45, 9 rules × 5 seeds)
- Rules that are discriminable in-domain generalize perfectly OOD (1.0).
- `sum_digits` — EXP-H's canonical OOD failure — abstains in 5/5 episodes
  because `sum_digits` and `digital_root` are unidentifiable on the training
  domain. This is an epistemic fix (refusing to guess), not a fitting trick.
- `nearest`, `same_shape`, `left_of` are reported as `REJECTED` because the
  typed int/list DSL has no scene representation; they are **not** counted as
  successful or as generalization evidence.

### I-F — Repair (n=36)
- False *verified* repairs: 0. Three repairs that passed visible examples were
  **detected** as non-equivalent by bounded exhaustive checking and
  adversarial counterexamples; 21 cases were explicitly `AMBIGUOUS` rather
  than labelled semantically correct.
- The repair generator received only visible examples and the buggy program;
  the hidden correct program was used only by the evaluation harness/verifier.

### I-G — Abstraction (n=120, 4 families × 3 seeds × 2 arms)
- Harvest produced real, validated, reusable abstractions (81 units; 29 reuse
  events) on shared-core *and* unrelated families, so the discovery mechanism
  transferable in the narrow sense of "structures are found and reused".
- However it produced **no measurable search or accuracy benefit** in this
  implementation: identical expansion totals and accuracy to baseline, with a
  marginally higher median fit time. This fails to replicate EXP-H H6's win.
  Likely causes: injected abstractions increase atom count and beam pressure;
  the cost model does not credit reuse; different search architecture. This
  requires further research, not promotion.

## 4. Failure taxonomy (339 episode rows)

`success 242, wrong-acceptance 5, non-identifiable 32, insufficient-evidence 5,
representation-limitation 15, search-explosion 34, false-retrieval 1,
false-repair-detected 3, undergeneralization 2`. Counts are recorded, never
collapsed or deleted. Complete per-episode data is in `latest.json`.

## 5. Verification and integrity

- `python -m pytest tests/test_exp_i.py -q` → **64 passed**.
- `python -m pytest -q` → **229 passed, 1 skipped** (the skip is pre-existing).
- Stage-A smoke → **13/13 pass**.
- Full campaign → 8/8 `completed`, 0 runner errors, all experiments with
  explicit `completion_status`; no timeout produced a `COMPLETED` result.
- Protected-file verification: **22/22 SHA-256 hashes identical** to the
  STAGE-0 manifest (`results/baseline_hashes.json`, `results/integrity.json`).
  CCM, `omega_nano/`, EXP-H source and all historical snapshots are untouched.
- Peak RSS per experiment: 205.3–207.1 MB (token-aligned with Python process
  overhead); concept-store bytes 954–1336 B.
- CPU wall time for the final full campaign: approximately **63 seconds**.

## 6. What remains

**Fixed on synthetic benchmarks:** wrong-twin acceptance, retrieval silent
substitution, conditional/exclusion representation, numeric OOD `sum_digits`
false confidence, false verified repairs, contradiction/blind-positive
safe-failure, global resource governance.

**Partially improved:** non-identifiability is now explicit and abstention is
calibrated; interference is reduced but retrieval collisions still occur
(1/60 silent substitution, 4/60 ambiguity); deep search abstains safely at
high depth but still accepts spurious programs at depth 2–5.

**Unresolved:** spurious shallow/mid-depth acceptance in the typed search;
abstraction harvest shows no efficiency or accuracy transfer benefit here.

**Fundamentally unidentifiable (from available evidence):** twin rules whose
support and validation domains coincide; `sum_digits` vs `digital_root` on
single digits. The only correct output is `NON_IDENTIFIABLE`, which the engine
now emits instead of guessing.

**Representation limitations:** scene relations (`nearest`, `same_shape`,
`left_of`) have no typed DSL program and are reported as rejected, not fitted.

**Computational limitations:** depth ≥8 chains are bounded-abstained; the
search is cost-budgeted and will not solve arbitrary deep compositions.

**Requires further research:** calibration under noisy/non-synthetic evidence;
a cost model that actually rewards abstraction reuse; adversarial robustness
of retrieval across larger concept pools; any transfer to natural input.

## 7. Integration-readiness assessment

**Not ready. No integration authorized.** `analysis.integration_readiness()`
records `integration_recommended = false` even though the synthetic-benchmark
evidence flag is true, for two decisive reasons: (1) all evidence is from a
small typed-DSL toy universe, with no natural-input validation; (2)EXP-I's own
deep-search and abstraction-transfer experiments are not cleared. Nothing here
supports a claim of general intelligence, natural-language understanding,
universal generalization, or any OMEGA-NANO capability. No efficiency figure
from EXP-I should be read as a system-level speed-up.

## 8. Exact next recommended research step

Run a **transfer-and-noise campaign** before any further architectural change:
(1) replace the synthetic twin/generator families with independently written
generators plus label noise and distribution drift, and re-measure I-A
calibration and abstention; (2) raise I-D to ≥10 seeds per depth and test a
reuse-crediting cost model for I-G to determine whether the abstraction
negative result is a cost-model artefact or a genuine limitation. Do not
promote any mechanism into CCM or OMEGA-NANO on the present evidence.

## 9. Reproduction

```powershell
cd "D:\ai model"
python -m constructive.exp_i.runner --stage A
python -m constructive.exp_i.runner --quick
python -m constructive.exp_i.runner --no-resume
python -m pytest tests/test_exp_i.py -q
python -m pytest -q
```

Every run writes a new immutable snapshot to
`constructive/exp_i/results/experiment-history/`; `latest.json` is regenerated
from the per-experiment files with `--merge-only`. Protected hashes are checked
by `python -c "from constructive.exp_i.runner import verify_integrity; print(verify_integrity())"`.
