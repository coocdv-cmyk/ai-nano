# EXP-K — Root-Cause Analysis (living document)

## K0 — Reproduced baseline (10 seeds, 16 new rule families, 800 datasets)

All targets are **reproduction targets from EXP-J**, not required outcomes.
EXP-K uses independent generators, so exact equality is not expected.

| Failure condition | EXP-J reference | EXP-K baseline (matched constructor, 10 seeds) | Reproduced? |
|---|---|---|---|
| Deep-composition incorrect acceptance | 105/280 = 0.375 | **39/70 = 0.557** (by depth: d2 0.6, d3 0.7, d4 0.6, d5 0.6, d8 0.3, d12 0.4, d16 0.7) | Yes — failure present, higher rate |
| OOD retrieval silent substitution | 1.00 | **0.40** (ambiguity 0.60) | Yes — failure present |
| Drift missed streams | 0.60 | **0.575** (gradual 0.575, abrupt 0.575) | Yes |
| Independent ECE | 0.1596 | **0.1699** (Brier 0.0516, n_fit=60, n_eval=60) | Yes |
| Misleading-support wrong acceptance | 0.08 | **0.05** | Yes |

**Integrity:** EXP-I 22/22 and EXP-J 84/84 protected hashes unchanged.
**Resources:** 800 datasets, 15.19 s wall, 33.64 MB peak RSS, CPU-only.

### Initial hypotheses to test in K1–K4

1. **Deep composition:** incorrect acceptance is driven by support/validation
   consistency being mistaken for semantic equivalence; shallow incidental fits
   survive because no intermediate contract or whole-program counterexample
   check is applied. To be isolated in K1 with first-error-depth tracing.
2. **Retrieval:** silent substitution occurs when type-compatible but
   semantically distinct concepts score closely and no explicit OOD rejection
   or compatibility constraint is applied. To be isolated in K2.
3. **Drift:** acceptance-based "detection" cannot notice drift when the label
   rule remains valid off-domain; no explicit change-point statistic exists.
   To be isolated in K3.
4. **Calibration:** raw confidence is a heuristic blend of support fit,
   validation accuracy and uniqueness, not a probability; binning on small
   splits leaves substantial ECE. To be isolated in K4.

These are hypotheses, not findings. Later stages will confirm or reject them.

## K1 — Deep composition (350 matched runs, 5 arms × 7 depths × 10 seeds)

| Arm | Incorrect acceptance | Correct acceptance | Abstention | Mean expansions |
|---|---|---|---|---|
| baseline | 0.557 (CI ±0.116) | 0.000 | 0.443 | 163.6 |
| shifted validation | 0.329 | 0.000 | 0.671 | 163.6 |
| CEGIS refinement | 0.329 | 0.000 | 0.671 | 268.4 |
| conservative abstention | 0.329 | 0.000 | 0.671 | 163.6 |
| oracle intermediate contracts | **0.000** | 0.000 | 1.000 | 163.6 |

**Findings.**
1. Incorrect acceptance is **reproduced** (39/70 baseline) and is not caused by
   resource exhaustion (0 timeouts) or representation limits (all chains are
   in-DSL). It is a validation blind spot: support-consistent, incidentally
   fitting programs survive.
2. First-error tracing shows the earliest divergence occurs at composition step
   **2** (median), i.e. early and propagates to the final answer.
3. Three deployable safeguards each cut incorrect acceptance by 0.229 absolute
   (41% relative) on 70 paired episodes, but **all buy it with coverage**:
   abstention rises 0.443 → 0.671, and correct acceptance stays at 0.000.
   CEGIS costs 64% more expansions for the same outcome as shifted validation.
4. The **oracle-contract arm eliminates all incorrect acceptance but abstains
   on everything**. This confirms that perfect intermediate verification would
   solve the safety problem, but it is not deployable (needs the hidden chain)
   and provides no useful coverage.
5. Hypothesis 1 is therefore **partially confirmed**: the cause is a validation
   blind spot, but the tested fixes trade safety for coverage rather than
   solving composition. `shifted_validation` is the best deployable arm on the
   safety/cost trade-off, yet it is not a solution.

**Resource:** 80.22 s wall, peak RSS measured per block, 0 timeouts.

## K2 — OOD retrieval (720 queries, 6 arms, 10 seeds)

| Arm | Silent substitution (OOD) | Correct abstention (OOD) | Correct top (familiar) | Wrong abstention (familiar) |
|---|---|---|---|---|
| baseline | **1.000** | 0.000 | 0.867 | 0.000 |
| type-aware only | 1.000 | 0.000 | 0.867 | 0.000 |
| OOD rejection | 0.020 | 0.980 | 0.367 | 0.633 |
| ambiguity abstention | 0.020 | 0.980 | 0.367 | 0.633 |
| provenance-compatible | 0.000 | 1.000 | 0.367 | 0.633 |
| independent verification | **0.000** | **1.000** | 0.333 | 0.667 |

**Findings.**
1. The EXP-J result is **fully reproduced** on independently generated
   type-matched held-out families: 100% silent substitution at baseline.
2. Hypothesis "missing type constraints" is **rejected**: type-aware ranking
   changes nothing, because learned and held-out families share input/output
   types. The cause is behavioral-similarity collision without an OOD rule.
3. Behavioral agreement thresholds and ambiguity abstention suppress silent
   substitution to ~2%, and provenance/independent verification suppress it to
   0%, but every such arm loses about two thirds of familiar coverage
   (0.867 → 0.333–0.367) through false rejection.
4. Therefore no tested safeguard solves retrieval: it converts silent wrong
   answers into honest abstentions, at a large coverage cost. The best safety
   arm is `independent_verification` (0% substitution, 100% OOD abstention),
   but it is the worst on familiar coverage.
5. Recall@2/precision@2 are dominated by the same trade-off; latency is small
   (<0.3 ms per query) and is not the limiting factor.

## K3 — Drift detection (2,400 stream-arm rows, 5 arms, 10 seeds)

| Arm | Detection recall | Miss rate | False-positive rate (stable) | Median delay |
|---|---|---|---|---|
| acceptance baseline | 0.207 | 0.793 | 0.325 | 1 |
| accuracy window | 0.184 | 0.816 | 0.225 | 0 |
| **feature window** | **0.998** | **0.002** | **0.000** | 0 |
| two-window | 0.868 | 0.132 | 0.000 | 1 |
| combined | 0.998 | 0.002 | 0.225 | 0 |

**Findings.**
1. The EXP-J failure is **reproduced and worsened** in this matched setting:
   acceptance-based detection misses 79% of drift streams, and raises false
   alarms on 32% of stable streams.
2. Accuracy-based detection fails for the same reason as K0: validation
   accuracy is often *higher* during drift (0.992) than before it (0.845),
   because the true rule still generalizes off-domain.
3. A simple input-feature statistic with a threshold fixed on separate
   calibration seeds detects 99.8% of drift streams with **zero** false alarms
   on stable streams. This is the clearest positive result so far. Hypothesis 3
   (no explicit change-point statistic) is **confirmed** for feature drift.
4. Caveat: all twelve named "drift kinds" are implemented as input-domain
   moves; genuine concept drift with unchanged inputs is not separately modeled.
   The positive result therefore applies to feature/parameter drift, not to all
   semantic drift. The `combined` arm shows that adding the accuracy signal
   reintroduces false alarms without improving recall.
5. Thresholds (`feature_threshold=19.96`, `window=2`, `accuracy_threshold=0.825`)
   were fixed on calibration seeds 101–103 before held-out evaluation.

## K4 — Calibration (140 calibration + 350 held-out rows, 5 methods)

| Method | ECE | Brier | NLL | Accuracy |
|---|---|---|---|---|
| raw | 0.1258 | 0.0841 | 0.2840 | 0.491 |
| temperature (T=0.75) | **0.0741** | 0.0770 | 0.2639 | 0.491 |
| isotonic | 0.0982 | **0.0679** | 0.8983 | 0.491 |
| binning | 0.0842 | 0.0768 | **0.2620** | 0.491 |
| conformal abstention | risk-controlled coverage (ECE not applicable) | | | |

Per-condition raw ECE: normal 0.185, deep **0.424**, OOD 0.163, noise 0.015,
contradictory 0.000, retrieval substitution **0.410**.

**Findings.**
1. EXP-J's calibration failure is **reproduced** (raw ECE 0.126 on this held-out
   split; 0.185 on the normal-condition subset alone).
2. Temperature scaling (fit on separate seeds) nearly halves ECE, and binning
   gives the best NLL. Isotonic improves Brier but **destroys NLL** on only 140
   calibration rows — a concrete small-sample warning.
3. No method changes accuracy: these are post-hoc probability transforms.
   Calibrating confidence is not the same as making the underlying decisions
   correct (Hypothesis 4 is **partially confirmed**).
4. Calibration **fails where it matters most**: deep compositions (ECE 0.424,
   temperature makes it worse) and retrieval substitutions (0.410) have near-zero
   accuracy yet high confidence. Contradictory cases are trivially calibrated
   because the engine emits confidence 0.
5. Conformal-style abstention is measured as a risk/coverage trade-off, not as
   a probability transform, and is not claimed to prove correctness.

