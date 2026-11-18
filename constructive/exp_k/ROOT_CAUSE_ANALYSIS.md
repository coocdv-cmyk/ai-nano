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

