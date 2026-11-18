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
