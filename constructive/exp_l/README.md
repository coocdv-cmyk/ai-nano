# EXP-L (isolated research package)

Generalization bridge, semantic drift, and selective reasoning reliability.

This package is deliberately isolated: it is not imported by OMEGA-NANO, CCM or
any earlier experiment, and it modifies no protected system. See the root
`EXP-L-REPORT.md` and `EXP-L-REPRODUCIBILITY.md` for evidence and commands.

## Modules

| Module | Purpose |
|---|---|
| `config.py` | Pre-registered config + config fingerprint |
| `determinism.py` | SHA-256-derived RNG, canonical hashing, fingerprints |
| `datasets.py` | Deterministic fixtures (split by family in later stages) |
| `metrics.py` | Shared, auditable metric definitions (added in L1+) |
| `integrity.py` | Read-only verification of protected hashes/manifests |
| `environment.py` | Reproducibility environment capture |
| `reporting.py` | Atomic artifacts, immutable snapshots, additive manifest |
| `runner.py` | CLI: `--stage --seed --quick --no-resume --merge-only` |

## Rules

- Documented hypothesis, preregistered config, fixed seeds, baseline, control,
  test condition, declared primary/secondary metrics, failure criterion,
  resource budget and reproducible artifact for every experiment.
- No final-test data is used for threshold/parameter selection.
- No oracle final answer may enter an operational decision.
- Abstention is not correctness; retrieval is not understanding; calibration
  is not correctness; synthetic performance is not real-world capability.
