# `constructive/exp_i/` — EXP-I research module

Isolated experimental research phase on root causes documented in
`CONSTRUCTIVE-EXP-H-REPORT.md`. **EXP-I is a research branch only.** It does not
modify CCM, `omega_nano/`, protected Phase-0 architecture, EXP-H source, or any
historical result. All code lives here; all outputs go to `results/`.

- No external APIs. No GPU. CPU-only, stdlib-only.
- Python 3.13.15, NumPy 2.5.3 present but not required.
- See `ROOT-CAUSE-MATRIX.md` (pre-implementation traceability) and
  `ARCHITECTURE.md` (design contract), and `EXP-I-REPORT.md` (measured results).

## Modules

| Module | Role |
|---|---|
| `types.py` | Immutable data, validation states, completion status |
| `config.py` | Bounded config, hardware probe, fingerprints, SHA-256 |
| `resources.py` | Global/per-experiment/per-fit deadlines, node budgets, cancellation, memory guard, structured timeout reporting |
| `dsl.py` | Typed AST: conditionals, booleans, comparisons, logic, bitwise exclusion, bounded arithmetic, list ops; canonicalization, complexity, safe execution |
| `evidence.py` | Scoring, disagreement, counterexamples, adaptive selection, contradiction/blind-positive detection, provenance |
| `validation.py` | Multi-sample validation, six states, calibration (ECE), abstention |
| `retrieval.py` | Typed signatures, structural/behavioral fingerprints, collision/ambiguity diagnostics, versioned immutable history, retirement |
| `search.py` | Memoized cost-aware typed search, type-distance pruning, CEGIS refinement, hard budgets |
| `generators.py` | Independent task families with disjoint support/validation/test splits |
| `generalization.py` | Numeric/structural OOD, domain boundaries |
| `repair.py` | Repair generation ⟂ verification, bounded exhaustive equivalence, adversarial counterexamples |
| `abstraction.py` | Subtree harvest, reusability, validation, provenance, retirement |
| `metrics.py` | n/mean/median/std/CI, accuracy/coverage, comparisons, taxonomy |
| `experiments.py` | Experiment definitions I-A … I-H |
| `analysis.py` | Cross-experiment summaries, conservative integration assessment |
| `runner.py` | Orchestration, immutable snapshots, integrity verification |
| `smoke.py` | Stage-A fast checks |

## Commands

```powershell
cd "D:\ai model"
python -m constructive.exp_i.runner --stage A        # 13 smoke checks
python -m constructive.exp_i.runner --quick          # fast 8-experiment pilot
python -m constructive.exp_i.runner                  # full campaign
python -m constructive.exp_i.runner --only I-A I-C   # selected experiments
python -m constructive.exp_i.runner --merge-only     # rebuild latest.json
python -m constructive.exp_i.runner --no-resume      # ignore cached results
python -m pytest tests/test_exp_i.py -q              # 64 EXP-I tests
python -m pytest -q                                  # full repo: 229 passed, 1 skipped
```

## Output contract

- `results/latest.json` — merged machine-readable result.
- `results/experiments/I*.json` — per-experiment current result.
- `results/experiment-history/I*-<stamp>.json` — **immutable** snapshots (never overwritten).
- `results/baseline_hashes.json` — SHA-256 of 22 protected files recorded at STAGE 0.
- `results/integrity.json` — re-verification at close-out.
- `results/*.txt` — test/campaign logs and metric extracts.

## Scientific protocol

Identical distributions and information budgets per comparison; independent
seeds; disjoint support/validation/test; test labels never used in generation,
candidate selection or calibration; skipped experiments remain `SKIPPED`;
timeouts are never recorded as success. Prefer ≥30 episodes per principal family
and ≥5 seeds for major comparisons (I-A/I-C use 30/60 episodes and ≥2 seeds;
I-D/I-G are smaller and their sample sizes are stated in the report).
