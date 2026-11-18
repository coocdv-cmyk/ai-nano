# `constructive/exp_h/` — EXP-H research module

Open-world intelligence, concept discovery and architectural evolution, built
*on top of* the untouched Constructive Concept Model (CCM).  This package is
fully isolated from `constructive/` and from `omega_nano/`; it uses the public
interfaces of `constructive` read-only and never monkey-patches them.

## Hard facts

- **No external LLM. No GPU. CPU-first.**
- The eight experiments H1–H8 are implemented and measured on this machine
  (i5-1135G7, 4C/8T, Windows 11, Python 3.13.15, NumPy 2.5.3).
- Results go to `results/latest.json` with every run preserved under
  `results/experiment-history/<id>-<stamp>.json` — nothing is overwritten.
- Failure modes are **classified**, never collapsed (see `metrics.FAILURE_TAXONOMY`).
- A tiny, honest pilot configuration exists (`--quick`); the full campaign is
  stage C, bounded by per-fit budgets.

## Experiments

| ID | Name | Where |
|---|---|---|
| H1 | Open-world primitive and concept discovery | `h1_discovery.py` |
| H2 | Deep composition and reasoning scaling | `h2_scaling.py` |
| H3 | Structural OOD and compositional generalization | `h3_ood.py` |
| H4 | Large-scale continual learning and interference | `h4_continual.py` |
| H5 | Programming intelligence and AST transformation | `h5_ast.py` + `ast_engine.py` |
| H6 | Autonomous representation and abstraction discovery | `h6_abstraction.py` |
| H7 | Adversarial validation, identifiability, robustness | `h7_adversarial.py` |
| H8 | Computational efficiency and architectural alternatives | `h8_efficiency.py` + `lexical_h8.py` |

## Layout

- `config.py` — bounds, hardware probe, fingerprinting
- `profiling.py` — wall time / traced heap / node accounting
- `metrics.py` — aggregate stats, 95 % CIs, failure taxonomy
- `episode.py` — uniform fit-and-measure harness (CCM + baselines)
- `engine.py` — hidden-program language for targets (safe, step-bounded)
- `generators.py` — independent task generators (H1..H4 tasks)
- `h4_pool.py` — the parametrized 86-concept pool for continual learning
- `ast_engine.py` — safe AST interpreter, template library, repair search
- `h5_ast.py` — H5 levels L1..L5
- `lexical_h8.py` — independent search implementation (beam-elite / eager-prune arms)
- `smoke.py` — Stage A smoke tests
- `runner.py` — orchestration (stages A/B/C; resume; merge)
- `analysis.py` — curves, taxonomy tallies

## Run

```powershell
python -m constructive.exp_h.runner --stage A       # smoke tests
python -m constructive.exp_h.runner --quick          # pilot, fast
python -m constructive.exp_h.runner                  # full campaign
python -m constructive.exp_h.runner --only H1 H3     # selected experiments
python -m constructive.exp_h.runner --no-resume      # ignore cached episodes
```

Tests:

```powershell
python -m pytest tests/test_exp_h.py -v
python -m pytest -q          # must remain 145+ passed
```

## Results

See `results/latest.json` for the raw data, `CONSTRUCTIVE-EXP-H-REPORT.md` for the
analysis, and `RESEARCH-BASELINE.md` for what CCM actually implements vs what this module
adds. Headline measured outcomes (CPU, i5-1135G7, synthetic tasks):

| Experiment | Outcome |
|---|---|
| H1 open-world discovery | vocab 0.81 acc, competence at k=5; exclusion **0.007**, conditional **0.048** (vocabulary/representation limits); nearest-NN control 0.078 |
| H2 deep composition | depth ≤2 correct; depth ≥3 incidental acceptances; spurious rates up to 0.67 from depth 3 |
| H3 OOD | length 0.867, structure 0.967, numeric 0.728 (`sum_digits` breaks at OOD) |
| H4 continual | 90–92 % learn-acceptance; forgetting 0.07 mean drop; interference 13/157, all parametric collisions |
| H5 AST repair | 100 % found, 0.92–1.00 semantically correct, false-repair ≤ 0.08 |
| H6 macro-harvest | +0.16 accuracy vs baseline, 6.1× faster fit, 2.66× fewer search nodes |
| H7 adversarial | corruption localised 6/6 (acc 1.0); twin ambiguity exposed at k ≤ 3 |
| H8 alternatives | beam-elite & eager-prune won on 12 easy episodes but 3× more nodes; no efficiency claim |

Nothing here supersedes or integrates into OMEGA-NANO. See the report for full
limitations.
