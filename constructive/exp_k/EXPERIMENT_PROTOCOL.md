# EXP-K Experiment Protocol

## 1. Scientific rules

1. All generators, reference labels and RNGs are EXP-K originals.
2. Train, validation and test are disjoint by construction and checked by
   `datasets.split_integrity`.
3. Test labels are used for evaluation only, never for generation, candidate
   selection, threshold choice or calibration.
4. Every treatment arm has a matched baseline arm using the same constructors,
   seeds and configuration.
5. 10 independent seeds for principal experiments; quick mode (2 seeds) is a
   pilot only and is never reported as confirmatory.
6. Report n, mean, median, std, 95% CI, coverage, false acceptance and false
   rejection separately. Preserve negative results and failed runs.
7. Acceptance thresholds and detector thresholds are fixed before inspecting
   held-out results, using a separate calibration split where applicable.
8. Budgets are hard caps. A timeout is never a success.

## 2. Stage flow

| Stage | Question | Matched baseline |
|---|---|---|
| K0 | Reproduce the four EXP-J failures and establish integrity/resource baseline | (this is the baseline) |
| K1 | Why are incorrect composed rules accepted? | K0 deep baseline |
| K2 | Why does retrieval silently substitute? | K0 retrieval baseline |
| K3 | Why is drift missed? | K0 drift baseline |
| K4 | Why is confidence miscalibrated? | K0 calibration baseline |
| K5 | Do fixes damage other mechanisms? | Factorial 2×2 arms |
| K6 | Do fixes survive adversarial conditions? | Matched adversarial baseline |
| K7 | Full reproduction and integrity | Re-run of all stages |

## 3. Commands

```bash
python -m constructive.exp_k.runner --stage A
python -m constructive.exp_k.runner --stage K0 --quick     # pilot
python -m constructive.exp_k.runner --stage K0 --no-resume # principal 10-seed run
python -m constructive.exp_k.runner --merge-only
python -m pytest -q tests/test_exp_k.py
python -m pytest -q
```

## 4. K0 recorded configuration

10 seeds (0–9), 16 rule families, 5 shifts, 800 generated datasets (all splits
disjoint), 5 validation observations, search beam 24, node budget 12,000,
fit deadline 2.5 s, depths 2/3/4/5/8/12/16. CPU-only; no external APIs.

## 5. Publication protocol

Each verified stage is committed and pushed to `origin/research/exp-k` with a
descriptive message; the remote commit is independently verified with
`git ls-remote` before the next stage begins. No force push, reset or history
rewrite is used.
