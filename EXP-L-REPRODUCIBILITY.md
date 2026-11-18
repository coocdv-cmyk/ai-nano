# EXP-L Reproducibility

## Environment

CPU-only Windows 11, Python 3.13.15. NumPy/SciPy/pytest/psutil are installed but
no external API, network service, GPU, or paid service is required. All core
experiments use explicit SHA-256-derived random generators; Python's randomized
string hash is never used as experiment randomness.

## From a clean clone

```bash
git clone https://github.com/coocdv-cmyk/ai-nano.git
cd ai-nano
git switch research/exp-l

python -m pytest -q
python -m pytest tests/test_exp_l.py -q

python -m constructive.exp_l.runner --stage L0 --no-resume
python -m constructive.exp_l.runner --stage L0 --merge-only
python -m constructive.exp_l.runner --stage L0 --quick   # pilot only; writes *-quick.json

python -m constructive.exp_l.runner --stage L1 --no-resume
python -m constructive.exp_l.runner --stage L2 --no-resume
python -m constructive.exp_l.runner --stage L3 --no-resume
python -m constructive.exp_l.runner --stage L4 --no-resume
python -m constructive.exp_l.runner --stage L5 --no-resume
python -m constructive.exp_l.runner --stage L6 --no-resume
python -m constructive.exp_l.runner --stage L7 --no-resume
python -m constructive.exp_l.runner --stage L6 --merge-only
```

`--quick` results are written to separate `*-quick.json` artifacts and snapshots
and never overwrite principal results. `--seed N` restricts a run to a single
seed for debugging.

## Determinism

Aggregate results are reproducible from `(seed, generator version, config,
dataset fingerprint)`. Timestamps and measured wall-clock times may differ
between runs; byte-for-byte equality of result files is not claimed. The L0
smoke fixture is verified reproducible both within a process and across
independent Python processes.

## Integrity

`constructive/exp_l/results/integrity.json` is an **additive** manifest; it never
replaces the EXP-I (22-file), EXP-J (84-file), or EXP-K manifests. Protected
source hashes and the EXP-K K0 dataset fingerprint are re-verified at every
milestone. Immutable snapshots are stored under
`constructive/exp_l/results/history/`; failed diagnostic runs are preserved.

## Known suite side effects

The pre-existing test suite regenerates some EXP-J/EXP-K quick-run JSON result
files and untracked smoke snapshots when executed. These are restored to the
committed state after a milestone run and are never staged as EXP-L work.
