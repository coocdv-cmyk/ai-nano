# EXP-K Reproducibility

## Environment

- CPU-only Windows 11, Python 3.10+ (developed on Python 3.13.15), NumPy optional.
- No external APIs and no network access required by the research runners.
- Exact hardware, Python version, config fingerprint, source hashes of the
  read-only systems under test and the git commit are recorded in every result
  block under `reproducibility`.

## From a fresh clone

```bash
git clone https://github.com/coocdv-cmyk/ai-nano.git
cd ai-nano
git switch research/exp-k
python -m pytest -q
python -m constructive.exp_k.runner --stage A

python -m constructive.exp_k.runner --stage K0 --no-resume
python -m constructive.exp_k.runner --stage K1 --no-resume
python -m constructive.exp_k.runner --stage K2 --no-resume
python -m constructive.exp_k.runner --stage K3 --no-resume
python -m constructive.exp_k.runner --stage K4 --no-resume
python -m constructive.exp_k.runner --stage K5 --no-resume
python -m constructive.exp_k.runner --stage K6 --no-resume
python -m constructive.exp_k.runner --stage K7 --no-resume
python -m constructive.exp_k.runner --merge-only
```

Use `--quick` only for a pilot. Quick runs write `*-quick.json` files and
**never** overwrite the principal `K*.json` results.

## Determinism

All generators use `random.Random(seed * constant + ...)`; no `hash()` of strings
enters data generation. K7 independently reran K2, K3 and K4 with the full
configuration and confirmed identical aggregate metrics and run counts. Result
files contain wall-clock timestamps, so byte-for-byte equality is not claimed.

## Manifests and snapshots

- `results/manifests/k0_dataset_manifest.json` — 800 generated datasets with
  disjoint train/validation/test splits and a dataset fingerprint.
- `results/history/K*-<stamp>.json` — immutable snapshots of every run,
  including a failed K6 diagnostic snapshot that was preserved and disclosed.
- `results/integrity.json` — re-verification of the EXP-I (22) and EXP-J (84)
  protected-file SHA-256 manifests, 106 files total.
- `results/latest.json` — merged view of all stages.

## Known constraints

- One K6 run failed with a `abs()`-on-list generator bug; the code was fixed and
  the failed snapshot retained.
- K5's OOD/drift-source factorial cells are saturated and uninformative; they
  are preserved in raw results and must not be quoted as positive evidence.
- No natural-input validation was performed. Results are benchmark-local.
