# Contributing to OMEGA-NANO

Thank you for helping continue this research. The project values reproducibility and historical integrity above
convenience: a result that cannot be reproduced, or a history that has been rewritten, is worse than no result.

## Ground rules

1. **Never modify protected files or historical results.** This includes:
   - the original core (`omega_nano/`, original `constructive/*.py`),
   - `constructive/exp_h/`, `constructive/exp_i/`, `constructive/exp_j/` source once an experiment is complete,
   - all `results/latest.json`, immutable `results/experiment-history/` snapshots, baseline hash manifests and
     SHA-256 records,
   - existing tests and final reports.
2. **Never rewrite history.** No `git push --force`, `git reset --hard`, rebasing published commits, or squashing.
3. **No secrets.** Never commit credentials, tokens, private keys, `.env` files with real values, local absolute paths
   that identify you, or machine-specific private data. The destination repository is public.
4. **No unsupported claims.** Do not describe the project as AGI, proven general intelligence, or a replacement for
   frontier models. Experimental results are valid only for the benchmarks and machines on which they were measured.

## Working on a change

### 1. Create an isolated branch

```bash
git switch -c exp/<your-topic>
```

Keep each experiment or fix on its own branch. Do not commit experiment code into the protected core.

### 2. Adding a new experiment

- Put all new code under a new isolated package such as `constructive/exp_k/`.
- Put new tests in `tests/test_exp_k.py`, results in `constructive/exp_k/results/`, and write a final report plus an
  architecture note.
- Import protected modules **read-only**. If you need a different behavior, implement it in your own package rather
  than editing the original.
- Record a STAGE-0 protected-file hash manifest before you begin, and verify it at close-out.
- Preserve raw per-episode data, not just aggregates. Report negative results and explicit completion states
  (including timeouts); never convert a timeout into a success.

### 3. Record reproducibility information

For every principal experiment, record:

- exact seeds and configuration (config fingerprint),
- generator/source fingerprints (including hashes of any module used as the system under test),
- hardware and software environment,
- train / validation / test separation,
- sample counts, mean, median, standard deviation and confidence intervals where appropriate,
- the exact command used to run the experiment,
- known limitations and failures.

### 4. Run tests before submitting

```bash
python -m pytest -q
python -m pytest tests/test_exp_k.py -q          # your new tests
python -m constructive.exp_k.runner --stage A    # if your package has a smoke stage
python -c "from constructive.exp_j.runner import verify_integrity; print(verify_integrity())"
```

A change is not ready if any test fails or if any protected hash differs from its baseline. If a protected hash changes,
stop, find out why, and do not submit.

### 5. Submitting

Commit with a clear message describing what changed and what evidence supports it. Push your branch normally (never
force). Open a pull request that links the relevant report and test output. Report failures honestly; a well-documented
negative result is a valid contribution.

## Licence note

No licence file has been authorized. Until the owner explicitly selects one, default copyright applies and the code may
not be reused or redistributed by others. See `LICENSE-AND-PROVENANCE-REGISTER.md`.
