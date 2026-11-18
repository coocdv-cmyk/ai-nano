# OMEGA EXP-J — Independent Transfer Validation Campaign

Isolated research package. Nothing here is integrated into CCM or
`omega_nano`. EXP-I source, results and snapshots are untouched and verified by
SHA-256 at every run.

## Quick start

```powershell
cd "D:\ai model"
python -m constructive.exp_j.runner --stage A        # invariants smoke (8 checks)
python -m constructive.exp_j.runner --quick          # 2-seed pilot
python -m constructive.exp_j.runner --no-resume      # full 10-seed campaign
python -m constructive.exp_j.runner --merge-only     # rebuild latest.json
python -m pytest tests/test_exp_j.py -q              # 35 apparatus tests
python -m pytest -q                                   # full suite (264 passed, 1 skipped)
python -c "from constructive.exp_j.runner import verify_integrity; print(verify_integrity())"
```

## What is measured

| ID | Question | Principal output |
|---|---|---|
| J1 | Does EXP-I validation/retrieval transfer to new rule families and shifted distributions? | wrong acceptance vs independent baseline, coverage, calibration, retrieval P/R |
| J2 | How does the engine behave under noise, corruption and drift? | accuracy–coverage, state discrimination, detection delay, recovery |
| J3 | Why does deep composition still accept spurious programs? | six-way outcome classification by depth, expansions/time per strategy |
| J4 | Do abstractions pay for themselves? | total lifecycle cost for five arms on two distributions |
| J5 | Do combined mechanisms regress? | controlled ablations with explicit regression list |
| J6 | What does it cost computationally? | wall time, RSS, expansions, memory/concept, scaling |

## Scientific policy

- 12 original rule families; train/validation/test disjoint by construction.
- Test labels are never used for training, selection or calibration.
- 10 independent seeds for principal evaluations; raw episodes preserved.
- No fabricated measurements; negative results are reported as such.
- No AGI, natural-language intelligence, universal-generalization or
  system-level efficiency claims.

See `ARCHITECTURE.md` for the module map and `../../EXP-J-REPORT.md` for the
measured findings and integration-readiness assessment.
