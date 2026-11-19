# EXP-015 — N5: release gate

Pre-registration: `RESEARCH/EXP-N-PLAN.md` §7.

**Gate.** Every number in the release matrix resolves to a machine-readable
result file **and** a recorded command; no claim without a `results.json`.

## Instrument

`matrix.py` defines 19 claims across tracks M1, M2, N0, N1, N2, N3 and N4.
Each row is *extracted* from its source JSON via an explicit path — no number is
hand-typed — and carries the command that produced that file. `matrix.json`
records the integrity check and `matrix.md` renders the table.

`serve.py` is a loopback-only UI/API that calls the same implementations the
matrix measures: `encode_text` (N3), `MemoryFabric.recall` (N0/N3) and
`Engine.generate` with the shipped N2 default (non-speculative). It accepts
text, never file paths.

## Results

- **19/19 claims resolved**, integrity clean.
- Live reproducibility: N3 `recall@5 = 0.7333` re-run into a fresh temp dir
  reproduced **exactly** (`identical: true`). The other 18 commands are recorded
  but were not re-executed in this session.
- API: `2 passed` in `tests/test_exp_n5.py` — matrix integrity plus
  `/`, `/api/matrix`, `/api/encode`, `/api/retrieve`, `/api/generate`.

Cost: 8.312 s wall, 21.32 MB peak parent RSS (the live N3 child is accounted in
EXP-013 at 57.82 MB), 0 repair-verifier calls. Full record:
`runs/n5-v1/results.json` and `runs/n5-v1/matrix.json`.

## Negative results and limitations

- Historical numbers lacking a machine-readable result file are excluded from
  the matrix. The EXP-005 v2 figure was found in `showdown_results.json` (not
  `results.json`) and included from there.
- Only one claim was live re-run; "command recorded" is traceability, not proof
  that every historical experiment still reproduces on this machine.
- The API is loopback-only and deliberately narrow: it does not expose repair
  over arbitrary workspaces, because that would make the demo a different code
  path from the measured one.

## Files

| File | Role |
| --- | --- |
| `matrix.py` | provenance-checked capability/efficiency matrix + live repro |
| `serve.py` | local UI/API over the measured code paths |
| `runs/n5-v1/matrix.json` | 19 rows, integrity, cost, live repro |
| `runs/n5-v1/matrix.md` | rendered release table |
| `runs/n5-v1/results.json` | N5 gate verdict |
