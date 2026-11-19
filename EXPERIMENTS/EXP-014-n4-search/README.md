# EXP-014 — N4: value-guided search over verified candidates

Pre-registration: `RESEARCH/EXP-N-PLAN.md` §6.

**Hypothesis.** Value-guided branching beats the linear `attempt 1..N` loop on
multi-function synthesis at equal verifier cost. Gate: higher pass rate at
equal budget, or equal pass rate at ≤ 60% of the verifier calls. Kill: if it
only wins by spending more calls, the linear loop stays.

## Instrument

`tasks.py` composes six **three-function** workspaces from fresh N4 themes
(three train, three held-out). Each has three localized targets and 4–7
candidate patches. `run_exp.py` applies the same candidate generator, the same
verifier and the same hard budget of **12 verifier calls** to both arms:

- `linear` — generator/priority order (the N1 engine's loop);
- `value` — a logistic ranker over static candidate features (operator,
  priority, localization hits, diff size, file length), fitted **only** from
  the linear arm's verified decisions on the train split. Ranking spends zero
  verifier calls.

N0 is cited only; none of its tasks entered this experiment.

## Results

| Arm | pass rate | mean verifier calls | attempts to first success |
| --- | --- | --- | --- |
| linear | 1.000 | 9.667 | 8.667 |
| value | 1.000 | 7.000 | 6.000 |

Value-guided ordering used **72.4%** of the linear arm's calls — a 27.6%
reduction — while keeping the pass rate at 100%. The pre-registered 60% gate
is therefore **not met**; the kill criterion is also not triggered (it wins by
spending *fewer* calls). Verdict: linear stays the default; value-guided
ordering is recorded as a promising direction that needs more training traces.

Cost: 42.262 s wall, 70.11 MB peak RSS, 78 verifier calls total, 0 unverified
edits. Command is recorded in `results.json`; fitted weights in
`runs/n4-v1/value_weights.npz`.

## Negative results and instrument defects (recorded, not deleted)

- Fresh `off_by_one` N4 variants are not repaired by the current operator;
  composing them made every task unsolvable and produced a vacuous 0%/0% gate.
  They were excluded from composition.
- An earlier run capped edits at 3 and reused N1's coarse dedup key
  `(operator, file, target)`, which skipped distinct patches for the same
  function; both arms scored 0%. Fixed to a 6-edit cap and a source-hash key.
- The final result is a **near miss**, not a success: 72.4% > 60%.

## Files

| File | Role |
| --- | --- |
| `tasks.py` | fresh three-function task composer |
| `run_exp.py` | equal-budget linear vs value search |
| `runs/n4-v1/results.json` | verdict, per-task calls, cost, negative results |
