# EXP-011 — N1-Heldout: Same-Family Surface-Transfer Probe

- **Branch:** `research/exp-n` (from `research/exp-m` @ `21b93b4`)
- **Date:** 2026-11-19 06:15:54 UTC
- **Hardware:** Intel Core i5-1135G7 @ 2.40 GHz (8 logical CPUs), 15.66 GB RAM, Windows 11 (10.0.26200),
  Python 3.13.15, CPU-only
- **Entrypoint:** `python EXPERIMENTS/EXP-011-n1-heldout/run_exp.py` (wall **27.782 s**, peak RSS **228.44 MB**)
- **Suite:** `task_suite/v1`, **29 files**, manifest SHA-256
  `4cf5c03cb5a615f5d1bd06a3d462d2cfc820a49af4fc3adf43c5c3ba685875f1`, integrity **OK**
- **Implementation:** the same `RepairEngine` measured in `EXP-010`; `results.json` (recorded output),
  `results/history/` (earlier runs, preserved)

---

## 1. What this experiment is

`EXP-010` measured the repair engine on the six tasks it was designed against. A 6/6 result there is necessary but
not sufficient: the engine could be pattern-matching the exact N0 names and call shapes. `EXP-011` asks a narrower
question:

> Do the **frozen** operators still work when the names, domains, module layouts, argument names and test phrasing
> change?

This is a **same-family surface-transfer probe**, not an open-world benchmark and not a test of unseen bug families.
All seven tasks were authored **after** the nine-operator registry was frozen for the recorded `EXP-010` run. The
registry was not modified between the two runs or after the `EXP-011` run. The tasks were validated in both
directions before freezing: the buggy repository fails its own suite, and the sealed fix passes it.

**Pre-registered gate** (recorded before the run):

1. transfer rate at least **50%** (at least 4 of 7);
2. **zero** accepted patches whose tests were not executed;
3. the frozen scorer agrees with the engine on every task;
4. the oracle still scores 7/7.

**Interpretation rule, fixed before the run:** a failure is a coverage gap and is recorded as such. It is not repaired
by editing the operators and re-running this suite; that would convert the held-out probe into a development set and
require a new, unseen suite.

---

## 2. Suite design

| Task | Domain | Bug family | Why it is not an N0 copy |
|---|---|---|---|
| `n1-h-01-window-slice` | text windows | boundary + off-by-one | `page`→`window`, `page_number`→`page`, `page_size`→`size`; returns a slice of a string-like sequence |
| `n1-h-02-buffer-default` | append-only buffer | mutable default | `events=[]`→`buffer=[]`; tests require in-place mutation of an explicit list as well as default isolation |
| `n1-h-03-resilience` | retry client | exception scope | two exception types in a separate `faults.py`; the correct one is `RecoverableError`, not `TransientError` |
| `n1-h-04-memo-locale` | memoised formatting | cache key | cache key is the first parameter only; second parameter is a template |
| `n1-h-05-router-closure` | route table | closure binding | loop is `for path, handler_id in table.items()`, not a name/code pair |
| `n1-h-06-points` | point distribution | boundary + integer remainder | `total`/`buckets`; negative total must be rejected; remainder order is checked |
| `n1-h-07-bisect-bound` | binary search | comparison | `while low < high` must become `low <= high`; no N0 task uses `comparison_flip` |

Leakage controls are the same as `EXP-009`: sealed fixes live under `task_suite/v1/.sealed/`, only `repo/` and `tests/`
are copied to the solver, and `verify_integrity(strict=True)` runs before scoring.

---

## 3. Results

### 3.1 Aggregate

| Metric | Result |
|---|---|
| Tasks solved | **7 / 7** (pass rate **1.0**) |
| Transfer gate (≥ 50%) | **passed** |
| Verifier calls, total | **26** |
| Verifier calls per solved task | mean **3.71**, max **6** |
| Candidate executions | 12 (9 kept edits, 3 rejected) |
| Accepted patches without execution | **0** |
| Independent scorer agrees | **yes** (7/7) |
| Oracle | **7 / 7** (4.273 s) |
| Gate | **passed** (all four conditions) |
| Wall time | **27.782 s** |
| Peak RSS | **228.44 MB** |

### 3.2 Per task

| Task | Failed | Calls | Accepted operators | Rejected during search |
|---|---|---|---|---|
| `n1-h-01-window-slice` | 4 → 0 | 6 | `boundary_guard`, `off_by_one` | `off_by_one +1` (both operands) |
| `n1-h-02-buffer-default` | 2 → 0 | 3 | `mutable_default` | — |
| `n1-h-03-resilience` | 2 → 0 | 4 | `exception_scope` → `RecoverableError` | `BadRequest` (2 → 3) |
| `n1-h-04-memo-locale` | 1 → 0 | 3 | `cache_key` | — |
| `n1-h-05-router-closure` | 1 → 0 | 3 | `closure_binding` | — |
| `n1-h-06-points` | 3 → 0 | 4 | `boundary_guard`, `integer_distribution` | — |
| `n1-h-07-bisect-bound` | 1 → 0 | 3 | `comparison_flip` | — |

### 3.3 Operator coverage across the two N1 experiments

| Operator | EXP-010 (N0) | EXP-011 (held-out) |
|---|---|---|
| `mutable_default` | accepted | accepted |
| `exception_scope` | accepted | accepted |
| `cache_key` | accepted | accepted |
| `closure_binding` | accepted | accepted |
| `boundary_guard` | accepted | accepted |
| `off_by_one` | accepted | accepted |
| `integer_distribution` | accepted | accepted |
| `mapping_fallback` | accepted | not exercised |
| `comparison_flip` | not exercised | accepted |

All nine registered operators now have at least one executed, independently scored success across the two runs.
`mapping_fallback` has only one task of evidence; `comparison_flip` has only one. That is coverage, not robustness.

---

## 4. What these numbers mean — and what they do not

1. **The operators are not keyed to the N0 names.** Every task uses different module names, function names, parameter
   names and test phrasing, and 7/7 still pass. The `resilience` task is the strongest single piece of evidence: the
   correct exception is `RecoverableError`, the engine first tries `BadRequest`, observes a worse suite, reverts, and
   then finds `RecoverableError` from the test bodies alone.
2. **This is not unseen-family generalization.** The seven tasks were authored by the same agent that wrote the
   operators, with the operator list in hand, and each task was chosen because a frozen operator could solve it. A bug
   outside the nine families is still refused rather than repaired. The correct description is *same-family surface
   transfer*.
3. **The suite is now spent as a held-out set.** Because its results are known, further operator tuning against it
   would be training, not evaluation. Any future N1 claim must use a new unseen suite; this one becomes a regression
   check.
4. **No real repository is involved.** These are still synthetic single-package workspaces with short functions. The
   `EXP-010` README §4 limitations apply unchanged: `ast.unparse` formatting loss, no OS sandbox, one accepted edit
   per `(operator, file, target)`, and a suite run as the dominant cost.
5. **The gate was lenient by design.** A 50% transfer threshold was set because a 7-task probe cannot estimate a rate
   precisely. The observed 7/7 is evidence that the operators generalise across surface forms, not evidence of a 100%
   real-world repair rate.

---

## 5. Files

```
EXPERIMENTS/EXP-011-n1-heldout/
  README.md                  this record
  run_exp.py                 runner (integrity check, oracle, engine over the held-out suite)
  results.json               recorded run (2026-11-19 06:15:54 UTC)
  results/history/           earlier runs, preserved, never overwritten
  task_suite/v1/
    manifest.json            SHA-256 of all 29 files
    code/<task-id>/          task.json, repo/, tests/
    .sealed/solutions/       sealed fixes, never copied to the solver
```

## 6. Reproduction

```bash
python EXPERIMENTS/EXP-011-n1-heldout/run_exp.py
```

Expected: `solved 7/7`, `verifier_calls_total 26`, `accepted_without_execution 0`,
`independent_scorer_agrees true`, `gate.passed true`. Timing and RSS vary with the machine; the gate booleans and the
call counts are the reproducible part.

## 7. Change policy

The suite is frozen at `v1`. Any edit to a task, test, task statement or sealed fix bumps it to `v2`, invalidates this
result, and requires a fresh oracle and run. `results.json` is never edited by hand; a new run copies the old file
into `results/history/` with a UTC timestamp first. Because this suite has been measured, it may be used as a
regression check but not as a held-out set for a future N1 claim.
