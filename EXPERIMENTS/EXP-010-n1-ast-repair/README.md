# EXP-010 — N1: Execution-Verified AST Repair

- **Branch:** `research/exp-n` (from `research/exp-m` @ `21b93b4`)
- **Date:** 2026-11-19 06:15:27 UTC
- **Hardware:** Intel Core i5-1135G7 @ 2.40 GHz (8 logical CPUs), 15.66 GB RAM, Windows 11 (10.0.26200),
  Python 3.13.15, CPU-only
- **Entrypoint:** `python EXPERIMENTS/EXP-010-n1-ast-repair/run_exp.py` (wall **25.241 s**, peak RSS **228.08 MB**)
- **Suite:** frozen N0 `task_suite/v1`, **32 files**, manifest SHA-256
  `a16e73d315fe7a8ccf6c693c624d70d0100b90f590eac6cd8794bcdf7db75ac8`, integrity **OK**
- **Implementation:** `omega_nano/brain/repair.py` (engine), `omega_nano/agent.py` (`repair_workspace`),
  `omega_nano/cli.py` (`brain repair`), `tests/test_exp_n1_repair.py`, `results.json` (recorded output),
  `results/history/` (earlier runs, preserved)

---

## 1. What this experiment is

`N1` tests the first open-world capability claim of `EXP-N`: can the system repair a multi-file Python repository
from **its failing test suite alone**, without reading a reference solution and without ever accepting an edit that
was not executed?

The six tasks are the frozen N0 code track (`EXP-009`), authored before any repair code existed. The product baseline
on those tasks is **0/6**: the agent routes to `workspace_agent`, changes no files, and ends `blocked`. N1 must move
that number with zero unverified patches.

**Pre-registered gate** (`RESEARCH/EXP-N-PLAN.md` §3):

1. code pass rate strictly above the N0 baseline (0/6);
2. **zero** accepted patches whose tests were not executed;
3. the frozen scorer agrees with the engine on every task;
4. the oracle still scores 6/6.

**Kill criterion:** if three iterations of the operator set leave the pass rate at the baseline, the mutation
approach is declared insufficient and `N4` must carry the track instead. (One iteration was enough; the criterion was
not triggered.)

---

## 2. Method

### 2.1 Localisation

1. Run `pytest -q tests` in the workspace, with a sanitised environment (no inherited secrets) and a timeout.
2. Parse the `FAILED`/`ERROR` lines and the full traceback.
3. Index the repository with `RepoGraph` (AST symbols + call edges).
4. Parse each failing test body and collect its calls, literal integer arguments, and any `pytest.raises(...)` scope.
5. Rank symbols by: called directly by a failing test (+10), named in an assertion message (+2), traceback frame
   (+20), one-hop callee of a direct target (+3).

Only the repository's own AST and the suite's observed behaviour are used. The sealed solutions directory is never
opened. `harness.prepare_workspace(..., oracle=False)` copies only `repo/` and `tests/`.

### 2.2 Operators

Nine AST transformations are registered, all deterministic and syntax-preserving at the tree level:

| Operator | What it infers from the tests |
|---|---|
| `mutable_default` | a mutable default argument must become `None` + a fresh value inside the body |
| `exception_scope` | a broad `except Exception` must be narrowed to a workspace exception type the tests exercise |
| `cache_key` | a memo key omits one or more parameters of the function |
| `closure_binding` | a lambda in a loop captures a late-bound loop variable |
| `boundary_guard` | an argument value that only appears inside `pytest.raises(...)` needs a guard |
| `off_by_one` | index arithmetic in a slice is shifted by ±1 |
| `integer_distribution` | `total // n` parts must distribute the remainder one unit at a time |
| `mapping_fallback` | a dict lookup must use the mapping's declared fallback value |
| `comparison_flip` | one control-flow comparison is wrong (`<`↔`<=`, `==`↔`!=`, …) |

The operators are the pre-registered N1 families (off-by-one, comparison, default-argument, closure-binding,
cache-key, rounding/remainder) plus the boundary-guard, exception-scope and mapping-fallback families the N0 tasks
require. `comparison_flip` is registered but is not exercised by any N0 task; it is exercised by the held-out probe
(`EXP-011`).

### 2.3 Search and the verification invariant

- Best-first over candidates ordered by operator priority and localisation rank.
- A candidate is written into an **isolated copy** of the workspace and the workspace's own suite is run.
- A run that passes is re-run once to confirm (guards against flaky tests).
- A run that strictly reduces the failing-test count is kept as the new base and the search continues.
- A run that does not improve is reverted immediately.
- The real workspace is written **only** after the suite passes; otherwise it is untouched.
- `accepted_without_execution` cross-checks every kept edit against the recorded attempt list and must be `0`.

Budget: `max_verifier_calls = 80`, `max_depth = 3`. Actual use was 24 calls in total.

---

## 3. Results

### 3.1 Aggregate

| Metric | Result |
|---|---|
| Tasks solved | **6 / 6** (pass rate **1.0**) |
| Product baseline (EXP-009) | **0 / 6** |
| Verifier calls, total | **24** |
| Verifier calls per solved task | mean **4.0**, max **6** |
| Candidate executions | 12 (9 kept edits, 3 rejected) |
| Accepted patches without execution | **0** |
| Independent scorer agrees | **yes** (6/6) |
| Oracle | **6 / 6** (3.875 s) |
| Gate | **passed** (all four conditions) |
| Wall time | **25.241 s** |
| Peak RSS | **228.08 MB** |

### 3.2 Per task

| Task | Bug family | Failed | Calls | Accepted operators | Rejected during search |
|---|---|---|---|---|---|
| `n0-code-01-pagination` | off-by-one + boundary | 5 → 0 | 6 | `boundary_guard`, `off_by_one` | `off_by_one +1` (both operands) |
| `n0-code-02-audit-default` | mutable default | 2 → 0 | 3 | `mutable_default` | — |
| `n0-code-03-retry-policy` | exception scope | 2 → 0 | 4 | `exception_scope` → `TransientError` | `FatalError` widening (2 → 3) |
| `n0-code-04-cache-key` | cache key + mapping fallback | 2 → 0 | 4 | `cache_key`, `mapping_fallback` | — |
| `n0-code-05-closure` | closure late binding | 2 → 0 | 3 | `closure_binding` | — |
| `n0-code-06-money-split` | boundary + integer remainder | 3 → 0 | 4 | `boundary_guard`, `integer_distribution` | — |

The `retry-policy` trace is the clearest evidence that the search is driven by execution, not by naming: the engine
first narrows the handler to `FatalError`, the suite gets **worse** (3 failures), the edit is reverted, and the next
candidate — `TransientError` — passes. The sealed fix was never read.

### 3.3 Operator coverage on this suite

Exercised and accepted: `boundary_guard`, `off_by_one`, `mutable_default`, `exception_scope`, `cache_key`,
`closure_binding`, `integer_distribution`, `mapping_fallback`.
Not exercised here: `comparison_flip` (no N0 task has a broken comparison; see `EXP-011`).

---

## 4. What these numbers mean — and what they do not

1. **The N1 gate is met, exactly as pre-registered.** The code track moves from 0/6 to 6/6 with zero unexecuted
   patches, and an independent scorer re-runs the suite after the engine returns.
2. **This is not open-world repository repair.** The six tasks are synthetic multi-file workspaces authored for N0,
   with short functions and one or two bugs each. The operator set was designed with those bug families in mind. A
   real repository has imports the indexer does not resolve dynamically, generated code, conftest fixtures, and bugs
   outside the nine families. Localisation reads the failing test body and traceback frames; a bug reached only through
   a test helper that is neither called directly by the test body nor visible in a traceback frame may not be ranked
   highly enough to patch. `EXP-011` measures same-family surface transfer only; neither experiment supports an
   open-world claim.
3. **The engine edits files with `ast.unparse`.** Comments and blank-line layout inside a rewritten file are not
   preserved, and string quoting may change. Semantics are preserved by construction and proven by the suite, but a
   patch is not merge-ready without review.
4. **The engine executes arbitrary code.** It runs the workspace's own tests with the caller's privileges. It is not
   an OS sandbox (repository defect #4). The isolated-copy rule protects the source tree from failed candidates; it
   does not protect the machine from hostile tests.
5. **Cost is modest but not free.** 24 verifier calls for six tasks means roughly four suite executions per solved
   task. On a larger repository, one suite run is the dominant cost; this experiment says nothing about scaling to a
   10-minute suite.
6. **A single accepted edit per function is a real limitation.** Search de-duplicates by `(operator, file, target)`, so
   a second bug fix from the same operator family in the same function is not attempted after the first is accepted.
   None of these tasks needs it; a real bug might.

---

## 5. Files

```
EXPERIMENTS/EXP-010-n1-ast-repair/
  README.md                  this record
  run_exp.py                 runner (integrity check, oracle, engine over the N0 code track)
  results.json               recorded run (2026-11-19 06:15:27 UTC)
  results/history/           earlier runs, preserved, never overwritten
```

Product code (shared with the rest of the repository, not experiment-local):

```
omega_nano/brain/repair.py   RepairEngine, operators, localisation, search, invariant
omega_nano/agent.py          OmegaAgent.repair_workspace
omega_nano/cli.py            python -m omega_nano brain repair --path ... --tests ... --max-calls ... --dry-run
tests/test_exp_n1_repair.py  6 tests: registry, parser, fresh workspace, refusal, green workspace, recorded gate
```

## 6. Reproduction

```bash
python EXPERIMENTS/EXP-010-n1-ast-repair/run_exp.py
python EXPERIMENTS/EXP-010-n1-ast-repair/run_exp.py --task n0-code-04-cache-key
python EXPERIMENTS/EXP-010-n1-ast-repair/run_exp.py --max-calls 40 --max-depth 2

# product path on any workspace with a failing suite
python -m omega_nano brain repair --path <workspace> --tests tests --max-calls 80
```

Expected: `solved 6/6`, `verifier_calls_total 24`, `accepted_without_execution 0`,
`independent_scorer_agrees true`, `gate.passed true`. Timing and RSS vary with the machine; the gate booleans and the
call counts are the reproducible part.

## 7. Change policy

The N0 suite is frozen. If any N0 task, test or sealed file changes, the suite becomes `v2`, this result is no longer
comparable, and a new baseline and N1 run are required. `results.json` is never edited by hand; a new run copies the
old file into `results/history/` with a UTC timestamp first. The operator set is part of the product
(`omega_nano/brain/repair.py`); changing it after this record does not change this result, but any later claim must
name the operator set it was measured with.
