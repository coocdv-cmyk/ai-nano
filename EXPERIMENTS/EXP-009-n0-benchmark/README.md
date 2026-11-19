# EXP-009 — N0: Sealed `EXP-N` Benchmark v1 (oracle + baseline)

- **Branch:** `research/exp-n`
- **Date:** 2026-11-19
- **Hardware:** Intel Core i5-1135G7 (8 logical CPUs), 15.66 GB RAM, Windows 11 (10.0.26200), Python 3.13.15, CPU-only
- **Entrypoint:** `python EXPERIMENTS/EXP-009-n0-benchmark/run_exp.py` (wall time **10.405 s**, peak RSS **232.47 MB**)
- **Suite:** `task_suite/v1`, **32 files**, manifest SHA-256
  `a16e73d315fe7a8ccf6c693c624d70d0100b90f590eac6cd8794bcdf7db75ac8`, integrity **OK**
- **Implementation:** `harness.py` (evaluator), `run_exp.py` (runner), `results.json` (recorded output),
  `results/history/` (earlier runs, preserved)

---

## 1. What this experiment is

`EXP-M` finished with `20/20` on author-designed sealed problems, all of which the symbolic families can express. That is
a strong result for a verifier-guided loop, but it does not say whether the system can touch an **open-world** task.
`EXP-009` is the measurement instrument for the whole `EXP-N` program: a small, frozen, contamination-controlled suite
whose only purpose is to be hard for today's system and stable for tomorrow's.

The suite is frozen **before** `N1` exists. Its scores are the floor that `N1`–`N4` must beat.

### Tracks

| Track | Content | Scoring |
|---|---|---|
| **C — repository repair** | 6 multi-file Python tasks; each ships a failing `pytest` suite and no reference fix in the solver's view | the task's own test suite passes after the candidate's edit |
| **R — verifiable reasoning** | 12 multi-step problems with exact answers (integer / reduced fraction) | SHA-256 of the canonical answer |
| **L — bilingual retrieval** | 36 short documents (18 English + 18 Arabic renderings of the same facts), 18 paraphrased queries | recall@5 and MRR against sealed labels; a topic-level view is reported separately |

### Leakage controls

- Reference fixes and answer labels live under `task_suite/v1/.sealed/`. `harness.prepare_workspace()` copies only
  `repo/` and `tests/`; `harness.assert_not_sealed()` guards every solve-time copy.
- Answers are stored **only as canonical SHA-256 digests** in the public files; the plaintext lives in `.sealed/`.
- `manifest.json` pins every file, including sealed artefacts. `verify_integrity()` runs before scoring and aborts on any
  mismatch.
- The seal is **methodological, not cryptographic**. The repository is public, so the guarantee is policy plus tooling:
  every training/corpus builder must exclude `EXPERIMENTS/EXP-009-n0-benchmark/`, and no solver path may read `.sealed/`
  (`harness.assert_not_sealed` enforces this in code).
- The suite must never enter a training corpus, replay buffer, skill library, or prompt. An edit invalidates the manifest
  and every baseline recorded against `v1`.

---

## 2. Oracle — proof that the suite is satisfiable

The oracle is the evaluator itself, using the sealed fixes/answers. If it is not 100%, no baseline number below means
anything.

| Track | Result | Wall time |
|---|---|---|
| Code repair | **6 / 6** | 3.581 s |
| Verifiable reasoning | **12 / 12** | included |
| Bilingual retrieval | **18 / 18** (recall@5 = 1.0, MRR = 1.0) | included |
| Overall | **`ok = true`** | 3.625 s |

Before freezing, each code task was also verified in the opposite direction: the buggy repository **fails** its own tests
(all 6 do), and the sealed fix **passes** them (all 6 do).

---

## 3. Baseline — what the product does today

The baseline is the real product, not a strawman: `OmegaAgent.handle()` for code and reasoning, `MemoryFabric` for
retrieval. The agent is given the task statement and the workspace path, and receives every numeric token it surfaces as
a reasoning candidate (best effort, last output first).

| Track | Baseline | Interpretation |
|---|---|---|
| Code repair | **0 / 6** | Every task routes to `workspace_agent`, tier `t1:rules`, and ends **blocked**. `files_changed = []` — the system correctly refuses to touch a workspace it does not understand. There is no patch operator and no traceback-driven localization yet. |
| Verifiable reasoning | **0 / 12** | Every problem routes to the durable workspace agent; no arithmetic/word-problem path exists outside the symbolic families. Candidate extraction was generous (up to 4 numeric tokens per task) and still found no correct answer. |
| Bilingual retrieval | **recall@5 = 0.6667**, **MRR = 0.625** (12/18) | Real, non-zero signal from the hashed encoder + BM25. The 6 misses return the **translation twin** of the target document (e.g. an Arabic query retrieves the Arabic document instead of its English counterpart). |
| Retrieval, topic view | **topic recall@5 = 1.0**, **topic MRR = 1.0** (18/18) | When the language of the document is not part of the question, the system finds the right fact every time. |

Cost: code track 3.823 s; code + reasoning wall 4.224 s (agent startup included); retrieval 0.048 s. Peak RSS
232.47 MB for the whole run.

---

## 4. What these numbers mean (and what they do not)

1. **The instrument works.** The oracle is 100%, integrity passes, and the two hard tracks score exactly zero for a
   structural reason that is visible in the logs (blocked route, no file changes). That is the correct starting point.
2. **`N1` has a clean target:** move code repair above 0/6 with zero unverified patches. The tasks were chosen to exercise
   six distinct bug families (off-by-one, mutable default, wrong exception scope, cache-key omission, closure late binding,
   integer remainder), all of which the `EXP-N` plan lists as AST operators.
3. **`N3` must not claim a retrieval win on this suite.** `topic recall@5` is already **1.0**: the hashed encoder saturates
   the topic-level task. The strict document-level score (0.6667) is depressed by translation twins, which is an artefact
   of the suite, not a capability gap. Therefore `v1` Track L is a **diagnostic, not a gate**: `N3` must author a harder
   held-out set (no translation twins, no shared surface tokens, near-duplicate facts) **before** it claims any semantic
   gain. This is recorded here, before any `N3` code exists.
4. **The reasoning baseline says nothing about the sealed `20/20`.** The `EXP-M` result came from structured problems with
   verifiers the solver could call. These 12 problems are plain text with no verifier scaffolding, and the product has no
   general arithmetic front end. Both statements are true at once.
5. **Cost is not yet the constraint.** A full evaluation cycle costs ~10 s and ~232 MB on an ordinary laptop. The binding
   constraint is capability, not throughput.

---

## 5. Files

```
EXPERIMENTS/EXP-009-n0-benchmark/
  README.md                  this pre-registration + result record
  harness.py                 suite loader, integrity check, verifiers, scoring
  run_exp.py                 oracle + baseline runner (writes results.json)
  results.json               recorded run (2026-11-19)
  results/history/           earlier runs, preserved, never overwritten
  task_suite/v1/
    manifest.json            SHA-256 of all 32 files
    code/<task-id>/          task.json, repo/, tests/
    reasoning.json           statements + canonical answer digests
    retrieval/               docs.json, queries.json
    .sealed/                 solutions/ (6), reasoning_answers.json, labels.json
```

## 6. Reproduction

```bash
python EXPERIMENTS/EXP-009-n0-benchmark/run_exp.py            # oracle + baseline
python EXPERIMENTS/EXP-009-n0-benchmark/run_exp.py --oracle-only
python EXPERIMENTS/EXP-009-n0-benchmark/run_exp.py --baseline-only
```

Expected: `oracle ok=True code=6/6`, `baseline code=0/6 reasoning=0/12 recall@5=0.6667 topic_recall@5=1.0`.

## 7. Change policy

Any modification to a task, a test, a document, a query, a label, or a sealed solution bumps the suite to `v2` and
invalidates this baseline. The old `results.json` is copied to `results/history/` with a UTC timestamp before a new run is
written. Historical numbers are never edited.
