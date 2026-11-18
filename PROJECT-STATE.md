# Project State — resume here

Last updated: 2026-11-18 (Phase 0 audit + research design session).
Read this first if you are a new agent or a returning human. It is deliberately short; the other documents carry the
detail.

---

## 1. In one paragraph

OMEGA-NANO has a **working, tested, durable agent foundation** and a **measured but weak language core**. The control
plane (planning, tools, verification, memory, permissions, crash recovery) is real and passes 126 tests. The neural core
is a 3.34 M-parameter byte-level decoder that reaches 1.8944 bits/byte (fp32) or 2.0554 (ternary) on a 20 MB Python-source
corpus — worse than zlib's 1.8296. Decode speed is 1,425 B/s single-stream / 5,027 B/s aggregate on the current laptop.
The corpus is **licence-contaminated** (LGPL code from `site-packages`), so neither it nor the checkpoints may be
redistributed. The project's next move is not to scale the model; it is to test whether a **Skill Fabric** can acquire
reusable skills from three examples, which is cheap, falsifiable, and CPU-native.

---

## 2. What works, what does not

| Capability | State | Evidence |
|---|---|---|
| Durable task engine, write-ahead checkpoints, crash recovery | **works** | `tests/`, 126 passed |
| Independent verification against the world (hashes, file contents) | **works** | `verify.py`, tests |
| Permission modes, scopes, approvals, emergency stop | **works** | `security.py`, tests |
| Bilingual rule reasoner (EN/AR) | **works** for fixed intents | `reasoner.py`, tests |
| Tool layer (10 tools) | **works** | `python -m omega_nano tools` |
| Memory fabric (lexical + int8 vector + fuzzy, provenance, versioning) | **works**, measured to 50k | `BENCHMARKS/results/memory.json` (different machine) |
| Native CPU decode engine (Numba, KV cache, batch) | **works** | EXP-003; fresh spot-check |
| Ternary training and inference | **works**, costs 0.16 bpb | EXP-003 |
| Language-model quality | **weak** — below zlib | EXP-003 |
| Adaptive reasoning quality | **unmeasured** | no benchmark |
| Software-engineering capability | **absent** (Phase 4) | no sandbox, no repo index |
| Skill acquisition / continual learning | **absent** | this package proposes the first test |
| Verified reasoning (Lean/Z3) | **absent** | proposed as VW-1 |
| Multimodal, computer-use, multi-agent | **absent** | not started |

---

## 3. Numbers you may quote, and how

Always attach the qualifier. Never quote a bare number.

| Number | Exact meaning | Machine | Source |
|---|---|---|---|
| **1.8944 bpb** | fp32 core, held-out val, 20 MB Python corpus | i7-7820HQ | EXP-003 |
| **2.0554 bpb** | ternary core, same split, same budget | i7-7820HQ | EXP-003 |
| **1.8296 / 1.4657 / 1.4341** | zlib-9 / bz2-9 / lzma-extreme on the same split | — | EXP-003 |
| **1,425 B/s** | ternary, **single-stream**, end-to-end decode incl. re-encode, 4 threads, n=400 | i5-1135G7 (fresh) | audit §5.5 |
| **5,027 B/s** | ternary, **aggregate, batch 8**, same settings | i5-1135G7 (fresh) | audit §5.5 |
| **841 / 2,865** | ternary, single (1 thread) / aggregate (batch 14), n=2000 | i7-7820HQ | EXP-003 |
| **3.2 MB** | ternary block weights (runtime); checkpoints are 12.8 MB because they store fp32 masters | — | EXP-003 |
| **126 passed, 1 skipped** | test suite, 16.54 s | i5-1135G7 (fresh) | audit §5.1 |
| **10k tok/s** | **ambition**, not a measurement; single-stream target | — | RC-001, constitution §5.2 |

**Never** say "5,027 tokens/second" without "aggregate, batch 8". **Never** compare 5,027 to 10,000; they are different
metrics. **Never** present a cross-machine ratio as a speedup.

---

## 4. Confirmed defects (do not rediscover these)

| # | Defect | Where | Severity |
|---|---|---|---|
| 1 | `fs.write` declares `idempotent=True` but `mode="append"` is not; retries duplicate content (reproduced: 3× `xxx`) | `tools/fs.py:125` | High (data corruption) |
| 2 | `LocalLLMReasoner.revise` puts raw tool output into the prompt — injection path | `reasoner.py:~295` | High (security) |
| 3 | `shell.run` inherits `os.environ`; allowlisted interpreters can read secrets | `tools/shell.py` | High |
| 4 | `shell.run` is not an OS sandbox | `tools/shell.py` | High |
| 5 | Audit log is not hash-chained; tamper-evident only by convention | `store.py` | Medium |
| 6 | Packaging omits `omega_nano.core` and the `torch` extra | `pyproject.toml` | Medium |
| 7 | Version mismatch: `pyproject` 0.1.0 vs `__init__` 0.2.0 | two files | Low |
| 8 | No CI; no automated gate for tests, the corpus manifest, or SPDX headers | repo | High |
| 9 | Zero git commits; all files untracked, so there is no authorship record | repo | High (provenance) |
| 10 | Corpus includes `site-packages`; contains LGPL-2.1 / Apache-2.0 | `core/data.py:26–52` | **Critical (legal)** |

Details and reproductions: `PHASE-0-REPOSITORY-AUDIT.md` §8.

---

## 5. Decisions that need the owner

These are outside an agent's authority. Nothing downstream can be published until 1 and 2 are resolved.

1. **Choose a project licence.** There is no `LICENSE` file, so the default is all rights reserved. Apache-2.0 is the
   natural fit for a research/engineering project that may be commercialised; AGPL-3.0 fits "local-first, user-controlled"
   but deters adoption. See the register §2.
2. **Approve the corpus rebuild.** It changes `core/data.py`, which is production code, and invalidates EXP-003's
   comparability (the old numbers must be kept as history).
3. **Approve the first git commit.** 0 commits means no authorship record; every file is untracked.
4. **Decide whether a pretrained local model may be used.** That would change Track E and the research narrative; it is a
   strategy decision, not a technical one.

---

## 6. What was done in this session

Documentation only. **No production code was modified, no experiment was run beyond the audit's read-only checks, nothing
was committed.**

Created:

- `PHASE-0-REPOSITORY-AUDIT.md` — factual audit, confirmed defects, maturity by layer, reproduction commands.
- `RESEARCH-OPPORTUNITIES.md` — 15-family portfolio, evidence grades, three opportunity cards, rejected directions.
- `ORIGINAL-LEARNING-MECHANISM.md` — Skill Fabric / BIRS: definitions, algorithm, admission tests, novelty, P1–P4.
- `OMEGA-EXPERIMENT-001.md` — pre-registered EXP-005 (not implemented).
- `RESEARCH-ROADMAP.md` — stages R0–R5, V0–V2, K0–K3, S0, E1, with gates and kill criteria.
- `BASELINE-AND-BENCHMARK-PLAN.md` — metric definitions, quality floors, banned claims, current ledger.
- `LICENSE-AND-PROVENANCE-REGISTER.md` — licence status, dependency inventory, corpus contamination, remediation.
- `PROJECT-STATE.md` — this file.

Updated:

- `ARCHITECTURE.md` — corrected to 0.2.0-dev, fixed the stale neural-core row, added data flows, module map, evidence map,
  and a companion-document list. This also satisfies the requested `CURRENT-ARCHITECTURE-MAP.md` deliverable without
  creating a duplicate.

---

## 7. The next single action

> **Implement and run EXP-005** (`OMEGA-EXPERIMENT-001.md`), after the owner resolves the corpus/licence decision.

If the owner has not decided, the next action is instead:

> **Create the first git commit** and a CI skeleton, because every subsequent result needs a code identity.

Rationale: EXP-005 is < 5 minutes of compute and can kill or confirm the project's central research bet. Nothing else on
the roadmap has that ratio of information to cost.

---

## 8. Resume commands

```bash
# confirm the environment
python -m omega_nano doctor
python -m omega_nano tools

# regression
python -m pytest -q                    # expect 126 passed, 1 skipped

# recorded benchmarks (different machine; do not compare as a ratio)
python BENCHMARKS/bench_engine.py
python BENCHMARKS/bench_memory.py --sizes 1000 10000 50000
python EXPERIMENTS/EXP-001-ternary-cpu/ternary_bench.py

# core quality (requires torch)
python EXPERIMENTS/EXP-003-omega-core-v0/evaluate.py

# state
git status --porcelain                 # expect everything untracked, 0 commits
```

State lives in `./.omega/` (override with `--home` or `OMEGA_HOME`). Tools are confined to `./.omega/workspace/`.

---

## 9. Document map

| Question | Document |
|---|---|
| What exists and what is broken? | `PHASE-0-REPOSITORY-AUDIT.md` |
| How does the system work? | `ARCHITECTURE.md` |
| What should we research and why? | `RESEARCH-OPPORTUNITIES.md` |
| What is the new learning mechanism? | `ORIGINAL-LEARNING-MECHANISM.md` |
| What is the first experiment? | `OMEGA-EXPERIMENT-001.md` |
| In what order, with which gates? | `RESEARCH-ROADMAP.md` |
| How do we measure anything? | `BASELINE-AND-BENCHMARK-PLAN.md` |
| What can we legally publish? | `LICENSE-AND-PROVENANCE-REGISTER.md` |
| Where do I resume? | this file |
| What is the mission? | `OMEGA-NANO-VISION-V3-ENGLISH.md` |

---

## 10. The one rule that matters

**Do not claim a capability that has not been measured on this machine, at a stated quality floor, with a reproducible
command.** Everything else in this repository is negotiable; that is not.
