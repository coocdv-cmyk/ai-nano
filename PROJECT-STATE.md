# Project State — resume here

Last updated: 2026-11-19 (`EXP-N` N0 frozen and baselined; N1 repair engine built, gated on N0, and probed on a
same-family held-out set on `research/exp-n`).
Read this first if you are a new agent or a returning human. It is deliberately short; the other documents carry the
detail.

---

## 1. In one paragraph

OMEGA-NANO now has **four verified layers**: (a) a durable, tested agent control plane (planning, tools, verification,
memory, permissions, crash recovery); (b) a **measured neural core that has passed the compression baseline** — `OmegaCoreV2`
(2.00 M parameters) reaches `1.7482 bpb` against `zlib-9`'s `1.8370` on a matched clean split (EXP-005), after the v0 core
(3.34 M) reached only `2.7841 bpb` under the same 300 s budget (its older `1.8944 bpb` came from the pre-cleanup,
licence-contaminated corpus and is kept as history, not as a comparison); (c) an execution-guided **neuro-symbolic brain**
that solved `10/10` sealed blind problems and `10/10` held-out problems with zero unverified answers (EXP-006/008); and
(d) a bounded, execution-verified **AST repair engine** that solved `6/6` frozen N0 code tasks (EXP-010) and `7/7`
same-family surface-transfer tasks (EXP-011) with zero unexecuted patches. The native CPU runtime reaches
`10,509–13,209 tok/s` single-stream and `23,284–25,177 tok/s` at batch 4 with `122,880–135,168` bytes of packed 2-bit
block weights (EXP-007/008). What does **not** exist yet is open-ended generalization: the 20/20 result is on
author-designed, verifier-checkable tasks, and N1 repairs synthetic workspaces with nine AST operators — neither touches
an arbitrary repository or free-form language. That gap is the rest of `EXP-N`.

---

## 2. What works, what does not

| Capability | State | Evidence |
|---|---|---|
| Durable task engine, write-ahead checkpoints, crash recovery | **works** | `tests/test_recovery.py`; 180 passed, 1 skipped (2026-11-19) |
| Independent verification against the world (hashes, file contents) | **works** | `verify.py`, `tests/` |
| Permission modes, scopes, approvals, emergency stop | **works** | `security.py`, `tests/test_security.py` |
| Bilingual rule reasoner (EN/AR) for fixed intents | **works** for fixed intents; abstains otherwise | `reasoner.py`, `tests/test_reasoner.py` |
| Tool layer (typed, validated, bounded, dynamic idempotency) | **works** | `tools/`, `tests/test_tools.py` |
| Memory fabric (lexical + int8 vector + fuzzy, provenance, versioning) | **works**, measured to 50k records | `BENCHMARKS/results/memory.json` (older machine) |
| Native CPU decode engine (Numba, ring-KV, batch) | **works**, greedy-identical to PyTorch | EXP-003, EXP-007 |
| Ternary / 2-bit packing (4× vs int8, 16× vs fp32) | **works** | `core/packed.py`, EXP-007 |
| Neural language modeling vs zlib-9 | **first pass**: v2 `1.7482` vs zlib-9 `1.8370` (2.00 M params, clean split) | EXP-005 |
| Adaptive reasoning quality | **measured on sealed suites only** (`20/20`), not open-world | EXP-006, EXP-008 |
| Sealed, contamination-free coding/math synthesis | **works** for the tested families | `brain/blind_eval.py` |
| Repository repair on multi-file Python bugs | **bounded capability**: 6/6 on the frozen N0 code track and 7/7 on a same-family held-out surface probe, zero unexecuted patches; **not open-world** — nine AST operators, synthetic workspaces | EXP-010, EXP-011; `brain/repair.py` |
| Open-ended natural language / world knowledge | **absent** — 0.8–2 M weights cannot store an encyclopedia | EXP-008 §Gaps |
| Continual learning without forgetting | **first signal**: wake-sleep replay `6.7749 → 3.0872 bpb` in `6.104 s`, single task | EXP-008 |
| Million-token context | **absent** | `EXP-P` target |
| Multimodal, computer-use, multi-agent | **absent** | not started |

---

## 3. Numbers you may quote, and how

Always attach the qualifier. Never quote a bare number.

| Number | Exact meaning | Machine | Source |
|---|---|---|---|
| **1.7482 bpb** | `v2_2.00M_5min`: 2,002,204 params, fp32, held-out val, 1.13 MB trained, 306.26 s CPU | i7-7820HQ | EXP-005 showdown |
| **1.8370 bpb** | `zlib-9` on the identical split | i7-7820HQ | EXP-005 showdown |
| **2.7841 bpb** | `v0_3.34M_5min` on the identical split (same 300 s budget) | i7-7820HQ | EXP-005 showdown |
| **1.5724 / 1.5540 bpb** | `bz2-9` / `lzma-extreme`, same split (both still ahead of v2) | i7-7820HQ | EXP-005 |
| **0.9609** | 2048-byte needle top-1 recall with the copy-gate memory (v2) | i7-7820HQ | EXP-005 showdown |
| **11.12 bpb** | Arabic 256-byte window, v2 (zlib-9 `4.5447`) — **the core is still far behind on Arabic bytes** | i7-7820HQ | EXP-005 showdown |
| **10/10 + 10/10** | sealed `BlindVault` + disjoint `BlindVault.heldout()` code/math problems, 100% verified | i7-7820HQ | EXP-006, EXP-008 |
| **13,209.0 tok/s** | native `v2` streaming, **single-stream**, greedy, 135,168 B packed blocks | i7-7820HQ | EXP-008 |
| **25,177.6 tok/s** | native `v2`, **batch 4**, same settings | i7-7820HQ | EXP-008 |
| **10,509.2 tok/s / 23,284.9 tok/s** | native `v2` incremental engine, single / batch 4, 122,880 B packed blocks | i7-7820HQ | EXP-007 |
| **2.23–9.41 ms** | TTFT across native v1/v2 configurations | i7-7820HQ | EXP-007/008 |
| **122.9–135.2 KB** | packed 2-bit block weights resident in L2 | i7-7820HQ | EXP-007/008 |
| **6.104 s** | wake-sleep distillation, 837 K params, `val_bpb 6.7749 → 3.0872` | i7-7820HQ | EXP-008 |
| **257.85 ms** | AST index of `omega_nano/`: 39 files, 426 symbols, 0 errors | i7-7820HQ | EXP-008 |
| **335.7 MB** | peak process RSS, full Python + Torch + SQLite runtime | i7-7820HQ | EXP-008 |
| **166 passed, 1 skipped** | active suite before N0, 89.31 s | i5-1135G7, Py 3.13.15 | 2026-11-19 |
| **173 passed, 1 skipped** | active suite after adding the N0 harness tests, 45.75 s | i5-1135G7, Py 3.13.15 | 2026-11-19 |
| **180 passed, 1 skipped** | active suite after adding the N0 + N1 tests, 44.14 s | i5-1135G7, Py 3.13.15 | 2026-11-19 |
| **6/6 · 24 verifier calls** | N1 `RepairEngine` on the frozen N0 code track; 0 unexecuted patches, scorer agrees, 25.241 s / 228.08 MB | i5-1135G7 | EXP-010 |
| **7/7 · 26 verifier calls** | N1 same-family held-out surface probe; operators frozen before authoring, 0 unexecuted patches, 27.782 s / 228.44 MB | i5-1135G7 | EXP-011 |
| **0/6 code · 0/12 reasoning · recall@5 0.6667** | N0 sealed baseline (`EXP-009`), product as-is; oracle `6/6`, `12/12`, `18/18` | i5-1135G7 | EXP-009 |
| **1.8944 / 2.0554 bpb** | *historical* v0 fp32 / ternary, 3.34 M, contaminated corpus | i7-7820HQ | EXP-003 (superseded, keep as history) |
| **1,425 / 5,027 B/s** | *historical* v0 ternary decode single / aggregate batch 8 | i5-1135G7 | Phase-0 audit §5.5 |

**Never** compare `13,209 tok/s` with `5,027 B/s`: different runtimes, different model sizes, and one is bytes while the
other is tokens. **Never** present a cross-machine ratio as a speedup. **Never** call `20/20` general intelligence; the
tasks were authored with verifiers. **Never** call N1 `6/6` or `7/7` open-world repository repair: both suites are
synthetic, the seven held-out tasks are same-family surface variants authored with the operator list in hand, and a bug
outside the nine operators is refused rather than fixed.

---

## 4. Confirmed defects — status

| # | Defect | Status |
|---|---|---|
| 1 | `fs.write` append claimed idempotent; retries duplicated content | **fixed (M0)** — `dynamic_idempotent` |
| 2 | `LocalLLMReasoner.revise` injected raw tool output into the prompt | **fixed (M0)** — `untrusted_tool_observation` |
| 3 | `shell.run` inherited `os.environ` (secret leak) | **fixed (M0)** — sanitized allowlist |
| 4 | `shell.run` is not an OS sandbox | **open** — N1 limits itself to isolated workspace copies, so the measurement is valid, but autonomous editing of the real workspace still blocks on this decision |
| 5 | Audit log is not hash-chained | **open** — tamper-evident only by convention |
| 6 | Packaging omitted `omega_nano.core` / extras | **fixed (M0)** — `pyproject.toml` 0.3.0 |
| 7 | Version mismatch `pyproject` vs `__init__` | **fixed (M0)** — both `0.3.0` |
| 8 | No CI gate | **fixed (M0)** — `.github/workflows/ci.yml` |
| 9 | Zero git commits | **fixed** — `master` + `research/exp-k|l|m|n` |
| 10 | Corpus included `site-packages` copyleft code | **fixed (M0)** — allowlist + SHA-256 manifest |
| 11 | `tests/test_exp_i/j/k/l.py` integrity pins predate `EXP-M`'s sanctioned `omega_nano/` changes, so `pytest tests` reports **8 failures** on every post-`EXP-M` branch (reproduced on a clean `21b93b4` worktree) | **open — owner decision**; the active suite is green (180 passed, 1 skipped). See `RESEARCH/EXP-N-PLAN.md` §8.1 |

---

## 5. Decisions that need the owner

1. **Choose a project licence.** Still no `LICENSE` file, so the default is all rights reserved. Apache-2.0 remains the
   natural fit. This gates any external publication of results or weights.
2. **Decide on the OS sandbox for `shell.run`** (defect #4). N1 already confines its search to an isolated workspace
   copy, so the measured result stands; autonomous editing of the **real** workspace, or executing third-party test
   code, still needs a job-object or container boundary.
3. **Confirm the `EXP-N` benchmark seal.** `EXPERIMENTS/EXP-009-n0-benchmark/task_suite/v1` is frozen and measured; its
   tasks must never enter a training corpus, and edits require a version bump plus a fresh baseline. `EXP-011`'s suite
   is now spent as a held-out set for the same reason.

---

## 6. Where the work lives

| Branch | Head | Content |
|---|---|---|
| `master` | `03149eb` | EXP-L L3–L7 merged, `فهم الخطة` master charter |
| `research/exp-l` | `b403ff0` | semantic drift, composition verification, retrieval, calibration, natural-input bridge, economics, audit |
| `research/exp-m` | `21b93b4` | `M0` integration + fixes, `M1` core v2, `M2` native engine, `M3` brain, `M4` V1 matrix + `HANDOFF.md` |
| `research/exp-n` | `21b93b4` + uncommitted | `EXP-N` plan, N0 sealed benchmark, N1 repair engine, EXP-010/011 records |

`constructive/` (EXP-H…L) is frozen historical research. `omega_nano/` is the product line.

---

## 7. The next single action

> **Commit N0+N1 on `research/exp-n` (owner approval required), then start `EXP-N` N2**: export the verified N1
> trajectories into a `(spec, trace, verified_code)` frame corpus and train the MTP heads.

`N0` is complete and measured: the suite is frozen (`manifest a16e73d3…`), the oracle is `6/6 · 12/12 · 18/18`, and the
product baseline is `0/6 code`, `0/12 reasoning`, `0.6667` strict recall@5 (`1.0` at topic level). `N1` moved the code
track to `6/6` at 24 verifier calls with zero unexecuted patches (EXP-010), and `7/7` on a same-family surface-transfer
probe (EXP-011). N1 is bounded, not open-world; the next program step is N2, which uses those verified trajectories.
Detail: `RESEARCH/EXP-N-PLAN.md` §3–§4.

Rationale: every later `EXP-N` claim (`N2` MTP acceptance, `N4` search gains) is measured against the frozen suite. A
benchmark defined after the improvements cannot validate them. N2's gate is MTP acceptance `> 40%` at
`> 10,509 tok/s` with greedy output identical to the non-speculative path.

---

## 8. Resume commands

```bash
# environment and state
python -m omega_nano doctor
python -m omega_nano tools

# regression (active suite)
python -m pytest tests/test_agent_v1.py tests/test_brain.py tests/test_core.py tests/test_engine.py \
  tests/test_llm.py tests/test_memory.py tests/test_reasoner.py tests/test_recovery.py \
  tests/test_security.py tests/test_state.py tests/test_tools.py tests/test_exp_n_benchmark.py \
  tests/test_exp_n1_repair.py -q
  # expect 180 passed, 1 skipped

# WARNING: do NOT run bare `pytest tests`. It collects the frozen constructive EXP-I..L suites, which
#   (a) rewrite recorded result files under constructive/ (violating the immutability rule), and
#   (b) pin pre-EXP-M hashes, so 8 tests fail on any post-EXP-M branch.
#   Reproduced on a clean 21b93b4 worktree; see PROJECT-STATE §4 defect 11 and RESEARCH/EXP-N-PLAN §8.1.
#   Use the explicit active-suite command above.

# recorded experiments
python EXPERIMENTS/EXP-005-m1-v2-core/run_exp.py
python EXPERIMENTS/EXP-007-m2-bitpacked-engine/run_exp.py
python EXPERIMENTS/EXP-008-m4-omega-v1-matrix/run_exp.py

# EXP-N N0 benchmark
python EXPERIMENTS/EXP-009-n0-benchmark/run_exp.py
python EXPERIMENTS/EXP-009-n0-benchmark/run_exp.py --oracle-only

# EXP-N N1 repair
python EXPERIMENTS/EXP-010-n1-ast-repair/run_exp.py
python EXPERIMENTS/EXP-011-n1-heldout/run_exp.py
python -m omega_nano brain repair --path <workspace> --tests tests --max-calls 80
```

State lives in `./.omega/` (override with `--home` or `OMEGA_HOME`). Tools are confined to `./.omega/workspace/`.

---

## 9. Document map

| Question | Document |
|---|---|
| What exists and what is broken? | `PHASE-0-REPOSITORY-AUDIT.md` (historical, 2026-11-18) |
| How does the system work now? | `ARCHITECTURE.md`, `RESEARCH/EXP-M-ARCHITECTURE.md` |
| What is the master vision and the seven programs? | `فهم الخطة`, `OMEGA-NANO-PROJECT-MISSION.md` |
| What was completed in EXP-M? | `RESEARCH/HANDOFF.md` |
| What is the next research program? | `RESEARCH/EXP-N-PLAN.md` |
| How do we measure anything? | `BASELINE-AND-BENCHMARK-PLAN.md`, `EXPERIMENTS/EXP-009-n0-benchmark/README.md` |
| What did N1 measure, and how far does it generalise? | `EXPERIMENTS/EXP-010-n1-ast-repair/README.md`, `EXPERIMENTS/EXP-011-n1-heldout/README.md` |
| What can we legally publish? | `LICENSE-AND-PROVENANCE-REGISTER.md`, `LICENSE` (missing) |
| Where do I resume? | this file |

---

## 10. The one rule that matters

**Do not claim a capability that has not been measured on this machine, at a stated quality floor, with a reproducible
command.** Everything else in this repository is negotiable; that is not.
