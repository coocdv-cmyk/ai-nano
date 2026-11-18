# PHASE-0 — Repository Audit (factual)

**Date:** 2026-11-18
**Root:** `D:\ai model`
**Auditor role:** Phase 0 discovery only. No production code was modified, deleted, or run against real user data.
**Method:** Read every source file and document; ran the test suite, `doctor`, `tools`, a fresh engine spot-check, a
corpus license scan, and the known `fs.write` append reproduction. All numbers below were observed in this session unless
explicitly labelled *recorded* (i.e. taken from the repository's historical JSON).

---

## 1. Checklist — every item the mission asked for

| Requested item | Finding | Where |
|---|---|---|
| Repository root | `D:\ai model`, git repo, branch `master` | this doc §4 |
| Project identity / version | `omega-nano` 0.1.0 in code; CHANGELOG claims 0.2.0 | §3.3 |
| Constitution / mission | `OMEGA-NANO-VISION-V3-ENGLISH.md`, 1,203 lines, v3.0 | §3.1 |
| README / agent instructions | present, accurate except test count and version | §6 |
| Architecture docs | `ARCHITECTURE.md` (58 lines) — **stale** on the neural core | §6.1 |
| Research docs | `RESEARCH/RC-001..003`, 3 research cards | §3.1 |
| Source code | 27 Python modules, 3,868 lines under `omega_nano/` + benchmark/experiment scripts | §3.2 |
| Model implementations | own byte-level decoder, 3.34 M params, fp32 and ternary arms | §5.4 |
| Experimental implementations | EXP-001 (kernels), EXP-002 (retrieval), EXP-003 (core) | §5 |
| Training pipeline | `omega_nano/core/train.py`, wall-clock budget, AdamW | §5.4 |
| Inference engine | Numba/LLVM token loop with KV cache | §5.5 |
| Data ingestion | corpus builder + memory bulk ingest; **no document parser** | §5.6 |
| Knowledge storage | SQLite + FTS5 + int8 vectors + trigram index | §5.6 |
| Memory implementations | versioned, provenance-tracked, hybrid retrieval | §5.6 |
| Benchmarking | 2 benchmark scripts, 2 result files | §5.1 |
| Evaluation | bits/byte + compressor baselines; no task-success benchmark | §5.4 |
| Datasets and provenance | Python stdlib corpus, **contaminated with third-party code** | §8 |
| Tests | 75 test functions → 127 cases; **126 passed, 1 skipped** | §5.2 |
| Runtime environment | Python 3.13.15, CPU-only, no C compiler | §2 |
| Dependencies | stdlib-first; numpy/numba/torch installed but **not all declared** | §3.4 |
| Hardware detection | `omega_nano/hw.py`, works | §2 |
| Checkpoints / weights | 2 × 13.4 MB `.pt`, present, gitignored | §5.4 |
| Experimental results | EXP-001/002/003 JSON, internally consistent | §5 |
| Licenses / attribution | **no LICENSE file; no NOTICE; corpus license risk** | §8 |
| Programming language / maturity | Python 3.10+; Phase 1 solid, Phase 2 first point, Phases 4–9 absent | §7 |

---

## 2. Environment actually used for this audit

Measured now, on the machine the repository lives on:

| Property | Value |
|---|---|
| OS | Windows 11 (10.0.26200) |
| CPU | 11th Gen Intel Core i5-1135G7 @ 2.40 GHz |
| Cores / threads | 4 physical / 8 logical |
| L2 / L3 cache | 5 MB / 8 MB (reported by WMI) |
| RAM | 15.66 GB |
| Free disk (D:) | 468.3 GB |
| Python | 3.13.15 |
| SQLite | 3.50.4 |
| numpy | 2.5.3 |
| numba | 0.68.0 |
| torch | 2.14.1+cpu, CUDA unavailable |
| C/C++ toolchain on PATH | **none** (`cl`, `clang`, `gcc`, `cmake` all absent) |
| GPU | none / integrated only; `torch.cuda.is_available() == False` |

**Critical methodological fact.** Every recorded performance number in the repository was measured on a *different*
machine: Intel i7-7820HQ (4C/8T), 7.41 GB RAM, Python 3.10.11, Windows 10. Those numbers are quoted, never presented
here as reproduced. The prior `MODEL_REVIEW.md` (18 Nov 2026, Arabic) already made this distinction; this audit confirms
it and adds one fresh spot-check (§5.5).

---

## 3. What exists

### 3.1 Documents

| File | Size | Purpose | Audit note |
|---|---|---|---|
| `OMEGA-NANO-VISION-V3-ENGLISH.md` | 54.6 KB, 1,203 lines | Constitution v3.0 | Coherent, unusually disciplined. Defines 16 architecture layers, 10 phases with evidence gates, anti-fabrication rules. |
| `README.md` | 4.4 KB | Overview + quick start | Accurate on behaviour; says "127 tests" where the executable count is 127 cases / 126 passed + 1 skipped. |
| `ARCHITECTURE.md` | 3.9 KB | "as implemented" | **Stale**: row 1 says neural core "not started" while `omega_nano/core/` exists and EXP-003 is measured. |
| `ROADMAP.md` | 2.3 KB | Phase gates + backlog | Honest; Phase 2 correctly called "first point measured". |
| `CHANGELOG.md` | 1.4 KB | Version history | Documents 0.2.0; code says 0.1.0. |
| `AGENT_INSTRUCTIONS.md` | 0.9 KB | Agent rules | Good: tests before/after, no unverified "done", ADR for permission changes. |
| `SECURITY.md` | 2.1 KB | Threat model | Good intent; understates that `shell.run` is not an OS sandbox. |
| `MODEL_REVIEW.md` | 46.3 KB | Prior full review (Arabic) | Independent, detailed, and materially correct. It is the best existing state document. |
| `DECISIONS/ADR-0001-phase1-foundation.md` | 1.8 KB | Why Python/SQLite/rules-first | Sound. |
| `RESEARCH/RC-001..003` | 7.9 KB total | 10k tok/s, memory fabric, "the brain" | Honest research cards; RC-003 explicitly refuses to claim frontier capability. |
| `BENCHMARKS/README.md` | 2.0 KB | Measurement protocol | States the <1.5× noise caveat. |
| `EXPERIMENTS/EXP-00{1,2,3}` | ~16 KB | Experiment records | Negative results preserved (2-bit hypothesis rejected). |

### 3.2 Source tree (line counts, verified)

```
omega_nano/
  engine.py            425   control loop, recovery, retries, revision, approvals
  memory.py            441   memory fabric, normalization, embedding, RRF
  reasoner.py          300   bilingual rule reasoner + OpenAI-compatible LLM adapter
  store.py             196   SQLite WAL, checksummed checkpoints, audit log, plan cache
  model.py             140   Task/Step dataclasses + state machine
  cli.py               178   run/resume/status/report/approve/stop/tools/doctor/mem
  verify.py            102   independent postcondition checks
  kernels.py            60   Numba int8×f32 dot, numpy fallback
  hw.py                 77   hardware detection
  config.py             52   paths, limits, shell allowlist
  security.py           66   modes, scopes, path guard
  report.py             59   Markdown task report
  tools/               546   base contract + fs, math, shell, memory, core tools
  core/model.py        143   OmegaCore: byte decoder, BitLinear, RoPE, SwiGLU
  core/engine.py       300   native Numba decode loop, KV cache, batched streams
  core/train.py        112   wall-clock-budgeted training
  core/data.py          90   corpus builder, file-level split, compressor baselines
tests/                 820   75 test functions (127 parametrized cases)
BENCHMARKS/            233   2 scripts
EXPERIMENTS/           253   EXP-001 script, EXP-003 evaluator
```

There is no vendored third-party source in `omega_nano/` (no `Copyright`/SPDX headers found). All code appears to be
original to this project.

### 3.3 Identity and version

- `pyproject.toml`: `version = "0.1.0"`, `dependencies = []`, packages `["omega_nano", "omega_nano.tools"]`.
- `omega_nano/__init__.py`: `__version__ = "0.1.0"`.
- `CHANGELOG.md`: `## 0.2.0 — 2026-11-17` includes the neural core.
- **No single trustworthy version.** The code is functionally 0.2.0; the packaging metadata says 0.1.0.

### 3.4 Dependencies

Declared: none required; optional `fast = [numpy, numba]`, `dev = [pytest]`.
Actually imported at runtime: `torch` (hard requirement of `omega_nano.core` and `core.complete`), `numpy` and `numba`
(optional with fallbacks), stdlib otherwise.
`torch`, `numpy`, `numba` are **not** declared in any dependency group. `pip install .` also omits `omega_nano.core`
because `packages` is an explicit list. Consequence: an installed package cannot run `core.complete`.

---

## 4. Git state

```
branch: master
commits: 0
status: every project file untracked (??)
ignored: .omega/, __pycache__/, *.pyc, .pytest_cache/, *.nbi, *.nbc, EXPERIMENTS/**/*.pt
```

This is the single largest project risk: there is no history, no diff, no rollback, and a single `git clean -fdx`
would destroy the entire program. The prior review flagged this as P0; it remains true.

---

## 5. What was actually executed in this audit (evidence)

### 5.1 Commands

```
python -m pytest -q --durations=8        → 126 passed, 1 skipped in 16.54 s
python -m omega_nano doctor              → hardware, empty task counts, 0 memory records
python -m omega_nano tools               → 10 tools listed with scopes/idempotency
fresh Engine.bench (this machine)        → see §5.5
corpus byte scan                         → third-party licence text present, see §8
fs.write append reproduction (temp home) → status failed, attempts 3, content 'xxx'
```

The skipped test is the symlink-escape test (Windows needs admin rights to create symlinks). It is a real coverage gap
on the development platform: the most important path-escape protection is not proven here.

### 5.2 Test suite composition

| File | Test functions | Covers |
|---|---|---|
| `test_engine.py` | 21 | end-to-end tasks, verification, retries, revision, approvals, timeouts |
| `test_memory.py` | 11 | ingest, ranking, versions, hard delete, Arabic, incremental index |
| `test_core.py` | 8 (parametrized) | ternary math, causality, engine↔PyTorch equivalence, context-window generation |
| `test_security.py` | 8 | modes, scopes, path escape, symlink (skipped on Windows) |
| `test_tools.py` | 7 | argument validation, refusal cases, malformed output |
| `test_reasoner.py` | 6 | EN/AR plan parsing |
| `test_recovery.py` | 6 | real process kill + resume, corruption recovery, duplicate delivery |
| `test_state.py` | 5 | legal/illegal transitions |
| `test_llm.py` | 3 | stub OpenAI-compatible server, fallback |
| `conftest.py` | — | isolated temp home per test |

The suite is real: `test_recovery.py` kills a subprocess with `os._exit(77)` and verifies resume without re-execution.

### 5.3 Agent loop benchmark (recorded, not re-run)

200 tasks/tier, 3 verified steps each, 10 durable checkpoints per task, i7-7820HQ:

| Metric | t1 rules | t0 plan cache |
|---|---|---|
| e2e p50 / p95 | 22.723 / 39.663 ms | 13.350 / 43.191 ms |
| planning p50 | 0.119 ms | 0.089 ms |
| tasks/s | 40.4 | 52.5 |
| resume after crash (incl. Python start) | 407.3 ms | — |

Honest reading: planning is not the bottleneck; durable SQLite checkpointing and disk I/O dominate. The t0/t1 gap is
mostly a workload artifact (t0 rewrites an identical file), which the benchmark document itself states.

### 5.4 Neural core (recorded, EXP-003)

Byte-level decoder: `d=256`, 4 layers, 4 heads, SwiGLU 704, RoPE, ctx 256, 3,344,640 params (3,211,264 in block
matrices). Trained 30 wall-clock minutes per arm on 19.9 MB of Python source, file-level split.

| Arm | val bits/byte | block weights | single-stream (1T) | aggregate (7T, 14 streams) |
|---|---|---|---|---|
| fp32 | **1.8944** | 12.8 MB | 402 B/s | 1,276 B/s |
| ternary | **2.0554** | 3.2 MB | 841 B/s | 2,865 B/s |

Baselines on the same validation bytes: uniform 8.0, order-2 n-gram 2.579, **zlib 1.830**, bz2 1.466, lzma 1.434.

Honest reading: the core learns (beats n-gram by 27%), but **loses to zlib** and is far from lzma. Samples are
locally plausible Python-shaped text, not coherent programs. Ternary costs 8.5% quality for 4× size and ~2.1× speed at
this scale. The 10k tok/s target is not reached (best recorded 2,865 aggregate).

### 5.5 Fresh spot-check on this machine (new evidence, narrow scope)

`Engine.load(...).bench(n_new=400, batch=8)`, 4 threads, `NUMBA_CACHE_DIR` in temp, single run, uncontrolled thermals:

| Arm | single-stream | aggregate batch-8 | first call incl. JIT |
|---|---|---|---|
| fp32 | 673 B/s | 1,655 B/s | 8.01 s |
| ternary | **1,425 B/s** | **5,027 B/s** | 4.55 s |

This is *not* a re-run of EXP-003 (400 vs 2,000 generated bytes, different thread count, one seed, different CPU). It
shows the pipeline still works on Python 3.13 / numpy 2.5 / numba 0.68 / torch 2.14, and that this newer CPU is
materially faster than the 2017 i7 for the same code. It does **not** replace the recorded numbers.

Two facts matter for the roadmap:
1. **5,027 aggregate tok/s at 4 threads is already half of the 10k target** on the aggregate metric, with a 3.3 M model.
2. **Single-stream is 1,425 tok/s**, ~7× short. Aggregate and single-stream must never be conflated (constitution §5.2).

### 5.6 Memory fabric (recorded, EXP-002)

50k records, 300 queries/style: paraphrase R@1 1.000, typo R@1 0.33 / R@5 0.83, Arabic R@1 1.000, p50 22.6 ms,
~1.25 KB/record, ~1,700 records/s ingest, brute-force vector search. The ablation is the valuable part: without the
gated trigram signal, typo R@5 collapses to 0.03 at 50k. The benchmark *favours lexical retrieval* by construction
(paraphrases reuse entity names), which the experiment document states explicitly. No ANN index exists; latency is linear.

---

## 6. Documentation drift and internal contradictions

| # | Claim in docs | Reality | Severity |
|---|---|---|---|
| 1 | `ARCHITECTURE.md`: neural core "not started" | `omega_nano/core/` implemented, trained, measured, exposed as `core.complete` | High — misleads every new agent |
| 2 | README: "127 tests" | 127 cases, **126 passed + 1 skipped** | Low |
| 3 | CHANGELOG 0.2.0 vs `pyproject`/`__init__` 0.1.0 | two versions in one tree | Medium |
| 4 | README: "no required dependencies" | `torch` is required for `core.complete` and `omega_nano.core` | Medium |
| 5 | SECURITY.md implies workspace confinement | `shell.run` runs allowlisted interpreters with full user privileges and inherited environment | **High** |
| 6 | Architecture invariant 5: "tool output never changes the plan" | true for successful steps and the rule reasoner; **false for `LocalLLMReasoner.revise`**, which serializes raw failed-step output into the prompt | **High** |
| 7 | ROADMAP Phase 0 "passed (empty repo → this document set)" | the repo was not empty at audit time; Phase 0 was previously self-certified without this audit | Low |
| 8 | `EXPERIMENTS/EXP-003` says corpus is "Python standard library sources" | corpus builder globs **all** `.py` under `Lib`, including `site-packages`; scan confirms third-party licence text | **High (legal)** |

---

## 7. Maturity by constitution layer

| # | Layer | Status | Evidence |
|---|---|---|---|
| 1 | Neural core | **v0 measured** (3.34 M params), below zlib | EXP-003 |
| 2 | Adaptive compute | **prototype** (3 planning tiers, plan cache) | `engine._plan` |
| 3 | Inference runtime | **prototype** (Numba decode loop, int8 storage; no SIMD blocking, no rolling cache) | `core/engine.py` |
| 4 | Memory fabric | **implemented and measured to 50k** | EXP-002 |
| 5 | Knowledge ingestion | **partial**: bulk text ingest, dedup; no document parsing | `memory.remember_many` |
| 6 | Retrieval | **implemented**: BM25 + hashed int8 vectors + gated trigram, RRF | `memory.recall` |
| 7 | Reasoning/planning | **prototype**: fixed bilingual intents; optional local LLM | `reasoner.py` |
| 8 | Tool execution | **implemented**: 10 typed tools, pre/post validation | `tools/` |
| 9 | Computer interaction | absent | — |
| 10 | Software engineering | absent (only a bounded shell) | — |
| 11 | Multimodal | absent | — |
| 12 | Verification | **implemented** for world-checkable effects | `verify.py` |
| 13 | Agent orchestration | **implemented** for single-agent durable tasks | `engine.py`, `store.py` |
| 14 | Security/governance | **prototype**: modes, scopes, audit, stop; not a sandbox | `security.py` |
| 15 | UI | CLI + Markdown reports | `cli.py`, `report.py` |
| 16 | Evaluation loop | **partial**: bits/byte, retrieval, loop timing; no task-success suite | `BENCHMARKS/` |

---

## 8. Confirmed defects and security findings

### 8.1 `fs.write` is declared idempotent but `mode="append"` is not — **reproduced now**

`omega_nano/tools/fs.py:125` sets `idempotent=True` unconditionally. The engine's retry path
(`engine.py:300`) therefore re-runs a failed append up to `max_retries`. Reproduction in a temp home:

```
plan: fs.write(path=a.txt, content=x, mode=append), expect file_endswith "ZZZ"
result: status failed, attempts 3, file content 'xxx'
```

Two independent problems: (a) the tool contract is false, so the "non-idempotent steps are escalated, not repeated"
guarantee does not hold for append; (b) verification failures cause real duplicated side effects. This is a correctness
bug in the project's strongest claimed invariant. It must be fixed before any autonomous file editing (Phase 4).

### 8.2 Prompt-injection path through plan revision

`reasoner.py:295` builds `{"goal": goal, "failed_step": asdict(failed), ...}` and sends it to the local LLM. `asdict`
includes `output`, which may contain the contents of a workspace file (`fs.read` returns `content`) or raw shell
output. Any file the agent reads can therefore inject instructions into the revision prompt. The architecture claim
"untrusted data never changes the plan" does not hold on this path. Fix: send a structured failure summary only (tool,
error class, short message), never raw untrusted content.

### 8.3 Environment leakage through `shell.run`

`tools/shell.py:64` passes `env={**os.environ, ...}`. Allowlisted `python`/`node` can print every API key and token,
and the output is stored in the task state and report. Fix: clean allowlisted environment; redact secrets from stored
output; document that `EXECUTE` equals the user's privileges.

### 8.4 `shell.run` is not a sandbox

`guard_path` confines the *file tools*. `shell.run` runs real interpreters with the user's full filesystem and network
rights. The constitution's "isolate untrusted code" is not met. This is documented as a limitation in `ARCHITECTURE.md`
but should be stated plainly in `SECURITY.md` and treated as a blocker for unattended autonomy.

### 8.5 Audit log is not tamper-evident

Append-only at the application level, but not hash-chained. Anyone with the SQLite file can rewrite history and
recompute the checksums. Checksums detect accidental corruption, not tampering.

### 8.6 Packaging

- `omega_nano.core` missing from `[tool.setuptools] packages`.
- `torch`/`numpy`/`numba` undeclared (no `core` extra).
- Version mismatch (§3.3).
- No `LICENSE`, no `NOTICE`, no `py.typed`, no CI workflow.

### 8.7 Corpus provenance

`core/data.py:28` globs `os.path.join(lib, "**", "*.py")`. On this machine that pattern matches **19,218** files, of
which **17,447** are under `site-packages`. The historical corpus (`meta.json`) was built from a Windows Store Python
3.10 install and contains 1,408 train / 91 val files. A byte scan of `train.bin` finds:

- 62 × "GNU Lesser General Public" (chardet-style LGPL-2.1 blocks), 4 in validation;
- 42 × "Apache License" (e.g. `distro`, `packaging`, `requests`);
- markers for `setuptools`, `distro`, `packaging`, `requests`/`certifi`.

So the corpus is **not** pure PSF-licensed standard library. This does not affect running the code, but it affects any
redistribution of the corpus or of weights trained on it, and it invalidates the README/EXP-003 description. See
`LICENSE-AND-PROVENANCE-REGISTER.md` for the register and remediation.

### 8.8 Reproducibility gaps

- `.omega/` corpus and `*.pt` weights are gitignored, with no SHA-256 recorded in `results.json`.
- No file list for the corpus; `max_mb` truncation is alphabetical, so corpus contents depend on filename order.
- No CI; all claims come from manual runs on one machine.
- No task-success benchmark on real work; all agent benchmarks are synthetic 3-step tasks.

---

## 9. What is genuinely functional (not conceptual)

1. **Durable control loop.** Write-ahead checkpoints, checksum-verified state, real crash recovery without re-execution,
   bounded retries, human approval, emergency stop, idempotency keys.
2. **Independent verification.** File effects are re-read from disk and hashed; the tool's own report is not trusted.
3. **Tool contract.** Typed parameters, rejection of unknown arguments, post-execution shape validation, explicit
   idempotency and scopes.
4. **Memory fabric.** Versioning, provenance, validity intervals, hard delete, Arabic normalization, RRF, measured
   ablations, incremental index.
5. **Exact math engine.** Arbitrary-precision integers and rationals via a whitelisted AST; no `eval`; bounded.
6. **Own neural core, end to end.** Data → training → ternary quantization → native inference → agent tool, with
   numerical equivalence to PyTorch tested.
7. **Bilingual rule reasoner.** Fast, deterministic, honest about unparsed goals.

## 10. What is only conceptual or absent

- Any capability on real software-engineering tasks.
- Browser/desktop control, multimodal input/output, multi-agent orchestration.
- A model that beats a general compressor or produces coherent text.
- 10k tok/s (single-stream or with a quality floor).
- Continuous learning beyond a hash-keyed plan cache and text memory.
- A task-success benchmark, an OS sandbox, CI, packaging that installs, a licence.

---

## 11. Preserve / repair / replace (Phase 0 recommendation)

**Preserve unchanged:** the engine/verification/store/security *invariants*, the memory fabric design, the exact math
engine, all experiment records and negative results, the constitution.

**Repair before new research (blocking):**
1. First git commit (and a second commit after every step of this plan).
2. `fs.write` idempotency (dynamic by mode) + regression test.
3. Packaging: include `omega_nano.core`, declare a `core` extra, unify version, add `LICENSE`.
4. Correct `ARCHITECTURE.md`, `SECURITY.md` (sandbox limits), corpus description.
5. Rebuild the corpus from an explicit allowlist with per-file SPDX + SHA-256.
6. Close the `revise` injection path and the `shell.run` environment leak.
7. Add CI (Linux + Windows) so the symlink test runs somewhere.

**Investigate as new architecture:** the learning mechanism in `ORIGINAL-LEARNING-MECHANISM.md`, not a bigger transformer.

**Do not do:** train a larger dense core on the contaminated 20 MB corpus and call it progress. The evidence says it
would still lose to lzma and would not answer the mission's central question.

---

## 12. Reproduction commands

```powershell
cd "D:\ai model"
python -m pytest -q
python -m omega_nano doctor
python -m omega_nano tools

# fresh engine spot-check (does not overwrite recorded results)
$env:NUMBA_CACHE_DIR="$env:TEMP\omega_numba_cache"; $env:NUMBA_NUM_THREADS="4"
python -c "from omega_nano.core.engine import Engine; e=Engine.load('EXPERIMENTS/EXP-003-omega-core-v0/runs/ternary/model.pt'); e.generate(b'def ',8); print(e.bench(n_new=400,batch=8))"

# corpus licence scan
python -c "tr=open('.omega/core_data/train.bin','rb').read(); print({s:tr.count(s.encode()) for s in ['GNU Lesser General Public','Apache License','site-packages']})"
```

## 13. Audit limitations

- Historical numbers were not re-measured; only one narrow spot-check was run.
- The corpus scan is a byte-level keyword scan, not an SPDX/file-level audit; it proves contamination exists, not its
  exact extent.
- `MODEL_REVIEW.md` was written on the same machine as this audit but before it; its findings were spot-checked, not
  fully re-derived.
- No dynamic security testing (fuzzing, injection payloads against the LLM path) was performed.
- No legal advice is given; the provenance register flags risk and required actions.
