# Architecture — as implemented (0.3.0-dev / EXP-M `M0`–`M4` + EXP-N `N0`–`N1`)

This describes what exists, not what is planned. The target architecture (16 layers) is in the constitution §4.
Updated 2026-11-19 on `research/exp-n` (base `21b93b4`) after `M0` integration + fixes, `M1` `OmegaCoreV2`, `M2` native
2-bit runtime, `M3` neuro-symbolic brain, `M4` integrated `OmegaAgent`, and the first `EXP-N` tracks: N0 sealed benchmark
and N1 execution-verified repair. Evidence is in `EXPERIMENTS/EXP-004…011`; the continuation plan is
`RESEARCH/EXP-N-PLAN.md`.

Status vocabulary: **measured** (implemented and benchmarked) · **implemented** (tested, not benchmarked) · **prototype**
(works on a narrow path) · **partial** (some sub-capability missing) · **absent**.

| # | Constitution layer | Status | Where |
|---|---|---|---|
| 1 | Neural core | **measured**: `v0` (`OmegaCore`) and `v1` (`LoopedOmegaCore`, ACT + MTP); `v2` (`OmegaCoreV2`: hash-ngram stem, patched loops, delta fast weights, causal copy gate) at **1.7482 bpb** vs zlib-9 `1.8370` | `core/model.py`, `core/model_v2.py`, EXP-005 |
| 2 | Adaptive compute controller | **measured**: 3 planning tiers (`t0` cache / `t1` rules / `t2` LLM) + ACT halting in `v1`/`v2` (`eval_avg_loops ≈ 1.02`) | `engine._plan`, `core/model.py`, `core/model_v2.py` |
| 3 | Inference runtime | **measured**: Numba v0 ring-KV; native v1/v2 with ACT early-exit, 2-bit LUT T-MAC, self-speculation, streaming; **10,509–13,209 tok/s** single-stream, greedy-identical to PyTorch | `core/engine.py`, `core/engine_v1_v2.py`, EXP-007/008 |
| 4 | Memory fabric | **implemented**, measured to 50k records; bilingual semantic encoder + `L1` drift guard + `L3` behavioral guard | `memory.py`, `semantic.py` |
| 5 | Knowledge ingestion | bulk ingest + dedup; clean copyleft-free corpus builder with SHA-256 provenance | `memory.remember_many`, `core/data.py` |
| 6 | Retrieval and search | **implemented**: lexical BM25 + int8 vector + gated fuzzy, RRF; AR→EN recall demonstrated | `memory.recall`, EXP-008 |
| 7 | Reasoning and planning | rule reasoner (EN/AR) + hardened LLM adapter + execution-guided `NeuroSymbolicSolver` + `SelfPlayEngine` | `reasoner.py`, `brain/` |
| 8 | Tool execution | **implemented**: typed, validated, bounded, dynamic idempotency, sanitized env | `tools/` |
| 9 | Computer interaction | not started (Phase 5) | — |
| 10 | Software engineering engine | **bounded capability**: AST symbol/call-graph indexer (39 files / 426 symbols / 258 ms), `CodeVerifier`, blind code synthesis, and execution-verified `RepairEngine` (9 AST operators; pytest-traceback + `RepoGraph` localisation) — **6/6** frozen N0 code tasks and **7/7** same-family held-out tasks, 0 unexecuted patches; **not open-world**, no OS sandbox | `brain/repo_graph.py`, `brain/verifiers.py`, `brain/repair.py`, EXP-010/011 |
| 11 | Multimodal | not started (Phase 7) | — |
| 12 | Verification and critique | **implemented**: world checks + SymPy/mpmath (50-digit) + isolated Python with AST security audit | `verify.py`, `brain/verifiers.py` |
| 13 | Agent orchestration | **implemented**: durable single-agent tasks + unified offline `OmegaAgent` (`agent`, `brain distill`, `brain eval`, `brain index-repo`, `brain repair`) | `engine.py`, `store.py`, `agent.py`, `cli.py` |
| 14 | Security and governance | **implemented**: modes, scopes, approvals, sanitized shell env, stop; **still not an OS sandbox** (defect #4) | `security.py`, `tools/shell.py` |
| 15 | User interface | CLI + markdown reports + `agent` / `brain` subcommands | `cli.py`, `report.py` |
| 16 | Evaluation loop | **measured**: `EXP-001`, `EXP-003`–`EXP-011`, sealed `BlindVault`, `EXP-N` N0–N1 | `BENCHMARKS/`, `EXPERIMENTS/`, `constructive/` |

## Core invariants
1. **The plan is never trusted.** Every step is validated against the tool registry, argument schemas and the permission
   mode before it runs, whichever reasoner produced it.
2. **A step is done only when verified.** Postconditions such as file hashes and file contents are checked against the
   world, not the tool's report.
3. **Write-ahead.** A step is checkpointed as `started` before execution. After a crash, one of three things happens:
   (a) the world postcondition already holds, so the step is marked done without re-running; (b) the tool is idempotent,
   so it is re-run; (c) neither, so the task is escalated to a human.
4. **State is checksummed.** Each save writes the task document and a numbered checkpoint, both carrying a SHA-256.
   If the current state is corrupt, the newest valid checkpoint is restored and the steps after it are re-verified.
5. **Untrusted data stays data.** File contents and process output are returned as `trust: untrusted_data` and never
   change the plan. Only the user's goal and the reasoner do.
6. **Every important action is logged** to an append-only event table, which feeds the per-task report.
7. **Unverified output never passes.** `PROVEN_EXACT` / `TESTS_PASSED` are only emitted by an executing verifier; a failed
   verification produces a counterexample or traceback, never a success.
8. **A patch is accepted only by execution.** `RepairEngine` searches in an isolated copy, writes to the real workspace
   only after the workspace's own suite passes (and passes a confirmation run), and records every rejected candidate.
   `accepted_without_execution` must be zero for a repair result to be valid.

## Task state machine
```
pending → planned → running ⇄ (awaiting_approval | paused) → done
              ↘ blocked (can't understand, or uncertain side effect)  ↘ failed
```
Illegal transitions raise. `done` and `failed` are final.

## Adaptive compute tiers
| Tier | Mechanism | Measured planning time (p50) |
|---|---|---|
| t0 | procedural plan cache, keyed by normalized goal; only verified plans are cached; a failure evicts the entry | 0.09 ms |
| t1 | deterministic bilingual rule reasoner | 0.12 ms |
| t2 | local OpenAI-compatible LLM (optional; `llm_backend: null` by default) | model-dependent |

## Learning loop — the self-evolving mind
```
problem ─► SkillLibrary.match ─► NeuroSymbolicSolver
                                   ├─ SymbolicTacticSolver (families: intervals, graph order, DP, windows, …)
                                   ├─ NeuralCodeProposer (v2 checkpoint)
                                   └─ NeuralPolicyRanker (persistent online weights)
                                        │  B diverse candidates
                                        ▼
                                  CodeVerifier / MathVerifier  (isolated subprocess, AST-guarded)
                                        ├─ fail ─► FailureDiagnosis ─► next attempt
                                        └─ pass ─► SkillLibrary.register_skill + add_replay
                                                     │
                                                     ▼
                                          Wake-Sleep distill (distill.py)
                                          replay frames → v2 weights (val_bpb ↓)
```
Measured: `10/10` sealed + `10/10` held-out (EXP-006/008); policy transfer across names `2 → 1` attempts;
wake-sleep `6.7749 → 3.0872 bpb` in `6.104 s` on 837 K params (EXP-008).

## Repair loop — N1 (execution-verified AST mutation)

```
failing pytest suite
  → parse FAILED/ERROR nodes + traceback sites
  → RepoGraph localisation (direct calls, assertion names, traceback frames, callees)
  → candidate AST patches from 9 operators
  → isolated workspace copy + pytest          ← budgeted verifier call
       ├─ suite passes → confirm run → write back to the real workspace
       ├─ fewer failures → keep as new base → continue
       └─ no improvement → revert, next candidate
```

Measured: `6/6` on the frozen N0 code track at 24 verifier calls (`EXP-010`); `7/7` on a same-family held-out surface
probe at 26 calls (`EXP-011`); zero accepted patches without an executed verification in both. The seven held-out tasks
were authored after the operator set was frozen, but by the same agent with the operator list in hand — **same-family
surface transfer, not open-world repair**.

## Known limitations (updated after N1)
- The rule reasoner covers a fixed set of intents. Anything else is blocked unless an `LLM` backend is configured.
- The neural core is byte-level (`vocab=256`, `ctx=256`). It beats zlib-9 on Python bytes but is **far behind on Arabic**
  (`11.12 bpb` vs zlib `4.5447` at 256 bytes) and has no world knowledge.
- `bz2-9` (`1.5724`) and `lzma-extreme` (`1.5540`) still beat `v2` (`1.7482`) on the same split; the compression claim is
  specifically against `zlib-9`.
- MTP **without trained heads** destroys throughput: `1,408.1 tok/s`, `1/595` accepted (EXP-007). `EXP-N` N2 must train
  the heads or leave speculation off.
- `shell.run` is **not** an OS sandbox: it uses a real process timeout and a sanitized environment allowlist only.
- The audit log is **not hash-chained** (defect #5).
- The repair engine covers **nine AST operator families** on synthetic Python workspaces; a bug outside them is refused,
  not guessed at. It does not handle dynamic imports, generated code, `conftest` fixtures, or multi-package layouts.
  Localisation reads failing test bodies and traceback frames; helper indirection beyond the one-hop callee boost can
  be missed, and boundary guards are inferred only from literal integer arguments.
- Patched files are emitted with `ast.unparse`: comments and blank-line layout inside a rewritten file are lost, and
  quoting may change. A patch is proven by the suite, not merge-ready.
- The repair engine executes the workspace's own tests with the caller's privileges; the isolated copy protects the
  source tree, not the machine (defect #4).
- Repair search de-duplicates by `(operator, file, target)`, so a second bug of the same operator family in the same
  function is not attempted once the first is accepted.
- The N1 held-out suite (`EXP-011`) is now spent; a future N1 claim needs a new unseen suite.
- The engine has only been measured on suites that finish in seconds; suite runtime dominates on a real repository and
  has not been profiled.
- In-process tool timeouts abandon the worker thread instead of killing it.
- One worker per task (lease + PID liveness). No distributed queue.
- Long-range recall is demonstrated by the copy-gate over a **2048-byte** horizon, not by an attention mechanism.

## Data flow — task execution (as implemented)

```
user goal
  │
  ├─ idempotency key? ──► store.find_by_key ──► return existing task (logged duplicate_delivery)
  │
  ├─ _plan(goal)
  │    ├─ t0: store.plan_cache[sha256(normalize(goal))]        (verified plans only; a failure evicts)
  │    ├─ t1: RuleReasoner.plan  (EN/AR, fixed intents + templates)
  │    └─ t2: LocalLLMReasoner.plan  (optional OpenAI-compatible local server)
  │         └─ _validate_plan: unknown tool / bad args / unresolvable template ⇒ BLOCKED
  │
  └─ for each step i
       ├─ resolve ${prev.x}/${sN.x} templates
       ├─ tool.needed_scopes → security.decide(mode, scopes)
       │     ├─ DENY ⇒ task FAILED
       │     └─ ASK  ⇒ step AWAITING_APPROVAL, task AWAITING_APPROVAL, stop (human `approve`)
       ├─ step = STARTED; store.save("step_started")      ← write-ahead checkpoint
       ├─ tool.validate(args) → tool.run(ctx, args)        ← bounded by a worker-thread timeout
       ├─ verify(ctx, expect, output)                     ← re-reads disk, recomputes SHA-256
       │     ├─ pass ⇒ step DONE, checkpoint
       │     └─ fail ⇒ log; retry if idempotent and attempts ≤ max_retries
       │                └─ else RuleReasoner.revise → (optional) LLM revise → replace step
       │                └─ else task FAILED
       └─ after all steps: task DONE, summary, _learn()
             ├─ plan_cache[signature] = verified plan
             └─ memory.remember("Completed task: …", kind="episodic")
```

Crash path: `STARTED` steps are re-checked against `world_checks(expect)` (file kinds only). If the effect already
exists, the step is marked done without re-execution. If not, and the tool is not idempotent, the task is BLOCKED for a
human. Correctness is verified against the world, not against the tool's report.

## Data flow — memory fabric

```
remember(text, kind, source, key, confidence)
  → normalize (Arabic diacritics/alef/taa/ya/digits, casefold)
  → embed (bilingual semantic encoder or hashed fallback → 512-d)
  → quantize int8 + per-record scale
  → SQLite transaction: memories + memories_fts (BM25) + memories_tri (trigram)
  → same key ⇒ old row valid_to=now (history preserved, not deleted)
  → same normalized text ⇒ duplicate; cosine ≥ 0.97 ⇒ near_duplicate
recall(query, k)
  → unknown-word probe (vocabulary prefix = inflection vs typo)
  → lexical BM25 top-pool  ⊕  int8 vector top-pool (Numba)  ⊕  gated trigram fuzzy
  → reciprocal-rank fusion  →  score × (0.5 + 0.5·confidence)  →  active rows only
```

## Module map (imports, simplified)

```
cli.py
 ├─ config.load_config
 ├─ engine.Engine
 │   ├─ store.Store          (SQLite WAL, checkpoints, events, plan_cache)
 │   ├─ memory.MemoryFabric  (SQLite FTS5 + int8 matrix)  ← semantic.encode_text
 │   ├─ reasoner.RuleReasoner / LocalLLMReasoner
 │   ├─ tools.default_registry  (fs, mathx, shell, memtools, coretools)
 │   ├─ security.decide / guard_path
 │   └─ verify.verify / world_checks
 ├─ agent.OmegaAgent
 │   ├─ brain.SkillLibrary / SelfPlayEngine / NeuroSymbolicSolver
 │   ├─ brain.CodeVerifier / MathVerifier
 │   ├─ brain.index_repository  (AST symbols + call graph → memory)
 │   ├─ brain.repair.RepairEngine  (N1: localisation + 9 AST operators + pytest search)
 │   └─ core.distill.sleep_distill  → core/model_v2 checkpoint
 ├─ report.py
 └─ hw.detect
core/ (optional; requires torch)
 ├─ data.py  (clean corpus, file-level split, compressor baselines, provenance manifest)
 ├─ model.py (OmegaCore v0, LoopedOmegaCore v1, BitLinear/STE, RoPE, SwiGLU)
 ├─ model_v2.py (OmegaCoreV2: hash-ngram stem, patched loops, delta fast weights, copy gate)
 ├─ packed.py (2-bit ternary pack, LUT T-MAC)
 ├─ train.py (AdamW, wall-clock cosine schedule)
 ├─ distill.py (wake-sleep replay distillation)
 └─ engine.py / engine_v1_v2.py (Numba decode, ring-KV, ACT early-exit, streaming)
```

## Evidence map

| Claim | Evidence | Reproduced on this machine? |
|---|---|---|
| Agent-loop durability | `tests/test_recovery.py` | tests: yes (2026-11-19); benchmark: no |
| 180 passed / 1 skipped (active suite; 166 at the `EXP-M` gate) | `python -m pytest … -q` | **yes** (44.14 s) |
| Memory at 50k | `BENCHMARKS/results/memory.json` (older machine) | no |
| v2 vs zlib-9 (1.7482 vs 1.8370) | `EXPERIMENTS/EXP-005-m1-v2-core/showdown_results.json` | no |
| Native speed / greedy identity | `EXPERIMENTS/EXP-007…/results.json`, `EXP-008…/results.json` | no |
| Sealed 20/20 | `EXP-008…/results.json` | no |
| N0 baseline (`0/6`, `0/12`, `0.6667` strict recall@5) | `EXP-009…/results.json` | **yes** (10.405 s) |
| N1 repair (`6/6` N0, `7/7` same-family held-out) | `EXP-010…/results.json`, `EXP-011…/results.json` | **yes** (25.241 s / 27.782 s) |
| Wake-sleep distillation | `EXP-008…/results.json` | no |
| `fs.write` append fix | `tests/test_tools.py` | **yes** |

## Companion documents

- `RESEARCH/EXP-M-ARCHITECTURE.md` — `M0`–`M4` specification and exit gates.
- `RESEARCH/HANDOFF.md` — verified `EXP-M` state and the `EXP-N` boot sequence.
- `RESEARCH/EXP-N-PLAN.md` — pre-registered `EXP-N` hypotheses, gates and kill criteria.
- `EXPERIMENTS/EXP-009-n0-benchmark/README.md` — the sealed N0 suite and its baseline.
- `EXPERIMENTS/EXP-010-n1-ast-repair/README.md` — N1 on the frozen N0 code track.
- `EXPERIMENTS/EXP-011-n1-heldout/README.md` — N1 same-family surface-transfer probe.
- `PROJECT-STATE.md` — concise resume document.
- `PHASE-0-REPOSITORY-AUDIT.md` — historical audit (defect status is tracked in `PROJECT-STATE.md` §4).
