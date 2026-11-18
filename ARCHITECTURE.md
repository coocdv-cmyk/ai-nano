# Architecture — as implemented (0.3.0-dev / EXP-M M0)

This describes what exists, not what is planned. The target architecture (16 layers) is in the constitution §4.
Corrected 2026-11-18 by the Phase 0 audit (`PHASE-0-REPOSITORY-AUDIT.md`) and updated on `research/exp-m` (`M0`) after integrating `local-v0.3-brain` (`LoopedOmegaCore`, `omega_nano/brain/`, `reasoner.py` fix) and resolving audit defects #1, #3, #6, #10.

Status vocabulary: **measured** (implemented and benchmarked) · **implemented** (tested, not benchmarked) · **prototype**
(works on a narrow path) · **partial** (some sub-capability missing) · **absent**.

| # | Constitution layer | Status | Where |
|---|---|---|---|
| 1 | Neural core | **v0 measured** (EXP-003, below zlib) + **v1 prototype** (`LoopedOmegaCore`: ACT + MTP) | `omega_nano/core/model.py` |
| 2 | Adaptive compute controller | prototype: 3 planning tiers + ACT halting in `v1` | `engine._plan`, `LoopedOmegaCore` |
| 3 | Inference runtime | **implemented**: Numba ring-KV + row4-unrolled matvec (`v0`); int8 retrieval kernel | `core/engine.py`, `kernels.py` |
| 4 | Memory fabric | **implemented**, measured to 50k records | `memory.py` |
| 5 | Knowledge ingestion | bulk ingest + dedup; clean copyleft-free corpus builder | `memory.remember_many`, `core/data.py` |
| 6 | Retrieval and search | **implemented**: lexical + vector + fuzzy, RRF | `memory.recall` |
| 7 | Reasoning and planning | rule reasoner (EN/AR) + hardened LLM adapter + `brain` self-play | `reasoner.py`, `brain/` |
| 8 | Tool execution | **implemented**: typed, validated, bounded, dynamic idempotency | `tools/` |
| 9 | Computer interaction | not started (Phase 5) | – |
| 10 | Software engineering engine | code verifier & blind code suite started; no repo indexer yet | `brain/verifiers.py` |
| 11 | Multimodal | not started (Phase 7) | – |
| 12 | Verification and critique | **implemented**: world checks + SymPy/mpmath + isolated Python | `verify.py`, `brain/verifiers.py` |
| 13 | Agent orchestration | **implemented** for single-agent durable tasks | `engine.py`, `store.py` |
| 14 | Security and governance | **implemented**: modes, scopes, approvals, sanitized shell env, stop | `security.py`, `tools/shell.py` |
| 15 | User interface | CLI + markdown reports + `brain` subcommands | `cli.py`, `report.py` |
| 16 | Evaluation loop | benchmarks + experiment records (`EXP-001..004`, `constructive/`) | `BENCHMARKS/`, `EXPERIMENTS/`, `constructive/` |

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
| t2 | local OpenAI-compatible LLM (optional) | model-dependent |

## Known limitations (updated M0)
- The rule reasoner covers a fixed set of intents. Anything else is blocked unless an LLM backend is configured.
- In-process tool timeouts abandon the worker thread instead of killing it. `shell.run` uses a real process timeout and now executes with a minimal sanitized environment allowlist (though it is still not a full OS container/sandbox).
- One worker per task (lease + PID liveness). There is no distributed queue yet.
- `fs.write` now marks `mode="append"` non-idempotent via `dynamic_idempotent` so verification failures never duplicate appended text on retry; `LocalLLMReasoner.revise` wraps tool outputs under `untrusted_tool_observation`.
- The neural core (`v0` / `v1` prototype) still operates at raw byte level (`vocab=256`, `ctx=256`) and has not yet cleared `zlib-9` on a full training run (`M1` target).
- The native Numba engine now uses a rolling ring-KV cache + row4-unrolled matvec for `v0`, while `v1` (`LoopedOmegaCore`) is not yet wired into the native CPU runtime (`M2` target).
- Memory retrieval is brute-force vector search using feature-hashing (`crc32`) rather than learned semantic embeddings (`M3` target).

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
  → embed (word uni/bigrams + char trigrams → signed feature hashing, 512-d)
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
 │   ├─ memory.MemoryFabric  (SQLite FTS5 + int8 matrix)
 │   ├─ reasoner.RuleReasoner / LocalLLMReasoner
 │   ├─ tools.default_registry  (fs, mathx, shell, memtools, coretools)
 │   │    └─ tools.base.Tool contract
 │   ├─ security.decide / guard_path
 │   └─ verify.verify / world_checks
 ├─ report.py
 └─ hw.detect
core/ (optional; requires torch)
 ├─ data.py  (corpus, file-level split, compressor baselines)
 ├─ model.py (OmegaCore, BitLinear/STE, RoPE, SwiGLU)
 ├─ train.py (AdamW, wall-clock cosine schedule)
 └─ engine.py (Numba decode loop, KV cache, batch)
```

## Evidence map

| Claim | Evidence | Reproduced in this audit? |
|---|---|---|
| Agent loop durability | `tests/test_recovery.py`, `BENCHMARKS/results/engine.json` | Tests: yes. Benchmark: no. |
| 126 passed / 1 skipped | `python -m pytest -q` | **yes** (16.54 s) |
| Memory at 50k | `BENCHMARKS/results/memory.json`, EXP-002 | no |
| Core quality/speed | `EXPERIMENTS/EXP-003/results.json` | Quality: no. Speed: fresh spot-check only. |
| Kernel speedups | `EXPERIMENTS/EXP-001/results.json` | no |
| `fs.write` append bug | audit §8.1 reproduction | **yes** |

## Companion documents (2026-11-18)

- `PHASE-0-REPOSITORY-AUDIT.md` — factual audit and confirmed defects.
- `RESEARCH-OPPORTUNITIES.md` — research portfolio and the three highest-value directions.
- `ORIGINAL-LEARNING-MECHANISM.md` — the rapid-acquisition mechanism (Skill Fabric / BIRS).
- `OMEGA-EXPERIMENT-001.md` — the first falsifiable experiment.
- `RESEARCH-ROADMAP.md` — staged research plan with gates and kill criteria.
- `BASELINE-AND-BENCHMARK-PLAN.md` — comparison protocol.
- `LICENSE-AND-PROVENANCE-REGISTER.md` — licences and corpus provenance.
- `PROJECT-STATE.md` — concise resume document.
