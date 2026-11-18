# Architecture — as implemented (0.2.0-dev)

This describes what exists, not what is planned. The target architecture (16 layers) is in the constitution §4.
Corrected 2026-11-18 by the Phase 0 audit (`PHASE-0-REPOSITORY-AUDIT.md`); the previous version wrongly said the neural
core was "not started" after `omega_nano/core/` had been implemented and measured as EXP-003.

Status vocabulary: **measured** (implemented and benchmarked) · **implemented** (tested, not benchmarked) · **prototype**
(works on a narrow path) · **partial** (some sub-capability missing) · **absent**.

| # | Constitution layer | Status | Where |
|---|---|---|---|
| 1 | Neural core | **v0 measured**: own 3.34 M-param byte-level decoder; below zlib in quality | `omega_nano/core/`, `EXPERIMENTS/EXP-003-omega-core-v0/` |
| 2 | Adaptive compute controller | prototype: 3 planning tiers | `engine._plan` |
| 3 | Inference runtime | prototype: native Numba decode loop with KV cache; int8 retrieval kernel in production | `core/engine.py`, `kernels.py` |
| 4 | Memory fabric | **implemented**, measured to 50k records | `memory.py` |
| 5 | Knowledge ingestion | bulk ingest + dedup; no document parsing yet | `memory.remember_many` |
| 6 | Retrieval and search | **implemented**: lexical + vector + fuzzy, RRF | `memory.recall` |
| 7 | Reasoning and planning | rule reasoner (EN/AR) + LLM adapter + revision | `reasoner.py` |
| 8 | Tool execution | **implemented**: typed, validated, bounded | `tools/` |
| 9 | Computer interaction | not started (Phase 5) | – |
| 10 | Software engineering engine | not started (Phase 4); shell tool exists | – |
| 11 | Multimodal | not started (Phase 7) | – |
| 12 | Verification and critique | **implemented** for world-checkable effects | `verify.py` |
| 13 | Agent orchestration | **implemented** for single-agent durable tasks | `engine.py`, `store.py` |
| 14 | Security and governance | **implemented**: modes, scopes, approvals, audit, stop | `security.py` |
| 15 | User interface | CLI + markdown reports | `cli.py`, `report.py` |
| 16 | Evaluation loop | benchmarks + experiment records | `BENCHMARKS/`, `EXPERIMENTS/` |

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

## Known limitations
- The rule reasoner covers a fixed set of intents. Anything else is blocked unless an LLM backend is configured.
- In-process tool timeouts abandon the worker thread instead of killing it. `shell.run` uses a real process timeout.
- `shell.run` uses an allowlist and deny patterns, **not an OS sandbox**. Allowlisted interpreters (python, node) can run
  arbitrary code and inherit the full process environment. That is why `EXECUTE` needs approval in every mode except
  `autonomous`. See SECURITY.md.
- One worker per task (lease + PID liveness). There is no distributed queue yet.
- `fs.write` declares `idempotent=True` for every mode, but `mode="append"` is not idempotent. A verification failure can
  therefore append the same content up to `max_retries` times. Reproduced 2026-11-18; see audit §8.1.
- `LocalLLMReasoner.revise` serializes the failed step (including raw `output`) into the prompt. Tool output can therefore
  influence a revised plan only on this path, contrary to invariant 5. See audit §8.2.
- The neural core is 3.34 M parameters, trained on ~20 MB for 30 minutes. It is a pipeline proof, not a usable language
  model: 1.89–2.06 bits/byte on held-out code, worse than zlib (1.83).
- The native engine re-encodes half the context window every `ctx/2` generated tokens and uses scalar matvecs; both are
  known, measurable speed losses.
- Memory retrieval is brute-force vector search: linear in record count, no ANN index.

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
