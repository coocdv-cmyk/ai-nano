# RC-004 — The Self-Evolving Mind: Continual Verifiable Self-Play, Looped Core & Zero-Hallucination Search

**Status:** active design + Phase A prototype (`OmegaCore v1`, `omega_nano/brain/`, `EXP-004/005`)  
**Constitution anchors:** §0.1, §2.2, §2.3, §3.1, §3.2, §3.5, §4.2, §6, §7

---

## 1. Research Question
Can a compact neural core (`< 0.0625%` of frontier parameter scale, targeting `10M–150M` parameters / `~3–40 MB` at 2-bit ternary) coupled with:
1. **Recurrent weight-shared depth + adaptive halting + multi-token prediction** (`OmegaCore v1`),
2. **External hierarchical semantic/graph memory + procedural skill compilation** (`MemoryFabric` + `Tier 0+`),
3. **Execution-guided test-time search & formal/symbolic verifiers** (`SymPy`, `mpmath`, isolated `pytest`), and
4. **Continual verified self-play** (Curriculum Proposer → Blind Solver → Formal Verifier → Skill Compiler + Replay Buffer),

continuously improve over time without human labeling, eliminate unverified hallucinations in verifiable domains, build and verify complete software projects, and solve complex mathematical problems blindly?

---

## 2. Why Conventional Tiny Transformers Fail
`EXP-003` (`OmegaCore v0`, 3.34M params, 4 fixed layers, byte-level) established two concrete baselines:
- Quality: `1.894 bpb` (`fp32`) and `2.055 bpb` (`ternary`) on Python stdlib — beating a trained order-2 n-gram (`2.579`) but trailing `zlib -9` (`1.830`).
- Speed: `841 B/s` single-stream (`810` at 7 threads) and `2,865 B/s` aggregate across 14 streams.

Root causes observed directly in code and measurements:
1. **Fixed shallow depth (`layers=4`):** Only 4 sequential composition steps per byte regardless of problem difficulty.
2. **Byte-level Arabic/Code tax:** UTF-8 Arabic characters use 2 bytes/letter (`~25 words` fit in `ctx=256`); predictable boilerplate consumes full Transformer passes.
3. **Zero search or verification feedback at inference:** Single-pass autoregressive sampling cannot catch or repair an algebraic or syntax error mid-stream.
4. **Static weights + exact-hash cache:** `t0:plan-cache` only hits identical normalized strings; solved problems do not generalize into reusable parameterized skills or replay training traces.
5. **Engine bottlenecks (`omega_nano/core/engine.py`):** Window re-encoding (`ctx//2` recompute every `ctx//2` steps = ~2x work), per-step heap allocations, and independent weight reads across batched streams (`14x` redundant DRAM/cache reads).

---

## 3. Core Architecture: The 4-Loop Flywheel

```
          ┌────────────────────────────────────────────────────────┐
          │   Loop 4: Sealed Blind Benchmark Vault                 │
          │   (No peeking at paper/reference; post-hoc evaluation) │
          └───────────────────────────┬────────────────────────────┘
                                      ▲
┌─────────────────────────────────────┴─────────────────────────────────────┐
│ Loop 1: Looped Neural Core (OmegaCore v1)                                 │
│  • Weight-shared recurrent block(s) × K loops (deep reasoning, tiny RAM)  │
│  • Loop-step embeddings + residual input anchor + ACT halting gate        │
│  • Multi-Token Prediction (MTP) heads (plan chunks & UTF-8 pairs)         │
└───────────────────┬─────────────────────────────────▲─────────────────────┘
                    │ candidates / tactics            │ verified replay traces
                    ▼                                 │
┌───────────────────────────────────────┐   ┌─────────┴─────────────────────┐
│ Loop 2: Zero-Hallucination Search     │   │ Loop 3: Every-Second Learning │
│  • Generate B diverse attempts/edits  │   │  • Fast (ms): Skill Compiler  │
│  • Run SymPy / mpmath / AST / pytest  ├──►│    & Lemma Memory (no weights)│
│  • Feed exact counterexample/traceback│   │  • Slow: Replay distillation  │
│  • Output ONLY proof-carrying claims  │   │    with anti-forgetting mix   │
└───────────────────────────────────────┘   └───────────────────────────────┘
```

### Key Architectural Invariants
1. **Proof-Carrying Output Rule (Zero Hallucination):** Never present an unverified mathematical derivation or code patch as verified truth. Every claim carries an explicit status: `PROVEN_EXACT`, `TESTS_PASSED`, `NUMERICAL_VERIFIED`, `REFUTED` (with counterexample), or `UNVERIFIED`.
2. **Contamination-Free Blind Evaluation:** Reference solutions/papers are physically isolated from the solver's workspace, tools, and memory index during problem solving.
3. **Separation of Knowledge and Compute:** Encyclopedic facts, repo symbol graphs, and compiled skills live in external storage (`MemoryFabric` / SQLite / disk); neural core weights remain tiny and cache-friendly.

---

## 4. Falsifiable Hypotheses & Cheapest Experiments

| ID | Hypothesis | Falsifier / Failure Criterion | Success Criterion |
|---|---|---|---|
| **H1 (Engine Speed)** | Removing window re-encode via Rolling Ring KV Cache + pre-allocated buffers + weight-shared batched GEMM increases aggregate & single-stream throughput significantly on `EXP-003` weights with identical short-context logits. | Single-stream speedup `< 1.3x` or aggregate `< 2x` or logits diverge `> 2e-3` from PyTorch. | Validated speedup on `EXP-003` checkpoints + all equivalence tests passing. |
| **H2 (Looped Core Quality)** | At equal parameter count (~3.3M) and equal wall-clock on Python stdlib, a weight-shared looped core with step embeddings, input anchoring, and MTP achieves lower `val_bpb` than `OmegaCore v0` (`1.894`). | `val_bpb` higher than `v0` across matched budgets. | Statistically reproducible drop in `val_bpb` per parameter. |
| **H3 (Self-Play Gain)** | Execution-guided search + skill compilation raises solve rate on a held-out blind math + Python suite across self-play cycles (`cycle N > cycle 0`) without forgetting previously solved skills. | Cycle accuracy stagnates (`<= cycle 0`) or prior tasks regress. | Monotonic or net positive improvement measured on blind eval with zero regressions. |
| **H4 (Semantic Memory)** | A compact ONNX multilingual encoder + AST/symbol graph beats CRC32 hashing on true synonym & EN↔AR cross-lingual benchmarks at `< 20 ms` CPU latency. | Fails paraphrase/cross-lingual recall or latency exceeds budget. | High recall on non-overlapping synonym/cross-lingual queries (RC-002). |

---

## 5. Honest Compute & Resource Envelope
- **What runs 100% locally on CPU (`i7-7820HQ`, 7.41 GB RAM):** Engine optimizations, `OmegaCore v1` 3M–20M comparative ablations, symbolic/execution verification, self-play skill compilation, and all inference after training.
- **What requires scaled compute (authorized GPU burst or teacher distillation):** Scaling the neural core to `30M–150M` parameters over millions of distilled & self-play verified traces cannot finish in practical wall-clock on a single 4-core CPU (`RC-003`). Research path: prove architecture efficiency at tiny scale locally first, distill/self-play verified traces, scale training only after architecture gates pass.
