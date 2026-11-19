# OMEGA-NANO — Sync & Next-Phase Handoff (`EXP-M -> EXP-N`)

> **Status note (2026-11-19):** `N0` and `N1` are complete and recorded on `research/exp-n`; resume from
> `PROJECT-STATE.md`, not from this historical handoff. N1's as-built verification runs the suite in an isolated
> workspace copy — it is **not** an OS sandbox; the deviation is recorded in `RESEARCH/EXP-N-PLAN.md` §3.

**Canonical Remote:** `https://github.com/coocdv-cmyk/ai-nano.git` (`origin`)  
**Policy Rule:** Never push unverified code. Push to `origin` only after full tests pass and benchmark outputs are recorded.

---

## 1. Current Verified State (`research/exp-m` @ `3df015c`, `master` @ `03149eb`)

Completed **Program `EXP-M` (`M0 → M4`)** verified on CPU-only hardware (`Intel i7-7820HQ`, `7.41 GB RAM`, Windows 10, Python `3.10.11`):

- **`M0` (`5a777a0`)**: Unified local `v0.3` brain/core with GitHub `origin/master` (`EXP-L L3–L7`); fixed idempotency (`fs.write append`), `shell.run` env leak, `reasoner.py` prompt-injection isolation, clean PSF corpus (`core/data.py`), and CI workflow.
- **`M1` (`2f2ab4d`, `EXP-005`)**: `OmegaCoreV2` (`omega_nano/core/model_v2.py`) — hash-ngram byte stem + strided coarse thought patches + Delta Fast Weights + causal CopyGate. At 2.00M params (`5 min` CPU) hit `1.7482 bpb` (beating `zlib-9` `1.8370` & `v0_3.34M` `2.7841`) with `0.9609–1.0000` 2048-byte recall.
- **`M3` (`d6c0161`, `EXP-006`)**:
  - Bilingual AR/EN/Code semantic encoder + `EXP-L` `L1` (`SemanticDriftMonitor`) + `L3` (`verify_behavioral_candidates`) in `omega_nano/semantic.py` + `omega_nano/memory.py`.
  - Compositional + AST-IO + CEGIS program synthesizer & persisted online `NeuralPolicyRanker` (`omega_nano/brain/synthesizer.py`, `skills.py`, `evolver.py`).
  - AST Python symbol & call-graph indexer (`omega_nano/brain/repo_graph.py`).
  - Measured `10/10` on `BlindVault` + `10/10` on `BlindVault.heldout()`, Cycle 1->2 (`1.08 -> 1.00` avg attempts), cross-name policy transfer (`2 -> 1` attempt), 3/3 drift rejection & self-healing, `35` files / `374` symbols indexed in `183.69 ms`.
- **`M2 + M4` (`3df015c`, `EXP-007` & `EXP-008`)**:
  - `omega_nano/core/packed.py`: 2-bit ternary packing (4× vs `int8`, 16× vs `fp32`) + multiplier-free T-MAC LUT.
  - `omega_nano/core/engine_v1_v2.py` & `engine.py`: Native Numba inference for `v0`, `v1` (ACT early-exit + lossless MTP speculation), and `v2` streaming (`nogil=True` + `ThreadPoolExecutor`). Reached **`10,509.2 – 13,209.0 tok/s`** single-stream (`23,284 – 25,177 tok/s` batch-4, `2.23–9.41 ms` TTFT, `122.9–135.2 KB` packed block weights, 100% greedy match vs PyTorch).
  - `omega_nano/core/distill.py` & `omega_nano/agent.py`: Wake-Sleep replay distillation (`val_bpb 6.7749 -> 3.0872` in `6.104 s` CPU on 837K params) + unified offline `OmegaAgent` & `cli.py` (`agent`, `brain distill`, `brain eval --heldout`, `brain index-repo`).
  - Full active test suite verified: **`166 passed, 1 skipped`** in `38.96 s`; `constructive/` remains untouched.

---

## 2. Roadmap for Next Phase (`EXP-N: Ultralight Reasoning Dominance`)

Goal: Keep weights ultra-tiny (`< 2M`–`5M`, L2/L3 cache-resident, zero GPU) while moving capability from fixed algorithmic synthesis to open repository repair, trained self-speculation, learned semantic embeddings, and tree search:

1. **`N1` — Open AST Program Synthesis & Local Repo Repair (`SWE-mini`)**:
   - Add AST-level mutation/patch operators (`ast.NodeTransformer`) driven by pytest tracebacks and failing assertions.
   - Connect `RepoGraph` context retrieval -> localized patch generation -> sandboxed `CodeVerifier` validation on multi-file broken Python modules.
2. **`N2` — Self-Generated Verified Corpus Scaling + Trained Speculative `MTP`**:
   - Note from `EXP-007`: untrained MTP heads drop throughput (`1,408.1 tok/s`, `1/595` accepted).
   - Generate thousands of verifier-proven `(spec, AST/trace, verified_code)` frames offline; train `OmegaCoreV2`/`v1` MTP heads until draft acceptance exceeds `40%+` with an adaptive speculation threshold gate (`accept_rate >= 0.25`).
   - Use distilled `OmegaCoreV2` as a strong open-ended proposal generator inside `NeuroSymbolicSolver`.
3. **`N3` — Tiny Learned Bilingual Metric Encoder + Hybrid Intent Router**:
   - Upgrade `omega_nano/semantic.py` from static concept anchors to a compact (`< 500 KB` `int8`) contrastive AR/EN/Code bi-encoder trained cleanly on bilingual/code pairs so arbitrary unseen synonyms generalize smoothly.
   - Pair `RuleReasoner` with a lightweight calibrated classifier (`hybrid intent + argument extractor`) while preserving zero-hallucination refusal on unparseable/out-of-scope goals.
4. **`N4` — Best-First / MCTS Execution-Guided Tree Search**:
   - Upgrade linear `attempt 1..N` loops in `SelfPlayEngine` to value-guided branching search (`NeuralPolicyRanker` + incremental partial-test pruning) for multi-step algorithmic and multi-function synthesis.
5. **`N5` — Held-Out Benchmark Suite V2 & Local Lightweight UI/API**:
   - Establish an independent, contamination-free `EXP-N` benchmark covering repo bug-fixing, multi-step bilingual workflows, trained MTP speedups, semantic zero-shot recall, and strict RSS/latency budgets before shipping `V2`.

---

## 3. Exact Work-PC Boot Sequence

```bash
git fetch --all
git checkout research/exp-m
git pull origin research/exp-m
python -m pytest tests/test_agent_v1.py tests/test_brain.py tests/test_core.py tests/test_engine.py tests/test_llm.py tests/test_memory.py tests/test_reasoner.py tests/test_recovery.py tests/test_security.py tests/test_state.py tests/test_tools.py -q
```
