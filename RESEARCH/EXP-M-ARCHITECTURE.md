# EXP-M — Architectural Evolution Specification (`OMEGA-NANO V1` Program)

**Branch:** `research/exp-m`  
**Status:** **COMPLETE (`M0 -> M4` verified across `EXP-005`, `EXP-006`, `EXP-007`, `EXP-008`)**  
**Constitution anchors:** `OMEGA-NANO-VISION-V3-ENGLISH.md` (§0–§5, §7, §11), `فهم الخطة` (Program 2 & Program 3)

---

## 1. Central Objective

Construct an ultra-compact, CPU-first integrated AI system (`OMEGA-NANO V1`) that breaks the conventional scaling-only bottleneck by combining:
1. **Cache-resident recurrent neural compute** (depth-as-time via `LoopedOmegaCore`, test-time fast weights, and entropy-driven patching),
2. **External semantic + procedural executable memory** (`MemoryFabric` + `SkillLibrary`),
3. **Zero-hallucination formal & execution verification** (`MathVerifier`, `CodeVerifier`, `verify.py`), and
4. **Wake-Sleep continual learning** (replay distillation from verified trajectories back into compact weights).

---

## 2. Core Components & Information Flow

1. **Input & Entropy Boundary (`M1`):** Dynamic patching over UTF-8 bytes to compress predictable boilerplate and extend effective context on Arabic, English, and code without giant vocab matrices.
2. **Cache-Resident Neural Core (`omega_nano/core/model.py`):** Weight-shared recurrent blocks (`LoopedOmegaCore`) with adaptive halting (ACT), Multi-Token Prediction (MTP), and ephemeral Test-Time Fast Weights (`M1`) sized to stay resident in CPU L2/L3 cache.
3. **Multiplier-Free CPU Runtime (`omega_nano/core/engine.py`):** Rolling Ring-KV cache (`M0`), ACT early-exit native path, and bit-packed/LUT matvec + self-speculative multi-token verification (`M2`).
4. **External Memory & Safety (`omega_nano/memory.py`, `omega_nano/brain/`):** Semantic embeddings replacing lexical hashing (`M3`), `EXP-L L3` behavioral retrieval guard, `EXP-L L1` mapping-consistency drift detection, `SkillLibrary`, and formal/sandboxed verifiers.
5. **Control & Governance (`omega_nano/engine.py`, `store.py`, `security.py`, `tools/`):** Durable write-ahead checkpointing, dynamic idempotency (`fs.write` append fixed in `M0`), sanitized subprocess environment (`shell.run` fixed in `M0`), and untrusted-output isolation (`reasoner.py` fixed in `M0`).

```
Goal / Task
  ├─► MemoryFabric (Semantic + L3 Behavioral Guard + AST Repo Graph)
  ├─► SkillLibrary (0.1 ms verified parameterized tactics)
  └─► LoopedOmegaCore v2 (Entropy-Patched + Fast Weights + Ring-KV CPU Runtime)
        │  proposes B diverse candidate plans / edits / proofs
        ▼
      Verifiers (MathVerifier / CodeVerifier / Disk World Checks)
        ├─ fail ─► exact counterexample / traceback ─► adaptive re-loop / revise
        └─ pass ─► SkillLibrary + Replay Buffer ─► Wake-Sleep distillation into Core
```

---

## 3. Learning, Reasoning, Memory, Resource & Failure Model

* **Learning procedure:** Supervised compression on clean, copyleft-free multi-domain data (`core/data.py` fixed in `M0`) + Wake-Sleep self-play replay distillation on 100%-verified trajectories only.
* **Reasoning procedure:** Adaptive compute routing (`t0` plan/skill cache -> `t1` deterministic rules -> `t2` looped core + verifier-guided search).
* **Memory model:** Short-term fast weights inside the core (`O(1)` RAM per step) + persistent SQLite FTS5 + int8 vector fabric outside weights (`~1.2 KB/record`).
* **Resource model:** Tier 1 target is ordinary CPU-only laptops (4 cores, 7–16 GB RAM, `< 10 MB` active block weights). All benchmarks report `single-stream tok/s`, `aggregate tok/s`, `TTFT`, `peak RAM`, and quality (`bpb` / exact match) together.
* **Failure handling:** Unverified outputs never receive `PROVEN_EXACT` or `TESTS_PASSED`; non-idempotent steps (`fs.write append`, `shell.run`) never blindly auto-retry; historical `constructive/` (`EXP-H..L`) modules remain frozen and unmodified.

---

## 4. Work Packages & Exit Gates (`M0 -> M4`)

| Stage | Deliverable | Exit Gate |
|---|---|---|
| **M0** | Branch unification (`local-v0.3-brain` + `master`), security/idempotency fixes (`reasoner.py`, `fs.write`, `shell.run`), clean corpus (`core/data.py`), CI | All `omega_nano` + `brain` + `core` tests green; 0 `site-packages`/copyleft in corpus |
| **M1** | `LoopedOmegaCore v2`: dynamic entropy patching + test-time fast weights | Beats `v0` & `zlib-9` on matched clean split & wall-clock; passes long-range (`>2000` byte) recall test |
| **M2** | Native Numba `v1/v2` engine with early-exit, bit-packed LUT/self-speculation | Greedy-identical to PyTorch; measurable single-stream & aggregate `tok/s` speedup on CPU |
| **M3** | Semantic `MemoryFabric.embed()` + `L3`/`L1` guards + neural-guided `SelfPlayEngine` + AST symbol graph | Raises blind coding solve rate (`> 0/5`), Cycle N > Cycle 0 without forgetting, cross-lingual recall |
| **M4** | Integrated `OMEGA-NANO V1` offline agent + full Capability & Efficiency Matrix | End-to-end offline tasks verified with documented RAM, speed, accuracy, and honest gaps |
