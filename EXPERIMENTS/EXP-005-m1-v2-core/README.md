# EXP-005 (`M1`) — OmegaCore v2: Patched, Fast-Weight Plastic & Associative-Copy Core

**Status:** measured (`M1` exit gate **PASSED**) · **Code:** `omega_nano/core/model_v2.py` · **Hardware:** Intel i7-7820HQ (4C/8T), 7.41 GB RAM, CPU-only

---

## Hypothesis
Replacing flat byte-only Transformer decoding (`OmegaCore v0`) with:
1. A causal multi-scale n-gram hash stem + depthwise conv (`O(T)`),
2. Strided thought patching (`patch_size=4`, 16× cheaper attention, 4× effective context),
3. Test-time Delta-Rule Fast Weights (`O(H * d_f^2)` plastic state), and
4. An `O(1)` Numba-accelerated causal associative suffix `CopyGate`

will beat `v0` at matched parameter & wall-clock budget, beat `zlib-9` (`1.8370 bpb`) on the clean `12.2 MB` PSF stdlib split, and solve long-range (`2,048`-byte) recall where `ctx=256` models fail.

---

## Part 1 — Matched-Budget Ablation (`~870K` params, `ctx=256`, `90s` CPU each)

Source: `results.json`.

| Arm | Params | Steps | Val `bpb` | English `ctx=256` | Arabic `ctx=256` | Long-Range 2,048b Acc |
|---|---|---|---|---|---|---|
| `v0_baseline` | 869,504 | 152 | 3.0608 | 3.9427 | 12.1485 | 0.0052 |
| `v2_pure_neural` *(no CopyGate)* | 869,650 | 138 | **2.4548** | **3.5855** | 10.5712 | 0.0052 |
| `v2_hybrid_ctx256` *(fp32)* | 869,916 | 126 | **2.2860** | 3.6418 | **7.7667** | **1.0000** (`1.0644 bpb`) |
| `v2_hybrid_ternary` *(BitLinear)* | 869,916 | 116 | 2.3720 | 3.6902 | 7.7998 | **1.0000** (`1.0901 bpb`) |

- Even **without** `CopyGate`, `v2_pure_neural` outperformed `v0` by **−0.606 bpb** at matched parameter count and wall-clock.
- With `CopyGate` restricted to the **same 256-byte window**, `v2_hybrid_ctx256` reached `2.2860 bpb` (**−0.775 bpb** vs `v0`), and at `2,048` bytes achieved **100% top-1 needle recall** across Python→Arabic bindings.

---

## Part 2 — 5-Minute Showdown vs `3.34M` `v0` and `zlib-9` (`1.8370 bpb`)

Source: `showdown_results.json`.

| Arm | Params | Budget | Steps | Final Val `bpb` | Long-Range 2,048b Acc |
|---|---|---|---|---|---|
| `v0_3.34M_5min` | 3,344,640 | 308.8s | 146 | 2.7841 | 0.0104 |
| **`v2_2.00M_5min` (40% smaller)** | **2,002,204** | **306.3s** | **276** | **1.7482** (**beats `zlib-9` 1.8370**) | **0.9609** (`0.8633 bpb`) |

- `v2_2.00M_5min` crossed `zlib-9` (`1.8223 < 1.8370`) at **185 seconds** and finished at **`1.7482 bpb`** — over **1.03 bpb** ahead of `v0_3.34M` (`2.7841`), with **40% fewer parameters**.

---

## Honest Limits & Pivot Decision
- `eval_avg_loops` settled around `1.023–1.031` under ACT penalty, meaning loop 2 contributed minimally; further depth efficiency belongs in `M2`.
- Arabic prose zero-shot compression stayed high (`7.77–11.12 bpb`) because the training corpus was Python-only.
- **Strategic takeaway:** Even beating `zlib-9` (`1.7482 bpb`), 5 minutes of next-byte LM training does **not** yield general coding or reasoning intelligence (`sample_def` still trails spaces). Per the project's non-traditional roadmap, we **stop scaling raw-byte LM training** here and pivot immediately to **Programmatic / Execution-Guided Synthesis, Semantic Concept Memory, and Wake-Sleep Skill Compilation (`M3`)**, using `v2` only as a compact learned policy/compression substrate.
