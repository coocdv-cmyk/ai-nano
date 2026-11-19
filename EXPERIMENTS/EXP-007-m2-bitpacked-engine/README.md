# EXP-007 — M2 Native 2-Bit Packed LUT, ACT Early-Exit, Self-Speculation & Streaming `v2`

- **Branch:** `research/exp-m`
- **Date:** 2026-11-18
- **Hardware:** Intel Core i7-7820HQ (CPU-only, 7 Numba threads), 7.41 GB RAM, Windows 10, Python 3.10.11
- **Script:** `run_exp.py` (`results.json`, wall time **17.44 s**)

## What M2 Implemented (`omega_nano/core/packed.py`, `engine_v1_v2.py`, `engine.py`)

1. **2-Bit Ternary Weight Packing & Multiplier-Free T-MAC LUT (`packed.py`):**
   - Packs `{-1, 0, +1}` block matrices into **2 bits/weight** (`uint8`, 4 weights/byte): **4× smaller** than `int8` and **16× smaller** than `fp32`.
   - Evaluates dot products via per-4-input group 256-entry float32 activation LUTs using additions/negations only (zero scalar multiplications inside the packed matvec loop).
2. **Unified Native CPU Inference for `arch='v0'`, `'v1'`, and `'v2'` (`engine.py` + `engine_v1_v2.py`):**
   - **`v1` (`LoopedOmegaCore`)**: True ACT early-exit halting with lazy KV backfill + lossless batched Self-Speculative MTP verification.
   - **`v2` (`OmegaCoreV2`)**: Modular O(1)-per-byte streaming engine (Causal Hash-Ngram Stem -> Depthwise Conv -> Strided Thought Patches -> Ring-KV Attention + incremental `DeltaFastWeights` -> incremental `CopyGate`), released via `nogil=True` + `ThreadPoolExecutor` for fast multi-core parallel batching without Parfors compile overhead.

## Measured Results (`results.json`)

| Architecture | Params | Block Weights RAM (Packed) | PyTorch CPU `tok/s` | Native Single `tok/s` | Native Batch-4 `tok/s` | Speedup vs PyTorch | Greedy Match |
|---|---:|---:|---:|---:|---:|---:|:---:|
| **`v0` Ternary** *(int8 / 2-bit LUT)* | 1.43M | 331.8 KB *(vs 5.31 MB fp32)* | 34.2 | **2,562.8** *(int8)* / **1,963.1** *(LUT)* | **7,311.2** *(int8)* / **4,512.7** *(LUT)* | **57.4×** *(LUT)* / **74.9×** *(int8)* | True |
| **`v1` Looped Ternary** *(ACT early-exit)* | 333K | 50.2 KB *(vs 802.8 KB fp32)* | 75.7 | **6,637.6** *(`1.159` avg loops)* | **18,535.1** | **87.7×** | True |
| **`v2` Streaming** *(real `EXP-005/v2_hybrid_ternary`)* | 870K | **122.9 KB** *(L2-cache resident!)* | 35.5 | **10,509.2** (`9.41 ms` TTFT) | **23,284.9** | **295.7×** | True |

## Honest Tradeoffs Observed

- **Int8 vs 2-bit Packed LUT at 1.3 MB:** While 2-bit packing compresses block weights **4× further than int8 (16× vs fp32)**, building the `256`-entry LUT per group on a single stream makes `v0` LUT `1,963.1 tok/s` slightly slower than `int8` SIMD matvec (`2,562.8 tok/s`) when matrices already fit inside L3 cache (`1.3 MB`). At larger parameter counts exceeding cache, 2-bit bandwidth wins dominate.
- **Self-Speculative MTP on Untrained Heads (`1,408.1 tok/s`, `1/595` accepted):** Untrained random MTP heads fail draft verification and add verification overhead; lossless speculative MTP only speeds up inference once the MTP heads are trained via Wake-Sleep distillation (`MTP acceptance rate > 25%`). Consequently, production inference should gate speculation on calibrated draft acceptance.
