# EXP-001 — Do ternary weights make CPU decode faster, or only smaller?

**Status:** measured · **Decision:** keep int8 as the CPU baseline format; continue 2-bit only with a SIMD-LUT kernel (EXP-004)
**Script:** `ternary_bench.py` · **Raw results:** `results.json` (latest run)

## Question
Constitution §5.3 warns that low-bit weights "do not automatically make inference faster." Single-token decode is a
matrix–vector product, which should be bound by memory bandwidth. If so, fewer weight bytes should mean
proportionally less time, but only when a kernel actually exploits the packed format.

## Hypotheses
- **H1:** int8 (1 B/weight) is ~4x faster than fp32 (4 B/weight) on layers too big for cache.
- **H2:** 2-bit packed ternary (0.25 B/weight) is ~16x faster than fp32 with straightforward kernels.
- **H0 (alternative to H2):** unpacking 2-bit codes costs more compute than the bandwidth it saves, so the gain is far below 16x.

## Method
Ternary W ∈ {-1, 0, +1} (uniform, ~1/3 zeros), activations x ~ N(0, 1), batch 1. Every kernel computes y = Wx for
the same W and is checked against an fp32 reference (max relative error reported). Median of 30 runs after 3
warm-ups. Kernels: numpy BLAS fp32, torch fp32 (MKL), torch dynamic int8 (fbgemm), and four Numba/LLVM
multi-threaded kernels written here (int8, 2-bit bit-decode, 2-bit 256-entry LUT, 2-bit column-major).

**Hardware:** Intel i7-7820HQ (Kaby Lake, 4C/8T, AVX2, 8 MB L3), 8 GB RAM, Windows 11, Python 3.10, numpy 1.26,
torch 2.1.2+cpu, numba 0.62.

## Results (two independent runs; speedup vs fp32 numpy)

| Shape (out×in) | Kernel | Bytes/weight | Run 1 ms | Run 2 ms | Speedup (run 1 / run 2) | Max rel. error |
|---|---|---|---|---|---|---|
| 4096×4096 | fp32 numpy | 4 | 5.52 | 4.62 | 1.0 / 1.0 | 0 |
| 4096×4096 | int8 torch dynamic | 1 | 1.45 | 1.26 | 3.8 / 3.7 | **1.7e-2** (quantizes activations) |
| 4096×4096 | **int8 Numba (ours)** | 1 | **1.33** | **0.99** | **4.1 / 4.7** | 3.9e-7 |
| 4096×4096 | 2-bit decode | 0.25 | 4.99 | 2.05 | 1.1 / 2.3 | 4.1e-7 |
| 4096×4096 | 2-bit LUT | 0.25 | 6.09 | 2.82 | 0.9 / 1.6 | 4.3e-7 |
| 11008×4096 | fp32 numpy | 4 | 13.69 | 15.53 | 1.0 / 1.0 | 0 |
| 11008×4096 | int8 torch dynamic | 1 | 3.56 | 8.64 | 3.9 / 1.8 | 2.0e-2 |
| 11008×4096 | **int8 Numba (ours)** | 1 | **4.12** | **5.23** | **3.3 / 3.0** | 5.0e-7 |
| 11008×4096 | 2-bit decode | 0.25 | 9.91 | 7.78 | 1.4 / 2.0 | 4.8e-7 |
| 11008×4096 | 2-bit LUT | 0.25 | 10.38 | 10.44 | 1.3 / 1.5 | 4.7e-7 |
| 11008×4096 | 2-bit column-major | 0.25 | – | 12.34 | – / 1.3 | 1.9e-6 |

At 2048×2048 the whole int8 matrix (4 MiB) fits in L3, so int8 Numba reached 40 GB/s, above DRAM bandwidth.
Cache-resident results are not representative of large models and are excluded from the conclusions.

## Interpretation
- **H1 supported.** int8 gives 3.0–4.7x on DRAM-resident layers, close to the 4x byte ratio. fp32 sustains ~12–14 GB/s,
  this machine's practical DRAM bandwidth. Our int8 kernel matches or beats fbgemm's dynamic int8 while staying exact,
  because it doesn't quantize activations.
- **H2 rejected, H0 supported.** 2-bit kernels reach only 1.3–2.5x despite 16x fewer bytes. They run at ~1–2 GB/s of
  weight traffic, so they're limited by unpacking compute, not memory. The LUT variant is limited by its gathers.
  Transposing for row-wise vectorization didn't help (the auto-vectorizer apparently didn't produce the hoped-for code;
  the assembly hasn't been inspected yet).
- **Variance is high** (up to 2.4x run to run for some large-shape kernels) on a laptop CPU with turbo, thermals and
  background load. Conclusions rest only on effects much larger than the variance.

## What this means for the north-star targets
At ~13 GB/s, single-stream decode speed is capped at about bandwidth ÷ bytes per token:

| Model weights | fp32 | int8 | 2-bit (if bandwidth-bound) |
|---|---|---|---|
| 1 B | ~3 tok/s | ~13 tok/s | ~52 tok/s |
| 100 M | ~33 tok/s | ~130 tok/s | ~520 tok/s |
| 5 M | ~650 tok/s | ~2,600 tok/s | ~10,400 tok/s |

So **>10,000 tokens/s single-stream on this CPU requires the per-token weight traffic to stay ≲1.3 MB.** Ways there:
a model of only a few million parameters, weights resident in cache, batched or speculative decoding that amortizes
each weight read over many tokens, or a reasoning core that doesn't read all weights per token (sparse / conditional
compute). This turns the target into concrete engineering constraints (see RESEARCH/RC-001).

## Next experiments
1. **EXP-004:** a 2-bit kernel with AVX2 `pshufb` 16-entry LUTs (T-MAC / bitnet.cpp TL-style) in C. Blocked on a C
   compiler: Visual Studio 2019 is present but has no C++ build tools. Install "Desktop development with C++" or use clang.
2. Inspect the Numba LLVM IR/assembly of `t2_colmajor_matvec` to see why it didn't vectorize as intended.
3. Batch-size sweep (1, 4, 16, 64) to measure how fast low-bit kernels become compute-bound.
4. Pin the power plan, raise repetitions, and report confidence intervals.
