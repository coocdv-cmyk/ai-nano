# RC-001 — Decomposing ">10,000 tokens/s on an ordinary CPU"

Research card per constitution §7. **Status:** analysis + first measurement (EXP-001).

1. **Question.** Under what conditions can OMEGA-NANO generate >10,000 tokens/s locally on a CPU-only laptop?
2. **Why it matters.** Speed is a north-star target (§0.2, §5.2) and determines whether long autonomous work is cheap enough to run locally.
3. **State of the art (to verify with a literature pass).** Low-bit CPU inference (bitnet.cpp, T-MAC LUT kernels, llama.cpp
   k-quants) and speculative and multi-token decoding. Exact figures have not been checked in this session and are not cited here.
4. **Known limit (measured).** Single-stream decode reads every weight once per token. This machine sustains ~13 GB/s, so
   tokens/s ≤ 13e9 / (bytes of weights touched per token).
5. **Core hypothesis.** 10k tok/s is reachable when bytes touched per token ≤ ~1.3 MB. That means either a small core
   (≈5 M ternary or ≈1.3 M int8 parameters touched per token) or amortizing weight reads across many tokens.
6. **Alternative routes.**
   - **A. Tiny cache-resident core.** A few million parameters fit in L2/L3, and bandwidth jumps (EXP-001 saw 40 GB/s
     cache-resident). Capability per parameter becomes the question: route hard steps to tools, memory and a larger model.
   - **B. Batched / speculative decoding.** One weight read serves k tokens (draft-and-verify, multi-token heads).
     Aggregate throughput is not the same as single-stream speed, and the two must be reported separately (§5.2).
   - **C. Conditional compute.** Mixture-of-experts or early exit touches a fraction of the weights per token.
   - **D. Non-generative paths.** Many "tokens" in agent work are tool output or deterministic code: the math engine and
     plan cache already produce results without generation (the plan-cache tier plans in ~0.09 ms).
7. **Cheapest falsifying experiments.** (a) A 2-bit SIMD-LUT kernel reaching ≥8x over fp32 on a 4096² layer (EXP-004);
   (b) the measured tokens/s of a 5 M-parameter ternary model end to end; (c) acceptance rate of speculative drafts on
   agent-style text.
8. **Success criterion.** A reproducible run with a stated model, hardware, batch size, context length and measurement
   method showing ≥10k tok/s, reported as single-stream or aggregate.
9. **Failure criterion.** If even cache-resident 2-bit kernels stay compute-bound below ~2 GB/s effective, route A is capped
   and the work shifts to B and C.
10. **Result so far.** int8 delivers the expected ~4x over fp32. Naive 2-bit kernels reach only 1.3–2.5x (compute-bound).
    See EXP-001.
11. **Decision.** Continue with a SIMD-LUT kernel. Keep int8 as the production CPU format until a 2-bit kernel beats it.
