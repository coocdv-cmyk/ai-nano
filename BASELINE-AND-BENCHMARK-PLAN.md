# Baseline and Benchmark Plan

Created 2026-11-18. This document exists because the repository has already produced numbers that are easy to
misread — and one of them (aggregate throughput) has already been quoted as if it were single-stream latency in the
roadmap. The plan fixes definitions, baselines, quality floors and reporting rules **before** the next measurement, so
that a future result cannot be narrated into something it is not.

It is a protocol, not a result. The measured ledger is in §10.

---

## 1. The problem this solves

Four distinct quantities are routinely collapsed into the phrase "tokens per second":

1. **Prefill throughput** — prompt tokens processed per second.
2. **Single-stream decode** — output tokens per second for one request, on one core or many.
3. **Aggregate throughput** — total output tokens per second across `B` concurrent requests.
4. **End-to-end rate** — output tokens divided by total wall clock, including queueing, tokenisation, prefill and
   any re-encoding.

They differ by 3–4× in this codebase (fresh measurement: 1,425 vs 5,027 B/s). A 10,000 tok/s claim is meaningful only if
it names which one, on which hardware, with which quality floor. The same applies to "faster": faster than what, at what
batch size, with what output?

The second problem is quality. A speedup that changes the model's output is not a speedup; it is a different model. Every
speed claim therefore carries a quality measurement from the **same checkpoint**, and every kernel change is checked for
token-identical greedy decoding on a fixed prompt set.

---

## 2. Metric definitions (normative)

Let a request have `n_p` prompt tokens, `n_o` output tokens, issue time `t_0`, first-token time `t_1`, last-token time
`t_n`. For a batch of `B` requests, `t_end` is the time the last request in the batch completes.

| Metric | Definition | Unit | What it is **not** |
|---|---|---|---|
| **TTFT** | `t_1 − t_0` | ms | Not prefill time (includes queueing + tokenisation + first sample) |
| **Prefill rate** | `n_p / (t_1 − t_0)` | tok/s | Not decode speed; not comparable across prompt lengths without stating `n_p` |
| **Single-stream decode** | `(n_o − 1) / (t_n − t_1)` | tok/s | Not end-to-end; excludes prefill and first token |
| **End-to-end rate** | `n_o / (t_n − t_0)` | tok/s | The number a user experiences; always ≤ decode rate |
| **Aggregate throughput** | `Σ_b n_o,b / (t_end − t_0)` | tok/s | Not latency. A system can have high aggregate and terrible TTFT |
| **Per-token latency** | `(t_n − t_1)/(n_o − 1)` | ms | Report p50 and p95, not just the mean |
| **Quality** | bits per byte on held-out data, or task exact-match | bpb / % | Must come from the same checkpoint and split as the speed number |
| **Energy** | joules per 1,000 tokens | J | **Not measurable here.** No power meter is available; do not report it |

**Mandatory qualifiers.** Any reported number must carry: metric name, batch size, `n_p`, `n_o`, thread count, CPU model,
and quality value. A number missing these is a draft, not a result.

**Batch size is part of the metric name.** "5,027 tok/s" without "aggregate, B=8" is not a result. "1,425 tok/s" without
"single-stream, 4 threads, end-to-end decode including re-encode" is not a result either.

---

## 3. Quality floors

A speed or efficiency claim is only admissible if it clears a quality floor on the same artefact. The floors are project
decisions, stated as such — they are not derived from first principles.

| Domain | Floor | Rationale |
|---|---|---|
| Core language model | **≤ 1.83 bpb** to claim "matches zlib"; **≤ 1.43 bpb** to claim "beats the best classical compressor on this corpus" | zlib-9 = 1.8296, bz2-9 = 1.4657, lzma-extreme = 1.4341 on the EXP-003 split |
| Core language model (any architecture claim) | Must beat the **same-size** baseline trained on the same tokens for the same wall clock | Prevents comparing a 30-minute run to a 3-hour run |
| Ternary / quantised core | Report the bpb delta versus fp32 **and** the byte reduction; a speed win with an unstated quality loss is not a win | EXP-003: 2.0554 vs 1.8944 bpb, 3.2 MB vs 12.8 MB block weights |
| Decode speed | Greedy output must be **token-identical** to the pre-change engine on a fixed prompt set, or the divergence must be reported with a quality delta | A kernel that changes results is a different model |
| Memory retrieval | Recall@10 must be reported on a labelled query set before latency is quoted as a capability | Latency without recall measures an index, not a memory |
| Agent tasks | Success = verified postcondition, not "the tool returned ok" | Invariant 2 of the architecture |
| Skill acquisition | Pre-registered thresholds in `OMEGA-EXPERIMENT-001.md` §7 | Prevents post-hoc success criteria |

**The floor is not negotiable after seeing the speed number.** If a 3× speedup arrives with a 0.4 bpb regression, the
correct report is "3× faster at 0.4 bpb worse", and the claim is a trade-off, not a win.

---

## 4. Hardware and environment disclosure

Every `results.json` must contain this block, filled from the machine that ran it:

```json
{
  "hardware": {
    "cpu": "11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz",
    "physical_cores": 4,
    "logical_cores": 8,
    "l3_cache_mb": 8,
    "ram_gb": 15.66,
    "os": "Windows 11 10.0.26200",
    "python": "3.13.15",
    "numpy": "2.5.3",
    "numba": "0.68.0",
    "torch": "2.14.1+cpu",
    "cuda": false,
    "simd": "AVX2 (NumPy X86_V3); AVX-512 not detected",
    "c_compiler": "none on PATH",
    "power": "mains, no meter",
    "thermal_state": "cold start, first measurement after 60 s idle"
  }
}
```

**Rule:** results from different hardware are never presented as a speedup ratio. They may be presented as two
independent measurements with the hardware string attached. The repository currently contains results from an
i7-7820HQ (EXP-001, EXP-003, BENCHMARKS) and from an i5-1135G7 (the 2026-11-18 spot-check); conflating them would be a
false claim.

**Thermals.** Laptop CPUs throttle. A benchmark that runs for 10 minutes measures a different machine at minute 9 than at
minute 1. Protocol: 60 s idle, then measure; report the first run and the steady-state run separately if they differ by
more than 10%.

---

## 5. Benchmark suites

Each suite has one question, a fixed protocol, explicit baselines, and a pass condition. Suites are versioned; changing a
suite invalidates comparison with earlier results and requires re-running the baseline.

### B1 — Test suite (regression, not capability)

- **Question:** does the system still behave as specified?
- **Command:** `python -m pytest -q`
- **Current:** 126 passed, 1 skipped in 16.54 s (2026-11-18). The skip is a Windows symlink permission test.
- **Pass:** zero failures. A skipped test is reported, never counted as a pass.
- **What it does not measure:** capability, quality, or speed.

### B2 — Agent loop (durability and correctness)

- **Question:** does a task survive crashes, retries and malformed output without duplicating effects?
- **Command:** `python BENCHMARKS/bench_engine.py`
- **Recorded baseline:** `BENCHMARKS/results/engine.json` (i7-7820HQ, 2026-11-17). Not re-run in the audit.
- **Metrics:** success rate, recovery rate, duplicate-effect count, p50/p95 task latency.
- **Pass:** 100% recovery without re-execution of completed steps; zero duplicate effects on idempotent tools; the
  `fs.write` append case must be **fixed** before this suite can claim zero duplicates (audit §8.1).

### B3 — Memory fabric

- **Question:** can the agent retrieve the right memory at scale, and at what latency?
- **Command:** `python BENCHMARKS/bench_memory.py --sizes 1000 10000 50000`
- **Recorded baseline:** `BENCHMARKS/results/memory.json`, measured to 50k records.
- **Metrics:** recall@1/@10 on a labelled query set, p50/p95 query latency, index size, ingest rate.
- **Gap:** there is **no labelled query set** yet. The recorded benchmark measures latency and synthetic recall, not
  semantic quality. Until a labelled set exists, no retrieval-quality claim is admissible.
- **Next:** build 300 queries with graded relevance over a clean corpus (blocked by corpus remediation).

### B4 — Core language-model quality

- **Question:** how well does the core predict held-out bytes, and at what size?
- **Command:** `python EXPERIMENTS/EXP-003-omega-core-v0/evaluate.py` (see the experiment README for exact flags)
- **Recorded baseline (EXP-003, i7-7820HQ):**

| Arm | val bpb | params | block weights | training |
|---|---|---|---|---|
| fp32 | **1.8944** | 3,344,640 | 12.8 MB | 30.7 min, 837 steps |
| ternary | **2.0554** | 3,344,640 | 3.2 MB | 30.9 min, 784 steps |
| zlib-9 | 1.8296 | — | — | — |
| bz2-9 | 1.4657 | — | — | — |
| lzma-extreme | 1.4341 | — | — | — |
| order-2 n-gram | 2.5793 | — | — | — |

- **Pass for a new architecture:** lower val bpb than the fp32 baseline at equal parameters, equal tokens and equal wall
  clock. Otherwise the result is "different", not "better".
- **Pass for a quantisation method:** report `(bpb, bytes)` as a pair; a Pareto improvement over the ternary point is a
  win, a movement along the frontier is a trade-off.

### B5 — Core decode speed

- **Question:** how fast does the engine generate, under each of the four metric definitions?
- **Command:** `python BENCHMARKS/bench_core_speed.py` (to be written; currently speed is reported inside EXP-003)
- **Protocol:** fixed prompt, `n_new = 400` for the quick run and `2000` for the full run, greedy sampling, warm-up call
  excluded, three repetitions, report median and p95.
- **Fresh spot-check (2026-11-18, i5-1135G7, 4 threads, `n_new = 400`):**

| Arm | Single-stream | Aggregate | Batch | First call (incl. JIT) |
|---|---|---|---|---|
| fp32 | 673 B/s | 1,655 B/s | 8 | 8.01 s |
| ternary | **1,425 B/s** | **5,027 B/s** | 8 | 4.55 s |

- **Recorded EXP-003 (i7-7820HQ, different machine, `n_new = 2000`):** ternary 841 B/s single (1 thread), 810 B/s single
  (7 threads), 2,865 B/s aggregate (batch 14); fp32 402 / 427 / 1,276.
- **Pass for a kernel change:** ≥ 1.5× on the named metric, identical greedy tokens, bpb unchanged.
- **Explicitly banned:** quoting 5,027 as "tokens per second" without "aggregate, batch 8"; comparing 5,027 to the 10k
  target, which is a single-stream ambition.

### B6 — Skill acquisition (EXP-005)

- **Question:** does behaviour-indexed residual synthesis beat its ablations at `k = 3`?
- **Protocol:** `OMEGA-EXPERIMENT-001.md`, pre-registered.
- **Metrics:** program-level exact match, nodes, wall clock, retrieval precision, admission precision.
- **Status:** designed, not implemented, not run. No number exists.

### B7 — Verified reasoning (future)

- **Question:** what fraction of formal goals can the workbench close, with zero unsound proofs?
- **Protocol:** fixed 30-goal set (Nat/Int arithmetic), Z3 + `simp`, 60 s timeout each.
- **Metrics:** solve rate, median time, kernel-checked proof count, unsound count (must be 0).
- **Status:** proposed (VW-1 in `RESEARCH-OPPORTUNITIES.md`). Not started.

---

## 6. Baseline taxonomy and fairness rules

### 6.1 Four tiers

| Tier | Meaning | Examples in this repo |
|---|---|---|
| **T0 — trivial** | The weakest thing that could possibly work; if the system cannot beat it, stop | identity transform; copy input; return most frequent output |
| **T1 — classical** | Strong non-neural baselines | zlib/bz2/lzma for compression; BM25 for retrieval; nearest-neighbour for skills |
| **T2 — same-family published** | The published method reimplemented at this scale | bitnet.cpp-style LUT matvec; DreamCoder-style enumerative synthesis; small seq2seq |
| **T3 — external systems** | Third-party systems run as-is | llama.cpp with a small GGUF model; Lean/Z3 |

**Rule:** every experiment reports at least one T0 and one T1 baseline. A result that only beats T2 is not yet a result,
because T2 may itself be a weak implementation. A result that loses to T0 is a bug or a dead end.

### 6.2 Fairness rules

1. **Same data, same split, same budget.** If an arm gets more supervision, it is labelled as such (see EXP-005 A6).
2. **Same hardware, same process.** Cross-machine numbers are reported separately.
3. **All arms reported.** Dropping a baseline after it loses is a protocol violation.
4. **Hyperparameters fixed before the run**, or tuned on a validation split that is disjoint from test. The tuning
   protocol is recorded.
5. **No "best of n" without declaring n.** If the best of 5 seeds is reported, all 5 are reported.
6. **No test-set peeking.** The test split is touched once per experiment version.
7. **No metric substitution.** If the pre-registered metric is exact match, character similarity cannot replace it after
   the fact.
8. **Negative results are results.** They are written to the experiment record with the same care as positives.

---

## 7. Statistical protocol

- **Unit of analysis** is stated per suite: task (B2), memory record (B3), program (B6), goal (B7). Never "token".
- **Report n.** A percentage without a denominator is not a measurement.
- **Report dispersion.** For continuous metrics: median and p95 (robust to the long tail that laptops produce). For
  proportions: Wilson interval, not normal approximation.
- **Paired designs where possible.** Same programs, same seeds, different arms; analyse differences, not independent
  means.
- **Seeds.** Minimum 3 for any learned component. All reported.
- **Multiplicity.** If more than two arms are compared, Holm-correct the p-values; the decision rules in each experiment
  are effect-size based and take precedence.
- **Effect size over significance.** A 2% improvement with p < 0.05 on 60 programs is not a capability.
- **No interim peeking.** One analysis per version.

---

## 8. Reporting template

Every result file and every README section follows this shape:

```
Claim:        one sentence, with the metric and the qualifier
Metric:       name, formula, batch size, n_p, n_o
Quality:      bpb or task score from the same checkpoint
Hardware:     §4 block
Command:      exact command, copy-pasteable
Result:       value, dispersion, n
Baseline:     the comparison, with its own quality and hardware
Verdict:      pass / partial / fail against a pre-stated criterion
Limitations:  what the number does not show
```

`results.json` is append-only in spirit: a new run writes a new file or a new seeded block. **Recorded results are never
overwritten to look better** (constitution §14). The existing `EXPERIMENTS/*/results.json` files are treated as immutable
history; the audit did not modify them.

---

## 9. Banned claims

The following are protocol violations, not style preferences:

1. "10,000 tokens/second" without specifying single-stream vs aggregate, batch size, and quality.
2. A speedup ratio between two different machines.
3. "Faster than X" where X was measured with a different `n_new`, batch size, or thread count.
4. A quality claim from a contaminated corpus presented as a capability claim.
5. "State of the art" without a reproduced comparison on a public benchmark.
6. "Works" from a single seed or a single example.
7. "Verified" when only the tool's own report was checked.
8. "Zero-shot" when the demonstration examples were in the prompt.
9. "Learns" when the only mechanism is a lookup table with the answer in it.
10. Any number whose command is not in the document and whose hardware string is absent.

---

## 10. Current baseline ledger

What the project can honestly claim as of 2026-11-18, with provenance. "Recorded" means it exists in a result file but
was not re-run in this audit; "fresh" means it was measured on the current machine during the audit.

| ID | Claim | Value | Provenance | Status |
|---|---|---|---|---|
| L1 | Test suite passes | 126 passed, 1 skipped, 16.54 s | fresh | reproduced |
| L2 | Agent loop durability | recorded benchmark | `BENCHMARKS/results/engine.json` | not re-run |
| L3 | Memory to 50k records | latency + synthetic recall | `BENCHMARKS/results/memory.json` | not re-run |
| L4 | Core fp32 quality | 1.8944 bpb | EXP-003 `results.json` | not re-run |
| L5 | Core ternary quality | 2.0554 bpb | EXP-003 `results.json` | not re-run |
| L6 | Classical compressors | 1.8296 / 1.4657 / 1.4341 bpb | EXP-003 `results.json` | not re-run |
| L7 | Ternary single-stream speed | 1,425 B/s | fresh, i5-1135G7, B=1, 4 threads | reproduced (narrow) |
| L8 | Ternary aggregate speed | 5,027 B/s | fresh, i5-1135G7, B=8 | reproduced (narrow) |
| L9 | Ternary weights | 3.2 MB block / 3.8 MB total | EXP-003 `results.json` | arithmetic check |
| L10 | `fs.write` append bug | appends up to 3× on retry | fresh reproduction | confirmed defect |
| L11 | Corpus licence contamination | 62 LGPL headers in train | fresh scan | confirmed defect |
| L12 | Skill acquisition | — | — | **no result** |
| L13 | Verified reasoning | — | — | **no result** |
| L14 | Energy per token | — | — | **not measurable here** |

**The honest summary:** the project has a solid agent foundation, a measured but weak language core, a real but
narrow speed point, and no evidence yet for the two research programmes it is about to pursue. That is the correct
starting state for an experiment, not a failure.

---

## 11. Reproduction commands

Bash syntax; on Windows PowerShell use the equivalents or Git Bash. `python -m pytest -q` and the module invocations are
portable as written.

```bash
# environment
python -m omega_nano doctor
python -m omega_nano tools

# regression
python -m pytest -q

# agent loop and memory (recorded on i7-7820HQ)
python BENCHMARKS/bench_engine.py
python BENCHMARKS/bench_memory.py --sizes 1000 10000 50000

# kernels (recorded on i7-7820HQ)
python EXPERIMENTS/EXP-001-ternary-cpu/ternary_bench.py

# core quality (requires torch)
python EXPERIMENTS/EXP-003-omega-core-v0/evaluate.py

# corpus licence scan (no writes)
python - <<'PY'
import pathlib, re
pat = re.compile(rb"GNU Lesser General Public|Apache License|MIT License")
for p in pathlib.Path("data").rglob("*"):
    if p.is_file() and p.suffix in {".py", ".txt", ".md"}:
        n = len(pat.findall(p.read_bytes()))
        if n: print(n, p)
PY
```

The spot-check from the audit used the engine's in-process `bench()` with `n_new = 400`, batch 8 and 4 threads. Its exact
invocation is recorded in `PHASE-0-REPOSITORY-AUDIT.md` §12.

---

## 12. Governance

- This plan is versioned. A change to a metric definition or a quality floor requires a note in `CHANGELOG.md` and a
  re-run of any result it invalidates.
- Results are immutable. Corrections are new records that cite the old one.
- A benchmark that has never been run on the current machine is marked "recorded", never "reproduced".
- The protocol applies to internal documents as well as external claims. A false number in a roadmap is still a false
  number.
