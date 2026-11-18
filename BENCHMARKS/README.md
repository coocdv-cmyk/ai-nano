# Benchmarks

All numbers were measured in this repository on: Intel i7-7820HQ (4C/8T, AVX2), 8 GB RAM, Windows 11, Python 3.10.
Raw JSON, including detected hardware, is in `results/`. Re-run the scripts to reproduce. Laptop results vary run to run
(turbo, thermals, background load); treat differences under ~1.5x as noise unless repeated.

## Agent loop (`bench_engine.py`, 200 tasks per tier)
Task: `calculate 2^N then save the result to rN.txt then check that file rN.txt contains 1`, i.e. 3 steps, each
independently verified, with 10 durable checkpoints per task.

| Metric | t1 rules | t0 plan cache |
|---|---|---|
| End-to-end p50 / p95 | 22.7 / 39.7 ms | 13.4 / 43.2 ms |
| Planning p50 | 0.12 ms | 0.09 ms |
| Execute + verify per step, p50 | 2.5 ms | 2.4 ms |
| Tasks/s (serial) | 40 | 53 |

Honest reading: the plan cache saves only ~0.03 ms of planning, because rule planning is already cheap. The end-to-end
gap mostly comes from the workload (the cached runs rewrite the same file with identical content), not from caching.
The dominant cost is durable checkpointing (≈2 KB × 10 transactions per task).
Crash recovery: the killed process exited with code 77; a fresh `resume` process (including Python start-up) completed
the task in 407 ms and **recovered the interrupted step without re-executing it**.

## Memory fabric (`bench_memory.py`)
See `EXPERIMENTS/EXP-002-int8-retrieval-kernel.md` for the full iteration history and ablation.

| Records | Paraphrase R@1 | Typo R@5 | Arabic R@1 | p50 / p95 (paraphrase) | Bytes/record | Ingest/s |
|---|---|---|---|---|---|---|
| 1,000 | 1.000 | 0.97 | 1.000 | 2.1 / 3.3 ms | 1,260 | 2,098 |
| 10,000 | 1.000 | 0.94 | 1.000 | 15.4 / 20.6 ms | 1,643 | 1,456 |
| 50,000 | 1.000 | 0.83 | 1.000 | 22.6 / 48.8 ms | 1,252 | 1,712 |

## CPU kernels (`EXPERIMENTS/EXP-001-ternary-cpu`)
int8 matvec: 3.0–4.7x over fp32 BLAS on DRAM-resident layers and exact. 2-bit ternary: 16x smaller but only
1.3–2.5x faster with the kernels so far (compute-bound unpacking).
