# Release matrix (EXP-015 / N5)

Every value is extracted from the cited result file; the command that produced it is recorded in `matrix.json`.

| Track | Metric | Value | Unit | Qualifier | Source |
| --- | --- | --- | --- | --- | --- |
| M1 | v2 clean-split bpb | 1.7482 | bpb | 2.00M params, 306.3 s CPU | `EXPERIMENTS/EXP-005-m1-v2-core/showdown_results.json` |
| M1 | zlib-9 baseline bpb | 1.837 | bpb | identical clean split | `EXPERIMENTS/EXP-005-m1-v2-core/results.json` |
| M2 | native v2 single-stream | 13209 | tok/s | packed 2-bit, greedy | `EXPERIMENTS/EXP-008-m4-omega-v1-matrix/results.json` |
| M2 | native v2 batch-4 | 25177.6 | tok/s | same checkpoint, batch 4 | `EXPERIMENTS/EXP-008-m4-omega-v1-matrix/results.json` |
| M2 | incremental engine baseline | 10509.2 | tok/s | the N2 throughput gate threshold | `EXPERIMENTS/EXP-007-m2-bitpacked-engine/results.json` |
| N0 | strict retrieval recall@5 | 0.6667 | fraction | sealed N0 v1, baseline before N1 | `EXPERIMENTS/EXP-009-n0-benchmark/results.json` |
| N0 | peak RSS | 232.47 | MB | full benchmark run | `EXPERIMENTS/EXP-009-n0-benchmark/results.json` |
| N1 | N0 code solved | 6 | tasks | 6 of 6, zero unexecuted patches | `EXPERIMENTS/EXP-010-n1-ast-repair/results.json` |
| N1 | N0 code verifier calls | 24 | calls | 6/6 solved | `EXPERIMENTS/EXP-010-n1-ast-repair/results.json` |
| N1 | held-out solved | 7 | tasks | 7 of 7, same-family surface probe | `EXPERIMENTS/EXP-011-n1-heldout/results.json` |
| N2 | fresh verified corpus | 108 | tasks | 108/108, leak check clean | `EXPERIMENTS/EXP-012-n2-mtp/results.json` |
| N2 | frozen base bpb | 2.1156 | bpb | N2 held-out probe | `EXPERIMENTS/EXP-012-n2-mtp/results.json` |
| N2 | engine accept rate | 0.3801 | fraction | self objective, 10k greedy probe | `EXPERIMENTS/EXP-012-n2-mtp/results.json` |
| N2 | speculative tok/s | 7512.4 | tok/s | self objective, slower than non-spec — gate failed | `EXPERIMENTS/EXP-012-n2-mtp/results.json` |
| N3 | learned recall@5 | 0.7333 | fraction | 393,216 B int8, fresh AR/EN held-out | `EXPERIMENTS/EXP-013-n3-bilingual/runs/n3-v1/results.json` |
| N3 | static recall@5 | 0.0833 | fraction | same held-out set, recorded before training | `EXPERIMENTS/EXP-013-n3-bilingual/runs/n3-v1/results.json` |
| N3 | out-of-scope false accept | 0.0625 | fraction | 16 OOS queries | `EXPERIMENTS/EXP-013-n3-bilingual/runs/n3-v1/results.json` |
| N4 | value-arm pass rate | 1 | fraction | three fresh three-function held-out tasks | `EXPERIMENTS/EXP-014-n4-search/runs/n4-v1/results.json` |
| N4 | value-arm verifier calls | 7 | calls | vs linear 9.667; 72.4% > 60% gate | `EXPERIMENTS/EXP-014-n4-search/runs/n4-v1/results.json` |

Integrity: 19/19 claims resolved, clean=True.
