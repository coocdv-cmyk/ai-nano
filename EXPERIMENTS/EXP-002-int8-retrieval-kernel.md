# EXP-002 — Memory Fabric: retrieval quality and latency under scale

**Status:** measured, adopted · **Benchmark:** `BENCHMARKS/bench_memory.py` · **Raw:** `BENCHMARKS/results/memory.json`

## Questions
1. Can int8-quantized hashed embeddings be searched fast enough on a CPU without a GPU or an ANN index?
2. How robust is hybrid retrieval to paraphrase, misspelling, and Arabic orthographic variation?

## Setup
Synthetic facts about invented entities, so no pretrained model can know them and the ground truth is exact. 10% are
Arabic. Queries come in three styles: paraphrase, typo (one interior character dropped from the entity name), and
Arabic with diacritics and alef variants. 300 queries per style. Single-threaded queries, warm cache, i7-7820HQ.

## Iterations (50k records, hybrid unless noted)

| Version | Change | Paraphrase R@1 | Typo R@5 | p50 latency |
|---|---|---|---|---|
| v0 | lexical BM25 + hashed vectors, numpy `int8 @ f32` | 0.993 | 0.007 | 84 ms |
| v1 | Numba int8×f32 kernel (`omega_nano/kernels.py`) | 0.993 | 0.007 | 19 ms |
| v2 | + trigram "fuzzy" signal on every unknown word | 0.993 | 0.837 | 46 ms (R@1 0.68 at 1k) |
| v3 | prefix stage as a separate list, 2% trigram filter | 0.82 | 0.353 | 25 ms |
| v4 | prefix terms merged into BM25, 8 rarest trigrams | 0.993 | 0.82 | 20 ms |
| **v5** | **+ question words as stopwords (EN/AR)** | **1.000** | **0.83** | **22.6 ms** |

Kernel micro-benchmark (50,000 × 512): numpy int8@f32 82 ms (it upcasts the whole matrix each call), numpy fp32
BLAS 10.7 ms, **Numba int8×f32 3.0 ms**, Numba int8×int8 2.2 ms (top-50 overlap 49/50). Max relative error of the
adopted kernel vs fp32: 2e-7.

## Final ablation (v5)

| N | Signals | Paraphrase R@1 | Typo R@1 / R@5 | Arabic R@1 | p50 ms (paraphrase) |
|---|---|---|---|---|---|
| 1k | hybrid (lexical+vector+fuzzy) | 1.000 | 0.54 / **0.97** | 1.000 | 2.1 |
| 1k | lexical+vector | 1.000 | 0.08 / 0.33 | 1.000 | 1.9 |
| 10k | hybrid | 1.000 | 0.69 / **0.94** | 1.000 | 15.4 |
| 10k | lexical+vector | 1.000 | 0.02 / 0.06 | 1.000 | 4.8 |
| 50k | hybrid | 1.000 | 0.33 / **0.83** | 1.000 | 22.6 |
| 50k | lexical only | 1.000 | 0.00 / 0.00 | 1.000 | 13.6 |
| 50k | vector only | 0.92 | 0.00 / 0.03 | 0.957 | 5.0 |

Storage: ~1.25–1.6 KB per record on disk, including text, normalized text, a 512-byte int8 vector, the BM25 index and
the trigram index. Bulk ingest: ~1,500–2,100 records/s.

## Interpretation and honest caveats
- The fuzzy signal is what makes misspelled entity names retrievable (0.7% → 83% R@5 at 50k). Gating it to genuinely
  unknown words matters: applying it to every unknown token (v2, v3) hurt normal queries.
- **This benchmark favors lexical retrieval:** paraphrase queries reuse the entity name verbatim, so lexical-only already
  scores 100%. The vector signal's value isn't demonstrated here. A synonym/semantic-paraphrase benchmark is needed
  before claiming it helps (RESEARCH/RC-002).
- Typo queries cost more (p50 59 ms at 50k) because trigram postings are long. A vocabulary-level spelling index
  (SymSpell-style) is the next candidate.
- Brute-force vector search is linear in N. Beyond ~1M records an ANN index or a coarse quantizer will be needed. Not
  yet measured.
