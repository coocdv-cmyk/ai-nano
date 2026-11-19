# EXP-012 — N2: verified corpus scaling + trained MTP speculation

Pre-registration: `RESEARCH/EXP-N-PLAN.md` §4.

**Hypothesis.** MTP speculation becomes a net speedup once the heads are trained
on verifier-proven frames. Gate: acceptance `> 40%`, throughput `> 10,509 tok/s`,
greedy output identical on a fixed 10k-token probe. Kill criterion: acceptance
`< 5%` at a corpus `>= 10x` the EXP-008 replay set.

## Instrument corrections (as-built, dated 2026-11-19)

1. **v1 off-by-one → v2 verification order.** The historic v1 engine compared an
   offset-1 draft against logits for the *next* position. The v2 path verifies
   draft 1 against the already-computed `scores_buf` before any forward runs,
   and drafts `2..K` against the batched forward outputs.

2. **A real base is required before training MTP heads.** The checkpoint actually
   shipped by EXP-008 is 40 steps / 625 bytes of wake corpus; its greedy output
   is degenerate, so a head imitating it would score a meaningless acceptance.
   This instrument therefore trains a v2 base on `core_data`, adapts it to the
   verified frames, freezes it, and only then trains the heads. Two objectives
   are compared on the same frozen base: `self` (imitate the base argmax) and
   `truth` (predict the corpus byte).

3. **Two native speculation bugs were found and fixed before measurement.**
   (a) Speculation could begin while still consuming the prompt, skipping input
   bytes whenever a draft was accepted; it is now gated by `pos >= T`.
   (b) Accepted drafts after the first were committed to model state but never
   written to `gen_out`, leaving uninitialised output bytes; every accepted
   draft `0..a-1` is now written. `tests/test_core.py::test_native_v2_speculation_is_lossless`
   and `check_spec.py` pin the byte-identical invariant for rejection-heavy,
   acceptance-heavy and adaptive-gate regimes.

4. **Verifier hermeticity.** Third-party pytest plugin autoload could shadow a
   workspace module name (`filters`, `renderers`, `formatters`, ...) and hang the
   verifier subprocess for 60 s, misrecording a solvable task as unsolvable.
   `omega_nano/brain/repair.py::_safe_env` now sets
   `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; the isolated verifier went from a 1.5 s
   test plus a 60 s hang to a 1.5 s test.

## Corpus

`corpus.py` authors 9 families x 12 fresh variants = 108 tasks. Eight variants
per family are train, the last four are a predeclared held-out probe. A frame
`<SPEC>/<TEST>/<FAIL>/<SOL>` is admitted only when `RepairEngine` produces a
patch that passes the workspace's own pytest and never writes without execution.
A 64-byte-shingle leak check compares both streams against every sealed N0 and
EXP-011 task-suite file; a hit aborts admission.

`build()` is resumable: each finished task is persisted immediately to
`records.jsonl` and `frames/<id>.bin`, and a rerun skips recorded tasks. Final
`train.bin`/`probe.bin` are assembled deterministically from cached frames in
family/variant order, so resuming never changes their byte layout.

## Negative results recorded (not deleted)

- The first two corpus runs produced no output and wrote nothing: stdout was
  block-buffered and the build wrote only at the end. Fixed by per-task flush and
  incremental persistence.
- Ten verifier timeouts (`final_failed=999`) from plugin autoload were archived
  verbatim to `records-failed-timeouts.jsonl`, root-caused (item 4 above) and
  retried. They are not presented as repair failures.
- The smoke measurement at 2000 tokens, run under corpus-build CPU contention,
  reached only 4,630 tok/s and 24.9% acceptance. It is a pipeline check, not an
  N2 result; all reported gates use the uncontended 10k-token run.
- **Patch-boundary hand-off defect.** With drafts allowed to occupy the final
  token of a patch, native output first diverged at the next patch boundary on
  trained checkpoints (byte 24 for a 64-byte prompt). Instrumentation showed
  identical per-position `xf` and coarse checksums through the previous patch,
  then a copy-gate score divergence for the boundary draft. Restricting drafts
  to non-final patch positions restores byte identity at every prompt length
  tested (60..68 bytes). The unrestricted variant is logged here as a defect,
  not quietly discarded.

## Results

Corpus: **108/108** tasks solved, 354 verifier calls, 72 train frames / 36 probe
frames, 56,531 / 28,306 bytes, leak check clean (35,875 sealed shingles, 0 hits).
The train stream is **90x** the 625-byte EXP-008 replay set, well above the 10x
kill floor. Base: **2.1156 bpb** on `probe.bin`, top-1 agreement with bytes
**62.81%**.

| Objective | mean prefix / K | conditional offsets 1/2/3 | all accepted | native accept | native tok/s (plain → spec) | speedup | 10k identity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `self` | 0.2927 | 0.609 / 0.310 / 0.423 | 0.0800 | 0.3801 | 8,604.9 → 7,512.4 | **0.873x** | identical |
| `truth` | 0.2747 | 0.579 / 0.302 / 0.396 | 0.0693 | 0.3581 | 9,181.8 → 8,481.4 | **0.9237x** | identical |

Peak RSS 291.7 / 291.96 MB; wall 7.55 s for the 10k runs; 362.5 s training.

**Gate verdict: FAILED.** Acceptance is above the 5% kill floor (so speculation
is not killed outright) but below the 40% gate, throughput is below 10,509
tok/s, and although greedy output is bit-identical, speculation is a **0.87x -
0.92x slowdown**, not a speedup. The extra draft projections and batched copy
steps cost more than the fine-path batching saves at this acceptance. The speed
claim therefore rests on the **non-speculative** engine, and `Engine.generate` / `generate_batch` now default to
`speculative=False` (the measurement instrument opts in explicitly). Full record:
`results.json`, `runs/mtp-v1/engine_results-mtp-{self,truth}.json`.

Note on metrics: `mean_prefix/K` (the pre-registered definition) is 29.3% /
27.5%, while the engine's `accepted/attempted` rate is 38.0% / 35.8%. The latter
is higher because drafts 2..K are only attempted after draft 1 matches. Both are
reported; neither clears 40%.

## Files

| File | Role |
| --- | --- |
| `corpus.py` | fresh verified corpus generator (resumable, leak-checked) |
| `train_mtp.py` | base training, MTP training (`self`/`truth`), PyTorch acceptance |
| `check_spec.py` | native speculation losslessness invariant |
| `measure_engine.py` | native throughput, acceptance, RSS, 10k identity |
| `repro_boundary_defect.py` | reproducible prompt-alignment sweep for the recorded hand-off defect |
| `results.json` | authoritative N2 result, gate verdict and cost record |
| `runs/corpus-v1/` | generated corpus, manifest, per-task records and frames |
| `runs/mtp-v1/` | trained checkpoints and measurement records |
