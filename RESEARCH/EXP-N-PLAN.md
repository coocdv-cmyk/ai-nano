# EXP-N — Pre-Registered Research Plan (`OMEGA-NANO V2` program)

**Branch:** `research/exp-n`
**Status:** active — N0 frozen and baselined; N1 engine built, gated, and probed on a held-out surface set
**Parent program:** `EXP-M` (`M0`–`M4`, `research/exp-m` @ `21b93b4`)
**Constitution anchors:** `فهم الخطة` (Program 3), `OMEGA-NANO-PROJECT-MISSION.md` §4, §8, §11

---

## 0. Why this program exists

`EXP-M` proved three things and left one gap:

1. A **2.00 M-parameter** core can beat `zlib-9` compression on a clean split (`1.7482` vs `1.8370` bpb, EXP-005).
2. The native CPU runtime can decode at **10,509–13,209 tok/s** single-stream with greedy-identical output (EXP-007/008).
3. A verifier-guided loop can reach **10/10 sealed + 10/10 held-out** on author-designed tasks with zero unverified answers.

The gap: **none of those tasks came from the open world.** There is no measurement of whether the system can repair a
real multi-file repository, reason through an unfamiliar multi-step problem, or retrieve bilingual knowledge beyond the
strings it was given. `EXP-N` exists to measure that gap, then close it in a controlled order.

**Program question:** can a `< 5 M`-parameter, CPU-only, verifier-guided system improve its open-world task success at a
cost that stays inside the Tier-1 budget (ordinary laptop, ≤ 8 GB RAM, no GPU)?

---

## 1. Pre-registration rules (apply to every N-track)

- Every track states its **hypothesis**, **metric**, **baseline**, **gate**, and **kill criterion** *before* implemention.
- The N0 suite is **frozen at v1**. Any change to a task bumps the suite version and **invalidates every earlier
  baseline**; the old `results.json` is preserved as history and is never overwritten.
- No task in the N0 suite may enter any training corpus, wake-sleep replay buffer, skill library, or few-shot prompt.
  The harness exposes only prompts and verifiers to a solver; solutions live under `.sealed/` and are read only by the
  evaluator.
- A result is reported with its **full cost**: wall time, peak RSS, verifier calls, attempts. A success obtained by
  spending 100× the baseline is a different result, not a better one.
- Negative and null results are recorded in `results.json` and in the README, not deleted.

---

## 2. N0 — Sealed benchmark v1 (the measurement instrument)

**Deliverable:** `EXPERIMENTS/EXP-009-n0-benchmark/` with:
`task_suite/v1` (hashed), `.sealed/` (solutions + labels), `harness.py`, `run_exp.py`, `results.json`.

**Tracks and scoring**

| Track | Content | Scoring | Oracle requirement |
|---|---|---|---|
| **C — repository repair** | 6 multi-file Python tasks, each with a failing `pytest` suite; the suite is the only specification | fraction of tasks whose full test suite passes after the candidate's edit | `6/6` |
| **R — verifiable reasoning** | 12 multi-step problems with exact answers (integer or reduced fraction), hashed | exact match after canonicalisation | `12/12` |
| **L — bilingual retrieval** | 36 short documents (AR/EN/code) + 18 paraphrased queries | recall@5 and MRR against sealed labels | `18/18` |

**Baseline procedure (recorded before any N1 code):**
1. Verify the suite manifest (SHA-256 of every file).
2. Run the oracle: sealed solution → full tests; sealed answer → hash check. Must be `C 6/6`, `R 12/12`, `L 18/18`.
3. Run the product baseline:
   - `C`: `OmegaAgent.handle(<task statement + workspace path>)` — the current route has no patch operator, so this
     measures the real product, not a strawman.
   - `R`: `OmegaAgent.handle(<problem statement>)`.
   - `L`: `MemoryFabric` ingest of all 36 documents, then `recall(query, k=5)`.
4. Record wall time and peak RSS for the whole run.

**Gate:** the run completes, integrity passes, oracle is 100%, and the baseline is written to `results.json`.
**Kill criterion (program-level):** if after N1–N4 the system cannot move the code track above the recorded baseline
*at equal or lower verifier budget*, the "open-world repair" claim is withdrawn and the architecture review reopens.

**Recorded status (2026-11-19):** gate passed — oracle `C 6/6`, `R 12/12`, `L 18/18`; baseline `C 0/6`, `R 0/12`,
`L recall@5 0.6667 / MRR 0.625`, topic view `1.0`, 10.405 s, 232.47 MB peak RSS. See
`EXPERIMENTS/EXP-009-n0-benchmark/README.md` §3–§4.

---

## 3. N1 — Open AST repair (`SWE-mini`)

**Hypothesis N1:** traceback-driven AST mutation plus `RepoGraph` context can repair multi-file bugs that the current
synthesizer cannot touch, without weakening the "verified or silent" guarantee.

- **Mechanism:** `ast.NodeTransformer` operators (off-by-one, comparison, default-argument, closure-binding, cache-key,
  rounding); failure localization from the pytest traceback + assertion diff; `RepoGraph` retrieval of the enclosing
  symbol and its callers; sandboxed `CodeVerifier` on the patched copy.
- **Metric:** N0 code track pass rate, attempts per solved task, verifier calls per solved task.
- **Gate:** pass rate strictly above the N0 baseline with **zero** accepted patches whose tests were not executed.
- **Kill criterion:** if 3 iterations of the operator set leave the pass rate at baseline, the mutation approach is
  declared insufficient and N4 (search) must carry the track instead.

**Recorded status (2026-11-19): gate passed.** `EXP-010` solved **6/6** N0 code tasks at **24 verifier calls** total
(mean 4.0, max 6), with **0** accepted patches lacking an executed verification, an independent scorer that agrees on
all six, and the oracle still at 6/6. Cost: **25.241 s**, **228.08 MB** peak RSS. See
`EXPERIMENTS/EXP-010-n1-ast-repair/README.md`.

**Held-out surface probe (`EXP-011`, 2026-11-19): gate passed.** Seven new same-family tasks, authored after the
nine-operator registry was frozen and run once without tuning: **7/7**, **26 verifier calls** (mean 3.71, max 6),
**0** unexecuted patches, scorer agrees, oracle 7/7. Cost: **27.782 s**, **228.44 MB** peak RSS. This is
**same-family surface transfer, not open-world repair**: the tasks were authored by the same agent with the operator
list in hand, and a bug outside the nine families is still refused rather than fixed. The suite is now spent as a
held-out set and becomes a regression check. See `EXPERIMENTS/EXP-011-n1-heldout/README.md`.

**As-built deviation from the pre-registration.** The mechanism bullet above says "sandboxed `CodeVerifier`". What was
built is an **isolated workspace copy** plus `pytest` in a subprocess with a sanitised environment, a timeout, and a
write-back only after the suite passes. That protects the source tree from failed candidates; it is **not** an OS
sandbox and does not contain hostile test code (repository defect #4). The N1 claim is scoped accordingly.

**Operator note.** The registry has nine families. `comparison_flip` belongs to the pre-registered comparison family,
is not exercised by any N0 task, and is exercised exactly once by `EXP-011` (`n1-h-07-bisect-bound`).
`mapping_fallback` is exercised once by N0 (`n0-code-04-cache-key`). One success each is coverage, not robustness.

## 4. N2 — Verified corpus scaling + trained MTP

**Hypothesis N2:** MTP speculation becomes a net speedup once the heads are trained on verifier-proven frames.

- **Mechanism:** generate `(spec, trace, verified_code)` frames from N1; train the existing MTP heads; add an adaptive
  acceptance gate (`accept_rate >= 0.25`).
- **Metric:** speculative acceptance rate, single-stream tok/s, greedy-exact agreement with the non-speculative path.
- **Gate:** acceptance `> 40%`, throughput `> 10,509 tok/s`, greedy output identical on a fixed 10k-token probe.
- **Kill criterion:** if acceptance stays `< 5%` after the corpus is ≥ 10× the EXP-008 replay set, speculation is
  disabled by default and the speed claim rests on the non-speculative engine only.

**As-built status (2026-11-19, `EXPERIMENTS/EXP-012-n2-mtp/`).** The pre-registration above is unchanged. The
instrument deviated in three recorded ways: the shipped EXP-008 checkpoint (40 steps / 625 B) was too degenerate to
be a meaningful speculation target, so a real v2 base was trained before freezing it and training the MTP heads;
native v2 speculation had two losslessness bugs (prompt-byte skipping, unwritten accepted drafts) that were fixed
and pinned by tests; and a patch-boundary copy-gate divergence forced drafts to stay off the final token of a patch.
Result: 108/108 verified fresh tasks, 90× the EXP-008 corpus, leak check clean, greedy output bit-identical over
10k tokens — but acceptance 29.3%/27.5% (`mean_prefix/K`) or 38.0%/35.8% (engine) and throughput 7,512/8,481 tok/s,
so **both quantitative gates fail** and speculation is a 0.87x-0.92x slowdown. The kill criterion did not trigger;
the speed claim rests on the non-speculative engine. Negative results and the boundary defect are recorded in the
EXP-012 README rather than discarded.

## 5. N3 — Learned bilingual encoder

**Hypothesis N3:** a `< 500 KB int8` contrastive bi-encoder generalises across unseen AR/EN synonyms where the static
concept anchors do not.

- **Metric:** recall@5 / MRR on a **new harder held-out set** + the N0 retrieval track as a sanity check.
- **Gate:** recall@5 above the N0 baseline with no increase in false-accept rate on out-of-scope queries.
- **Instrument correction (recorded after the N0 baseline, before any N3 code):** Track L `v1` cannot serve as the N3
  gate. The strict document-level baseline (`0.6667`) is depressed by translation twins — the misses return the correct
  fact in the other language — and the topic-level view is already saturated at `recall@5 = 1.0`. `N3` must therefore
  author a held-out retrieval set with (a) no translation twins, (b) no shared surface tokens between query and target,
  and (c) near-duplicate distractor facts, and record its own baseline before claiming a semantic gain. Track L `v1`
  remains a regression check, not evidence of improvement.
- **Kill criterion:** if it cannot beat the N0 baseline while staying under 500 KB, the static encoder stays and the
  bilingual claim is narrowed to the measured vocabulary.
- **Hard constraint:** the Arabic byte-model deficit (`11.12 bpb` vs zlib `4.5447`, EXP-005) must be reported alongside
  any retrieval gain; retrieval success does not repair the language model.

**As-built status (2026-11-19, `EXPERIMENTS/EXP-013-n3-bilingual/`).** The pre-registration above is unchanged. A fresh
62-concept train / 30-concept held-out inventory was authored with exact-string disjointness; the probe has one gold
document per query, zero shared query/target surface tokens and two near-duplicate distractors per gold. The static
baseline was recorded before any training: recall@5 `0.0833`, false-accept `0.8125` — effectively non-functional on
this set. A tied char-ngram → int8 encoder (393,216 bytes) then reached recall@5 `0.7333` (AR `0.7000`, EN `0.7667`),
MRR `0.3454`, false-accept `0.0625`, beating both the static baseline and the cited N0 `0.6667`; all four gates pass.
Caveats recorded, not hidden: the static false-accept comparator is weak, the first hyperparameter attempt diverged and
was recorded, the N0 comparison is cross-suite, the probe is synthetic, and the Arabic byte deficit is unchanged.

## 6. N4 — Best-first / MCTS over verified candidates

**Hypothesis N4:** value-guided branching beats linear `attempt 1..N` on multi-function synthesis at equal verifier cost.

- **Metric:** N0 code pass rate at a fixed budget of verifier calls (not fixed wall time), plus attempts-to-first-success.
- **Gate:** higher pass rate at equal budget, or equal pass rate at ≤ 60% of the verifier calls.
- **Kill criterion:** if branching only wins by spending more verifier calls, the linear loop stays.

**As-built status (2026-11-19, `EXPERIMENTS/EXP-014-n4-search/`).** The pre-registration above is unchanged. Six fresh
three-function workspaces (three train / three held-out) were composed from N4-only themes; N0 was cited, not used.
Both arms used the same generator, verifier and 12-call budget. The value arm was fitted only from the linear arm's
train-split decisions and spent no verifier calls at ranking time. Final held-out result: both arms `1.000` pass, with
**9.667 → 7.000** mean verifier calls (72.4% of linear, a 27.6% reduction). This misses the 60% gate, so the gate is
**not met**, while the kill criterion is not triggered either (it wins by spending fewer calls). The linear loop stays
the default. Instrument defects were recorded rather than hidden: unsolvable `off_by_one` variants were excluded from
composition, and an earlier 0%/0% result came from a 3-edit cap plus N1's coarse dedup key.

## 7. N5 — `V2` release gate

- Full capability + efficiency matrix (accuracy, tok/s, TTFT, peak RSS, model bytes, verifier calls) for every track.
- A local UI/API that exercises the same code paths the matrix measures.
- **Gate:** every number reproducible from a command recorded in the matrix; no claim without a `results.json`.

---

## 8. Order and budget

```
N0 (this turn)  →  N1  →  N2  →  N3  →  N4  →  N5
```

`N1` first because it supplies the trajectories that `N2` trains on and the multi-step stress that `N4` searches over.
`N3` is independent and may run in parallel with `N2` once `N0` is frozen.

| Track | Expected wall time | Expected RAM | New dependencies |
|---|---|---|---|
| N0 | < 2 min | < 400 MB | none |
| N1 | 1–3 days | < 500 MB | none (`ast`, `pytest`, existing verifier) |
| N2 | hours of CPU, background | < 1 GB | none (torch already optional) |
| N3 | 1–2 days | < 600 MB | none (pure numpy) |
| N4 | 2–4 days | < 600 MB | none |
| N5 | 1 day + benchmark runs | < 500 MB | none |

No GPU is required by any track. No new third-party dependency is introduced before a licence review.

### 8.1 Known repository-level issue (pre-existing, not caused by `EXP-N`)

`python -m pytest tests` collects `tests/test_exp_i.py`, `test_exp_j.py`, `test_exp_k.py` and `test_exp_l.py`, whose
integrity assertions pin the SHA-256 of `omega_nano/` files as they stood during `EXP-I`…`EXP-L`. `EXP-M`'s sanctioned
`M0` changes (security fixes, corpus rebuild, new brain modules) invalidate those pins, so **8 tests fail on
`research/exp-m` and on every branch derived from it** — reproduced on a clean worktree of `21b93b4` with no local edits.
The **active suite** (`RESEARCH/HANDOFF.md` §3 plus `tests/test_exp_n_benchmark.py` and
`tests/test_exp_n1_repair.py`) is green at **180 passed, 1 skipped**.
Repairing the pins requires an owner decision, because the integrity manifests are historical artefacts and must not be
rewritten silently.

---

## 9. What would count as evidence for the mission

A defensible milestone is not "20/20 on our own problems" again. It is, on the frozen suite and with a recorded command:

- **N1:** code repair above baseline with zero unverified patches. **Met 2026-11-19, bounded:** `6/6` N0 and `7/7`
  same-family held-out, 0 unexecuted patches (`EXP-010`/`EXP-011`). This is not an open-world claim.
- **N2:** `> 40%` MTP acceptance at `> 10,509 tok/s` with identical greedy output.
- **N3:** bilingual recall above baseline under 500 KB.
- **N4:** better success per verifier call, not per wall-clock minute.

Only after `N1`–`N4` pass should `EXP-O` (multi-domain capability) be opened. The program does not advance because a
milestone was named; it advances because a gate was measured.
