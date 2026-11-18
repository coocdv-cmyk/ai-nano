# OMEGA EXPERIMENT 001 — Skill Fabric / BIRS

Internal ID: **EXP-005** (EXP-004 is reserved in `ROADMAP.md` for the AVX2 C kernel, which is blocked by the missing
toolchain).

| Field | Value |
|---|---|
| Status | **Designed. Not implemented. Not run.** No result exists. |
| Date designed | 2026-11-18 |
| Owner | OMEGA-NANO research |
| Type | Pre-registered, CPU-only, synthetic, falsification-first |
| Depends on | nothing (self-contained); informs `ORIGINAL-LEARNING-MECHANISM.md` |
| Estimated implementation | 600–900 LOC, 1–2 days |
| Estimated run time | < 5 min CPU, < 1 GB RAM, no network, no GPU |
| Location when built | `EXPERIMENTS/EXP-005-skill-fabric/` (does **not** exist yet) |

> This document is the *pre-registration*. Thresholds, arms and metrics are fixed here, before any code is written, so that
> the result cannot be reinterpreted after the fact. Any deviation must be recorded in §14 with a reason and a timestamp.

---

## 1. Research question

> **Does retrieving an executable skill by its measured behaviour, and synthesising only the residual against it, improve
> sample-efficient program acquisition compared with retrieval alone, synthesis from scratch, and a small neural
> sequence model — at three demonstrations per task?**

A second, subordinate question:

> **Does behavioural indexing reduce search cost at equal quality compared with retrieval on surface features or random
> selection?**

---

## 2. Hypothesis

**H1 (primary).** At `k = 3` demonstrations, on programs whose nearest available skill shares a sub-behaviour but not the
whole program, BIRS achieves a higher held-out exact-match rate than each of its ablations and baselines.

**H2 (secondary).** Behavioural indexing reduces the number of search nodes required to reach a given solve rate by at
least one order of magnitude relative to from-scratch synthesis, because the retrieved skill supplies a correct prefix of
the target program.

**H0 (null).** Any observed advantage is explained by the extra search budget, the DSL's accidental structure, or
evaluation leakage. The design below is built specifically to make H0 detectable.

**What is explicitly *not* claimed.** That this generalises to natural language, to open-ended code, or to "general
intelligence". The claim is scoped to example-driven transformations in a known DSL. Anything broader is a later
experiment with a different design.

---

## 3. Task family and DSL

### 3.1 Domain

Strings over the alphabet `Σ = {a, b, c, 1, 2}` with length 4–16. The restricted alphabet keeps the search space finite
and the failure modes visible; it is a deliberate simplification, and §12 states what it costs in external validity.

### 3.2 DSL

Unary string functions, composed left-to-right. Programs are sequences of 1–4 primitives.

| Primitive | Semantics | Arity |
|---|---|---|
| `id` | identity | 0 |
| `upper` / `lower` | ASCII case | 0 |
| `rev` | reverse | 0 |
| `sort` | sort characters ascending | 0 |
| `dedup` | collapse consecutive duplicates | 0 |
| `uniq` | remove all but first occurrence of each character | 0 |
| `take(n)` | first `n` characters | 1 (n ∈ 1..8) |
| `drop(n)` | drop first `n` characters | 1 (n ∈ 1..8) |
| `rep(a,b)` | replace all `a` with `b` | 2 (a,b ∈ Σ) |
| `del(c)` | delete all `c` | 1 (c ∈ Σ) |
| `keep(S)` | keep only characters in `S` | 1 (S ⊆ Σ, card(S) ≤ 2) |
| `rot(n)` | rotate left by `n` | 1 (n ∈ 1..4) |
| `swap` | swap adjacent pairs | 0 |
| `dup` | duplicate every character | 0 |
| `cat(c)` | append constant `c` | 1 (c ∈ Σ) |
| `pre(c)` | prepend constant `c` | 1 (c ∈ Σ) |
| `mask(c,n)` | replace positions `n mod len` with `c` | 2 |
| `iflen(n, f, g)` | if `len ≥ n` apply `f` else `g` | 3 (n ∈ 4..12, f,g primitive) |

`iflen` is the only branching primitive and is included because it creates genuinely non-compositional behaviour: a
residual over a retrieved skill sometimes *needs* a branch, and a DSL without one would make BIRS look better than it is.

The **program space** at depth ≤ 4 is on the order of `10^5–10^6` programs after type and length pruning. That is small
enough for exhaustive search within budget and large enough that naive enumeration is slow — the regime in which retrieval
can plausibly help.

### 3.3 Search semantics

Cost-bounded depth-first enumeration with:

- **observational pruning:** if two partial programs agree on all `k` demo inputs, expand only the cheaper one
  (behavioural equivalence on demos);
- **type/length pruning:** `take`, `drop`, `rot` are no-ops on short strings and are skipped where provably inert;
- **fuel:** a global node counter, not a wall-clock timer, so results are reproducible across machines. Wall clock is
  recorded separately.

Ties in cost are broken by a fixed total order on program strings, so the search is deterministic.

---

## 4. Data generation and splits

### 4.1 Generator

Programs are sampled from the DSL with a fixed seed, subject to:

1. **Non-triviality:** the program changes at least 60% of inputs drawn from its distribution.
2. **Stability:** it is defined and terminating on all sampled inputs.
3. **Identifiability:** among all programs of cost ≤ `cost(target) + 2`, the target is the **unique** minimum-cost program
   consistent with its 3 demonstrations. Programs failing this filter are discarded at generation time.

Filter 3 is essential. Without it, "held-out exact match" is ill-posed: many different programs fit three examples, and a
correct synthesiser could be scored wrong for choosing a different consistent program. The filter makes the task
well-posed by construction, and the discard rate is reported (it is itself a finding about the DSL).

### 4.2 Splits

| Split | Programs | Purpose |
|---|---|---|
| `train` | 60 | The skill library is built from these; each contributes 3 demos |
| `test-id` | 60 | **Primary endpoint.** Disjoint programs, same operator set |
| `test-hard` | 30 | Target shares a *middle* sub-program with a train program but not a prefix or suffix |
| `test-ood-op` | 20 | Uses `rot` and `keep`, which are present in the DSL but **never used by any `train` program** — no retrieved skill can help, so every arm must fall back to synthesis. An out-of-distribution probe, reported separately, never pooled with the primary metric |

**Split by program, never by example.** A program's inputs never appear in two splits. This is the leakage control that
most program-synthesis benchmarks get wrong.

Each program contributes:

- 3 demonstrations `(x_i, y_i)`;
- 10 held-out inputs for evaluation (labels never shown to any arm);
- 16 public probe inputs `P` used for signatures (shared across the library, disjoint from demos and evaluation);
- 16 admission probes `P_adm` generated per task, used only by the admission tests.

### 4.3 Curriculum construction for `test-hard`

For each hard target, the generator finds a train program `s` such that the target's AST contains a subtree of `s` of size
≥ 2 but the target's prefix differs. If no such pair exists, the target is regenerated. The construction is recorded per
program so retrieval precision can be scored against ground truth.

### 4.4 Seeds

Three generation seeds: `{11, 23, 37}`. Each seed produces a complete, independent dataset (programs, demos, probes). All
arms see the same dataset within a seed. Results are reported per seed and as a mean with range; no seed is dropped.

---

## 5. Arms

Seven arms. The first five are the comparison; the last two are diagnostics and are never used to claim a win.

| Arm | Name | What it does | Supervision |
|---|---|---|---|
| A1 | **SYN** | Enumerate from `id`; no library | 3 demos |
| A2 | **RET** | Retrieve top-1 skill by behaviour; apply it; **no residual** | 3 demos + library |
| A3 | **BIRS** | Retrieve top-1 by behaviour; synthesise residual `r` with `s∘r` | 3 demos + library |
| A4 | **BIRS-perm** | Same as A3, but each target receives the skill retrieved for the *next* target in a fixed permutation. Preserves the retrieved-skill distribution and the compute, destroys the target–skill correspondence | 3 demos + library |
| A5 | **NN-k** | Char-level GRU encoder–decoder with attention, ~200k params, trained on the 180 demo pairs from `train` | 3 demos (matched) |
| A6 | **NN-full** *(diagnostic)* | Same architecture, trained on 50 pairs per train program (3,000 pairs) | supervision-rich |
| A7 | **ORACLE** *(diagnostic)* | Retrieves the train program with maximum shared AST subtree; residual synthesis | ground-truth retrieval |

**Why A4 exists.** If BIRS beats RET but BIRS-perm matches BIRS, the gain comes from *having a library at all*, not from
behavioural indexing. A4 is the control that separates those explanations. It is the single most important arm in the
design.

**Why A6 exists.** A reviewer will ask whether the neural baseline was starved. A6 answers that by giving it 16× more
supervision. If A6 still loses to A3, the result is strong; if A6 wins, the honest conclusion is that BIRS wins only in
the low-supervision regime — which is still the regime OMEGA cares about, but the claim must be stated that way.

**Budget parity.** Every search arm (A1–A4, A7) gets the same node budget: `N = 200,000` nodes and `T = 2 s` per program.
A3 does not get more compute than A1; it gets a better starting point. If A3 finishes early, the unused budget is
recorded, not redistributed.

**Neural baseline details.** 1-layer GRU, hidden 256, embedding 64, Luong attention, teacher forcing, 20 epochs, Adam
3e-4, batch 32, early stopping on a 10% held-out slice of the training pairs. Decoding is greedy. Three training seeds per
generation seed; the best-on-validation checkpoint is used. Training time must stay under 60 s per run on this CPU.

---

## 6. Metrics

### 6.1 Primary

**Program-level exact match (PEM).** A program counts as solved iff the arm's induced program produces the correct output
on **all 10** held-out inputs. Reported as `solved / 60` on `test-id`, averaged over 3 generation seeds.

Program-level, not input-level, because the object of interest is a *program*, and per-input accuracy hides the difference
between "almost right" and "right".

### 6.2 Secondary

| Metric | Definition | Why it matters |
|---|---|---|
| Search nodes | Median and p95 nodes expanded per solved program | Tests H2 directly |
| Wall clock | Median and p95 seconds per program | Practical feasibility |
| Retrieval precision@1 | Fraction where the retrieved skill shares ≥ 1 AST subtree with the target | Validates the behavioural index itself |
| Signature collision rate | Fraction of distinct train programs with identical `σ` on 16 probes | Tests whether the key is discriminative |
| Residual depth | Distribution of `len(r)` when retrieval is correct vs incorrect | Shows whether the residual is doing real work |
| Expressibility failures | Fraction of targets where no program of cost ≤ target+2 was found | Separates DSL limits from retrieval limits |
| Demo→held-out gap | PEM on demos minus PEM on held-out, per program | Detects overfitting |
| Admission precision | Fraction of admitted skills that are non-equivalent to the true program on 100 fresh probes | **Safety metric** |
| Library compression | `L(S)` after 60 admissions, in bits | Tests whether MDL admission is doing anything |
| NN training time | Seconds per run | Budget compliance |

### 6.3 Reported but not used for decisions

Per-input accuracy; BLEU-style character similarity (it rewards near-misses and is misleading for exact tasks); training
loss curves; node counts per operator.

---

## 7. Pre-registered decision rules

Evaluated in order. The first matching rule is the result.

| # | Condition | Verdict |
|---|---|---|
| R0 | Any arm's held-out labels were visible to any other arm, or the split was by example | **Invalid run.** Fix and rerun; do not report |
| R1 | `BIRS ≥ 0.60` PEM **and** `BIRS ≥ 2.0 × max(A1, A2, A5)` | **P1 confirmed.** Proceed to EXP-006 (library learning) |
| R2 | `BIRS ≥ A1 + 0.10` but ratio `< 2.0` | **Partial.** Mechanism helps; the "2×" claim is withdrawn. Revise the mechanism document and design EXP-006 around cost, not quality |
| R3 | `BIRS ≤ A1 + 0.05` | **P1 refuted.** Kill the residual-synthesis programme; keep the behavioural index as a retrieval feature and report the negative result |
| R4 | `BIRS > A1 + 0.05` but `abs(BIRS - A4) < 0.05` | **Attribution failure.** The gain is "library, not behaviour". Rewrite the claim; test a stronger index in EXP-006 |
| R5 | `PEM(BIRS) ≥ 0.60` on `test-id` but `< 0.20` on `test-hard` | **Brittle.** The mechanism only works when the shared prefix is aligned. Report as scoped to prefix-sharing tasks |
| R6 | Admission precision `< 1.0` | **Unsafe.** The admission tests are insufficient; block any integration until a stronger test exists |

H2 is evaluated separately:

| # | Condition | Verdict |
|---|---|---|
| S1 | Median nodes(BIRS) ≤ 0.10 × nodes(SYN) at equal PEM | **H2 confirmed** |
| S2 | 0.10× < ratio ≤ 0.50× | **Partially confirmed** |
| S3 | ratio > 0.50× | **H2 refuted.** Retrieval does not pay for itself in search cost |

No p-hacking: the thresholds above are fixed. A p-value may be reported alongside, but it does not override an effect-size
rule.

---

## 8. Statistical treatment

- Unit of analysis: **program**, paired across arms within a seed.
- Primary estimate: mean PEM across the 60 `test-id` programs, averaged over 3 seeds; report the per-seed values, not just
  the mean.
- Uncertainty: 10,000-resample paired bootstrap over programs, 95% percentile interval for the difference
  `PEM(BIRS) − PEM(baseline)`.
- Paired binary comparison: McNemar's test for BIRS vs each baseline, reported with the discordant-pair counts (not just
  the p-value).
- Multiplicity: three primary comparisons (vs A1, A2, A5). Holm correction applied to the reported p-values; the decision
  rules in §7 are effect-size based and do not depend on it.
- No interim analysis. One run per seed; reruns only for implementation bugs, and every rerun is logged.

---

## 9. Leakage and validity controls

| Risk | Control | Check |
|---|---|---|
| Example-level leakage | Split by program; assert `inputs(train) ∩ inputs(test) = ∅` | Automated assertion, fails the run |
| Operator leakage | `rot`, `keep` never used in `train`; `test-ood-op` reported separately | Assert zero `train` programs use them |
| Probe leakage | `P`, `P_adm`, demos and eval inputs are pairwise disjoint | Assert at generation |
| Tuning on test | `λ`, `δ`, depth caps and node budgets fixed here; tuned only on `train` | Config hash recorded in `results.json` |
| Neural baseline starved | A6 with 16× supervision | Report both |
| Retrieval given ground truth | Retrieval uses only `β(E)` and demo-input agreement | Code review + unit test that `β` never sees the target program |
| Search budget confound | Equal node budget; unused budget recorded | Assert per-arm counters |
| Seed shopping | Three seeds fixed in advance; all reported | `results.json` contains all three |
| DSL overfit to benchmark | Grammar and generator published in the repo; `test-ood-op` measures the edge | Report OOD separately |
| Hardware variance | Node counts, not wall clock, are the search metric | Wall clock reported with hardware string |

**Threats that cannot be controlled in this design** (stated now, not discovered later):

1. The DSL is small and synthetic; success may not transfer to real string/data tasks.
2. Three demonstrations may be too few to identify many real-world programs; the identifiability filter hides this by
   discarding ambiguous cases, and the discard rate is a hidden selection effect.
3. A 200k-parameter GRU is a weak neural baseline; a pretrained code model would be stronger but is not available
   offline and would violate the no-download constraint.
4. The behavioural feature vector `β` was designed by the same person who designed the DSL. It may encode DSL-specific
   regularities. The `test-ood-op` split is the only partial check.

---

## 10. Budget and environment

| Resource | Limit | Enforcement |
|---|---|---|
| Wall clock, full grid (7 arms × 3 seeds × 170 programs) | ≤ 5 min | Harness aborts at 300 s and records partial results as **incomplete**, not as a result |
| RAM | ≤ 1 GB | `tracemalloc` high-water mark recorded |
| CPU | 1 process, ≤ 4 threads | Thread count recorded |
| Network | none | No network calls in the code path |
| New dependencies | none beyond stdlib + NumPy (already present) | Import audit |
| Disk | ≤ 50 MB | Results are JSON + code |

Hardware disclosure is mandatory in `results.json`: CPU model, core/thread count, RAM, OS build, Python, NumPy version,
and whether AVX2 was detected. A number without this string is not a result.

---

## 11. Implementation plan (600–900 LOC)

Not built in this session — the assignment is documentation-only. The plan is specified so implementation is mechanical.

| Module | ~LOC | Contents |
|---|---|---|
| `dsl.py` | 150 | AST nodes, typed constructors, interpreter with fuel, printer/parser, canonical ordering |
| `search.py` | 200 | Enumerative search, observational pruning, residual mode `s∘r`, node accounting, memoisation |
| `profile.py` | 110 | `β` features, `σ` signatures, probe generation, index, collision statistics |
| `admission.py` | 120 | MDL accounting, invariance probes, regression capture, replay, versioned library |
| `generator.py` | 130 | Program sampler, identifiability filter, split construction, hard-pair mining |
| `arms.py` | 90 | The seven arms behind one interface; budget accounting |
| `run.py` | 90 | Grid, seeds, timing, `results.json` writer, assertions from §9 |
| `tests/test_skills.py` | 150 | Determinism, fuel termination, no-import sandbox, replay catches mutants, MDL monotonicity |

**Interface contract.** Every arm exposes `solve(demos, library, budget) -> (program, stats)`. The harness never inspects
arm internals, so a future arm (e.g., an LLM proposer) can be added without changing the evaluation.

**Testing before the run.** The suite must pass, including:

- a mutant skill that differs on one probe is rejected by replay;
- a semantics-preserving refactor is accepted;
- a program that loops is terminated by fuel, not by wall clock;
- the interpreter rejects attribute access, imports, and dunder names;
- two runs with the same seed produce byte-identical `results.json` (modulo timings).

---

## 12. What the result would mean

| Outcome | Interpretation | Next action |
|---|---|---|
| R1 | The composition is real at this scale | EXP-006: multi-stage library learning (P3), then a real-task replication on CSV/log cleaning |
| R2 | Real but smaller than claimed | Keep the mechanism, drop the 2× headline, focus EXP-006 on cost and reuse |
| R3 | The central claim is false at k=3 | Publish the negative result; pivot to VW-1 (verified workbench), which does not depend on this mechanism |
| R4 | Libraries help, behaviour indexing does not | Simplify: use a plain code library with a cheap index; test richer signatures later |
| R5 | Works only for aligned prefixes | Scope the claim; design a residual *alignment* mechanism as the next question |
| R6 | Admission is unsafe | Stop. No production integration. Redesign admission before any further work |

**The most valuable outcome is R3 or R6.** A clean negative at 5 minutes of CPU is worth more than a vague positive, because
it redirects the programme before months are spent. The document says this now so that a negative result cannot later be
narrated as a success.

---

## 13. Deliverables when run

```
EXPERIMENTS/EXP-005-skill-fabric/
  README.md          # design snapshot, deviations, one-page result
  code/              # the modules above
  results.json       # all arms × seeds × metrics + hardware string + config hash
  runs/<seed>/       # per-program records: target hash, retrieved skill, residual, nodes, time
  FAILURES.md        # every program all arms failed on, with the target program
```

`results.json` schema (fixed now):

```json
{
  "experiment": "EXP-005-skill-fabric",
  "design_version": "1.0",
  "config_hash": "...",
  "hardware": {"cpu": "...", "threads": 0, "ram_gb": 0, "os": "...", "python": "...", "numpy": "...", "avx2": true},
  "seeds": [11, 23, 37],
  "splits": {"train": 60, "test-id": 60, "test-hard": 30, "test-ood-op": 20},
  "arms": {
    "BIRS": {"pem": {"11": 0.0, "23": 0.0, "37": 0.0}, "nodes_median": 0, "wall_median_s": 0.0,
             "retrieval_precision": 0.0, "admission_precision": 0.0}
  },
  "verdict": "R1|R2|R3|R4|R5|R6",
  "notes": []
}
```

---

## 14. Deviations log

Empty by construction. Any change to §3–§9 after the first run must be appended here with date, reason, and the fact that
it was made after seeing data. A post-hoc change that is not logged invalidates the pre-registration.

| Date | Section | Change | Seen data before change? |
|---|---|---|---|
| — | — | — | — |

---

## 15. Relationship to the rest of the programme

- If **R1/R2**, the Skill Fabric becomes the project's primary research bet, and EXP-006 tests compounding (P3) on a
  multi-stage curriculum.
- If **R3**, the verified workbench (VW-1) becomes primary, and the fabric is reduced to a plain code cache.
- Either way, the **kernel work (KC-1)** proceeds in parallel because it is orthogonal and its success criterion is a
  number, not a capability.
- The **corpus remediation** must precede any external publication of results, because a synthetic benchmark is not
  immune to the licence problem if its generator or examples are derived from contaminated data. EXP-005's generator is
  written from scratch, so the experiment itself is clean; the surrounding claims are not.
