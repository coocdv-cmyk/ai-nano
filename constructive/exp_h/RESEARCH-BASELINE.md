# EXP-H — Research baseline: what CCM actually is

Verified against source on 2026-11-18. Nothing here is assumed; every claim
cites the file. No interaction with `omega_nano/` is described — that codebase
is a separate project and is untouched.

## 1. What CCM actually implements

**Values and language** — `constructive/core.py` (stdlib-only):

- `Entity`, `Scene`, frozen dataclasses; `Scene.by_id(id)` raises `EvalError` on
  a missing entity.
- `Term` (immutable `op/args/value`), `infer_type`, `well_typed`,
  `tree_edit_distance`, `value_key`, `same_value`, `term_cost`, boolean-aware
  `value_key` (a bool is never equal to 0/1).
- Environment constants derived per-fit: `build_prims(int_constants=range(-20,21))`
  is available; the learner itself calls `build_prims(_constants(examples))`,
  i.e. only constants ≤ 0..12 (plus a small literal set) are indexed.
  Primitives are **curated and fixed**: unary arithmetic (`add1`, `sub1`,
  `double`, `square`, `abs`, `neg`, `sum_digits`, `digital_root`, `is_prime`,
  `even`, `odd`, `fib`, `fact`), constant-shifted ops (`add(k)`, `mul(k)`,
  `mod(k)`, `gt(k)`, `lt(k)`), list ops, pair ops, scene relations, logic, counts.
- No conditional expression, no loops, no list comprehension taking a computed
  function — nothing with control flow or recursion.

**Learner** — `constructive/model.py`:

- `Concept = typed frozen program with (params, ret, input_type, arg_types,
  guard, confidence, support, validation set, embedding, parent, note)`.
- `Searcher`: bottom-up enumeration with **value-level evaluation**
  (`lift1`/`lift2` replace recursive re-evaluation of subtrees),
  a **type-relevance pruning** set (a candidate whose type cannot reach the
  target type within the remaining budget is discarded), a capped
  **mid-level bucket** (4000) and a **final-level beam** (default 45),
  **observational deduplication** keyed on
  `(type, outputs, probes)`, and **deduplicated alternatives** preserved (up to
  8) to be compared on held-out evidence.
- Primitives first, then concepts-as-atoms (1- and 2-arg).
- Validated acceptance: exact training fit → held-out validation must be 100 %;
  only then is the winner accepted (or re-used, if it is exactly a call to an
  existing concept). Otherwise the concept is rejected or parked in
  `hypotheses/`.
- Posterior: unnormalised log-losses with `eps`, `beta=1`, softmax-like
  normalisation; margin < `abstain_margin = 0.12` with a surviving rival →
  ambiguous → not adopted as a concept; it stays an unvalidated hypothesis.
- Quarantine salvage (all-but-one): a rule fitting n−1 of n examples and
  validating on held-out evidence is accepted with the inconsistent example
  **quarantined as the identified violation**.
- Repair (`revise`): a global body replacement only if it preserves every old
  example (distance ≤ 2); otherwise a scoped child concept with a guard, and
  the parent locally restricted with the **negated** guard; otherwise reject.
- `compose`: a cost-restricted fresh search (`max_cost=2`, concept units),
  requiring an actual concept call; structurally, not by naming.
- Retention bookkeeping: `retention()`, `forgetting()`, `feedback()`,
  online perceptron-like update on 17 weights of a retrieval prior.

**Retrieval and prediction** — `ConstructiveModel.predict_episode`: behaviour-matched
ranking on the support set (a stored concept explains the support if ≥ 75 %),
abstain if nothing explains it, refuses if two equally-matching concepts
disagree on the query (`ambiguous with …`), never guesses without evidence.

**World/canon** — `constructive/worlds.py`: task generators reject data
explainable by distractors, derived hidden rules expressed in the same DSL.

## 2. What CCM does NOT implement (verified in source)

| Missing capability | Consequence for EXP-H |
|---|---|
| No conditional/branching in DSL | H1/H2 conditional trees unreachable by construction — expected 0% (measured, not assumed) |
| No bitwise operators | H1 `exclusion` family unreachable in baseline |
| No abstraction/macro harvesting | H6 needs an *experimental* extension (`h6_abstraction.py`, isolated) |
| No invention of primitives outside DSL | H1 `exclusion` measures vocabulary-limitation failure, by construction |
| No sub-concept storage from a winner | reuse currently repeats-cost; H8 studies this |
| No test-time hypothesis search | abstention only from stored concepts — no active counterexample search |
| No A* / cost-guided search / incremental validation | H8 is a fair-testing ground for these |
| No indentification machinery (twins distinguishable) | H1/H7 measurability depends on probes + held-out validation |
| No runtime resource cap enforcement against adversarial big-ints | H2 chains with unbounded arithmetic are a documented risk |

## 3. Computational limitations

- Default `max_cost=3`, `beam=45`, `max_nodes=250_000`,
  `time_budget_s=2.0` in `CCM_KW`; the benchmark uses 10.0 s / 600 k nodes.
- Final beam prunes diversity first (`-variation, -n_ok`), capping branches
  evolutionary; mid-level cap 4000 keeps enough variety.
- Search cost for a full list-rule (cost 3) ≈ 60–100 k nodes ≈ 0.6–3 s;
  cost-4 flat search explodes (minutes at 4 M nodes) — verified empirically
  and used as evidence for composition (`constructive/bench.py` Experiment C).
- Memory is not profiled inside the evaluator: `tracemalloc` only shows Python
  allocations; children (None), candidates and trees are short-lived.
- No claim of O(1) or polynomial in depth; deep alternating chains blow up.
- Constants discovered are bounded by "those in example values ≤ 12", so
  `add(11)` is usually unlearnable.

## 4. Which EXP-H experiments run unchanged, which need isolation

| Experiment | Needs | Isolation |
|---|---|---|
| H1 open-world discovery | existing CCM | none — reuse untouched |
| H2 deep composition | existing CCM (with small `max_cost`) | none |
| H3 OOD generalization | existing CCM | none |
| H4 continual | existing CCM | none (pool in `h4_pool.py`) |
| H5 programming intelligence | **new** AST engine | `ast_engine.py` (isolated) |
| H6 abstraction discovery | **extension**: macro harvest | `h6_abstraction.py` (isolated) |
| H7 adversarial validation | existing CCM + oracle | `h7_adversarial.py` (isolated) |
| H8 efficiency | CCM arm parameters + **one** independent search | `lexical_h8.py` (independent) |

## 5. Verified environment

- CPU: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40 GHz, 4 cores / 8 threads
- RAM: 16.82 GB; OS: Windows 11 26200
- Python 3.13.15, numpy 2.5.3, no CUDA, no C compiler on PATH
- tracemalloc (Python-level) only; machine not pinned; no NUMA tuning

## 6. Base-line decisions

- CCM baseline runs untouched; any alternative is an *independent
  implementation*, never a silent replacement.
- Baselines (`baselines.py`) see the same held-out example as training data
  (fair label budget) and CCM sees it exactly once for acceptance.
- Memoriser = regression control; kNN = nearest-example; MLP = small numpy
  net with per-entity softmax for relational outputs; none is handicapped
  (audit included in report §6).
- Validation example is never used for training inside CCM; baselines receive
  it explicitly (they must).
- Unlabelled probes go only to CCM; documented asymmetry, harmless to labels.

## 7. Git status at inspection time

- Branch `master`, **zero commits** (all files untracked, `git diff --cached`
  empty). Nothing is staged. No file outside `constructive/exp_h/`,
  `tests/test_exp_h.py` and `CONSTRUCTIVE-EXP-H-REPORT.md` is touched.
  `omega_nano/` is unmodified (verified by post-11:00 mtime scan).
- `constructive/exp_h/results/` is populated by the runner; `experiment-history/`
  retains every run for audit.
