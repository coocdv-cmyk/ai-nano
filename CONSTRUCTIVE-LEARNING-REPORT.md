# OMEGA Constructive Intelligence — Phase 1 report

| | |
|---|---|
| **Date** | 2026-11-18 |
| **Scope** | An isolated research module (`constructive/`), not a change to the OMEGA-NANO runtime |
| **Status** | Works as designed on synthetic few-shot tasks; honest negatives reported; **not** a result close to AGI, and not a 100,000× speedup claim |
| **Evidence** | `constructive/results/latest.json` (stamp 2026-11-18T11:23:03, runtime 151 s; byte-identical to `results-20261118-112535.json`) |
| **Tests** | `tests/test_constructive.py` → 19 passed; full repo `python -m pytest -q` → **145 passed, 1 skipped** |
| **Environment** | Intel64 Family 6 Model 140 Stepping 1 (i5-1135G7 class), 4C/8T, 16.82 GB RAM, Windows 11 26200, Python 3.13.15, numpy 2.5.3, no CUDA, no C compiler, mains-powered |

All numbers below come from the harness; nothing was hand-tuned to a result. Baselines
were fixed before reading results, then audited and found weak for structural reasons —
that audit is part of the story and is reported in §6.

---

## 1. What this module is, and what it is not

**Is.** A small CPU-only learner with an explicit internal representation of concepts
(typed programs with validation sets, guards and confidences), search-based rule
acquisition, compositional use of stored concepts, abstention on insufficient evidence,
quarantine-based error localization, and scoped repair. It learns from **k = 3–4 labelled
examples** and answers 20 held-out test items.

**Is not.** It is not the omega-nano language model, not an agent architecture, not an
EXP-005 implementation, not a symbolic-only rule engine with the learning bolted on, and
not evidence about general intelligence. `omega_nano/` was not modified. The synthetic
task ontology overlaps with the programs the learner uses (both come from the same
DSL); these results demonstrate internal consistency and sample-efficiency of the
*mechanism*, not transfer to natural tasks.

## 2. The mechanism in 200 words

A **Concept** is a frozen, typed program in a small term language over scenes, lists and
numbers (`constructive/core.py`, `constructive/model.py`). Learning is a **concept
graph**, not a weight vector.

1. **Induce.** Type-directed bottom-up enumeration builds programs from subprogram
   *values* (never re-evaluating subtrees). Observationally equivalent programs are
   deduplicated; hypotheses that agree on training inputs but disagree on *unlabelled*
   probe inputs are retained as distinct hypotheses, so disagreement stays visible.
2. **Validate.** A held-out example gates acceptance. Fitting the support set is never
   sufficient. Every accepted concept carries its validation evidence.
3. **Abstain.** Low posterior margin, or irresolvable ambiguity at low support → refuse.
4. **Compose.** Validated concepts are *single units* in a cost-restricted search, so
   two-step behaviour is assembled, not rediscovered.
5. **Repair.** Global rewrite only if it preserves all old examples within edit distance
   2; otherwise a scoped child concept guarded by a predicate that separates new from old
   inputs, and the parent locally restricted with the negated guard; otherwise reject. A
   rule fitting all but one example is accepted with that example **quarantined** as the
   identified violation.
6. **Local memory.** A new concept is appended with its own body and validation set;
   existing concepts are never retrained.

The only learned parameters anywhere are 17 weights of a linear retrieval prior
(Experimental G). Ablation shows they change **nothing** — see §6.4.

## 3. Experimental design

* **Generators.** Every task derives from a hidden program in the same DSL. Support,
  validation (1 example) and test (20 items) are disjoint index ranges; duplicates are
  rejected. Each task re-rejects training data that **any** primitive in a deliberately
  *stronger* distractor vocabulary (constants ±20, `lt(20)`, `mod(9)`…) already explains —
  the intended rule must be identifiable among single primitives.
* **Systems.** CCM (the mechanism above) vs. **memoriser**, **nearest-neighbour**
  (role-transfer for scenes), and a small **numpy MLP** (per-entity scorer with a softmax
  for relational; target-standardised regression for numbers; padded list encoding; L2).
* **Fair label budget.** CCM induces on the k support examples and uses the held-out
  example only to accept/reject. Baselines receive **both** the support and the held-out
  example as training data. CCM receives unlabelled probes (no labels); baselines receive
  none. This is the only information asymmetry and it favours baselines in one way
  (labels) and the model in another (unlabelled inputs) — documented, not hidden.
* **Metrics.** Accuracy (exact), coverage (answered/abstained), tolerance accuracy
  (within max(1, 1%)) and MAE for numeric targets, examples-to-competence (smallest k
  whose mean test accuracy reaches **0.8**), CPU time, traced peak memory, search nodes.
* **Seeds.** 3 per condition (10 for the noise experiment). A "100,000×" efficiency target
  is a research aspiration and is **not** assumed; nothing in this report is conditioned
  on it.

## 4. Environment and reproducibility

```
cd "D:\ai model"
python -m pytest tests/test_constructive.py -q     # 19 passed
python -c "import constructive.bench as b; r=b.run_all()"
python -m pytest -q                                 # 145 passed, 1 skipped
```

Results: `constructive/results/latest.json` (plus `results-<stamp>.json` per run;
`latest.json` is a byte copy of the newest run). Peak memory is *traced Python heap*
(tracemalloc), excluding native allocations. Wall times are single-machine, cold,
unpinned. Fast runs (~0.004 s) are dominated by Python call overhead; treat them
as "sub-millisecond-ish", not as a latency claim.

## 5. Results

### 5.1 Experiment A — relational concepts (scenes, target = entity)

| System | n | Acc | Coverage | Fit (median) | Peak heap |
|---|---|---|---|---|---|
| **CCM** | 27 | **1.000** | 1.00 | 0.0036 s | 55 KB |
| Memoriser | 27 | 0.000 | 0.00 | 0.000 s | 1 KB |
| MLP (24×16, softmax scorer) | 27 | 0.378 | 1.00 | 0.237 s | 31 KB |
| Nearest-neighbour | 27 | 0.331 | 1.00 | 0.0005 s | 5 KB |

Per-rule CCM = **1.00 for all 9 rules** (`left_of`, `right_of`, `above`, `below`,
`same_color`, `same_shape`, `nearest`, `biggest`, `smallest`). Competence threshold
reached at **k=3** for `left_of`/`same_shape`; 1.0 at k=4, 6, 8, 12, 20, 40, 80.

### 5.2 Experiment B — rule discovery on lists (target = int)

| System | n | Acc | Tol-Acc | Median MAE | Coverage | Fit (median) |
|---|---|---|---|---|---|---|
| **CCM** | 27 | **1.000** | 1.00 | 0.0 | 1.00 | 0.010 s |
| MLP | 27 | 0.150 | 0.417 | 2.6 | 1.00 | 0.045 s |
| Nearest-neighbour | 27 | 0.137 | 0.424 | 2.5 | 1.00 | 0.000 s |

Per-rule CCM = **1.00 for all 9 rules** (`sum_list`, `max_list`, `range_list`,
`count_even`, `sum_squares`, `head`, `sum_of_evens`, `distinct_count`, `sum_top3`).
See also **Experiment D_pairs** (two-argument arithmetic: `sum_pair`, `diff_pair`,
`gcd_pair`, `max_pair`): **CCM 12/12**, MLP 0.388, KNN 0.142.

Learning curves (list rules, n_test=12): competence threshold reached by CCM at **k=2**
for `sum_list` and `sum_of_evens`; MLP remains below threshold at every k ≤ 24
(0.167 at k=24 for `sum_list`).

### 5.3 Experiment C — compositional learning

Protocol: teach A (`evens`) and B (`sum_top3`, cost 3) separately (k=4 each), then induce
T = `sum_top3(evens(x))` with the A and B concepts as units under a **cost-2** fresh
search. Controls: fresh model, same search, no concepts; flat search at cost 4.

| Step | 4/4 seeds | Test acc | Time | Notes |
|---|---|---|---|---|
| learn A | ✅ | **1.00** | 0.011 s | `evens(x)` |
| learn B | ✅ | **1.00** | 2.98 s | `sum_list(take(3)(sort_desc(x)))`, ~60 k nodes |
| compose T = `cB(cA(x))` | ✅ | **1.00** | **0.10 s** | one validated concept |
| compose, no concepts | ❌ 0/4 | 0.00 | 0.10 s | search cost-2 finds only `4` |
| flat search, cost 4, 600 k nodes / 10 s budget | ❌ 0/4 | 0.00 | ~10 s each | all timed out and rejected |

One-off measurement outside the harness (documented honestly): with a **120 s / 4 M
nodes** flat cost-4 budget, seed 1 *accepted* the spurious
`sum_list(filter_gt(2)(take(4)(evens(x))))`, passing the single validation example at
**45 % test accuracy** — held-out validation is a gate, not a guarantee; the effective
defense here is the cost-2 composition path.

### 5.4 Experiment D — mathematical structure (single ints / pairs)

| System | n | Acc | Tol-Acc | MAE | Coverage | Fit (median) |
|---|---|---|---|---|---|---|
| **CCM** | 24 | **1.000** | 1.00 | 0.0 | 1.00 | 0.004 s |
| MLP | 24 | 0.240 | 0.317 | 7.4 | 1.00 | 0.045 s |
| Nearest-neighbour | 24 | 0.210 | 0.279 | 12.4 | 1.00 | 0.0001 s |
| *pairs* CCM | 12 | **1.000** | | | 1.00 | 0.004 s |

Per-rule CCM = **1.00 for all 8 single-int rules** (`square`, `sum_digits`,
`digital_root`, `fib`, `is_prime`, `even`, `add(3)`, `mul(3)`) and **1.00 for all 4 pair
rules** (`sum_pair`, `diff_pair`, `gcd_pair`, `max_pair`). The two rules where baselines
look respectable are class-skewed or one-feature (`is_prime` 0.82, `even` 0.70/0.65 for
MLP/kNN, `smallest` 0.82) — majority-class behaviour, not rule learning (see §6).

### 5.5 Experiment E — continual learning (5 scene tasks, sequential, same model)

| System | Final mean acc (5 tasks) | Forgetting | First-task at end |
|---|---|---|---|
| **CCM** | **1.000** | **0.000** | **1.000** |
| MLP sequential fine-tune | 0.267 | 0.350 | 0.233 |

The MLP fine-tunes per task on the union of everything seen so far for that task; each
new task erodes earlier competence (forgetting 0.35). CCM stores one concept per task and
retrieves behaviourally — nothing is retrained. This is the strongest retention result
and the cleanest demonstration that a concept graph and a weight vector differ in kind.

### 5.6 Experiment F — error-driven correction

| Scenario (n) | Result | Details |
|---|---|---|
| One label corrupted (**10 seeds**, k=4) | **10/10 localised** | accepted with **exactly** the corrupted index quarantined; test accuracy **1.000** |
| Contradictory evidence (4 seeds) | **4/4 rejected** | coverage 0.0; model refuses rather than accepts a mid-split hypothesis |
| Mirrored-scene shift (4 seeds) | **scoped repair 4/4** | old knowledge intact: `mirror(x)` guard on child, parent restricted to `not(mirror(x))`; old retention **1.00**; new accuracy **1.00** |

Noise-localisation fit median 0.41 s; peak traced heap 3.98 MB (the quarantine path
enumerates near-miss candidates). The contradiction case is *rejected*, not learned —
that is the correct conservative outcome, and the failure-mode boundary is stated in §7.

### 5.7 Experiment G — is the learned prior doing anything? (ablation)

Mixed continual suite (scene → list → int, 6 tasks), 3 seeds:

| | with prior | prior zeroed |
|---|---|---|
| accuracy | 1.000 | 1.000 |
| identical predictions | 100 % | — |
| concepts formed | 6 | 6 |
| parameters | 34 (17 prior) | — |
| stored DSL nodes | 17 | — |

The 17 weights change **no** prediction. They are kept, reported, and should be deleted
or replaced in any future version. The model's intelligence lives in the concept graph,
not in these weights.

### 5.8 Learning curves and the 100,000× target

`left_of` / `same_shape` (3 seeds):

| k | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 20 | 40 | 80 |
|---|---|---|---|---|---|---|---|---|---|---|
| CCM `left_of` | .38 | .67 | **1.0** | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| MLP `left_of` | .22 | .23 | .25 | .23 | .33 | .40 | .30 | .30 | .52 | .70 |
| CCM `same_shape` | .07 | .67 | **1.0** | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| MLP `same_shape` | .18 | .22 | .27 | .28 | .25 | .23 | .22 | .25 | .28 | .23 |

Competence (≥0.8) at **k=3** for CCM; MLP does not reach it anywhere in the measured
range (best 0.70 at k=80). List curves behave identically (§5.2). Memory and
time-to-competence: k=3 fits cost ≤0.01 s and ≤200 KB.

**Benchmarks vs. target.** The 100,000× number was never assumed and is **not met**;
no experiment in this project produces it, and none should be read as supporting it.
The honest, measured version:

* CCM is competent from **2–3 examples**; no tested baseline reaches competence even at
  **k=80** — an unbounded-in-the-measured-range example-efficiency advantage, i.e. a
  minimum observed sample-efficiency ratio on the order of **27–40×**, not 100,000×.
* Where a baseline did reach quality (6 of 78 matching A/B/D episodes), CCM was faster
  in 4 of them (**~11× to ~134×** fit-time, `smallest`/`is_prime`/`even`); the other 2
  were unbalanced class-prior near-trivial tasks where `nearest_neighbor` is
  effectively free (see §6).
* The curves-speedup table in the harness is therefore *empty* — correctly: no
  matched-quality ratio exists when no peer crosses the bar in the measured range.

## 6. Findings, including the ones that look bad

1. **The baselines needed repair, not the CCM.** The MLP originally showed a
   *memorised-training* pattern (train loss ≈ 3e-6, test ≈ 8/20 at k=20) because
   per-scene gradients were summed, making the step size grow with k; targets were
   unscaled; and the list encoding was a bag-of-stats that forced the network to
   multiply two features to recover `sum_list`. It was rebuilt: per-scene **averaged**
   gradients, L2, target standardisation, a padded slot encoding for lists, and true
   sequential fine-tuning for Experiment E. The baseline then behaves **unchanged in
   outcome** — it still fails every fair few-shot comparison — but it is now a
   defensible baseline, not a straw man.
2. **Memoriser = 0.000 everywhere** (as designed).
3. **Duplicate hypotheses can hide the right rule.** Early results rejected `digital_root`
   because the true program was a *deduplicated alternative* of a cheap near-miss.
   Deduplication now keeps up to 8 alternatives per behavioural class, and alternatives
   are tried against held-out validation before acceptance. This was a real bug found by
   the harness, not a cosmetic change.
4. **A held-out example can be fooled.** (§5.3, seed 1) The spurious rule passed one
   validation example. Defence-in-depth is cost and composition budget, not this one
   example. Validation must remain a gate, never a guarantee claim.
5. **Ambiguity is real at tiny support.** `above` and `is_prime` have legitimate
   observationally-distinct twins at k≤3; the model abstains rather than guess (by
   design). k=4 resolves it.
6. **Cost-4 flat search is intractable-ish.** A `cost-4` cold search with a 4 M-node
   budget still found only an overfit, 45 % rule in one seed and timed out in the bench.
   Search cost, not learning, is the bottleneck.
7. **The checker caught class-skew wins.** In 78 A/B/D episodes a baseline reached ≥0.8
   only on majority-class/single-feature rules (`is_prime`, `even`, `smallest`); two of
   those episodes were faster for the baseline by construction (NN lookup ≈ 0.000 s).
   Reported rather than suppressed.
8. **The prior is redundant.** §5.7: 100 % identical predictions. Negative result, kept
   in the open.
9. **Contradictory evidence → refusal.** Correct conservative behaviour; it is not a
   learning mechanism for mixed contexts (stated, not hidden).
10. **Numbers-as-exact-integers are strict for continuous models**; we report
    tolerance-accuracy and MAE alongside exact match precisely for this reason.

## 7. Honest limits and failure modes

* **Synthetic ontology overlap.** Tasks and DSL come from the same vocabulary; success
  measures mechanism sample-efficiency under a shared ontology, not transfer to natural
  input.
* **Hand-designed primitives and constants.** The learner does not invent primitives;
  anything outside the DSL is unreachable by construction.
* **Search complexity.** A cost-3 search over scenes (~630 nodes) costs ~0.04 s at
  k=4; cost-3 list rules cost ~60–100 k nodes (~0.6–3 s); cost-4 flat search, no. This
  is why the mechanism is worth something *despite* the small DSL.
* **Quarantine assumes ≤1 inconsistency in a few-example support set**; two corrupted
  labels at k=4 are rejected (coverage 0.0), which is correct but uninformative.
* **Junk-accept channel not fully modeled.** Validation gate + composition budget is the
  defence; a larger held-out set would harden it at the cost of more evidence per concept.
* **Seeds.** 3 per cell (10 for noise); means hide variance; `results/latest.json` carries
  per-episode rows for replication.
* **No claim for omega-nano integration, natural language, or the EXP-005 plan.** The
  corpus-licence contamination documented in `PHASE-0-REPOSITORY-AUDIT.md` is unaffected
  by this module and remains unresolved. This module uses synthetic data only, so it is
  not contaminated by that issue and also not evidence about resolving it.
* **`is_prime`/`even` baselines match via class prior**; the harness deliberately shows
  both exact and tolerance metrics so this effect is visible, not buried.

## 8. Scorecard against the required experiments

| Required experiment | Status | Evidence |
|---|---|---|
| A Relational concepts | ✅ pass | 27/27, O(10 ms) |
| B Rule discovery | ✅ pass | 27/27 |
| C Compositional learning | ✅ pass | 4/4 seeds + 2 negative controls |
| D Mathematical structure | ✅ pass | 36/36 (ints + pairs) |
| E Continual learning | ✅ pass | forgetting 0.000 vs 0.35 (MLP) |
| F Error-driven correction | ✅ pass | 10/10 localised; 4/4 contradictions rejected; 4/4 scoped repair |
| G (added) prior ablation | ✅ honest negative | no effect |

| Required artifact | Status |
|---|---|
| concept representation | ✅ |
| constructive rule acquisition | ✅ |
| validation / rejection | ✅ |
| compositional reasoning | ✅ |
| local knowledge integration | ✅ (scoped guards, no retraining) |
| retention + forgetting evaluation | ✅ (E, G) |
| self-correction | ✅ (F mirror) |
| synthetic generators | ✅ |
| fair baselines | ✅ (repaired MLP, NN role-transfer, memoriser; label budget fair) |
| benchmark + tests + runner | ✅ (`constructive/bench.py`, 19 tests, 145 full-repo) |
| speedups only at matched quality | ✅ (curves tooling + 6 episode-level matches reported) |
| 100,000× honesty | ✅ not met; reported as ≥27–40× observed sample-efficiency, nothing more |

## 9. What would count as progress from here

1. **Delete the prior** and re-benchmark; if accuracy stays 1.0 everywhere and the suite
   stays green, keep it deleted.
2. **Cheaper synthesis.** Value-level composition brought one rule from 10.7 s to 0.6 s;
   further dedup/analogy pruning or a cached feature index is the obvious next lever.
3. **A non-DSL task**, even a toy one outside the ontology, to measure whether the
   mechanism (as opposed to the shared vocabulary) generalises.
4. **Independent re-run** of this suite by a different machine/fresh session to rule out
   systematic tuning effects.
5. **Retained contradiction handling**: learn *two* scoped sub-contexts from a 2-vs-2
   contradiction instead of refusing — a legitimate constructive-learning problem, and a
   genuinely hard one.

## 10. Exact files

Created in this phase (all outside `omega_nano/`):

```
constructive/__init__.py
constructive/core.py
constructive/model.py
constructive/worlds.py
constructive/baselines.py
constructive/bench.py
constructive/README.md
constructive/results/results-20261118-112535.json   (stamped snapshot; latest.json identical)
tests/test_constructive.py
CONSTRUCTIVE-LEARNING-REPORT.md                   (this file)
```

Untouched: `omega_nano/`, all pre-existing tests, `ROADMAP.md`, `CHANGELOG.md`,
`README.md`, all Phase-0 documents (they stand as historical context, not as results).

## 11. The one-sentence version

On synthetic few-shot tasks the constructive mechanism learns the correct, reusable,
validated concept from **2–4 labelled examples** with ~0.001–3 s of CPU, composes it,
repairs itself without destroying old knowledge, and forgets nothing; simple baselines
need more than 80 examples to get close and never arrive. Nothing in this experiment
supports romantic claims; it does support the strong, testable claim that the mechanism
— explicit programs plus held-out validation plus composition — is a real learning
algorithm with a measurably different efficiency profile from fine-tuned weights.
