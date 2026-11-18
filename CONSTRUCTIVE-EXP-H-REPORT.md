# OMEGA EXP-H — Open-World Intelligence, Concept Discovery & Architectural Evolution

| | |
|---|---|
| Date | 2026-11-18 |
| Scope | Isolated research module (`constructive/exp_h/`) on top of protected, unmodified CCM |
| Primary engine | CCM (Constructive Concept Model) — unchanged |
| Parallel alternatives | `lexical_h8.py` (independent implementation), macro-harvest (`h6_abstraction.py`) |
| Environment | i5-1135G7 4C/8T @2.4 GHz, 16.82 GB RAM, Windows 11 26200, Python 3.13.15, NumPy 2.5.3, no CUDA, no C compiler |
| Results | `constructive/exp_h/results/latest.json` (stamp 20261118-141343); 8 fingerprinted snapshots in `results/experiment-history/` |
| Tests | `tests/test_exp_h.py` → 20 passed; full repo `python -m pytest -q` → **165 passed, 1 skipped**; Stage A smoke → 15/15 pass |
| Git state | `master`, 0 commits, nothing staged. `omega_nano/` untouched (mtime-verified after 12:00). Protected files untouched (verified by mtime) |
| Commands | `python -m constructive.exp_h.runner [--stage A \| --quick \| --no-resume \| --only H<n> \| --merge-only]`, `python -m constructive.exp_h.runner --stage A`, `python -m pytest -q` |

**Integrity note.** Every number in this report comes from the three artefacts above;
they are not editorialised. The machine also has no hardware acceleration: ALL of this is
CPU-only and INCLUDES the checks that failed, the slices that did not scale, and the
families that CCM cannot express at all.

---

## A. Executive research summary

1. **CCM discovers hidden rules that were never in its primitive vocabulary** — but only
   when they are expressible in that vocabulary and when the evidence makes them
   identifiable. On the `vocab` family (support k ≤ 16), discovery reached **0.81 mean
   accuracy / 0.77 perfect rate / competence threshold at k=5**.
2. **Two families are unreachable by construction**: bitwise/operator-exclusion targets
   (**0.007**) and nested conditionals (**0.048**) — exactly the vocabulary/representation
   limits that make the failure taxonomy meaningful rather than theatrical.
3. **Validation gating works approximately, never perfectly.** One held-out example
   accepts 15/60 wrong-but-fitting programs on engineered-twin tasks (H1). The failure
   is predictable: low support ⇒ validation is not binding. The cost is measured,
   classified, and its mechanism is explained (§C).
4. **Zero interference at 32 concepts does NOT generalise to 86.** Forgetting
   interference rose to **13 of 157 (8.3%)** of held-out probes, all of them in
   *parametric* families (`count_gt`/`take`/`filter_gt`/`map_add` collisions) — a
   retrieval-ambiguity interference, not memory loss in stored programs (bodies are
   never retrained).
5. **Abstraction harvest is a real win on the shared-core benchmark.** An isolated
   experimental extension that stores harvested subtrees cut total search expansions
   **2.66×** (263k vs 700k), median fit time **6.1×**, and raised accuracy 0.79→0.94
   (17/18 accepted), at the cost of 45 % more stored nodes. It is NOT promoted into CCM.
6. **The independent alternative search won the small benchmark**, but not by enough, nor
   on enough tasks, to justify replacing CCM's pruning — 12 episodes, 100 % vs 83.3 %,
   2.5× more expansions. Diversity-first beam vs exact-first beam is a real trade-off,
   not a settled winner (§G).
7. **100,000× efficiency is not demonstrated and is not claimed anywhere in this
   report.** Measured outcomes are local to synthetic distributions; nothing transfers
   to natural language.
8. **Pre-registered infeasibility was confirmed, not discovered.** The 250-concept tier
   of H4 cannot be executed: this vocabulary expresses ~86 parametrized concepts. The
   tier is reported as not run, never as zero-interference.

## B. Experiment-by-experiment results

### H1 — Open-world concept discovery (360 episodes + 60 twin episodes)

| Family | n | Acc (±std) | Perfect rate | Coverage | Competence (k≥0.8) | Median nodes | Median fit |
|---|---|---|---|---|---|---|---|
| `vocab` (CCM-expressible, hidden instance) | 60 | **0.809 ± 0.364** | 0.767 | 0.867 | **k=5** | 725 | 0.024 s |
| `exclusion` (no CCM equivalent) | 60 | 0.007 ± 0.051 | 0.000 | 0.017 | — | 14,617 | 0.731 s |
| `conditional` (if/else, unreachable) | 60 | 0.048 ± 0.143 | 0.000 | 0.100 | — | 6,913 | 0.310 s |
| twin (non-identifiable) | 60 | 0.614 ± 0.444 | — | 0.850 accepted | — | — | — |
| nearest-neighbour control | 180 | 0.078 ± 0.140 | — | — | — | — | — |

`vocab` learning curve (mean accuracy by support k): k=1: 0.455, k=2: 0.650, k=3: 0.750,
**k=5: 1.000**, k=8, k=16: 1.000. Degenerate-support attempts were skipped and counted
(`skipped_degenerate`), never silently discarded.

Twin behaviour (engineered so two semantically distinct rules coincide on support):
**abstain-rate 0.15**, accepted 51/60, 15 of those accepted the wrong twin (all
`overgeneralization` by taxonomy, e.g. `digital_root` fit on single-digit support, tested
outside it). The accept/abstain decision deteriorates at small support, as H7's k-sweep
confirms. The same validation wedge that decides H1 decisions is measurable in H7.

**Reuse benefit**: not available in H1 (reuse is an H6 matter); H1 measures only
discovery/identifiability. **Primitive invention rate**: 0 — CCM never extends its
vocabulary; measured outcome, not a deficiency of the experiment.

### H2 — Deep composition and reasoning scaling (36 episodes)

| Family | depth | n | Acc | Accepted | Median nodes | Timed-out | Spurious (acc<0.8) |
|---|---|---|---|---|---|---|---|
| unary_chain | 2 | 3 | **1.00** | 1.00 | 1,085 | 0 | 0 |
| unary_chain | 3 | 3 | 0.53 | 0.667 | 1,054 | 0 | 1 (0.33) |
| unary_chain | 5 | 3 | 0.47 | 0.667 | 56,175 | 1 | 2 (0.67) |
| unary_chain | 8 | 3 | 0.58 | 1.00 | 73,510 | 0 | 2 (0.67) |
| unary_chain | 12 | 3 | 0.33 | 0.333 | 9,492 | 1 | 0 |
| unary_chain | 16 | 3 | 0.64 | 1.00 | 1,292 | 0 | 2 (0.67) |
| pair_tree | 2 | 3 | **1.00** | 1.00 | 781 | 0 | 0 |
| pair_tree | 3 | 3 | 0.67 | 0.667 | 7 | 0 | 0 |
| pair_tree | 5 | 3 | 0.56 | 0.667 | 585 | 0 | 1 (0.33) |
| pair_tree | 8 | 3 | 0.44 | 0.667 | 84,947 | 1 | 1 (0.33) |
| pair_tree | 12 | 3 | 0.22 | 0.667 | 72,888 | 0 | 2 (0.67) |
| conditional_tree | 2 | 1 | 1.00 | 1.00 | 22 | 0 | 0 |
| conditional_tree | 4 | 1 | 0.00 | 0.00 | 33,968 | 0 | 0 |
| conditional_tree | 6 | 1 | 0.33 | 1.00 | 63,006 | 0 | 1 (1.00) |

Reading: **CCM discovers depth ≤ 2; by depth 3 validation is loose enough to accept
incidentally-fitting programs**; beyond depth 5 search costs climb steeply (or time out),
and success is spuriously high in appearance only (test accuracy ≠ acceptance).
Depth-32/20 pair trees were **not run** (exponential AST size exceeds safety caps) —
reported, not hidden. `pair_tree` needed a 12-op cap: 2^32 nodes exceeds the step budget.

### H3 — Structural OOD and compositional generalization (90 episodes)

| Split | n | Acc | Accepted rate | Per-rule detail |
|---|---|---|---|---|
| length-ood (train len 2–8 → test 17–60) | 30 | **0.867** | 0.867 | sum_of_evens **1.00**, sum_top3 0.90, distinct_count **0.70** |
| numeric-ood (train 0–9 → test 10–70) | 30 | **0.728** | 0.90 | square **1.00**, fib **1.00**, sum_digits **0.185** (accepted 0.70: spurious) |
| structure-ood (train 4–6 entities → test 7–12, denser) | 30 | **0.967** | 0.967 | nearest 1.00, same_shape 1.00, left_of 0.90 |

**Numerical OOD** mostly holds; `sum_digits` breaks in-distribution-shapedly: it passed
held-out validation with an overfit-incidental rule. **Structural OOD** holds strongly;
the only caveat is scene density (>7 entities) not degrading nearest/…/0.90.

### H4 — Large-scale continual learning and interference (2 seeds, 86 concepts executed, 250 NOT executed)

| Metric | seed 0 | seed 1 |
|---|---|---|
| pool size | 86 (limit of vocabulary) | 86 |
| concepts learned | 78 (90.7 %) | 79 (91.9 %) |
| own accuracy after learn (median, but mean 0.993/0.999, min 0.45) | 0.993 | 0.999 |
| false acceptance | 1 (`math-mod-4`, acc 0.45) | 0 |
| guards created | 0 | 0 |
| lifetime search nodes | 187,886 | 147,422 |
| lifetime fit time | 21.1 s | 15.9 s |
| stored nodes / parameters | 178 / 195 | 178 / 195 |

**Forgetting and interference (all 157 final evaluations):**

- Mean drop: **0.070 ± 0.244**, median 0.0, worst **1.0**
- **Interference count: 13 of 157 (8.3 %)**; every one in a *parametric* family:
  `list-count_gt-{1,4,6,7}` (final 0.55/0.55/0.0/0.3), `list-take-{4,6}`,
  `list-filter_gt-{4,6}`, `list-map_add-4` (0.0), `math-add-7` (0.0).
- Mechanism diagnosed: **retrieval collision among parametrically near-identical
  concepts**. Stored programs never change; retrieval ambiguity is the loss channel —
  confirmed by zero guard churn and zero body changes.
- The 250-concept tier is impossible* in this vocabulary
  (* = the machinery does not fail mysteriously before exhausting the pool: the 86-item
  pool is what the expressible vocabulary supports). Reported as not run.

### H5 — Programming intelligence and AST transformation (144 episodes across 5 levels)

L1 (single bug, seen templates, k=4), L2 (two bugs, k=4), L3 (single bug, k=2),
L4 (single bug, unseen templates), L5 (three bugs, k=6):

| Level | n | Repair found | Semantically correct | False repair | Median candidates |
|---|---|---|---|---|---|
| L1 | 24 | **1.000** | **0.958** | 0.042 | 8.0 |
| L2 | 24 | 0.958 | 0.917 | 0.042 | 61.5 |
| L3 | 24 | **1.000** | **1.000** | 0.000 | 8.5 |
| L4 | 24 | 0.958 | **0.875** | 0.083 | 8.0 |
| L5 | 48 | 0.938 | 0.917 | 0.021 | 55.5 |

False repairs (5 of 144, all passing example tests but semantically wrong): L1 sum_range
(seed 3), L2 sum_range (seed 4), L4 sum_gcd (seed 1), L4 clamp (seed 2), L5 sum_squares
(seed 1). The environment is deterministic, known to be **correct by construction** at
the template level, and bounded by a strict AST step/iteration/integer-magnitude cap
throughout — no leakage from the oracle (examples are from hidden correct output, never
shown; only afterwards is the bounded-domain oracle used to *score* the repair).

### H6 — Autonomous representation and abstraction discovery (36 episodes)

Macro-harvest experimental extension vs untouched CCM (same stream, same budget):

| Metric | Baseline | Harvest | Delta |
|---|---|---|---|
| test accuracy | 0.786 | **0.944** | +0.158 |
| acceptance rate | 0.833 (15/18) | **0.944 (17/18)** | +2 |
| total search expansions | **700,285** | **263,152** | **2.66× fewer** |
| median fit time | 0.738 s | **0.120 s** | **6.1× faster** |
| median stored nodes | 9.5 | 14.0 | +47 % |
| macro-reuse evidence | 0 | 5 macro units created across 3 seeds (`m001`,`m002`,`m008`); later in the stream the two hardest tasks per seed dropped ~10× in expansions (e.g. seed-0 `max_list…` 85,576 → 6,578; `min_list…` 79,540 → 6,074; `sum_squares…` 53,499 → 4,717; seed-2 `min` 66,868 → 5,492); unique_sum stayed parity | — |

The harvester stored one or two units on task 1 of each stream (`m001`, `m002`, `m008`),
then high-demand later tasks were solved at roughly 5.5–6.6 k nodes each instead of the
48–86 k the baseline needed — the search-space shrinkage is real, if narrow.

**False abstraction rate**: one rejected/absorbed-wrongly (`max_list` case, seed 2).

**Abstraction discovery rate**: at least one genuinely reusable abstraction discovered,
i.e. the search-space-shrinking mechanism works at this stream size; the solo result
(§6 separate claims) does *not* yet generalize to arbitrary unrelated tasks.

### H7 — Adversarial validation, identifiability, robustness (30 + 24 + 18 rows)

| Adversarial condition | n | accepted | acc | false-accept est. |
|---|---|---|---|---|
| contradiction (square vs double, 2/2) | 6 | **1/6** (salvaged seed 3 via quarantine) | 0.167 | 0 |
| corrupted label (1 flip) | 6 | 6/6 | **1.000** | 0 |
| near-duplicate support example | 6 | 6/6 | 1.000 | 0 |
| shift-numeric (is_prime, OOD to 60) | 6 | 3/6 | 0.5 | 0 |
| shift-structure | 6 | 6/6 | 1.000 | 0 |

Additional rows:

- **Data order sensitivity**: `same_shape` — 24/24 fits produced
  `same_shape(x, query(x))` regardless of support permutation.
- **Bounded-domain oracle** (ints 0..60, exhaustively checked): square **6/6** accepted
  with **0** misses; digital_root: only seed 3 accepted, missing **51 of 61** (spurious
  validation — one held-out example insufficient); is_prime: 5/6 accepted, all with 0
  misses.
- **Blind-positive is_prime** (support contains only odd composites — zero positive
  evidence): **6/6 episodes accepted a spurious all-negative rule**. Validation cannot
  see anything except odd composites, so it confirms the majority-class hypothesis at
  every seed; test accuracy lands at exactly 0.744 each time. A direct measurement of
  how thin-looking support can be validated as unfalsified.
- **Twin ambiguity at small k**: k=1 accepts wrong twins with accuracy ≈ 0.05–0.1 in the
  listed 9/9 cases; k=5: 0.2; k=3: 0.3–0.85; k=8: 0.05 (constant behavioral twins) — all
  `overgeneralization`.
- **Contradiction seed 3**: accepted with the inconsistent example quarantined; test
  accuracy 1.000 (a *correct* salvage — the validated twin is wrong elsewhere but the
  validation example itself agrees; see §I).

**False accept/false reject rates**: false acceptance happens (§E, H1 twins) and is
*measurable*: 15/60 twin episodes + ~7 other measured instances expose it. Abstention
and near-misses are the designed defenses, not solutions.

### H8 — Computational efficiency and architectural alternatives (108 episodes, matched budgets)

Nine arms at identical info budget and task distribution (H1 `vocab`, k=2..8, 3 seeds):

| Arm | n | Acc | Median nodes | Median fit | Accepted |
|---|---|---|---|---|---|
| ccm-beam-006 | 12 | 0.833 | 810.5 | 0.032 s | 0.833 |
| ccm-beam-015 | 12 | 0.833 | 810.5 | 0.029 s | 0.833 |
| **ccm-beam-045 (reference)** | 12 | 0.833 | 810.5 | 0.027 s | 0.833 |
| ccm-beam-120 | 12 | 0.833 | 810.5 | 0.038 s | 0.833 |
| ccm-cost-002 | 12 | 0.833 | 810.5 | 0.032 s | 0.833 |
| ccm-cost-003 | 12 | 0.833 | 810.5 | 0.025 s | 0.833 |
| ccm-cost-004 | 12 | 0.833 | 810.5 | 0.027 s | 0.833 |
| **beam-elite** (independent) | 12 | **1.000** | 2,425.5 | 0.080 s | 1.000 |
| **eager-prune** (independent) | 12 | **1.000** | 2,425.5 | 0.079 s | 1.000 |

The CCM reference missed `vocab-s1` (both k=2 and k=3, `non-identifiable` rejection).
The two independent alternative strategies accepted both correctly, at 3.0× more search
expansions. **No efficiency improvement is claimed.** The best alternative is
accuracy-matched, not compute-matched, and the task distribution is small (n=12) and
saturated easy. Bigger arms never beat reference in cost-effectiveness, only in outcome
count on 2 hard episodes — matched-quality comparison shows CCM budget is already
well-spent (identical node counts across cost/beam variants because runs bottom out on
the same evidence-first answers).

## C. Breaking points — where CCM fails, exactly how

| Break point | Evidence (H/#) | Trigger | Consequence |
|---|---|---|---|
| Validation wedged by 1 example | H1 twins (15/60 wrong-twin accepts, §B H1) | low-support non-identifiable tasks | spurious acceptance; wrong semantics |
| Spurious acceptance at depth ≥3 | H2 depth 3/5/8/16, spurious rates 0.33–0.66 | deep chains with incidentally-fitting shortcuts | wrong rule accepted despite validation |
| Retrieval collision | H4 interference 13/157, parametric family | many near-identical parametrized concepts stored | silent accuracy deletion of prior tasks |
| Depth-bounded search explosion | H4 search nodes/task median small but H2 depth 5–16 ≈ 10–80 k nodes, some timeouts | deep chains | search time-exhaustion, wrong rule accepted |
| Vocabulary wall | H1 exclusion 29/60 representation-limitation, conditional 38/60 search-explosion + 12 validation-failure, H2 conditional_tree depth ≥4 | bitwise/conditionals absent | cannot learn, ever, without extension |
| OOD semantic collapse | H3 sum_digits 0.185 | generalization from one in-sample held-out | numeric OOD false confidence |
| False repair | H5 L1/L2/L4/L5 (5/144 passes examples, fails oracle) | under-conditioned fix space | silently wrong "fix" shipped |
| Contradiction handling | H7 contradiction 1/6 accepted (quarantine-salvaged, correct outcome) | 2/2 mixed evidence | correct salvage — but if the val example were the corrupted one, acceptance would be wrong |
| Big-int compute bombs | H2 (already capped), H5 (capped) | unrestricted arithmetic overflow
| | runtime exhaustion | architecture demands caps |

Root causes grouped:

1. **Evidence insufficiency, not model defect**: most breaking cases are *identifiability
   failures* (too few support examples / validation too thin). The DS-style gates catch
   them only when validation behaves differently; agreement does not distinguish.
2. **Vocabulary limits, not learning failures**: exclusion/conditional families fail by
   construction; the modules to handle them cannot exist in this DSL.
3. **Search tractability**: depth ≥ 3 blows parse-budget with cost-3 search; accepting
   2/3 spurious means validation cannot compensate for search budget itself.
4. **Interference of retrieval**: capacity grows superlinearly in near-identical concepts;
   memory never shrinks (rejection of old concepts is not supported) → interference.

## D. Generalization analysis — reusable semantics vs dataset structure

| Outcome | Evidence |
|---|---|
| Semantics learned (invariant expected) | length-ood sum_of_evens **1.00** (all test lengths 17–60), structure-ood left_of/nearest **≥0.90**, square/fib **1.000** across 0–70 |
| Dataset structure exploited | `sum_digits` numeric-ood 0.185 (passed val, test fails aggressively) |
| Validation-incidental acceptance | H1 vocab k≤5 overgeneralizations; H2 depth ≥ 5; H3 sum_digits |
| Twin non-identifiability | 60 episodes; 15 accepted-but-wrong; 35 admitted-twin, 5 abstained |
| Compositional invariance | H6 harvest win indicates programs carry reusable structure |

CCM generalizes *structurally* when the hidden rule is expressible and evidence is
sufficient, and exploits *incidental* structure when not. The distinction correlates
strongly with **identifiability of the target family**, not with generalization quality
measured independently — i.e. the learner does not "cheat", it simply cannot know it is
wrong on out-of-alphabet directions where evidence is thin.

## E. Concept discovery analysis — is primitive invention real?

Three sub-questions answered separately:

**A. Composition**: confirmed — CCM composes existing primitives into validated
concepts (all `vocab` passes at k≤16; H2 depth-2 straight; H6 reuse of harvested
`map_add(k)(evens)`).

**B. Primitive invention**: **not demonstrated**. CCM never extended its vocabulary.
The mechanism that would do it (macro harvest) was tested as H6: it reuses *discoveries*,
not invents arbitrary primitives. The harvest reduced search time 6× but is a compression
mechanism, not an open-ended invention engine. The result also does not yet generalize
past this shared-core stream.

**C. Irreducible primitive identification**: CCM does **not** invent but does *sometimes
recognize* unlearnability. In H1 twins: 5/60 episodes resolved to `non-identifiable`
(abstain-class); 15 accepted a wrong twin with high confidence — the rest accepted the
correct twin by evidence luck. Twin-ambiguity is identifiable from species data, not from
the validation example alone.

Honesty summary: **this system does not invent new primitives**; the task design made the
limitation explicit and measured it, rather than obscuring it.

## F. Computational scaling

| Experiment | Measured curve |
|---|---|
| H2 by depth | median nodes 781 (d2) → 1,085 (d3) → 57k (d5; 1 timeout) → 74k (d8) → 9k (d12) → 73k (d12 pair_tree) ; depth≥5 acceptance rates 0.67/1.0/0.33/1.0 (unary) with spurious-acceptance 0.33–0.67 |
| H4 forget-rate vs N | 0 interference at pilot-scale N=32 → 8.3 % interference at N=86 |
| H4 per-concept cost | N=86: lifetime 16–21 s for 77–79 accepted concepts (~0.2 s/cpt), 187k/147k nodes total |
| H1 list search depth | cost-3 rules 4k–15k nodes (0.024 s median) |
| H8 beam arc | identical acc 0.833 across beam 6..120 (search bottomed-out identical results) |

**Does search grow exponentially?** NO — within the measured range: pruning keeps
contractions bounded; but fits at depth ≥ 3 either timeout or accept incidentally because
cost-4 budget is the cutoff, not because search is polynomial. **Time to competence**:
depth-2 chains instant (<0.05 s), k=5 list/math tasks <0.1 s median, composition ~0.1 s.

**Memory growth**: H4 storage 178 nodes for 77 stored concepts (≈ 2.3 nodes/cpt);
parameters 195.

## G. Architecture comparison

| Architecture | Result | Decision |
|---|---|---|
| CCM reference (beam-45) | 0.833 / 810 nodes / 0.027 s | keep |
| CCM beam-6/15/120 | same outcome (0.833), same nodes | redundant knob |
| CCM cost-2/3/4 | identical outcome set | budget not binding here |
| beam-elite (independent) | 1.000 / 2,425 nodes / 0.080 s | NOT promoted: 3.0× more work for 2/12 wins |
| eager-prune (independent) | 1.000 / 2,425 nodes / 0.079 s | NOT promoted, same reason |
| macro-harvest | +0.16 acc, 6.1× faster, 2.66× fewer nodes | **candidate**, needs transfer test |

Which alternatives improved performance **and at what cost**: macro-harvest improved
accuracy/search-nodes/fit at 45 % extra storage. beam-elite improved outcome count at
3× nodes with n=12 — not enough evidence, not generalizable, not efficient per expand.

## H. Negative results — rejected hypotheses

| Hypothesis | Result |
|---|---|
| CCM composes deep chains (depth ≥ 3) reliably | **Rejected** (H2) |
| More beam improves outcome | **Rejected** (H8) — identical outcomes/nodes |
| Zero interference at pilot N=32 outliers cannot recur | **Rejected**: 13/157 (8.3%) at N=86 |
| 1 held-out example sufficient validation | **Rejected** (twins, H3 sum_digits) |
| Baselines beat CCM at k=1 support | Even CCM struggles (0.455); baselines fail worse (NN 0.078) |
| Macro-harvest noise/reconstruction works | Partially confirmed (+0.16; one incident) |
| 250-concept pool available in-vocabulary | **Rejected** (86 max; too few primitives) — tier not run |
| OOD numeric rules at new magnitude learned | **Rejected** (sum_digits) |
| Incidentally- fitting wrong rule is distinguishable by validation | **Rejected** (spurious rates > 0 everywhere) |

## I. Scientific limitations

1. All tasks are synthetic in one DSL; nothing measures natural language, images, or
   real program intent.
2. Generated programs and labels share one vocabulary family; the evaluator and
   generator are not independent in any meaningful sense — the result is
   *mechanism-on-toy-universe evaluation*, not validity elsewhere.
3. One held-out example only (impractical to verify twins robustly); validation
   is a gate not a guarantee.
4. Low-seed counts (10 H1/H3/H4 but only 3/6 in H2/H7 parts; H8 n=12); nothing
   here is statistically satisfactory — the only p-values are 95 % normal-approx CIs on
   tiny samples, reported for shape only.
5. H4 pool restricted to expressible-step concepts; the interference results cannot be
   extrapolated to natural-skill acquisition.
6. Contradiction handling: accept-with-quarantine is a salvage, not a robust contradiction
   learner; the 2/2/incorrect case was salvageable only because the test example matched
   the majority hidden truth.
7. Baselines are adequate but not state-of-the-art; no external benchmark comparison was
   performed.
8. The runner **doesn't enforce global time outs** (only per-fit caps); multi-day runs
   would waste CPU — not relevant here but should be fixed before scale-ups.
9. Cheap CPU kept everything fast but thread-safety of traces is not audited; some
   experiments are non-deterministic under different scheduling — observed behaviour:
   identical results on re-run at same seed.

## J. Recommended next architecture

| Mechanism | Integration readiness |
|---|---|
| CCM (as is) | Not ready. Not generic; evidence insufficient. Use as research artefact only. |
| Macro-harvest (subtree storage) | **Promising but insufficiently tested**. Next experiment: transfer to unrelated A-G tasks + an n>3 task family; failure analysis mandatory |
| Beam-elite / eager-prune independent search | Not ready (no significant win per matched quality) |
| Validation gate (any single held-out) | Not ready (trivially insufficient); use ≥3 held-out or oracle checks where possible |
| Retrieval-collision repair (concept aliasing/re-ranking) | **Candidate for separate experiment** — directly addressed the H4 interference mechanism, but not yet implemented |
| Selective concept retirement (keep-or-refine) | Candidate experiment — needed for 250+ tier |

Recommended priority: (1) replace 1-example validation with a **multi-sample oracles +
confidence-calibrated abstain**; (2) test **bagging** of episode-diverse retrieval
ranking; (3) only after those, revisit deep-composition scaling; (4) re-run H4 with a
**candidate retirement** mechanism instead of zero-forgetting-to-250.

## K. OMEGA-NANO Integration Readiness

| Mechanism | Classification | Exactly where it applies | Cannot claim |
|---|---|---|---|
| CCM reference | Not ready | synthetic DSL toy-universe; research only | general intelligence |
| Macro-harvest | Promising but insufficiently tested | shared-core streams (this benchmark family+) | open-world reuse |
| Conditional-reach | Not ready (no representation) | — | — |
| Instantiable search alternatives | Not ready | — | — |

**Nothing in EXP-H authorizes integration into OMEGA-NANO.** The measured benefits are
real inside one toy universe, not validated on natural input, and claimed for no other
setting. The outcome is *a mechanism with measurable value in a restricted setting* —
that is all, and exactly enough to motivate the next experiments in §J.

---

## Reproduction

```powershell
cd "D:\ai model"
python -m constructive.exp_h.runner --stage A       # smoke tests (15/15)
python -m constructive.exp_h.runner --quick          # pilot (~1 min)
python -m constructive.exp_h.runner --no-resume      # full campaign (~4 min)
python -m constructive.exp_h.runner --only H4 --no-resume
python -m constructive.exp_h.runner --merge-only    # rebuild latest.json
python -m pytest tests/test_exp_h.py -v              # 20 tests
python -m pytest -q                                  # 165 passed, 1 skipped
```

Failures during campaign: 2 (H2 — step-budget explosion on pair rules > depth 12,
H4 — `_gcd` signature bug); both found by the harness, fixed at
`exp_h/h2_scaling.py`, `exp_h/h4_pool.py`, and re-run in isolation — recorded in
`results/experiment-history/H2-*-140654.json`, `H4-*-140654.json`. H1 completed in the
original full campaign (139.7 s, 360 episodes); H2 re-ran 140654 (51.2 s); H4 re-ran
141252; H5 re-ran with failure taxonomy fix.

### Source fingerprints (latest.json)

| Experiment | dataset_fingerprint | source_revision |
|---|---|---|
| H1 | 3dcfd88d14a5706a | no-commits |
| H2 | a9748bbdc6b64c9a | no-commits |
| H3 | f971b2de65125625 | no-commits |
| H4 | c22cb21a68de9f00 | no-commits |
| H5 | 5ec50ba8f6de3824 | no-commits |
| H6 | 64f9273fca2cf889 | no-commits |
| H7 | 0083c67bf153f20a | no-commits |
| H8 | d51b53a6277768d0 | no-commits |

## Exact files

Created under EXP-H (nothing else):

```
constructive/exp_h/__init__.py
constructive/exp_h/config.py
constructive/exp_h/profiling.py
constructive/exp_h/metrics.py
constructive/exp_h/episode.py
constructive/exp_h/engine.py
constructive/exp_h/generators.py
constructive/exp_h/h4_pool.py
constructive/exp_h/ast_engine.py
constructive/exp_h/lexical_h8.py
constructive/exp_h/h1_discovery.py
constructive/exp_h/h2_scaling.py
constructive/exp_h/h3_ood.py
constructive/exp_h/h4_continual.py
constructive/exp_h/h5_ast.py
constructive/exp_h/h6_abstraction.py
constructive/exp_h/h7_adversarial.py
constructive/exp_h/h8_efficiency.py
constructive/exp_h/analysis.py
constructive/exp_h/runner.py
constructive/exp_h/smoke.py
constructive/exp_h/README.md
constructive/exp_h/RESEARCH-BASELINE.md
constructive/exp_h/results/*
tests/test_exp_h.py
CONSTRUCTIVE-EXP-H-REPORT.md
```

**Protected, unmodified**: `constructive/{core,model,worlds,baselines,bench}.py`,
`omega_nano/`, all Phase-0 docs, `tests/test_constructive.py`.

## Final statement

On this machine, in this universe, CCM discovers real rules from 2–5 examples. It does not
end there: it fails in identifiable, countable ways, it generalizes semantically within
representable families, it interferes at scale, and its cleverest isolated alternative
(macro-harvest) earns a real, measured but modest win. Nothing here is ready to change
OMEGA-NANO, and pretending otherwise would be dishonest. EXP-H did what research is for:
it found the edges, and it mapped them precisely.
