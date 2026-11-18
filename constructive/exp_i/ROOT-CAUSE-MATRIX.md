# OMEGA EXP-I — Root-Cause Traceability Matrix

> **Status:** STAGE 1 (pre-implementation). All baseline numbers are transcribed from
> `CONSTRUCTIVE-EXP-H-REPORT.md` / `constructive/exp_h/results/latest.json` and were
> independently confirmed against the repository on 2026-11-18. No EXP-I implementation
> existed when this matrix was authored. Measurements marked *target* are acceptance
> criteria, not achieved results; achieved results are recorded later in `EXP-I-REPORT.md`.

## Traceability legend

- **Solvable?** `yes` = addressable by evidence/architecture within this DSL;
  `partial` = mitigable but bounded by representation or information theory;
  `no` = fundamentally unidentifiable from available evidence or impossible in the
  current representation without an explicit extension.
- **Baseline** = EXP-H recorded value. **Target** = EXP-I acceptance criterion.

---

## A. Insufficient validation and false acceptance

| Field | Value |
|---|---|
| **Failure identifier** | F-A1 (wrong-twin acceptance); F-A2 (spurious depth acceptance); F-A3 (blind-positive) |
| **Original experiment** | H1 twins (60 episodes), H2 depth 3–16, H7 blind-positive |
| **Exact observed failure** | 15/60 wrong twins accepted with high confidence; spurious-acceptance rates 0.33–0.67 at depth ≥3; 6/6 blind-positive `is_prime` episodes accepted an all-negative rule |
| **Measured baseline** | wrong-twin acceptance 15/60 (=0.25 of accepted, 0.15 of abstain-rate framing); abstain rate 0.15; false acceptance 15/60; blind-positive 6/6 spurious |
| **Root cause** | A single validation example that lies inside the support-consistent region cannot discriminate observationally distinct hypotheses. Validation agreement is conflated with semantic correctness. Confidence score is an uncalibrated log-loss margin. |
| **Proposed architectural intervention** | Independent Evidence Engine: separate support/validation pools; adaptive multi-sample selection by candidate disagreement (query-by-committee); explicit `NON_IDENTIFIABLE` state; calibrated confidence (empirical binning); explicit abstention thresholds; provenance; contradiction detection. |
| **Expected measurable improvement** | *target* wrong-twin acceptance ≤ 5/60 while coverage stays ≥ 0.6; false-rejection reported separately; ECE < 0.15 on independent calibration distribution. |
| **Possible regressions** | Over-abstention lowers coverage; adaptive selection can overfit validation pool; calibration estimated on wrong split. |
| **Required tests** | misleading single support; observationally equivalent hypotheses; contradictory labels; corrupted validation; blind-positive; calibration on independent distribution; accuracy/coverage trade-off not "reject everything". |
| **Acceptance criteria** | Wrong-twin acceptance < baseline; coverage not collapsed; calibration measured; non-identifiability explicitly reported; neither accuracy nor coverage fabricated. |
| **Solvable?** | partial — solvable when a discriminating validation observation exists in the declared domain; if the domain itself is ambiguous, only `NON_IDENTIFIABLE` is correct. |

## B. Deep composition and search explosion

| Field | Value |
|---|---|
| **Failure identifier** | F-B1 (spurious depth acceptance); F-B2 (search explosion / timeout) |
| **Original experiment** | H2 (36 episodes) |
| **Exact observed failure** | Unary chains depth 3–16: spurious acceptance 0.33–0.67; pair_tree depth 12 acc 0.22 with ~73k nodes; depth-32/20 not run (exponential AST) |
| **Measured baseline** | depth ≤2 acc 1.00; depth ≥3 median nodes 1k–74k, 2 timeouts; spurious up to 0.67 |
| **Root cause** | Bottom-up enumeration without structural decomposition or memoization; validation too weak to reject incidentally-fitting deep shortcuts; no cost-aware expansion or search budget allocation. |
| **Proposed architectural intervention** | Typed AST search with subexpression caching, memoized evaluation, hierarchical decomposition, cost-aware expansion, admissible pruning only where justified, counterexample-guided refinement, incremental validation, explicit search budget. Macro-harvest as candidate, not assumed winner. |
| **Expected measurable improvement** | *target* depth 2/3/5 correct acceptance ≥ baseline at ≤ 0.5× expansions; at depths 8/12/16 correct abstention instead of spurious acceptance; no unbounded growth. |
| **Possible regressions** | Memoization key errors; over-pruning removes valid solutions; caching increases memory. |
| **Required tests** | unary chains d∈{2,3,5,8,12,16}; pair trees; unseen compositions; shared-core; unrelated families; transfer with/without abstractions. |
| **Acceptance criteria** | Expansions/time/memory/timeouts reported; spurious acceptance ≤ baseline; abstention counted as success when unsupported; budgets capped. |
| **Solvable?** | partial — depth-2/3 solvable; very deep chains bounded by computational budget, must abstain. |

## C. Parametric retrieval collisions and continual-learning interference

| Field | Value |
|---|---|
| **Failure identifier** | F-C1 (parametric retrieval collision) |
| **Original experiment** | H4 (2 seeds, 86 concepts) |
| **Exact observed failure** | 13/157 (8.3%) held-out probes lost accuracy; all in `count_gt`, `take`, `filter_gt`, `map_add`, `add` families; stored bodies never changed (zero guard churn) |
| **Measured baseline** | interference 13/157; false acceptance 1 (seed 0); 86-concept capacity (250 tier not run) |
| **Root cause** | Retrieval matched on behaviour over a thin support set, so near-identical parameterizations became ambiguous; no typed signature, no parameter-aware matching, no behavioral fingerprint over a diagnostic input set, no ambiguity escalation. |
| **Proposed architectural intervention** | Independent retrieval with typed concept signatures, parameter-aware matching, structural + behavioral fingerprints, context-sensitive ranking, ambiguity detection with "retrieve both, request evidence", versioned concept records with immutable history, alias detection, collision diagnostics, selective retirement and refinement. |
| **Expected measurable improvement** | *target* interference rate < 8.3%; false retrieval rate < baseline; precision/recall reported; both candidates returned on collision instead of silent substitution. |
| **Possible regressions** | Storing more signatures increases memory and latency; aggressive retirement can delete useful concepts. |
| **Required tests** | exact H4 collision families; ≥2 independent seeds; pool expansion only where DSL supports; parametric near-duplicates; ambiguous context; retirement/refinement history. |
| **Acceptance criteria** | Interference, false retrieval, precision, recall, memory, latency all measured; no silent substitution; no unsupported 250-concept claim. |
| **Solvable?** | yes (for this failure channel) — it is a retrieval ambiguity, fixable by disambiguating evidence; capacity beyond ~86 is a *representation* limit, not solved here. |

## D. Missing conditional and operator-exclusion representations

| Field | Value |
|---|---|
| **Failure identifier** | F-D1 (exclusion unreachable); F-D2 (conditional unreachable) |
| **Original experiment** | H1 exclusion/conditional; H2 conditional_tree |
| **Exact observed failure** | exclusion acc 0.007, conditional acc 0.048; conditional_tree depth ≥4 collapses |
| **Measured baseline** | exclusion 0.007; conditional 0.048 |
| **Root cause** | CCM vocabulary has no conditionals, boolean predicates, comparisons, logical connectives, or bitwise/operator-exclusion operators. Problems fail by construction, not by learning. |
| **Proposed architectural intervention** | Independent extensible typed DSL: conditional expressions, boolean predicates, comparisons, conjunction/disjunction/negation, operator exclusion (bitwise), bounded arithmetic, parameterized transforms, nested composition, structured list ops; strict AST typing, static validation, canonicalization, complexity/depth/magnitude limits, safe execution. |
| **Expected measurable improvement** | *target* express the hidden rules for exclusion/conditional families; correctly distinguish "expressible but not discovered" vs "not expressible" vs "unidentifiable" vs "exceeds computational limits". |
| **Possible regressions** | Larger search space increases explosion; type system bugs; unsafe execution if caps fail. |
| **Required tests** | unseen compositions; held-out operator combinations; deep nested conditionals; type-safety violations; depth/magnitude bombs; safe execution; expressiveness classification. |
| **Acceptance criteria** | Hidden rules expressible in DSL; genuine composition of learned operators into unseen structure; no primitive added merely to pass a benchmark example. |
| **Solvable?** | yes (representation extension) — the DSL is designed to express these. |

## E. Numeric out-of-distribution generalization

| Field | Value |
|---|---|
| **Failure identifier** | F-E1 (`sum_digits` numeric-OOD collapse) |
| **Original experiment** | H3 numeric-ood (30 episodes) |
| **Exact observed failure** | `sum_digits` OOD accuracy 0.185 despite 0.70 acceptance — passed single held-out example with an overfit incidental rule |
| **Measured baseline** | numeric-OOD 0.728 overall; `sum_digits` 0.185; square/fib 1.000 |
| **Root cause** | Validation on in-range examples cannot detect OOD failure; no notion of validated domain boundaries; confidence not tied to domain of validity. |
| **Proposed architectural intervention** | Generalization experiments with independent generator families; domain-boundary tracking; OOD detection and abstention; independent ID/OOD/calibration splits; no shared generator shortcuts between train and eval. |
| **Expected measurable improvement** | *target* system abstains or flags low confidence when extrapolating beyond validated domain; OOD false acceptance lower than baseline; domain boundaries reported. |
| **Possible regressions** | Over-abstention on legitimately generalizable rules; domain heuristic misfires. |
| **Required tests** | magnitude shift; negatives; zero; large bounded ints; length/value/sparsity shifts; distribution shift; compositional OOD; independent generators; listed rules (sum_digits, square, fib, is_prime, sum_of_evens, distinct_count, nearest, same_shape, left_of). |
| **Acceptance criteria** | ID accuracy, OOD accuracy, calibration, false acceptance, coverage, domain boundaries all recorded; narrow-distribution generalization not overclaimed. |
| **Solvable?** | partial — genuine semantic rules generalize; incidental fits must be rejected, not repaired. |

## F. False-positive program repairs

| Field | Value |
|---|---|
| **Failure identifier** | F-F1 (false verified repair) |
| **Original experiment** | H5 (144 episodes) |
| **Exact observed failure** | 5/144 repairs pass visible examples but are semantically wrong (L1 sum_range, L2 sum_range, L4 sum_gcd, L4 clamp, L5 sum_squares) |
| **Measured baseline** | false repair 5/144; semantically correct 0.92–1.00 per level |
| **Root cause** | Repair generation and verification are coupled; passing visible examples is treated as correctness; no adversarial counterexamples or bounded exhaustive equivalence check at verification time. |
| **Proposed architectural intervention** | Separate generation from verification; bounded exhaustive verification over finite domains; adversarial counterexample search; detect multiple plausible repairs; reject memorization-only repairs; explicit uncertainty states; provenance; no oracle leakage into generation. |
| **Expected measurable improvement** | *target* zero false *verified* repairs on the original bounded benchmark; unverifiable cases explicitly reported, never labelled correct. |
| **Possible regressions** | Legitimate repairs rejected when domain coverage is incomplete; higher verification cost. |
| **Required tests** | original 5 false-repair cases; multiple plausible repairs; corrupted evidence; adversarial counterexamples; oracle-leak check; verification coverage. |
| **Acceptance criteria** | False verified repair = 0; unverifiable reported; generation has no hidden oracle access. |
| **Solvable?** | yes (bounded domain) — exhaustive finite-domain verification is decisive within the declared domain. |

## G. Contradiction handling and corrupted evidence

| Field | Value |
|---|---|
| **Failure identifier** | F-G1 (contradiction handling); F-G2 (corrupted label) |
| **Original experiment** | H7 (contradiction 6, corrupted label 6, blind-positive 6) |
| **Exact observed failure** | Contradiction 1/6 accepted via quarantine salvage — correct only because validation example matched majority truth; if validation were the corrupted one, acceptance would be wrong |
| **Measured baseline** | contradiction accepted 1/6; corrupted label localised 6/6 (acc 1.0) |
| **Root cause** | Quarantine is a salvage heuristic, not a contradiction learner; no explicit `CONTRADICTORY` state; no distinction between corrupted evidence and genuinely multi-valued rules. |
| **Proposed architectural intervention** | Evidence Engine contradiction detection; explicit `CONTRADICTORY` validation state; quarantine as a *provenance-tracked* hypothesis with reduced confidence; corrupted-example localization; blind-positive detection (zero positive evidence). |
| **Expected measurable improvement** | *target* contradictory evidence never yields a high-confidence ACCEPTED; corrupted example localized; blind-positive abstains. |
| **Possible regressions** | Legitimate noise causes excess abstention. |
| **Required tests** | contradictory labels; corrupted validation; blind-positive datasets; randomized support ordering. |
| **Acceptance criteria** | Explicit contradictory/insufficient states; no confident answer on corrupted evidence. |
| **Solvable?** | partial — detectable; disambiguation requires clean evidence, otherwise abstain. |

## H. Abstraction discovery and cross-family transfer

| Field | Value |
|---|---|
| **Failure identifier** | F-H1 (false abstraction); F-H2 (narrow transfer) |
| **Original experiment** | H6 (36 episodes, 3 seeds) |
| **Exact observed failure** | 1 false abstraction (`max_list`, seed 2); win confined to shared-core stream; solo result does not generalize to unrelated tasks |
| **Measured baseline** | accuracy 0.786→0.944; expansions 700k→263k (2.66×); fit 0.738→0.120 s (6.1×); stored nodes +45%; 1 false abstraction |
| **Root cause** | Frequent-subtree storage is a compression heuristic with weak validation; reusability measured only on the stream that generated the abstraction. |
| **Proposed architectural intervention** | Abstraction subsystem: subtree extraction, reusability scoring, parameter inference, independent abstraction validation, provenance, false-abstraction rejection, retirement; test transfer to unrelated A–G families, multi-abstraction compositions, adversarially similar structures. |
| **Expected measurable improvement** | *target* quantify transfer benefit vs untouched baseline on unrelated families; false-abstraction rejection demonstrated; report when no benefit exists. |
| **Possible regressions** | Storage overhead; false abstractions hurt accuracy; retirement removes useful units. |
| **Required tests** | original shared-core stream; unrelated families; new structures; multi-abstraction; adversarial near-duplicates. |
| **Acceptance criteria** | Transfer vs baseline measured across families; false abstractions rejected; no promotion into CCM/OMEGA-NANO. |
| **Solvable?** | partial — reusable structure exists on shared-core tasks; broad transfer is an empirical question to be measured, not assumed. |

## I. Global computational resource management

| Field | Value |
|---|---|
| **Failure identifier** | F-I1 (no global timeout); F-I2 (no cancellation/cleanup) |
| **Original experiment** | EXP-H §I.8 runner limitation |
| **Exact observed failure** | Runner enforces only per-fit caps; no global wall-clock deadline; long runs could waste CPU; no structured timeout reporting or cancellation propagation |
| **Measured baseline** | global timeout absent; per-fit caps only |
| **Root cause** | Resource governance was local to fits, not centralized across experiment / campaign. |
| **Proposed architectural intervention** | Central ResourceGovernor: global/campaign/per-experiment/per-fit deadlines, search-node budgets, safe memory limits, cancellation propagation, deterministic cleanup, failure isolation, structured timeout reporting, CPU worker limits, reproducibility metadata. |
| **Expected measurable improvement** | *target* every long-running experiment safely terminable; no partial result marked successful after timeout; valid partial diagnostics preserved and marked incomplete. |
| **Possible regressions** | Overly aggressive deadlines truncate valid experiments; cancellation complexity. |
| **Required tests** | resource exhaustion; interrupted experiments; repeated identical runs; timeout never yields false success; deterministic cleanup. |
| **Acceptance criteria** | Deadlines enforced and reported; incomplete runs flagged; protected files unchanged; reproducibility metadata present. |
| **Solvable?** | yes — engineering control, fully implementable in-process. |

---

## Cross-cutting constraints (apply to every row)

- Never modify original CCM, `omega_nano/`, protected Phase-0 architecture, EXP-H source or results.
- No external cloud APIs, no GPU, CPU-only, existing Python environment.
- No hidden-oracle leakage; test labels never used in training or candidate selection.
- Baselines use identical task distributions, information budgets, evaluation domains, independent seeds.
- Report sample count, mean, median, std, CI, baseline, new result, absolute/relative improvement, failure count.
- Prefer ≥30 episodes per principal family and ≥5 seeds where practical; state limitations otherwise.
- A confident unsupported answer is a failure; abstention is a legitimate, measured outcome.
