# OMEGA EXP-I — Architecture

> Research branch only. Nothing here modifies CCM, `omega_nano/`, protected Phase-0
> architecture, or EXP-H source/results. All code lives in `constructive/exp_i/`; all
> outputs in `constructive/exp_i/results/`.

## 1. Design contract

EXP-I is an **independent** experimental system, not an extension of CCM. It reuses
EXP-H *read-only* only where a baseline comparison requires it. The system is built so
that every claim is backed by a recorded measurement, and so that "I don't know" is a
first-class, tested outcome.

Four epistemic outcomes are kept strictly distinct:
1. **Expressible but not discovered** — a search/resource failure.
2. **Not expressible** — a representation limitation of the declared DSL.
3. **Not identifiable** — the evidence cannot single out one hypothesis.
4. **Exceeds computational limits** — a bounded-resource abstention.

## 2. Module map and dependency order

```
types.py        immutable data: Example, Task, ValidationState, Result, enums
config.py       ExpIConfig, hardware probe, fingerprint/hash helpers
resources.py    ResourceGovernor (deadlines, budgets, cancellation, completion status)
dsl.py          typed AST nodes, type inference, static validation, canonicalization,
                complexity/depth/magnitude limits, SAFE execution
evidence.py     scoring, disagreement, counterexample generation, adaptive selection,
                informativeness/redundancy, contradiction detection, provenance
validation.py   multi-sample validation engine: states, calibration, abstention
retrieval.py    concept store: typed signatures, structural/behavioral fingerprints,
                ranking, ambiguity, collisions, versioned immutable history, retirement
search.py       typed AST search: memoization, subexpression cache, cost-aware
                expansion, hierarchical decomposition, CEGIS-style refinement
generators.py   independent task generator families (no shared evaluator shortcut)
generalization.py  numeric/structural OOD experiments and domain-boundary tracking
repair.py       repair generation ⟂ verification; bounded exhaustive equivalence
abstraction.py  subtree extraction, reusability, validation, provenance, retirement
metrics.py      aggregation, CI, calibration (ECE), accuracy/coverage, comparisons
experiments.py  experiment A–H definitions, each returns a measured block
analysis.py     taxonomy tallies, cross-experiment summaries
runner.py       orchestration, snapshots, immutable history, latest.json, CLI
smoke.py        fast stage-A smoke checks
__init__.py
```

Independence: `dsl.py` and `resources.py` have zero intra-package dependencies beyond
`types/config`. `evidence/validation` depend on `dsl/types`. `retrieval/search` depend
on `dsl/evidence`. `generalization/repair/abstraction` compose the lower layers.
`experiments/runner` sit on top. This keeps each module independently testable.

## 3. Data model (`types.py`)

- `Example(x, y)` — one labelled observation; `x` may be int, list[int], bool, pair.
- `Task(name, family, train, validation_pool, test, input_type, output_type, meta)` —
  support, validation and test are **disjoint** by construction. `validation_pool` is a
  separate labelled set used by the adaptive validator; it is never the test set and
  never used to train candidate generators.
- `ValidationState` enum with the six required states.
- `ValidationResult(state, confidence, calibrated_confidence, evidence, provenance, ...)`.
- `CompletionStatus` enum: `COMPLETED`, `PARTIAL`, `TIMED_OUT`, `ERROR`, `SKIPPED`.
  A timeout can never be mapped to `COMPLETED`.
- `Result` wrappers carry provenance, budgets and completion status.

## 4. Evidence & validation (EXP-I-A)

- A **candidate** is a typed DSL program. The evidence engine scores candidates against
  support, then runs adaptive multi-sample validation: pick the validation observation
  that maximises disagreement among still-consistent candidates (query-by-committee),
  reveal its label, eliminate inconsistent candidates, repeat until budget exhausted.
- Non-identifiability: if ≥2 surviving candidates disagree on any domain point and no
  further discriminating observation is available, return `NON_IDENTIFIABLE`.
- Contradiction: if no candidate is consistent with support, return `CONTRADICTORY`.
- Low support / weak information: `INSUFFICIENT_EVIDENCE`.
- `RESOURCE_LIMITED` when the validation/search budget is exhausted before a decision.
- Confidence calibration: empirical binning fit on a dedicated calibration split,
  reported via Expected Calibration Error (ECE); abstention threshold is chosen from
  calibration, not an arbitrary constant.
- Provenance records which examples were informative vs redundant and which eliminated
  which candidate. Agreement on observed samples is explicitly *not* proof of semantics.

## 5. Typed DSL (EXP-I-B)

Nodes carry explicit input/output types; composition is type-checked. Supported
families: constants, variable, bounded arithmetic (`add/sub/mul/mod/div` with integer
magnitude cap), comparison (`lt/le/gt/ge/eq/ne`), boolean predicates (`is_prime`,
`even`, `odd`, `is_zero`), connectives (`and/or/not`), conditional (`if`), bitwise /
operator-exclusion (`band/bor/bxor/shl/shr`), parameterized list transforms
(`map_add/map_mul/filter_gt/count_gt/take/drop`), reductions (`sum/max/min/len`),
structured list ops (`sort/reverse/unique/evens/odds`). Static validation rejects
ill-typed trees; canonicalization gives a stable structural key; complexity estimation
feeds search cost; depth/node/magnitude caps make execution safe. The DSL reports which
of the four epistemic outcomes applies to a target.

## 6. Retrieval (EXP-I-C)

A `ConceptRecord` stores an immutable `body` plus versioned refinements. Each record
has a typed signature, a structural fingerprint (canonical AST hash), and a behavioral
fingerprint (outputs on a fixed diagnostic input set spanning the parameter ranges of
the H4 collision families). Retrieval ranks by signature compatibility, parameter
distance and behavioral agreement on the query context. If two candidates are similar
but behaviorally distinct, both are returned with `ambiguous=True` and the engine
requests more evidence — never silent substitution. Alias detection flags behavioral
duplicates; retirement marks records inactive while preserving history.

## 7. Search (EXP-I-D)

Bottom-up enumeration over typed expressions with: a global subexpression cache keyed
by canonical form, memoized evaluation keyed by (canonical-form, input-key), cost-aware
expansion order, hierarchical decomposition of the target type, and admissible pruning
only when a type-distance bound is provable. Counterexample-guided refinement adds
discriminating support points when a candidate passes support but fails validation.
Expansions, wall time, peak memory, timeout rate, spurious/correct acceptance and
abstraction reuse are all measured. Budgets are hard caps, never silently raised.

## 8. Generators & generalization (EXP-I-E)

Generators are written independently of the evaluator: each family has its own label
function and RNG, and train/eval use separate generator instances with disjoint
parameter ranges. No implementation object is shared between generator and evaluator.
Numeric-OOD experiments sweep magnitude, sign, zero, length, value distribution,
sparsity and distribution shift; domain boundaries are inferred from validation coverage
and extrapolations beyond them trigger abstention.

## 9. Repair (EXP-I-F)

`generate_repair` receives only visible examples and the buggy program. `verify_repair`
is a separate component that (a) checks bounded exhaustive equivalence on the declared
finite domain, (b) searches adversarial counterexamples, (c) detects when multiple
distinct repairs satisfy visible evidence, and (d) returns an explicit uncertainty state.
A repair passing visible examples alone is never labelled semantically correct. The
oracle (hidden correct program) is used only by the evaluation harness, never by the
generator.

## 10. Abstraction (EXP-I-G)

Subtrees of validated solutions are candidate abstractions; a reusability score weights
frequency, type-generality and behavioral distinctiveness. Each abstraction is validated
on an independent task before storage; provenance records origin. Adversarially similar
but semantically different structures are used to test false-abstraction rejection.
Macros are search units only and are never promoted into CCM or OMEGA-NANO.

## 11. Resource governance (EXP-I-H)

`ResourceGovernor` owns a global wall-clock deadline, per-experiment and per-fit
deadlines, a search-node budget, an optional memory guard, cancellation flags and
structured timeout reports. Each long operation checks `governor.checkpoint()`; exceeding
a budget raises `ResourceLimit` (captured, never converted to success). Cleanup is
deterministic via context managers. Every result carries `CompletionStatus` and
reproducibility metadata (config fingerprint, seeds, source hashes, hardware).

## 12. Persistence & integrity

- `results/latest.json` — merged, machine-readable campaign result.
- `results/experiments/I*.json` — per-experiment current result.
- `results/experiment-history/I*-<stamp>.json` — **immutable** snapshots; never overwritten.
- `results/baseline_hashes.json` — SHA-256 of all protected files, recorded at STAGE 0.
- `results/integrity.json` — re-verification of protected hashes at close-out.

## 13. Scientific protocol

Each experiment reports n, mean, median, std, 95% CI, baseline, new result, absolute and
relative improvement, and failure count, with ≥30 episodes per principal family where
practical and ≥5 seeds for major comparisons; limitations are stated where resources
force fewer. Development and test data are separated; test labels are never consulted by
generation, candidate selection or calibration. Skipped experiments remain `SKIPPED`.
