# `constructive/` — OMEGA Constructive Intelligence (OCI), isolated research module

A small, CPU-only learning system that builds reusable **concepts** from a few
examples, **composes** them, and **repairs** itself locally. It is not a
language model, not an agent, and has no external LLM dependency. The model
itself is the learner; the retrieval prior is 17 weights and is ablated to
inertness.

> Status: experimental research code. Not integrated into `omega_nano/`.
> Synthetic tasks only. Every number below was measured with the commands in
> "Reproduce" (results artifact: `constructive/results/latest.json`,
> timestamp 2026-11-18T11:23:03).

## Mechanism in one paragraph

A **Concept** is a frozen, typed program in a small term language over scenes,
integer lists and numbers, with an optional guard predicate and a validation
set. Learning is a **concept graph**, not a weight vector:

1. **Induce** — type-directed bottom-up enumeration (programs are built from
   subprogram *values*, not re-evaluated subtrees) with observational
   deduplication. Hypotheses that agree on the training inputs but disagree on
   unlabelled probe inputs are kept as distinct hypotheses so disagreement
   stays visible.
2. **Validate** — a held-out example gates acceptance; held-out evidence is
   required for a hypothesis to become a concept. "It fits the examples" is
   never sufficient.
3. **Abstain** — posterior margin below threshold, or irresolvable ambiguity
   at low support, and the model refuses instead of guessing.
4. **Compose** — previously validated concepts are used as single units in a
   cost-restricted search, so multi-step behaviour is assembled rather than
   rediscovered.
5. **Repair** — smallest justified change: a global rewrite only if it keeps
   every old example and is within edit distance 2; otherwise a scoped child
   concept with a learned guard, and the parent locally restricted with the
   negated guard; otherwise reject. A rule fitting all but one example is
   accepted with the inconsistent example **quarantined** as the identified
   violation.
6. **Local memory** — a new concept is appended with its own body, validation
   set and confidence. Existing concepts are never retrained.

## Measured results (3 seeds per condition unless noted)

| Experiment | Constructive model | best simple baseline |
|---|---|---|
| A relational rules (9 rules) | **100%**, 27/27, fit 0.004 s | MLP 0.378, kNN 0.331, memoriser 0.000 |
| B rule discovery on lists (9 rules) | **100%**, 27/27, fit 0.010 s | MLP 0.150, kNN 0.137 |
| B pairs too (arithmetic) | **100%**, 12/12 | MLP 0.388, kNN 0.142 |
| D single-integer rules | **100%**, 24/24, fit 0.004 s | MLP 0.240, kNN 0.210 |
| C composition (A→B) | **100%**, fit 0.10 s, 4/4 seeds | no-concepts control **0/4**; flat cost-4 search 0/4 |
| E continual (5 scene tasks, sequential) | final **1.00**, forgetting **0.00** | MLP sequential fine-tune final **0.267**, forgetting **0.35** |
| F one corrupted label (10 seeds) | **10/10 localised**, test **100%** | — |
| F contradictory labels | **rejected 4/4**, coverage 0.0 | — |
| F mirrored shift | **scoped repair 4/4**, old retention 1.00, new accuracy 1.00 | — |
| G learned-prior ablation | identical predictions with and without prior | 0.0 effect, 6 concepts, 34 parameters, 17 stored nodes |

**Learning curves.** On `left_of` and `same_shape` the model reaches the 0.8
competence threshold at **k=3**; it stays at 1.0 through k=80. The MLP does not
reach the threshold anywhere in the measured range: 0.30 (k=3) → 0.52 (k=40) →
**0.70** (k=80, left_of); 0.25 (k=80, same_shape). Same for list rules at
k≤24. Memoriser 0.0 everywhere. **No baseline reaches competence in the
measured range**, so no speedup is claimed from the curves.

**Matched quality (where a baseline did reach ≥0.8).** In 6 of 72 episodes a
baseline matched CCM's quality; in 4 of those the constructive model was
faster: `rel-smallest` (MLP at k=3–4, ratio **~126–134×**), `math-is_prime`
(MLP, **15×**), `math-even` (MLP, **11×**). In the other 2 episodes,
`nearest_neighbor` matched the majority-class rule and was effectively free
(baseline ≈ lookup). Nothing here supports a 100,000× claim.

## Reproduce

```bash
cd "D:\ai model"
python -m pytest tests/test_constructive.py -q      # 19 tests
python -c "import constructive.bench as b; r=b.run_all()"
python -m pytest -q                                  # full repo suite, 145 passed / 1 skipped
```

Results are written to `constructive/results/results-<stamp>.json` and
`constructive/results/latest.json`.

## Files

| File | Role |
|---|---|
| `constructive/core.py` | values, terms, primitives, typing, evaluation (stdlib only) |
| `constructive/model.py` | search, concepts, retrieval prior, composition, guards, repair, quarantine |
| `constructive/worlds.py` | synthetic task generators with distractor rejection |
| `constructive/baselines.py` | memoriser, nearest-neighbour, small numpy MLP |
| `constructive/bench.py` | experiments A–G, learning curves, JSON output |
| `tests/test_constructive.py` | 19 tests, all passing |
