# Original Learning Mechanism — Skill Fabric / Behaviour-Indexed Residual Synthesis (BIRS)

Status: **proposal**, not an implemented capability. Nothing in this document has been measured. The first test is
`OMEGA-EXPERIMENT-001.md` (internal ID EXP-005). The distinction between vision, hypothesis, proposal, prototype,
measured and validated is maintained throughout, as the constitution requires.

Created 2026-11-18.

---

## 1. Why a new learning mechanism is needed

The repository currently learns in exactly two ways, both thin:

1. **Plan caching** (`engine._learn`): a verified plan is stored under `sha256(normalize(goal))` and replayed verbatim.
   This is memorisation of a whole episode. It generalises to nothing.
2. **Episodic memory** (`memory.remember`): the completed task is stored as text and can be recalled by lexical or
   vector similarity. This is retrieval of a description. It cannot be executed.

Neither mechanism can do the thing that matters: **turn three demonstrations of a new transformation into a reusable,
verifiable procedure, and get better at the next one because of it.** The 3.34 M-parameter core cannot supply this either
— it is 0.16 bits/byte worse than zlib on held-out code and has no instruction-following ability.

So the question is not "which model should we train". It is: **what is the smallest learning mechanism that produces
compounding capability on a CPU, with verification, from a handful of examples?**

The proposal is a **Skill Fabric**: an agent-managed library of executable skills, indexed by *behaviour* rather than
text, extended by *residual* synthesis rather than from-scratch search, and admitted only through compression, invariance
and non-interference tests.

---

## 2. An operational taxonomy of learning

Six modes, ordered by abstraction. Each has a definition, a measurable signature, a cost profile, and a characteristic
failure. This taxonomy is the vocabulary the rest of the document uses; it is deliberately operational rather than
cognitive.

| Mode | Definition | Signature | Cost | Typical failure |
|---|---|---|---|---|
| **A. Memorisation** | Store `(x, y)` pairs and return `y` for an exact or near-exact `x` | Exact-match recall on seen inputs | O(1) write | Zero generalisation; leakage |
| **B. Retrieval** | Index items and return the nearest neighbour(s) for a query | Top-k recall / precision | O(log n) query | Semantic gap: similar surface, different function |
| **C. Procedure acquisition** | Induce an executable program `p` from demonstrations such that `p(x_i) = y_i` | Held-out exact match on *new inputs* | Search, super-linear in depth | Overfitting the demos; wrong inductive bias |
| **D. Concept formation** | Factor repeated substructure across procedures into a reusable operator with a contract | Reuse rate; description-length reduction | Amortised, periodic | Bloat; abstraction that never pays for itself |
| **E. Reasoning** | Compose acquired concepts under a verifier to solve a novel task | Solve rate at fixed budget | Search over compositions | Combinatorial explosion |
| **F. Discovery** | Generate new tasks or conjectures that are neither too easy nor impossible, and learn from them | Learning-curve slope per generated task | Expensive | Curriculum collapse; reward hacking |

The ladder is not a claim that cognition works this way. It is a claim about **what can be measured** with the
instrumentation this project actually has.

**Where OMEGA sits today:** A and B exist in `memory.py`; a degenerate C exists for *whole plans* (plan cache); D, E and
F are absent. The gap between B and C is the entire problem.

---

## 3. Formal definitions

### 3.1 Tasks, skills, contracts

A **task family** `T` is a distribution over pairs `(x, y)` with `x ∈ X`, `y ∈ Y`. A **demonstration set** is
`E = {(x_i, y_i)}_{i=1..k}` drawn i.i.d. from `T`.

A **skill** is a tuple

```
s = (code, τ, pre, post, cost, prov)
```

where `code` is a pure function `X ⇀ Y` in a restricted DSL, `τ` is its type signature, `pre`/`post` are predicates,
`cost` is a measured or estimated cost model, and `prov` is provenance (author, date, source task, hash).

A skill is **pure** if it performs no I/O, has no randomness, and is deterministic on its domain. Purity is not a stylistic
preference; it is what makes verification, caching and non-interference replay meaningful. Impure skills are allowed in the
system but are outside the fabric's admission guarantees (see §8).

### 3.2 Behavioural profile

Let `P = (p_1, …, p_m)` be a **public probe suite**: inputs chosen to exercise the DSL's domain, fixed for a library
version, and disjoint from any evaluation set.

The **signature** of a skill is

```
σ_P(s) = ( s(p_1), …, s(p_m) )        with ⊥ where s is undefined
```

Two skills are **behaviourally equivalent on P** iff `σ_P(s) = σ_P(t)`. This is an *approximation* of extensional
equality: it can prove difference (one witness suffices) but never equality.

For a *target* described only by `k` demonstrations we cannot evaluate `σ_P(f*)` — we have no oracle. So the target side
uses a **behavioural feature vector** computable from the demonstrations alone:

```
β(E) = ( Δlen, edit-ratio, is-subsequence, char-class transitions, repetition structure, … )
```

`β` is deliberately cheap and coarse. It is a **gate**, not a decision: it narrows the candidate set to `c` skills, which
are then checked exactly on the demo inputs (where the target's behaviour *is* known).

**Retrieval score** for skill `s` given demos `E`:

```
r(s, E) = α · d_β(β(E), β(Ē_s))  +  (1-α) · d_σ( s|_X_E , f*|_X_E )
```

where `X_E` are the demo inputs, `Ē_s` are the demos that originally produced `s`, and `d_σ` is normalised Hamming with a
penalty for domain mismatch. The second term is exact and cheap because `k` is small.

### 3.3 Residual synthesis

Given a retrieved skill `s` and demos `E`, find a residual program `r` in DSL `L` such that

```
∀(x_i, y_i) ∈ E :  s(r(x_i)) = y_i
```

The search is cost-bounded: `|r| ≤ d`, `nodes ≤ N`, `time ≤ T`. If `s = id`, the residual is the whole program, so
**BIRS strictly contains from-scratch synthesis as a special case**. This matters for the experiment: the arms are nested,
not independent.

The objective is not "fit the demos" (infinitely many programs do). It is

```
r* = argmin_{r ∈ L, r ⊨ E}   cost(r) + λ · complexity(r)
```

with `complexity` measured in DSL tokens. `λ` is fixed before the experiment, not tuned on the test set.

### 3.4 Admission

A candidate skill (or abstraction) is admitted only if all three tests pass.

**Test 1 — Compression (MDL).** Let `L(S)` be the total description length of the library. Admitting `a` yields `S'`.
Admit only if

```
L(S') + L(a)  <  L(S) - δ
```

with `δ` a fixed churn penalty (8 bits in EXP-005). This is the DreamCoder/Stitch compression criterion, adopted
deliberately: it is the only known objective that reliably prevents library bloat without a hand-tuned utility function.

For a *new skill* that no existing skill can express, the comparison is against storing the demonstrations plus a fresh
program; for an *abstraction*, it is against the library before refactoring. The two cases have different baselines and
must be reported separately.

**Test 2 — Invariance.** From the demonstrations, infer a set of input perturbations `I` that the task appears to ignore
(e.g., a character that never affects the output, a whitespace convention). Require

```
∀ x ∈ dom(s∘r), ∀ π ∈ I :  s(r(π(x))) = s(r(x))
```

This is a **falsification test**, not a proof of correctness. It catches the common overfit where the synthesised program
keys on an irrelevant literal. It cannot catch a program that is wrong on inputs nobody thought to probe, and the document
must not pretend otherwise.

Two guards against a vacuous pass:
- **Non-constancy**: unless all demonstrated outputs are equal, `s∘r` must not be constant on `P`.
- **Totality**: `s∘r` must be defined and deterministic on all of `P`, not raise, and not consume more than the fuel
  budget.

**Test 3 — Non-interference.** Adding a *new* skill cannot change existing behaviour, because skills are pure and
separately namespaced — the test is trivially satisfied and is reported as such. The test is substantive for
**refactoring**: when an abstraction rewrites existing skills to call it, every previously admitted skill must be replayed
against its own regression probes and produce byte-identical outputs.

```
∀ s_j ∈ S, ∀ q ∈ R_j :  s_j'(q) = s_j(q)
```

where `R_j` is the regression suite captured at `s_j`'s admission. A single mismatch rejects the whole refactor. This is
the mechanism's safety property, and it is the one piece that is genuinely specific to an *agent* library rather than a
batch program-synthesis system: the library is live, and a bad abstraction could silently corrupt previously working
capability.

### 3.5 The fabric

```
F = ( S, Index_β, Index_σ, R, V )
```

`S` is the skill set; `Index_β` and `Index_σ` are the two retrieval indexes; `R` maps each skill to its regression suite;
`V` is the verifier interface. The fabric is versioned and append-only: admitted skills are never silently mutated; a
refactor creates a new library version with a replay proof.

---

## 4. The algorithm

```
ACQUIRE(task, demos E, fabric F, budget B):

  # 1. Cheap gate
  cands ← top_c( F.Index_β, β(E), c=8 )
  cands ← rank( cands, by r(s,E) )                  # exact check on demo inputs
  if |cands| = 0: cands ← [ identity ]

  # 2. Residual synthesis over candidates, interleaved by cost
  best ← ⊥
  for s in cands (best-first by r):
      r ← SEARCH( L, E, objective = cost + λ·complexity,
                  constraint = λx. s(r(x)) = y,
                  budget = B / |cands| )
      if r ≠ ⊥ and better(best, (s,r)): best ← (s,r)
      if cost(best) ≤ τ_stop: break

  # 3. If nothing composes, fall back to whole-program synthesis
  if best = ⊥: best ← (identity, SEARCH(L, E, budget = B_remaining))

  # 4. Admission
  if not CONSISTENT(best, E):            return REJECT("example mismatch")
  if not INVARIANT(best, E):             return REJECT("invariance")
  if not TOTAL_AND_DETERMINISTIC(best):  return REJECT("partial or nondeterministic")

  skill ← materialise(best, prov = hash(E) + fabric.version)
  F' ← F + skill
  if L(F') ≥ L(F) - δ:                    return REJECT("does not compress")

  # 5. Opportunistic abstraction (mode D)
  if |F'.S| mod N_refactor = 0:
      a ← MINE_COMMON_SUBSTRUCTURE(F'.S)          # anti-unification over ASTs
      if a ≠ ⊥ and REPLAY_EQUIVALENT(F'.S, refactor(F'.S, a)):
          F' ← refactor(F'.S, a) + a              # non-interference proved
      else:
          discard(a)

  return ACCEPT(skill, F')
```

Three properties of this loop matter more than its details:

1. **The fallback is the baseline.** If retrieval is useless, `cands = [identity]` and the algorithm is ordinary
   synthesis. So BIRS cannot be *worse* than synthesis by construction — only slower. The experiment must therefore
   measure **search cost at equal quality**, not just quality.
2. **Abstraction is opportunistic, not per-task.** Refactoring on every admission would make cost unpredictable and
   replay expensive; batching it makes the non-interference proof tractable.
3. **Nothing is admitted without a regression suite.** Even the first skill gets one, captured from the public probes.

---

## 5. What is actually novel here

Intellectual honesty requires separating components from composition.

| Component | Prior art | Status |
|---|---|---|
| Compression-driven library learning | DreamCoder (2020), Stitch (2024) | Established, grade B |
| Anti-unification / abstraction refactoring | DreamCoder, LILO | Established |
| Example-driven string transformation | FlashFill (2011), PROSE | Established, industrial |
| Behavioural equivalence checking | Property-based testing, QuickCheck | Established |
| Retrieval of code by description | RAG, code search | Established |
| Skill libraries for agents | Voyager (2023) and successors | Established, grade C |
| **Behaviour-space retrieval key** (`σ` + `β` instead of text embedding) | — | **Proposed** |
| **Residual-only synthesis over retrieved skills** | — | **Proposed as an integrated loop** |
| **MDL + invariance + non-interference admission for a live agent library** | parts in DreamCoder; replay is new here | **Proposed** |

**The honest novelty claim:**

> Existing library-learning systems learn abstractions from a *batch* of tasks and evaluate on held-out tasks. Existing
> agent skill libraries store code with *textual* descriptions and retrieve by language similarity. Neither indexes by
> measured behaviour, neither synthesises only the residual against a retrieved skill, and neither admits a refactor only
> after replay-proving non-interference on the live library.

That is a *composition* claim, and it is worth testing because the composition makes a different empirical prediction
(§6), not because the parts are new.

**What would make it not novel at all:** if a 2024–2026 system already does behaviour-indexed retrieval plus residual
synthesis plus replay-checked abstraction. The literature review in `RESEARCH-OPPORTUNITIES.md` §9 did not find one, but
absence of evidence is not evidence of absence. A systematic search is a prerequisite before any external publication.

---

## 6. Falsifiable predictions

Each prediction names the measurement that would refute it. If a prediction fails, the mechanism is revised or abandoned;
it is not re-narrated.

**P1 — Sample efficiency.** At `k = 3` demonstrations, on programs whose nearest neighbour shares a sub-behaviour but not
the whole program, BIRS achieves **≥ 2× the held-out exact-match rate** of the best of {retrieval-only, synthesis-only,
small neural seq2seq}. Refuted if the ratio is < 1.25×.

*Why this is the load-bearing prediction:* it is the entire reason to build the mechanism. If residual synthesis does not
beat both of its parents, the composition adds complexity for nothing.

**P2 — Search efficiency.** At equal solve rate, BIRS expands **≥ 10× fewer search nodes** than from-scratch synthesis,
because the retrieved skill supplies the shared prefix of the program. Refuted if node reduction is < 2×.

*This can fail independently of P1.* A mechanism can be more sample-efficient and slower; that is still useful, but it
changes where it belongs in the stack.

**P3 — Compounding.** Abstractions admitted by the compression test are **reused by ≥ 30% of later tasks** in the same
family. Refuted if reuse is < 10%, which would mean the DSL's natural abstractions are not the ones the curriculum needs.

*This is the only prediction that tests mode D rather than mode C.* It requires a multi-stage experiment (EXP-005 is
single-stage); P3 is therefore scheduled for EXP-006.

**P4 — Safety.** Non-interference replay detects **100%** of injected semantic regressions (mutants that change behaviour
on at least one regression probe) with **0 false rejections** on equivalent refactors. Refuted by a single escape or by
false-rejection rate > 5%.

*Method:* mutate an admitted skill in `n = 20` behaviour-changing ways and in `n = 20` semantics-preserving ways, run the
replay, and report the confusion matrix. This is a standard mutation-testing design and needs no new theory.

**A prediction that is deliberately absent.** There is no prediction about "general intelligence" or transfer to arbitrary
domains. The mechanism is scoped to example-driven transformations with a checkable DSL. Overclaiming here would violate
the constitution's no-false-breakthrough rule, and it would make the experiment unfalsifiable.

---

## 7. Why not just use fast weights / test-time training?

Because the two mechanisms make different trade-offs, and the project should pick the one it can verify.

| Property | Fast weights (F03) | Skill Fabric (this document) |
|---|---|---|
| Representation | Continuous parameter delta | Discrete program + contract |
| Verifiable | No (behaviour is emergent) | Yes (execute and compare) |
| Auditable | No | Yes (code, provenance, hash) |
| Forgetting | Catastrophic, hard to bound | None for existing skills; replay-checked |
| Generalisation | Interpolative | Compositional |
| Cost | Gradient steps per adaptation | Search per adaptation |
| Failure mode | Silent degradation | Explicit reject |

The fabric is the *auditable* form of adaptation. A continuous memory may later be added *underneath* it (as a retrieval
index over skill embeddings), but it should not be the primary mechanism until the discrete one is measured. This is a
sequencing decision, not a claim that fast weights are wrong.

---

## 8. Failure modes and safety

A mechanism that synthesises and executes code is a security boundary. The fabric is only acceptable if it is built with
the same discipline as `tools/` and `security.py`.

| Failure | Detection | Mitigation |
|---|---|---|
| DSL cannot express the target | Residual search exhausts budget; falls back to synthesis; records `expressibility_fail` | Report the rate. A high rate means the DSL, not the retrieval, is the bottleneck |
| Signature collision (different function, same `σ`) | Residual cost is large despite a close match | Treat cost blow-up as a retrieval miss; widen `P`; report collision rate |
| Overfit to demos | Held-out exact match collapses | Invariance probes; report the demo→held-out gap per program, not just the mean |
| Abstraction bloat | `L(S)` grows without reuse | MDL `δ`; evict unused skills after `N` tasks; P3 measures it |
| Skill poisoning | A malicious or wrong skill enters the library | Provenance + admission + regression replay + human approval for anything impure |
| Unsafe execution | Synthesised code imports `os`, loops forever, or reads files | AST whitelist, no imports/attributes outside an allowlist, fuel counter, wall-clock cap, subprocess-free interpreter |
| Benchmark overfitting | Suspiciously high exact match | Hold out **operators**, not just programs; publish the DSL grammar and the generator seed |
| Silent refactor corruption | Existing skill changes answer | Non-interference replay rejects the refactor atomically (§3.4) |

**Non-negotiable design rules:**

1. **Skills are pure.** No I/O, no clock, no randomness, no environment access. Anything impure is a `tool`, with the
   existing permission model, not a `skill`.
2. **The interpreter is not Python's `eval`.** A whitelisted AST walker with a fuel budget. `eval` on synthesised code is
   an arbitrary-code-execution primitive and would be a worse vulnerability than the `shell.run` finding in audit §8.4.
3. **Admission is a state transition, not a side effect.** It writes a new library version with the replay proof; it never
   mutates in place.
4. **Every skill is hash-addressed** and its provenance chain is recorded in the same append-only store as tasks.
5. **Human approval gates any skill whose contract touches the world.** Pure string/number skills can be auto-admitted;
   everything else cannot.

These rules are constraints on the *implementation*, and EXP-005 is designed to respect them from the first line of code.

---

## 9. Where it plugs into OMEGA-NANO

| Constitution layer | Contribution |
|---|---|
| 4 Memory fabric | New record kind `skill` with a behavioural profile and a regression suite; the existing SQLite/WAL store and provenance model are reused |
| 6 Retrieval | Two new indexes (`β`, `σ`) beside BM25/vector/trigram; retrieval becomes function-aware, not just text-aware |
| 7 Reasoning | Planning consults the fabric before searching from scratch; a plan step can call an admitted skill |
| 12 Verification | The invariance and non-interference tests are new postcondition kinds; `verify.py` gains `behaviour_equivalent` |
| 13 Orchestration | Admission becomes a durable, checkpointed task step, not an in-memory mutation |
| 14 Security | Purity + AST whitelist + fuel + provenance are enforced at admission and at call time |
| 16 Evaluation | EXP-005 defines the first acquisition benchmark |

**What it does not replace.** It does not replace the core model, the tool layer, or the reasoner. It is a fourth kind of
memory — *executable* memory — and a policy for when to trust it.

---

## 10. Open questions

Recorded so they are not forgotten, and so a later document can close them with evidence rather than opinion.

1. **Signature sufficiency.** How many probes `m` are needed before `σ` distinguishes skills that matter? This is
   empirical and task-dependent; EXP-005 should report the collision rate as `m` varies over {8, 16, 32}.
2. **Residual ordering.** Is it better to retrieve one skill and synthesise a deep residual, or retrieve several and
   synthesise shallow ones? The current algorithm does the latter implicitly; it is not obviously right.
3. **Abstraction timing.** Per-task refactoring gives faster compounding but makes cost unpredictable. The `N_refactor`
   parameter is a guess.
4. **Cross-family transfer.** A skill learned for CSV cleanup may help with log parsing. The behavioural index should
   capture this; nothing in the current design proves it.
5. **Interaction with the reasoner.** If the rule reasoner can already plan the task, the fabric is unnecessary. The fabric
   matters exactly where the reasoner fails — which is also where evaluation is hardest.
6. **Licence of learned artefacts.** A skill synthesised from demonstrations is a derived work of the demonstrations. If
   the corpus is contaminated, so is the fabric. This is one more reason the corpus must be rebuilt first.

---

## 11. Summary in one paragraph

OMEGA-NANO can already plan, act, verify and remember. What it cannot do is **acquire**. The Skill Fabric turns
demonstrations into pure, typed, hash-addressed programs; indexes them by measured behaviour rather than by description;
retrieves the closest one and synthesises only the residual; and admits new skills — and, more importantly, new
*abstractions* — only when they compress the library, survive invariance probes, and replay-prove that they did not break
anything already working. The claim is not that this is a new paradigm. The claim is that this specific composition
produces a measurable sample-efficiency advantage on a CPU, and that the experiment to test it costs less than a day.
