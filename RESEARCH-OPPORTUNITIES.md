# Research Opportunities — an evidence-graded portfolio

Created 2026-11-18 by the Phase 0 audit. Companion to `PHASE-0-REPOSITORY-AUDIT.md` (what exists) and
`ARCHITECTURE.md` (how it works). This document answers one question: **given the repository as it actually is, which
research directions are worth the next unit of effort, and which are not?**

Nothing here is a claim of capability. Every direction carries an evidence grade, a cost, and a falsifiable first test.

---

## 1. Constraints that shape the portfolio

Any direction that ignores these is fantasy. They are measured, not assumed (audit §2, §5).

| Constraint | Value | Consequence |
|---|---|---|
| CPU | Intel i5-1135G7, 4 cores / 8 threads, 8 MB L3 | No GPU path. Throughput must come from cache residency, int8/2-bit kernels, batching |
| SIMD | AVX2 present; AVX-512 **not** detected by NumPy runtime (`X86_V3` found, `X86_V4` not) | No VNNI dot-product instructions. Int8 matmul needs `vpmaddubsw`/`vpmaddwd` style kernels |
| RAM | 15.66 GB | Models up to ~1 B params in 4-bit are *storable*; they are not *trainable* here |
| Compiler | No C/C++ toolchain on PATH | C kernels (EXP-004) are blocked until a toolchain is installed |
| Training corpus | 19.9 MB train / 1.9 MB val, and **licence-contaminated** (see register) | No credible scaling curve. Corpus must be rebuilt before any quality claim is published |
| Existing core | 3.34 M params, 1.8944 bpb fp32 / 2.0554 bpb ternary | Worse than zlib (1.8296). It is a pipeline proof, not a language model |
| Existing agent | 126 tests pass, durable loop, tools, memory, verification | The strongest asset in the repo. Research should *use* it, not bypass it |
| Existing speed | 1,425 B/s single-stream, 5,027 B/s aggregate (batch 8), fresh spot-check | Aggregate is 3.5× single-stream. These are different metrics and must never be conflated |
| Time budget | One researcher, CPU-only | Prefer experiments that produce a signal in < 1 hour |

**The strategic reading.** OMEGA-NANO cannot win by out-scaling anyone. It can win by being *the system that acquires
competence fastest per unit of compute and per example* — because that is the one axis where a 4-core laptop is not
obviously outclassed. The portfolio is therefore weighted toward **acquisition mechanisms**, not architecture beauty.

---

## 2. How directions were graded

### 2.1 Evidence grades

| Grade | Meaning | Typical example |
|---|---|---|
| **A** | Independently replicated, peer-reviewed, public code and numbers | BitNet b1.58 energy/latency claims partly reach A; most do not |
| **B** | Peer-reviewed or top-lab technical report, public code, no independent replication | BitNet b1.58 2B4T, T-MAC, Mamba-2 |
| **C** | Preprint with public code, credible lab, partial replication | Titans, Coconut, LILO |
| **D** | Preprint or blog, no code or no replication | most "new architecture" papers |
| **E** | Analysis or concept only; no measured system | RC-001-style bandwidth arguments |

A grade is about *the claim*, not the venue. "Transformers are all you need" is not grade A because it is famous.

### 2.2 Direction scoring

Each family is scored on six axes, 1–5, higher is better:

- **Impact** — expected contribution to the constitution's definition of success.
- **Evidence** — how much is already established (grade mapped to 1–5).
- **CPU fit** — can it be honestly tested on 4 cores, no GPU.
- **Time-to-signal** — 5 = decisive result in a day, 1 = months.
- **Combinability** — does it strengthen the existing agent stack rather than replace it.
- **Cost** — 5 = cheap, 1 = expensive (inverted cost).

The total is a prioritisation aid, not a truth. It is recorded so later revisions can be audited.

---

## 3. The portfolio at a glance

| ID | Family | What it would buy | Impact | Evidence | CPU fit | Time | Comb. | Cost | Total | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F01 | Sparse conditional compute / MoE | More capacity per FLOP | 3 | 4 | 3 | 3 | 3 | 3 | 19 | **Second tier** |
| F02 | Linear recurrent / SSM hybrids | Cheaper long context, O(1) decode state | 3 | 4 | 4 | 2 | 3 | 3 | 19 | **Second tier** |
| F03 | Test-time memory / fast weights | Learning without retraining | 2 | 3 | 3 | 2 | 4 | 2 | 16 | Watch |
| F04 | Latent / recurrent-depth reasoning | More reasoning per token | 2 | 2 | 3 | 2 | 3 | 2 | 14 | Watch |
| F05 | Diffusion / parallel decoding | Latency via parallel tokens | 2 | 3 | 2 | 2 | 2 | 2 | 13 | Reject for now |
| F06 | Low-bit weights + LUT kernels | Memory and speed on CPU | 4 | 4 | 5 | 4 | 5 | 4 | **26** | **Top 3** |
| F07 | Retrieval-augmented generation | Knowledge without weights | 3 | 4 | 5 | 5 | 5 | 5 | **27** | **Top 3 (as substrate)** |
| F08 | Program synthesis + library learning | Compositional skill acquisition | 3 | 3 | 5 | 3 | 5 | 4 | **23** | **Top 1** |
| F09 | Neuro-symbolic verification | Correctness on checkable domains | 3 | 4 | 5 | 3 | 5 | 3 | **23** | **Top 2** |
| F10 | Continual learning / adapters | Adaptation without forgetting | 2 | 3 | 4 | 3 | 4 | 3 | 19 | Second tier |
| F11 | Explicit memory layers (PKM) | Parameter-efficient knowledge | 2 | 3 | 3 | 2 | 3 | 2 | 15 | Watch |
| F12 | World models / latent planning | Cheap lookahead | 2 | 2 | 2 | 1 | 3 | 1 | 11 | Reject for now |
| F13 | Distillation / synthetic data / self-play | Cheap capability transfer | 2 | 3 | 3 | 3 | 4 | 2 | 17 | Second tier |
| F14 | Verifier-guided search / process supervision | Reliability per FLOP | 2 | 4 | 5 | 4 | 5 | 4 | **24** | **Top 2 (paired)** |
| F15 | Spiking / neuromorphic / optical | Energy at the edge | 2 | 1 | 1 | 1 | 1 | 1 | 7 | Reject |

Top 3 as *research programmes* (not as isolated tricks): **F08 + F07 → Skill Fabric / BIRS**; **F09 + F14 → verified
symbolic workbench**; **F06 → cache-resident ternary core**. F07 and F14 are force multipliers inside the other two, which
is why they score high but are not separate programmes.

---

## 4. Per-family analysis

### 4.0 The Transformer question, answered without assuming the conclusion

The constitution forbids treating the Transformer as either the answer or the wrong answer. The evidence supports a
narrower statement:

- **At frontier scale**, the Transformer is the only architecture with multiple independent, reproducible wins across
  language, code, vision and audio. That is strong evidence that it is *sufficient*, not that it is *optimal*.
- **At 3.3 M parameters and 20 MB of training data**, architecture comparisons are dominated by optimisation noise, corpus
  artefacts and hyperparameter luck. A 10% bpb difference between two architectures at this scale is not evidence about
  architectures; it is evidence about seeds.
- **The alternatives are not speculative.** SSMs, linear attention, hybrid stacks and latent recurrence have public code
  and credible results at scale (F02, F04). They are real options, not contrarianism.
- **The current core's weakness is not architectural.** It is undertrained, trained on a contaminated 20 MB corpus, and
  decoded by an engine that re-encodes half the context every `ctx/2` tokens. Fixing those three things would move the
  number more than swapping the block type.

**Therefore:** keep the Transformer core for now, because it is the only one implemented and measured here, and because
the project cannot afford an architecture experiment that it cannot arbitrate. Spend the next research budget on
**acquisition (F08) and verification (F09)**, which are measurable at this scale. Revisit architecture at Stage R6, when a
clean corpus and a three-point scaling curve exist. This is a sequencing decision under uncertainty, not a verdict that
the Transformer is the final answer — and it is explicitly reversible.

Each block below follows the same order: mechanism and mathematics, compute profile, CPU viability, incremental viability inside
this repository, generalisation evidence, combinability, and what it would mean for OMEGA-NANO.

### F01 — Sparse conditional computation and mixture-of-experts

**Mechanism.** Replace one dense FFN with `E` experts and a router `g(x) = softmax(W_r x)`, activating the top-k:
`y = Σ_{i∈TopK(g)} g_i(x) · E_i(x)`. Capacity scales with `E`; FLOPs scale with `k`. Switch Transformer, GShard and
Mixtral established that this trains; DeepSeek-V3 established fine-grained experts plus shared experts at frontier scale.

**Compute.** Training is communication- and memory-bound, not FLOP-bound. Inference cost is `k/E` of dense plus routing
overhead, but *memory* is still `E` experts. On a laptop, the win is only real if the inactive experts stay on disk, which
destroys latency.

**CPU viability.** Poor at small scale. With 3.3 M parameters total, there is nothing to sparsify: routing overhead would
exceed the saved FLOPs.

**Incremental viability here.** Low. It would require a larger core first, and the larger core is blocked by corpus quality.

**Generalisation evidence.** Strong at scale (grade B/A for "it works at 100B+"), weak for "it works at 10M on 20MB".
Expert collapse and load-balancing instability are well documented; the fixes (auxiliary loss, capacity factors) are
hyperparameter-heavy.

**Combinability.** Moderate: experts could host skills, which rhymes with F08/F14. But that is a much later architecture.

**OMEGA angle.** Keep as a Phase 3+ option if the core ever exceeds ~100 M parameters. Not now. The honest statement is:
*conditional compute is the right idea at the wrong scale.*

### F02 — Linear recurrent and state-space hybrids

**Mechanism.** Replace quadratic attention with a recurrence whose state is a fixed-size matrix, e.g.
`h_t = A_t h_{t-1} + B_t x_t`, `y_t = C_t h_t`. Mamba-2's structured state-space duality makes this trainable in parallel
as a matrix multiplication. Hybrids interleave a few attention layers with many SSM layers (Jamba, Griffin, Samba).

**Compute.** Training is compute-efficient; decode is `O(1)` state per stream, which is exactly what a CPU wants — no KV
cache growth. Prefill is a scan, not a matmul with a growing cache.

**CPU viability.** Good in principle. But a 3.3 M-parameter SSM is not obviously better than a 3.3 M-parameter transformer
on 20 MB of data, and the implementation risk (numerical stability of the scan, chunked kernels) is high.

**Incremental viability here.** Medium. `core/engine.py` already has a KV cache; swapping in a recurrent state is a
contained change. The blocker is *evidence*: we would be testing an architecture claim with a corpus too small to separate
architecture from noise.

**Generalisation evidence.** Grade B that SSMs match transformers at scale on language modelling; grade C/D on
length generalisation beyond training distribution. The "Mamba beats transformers" narrative is not settled.

**Combinability.** High with the kernel work (F06): a state-based decoder removes the re-encode-every-`ctx/2` cost that
currently wastes ~half the compute.

**OMEGA angle.** A legitimate Phase 2 architecture experiment *after* the corpus is fixed and after EXP-005 has told us
whether acquisition, not architecture, is the bottleneck. Do not run it first; it would consume weeks and answer a
question we cannot yet measure cleanly.

### F03 — Test-time memory and fast weights

**Mechanism.** Update some parameters or memory slots during inference, driven by a surprise/associative loss:
`M_t = M_{t-1} - η ∇ℓ(M_{t-1}; x_t)`. Titans formalises a neural long-term memory module with a surprise metric and
shows gains up to 2 M tokens; TTT does the same with a self-supervised inner loop; kNN-LM does it non-parametrically.

**Compute.** Extra forward/backward per chunk at inference. On CPU this is 2–3× the cost of a forward pass for the memory
module alone.

**CPU viability.** Medium-low. The idea is attractive because it promises learning without retraining, but per-token
gradient updates on 4 cores would dominate latency.

**Incremental viability here.** Medium. The repository already has an external memory (`memory.py`) that is *cheaper and
more inspectable* than fast weights. The interesting version is hybrid: neural memory for similarity, symbolic memory for
provenance.

**Generalisation evidence.** Grade C. Impressive long-context results from one lab; independent replication is thin;
forgetting and stability of the update rule are open.

**Combinability.** High conceptually with the Skill Fabric: both are "learn at use time". Risk of double-counting the same
benefit.

**OMEGA angle.** The *principle* — adapt at inference — is core to OMEGA. The *implementation* should be the Skill Fabric
(discrete, verifiable, auditable) before it is fast weights (continuous, opaque, unauditable). See
`ORIGINAL-LEARNING-MECHANISM.md` §7.

### F04 — Latent and recurrent-depth reasoning

**Mechanism.** Spend more compute per token instead of more tokens: loop a block `r` times (`h ← f(h, e)`), or reason in
continuous space without emitting chain-of-thought tokens (Coconut), or use two coupled recurrent modules at different
timescales (HRM). The claim is that latent recurrence buys depth without vocabulary overhead.

**Compute.** `r×` the forward cost at inference; training is backprop-through-time, which is memory-hungry.

**CPU viability.** Medium. Small models can loop cheaply; the problem is that we cannot yet train a small model well enough
to see a reasoning gain.

**Incremental viability here.** Low. No reasoning benchmark exists in the repo yet; adding one is a prerequisite.

**Generalisation evidence.** Grade C/D. Results are striking but narrow (ARC-style puzzles, specific maze tasks), and
"latent reasoning" is hard to falsify because the intermediate state is unreadable.

**Combinability.** Low with the current stack: the agent's reasoning is currently symbolic/rule-based and auditable.

**OMEGA angle.** Interesting for Phase 6 (adaptive reasoning) *if* the project first builds a reasoning benchmark with a
measured quality floor. Without that, it is unfalsifiable.

### F05 — Diffusion and parallel decoding

**Mechanism.** Generate a whole block by iterative denoising rather than left-to-right: sample `x_T ~ N(0,I)`, apply
`x_{t-1} = f_θ(x_t, t)`, or mask-and-unmask for discrete diffusion (LLaDA). The appeal is `n` tokens in `k ≪ n` network
evaluations.

**Compute.** Training is comparable to autoregressive; inference trades sequential depth for parallel width.

**CPU viability.** Poor for a small model: each denoising step is a full forward pass, and the quality/compute frontier at
3 M parameters is unknown and likely bad.

**Incremental viability here.** Low. The entire `core/engine.py` decode loop assumes causal left-to-right with a KV cache.

**Generalisation evidence.** Grade C. LLaDA and Mercury show it can work at scale; whether it beats autoregressive at
equal compute is contested.

**Combinability.** Low.

**OMEGA angle.** Reject for now. Revisit only if the project has a measured latency problem that speculation cannot fix.
Note that OMEGA's latency problem is currently *kernel efficiency*, not sequential depth: 1,425 B/s is far below what the
hardware should do.

### F06 — Low-bit weights and LUT kernels

**Mechanism.** Constrain weights to `{-1, 0, +1}` (ternary) or 2-bit codes, and compute matvecs by table lookup rather than
multiply-accumulate. BitNet b1.58 shows ternary training with straight-through estimators is viable at 2B–3B scale;
bitnet.cpp reports 2.37–6.17× x86 speedups; T-MAC reports up to 6.6× with LUT-based mixed-precision kernels.

**Compute.** Arithmetic intensity drops; the problem becomes memory movement and instruction count. For a 3.34 M-parameter
model at 2 bits, the block weights are 3.2 MB and the full payload 3.8 MB — **it fits in the 8 MB L3**. That is the single
most important fact in this document for speed work.

**CPU viability.** Excellent, with caveats: AVX2 only here, no VNNI. `vpmaddubsw` + `vpmaddwd` can emulate int8 dot
products at roughly 2–4× scalar throughput; Numba already compiles the current loops.

**Incremental viability here.** Very high. EXP-001 already measured ternary kernels; EXP-003 already trained a ternary
core. The missing piece is a *blocked, cache-resident* kernel and removal of the window re-encode.

**Generalisation evidence.** Grade B for "ternary training works at scale"; grade C for "2-bit is lossless enough at 3 M
parameters on 20 MB" — EXP-003 says it costs 0.16 bpb here, which is a real regression, not noise.

**Combinability.** High: orthogonal to every other direction. It makes all of them cheaper.

**OMEGA angle.** The 10k tok/s ambition is a *systems* claim, and this is the only family that directly attacks it. Note
precisely what must be measured: single-stream decode, prefill, TTFT, and aggregate at batch `B` are four different
numbers. The recorded 2,865 aggregate at batch 14 (i7-7820HQ) and the fresh 5,027 aggregate at batch 8 (i5-1135G7) do not
support a 10k *single-stream* claim.

### F07 — Retrieval-augmented generation and hybrid retrieval

**Mechanism.** Condition generation on retrieved evidence: `p(y|x) ≈ Σ_z p(z|x) p(y|x,z)`. Variants differ in the index
(dense, sparse, late interaction), the granularity (token, chunk, document), and whether retrieval is trained jointly
(RETRO, REALM) or bolted on (RAG).

**Compute.** One embedding + ANN query per retrieval. Cheap relative to generation.

**CPU viability.** Excellent. The repository already has BM25 + int8 vector + trigram fusion with RRF, measured to 50k
records.

**Incremental viability here.** Maximum. It exists and is the most mature layer after the agent loop.

**Generalisation evidence.** Grade A that retrieval helps factual QA and reduces hallucination when the retriever is good;
grade B/C that it helps *reasoning* (it often hurts by injecting distractors).

**Combinability.** Maximum. It is the substrate for F08.

**OMEGA angle.** The unexploited insight is that retrieval should index **behaviour**, not text. A memory of "this program
transformed inputs like this" is a different object from a memory of "this paragraph said that". That gap is the Skill
Fabric thesis.

### F08 — Program synthesis and library learning

**Mechanism.** Given input/output examples, search a program space for a program consistent with them, then **refactor
repeated substructure into reusable abstractions**. DreamCoder does this with a learned recognition model and a
compression objective; Stitch makes library learning fast enough to be practical; LILO uses syntax-guided decompilation
to turn latent programs into readable code. FlashFill showed version-space algebra can synthesise string transformations
from very few examples.

**Compute.** Search is CPU-bound and embarrassingly parallel. Cost is dominated by the DSL and the search bias, not by
matrix multiplication. This is the rare family where a laptop is a legitimate research instrument.

**CPU viability.** Excellent. No GPU required at all.

**Incremental viability here.** High, but not trivial: the repo has no DSL, no synthesizer, and no benchmark of
transformation tasks.

**Generalisation evidence.** Grade B that library learning improves sample efficiency on curated domains; grade C that it
transfers to open-ended real-world tasks. The honest summary: *it works when the DSL is right, and the DSL is the hard
part.*

**Combinability.** Very high: it plugs into the agent as a new tool plus a memory kind, and it uses the existing
verification layer as its correctness oracle.

**OMEGA angle.** This is the most defensible originality bet available to a CPU-only project. The novel composition is
described in `ORIGINAL-LEARNING-MECHANISM.md`; the first test is `OMEGA-EXPERIMENT-001.md`. Crucially, the claim is
*not* "we invented program synthesis" — it is "behaviour-indexed retrieval plus residual synthesis beats both pure
retrieval and pure synthesis at k=3 examples, under a fixed MDL budget". That is falsifiable in a day.

### F09 — Neuro-symbolic verification

**Mechanism.** Pair a generator with a checker that cannot be fooled by fluent text: a proof kernel (Lean 4/mathlib), an
SMT solver (Z3), a CAS (SymPy), or a unit-test runner. Generation proposes; verification disposes. AlphaGeometry and
AlphaProof are the existence proofs: a neural model plus a symbolic verifier reached IMO silver-medal level in 2024, and
the 2025 Nature paper documents the system.

**Compute.** Search plus solver time. Solvers are CPU-native and often the bottleneck, which is fine here.

**CPU viability.** Excellent. Lean and Z3 run on a laptop; mathlib compilation is heavy but incremental.

**Incremental viability here.** Medium-high. `verify.py` already does world-checkable postconditions; the generalisation is
to *mathematical* postconditions. The repository also has `mathx` for exact arithmetic, which is a seed.

**Generalisation evidence.** Grade A that verified search produces correct proofs in formal domains; grade C that the
approach transfers to informal domains, because formalisation is the bottleneck.

**Combinability.** Very high: it turns the agent's "verification" invariant from a file-hash check into a semantic check.

**OMEGA angle.** This is the second programme. It gives OMEGA something no chatbot has: **a capability that is either
correct or rejected, with a proof object attached**. The strategic value is trust per FLOP.

### F10 — Continual learning and adapter composition

**Mechanism.** Learn a task without rewriting the base model: low-rank adapters (`W + BA`), prefix tuning, or
elastic weight consolidation (`L = L_task + λ Σ F_i (θ_i - θ*_i)²`). Composition is the hard part: merging adapters is
non-linear and lossy.

**Compute.** Cheap to train an adapter; expensive to evaluate forgetting.

**CPU viability.** Good for small models.

**Incremental viability here.** Medium: PyTorch is present, but the core is 3.3 M parameters and already weak.

**Generalisation evidence.** Grade B for "adapters work"; grade C for "composition is principled". Task arithmetic and
model merging are empirically fragile.

**Combinability.** High in principle, and it is the continuous analogue of the Skill Fabric. The project should choose one
as the primary mechanism and treat the other as a baseline, not run both.

**OMEGA angle.** The Skill Fabric is the auditable version of the same ambition: adapt without catastrophic forgetting,
but represent the adaptation as a program with a contract, not a weight delta.

### F11 — Explicit memory layers

**Mechanism.** Decouple knowledge capacity from FLOPs with a large key-value table addressed by product keys
(`score(k) = Σ_j ⟨k_j, k_{i,j}⟩`) or a sparse memory. Memory Layers and PKM showed gains at fixed compute.

**Compute.** Memory lookup is cheap; the table is large.

**CPU viability.** Medium: the table must be in RAM, and random access is cache-hostile.

**Incremental viability here.** Low. It is an architecture change that only pays off with many more parameters than 3.3 M.

**Generalisation evidence.** Grade B/C. Gains are real but modest, and modern LLMs largely absorbed them into MoE.

**Combinability.** Low now, moderate later.

**OMEGA angle.** Watch. The repository's *external* memory already provides capacity without parameters; the neural version
would only be interesting if the core grew.

### F12 — World models and latent planning

**Mechanism.** Learn `p(s_{t+1} | s_t, a_t)` and plan in latent space (Dreamer-style) or search over a learned model.
Appealing for agents that must act without trial and error in the real world.

**Compute.** High: environment interaction plus model learning.

**CPU viability.** Poor for anything beyond toy environments.

**Incremental viability here.** Very low. There is no environment, no simulator, and no action space beyond tools.

**Generalisation evidence.** Grade B in RL benchmarks; grade D for open-ended computer use.

**Combinability.** Low.

**OMEGA angle.** Reject for now. The agent's "world" is a filesystem and a shell; a world model is not the bottleneck,
verification is.

### F13 — Distillation, synthetic data and self-play

**Mechanism.** Compress a teacher into a student (`L = KL(p_T || p_S)`), or generate training data with a stronger system,
filter it by verification, and train on the survivors. AlphaProof-style systems depend on this loop.

**Compute.** Teacher inference dominates. On CPU, a local 7B teacher is slow but feasible in batch.

**CPU viability.** Medium. Distillation into a 3 M model is cheap; generating the corpus is not.

**Incremental viability here.** Medium-high: it is the natural way to get a better corpus without downloading one.

**Generalisation evidence.** Grade B that distillation works; grade C that self-generated data improves reasoning rather
than teaching the student to imitate the teacher's errors.

**Combinability.** High with F08/F09: verified synthesis produces high-precision data by construction.

**OMEGA angle.** This is the flywheel: **verify → filter → train → verify better**. It only becomes safe once the corpus
licence problem is fixed, because synthetic data inherits the licence of its seed corpus.

### F14 — Verifier-guided search and process supervision

**Mechanism.** Sample `n` candidates, rank or prune with a verifier or a process reward model, return the best:
`ŷ = argmax_{y ∈ Y} V(x, y)`. Cheap, orthogonal to architecture, and it converts compute into reliability.

**Compute.** `n×` generation plus verification. On CPU, `n` is small (4–16).

**CPU viability.** Excellent, *if* the verifier is cheap. Symbolic verifiers are; neural reward models are not.

**Incremental viability here.** High. `verify.py` is already the seed of a verifier; best-of-n can be added to the engine
without touching the core.

**Generalisation evidence.** Grade A that best-of-n with a good verifier improves pass@1 on code and math; grade C that
learned process reward models generalise beyond their training distribution.

**Combinability.** Maximum. It strengthens every other family.

**OMEGA angle.** This is the reliability multiplier. It is paired with F09 in the second programme because a verifier
without a generator is useless, and a generator without a verifier is untrustworthy.

### F15 — Spiking, neuromorphic and optical substrates

**Mechanism.** Event-driven computation (`Σ w_i · spike_i(t)`), or analog photonic matmuls. The promise is orders of
magnitude in energy.

**Compute.** Fundamentally different; not benchmarkable against dense matmul without a hardware target.

**CPU viability.** None: there is no neuromorphic hardware here, and simulators are slower than the dense baseline.

**Incremental viability here.** Zero.

**Generalisation evidence.** Grade D/E for language modelling at competitive quality. The field is real but pre-competitive.

**Combinability.** None.

**OMEGA angle.** Reject. It is a hardware bet, not a software research direction, and OMEGA has no hardware programme.

---

## 5. The three highest-value opportunities

The template has 15 fields so that two opportunities can be compared without hand-waving. Fields 8–13 are the ones that
make a proposal falsifiable.

### 5.1 Opportunity card — SF-1: Skill Fabric / Behaviour-Indexed Residual Synthesis (BIRS)

| # | Field | Content |
|---|---|---|
| 1 | **ID and name** | SF-1 — Skill Fabric / Behaviour-Indexed Residual Synthesis |
| 2 | **Thesis** | A system that retrieves *executable behaviour* by behavioural signature and synthesises only the residual can learn new string/data transformations from ~3 examples far more sample-efficiently than either retrieval or synthesis alone. |
| 3 | **Problem** | The agent can plan and verify, but it cannot *acquire a reusable skill* from a handful of demonstrations. Every new task starts from zero. The core model cannot supply this at 3.3 M parameters. |
| 4 | **Mechanism** | (a) Describe each candidate skill by a behavioural signature over a probe suite; (b) retrieve the `k` skills whose signatures are closest to the target's observed behaviour; (c) synthesise a residual program `r` such that `skill ∘ r` fits the examples; (d) admit `r` only if it compresses (MDL), is invariant under held-out probes, and does not perturb existing skills (non-interference); (e) store the admitted program with its signature and provenance. |
| 5 | **Mathematics** | Signature `σ(f) = (f(p_1), …, f(p_m))` on probes `P`. Retrieval `S* = argmin_{S} d(σ(S), σ(f*))`. Residual search `r* = argmin_{r ∈ DSL} L(σ(S* ∘ r), σ(f*)) + λ·len(r)`. Admission iff `MDL(S* ∘ r*) < MDL(S*) + MDL(r*) - δ` and `∀q ∈ Q_heldout: S*∘r*(q) = f*(q)`. |
| 6 | **Novelty** | Each part is prior art: DreamCoder/Stitch (library learning + compression), FlashFill (version-space string synthesis), LILO (decompilation), RAG (retrieval), case-based reasoning (behavioural reuse). The **specific composition** — behaviour-space retrieval key, residual-only synthesis, MDL/invariance/non-interference admission, and replay-based non-interference on an agent's live skill set — is not, to our knowledge, packaged and measured as one mechanism. Claim novelty for the *empirical prediction*, not the components. |
| 7 | **Evidence grade** | B for the components, **E for the composition** (no one has measured this exact loop). That is precisely why it is worth an experiment. |
| 8 | **Compute cost** | Search only; no training. Target < 5 min CPU for the full EXP-005 grid. |
| 9 | **CPU fit** | Excellent. Enumerative/version-space search is branch-and-bound on 4 cores. |
| 10 | **Integration point** | New `skills/` package beside `tools/`; a new memory kind `skill`; `verify.py` as the oracle; the engine learns to consult the fabric before planning from scratch. |
| 11 | **Falsifiable prediction (P1)** | With 3 demonstrations, BIRS reaches ≥ 2× the held-out exact-match rate of the best of {retrieval-only, synthesis-only, neural seq2seq} on programs whose nearest neighbour shares a *sub-behaviour* but not the whole program. |
| 12 | **Cheapest decisive test** | `OMEGA-EXPERIMENT-001.md` (EXP-005): a generated string-transformation DSL, 60 train / 60 held-out programs, split by program, 5 arms, 3 seeds. |
| 13 | **Success criteria** | Pre-registered in EXP-005 §7: exact-match ≥ 0.60 on held-out programs, ≥ 2× the strongest baseline, zero admitted non-equivalent skills on the probe suite, < 5 min wall clock. |
| 14 | **Kill criteria** | If residual synthesis adds < 10 percentage points over retrieval-only at equal budget, the thesis is false and the programme stops. If admission admits a single non-equivalent skill, the mechanism is unsafe as specified. |
| 15 | **Risks** | Overfitting the DSL to the benchmark (mitigate: hold out *operators*, not just programs); signature collisions (mitigate: report collision rate); skill poisoning (mitigate: non-interference replay + provenance); licence contamination of any learned artefact (mitigate: synthetic data only). |

### 5.2 Opportunity card — VW-1: Verified symbolic-mathematical workbench

| # | Field | Content |
|---|---|---|
| 1 | **ID and name** | VW-1 — Verified symbolic-mathematical workbench |
| 2 | **Thesis** | A compact generator coupled to Lean 4/Z3/SymPy can solve a measurable class of formal problems at a quality no unverified model of the same size can match, because the verifier absorbs the correctness burden. |
| 3 | **Problem** | OMEGA's verification layer checks file effects, not *semantic* correctness. That caps trust at "the file exists", not "the answer is right". |
| 4 | **Mechanism** | Tactic/lemma search over Lean, SMT queries over Z3, exact algebra over SymPy; a small model or enumerator proposes; the kernel checks; failures become training data; successful proofs are cached with dependencies. |
| 5 | **Mathematics** | Search over proof states `Γ ⊢ G`; each tactic `t` yields subgoals; success iff all leaves close. Cost model: `cost(proof) = Σ depth(t_i) + solver_time`. Quality floor: zero unverified answers. |
| 6 | **Novelty** | Low as a concept (AlphaProof, LeanDojo, COPRA exist). High as an *integration*: none of them ship as a local-first agent tool with provenance and a permission model. The research contribution would be the cost/quality frontier at small scale, not the paradigm. |
| 7 | **Evidence grade** | A for "neural + symbolic reaches IMO silver"; B for "small models can be useful tactic proposers with retrieval"; C for "this transfers to informal software tasks". |
| 8 | **Compute cost** | Lean/mathlib installation is heavy (GBs); Z3/SymPy are light. Search is CPU-bound. |
| 9 | **CPU fit** | Good for Z3/SymPy, medium for Lean (compilation and `simp` can be slow). |
| 10 | **Integration point** | New tools: `lean.check`, `smt.check`, `cas.simplify`; `verify.py` gains a `proof` postcondition kind. |
| 11 | **Falsifiable prediction (P2)** | On a fixed set of 100 Lean exercises at the level of `mathlib` basics, retrieval + small proposer + kernel check solves ≥ 30% within 60 s/problem, with 0 unsound proofs. |
| 12 | **Cheapest decisive test** | 30 `Nat`/`Int` arithmetic goals; Z3 and `simp` only; measure solve rate and time. No neural model needed for the first signal. |
| 13 | **Success criteria** | ≥ 90% of the 30 goals closed by automation; median < 2 s; every proof kernel-checked. |
| 14 | **Kill criteria** | If automation closes < 50% of trivially stated goals, the workbench is not yet useful and the effort belongs in the Skill Fabric instead. |
| 15 | **Risks** | Toolchain fragility on Windows; mathlib size; the temptation to claim "formal verification" for a tactic that only rewrites. Mitigate with explicit kernel-check evidence per proof. |

### 5.3 Opportunity card — KC-1: Cache-resident ternary core and honest 10k tok/s path

| # | Field | Content |
|---|---|---|
| 1 | **ID and name** | KC-1 — Cache-resident ternary core with blocked AVX2 kernels |
| 2 | **Thesis** | The 3.34 M-parameter ternary core fits in 8 MB L3 (3.2 MB block weights); with blocked int8/2-bit matvecs, no window re-encode, and a rolling KV cache, single-stream decode on this laptop can approach 10k B/s without a GPU. |
| 3 | **Problem** | Fresh measurement: 1,425 B/s single-stream, 5,027 B/s aggregate. The gap to the 10k ambition is a kernel and memory-layout problem, not a model problem. |
| 4 | **Mechanism** | (a) Store ternary blocks cache-line aligned; (b) decode 2-bit codes with `vpmaddubsw`/`vpmaddwd`; (c) keep the whole weight set resident; (d) replace the `ctx/2` re-encode with a true rolling cache; (e) batch only where it does not trade single-stream latency. |
| 5 | **Mathematics** | Decode is bandwidth/instruction bound: `tokens/s ≈ min(BW / bytes_per_token, IPC / instr_per_token)`. With 3.8 MB per token and ~100 GB/s L3, the bandwidth ceiling is ~26k B/s; the current 1,425 is ~5% of that ceiling, so instructions dominate. |
| 6 | **Novelty** | Low. bitnet.cpp and T-MAC already do this. The contribution is a *reproducible local measurement* and an honest decomposition of single-stream vs aggregate. |
| 7 | **Evidence grade** | B for the kernel techniques; **measured-but-narrow** for the current system. |
| 8 | **Compute cost** | Zero training; hours of engineering. |
| 9 | **CPU fit** | Maximum. |
| 10 | **Integration point** | `core/engine.py`, `omega_nano/kernels.py`, optionally a C extension (blocked: no compiler). |
| 11 | **Falsifiable prediction (P3)** | A blocked AVX2 ternary matvec plus rolling cache reaches ≥ 4,000 B/s single-stream on this laptop at the same output distribution (bit-identical greedy decode on a fixed prompt). |
| 12 | **Cheapest decisive test** | Microbenchmark the matvec in isolation (Numba, no C), then swap it into the decode loop and compare greedy outputs token-for-token. |
| 13 | **Success criteria** | ≥ 2.8× single-stream speedup, identical greedy tokens, p95 latency reported, no quality change (bpb unchanged). |
| 14 | **Kill criteria** | If the matvec is < 2× faster in isolation, the bottleneck is elsewhere (Python loop, sampling, re-encode) and the effort moves there. |
| 15 | **Risks** | Benchmarking on a thermally throttled laptop; conflating batch throughput with latency; claiming 10k when only aggregate crosses it. Mitigate with the protocol in `BASELINE-AND-BENCHMARK-PLAN.md`. |

**Ordering.** SF-1 first (it is the originality bet and the cheapest decisive test), VW-1 second (it is the trust
multiplier and mostly integration), KC-1 third (it is engineering with a clear ceiling, and it is the only one whose
success is a number rather than a capability).

---

## 6. Rejected directions, and why

Rejection here means "not now, for stated reasons", not "impossible". Each rejection is revisitable if the stated
condition changes.

| Direction | Why rejected now | Revisit if |
|---|---|---|
| Scale the existing 3.3 M core to 100 M+ | Corpus is 20 MB and licence-contaminated; a scaling curve on it would measure the corpus, not the architecture | Corpus rebuilt from an allowlist with ≥ 1 GB of clean text |
| Download and wrap a 7B model as "OMEGA" | It is a legitimate product decision but not a research contribution; it also hides every mechanism the project claims to study | The project chooses an integration-first strategy explicitly |
| Full Transformer replacement with a novel architecture | Architecture differences are unmeasurable at 20 MB; the repo has no clean benchmark to arbitrate | A benchmark with a quality floor exists |
| Train a reasoning model with RL | Requires a verifier-rich task distribution and far more compute than available | VW-1 exists and produces verified trajectories |
| Quantum / analog / optical acceleration | No hardware, no simulator budget, no falsifiable local test | Hardware access appears |
| "Just use the cloud" | Violates local-first and the resource strategy; also unfalsifiable as a local capability claim | A specific experiment justifies authorised cloud compute |
| Benchmark-chasing on public leaderboards | The constitution forbids claims without local reproduction; leaderboard deltas are not mechanisms | A local harness reproduces the leaderboard task |

---

## 7. What the field agrees on, and what is contested

This section exists to prevent the project from mistaking consensus for proof, or controversy for opportunity.

**Agreement (usable as premises):**
1. Compute-optimal training matters; most small models in 2026 are undertrained relative to their parameter count.
2. Data quality beats data quantity at small scale; deduplication and licence hygiene are part of quality.
3. Retrieval reduces factual error when the retriever is in-domain; it is not a substitute for reasoning.
4. Verification converts compute into reliability more cheaply than scaling generation.
5. Low-bit inference is a memory-bandwidth optimisation before it is an arithmetic one.
6. Evaluation leakage is the default failure mode of self-reported benchmarks.

**Contested (do not treat as premises):**
1. Whether SSMs/hybrids beat transformers at equal compute — evidence is scale- and task-dependent.
2. Whether latent reasoning beats explicit chain-of-thought — the fair comparison is hard to construct.
3. Whether test-time training generalises beyond the lab that proposed it.
4. Whether library learning transfers outside curated DSLs.
5. Whether "emergent" abilities are discontinuities or metric artefacts.
6. Whether 2-bit weights are viable below ~1B parameters — EXP-003's 0.16 bpb regression is a data point, not a verdict.

**The project's own uncertain claim**, stated so it can be attacked: *sample-efficient skill acquisition is a more
tractable axis of advantage for a CPU-only system than architectural novelty.* EXP-005 is designed to break this claim,
not to confirm it.

---

## 8. Portfolio allocation (next 90 days, one researcher)

| Programme | Share | Deliverable | Gate |
|---|---|---|---|
| SF-1 Skill Fabric | 50% | EXP-005 result + admitted skill set | P1 confirmed or killed |
| VW-1 Verified workbench | 25% | Z3/SymPy tools + 30-goal result | P2 first signal |
| KC-1 Kernel/cache | 15% | Blocked matvec + rolling cache | P3 ≥ 2.8× |
| Hygiene (corpus, packaging, CI) | 10% | Clean corpus + green CI | Required before any external claim |

Hygiene is not research, but it gates every claim. A result measured on a contaminated corpus cannot be published, and a
result that cannot be reproduced on a clean checkout is not a result.

---

## 9. Source log and verification status

Verified in this session (2026-11-18) by direct retrieval, not from memory:

| Source | Identifier | What was checked |
|---|---|---|
| BitNet b1.58 (Wang et al.) | arXiv 2402.17764 | Ternary weights `{-1,0,1}`, STE training, matched-perplexity claim |
| BitNet b1.58 2B4T | arXiv 2504.12285 | 2B-scale ternary model, public release, CPU latency claims |
| T-MAC | arXiv 2407.00088 | LUT-based mixed-precision CPU kernels, up to 6.6× claim, int1/int2 support |
| bitnet.cpp | github.com/microsoft/BitNet | MIT licence; reported 2.37–6.17× x86 speedups |
| Mamba-2 | arXiv 2405.21060 | Structured state-space duality; linear-time sequence mixing |
| Titans | arXiv 2501.00663 | Test-time neural memory, surprise metric, >2M context results |
| DreamCoder | arXiv 2006.08381 | Wake-sleep library learning with compression objective |
| Stitch | PLDI 2024 / arXiv | Fast Bayesian library learning; orders-of-magnitude speedup over DreamCoder |
| LILO | arXiv 2310.19791 | Syntax-guided decompilation of latent programs |
| AlphaProof / AlphaGeometry 2 | DeepMind IMO 2024 report; Nature 2025 | Silver-medal performance with a symbolic verifier; formal Lean proofs |
| Lean 4 / mathlib4 | github.com/leanprover/lean4, leanprover-community/mathlib4 | Apache-2.0 licences; kernel-checked proofs |
| llama.cpp | github.com/ggml-org/llama.cpp | MIT licence; local CPU inference reference |

Standard references used as background but **not** load-bearing for any number in this document (bibliographic details
should be re-confirmed before external publication): Switch Transformer, GShard, Mixtral, DeepSeek-V3, Jamba, Griffin,
Samba, Coconut, HRM, LLaDA, RETRO, kNN-LM, REALM, product-key memory, Memory Layers, Dreamer, FlashFill, EWC, LoRA,
AlphaCode, LeanDojo, COPRA.

**Rule for future documents:** a citation may support a *qualitative* statement at grade B/C, but no quantitative claim may
be attributed to a source that was not opened and checked in the session that makes the claim.
