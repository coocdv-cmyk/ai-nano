# Research Roadmap — staged, gated, killable

Created 2026-11-18. This is the **research** roadmap: what to investigate, in what order, with which gates, and what would
make us stop. The **phase** roadmap (constitution §11, `ROADMAP.md`) tracks product maturity; the two are related but not
the same. A phase can be "passed" while the research programme that matters is still unproven, and that is the current
situation.

Governing rule: **no stage begins until the previous gate is met or explicitly waived in writing.** The most common
failure mode of ambitious projects is not a wrong hypothesis; it is running five hypotheses at once so that no result is
attributable.

---

## 1. Principles

1. **One variable per experiment.** If two things change, the result explains neither.
2. **A gate is a measurement, not an opinion.** "Looks promising" is not a gate.
3. **Every stage has a kill criterion, written before it starts.** A stage without one is a hobby.
4. **Cheapest decisive test first.** If a 5-minute experiment can kill the idea, run it before the 5-week one.
5. **Hygiene gates claims, not curiosity.** You may explore on a dirty corpus; you may not *publish* a capability claim
   from one.
6. **Integration follows evidence.** A mechanism enters `omega_nano/` only after it has won outside it.
7. **Negative results are deliverables.** They are written up with the same care and they redirect the programme.

---

## 2. The tracks

| Track | Question | Why it exists | Primary risk |
|---|---|---|---|
| **A — Skill Fabric (SF)** | Can the agent acquire reusable skills from ~3 examples? | The originality bet; the only axis where CPU-only is not a handicap | The DSL, not the mechanism, may be doing the work |
| **B — Verified workbench (VW)** | Can the agent be *provably* right on checkable domains? | Trust per FLOP; differentiates from fluent chatbots | Formalisation bottleneck; toolchain fragility |
| **C — Kernel/core (KC)** | How fast can this laptop actually decode, honestly? | The 10k ambition is a systems claim; it is measurable | Optimising a model too weak to matter |
| **D — Hygiene (HY)** | Can any claim be reproduced and legally published? | Every other track's results depend on it | Boring, therefore skipped |
| **E — Agent capability (AC)** | Can the agent do real repository work? | The constitution's top strategic priority (Phase 4) | Blocked by sandbox and by core weakness |

Tracks A, B and C are research. D is infrastructure. E is product. The roadmap sequences A and C first because they are
cheap and falsifiable; B second because it is mostly integration; E after the sandbox exists.

---

## 3. Stages

### Stage R0 — Truth and hygiene *(partially complete)*

**Question:** what is actually here, and can anything be claimed from it?

| Item | Status |
|---|---|
| Factual repository audit | **done** (`PHASE-0-REPOSITORY-AUDIT.md`) |
| Architecture map corrected | **done** (`ARCHITECTURE.md`) |
| Licence/provenance register | **done** (`LICENSE-AND-PROVENANCE-REGISTER.md`) |
| Research portfolio and mechanism design | **done** (this package) |
| First git commit | **not done** — 0 commits, all files untracked |
| Corpus rebuilt from an allowlist | **not done** — blocked on owner approval (changes production code) |
| CI (tests + SPDX + manifest) | **not done** |
| Packaging fixed (`omega_nano.core`, `torch` extra) | **not done** |

**Gate R0:** a clean checkout plus `python -m pytest -q` is green, a manifest covers every byte of the corpus, and the
corpus scan returns zero copyleft markers.

**Kill criterion:** none. This stage is not optional. If it cannot be completed, nothing downstream may be published.

**Cost:** 2–4 days, mostly waiting on the owner's licence and corpus decisions.

---

### Stage R1 — Measurement substrate

**Question:** can a number be trusted when it is produced?

Deliverables:

1. `BENCHMARKS/bench_core_speed.py` implementing §2 of `BASELINE-AND-BENCHMARK-PLAN.md` (four metrics, p50/p95, warm-up,
   greedy-identity check).
2. A 300-query labelled retrieval set over a clean corpus (B3).
3. A results schema that enforces the hardware block and the quality floor.
4. A `make bench` (or `python -m tools.bench`) entry point that runs B1, B2, B4, B5 and writes one signed result file.

**Gate R1:** re-running the harness on the current machine reproduces the fresh spot-check within 15% on single-stream
decode and produces identical greedy tokens across two runs.

**Kill criterion:** if the harness cannot reproduce its own number twice, the measurement problem is deeper than the
research problem, and Track C stops until it is fixed.

**Cost:** 3–5 days.

---

### Stage R2 — EXP-005: single-stage skill acquisition *(the decisive experiment)*

**Question:** does BIRS beat its ablations at `k = 3`?

Protocol: `OMEGA-EXPERIMENT-001.md`, pre-registered.

**Gate R2:** verdict R1 or R2 (P1 confirmed or partial).
**Kill criterion:** verdict R3 (BIRS ≤ SYN + 0.05). If killed, Track A stops at single-stage and the effort moves to
Track B.

**Cost:** 2 days implementation + 5 minutes compute.

**Why first:** it is the cheapest test of the most load-bearing claim in the package. If it fails, the project saves months.

---

### Stage R3 — EXP-006: compounding and safety

**Question:** do admitted abstractions get reused (P3), and does non-interference replay catch regressions (P4)?

Protocol (designed only if R2 passes):

- Three-stage curriculum: 20 programs per stage; the library persists across stages.
- Measure reuse rate, description-length trajectory, and the mutation-testing confusion matrix for replay.
- Ablate the compression test (admit everything) to see whether MDL actually prevents bloat.

**Gate R3:** reuse ≥ 30% (P3) **and** replay catches 100% of semantic mutants with ≤ 5% false rejection (P4).
**Kill criterion:** reuse < 10%, or any mutant escapes. An escaping mutant is a safety failure, not a quality failure.

**Cost:** 3–4 days.

---

### Stage R4 — Real-task replication

**Question:** does the mechanism survive contact with data that was not generated by its own DSL?

Tasks: CSV normalisation, log-line parsing, filename cleanup, small refactors with test oracles — each with 3
demonstrations and a held-out set built from real files in the workspace.

**Gate R4:** BIRS ≥ 1.5× the best baseline on ≥ 2 of 4 task families, with admission precision 1.0.
**Kill criterion:** no family shows a gain. The synthetic result was a DSL artefact.

**Cost:** 5–8 days. This is where most research programmes die, and it is scheduled deliberately before integration.

---

### Stage R5 — Integration into the agent

Only after R4:

- `skills/` package with the pure-function contract and the AST-whitelisted interpreter.
- New memory kind `skill`, versioned, with provenance and regression suites.
- Admission as a durable, checkpointed task step (not an in-memory mutation).
- A `skill.call` tool with the existing permission model; pure skills auto-allowed, impure ones are tools.
- Report integration: the task report shows which skills were used and which were admitted.

**Gate R5:** 20 consecutive real tasks in which no admitted skill breaks a previously working task (non-interference in
production), and the audit log records every admission with a hash.
**Kill criterion:** any silent behaviour change in an existing skill. The fabric is rolled back and the replay design is
revised.

**Cost:** 8–12 days.

---

### Stage V0–V2 — Verified workbench (parallel, small)

| Sub-stage | Deliverable | Gate | Cost |
|---|---|---|---|
| **V0** | Z3 + SymPy tools; 30 arithmetic goals; solve rate and time | ≥ 90% closed, 0 unsound, median < 2 s | 2 days |
| **V1** | Lean 4 + a 100-exercise set; retrieval + `simp`/`omega`; kernel-checked proofs only | ≥ 30% solved in 60 s, 0 unsound | 5–8 days (toolchain risk) |
| **V2** | `verify.py` gains a `proof` postcondition; the report carries the proof object | End-to-end task with a verified proof | 3 days |

**Kill criterion:** V0 closes < 50% of trivially stated goals — then the bottleneck is tooling, not research, and the
track pauses.

**Sequencing:** V0 runs in parallel with R2 because it is independent and cheap. V1 waits for the Lean toolchain decision.

---

### Stage K0–K3 — Kernel and core speed (parallel, engineering)

| Sub-stage | Deliverable | Gate | Cost |
|---|---|---|---|
| **K0** | Isolated blocked int8/2-bit matvec in Numba; microbenchmark vs current | ≥ 2× in isolation, or stop and profile elsewhere | 2 days |
| **K1** | Rolling KV cache replacing the `ctx/2` re-encode | Greedy tokens identical; end-to-end decode ≥ 2× | 3–4 days |
| **K2** | Integrated engine; full four-metric report | ≥ 4,000 B/s single-stream, bpb unchanged | 3 days |
| **K3** | Honest 10k claim or an honest ceiling | 10k single-stream **or** a written bandwidth/instruction analysis of why not | 2 days |

**Kill criterion:** K0 < 2× in isolation. Then the bottleneck is not the matvec, and the effort moves to the decode loop,
sampling, or Python overhead — measured, not guessed.

**Note on EXP-004.** The C/AVX2 kernel experiment remains blocked by the absent compiler. K0 is the Numba-only path; it is
strictly weaker but needs no toolchain. If a compiler appears, EXP-004 subsumes K0 and should be run instead.

---

### Stage S0 — Sandbox *(prerequisite for Track E, not for A–C)*

`shell.run` is an allowlist plus deny patterns, not an OS sandbox; allowlisted interpreters inherit the full environment
(audit §8.3–8.4). Before the agent runs autonomous repository tasks, it needs a real boundary: Windows Job Object with a
restricted token and a scrubbed environment, or a container.

**Gate S0:** a deliberately malicious command cannot read an environment secret, cannot write outside the workspace, and
cannot outlive its timeout.
**Cost:** 5–10 days, platform-specific.

---

### Stage E1 — First real software-engineering capability

Only after S0 and R5:

- Repository indexing tool (file map, symbols, dependency graph).
- A benchmark of 20 real repository tasks with verified tests as postconditions.
- Success = tests pass, not "the model said so".

**Gate E1:** ≥ 50% task success on the 20-task set, zero unverified claims of success.
**Cost:** 10–15 days.

---

## 4. Sequencing

```
R0 ──► R1 ──► R2 ──► R3 ──► R4 ──► R5 ──┐
                 │                       ├──► E1 (needs S0)
                 └── (if killed) ──► V0 ─┘
V0 ──────────────────────────────────────► V1 ──► V2
K0 ──► K1 ──► K2 ──► K3
S0 ──────────────────────────────────────────────► E1
```

Parallelism is deliberate but bounded: **at most three active tracks**, and never two experiments that share a variable.

---

## 5. Calendar view

| Window | Focus | Exit condition |
|---|---|---|
| **Week 1** | R0 completion: first commit, corpus decision, CI skeleton | Clean checkout, green CI |
| **Weeks 2–3** | R1 harness + R2 implementation | Harness reproduces itself; EXP-005 runs |
| **Week 4** | R2 result and write-up | Verdict R1–R6 recorded |
| **Weeks 5–6** | R3 (if R2 passes) **or** V0/V1 (if not); K0–K1 in parallel | P3/P4 measured, or Z3/Lean signal |
| **Weeks 7–9** | R4 real-task replication; K2–K3 | Gains survive real data; speed claim settled |
| **Weeks 10–12** | R5 integration or Track B integration; S0 sandbox | Agent uses skills or proofs in a real task |
| **Months 4–6** | E1 repository tasks; corpus expansion to ≥ 200 MB clean; core scaling to 20–50 M params | Task success measured; scaling curve has ≥ 3 points |
| **Months 7–12** | Architecture experiment (SSM/hybrid) *only if* the scaling curve shows the core is the bottleneck; multimodal only if a benchmark exists | One architecture question answered with a measured quality floor |

**What does not appear in the calendar:** training a frontier model, downloading a large model and calling it OMEGA,
multi-agent orchestration, computer-use, multimodal. They are not rejected forever; they are downstream of gates that have
not been met.

---

## 6. Resource allocation

| Track | Share of research time | Rationale |
|---|---|---|
| A — Skill Fabric | 45% | Highest originality per CPU-hour |
| C — Kernel/core | 20% | Orthogonal, measurable, unblocks every latency claim |
| B — Verified workbench | 20% | Trust differentiator; mostly integration |
| D — Hygiene | 10% | Gates publication |
| E — Agent capability | 5% until S0 | Premature without a sandbox |

---

## 7. Evidence triggers that reorder the roadmap

The roadmap is not a promise; it is a conditional plan. These are the conditions under which it changes.

| Trigger | Reordering |
|---|---|
| EXP-005 killed (R3) | Track A drops to 10%; Track B rises to 50%; the fabric becomes a plain code cache |
| EXP-005 strong (R1) with node reduction ≥ 10× | Track A rises to 60%; R4 pulled forward |
| K0 shows < 2× | Track C pauses; the bottleneck analysis replaces it; the 10k target is restated as a ceiling |
| K1 reaches ≥ 8k single-stream | The speed narrative changes from "ambition" to "measured"; publish the decomposition |
| A compiler becomes available | EXP-004 replaces K0; re-baseline everything |
| Corpus rebuild blocked for > 30 days | No external publication; internal work continues; the register records the delay |
| A pretrained local model becomes acceptable to the project | E1 unblocks without R5; Track A becomes a complement rather than a substitute |

---

## 8. Mapping to the constitution's phases

| Constitution phase | Research stages that actually satisfy it |
|---|---|
| 0 Mission and audit | R0 |
| 1 Local agent slice | already passed; B1/B2 maintain it |
| 2 Core model and inference baselines | R1 (measurement) + K0–K3; a *credible* core needs the clean corpus and a scaling curve, neither of which exists |
| 3 Memory and knowledge fabric | R1 labelled set; R5 adds executable memory |
| 4 Software engineering | S0 + E1 |
| 5 Browser and desktop | not scheduled; needs S0 first |
| 6 Adaptive reasoning | V0–V2 and R3 (skill selection is a form of adaptive compute) |
| 7 Multimodal | not scheduled; no benchmark, no data, no compute |
| 8 Multi-agent and long autonomy | not scheduled; single-agent durability is the current ceiling |
| 9 Frontier competition | not scheduled; nothing to compete with yet |

The honest reading: the project is at Phase 1–2 with a strong control plane and a weak core. The research roadmap is
designed to produce a *credible* Phase 2/3 result rather than to skip ahead to Phase 6.

---

## 9. What we will not do (for the next 90 days)

1. Train anything larger than 10 M parameters.
2. Download a dataset or a model without a licence review and explicit approval.
3. Add a dependency that is not already installed.
4. Claim a capability from a corpus that has not passed the allowlist gate.
5. Run two experiments that share a variable.
6. Integrate a mechanism into `omega_nano/` before it has passed a gate outside it.
7. Rewrite a recorded result.
8. Start multimodal, multi-agent, or computer-use work.

---

## 10. The 12-month thesis, stated so it can be wrong

> By late 2027, OMEGA-NANO will have demonstrated that a CPU-only system with a small core, a verified tool layer, and an
> executable skill fabric can acquire new transformation and verification skills from a handful of examples, reuse them
> across tasks, and prove that reuse did not break anything — at a quality that a same-size model without the fabric
> cannot reach.

If EXP-005 and EXP-006 fail, that thesis is false, and the fallback thesis is:

> A CPU-only system can be *provably correct* on a narrow but useful class of formal and data-transformation tasks, even
> if it cannot acquire skills sample-efficiently.

Both are worth publishing. Neither requires a GPU. That is the point.
