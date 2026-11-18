# Roadmap — evidence gates (constitution §11)

Updated 2026-11-18 by the Phase 0 audit. The **research** sequence (experiments, gates, kill criteria) is in
`RESEARCH-ROADMAP.md`; this file tracks product maturity against the constitution's phases.

| Phase | Gate | Status |
|---|---|---|
| 0 Mission & repo audit | can explain the current state and the next experiment | **passed 2026-11-18** — full audit in `PHASE-0-REPOSITORY-AUDIT.md`; architecture corrected; research plan defined |
| 1 Local agent slice | end-to-end behaviour, automated tests, recovery tests | **passed**: 126 passed, 1 skipped (127 collected); includes real kill-and-resume |
| 2 Core model & inference baselines | measured local core with a profile | **first point measured, below the quality floor**: 3.34 M-param byte-level core, 1.8944 bpb fp32 / 2.0554 ternary vs zlib 1.8296 (EXP-003). Decode 1,425 B/s single-stream / 5,027 B/s aggregate (batch 8) on the current laptop. Not a usable language model |
| 3 Memory & knowledge fabric | retrieve evidence, cite provenance, update, survive restart | **largely met** at 50k records (recorded on a different machine); no labelled query set; document ingestion missing |
| 4 Software engineering | real repository tasks with verified tests | not started; blocked by the missing OS sandbox (S0) |
| 5 Browser & desktop | bounded workflows under permissions | not started |
| 6 Adaptive reasoning | quality gains over baselines at acceptable cost | planning tiers exist; **no quality benchmark**, so the gate cannot be evaluated |
| 7 Multimodal | per-modality quality/latency tests | not started |
| 8 Multi-agent & long autonomy | long-horizon tasks with measured reliability | single-agent durability only |
| 9 Frontier competition | independent, repeatable results | not started |

**Reading the table honestly.** Phase 2 is not "passed" because the model is worse than zlib; it has a measured point on a
contaminated corpus. Phase 3 is strong on mechanism and weak on evaluation. Phase 6 looks advanced because tiers exist, but
a tier without a quality benchmark is an architecture, not a capability.

## Research tracks (next 90 days)

The research roadmap supersedes the old backlog for anything that produces a claim. Detail: `RESEARCH-ROADMAP.md`.

| Track | Stage | Gate |
|---|---|---|
| A — Skill Fabric | EXP-005 (designed, not run) | ≥ 2× the best baseline at k=3, or kill |
| B — Verified workbench | V0 (30 Z3/SymPy goals) | ≥ 90% closed, 0 unsound |
| C — Kernel/core | K0 (blocked matvec), K1 (rolling cache) | ≥ 2× in isolation; ≥ 4,000 B/s single-stream |
| D — Hygiene | corpus allowlist, first commit, CI | zero copyleft markers; green CI |

## Prioritized backlog (highest value first)

0. **Owner decisions:** project licence, corpus rebuild approval, first git commit. Nothing may be published before these.
1. **Implement and run EXP-005** (`OMEGA-EXPERIMENT-001.md`): the cheapest test of the project's central research claim.
2. **Repair the confirmed defects** (audit §8), highest severity first:
   - `fs.write` append idempotency (data corruption, reproduced);
   - `LocalLLMReasoner.revise` prompt-injection path;
   - `shell.run` environment leakage and missing OS sandbox;
   - hash-chained audit log.
3. **Corpus remediation** (`LICENSE-AND-PROVENANCE-REGISTER.md` §9): allowlist + SPDX + SHA-256, then re-train and
   re-measure. Required before any external quality claim.
4. **Core engine speed** (K0–K3): blocked int8/2-bit matvec and a rolling KV cache. Target ≥ 4,000 B/s single-stream with
   identical greedy output; the 10k ambition is only claimable after a measured decomposition.
5. **EXP-004: AVX2 SIMD-LUT 2-bit kernel** in C — still blocked by the absent compiler. Supersedes K0 if a toolchain
   appears.
6. **OS-level sandbox for `shell.run`** (Windows job object / restricted token). Blocks Phase 4 and any autonomous coding.
7. **Connect a real local model to tier t2** (llama.cpp / Ollama) and build a 50-goal EN+AR task benchmark comparing the
   rule and LLM tiers on success, verification-pass rate and latency.
8. **Semantic-paraphrase retrieval benchmark** (300 labelled queries) to justify or replace the hashed vector signal.
9. **Repository indexing tool** (file map, symbols, dependency graph) into project memory — the first Phase 4 step.
10. **Document ingestion** (Markdown/PDF/text → chunks with provenance) into the memory fabric.
11. **CI**: tests, SPDX headers, corpus manifest check, packaging smoke test.
12. Multi-worker task queue with leases; distributed recovery.

## What is not on the backlog, deliberately

Training a model larger than 10 M parameters; downloading datasets or checkpoints without a licence review; multimodal,
computer-use or multi-agent work; any architecture experiment that cannot be measured on the current corpus. These are
downstream of gates that have not been met, not permanent refusals.
