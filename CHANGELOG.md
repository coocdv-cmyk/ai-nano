# Changelog

## 0.3.0-dev — 2026-11-18 (`research/exp-m` M0 Foundation & Hygiene)

### Added
- `RESEARCH/EXP-M-ARCHITECTURE.md` — Program 2/3 specification (`M0`–`M4`) for `OMEGA-NANO V1`.
- `omega_nano/brain/` (from `local-v0.3-brain`): `MathVerifier` (SymPy/mpmath), `CodeVerifier` (AST + isolated Python), `SkillLibrary`, `SelfPlayEngine`, `BlindVault`, and `omega brain` CLI subcommands.
- `omega_nano/core/model.py`: `LoopedOmegaCore` (`v1`: weight-shared loops, ACT halting, MTP) + `build_model`.
- `RESEARCH/RC-004-self-evolving-mind.md` & `EXPERIMENTS/EXP-004-looped-mind-and-speed/`.
- `.github/workflows/ci.yml` — cross-platform CI matrix for `omega_nano`, `core`, and `brain`.

### Fixed (resolving Phase 0 audit defects)
- `fs.write` (`mode="append"`) now reports `is_idempotent(...) == False` via `Tool.dynamic_idempotent`, preventing repeated appends on retry or uncertain crash recovery.
- `LocalLLMReasoner.revise` now isolates failed tool output inside `untrusted_tool_observation` (closing the prompt-injection path).
- `shell.run` now passes a sanitized allowlist of OS environment variables instead of leaking `os.environ` secrets.
- `omega_nano/core/data.py` now strictly excludes `site-packages`, `ensurepip`, and copyleft license headers, and writes SHA-256 provenance manifests.
- `omega_nano/core/engine.py` updated with rolling ring-KV cache and row4-unrolled `_matvec` for `v0`.
- `pyproject.toml` updated to version `0.3.0`, exporting `omega_nano.core` and `omega_nano.brain` plus `core` and `brain` extras.

## Unreleased — 2026-11-18 (documentation and audit only)

No production code was modified in this entry. All changes are documents.

### Added
- `PHASE-0-REPOSITORY-AUDIT.md` — factual audit: environment, source inventory, git state, executed commands, maturity by
  constitution layer, confirmed defects with reproductions, preservation/repair/replace recommendation.
- `RESEARCH-OPPORTUNITIES.md` — 15-family research portfolio with evidence grades, three opportunity cards
  (Skill Fabric/BIRS, verified symbolic workbench, cache-resident ternary core), rejected directions, verified-source log.
- `ORIGINAL-LEARNING-MECHANISM.md` — Skill Fabric / Behaviour-Indexed Residual Synthesis: A–F learning taxonomy, formal
  definitions, algorithm, MDL/invariance/non-interference admission, novelty analysis, falsifiable predictions P1–P4.
- `OMEGA-EXPERIMENT-001.md` — pre-registered EXP-005 (designed, not implemented, not run), including arms, splits, leakage
  controls, decision rules and a 600–900 LOC implementation plan.
- `RESEARCH-ROADMAP.md` — gated research stages R0–R5, V0–V2, K0–K3, S0, E1, with costs and kill criteria.
- `BASELINE-AND-BENCHMARK-PLAN.md` — normative metric definitions (TTFT, prefill, single-stream, aggregate, end-to-end),
  quality floors, fairness rules, banned claims, current baseline ledger.
- `LICENSE-AND-PROVENANCE-REGISTER.md` — licence status, dependency inventory, corpus contamination analysis, remediation
  plan R1–R7.
- `PROJECT-STATE.md` — concise resume document.

### Changed
- `ARCHITECTURE.md` — corrected version to 0.2.0-dev; fixed the stale "neural core not started" row; added task and
  memory data flows, module map, evidence map and companion-document list. Also serves as the requested current
  architecture map (no duplicate file created).
- `ROADMAP.md` — Phase 0 marked passed with evidence; Phase 2 restated with actual numbers and qualifiers; research tracks
  and a re-prioritised backlog added.
- `README.md` — added a "start here" pointer.

### Findings recorded, not fixed
- `fs.write` append-mode idempotency (reproduced: content appended up to three times on retry).- Prompt-injection path through `LocalLLMReasoner.revise`.
- `shell.run` inherits the process environment and is not an OS sandbox.
- Audit log is not tamper-evident; packaging omits `omega_nano.core` and the `torch` extra; version mismatch.
- Corpus is contaminated with LGPL-2.1/Apache-2.0 code from `site-packages`; corpus and checkpoints must not be
  redistributed until rebuilt from an allowlist.
- No `LICENSE` file and zero git commits.

## 0.2.0 — 2026-11-17
- OMEGA core v0 (`omega_nano/core/`): own byte-level decoder with optional ternary weights (STE training), CPU
  training under a wall-clock budget, file-level train/val split, compressor and n-gram baselines.
- Native Numba inference engine (KV cache, rotary attention, SwiGLU, sampling, parallel streams), equivalent to PyTorch.
- `core.complete` tool and "complete ... / كمل ..." intents: the agent can use its own neural core.
- EXP-003 results; RC-003 (decomposition of "the brain").

## 0.1.0 — 2026-11-17
- Phase 1 control foundation: durable task engine with write-ahead checkpoints, checksum-verified state, crash
  recovery without re-execution, approvals, emergency stop, retries with bounded backoff, plan revision, duplicate-delivery
  idempotency and transparent reports.
- Tools: workspace filesystem (atomic writes, recoverable delete), exact arbitrary-precision math, bounded shell, memory.
- Reasoners: bilingual (English/Arabic) rule reasoner; adapter for local OpenAI-compatible LLMs.
- Memory fabric: versioned, provenance-tracked memory with lexical + int8-vector + gated fuzzy retrieval, hard delete,
  bulk ingest, Arabic normalization, Numba int8 kernel.
- Benchmarks: engine loop, memory at 1k/10k/50k. Experiments: EXP-001 (ternary CPU kernels), EXP-002 (retrieval).
- 118 tests.
