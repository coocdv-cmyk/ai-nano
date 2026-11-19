# EXP-006 — M3 Execution-Guided Neuro-Symbolic & Bilingual Semantic Brain

- **Branch:** `research/exp-m`
- **Date:** 2026-11-18
- **Hardware:** CPU-only (Intel Core i7-7820HQ, 7.41 GB RAM, Windows 10, Python 3.10.11)
- **Script:** `run_exp.py` (`results.json`, wall time **9.83 s**)

## What Changed in M3

1. **Bilingual Semantic Memory (`omega_nano/semantic.py` + `memory.py`):**
   - Replaced blind raw-byte hashing with bilingual AR/EN stemming, camelCase/snake_case identifier splitting, and curated CS/assistant concept anchors inside a 512-dim int8 vector index fused via weighted RRF with SQLite FTS5 + char trigrams.
2. **Execution-Guided Program Synthesizer (`omega_nano/brain/synthesizer.py`):**
   - Replaces static single-template lookup with specification analysis, 9 compositional algorithmic hypothesis families (plus bottom-up typed I/O expression enumeration), online bilinear `NeuralPolicyRanker` persisted in SQLite (`kv_state`), and CEGIS counterexample-guided repair (`diagnose_failure`) from real `CodeVerifier` errors.
3. **EXP-L Safety Integration (`omega_nano/brain/evolver.py`):**
   - `L3` behavioral guard (`verify_behavioral_candidates`) tests all regex-matched compiled skills against the live verifier before trusting any cached skill.
   - `L1` `SemanticDriftMonitor` tracks unexpected skill verification drops via sliding-window consistency + CUSUM and triggers automatic self-healing re-synthesis.
4. **AST Repo Symbol & Call-Graph Indexer (`omega_nano/brain/repo_graph.py`):**
   - Indexes Python classes, functions, methods, signatures, docstrings, and direct call edges into `RepoGraph` and `MemoryFabric(kind="project")`.

## Measured Results (`results.json`)

| Metric | Baseline (`SymbolicTacticSolver`) | M3 (`NeuroSymbolicSolver`) |
|---|---:|---:|
| `BlindVault` Total (Math + Code) | **5 / 10 (50.0%)** | **10 / 10 (100.0%)** |
| `BlindVault` Math | 5 / 5 | 5 / 5 |
| `BlindVault` Code | 0 / 5 | **5 / 5** (avg **1.30** attempts) |
| `BlindVault.heldout()` (10 unseen code tasks) | — | **10 / 10 (100.0%)** (avg **2.20** attempts) |
| Self-Play Cycle 1 -> Cycle 2 (`n=12`, diverse code) | — | **12/12 (`1.08` att) -> 12/12 (`1.00` att)** |
| Cross-Name Policy Transfer (reopen fresh DB, unseen fn name) | `2` attempts (cold) | **`1` attempt** (418 features persisted) |
| Spec Drift (`L3` reject + `L1` CUSUM + self-heal) | Silent failure risk | **3/3 rejected, 3/3 healed, drift flagged, `1` att post-heal** |
| Repo Indexer (`omega_nano/`, 35 files, 374 symbols) | — | **183.69 ms** (`0` errors), recall **2.96 ms** |
| Bilingual & Identifier Semantic Probes | — | **4 / 4 Top-3 hits** (`0.88–1.88 ms`) |

## Honest Scope & Limits

- Compositional synthesis currently spans 9 algorithmic families (intervals, caches, DAGs/graphs, stack machines, UTF-8/budget chunkers, sliding windows/subarrays, binary search, 1D/string DP, string transforms) plus inline I/O AST enumeration; novel multi-module open-ended synthesis beyond those families awaits M2 (fine-tuned `OmegaCoreV2` neural proposal generator) and M4 (full autonomous loop).
- Bilingual semantic recall uses deterministic morphology + curated domain concept clusters rather than an unconstrained neural bilingual embedding transformer; arbitrary out-of-domain synonyms with zero surface/root overlap still require M2 semantic distillation.
