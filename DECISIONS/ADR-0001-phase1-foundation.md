# ADR-0001 — Phase 1 foundation: Python + SQLite, stdlib-first, rules before models

**Date:** 2026-11-17 · **Status:** accepted

## Context
The repository contained only the constitution. §12 sets the first milestone: a local, resumable, tool-using agent with a
replaceable reasoning backend. Hardware: a CPU-only laptop with 8 GB RAM. Python 3.10, numpy, numba and torch are available;
no C toolchain and no local LLM server.

## Decisions
1. **Python, standard library first.** Runs anywhere, is fast to iterate on, and numpy/numba are optional accelerators with
   pure-Python fallbacks (tested). Hot paths move to native kernels when measurements justify it, as done for vector search.
2. **SQLite in WAL mode as the durable store,** one JSON state document per task, checksummed, plus numbered checkpoints.
   Alternatives considered: plain JSON files (no atomic multi-record writes) and an external DB (a network dependency,
   which conflicts with §4.1 offline-first). SQLite also provides FTS5 for retrieval at no extra cost.
3. **A deterministic rule reasoner as tier 1,** with LLMs behind an OpenAI-compatible HTTP adapter. This makes the whole
   control loop testable without a model, and it never silently routes work to a proprietary cloud model (§1.3). Unparsed
   goals are blocked, not guessed.
4. **Verification is separate from execution,** and the world is the source of truth (§4.1, "separate execution from
   verification").
5. **Hashed embeddings instead of a downloaded encoder** for v0: no download, deterministic, CPU-cheap. Swappable behind
   `embed()`. RC-002 tracks the evidence needed to replace it.

## Consequences
- Open-ended natural-language capability depends on attaching a local model (backlog item 1).
- All Phase 1 gate items are covered by automated tests, including real process-kill recovery.
