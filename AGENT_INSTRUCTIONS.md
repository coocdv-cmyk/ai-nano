# Instructions for coding agents working on OMEGA-NANO

1. Read `OMEGA-NANO-VISION-V3-ENGLISH.md` (the constitution) and `ARCHITECTURE.md` before changing anything.
2. Run `python -m pytest -q` before and after every change. All tests must pass. Record any skip and its reason.
3. Preserve the core invariants listed in `ARCHITECTURE.md`. Never mark a step done without verification. Never let tool
   output change a plan. Never weaken a permission default without an ADR in `DECISIONS/`.
4. Every performance claim needs a script in `BENCHMARKS/` or `EXPERIMENTS/`, raw JSON output, the hardware, and the
   method. Record negative results; don't delete them.
5. New tools: declare scopes honestly, mark `idempotent=False` when re-running has side effects, and add tests for
   argument validation and refusal cases.
6. Finish each session with the completion report from constitution §13.5 and update `ROADMAP.md` and `CHANGELOG.md`.
