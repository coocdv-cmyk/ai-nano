# OMEGA-NANO

A research and engineering program for a compact, efficient, local-first, autonomous AI system.
The mission and rules of engagement are in [OMEGA-NANO-VISION-V3-ENGLISH.md](OMEGA-NANO-VISION-V3-ENGLISH.md) (the constitution).

> **Start here:** [PROJECT-STATE.md](PROJECT-STATE.md) is the five-minute resume. It links the audit, the architecture,
> the research portfolio, the first experiment and the licence/provenance findings.

**Current state: Phase 1, the control foundation, is implemented and tested; Phase 2 has its first measured point** — an own 3.3M-parameter core trained from scratch on this laptop (EXP-003). It is a local, resumable, tool-using agent
with a replaceable reasoning backend, independent verification, durable state, crash recovery, a permission system,
and a persistent memory fabric. Plus the first two measured experiments (CPU low-bit kernels, retrieval at scale).
It is **not** a frontier model, and nothing here claims frontier capability. See [ROADMAP.md](ROADMAP.md) for what is done and what isn't.

## Quick start

Python 3.10+, no required dependencies. Optional: `numpy` (fast vector search) and `numba` (native int8 kernel).

```bash
python -m omega_nano run "calculate 2^200 / 3 then save the result to big.txt"
```

```bash
python -m omega_nano run "احسب ١٢٣٤٥ * ٦٧٨٩ ثم احفظ النتيجة في ناتج.txt"
```

```bash
python -m omega_nano run "remember that the staging server port is 8443"
```

```bash
python -m omega_nano run "what do you know about staging server"
```

```bash
python -m omega_nano run "run python --version"
```

The last command needs approval (process execution in the default `workspace` mode):

```bash
python -m omega_nano approve <task_id>
```

Other commands: `status`, `report <id>`, `resume`, `stop` / `unstop` (emergency stop), `tools`, `doctor`,
`mem add|search|forget|history|stats`. State lives in `./.omega/` (override with `--home` or `OMEGA_HOME`). All file
tools are confined to `./.omega/workspace/`.

Open-ended goals that the offline rule reasoner can't parse are **blocked honestly** rather than guessed. To enable an
LLM planning tier, point OMEGA-NANO at any local OpenAI-compatible server (llama.cpp, Ollama, LM Studio, vLLM):

```bash
OMEGA_LLM_URL=http://127.0.0.1:8080 OMEGA_LLM_MODEL=qwen python -m omega_nano run "tidy up my notes folder"
```

## What one task goes through

```
receive → understand (EN/AR) → plan via adaptive tier (t0 plan cache → t1 rules → t2 local LLM)
→ validate plan against the tool registry → per step:
    permission check (allow / ask human / deny) → write-ahead checkpoint → bounded execution (timeout)
    → observe → independent verification against the world (re-read disk, hash) → retry with backoff
    → revise plan → escalate
→ checkpoint → transparent report → learn (verified plans → plan cache; outcome → episodic memory)
```

## Tests and benchmarks

```bash
python -m pytest -q
```

The current suite reports **264 passed, 1 skipped** (the skip is pre-existing: symlink creation needs admin rights on Windows).
It covers the state machine, checksum corruption and checkpoint restore, path-escape and shell-injection refusal, exact math,
plan parsing in English and Arabic, retries, timeouts, malformed tool output, disk-full errors, prompt injection in files,
emergency stop, duplicate delivery, and real crash-and-resume runs where a subprocess is killed mid-task, plus the
isolated constructive research packages (see below). Historical test counts in the reports reflect the suite at the time
each report was written; the reports are preserved unchanged.

```bash
python BENCHMARKS/bench_engine.py
```

```bash
python BENCHMARKS/bench_memory.py --sizes 1000 10000 50000
```

```bash
python EXPERIMENTS/EXP-001-ternary-cpu/ternary_bench.py
```

Measured on an i7-7820HQ laptop (4C/8T, 8 GB, no GPU). See [BENCHMARKS/README.md](BENCHMARKS/README.md).

## Layout

| Path | What |
|---|---|
| `omega_nano/engine.py` | control loop, recovery, retries, revision, approvals |
| `omega_nano/reasoner.py` | rule reasoner (EN/AR) and local LLM reasoner |
| `omega_nano/verify.py` | independent postcondition checks |
| `omega_nano/store.py` | SQLite WAL state, checksummed checkpoints, audit log, plan cache |
| `omega_nano/memory.py` | memory fabric: versioned, provenance-tracked, hybrid retrieval |
| `omega_nano/kernels.py` | native CPU kernels (Numba) |
| `omega_nano/security.py` | permission modes, scopes, workspace path guard |
| `omega_nano/tools/` | typed tools: fs, exact math, shell, memory |
| `ARCHITECTURE.md`, `SECURITY.md`, `ROADMAP.md` | design, threat model, phases |
| `DECISIONS/`, `RESEARCH/`, `EXPERIMENTS/`, `BENCHMARKS/` | ADRs, research cards, experiments, results |

## Original core vs. isolated experimental research

The repository has two deliberately separate layers:

1. **Original project core and control plane** — `omega_nano/` (engine, reasoner, verifier, store, memory, security, tools),
   the original `constructive/` modules, `tests/`, `BENCHMARKS/`, `EXPERIMENTS/`, `DECISIONS/` and `RESEARCH/`. This is the
   implemented system described in `ARCHITECTURE.md`, `SECURITY.md` and `PROJECT-STATE.md`.
2. **Isolated constructive research packages** — `constructive/exp_h/`, `constructive/exp_i/` and `constructive/exp_j/`.
   These are self-contained experiments that use the original core **read-only**; they do not modify it and are not
   integrated into OMEGA-NANO. Each package owns its source, results, immutable snapshots and final report:

   | Package | Report | Question it tested |
   |---|---|---|
   | `constructive/exp_h/` | `CONSTRUCTIVE-EXP-H-REPORT.md` | Baseline discovery/scaling/OOD/abstraction probes on the constructive task world |
   | `constructive/exp_i/` | `EXP-I-REPORT.md` | Independent typed-DSL evidence, retrieval, search, repair and resource-governance mechanisms |
   | `constructive/exp_j/` | `EXP-J-REPORT.md` | Independent transfer, noise/drift, deep composition, abstraction economics and mechanism interactions |

   Their reports are historical records: results, snapshots, manifests and SHA-256 files are preserved exactly as
   generated. Do not regenerate or edit them.

## Reproducing the experiments

All principal research packages are CPU-only and deterministic where seeded. Exact commands are in each report and in
`constructive/exp_j/README.md`:

```bash
python -m pytest tests/test_exp_i.py -q
python -m constructive.exp_i.runner --no-resume

python -m pytest tests/test_exp_j.py -q
python -m constructive.exp_j.runner --stage A
python -m constructive.exp_j.runner --no-resume
python -m constructive.exp_j.runner --merge-only

python -c "from constructive.exp_j.runner import verify_integrity; print(verify_integrity())"
```

**Inspecting historical results:** start with `PROJECT-STATE.md`, then the report for the experiment of interest.
Machine-readable results are under each package's `results/latest.json`; immutable per-run snapshots are under
`results/experiment-history/`. STAGE-0 baseline hash manifests and integrity reports record which files are protected.
Never overwrite a snapshot or a historical `results/*.json`.

## Limitations and scope

These are benchmark-local research results, not claims of general intelligence, natural-language understanding,
universal generalization, or replacement for any frontier model. EXP-I and EXP-J explicitly report **unresolved
limitations** (for example spurious acceptance in deep composition, no measured abstraction benefit, and calibration
that is not solved). EXP-J's integration-readiness assessment is **not recommended**. Read the "What remains" and
"Integration-readiness" sections of `EXP-I-REPORT.md` and `EXP-J-REPORT.md` before quoting any number. Hardware and
machine-specific paths in reports are disclosures of the machines that produced the measurements, not requirements.

Experimental runners are written for CPU-only execution on the test machines; optional `numpy`/`numba` acceleration is
described in `pyproject.toml` and `BENCHMARKS/README.md`. No external API or network access is required by the research
runners.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: work on isolated branches, never modify protected files or historical
results, run the test suite before submitting, and record seeds, configuration, metrics and reproduction steps.

## Licence

No `LICENSE` file is present, so default copyright applies ("all rights reserved"). A public repository is **not** a
licence grant. The owner must explicitly select a licence before anyone may legally reuse or redistribute the code.
See `LICENSE-AND-PROVENANCE-REGISTER.md` for the full finding, including third-party-licence contamination of the
EXP-003 training corpus and checkpoints (those generated data files are not published in this repository).
