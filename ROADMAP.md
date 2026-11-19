# Roadmap — evidence gates (constitution §11)

Updated 2026-11-19, after `EXP-M` (`M0`–`M4`) completed on `research/exp-m` and `EXP-N` N0–N1 completed on
`research/exp-n`. The **research** sequence (experiments, gates, kill criteria) is in `RESEARCH/EXP-N-PLAN.md`; this file
tracks product maturity against the constitution's phases.

| Phase | Gate | Status |
|---|---|---|
| 0 Mission & repo audit | can explain the current state and the next experiment | **passed 2026-11-18** — `PHASE-0-REPOSITORY-AUDIT.md`; 9 of 10 audit defects repaired in `M0` |
| 1 Local agent slice | end-to-end behaviour, automated tests, recovery tests | **passed**: 166 passed, 1 skipped at the `EXP-M` gate (2026-11-19); active suite now **180 passed, 1 skipped** after the N0/N1 tests |
| 2 Core model & inference baselines | measured local core with a profile | **compression gate passed on a clean split** (EXP-005): `v2` 2.00 M → **1.7482 bpb** vs `zlib-9` **1.8370**; `v0` 3.34 M → **2.7841** under the same 300 s budget. Native runtime **10,509–13,209 tok/s** single-stream (EXP-007/008). Still **not** a usable language model: Arabic 256-byte windows are **11.12 bpb** vs zlib `4.5447`, and `bz2`/`lzma` remain ahead |
| 3 Memory & knowledge fabric | retrieve evidence, cite provenance, update, survive restart | **largely met** at 50k records (older machine); bilingual AR→EN recall demonstrated (EXP-008); no labelled open-world query set yet |
| 4 Software engineering | real repository tasks with verified tests | **bounded capability (`EXP-N` N1)**: `RepairEngine` solved **6/6** frozen N0 code tasks (24 verifier calls) and **7/7** same-family held-out surface tasks (26 calls), with **0** unexecuted patches. Nine AST operators + traceback/`RepoGraph` localisation; **not open-world** — synthetic workspaces, no OS sandbox |
| 5 Browser & desktop | bounded workflows under permissions | not started |
| 6 Adaptive reasoning | quality gains over baselines at acceptable cost | **partially measured**: 20/20 on sealed author-designed suites (EXP-006/008); **no open-world benchmark** — N0 provides a frozen, contamination-controlled diagnostic, not open-world coverage |
| 7 Multimodal | per-modality quality/latency tests | not started |
| 8 Multi-agent & long autonomy | long-horizon tasks with measured reliability | single-agent durability only |
| 9 Frontier competition | independent, repeatable results | not started |

**Reading the table honestly.** Phase 2 passed the *compression* gate, not the *language* gate: beating `zlib-9` on Python
bytes says the representation is sound, not that the model can converse, reason in Arabic, or hold world knowledge. Phase 6
looks advanced because the sealed suites pass; they were authored with verifiers and are deliberately narrow. Phase 4 now
has a measured repair capability, but on six synthetic N0 tasks plus seven same-family surface variants — it is a
beachhead, not open-world repository engineering. The next honest gate is N2 (verified corpus + trained MTP).

## Research programs

| Program | Family | Status |
|---|---|---|
| Foundation validation | `EXP-L` (L0–L7) | **complete**, merged to `master` (`03149eb`) |
| Architectural evolution | `EXP-M` (M0–M4) | **complete**, `research/exp-m` (`21b93b4`) |
| Integrated model / reasoning dominance | `EXP-N` (N0–N5) | **active** — N0 frozen/baselined; N1 gated (`6/6` N0, `7/7` held-out surface) |
| General capability & learning | `EXP-O` | planned |
| Memory, context, autonomy | `EXP-P` | planned |
| Extreme efficiency | `EXP-Q` | planned |
| Global competitive evaluation | `EXP-R` | planned |

### `EXP-N` tracks (detail: `RESEARCH/EXP-N-PLAN.md`)

| Track | Scope | Gate |
|---|---|---|
| **N0** | sealed `EXP-009` benchmark v1 (repo repair, verifiable reasoning, bilingual retrieval) + baseline | **met**: suite hashed, oracle `100%`, baseline recorded **before** any N1 change (`0/6`, `0/12`, `0.6667` strict recall@5) |
| **N1** | AST patch operators, traceback→localization, execution verification on multi-file bugs | **met (bounded)**: `6/6` N0 at 24 verifier calls, `7/7` same-family held-out at 26 calls, zero unexecuted patches; **not open-world** |
| **N2** | verified corpus scaling + trained MTP heads | **evaluated, gates failed**: 108/108 fresh verified tasks, 90x EXP-008 corpus, greedy-identical over 10k tokens, but acceptance 29–38% and 0.87x–0.92x speed (< 10,509 tok/s); speculation not adopted, speed claim rests on non-spec engine |
| **N3** | learned bilingual metric encoder `< 500 KB int8` | zero-shot AR/EN synonym recall beats the N0 retrieval baseline, honest abstention preserved |
| **N4** | best-first / MCTS over verified candidates | improve multi-function synthesis success at equal verifier budget |
| **N5** | `V2` release gate: capability + efficiency matrix, local UI/API | every number reproducible from a recorded command |

## Prioritized backlog (highest value first)

0. **Owner decisions:** project licence (still no `LICENSE`), OS sandbox for `shell.run` (defect #4), benchmark-seal
   confirmation.
1. **`EXP-N` N2 — done, negative result**. 108/108 verified fresh frames, two MTP objectives trained, native
   speculation lossless but 0.87x–0.92x and below both quantitative gates. Keep speculation disabled by default and
   record the defect (patch-boundary copy hand-off) before revisiting it. Results: `EXPERIMENTS/EXP-012-n2-mtp/results.json`.
2. **`EXP-N` N4** — value-guided search over verified candidates at equal verifier budget (can start from the N1 engine
   independently of N2).
3. **Defect #5** — hash-chain the audit log.
4. **Arabic byte efficiency** (`11.12 bpb` vs zlib `4.5447`): the single worst measured number in the project; `N3` must
   address it or the bilingual claim is empty.
5. **Sandbox** for `shell.run` (job object / restricted token) before any autonomous editing of the real workspace.
6. **Corpus expansion** under the same licence policy (multi-domain: Python, Markdown, Arabic prose, English prose).
7. **Document ingestion** (Markdown/PDF/text → provenance-bearing chunks) into the memory fabric.
8. **Long-context measurement** beyond the 2048-byte copy-gate horizon.
9. Multi-worker task queue with leases; distributed recovery.

## What is not on the backlog, deliberately

Training a model larger than 10 M parameters; downloading datasets or checkpoints without a licence review; multimodal,
computer-use or multi-agent work; any architecture experiment that cannot be measured on the current corpus. These are
downstream of gates that have not been met, not permanent refusals.
