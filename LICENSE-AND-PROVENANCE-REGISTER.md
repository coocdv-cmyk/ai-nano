# Licence and Provenance Register

Created 2026-11-18. Scope: the repository, its declared dependencies, the training corpus, and every artefact derived
from them. This is an engineering register, not legal advice. It records what is **known**, what is **unknown**, and what
must be true before the project may redistribute anything.

---

## 1. Summary — three blockers

| # | Finding | Severity | Consequence |
|---|---|---|---|
| 1 | **No `LICENSE` file** in the repository | High | Default copyright applies: all rights reserved. Nobody may legally reuse, redistribute or contribute without the owner's permission. The project is not open source, despite being public-looking |
| 2 | **The training corpus is not pure PSF.** It is every `*.py` under the local Python `Lib/`, which includes `site-packages` | **Critical** | The corpus contains LGPL-2.1, Apache-2.0 and BSD-licensed third-party code. The EXP-003 checkpoints are derived works of it. Neither corpus nor weights may be redistributed, and any quality claim built on them is contaminated |
| 3 | **No provenance manifest** (SPDX identifiers, SHA-256 per file, generator version) | High | Even if the corpus were clean, there is no artefact that proves it. Reproducibility and auditability both fail |

Two secondary findings:

4. **Zero git commits.** Every file is untracked. There is no authorship record, no history, and no way to attribute a
   change. This is a provenance failure independent of licensing.
5. **Recorded results were produced on a different machine** (i7-7820HQ) from the current one (i5-1135G7), and the result
   files do not contain a corpus hash. They cannot be tied to a corpus version.

---

## 2. Project licence status

- **File present:** none. `Get-ChildItem` at the repository root lists no `LICENSE`, `LICENCE`, `COPYING` or `NOTICE`.
- **`pyproject.toml`:** declares package metadata but no `license` field and no `classifiers` for licensing. (Verified by
  reading the file; it is 447 bytes.)
- **Effect under the Berne Convention:** copyright subsists automatically. Without a grant, the default is "all rights
  reserved". A public repository is not a licence.
- **What this blocks:** redistribution, packaging for PyPI, accepting outside contributions, publishing weights, and any
  claim that the project is "open".
- **What this does not block:** local use, internal research, and the documentation in this package.

**Required decision (owner, not the agent):** choose a licence. The plausible options and their consequences:

| Option | Effect | Fit for this project |
|---|---|---|
| Apache-2.0 | Permissive + explicit patent grant + trademark clause | Good for a research/engineering project that may be commercialised |
| MIT | Shortest permissive licence | Good if simplicity matters more than patents |
| AGPL-3.0 | Copyleft over network use | Consistent with "local-first, user-controlled", but hostile to some adopters |
| Dual / delayed open | Research licence now, open later | Requires legal drafting; not recommended without counsel |

**No licence file may be added by an agent without explicit authorisation.** This register records the decision point; it
does not make the decision.

---

## 3. Dependency inventory

Verified from installed package metadata on 2026-11-18 (aggregate fields, not file-by-file audits):

| Dependency | Version | Licence | Obligations if redistributed | Verified |
|---|---|---|---|---|
| Python | 3.13.15 | PSF-2.0 | Notice; no copyleft | metadata |
| NumPy | 2.5.3 | BSD-3-Clause | Notice, no endorsement | metadata |
| Numba | 0.68.0 | BSD-2-Clause | Notice | metadata |
| llvmlite | (Numba dep) | BSD-2-Clause | Notice | metadata |
| PyTorch | 2.14.1+cpu | BSD-3-Clause | Notice; **plus** bundled third-party notices | metadata |
| pytest | (dev) | MIT | Notice | metadata |

All declared dependencies are permissive. **The dependency tree is not the problem.** The corpus is.

**Caveat.** `torch` wheels bundle MKL, oneDNN and other components with their own notices. A redistributed binary must
carry PyTorch's `LICENSE` and `NOTICE` files. This register does not attempt a file-level audit of the wheel; that is a
packaging task, tracked in §9 R6.

---

## 4. Corpus provenance — the critical finding

### 4.1 How the corpus is actually built

`omega_nano/core/data.py:26–52`:

```python
lib = os.path.dirname(os.__file__)                      # …/Python313/Lib
files = sorted(glob.glob(os.path.join(lib, "**", "*.py"), recursive=True))
```

`Lib/**/*.py` includes `Lib/site-packages/**`. The module docstring says:

> "v0 corpus: the Python standard library sources installed on this machine (PSF licence, local, no download)."

That statement is **false as implemented**. The standard library is PSF-2.0; `site-packages` is a mixture of whatever the
user installed, under many licences.

### 4.2 What the generated corpus contained

Measured during the audit by scanning the generated `train.bin`/`val.bin` for licence headers:

| Marker | Train occurrences | Val occurrences |
|---|---|---|
| `GNU Lesser General Public` | **62** | 4 |
| `Apache License` | 42 (combined total; not separated by split) | (included) |
| `MIT License` | present | present |
| `BSD` | present | present |

Size and shape: **19,888,741 bytes train / 1,921,027 bytes val**, from **1,408 / 91 files**. The split is by
`sha256(relative path)` bucketed at 5%, so it is deterministic but not stratified by licence.

### 4.3 Which packages are implicated

The scan and the file list show at minimum:

| Package | Licence | Copyleft? |
|---|---|---|
| `chardet` | LGPL-2.1 | **Yes — weak copyleft** |
| `requests` | Apache-2.0 | No, but notice + patent terms |
| `distro` | Apache-2.0 | No |
| `packaging` | Apache-2.0 / BSD dual | No |
| assorted `pip`/`setuptools` vendored code | MIT/BSD/PSF | No |

The LGPL presence is the decisive one. LGPL-2.1 requires that modifications to the licensed work be released under the
same licence and that users be able to relink. A byte-level language model trained on LGPL source is a legally unsettled
case; the conservative and correct engineering position is: **do not redistribute the corpus or any model trained on it
until it is rebuilt from an allowlist.**

### 4.4 Why "trained from scratch on this laptop" is not a defence

Copyright protects the expression, not the learning algorithm. Training on a work does not launder the licence of the
work, and the resulting weights may be treated as a derived work in some jurisdictions. This is unsettled law, which is
exactly why a research project that intends to publish must avoid the question rather than gamble on it.

### 4.5 Consequences for existing artefacts

| Artefact | Path | Status |
|---|---|---|
| Corpora `train.bin`, `val.bin`, `meta.json` | generated into the data dir, not committed | **Contaminated.** Do not commit, do not redistribute |
| fp32 checkpoint | `EXPERIMENTS/EXP-003-omega-core-v0/runs/fp32/model.pt` (13,388,793 B) | Derived work. Local research only |
| ternary checkpoint | `EXPERIMENTS/EXP-003-omega-core-v0/runs/ternary/model.pt` (13,388,793 B — the checkpoint stores fp32 masters, so the ternary payload size in `results.json` is the *runtime* size, not the file size) | Derived work. Local research only |
| EXP-003 quality numbers | `results.json` | Valid as an internal measurement of *this* corpus; **not publishable** as a capability claim |
| EXP-001 kernel numbers | `EXPERIMENTS/EXP-001-ternary-cpu/results.json` | Kernel microbenchmarks; no corpus dependency; publishable as systems measurements |
| EXP-002 memory numbers | `EXPERIMENTS/EXP-002-int8-retrieval-kernel.md` | Synthetic data; no corpus dependency; publishable with the synthetic set |

This is the single most important distinction in the register: **kernel and memory results are clean; language-model
quality results are not.**

---

## 5. Model artefact provenance

### 5.1 What exists

- EXP-003 produced two checkpoints from the same corpus, same architecture, same wall-clock budget.
- The `results.json` files record hardware, hyperparameters, step counts, bpb, and samples. They do **not** record:
  - a SHA-256 of the corpus,
  - the corpus file list,
  - the Python executable path or the `Lib` directory used,
  - the exact commit of the code that produced them (there is no commit).
- The `.pt` files therefore cannot be tied to a corpus version. Even a benign corpus change would silently invalidate a
  comparison.

### 5.2 What a checkpoint record must contain (going forward)

```json
{
  "checkpoint_sha256": "…",
  "corpus_manifest_sha256": "…",
  "corpus_files": 1408,
  "corpus_bytes": 19888741,
  "corpus_licence_allowlist": ["PSF-2.0", "MIT", "BSD-3-Clause", "Apache-2.0"],
  "code_commit": "…",
  "python": "3.13.15",
  "torch": "2.14.1+cpu",
  "seed": 11,
  "wall_clock_s": 1844
}
```

None of these fields exist today. That is a reproducibility gap, not a legal one, but it compounds the legal one: without
a manifest there is no way to prove that a future clean model was not trained on the old corpus.

---

## 6. Documentation and third-party content

| Item | Status |
|---|---|
| `OMEGA-NANO-VISION-V3-ENGLISH.md` | Original project document; copyright the owner |
| `MODEL_REVIEW.md` | Original review, Arabic; copyright the author |
| Papers cited in `RESEARCH-OPPORTUNITIES.md` | Cited, not reproduced. Short quotations fall under fair dealing/fair use; no long excerpts are included |
| Code snippets from other projects | None copied. Algorithms are described, not transcribed |
| `PHASE-0-REPOSITORY-AUDIT.md` and this package | Original; same licence as the project (undecided, see §2) |

**A note on the name.** "OMEGA-NANO" is used as a project name. No trademark search has been performed. Before any public
release, a trademark check is required; that is outside this register's scope.

---

## 7. Risk register

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| L1 | LGPL code in the corpus makes weights unredistributable | **Certain** (confirmed) | High | Rebuild corpus from allowlist (§9 R2) |
| L2 | No project licence means no one can legally adopt the code | Certain | High | Owner chooses a licence (§2) |
| L3 | Apache-2.0 notice requirements unmet if binaries are shipped | Likely | Medium | Generate `THIRD_PARTY_NOTICES` at package time (§9 R6) |
| L4 | Weights published without a manifest, then corpus changes | Medium | High | Manifest + checkpoint hash (§5.2) |
| L5 | A contributor adds code of unknown provenance | Medium | Medium | DCO/CLA + SPDX header check in CI (§9 R5) |
| L6 | Third-party benchmark data leaked into the repo | Low | High | Dataset cards + hashes; no data without a licence field |
| L7 | Model output reproduces memorised LGPL code verbatim | Low at 3.3 M params, rising with scale | High | Deduplication + memorisation probes before release (§9 R7) |
| L8 | Trademark conflict on the project name | Unknown | Medium | Search before public release |

---

## 8. What may and may not be done today

**Permitted now (no new authorisation needed):**
- Local research, experiments, and documentation.
- Publishing *kernel* and *memory* benchmark numbers, which do not depend on the corpus.
- Describing the architecture and the algorithms.
- Internal comparison of checkpoints, as long as the corpus is unchanged.

**Not permitted without owner action:**
- Publishing or redistributing the corpus, the checkpoints, or any model trained on them.
- Claiming a language-model quality result as a capability result.
- Adding a `LICENSE` file, which is a legal decision, not a formatting one.
- Packaging for distribution (PyPI, wheels) before §9 R6.
- Accepting outside code contributions before §9 R5.

**Not permitted at all, by the constitution:**
- Rewriting recorded results to hide the contamination.
- Deleting the contaminated artefacts (they are evidence, and deletion is destructive). They must be **quarantined and
  labelled**, not erased.

---

## 9. Remediation plan

Ordered by dependency. Each item has an acceptance criterion that a reviewer can check without trusting the author.

### R1 — Record the current state (do first, costs minutes)

- Compute SHA-256 of every file in the repo and of the generated corpora and checkpoints.
- Write `PROVENANCE/current-state.json` with the hashes and the git status (`0 commits`).
- **Acceptance:** a manifest exists; re-running the hash command reproduces it exactly.

### R2 — Rebuild the corpus from an allowlist (the critical fix)

- Replace `Lib/**/*.py` with an explicit **allowlist** of directories and files, excluding `site-packages` by
  construction (not by a filter that can be forgotten).
- Record per-file: path, bytes, SHA-256, SPDX identifier, source URL/version.
- Reject any file whose SPDX identifier is not in `{PSF-2.0, MIT, BSD-2-Clause, BSD-3-Clause, Apache-2.0, CC0-1.0}`.
- **Acceptance:** a scan for `GNU Lesser General Public`, `GNU General Public`, and `AGPL` returns **zero** hits in the
  rebuilt `train.bin` and `val.bin`; the manifest covers 100% of bytes.

### R3 — Re-train and re-measure

- Re-run EXP-003 on the clean corpus with the same architecture and budget.
- Publish the new bpb alongside the old one, labelled "different corpus; not comparable".
- **Acceptance:** the new checkpoint records `corpus_manifest_sha256`; the old number is retained as history.

### R4 — Quarantine the old artefacts

- Move the contaminated checkpoints and corpora to `QUARANTINE/` with a `README` explaining why.
- Do not delete. Do not train on them again.
- **Acceptance:** no code path can reach them by default; the default corpus builder refuses to run without an allowlist.

### R5 — Provenance gates in CI

- A check that every source file has an SPDX header.
- A check that no file over 1 MB is committed without a manifest entry.
- A check that the corpus builder's allowlist is non-empty and that its output hashes match the manifest.
- **Acceptance:** CI fails on a deliberately added unlicensed file (test the gate, do not assume it works).

### R6 — Packaging notices

- Generate `THIRD_PARTY_NOTICES.md` from installed metadata at build time.
- Include PyTorch's bundled notices if a binary is shipped.
- **Acceptance:** a wheel built from the repo contains the notices and the project licence.

### R7 — Memorisation probes before any weight release

- Prompt the model with prefixes from the training corpus and measure verbatim continuation rate.
- Report the rate; if it is non-trivial, do not release weights.
- **Acceptance:** a documented probe with a threshold chosen before the measurement.

---

## 10. Verification commands (read-only)

The snippets are shown in bash syntax. On Windows PowerShell use `Get-ChildItem` / `Select-String`, or run them under
Git Bash or WSL. The Python one-liners are portable if invoked as `python -c "..."`.

```bash
# 1. Is there a licence file?
ls LICENSE LICENCE COPYING NOTICE 2>/dev/null || echo "no licence file"

# 2. What does pyproject declare?
grep -n -i "license\|classifier" pyproject.toml

# 3. Dependency licences (aggregate metadata)
python - <<'PY'
from importlib.metadata import metadata
for p in ["numpy", "numba", "torch", "pytest", "llvmlite"]:
    m = metadata(p)
    print(p, m.get("Version"), "|", m.get("License"), "|", m.get("License-Expression"))
PY

# 4. Corpus contamination scan (operates on the generated bins; no writes)
python - <<'PY'
from pathlib import Path
markers = [b"GNU Lesser General Public", b"GNU General Public", b"GNU Affero"]
for name in ["train.bin", "val.bin"]:
    for p in Path(".").rglob(name):
        data = p.read_bytes()
        hits = {m.decode(): data.count(m) for m in markers}
        print(p, len(data), hits)
PY

# 5. Corpus builder scope (prove site-packages is included)
python - <<'PY'
import os, glob
lib = os.path.dirname(os.__file__)
files = glob.glob(os.path.join(lib, "**", "*.py"), recursive=True)
sp = [f for f in files if "site-packages" in f]
print("total", len(files), "site-packages", len(sp))
PY

# 6. Git provenance
git log --oneline | head -5 || echo "no commits"
git status --porcelain | wc -l
```

---

## 11. Decision log

| Date | Decision | Owner | Status |
|---|---|---|---|
| 2026-11-18 | Record contamination as a blocker; do not remediate destructively | audit | done |
| 2026-11-18 | Do not add a licence file without owner approval | audit | done |
| — | Choose the project licence | owner | **open** |
| — | Approve corpus rebuild (changes `data.py`, which is production code) | owner | **open** |
| — | Approve re-training and publication of new numbers | owner | **open** |

---

## 12. Limitations of this register

- It is not a legal opinion. The LGPL analysis is a conservative engineering position, not a court ruling.
- Dependency licences were read from package metadata, not from file-level audits. Wheels bundle more than their metadata
  declares.
- The corpus scan counts **markers**, which can produce false positives (a string in a test fixture) and false negatives
  (a licence without the standard header). It is sufficient to prove contamination, not to prove cleanliness.
- Only the Python `Lib` corpus was examined. Any future corpus (papers, web text, code repositories) needs its own
  register.
- The absence of a licence file was verified at the repository root; a licence could exist elsewhere, but none was found
  in a recursive listing of tracked-looking paths.
