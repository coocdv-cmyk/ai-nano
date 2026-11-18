# GITHUB-PUBLICATION-REPORT

## Publication outcome: **PUBLISHED AND VERIFIED**

| Item | Value |
|---|---|
| Repository URL | https://github.com/coocdv-cmyk/ai-nano.git |
| Visibility | Public (pre-existing setting; **not** changed by this publication) |
| Published branch | `master` (now tracks `origin/master`) |
| Published commit (initial) | `66300138543fac8b5b9acfe3ffa4416802c77243` |
| Commit message | `Initial complete publication of OMEGA-NANO research project` |
| Total tracked files | **312** |
| Total published size | **20,547,764 bytes (~19.6 MB)** of tracked files as checked out; local staging size 20,527,459 bytes (~19.58 MB). Difference is Windows CRLF checkout normalization. |
| Verification method | `git ls-remote origin` + fresh `git clone` into an independent temp directory, then file/hash inspection of the clone |
| Status of remote before push | Empty (no refs); confirmed with `git ls-remote` (exit 0, no output) before adding the remote |

## Important published directories and assets

| Path | Present remotely | Notes |
|---|---|---|
| `omega_nano/` | yes | Original control-plane implementation, preserved unchanged |
| `constructive/` (original modules) | yes | Original constructive core, preserved unchanged |
| `constructive/exp_h/` | yes | 41 immutable experiment snapshots |
| `constructive/exp_i/` | yes | 48 immutable experiment snapshots |
| `constructive/exp_j/` | yes | 12 immutable experiment snapshots |
| `tests/` | yes | 14 test files including `test_exp_h.py`, `test_exp_i.py`, `test_exp_j.py` |
| `BENCHMARKS/`, `EXPERIMENTS/`, `RESEARCH/`, `DECISIONS/` | yes | Benchmarks, experiment records, research cards, ADRs |
| Reports | yes | `README.md`, `ARCHITECTURE.md`, `SECURITY.md`, `CONTRIBUTING.md`, `CONSTRUCTIVE-EXP-H-REPORT.md`, `EXP-I-REPORT.md`, `EXP-J-REPORT.md`, `LICENSE-AND-PROVENANCE-REGISTER.md`, `PROJECT-STATE.md`, and other root documents |
| Machine-readable results | yes | Every package's `results/latest.json`, per-experiment files, all historical snapshots |
| Integrity assets | yes | Both STAGE-0 baseline hash manifests and both integrity reports |

All 27 required documentation/research assets were checked in the fresh clone; none were missing.

## Security audit outcome

- **Manual secret scan** of every file (including ignored files) for API keys, tokens, passwords, private-key headers,
  GitHub token patterns and cloud key patterns: **no matches**.
- **Filename scan** for `.env`, `id_rsa`, `*.pem`, `*.key`, `*.pfx`, cookies/session files: **no secret files found**
  (the only name match was a research card titled "ten-thousand-tokens-per-second", which is not a credential).
- **Local absolute-path scan** for personal home directories: **no matches**.
- **Git history scan:** the local repository had **zero commits** before this publication, so there was no history
  containing secrets; the initial commit is the first commit.
- **Secret scanners:** `gitleaks`, `trufflehog` and `git-secrets` are not installed on this machine; the audit used
  exhaustive regex and filename scans instead. This limitation is recorded here rather than hidden.
- **Credential handling:** authentication used the existing GitHub CLI credential helper (account `coocdv-cmyk`).
  No credential was printed, pasted into chat, or written to any file.
- **Published tree re-check:** 0 tracked files matching secret filenames or `.pem/.key/.pfx/.p12/.bin/.pt` patterns.
- **Git config note:** global `safe.directory` entries on this machine point at unrelated local paths; they are global
  configuration and were **not** copied into this repository. The local commit identity was set to the neutral
  `OMEGA-NANO Research <noreply@github.com>` to avoid publishing personal information.

## Test results (actually run during this task, not copied from reports)

| Check | Exact command | Actual result |
|---|---|---|
| Full suite | `python -m pytest -q` | **264 passed, 1 skipped in 50.60s** |
| EXP-J smoke | `python -m constructive.exp_j.runner --stage A` | **8/8 pass, 0 failures** |
| EXP-I integrity | `python -c "from constructive.exp_i.runner import verify_integrity; ..."` | **True, 22 protected files** |

Historical reference points from the latest research report (EXP-J tests 35/35, full suite 264 passed / 1 skipped,
Stage-A 8/8, campaign 6/6 completed, protected files 84/84) were **not treated as permission to modify files**; the
actual available tests were run and their real results are recorded above.

## Hash and integrity verification

| Manifest | Files | Result |
|---|---|---|
| `constructive/exp_i/results/baseline_hashes.json` (EXP-I STAGE 0) | 22 | **all unchanged** |
| `constructive/exp_j/results/baseline_hashes.json` (EXP-J STAGE 0) | 84 | **all unchanged** |

- Rechecked after the push: both manifests still show zero changed files; working tree clean.
- No protected source file, experiment snapshot, historical result, or SHA-256 record was modified.
- Runtime-generated verification artifacts (`results/integrity.json`, `results/stage-a-smoke.json`) are designed to be
  rewritten by their verification commands; their baseline manifests were not changed.
- No `git push --force`, `git reset --hard`, history rewrite, or destructive cleanup was used at any point.

## Intentionally excluded files (with reasons)

These files were **not silently discarded**; they are listed here explicitly.

| Excluded path/pattern | Reason |
|---|---|
| `.omega/` (local runtime DBs, `core_data/train.bin`, `core_data/val.bin`, `baselines.json`, `meta.json`) | Local machine state and a training corpus documented in `LICENSE-AND-PROVENANCE-REGISTER.md` as containing LGPL/Apache/BSD third-party code. It may not legally be redistributed. Already ignored by the project's pre-existing `.gitignore`. |
| `EXPERIMENTS/**/*.pt` (two 12.8 MB checkpoints) | Model weights derived from the licence-contaminated corpus; per the provenance register, neither corpus nor weights may be redistributed. Already ignored by the pre-existing rule. |
| `__pycache__/`, `*.pyc`, `*.nbi`, `*.nbc` | Generated bytecode/cache, not source or results. |
| `.pytest_cache/`, `.claude/`, `.idea/`, `.vscode/`, `.venv/`, `*.log`, `*.tmp`, `*.bak`, OS metadata | Local caches, IDE/agent settings and transient files; excluded by the updated `.gitignore`. |
| `LICENSE` | **No licence file was created.** No licence has been selected or authorized by the owner. The repository is public but all rights remain reserved; see `LICENSE-AND-PROVENANCE-REGISTER.md`. |

The `.gitignore` was extended with secret/venv/IDE/temp/OS exclusions while **preserving every existing rule**.
Research assets (experiment source, reports, `results/latest.json`, immutable snapshots, manifests, SHA-256 records,
tests and benchmarks) are explicitly **not** ignored and are all published.

## Unresolved issues / owner actions

1. **Licence decision required.** No `LICENSE` file exists by design. The owner must choose a licence before others may
   legally reuse the code. See `LICENSE-AND-PROVENANCE-REGISTER.md`.
2. **Licence-contaminated corpus and weights remain local only.** Do not upload `.omega/core_data/*.bin` or
   `EXPERIMENTS/**/*.pt` from any machine without resolving the third-party licence question.
3. **Research limitations are unresolved by design** (see `EXP-I-REPORT.md` and `EXP-J-REPORT.md`): calibration is not
   solved, deep-composition spurious acceptance persists, drift detection is unreliable, and abstraction lifecycle cost
   is positive. These are reported, not hidden, and no integration is authorized.
4. **No secret-scanner binary was available**; the audit was manual/regex-based. Re-run a Gitleaks scan on any future
   machine that has it installed.
5. **Commit attribution** uses the neutral identity above; if the owner wants personal attribution, set `user.name` and
   `user.email` locally and make a new commit (do not amend published history).

## Cloning and continuing on another laptop (exact instructions)

```bash
# 1. Clone (HTTPS; GitHub CLI / credential manager handles authentication)
git clone https://github.com/coocdv-cmyk/ai-nano.git
cd ai-nano

# 2. Confirm you are on the published branch and commit
git branch --show-current                 # master
git log -1 --format='%H %s'               # 66300138543fac8b5b9acfe3ffa4416802c77243 Initial complete publication...

# 3. Set your own local identity (do not use shared/global config with personal data)
git config --local user.name "Your Name"
git config --local user.email "your-account-email@example.com"

# 4. Create an isolated branch before any work
git switch -c exp/your-topic

# 5. Environment: Python 3.10+, no required dependencies for the control plane
python -m pip install -e ".[dev]"         # optional; pytest only
# Optional accelerators: numpy (fast vector search), numba (native int8 kernel)

# 6. Verify integrity and tests before changing anything
python -m pytest -q
python -c "from constructive.exp_j.runner import verify_integrity; print(verify_integrity())"

# 7. Run a research package (example)
python -m constructive.exp_j.runner --stage A
python -m constructive.exp_j.runner --no-resume

# 8. Work normally; never rewrite history
git add -A
git commit -m "Describe the change and its evidence"
git push -u origin exp/your-topic
```

The generated `.omega/`, `.pt` checkpoints and corpus binaries are **not** in the clone. If a new machine needs to run
the original core experiments that produced those artifacts, regenerate them locally from the documented procedures in
`EXPERIMENTS/EXP-003-omega-core-v0/README.md` and `BENCHMARKS/README.md` — do not copy them from another machine's
private storage without resolving the licence issue first.
