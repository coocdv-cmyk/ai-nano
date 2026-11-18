# Security model

## Permission modes (per task)

| Scope → / Mode ↓ | read | write | execute | destructive |
|---|---|---|---|---|
| `read_only` | allow | deny | deny | deny |
| `ask` | allow | ask | ask | ask |
| `workspace` (default) | allow | allow | ask | ask |
| `autonomous` | allow | allow | allow | ask |

Destructive actions **always** need a human, in every mode. Scopes can be raised per call: `fs.write` that would
overwrite an existing file with different content becomes `destructive`. Approvals are per step, persist across restarts,
and are recorded in the audit log. `deny` fails the task without side effects.

## Controls
- **Workspace confinement:** every path is resolved and must stay under `.omega/workspace`. Absolute paths, drive letters,
  UNC paths, `..` escapes, NUL bytes and symlink escapes are rejected (tested).
- **Least-destructive delete:** `fs.delete` moves to `.omega/trash/<timestamp>/`, so it is recoverable.
- **Process execution:** program allowlist; deny patterns (recursive delete, format, shutdown, registry edits, curl/wget,
  netsh, ...); shell metacharacters outside quotes refused; programs run with argv (no shell); hard timeout; output capped.
- **Emergency stop:** `python -m omega_nano stop` pauses every task before its next step. `unstop` + `resume` continues.
- **Prompt-injection posture:** tool outputs are labeled untrusted and never alter the plan (tested with a malicious file).
- **No secrets in state:** nothing reads environment secrets. Logs hold tool arguments and outputs, so don't put secrets in goals.

## Known gaps (not yet mitigated)
1. **No OS sandbox.** `python`/`node`/`npm`/`git` on the allowlist can do anything the user can. Planned: run `shell.run`
   in a restricted job object / container with no network and a workspace-only filesystem.
2. In-process tool timeouts can't forcibly kill a runaway Python thread.
3. The audit log is append-only by convention, not cryptographically chained. Planned: hash-chain the event table.
4. The LLM tier sends goal text and tool descriptions to `OMEGA_LLM_URL`. Point it only at a local or trusted endpoint.
