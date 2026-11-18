"""Transparent execution report: what was understood, planned, done, verified, and what remains uncertain."""

from __future__ import annotations

import json
import time

from .model import StepStatus, Task, TaskStatus
from .store import Store


def _ts(t: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) if t else "-"


def render(task: Task, store: Store) -> str:
    u = task.understanding
    lines = [
        f"# Task {task.id} — {task.status.value.upper()}",
        "",
        f"**Goal:** {task.goal}",
        f"**Mode:** `{task.mode}` · **Plan tier:** `{task.tier}` · **Reasoner:** `{u.get('reasoner')}` · "
        f"**Planning time:** {u.get('plan_ms')} ms · **Revisions:** {task.revisions}",
        f"**Summary:** {task.summary or '-'}",
        "",
        "## Understanding",
        f"- Language: {u.get('language', '-')}; confidence: {u.get('confidence')}",
    ]
    for c in u.get("unparsed", []):
        lines.append(f"- Not understood: `{c}`")
    lines += ["", "## Steps", "", "| # | Tool | Status | Attempts | Verification | Time (ms) |",
              "|---|---|---|---|---|---|"]
    for s in task.steps:
        ver = "; ".join(("PASS " if v["passed"] else "FAIL ") + v["detail"] for v in s.verification) or "-"
        dur = round((s.ended_at - s.started_at) * 1000, 2) if s.started_at and s.ended_at else "-"
        status = s.status.value + (" (recovered)" if s.recovered else "")
        lines.append(f"| {s.idx} | `{s.tool}` | {status} | {s.attempts} | {ver.replace('|', '/')} | {dur} |")
    lines += ["", "## Step detail", ""]
    for s in task.steps:
        lines.append(f"### {s.idx}. `{s.tool}` — {s.why}")
        lines.append(f"- Args: `{json.dumps(s.args, ensure_ascii=False)[:300]}`")
        if s.output is not None:
            lines.append(f"- Output: `{json.dumps(s.output, ensure_ascii=False)[:500]}`")
        if s.error:
            lines.append(f"- Error: {s.error}")
        lines.append("")
    uncertain = [s for s in task.steps if s.status != StepStatus.DONE]
    lines += ["## Uncertain / not done", ""]
    if task.status == TaskStatus.DONE and not uncertain:
        lines.append("Nothing: every step's postconditions were independently verified.")
    else:
        for s in uncertain:
            lines.append(f"- Step {s.idx} `{s.tool}` is `{s.status.value}`" + (f": {s.error}" if s.error else ""))
        if not uncertain:
            lines.append(f"- Task status is `{task.status.value}`.")
    lines += ["", "## Audit log", ""]
    for e in store.events(task.id):
        lines.append(f"- {_ts(e['ts'])} `{e['kind']}` {json.dumps(e['data'], ensure_ascii=False)[:200]}")
    return "\n".join(lines) + "\n"
