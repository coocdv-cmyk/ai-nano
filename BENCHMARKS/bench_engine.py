"""Agent-loop overhead benchmark: how much the control foundation costs per task and per step.

Measures, in-process on a fresh state directory:
  * end-to-end latency of a 3-step verified task (compute -> write -> verify-read), plan tier t1 (rules)
  * the same goal again via the procedural plan cache (tier t0)
  * planning time per tier, per-step overhead, checkpoint bytes per save
  * crash-recovery wall time: a subprocess is killed mid-task and a new one resumes it

Usage: python BENCHMARKS/bench_engine.py --runs 200
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omega_nano.config import load_config  # noqa: E402
from omega_nano.engine import Engine  # noqa: E402
from omega_nano.hw import detect  # noqa: E402


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[int(p * (len(xs) - 1))], 3)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "BENCHMARKS" / "results" / "engine.json"))
    a = ap.parse_args()
    eng = Engine(load_config(tempfile.mkdtemp(prefix="omega_engine_")))
    res: dict = {"benchmark": "engine-loop-v1", "date": time.strftime("%Y-%m-%d %H:%M"), "hardware": detect()}

    for tier, unique in (("t1:rules", True), ("t0:plan-cache", False)):
        e2e, plan, steps = [], [], []
        for i in range(a.runs):
            n = i if unique else 0
            goal = f"calculate 2^{100 + n} then save the result to r{n}.txt then check that file r{n}.txt contains 1"
            t0 = time.perf_counter()
            task = eng.submit(goal)
            task = eng.run(task.id)
            e2e.append((time.perf_counter() - t0) * 1000)
            assert task.status.value == "done", task.summary
            assert task.tier == tier or (not unique and i == 0), task.tier
            plan.append(task.understanding["plan_ms"])
            steps += [(s.ended_at - s.started_at) * 1000 for s in task.steps]
        res[tier] = {"runs": a.runs, "e2e_p50_ms": pct(e2e, .5), "e2e_p95_ms": pct(e2e, .95),
                     "plan_p50_ms": pct(plan, .5), "step_exec_verify_p50_ms": pct(steps, .5),
                     "tasks_per_s": round(1000 / statistics.mean(e2e), 1)}
        print(tier, res[tier], flush=True)

    rows = eng.store.conn.execute("SELECT AVG(LENGTH(state_json)), COUNT(*) FROM checkpoints").fetchone()
    res["checkpoint_avg_bytes"], res["checkpoints"] = round(rows[0]), rows[1]
    res["saves_per_3step_task"] = round(rows[1] / (2 * a.runs), 1)
    eng.close()

    home = tempfile.mkdtemp(prefix="omega_crash_")
    env = {**os.environ, "OMEGA_HOME": home, "OMEGA_FAULT": "crash_after_exec:1", "PYTHONIOENCODING": "utf-8"}
    cmd = [sys.executable, "-m", "omega_nano"]
    goal = "write file a.txt with A then write file b.txt with B then write file c.txt with C"
    t0 = time.perf_counter()
    code = subprocess.run([*cmd, "run", goal], cwd=ROOT, env=env, capture_output=True).returncode
    t1 = time.perf_counter()
    env.pop("OMEGA_FAULT")
    p = subprocess.run([*cmd, "resume"], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    t2 = time.perf_counter()
    res["crash_recovery"] = {"crash_exit_code": code, "resume_wall_ms_incl_python_startup": round((t2 - t1) * 1000, 1),
                             "run_until_crash_wall_ms": round((t1 - t0) * 1000, 1),
                             "recovered_without_reexecution": "(recovered)" in p.stdout,
                             "final_ok": "[OK]" in p.stdout}
    print("crash_recovery", res["crash_recovery"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
