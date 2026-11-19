"""EXP-008: M4 Integrated OMEGA-NANO V1 Offline Capability & Efficiency Matrix.

End-to-end CPU benchmark uniting M0..M4:
  * M0: Checkpointed workspace & bilingual tool execution + safety
  * M1/M2: OmegaCoreV2 native streaming + 2-bit packed LUT + Wake-Sleep distillation
  * M3: BlindVault + Held-out verification + SkillLibrary + L1/L3 guards + RepoGraph + Semantic Memory
  * M4: Unified OmegaAgent telemetry (solve rate, speed, RSS RAM, byte footprint, honest limits)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega_nano.agent import OmegaAgent, _rss_mb
from omega_nano.brain import BlindVault
from omega_nano.core.engine import Engine


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    t_all = time.perf_counter()

    with tempfile.TemporaryDirectory() as td:
        home = Path(td) / "omega_home"
        agent = OmegaAgent(home=home)
        try:
            # 1. Seed a real workspace module & index full package + workspace
            (agent.cfg.workspace / "app_utils.py").write_text(
                "def calculate_tax(amount, rate=0.15):\n    return round(amount * rate, 2)\n",
                encoding="utf-8",
            )
            ws_graph = agent.index_codebase(agent.cfg.workspace)
            pkg_graph = agent.index_codebase(ROOT / "omega_nano")

            # 2. End-to-end bilingual workspace execution (math -> disk SHA-256 -> verify)
            t0 = time.perf_counter()
            ws_task = agent.handle("احسب 125 * 8 ثم احفظ النتيجة في out.txt")
            ws_read = agent.handle("check that file out.txt contains 1000")
            ws_ms = (time.perf_counter() - t0) * 1000.0

            # 3. Live coding problem solved & verified inside OmegaAgent
            live_code = agent.handle(
                "Write Python function count_even(nums: list[int]) -> int such that [1, 2, 4, 5] -> 2.\n"
                "tests:\nassert count_even([1, 2, 4, 5]) == 2\n"
                "assert count_even([1, 3, 5]) == 0\n"
            )
            # 4. BlindVault (5 math + 5 code) & Independent Held-Out (10 unseen code)
            t0 = time.perf_counter()
            blind_rep = BlindVault().evaluate(lambda p: agent.solve_problem(p, max_attempts=5))
            blind_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            held_rep = BlindVault.heldout().evaluate(lambda p: agent.solve_problem(p, max_attempts=5))
            held_ms = (time.perf_counter() - t0) * 1000.0

            # 5. Wake phase -> Sleep phase: distill verified replay traces into tiny ternary v2 core
            sleep_stats = agent.sleep(budget_s=10.0, steps_cap=40)

            # 6. Native CPU speed & packed memory of distilled v2 core
            eng_v2 = Engine.load(agent.core_checkpoint, packed=True)
            v2_bench = eng_v2.bench(n_new=240, batch=4, prompt=b"<SPEC>\nWrite Python function ")

            # 7. Bilingual semantic recall on indexed + conversational facts
            agent.handle("سجل عندك ان قاعدة البيانات المحلية تستخدم ملف sqlite")
            mem_hits = agent.engine.memory.recall("where is local database stored?", kind="semantic", k=3)
            proj_hits = agent.engine.memory.recall("calculate_tax", kind="project", k=3)

            st = agent.status()
            matrix = {
                "exp_id": "EXP-008",
                "title": "M4 Integrated OMEGA-NANO V1 Offline Capability & Efficiency Matrix",
                "status_snapshot": st,
                "workspace_agent": {
                    "write_ok": ws_task["ok"],
                    "verify_ok": ws_read["ok"],
                    "tier": ws_task["tier"],
                    "elapsed_ms": round(ws_ms, 2),
                },
                "live_code_ast_io": {
                    "passed": live_code["ok"],
                    "verified": live_code["verified"],
                    "attempts": live_code.get("attempts", 0),
                },
                "blind_vault": {
                    "passed": blind_rep["passed"],
                    "total": blind_rep["total"],
                    "pass_rate": blind_rep["pass_rate"],
                    "elapsed_ms": round(blind_ms, 2),
                },
                "heldout_code": {
                    "passed": held_rep["passed"],
                    "total": held_rep["total"],
                    "pass_rate": held_rep["pass_rate"],
                    "elapsed_ms": round(held_ms, 2),
                },
                "wake_sleep_v2": sleep_stats,
                "native_v2_runtime": v2_bench,
                "repo_graph": {
                    "workspace": ws_graph,
                    "omega_nano": pkg_graph,
                    "repo_recall_hit": bool(proj_hits and "calculate_tax" in proj_hits[0].text),
                },
                "bilingual_memory_recall_hit": bool(mem_hits and "sqlite" in mem_hits[0].text.lower()),
                "peak_rss_mb": _rss_mb(),
                "total_wall_seconds": round(time.perf_counter() - t_all, 2),
            }
        finally:
            agent.close()

    (out_dir / "results.json").write_text(json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(matrix, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
