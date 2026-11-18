"""EXP-006: M3 Execution-Guided Neuro-Symbolic & Bilingual Semantic Brain Benchmark.

Measures on CPU (no external APIs, no hardcoded BlindVault problem IDs):
  1. Baseline SymbolicTacticSolver vs M3 NeuroSymbolicSolver on BlindVault (5 math + 5 code).
  2. M3 generalization on independent BlindVault.heldout() coding suite (10 distinct problems).
  3. Diverse self-play Cycle 1 -> Cycle 2 solve rate & attempt reduction + cross-name PolicyRanker persistence.
  4. L3 behavioral guard + L1 SemanticDriftMonitor detection & self-healing under spec drift.
  5. RepoGraph AST indexing speed & semantic code recall over `omega_nano/`.
  6. True Arabic synonym, Arabic<->English cross-lingual, and code identifier semantic recall.
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

from omega_nano.brain import (
    BlindProblem,
    BlindVault,
    CodeVerifier,
    NeuroSymbolicSolver,
    SelfPlayEngine,
    SkillLibrary,
    SymbolicTacticSolver,
    index_repository,
)
from omega_nano.memory import MemoryFabric
from omega_nano.semantic import SemanticDriftMonitor


def bench_blind_and_heldout(tmp: Path) -> dict:
    # 1. Baseline SymbolicTacticSolver (fresh isolated DB)
    m_base = MemoryFabric(tmp / "base_mem.db")
    s_base = SkillLibrary(tmp / "base_brain.db", memory=m_base)
    sp_base = SelfPlayEngine(s_base)
    base_solver = SymbolicTacticSolver()
    t0 = time.perf_counter()
    base_rep = BlindVault().evaluate(lambda p: sp_base.solve(p, base_solver, max_attempts=5))
    base_ms = (time.perf_counter() - t0) * 1000.0
    s_base.close()
    m_base.close()

    # 2. M3 NeuroSymbolicSolver (fresh isolated DB)
    m_m3 = MemoryFabric(tmp / "m3_mem.db")
    s_m3 = SkillLibrary(tmp / "m3_brain.db", memory=m_m3)
    sp_m3 = SelfPlayEngine(s_m3)
    m3_solver = NeuroSymbolicSolver()
    t0 = time.perf_counter()
    m3_rep = BlindVault().evaluate(lambda p: sp_m3.solve(p, m3_solver, max_attempts=5))
    m3_ms = (time.perf_counter() - t0) * 1000.0

    # 3. Independent held-out 10 coding tasks (no BlindVault overlap)
    t0 = time.perf_counter()
    held_rep = BlindVault.heldout().evaluate(lambda p: sp_m3.solve(p, m3_solver, max_attempts=5))
    held_ms = (time.perf_counter() - t0) * 1000.0
    s_m3.close()
    m_m3.close()

    return {
        "baseline_blind": {
            "passed": base_rep["passed"],
            "total": base_rep["total"],
            "pass_rate": base_rep["pass_rate"],
            "math_passed": sum(1 for r in base_rep["results"] if r["domain"] == "math" and r["passed"]),
            "code_passed": sum(1 for r in base_rep["results"] if r["domain"] == "code" and r["passed"]),
            "elapsed_ms": round(base_ms, 2),
        },
        "m3_blind": {
            "passed": m3_rep["passed"],
            "total": m3_rep["total"],
            "pass_rate": m3_rep["pass_rate"],
            "math_passed": sum(1 for r in m3_rep["results"] if r["domain"] == "math" and r["passed"]),
            "code_passed": sum(1 for r in m3_rep["results"] if r["domain"] == "code" and r["passed"]),
            "avg_attempts": round(sum(r["attempts"] for r in m3_rep["results"]) / m3_rep["total"], 2),
            "elapsed_ms": round(m3_ms, 2),
        },
        "m3_heldout": {
            "passed": held_rep["passed"],
            "total": held_rep["total"],
            "pass_rate": held_rep["pass_rate"],
            "avg_attempts": round(sum(r["attempts"] for r in held_rep["results"]) / held_rep["total"], 2),
            "elapsed_ms": round(held_ms, 2),
            "details": [
                {"id": r["id"], "passed": r["passed"], "attempts": r["attempts"], "source": r["source"]}
                for r in held_rep["results"]
            ],
        },
    }


def bench_selfplay_and_policy(tmp: Path) -> dict:
    mem = MemoryFabric(tmp / "sp_mem.db")
    db = tmp / "sp_brain.db"
    skills = SkillLibrary(db, memory=mem)
    sp = SelfPlayEngine(skills)
    solver = NeuroSymbolicSolver()

    c1 = sp.run_cycle(cycle=1, solver=solver, n_tasks=12, level=2, max_attempts=4)
    c2 = sp.run_cycle(cycle=2, solver=solver, n_tasks=12, level=2, max_attempts=4)

    # Cross-name policy transfer across fresh DB reopen
    p_db = tmp / "pol.db"
    s1 = SkillLibrary(p_db)
    sp1 = SelfPlayEngine(s1)
    rpn_v1 = BlindProblem(
        id="exp-rpn-1", domain="code", title="Stack RPN V1", difficulty=1,
        statement="Write Python function run_rpn_alpha(tokens: list[str]) -> int for Reverse Polish Notation (+ - * /).",
        verify=lambda code: CodeVerifier.verify_python(
            code,
            "assert run_rpn_alpha(['-7', '2', '/']) == -3\nassert run_rpn_alpha(['2', '1', '+', '3', '*']) == 9\n",
        ),
    )
    res_cold = sp1.solve(rpn_v1, NeuroSymbolicSolver(), max_attempts=5)
    policy_feats = s1.stats()["policy_features"]
    s1.close()

    s2 = SkillLibrary(p_db)
    sp2 = SelfPlayEngine(s2)
    rpn_v2 = BlindProblem(
        id="exp-rpn-2", domain="code", title="Stack RPN Unseen Name", difficulty=1,
        statement="Write Python function eval_postfix_beta(tokens: list[str]) -> int for Reverse Polish Notation (+ - * /).",
        verify=lambda code: CodeVerifier.verify_python(
            code,
            "assert eval_postfix_beta(['-7', '2', '/']) == -3\nassert eval_postfix_beta(['4', '13', '5', '/', '+']) == 6\n",
        ),
    )
    res_warm = sp2.solve(rpn_v2, NeuroSymbolicSolver(), max_attempts=5)
    s2.close()
    skills.close()
    mem.close()

    return {
        "cycle_1": c1,
        "cycle_2": c2,
        "policy_transfer": {
            "cold_attempts": res_cold["attempts"],
            "warm_unseen_name_attempts": res_warm["attempts"],
            "warm_passed": res_warm["passed"],
            "policy_features_persisted": policy_feats,
        },
    }


def bench_drift_and_l3_guard(tmp: Path) -> dict:
    skills = SkillLibrary(tmp / "drift_brain.db")
    monitor = SemanticDriftMonitor(window=6, baseline_rate=0.9, threshold=1.4)
    sp = SelfPlayEngine(skills, drift_monitor=monitor)
    solver = NeuroSymbolicSolver()

    records = []
    for idx in range(3):
        fn = f"merge_exp_{idx}"
        stale = (
            f"def {fn}(intervals):\n"
            "    if not intervals: return []\n"
            "    pts = sorted(intervals)\n"
            "    out = [list(pts[0])]\n"
            "    for s, e in pts[1:]:\n"
            "        if s < out[-1][1]: out[-1][1] = max(out[-1][1], e)\n"
            "        else: out.append([s, e])\n"
            "    return out\n"
        )
        skills.register_skill(f"code_{fn}", "code", rf"\b{fn}\b", {"kind": "code_template", "code": stale}, "PROVEN")
        shifted = BlindProblem(
            id=f"drift-exp-{idx}", domain="code", title=f"Shift {idx}", difficulty=1,
            statement=f"Write Python function {fn}(intervals: list[list[int]]) -> list[list[int]] merging overlapping/touching intervals.",
            verify=lambda code, F=fn: CodeVerifier.verify_python(
                code,
                f"assert {F}([[1, 3], [3, 5]]) == [[1, 5]]\n"
                f"assert {F}([[1, 2], [4, 6]]) == [[1, 2], [4, 6]]\n",
            ),
        )
        res = sp.solve(shifted, solver, max_attempts=4)
        records.append({
            "step": idx + 1,
            "l3_rejected_stale_skill": res.get("l3_rejected_skill", False),
            "repaired_and_passed": res["passed"],
            "drift_flagged": res["drift"],
        })

    recheck = sp.solve(
        BlindProblem(
            id="drift-healed", domain="code", title="Healed Recheck", difficulty=1,
            statement="Write Python function merge_exp_0(intervals: list[list[int]]) -> list[list[int]] merging overlapping/touching intervals.",
            verify=lambda code: CodeVerifier.verify_python(code, "assert merge_exp_0([[1, 3], [3, 5]]) == [[1, 5]]\n"),
        ),
        solver, max_attempts=4,
    )
    skills.close()
    return {
        "steps": records,
        "any_drift_flagged": any(r["drift_flagged"] for r in records),
        "all_self_healed": all(r["repaired_and_passed"] for r in records),
        "post_heal_attempts": recheck["attempts"],
        "post_heal_source": recheck["source"],
    }


def bench_repo_and_semantic_memory(tmp: Path) -> dict:
    mem = MemoryFabric(tmp / "repo_mem.db")
    sem_mem = MemoryFabric(tmp / "sem_mem.db")
    try:
        graph = index_repository(ROOT / "omega_nano", memory=mem)
        summary = graph.summarize()

        # Inject a snake_case symbol into the 374-symbol repo index and retrieve via natural language
        mem.remember("def fetch_user_profile_by_id(user_id): reads cache", kind="project", key="fn_profile")

        t0 = time.perf_counter()
        sym_hits = mem.recall("ExecutionGuidedSynthesizer", kind="project", k=3)
        repo_recall_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        id_hits = mem.recall("user profile lookup", kind="project", k=3)
        id_ms = (time.perf_counter() - t0) * 1000.0

        # Bilingual semantic recall probes on isolated memory with competing distractors
        sem_mem.remember("صيانة المحرك تحتاج تغيير زيت بانتظام", kind="semantic", key="engine_care")
        sem_mem.remember("يتم حفظ نسخة الداتا الاحتياطية على الخادم المحلي", kind="semantic", key="backup_loc")
        sem_mem.remember("كلمة المرور والرموز السرية محفوظة في متغيرات البيئة", kind="semantic", key="secrets_cfg")
        sem_mem.remember("درجة غليان الماء تساوي مئة مئوية", kind="semantic", key="water_boil")

        probes = [
            ("تصليح موتور السيارة", "engine_care", "arabic_true_synonym", sem_mem),
            ("where is backup data stored?", "backup_loc", "crosslingual_en_to_ar", sem_mem),
            ("فين مكان حفظ كلمات السر؟", "secrets_cfg", "arabic_paraphrase_secrets", sem_mem),
            ("user profile lookup", "fn_profile", "code_identifier_split_in_374_symbols", mem),
        ]
        probe_results = []
        hits_top3 = 0
        for q, want_key, tag, store in probes:
            t0 = time.perf_counter()
            recs = store.recall(q, k=3)
            ms = (time.perf_counter() - t0) * 1000.0
            matched = any(r.key == want_key for r in recs)
            if matched:
                hits_top3 += 1
            probe_results.append({"probe": tag, "top_keys": [r.key for r in recs], "hit_top3": matched, "ms": round(ms, 2)})

        return {
            "repo_graph": summary,
            "repo_recall_top_hit": sym_hits[0].text[:100] if sym_hits else None,
            "repo_recall_ms": round(repo_recall_ms, 2),
            "identifier_split_in_repo_hit": any(r.key == "fn_profile" for r in id_hits),
            "identifier_split_ms": round(id_ms, 2),
            "semantic_probes_passed": hits_top3,
            "semantic_probes_total": len(probes),
            "semantic_probes": probe_results,
        }
    finally:
        mem.close()
        sem_mem.close()


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    t_all = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        results = {
            "exp_id": "EXP-006",
            "title": "M3 Execution-Guided Neuro-Symbolic & Bilingual Semantic Brain",
            "blind_and_heldout": bench_blind_and_heldout(tmp),
            "selfplay_and_policy": bench_selfplay_and_policy(tmp),
            "l3_guard_and_l1_drift": bench_drift_and_l3_guard(tmp),
            "repo_and_semantic_memory": bench_repo_and_semantic_memory(tmp),
        }
    results["total_wall_seconds"] = round(time.perf_counter() - t_all, 2)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
