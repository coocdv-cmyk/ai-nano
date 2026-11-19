"""Command-line interface: `python -m omega_nano <command>`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import load_config
from .engine import Engine
from .model import TaskStatus
from .report import render

ICON = {"done": "OK", "failed": "FAIL", "blocked": "BLOCKED", "awaiting_approval": "APPROVAL?",
        "paused": "PAUSED", "running": "RUNNING", "planned": "PLANNED"}


def _print_task(task, verbose: bool = True) -> None:
    print(f"[{ICON.get(task.status.value, task.status.value)}] task {task.id}  tier={task.tier}  mode={task.mode}")
    if verbose:
        for s in task.steps:
            mark = {"done": "+", "failed": "x", "awaiting_approval": "?", "started": "~",
                    "denied": "-"}.get(s.status.value, " ")
            extra = ""
            o = s.output or {}
            for k in ("value", "answer", "completion", "path", "count", "exit_code"):
                if o.get(k) is not None:
                    extra = f"{k}={str(o[k])[:100]}"
                    break
            print(f"  [{mark}] {s.idx}. {s.tool:<16} {extra}{'  (recovered)' if s.recovered else ''}")
    if task.summary:
        print(f"  {task.summary}")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="omega", description="OMEGA-NANO local resumable agent")
    ap.add_argument("--home", help="state directory (default: ./.omega or $OMEGA_HOME)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="submit and execute a task")
    r.add_argument("goal")
    r.add_argument("--mode", default="workspace", choices=["read_only", "ask", "workspace", "autonomous"])
    r.add_argument("--key", help="idempotency key: re-delivering the same key never runs twice")
    r.add_argument("--plan", help="JSON file with an explicit plan [{tool,args,expect,why}]")
    r.add_argument("--report", action="store_true", help="print the full report afterwards")

    sub.add_parser("resume", help="resume every interrupted task")
    a = sub.add_parser("approve", help="approve the pending step of a task")
    a.add_argument("task_id")
    d = sub.add_parser("deny", help="deny the pending step of a task")
    d.add_argument("task_id")
    st = sub.add_parser("status", help="show one task, or list tasks")
    st.add_argument("task_id", nargs="?")
    rp = sub.add_parser("report", help="full markdown report for a task")
    rp.add_argument("task_id")
    rp.add_argument("--out", help="write the report to this file")
    sub.add_parser("stop", help="emergency stop: pause all tasks before their next step")
    sub.add_parser("unstop", help="clear the emergency stop")
    sub.add_parser("tools", help="list tools, scopes and parameters")
    sub.add_parser("doctor", help="hardware, environment and state health")

    m = sub.add_parser("mem", help="memory fabric")
    msub = m.add_subparsers(dest="mcmd", required=True)
    ma = msub.add_parser("add")
    ma.add_argument("text")
    ma.add_argument("--kind", default="semantic")
    ma.add_argument("--key")
    ma.add_argument("--source", default="cli")
    ms = msub.add_parser("search")
    ms.add_argument("query")
    ms.add_argument("-k", type=int, default=5)
    ms.add_argument("--kind")
    ms.add_argument("--history", action="store_true")
    mf = msub.add_parser("forget")
    mf.add_argument("id", type=int)
    mh = msub.add_parser("history")
    mh.add_argument("key")
    msub.add_parser("stats")

    b = sub.add_parser("brain", help="self-evolving verifier search, skill compiler, and blind evaluation")
    bsub = b.add_subparsers(dest="bcmd", required=True)
    be = bsub.add_parser("eval", help="run sealed blind math+code evaluation")
    be.add_argument("--domain", choices=["math", "code"])
    be.add_argument("--attempts", type=int, default=4)
    be.add_argument("--heldout", action="store_true", help="evaluate on independent held-out coding suite")
    bs = bsub.add_parser("selfplay", help="run self-play curriculum cycles")
    bs.add_argument("--cycles", type=int, default=2)
    bs.add_argument("--tasks", type=int, default=8)
    bs.add_argument("--level", type=int, default=2)
    bir = bsub.add_parser("index-repo", help="index Python repository symbols and call graph into memory")
    bir.add_argument("--path", default=".", help="repository or module path to index")
    bir.add_argument("--no-memory", action="store_true", help="skip persisting indexed symbols into MemoryFabric")
    brp = bsub.add_parser("repair", help="execution-verified AST repair of a failing Python workspace")
    brp.add_argument("--path", default=".", help="workspace root containing the failing tests")
    brp.add_argument("--tests", default="tests", help="test directory or node path to run (default: tests)")
    brp.add_argument("--max-calls", type=int, default=80, help="verifier-call budget (default: 80)")
    brp.add_argument("--dry-run", action="store_true", help="search but do not write the accepted patch")
    brp.add_argument("--attempts", dest="repair_attempts", type=int, default=3, help="maximum composed edits (default: 3)")
    bd = bsub.add_parser("distill", help="run Wake-Sleep replay distillation into compact v2 core")
    bd.add_argument("--seconds", type=float, default=6.0)
    bd.add_argument("--steps", type=int, default=80)
    bsub.add_parser("stats", help="show compiled skill & replay buffer statistics")

    ag = sub.add_parser("agent", help="run unified OMEGA-NANO V1 offline agent on a goal")
    ag.add_argument("goal")
    ag.add_argument("--mode", default="workspace", choices=["read_only", "ask", "workspace", "autonomous"])
    ag.add_argument("--index", action="store_true", help="index workspace Python symbols before running")

    args = ap.parse_args(argv)
    cfg = load_config(args.home)
    eng = Engine(cfg)
    try:
        return _dispatch(args, eng, cfg)
    finally:
        eng.close()


def _dispatch(args, eng: Engine, cfg) -> int:
    if args.cmd == "run":
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8")) if args.plan else None
        t0 = time.perf_counter()
        task = eng.submit(args.goal, mode=args.mode, idem_key=args.key, plan=plan)
        if task.status == TaskStatus.PLANNED:
            task = eng.run(task.id)
        _print_task(task)
        print(f"  elapsed {round((time.perf_counter() - t0) * 1000, 1)} ms · state in {cfg.home}")
        if task.status == TaskStatus.AWAITING_APPROVAL:
            print(f"  -> approve with: python -m omega_nano approve {task.id}")
        if args.report:
            print(render(task, eng.store))
        return 0 if task.status == TaskStatus.DONE else 2
    if args.cmd == "resume":
        tasks = eng.resume_all()
        if not tasks:
            print("nothing to resume")
        for t in tasks:
            _print_task(t)
        return 0
    if args.cmd in ("approve", "deny"):
        task = eng.approve(args.task_id, deny=args.cmd == "deny")
        _print_task(task)
        return 0 if task.status == TaskStatus.DONE else 2
    if args.cmd == "status":
        if args.task_id:
            _print_task(eng.store.load(args.task_id)[0])
        else:
            for row in eng.store.list_tasks()[:50]:
                print(f"{row['id']}  {row['status']:<18} {row['goal'][:80]}")
        return 0
    if args.cmd == "report":
        text = render(eng.store.load(args.task_id)[0], eng.store)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"report written to {args.out}")
        else:
            print(text)
        return 0
    if args.cmd == "stop":
        eng.stop()
        print("emergency stop set: running tasks pause before their next step")
        return 0
    if args.cmd == "unstop":
        eng.clear_stop()
        print("emergency stop cleared; run `resume` to continue paused tasks")
        return 0
    if args.cmd == "tools":
        for t in eng.tools.all():
            d = t.describe()
            params = ", ".join(f"{k}:{v['type']}{'' if v['required'] else '?'}" for k, v in d["params"].items())
            print(f"{d['name']:<16} scopes={','.join(d['scopes']) or '-':<12} idempotent={d['idempotent']!s:<5} ({params})")
            print(f"{'':16} {d['description']}")
        return 0
    if args.cmd == "doctor":
        from .hw import detect
        print(json.dumps({"hardware": detect(), "home": str(cfg.home), "workspace": str(cfg.workspace),
                          "llm_backend": cfg.llm_url or None, "emergency_stop": cfg.stop_file.exists(),
                          "tasks": {s.value: len(eng.store.list_tasks({s})) for s in TaskStatus},
                          "memory": eng.memory.stats()}, indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "mem":
        mem = eng.memory
        if args.mcmd == "add":
            print(json.dumps(mem.remember(args.text, kind=args.kind, key=args.key, source=args.source)))
        elif args.mcmd == "search":
            t0 = time.perf_counter()
            hits = mem.recall(args.query, k=args.k, kind=args.kind, include_history=args.history)
            ms = round((time.perf_counter() - t0) * 1000, 2)
            for h in hits:
                hist = "" if h.valid_to is None else "  [superseded]"
                print(f"#{h.id} score={h.score:.4f} cos={h.cosine:.3f} lex={h.lexical_rank} vec={h.vector_rank} "
                      f"conf={h.confidence} src={h.source}{hist}\n    {h.text}")
            print(f"({len(hits)} hits in {ms} ms)")
        elif args.mcmd == "forget":
            print("deleted" if mem.forget(args.id) else "no such memory")
        elif args.mcmd == "history":
            print(json.dumps(mem.history(args.key), indent=2, ensure_ascii=False))
        elif args.mcmd == "stats":
            print(json.dumps(mem.stats(), indent=2))
        return 0
    if args.cmd == "brain":
        from .brain import BlindVault, NeuroSymbolicSolver, SelfPlayEngine, SkillLibrary, index_repository

        if args.bcmd == "index-repo":
            graph = index_repository(
                Path(args.path),
                memory=None if args.no_memory else eng.memory,
            )
            print(json.dumps(graph.summarize(), indent=2, ensure_ascii=False))
            return 0

        if args.bcmd == "repair":
            from .brain import RepairEngine

            report = RepairEngine(
                tests_dir=args.tests,
                max_verifier_calls=args.max_calls,
                max_depth=args.repair_attempts,
            ).repair(args.path, apply=not args.dry_run)
            print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False, default=str))
            return 0 if report.solved else 2

        skills = SkillLibrary(cfg.home / "brain.db", memory=eng.memory)
        try:
            sp = SelfPlayEngine(skills)
            ckpt = cfg.home / "core_v2.pt"
            from .core.distill import NeuralCodeProposer, sleep_distill
            solver = NeuroSymbolicSolver(neural_proposer=NeuralCodeProposer(ckpt))
            if args.bcmd == "eval":
                vault = BlindVault.heldout() if args.heldout else BlindVault()
                report = vault.evaluate(
                    lambda p: sp.solve(p, solver, max_attempts=args.attempts), domain=args.domain
                )
                print(json.dumps(report, indent=2, ensure_ascii=False))
            elif args.bcmd == "selfplay":
                cycles = [sp.run_cycle(c, solver, n_tasks=args.tasks, level=args.level) for c in range(1, args.cycles + 1)]
                print(json.dumps(cycles, indent=2, ensure_ascii=False))
            elif args.bcmd == "distill":
                res = sleep_distill(
                    ckpt,
                    checkpoint_in=ckpt if ckpt.is_file() else None,
                    skills=skills,
                    memory=eng.memory,
                    budget_s=args.seconds,
                    steps_cap=args.steps,
                )
                print(json.dumps(res, indent=2, ensure_ascii=False))
            elif args.bcmd == "stats":
                print(json.dumps(skills.stats(), indent=2))
        finally:
            skills.close()
        return 0
    if args.cmd == "agent":
        from .agent import OmegaAgent

        agent = OmegaAgent(cfg=cfg)
        try:
            if args.index:
                agent.index_codebase(cfg.workspace)
            print(json.dumps(agent.handle(args.goal, mode=args.mode), indent=2, ensure_ascii=False, default=str))
        finally:
            agent.close()
        return 0
    return 1
