"""EXP-004: Engine ring-KV speedup, OmegaCore v1 (looped + ACT + MTP) vs v0 ablation, and Self-Play Brain.

Usage: python EXPERIMENTS/EXP-004-looped-mind-and-speed/run_exp.py --seconds 45
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from omega_nano.brain import BlindVault, SelfPlayEngine, SkillLibrary  # noqa: E402
from omega_nano.brain.evolver import SymbolicTacticSolver  # noqa: E402
from omega_nano.core.data import batches, bits_per_byte, load  # noqa: E402
from omega_nano.core.engine import Engine  # noqa: E402
from omega_nano.core.model import CoreConfig, build_model  # noqa: E402
from omega_nano.hw import detect  # noqa: E402

DATA = ROOT / ".omega" / "core_data"


@torch.no_grad()
def evaluate_bpb(model: torch.nn.Module, val: np.ndarray, ctx: int, n_batches: int = 32, batch: int = 16) -> float:
    model.eval()
    rng = np.random.default_rng(1234)
    it = batches(val, batch, ctx, rng)
    tot = 0.0
    for _ in range(n_batches):
        x, y = next(it)
        _, loss = model(torch.from_numpy(x), torch.from_numpy(y))
        tot += loss.item()
    model.train()
    return bits_per_byte(tot / n_batches)


def train_arm(name: str, cfg: CoreConfig, train: np.ndarray, val: np.ndarray, seconds: float,
              batch: int = 16, lr: float = 2e-3, threads: int = 4) -> dict:
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    rng = np.random.default_rng(0)
    it = batches(train, batch, cfg.ctx, rng)
    t0 = time.perf_counter()
    step = 0
    warmup = 25
    while True:
        elapsed = time.perf_counter() - t0
        if elapsed >= seconds:
            break
        frac = elapsed / seconds
        cur_lr = lr * min(1.0, (step + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        x, y = next(it)
        _, loss = model(torch.from_numpy(x), torch.from_numpy(y))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
    total = time.perf_counter() - t0
    val_bpb = evaluate_bpb(model, val, cfg.ctx)
    out = {
        "name": name,
        "config": cfg.to_dict(),
        "params": model.n_params(),
        "steps": step,
        "seconds": round(total, 2),
        "train_bytes": step * batch * cfg.ctx,
        "val_bpb": round(val_bpb, 4),
    }
    if hasattr(model, "last_avg_loops"):
        out["eval_avg_loops"] = round(model.last_avg_loops, 3)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=45.0, help="wall-clock budget per model arm")
    a = ap.parse_args()

    out: dict = {"experiment": "EXP-004-looped-mind-and-speed", "date": time.strftime("%Y-%m-%d %H:%M"),
                 "hardware": detect()}

    ckpt = ROOT / "EXPERIMENTS" / "EXP-003-omega-core-v0" / "runs" / "ternary" / "model.pt"
    if ckpt.is_file():
        eng = Engine.load(ckpt)
        b14 = eng.bench(n_new=2000, batch=14)
        b28 = eng.bench(n_new=2000, batch=28)
        out["engine_speed_ternary_v0"] = {**b14, "aggregate_tok_s_batch28": b28["aggregate_tok_s_batch28"]}
        print("Engine speed:", out["engine_speed_ternary_v0"], flush=True)

    if (DATA / "train.bin").exists():
        train, val = load(DATA)
        arms = [
            ("v0_4layer_fp32", CoreConfig(d=128, layers=4, heads=4, hidden=352, ctx=128, arch="v0")),
            ("v1_looped_half_weights", CoreConfig(d=128, layers=2, heads=4, hidden=352, ctx=128,
                                                  arch="v1", n_loops=3, mtp_k=3, ponder_weight=0.005)),
        ]
        out["core_ablation"] = []
        for name, cfg in arms:
            res = train_arm(name, cfg, train, val, seconds=a.seconds)
            out["core_ablation"].append(res)
            print("Arm:", res["name"], "params=", res["params"]["total"],
                  "steps=", res["steps"], "val_bpb=", res["val_bpb"], flush=True)

    with tempfile.TemporaryDirectory(prefix="omega_exp4_") as tmp:
        skills = SkillLibrary(Path(tmp) / "brain.db")
        sp = SelfPlayEngine(skills)
        solver = SymbolicTacticSolver()
        c1 = sp.run_cycle(1, solver, n_tasks=8, level=2, max_attempts=3)
        c2 = sp.run_cycle(2, solver, n_tasks=8, level=2, max_attempts=3)
        blind = BlindVault().evaluate(lambda p: sp.solve(p, solver, max_attempts=3))
        out["brain_selfplay_and_blind"] = {"cycle_1": c1, "cycle_2": c2, "blind_eval": blind}
        print("Brain self-play:", c1["solve_rate"], "->", c2["solve_rate"],
              "avg_attempts:", c1["avg_attempts"], "->", c2["avg_attempts"],
              "blind:", blind["passed"], "/", blind["total"], flush=True)
        skills.close()

    res_path = Path(__file__).with_name("results.json")
    res_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", res_path)


if __name__ == "__main__":
    main()
