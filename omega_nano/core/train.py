"""Train an OmegaCore arm on CPU under a wall-clock budget.

python -m omega_nano.core.train --arm ternary --minutes 20
Writes runs/<arm>/model.pt and runs/<arm>/log.json (loss curve, val bits/byte, timing).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .data import baselines, batches, bits_per_byte, build_corpus, load
from .model import CoreConfig, build_model

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / ".omega" / "core_data"
RUNS = ROOT / "EXPERIMENTS" / "EXP-003-omega-core-v0" / "runs"


@torch.no_grad()
def evaluate(model: OmegaCore, val: np.ndarray, ctx: int, n_batches: int = 24, batch: int = 16) -> float:
    model.eval()
    rng = np.random.default_rng(1234)  # identical eval windows for every arm and every eval
    it = batches(val, batch, ctx, rng)
    tot = 0.0
    for _ in range(n_batches):
        x, y = next(it)
        _, loss = model(torch.from_numpy(x), torch.from_numpy(y))
        tot += loss.item()
    model.train()
    return bits_per_byte(tot / n_batches)


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["fp32", "ternary"], required=True)
    ap.add_argument("--arch", choices=["v0", "v1", "v2"], default="v0")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=704)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--n-loops", type=int, default=1)
    ap.add_argument("--mtp-k", type=int, default=1)
    ap.add_argument("--patch-size", type=int, default=4)
    ap.add_argument("--no-copy-gate", action="store_true")
    ap.add_argument("--no-fast-weights", action="store_true")
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=100)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    if not (DATA / "train.bin").exists():
        print("building corpus:", build_corpus(DATA), flush=True)
    train, val = load(DATA)
    base_file = DATA / "baselines.json"
    if not base_file.exists():
        base_file.write_text(json.dumps(baselines(train, val), indent=2), encoding="utf-8")

    cfg = CoreConfig(
        d=a.d, layers=a.layers, hidden=a.hidden, ctx=a.ctx, ternary=a.arm == "ternary",
        arch=a.arch, n_loops=a.n_loops, mtp_k=a.mtp_k,
        patch_size=a.patch_size, use_copy_gate=not a.no_copy_gate,
        use_fast_weights=not a.no_fast_weights,
    )
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.05)
    out = RUNS / (a.out_name or a.arm)
    out.mkdir(parents=True, exist_ok=True)
    log = {"arm": a.arm, "config": cfg.to_dict(), "params": model.n_params(), "args": vars(a),
           "baselines_bpb": json.loads(base_file.read_text()), "curve": []}
    print(f"[{a.arm}] params {log['params']}  budget {a.minutes} min", flush=True)

    budget = a.minutes * 60
    warmup = 50
    rng = np.random.default_rng(a.seed)
    it = batches(train, a.batch, cfg.ctx, rng)
    t0 = time.perf_counter()
    step = 0
    while True:
        elapsed = time.perf_counter() - t0
        if elapsed >= budget:
            break
        # LR schedule over wall-clock time so both arms get the same budget, not the same step count.
        frac = elapsed / budget
        lr = a.lr * min(1.0, (step + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = next(it)
        _, loss = model(torch.from_numpy(x), torch.from_numpy(y))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % a.eval_every == 0:
            vb = evaluate(model, val, cfg.ctx)
            pt = {"step": step, "minutes": round((time.perf_counter() - t0) / 60, 2), "train_bpb":
                  round(bits_per_byte(loss.item()), 4), "val_bpb": round(vb, 4), "lr": round(lr, 6)}
            log["curve"].append(pt)
            print(pt, flush=True)
            torch.save({"config": cfg.to_dict(), "state": model.state_dict()}, out / "model.pt")
            (out / "log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    vb = evaluate(model, val, cfg.ctx, n_batches=64)
    total = time.perf_counter() - t0
    log["final"] = {"steps": step, "minutes": round(total / 60, 2), "val_bpb": round(vb, 4),
                    "train_tokens": step * a.batch * cfg.ctx,
                    "tokens_per_s_training": round(step * a.batch * cfg.ctx / total)}
    torch.save({"config": cfg.to_dict(), "state": model.state_dict()}, out / "model.pt")
    (out / "log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print("FINAL", log["final"], flush=True)
    print(model.generate(b"def ", 200).decode("utf-8", errors="replace"), flush=True)


if __name__ == "__main__":
    main()
