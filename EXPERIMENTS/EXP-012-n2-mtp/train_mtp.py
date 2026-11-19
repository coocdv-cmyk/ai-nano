"""EXP-012 N2: train a v2 base, train its MTP heads on verifier-proven frames,
and measure speculative acceptance on a held-out same-family split.

The N2 pre-registration (`RESEARCH/EXP-N-PLAN.md` §4) trains the **existing**
MTP heads on N1 frames.  The checkpoint the repository actually ships is
EXP-008's 40-step / 625-byte distillation, which is too weak to be a meaningful
speculation target: its greedy output is degenerate, so a head that imitates it
would score a meaningless acceptance rate.  The as-built instrument therefore
does what the checkpoint cannot: it trains a real v2 base on the project's own
`core_data`, adapts it to the verified repair frames, freezes it, and only then
trains the MTP heads.  This deviation is recorded in the EXP-012 README.

Two MTP objectives are trained and compared:

* ``self``  — the draft must reproduce the frozen base's own greedy token
  (self-distillation; the direct objective for lossless speculation);
* ``truth`` — the draft must reproduce the corpus byte (the objective the
  PyTorch MTP loss already uses).

Acceptance is measured exactly as the engine consumes drafts: the draft from
``hf[t]`` at offset ``k`` is compared with ``argmax(scores[t+k-1])``, i.e. the
base's greedy prediction for the token at ``t+k``.

Usage:
    python EXPERIMENTS/EXP-012-n2-mtp/train_mtp.py \
        --corpus EXPERIMENTS/EXP-012-n2-mtp/runs/corpus-v1 \
        --out EXPERIMENTS/EXP-012-n2-mtp/runs/mtp-v1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega_nano.core.data import bits_per_byte, load  # noqa: E402
from omega_nano.core.model import CoreConfig, RMSNorm, build_model  # noqa: E402


def make_cfg(*, mtp_k: int = 4, ctx: int = 256, copy_window: int = 2048) -> CoreConfig:
    """The EXP-008/EXP-005 v2 shape, unchanged except for the MTP depth."""
    return CoreConfig(
        vocab=256, d=128, layers=2, heads=4, hidden=320, ctx=ctx,
        ternary=True, arch="v2", n_loops=1, mtp_k=mtp_k,
        patch_size=4, hash_buckets=1024, fast_dim=16,
        use_fast_weights=True, use_copy_gate=True, copy_window=copy_window,
    )


def load_bytes(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype=np.uint8).astype(np.int64)


def sample_windows(arr: np.ndarray, batch: int, ctx: int, rng: np.random.Generator):
    if arr.size <= ctx + 1:
        arr = np.tile(arr, ((ctx + 2) // max(1, arr.size)) + 2)
    hi = arr.size - ctx - 1
    starts = rng.integers(0, max(1, hi), size=batch)
    x = np.stack([arr[s:s + ctx] for s in starts])
    y = np.stack([arr[s + 1:s + 1 + ctx] for s in starts])
    return x, y


def attach_mtp(model: nn.Module, k: int, seed: int) -> None:
    """Fresh MTP heads with exactly the module shape `OmegaCoreV2` builds."""
    d = model.cfg.d
    torch.manual_seed(seed)
    model.mtp_projs = nn.ModuleList(
        nn.Sequential(RMSNorm(d), nn.Linear(d, d, bias=False))
        for _ in range(max(0, k - 1))
    )


def _ce(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(scores.reshape(-1, scores.size(-1)), targets.reshape(-1))


def train_base(
    model: nn.Module,
    arr: np.ndarray,
    *,
    seconds: float,
    ctx: int,
    batch: int,
    lr: float,
    seed: int,
    tag: str,
    log_every: int = 100,
) -> dict[str, Any]:
    """Train the base only (MTP aux loss disabled), cosine-decayed by wall clock."""
    mtp = model.mtp_projs
    model.mtp_projs = nn.ModuleList()  # `forward` skips the MTP aux term
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    rng = np.random.default_rng(seed)
    model.train()
    t0 = time.perf_counter()
    step = 0
    losses: list[float] = []
    while True:
        elapsed = time.perf_counter() - t0
        if elapsed >= seconds:
            break
        frac = elapsed / seconds
        cur_lr = lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        x, y = sample_windows(arr, batch, ctx, rng)
        scores, _ = model(torch.from_numpy(x), torch.from_numpy(y))
        yt = torch.from_numpy(y)
        if model.use_copy_gate:  # `scores` are log-probabilities
            loss = F.nll_loss(scores.reshape(-1, scores.size(-1)), yt.reshape(-1))
        else:
            loss = _ce(scores, yt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        losses.append(float(loss.detach()))
        if step % log_every == 0:
            tail = float(np.mean(losses[-log_every:]))
            print(f"  [{tag}] step {step:5d}  loss {tail:.4f}  bpb {tail / math.log(2):.4f}  "
                  f"{elapsed:.1f}s", flush=True)
    model.mtp_projs = mtp
    return {
        "tag": tag, "steps": step, "seconds": round(time.perf_counter() - t0, 3),
        "first_loss": round(float(np.mean(losses[:max(1, len(losses) // 10)])), 4) if losses else None,
        "last_loss": round(float(np.mean(losses[-max(1, len(losses) // 10):])), 4) if losses else None,
    }


def eval_bpb(model: nn.Module, arr: np.ndarray, *, ctx: int, batch: int = 8,
             n_batches: int = 24, seed: int = 1234) -> float:
    model.eval()
    rng = np.random.default_rng(seed)
    tot = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = sample_windows(arr, batch, ctx, rng)
            scores, _ = model(torch.from_numpy(x), torch.from_numpy(y))
            yt = torch.from_numpy(y)
            if model.use_copy_gate:
                loss = F.nll_loss(scores.reshape(-1, scores.size(-1)), yt.reshape(-1))
            else:
                loss = _ce(scores, yt)
            tot += float(loss)
    return bits_per_byte(tot / max(1, n_batches))


def train_mtp(
    model: nn.Module,
    arr: np.ndarray,
    *,
    objective: str,
    seconds: float,
    ctx: int,
    batch: int,
    k: int,
    lr: float = 1e-3,
    seed: int = 11,
    log_every: int = 50,
) -> dict[str, Any]:
    """Freeze the base and train only `mtp_projs` against `self` or `truth`."""
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.mtp_projs.parameters():
        p.requires_grad_(True)
    model.eval()  # no MTP aux loss; deterministic, copy gate active
    captured: dict[str, torch.Tensor] = {}
    handle = model.norm.register_forward_pre_hook(
        lambda _m, args: captured.__setitem__("hf", args[0])
    )
    opt = torch.optim.AdamW(model.mtp_projs.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    step = 0
    losses: list[float] = []
    try:
        while True:
            elapsed = time.perf_counter() - t0
            if elapsed >= seconds:
                break
            frac = elapsed / seconds
            cur_lr = lr * (0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * frac)))
            for g in opt.param_groups:
                g["lr"] = cur_lr
            x, y = sample_windows(arr, batch, ctx, rng)
            with torch.no_grad():
                scores, _ = model(torch.from_numpy(x))
                hf = captured["hf"]
            T = hf.shape[1]
            yt = torch.from_numpy(y)
            loss = torch.zeros(())
            used = 0
            for off in range(1, k + 1):
                if T <= off:
                    break
                logits = model.head(model.mtp_projs[off - 1](hf[:, :T - off, :]))
                if objective == "self":
                    target = scores[:, off - 1:T - 1, :].argmax(-1)
                elif objective == "truth":
                    target = yt[:, off - 1:T - 1]
                else:  # pragma: no cover - CLI guards it
                    raise ValueError(objective)
                loss = loss + _ce(logits, target)
                used += 1
            if used == 0:
                continue
            loss = loss / used
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.mtp_projs.parameters(), 1.0)
            opt.step()
            step += 1
            losses.append(float(loss.detach()))
            if step % log_every == 0:
                tail = float(np.mean(losses[-log_every:]))
                print(f"  [mtp:{objective}] step {step:5d}  loss {tail:.4f}  {elapsed:.1f}s",
                      flush=True)
    finally:
        handle.remove()
    return {
        "objective": objective, "steps": step,
        "seconds": round(time.perf_counter() - t0, 3),
        "first_loss": round(float(np.mean(losses[:max(1, len(losses) // 10)])), 4) if losses else None,
        "last_loss": round(float(np.mean(losses[-max(1, len(losses) // 10):])), 4) if losses else None,
    }


@torch.no_grad()
def measure_acceptance(
    model: nn.Module,
    arr: np.ndarray,
    *,
    ctx: int,
    k: int,
    batch: int = 8,
    n_batches: int = 32,
    seed: int = 4242,
    margin: int = 8,
) -> dict[str, Any]:
    """Conditional per-offset acceptance and prefix length on held-out bytes.

    `drafts[o-1][b, t]` should equal the base's greedy token at `t+o`, i.e.
    `base_pred[b, t+o-1]`; that is exactly the comparison the native verifier
    will make, including the copy gate.
    """
    model.eval()
    captured: dict[str, torch.Tensor] = {}
    handle = model.norm.register_forward_pre_hook(
        lambda _m, args: captured.__setitem__("hf", args[0])
    )
    reached = np.zeros(k + 1, np.int64)   # reached[j] = positions where 1..j-1 matched
    accepted = np.zeros(k + 1, np.int64)  # accepted[j] = draft j matched
    prefix_hist = np.zeros(k + 1, np.int64)
    top1 = 0
    top1_n = 0
    rng = np.random.default_rng(seed)
    try:
        for _ in range(n_batches):
            x, y = sample_windows(arr, batch, ctx, rng)
            scores, _ = model(torch.from_numpy(x))
            hf = captured["hf"]
            base_pred = scores.argmax(-1).numpy()  # (B, T); base_pred[t] is the token at t+1
            drafts = [model.head(model.mtp_projs[off - 1](hf)).argmax(-1).numpy()
                      for off in range(1, k + 1)]
            top1 += int((base_pred[:, :ctx - 1] == y[:, :ctx - 1]).sum())
            top1_n += int(base_pred[:, :ctx - 1].size)
            for b in range(base_pred.shape[0]):
                for t in range(margin, ctx - k - 1):
                    pref = 0
                    for off in range(1, k + 1):
                        if drafts[off - 1][b, t] == base_pred[b, t + off - 1]:
                            pref += 1
                        else:
                            break
                    prefix_hist[pref] += 1
                    for off in range(1, k + 1):
                        if pref >= off - 1:
                            reached[off] += 1
                        if pref >= off:
                            accepted[off] += 1
    finally:
        handle.remove()

    rounds = int(prefix_hist.sum())
    conditional = {
        f"offset{j}": round(float(accepted[j] / max(1, reached[j])), 4)
        for j in range(1, k + 1)
    }
    mean_prefix = float(sum(j * prefix_hist[j] for j in range(k + 1)) / max(1, rounds))
    return {
        "rounds": rounds,
        "k": k,
        "conditional_accept_rate": conditional,
        "mean_prefix_len": round(mean_prefix, 4),
        "accept_rate_per_draft": round(mean_prefix / k, 4),
        "all_accepted_rate": round(float(prefix_hist[k] / max(1, rounds)), 4),
        "prefix_hist": {str(j): int(prefix_hist[j]) for j in range(k + 1)},
        "base_top1_vs_bytes": round(top1 / max(1, top1_n), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the EXP-012 N2 base + MTP heads")
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--core-data", type=Path, default=ROOT / ".omega" / "core_data")
    ap.add_argument("--base-seconds", type=float, default=180.0)
    ap.add_argument("--ft-seconds", type=float, default=60.0)
    ap.add_argument("--mtp-seconds", type=float, default=60.0)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--objectives", default="self,truth")
    ap.add_argument("--eval-batches", type=int, default=32)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    train_bytes = (args.corpus / "train.bin").read_bytes()
    probe_bytes = (args.corpus / "probe.bin").read_bytes()
    train_arr = load_bytes(args.corpus / "train.bin")
    probe_arr = load_bytes(args.corpus / "probe.bin")
    print(f"corpus: train {len(train_bytes)} B, probe {len(probe_bytes)} B", flush=True)

    cfg = make_cfg(mtp_k=args.k + 1, ctx=args.ctx)
    model = build_model(cfg)
    print(f"params: {model.n_params()}", flush=True)

    base_log: list[dict[str, Any]] = []
    if (args.core_data / "train.bin").is_file() and args.base_seconds > 0:
        core_train = load(args.core_data)[0].astype(np.int64)
        base_log.append(train_base(model, core_train, seconds=args.base_seconds, ctx=args.ctx,
                                   batch=args.batch, lr=2e-3, seed=args.seed, tag="core"))
    if args.ft_seconds > 0:
        base_log.append(train_base(model, train_arr, seconds=args.ft_seconds, ctx=args.ctx,
                                   batch=args.batch, lr=6e-4, seed=args.seed + 1, tag="frames"))

    bpb_probe = eval_bpb(model, probe_arr, ctx=args.ctx, batch=args.batch,
                         n_batches=args.eval_batches)
    print(f"base probe bpb: {bpb_probe:.4f}", flush=True)

    # Freeze a copy of the base for every MTP objective so the comparison is exact.
    base_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
    corpus_info = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    results: dict[str, Any] = {
        "exp_id": "EXP-012",
        "corpus": {
            "train_bytes": len(train_bytes), "probe_bytes": len(probe_bytes),
            "train_sha256": corpus_info["streams"]["train.bin"]["sha256"],
            "probe_sha256": corpus_info["streams"]["probe.bin"]["sha256"],
            "sealed_leak_check": corpus_info["sealed_leak_check"],
        },
        "cfg": cfg.to_dict(),
        "params": model.n_params(),
        "base_training": base_log,
        "base_probe_bpb": round(bpb_probe, 4),
        "mtp": {},
    }

    for objective in [o.strip() for o in args.objectives.split(",") if o.strip()]:
        model.load_state_dict(base_state)
        attach_mtp(model, args.k + 1, seed=args.seed + 100)
        log = train_mtp(model, train_arr, objective=objective, seconds=args.mtp_seconds,
                        ctx=args.ctx, batch=args.batch, k=args.k, seed=args.seed + 200)
        acc = measure_acceptance(model, probe_arr, ctx=args.ctx, k=args.k,
                                 batch=args.batch, n_batches=args.eval_batches)
        ckpt = args.out / f"mtp-{objective}.pt"
        torch.save({"config": cfg.to_dict(), "state": model.state_dict(),
                    "n2": {"objective": objective, "acceptance": acc, "train": log}},
                   ckpt)
        results["mtp"][objective] = {"train": log, "acceptance": acc, "checkpoint": ckpt.name}
        print(f"[{objective}] acceptance {json.dumps(acc)}", flush=True)

    (args.out / "train_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "cfg"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
