"""EXP-005 (M1): OmegaCore v2 vs OmegaCore v0 under matched params, context, and CPU wall-clock.

Evaluates:
  1. Pure neural v2 (Hash-Ngram Stem + Strided Looped Thought Core + Delta Fast Weights)
     against v0 at matched ~870K parameters and matched ctx=256.
  2. Full hybrid v2 (adding Causal Associative CopyGate) at matched ctx=256 (fp32 and ternary).
  3. Long-range associative recall at 2,048 bytes (8x beyond v0's 256-byte horizon), across
     Python identifiers and Arabic UTF-8 phrases.
  4. Held-out compression (bits/byte) on clean Python stdlib, English, and Arabic prose vs zlib-9.

Usage:
  python EXPERIMENTS/EXP-005-m1-v2-core/run_exp.py --seconds 90
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from omega_nano.core.data import baselines, batches, bits_per_byte, build_corpus, load  # noqa: E402
from omega_nano.core.model import CoreConfig, build_model  # noqa: E402
from omega_nano.hw import detect  # noqa: E402

HERE = Path(__file__).parent
DATA = ROOT / ".omega" / "core_data"
RUNS = HERE / "runs"


ARABIC_EVAL = (
    "تعمل نواة أوميغا على دمج الذاكرة الخارجية مع التفكير التكراري المتكيف لتقليل استهلاك الذاكرة "
    "وزيادة الدقة على المعالجات العادية بدون الحاجة إلى بطاقة رسوميات خارجية. "
    "عندما يتكرر مفهوم مهم داخل السياق الطويل، يجب على الشبكة استرجاعه بدقة عالية وتجنب التخمين غير المتحقق منه.\n"
) * 12

ENGLISH_EVAL = (
    "OMEGA-NANO separates persistent world knowledge from compact cache-resident reasoning weights. "
    "Dynamic patching compresses predictable boilerplate while fast-weight plasticity binds novel rules "
    "and exact identifiers across thousands of bytes on ordinary CPU hardware without GPU acceleration.\n"
) * 12


@torch.no_grad()
def evaluate_bpb(model: torch.nn.Module, val: np.ndarray, ctx: int, n_batches: int = 48, batch: int = 16) -> float:
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


@torch.no_grad()
def evaluate_text_bpb(model: torch.nn.Module, raw: bytes, max_ctx: int) -> dict[str, float]:
    model.eval()
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
    n_tok = min(len(arr) - 1, max_ctx)
    x = torch.from_numpy(arr[:n_tok]).unsqueeze(0)
    y = torch.from_numpy(arr[1 : n_tok + 1]).unsqueeze(0)
    _, loss = model(x, y)
    zlib_bpb = len(zlib.compress(raw[: n_tok + 1], 9)) * 8.0 / (n_tok + 1)
    model.train()
    return {"model_bpb": round(bits_per_byte(loss.item()), 4), "zlib9_bpb": round(zlib_bpb, 4), "bytes": int(n_tok)}


@torch.no_grad()
def evaluate_long_range_recall(model: torch.nn.Module, val: np.ndarray, model_ctx: int,
                               total_len: int = 2048, n_trials: int = 16) -> dict[str, float]:
    """Place a unique Python/Arabic key-value binding near byte 40 and test recall near byte 1900."""
    model.eval()
    rng = np.random.default_rng(99)
    filler_pool = val.tobytes()
    bpb_list: list[float] = []
    acc_list: list[float] = []

    for i in range(n_trials):
        secret_id = f"omega_secret_token_{1000 + i * 37}"
        ar_tag = f"رمز_المرجع_{200 + i * 13}"
        needle = f"\n# binding: {secret_id} => {ar_tag}\n".encode("utf-8")
        query = f"\n# recall:  {secret_id} => ".encode("utf-8")
        target = f"{ar_tag}\n".encode("utf-8")

        start = int(rng.integers(0, max(1, len(filler_pool) - total_len)))
        filler = filler_pool[start : start + total_len - len(needle) - len(query) - len(target) - 64]
        head_pad = filler[:40]
        mid_pad = filler[40:]
        full_seq = head_pad + needle + mid_pad + query + target

        arr = np.frombuffer(full_seq, dtype=np.uint8).astype(np.int64)
        # Keep only the suffix visible to the model's context budget
        visible_start = max(0, len(arr) - 1 - model_ctx)
        x = torch.from_numpy(arr[visible_start:-1]).unsqueeze(0)
        y = torch.from_numpy(arr[visible_start + 1 :]).unsqueeze(0)
        scores, _ = model(x)

        t_len = len(target)
        t_scores = scores[0, -t_len:, :]
        t_targets = y[0, -t_len:]
        ce = torch.nn.functional.cross_entropy(t_scores, t_targets) if not getattr(model, "use_copy_gate", False) \
            else torch.nn.functional.nll_loss(t_scores, t_targets)
        bpb_list.append(bits_per_byte(ce.item()))
        acc_list.append(float((t_scores.argmax(dim=-1) == t_targets).float().mean().item()))

    model.train()
    return {
        "eval_horizon_bytes": total_len,
        "model_visible_window": model_ctx,
        "needle_bpb": round(float(np.mean(bpb_list)), 4),
        "needle_top1_acc": round(float(np.mean(acc_list)), 4),
    }


def train_arm(name: str, cfg: CoreConfig, train: np.ndarray, val: np.ndarray,
              seconds: float, batch: int = 16, lr: float = 2e-3, threads: int = 4) -> tuple[dict, torch.nn.Module]:
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    rng = np.random.default_rng(0)
    it = batches(train, batch, cfg.ctx, rng)
    curve = []
    t0 = time.perf_counter()
    step = 0
    warmup = 30
    next_eval = 15.0

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
        if time.perf_counter() - t0 >= next_eval:
            vb = evaluate_bpb(model, val, cfg.ctx, n_batches=24)
            curve.append({"elapsed_s": round(time.perf_counter() - t0, 1), "step": step,
                          "train_bpb": round(bits_per_byte(loss.item()), 4), "val_bpb": round(vb, 4)})
            next_eval += 15.0

    total = time.perf_counter() - t0
    final_val = evaluate_bpb(model, val, cfg.ctx, n_batches=48)

    # Measure long-range recall at 2,048 bytes: v0 is capped at cfg.ctx (256), while v2 sees 2,048.
    long_win = 2048 if cfg.arch == "v2" else cfg.ctx
    long_recall = evaluate_long_range_recall(model, val, model_ctx=long_win, total_len=2048)
    long_recall_2048_mem = None
    if cfg.arch == "v2" and cfg.use_copy_gate:
        prev_cw = model.cfg.copy_window
        model.cfg.copy_window = 2048
        long_recall_2048_mem = evaluate_long_range_recall(model, val, model_ctx=2048, total_len=2048)
        model.cfg.copy_window = prev_cw

    # Bilingual + code micro-eval (using matched ctx=256 for fair local compression comparison)
    ar_eval = evaluate_text_bpb(model, ARABIC_EVAL.encode("utf-8"), max_ctx=cfg.ctx)
    en_eval = evaluate_text_bpb(model, ENGLISH_EVAL.encode("utf-8"), max_ctx=cfg.ctx)

    out_dir = RUNS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.to_dict(), "state": model.state_dict()}, out_dir / "model.pt")

    summary = {
        "name": name,
        "config": cfg.to_dict(),
        "params": model.n_params(),
        "steps": step,
        "seconds": round(total, 2),
        "train_bytes": step * batch * cfg.ctx,
        "train_tok_s": round(step * batch * cfg.ctx / total),
        "final_val_bpb": round(final_val, 4),
        "curve": curve,
        "long_range_2048": long_recall,
        "long_range_2048_copy_mem": long_recall_2048_mem,
        "arabic_ctx256": ar_eval,
        "english_ctx256": en_eval,
    }
    if hasattr(model, "last_avg_loops"):
        summary["eval_avg_loops"] = round(float(model.last_avg_loops), 3)
    return summary, model


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    if not (DATA / "train.bin").exists():
        build_corpus(DATA)
    train, val = load(DATA)
    base_file = DATA / "baselines.json"
    base_bpb = json.loads(base_file.read_text(encoding="utf-8")) if base_file.exists() else baselines(train, val)

    arms = [
        ("v0_baseline", CoreConfig(d=128, layers=4, heads=4, hidden=352, ctx=256, arch="v0")),
        ("v2_pure_neural", CoreConfig(
            d=128, layers=2, heads=4, hidden=256, ctx=256, arch="v2",
            n_loops=2, mtp_k=1, patch_size=4, hash_buckets=2048, fast_dim=16,
            use_fast_weights=True, use_copy_gate=False,
        )),
        ("v2_hybrid_ctx256", CoreConfig(
            d=128, layers=2, heads=4, hidden=256, ctx=256, arch="v2",
            n_loops=2, mtp_k=1, patch_size=4, hash_buckets=2048, fast_dim=16,
            use_fast_weights=True, use_copy_gate=True, copy_window=256,
        )),
        ("v2_hybrid_ternary", CoreConfig(
            d=128, layers=2, heads=4, hidden=256, ctx=256, arch="v2", ternary=True,
            n_loops=2, mtp_k=1, patch_size=4, hash_buckets=2048, fast_dim=16,
            use_fast_weights=True, use_copy_gate=True, copy_window=256,
        )),
    ]

    results = {
        "experiment": "EXP-005-m1-v2-core",
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "hardware": detect(),
        "corpus_meta": json.loads((DATA / "meta.json").read_text(encoding="utf-8")),
        "baselines_bpb": base_bpb,
        "arms": [],
    }

    for name, cfg in arms:
        print(f"\n=== Training {name} ({a.seconds}s) ===", flush=True)
        res, _ = train_arm(name, cfg, train, val, seconds=a.seconds, threads=a.threads)
        results["arms"].append(res)
        lr_show = res['long_range_2048_copy_mem'] or res['long_range_2048']
        print(
            f"{name}: params={res['params']['total']:,} steps={res['steps']} "
            f"val_bpb={res['final_val_bpb']} long2048_acc={lr_show['needle_top1_acc']} "
            f"ar_bpb={res['arabic_ctx256']['model_bpb']} en_bpb={res['english_ctx256']['model_bpb']}",
            flush=True,
        )

    out_path = HERE / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved results to", out_path, flush=True)


if __name__ == "__main__":
    main()
