"""EXP-012 N2: measure native v2 speculation on the trained checkpoint.

Reports, on one fixed 10k-token greedy probe:

* speculative vs non-speculative bytes (must be identical);
* single-stream tok/s for both paths (median of `--reps`);
* drafts, accepted drafts, accept rate, batch forwards, gate state;
* process peak RSS sampled during the runs.

The pre-registered N2 gate is acceptance > 40% and throughput > 10,509 tok/s
while staying byte-identical (`RESEARCH/EXP-N-PLAN.md` §4).

Run:
    python EXPERIMENTS/EXP-012-n2-mtp/measure_engine.py \
        --ckpt EXPERIMENTS/EXP-012-n2-mtp/runs/mtp-v1/mtp-self.pt \
        --corpus EXPERIMENTS/EXP-012-n2-mtp/runs/corpus-v1 \
        --out EXPERIMENTS/EXP-012-n2-mtp/runs/mtp-v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega_nano.core.engine import Engine  # noqa: E402


class RssSampler(threading.Thread):
    def __init__(self, interval_s: float = 0.02) -> None:
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.peak = 0
        self._stop = threading.Event()

    def run(self) -> None:
        proc = psutil.Process()
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, proc.memory_info().rss)
            except psutil.Error:
                break
            time.sleep(self.interval_s)

    def stop(self) -> int:
        self._stop.set()
        self.join(timeout=2.0)
        return self.peak


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_once(eng: Engine, prompt: bytes, n: int, *, speculative: bool, adaptive: bool,
             seed: int = 0) -> tuple[bytes, float, dict]:
    eng.spec_adaptive = adaptive
    t0 = time.perf_counter()
    out = eng.generate(prompt, n, temperature=0, seed=seed, speculative=speculative)
    dt = time.perf_counter() - t0
    return out, dt, dict(eng.last_stats)


def summarise(name: str, runs: list[tuple[bytes, float, dict]], n: int) -> dict:
    times = [r[1] for r in runs]
    out = runs[-1][0]
    stats = runs[-1][2]
    return {
        "path": name,
        "n_tokens": n,
        "tok_per_s_median": round(n / statistics.median(times), 1),
        "tok_per_s_min": round(n / max(times), 1),
        "tok_per_s_max": round(n / min(times), 1),
        "seconds": [round(t, 4) for t in times],
        "sha256": sha(out),
        "bytes_prefix_hex": out[:32].hex(),
        "stats": stats,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure EXP-012 N2 native speculation")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tokens", type=int, default=10_000)
    ap.add_argument("--prompt-bytes", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    out_dir = args.out or args.ckpt.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = (args.corpus / "probe.bin").read_bytes()
    if len(probe) < args.prompt_bytes + 16:
        raise SystemExit(f"probe.bin too short: {len(probe)} B")
    prompt = probe[:args.prompt_bytes]

    eng = Engine.load(args.ckpt, packed=True)
    cfg = eng.w.cfg
    print(f"checkpoint: {args.ckpt}  arch={cfg.arch}  mtp={getattr(eng.w, 'mtp_count', 0)}  "
          f"patch={cfg.patch_size}  d={cfg.d}  layers={cfg.layers}", flush=True)

    # Numba/JIT warm-up, excluded from all timings.
    eng.generate(prompt, 24, temperature=0, speculative=False)
    eng.generate(prompt, 24, temperature=0, speculative=True)

    sampler = RssSampler()
    sampler.start()
    plain_runs: list[tuple[bytes, float, dict]] = []
    spec_runs: list[tuple[bytes, float, dict]] = []
    for i in range(args.reps):
        plain_runs.append(run_once(eng, prompt, args.tokens, speculative=False, adaptive=False))
        print(f"  plain rep {i + 1}: {args.tokens / plain_runs[-1][1]:,.1f} tok/s", flush=True)
        spec_runs.append(run_once(eng, prompt, args.tokens, speculative=True, adaptive=False))
        print(f"  spec  rep {i + 1}: {args.tokens / spec_runs[-1][1]:,.1f} tok/s  "
              f"drafts={spec_runs[-1][2].get('speculative_drafts')} "
              f"accepted={spec_runs[-1][2].get('speculative_accepted')}", flush=True)
    prod = run_once(eng, prompt, min(args.tokens, 2_000), speculative=True, adaptive=True)
    peak = sampler.stop()

    plain = summarise("non_speculative", plain_runs, args.tokens)
    spec = summarise("speculative", spec_runs, args.tokens)
    identical = all(r[0] == plain_runs[0][0] for r in plain_runs + spec_runs)
    first_div = next(
        (i for i, (a, b) in enumerate(zip(plain_runs[0][0], spec_runs[0][0])) if a != b),
        None,
    )
    rate = spec["stats"].get("spec_accept_rate", 0.0)
    speedup = spec["tok_per_s_median"] / max(1e-9, plain["tok_per_s_median"])
    result = {
        "exp_id": "EXP-012",
        "instrument": "measure_engine",
        "checkpoint": str(args.ckpt),
        "config": cfg.to_dict(),
        "mtp_count": int(getattr(eng.w, "mtp_count", 0)),
        "probe": {"prompt_bytes": len(prompt), "n_tokens": args.tokens, "reps": args.reps},
        "non_speculative": plain,
        "speculative": spec,
        "identical_greedy": identical,
        "first_divergence": first_div,
        "speedup": round(speedup, 4),
        "accept_rate": rate,
        "peak_rss_mb": round(peak / 2**20, 2),
        "production_default": {
            "n_tokens": prod[0].__len__(),
            "seconds": round(prod[1], 4),
            "stats": prod[2],
        },
        "gates": {
            "acceptance_gt_0.40": bool(rate > 0.40),
            "throughput_gt_10509": bool(spec["tok_per_s_median"] > 10_509),
            "greedy_identical_10k": bool(identical),
        },
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    (out_dir / "engine_results.json").write_text(payload, encoding="utf-8")
    (out_dir / f"engine_results-{args.ckpt.stem}.json").write_text(payload, encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2), flush=True)
    if not identical:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
