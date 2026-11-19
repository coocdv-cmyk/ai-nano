"""EXP-012 N2 invariant check: native v2 speculation must be lossless.

Speculative decoding is only admissible if, for greedy decoding, the token
stream is bit-identical to the sequential path.  This check exercises three
regimes on CPU-sized random models (no training required):

``reject``  — random MTP heads: nearly every draft is rejected, so the undo
              log (ring buffer, n-gram tables, conv history) runs constantly.
``accept``  — a degenerate base whose argmax is a fixed token, plus MTP heads
              that produce the same token: every draft is accepted, so the
              commit path and patch-boundary `pending_patch` hand-off run.
``adaptive``— the production gate must disable speculation after >= 64 drafts
              with a < 25% acceptance rate, without changing the output.

Run:
    python EXPERIMENTS/EXP-012-n2-mtp/check_spec.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega_nano.core.engine import Engine  # noqa: E402
from omega_nano.core.model import CoreConfig, build_model  # noqa: E402


def make_model(seed: int, *, mtp_k: int = 4, use_copy: bool = True) -> torch.nn.Module:
    torch.manual_seed(seed)
    cfg = CoreConfig(
        d=64, layers=1, heads=2, hidden=96, ctx=64, arch="v2",
        n_loops=1, mtp_k=mtp_k, patch_size=4, hash_buckets=256, fast_dim=8,
        use_fast_weights=True, use_copy_gate=use_copy, copy_window=2048, ternary=True,
    )
    model = build_model(cfg)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                p.normal_(0, 0.14)
    return model


def save(model: torch.nn.Module, path: Path) -> Engine:
    torch.save({"config": model.cfg.to_dict(), "state": model.state_dict()}, path)
    return Engine.load(path, packed=True)


def run_case(name: str, engine: Engine, prompt: bytes, n: int, *,
             adaptive: bool, degenerate_head: int | None = None) -> dict:
    engine.spec_adaptive = adaptive
    engine.generate(prompt, 1, temperature=0, speculative=False)  # Numba warm-up
    t0 = time.perf_counter()
    plain = engine.generate(prompt, n, temperature=0, speculative=False)
    t_plain = time.perf_counter() - t0
    t0 = time.perf_counter()
    spec = engine.generate(prompt, n, temperature=0, speculative=True)
    t_spec = time.perf_counter() - t0
    stats = dict(engine.last_stats)
    ok = plain == spec
    print(f"[{name}] identical={ok} plain={t_plain:.3f}s spec={t_spec:.3f}s "
          f"drafts={stats.get('speculative_drafts')} accepted={stats.get('speculative_accepted')} "
          f"rate={stats.get('spec_accept_rate')} disabled={stats.get('spec_disabled')}",
          flush=True)
    if not ok:
        for i, (a, b) in enumerate(zip(plain, spec)):
            if a != b:
                print(f"  first divergence at byte {i}: plain={a} spec={b}", flush=True)
                break
        print(f"  plain={plain[:48]!r}\n  spec ={spec[:48]!r}", flush=True)
    return {
        "case": name, "identical": ok, "n": n,
        "plain_s": round(t_plain, 4), "spec_s": round(t_spec, 4),
        "stats": stats,
        "plain_sha": plain.hex(), "spec_sha": spec.hex(),
    }


def main() -> None:
    tmp = Path(__file__).resolve().parent / "runs" / "spec-check"
    tmp.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    # 1. Random heads: rejection-heavy undo path, adaptive gate off.
    m = make_model(seed=21)
    eng = save(m, tmp / "random.pt")
    prompt = b"def repair(rows):\n    return rows[0]\n# patch\n"
    results.append(run_case("reject", eng, prompt, 192, adaptive=False))

    # 2. All-zero base: argmax is 0 everywhere, and every MTP head also emits 0,
    #    so every draft is accepted.  Contrast with case 1, which never accepts.
    m = make_model(seed=22, use_copy=False)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
    eng = save(m, tmp / "accept.pt")
    results.append(run_case("accept", eng, b"alpha beta gamma delta\n", 97, adaptive=False))

    # 3. Production gate: random heads must trip it and stay lossless.
    m = make_model(seed=23)
    eng = save(m, tmp / "adaptive.pt")
    results.append(run_case("adaptive", eng, b"x = 0\nwhile x < 9:\n    x += 1\n", 700,
                            adaptive=True))

    verdict = all(r["identical"] for r in results)
    if verdict:
        accept = results[1]["stats"]
        adaptive = results[2]["stats"]
        verdict = bool(
            accept.get("spec_accept_rate") == 1.0
            and int(accept.get("spec_batch_forwards", 0)) > 0
            and adaptive.get("spec_disabled") is True
        )
    out = {
        "exp_id": "EXP-012",
        "instrument": "check_spec",
        "verdict": verdict,
        "cases": results,
        "note": "accept case must reach rate 1.0; adaptive must disable after >=64 drafts",
    }
    (tmp / "check_spec.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict,
                      "cases": [{k: r[k] for k in ("case", "identical", "stats")} for r in results]},
                     indent=2), flush=True)
    raise SystemExit(0 if verdict else 1)


if __name__ == "__main__":
    main()
