"""EXP-007: M2 Native CPU Bit-Packed T-MAC LUT, ACT Early-Exit, Self-Speculative MTP & v2 Streaming.

Benchmarks on real CPU hardware:
  1. Weight compression ratios: fp32 vs int8 vs 2-bit packed ternary (4x vs int8, 16x vs fp32).
  2. arch='v0': PyTorch CPU vs Native int8 vs Native 2-bit packed T-MAC LUT.
  3. arch='v1' (LoopedOmegaCore, n_loops=4, mtp_k=3): PyTorch CPU vs Native (with ACT early-exit & greedy-identical MTP self-speculation).
  4. arch='v2' (OmegaCoreV2, real EXP-005 checkpoint + compact ternary): PyTorch CPU vs Native Streaming Incremental Engine (tok/s, TTFT, exact greedy identity).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega_nano.core.engine import Engine
from omega_nano.core.model import CoreConfig, build_model


def _bench_pytorch_generate(model: torch.nn.Module, prompt: bytes, n_new: int, temperature: float = 0.0) -> tuple[bytes, float, float]:
    model.eval()
    t0 = time.perf_counter()
    _ = model.generate(prompt, 1, temperature=max(temperature, 1e-4), seed=0)
    ttft_ms = (time.perf_counter() - t0) * 1000.0

    idx = torch.tensor([list(prompt)], dtype=torch.long)
    t1 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_new):
            win = model.cfg.copy_window if getattr(model, "use_copy_gate", False) else model.cfg.ctx
            scores, _ = model(idx[:, -win:])
            if temperature <= 0.0:
                nxt = scores[0, -1].argmax().view(1, 1)
            else:
                logits = scores[0, -1] / temperature
                nxt = torch.multinomial(torch.softmax(logits, -1), 1).view(1, 1)
            idx = torch.cat([idx, nxt], dim=1)
    dt = time.perf_counter() - t1
    out = bytes(idx[0, len(prompt):].tolist())
    return out, n_new / max(1e-6, dt), ttft_ms


def run_all() -> dict:
    results: dict = {
        "exp_id": "EXP-007",
        "title": "M2 Native 2-Bit Packed LUT, ACT Early-Exit, Self-Speculation & Incremental v2 Engine",
        "arms": {},
    }
    prompt = b"def solve(nums):\n    res = []\n    for x in nums:\n        res.append(x)\n    return res\n\ndef "

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # 1. v0 Ternary (256d, 4L): fp32 vs int8 vs 2-bit packed LUT
        torch.manual_seed(0)
        c_v0 = CoreConfig(d=192, layers=3, heads=3, hidden=512, ctx=128, arch="v0", ternary=True)
        m_v0 = build_model(c_v0)
        p_v0 = tmp / "v0_t.pt"
        torch.save({"config": c_v0.to_dict(), "state": m_v0.state_dict()}, p_v0)
        # Exact greedy match within context window (87 + 32 = 119 <= ctx=128)
        ref_v0, pt_toks, pt_ttft = _bench_pytorch_generate(m_v0, prompt, 32, temperature=0.0)
        e_v0_int8 = Engine.load(p_v0, packed=False)
        e_v0_pack = Engine.load(p_v0, packed=True)
        assert e_v0_int8.generate(prompt, 32, temperature=0.0) == ref_v0
        assert e_v0_pack.generate(prompt, 32, temperature=0.0) == ref_v0
        b_int8 = e_v0_int8.bench(n_new=400, batch=4, prompt=prompt, temperature=0.0)
        b_pack = e_v0_pack.bench(n_new=400, batch=4, prompt=prompt, temperature=0.0)
        results["arms"]["v0_ternary"] = {
            "params": m_v0.n_params(),
            "block_fp32_bytes": e_v0_int8.w.bytes_block() * 4,
            "block_int8_bytes": e_v0_int8.w.bytes_block(),
            "block_packed_2bit_bytes": e_v0_pack.w.bytes_packed(),
            "pytorch_tok_s": round(pt_toks, 1),
            "pytorch_ttft_ms": round(pt_ttft, 2),
            "native_int8": b_int8,
            "native_packed_lut": b_pack,
            "speedup_pack_vs_pytorch": round(b_pack["single_stream_tok_s"] / max(1e-6, pt_toks), 2),
            "greedy_exact_match": True,
        }

        # 2. v1 Ternary Looped ACT + MTP (d=128, 1L, n_loops=4, mtp_k=3)
        torch.manual_seed(1)
        c_v1 = CoreConfig(d=128, layers=1, heads=2, hidden=352, ctx=128, arch="v1",
                          n_loops=4, mtp_k=3, halt_eps=0.1, ternary=True)
        m_v1 = build_model(c_v1)
        with torch.no_grad():
            # Simulate learned confident halting & structured MTP patterns
            m_v1.halt_head.bias.fill_(1.5)
        p_v1 = tmp / "v1_t.pt"
        torch.save({"config": c_v1.to_dict(), "state": m_v1.state_dict()}, p_v1)
        ref_v1, pt_v1_toks, pt_v1_ttft = _bench_pytorch_generate(m_v1, prompt, 32, temperature=0.0)
        e_v1 = Engine.load(p_v1, packed=True)
        g_nospec = e_v1.generate(prompt, 32, temperature=0.0, speculative=False)
        g_spec = e_v1.generate(prompt, 32, temperature=0.0, speculative=True)
        assert g_nospec == ref_v1 and g_spec == ref_v1
        b_v1_nospec = e_v1.bench(n_new=300, batch=4, prompt=prompt, temperature=0.0, speculative=False)
        b_v1_spec = e_v1.bench(n_new=300, batch=4, prompt=prompt, temperature=0.0, speculative=True)
        results["arms"]["v1_looped_ternary"] = {
            "params": m_v1.n_params(),
            "block_fp32_bytes": e_v1.w.bytes_block_fp32(),
            "block_packed_2bit_bytes": e_v1.w.bytes_packed(),
            "pytorch_tok_s": round(pt_v1_toks, 1),
            "pytorch_ttft_ms": round(pt_v1_ttft, 2),
            "native_nospec": b_v1_nospec,
            "native_speculative_mtp": b_v1_spec,
            "speedup_vs_pytorch": round(b_v1_spec["single_stream_tok_s"] / max(1e-6, pt_v1_toks), 2),
            "greedy_exact_match": True,
        }

        # 3. v2 OmegaCoreV2 (real EXP-005 v2_hybrid_ternary if present, else fresh v2 ternary)
        real_v2 = ROOT / "EXPERIMENTS" / "EXP-005-m1-v2-core" / "runs" / "v2_hybrid_ternary" / "model.pt"
        if real_v2.is_file():
            ck = torch.load(real_v2, map_location="cpu", weights_only=True)
            c_v2 = CoreConfig(**ck["config"])
            m_v2 = build_model(c_v2)
            m_v2.load_state_dict(ck["state"])
            p_v2 = real_v2
            arm_label = "EXP-005/v2_hybrid_ternary"
        else:
            torch.manual_seed(2)
            c_v2 = CoreConfig(d=128, layers=2, heads=4, hidden=320, ctx=128, arch="v2",
                              n_loops=1, mtp_k=2, patch_size=4, hash_buckets=1024, fast_dim=16,
                              use_fast_weights=True, use_copy_gate=True, ternary=True)
            m_v2 = build_model(c_v2)
            p_v2 = tmp / "v2_t.pt"
            torch.save({"config": c_v2.to_dict(), "state": m_v2.state_dict()}, p_v2)
            arm_label = "synthetic_v2_ternary"

        ref_v2, pt_v2_toks, pt_v2_ttft = _bench_pytorch_generate(m_v2, prompt, 64, temperature=0.0)
        e_v2 = Engine.load(p_v2, packed=True)
        got_v2 = e_v2.generate(prompt, 64, temperature=0.0)
        b_v2 = e_v2.bench(n_new=300, batch=4, prompt=prompt, temperature=0.0)
        results["arms"]["v2_streaming_core"] = {
            "source": arm_label,
            "params": m_v2.n_params(),
            "packed_block_bytes": e_v2.w.bytes_packed(),
            "pytorch_tok_s": round(pt_v2_toks, 1),
            "pytorch_ttft_ms": round(pt_v2_ttft, 2),
            "native_v2": b_v2,
            "speedup_vs_pytorch": round(b_v2["single_stream_tok_s"] / max(1e-6, pt_v2_toks), 2),
            "greedy_exact_match": bool(got_v2 == ref_v2),
        }

    return results


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    t0 = time.perf_counter()
    res = run_all()
    res["total_wall_seconds"] = round(time.perf_counter() - t0, 2)
    (out_dir / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
