"""Native CPU inference engine for OmegaCore (Numba/LLVM, no Python per token).

Weights are exported once: ternary arms store each block matrix as int8 codes
{-1, 0, 1} plus one fp32 scale (1 byte/weight, the format EXP-001 found
fastest on this CPU); fp32 arms store float32 with scale 1. The whole decode
loop — embedding, RMSNorm, rotary attention with a KV cache, SwiGLU, head,
sampling — runs inside one compiled function. `generate_batch` decodes
independent sequences in parallel across cores (aggregate throughput); single
sequences report single-stream speed. The two are never mixed in reporting.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import os
from pathlib import Path
import time

import numba
import numpy as np
from numba import prange

from .engine_v1_v2 import (
    V1Weights,
    V2Weights,
    _v1_generate,
    _v1_logits_all,
    _v2_run,
    export_v1,
    export_v2,
)
from .model import CoreConfig, ternarize
from .packed import matvec_packed_lut, pack_ternary


@dataclass
class Weights:
    cfg: CoreConfig
    emb: np.ndarray
    n1: np.ndarray
    n2: np.ndarray
    wqkv: np.ndarray
    sqkv: np.ndarray
    wo: np.ndarray
    so: np.ndarray
    wg: np.ndarray
    sg: np.ndarray
    wu: np.ndarray
    su: np.ndarray
    wd: np.ndarray
    sd: np.ndarray
    nf: np.ndarray
    head: np.ndarray
    cos: np.ndarray
    sin: np.ndarray
    pqkv: np.ndarray | None = None
    po: np.ndarray | None = None
    pg: np.ndarray | None = None
    pu: np.ndarray | None = None
    pd: np.ndarray | None = None

    def args(self):
        return (self.emb, self.n1, self.n2, self.wqkv, self.sqkv, self.wo, self.so, self.wg, self.sg, self.wu,
                self.su, self.wd, self.sd, self.nf, self.head, self.cos, self.sin)

    def packed_args(self):
        return (self.pqkv, self.po, self.pg, self.pu, self.pd)

    def bytes_block(self) -> int:
        return sum(a.nbytes for a in (self.wqkv, self.wo, self.wg, self.wu, self.wd))

    def bytes_packed(self) -> int:
        if self.pqkv is not None:
            return sum(a.nbytes for a in (self.pqkv, self.po, self.pg, self.pu, self.pd))
        return self.bytes_block()

    def bytes_total(self) -> int:
        return sum(a.nbytes for a in self.args())


def export(path: str | Path) -> Weights:
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=True)  # never unpickle arbitrary objects
    cfg = CoreConfig(**ck["config"])
    if cfg.arch != "v0":
        raise ValueError(f"export() handles arch='v0'; use Engine.load() for arch={cfg.arch!r}")
    st = ck["state"]

    def mat(name):
        w = st[name].float()
        if cfg.ternary:
            codes, s = ternarize(w)
            return codes.to(torch.int8).numpy(), float(s)
        return w.numpy().astype(np.float32), 1.0

    def stack(kind):
        ws, ss = zip(*(mat(f"blocks.{i}.{kind}.weight") for i in range(cfg.layers)))
        return np.ascontiguousarray(np.stack(ws)), np.array(ss, dtype=np.float32)

    wqkv, sqkv = stack("qkv")
    wo, so = stack("proj")
    wg, sg = stack("gate")
    wu, su = stack("up")
    wd, sd = stack("down")
    pqkv = po = pg = pu = pd = None
    if cfg.ternary:
        pqkv = np.ascontiguousarray(np.stack([pack_ternary(wqkv[i])[0] for i in range(cfg.layers)]))
        po = np.ascontiguousarray(np.stack([pack_ternary(wo[i])[0] for i in range(cfg.layers)]))
        pg = np.ascontiguousarray(np.stack([pack_ternary(wg[i])[0] for i in range(cfg.layers)]))
        pu = np.ascontiguousarray(np.stack([pack_ternary(wu[i])[0] for i in range(cfg.layers)]))
        pd = np.ascontiguousarray(np.stack([pack_ternary(wd[i])[0] for i in range(cfg.layers)]))

    hd = cfg.d // cfg.heads
    inv = 1.0 / 10000.0 ** (np.arange(0, hd, 2, dtype=np.float64) / hd)
    ang = np.outer(np.arange(cfg.ctx, dtype=np.float64), inv)
    f = lambda k: st[k].float().numpy().astype(np.float32)  # noqa: E731
    return Weights(cfg, f("emb.weight"), np.stack([f(f"blocks.{i}.n1.weight") for i in range(cfg.layers)]),
                   np.stack([f(f"blocks.{i}.n2.weight") for i in range(cfg.layers)]), wqkv, sqkv, wo, so, wg, sg,
                   wu, su, wd, sd, f("norm.weight"), f("head.weight"), np.cos(ang).astype(np.float32),
                   np.sin(ang).astype(np.float32), pqkv, po, pg, pu, pd)


@numba.njit(fastmath=True, cache=True, inline="always")
def _matvec(w, s, x, out):
    O = w.shape[0]
    I = w.shape[1]
    o = 0
    lim = O - (O & 3)
    while o < lim:
        s0 = np.float32(0.0)
        s1 = np.float32(0.0)
        s2 = np.float32(0.0)
        s3 = np.float32(0.0)
        for j in range(I):
            xj = x[j]
            s0 += np.float32(w[o, j]) * xj
            s1 += np.float32(w[o + 1, j]) * xj
            s2 += np.float32(w[o + 2, j]) * xj
            s3 += np.float32(w[o + 3, j]) * xj
        out[o] = s0 * s
        out[o + 1] = s1 * s
        out[o + 2] = s2 * s
        out[o + 3] = s3 * s
        o += 4
    while o < O:
        acc = np.float32(0.0)
        for j in range(I):
            acc += np.float32(w[o, j]) * x[j]
        out[o] = acc * s
        o += 1


@numba.njit(fastmath=True, cache=True, inline="always")
def _rmsnorm(x, w, out):
    ss = np.float32(0.0)
    for i in range(x.shape[0]):
        ss += x[i] * x[i]
    r = np.float32(1.0) / np.sqrt(ss / x.shape[0] + np.float32(1e-6))
    for i in range(x.shape[0]):
        out[i] = x[i] * r * w[i]


@numba.njit(fastmath=True, cache=True)
def _step(tok, pos, kc, vc, emb, n1, n2, wqkv, sqkv, wo, so, wg, sg, wu, su, wd, sd, nf, head, cos, sin,
          heads, logits, h, qkv, att, tmp, g, u, scores,
          use_packed, pqkv, po, pg, pu, pd, lut_d, lut_h):
    """Run one token through the model at position `pos` using a rolling ring KV cache."""
    L = wqkv.shape[0]
    ctx = kc.shape[1]
    d = emb.shape[1]
    hd = d // heads
    half = hd // 2
    hidden = wg.shape[1]
    slot = pos % ctx
    valid_len = pos + 1 if pos < ctx else ctx
    x = emb[tok].copy()
    inv_sqrt = np.float32(1.0 / math.sqrt(hd))
    for l in range(L):
        _rmsnorm(x, n1[l], h)
        if use_packed:
            matvec_packed_lut(pqkv[l], sqkv[l], h, d, lut_d, qkv)
        else:
            _matvec(wqkv[l], sqkv[l], h, qkv)
        # rotary on q and k (pairs of even/odd dims within each head)
        for hh in range(heads):
            b = hh * hd
            for i in range(half):
                c = cos[pos, i]
                s = sin[pos, i]
                q1 = qkv[b + 2 * i]
                q2 = qkv[b + 2 * i + 1]
                qkv[b + 2 * i] = q1 * c - q2 * s
                qkv[b + 2 * i + 1] = q1 * s + q2 * c
                k1 = qkv[d + b + 2 * i]
                k2 = qkv[d + b + 2 * i + 1]
                kc[l, slot, b + 2 * i] = k1 * c - k2 * s
                kc[l, slot, b + 2 * i + 1] = k1 * s + k2 * c
        for i in range(d):
            vc[l, slot, i] = qkv[2 * d + i]
        for hh in range(heads):
            b = hh * hd
            m = np.float32(-1e30)
            for t in range(valid_len):
                acc = np.float32(0.0)
                for i in range(hd):
                    acc += qkv[b + i] * kc[l, t, b + i]
                acc *= inv_sqrt
                scores[t] = acc
                if acc > m:
                    m = acc
            z = np.float32(0.0)
            for t in range(valid_len):
                e = np.exp(scores[t] - m)
                scores[t] = e
                z += e
            inv_z = np.float32(1.0) / z
            for i in range(hd):
                att[b + i] = np.float32(0.0)
            for t in range(valid_len):
                w_att = scores[t] * inv_z
                for i in range(hd):
                    att[b + i] += w_att * vc[l, t, b + i]
        if use_packed:
            matvec_packed_lut(po[l], so[l], att, d, lut_d, tmp)
        else:
            _matvec(wo[l], so[l], att, tmp)
        for i in range(d):
            x[i] += tmp[i]
        _rmsnorm(x, n2[l], h)
        if use_packed:
            matvec_packed_lut(pg[l], sg[l], h, d, lut_d, g)
            matvec_packed_lut(pu[l], su[l], h, d, lut_d, u)
        else:
            _matvec(wg[l], sg[l], h, g)
            _matvec(wu[l], su[l], h, u)
        for i in range(hidden):
            gi = g[i]
            g[i] = gi / (np.float32(1.0) + np.exp(-gi)) * u[i]
        if use_packed:
            matvec_packed_lut(pd[l], sd[l], g, hidden, lut_h, tmp)
        else:
            _matvec(wd[l], sd[l], g, tmp)
        for i in range(d):
            x[i] += tmp[i]
    _rmsnorm(x, nf, h)
    _matvec(head, np.float32(1.0), h, logits)


@numba.njit(fastmath=True, cache=True)
def _sample(logits, temperature, u):
    if temperature <= 0:
        return np.argmax(logits)
    m = logits.max()
    p = np.exp((logits - m) / temperature)
    c = np.cumsum(p)
    r = u * c[-1]
    for i in range(c.shape[0]):
        if c[i] >= r:
            return i
    return c.shape[0] - 1


@numba.njit(fastmath=True, cache=True)
def _generate(prompt, n_new, temperature, seed, out, emb, n1, n2, wqkv, sqkv, wo, so, wg, sg, wu, su, wd, sd,
              nf, head, cos, sin, heads, ctx, use_packed, pqkv, po, pg, pu, pd):
    d = emb.shape[1]
    L = wqkv.shape[0]
    hidden = wg.shape[1]
    kc = np.zeros((L, ctx, d), np.float32)
    vc = np.zeros((L, ctx, d), np.float32)
    logits = np.empty(head.shape[0], np.float32)
    h = np.empty(d, np.float32)
    qkv = np.empty(3 * d, np.float32)
    att = np.empty(d, np.float32)
    tmp = np.empty(d, np.float32)
    g = np.empty(hidden, np.float32)
    u = np.empty(hidden, np.float32)
    scores = np.empty(ctx, np.float32)
    lut_d = np.empty(((d + 3) // 4, 256), np.float32)
    lut_h = np.empty(((hidden + 3) // 4, 256), np.float32)
    np.random.seed(seed)
    start = prompt.shape[0] - ctx if prompt.shape[0] > ctx else 0
    pos = 0
    for i in range(start, prompt.shape[0]):
        _step(prompt[i], pos, kc, vc, emb, n1, n2, wqkv, sqkv, wo, so, wg, sg, wu, su, wd, sd, nf, head, cos, sin,
              heads, logits, h, qkv, att, tmp, g, u, scores, use_packed, pqkv, po, pg, pu, pd, lut_d, lut_h)
        pos += 1
    for k in range(n_new):
        tok = _sample(logits, temperature, np.random.random())
        out[k] = tok
        _step(tok, pos, kc, vc, emb, n1, n2, wqkv, sqkv, wo, so, wg, sg, wu, su, wd, sd, nf, head, cos, sin,
              heads, logits, h, qkv, att, tmp, g, u, scores, use_packed, pqkv, po, pg, pu, pd, lut_d, lut_h)
        pos += 1


@numba.njit(fastmath=True, cache=True, parallel=True)
def _generate_batch(prompts, n_new, temperature, seed, out, emb, n1, n2, wqkv, sqkv, wo, so, wg, sg, wu, su, wd,
                    sd, nf, head, cos, sin, heads, ctx, use_packed, pqkv, po, pg, pu, pd):
    for b in prange(prompts.shape[0]):
        _generate(prompts[b], n_new, temperature, seed + b, out[b], emb, n1, n2, wqkv, sqkv, wo, so, wg, sg, wu,
                  su, wd, sd, nf, head, cos, sin, heads, ctx, use_packed, pqkv, po, pg, pu, pd)


@numba.njit(fastmath=True, cache=True)
def _logits_all(seq, emb, n1, n2, wqkv, sqkv, wo, so, wg, sg, wu, su, wd, sd, nf, head, cos, sin, heads,
                use_packed, pqkv, po, pg, pu, pd):
    ctx = cos.shape[0]
    d = emb.shape[1]
    hidden = wg.shape[1]
    kc = np.zeros((wqkv.shape[0], ctx, d), np.float32)
    vc = np.zeros((wqkv.shape[0], ctx, d), np.float32)
    out = np.empty((seq.shape[0], head.shape[0]), np.float32)
    logits = np.empty(head.shape[0], np.float32)
    h = np.empty(d, np.float32)
    qkv = np.empty(3 * d, np.float32)
    att = np.empty(d, np.float32)
    tmp = np.empty(d, np.float32)
    g = np.empty(hidden, np.float32)
    u = np.empty(hidden, np.float32)
    scores = np.empty(ctx, np.float32)
    lut_d = np.empty(((d + 3) // 4, 256), np.float32)
    lut_h = np.empty(((hidden + 3) // 4, 256), np.float32)
    for p in range(seq.shape[0]):
        _step(seq[p], p, kc, vc, emb, n1, n2, wqkv, sqkv, wo, so, wg, sg, wu, su, wd, sd, nf, head, cos, sin, heads,
              logits, h, qkv, att, tmp, g, u, scores, use_packed, pqkv, po, pg, pu, pd, lut_d, lut_h)
        out[p] = logits
    return out


_DUMMY_PACK = np.zeros((1, 1, 1), dtype=np.uint8)


class Engine:
    """Unified Native CPU Engine supporting arch v0 (int8/2-bit packed), v1 (ACT+MTP), and v2 (Patched Fast-Weight Copy)."""

    def __init__(self, weights: Weights | V1Weights | V2Weights, packed: bool = True):
        self.w = weights
        self.arch = weights.cfg.arch
        self.packed = bool(packed and weights.cfg.ternary)
        hd = weights.cfg.d // weights.cfg.heads
        self._inv = 1.0 / 10000.0 ** (np.arange(0, hd, 2, dtype=np.float64) / hd)
        self.last_stats: dict[str, float | int | str] = {}
        # N2 adaptive speculation gate.  The instrument disables it to measure a
        # pure speculative run; production keeps it on.
        self.spec_adaptive = True

    def _rope(self, max_pos: int) -> tuple[np.ndarray, np.ndarray]:
        if max_pos <= self.w.cos.shape[0]:
            return self.w.cos, self.w.sin
        ang = np.outer(np.arange(max_pos, dtype=np.float64), self._inv)
        return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

    @classmethod
    def load(cls, path, packed: bool = True) -> "Engine":
        import torch

        ck = torch.load(path, map_location="cpu", weights_only=True)
        cfg = CoreConfig(**ck["config"])
        st = ck["state"]
        if cfg.arch == "v0":
            return cls(export(path), packed=packed)
        if cfg.arch == "v1":
            return cls(export_v1(cfg, st), packed=packed)
        if cfg.arch == "v2":
            return cls(export_v2(cfg, st), packed=packed)
        raise ValueError(f"unsupported arch={cfg.arch!r}")

    def _v0_pack_tuple(self):
        w = self.w
        assert isinstance(w, Weights)
        if self.packed and w.pqkv is not None:
            return True, w.pqkv, w.po, w.pg, w.pu, w.pd
        return False, _DUMMY_PACK, _DUMMY_PACK, _DUMMY_PACK, _DUMMY_PACK, _DUMMY_PACK

    def logits(self, seq: bytes, early_exit: bool = True) -> np.ndarray:
        arr = np.frombuffer(seq, np.uint8).astype(np.int64)
        if self.arch == "v0":
            cos, sin = self._rope(len(seq) + 1)
            args = (*self.w.args()[:-2], cos, sin)
            return _logits_all(arr, *args, self.w.cfg.heads, *self._v0_pack_tuple())
        if self.arch == "v1":
            w = self.w
            assert isinstance(w, V1Weights)
            cos, sin = self._rope(len(seq) + 1)
            out, avg_l = _v1_logits_all(
                arr, early_exit, w.is_ternary, w.cfg.n_loops, w.cfg.heads, np.float32(w.cfg.halt_eps),
                w.emb, w.n1, w.n2, w.wqkv, w.pqkv, w.sqkv, w.wo, w.po, w.so,
                w.wg, w.pg, w.sg, w.wu, w.pu, w.su, w.wd, w.pd, w.sd,
                w.loop_emb, w.anchor_w, w.anchor_b, w.halt_norm, w.halt_w, w.halt_b,
                w.nf, w.head, cos, sin,
            )
            self.last_stats = {"arch": "v1", "avg_loops": round(avg_l, 4)}
            return out
        w = self.w
        assert isinstance(w, V2Weights)
        ps = max(2, w.cfg.patch_size)
        cos, sin = self._rope(max(w.cos.shape[0], (len(seq) // ps) + 8))
        out, _ = _v2_run(
            arr, 0, 0.0, 0, early_exit, True,
            ps, max(256, w.cfg.hash_buckets), max(1, w.cfg.n_loops), w.cfg.heads, w.cfg.fast_dim,
            w.cfg.copy_window, np.float32(w.cfg.halt_eps), w.is_ternary, w.use_fw, w.use_copy_gate,
            w.emb, w.emb_h2, w.emb_h3, w.emb_h4, w.ngram_proj, w.stem_norm, w.dw_w, w.dw_b, w.dw_proj,
            w.patch_scorer_w, w.patch_scorer_b, w.patch_proj, w.bos_thought,
            w.n1, w.n2, w.wqkv, w.pqkv, w.sqkv, w.wo, w.po, w.so, w.wg, w.pg, w.sg, w.wu, w.pu, w.su, w.wd, w.pd, w.sd,
            w.n_fw, w.fw_wqkv, w.fw_pqkv, w.fw_sqkv, w.fw_gates_w, w.fw_gates_b, w.fw_wo, w.fw_po, w.fw_so,
            w.loop_emb, w.anchor_w, w.anchor_b, w.halt_norm, w.halt_w, w.halt_b,
            w.offset_emb, w.cross_gate_w, w.cross_gate_b, w.fine_norm,
            w.fine_gate_w, w.fine_gate_p, w.fine_gate_s, w.fine_up_w, w.fine_up_p, w.fine_up_s,
            w.fine_down_w, w.fine_down_p, w.fine_down_s, w.norm, w.head, w.copy_w, w.copy_b,
            w.mtp_norms, w.mtp_ws, w.mtp_count, False, False, np.zeros(6, np.int64), cos, sin,
        )
        return out

    def generate(self, prompt: bytes, n: int, temperature: float = 0.8, seed: int = 0,
                 speculative: bool = False) -> bytes:
        # EXP-012 N2: native v2 speculation is lossless but 0.87x-0.92x on the
        # trained checkpoints, so it is off by default. Measurement instruments
        # pass speculative=True explicitly; callers can opt in.
        arr = np.frombuffer(prompt, np.uint8).astype(np.int64)
        ctx = self.w.cfg.ctx
        if self.arch == "v0":
            cos, sin = self._rope(min(len(prompt), ctx) + n + 1)
            args = (*self.w.args()[:-2], cos, sin)
            out = np.empty(n, np.int64)
            _generate(arr, n, temperature, seed, out, *args, self.w.cfg.heads, ctx, *self._v0_pack_tuple())
            return bytes(out.astype(np.uint8).tolist())
        if self.arch == "v1":
            w = self.w
            assert isinstance(w, V1Weights)
            cos, sin = self._rope(min(len(prompt), ctx) + n + 4)
            out, avg_l, td, ad = _v1_generate(
                arr, n, temperature, seed, speculative,
                w.is_ternary, w.cfg.n_loops, w.cfg.heads, np.float32(w.cfg.halt_eps),
                w.emb, w.n1, w.n2, w.wqkv, w.pqkv, w.sqkv, w.wo, w.po, w.so,
                w.wg, w.pg, w.sg, w.wu, w.pu, w.su, w.wd, w.pd, w.sd,
                w.loop_emb, w.anchor_w, w.anchor_b, w.halt_norm, w.halt_w, w.halt_b,
                w.nf, w.head, w.mtp_norms, w.mtp_ws, w.mtp_count, cos, sin,
            )
            self.last_stats = {
                "arch": "v1", "avg_loops": round(avg_l, 4),
                "speculative_drafts": int(td), "speculative_accepted": int(ad),
                "spec_accept_rate": round(ad / max(1, td), 4) if td else 0.0,
            }
            return bytes(out.astype(np.uint8).tolist())
        w = self.w
        assert isinstance(w, V2Weights)
        ps = max(2, w.cfg.patch_size)
        cos, sin = self._rope(max(w.cos.shape[0], ((len(prompt) + n) // ps) + 8))
        stats = np.zeros(6, np.int64)
        _, out = _v2_run(
            arr, n, temperature, seed, True, False,
            ps, max(256, w.cfg.hash_buckets), max(1, w.cfg.n_loops), w.cfg.heads, w.cfg.fast_dim,
            w.cfg.copy_window, np.float32(w.cfg.halt_eps), w.is_ternary, w.use_fw, w.use_copy_gate,
            w.emb, w.emb_h2, w.emb_h3, w.emb_h4, w.ngram_proj, w.stem_norm, w.dw_w, w.dw_b, w.dw_proj,
            w.patch_scorer_w, w.patch_scorer_b, w.patch_proj, w.bos_thought,
            w.n1, w.n2, w.wqkv, w.pqkv, w.sqkv, w.wo, w.po, w.so, w.wg, w.pg, w.sg, w.wu, w.pu, w.su, w.wd, w.pd, w.sd,
            w.n_fw, w.fw_wqkv, w.fw_pqkv, w.fw_sqkv, w.fw_gates_w, w.fw_gates_b, w.fw_wo, w.fw_po, w.fw_so,
            w.loop_emb, w.anchor_w, w.anchor_b, w.halt_norm, w.halt_w, w.halt_b,
            w.offset_emb, w.cross_gate_w, w.cross_gate_b, w.fine_norm,
            w.fine_gate_w, w.fine_gate_p, w.fine_gate_s, w.fine_up_w, w.fine_up_p, w.fine_up_s,
            w.fine_down_w, w.fine_down_p, w.fine_down_s, w.norm, w.head, w.copy_w, w.copy_b,
            w.mtp_norms, w.mtp_ws, w.mtp_count, speculative, self.spec_adaptive,
            stats, cos, sin,
        )
        self.last_stats = {
            "arch": "v2",
            "speculative_drafts": int(stats[0]),
            "speculative_accepted": int(stats[1]),
            "spec_rounds": int(stats[2]),
            "spec_accept_rate": round(float(stats[1]) / max(1, int(stats[0])), 4),
            "spec_batch_forwards": int(stats[4]),
            "spec_first_rejections": int(stats[5]),
            "spec_disabled": bool(stats[3]),
        }
        return bytes(out[:n].astype(np.uint8).tolist())

    def generate_batch(self, prompts: list[bytes], n: int, temperature: float = 0.8, seed: int = 0,
                       speculative: bool = False) -> list[bytes]:
        if self.arch == "v0":
            ctx = self.w.cfg.ctx
            L = min(len(p) for p in prompts)
            arr = (np.stack([np.frombuffer(p[-L:], np.uint8).astype(np.int64) for p in prompts])
                   if L > 0 else np.zeros((len(prompts), 0), np.int64))
            out = np.empty((len(prompts), n), np.int64)
            cos, sin = self._rope(min(L, ctx) + n + 1)
            args = (*self.w.args()[:-2], cos, sin)
            _generate_batch(arr, n, temperature, seed, out, *args, self.w.cfg.heads, ctx, *self._v0_pack_tuple())
            return [bytes(r.astype(np.uint8).tolist()) for r in out]
        # v1 & v2 release the GIL inside nogil=True Numba kernels -> true multi-core parallelism without Parfors compile cost
        workers = min(len(prompts), os.cpu_count() or 4)
        if workers <= 1:
            return [self.generate(p, n, temperature=temperature, seed=seed + i, speculative=speculative)
                    for i, p in enumerate(prompts)]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(self.generate, p, n, temperature=temperature, seed=seed + i, speculative=speculative)
                for i, p in enumerate(prompts)
            ]
            return [f.result() for f in futs]

    def bench(self, n_new: int = 1000, batch: int = 8, prompt: bytes = b"def ",
              temperature: float = 0.0, speculative: bool = True) -> dict:
        self.generate(prompt, 8, temperature=temperature, speculative=speculative)
        self.generate_batch([prompt] * batch, 8, temperature=temperature, speculative=speculative)
        t0 = time.perf_counter()
        self.generate(prompt, 1, temperature=temperature, speculative=speculative)
        ttft_ms = (time.perf_counter() - t0) * 1000.0
        t = time.perf_counter()
        self.generate(prompt, n_new, temperature=temperature, speculative=speculative)
        single = n_new / (time.perf_counter() - t)
        t = time.perf_counter()
        self.generate_batch([prompt] * batch, n_new, temperature=temperature, speculative=speculative)
        agg = batch * n_new / (time.perf_counter() - t)
        block_bytes = self.w.bytes_packed() if hasattr(self.w, "bytes_packed") else 0
        return {
            "arch": self.arch,
            "ternary": bool(self.w.cfg.ternary),
            "packed_lut": self.packed,
            "single_stream_tok_s": round(single, 1),
            f"aggregate_tok_s_batch{batch}": round(agg, 1),
            "ttft_ms": round(ttft_ms, 2),
            "threads": numba.get_num_threads(),
            "packed_block_bytes": int(block_bytes),
            "n_new": n_new,
            "stats": dict(self.last_stats),
        }
