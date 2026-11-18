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

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numba
import numpy as np
from numba import prange

from .model import CoreConfig, ternarize


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

    def args(self):
        return (self.emb, self.n1, self.n2, self.wqkv, self.sqkv, self.wo, self.so, self.wg, self.sg, self.wu,
                self.su, self.wd, self.sd, self.nf, self.head, self.cos, self.sin)

    def bytes_block(self) -> int:
        return sum(a.nbytes for a in (self.wqkv, self.wo, self.wg, self.wu, self.wd))

    def bytes_total(self) -> int:
        return sum(a.nbytes for a in self.args())


def export(path: str | Path) -> Weights:
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=True)  # never unpickle arbitrary objects
    cfg = CoreConfig(**ck["config"])
    if cfg.arch != "v0":
        raise ValueError(f"native CPU engine supports arch='v0'; got arch={cfg.arch!r} (use PyTorch model directly)")
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
    hd = cfg.d // cfg.heads
    inv = 1.0 / 10000.0 ** (np.arange(0, hd, 2, dtype=np.float64) / hd)
    ang = np.outer(np.arange(cfg.ctx, dtype=np.float64), inv)
    f = lambda k: st[k].float().numpy().astype(np.float32)  # noqa: E731
    return Weights(cfg, f("emb.weight"), np.stack([f(f"blocks.{i}.n1.weight") for i in range(cfg.layers)]),
                   np.stack([f(f"blocks.{i}.n2.weight") for i in range(cfg.layers)]), wqkv, sqkv, wo, so, wg, sg,
                   wu, su, wd, sd, f("norm.weight"), f("head.weight"), np.cos(ang).astype(np.float32),
                   np.sin(ang).astype(np.float32))


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
          heads, logits, h, qkv, att, tmp, g, u, scores):
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
        _matvec(wo[l], so[l], att, tmp)
        for i in range(d):
            x[i] += tmp[i]
        _rmsnorm(x, n2[l], h)
        _matvec(wg[l], sg[l], h, g)
        _matvec(wu[l], su[l], h, u)
        for i in range(hidden):
            gi = g[i]
            g[i] = gi / (np.float32(1.0) + np.exp(-gi)) * u[i]
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
              nf, head, cos, sin, heads, ctx):
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
    np.random.seed(seed)
    start = prompt.shape[0] - ctx if prompt.shape[0] > ctx else 0
    pos = 0
    for i in range(start, prompt.shape[0]):
        _step(prompt[i], pos, kc, vc, emb, n1, n2, wqkv, sqkv, wo, so, wg, sg, wu, su, wd, sd, nf, head, cos, sin,
              heads, logits, h, qkv, att, tmp, g, u, scores)
        pos += 1
    for k in range(n_new):
        tok = _sample(logits, temperature, np.random.random())
        out[k] = tok
        _step(tok, pos, kc, vc, emb, n1, n2, wqkv, sqkv, wo, so, wg, sg, wu, su, wd, sd, nf, head, cos, sin,
              heads, logits, h, qkv, att, tmp, g, u, scores)
        pos += 1


@numba.njit(fastmath=True, cache=True, parallel=True)
def _generate_batch(prompts, n_new, temperature, seed, out, emb, n1, n2, wqkv, sqkv, wo, so, wg, sg, wu, su, wd,
                    sd, nf, head, cos, sin, heads, ctx):
    for b in prange(prompts.shape[0]):
        _generate(prompts[b], n_new, temperature, seed + b, out[b], emb, n1, n2, wqkv, sqkv, wo, so, wg, sg, wu,
                  su, wd, sd, nf, head, cos, sin, heads, ctx)


@numba.njit(fastmath=True, cache=True)
def _logits_all(seq, emb, n1, n2, wqkv, sqkv, wo, so, wg, sg, wu, su, wd, sd, nf, head, cos, sin, heads):
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
    for p in range(seq.shape[0]):
        _step(seq[p], p, kc, vc, emb, n1, n2, wqkv, sqkv, wo, so, wg, sg, wu, su, wd, sd, nf, head, cos, sin, heads,
              logits, h, qkv, att, tmp, g, u, scores)
        out[p] = logits
    return out


class Engine:
    def __init__(self, weights: Weights):
        self.w = weights
        hd = weights.cfg.d // weights.cfg.heads
        self._inv = 1.0 / 10000.0 ** (np.arange(0, hd, 2, dtype=np.float64) / hd)

    def _rope(self, max_pos: int) -> tuple[np.ndarray, np.ndarray]:
        if max_pos <= self.w.cos.shape[0]:
            return self.w.cos, self.w.sin
        ang = np.outer(np.arange(max_pos, dtype=np.float64), self._inv)
        return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

    @classmethod
    def load(cls, path) -> "Engine":
        return cls(export(path))

    def logits(self, seq: bytes) -> np.ndarray:
        cos, sin = self._rope(len(seq) + 1)
        args = (*self.w.args()[:-2], cos, sin)
        return _logits_all(np.frombuffer(seq, np.uint8).astype(np.int64), *args, self.w.cfg.heads)

    def generate(self, prompt: bytes, n: int, temperature: float = 0.8, seed: int = 0) -> bytes:
        ctx = self.w.cfg.ctx
        cos, sin = self._rope(min(len(prompt), ctx) + n + 1)
        args = (*self.w.args()[:-2], cos, sin)
        out = np.empty(n, np.int64)
        _generate(np.frombuffer(prompt, np.uint8).astype(np.int64), n, temperature, seed, out, *args,
                  self.w.cfg.heads, ctx)
        return bytes(out.astype(np.uint8).tolist())

    def generate_batch(self, prompts: list[bytes], n: int, temperature: float = 0.8, seed: int = 0) -> list[bytes]:
        ctx = self.w.cfg.ctx
        L = min(len(p) for p in prompts)
        arr = (np.stack([np.frombuffer(p[-L:], np.uint8).astype(np.int64) for p in prompts])
               if L > 0 else np.zeros((len(prompts), 0), np.int64))
        cos, sin = self._rope(min(L, ctx) + n + 1)
        args = (*self.w.args()[:-2], cos, sin)
        out = np.empty((len(prompts), n), np.int64)
        _generate_batch(arr, n, temperature, seed, out, *args, self.w.cfg.heads, ctx)
        return [bytes(r.astype(np.uint8).tolist()) for r in out]

    def bench(self, n_new: int = 2000, batch: int = 14, prompt: bytes = b"def ") -> dict:
        self.generate(prompt, 8)
        self.generate_batch([prompt] * batch, 8)
        t = time.perf_counter()
        self.generate(prompt, n_new, temperature=0.8)
        single = n_new / (time.perf_counter() - t)
        t = time.perf_counter()
        self.generate_batch([prompt] * batch, n_new, temperature=0.8)
        agg = batch * n_new / (time.perf_counter() - t)
        return {"single_stream_tok_s": round(single), f"aggregate_tok_s_batch{batch}": round(agg),
                "threads": numba.get_num_threads(), "block_weight_bytes": self.w.bytes_block(),
                "total_weight_bytes": self.w.bytes_total(), "n_new": n_new,
                "note": "tokens are bytes; rolling ring KV cache + row4-unrolled matvec"}
