"""Native Numba CPU inference engines for OmegaCore v1 (Looped ACT + MTP) & v2 (Patched Fast-Weight CopyCore).

Implements modular <=24-arg Numba kernels (fast LLVM compile, zero CALL_FUNCTION_EX):
  * 2-bit ternary packed T-MAC LUT matvec (4x smaller than int8, 16x smaller than fp32)
  * Real adaptive halting (ACT early-exit) with lazy KV backfill for exact PyTorch identity
  * Exact lossless Self-Speculative Multi-Token (MTP) batched verification for v1
  * Incremental O(1)-state streaming for v2: Causal Hash-Ngram Stem -> Depthwise Conv ->
    Strided Thought Patches -> Ring-KV Attention + DeltaFastWeights -> Causal CopyGate
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numba
import numpy as np

from .packed import matvec_fp32, matvec_packed_batch, matvec_packed_lut, pack_ternary

if TYPE_CHECKING:
    from .model import CoreConfig


@numba.njit(fastmath=True, cache=True, nogil=True)
def _lin(is_ternary: bool, wf: np.ndarray, wp: np.ndarray, s: np.float32,
         x: np.ndarray, lut: np.ndarray, out: np.ndarray) -> None:
    if is_ternary:
        matvec_packed_lut(wp, s, x, x.shape[0], lut, out)
    else:
        matvec_fp32(wf, s, x, out)


@numba.njit(fastmath=True, cache=True, nogil=True)
def _lin_bias(wf: np.ndarray, bf: np.ndarray, x: np.ndarray, out: np.ndarray) -> None:
    O = wf.shape[0]
    I = wf.shape[1]
    for o in range(O):
        acc = bf[o]
        for j in range(I):
            acc += wf[o, j] * x[j]
        out[o] = acc


@numba.njit(fastmath=True, cache=True, nogil=True)
def _rmsnorm(x: np.ndarray, w: np.ndarray, out: np.ndarray) -> None:
    ss = np.float32(0.0)
    for i in range(x.shape[0]):
        ss += x[i] * x[i]
    r = np.float32(1.0) / np.sqrt(ss / x.shape[0] + np.float32(1e-6))
    for i in range(x.shape[0]):
        out[i] = x[i] * r * w[i]


@numba.njit(fastmath=True, cache=True, nogil=True)
def _sample(logits: np.ndarray, temperature: float, u: float) -> int:
    if temperature <= 0.0:
        return int(np.argmax(logits))
    m = logits.max()
    p = np.exp((logits - m) / temperature)
    c = np.cumsum(p)
    r = u * c[-1]
    for i in range(c.shape[0]):
        if c[i] >= r:
            return i
    return c.shape[0] - 1


def _export_linear(st: dict, key: str, ternary: bool):
    w = st[key].float().numpy().astype(np.float32)
    if ternary:
        t = st[key].float()
        scale = float(t.abs().mean().clamp(min=1e-8).item())
        codes = (t / scale).round().clamp(-1, 1).to(t.dtype).numpy().astype(np.int8)
        wp, _ = pack_ternary(codes)
        return w[:1, :1].copy(), wp, np.float32(scale)
    return np.ascontiguousarray(w), np.zeros((1, 1), dtype=np.uint8), np.float32(1.0)


def _stack_linears(st: dict, count: int, fmt: str, ternary: bool):
    wfs, wps, ss = [], [], []
    for i in range(count):
        wf, wp, s = _export_linear(st, fmt.format(i), ternary)
        wfs.append(wf)
        wps.append(wp)
        ss.append(s)
    return (
        np.ascontiguousarray(np.stack(wfs)),
        np.ascontiguousarray(np.stack(wps)),
        np.array(ss, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Shared Attention + SwiGLU Kernels (small arg lists, zero duplication)
# ---------------------------------------------------------------------------

@numba.njit(fastmath=True, cache=True, nogil=True)
def _attn_loop_step(
    is_ternary: bool, l: int, t: int, pos: int, valid_len: int, heads: int, ctx: int, inv_sqrt: np.float32,
    x: np.ndarray, kc: np.ndarray, vc: np.ndarray,
    n1: np.ndarray, wqkv: np.ndarray, pqkv: np.ndarray, sqkv: np.ndarray,
    wo: np.ndarray, po: np.ndarray, so: np.ndarray, cos: np.ndarray, sin: np.ndarray,
    h: np.ndarray, qkv: np.ndarray, att: np.ndarray, tmp: np.ndarray, scores: np.ndarray, lut_d: np.ndarray,
) -> None:
    d = x.shape[0]
    hd = d // heads
    half = hd // 2
    slot = pos % ctx

    _rmsnorm(x, n1[l], h)
    _lin(is_ternary, wqkv[l], pqkv[l], sqkv[l], h, lut_d, qkv)
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
            kc[t, l, slot, b + 2 * i] = k1 * c - k2 * s
            kc[t, l, slot, b + 2 * i + 1] = k1 * s + k2 * c
    for i in range(d):
        vc[t, l, slot, i] = qkv[2 * d + i]

    for hh in range(heads):
        b = hh * hd
        m = np.float32(-1e30)
        for j in range(valid_len):
            js = j % ctx
            acc = np.float32(0.0)
            for i in range(hd):
                acc += qkv[b + i] * kc[t, l, js, b + i]
            acc *= inv_sqrt
            scores[j] = acc
            if acc > m:
                m = acc
        z = np.float32(0.0)
        for j in range(valid_len):
            e = np.exp(scores[j] - m)
            scores[j] = e
            z += e
        inv_z = np.float32(1.0) / z
        for i in range(hd):
            att[b + i] = np.float32(0.0)
        for j in range(valid_len):
            js = j % ctx
            w_att = scores[j] * inv_z
            for i in range(hd):
                att[b + i] += w_att * vc[t, l, js, b + i]

    _lin(is_ternary, wo[l], po[l], so[l], att, lut_d, tmp)
    for i in range(d):
        x[i] += tmp[i]


@numba.njit(fastmath=True, cache=True, nogil=True)
def _swiglu_step(
    is_ternary: bool, l: int, x: np.ndarray, n2: np.ndarray,
    wg: np.ndarray, pg: np.ndarray, sg: np.ndarray,
    wu: np.ndarray, pu: np.ndarray, su: np.ndarray,
    wd: np.ndarray, pd: np.ndarray, sd: np.ndarray,
    h: np.ndarray, g: np.ndarray, u: np.ndarray, tmp: np.ndarray,
    lut_d: np.ndarray, lut_h: np.ndarray,
) -> None:
    d = x.shape[0]
    hidden = u.shape[0]
    _rmsnorm(x, n2[l], h)
    _lin(is_ternary, wg[l], pg[l], sg[l], h, lut_d, g)
    _lin(is_ternary, wu[l], pu[l], su[l], h, lut_d, u)
    for i in range(hidden):
        gi = g[i]
        g[i] = gi / (np.float32(1.0) + np.exp(-gi)) * u[i]
    _lin(is_ternary, wd[l], pd[l], sd[l], g, lut_h, tmp)
    for i in range(d):
        x[i] += tmp[i]


# ---------------------------------------------------------------------------
# Arch v1: LoopedOmegaCore (ACT Early-Exit + Self-Speculative MTP)
# ---------------------------------------------------------------------------

@dataclass
class V1Weights:
    cfg: CoreConfig
    is_ternary: bool
    emb: np.ndarray
    n1: np.ndarray
    n2: np.ndarray
    wqkv: np.ndarray
    pqkv: np.ndarray
    sqkv: np.ndarray
    wo: np.ndarray
    po: np.ndarray
    so: np.ndarray
    wg: np.ndarray
    pg: np.ndarray
    sg: np.ndarray
    wu: np.ndarray
    pu: np.ndarray
    su: np.ndarray
    wd: np.ndarray
    pd: np.ndarray
    sd: np.ndarray
    loop_emb: np.ndarray
    anchor_w: np.ndarray
    anchor_b: np.ndarray
    halt_norm: np.ndarray
    halt_w: np.ndarray
    halt_b: np.ndarray
    nf: np.ndarray
    head: np.ndarray
    mtp_norms: np.ndarray
    mtp_ws: np.ndarray
    mtp_count: int
    cos: np.ndarray
    sin: np.ndarray

    def bytes_block_fp32(self) -> int:
        L = self.cfg.layers
        d, hid = self.cfg.d, self.cfg.hidden
        return 4 * L * (3 * d * d + d * d + 3 * hid * d)

    def bytes_packed(self) -> int:
        if self.is_ternary:
            return sum(a.nbytes for a in (self.pqkv, self.po, self.pg, self.pu, self.pd))
        return sum(a.nbytes for a in (self.wqkv, self.wo, self.wg, self.wu, self.wd))


def export_v1(cfg: CoreConfig, st: dict) -> V1Weights:
    f = lambda k: np.ascontiguousarray(st[k].float().numpy().astype(np.float32))
    L = cfg.layers
    wqkv, pqkv, sqkv = _stack_linears(st, L, "blocks.{}.qkv.weight", cfg.ternary)
    wo, po, so = _stack_linears(st, L, "blocks.{}.proj.weight", cfg.ternary)
    wg, pg, sg = _stack_linears(st, L, "blocks.{}.gate.weight", cfg.ternary)
    wu, pu, su = _stack_linears(st, L, "blocks.{}.up.weight", cfg.ternary)
    wd, pd, sd = _stack_linears(st, L, "blocks.{}.down.weight", cfg.ternary)
    n1 = np.ascontiguousarray(np.stack([f(f"blocks.{i}.n1.weight") for i in range(L)]))
    n2 = np.ascontiguousarray(np.stack([f(f"blocks.{i}.n2.weight") for i in range(L)]))

    M = max(0, cfg.mtp_k - 1)
    if M > 0:
        mtp_norms = np.ascontiguousarray(np.stack([f(f"mtp_projs.{k}.0.weight") for k in range(M)]))
        mtp_ws = np.ascontiguousarray(np.stack([f(f"mtp_projs.{k}.1.weight") for k in range(M)]))
    else:
        mtp_norms = np.zeros((1, cfg.d), dtype=np.float32)
        mtp_ws = np.zeros((1, cfg.d, cfg.d), dtype=np.float32)

    hd = cfg.d // cfg.heads
    inv = 1.0 / 10000.0 ** (np.arange(0, hd, 2, dtype=np.float64) / hd)
    ang = np.outer(np.arange(cfg.ctx, dtype=np.float64), inv)
    return V1Weights(
        cfg=cfg, is_ternary=bool(cfg.ternary), emb=f("emb.weight"), n1=n1, n2=n2,
        wqkv=wqkv, pqkv=pqkv, sqkv=sqkv, wo=wo, po=po, so=so,
        wg=wg, pg=pg, sg=sg, wu=wu, pu=pu, su=su, wd=wd, pd=pd, sd=sd,
        loop_emb=f("loop_emb"), anchor_w=f("anchor_gate.weight"), anchor_b=f("anchor_gate.bias"),
        halt_norm=f("halt_norm.weight"), halt_w=f("halt_head.weight"), halt_b=f("halt_head.bias"),
        nf=f("norm.weight"), head=f("head.weight"), mtp_norms=mtp_norms, mtp_ws=mtp_ws,
        mtp_count=M, cos=np.cos(ang).astype(np.float32), sin=np.sin(ang).astype(np.float32),
    )


@numba.njit(fastmath=True, cache=True, nogil=True)
def _v1_prep_anchor(x: np.ndarray, x0: np.ndarray, loop_v: np.ndarray,
                    anchor_w: np.ndarray, anchor_b: np.ndarray,
                    cat2d: np.ndarray, gate: np.ndarray, out: np.ndarray) -> None:
    d = x.shape[0]
    for i in range(d):
        cat2d[i] = x[i]
        cat2d[d + i] = x0[i]
    _lin_bias(anchor_w, anchor_b, cat2d, gate)
    for i in range(d):
        gs = np.float32(1.0) / (np.float32(1.0) + np.exp(-gate[i]))
        out[i] = x[i] + gs * x0[i] + loop_v[i]


@numba.njit(fastmath=True, cache=True, nogil=True)
def _v1_logits_all(
    seq: np.ndarray, early_exit: bool, is_ternary: bool, n_loops: int, heads: int, halt_eps: np.float32,
    emb: np.ndarray, n1: np.ndarray, n2: np.ndarray,
    wqkv: np.ndarray, pqkv: np.ndarray, sqkv: np.ndarray,
    wo: np.ndarray, po: np.ndarray, so: np.ndarray,
    wg: np.ndarray, pg: np.ndarray, sg: np.ndarray,
    wu: np.ndarray, pu: np.ndarray, su: np.ndarray,
    wd: np.ndarray, pd: np.ndarray, sd: np.ndarray,
    loop_emb: np.ndarray, anchor_w: np.ndarray, anchor_b: np.ndarray,
    halt_norm: np.ndarray, halt_w: np.ndarray, halt_b: np.ndarray,
    nf: np.ndarray, head: np.ndarray, cos: np.ndarray, sin: np.ndarray,
):
    T = seq.shape[0]
    L = wqkv.shape[0]
    ctx = cos.shape[0]
    d = emb.shape[1]
    hidden = wg.shape[1] if not is_ternary else pg.shape[1]
    inv_sqrt = np.float32(1.0 / math.sqrt(d // heads))

    kc = np.zeros((n_loops, L, ctx, d), np.float32)
    vc = np.zeros((n_loops, L, ctx, d), np.float32)
    x0_hist = np.zeros((ctx, d), np.float32)
    xf_hist = np.zeros((ctx, d), np.float32)
    kv_ready = np.full(ctx, -1, np.int32)

    out = np.empty((T, head.shape[0]), np.float32)
    logits = np.empty(head.shape[0], np.float32)
    out_state = np.empty(d, np.float32)
    x = np.empty(d, np.float32)
    cat2d = np.empty(2 * d, np.float32)
    gate = np.empty(d, np.float32)
    xtmp = np.empty(d, np.float32)
    h = np.empty(d, np.float32)
    qkv = np.empty(3 * d, np.float32)
    att = np.empty(d, np.float32)
    tmp = np.empty(d, np.float32)
    g = np.empty(hidden, np.float32)
    u = np.empty(hidden, np.float32)
    scores = np.empty(ctx, np.float32)
    lut_d = np.empty(((d + 3) // 4, 256), np.float32)
    lut_h = np.empty(((hidden + 3) // 4, 256), np.float32)
    halt_buf = np.empty(1, np.float32)

    ponder_sum = 0.0
    for pos in range(T):
        tok = int(seq[pos])
        valid_len = pos + 1 if pos < ctx else ctx
        slot = pos % ctx
        for i in range(d):
            v = emb[tok, i]
            x[i] = v
            x0_hist[slot, i] = v
            out_state[i] = np.float32(0.0)
        rem = np.float32(1.0)
        ponder = np.float32(0.0)
        start_p = pos - valid_len + 1

        for t in range(n_loops):
            for pp in range(start_p, pos):
                ps = pp % ctx
                if kv_ready[ps] < t:
                    vl_p = pp + 1 if pp < ctx else ctx
                    _v1_prep_anchor(xf_hist[ps], x0_hist[ps], loop_emb[t], anchor_w, anchor_b, cat2d, gate, xtmp)
                    for l in range(L):
                        _attn_loop_step(is_ternary, l, t, pp, vl_p, heads, ctx, inv_sqrt, xtmp, kc, vc,
                                        n1, wqkv, pqkv, sqkv, wo, po, so, cos, sin, h, qkv, att, tmp, scores, lut_d)
                        _swiglu_step(is_ternary, l, xtmp, n2, wg, pg, sg, wu, pu, su, wd, pd, sd, h, g, u, tmp, lut_d, lut_h)
                    kv_ready[ps] = t

            _v1_prep_anchor(x, x0_hist[slot], loop_emb[t], anchor_w, anchor_b, cat2d, gate, x)
            for l in range(L):
                _attn_loop_step(is_ternary, l, t, pos, valid_len, heads, ctx, inv_sqrt, x, kc, vc,
                                n1, wqkv, pqkv, sqkv, wo, po, so, cos, sin, h, qkv, att, tmp, scores, lut_d)
                _swiglu_step(is_ternary, l, x, n2, wg, pg, sg, wu, pu, su, wd, pd, sd, h, g, u, tmp, lut_d, lut_h)
            kv_ready[slot] = t

            _rmsnorm(x, halt_norm, h)
            _lin_bias(halt_w, halt_b, h, halt_buf)
            p_halt = np.float32(1.0) / (np.float32(1.0) + np.exp(-halt_buf[0]))
            is_last = t == n_loops - 1
            q_w = rem if is_last else rem * p_halt
            rem = rem * (np.float32(1.0) - p_halt)
            for i in range(d):
                out_state[i] += q_w * x[i]
            ponder += np.float32(t + 1) * q_w
            if early_exit and not is_last:
                if (np.float32(1.0) - rem) >= (np.float32(1.0) - halt_eps):
                    for i in range(d):
                        out_state[i] += rem * x[i]
                    ponder += np.float32(t + 1) * rem
                    break

        for i in range(d):
            xf_hist[slot, i] = x[i]
        _rmsnorm(out_state, nf, h)
        matvec_fp32(head, np.float32(1.0), h, logits)
        out[pos] = logits
        ponder_sum += float(ponder)
    return out, (ponder_sum / max(1, T))


@numba.njit(fastmath=True, cache=True, nogil=True)
def _v1_generate(
    prompt: np.ndarray, n_new: int, temperature: float, seed: int, speculative: bool,
    is_ternary: bool, n_loops: int, heads: int, halt_eps: np.float32,
    emb: np.ndarray, n1: np.ndarray, n2: np.ndarray,
    wqkv: np.ndarray, pqkv: np.ndarray, sqkv: np.ndarray,
    wo: np.ndarray, po: np.ndarray, so: np.ndarray,
    wg: np.ndarray, pg: np.ndarray, sg: np.ndarray,
    wu: np.ndarray, pu: np.ndarray, su: np.ndarray,
    wd: np.ndarray, pd: np.ndarray, sd: np.ndarray,
    loop_emb: np.ndarray, anchor_w: np.ndarray, anchor_b: np.ndarray,
    halt_norm: np.ndarray, halt_w: np.ndarray, halt_b: np.ndarray,
    nf: np.ndarray, head: np.ndarray, mtp_norms: np.ndarray, mtp_ws: np.ndarray, mtp_count: int,
    cos: np.ndarray, sin: np.ndarray,
):
    L = wqkv.shape[0]
    ctx = cos.shape[0]
    d = emb.shape[1]
    V = head.shape[0]
    hd = d // heads
    half = hd // 2
    hidden = wg.shape[1] if not is_ternary else pg.shape[1]
    inv_sqrt = np.float32(1.0 / math.sqrt(hd))

    kc = np.zeros((n_loops, L, ctx, d), np.float32)
    vc = np.zeros((n_loops, L, ctx, d), np.float32)
    x0_hist = np.zeros((ctx, d), np.float32)
    xf_hist = np.zeros((ctx, d), np.float32)
    kv_ready = np.full(ctx, -1, np.int32)
    out = np.empty(n_new, np.int64)
    logits = np.empty(V, np.float32)
    draft_logits = np.empty(V, np.float32)
    out_state = np.empty(d, np.float32)
    x = np.empty(d, np.float32)
    cat2d = np.empty(2 * d, np.float32)
    gate = np.empty(d, np.float32)
    xtmp = np.empty(d, np.float32)
    h = np.empty(d, np.float32)
    proj_h = np.empty(d, np.float32)
    qkv = np.empty(3 * d, np.float32)
    att = np.empty(d, np.float32)
    tmp = np.empty(d, np.float32)
    g = np.empty(hidden, np.float32)
    u = np.empty(hidden, np.float32)
    scores = np.empty(ctx, np.float32)
    lut_d = np.empty(((d + 3) // 4, 256), np.float32)
    lut_h = np.empty(((hidden + 3) // 4, 256), np.float32)
    halt_buf = np.empty(1, np.float32)

    max_B = 1 + max(1, mtp_count)
    toks_b = np.empty(max_B, np.int64)
    logits_b = np.empty((max_B, V), np.float32)
    out_states_b = np.empty((max_B, d), np.float32)
    xb = np.empty((max_B, d), np.float32)
    x0b = np.empty((max_B, d), np.float32)
    hin_b = np.empty((max_B, d), np.float32)
    hb = np.empty((max_B, d), np.float32)
    qkv_b = np.empty((max_B, 3 * d), np.float32)
    att_b = np.empty((max_B, d), np.float32)
    tmp_b = np.empty((max_B, d), np.float32)
    gb = np.empty((max_B, hidden), np.float32)
    ub = np.empty((max_B, hidden), np.float32)
    luts_d = np.empty((max_B, (d + 3) // 4, 256), np.float32)
    luts_h = np.empty((max_B, (hidden + 3) // 4, 256), np.float32)
    rem_b = np.empty(max_B, np.float32)
    halted_b = np.empty(max_B, np.int32)
    drafts = np.empty(max(1, mtp_count), np.int64)

    np.random.seed(seed)
    start = prompt.shape[0] - ctx if prompt.shape[0] > ctx else 0
    pos = 0
    ponder_sum = 0.0
    steps = 0

    for i in range(start, prompt.shape[0]):
        tok = int(prompt[i])
        valid_len = pos + 1 if pos < ctx else ctx
        slot = pos % ctx
        for j in range(d):
            v = emb[tok, j]
            x[j] = v
            x0_hist[slot, j] = v
            out_state[j] = np.float32(0.0)
        rem = np.float32(1.0)
        ponder = np.float32(0.0)
        start_p = pos - valid_len + 1
        for t in range(n_loops):
            for pp in range(start_p, pos):
                ps = pp % ctx
                if kv_ready[ps] < t:
                    vl_p = pp + 1 if pp < ctx else ctx
                    _v1_prep_anchor(xf_hist[ps], x0_hist[ps], loop_emb[t], anchor_w, anchor_b, cat2d, gate, xtmp)
                    for l in range(L):
                        _attn_loop_step(is_ternary, l, t, pp, vl_p, heads, ctx, inv_sqrt, xtmp, kc, vc,
                                        n1, wqkv, pqkv, sqkv, wo, po, so, cos, sin, h, qkv, att, tmp, scores, lut_d)
                        _swiglu_step(is_ternary, l, xtmp, n2, wg, pg, sg, wu, pu, su, wd, pd, sd, h, g, u, tmp, lut_d, lut_h)
                    kv_ready[ps] = t
            _v1_prep_anchor(x, x0_hist[slot], loop_emb[t], anchor_w, anchor_b, cat2d, gate, x)
            for l in range(L):
                _attn_loop_step(is_ternary, l, t, pos, valid_len, heads, ctx, inv_sqrt, x, kc, vc,
                                n1, wqkv, pqkv, sqkv, wo, po, so, cos, sin, h, qkv, att, tmp, scores, lut_d)
                _swiglu_step(is_ternary, l, x, n2, wg, pg, sg, wu, pu, su, wd, pd, sd, h, g, u, tmp, lut_d, lut_h)
            kv_ready[slot] = t
            _rmsnorm(x, halt_norm, h)
            _lin_bias(halt_w, halt_b, h, halt_buf)
            p_halt = np.float32(1.0) / (np.float32(1.0) + np.exp(-halt_buf[0]))
            is_last = t == n_loops - 1
            q_w = rem if is_last else rem * p_halt
            rem = rem * (np.float32(1.0) - p_halt)
            for j in range(d):
                out_state[j] += q_w * x[j]
            ponder += np.float32(t + 1) * q_w
            if not is_last and (np.float32(1.0) - rem) >= (np.float32(1.0) - halt_eps):
                for j in range(d):
                    out_state[j] += rem * x[j]
                ponder += np.float32(t + 1) * rem
                break
        for j in range(d):
            xf_hist[slot, j] = x[j]
        _rmsnorm(out_state, nf, h)
        matvec_fp32(head, np.float32(1.0), h, logits)
        ponder_sum += float(ponder)
        pos += 1
        steps += 1

    k_idx = 0
    total_drafts = 0
    accepted_drafts = 0

    while k_idx < n_new:
        tok = _sample(logits, temperature, np.random.random())
        out[k_idx] = tok
        k_idx += 1
        if k_idx >= n_new:
            break

        if speculative and mtp_count > 0 and temperature <= 0.0 and (pos + 1 + mtp_count) <= cos.shape[0]:
            n_d = min(mtp_count, n_new - k_idx)
            toks_b[0] = tok
            for m_i in range(n_d):
                _rmsnorm(out_state, mtp_norms[m_i], h)
                matvec_fp32(mtp_ws[m_i], np.float32(1.0), h, proj_h)
                matvec_fp32(head, np.float32(1.0), proj_h, draft_logits)
                d_tok = int(np.argmax(draft_logits))
                drafts[m_i] = d_tok
                toks_b[1 + m_i] = d_tok
            total_drafts += n_d

            B = 1 + n_d
            for b in range(B):
                tok_i = int(toks_b[b])
                s_b = (pos + b) % ctx
                rem_b[b] = np.float32(1.0)
                halted_b[b] = 0
                for i in range(d):
                    v = emb[tok_i, i]
                    xb[b, i] = v
                    x0b[b, i] = v
                    x0_hist[s_b, i] = v
                    out_states_b[b, i] = np.float32(0.0)

            for t in range(n_loops):
                start_p = pos - min(pos, ctx - 1)
                for pp in range(start_p, pos):
                    ps = pp % ctx
                    if kv_ready[ps] < t:
                        vl_p = pp + 1 if pp < ctx else ctx
                        _v1_prep_anchor(xf_hist[ps], x0_hist[ps], loop_emb[t], anchor_w, anchor_b, cat2d, gate, xtmp)
                        for l in range(L):
                            _attn_loop_step(is_ternary, l, t, pp, vl_p, heads, ctx, inv_sqrt, xtmp, kc, vc,
                                            n1, wqkv, pqkv, sqkv, wo, po, so, cos, sin, h, qkv, att, tmp, scores, lut_d)
                            _swiglu_step(is_ternary, l, xtmp, n2, wg, pg, sg, wu, pu, su, wd, pd, sd, h, g, u, tmp, lut_d, lut_h)
                        kv_ready[ps] = t

                for b in range(B):
                    _v1_prep_anchor(xb[b], x0b[b], loop_emb[t], anchor_w, anchor_b, cat2d, gate, hin_b[b])

                for l in range(L):
                    for b in range(B):
                        _rmsnorm(hin_b[b], n1[l], hb[b])
                    if is_ternary:
                        matvec_packed_batch(pqkv[l], sqkv[l], hb, B, d, luts_d, qkv_b)
                    else:
                        for b in range(B):
                            matvec_fp32(wqkv[l], sqkv[l], hb[b], qkv_b[b])

                    for b in range(B):
                        p_b = pos + b
                        s_b = p_b % ctx
                        vlen_b = p_b + 1 if p_b < ctx else ctx
                        for hh in range(heads):
                            off = hh * hd
                            for i in range(half):
                                c = cos[p_b, i]
                                s = sin[p_b, i]
                                q1 = qkv_b[b, off + 2 * i]
                                q2 = qkv_b[b, off + 2 * i + 1]
                                qkv_b[b, off + 2 * i] = q1 * c - q2 * s
                                qkv_b[b, off + 2 * i + 1] = q1 * s + q2 * c
                                k1 = qkv_b[b, d + off + 2 * i]
                                k2 = qkv_b[b, d + off + 2 * i + 1]
                                kc[t, l, s_b, off + 2 * i] = k1 * c - k2 * s
                                kc[t, l, s_b, off + 2 * i + 1] = k1 * s + k2 * c
                        for i in range(d):
                            vc[t, l, s_b, i] = qkv_b[b, 2 * d + i]

                        for hh in range(heads):
                            off = hh * hd
                            m = np.float32(-1e30)
                            for j in range(vlen_b):
                                js = j % ctx
                                acc = np.float32(0.0)
                                for i in range(hd):
                                    acc += qkv_b[b, off + i] * kc[t, l, js, off + i]
                                acc *= inv_sqrt
                                scores[j] = acc
                                if acc > m:
                                    m = acc
                            z = np.float32(0.0)
                            for j in range(vlen_b):
                                e = np.exp(scores[j] - m)
                                scores[j] = e
                                z += e
                            inv_z = np.float32(1.0) / z
                            for i in range(hd):
                                att_b[b, off + i] = np.float32(0.0)
                            for j in range(vlen_b):
                                js = j % ctx
                                w_att = scores[j] * inv_z
                                for i in range(hd):
                                    att_b[b, off + i] += w_att * vc[t, l, js, off + i]

                    if is_ternary:
                        matvec_packed_batch(po[l], so[l], att_b, B, d, luts_d, tmp_b)
                    else:
                        for b in range(B):
                            matvec_fp32(wo[l], so[l], att_b[b], tmp_b[b])
                    for b in range(B):
                        for i in range(d):
                            hin_b[b, i] += tmp_b[b, i]
                        _rmsnorm(hin_b[b], n2[l], hb[b])

                    if is_ternary:
                        matvec_packed_batch(pg[l], sg[l], hb, B, d, luts_d, gb)
                        matvec_packed_batch(pu[l], su[l], hb, B, d, luts_d, ub)
                    else:
                        for b in range(B):
                            matvec_fp32(wg[l], sg[l], hb[b], gb[b])
                            matvec_fp32(wu[l], su[l], hb[b], ub[b])
                    for b in range(B):
                        for i in range(hidden):
                            gi = gb[b, i]
                            gb[b, i] = gi / (np.float32(1.0) + np.exp(-gi)) * ub[b, i]
                    if is_ternary:
                        matvec_packed_batch(pd[l], sd[l], gb, B, hidden, luts_h, tmp_b)
                    else:
                        for b in range(B):
                            matvec_fp32(wd[l], sd[l], gb[b], tmp_b[b])
                    for b in range(B):
                        for i in range(d):
                            hin_b[b, i] += tmp_b[b, i]

                all_done = True
                is_last = t == n_loops - 1
                for b in range(B):
                    kv_ready[(pos + b) % ctx] = t
                    if halted_b[b] == 0:
                        for i in range(d):
                            xb[b, i] = hin_b[b, i]
                    _rmsnorm(xb[b], halt_norm, hb[0])
                    _lin_bias(halt_w, halt_b, hb[0], halt_buf)
                    p_halt = np.float32(1.0) / (np.float32(1.0) + np.exp(-halt_buf[0]))
                    q_w = rem_b[b] if is_last else rem_b[b] * p_halt
                    rem_b[b] = rem_b[b] * (np.float32(1.0) - p_halt)
                    for i in range(d):
                        out_states_b[b, i] += q_w * xb[b, i]
                    ponder_sum += float(np.float32(t + 1) * q_w)
                    if not is_last and (np.float32(1.0) - rem_b[b]) >= (np.float32(1.0) - halt_eps):
                        halted_b[b] = 1
                    elif is_last:
                        halted_b[b] = 1
                    if halted_b[b] == 0:
                        all_done = False
                if not is_last and all_done:
                    for b in range(B):
                        for i in range(d):
                            out_states_b[b, i] += rem_b[b] * xb[b, i]
                        ponder_sum += float(np.float32(t + 1) * rem_b[b])
                    break

            for b in range(B):
                s_b = (pos + b) % ctx
                for i in range(d):
                    xf_hist[s_b, i] = xb[b, i]
                _rmsnorm(out_states_b[b], nf, hb[0])
                matvec_fp32(head, np.float32(1.0), hb[0], logits_b[b])

            matched = 0
            for m_i in range(n_d):
                pred = int(np.argmax(logits_b[m_i]))
                if pred == int(drafts[m_i]):
                    accepted_drafts += 1
                    out[k_idx] = pred
                    k_idx += 1
                    matched += 1
                    if k_idx >= n_new:
                        break
                else:
                    break
            pos += 1 + matched
            steps += 1 + matched
            for i in range(V):
                logits[i] = logits_b[matched, i]
            for i in range(d):
                out_state[i] = out_states_b[matched, i]
        else:
            valid_len = pos + 1 if pos < ctx else ctx
            slot = pos % ctx
            for j in range(d):
                v = emb[tok, j]
                x[j] = v
                x0_hist[slot, j] = v
                out_state[j] = np.float32(0.0)
            rem = np.float32(1.0)
            ponder = np.float32(0.0)
            start_p = pos - valid_len + 1
            for t in range(n_loops):
                for pp in range(start_p, pos):
                    ps = pp % ctx
                    if kv_ready[ps] < t:
                        vl_p = pp + 1 if pp < ctx else ctx
                        _v1_prep_anchor(xf_hist[ps], x0_hist[ps], loop_emb[t], anchor_w, anchor_b, cat2d, gate, xtmp)
                        for l in range(L):
                            _attn_loop_step(is_ternary, l, t, pp, vl_p, heads, ctx, inv_sqrt, xtmp, kc, vc,
                                            n1, wqkv, pqkv, sqkv, wo, po, so, cos, sin, h, qkv, att, tmp, scores, lut_d)
                            _swiglu_step(is_ternary, l, xtmp, n2, wg, pg, sg, wu, pu, su, wd, pd, sd, h, g, u, tmp, lut_d, lut_h)
                        kv_ready[ps] = t
                _v1_prep_anchor(x, x0_hist[slot], loop_emb[t], anchor_w, anchor_b, cat2d, gate, x)
                for l in range(L):
                    _attn_loop_step(is_ternary, l, t, pos, valid_len, heads, ctx, inv_sqrt, x, kc, vc,
                                    n1, wqkv, pqkv, sqkv, wo, po, so, cos, sin, h, qkv, att, tmp, scores, lut_d)
                    _swiglu_step(is_ternary, l, x, n2, wg, pg, sg, wu, pu, su, wd, pd, sd, h, g, u, tmp, lut_d, lut_h)
                kv_ready[slot] = t
                _rmsnorm(x, halt_norm, h)
                _lin_bias(halt_w, halt_b, h, halt_buf)
                p_halt = np.float32(1.0) / (np.float32(1.0) + np.exp(-halt_buf[0]))
                is_last = t == n_loops - 1
                q_w = rem if is_last else rem * p_halt
                rem = rem * (np.float32(1.0) - p_halt)
                for j in range(d):
                    out_state[j] += q_w * x[j]
                ponder += np.float32(t + 1) * q_w
                if not is_last and (np.float32(1.0) - rem) >= (np.float32(1.0) - halt_eps):
                    for j in range(d):
                        out_state[j] += rem * x[j]
                    ponder += np.float32(t + 1) * rem
                    break
            for j in range(d):
                xf_hist[slot, j] = x[j]
            _rmsnorm(out_state, nf, h)
            matvec_fp32(head, np.float32(1.0), h, logits)
            ponder_sum += float(ponder)
            pos += 1
            steps += 1

    return out, (ponder_sum / max(1, steps)), total_drafts, accepted_drafts


# ---------------------------------------------------------------------------
# Arch v2: OmegaCoreV2 (Modular Small-Arg Streaming Kernels)
# ---------------------------------------------------------------------------

@dataclass
class V2Weights:
    cfg: CoreConfig
    is_ternary: bool
    emb: np.ndarray
    emb_h2: np.ndarray
    emb_h3: np.ndarray
    emb_h4: np.ndarray
    ngram_proj: np.ndarray
    stem_norm: np.ndarray
    dw_w: np.ndarray
    dw_b: np.ndarray
    dw_proj: np.ndarray
    patch_scorer_w: np.ndarray
    patch_scorer_b: np.ndarray
    patch_proj: np.ndarray
    bos_thought: np.ndarray
    n1: np.ndarray
    n2: np.ndarray
    wqkv: np.ndarray
    pqkv: np.ndarray
    sqkv: np.ndarray
    wo: np.ndarray
    po: np.ndarray
    so: np.ndarray
    wg: np.ndarray
    pg: np.ndarray
    sg: np.ndarray
    wu: np.ndarray
    pu: np.ndarray
    su: np.ndarray
    wd: np.ndarray
    pd: np.ndarray
    sd: np.ndarray
    use_fw: bool
    n_fw: np.ndarray
    fw_wqkv: np.ndarray
    fw_pqkv: np.ndarray
    fw_sqkv: np.ndarray
    fw_gates_w: np.ndarray
    fw_gates_b: np.ndarray
    fw_wo: np.ndarray
    fw_po: np.ndarray
    fw_so: np.ndarray
    loop_emb: np.ndarray
    anchor_w: np.ndarray
    anchor_b: np.ndarray
    halt_norm: np.ndarray
    halt_w: np.ndarray
    halt_b: np.ndarray
    offset_emb: np.ndarray
    cross_gate_w: np.ndarray
    cross_gate_b: np.ndarray
    fine_norm: np.ndarray
    fine_gate_w: np.ndarray
    fine_gate_p: np.ndarray
    fine_gate_s: np.float32
    fine_up_w: np.ndarray
    fine_up_p: np.ndarray
    fine_up_s: np.float32
    fine_down_w: np.ndarray
    fine_down_p: np.ndarray
    fine_down_s: np.float32
    norm: np.ndarray
    head: np.ndarray
    use_copy_gate: bool
    copy_w: np.ndarray
    copy_b: np.ndarray
    cos: np.ndarray
    sin: np.ndarray

    def bytes_packed(self) -> int:
        if self.is_ternary:
            return (
                sum(a.nbytes for a in (self.pqkv, self.po, self.pg, self.pu, self.pd, self.fw_pqkv, self.fw_po))
                + self.fine_gate_p.nbytes + self.fine_up_p.nbytes + self.fine_down_p.nbytes
            )
        return (
            sum(a.nbytes for a in (self.wqkv, self.wo, self.wg, self.wu, self.wd, self.fw_wqkv, self.fw_wo))
            + self.fine_gate_w.nbytes + self.fine_up_w.nbytes + self.fine_down_w.nbytes
        )


def export_v2(cfg: CoreConfig, st: dict) -> V2Weights:
    f = lambda k: np.ascontiguousarray(st[k].float().numpy().astype(np.float32))
    L = cfg.layers
    t = bool(cfg.ternary)

    wqkv, pqkv, sqkv = _stack_linears(st, L, "blocks.{}.qkv.weight", t)
    wo, po, so = _stack_linears(st, L, "blocks.{}.proj.weight", t)
    wg, pg, sg = _stack_linears(st, L, "blocks.{}.gate.weight", t)
    wu, pu, su = _stack_linears(st, L, "blocks.{}.up.weight", t)
    wd, pd, sd = _stack_linears(st, L, "blocks.{}.down.weight", t)
    n1 = np.ascontiguousarray(np.stack([f(f"blocks.{i}.n1.weight") for i in range(L)]))
    n2 = np.ascontiguousarray(np.stack([f(f"blocks.{i}.n2.weight") for i in range(L)]))

    if cfg.use_fast_weights:
        n_fw = np.ascontiguousarray(np.stack([f(f"blocks.{i}.n_fw.weight") for i in range(L)]))
        fw_wqkv, fw_pqkv, fw_sqkv = _stack_linears(st, L, "blocks.{}.fw.qkv.weight", t)
        fw_gates_w = np.ascontiguousarray(np.stack([f(f"blocks.{i}.fw.gates.weight") for i in range(L)]))
        fw_gates_b = np.ascontiguousarray(np.stack([f(f"blocks.{i}.fw.gates.bias") for i in range(L)]))
        fw_wo, fw_po, fw_so = _stack_linears(st, L, "blocks.{}.fw.out_proj.weight", t)
    else:
        n_fw = np.zeros((L, cfg.d), dtype=np.float32)
        fw_wqkv = np.zeros((L, 1, 1), dtype=np.float32)
        fw_pqkv = np.zeros((L, 1, 1), dtype=np.uint8)
        fw_sqkv = np.ones(L, dtype=np.float32)
        fw_gates_w = np.zeros((L, 1, 1), dtype=np.float32)
        fw_gates_b = np.zeros((L, 1), dtype=np.float32)
        fw_wo = np.zeros((L, 1, 1), dtype=np.float32)
        fw_po = np.zeros((L, 1, 1), dtype=np.uint8)
        fw_so = np.ones(L, dtype=np.float32)

    fg_w, fg_p, fg_s = _export_linear(st, "fine_gate.weight", t)
    fu_w, fu_p, fu_s = _export_linear(st, "fine_up.weight", t)
    fd_w, fd_p, fd_s = _export_linear(st, "fine_down.weight", t)

    if cfg.use_copy_gate:
        copy_w = f("copy_gate_net.weight")
        copy_b = f("copy_gate_net.bias")
    else:
        copy_w = np.zeros((2, cfg.d + 4), dtype=np.float32)
        copy_b = np.zeros(2, dtype=np.float32)

    dw_w = f("dw_conv.weight")[:, 0, :]
    max_patches = max(256, (max(cfg.ctx, 4096) // max(2, cfg.patch_size)) + 16)
    hd = cfg.d // cfg.heads
    inv = 1.0 / 10000.0 ** (np.arange(0, hd, 2, dtype=np.float64) / hd)
    ang = np.outer(np.arange(max_patches, dtype=np.float64), inv)

    return V2Weights(
        cfg=cfg, is_ternary=t,
        emb=f("emb.weight"), emb_h2=f("emb_h2.weight"), emb_h3=f("emb_h3.weight"), emb_h4=f("emb_h4.weight"),
        ngram_proj=f("ngram_proj.weight"), stem_norm=f("stem_norm.weight"),
        dw_w=np.ascontiguousarray(dw_w), dw_b=f("dw_conv.bias"), dw_proj=f("dw_proj.weight"),
        patch_scorer_w=f("patch_scorer.weight")[0], patch_scorer_b=f("patch_scorer.bias")[0:1],
        patch_proj=f("patch_proj.weight"), bos_thought=f("bos_thought")[0, 0],
        n1=n1, n2=n2, wqkv=wqkv, pqkv=pqkv, sqkv=sqkv, wo=wo, po=po, so=so,
        wg=wg, pg=pg, sg=sg, wu=wu, pu=pu, su=su, wd=wd, pd=pd, sd=sd,
        use_fw=bool(cfg.use_fast_weights), n_fw=n_fw, fw_wqkv=fw_wqkv, fw_pqkv=fw_pqkv, fw_sqkv=fw_sqkv,
        fw_gates_w=fw_gates_w, fw_gates_b=fw_gates_b, fw_wo=fw_wo, fw_po=fw_po, fw_so=fw_so,
        loop_emb=f("loop_emb"), anchor_w=f("anchor_gate.weight"), anchor_b=f("anchor_gate.bias"),
        halt_norm=f("halt_norm.weight"), halt_w=f("halt_head.weight"), halt_b=f("halt_head.bias"),
        offset_emb=f("offset_emb.weight"), cross_gate_w=f("cross_gate.weight"), cross_gate_b=f("cross_gate.bias"),
        fine_norm=f("fine_norm.weight"),
        fine_gate_w=fg_w, fine_gate_p=fg_p, fine_gate_s=fg_s,
        fine_up_w=fu_w, fine_up_p=fu_p, fine_up_s=fu_s,
        fine_down_w=fd_w, fine_down_p=fd_p, fine_down_s=fd_s,
        norm=f("norm.weight"), head=f("head.weight"),
        use_copy_gate=bool(cfg.use_copy_gate), copy_w=copy_w, copy_b=copy_b,
        cos=np.cos(ang).astype(np.float32), sin=np.sin(ang).astype(np.float32),
    )


@numba.njit(fastmath=True, cache=True, nogil=True)
def _v2_stem_step(
    tok: int, pos: int, hash_buckets: int,
    emb: np.ndarray, emb_h2: np.ndarray, emb_h3: np.ndarray, emb_h4: np.ndarray,
    ngram_proj: np.ndarray, stem_norm: np.ndarray, dw_w: np.ndarray, dw_b: np.ndarray, dw_proj: np.ndarray,
    tok_ring: np.ndarray, xf0_hist: np.ndarray, xf: np.ndarray,
    ng_cat: np.ndarray, e_ng: np.ndarray, xf0: np.ndarray, c_out: np.ndarray, tmp: np.ndarray,
) -> tuple[int, int, int]:
    d = emb.shape[1]
    d_hash = emb_h2.shape[1]
    mask = hash_buckets - 1
    r_cap = tok_ring.shape[0]
    tok_ring[pos % r_cap] = tok

    m1 = int(tok_ring[(pos - 1) % r_cap]) if pos >= 1 else 0
    m2 = int(tok_ring[(pos - 2) % r_cap]) if pos >= 2 else 0
    m3 = int(tok_ring[(pos - 3) % r_cap]) if pos >= 3 else 0

    h2 = ((m1 * 131 + tok) * 2654435761) & mask
    h3 = (((m2 * 131 + m1) * 131 + tok) * 2654435761) & mask
    h4 = ((((m3 * 131 + m2) * 131 + m1) * 131 + tok) * 2654435761) & mask

    for i in range(d_hash):
        ng_cat[i] = emb_h2[h2, i]
        ng_cat[d_hash + i] = emb_h3[h3, i]
        ng_cat[2 * d_hash + i] = emb_h4[h4, i]
    matvec_fp32(ngram_proj, np.float32(1.0), ng_cat, e_ng)
    for i in range(d):
        xf[i] = emb[tok, i] + e_ng[i]
    _rmsnorm(xf, stem_norm, xf0)

    for i in range(d):
        s_conv = (
            dw_w[i, 0] * xf0_hist[0, i]
            + dw_w[i, 1] * xf0_hist[1, i]
            + dw_w[i, 2] * xf0_hist[2, i]
            + dw_w[i, 3] * xf0[i]
            + dw_b[i]
        )
        c_out[i] = s_conv / (np.float32(1.0) + np.exp(-s_conv))
        xf0_hist[0, i] = xf0_hist[1, i]
        xf0_hist[1, i] = xf0_hist[2, i]
        xf0_hist[2, i] = xf0[i]
    matvec_fp32(dw_proj, np.float32(1.0), c_out, tmp)
    for i in range(d):
        xf[i] = xf0[i] + tmp[i]
    return m1, m2, m3


@numba.njit(fastmath=True, cache=True, nogil=True)
def _v2_copy_step(
    tok: int, pos: int, m1: int, m2: int, m3: int, copy_window: int,
    tok_ring: np.ndarray, last2: np.ndarray, last3: np.ndarray, last4: np.ndarray,
    copy_w: np.ndarray, copy_b: np.ndarray, h: np.ndarray, logits: np.ndarray, out_scores: np.ndarray,
    cg_in: np.ndarray, p_neural: np.ndarray,
) -> None:
    d = h.shape[0]
    r_cap = tok_ring.shape[0]
    best_pos = -1
    best_len = 0
    sec_pos = -1
    sec_len = 0
    max_match = 16
    t3_mask = 32767
    t4_mask = 32767

    if pos >= 3:
        h4c = ((((m3 * 131 + m2) * 131 + m1) * 131 + tok) * 2654435761) & t4_mask
        j = int(last4[h4c])
        if j >= 3 and (pos - j) <= copy_window and (pos - j) < r_cap - 2:
            ml = 0
            while ml < max_match and (pos - ml) >= 0 and (j - ml) >= 0 and tok_ring[(pos - ml) % r_cap] == tok_ring[(j - ml) % r_cap]:
                ml += 1
            if ml >= 4:
                best_pos, best_len = j, ml

    if pos >= 2:
        h3c = (((m2 * 131 + m1) * 131 + tok) * 2654435761) & t3_mask
        j = int(last3[h3c])
        if j >= 2 and (pos - j) <= copy_window and (pos - j) < r_cap - 2:
            ml = 0
            while ml < max_match and (pos - ml) >= 0 and (j - ml) >= 0 and tok_ring[(pos - ml) % r_cap] == tok_ring[(j - ml) % r_cap]:
                ml += 1
            if ml >= 3:
                if ml > best_len:
                    sec_pos, sec_len = best_pos, best_len
                    best_pos, best_len = j, ml
                elif j != best_pos and int(tok_ring[(j + 1) % r_cap]) != int(tok_ring[(best_pos + 1) % r_cap]):
                    sec_pos, sec_len = j, ml

    if pos >= 1:
        k2 = (m1 << 8) | tok
        j = int(last2[k2])
        if j >= 1 and (pos - j) <= copy_window and (pos - j) < r_cap - 2:
            ml = 0
            while ml < max_match and (pos - ml) >= 0 and (j - ml) >= 0 and tok_ring[(pos - ml) % r_cap] == tok_ring[(j - ml) % r_cap]:
                ml += 1
            if ml >= 2:
                if ml > best_len:
                    sec_pos, sec_len = best_pos, best_len
                    best_pos, best_len = j, ml
                elif sec_len == 0 and j != best_pos and (best_pos < 0 or int(tok_ring[(j + 1) % r_cap]) != int(tok_ring[(best_pos + 1) % r_cap])):
                    sec_pos, sec_len = j, ml

    b1 = int(tok_ring[(best_pos + 1) % r_cap]) if best_pos >= 0 else 0
    b2 = int(tok_ring[(sec_pos + 1) % r_cap]) if sec_pos >= 0 else 0
    m1f = float(best_len)
    m2f = float(sec_len)
    d1f = float(pos - best_pos) if best_pos >= 0 else 0.0

    if pos >= 1:
        last2[(m1 << 8) | tok] = pos
    if pos >= 2:
        last3[(((m2 * 131 + m1) * 131 + tok) * 2654435761) & t3_mask] = pos
    if pos >= 3:
        last4[((((m3 * 131 + m2) * 131 + m1) * 131 + tok) * 2654435761) & t4_mask] = pos

    V = logits.shape[0]
    mlog = np.float32(-1e30)
    for v_i in range(V):
        if logits[v_i] > mlog:
            mlog = logits[v_i]
    z_l = np.float32(0.0)
    for v_i in range(V):
        ev = np.exp(logits[v_i] - mlog)
        p_neural[v_i] = ev
        z_l += ev
    inv_zl = np.float32(1.0) / z_l
    for v_i in range(V):
        p_neural[v_i] *= inv_zl

    for i in range(d):
        cg_in[i] = h[i]
    hm1 = np.float32(1.0) if m1f > 0.0 else np.float32(0.0)
    hm2 = np.float32(1.0) if m2f > 0.0 else np.float32(0.0)
    cg_in[d] = hm1
    cg_in[d + 1] = np.float32(math.log1p(m1f) / 3.0)
    cg_in[d + 2] = np.float32(math.log1p(m2f) / 3.0)
    cg_in[d + 3] = np.float32(math.log1p(d1f) / 9.0)
    g_raw = np.empty(2, np.float32)
    _lin_bias(copy_w, copy_b, cg_in, g_raw)
    alpha = (np.float32(1.0) / (np.float32(1.0) + np.exp(-g_raw[0]))) * hm1
    w_sec = (np.float32(1.0) / (np.float32(1.0) + np.exp(-g_raw[1]))) * np.float32(0.45) * hm2
    w_pri = np.float32(1.0) - w_sec

    one_ma = np.float32(1.0) - alpha
    for v_i in range(V):
        p_neural[v_i] *= one_ma
    p_neural[b1] += alpha * w_pri
    p_neural[b2] += alpha * w_sec
    for v_i in range(V):
        pm = p_neural[v_i]
        out_scores[v_i] = np.log(pm if pm > np.float32(1e-7) else np.float32(1e-7))


@numba.njit(fastmath=True, cache=True, nogil=True)
def _v2_fw_step(
    is_ternary: bool, l: int, t: int, heads: int, fast_dim: int,
    x: np.ndarray, n_fw: np.ndarray, fw_wqkv: np.ndarray, fw_pqkv: np.ndarray, fw_sqkv: np.ndarray,
    fw_gates_w: np.ndarray, fw_gates_b: np.ndarray, fw_wo: np.ndarray, fw_po: np.ndarray, fw_so: np.ndarray,
    fw_W: np.ndarray, h: np.ndarray, fqkv: np.ndarray, fg: np.ndarray, fmp: np.ndarray, tmp: np.ndarray,
    lut_d: np.ndarray, lut_fd: np.ndarray,
) -> None:
    d = x.shape[0]
    fd = heads * fast_dim
    _rmsnorm(x, n_fw[l], h)
    _lin(is_ternary, fw_wqkv[l], fw_pqkv[l], fw_sqkv[l], h, lut_d, fqkv)
    _lin_bias(fw_gates_w[l], fw_gates_b[l], h, fg)
    for hh in range(heads):
        lam = np.float32(1.0) / (np.float32(1.0) + np.exp(-fg[hh]))
        beta = np.float32(1.0) / (np.float32(1.0) + np.exp(-fg[heads + hh]))
        base = hh * fast_dim
        nq = np.float32(0.0)
        nk = np.float32(0.0)
        for i in range(fast_dim):
            qi = fqkv[base + i]
            ki = fqkv[fd + base + i]
            nq += qi * qi
            nk += ki * ki
        iq = np.float32(1.0) / np.sqrt(nq + np.float32(1e-12))
        ik = np.float32(1.0) / np.sqrt(nk + np.float32(1e-12))
        for r in range(fast_dim):
            acc_p = np.float32(0.0)
            for c_i in range(fast_dim):
                acc_p += fw_W[t, l, hh, r, c_i] * (fqkv[fd + base + c_i] * ik)
            err_r = beta * (fqkv[2 * fd + base + r] - acc_p)
            for c_i in range(fast_dim):
                k_norm = fqkv[fd + base + c_i] * ik
                fw_W[t, l, hh, r, c_i] = lam * fw_W[t, l, hh, r, c_i] + err_r * k_norm
        for r in range(fast_dim):
            acc_m = np.float32(0.0)
            for c_i in range(fast_dim):
                acc_m += fw_W[t, l, hh, r, c_i] * (fqkv[base + c_i] * iq)
            fmp[base + r] = acc_m
    if is_ternary:
        matvec_packed_lut(fw_po[l], fw_so[l], fmp, fd, lut_fd, tmp)
    else:
        matvec_fp32(fw_wo[l], fw_so[l], fmp, tmp)
    for i in range(d):
        x[i] += tmp[i]


@numba.njit(fastmath=True, cache=True, nogil=True)
def _v2_pool_patch(patch_size: int, chunk: np.ndarray, patch_scorer_w: np.ndarray, patch_scorer_b: np.float32,
                   patch_proj: np.ndarray, tmp: np.ndarray, z0: np.ndarray) -> None:
    d = patch_proj.shape[0]
    ps = np.empty(patch_size, np.float32)
    m_ps = np.float32(-1e30)
    for s_i in range(patch_size):
        acc = patch_scorer_b
        for i in range(d):
            acc += patch_scorer_w[i] * chunk[s_i, i]
        ps[s_i] = acc
        if acc > m_ps:
            m_ps = acc
    z_ps = np.float32(0.0)
    for s_i in range(patch_size):
        ps[s_i] = np.exp(ps[s_i] - m_ps)
        z_ps += ps[s_i]
    for i in range(d):
        tmp[i] = np.float32(0.0)
    for s_i in range(patch_size):
        w_s = ps[s_i] / z_ps
        for i in range(d):
            tmp[i] += w_s * chunk[s_i, i]
    matvec_fp32(patch_proj, np.float32(1.0), tmp, z0)


@numba.njit(fastmath=True, cache=True, nogil=True)
def _v2_run(
    seq: np.ndarray, n_new: int, temperature: float, seed: int, early_exit: bool, is_logits_mode: bool,
    patch_size: int, hash_buckets: int, n_loops: int, heads: int, fast_dim: int, copy_window: int,
    halt_eps: np.float32, is_ternary: bool, use_fw: bool, use_copy: bool,
    emb: np.ndarray, emb_h2: np.ndarray, emb_h3: np.ndarray, emb_h4: np.ndarray,
    ngram_proj: np.ndarray, stem_norm: np.ndarray, dw_w: np.ndarray, dw_b: np.ndarray, dw_proj: np.ndarray,
    patch_scorer_w: np.ndarray, patch_scorer_b: np.ndarray, patch_proj: np.ndarray, bos_thought: np.ndarray,
    n1: np.ndarray, n2: np.ndarray,
    wqkv: np.ndarray, pqkv: np.ndarray, sqkv: np.ndarray,
    wo: np.ndarray, po: np.ndarray, so: np.ndarray,
    wg: np.ndarray, pg: np.ndarray, sg: np.ndarray,
    wu: np.ndarray, pu: np.ndarray, su: np.ndarray,
    wd: np.ndarray, pd: np.ndarray, sd: np.ndarray,
    n_fw: np.ndarray, fw_wqkv: np.ndarray, fw_pqkv: np.ndarray, fw_sqkv: np.ndarray,
    fw_gates_w: np.ndarray, fw_gates_b: np.ndarray,
    fw_wo: np.ndarray, fw_po: np.ndarray, fw_so: np.ndarray,
    loop_emb: np.ndarray, anchor_w: np.ndarray, anchor_b: np.ndarray,
    halt_norm: np.ndarray, halt_w: np.ndarray, halt_b: np.ndarray,
    offset_emb: np.ndarray, cross_gate_w: np.ndarray, cross_gate_b: np.ndarray,
    fine_norm: np.ndarray,
    fine_gate_w: np.ndarray, fine_gate_p: np.ndarray, fine_gate_s: np.float32,
    fine_up_w: np.ndarray, fine_up_p: np.ndarray, fine_up_s: np.float32,
    fine_down_w: np.ndarray, fine_down_p: np.ndarray, fine_down_s: np.float32,
    norm: np.ndarray, head: np.ndarray, copy_w: np.ndarray, copy_b: np.ndarray,
    cos: np.ndarray, sin: np.ndarray,
):
    T = seq.shape[0]
    V = head.shape[0]
    d = emb.shape[1]
    d_hash = emb_h2.shape[1]
    L = wqkv.shape[0]
    hidden = wg.shape[1] if not is_ternary else pg.shape[1]
    fine_hid = fine_gate_w.shape[0] if not is_ternary else fine_gate_p.shape[0]
    max_p = cos.shape[0]
    r_cap = max(8192, copy_window + 256)
    inv_sqrt = np.float32(1.0 / math.sqrt(d // heads))

    tok_ring = np.zeros(r_cap, np.int64)
    last2 = np.full(65536, -1, np.int32)
    last3 = np.full(32768, -1, np.int32)
    last4 = np.full(32768, -1, np.int32)
    xf0_hist = np.zeros((3, d), np.float32)
    patch_chunk = np.zeros((patch_size, d), np.float32)
    cur_coarse = bos_thought.copy()
    kc = np.zeros((n_loops, L, max_p, d), np.float32)
    vc = np.zeros((n_loops, L, max_p, d), np.float32)
    fw_W = np.zeros((n_loops, L, heads, fast_dim, fast_dim), np.float32)

    out_logits = np.empty((T if is_logits_mode else 1, V), np.float32)
    gen_out = np.empty(max(1, n_new), np.int64)
    scores_buf = np.empty(V, np.float32)

    ng_cat = np.empty(3 * d_hash, np.float32)
    e_ng = np.empty(d, np.float32)
    xf0 = np.empty(d, np.float32)
    c_out = np.empty(d, np.float32)
    xf = np.empty(d, np.float32)
    cat2d = np.empty(2 * d, np.float32)
    cg = np.empty(d, np.float32)
    hf = np.empty(d, np.float32)
    h = np.empty(d, np.float32)
    fg_buf = np.empty(fine_hid, np.float32)
    fu_buf = np.empty(fine_hid, np.float32)
    tmp = np.empty(d, np.float32)
    logits = np.empty(V, np.float32)
    z0 = np.empty(d, np.float32)
    px = np.empty(d, np.float32)
    p_gate = np.empty(d, np.float32)
    ph = np.empty(d, np.float32)
    qkv = np.empty(3 * d, np.float32)
    att = np.empty(d, np.float32)
    ptmp = np.empty(d, np.float32)
    pg_buf = np.empty(hidden, np.float32)
    pu_buf = np.empty(hidden, np.float32)
    scores = np.empty(max_p, np.float32)
    fd = heads * fast_dim
    fqkv = np.empty(3 * fd, np.float32)
    f_g = np.empty(2 * heads, np.float32)
    fmp = np.empty(fd, np.float32)
    lut_d = np.empty(((d + 3) // 4, 256), np.float32)
    lut_h = np.empty(((hidden + 3) // 4, 256), np.float32)
    lut_fh = np.empty(((fine_hid + 3) // 4, 256), np.float32)
    lut_fd = np.empty(((fd + 3) // 4, 256), np.float32)
    cg_in = np.empty(d + 4, np.float32)
    p_neural = np.empty(V, np.float32)
    halt_buf = np.empty(1, np.float32)

    np.random.seed(seed)
    total_steps = T if is_logits_mode else (T + n_new)
    for pos in range(total_steps):
        if pos < T:
            tok = int(seq[pos])
        else:
            tok = _sample(scores_buf, temperature, np.random.random())
            gen_out[pos - T] = tok

        m1, m2, m3 = _v2_stem_step(
            tok, pos, hash_buckets, emb, emb_h2, emb_h3, emb_h4, ngram_proj, stem_norm,
            dw_w, dw_b, dw_proj, tok_ring, xf0_hist, xf, ng_cat, e_ng, xf0, c_out, tmp,
        )

        off = pos % patch_size
        for i in range(d):
            xf_o = xf[i] + offset_emb[off, i]
            cat2d[i] = xf_o
            cat2d[d + i] = cur_coarse[i]
            hf[i] = xf_o
        _lin_bias(cross_gate_w, cross_gate_b, cat2d, cg)
        for i in range(d):
            gs = np.float32(1.0) / (np.float32(1.0) + np.exp(-cg[i]))
            hf[i] += gs * cur_coarse[i]

        _rmsnorm(hf, fine_norm, h)
        _lin(is_ternary, fine_gate_w, fine_gate_p, fine_gate_s, h, lut_d, fg_buf)
        _lin(is_ternary, fine_up_w, fine_up_p, fine_up_s, h, lut_d, fu_buf)
        for i in range(fine_hid):
            gi = fg_buf[i]
            fg_buf[i] = gi / (np.float32(1.0) + np.exp(-gi)) * fu_buf[i]
        _lin(is_ternary, fine_down_w, fine_down_p, fine_down_s, fg_buf, lut_fh, tmp)
        for i in range(d):
            hf[i] += tmp[i]
        _rmsnorm(hf, norm, h)
        matvec_fp32(head, np.float32(1.0), h, logits)

        if use_copy:
            _v2_copy_step(tok, pos, m1, m2, m3, copy_window, tok_ring, last2, last3, last4,
                          copy_w, copy_b, h, logits, scores_buf, cg_in, p_neural)
        else:
            for v_i in range(V):
                scores_buf[v_i] = logits[v_i]

        if is_logits_mode:
            out_logits[pos] = scores_buf

        for i in range(d):
            patch_chunk[off, i] = xf[i]
        if off == patch_size - 1:
            p_idx = pos // patch_size
            valid_p = p_idx + 1 if p_idx < max_p else max_p
            _v2_pool_patch(patch_size, patch_chunk, patch_scorer_w, patch_scorer_b[0], patch_proj, ptmp, z0)
            for i in range(d):
                px[i] = z0[i]
                cur_coarse[i] = np.float32(0.0)
            rem = np.float32(1.0)
            for t in range(n_loops):
                _v1_prep_anchor(px, z0, loop_emb[t], anchor_w, anchor_b, cat2d, p_gate, px)
                for l in range(L):
                    _attn_loop_step(is_ternary, l, t, p_idx, valid_p, heads, max_p, inv_sqrt, px, kc, vc,
                                    n1, wqkv, pqkv, sqkv, wo, po, so, cos, sin, ph, qkv, att, ptmp, scores, lut_d)
                    if use_fw:
                        _v2_fw_step(is_ternary, l, t, heads, fast_dim, px, n_fw, fw_wqkv, fw_pqkv, fw_sqkv,
                                    fw_gates_w, fw_gates_b, fw_wo, fw_po, fw_so, fw_W, ph, fqkv, f_g, fmp, ptmp, lut_d, lut_fd)
                    _swiglu_step(is_ternary, l, px, n2, wg, pg, sg, wu, pu, su, wd, pd, sd, ph, pg_buf, pu_buf, ptmp, lut_d, lut_h)
                _rmsnorm(px, halt_norm, ph)
                _lin_bias(halt_w, halt_b, ph, halt_buf)
                p_halt = np.float32(1.0) / (np.float32(1.0) + np.exp(-halt_buf[0]))
                is_last = t == n_loops - 1
                q_w = rem if is_last else rem * p_halt
                rem = rem * (np.float32(1.0) - p_halt)
                for i in range(d):
                    cur_coarse[i] += q_w * px[i]
                if early_exit and not is_last and (np.float32(1.0) - rem) >= (np.float32(1.0) - halt_eps):
                    for i in range(d):
                        cur_coarse[i] += rem * px[i]
                    break

    return out_logits, gen_out
