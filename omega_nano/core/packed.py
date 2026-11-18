"""2-bit ternary weight packing & multiplier-free T-MAC LUT kernels for Numba CPU.

Packs ternary weight codes {-1, 0, +1} at 2 bits per weight (4 weights per uint8,
4x smaller than int8 and 16x smaller than fp32):
  0 -> 0b00 ( 0)
  +1 -> 0b01 (+1)
  -1 -> 0b10 (-1)

At inference time, builds a tiny float32 activation lookup table per 4-input group
using only additions/negations, then accumulates matvec rows via table lookups with
zero scalar multiplications.
"""
from __future__ import annotations

import numba
import numpy as np


def pack_ternary(w_int8: np.ndarray) -> tuple[np.ndarray, int]:
    """Pack a 2D int8 matrix of values in {-1, 0, 1} into [O, ceil(I/4)] uint8."""
    O, I = w_int8.shape
    pad = (-I) & 3
    if pad:
        padded = np.zeros((O, I + pad), dtype=np.int8)
        padded[:, :I] = w_int8
    else:
        padded = np.ascontiguousarray(w_int8, dtype=np.int8)
    G = padded.shape[1] // 4
    blk = padded.reshape(O, G, 4)
    # 0->0, 1->1, -1->2 (via code = np.where(b>0, 1, np.where(b<0, 2, 0)))
    codes = np.zeros_like(blk, dtype=np.uint8)
    codes[blk > 0] = 1
    codes[blk < 0] = 2
    packed = (
        codes[:, :, 0]
        | (codes[:, :, 1] << np.uint8(2))
        | (codes[:, :, 2] << np.uint8(4))
        | (codes[:, :, 3] << np.uint8(6))
    )
    return np.ascontiguousarray(packed, dtype=np.uint8), I


def unpack_ternary(packed: np.ndarray, orig_in: int) -> np.ndarray:
    """Inverse of `pack_ternary` for testing/verification."""
    O, G = packed.shape
    out = np.zeros((O, G * 4), dtype=np.int8)
    lut_val = np.array([0, 1, -1, 0], dtype=np.int8)
    for k in range(4):
        out[:, k::4] = lut_val[(packed >> np.uint8(2 * k)) & np.uint8(3)]
    return out[:, :orig_in]


@numba.njit(fastmath=True, cache=True, nogil=True)
def _build_lut4(x: np.ndarray, I: int, lut: np.ndarray) -> None:
    """Populate `lut[G, 256]` for 4-activation groups using only +/- (no multiplies)."""
    G = lut.shape[0]
    for g in range(G):
        base = g * 4
        x0 = x[base] if base < I else np.float32(0.0)
        x1 = x[base + 1] if base + 1 < I else np.float32(0.0)
        x2 = x[base + 2] if base + 2 < I else np.float32(0.0)
        x3 = x[base + 3] if base + 3 < I else np.float32(0.0)

        v0_0 = np.float32(0.0)
        v0_1 = x0
        v0_2 = -x0
        v1_0 = np.float32(0.0)
        v1_1 = x1
        v1_2 = -x1
        v2_0 = np.float32(0.0)
        v2_1 = x2
        v2_2 = -x2
        v3_0 = np.float32(0.0)
        v3_1 = x3
        v3_2 = -x3

        for d3 in range(3):
            a3 = v3_0 if d3 == 0 else (v3_1 if d3 == 1 else v3_2)
            s3 = d3 << 6
            for d2 in range(3):
                a23 = a3 + (v2_0 if d2 == 0 else (v2_1 if d2 == 1 else v2_2))
                s23 = s3 | (d2 << 4)
                for d1 in range(3):
                    a123 = a23 + (v1_0 if d1 == 0 else (v1_1 if d1 == 1 else v1_2))
                    s123 = s23 | (d1 << 2)
                    lut[g, s123] = a123
                    lut[g, s123 | 1] = a123 + v0_1
                    lut[g, s123 | 2] = a123 + v0_2


@numba.njit(fastmath=True, cache=True, nogil=True)
def matvec_packed_lut(packed: np.ndarray, scale: np.float32, x: np.ndarray, I: int,
                      lut: np.ndarray, out: np.ndarray) -> None:
    """Compute `out = scale * (W_ternary @ x)` via 2-bit packed weights & 4-input LUT."""
    _build_lut4(x, I, lut)
    O = packed.shape[0]
    G = packed.shape[1]
    o = 0
    lim = O - (O & 3)
    while o < lim:
        s0 = np.float32(0.0)
        s1 = np.float32(0.0)
        s2 = np.float32(0.0)
        s3 = np.float32(0.0)
        for g in range(G):
            s0 += lut[g, packed[o, g]]
            s1 += lut[g, packed[o + 1, g]]
            s2 += lut[g, packed[o + 2, g]]
            s3 += lut[g, packed[o + 3, g]]
        out[o] = s0 * scale
        out[o + 1] = s1 * scale
        out[o + 2] = s2 * scale
        out[o + 3] = s3 * scale
        o += 4
    while o < O:
        acc = np.float32(0.0)
        for g in range(G):
            acc += lut[g, packed[o, g]]
        out[o] = acc * scale
        o += 1


@numba.njit(fastmath=True, cache=True, nogil=True)
def matvec_packed_batch(packed: np.ndarray, scale: np.float32, xs: np.ndarray, B: int, I: int,
                        luts: np.ndarray, outs: np.ndarray) -> None:
    """Compute B matrix-vector products sharing a single pass over `packed` weights in L1/L2 cache."""
    G = packed.shape[1]
    for b in range(B):
        _build_lut4(xs[b], I, luts[b])
    O = packed.shape[0]
    if B == 2:
        for o in range(O):
            a0 = np.float32(0.0)
            a1 = np.float32(0.0)
            for g in range(G):
                idx = packed[o, g]
                a0 += luts[0, g, idx]
                a1 += luts[1, g, idx]
            outs[0, o] = a0 * scale
            outs[1, o] = a1 * scale
    elif B == 3:
        for o in range(O):
            a0 = np.float32(0.0)
            a1 = np.float32(0.0)
            a2 = np.float32(0.0)
            for g in range(G):
                idx = packed[o, g]
                a0 += luts[0, g, idx]
                a1 += luts[1, g, idx]
                a2 += luts[2, g, idx]
            outs[0, o] = a0 * scale
            outs[1, o] = a1 * scale
            outs[2, o] = a2 * scale
    else:
        for o in range(O):
            for b in range(B):
                outs[b, o] = np.float32(0.0)
            for g in range(G):
                idx = packed[o, g]
                for b in range(B):
                    outs[b, o] += luts[b, g, idx]
            for b in range(B):
                outs[b, o] *= scale


@numba.njit(fastmath=True, cache=True, nogil=True)
def matvec_fp32(w: np.ndarray, s: np.float32, x: np.ndarray, out: np.ndarray) -> None:
    """Dense fp32 matrix-vector product (for non-ternary layers/projections)."""
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
            s0 += w[o, j] * xj
            s1 += w[o + 1, j] * xj
            s2 += w[o + 2, j] * xj
            s3 += w[o + 3, j] * xj
        out[o] = s0 * s
        out[o + 1] = s1 * s
        out[o + 2] = s2 * s
        out[o + 3] = s3 * s
        o += 4
    while o < O:
        acc = np.float32(0.0)
        for j in range(I):
            acc += w[o, j] * x[j]
        out[o] = acc * s
        o += 1
