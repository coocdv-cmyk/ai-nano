"""EXP-001: Do ternary weights make CPU decode faster, or only smaller?

Hypothesis: single-token decode is a matrix-vector product, bound by memory
bandwidth, so cutting weight bytes (fp32 4 B -> int8 1 B -> 2-bit 0.25 B)
should cut time roughly in proportion, but only if a kernel actually exploits
the packed representation. Constitution §5.3: "low-bit or ternary weights do
not automatically make inference faster ... measure the real implementation."

Kernels (all produce y = W x for the same ternary W, checked against fp32):
  fp32-numpy      numpy BLAS matvec on the fp32 copy of W            (baseline)
  fp32-torch      torch.nn.functional.linear, MKL                      (baseline)
  int8-torch-dyn  torch dynamic-quantized int8 Linear (fbgemm)         (strong low-bit baseline)
  i8-numba        W stored as int8 {-1,0,1}, Numba parallel kernel     (1 B/weight)
  t2-decode       W packed 4 weights/byte, decoded with bit ops         (0.25 B/weight)
  t2-lut          W packed 4 weights/byte; per group of 4 activations a 256-entry
                  table of all signed partial sums is built once per token, then
                  each weight byte is one table lookup + add          (0.25 B/weight)
  t2-colmajor     same 2-bit codes stored transposed [groups, out] so the decode
                  vectorizes across output rows                       (0.25 B/weight)

Usage: python EXPERIMENTS/EXP-001-ternary-cpu/ternary_bench.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from omega_nano.hw import detect  # noqa: E402


@numba.njit(parallel=True, fastmath=True, cache=True)
def i8_matvec(w, x, y):
    for o in numba.prange(w.shape[0]):
        acc = np.float32(0.0)
        for j in range(w.shape[1]):
            acc += np.float32(w[o, j]) * x[j]
        y[o] = acc


@numba.njit(parallel=True, fastmath=True, cache=True)
def t2_decode_matvec(wp, x, y):
    # code 0 -> 0, 1 -> +1, 2 -> -1
    for o in numba.prange(wp.shape[0]):
        acc = np.float32(0.0)
        for g in range(wp.shape[1]):
            b = np.int32(wp[o, g])
            base = 4 * g
            for k in range(4):
                c = (b >> (2 * k)) & 3
                acc += x[base + k] * np.float32((c & 1) - (c >> 1))
        y[o] = acc


@numba.njit(parallel=True, fastmath=True, cache=True)
def build_lut(x, lut):
    for g in numba.prange(lut.shape[0]):
        x0, x1, x2, x3 = x[4 * g], x[4 * g + 1], x[4 * g + 2], x[4 * g + 3]
        for code in range(256):
            s = np.float32(0.0)
            for k in range(4):
                c = (code >> (2 * k)) & 3
                v = np.float32((c & 1) - (c >> 1))
                xv = x0 if k == 0 else (x1 if k == 1 else (x2 if k == 2 else x3))
                s += v * xv
            lut[g, code] = s


@numba.njit(parallel=True, fastmath=True, cache=True)
def t2_lut_matvec(wp, lut, y):
    for o in numba.prange(wp.shape[0]):
        acc = np.float32(0.0)
        for g in range(wp.shape[1]):
            acc += lut[g, wp[o, g]]
        y[o] = acc


TILE = 64


@numba.njit(parallel=True, fastmath=True, cache=True)
def t2_colmajor_matvec(wpt, x, y):
    # wpt is [groups, out]: for a fixed group the codes of consecutive rows are
    # contiguous, so the innermost loop runs over rows and LLVM vectorizes the
    # decode (shift, mask, int->float, FMA) across 8 rows per AVX2 instruction.
    n_out = wpt.shape[1]
    n_tiles = (n_out + TILE - 1) // TILE
    for t in numba.prange(n_tiles):
        lo = t * TILE
        hi = min(lo + TILE, n_out)
        acc = np.zeros(TILE, dtype=np.float32)
        for g in range(wpt.shape[0]):
            x0, x1, x2, x3 = x[4 * g], x[4 * g + 1], x[4 * g + 2], x[4 * g + 3]
            for o in range(lo, hi):
                b = np.int32(wpt[g, o])
                c0 = b & 3
                c1 = (b >> 2) & 3
                c2 = (b >> 4) & 3
                c3 = (b >> 6) & 3
                acc[o - lo] += (x0 * np.float32((c0 & 1) - (c0 >> 1)) + x1 * np.float32((c1 & 1) - (c1 >> 1))
                                + x2 * np.float32((c2 & 1) - (c2 >> 1)) + x3 * np.float32((c3 & 1) - (c3 >> 1)))
        for o in range(lo, hi):
            y[o] = acc[o - lo]


def pack2(w: np.ndarray) -> np.ndarray:
    codes = np.where(w == 1, 1, np.where(w == -1, 2, 0)).astype(np.uint8)
    c = codes.reshape(w.shape[0], -1, 4)
    return (c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)).astype(np.uint8)


def timeit(fn, reps: int) -> tuple[float, float]:
    for _ in range(3):
        fn()
    xs = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t) * 1000)
    return statistics.median(xs), statistics.pstdev(xs)


def run_shape(out_f: int, in_f: int, reps: int) -> dict:
    import torch

    rng = np.random.default_rng(0)
    w = rng.integers(-1, 2, size=(out_f, in_f)).astype(np.int8)  # ~1/3 zeros, like trained ternary nets
    x = rng.standard_normal(in_f).astype(np.float32)
    w32 = w.astype(np.float32)
    ref = w32 @ x
    wp = pack2(w)
    wpt = np.ascontiguousarray(wp.T)
    lut = np.empty((in_f // 4, 256), dtype=np.float32)
    y = np.empty(out_f, dtype=np.float32)
    tw = torch.from_numpy(w32)
    tx = torch.from_numpy(x).unsqueeze(0)
    lin = torch.nn.Linear(in_f, out_f, bias=False)
    with torch.no_grad():
        lin.weight.copy_(tw)
    qlin = torch.ao.quantization.quantize_dynamic(torch.nn.Sequential(lin), {torch.nn.Linear}, dtype=torch.qint8)

    def lut_full():
        build_lut(x, lut)
        t2_lut_matvec(wp, lut, y)

    kernels = {
        "fp32-numpy": (lambda: w32 @ x, w32.nbytes, lambda: w32 @ x),
        "fp32-torch": (lambda: torch.nn.functional.linear(tx, tw), w32.nbytes,
                       lambda: torch.nn.functional.linear(tx, tw).numpy()[0]),
        "int8-torch-dyn": (lambda: qlin(tx), out_f * in_f, lambda: qlin(tx).detach().numpy()[0]),
        "i8-numba": (lambda: i8_matvec(w, x, y), w.nbytes, lambda: (i8_matvec(w, x, y), y.copy())[1]),
        "t2-decode": (lambda: t2_decode_matvec(wp, x, y), wp.nbytes,
                      lambda: (t2_decode_matvec(wp, x, y), y.copy())[1]),
        "t2-lut": (lut_full, wp.nbytes, lambda: (lut_full(), y.copy())[1]),
        "t2-colmajor": (lambda: t2_colmajor_matvec(wpt, x, y), wpt.nbytes,
                        lambda: (t2_colmajor_matvec(wpt, x, y), y.copy())[1]),
    }
    res = {"shape": [out_f, in_f], "weights": out_f * in_f, "kernels": {}}
    with torch.no_grad():
        for name, (fn, nbytes, get) in kernels.items():
            med, sd = timeit(fn, reps)
            err = float(np.abs(get() - ref).max() / (np.abs(ref).max() + 1e-9))
            res["kernels"][name] = {"median_ms": round(med, 4), "stdev_ms": round(sd, 4), "weight_bytes": nbytes,
                                    "effective_GBps": round(nbytes / (med / 1000) / 1e9, 2),
                                    "max_rel_err_vs_fp32": float(f"{err:.2e}")}
    base = res["kernels"]["fp32-numpy"]["median_ms"]
    for k in res["kernels"].values():
        k["speedup_vs_fp32_numpy"] = round(base / k["median_ms"], 2)
    return res


def main() -> None:
    import torch

    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    shapes = [(2048, 2048), (4096, 4096), (11008, 4096)]
    out = {"experiment": "EXP-001-ternary-cpu", "date": time.strftime("%Y-%m-%d %H:%M"), "hardware": detect(),
           "threads": numba.config.NUMBA_NUM_THREADS, "batch": 1, "method": "median of 30 runs after 3 warmups",
           "results": []}
    for o, i in shapes:
        r = run_shape(o, i, reps=30)
        out["results"].append(r)
        print(f"\n{o}x{i} ({r['weights'] / 1e6:.1f}M weights)")
        for name, k in r["kernels"].items():
            print(f"  {name:<15} {k['median_ms']:>8.3f} ms  {k['weight_bytes'] / 2**20:>7.1f} MiB  "
                  f"{k['effective_GBps']:>6.2f} GB/s  x{k['speedup_vs_fp32_numpy']:<5} err {k['max_rel_err_vs_fp32']}")
    Path(__file__).with_name("results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
