"""Native CPU kernels (Numba/LLVM, multi-threaded), with numpy fallbacks.

Measured on an i7-7820HQ, 50k x 512 int8 matrix times a float32 vector:
numpy int8@f32 82 ms (it upcasts the whole matrix per call), numpy f32 BLAS
10.7 ms, this kernel 3.0 ms with max relative error 2e-7 versus f32.
See EXPERIMENTS/EXP-002-int8-retrieval-kernel.md.
"""

from __future__ import annotations

import os

try:
    import numpy as np
except ImportError:
    np = None

_dot = None
_tried = False
JIT_MIN_ROWS = 4096  # below this the JIT's one-time compile cost isn't worth it


def _compile():
    global _dot, _tried
    _tried = True
    if np is None or os.environ.get("OMEGA_NO_JIT"):
        return
    try:
        import numba
    except ImportError:
        return

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def dot_i8_f32(m, q, scales, out):
        for i in numba.prange(m.shape[0]):
            acc = np.float32(0.0)
            for j in range(m.shape[1]):
                acc += np.float32(m[i, j]) * q[j]
            out[i] = acc * scales[i]

    _dot = dot_i8_f32


def scaled_dot(mat, q, scales):
    """(int8 [n, d] @ float32 [d]) * scales[n] -> float32 [n]."""
    n = mat.shape[0]
    if n >= JIT_MIN_ROWS:
        if not _tried:
            _compile()
        if _dot is not None:
            out = np.empty(n, dtype=np.float32)
            _dot(mat, q, scales, out)
            return out
    return (mat @ q) * scales


def backend() -> str:
    if not _tried:
        _compile()
    return "numba-int8" if _dot is not None else ("numpy" if np is not None else "pure-python")
