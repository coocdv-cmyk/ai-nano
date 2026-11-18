"""Training corpus and the baselines an LM must beat.

v0 corpus: the Python standard library sources installed on this machine (PSF
licence, local, no download). Code is the constitution's top-priority domain.
Train/validation split is by *file* (hash of the path), so no file contributes
to both, which prevents leakage.
"""

from __future__ import annotations

import bz2
import glob
import hashlib
import json
import lzma
import math
import os
import zlib
from pathlib import Path

import numpy as np

SEP = b"\n\x00\n"  # file boundary marker


def build_corpus(out_dir: Path, val_fraction: float = 0.05, max_mb: float = 64.0) -> dict:
    lib = os.path.dirname(os.__file__)
    files = sorted(glob.glob(os.path.join(lib, "**", "*.py"), recursive=True))
    train, val = bytearray(), bytearray()
    n_train = n_val = 0
    for f in files:
        try:
            data = Path(f).read_bytes()
        except OSError:
            continue
        rel = os.path.relpath(f, lib).replace("\\", "/")
        bucket = int(hashlib.sha256(rel.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if bucket < val_fraction:
            val += data + SEP
            n_val += 1
        else:
            train += data + SEP
            n_train += 1
        if len(train) + len(val) > max_mb * 2**20:
            break
    out_dir.mkdir(parents=True, exist_ok=True)
    np.frombuffer(bytes(train), dtype=np.uint8).tofile(out_dir / "train.bin")
    np.frombuffer(bytes(val), dtype=np.uint8).tofile(out_dir / "val.bin")
    meta = {"source": f"Python {lib} *.py", "files_train": n_train, "files_val": n_val,
            "bytes_train": len(train), "bytes_val": len(val), "split": "sha256(path) per file"}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load(out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    return (np.fromfile(out_dir / "train.bin", dtype=np.uint8), np.fromfile(out_dir / "val.bin", dtype=np.uint8))


def baselines(train: np.ndarray, val: np.ndarray) -> dict:
    """Bits per byte on the validation set for reference compressors and a trained n-gram."""
    raw = val.tobytes()
    out = {
        "uniform": 8.0,
        "zlib-9": len(zlib.compress(raw, 9)) * 8 / len(raw),
        "bz2-9": len(bz2.compress(raw, 9)) * 8 / len(raw),
        "lzma-extreme": len(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)) * 8 / len(raw),
    }
    # Order-2 byte model trained on train, add-k smoothed, evaluated on val (a fair "learned" baseline).
    k = 0.05
    t = train.astype(np.int64)
    counts = np.bincount((t[:-2] << 16) | (t[1:-1] << 8) | t[2:], minlength=1 << 24).astype(np.float32)
    counts = counts.reshape(65536, 256)
    totals = counts.sum(1)
    v = val.astype(np.int64)
    ctx = (v[:-2] << 8) | v[1:-1]
    p = (counts[ctx, v[2:]] + k) / (totals[ctx] + 256 * k)
    out["order2-ngram (trained)"] = float(-np.log2(p).mean())
    return {k_: round(v_, 4) for k_, v_ in out.items()}


def batches(data: np.ndarray, batch: int, ctx: int, rng: np.random.Generator):
    while True:
        ix = rng.integers(0, len(data) - ctx - 1, size=batch)
        x = np.stack([data[i:i + ctx] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1:i + ctx + 1] for i in ix]).astype(np.int64)
        yield x, y


def bits_per_byte(nats: float) -> float:
    return nats / math.log(2)
