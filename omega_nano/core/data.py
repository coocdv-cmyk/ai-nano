"""Training corpus and the baselines an LM must beat.

Clean provenance-tracked corpus builder: excludes third-party ``site-packages``
and copyleft-licensed sources, records SHA-256 split manifests, and supports
additional clean local sources (e.g. verified replay traces or bilingual data).
Train/validation split is by *file* (hash of the relative path), so no file
contributes to both, preventing leakage.
"""

from __future__ import annotations

import bz2
import hashlib
import json
import lzma
import math
import os
import zlib
from pathlib import Path
from typing import Sequence

import numpy as np

SEP = b"\n\x00\n"  # file boundary marker
EXCLUDED_DIRS = frozenset({
    "site-packages", "dist-packages", "ensurepip", "__pycache__",
    "venv", ".venv", "pip", "setuptools", "pkg_resources",
})
COPYLEFT_MARKERS = (
    b"gnu lesser general public",
    b"gnu general public license",
    b"gnu affero general public",
)


def is_clean_source(rel_parts: tuple[str, ...], header_bytes: bytes) -> tuple[bool, str | None]:
    """Verify a candidate corpus file is outside excluded dirs and free of copyleft headers."""
    if any(part in EXCLUDED_DIRS for part in rel_parts):
        return False, "excluded_dir"
    head = header_bytes[:4096].lower()
    for marker in COPYLEFT_MARKERS:
        if marker in head:
            return False, f"copyleft:{marker.decode('ascii')}"
    return True, None


def _collect_py_files(root: Path, label: str) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        out.append((f"{label}:{rel.as_posix()}", path))
    return out


def build_corpus(
    out_dir: Path,
    val_fraction: float = 0.05,
    max_mb: float = 64.0,
    stdlib_root: Path | None = None,
    extra_sources: Sequence[tuple[str, Path]] | None = None,
) -> dict:
    lib = Path(stdlib_root) if stdlib_root is not None else Path(os.path.dirname(os.__file__))
    candidates = _collect_py_files(lib, "stdlib")
    if extra_sources:
        for label, src_path in extra_sources:
            p = Path(src_path)
            if p.is_file():
                candidates.append((f"{label}:{p.name}", p))
            elif p.is_dir():
                candidates.extend(_collect_py_files(p, label))

    train, val = bytearray(), bytearray()
    n_train = n_val = 0
    rejected: list[dict[str, str]] = []
    for rel_key, path in candidates:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        rel_no_label = rel_key.split(":", 1)[-1]
        ok, reason = is_clean_source(tuple(rel_no_label.split("/")), data)
        if not ok:
            rejected.append({"path": rel_key, "reason": reason or "rejected"})
            continue
        bucket = int(hashlib.sha256(rel_key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        if bucket < val_fraction:
            val += data + SEP
            n_val += 1
        else:
            train += data + SEP
            n_train += 1
        if len(train) + len(val) > max_mb * 2**20:
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    train_bytes = bytes(train)
    val_bytes = bytes(val)
    np.frombuffer(train_bytes, dtype=np.uint8).tofile(out_dir / "train.bin")
    np.frombuffer(val_bytes, dtype=np.uint8).tofile(out_dir / "val.bin")
    meta = {
        "source": f"Python clean stdlib ({lib}) excluding site-packages",
        "license_policy": "PSF-2.0 stdlib + explicitly allowed clean sources; no site-packages, no GPL/LGPL/AGPL",
        "excluded_dirs": sorted(EXCLUDED_DIRS),
        "rejected_count": len(rejected),
        "rejected_sample": rejected[:20],
        "files_train": n_train,
        "files_val": n_val,
        "bytes_train": len(train_bytes),
        "bytes_val": len(val_bytes),
        "train_sha256": hashlib.sha256(train_bytes).hexdigest(),
        "val_sha256": hashlib.sha256(val_bytes).hexdigest(),
        "split": "sha256(rel_key) per file",
    }
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
