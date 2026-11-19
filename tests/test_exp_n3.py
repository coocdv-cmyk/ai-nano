"""EXP-013 (N3) instrument invariants — no pre-registered gate is asserted here."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "EXPERIMENTS" / "EXP-013-n3-bilingual"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from concepts import HELD_OUT, TRAIN  # noqa: E402
from run_exp import D, F, LearnedEncoder, build_probe, build_train_validation, features  # noqa: E402


def test_heldout_probe_is_disjoint_token_free_and_pinned():
    train_terms = {w for row in TRAIN for w in row}
    held_terms = {w for row in HELD_OUT for w in row}
    assert train_terms.isdisjoint(held_terms), "held-out concept leaked into training"

    probe = build_probe()
    assert (len(probe["docs"]), len(probe["queries"]), len(probe["oos"])) == (90, 60, 16)
    assert probe["doc_sha256"] == "1467df1db51dd264c7d0dfd18df38ce603a5f9f06faf24cd44a20d4eaee3558e"
    assert probe["query_sha256"] == "e89daed15769bd69537992e5da271c7b207047cf1e73ec97a79eaddf416fd11c"
    for q, gold in zip(probe["queries"], probe["answers"]):
        assert not set(q.split()) & set(probe["docs"][gold].split()), q


def test_learned_encoder_is_int8_deterministic_and_under_cap():
    X, groups = build_train_validation()
    enc = LearnedEncoder(seed=7)
    enc.train(X, groups, steps=12, concepts_per_batch=8, lr=0.005, tau=0.2, seed=9)
    n_bytes = enc.quantize()
    assert n_bytes == F * D == 393_216 and n_bytes < 500_000
    assert enc.Eq.dtype == np.int8 and enc.Eq.shape == (F, D)
    Xq = np.stack([features("قوي جسر"), features("mighty overpass"),
                   features("مركبة فضائية")]).astype(np.float32)
    a = enc.encode_int8(Xq)
    b = enc.encode_int8(Xq)
    assert np.allclose(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
