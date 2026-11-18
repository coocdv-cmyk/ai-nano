"""OMEGA core v0: model semantics, ternary quantization, and native-engine equivalence with PyTorch."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numba")

from omega_nano.core.data import baselines  # noqa: E402
from omega_nano.core.engine import Engine  # noqa: E402
from omega_nano.core.model import CoreConfig, OmegaCore, ternarize  # noqa: E402

SMALL = dict(d=64, layers=2, heads=2, hidden=96, ctx=32)


def _ckpt(tmp_path, ternary):
    torch.manual_seed(0)
    cfg = CoreConfig(ternary=ternary, **SMALL)
    m = OmegaCore(cfg)
    with torch.no_grad():  # larger weights so logits are far from uniform and differences show
        for p in m.parameters():
            if p.dim() == 2:
                p.normal_(0, 0.15)
    path = tmp_path / f"m{int(ternary)}.pt"
    torch.save({"config": cfg.to_dict(), "state": m.state_dict()}, path)
    return m, path


def test_ternarize_values_and_scale():
    w = torch.tensor([[0.9, -0.05, -1.4], [0.2, 0.0, 3.0]])
    codes, s = ternarize(w)
    assert set(codes.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert torch.isclose(s, w.abs().mean())


def test_ternary_forward_uses_quantized_weights_but_trains_latent():
    m = OmegaCore(CoreConfig(ternary=True, **SMALL))
    x = torch.randint(0, 256, (2, 16))
    _, loss = m(x, x)
    loss.backward()
    g = m.blocks[0].qkv.weight.grad
    assert g is not None and g.abs().sum() > 0  # straight-through estimator passes gradients


def test_causality():
    m = OmegaCore(CoreConfig(**SMALL)).eval()
    a = torch.randint(0, 256, (1, 20))
    b = a.clone()
    b[0, 15:] = (b[0, 15:] + 7) % 256
    with torch.no_grad():
        la, lb = m(a)[0], m(b)[0]
    assert torch.allclose(la[0, :15], lb[0, :15], atol=1e-5)  # the future can't change the past


@pytest.mark.parametrize("ternary", [False, True])
def test_native_engine_matches_torch(tmp_path, ternary):
    m, path = _ckpt(tmp_path, ternary)
    seq = bytes(np.random.default_rng(1).integers(0, 256, 24).astype(np.uint8))
    with torch.no_grad():
        ref = m.eval()(torch.tensor([list(seq)]))[0][0].numpy()
    got = Engine.load(path).logits(seq)
    assert np.abs(got - ref).max() < 2e-3 * max(1.0, np.abs(ref).max())
    assert (got.argmax(1) == ref.argmax(1)).all()


def test_engine_storage_is_one_byte_per_ternary_weight(tmp_path):
    _, p0 = _ckpt(tmp_path, False)
    _, p1 = _ckpt(tmp_path, True)
    e0, e1 = Engine.load(p0), Engine.load(p1)
    assert e1.w.wqkv.dtype == np.int8 and set(np.unique(e1.w.wqkv)) <= {-1, 0, 1}
    assert e0.w.bytes_block() == 4 * e1.w.bytes_block()


def test_greedy_generation_matches_torch(tmp_path):
    m, path = _ckpt(tmp_path, True)
    prompt = b"def f(x):"
    idx = torch.tensor([list(prompt)])
    with torch.no_grad():
        for _ in range(12):
            nxt = m.eval()(idx)[0][0, -1].argmax()
            idx = torch.cat([idx, nxt.view(1, 1)], 1)
    want = bytes(idx[0, len(prompt):].tolist())
    assert Engine.load(path).generate(prompt, 12, temperature=0) == want


def test_generation_past_context_window_and_batch(tmp_path):
    _, path = _ckpt(tmp_path, True)
    e = Engine.load(path)
    out = e.generate(b"x" * 50, 100, temperature=0.9, seed=3)  # ctx is 32: exercises window re-encode
    assert len(out) == 100
    outs = e.generate_batch([b"ab", b"cd", b"ef"], 40, temperature=0.9)
    assert [len(o) for o in outs] == [40, 40, 40]


def test_baselines_on_toy_data():
    rng = np.random.default_rng(0)
    text = np.frombuffer(b"abcabcabd" * 2000, np.uint8)
    b = baselines(text[:15000], text[15000:])
    assert b["order2-ngram (trained)"] < 1.0 < b["uniform"]
    noise = rng.integers(0, 256, 20000).astype(np.uint8)
    assert baselines(noise[:15000], noise[15000:])["zlib-9"] > 7.9
