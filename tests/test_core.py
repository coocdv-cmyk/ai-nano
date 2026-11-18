"""OMEGA core v0: model semantics, ternary quantization, and native-engine equivalence with PyTorch."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numba")

from omega_nano.core.data import baselines, build_corpus  # noqa: E402
from omega_nano.core.engine import Engine  # noqa: E402
from omega_nano.core.model import CoreConfig, LoopedOmegaCore, OmegaCore, build_model, ternarize  # noqa: E402
from omega_nano.core.model_v2 import OmegaCoreV2, _extract_copy_numba, _extract_copy_py  # noqa: E402
from omega_nano.core.packed import pack_ternary, unpack_ternary  # noqa: E402

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


def test_looped_v1_causality_mtp_and_adaptive_halting(tmp_path):
    cfg = CoreConfig(d=64, layers=1, heads=2, hidden=96, ctx=32, arch="v1", n_loops=4, mtp_k=3, halt_eps=0.1)
    m = build_model(cfg)
    assert isinstance(m, LoopedOmegaCore)

    # Training computes joint loss (next-byte + MTP + ponder); eval returns pure next-byte CE.
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    m.train()
    _, train_loss = m(x, y)
    train_loss.backward()
    assert m.loop_emb.grad is not None and m.halt_head.weight.grad is not None

    m.eval()
    with torch.no_grad():
        _, eval_loss = m(x, y)
        assert eval_loss.isfinite()
        # Strict causality even with early-exit adaptive halting enabled.
        a = torch.randint(0, 256, (1, 20))
        b = a.clone()
        b[0, 15:] = (b[0, 15:] + 7) % 256
        la, lb = m(a)[0], m(b)[0]
        assert torch.allclose(la[0, :15], lb[0, :15], atol=1e-5)

    # Generating bytes works cleanly, and native engine loads v1 natively matching PyTorch.
    out = m.generate(b"def ", 8, temperature=0.8, seed=1)
    assert len(out) == 8
    ck = tmp_path / "v1.pt"
    torch.save({"config": cfg.to_dict(), "state": m.state_dict()}, ck)
    eng = Engine.load(ck)
    seq = bytes(np.random.default_rng(4).integers(0, 256, 20).astype(np.uint8))
    with torch.no_grad():
        ref_v1 = m(torch.tensor([list(seq)]))[0][0].numpy()
    got_v1 = eng.logits(seq)
    assert np.abs(got_v1 - ref_v1).max() < 2e-3 * max(1.0, np.abs(ref_v1).max())
    assert (got_v1.argmax(1) == ref_v1.argmax(1)).all()


def test_clean_corpus_excludes_site_packages_and_copyleft(tmp_path):
    lib = tmp_path / "lib"
    (lib / "site-packages").mkdir(parents=True)
    (lib / "ensurepip").mkdir(parents=True)
    (lib / "site-packages" / "third_party.py").write_text("SECRET_THIRD_PARTY = 1\n", encoding="utf-8")
    (lib / "ensurepip" / "bundled.py").write_text("BUNDLED_PIP = 1\n", encoding="utf-8")
    (lib / "copyleft_mod.py").write_text("# GNU Lesser General Public License\nCOPYLEFT = 1\n", encoding="utf-8")
    for i in range(40):
        (lib / f"clean_{i}.py").write_text(f"def f_{i}(x):\n    return x + {i}\n", encoding="utf-8")

    out = tmp_path / "corpus"
    meta = build_corpus(out, val_fraction=0.2, stdlib_root=lib)
    combined = (out / "train.bin").read_bytes() + (out / "val.bin").read_bytes()
    assert b"SECRET_THIRD_PARTY" not in combined
    assert b"BUNDLED_PIP" not in combined
    assert b"COPYLEFT" not in combined
    assert b"def f_0" in combined
    assert meta["rejected_count"] == 1
    assert len(meta["train_sha256"]) == 64 and len(meta["val_sha256"]) == 64


def test_v2_copy_extractor_matches_python_and_is_causal():
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 12, (2, 120), dtype=np.int64)
    py_res = _extract_copy_py(arr, window=64)
    if _extract_copy_numba is not None:
        nb_res = _extract_copy_numba(arr, window=64)
        for p, n in zip(py_res, nb_res):
            assert np.allclose(p, n)

    mut = arr.copy()
    mut[:, 55:] = (mut[:, 55:] + 3) % 12
    a_res = _extract_copy_py(arr, window=64)
    b_res = _extract_copy_py(mut, window=64)
    for a_t, b_t in zip(a_res, b_res):
        assert np.allclose(a_t[:, :55], b_t[:, :55])


@pytest.mark.parametrize("use_copy", [False, True])
def test_v2_patched_fast_weights_causality_grads_and_long_context(use_copy):
    cfg = CoreConfig(
        d=64, layers=1, heads=2, hidden=96, ctx=32, arch="v2",
        n_loops=2, mtp_k=2, patch_size=4, hash_buckets=256, fast_dim=8,
        use_fast_weights=True, use_copy_gate=use_copy, copy_window=2048, ternary=True,
    )
    m = build_model(cfg)
    assert isinstance(m, OmegaCoreV2)

    x = torch.randint(0, 256, (2, 24))
    y = torch.randint(0, 256, (2, 24))
    m.train()
    _, loss = m(x, y)
    assert loss.isfinite()
    loss.backward()
    assert m.emb_h2.weight.grad is not None and m.dw_conv.weight.grad is not None
    assert m.blocks[0].fw.gates.weight.grad is not None
    if use_copy:
        assert m.copy_gate_net.weight.grad is not None

    m.eval()
    with torch.no_grad():
        # Strict causality across every length & patch boundary
        a = torch.randint(0, 256, (1, 25))
        for cut in (1, 4, 15, 16, 17):
            b = a.clone()
            b[0, cut:] = (b[0, cut:] + 13) % 256
            la, lb = m(a)[0], m(b)[0]
            assert torch.allclose(la[0, :cut], lb[0, :cut], atol=1e-5), f"leak at cut={cut}"

        # Handles short sequences < patch_size and long sequences > ctx (e.g. 600 > 32)
        for length in (1, 3, 4, 5, 600):
            s = torch.randint(0, 256, (1, length))
            scores, _ = m(s)
            assert scores.shape == (1, length, 256) and scores.isfinite().all()

    out = m.generate(b"def f(", 12, temperature=0.8, seed=2)
    assert len(out) == 12


def test_pack_ternary_roundtrip_and_v0_compression(tmp_path):
    rng = np.random.default_rng(11)
    w = rng.choice([-1, 0, 1], size=(64, 99)).astype(np.int8)
    packed, orig_in = pack_ternary(w)
    assert packed.dtype == np.uint8 and orig_in == 99 and packed.shape == (64, 25)
    assert np.array_equal(unpack_ternary(packed, orig_in), w)

    _, p1 = _ckpt(tmp_path, True)
    e1 = Engine.load(p1, packed=True)
    assert e1.w.bytes_packed() == e1.w.bytes_block() // 4


def test_native_v1_ternary_speculative_matches_pytorch(tmp_path):
    torch.manual_seed(5)
    cfg = CoreConfig(d=64, layers=1, heads=2, hidden=96, ctx=32, arch="v1",
                     n_loops=3, mtp_k=3, halt_eps=0.1, ternary=True)
    m = build_model(cfg)
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 2:
                p.normal_(0, 0.18)
    ck = tmp_path / "v1t.pt"
    torch.save({"config": cfg.to_dict(), "state": m.state_dict()}, ck)

    eng = Engine.load(ck, packed=True)
    assert eng.w.bytes_packed() == eng.w.bytes_block_fp32() // 16

    seq = b"def add(a, b):\n    return a"
    with torch.no_grad():
        ref_logits = m.eval()(torch.tensor([list(seq)]))[0][0].numpy()
    got_logits = eng.logits(seq)
    assert np.abs(got_logits - ref_logits).max() < 2e-3 * max(1.0, np.abs(ref_logits).max())

    # Greedy generation identical between PyTorch, native non-speculative, and native speculative MTP
    prompt = b"def f(x):"
    idx = torch.tensor([list(prompt)])
    with torch.no_grad():
        for _ in range(14):
            nxt = m.eval()(idx)[0][0, -1].argmax()
            idx = torch.cat([idx, nxt.view(1, 1)], 1)
    want = bytes(idx[0, len(prompt):].tolist())

    assert eng.generate(prompt, 14, temperature=0, speculative=False) == want
    assert eng.generate(prompt, 14, temperature=0, speculative=True) == want
    outs = eng.generate_batch([prompt, b"class A: "], 8, temperature=0)
    assert len(outs) == 2 and all(len(o) == 8 for o in outs)


@pytest.mark.parametrize("ternary", [False, True])
def test_native_v2_matches_pytorch_and_greedy(tmp_path, ternary):
    torch.manual_seed(7)
    cfg = CoreConfig(
        d=64, layers=1, heads=2, hidden=96, ctx=32, arch="v2",
        n_loops=1, mtp_k=2, patch_size=4, hash_buckets=256, fast_dim=8,
        use_fast_weights=True, use_copy_gate=True, copy_window=2048, ternary=ternary,
    )
    m = build_model(cfg)
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() >= 2:
                p.normal_(0, 0.14)
    ck = tmp_path / f"v2_{int(ternary)}.pt"
    torch.save({"config": cfg.to_dict(), "state": m.state_dict()}, ck)

    eng = Engine.load(ck, packed=True)
    seq = b"def solve(arr):\n    return arr + arr"
    with torch.no_grad():
        ref = m.eval()(torch.tensor([list(seq)]))[0][0].numpy()
    got = eng.logits(seq)
    assert np.abs(got - ref).max() < 3e-3 * max(1.0, np.abs(ref).max())
    assert (got.argmax(1) == ref.argmax(1)).all()

    prompt = b"def run(x):\n    x = x + 1\n    def run(x):\n"
    idx = torch.tensor([list(prompt)])
    with torch.no_grad():
        for _ in range(12):
            nxt = m.eval()(idx)[0][0, -1].argmax()
            idx = torch.cat([idx, nxt.view(1, 1)], 1)
    want = bytes(idx[0, len(prompt):].tolist())
    assert eng.generate(prompt, 12, temperature=0) == want
    batch_out = eng.generate_batch([prompt, prompt], 10, temperature=0.8, seed=1)
    assert len(batch_out) == 2 and all(len(b) == 10 for b in batch_out)
