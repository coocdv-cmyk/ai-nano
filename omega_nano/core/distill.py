"""Wake-Sleep Replay Distiller & Neural Code Proposer (M2 <-> M3 Bridge).

Closes the loop between the Symbolic/Execution Brain (M3) and compact
OmegaCoreV2 weights (M1/M2):
  * Wake phase: verified solutions from `SkillLibrary.replay_traces` and
    `MemoryFabric` are formatted into `<SPEC> ... <SOL> ... </SOL>` bytes.
  * Sleep phase: bounded CPU AdamW distillation compresses verified traces
    into tiny cache-resident OmegaCoreV2 weights (`before_val_bpb` -> `after_val_bpb`).
  * Neural Proposal: `NeuralCodeProposer` uses the native CPU `Engine` to
    generate open-ended Python code candidates verified strictly by `CodeVerifier`.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .data import bits_per_byte
from .model import CoreConfig, build_model

_SEED_PATTERNS: tuple[bytes, ...] = (
    b"<SPEC>\nWrite Python function clamp(x: int, lo: int, hi: int) -> int.\n<SOL>\ndef clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n</SOL>\n",
    b"<SPEC>\nWrite Python function dot_product(a: list[int], b: list[int]) -> int.\n<SOL>\ndef dot_product(a, b):\n    return sum(x * y for x, y in zip(a, b))\n</SOL>\n",
    b"<SPEC>\nWrite Python function is_palindrome(s: str) -> bool.\n<SOL>\ndef is_palindrome(s):\n    clean = [c.casefold() for c in s if c.isalnum()]\n    return clean == clean[::-1]\n</SOL>\n",
    b"<SPEC>\nWrite Python function flatten(grid: list[list[int]]) -> list[int].\n<SOL>\ndef flatten(grid):\n    return [v for row in grid for v in row]\n</SOL>\n",
)


def build_wake_corpus(
    skills: Any | None = None,
    extra_frames: list[bytes] | None = None,
) -> bytes:
    """Build deterministic training stream from verified SkillLibrary replay traces."""
    frames: list[bytes] = []
    seen: set[bytes] = set()

    if skills is not None and hasattr(skills, "conn"):
        try:
            with skills._lock:
                rows = skills.conn.execute(
                    "SELECT statement, solution, proof_status FROM replay_traces "
                    "WHERE proof_status IN ('PROVEN', 'PROVEN_EXACT', 'TESTS_PASSED') "
                    "ORDER BY id DESC LIMIT 400"
                ).fetchall()
            for stmt, sol, _ in rows:
                frame = (
                    f"<SPEC>\n{stmt.strip()}\n<SOL>\n{sol.strip()}\n</SOL>\n"
                ).encode("utf-8", errors="replace")
                if frame not in seen:
                    seen.add(frame)
                    frames.append(frame)
        except Exception:
            pass

    for extra in (extra_frames or ()):
        if extra not in seen:
            seen.add(extra)
            frames.append(extra)

    # Ensure sufficient non-trivial frames if caller starts on an empty DB
    if len(frames) < 4:
        for seed_f in _SEED_PATTERNS:
            if seed_f not in seen:
                seen.add(seed_f)
                frames.append(seed_f)

    return b"\n".join(frames)


def _sample_windows(data: bytes, batch: int, ctx: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    if arr.size <= ctx + 1:
        reps = ((ctx + 2) // max(1, arr.size)) + 2
        arr = np.tile(arr, reps)
    hi = arr.size - ctx - 1
    starts = rng.integers(0, max(1, hi), size=batch)
    x = np.stack([arr[s : s + ctx] for s in starts])
    y = np.stack([arr[s + 1 : s + 1 + ctx] for s in starts])
    return x, y


@torch.no_grad()
def _eval_stream(model: torch.nn.Module, val_bytes: bytes, ctx: int, batch: int = 8, n_batches: int = 8) -> float:
    was_train = model.training
    model.eval()
    rng = np.random.default_rng(1234)
    tot = 0.0
    for _ in range(n_batches):
        x, y = _sample_windows(val_bytes, batch, ctx, rng)
        _, loss = model(torch.from_numpy(x), torch.from_numpy(y))
        tot += float(loss.item())
    if was_train:
        model.train()
    return bits_per_byte(tot / max(1, n_batches))


def sleep_distill(
    checkpoint_out: str | Path,
    *,
    checkpoint_in: str | Path | None = None,
    skills: Any | None = None,
    memory: Any | None = None,
    extra_frames: list[bytes] | None = None,
    budget_s: float = 8.0,
    steps_cap: int = 80,
    batch_size: int = 8,
    lr: float = 1.5e-3,
    seed: int = 0,
    ternary: bool = True,
) -> dict[str, Any]:
    """Run Wake-Sleep replay distillation on CPU into a compact OmegaCoreV2 checkpoint."""
    torch.manual_seed(seed)
    out_path = Path(checkpoint_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    corpus = build_wake_corpus(skills=skills, extra_frames=extra_frames)
    # Repeat conservatively only for splitting if corpus is shorter than ~2*ctx; val slice stays strictly disjoint.
    if len(corpus) < 1024:
        corpus = corpus * ((1024 // max(1, len(corpus))) + 1)
    split = max(int(len(corpus) * 0.8), len(corpus) - max(384, len(corpus) // 5))
    train_bytes = corpus[:split]
    val_bytes = corpus[split:]

    if checkpoint_in is not None and Path(checkpoint_in).is_file():
        ck = torch.load(checkpoint_in, map_location="cpu", weights_only=True)
        cfg = CoreConfig(**ck["config"])
        model = build_model(cfg)
        model.load_state_dict(ck["state"])
    else:
        cfg = CoreConfig(
            vocab=256, d=128, layers=2, heads=4, hidden=320, ctx=128,
            ternary=ternary, arch="v2", n_loops=1, mtp_k=2,
            patch_size=4, hash_buckets=1024, fast_dim=16,
            use_fast_weights=True, use_copy_gate=True, copy_window=2048,
        )
        model = build_model(cfg)

    eval_ctx = min(cfg.ctx, max(32, len(val_bytes) - 2))
    before_bpb = _eval_stream(model, val_bytes, ctx=eval_ctx, batch=min(4, batch_size), n_batches=6)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.02)
    rng = np.random.default_rng(seed)
    model.train()
    t0 = time.perf_counter()
    steps = 0
    tokens = 0
    train_ctx = min(cfg.ctx, max(32, len(train_bytes) - 2))

    while steps < steps_cap and (time.perf_counter() - t0) < budget_s:
        frac = min(1.0, (time.perf_counter() - t0) / max(0.05, budget_s))
        cur_lr = lr * (0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        x_np, y_np = _sample_windows(train_bytes, batch_size, train_ctx, rng)
        _, loss = model(torch.from_numpy(x_np), torch.from_numpy(y_np))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        steps += 1
        tokens += int(x_np.size)

    after_bpb = _eval_stream(model, val_bytes, ctx=eval_ctx, batch=min(4, batch_size), n_batches=6)
    elapsed_s = time.perf_counter() - t0

    meta = {
        "before_val_bpb": round(before_bpb, 4),
        "after_val_bpb": round(after_bpb, 4),
        "delta_bpb": round(after_bpb - before_bpb, 4),
        "steps": steps,
        "tokens": tokens,
        "train_bytes": len(train_bytes),
        "val_bytes": len(val_bytes),
        "elapsed_s": round(elapsed_s, 3),
        "params": model.n_params(),
        "checkpoint": str(out_path),
    }

    tmp_file = out_path.with_suffix(".pt.tmp")
    torch.save({"config": cfg.to_dict(), "state": model.state_dict(), "wake_sleep_meta": meta}, tmp_file)
    tmp_file.replace(out_path)

    if memory is not None and hasattr(memory, "remember"):
        try:
            memory.remember(
                f"Wake-Sleep v2 distillation: val_bpb {meta['before_val_bpb']} -> {meta['after_val_bpb']} in {steps} steps",
                kind="episodic",
                tags=["wake_sleep", "v2", "distill"],
                source="sleep_distill",
            )
        except Exception:
            pass

    return meta


def _extract_python_block(prefix: str, generated: str) -> str | None:
    raw = prefix + generated.split("</SOL>")[0].split("<SPEC>")[0]
    lines: list[str] = []
    started = False
    for ln in raw.splitlines():
        if not started and ln.startswith(("def ", "class ", "from ", "import ")):
            started = True
        if not started:
            continue
        if lines and ln and not ln.startswith((" ", "\t", ")", "]", "}", "def ", "class ", "from ", "import ")):
            break
        lines.append(ln)
    code = "\n".join(lines).strip() + "\n"
    if not code.strip():
        return None
    try:
        ast.parse(code)
    except SyntaxError:
        return None
    return code


class NeuralCodeProposer:
    """Open-ended CPU neural code proposal backend backed by Native Engine (never trusted unverified)."""

    def __init__(self, checkpoint_path: str | Path, max_bytes: int = 240):
        self.checkpoint_path = Path(checkpoint_path)
        self.max_bytes = max_bytes
        self._engine: Any | None = None

    def _get_engine(self):
        if self._engine is None and self.checkpoint_path.is_file():
            from .engine import Engine

            self._engine = Engine.load(self.checkpoint_path, packed=True)
        return self._engine

    def __call__(self, problem: Any, history: list[Any]) -> str | None:
        eng = self._get_engine()
        if eng is None or getattr(problem, "domain", "code") != "code":
            return None
        stmt = getattr(problem, "statement", str(problem))
        prefix = "class " if "class " in stmt else "def "
        prompt = f"<SPEC>\n{stmt}\n<SOL>\n{prefix}".encode("utf-8")
        attempt = len(history)
        temp = 0.0 if attempt == 0 else min(0.85, 0.25 * attempt)
        gen = eng.generate(prompt, self.max_bytes, temperature=temp, seed=attempt + 1)
        return _extract_python_block(prefix, gen.decode("utf-8", errors="replace"))
