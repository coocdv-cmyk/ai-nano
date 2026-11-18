"""Expose the project's own neural core to plans: code/text continuation by OmegaCore."""

from __future__ import annotations

import os
from pathlib import Path

from .base import Param, Tool, ToolContext, ToolError

_ENGINES: dict[str, object] = {}
_ROOT = Path(__file__).resolve().parents[2]
_CANDIDATE_MODELS: tuple[Path, ...] = (
    _ROOT / "EXPERIMENTS" / "EXP-005-m1-v2-core" / "runs" / "v2_2.00M_5min" / "model.pt",
    _ROOT / "EXPERIMENTS" / "EXP-005-m1-v2-core" / "runs" / "v2_hybrid_ternary" / "model.pt",
    _ROOT / "EXPERIMENTS" / "EXP-003-omega-core-v0" / "runs" / "ternary" / "model.pt",
)
DEFAULT = next((p for p in _CANDIDATE_MODELS if p.is_file()), _CANDIDATE_MODELS[-1])


def _resolve_model(ctx: ToolContext) -> Path:
    if env := os.environ.get("OMEGA_CORE_MODEL"):
        return Path(env)
    sleep_ckpt = ctx.home / "core_v2.pt"
    if sleep_ckpt.is_file():
        return sleep_ckpt
    return next((p for p in _CANDIDATE_MODELS if p.is_file()), _CANDIDATE_MODELS[-1])


def _engine(path: Path):
    key = str(path)
    if key not in _ENGINES:
        try:
            from ..core.engine import Engine
        except ImportError as e:
            raise ToolError(f"core.complete needs numpy, numba and torch: {e}") from None
        if not path.is_file():
            raise ToolError(f"no trained core at {path}; train one with python -m omega_nano.core.train")
        _ENGINES[key] = Engine.load(path, packed=True)
    return _ENGINES[key]


def _complete(ctx: ToolContext, prompt: str, n: int, temperature: float, seed: int) -> dict:
    path = _resolve_model(ctx)
    eng = _engine(path)
    n = max(1, min(n, 4096))
    out = eng.generate(prompt.encode("utf-8"), n, temperature=temperature, seed=seed)
    return {
        "ok": True,
        "completion": out.decode("utf-8", errors="replace"),
        "bytes": n,
        "model": f"{path.parent.name} ({eng.arch})",
        "trust": "model_generated_unverified",
    }


TOOLS = [
    Tool(
        name="core.complete",
        description="Continue text/code with OMEGA's own byte-level neural core (small, trained locally).",
        params={"prompt": Param("str"), "n": Param("int", required=False, default=200),
                "temperature": Param("float", required=False, default=0.7),
                "seed": Param("int", required=False, default=0)},
        scopes=frozenset(), fn=_complete,
    )
]
