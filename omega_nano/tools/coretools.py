"""Expose the project's own neural core to plans: code/text continuation by OmegaCore."""

from __future__ import annotations

import os
from pathlib import Path

from .base import Param, Tool, ToolContext, ToolError

_ENGINES: dict[str, object] = {}
DEFAULT = Path(__file__).resolve().parents[2] / "EXPERIMENTS" / "EXP-003-omega-core-v0" / "runs" / "ternary" / "model.pt"


def _engine(path: Path):
    key = str(path)
    if key not in _ENGINES:
        try:
            from ..core.engine import Engine
        except ImportError as e:
            raise ToolError(f"core.complete needs numpy, numba and torch: {e}") from None
        if not path.is_file():
            raise ToolError(f"no trained core at {path}; train one with python -m omega_nano.core.train")
        _ENGINES[key] = Engine.load(path)
    return _ENGINES[key]


def _complete(ctx: ToolContext, prompt: str, n: int, temperature: float, seed: int) -> dict:
    path = Path(os.environ.get("OMEGA_CORE_MODEL", DEFAULT))
    eng = _engine(path)
    n = max(1, min(n, 4096))
    out = eng.generate(prompt.encode("utf-8"), n, temperature=temperature, seed=seed)
    return {"ok": True, "completion": out.decode("utf-8", errors="replace"), "bytes": n,
            "model": path.parent.name, "trust": "model_generated_unverified"}


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
