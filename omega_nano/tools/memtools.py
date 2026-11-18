"""Tools that expose the Memory Fabric to plans."""

from __future__ import annotations

from ..security import Scope
from .base import Param, Tool, ToolContext, ToolError


def _need(ctx: ToolContext):
    if ctx.memory is None:
        raise ToolError("memory fabric is not available")
    return ctx.memory


def _remember(ctx: ToolContext, text: str, kind: str, key: str | None, source: str) -> dict:
    res = _need(ctx).remember(text, kind=kind, key=key, source=source)
    return {"ok": True, **res}


def _recall(ctx: ToolContext, query: str, k: int) -> dict:
    # Episodic records of past tasks are excluded so they don't crowd out facts.
    hits = _need(ctx).recall(query, k=max(1, min(k, 50)), exclude=("episodic",))
    return {"ok": True, "hits": [h.to_dict() for h in hits], "count": len(hits),
            "answer": hits[0].text if hits else None}


TOOLS = [
    Tool(
        name="memory.remember",
        description="Store a fact with provenance. Re-using a key supersedes the old version (kept as history).",
        params={"text": Param("str"), "kind": Param("str", required=False, default="semantic"),
                "key": Param("str", required=False), "source": Param("str", required=False, default="user")},
        scopes=frozenset({Scope.WRITE}), fn=_remember,
    ),
    Tool(
        name="memory.recall",
        description="Hybrid lexical + vector retrieval over memory, with provenance and confidence.",
        params={"query": Param("str"), "k": Param("int", required=False, default=5)},
        scopes=frozenset({Scope.READ}), fn=_recall,
    ),
]
