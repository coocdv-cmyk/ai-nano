"""Tool contract.

A tool is a typed, bounded action. It declares its parameters, the scopes it
needs, and whether re-running it is safe (idempotent). Arguments are validated
before execution and outputs are validated after, so a malformed tool response
becomes an observed failure instead of corrupting state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..security import Scope

_TYPES: dict[str, tuple[type, ...]] = {
    "str": (str,), "int": (int,), "float": (int, float), "bool": (bool,),
    "list": (list,), "dict": (dict,), "any": (object,),
}


class ToolError(Exception):
    """Raised for invalid arguments or tool-level failures that must be reported, not crash the agent."""


@dataclass
class Param:
    type: str
    required: bool = True
    default: Any = None
    doc: str = ""


@dataclass
class ToolContext:
    workspace: Path
    home: Path
    memory: Any = None  # MemoryFabric, optional
    config: Any = None


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, Param]
    scopes: frozenset[Scope]
    fn: Callable[..., dict[str, Any]]
    idempotent: bool = True
    # Optional hook that raises the needed scopes for specific arguments, e.g. an
    # overwrite of an existing file becomes DESTRUCTIVE.
    dynamic_scopes: Callable[[ToolContext, dict[str, Any]], frozenset[Scope]] | None = None
    examples: list[str] = field(default_factory=list)

    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ToolError(f"{self.name}: arguments must be an object")
        unknown = set(args) - set(self.params)
        if unknown:
            raise ToolError(f"{self.name}: unknown argument(s) {sorted(unknown)}")
        clean: dict[str, Any] = {}
        for pname, p in self.params.items():
            if pname not in args or args[pname] is None:
                if p.required:
                    raise ToolError(f"{self.name}: missing required argument '{pname}'")
                clean[pname] = p.default
                continue
            v = args[pname]
            if p.type == "int" and isinstance(v, bool):
                raise ToolError(f"{self.name}: '{pname}' must be int")
            if not isinstance(v, _TYPES[p.type]):
                raise ToolError(f"{self.name}: '{pname}' must be {p.type}, got {type(v).__name__}")
            clean[pname] = v
        return clean

    def needed_scopes(self, ctx: ToolContext, args: dict[str, Any]) -> frozenset[Scope]:
        extra = self.dynamic_scopes(ctx, args) if self.dynamic_scopes else frozenset()
        return self.scopes | extra

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        out = self.fn(ctx, **args)
        if not isinstance(out, dict) or not isinstance(out.get("ok"), bool):
            raise ToolError(f"{self.name}: malformed tool response (expected an object with boolean 'ok')")
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "params": {k: {"type": p.type, "required": p.required, "doc": p.doc} for k, p in self.params.items()},
            "scopes": sorted(s.value for s in self.scopes),
            "idempotent": self.idempotent,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"unknown tool '{name}'")
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> list[Tool]:
        return [self._tools[n] for n in self.names()]


def default_registry() -> ToolRegistry:
    from . import coretools, fs, mathx, memtools, shell

    reg = ToolRegistry()
    for mod in (fs, mathx, shell, memtools, coretools):
        for tool in mod.TOOLS:
            reg.register(tool)
    return reg
