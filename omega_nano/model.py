"""Task and step records plus the explicit task state machine."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    PLANNED = "planned"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    PAUSED = "paused"  # emergency stop
    BLOCKED = "blocked"  # cannot proceed without a human or a better reasoner
    DONE = "done"
    FAILED = "failed"


class StepStatus(str, Enum):
    PENDING = "pending"
    STARTED = "started"  # write-ahead marker: execution may or may not have happened
    DONE = "done"
    FAILED = "failed"
    AWAITING_APPROVAL = "awaiting_approval"
    DENIED = "denied"


TERMINAL = {TaskStatus.DONE, TaskStatus.FAILED}

# Every legal transition. Anything else is a bug and raises.
TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.PLANNED, TaskStatus.BLOCKED, TaskStatus.FAILED},
    TaskStatus.PLANNED: {TaskStatus.RUNNING, TaskStatus.PAUSED, TaskStatus.FAILED},
    TaskStatus.RUNNING: {
        TaskStatus.RUNNING, TaskStatus.AWAITING_APPROVAL, TaskStatus.PAUSED,
        TaskStatus.BLOCKED, TaskStatus.DONE, TaskStatus.FAILED,
    },
    TaskStatus.AWAITING_APPROVAL: {TaskStatus.RUNNING, TaskStatus.FAILED, TaskStatus.PAUSED},
    TaskStatus.PAUSED: {TaskStatus.RUNNING, TaskStatus.FAILED},
    TaskStatus.BLOCKED: {TaskStatus.RUNNING, TaskStatus.FAILED, TaskStatus.PLANNED},
    TaskStatus.DONE: set(),
    TaskStatus.FAILED: set(),
}


class IllegalTransition(RuntimeError):
    pass


@dataclass
class StepSpec:
    """What the reasoner proposes: one bounded tool call plus its postcondition."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    expect: list[dict[str, Any]] = field(default_factory=lambda: [{"kind": "ok"}])
    why: str = ""


@dataclass
class Step:
    idx: int
    tool: str
    args: dict[str, Any]
    expect: list[dict[str, Any]]
    why: str = ""
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    approved: bool = False
    output: dict[str, Any] | None = None
    error: str | None = None
    verification: list[dict[str, Any]] = field(default_factory=list)
    recovered: bool = False
    started_at: float | None = None
    ended_at: float | None = None

    @classmethod
    def from_spec(cls, idx: int, spec: StepSpec) -> "Step":
        return cls(idx=idx, tool=spec.tool, args=dict(spec.args), expect=list(spec.expect), why=spec.why)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Step":
        d = dict(d)
        d["status"] = StepStatus(d["status"])
        return cls(**d)


@dataclass
class Task:
    id: str
    goal: str
    status: TaskStatus = TaskStatus.PENDING
    mode: str = "workspace"
    idem_key: str | None = None
    understanding: dict[str, Any] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
    revisions: int = 0
    tier: str | None = None  # which adaptive-compute tier produced the plan
    summary: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    lease_until: float = 0.0
    lease_pid: int = 0  # process holding the lease; a dead owner releases it immediately

    def transition(self, new: TaskStatus) -> None:
        if new not in TRANSITIONS[self.status]:
            raise IllegalTransition(f"{self.status.value} -> {new.value}")
        self.status = new
        self.updated_at = time.time()

    @property
    def cursor(self) -> int:
        for s in self.steps:
            if s.status != StepStatus.DONE:
                return s.idx
        return len(self.steps)

    def to_state(self) -> dict[str, Any]:
        return {
            "id": self.id, "goal": self.goal, "status": self.status.value, "mode": self.mode,
            "idem_key": self.idem_key, "understanding": self.understanding,
            "steps": [s.to_dict() for s in self.steps], "revisions": self.revisions,
            "tier": self.tier, "summary": self.summary, "created_at": self.created_at,
            "updated_at": self.updated_at, "lease_until": self.lease_until, "lease_pid": self.lease_pid,
        }

    @classmethod
    def from_state(cls, d: dict[str, Any]) -> "Task":
        d = dict(d)
        d["status"] = TaskStatus(d["status"])
        d["steps"] = [Step.from_dict(s) for s in d["steps"]]
        return cls(**d)
