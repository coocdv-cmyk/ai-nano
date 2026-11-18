"""Runtime configuration: where state lives and how the agent is bounded."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    home: Path
    max_retries: int = 2  # retries per step before asking the reasoner to revise
    max_revisions: int = 2  # reasoner revisions per task before failing
    step_timeout_s: float = 30.0
    backoff_base_s: float = 0.05  # bounded exponential backoff between retries
    backoff_max_s: float = 1.0
    lease_s: float = 30.0  # a RUNNING task whose lease expired is considered crashed
    llm_url: str | None = None  # OpenAI-compatible local endpoint (llama.cpp, Ollama, vLLM)
    llm_model: str | None = None
    shell_allow: tuple[str, ...] = field(
        default_factory=lambda: (
            "python", "py", "git", "echo", "dir", "ls", "type", "cat",
            "node", "npm", "pytest", "where", "whoami", "ver",
        )
    )

    @property
    def db_path(self) -> Path:
        return self.home / "state.db"

    @property
    def workspace(self) -> Path:
        return self.home / "workspace"

    @property
    def stop_file(self) -> Path:
        return self.home / "STOP"

    def ensure(self) -> "Config":
        self.home.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)
        return self


def load_config(home: str | os.PathLike | None = None) -> Config:
    root = Path(home or os.environ.get("OMEGA_HOME") or Path.cwd() / ".omega").resolve()
    return Config(
        home=root,
        llm_url=os.environ.get("OMEGA_LLM_URL"),
        llm_model=os.environ.get("OMEGA_LLM_MODEL"),
    ).ensure()
