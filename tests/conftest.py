import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omega_nano.config import load_config  # noqa: E402
from omega_nano.engine import Engine  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    c = load_config(tmp_path / "home")
    c.backoff_base_s = 0.001
    return c


@pytest.fixture
def engine(cfg):
    e = Engine(cfg)
    yield e
    e.close()


def run_goal(engine, goal, **kw):
    task = engine.submit(goal, **kw)
    if task.status.value == "planned":
        task = engine.run(task.id)
    return task


@pytest.fixture
def cli(tmp_path):
    home = tmp_path / "clihome"

    def _cli(*args, fault=None):
        env = {**os.environ, "OMEGA_HOME": str(home), "PYTHONIOENCODING": "utf-8"}
        env.pop("OMEGA_FAULT", None)
        if fault:
            env["OMEGA_FAULT"] = fault
        p = subprocess.run([sys.executable, "-m", "omega_nano", *args], cwd=str(ROOT), env=env,
                           capture_output=True, text=True, encoding="utf-8", timeout=120)
        return p.returncode, p.stdout + p.stderr

    _cli.home = home
    return _cli
