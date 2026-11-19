"""End-to-end tests for M2 Wake-Sleep distillation and M4 unified OMEGA-NANO V1 OmegaAgent."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numba")

from omega_nano.agent import OmegaAgent  # noqa: E402
from omega_nano.brain import BlindProblem, CodeVerifier, SkillLibrary  # noqa: E402
from omega_nano.core.distill import NeuralCodeProposer, sleep_distill  # noqa: E402
from omega_nano.core.engine import Engine  # noqa: E402
from omega_nano.memory import MemoryFabric  # noqa: E402


def test_wake_sleep_distillation_improves_bpb_and_loads_natively(tmp_path):
    mem = MemoryFabric(tmp_path / "mem.db")
    skills = SkillLibrary(tmp_path / "brain.db", memory=mem)
    skills.add_replay(
        "r1", "code",
        "Write Python function sq_sum(nums: list[int]) -> int.",
        "def sq_sum(nums):\n    return sum(x * x for x in nums)\n",
        1, "TESTS_PASSED", cycle=1,
    )
    ckpt = tmp_path / "core_v2.pt"
    stats = sleep_distill(
        ckpt, skills=skills, memory=mem, budget_s=10.0, steps_cap=30, batch_size=4, lr=2e-3, ternary=True
    )
    assert ckpt.is_file()
    assert stats["steps"] >= 5
    assert stats["after_val_bpb"] < stats["before_val_bpb"], stats

    eng = Engine.load(ckpt, packed=True)
    assert eng.arch == "v2"
    proposer = NeuralCodeProposer(ckpt, max_bytes=120)
    # Smoke call must run natively on CPU without crashing (whether proposal is accepted is verifier-checked).
    prob = BlindProblem(
        id="smoke", domain="code", title="Smoke", difficulty=1,
        statement="Write Python function clamp(x: int, lo: int, hi: int) -> int.",
        verify=lambda code: CodeVerifier.verify_python(code, "assert clamp(5, 0, 3) == 3\n"),
    )
    _ = proposer(prob, [])
    skills.close()
    mem.close()


def test_omega_agent_end_to_end_offline_capabilities(tmp_path):
    home = tmp_path / "omega_home"
    agent = OmegaAgent(home=home)
    try:
        # 1. Seed a tiny workspace python module for RepoGraph indexing
        (agent.cfg.workspace / "helpers.py").write_text(
            "def compute_invoice_total(items):\n    '''Sum line items.'''\n    return sum(items)\n",
            encoding="utf-8",
        )
        idx_stats = agent.index_codebase()
        assert idx_stats["files_indexed"] >= 1 and idx_stats["total_symbols"] >= 1

        # 2. Multi-step checkpointed math + disk verification task
        res_ws = agent.handle("calculate 12 * 8 then save the result to calc.txt")
        assert res_ws["ok"] and res_ws["verified"] and res_ws["domain"] == "workspace_agent"
        assert res_ws["rss_mb"] > 0
        assert (agent.cfg.workspace / "calc.txt").read_text(encoding="utf-8").strip() == "96"

        # 3. Live verified Python coding specification (synthesizer + CodeVerifier)
        code_goal = (
            "Write Python function leftmost_pos(nums: list[int], target: int) -> int "
            "returning the binary search lower bound index in sorted nums.\n"
            "tests:\n"
            "assert leftmost_pos([1, 2, 2, 5], 2) == 1\n"
            "assert leftmost_pos([1, 3, 5], 4) == 2\n"
        )
        res_code = agent.handle(code_goal)
        assert res_code["ok"] and res_code["verified"] and res_code["domain"] == "code_synthesis"

        # 4. Bilingual semantic remember + recall
        r_mem = agent.handle("سجل عندك ان مشروع اوميجا يعمل محليا بدون انترنت")
        assert r_mem["ok"]
        r_rec = agent.handle("احكيلي عن مشروع اوميجا")
        assert r_rec["ok"]

        # 5. Offline Wake-Sleep sleep() step builds v2 core checkpoint cleanly
        sleep_stats = agent.sleep(budget_s=8.0, steps_cap=16)
        assert sleep_stats["steps"] >= 4 and agent.status()["core_ready"] is True
    finally:
        agent.close()
