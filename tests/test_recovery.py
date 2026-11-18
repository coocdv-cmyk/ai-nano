"""Restart and crash tests: a real process is killed mid-task (os._exit) and a fresh process resumes."""

import sqlite3


def test_crash_after_execution_recovers_without_reexecution(cli):
    code, out = cli("run", "write file c1.txt with one then append two to file c2.txt then write file c3.txt with three",
                    fault="crash_after_exec:1")
    assert code == 77
    # The append (non-overwrite, so re-running it would duplicate text) already happened before the crash.
    assert (cli.home / "workspace" / "c2.txt").read_text(encoding="utf-8") == "two\n"
    code, out = cli("resume")
    assert code == 0 and "[OK]" in out and "(recovered)" in out, out
    assert (cli.home / "workspace" / "c2.txt").read_text(encoding="utf-8") == "two\n"  # not appended twice
    assert (cli.home / "workspace" / "c3.txt").read_text(encoding="utf-8") == "three"


def test_crash_before_execution_reexecutes_idempotent_step(cli):
    code, _ = cli("run", "write file a.txt with A then write file b.txt with B", fault="crash_after_start:1")
    assert code == 77 and not (cli.home / "workspace" / "b.txt").exists()
    code, out = cli("resume")
    assert "[OK]" in out and (cli.home / "workspace" / "b.txt").read_text(encoding="utf-8") == "B"


def test_non_idempotent_step_escalates_to_human_after_crash(cli):
    code, _ = cli("run", "run python -c \"print(1)\"", "--mode", "autonomous", fault="crash_after_start:0")
    assert code == 77
    code, out = cli("resume")
    assert "[BLOCKED]" in out and "not safe to repeat" in out
    task_id = out.split("task ")[1].split()[0]
    code, out = cli("approve", task_id)
    assert code == 0 and "[OK]" in out


def test_resume_after_corrupted_current_state(cli):
    code, _ = cli("run", "write file x.txt with X then write file y.txt with Y", fault="crash_after_exec:1")
    assert code == 77
    db = sqlite3.connect(str(cli.home / "state.db"))
    db.execute("UPDATE tasks SET state_json = state_json || ' '")  # checksum now mismatches
    db.commit()
    db.close()
    code, out = cli("resume")
    assert "[OK]" in out, out
    code, out = cli("report", out.split("task ")[1].split()[0])
    assert "corruption_detected" in out and "restored_checkpoint" in out


def test_duplicate_delivery_across_processes(cli):
    cli("run", "append line to file log.txt", "--key", "msg-42")
    cli("run", "append line to file log.txt", "--key", "msg-42")
    assert (cli.home / "workspace" / "log.txt").read_text(encoding="utf-8") == "line\n"


def test_approval_survives_restart(cli):
    code, out = cli("run", "write file p.txt with pending", "--mode", "ask")
    assert code == 2 and "APPROVAL?" in out
    task_id = out.split("task ")[1].split()[0]
    code, out = cli("resume")
    assert "nothing to resume" in out  # waiting tasks are not auto-run
    code, out = cli("approve", task_id)
    assert "[OK]" in out and (cli.home / "workspace" / "p.txt").exists()
