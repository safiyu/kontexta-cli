import asyncio
import subprocess
import types

import pytest

from kontexta_cli.agents import brainstorm as bs
from kontexta_cli.agents.brainstorm import BrainstormWorkflow as B
from kontexta_cli.agents.subagent import SubagentManager, SubagentTask
from kontexta_cli.core.coordinator import CudaCoordinator
from kontexta_cli.core.sudo_manager import SudoManager
from kontexta_cli.terminal import say


@pytest.fixture(autouse=True)
def say_state():
    say.configure(json_output=False, verbose=False)
    yield
    say.configure(json_output=False, verbose=False)


def test_sudo_prompt_shows_danger_card(monkeypatch, capsys):
    SudoManager.cleanup()
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "pw")
    monkeypatch.setattr(SudoManager, "verify_and_authenticate", classmethod(lambda cls, p: (True, "ok")))
    assert SudoManager.prompt_password_entry(reason="sudo apt install htop")
    out = capsys.readouterr().out
    assert "sudo required" in out
    assert "sudo apt install htop" in out
    assert "this command needs root privileges" in out
    assert "password prompt follows" in out
    assert "unlocked for 15 minutes" in out
    SudoManager.cleanup()


def test_sudo_prompt_failure_lines(monkeypatch, capsys):
    SudoManager.cleanup()
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "bad")
    monkeypatch.setattr(SudoManager, "verify_and_authenticate", classmethod(lambda cls, p: (False, "Incorrect sudo password.")))
    assert SudoManager.prompt_password_entry(reason="sudo id", max_attempts=2) is False
    out = capsys.readouterr().out
    assert "1 attempt(s) remaining" in out
    assert "maximum attempts exceeded" in out
    SudoManager.cleanup()


class FakeSession:
    json_output = False
    endpoints = {}
    history = []
    laya = None

    def __init__(self, repo):
        self.repo = repo
        self.active_target = "9000"
        self.cloud_calls = 0

    async def chat_cloud_oauth(self, prompt, with_project_context=True):
        self.cloud_calls += 1
        return "# Plan\n- edit a.txt" if self.cloud_calls == 1 else "VERIFIED_CLEAN"

    async def chat_turn(self, prompt, target=None):
        (self.repo / "a.txt").write_text("agent edit\n")


def _run_workflow(git_repo, monkeypatch, answers):
    monkeypatch.setattr(bs.kontexta_sync, "sync_event", lambda *a, **k: True)
    it = iter(answers)
    monkeypatch.setattr(bs, "prompt_input", lambda prompt="": next(it))
    monkeypatch.setattr(bs.BrainstormWorkflow, "_show_markdown", staticmethod(lambda *a, **k: None))
    session = FakeSession(git_repo)
    asyncio.run(B.execute_workflow("tweak a", session, str(git_repo)))


def test_gate_cards_and_phase_headings(git_repo, monkeypatch, capsys):
    _run_workflow(git_repo, monkeypatch, ["y", "n"])
    out = capsys.readouterr().out
    assert "Spec-driven workflow" in out and "tweak a" in out
    assert "Gate 1: spec approval" in out and "[y]es / [n]o" in out
    assert "Phase 2/5" in out and "Phase 5/5" in out
    assert "Workflow complete: ready to commit" in out
    assert "files changed" in out and "feat(tweak_a): tweak a" in out
    assert "Changes left in working directory uncommitted." in out
    assert "╭" not in out


def test_rejected_plan_stops_at_gate_one(git_repo, monkeypatch, capsys):
    _run_workflow(git_repo, monkeypatch, ["n"])
    out = capsys.readouterr().out
    assert "Gate 1: spec approval" in out
    assert "Plan not approved" in out
    assert "Phase 2/5" not in out


def test_commit_gate_commits_and_reports(git_repo, monkeypatch, capsys):
    _run_workflow(git_repo, monkeypatch, ["y", "y"])
    out = capsys.readouterr().out
    assert "Changes successfully committed" in out
    log = subprocess.run(["git", "log", "-1", "--format=%s"], cwd=git_repo, capture_output=True, text=True).stdout
    assert log.startswith("feat(tweak_a)")


def test_step_lines_quiet_and_verbose(capsys):
    say.step("working memory: 4 older turns distilled via granite")
    say.detail("long form only when verbose")
    out = capsys.readouterr().out
    assert "working memory" in out and "long form" not in out
    say.configure(verbose=True)
    say.detail("long form only when verbose")
    assert "long form" in capsys.readouterr().out


def test_compact_history_reports_one_line(session, monkeypatch, capsys):
    from kontexta_cli.agents.compactor import ContextCurator
    session.json_output = False
    session.history = [{"role": "user", "target": "9000", "content": f"m{i}"} for i in range(8)]
    monkeypatch.setattr(ContextCurator, "synthesize_working_memory", classmethod(lambda cls, s, older, silent=False: ("summary", "test-tier")))
    removed = ContextCurator.compact_history(session, keep_recent=2)
    out = capsys.readouterr().out
    assert removed == 6
    assert "synthesizing working memory" in out
    assert "History compacted via test-tier" in out and "recent kept verbatim" in out


def test_context_summary_card(session, capsys):
    from kontexta_cli.agents.compactor import ContextCurator
    session.json_output = False
    ContextCurator.print_context_summary(session)
    out = capsys.readouterr().out
    assert "Context window" in out and "headroom" in out and "autocompact" in out
    assert "╭" not in out


@pytest.fixture
def online_session(session):
    session.endpoints["9002"].update(status="online", filename="granite.gguf", model="/m/granite.gguf", url="http://localhost:9002/v1")
    session.json_output = False
    return session


def _worker(task, session):
    async def go():
        CudaCoordinator.reserve_for_subagent()
        await SubagentManager._run_worker(task, session)
    asyncio.run(go())


def test_subagent_completed_line(online_session, monkeypatch, capsys):
    async def fake_run(cls, task, session, ep):
        return "all good"
    monkeypatch.setattr(SubagentManager, "_run_with_tools", classmethod(fake_run))
    task = SubagentTask(7, "inspect things")
    _worker(task, online_session)
    out = capsys.readouterr().out
    assert task.status == "completed"
    assert "Subagent #7 done in" in out and "inspect things" in out
    assert "/subagent view 7" in out


def test_subagent_failed_line(online_session, monkeypatch, capsys):
    async def boom(cls, task, session, ep):
        raise RuntimeError("kaput")
    async def boom_plain(cls, task, ep):
        raise RuntimeError("kaput")
    monkeypatch.setattr(SubagentManager, "_run_with_tools", classmethod(boom))
    monkeypatch.setattr(SubagentManager, "_run_plain", classmethod(boom_plain))
    task = SubagentTask(8, "break things")
    _worker(task, online_session)
    out = capsys.readouterr().out
    assert task.status == "failed"
    assert "Subagent #8 failed after" in out and "kaput" in out


def test_subagent_cancelled_line(online_session, monkeypatch, capsys):
    async def slow(cls, task, session, ep):
        raise asyncio.CancelledError()
    monkeypatch.setattr(SubagentManager, "_run_with_tools", classmethod(slow))
    task = SubagentTask(9, "slow things")
    _worker(task, online_session)
    out = capsys.readouterr().out
    assert task.status == "cancelled"
    assert "subagent #9 cancelled" in out


def test_subagent_task_table_and_detail(monkeypatch, capsys):
    SubagentManager._tasks.clear()
    t = SubagentTask(3, "list files")
    t.status, t.result, t.duration = "completed", "a.txt b.txt", 1.5
    SubagentManager._tasks.append(t)
    SubagentManager.print_tasks(types.SimpleNamespace(json_output=False))
    out = capsys.readouterr().out
    assert "Background subagents" in out and "list files" in out and "a.txt b.txt" in out
    SubagentManager.print_detail(3)
    out = capsys.readouterr().out
    assert "Subagent #3" in out and "a.txt b.txt" in out and "/subagent inject 3" in out
    SubagentManager.print_detail(99)
    assert "not found" in capsys.readouterr().out
    SubagentManager._tasks.clear()


def test_background_denial_warns(online_session, monkeypatch, capsys):
    import kontexta_cli.agents.subagent as sa
    from test_subagent import _fake_agent
    captured = {}

    class Fake(_fake_agent("ok")):
        def __init__(self, config):
            captured["policies"] = config.kwargs["policies"]
            super().__init__(config)

    monkeypatch.setattr(sa, "Agent", Fake)
    task = SubagentTask(4, "danger")
    _worker(task, online_session)
    capsys.readouterr()
    handler = captured["policies"][1]
    assert handler(types.SimpleNamespace(name="run_command"), {"CommandLine": "sudo id"}) is False
    out = capsys.readouterr().out
    assert "Subagent #4: denied run_command" in out and "sudo is not allowed" in out
    assert handler(types.SimpleNamespace(name="run_command"), {"CommandLine": "rm -rf /"}) is False
    assert "not auto-approved as safe" in capsys.readouterr().out
