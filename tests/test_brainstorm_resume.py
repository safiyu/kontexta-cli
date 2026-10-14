import asyncio
import os
import subprocess
import sys

import pytest

from myagy.agents import brainstorm as bs
from myagy.agents.brainstorm import BrainstormWorkflow as B
from myagy.terminal import say

PLAN = "# Plan\n\n## Milestones\n\n1. add a\n2. add b\n   - detail line\n\n## Risks\n\n- none\n"


@pytest.fixture(autouse=True)
def say_state():
    say.configure(json_output=False, verbose=False)
    yield
    say.configure(json_output=False, verbose=False)


def test_normalize_plan_checks_items_and_keeps_details():
    out = B._normalize_plan(PLAN)
    assert "- [ ] add a" in out and "- [ ] add b" in out and "   - detail line" in out
    assert "- none" in out
    assert [t for _, t in B._milestones(out)] == ["add a", "add b"]


def test_normalize_plan_adds_fallback_milestone():
    out = B._normalize_plan("# Plan\n- edit a.txt")
    assert B._milestones(out) == [(False, "Implement the full plan")]


def test_mark_done_ticks_only_the_requested_milestone(tmp_path):
    p = tmp_path / "plan.md"
    p.write_text(B._normalize_plan(PLAN))
    B._mark_done(str(p), 1, " (skipped)")
    assert B._milestones(p.read_text()) == [(False, "add a"), (True, "add b (skipped)")]


def test_meta_roundtrip_and_strip(tmp_path):
    p = tmp_path / "plan.md"
    p.write_text("# Plan\n")
    B._meta_set(str(p), "idea", "do x")
    B._meta_set(str(p), "approved", "1")
    B._meta_set(str(p), "approved", "2")
    text = p.read_text()
    assert B._meta_get(text, "idea") == "do x" and B._meta_get(text, "approved") == "2"
    assert B._strip_meta(text) == "# Plan"


def _git(repo, *a):
    return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True)


def test_pull_skipped_when_dirty_or_no_upstream(git_repo):
    assert B._pull_latest(str(git_repo), {"x"}) is True
    assert B._pull_latest(str(git_repo), set()) is True


def test_pull_fast_forwards_from_remote(git_repo, tmp_path_factory):
    remote = tmp_path_factory.mktemp("remote")
    other = tmp_path_factory.mktemp("other")
    _git(remote, "init", "-q", "--bare")
    branch = _git(git_repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    _git(git_repo, "remote", "add", "origin", str(remote))
    _git(git_repo, "push", "-q", "-u", "origin", branch)
    subprocess.run(["git", "clone", "-q", str(remote), str(other / "c")], check=True)
    c = other / "c"
    _git(c, "config", "user.email", "t@example.com")
    _git(c, "config", "user.name", "tester")
    (c / "new.txt").write_text("n\n")
    _git(c, "add", ".")
    _git(c, "commit", "-qm", "remote change")
    _git(c, "push", "-q")
    assert B._pull_latest(str(git_repo), set()) is True
    assert (git_repo / "new.txt").exists()


def test_pull_failure_stops_workflow(git_repo, tmp_path_factory):
    remote = tmp_path_factory.mktemp("remote")
    _git(remote, "init", "-q", "--bare")
    branch = _git(git_repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    _git(git_repo, "remote", "add", "origin", str(remote))
    _git(git_repo, "push", "-q", "-u", "origin", branch)
    _git(git_repo, "remote", "set-url", "origin", str(remote) + "_missing")
    assert B._pull_latest(str(git_repo), set()) is False


class FlakySession:
    json_output = False
    endpoints = {}
    history = []

    def __init__(self, repo, fail_first_turn=False, drafts_plan=True):
        self.repo = repo
        self.active_target = "9000"
        self.cloud_calls = 0
        self.turns = []
        self.fail_first_turn = fail_first_turn
        self.drafts_plan = drafts_plan

    async def chat_cloud_oauth(self, prompt, with_project_context=True):
        self.cloud_calls += 1
        return PLAN if self.drafts_plan and self.cloud_calls == 1 else "VERIFIED_CLEAN"

    async def chat_turn(self, prompt, target=None):
        if self.fail_first_turn and not self.turns:
            self.turns.append(("boom", target))
            raise RuntimeError("unexpected EOF")
        self.turns.append((prompt, target))
        (self.repo / f"f{len(self.turns)}.txt").write_text("x\n")

    async def reset_local_agent(self):
        pass


def _patch(monkeypatch, answers):
    monkeypatch.setattr(bs.kontexta_sync, "sync_event", lambda *a, **k: True)
    it = iter(answers)
    monkeypatch.setattr(bs, "prompt_input", lambda prompt="": next(it))
    monkeypatch.setattr(B, "_show_markdown", staticmethod(lambda *a, **k: None))


def test_crash_mid_build_is_resumable_and_plan_is_in_repo(git_repo, monkeypatch, capsys):
    _patch(monkeypatch, ["y", "r", "n"])
    first = FlakySession(git_repo, fail_first_turn=True)
    asyncio.run(B.execute_workflow("build ab", first, str(git_repo)))
    out = capsys.readouterr().out
    assert "Turn failed" in out and "resume" in out.lower()

    plans = list((git_repo / "docs" / "plans").glob("build_ab_*.md"))
    assert len(plans) == 1
    assert all(not done for done, _ in B._milestones(plans[0].read_text()))

    second = FlakySession(git_repo, drafts_plan=False)
    asyncio.run(B.execute_workflow("build ab", second, str(git_repo)))
    assert second.cloud_calls == 1  # review only, no new plan
    assert len(second.turns) >= 2
    assert all(done for done, _ in B._milestones(plans[0].read_text()))
    assert "docs/plans" not in (B._get_git_diff(str(git_repo)) or "")


def test_esc_stops_only_the_turn(monkeypatch):
    class FakeEsc:
        def __init__(self, on_escape=None):
            self.on_escape = on_escape
            self.interrupted = False

        def __enter__(self):
            loop = asyncio.get_running_loop()

            def fire():
                self.interrupted = True
                self.on_escape()

            loop.call_later(0.05, fire)
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(bs, "EscListener", FakeEsc)

    class S:
        async def reset_local_agent(self):
            pass

    async def slow():
        await asyncio.sleep(5)

    status, _ = asyncio.run(B._interruptible(S(), slow()))
    assert status == "stopped"


def test_outer_cancel_still_propagates(monkeypatch):
    class Quiet:
        interrupted = False

        def __init__(self, on_escape=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(bs, "EscListener", Quiet)

    async def scenario():
        t = asyncio.ensure_future(B._interruptible(object(), asyncio.sleep(5)))
        await asyncio.sleep(0.05)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t

    asyncio.run(scenario())


def test_control_switches_model_before_next_milestone(monkeypatch):
    from myagy.terminal import cli

    calls = []

    async def fake_switch(session, arg, ctx):
        calls.append(arg)
        session.active_target = "9001"

    monkeypatch.setitem(cli.COMMANDS, "/switch", fake_switch)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(B, "_read_line_timeout", staticmethod(lambda s: "/switch qwen"))

    class S:
        json_output = False
        active_target = "9000"

    s = S()
    assert asyncio.run(B._control(s, "Next: m1")) == "continue"
    assert calls == ["qwen"] and s.active_target == "9001"


def test_control_quits_and_skips(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)

    class S:
        json_output = False
        active_target = "9000"

    for typed, expected in (("q", "quit"), ("s", "skip"), ("", "continue")):
        monkeypatch.setattr(B, "_read_line_timeout", staticmethod(lambda sec, t=typed: t))
        assert asyncio.run(B._control(S(), "Next: m1")) == expected
