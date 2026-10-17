import asyncio
import json

import pytest

from kontexta_cli.core import session as session_mod
from kontexta_cli.core.prefs import load_prefs


class FakeStream:
    def __init__(self, text):
        self.text = text

    def __aiter__(self):
        async def gen():
            for tok in self.text.split(" "):
                yield tok + " "
        return gen()


class ScriptedAgent:
    """Agent whose chat() raises or replies according to a script of outcomes."""

    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = 0
        self.closed = False

    async def chat(self, prompt):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return FakeStream(outcome)

    async def __aexit__(self, *exc):
        self.closed = True


def _online(session):
    for ep in session.endpoints.values():
        ep.update(status="online", filename="m.gguf", model="/m/m.gguf", name_friendly="m")


def _bind_agent(session, monkeypatch, agent):
    async def get_agent(target):
        return agent
    monkeypatch.setattr(session, "get_or_switch_local_agent", get_agent)


CTX_ERR = RuntimeError("exceed_context_size_error: request exceeds the available context size")


def test_context_exceeded_on_last_attempt_escalates_instead_of_dropping_turn(session, monkeypatch):
    _online(session)
    session.enable_hooks = False
    # 3 normal attempts fail, then total distillation attempt also fails, triggering cloud fallback
    agent = ScriptedAgent([CTX_ERR, CTX_ERR, CTX_ERR, CTX_ERR])
    _bind_agent(session, monkeypatch, agent)
    monkeypatch.setattr(session, "compact_history", lambda keep_recent=None, silent=False: 1)
    session.history = [{"role": "user", "target": "9000", "content": "a"}, {"role": "assistant", "target": "9000", "content": "b"}, {"role": "user", "target": "9000", "content": "c"}]
    monkeypatch.setattr(session.laya, "choose_escalation_target", lambda *a, **k: ("cloud", 0.9))

    async def cloud(prompt, with_project_context=True):
        return "cloud answer"
    monkeypatch.setattr(session, "chat_cloud_oauth", cloud)

    out = asyncio.run(session.chat("big prompt", force_target="9000"))
    assert out == "cloud answer"
    assert agent.calls == 4
    assert session.history[-1] == {"role": "assistant", "target": "cloud", "content": "cloud answer"}
    assert session.stats["cloud"]["turns"] == 1


def test_context_exceeded_forces_total_distillation_and_succeeds_on_local(session, monkeypatch):
    _online(session)
    session.enable_hooks = False
    # 3 normal attempts fail, then post-total-distillation attempt succeeds locally
    agent = ScriptedAgent([CTX_ERR, CTX_ERR, CTX_ERR, "distilled answer"])
    _bind_agent(session, monkeypatch, agent)
    compact_calls = []

    def mock_compact(keep_recent=None, silent=False):
        compact_calls.append(keep_recent)
        return 1

    monkeypatch.setattr(session, "compact_history", mock_compact)
    session.history = [{"role": "user", "target": "9000", "content": "a"}, {"role": "assistant", "target": "9000", "content": "b"}]

    out = asyncio.run(session.chat("big prompt", force_target="9000"))
    assert out == "distilled answer"
    assert agent.calls == 4
    # The final compaction before retry was total distillation (keep_recent=0)
    assert compact_calls[-1] == 0
    assert session.history[-1] == {"role": "assistant", "target": "9000", "content": "distilled answer"}
    assert session.stats["9000"]["turns"] == 1


def test_context_exceeded_normal_compaction_zero_triggers_immediate_total_distillation(session, monkeypatch):
    _online(session)
    session.enable_hooks = False
    # Attempt 1 fails, normal compaction (keep_recent=2 and 1) returns 0,
    # immediately triggers total distillation (keep_recent=0) which succeeds!
    agent = ScriptedAgent([CTX_ERR, "distilled answer on attempt 2"])
    _bind_agent(session, monkeypatch, agent)
    compact_calls = []

    def mock_compact(keep_recent=None, silent=False):
        compact_calls.append(keep_recent)
        if keep_recent == 0:
            return 2  # Total distillation frees 2 entries
        return 0  # Normal compaction cannot free anything

    monkeypatch.setattr(session, "compact_history", mock_compact)
    session.history = [{"role": "user", "target": "9000", "content": "a"}, {"role": "assistant", "target": "9000", "content": "b"}]

    out = asyncio.run(session.chat("big prompt", force_target="9000"))
    assert out == "distilled answer on attempt 2"
    assert agent.calls == 2
    # keep_recent=2, then keep_recent=0 (total distillation)
    assert 0 in compact_calls
    assert session.history[-1] == {"role": "assistant", "target": "9000", "content": "distilled answer on attempt 2"}
    assert session.stats["9000"]["turns"] == 1


def test_context_exceeded_with_failed_escalation_raises_and_counts_error(session, monkeypatch):
    _online(session)
    session.enable_hooks = False
    agent = ScriptedAgent([CTX_ERR, CTX_ERR, CTX_ERR, CTX_ERR])
    _bind_agent(session, monkeypatch, agent)
    monkeypatch.setattr(session, "compact_history", lambda keep_recent=None, silent=False: 1)
    monkeypatch.setattr(session.laya, "choose_escalation_target", lambda *a, **k: ("cloud", 0.9))

    async def cloud(prompt, with_project_context=True):
        raise RuntimeError("Cloud model unreachable")
    monkeypatch.setattr(session, "chat_cloud_oauth", cloud)

    with pytest.raises(RuntimeError):
        asyncio.run(session.chat("big prompt", force_target="9000"))
    assert session.stats["9000"]["errors"] == 1
    assert "turns" not in session.stats["9000"]
    assert session.history == []


def test_empty_response_is_not_recorded_as_success(session, monkeypatch):
    _online(session)
    session.enable_hooks = False
    _bind_agent(session, monkeypatch, ScriptedAgent([""]))
    assert asyncio.run(session.chat("hi", force_target="9000")) == ""
    assert session.stats["9000"].get("turns", 0) == 0
    assert session.stats["9000"]["errors"] == 1


def test_error_stats_use_routed_target_not_auto(session, monkeypatch):
    _online(session)
    session.enable_hooks = False
    session.active_target = "auto"
    monkeypatch.setattr(session.laya, "is_git_action", lambda p: (False, 0.0))
    monkeypatch.setattr(session.laya, "is_build_action", lambda p: (False, 0.0))
    monkeypatch.setattr(session.laya, "is_heavy_reasoning", lambda p: (False, 0.0))
    monkeypatch.setattr(session.laya, "route_prompt", lambda p: ("9000", 0.9))
    _bind_agent(session, monkeypatch, ScriptedAgent([RuntimeError("boom")]))

    with pytest.raises(RuntimeError):
        asyncio.run(session.chat("hello"))
    assert "auto" not in session.stats
    assert session.stats["9000"]["errors"] == 1


def test_unknown_force_target_raises_value_error_and_is_not_a_phantom_row(session):
    session.enable_hooks = False
    with pytest.raises(ValueError, match="Unknown target"):
        asyncio.run(session.chat("hi", force_target="nope"))
    assert "nope" not in session.stats
    assert session.stats["unrouted"]["errors"] == 1


def test_chat_turn_is_defined_once_and_returns_result(session):
    assert session_mod.MultiGpuHybridSession.__dict__["chat_turn"] is not None
    src = open(session_mod.__file__, encoding="utf-8").read()
    assert src.count("async def chat_turn(") == 1

    async def fake(prompt, force_target=None):
        return "reply"
    session.chat = fake
    assert asyncio.run(session.chat_turn("x", target="9000")) == "reply"


def test_dropping_agent_from_sync_code_still_closes_it(session):
    agent = ScriptedAgent([])
    session._active_local_agent = agent
    session.set_dangerously_skip_permissions(not session.dangerously_skip_permissions)
    assert session._active_local_agent is None and not agent.closed
    asyncio.run(session.close())
    assert agent.closed


def test_reset_local_agent_closes_active_agent(session):
    agent = ScriptedAgent([])
    session._active_local_agent = agent
    asyncio.run(session.reset_local_agent())
    assert agent.closed and session._active_local_agent is None


def test_autocompact_threshold_counts_turns_not_entries(session, monkeypatch):
    _online(session)
    session.enable_hooks = False
    session.autocompact_threshold = 3
    calls = []
    monkeypatch.setattr(session, "compact_history", lambda keep_recent=None, silent=False: calls.append(1) or 0)
    monkeypatch.setattr(session, "trigger_background_synthesis", lambda: None)
    agent = ScriptedAgent(["one", "two", "three"])
    _bind_agent(session, monkeypatch, agent)

    for i in range(3):
        asyncio.run(session.chat(f"q{i}", force_target="9000"))
        # entries reach 2, 4, 6; the threshold of 3 turns must only fire on the third turn
        assert len(calls) == (1 if i == 2 else 0)


def test_compact_history_wrapper_converts_turns_to_entries(session, monkeypatch):
    seen = {}

    def fake(sess, keep_recent=None, silent=False):
        seen["keep"] = keep_recent
        return 0

    monkeypatch.setattr(session_mod.ContextCurator, "compact_history", staticmethod(fake))
    session.compact_history(keep_recent=3)
    assert seen["keep"] == 6
    session.max_recent_turns = 4
    session.compact_history()
    assert seen["keep"] == 8


def test_system_context_window_keeps_max_recent_turns_in_turns(session, monkeypatch):
    monkeypatch.setattr(session, "summarize_history_sync", lambda older: "SUMMARY")
    session.max_recent_turns = 1
    session.history = [
        {"role": "user", "target": "9000", "content": "old-q"},
        {"role": "assistant", "target": "9000", "content": "old-a"},
        {"role": "user", "target": "9000", "content": "new-q"},
        {"role": "assistant", "target": "9000", "content": "new-a"},
    ]
    ctx = session.build_system_context("9000")
    assert "SUMMARY" in ctx and "new-q" in ctx and "new-a" in ctx
    assert "old-q" not in ctx


def test_close_cancels_pending_synthesis_task(session):
    async def go():
        session._synthesis_task = asyncio.create_task(asyncio.sleep(60))
        task = session._synthesis_task
        await session.close()
        return task

    task = asyncio.run(go())
    assert task.cancelled()


def test_cancelled_cloud_turn_survives_already_dead_process(session, monkeypatch):
    class DeadProc:
        returncode = 0

        class _Stream:
            async def readline(self):
                await asyncio.sleep(60)

        stdout = _Stream()
        stderr = _Stream()

        def kill(self):
            raise ProcessLookupError

        async def wait(self):
            return 0

    async def fake_exec(*a, **k):
        return DeadProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def go():
        task = asyncio.create_task(session.chat_cloud_oauth("p", with_project_context=False))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())


def test_checkpoint_snapshot_runs_off_the_event_loop(session, git_repo, monkeypatch):
    import threading
    main_thread = threading.get_ident()
    seen = []
    real = session_mod.Checkpoints.snapshot

    def spy(root):
        seen.append(threading.get_ident())
        return real(root)

    monkeypatch.setattr(session_mod.Checkpoints, "snapshot", staticmethod(spy))
    session.enable_hooks = False

    async def impl(prompt, force_target=None):
        (git_repo / "a.txt").write_text("changed\n")
        return "ok"
    session._chat_impl = impl
    asyncio.run(session.chat("x"))
    assert len(seen) == 2 and all(t != main_thread for t in seen)


def test_print_status_plain_mode_does_not_crash(session, capsys, monkeypatch):
    monkeypatch.setattr(session, "refresh_endpoints", lambda: session.endpoints)
    monkeypatch.setattr(session_mod, "RICH_AVAILABLE", False)
    session.enable_kontexta = False
    session.json_output = False
    session.print_status()
    out = capsys.readouterr().out
    assert "Codebase AST Map" in out and "Conversation Turns" in out


def test_notice_goes_to_stderr_in_json_mode(session, capsys):
    session.json_output = True
    session._notice("hello")
    captured = capsys.readouterr()
    assert captured.out == "" and "hello" in captured.err
    session.json_output = False
    session._notice("again")
    assert "again" in capsys.readouterr().out


def test_prefs_reject_bool_for_int_keys_and_int_for_bool_keys(tmp_path):
    path = str(tmp_path / "c.json")
    with open(path, "w") as f:
        json.dump({"autocompact_threshold": True, "max_tool_steps_per_turn": 3, "verbose": 1, "show_metrics": False}, f)
    assert load_prefs(path) == {"max_tool_steps_per_turn": 3, "show_metrics": False}


@pytest.mark.parametrize("err", [
    RuntimeError("unexpected EOF: unexpected EOF"),
    ConnectionResetError("Connection reset by peer"),
    BrokenPipeError("Broken pipe"),
    RuntimeError("ClientPayloadError: Response payload is not completed"),
    RuntimeError("RemoteProtocolError: peer closed connection"),
    RuntimeError("unexpected EOF"),
])
def test_unexpected_eof_and_connection_drops_trigger_escalation(session, monkeypatch, err):
    _online(session)
    session.enable_hooks = False
    agent = ScriptedAgent([err])
    _bind_agent(session, monkeypatch, agent)
    monkeypatch.setattr(session.laya, "choose_escalation_target", lambda *a, **k: ("cloud", 0.95))

    async def cloud(prompt, with_project_context=True):
        return "escalated cloud reply"

    monkeypatch.setattr(session, "chat_cloud_oauth", cloud)

    out = asyncio.run(session.chat("build test harness", force_target="9000"))
    assert out == "escalated cloud reply"
    assert session.history[-1] == {"role": "assistant", "target": "cloud", "content": "escalated cloud reply"}
    assert session.stats["cloud"]["turns"] == 1


def test_turn_autosaves_latest_session(session, monkeypatch, tmp_path):
    _online(session)
    session.enable_hooks = False
    saved = []
    monkeypatch.setattr(session, "save_session", lambda name=None: saved.append(name) or "/tmp/session.json")
    _bind_agent(session, monkeypatch, ScriptedAgent(["test reply"]))

    asyncio.run(session.chat("hello"))
    assert "latest_session" in saved


def test_turn_error_resets_local_agent(session, monkeypatch):
    _online(session)
    session.enable_hooks = False
    agent = ScriptedAgent([RuntimeError("Fatal endpoint drop")])
    _bind_agent(session, monkeypatch, agent)
    # No escalation targets so it raises
    monkeypatch.setattr(session.laya, "choose_escalation_target", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no candidates")))

    with pytest.raises(RuntimeError):
        asyncio.run(session.chat("boom", force_target="9000"))
    assert session._active_local_agent is None


def test_wedged_non_idle_agent_is_reset_before_reuse(session, monkeypatch):
    _online(session)
    session.enable_hooks = False

    class FakeConn:
        is_idle = False

    class FakeConv:
        _connection = FakeConn()

    class WedgedAgent:
        _conversation = FakeConv()
        closed = False
        async def __aexit__(self, *args):
            self.closed = True

    wedged = WedgedAgent()
    session._active_local_agent = wedged
    session._current_agent_target = "9000"
    session._active_local_model = session.endpoints["9000"]["model"]

    # When get_or_switch_local_agent runs, it should detect is_idle is False and reset
    reset_called = []
    real_reset = session.reset_local_agent

    async def spy_reset():
        reset_called.append(True)
        await real_reset()

    monkeypatch.setattr(session, "reset_local_agent", spy_reset)

    class FreshAgent:
        _conversation = None
        async def __aenter__(self):
            return self

    monkeypatch.setattr(session_mod, "Agent", lambda cfg: FreshAgent())
    fresh = asyncio.run(session.get_or_switch_local_agent("9000"))
    assert reset_called == [True]
    assert fresh is not wedged


