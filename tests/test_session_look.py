import asyncio
from types import SimpleNamespace

import pytest

from google.antigravity import types as sdk_types

from myagy.core import session as session_mod
from myagy.core.session import _pt_style
from myagy.terminal import say
from myagy.terminal.ui import UI


@pytest.fixture(autouse=True)
def restore_theme():
    yield
    UI.set_theme("dark")


@pytest.fixture
def loud(session):
    session.json_output = False
    session.enable_hooks = False
    session.continuous_curation = False
    return session


def _route_with(session, routed="9000", conf=0.91):
    session.laya.is_git_action = lambda p: (False, 0.0)
    session.laya.is_build_action = lambda p: (False, 0.0)
    session.laya.route_prompt = lambda p: (routed, conf)


def test_routing_is_one_quiet_line_then_detail_when_verbose(loud, capsys):
    _route_with(loud)
    assert asyncio.run(loud._route_auto("explain this")) == "9000"
    quiet = capsys.readouterr().out
    assert "routed to ROCm:9000 (laya 91%)" in quiet
    assert quiet.strip().count("\n") == 0
    assert "Laya System 1" not in quiet

    loud.verbose = True
    asyncio.run(loud._route_auto("explain this"))
    verbose = capsys.readouterr().out
    assert "routed to ROCm:9000" in verbose and "Laya System 1 (~33ms)" in verbose


def test_routing_chatter_goes_to_stderr_in_json_mode(session, capsys):
    _route_with(session)
    asyncio.run(session._route_auto("x"))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "routed to ROCm:9000" in captured.err


def test_verbose_and_json_flags_configure_say(session):
    session.verbose = True
    session.json_output = True
    assert say.is_verbose() and say.is_json()
    session.json_output = False
    session.verbose = False
    assert not say.is_verbose() and not say.is_json()


def test_pt_style_converts_256_colors():
    assert _pt_style("\033[38;5;39m") == "fg:#00afff"
    assert _pt_style("\033[1;38;5;214m") == "fg:#ffaf00 bold"
    assert _pt_style("") == ""
    assert _pt_style("\033[2m") == ""


def test_toolbar_fragments_content(session, monkeypatch):
    session.endpoints["9000"].update(status="online", name_friendly="qwen27", filename="qwen27.gguf")
    session.active_target = "9000"
    monkeypatch.setattr(session, "get_context_metrics", lambda t=None: {"tokens": 12400, "max_ctx": 131072, "percentage": 9.5})
    monkeypatch.setattr(session_mod.SubagentManager, "get_queue_status", classmethod(lambda cls: {"running": 1, "queued": 1}))
    session.use_laya_adaptive_permissions = True
    session.kontexta_sync = True

    frags = session.toolbar_fragments()
    assert all(isinstance(f, tuple) and len(f) == 2 and all(isinstance(x, str) for x in f) for f in frags)
    text = "".join(t for _, t in frags)
    for expected in ("ROCm:9000", "qwen27", "12.4k/131k ctx", "laya", "2 subagents", "memory on"):
        assert expected in text


def test_toolbar_hides_optional_segments_and_uses_theme_colors(session, monkeypatch):
    session.active_target = "cloud"
    monkeypatch.setattr(session, "get_context_metrics", lambda t=None: {"tokens": 900, "max_ctx": 1048576, "percentage": 0.1})
    monkeypatch.setattr(session_mod.SubagentManager, "get_queue_status", classmethod(lambda cls: {"running": 0, "queued": 0}))
    session.kontexta_sync = False
    session.use_laya_adaptive_permissions = False
    session._dangerously_skip_permissions = True
    monkeypatch.setattr(UI, "CLOUD", "\033[38;5;75m")

    frags = session.toolbar_fragments()
    text = "".join(t for _, t in frags)
    assert "subagent" not in text and "memory" not in text
    assert "CLOUD" in text and "auto-approve" in text and "900/1M ctx" in text
    assert any("fg:#5fafff" in style for style, _ in frags)


def test_toolbar_context_is_cached(session, monkeypatch):
    calls = []
    monkeypatch.setattr(session, "get_context_metrics", lambda t=None: calls.append(1) or {"tokens": 1, "max_ctx": 2, "percentage": 50.0})
    session.toolbar_fragments()
    session.toolbar_fragments()
    assert len(calls) == 1


def test_print_header_content_and_size(loud, capsys, monkeypatch):
    monkeypatch.setattr(session_mod.kontexta_sync, "enabled", lambda *a, **k: False)
    loud.enable_kontexta = False
    loud.endpoints["9000"].update(status="online", name_friendly="qwen27")
    loud.endpoints["9001"].update(status="offline")
    loud.print_header()
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert 4 <= len(lines) <= 6
    for expected in ("ROCm:9000", "qwen27", "CUDA:9001", "offline", "Cloud", "target", "permissions", "repo", "memory"):
        assert expected in out


def test_print_header_silent_in_json_mode(session, capsys):
    session.print_header()
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


class _FakeAgent:
    def __init__(self, parts):
        self.parts = parts

    async def chat(self, prompt):
        async def gen():
            for p in self.parts:
                yield p
        return gen()


def _run_turn(session, parts):
    session.endpoints["9000"].update(status="online", model="m.gguf", filename="m.gguf", name_friendly="m")
    session.active_target = "9000"

    async def fake_agent(target):
        return _FakeAgent(parts)

    session.get_or_switch_local_agent = fake_agent
    return asyncio.run(session._chat_impl("hello"))


def test_last_answer_set_after_turn(loud):
    assert loud.last_answer == ""
    assert _run_turn(loud, ["Hello ", "world"]) == "Hello world"
    assert loud.last_answer == "Hello world"


def test_last_answer_ignores_empty_responses(loud):
    loud.last_answer = "previous"
    _run_turn(loud, [])
    assert loud.last_answer == "previous"


def test_turn_prints_header_and_no_old_banner(loud, capsys):
    _run_turn(loud, ["ok"])
    out = capsys.readouterr().out
    assert "ROCm:9000" in out and "m.gguf" in out
    assert "┌──" not in out


def test_set_theme_persists_and_validates(session, monkeypatch):
    saved = []
    monkeypatch.setattr(session_mod, "save_prefs", lambda values, path=None: saved.append(dict(values)) or True)
    assert session.set_theme("light") is True
    assert session.theme == "light" and UI.theme_name == "light"
    assert saved and saved[-1]["theme"] == "light"
    before = len(saved)
    assert session.set_theme("neon") is False
    assert len(saved) == before and session.theme == "light"


def test_theme_pref_is_applied_and_reset(session):
    from myagy.core.prefs import PREF_DEFAULTS
    assert PREF_DEFAULTS["theme"] == "dark"
    session.apply_preferences({"theme": "mono"})
    assert UI.theme_name == "mono" and session.preferences()["theme"] == "mono"
    session.apply_preferences(PREF_DEFAULTS)
    assert UI.theme_name == "dark"


def _handler_session(session, laya_risk=None):
    session.json_output = False
    session.use_laya_adaptive_permissions = laya_risk is not None
    session._dangerously_skip_permissions = False
    if laya_risk is not None:
        session.laya.evaluate_action_safety = lambda summary: (False, laya_risk)
    return session


def test_permission_card_shown_and_denied(session, monkeypatch, capsys):
    _handler_session(session)
    monkeypatch.setattr(session_mod, "prompt_input", lambda prompt: "n")
    tool = SimpleNamespace(name="write_file")
    assert session.permission_prompt_handler(tool, {"TargetFile": "x.py", "Instruction": "rewrite"}) is False
    out = capsys.readouterr().out
    for expected in ("Permission required", "write_file", "TargetFile", "x.py", "[y]es / [n]o / [a]lways", "denied"):
        assert expected in out


def test_permission_card_shows_laya_risk(session, monkeypatch, capsys):
    _handler_session(session, laya_risk=0.42)
    monkeypatch.setattr(session_mod, "prompt_input", lambda prompt: "y")
    tool = SimpleNamespace(name="run_command")
    assert session.permission_prompt_handler(tool, {"CommandLine": "make deploy"}) is True
    out = capsys.readouterr().out
    assert "0.42" in out and "Laya flagged elevated risk" in out and "Command" in out and "make deploy" in out


def test_always_remembers_only_the_exact_call(session, monkeypatch, capsys):
    _handler_session(session)
    answers = iter(["a"])
    monkeypatch.setattr(session_mod, "prompt_input", lambda prompt: next(answers))
    tool = SimpleNamespace(name="run_command")
    assert session.permission_prompt_handler(tool, {"CommandLine": "make test"}) is True

    def boom(prompt):
        raise AssertionError("must not prompt again for the same call")

    monkeypatch.setattr(session_mod, "prompt_input", boom)
    assert session.permission_prompt_handler(tool, {"CommandLine": "make test"}) is True

    monkeypatch.setattr(session_mod, "prompt_input", lambda prompt: "n")
    assert session.permission_prompt_handler(tool, {"CommandLine": "make deploy"}) is False


def test_laya_safe_action_is_a_single_quiet_line(session, monkeypatch, capsys):
    session.json_output = False
    session.use_laya_adaptive_permissions = True
    session.laya.evaluate_action_safety = lambda summary: (True, 0.97)
    assert session.permission_prompt_handler(SimpleNamespace(name="grep"), {"Query": "x"}) is True
    out = capsys.readouterr().out
    assert out.strip().count("\n") == 0 and "auto-approved grep" in out


def test_sudo_card_and_denial(session, monkeypatch, capsys):
    session.json_output = False
    monkeypatch.setattr(session_mod.SudoManager, "is_authenticated", classmethod(lambda cls: False))
    monkeypatch.setattr(session_mod, "prompt_input", lambda prompt: "n")
    tool = SimpleNamespace(name="run_command")
    assert session.permission_prompt_handler(tool, {"CommandLine": "sudo systemctl restart x"}) is False
    out = capsys.readouterr().out
    assert "Sudo elevation required" in out and "sudo systemctl restart x" in out and "denied" in out


def _hooks(session):
    session.enable_hooks = False
    return session.build_tool_hooks()


def test_tool_call_prints_one_completion_row(loud, capsys):
    pre, post = _hooks(loud)
    assert pre(sdk_types.ToolCall(name="grep", args={"Query": "foo"})).allow
    assert capsys.readouterr().out == ""
    post(sdk_types.ToolResult(result="a\nb\nc"))
    out = capsys.readouterr().out
    assert out.count("\n") == 1
    assert "✓" in out and "grep" in out and "foo" in out and "3 lines" in out


def test_failed_tool_row_shows_error(loud, capsys):
    pre, post = _hooks(loud)
    pre(sdk_types.ToolCall(name="run_command", args={"CommandLine": "false"}))
    post(sdk_types.ToolResult(error="boom: exit 1"))
    out = capsys.readouterr().out
    assert "✗" in out and "boom" in out and "run_command" in out


def test_verbose_tool_output_is_truncated_with_marker(loud, capsys):
    loud.verbose = True
    pre, post = _hooks(loud)
    pre(sdk_types.ToolCall(name="read", args={"AbsolutePath": "/x"}))
    capsys.readouterr()
    post(sdk_types.ToolResult(result="\n".join(f"line {i}" for i in range(25))))
    out = capsys.readouterr().out
    assert "line 9" in out and "line 10" not in out and "+15 lines" in out


def test_tool_rows_are_silent_in_json_mode(session, capsys):
    pre, post = _hooks(session)
    pre(sdk_types.ToolCall(name="grep", args={"Query": "foo"}))
    post(sdk_types.ToolResult(result="x"))
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_pending_tools_cleared_each_turn_and_loop_guard_warns(loud, capsys):
    pre, _ = _hooks(loud)
    call = sdk_types.ToolCall(name="grep", args={"Query": "same"})
    assert pre(call).allow and pre(call).allow and pre(call).allow is not None
    out = capsys.readouterr().out
    assert "Loop guard" in out
    loud._tool_pending.append({"id": None, "name": "stale", "summary": "", "t0": None, "step": 1})
    _run_turn(loud, ["ok"])
    assert loud._tool_pending == []


def test_print_help_removed(session):
    assert not hasattr(session, "print_help")


def test_status_and_stats_render_through_say(loud, capsys, monkeypatch):
    monkeypatch.setattr(loud, "refresh_endpoints", lambda: loud.endpoints)
    loud.enable_kontexta = False
    loud.print_status()
    out = capsys.readouterr().out
    assert "Status" in out and "Kontexta Journal Sync" in out and "Theme" in out

    loud.print_stats()
    assert "no turns recorded" in capsys.readouterr().out
    loud.stats["9000"] = {"turns": 2, "tokens": 50, "seconds": 3.0, "tps_sum": 20.0, "tps_n": 1}
    loud.print_stats()
    out = capsys.readouterr().out
    assert "Session usage" in out and "ROCm:9000" in out
