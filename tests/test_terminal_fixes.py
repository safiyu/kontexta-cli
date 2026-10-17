import asyncio
import io
import threading

import pytest

rich_console = pytest.importorskip("rich.console")

from myagy.terminal import cli, ui, say
from myagy.terminal.esc_listener import EscListener


@pytest.fixture(autouse=True)
def reset_say():
    say.configure(json_output=False, verbose=False)
    yield
    say.configure(json_output=False, verbose=False)


def _capture(monkeypatch, width=60):
    buf = io.StringIO()
    monkeypatch.setattr(ui, "console", rich_console.Console(file=buf, force_terminal=False, width=width))
    return buf


def _stream(text, monkeypatch):
    buf = _capture(monkeypatch)
    r = ui.StreamRenderer(markdown=True)
    for ch in text:
        r.feed(ch)
    r.stop()
    return r, buf.getvalue()


def test_table_streamed_char_by_char_renders_as_table(monkeypatch):
    _, out = _stream("| a | b |\n| - | - |\n| 1 | 2 |\n\n", monkeypatch)
    assert "| a | b |" not in out
    assert "a" in out and "2" in out and ("┃" in out or "│" in out or "─" in out)


def test_list_item_continuation_stays_in_one_block(monkeypatch):
    _, out = _stream("- item\n  continued line\n- second\n\n", monkeypatch)
    assert "item continued line" in " ".join(out.split())
    assert out.count("•") == 2


def test_setext_heading_is_not_split_into_paragraph_and_rule(monkeypatch):
    _, out = _stream("Title\n-----\n\nbody\n\n", monkeypatch)
    assert "Title" in out and "-----" not in out and "body" in out


def test_inline_triple_backticks_do_not_open_code_block(monkeypatch):
    r, out = _stream("use ```x``` inline\nnext line\n\n", monkeypatch)
    assert "next line" in out and "inline" in out
    assert r._in_code_block is False


def test_stop_flushes_unclosed_code_block(monkeypatch):
    _, out = _stream("```python\nprint(1)\n", monkeypatch)
    assert "print(1)" in out


def test_stop_flushes_partial_last_line(monkeypatch):
    _, out = _stream("- a\n- b without newline", monkeypatch)
    assert "b without newline" in out


def test_all_registered_commands_and_aliases_complete():
    for name in ("/build", "/test", "/git-shift", "/skip-permissions", "/mode", "/paste", "/exit"):
        assert name in cli.SLASH_COMMANDS
    assert set(cli.COMMANDS) <= set(cli.SLASH_COMMANDS)


class FakeSession:
    json_output = True
    verbose = False
    multiline_input = True
    active_target = "auto"
    history = []
    endpoints = {"9002": {"status": "online", "filename": "g.gguf"}}

    def __init__(self):
        self.spawned = []
        self.cancelled = []
        self.chats = []
        self.refresh_thread = None

    def get_catalog(self):
        return []

    def format_context_pill(self, target):
        return "0"

    def get_context_metrics(self, target):
        return {"tokens": 0}

    def save_preferences(self):
        return True

    def spawn_subagent(self, text):
        self.spawned.append(text)

    def cancel_subagent(self, n):
        self.cancelled.append(n)
        return True

    def print_subagent_detail(self, n):
        pass

    def refresh_endpoints(self):
        self.refresh_thread = threading.current_thread()

    async def chat(self, prompt, force_target=None):
        self.chats.append(prompt)


@pytest.mark.parametrize("arg", ["view foo", "view", "cancel abc", "kill", "stop x"])
def test_spawn_view_cancel_with_bad_arg_prints_usage(arg, capsys):
    s = FakeSession()
    asyncio.run(cli.COMMANDS["/spawn"](s, arg, cli._LoopCtx(None)))
    assert s.spawned == [] and s.cancelled == []
    assert "Usage" in capsys.readouterr().out


def test_spawn_free_text_still_spawns():
    s = FakeSession()
    asyncio.run(cli.COMMANDS["/spawn"](s, "inspect the logs", cli._LoopCtx(None)))
    assert s.spawned == ["inspect the logs"]


def test_compactor_status_refreshes_off_the_event_loop():
    s = FakeSession()
    asyncio.run(cli.COMMANDS["/compactor"](s, "status", cli._LoopCtx(None)))
    assert s.refresh_thread is not None and s.refresh_thread is not threading.main_thread()


def _run_loop(session, monkeypatch, inputs):
    it = iter(inputs)

    def fake(prompt, multiline):
        v = next(it)
        if isinstance(v, BaseException):
            raise v
        return v

    monkeypatch.setattr(cli, "read_input_prompt", fake)
    asyncio.run(cli.interactive_loop(session))


def test_unknown_at_model_warns_and_is_not_sent(monkeypatch, capsys):
    s = FakeSession()
    _run_loop(s, monkeypatch, ["@nosuchmodel do the thing", "/exit"])
    assert s.chats == []
    captured = capsys.readouterr()
    assert "Unknown model target '@nosuchmodel'" in captured.out + captured.err


def test_ctrl_c_at_prompt_first_warns_second_exits(monkeypatch, capsys):
    s = FakeSession()
    calls = []

    def fake(prompt, multiline):
        calls.append(1)
        raise cli.PromptInterrupted()

    monkeypatch.setattr(cli, "read_input_prompt", fake)
    asyncio.run(asyncio.wait_for(cli.interactive_loop(s), 5))
    assert len(calls) == 2
    captured = capsys.readouterr()
    assert "again to exit" in captured.out + captured.err


def test_ctrl_c_counter_resets_after_real_input(monkeypatch):
    s = FakeSession()
    _run_loop(s, monkeypatch, [cli.PromptInterrupted(), "/status-not-a-command", cli.PromptInterrupted(), "/exit"])


def test_first_line_keyboard_interrupt_raises_prompt_interrupted(monkeypatch):
    def boom(p):
        raise KeyboardInterrupt
    monkeypatch.setattr(cli, "_read_line", boom)
    with pytest.raises(cli.PromptInterrupted):
        cli.read_input_prompt("> ")


def test_reload_flags_come_from_preferences():
    flags = cli._reload_flags(
        {"cloud_model": "m", "active_target": "9002", "dangerously_skip_permissions": False,
         "verbose": True, "enable_mcp": False, "enable_kontexta": False, "multiline_input": False},
        json_output=True,
    )
    assert flags == ["--cloud-model", "m", "--target", "9002", "--safe", "--verbose", "--json",
                     "--no-mcp", "--no-kontexta", "--single-line"]
    assert cli._reload_flags({"active_target": "some-model-id"}) == []


def test_esc_listener_exit_joins_thread_before_restoring_terminal():
    order = []

    class FakeThread:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            order.append("join")

    lst = EscListener()
    lst._thread = FakeThread()
    lst._cleanup_terminal = lambda: order.append("cleanup")
    lst.__exit__(None, None, None)
    assert order == ["join", "cleanup"]


def test_resume_command_loads_latest_session(monkeypatch):
    class Sess:
        def __init__(self):
            self.loaded = None
            self.history = ["h1", "h2"]
        def list_sessions(self):
            return [{"file": "latest_session.json", "turns": 1, "saved_at": "2026-10-16", "size_kb": 1.0}]
        def load_session(self, name):
            self.loaded = name
            return True

    s = Sess()
    cmd = cli.COMMANDS["/resume"]
    asyncio.run(cmd(s, "", None))
    assert s.loaded == "latest_session.json"

