import pytest

from myagy.terminal import cli


def test_slash_command_returns_immediately(monkeypatch):
    monkeypatch.setattr(cli, "_read_line", lambda p: "/stats")
    assert cli.read_input_prompt("> ") == "/stats"


def test_pasted_multiline_buffer_is_not_reprompted(monkeypatch):
    monkeypatch.setattr(cli, "_read_line", lambda p: "line one\n\nline three")
    monkeypatch.setattr("builtins.input", lambda p="": pytest.fail("continuation prompt must not run"))
    assert cli.read_input_prompt("> ") == "line one\n\nline three"


def test_continuation_lines_until_blank(monkeypatch):
    monkeypatch.setattr(cli, "_read_line", lambda p: "first")
    lines = iter(["second", ""])
    monkeypatch.setattr("builtins.input", lambda p="": next(lines))
    monkeypatch.setattr(cli.select, "select", lambda *a, **k: ([], [], []))
    assert cli.read_input_prompt("> ") == "first\nsecond"


def test_single_line_mode(monkeypatch):
    monkeypatch.setattr(cli, "_read_line", lambda p: "  hello  ")
    assert cli.read_input_prompt("> ", multiline=False) == "hello"


def test_eof_returns_empty(monkeypatch):
    def eof(p):
        raise EOFError
    monkeypatch.setattr(cli, "_read_line", eof)
    assert cli.read_input_prompt("> ") == ""


def test_completer_suggests_commands_and_targets(monkeypatch):
    pytest.importorskip("prompt_toolkit")
    monkeypatch.setattr(cli, "_pt_session", None)
    monkeypatch.setattr(cli, "HISTORY_PATH", "/tmp/myagy-test-history")
    cli.set_completion_targets(["gemma4"])
    pt = cli._get_pt_session()
    assert pt is not None
    from prompt_toolkit.document import Document

    def complete(text):
        return [c.text for c in pt.completer.get_completions(Document(text), None)]

    assert "/diff" in complete("/di") and "/undo" in complete("/un")
    assert "/exit" in complete("/ex") and "/quit" in complete("/qu")
    assert "@gemma4" in complete("@ge") and "@cloud" in complete("@cl")
    assert complete("hello") == [] and complete("/model ge") == []


def test_no_prompt_toolkit_falls_back_to_input(monkeypatch):
    monkeypatch.setattr(cli, "_pt_session", False)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda p="": "plain")
    assert cli._read_line("> ") == "plain"


def _pt_with_pipe(monkeypatch, tmp_path):
    pytest.importorskip("prompt_toolkit")
    monkeypatch.setattr(cli, "_pt_session", None)
    monkeypatch.setattr(cli, "HISTORY_PATH", str(tmp_path / "hist"))
    return cli._get_pt_session()


def _feed_and_prompt(pt, chunks, prompt="> "):
    import threading
    import time
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    with create_pipe_input() as inp:
        def feed():
            time.sleep(0.4)
            for chunk in chunks:
                inp.send_text(chunk)
                time.sleep(0.5)
        threading.Thread(target=feed, daemon=True).start()
        session = PromptSession(input=inp, output=DummyOutput(), completer=pt.completer, history=pt.history)
        return session.prompt(prompt)


def test_tab_completes_slash_command(monkeypatch, tmp_path):
    pt = _pt_with_pipe(monkeypatch, tmp_path)
    from prompt_toolkit.formatted_text import ANSI
    assert _feed_and_prompt(pt, ["/hel", "\t", "\r"], ANSI("\x1b[1;38;5;39m[x]\x1b[0m You \x1b[39m>\x1b[0m ")) == "/help"


def test_up_arrow_recalls_history(monkeypatch, tmp_path):
    pt = _pt_with_pipe(monkeypatch, tmp_path)
    assert _feed_and_prompt(pt, ["remember me", "\r"]) == "remember me"
    assert _feed_and_prompt(pt, ["\x1b[A", "\r"]) == "remember me"
    assert "remember me" in (tmp_path / "hist").read_text()


def test_slash_exit_command_returns_immediately(monkeypatch):
    monkeypatch.setattr(cli, "_read_line", lambda p: "/exit")
    assert cli.read_input_prompt("> ") == "/exit"


@pytest.mark.asyncio
async def test_interactive_loop_exits_on_slash_exit(monkeypatch):
    class DummySession:
        json_output = True
        multiline_input = True
        active_target = "auto"

        def get_catalog(self):
            return []

        def format_context_pill(self, target):
            return "0 tokens"

        def get_context_metrics(self, target):
            return {"tokens": 0}

    session = DummySession()
    monkeypatch.setattr(cli, "read_input_prompt", lambda prompt, multiline: "/exit")
    await cli.interactive_loop(session)
