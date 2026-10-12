import asyncio

import pytest

from myagy.terminal import cli, say
from myagy.terminal.ui import UI


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    say.configure(json_output=False, verbose=False)
    monkeypatch.setattr(cli, "_toolbar_provider", None)
    original = UI.theme_name
    yield
    UI.set_theme(original)
    say.configure(json_output=False, verbose=False)


class FakeSession:
    json_output = True
    verbose = False
    multiline_input = True
    active_target = "9000"
    render_markdown = False
    last_answer = ""
    history = []

    def __init__(self):
        self.themes = []
        self.reset_calls = 0
        self.header_calls = 0

    def toolbar_fragments(self):
        return [("bold", "ROCm:9000"), ("", " | 1.2k ctx")]

    def print_header(self):
        self.header_calls += 1

    def set_theme(self, name):
        self.themes.append(name)
        return UI.set_theme(name)

    def get_catalog(self):
        return []

    def get_context_metrics(self, target):
        return {"tokens": 1234}

    def save_preferences(self):
        return True

    async def reset_local_agent(self):
        self.reset_calls += 1


def _run(name, session, arg=""):
    return asyncio.run(cli.COMMANDS[name](session, arg, cli._LoopCtx(None)))


def _out(capsys):
    captured = capsys.readouterr()
    return captured.out + captured.err


# ── registry and /help ────────────────────────────────────────────────

def test_every_command_has_a_known_group_and_help_text():
    for meta in cli.COMMAND_META:
        assert meta["group"] in cli.GROUP_ORDER, meta["names"]
        assert meta["help"], meta["names"]


def test_help_lists_every_registered_command_and_alias(capsys):
    _run("/help", FakeSession())
    out = _out(capsys)
    for name in cli.COMMANDS:
        assert name in out, name
    assert "/paste" in out
    for group_name in cli.GROUP_ORDER:
        assert group_name in out


def test_help_groups_are_built_from_registry():
    groups = cli.help_groups()
    listed = {alias.strip() for rows in groups.values() for names, _ in rows for alias in names.split(",")}
    assert set(cli.COMMANDS) <= listed
    assert "Other" not in groups


def test_help_can_filter_one_group(capsys):
    _run("/help", FakeSession(), "git")
    out = _out(capsys)
    assert "/diff" in out and "/undo" in out
    assert "/theme" not in out


def test_help_unknown_group_warns(capsys):
    _run("/help", FakeSession(), "nonsense")
    assert "Unknown group" in _out(capsys)


def test_new_commands_complete():
    for name in ("/theme", "/last", "/memory", "/help"):
        assert name in cli.SLASH_COMMANDS


# ── /theme ────────────────────────────────────────────────────────────

def test_theme_without_arg_lists_themes_and_current(capsys):
    _run("/theme", FakeSession())
    out = _out(capsys)
    for theme in UI.THEMES:
        assert theme in out
    assert "yes" in out


def test_theme_switches_through_session(capsys):
    s = FakeSession()
    _run("/theme", s, "light")
    assert s.themes == ["light"] and UI.theme_name == "light"
    assert "Theme set to light" in _out(capsys)


def test_theme_unknown_warns_and_keeps_current(capsys):
    s = FakeSession()
    before = UI.theme_name
    _run("/theme", s, "neon")
    assert UI.theme_name == before
    assert "Unknown theme 'neon'" in _out(capsys)


# ── /last ─────────────────────────────────────────────────────────────

def test_last_without_answer_warns(capsys):
    _run("/last", FakeSession())
    assert "No previous answer" in _out(capsys)


def test_last_reshows_full_answer(capsys):
    s = FakeSession()
    s.last_answer = "line one\nline two\nline three"
    _run("/last", s)
    out = _out(capsys)
    assert "line one" in out and "line three" in out


def test_last_renders_markdown_through_stream_renderer(monkeypatch):
    rich_console = pytest.importorskip("rich.console")
    import io
    from myagy.terminal import ui
    buf = io.StringIO()
    monkeypatch.setattr(ui, "console", rich_console.Console(file=buf, force_terminal=False, width=60))
    s = FakeSession()
    s.render_markdown = True
    s.last_answer = "# Title\n\nbody text\n"
    _run("/last", s)
    assert "Title" in buf.getvalue() and "body text" in buf.getvalue()
    assert "# Title" not in buf.getvalue()


# ── prompt and toolbar ────────────────────────────────────────────────

def test_prompt_is_plain_arrow_without_status(monkeypatch):
    s = FakeSession()
    monkeypatch.setattr(UI, "_use_color", False)
    assert cli._prompt_string(s) == "> "
    monkeypatch.setattr(UI, "_use_color", True)
    prompt = cli._prompt_string(s)
    assert "❯" in prompt and "You" not in prompt and "ctx" not in prompt


def test_plain_pill_shows_target_and_context():
    assert cli._plain_pill(FakeSession()) == "[ROCm:9000 | 1.2k ctx]"


def test_toolbar_fragments_come_from_provider_and_survive_errors():
    cli.set_toolbar(lambda: [("", "hello")])
    assert cli._toolbar_fragments() == [("", "hello")]

    def boom():
        raise RuntimeError("nope")

    cli.set_toolbar(boom)
    assert cli._toolbar_fragments() == []


def test_interactive_loop_registers_session_toolbar(monkeypatch):
    s = FakeSession()
    monkeypatch.setattr(cli, "read_input_prompt", lambda prompt, multiline: "/exit")
    asyncio.run(cli.interactive_loop(s))
    assert cli._toolbar_fragments() == s.toolbar_fragments()


def test_interactive_loop_calls_print_header_unless_json(monkeypatch):
    monkeypatch.setattr(cli, "read_input_prompt", lambda prompt, multiline: "/exit")
    quiet = FakeSession()
    asyncio.run(cli.interactive_loop(quiet))
    assert quiet.header_calls == 0
    loud = FakeSession()
    loud.json_output = False
    asyncio.run(cli.interactive_loop(loud))
    assert loud.header_calls == 1


def test_pt_prompt_receives_bottom_toolbar_with_refresh(monkeypatch):
    pytest.importorskip("prompt_toolkit")
    seen = {}

    class FakePT:
        def prompt(self, message, **kwargs):
            seen.update(kwargs)
            return "typed"

    monkeypatch.setattr(cli, "_get_pt_session", lambda: FakePT())
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True, raising=False)
    cli.set_toolbar(lambda: [("bold", "ROCm"), ("", " | 3 ctx")])
    assert cli._read_line("> ") == "typed"
    assert seen["refresh_interval"] == 1.0
    assert "".join(text for _, text in seen["bottom_toolbar"]()) == "ROCm | 3 ctx"


def test_no_toolbar_kwarg_when_no_provider(monkeypatch):
    pytest.importorskip("prompt_toolkit")
    seen = {}

    class FakePT:
        def prompt(self, message, **kwargs):
            seen["kwargs"] = kwargs
            return "typed"

    monkeypatch.setattr(cli, "_get_pt_session", lambda: FakePT())
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True, raising=False)
    assert cli._read_line("> ") == "typed"
    assert seen["kwargs"] == {}


def test_plain_pill_printed_before_prompt_without_toolbar(monkeypatch, capsys):
    s = FakeSession()
    s.json_output = False
    prompts = []

    def fake(prompt, multiline):
        prompts.append(prompt)
        return "/exit"

    monkeypatch.setattr(cli, "read_input_prompt", fake)
    asyncio.run(cli.interactive_loop(s))
    out = _out(capsys)
    assert "[ROCm:9000 | 1.2k ctx]" in out
    assert "You" not in prompts[0]


# ── destructive confirmations ─────────────────────────────────────────

def test_clear_asks_first_and_keeps_history_on_no(monkeypatch, capsys):
    s = FakeSession()
    s.history = [{"role": "user", "content": "x"}]
    s._cached_curated_memory = "keep"
    monkeypatch.setattr(cli, "prompt_input", lambda p: "n")
    _run("/clear", s)
    out = _out(capsys)
    assert "[y]es / [n]o" in out and "Clear cancelled" in out
    assert s.history and s.reset_calls == 0


def test_clear_confirmed_clears(monkeypatch, capsys):
    s = FakeSession()
    s.history = [{"role": "user", "content": "x"}]
    s._cached_curated_memory = "keep"
    monkeypatch.setattr(cli, "prompt_input", lambda p: "y")
    _run("/clear", s)
    assert s.history == [] and s._cached_curated_memory is None and s.reset_calls == 1
    assert "history cleared" in _out(capsys)


def test_clear_yes_flag_skips_prompt(monkeypatch):
    s = FakeSession()
    s.history = [{"role": "user", "content": "x"}]
    s._cached_curated_memory = None
    monkeypatch.setattr(cli, "prompt_input", lambda p: pytest.fail("must not prompt"))
    _run("/clear", s, "-y")
    assert s.history == []


def test_clear_with_empty_history_does_not_prompt(monkeypatch):
    s = FakeSession()
    s._cached_curated_memory = None
    monkeypatch.setattr(cli, "prompt_input", lambda p: pytest.fail("must not prompt"))
    _run("/clear", s)
    assert s.reset_calls == 1


def test_undo_shows_card_and_applies_on_yes(monkeypatch, capsys):
    s = FakeSession()
    applied = []
    s.undo_plan = lambda: {"checkpoint": {"prompt": "edit things"}, "files": {"a.py": "M", "b.py": "A"}}
    s.apply_undo = lambda plan: applied.append(plan) or ["a.py", "b.py"]
    monkeypatch.setattr(cli, "prompt_input", lambda p: "yes")
    _run("/undo", s)
    out = _out(capsys)
    assert "edit things" in out and "a.py" in out and "delete (created by turn)" in out and "[y]es / [n]o" in out
    assert len(applied) == 1 and "Reverted 2 file(s)" in out


def test_undo_declined_does_not_apply(monkeypatch, capsys):
    s = FakeSession()
    s.undo_plan = lambda: {"checkpoint": {"prompt": "p"}, "files": {"a.py": "M"}}
    s.apply_undo = lambda plan: pytest.fail("must not apply")
    monkeypatch.setattr(cli, "prompt_input", lambda p: "")
    _run("/undo", s)
    assert "Undo cancelled" in _out(capsys)


def test_prefs_reset_confirms_before_resetting(monkeypatch, capsys):
    s = FakeSession()
    resets = []
    s.apply_preferences = lambda prefs: resets.append(prefs)
    monkeypatch.setattr(cli, "reset_prefs", lambda: resets.append("file"))
    monkeypatch.setattr(cli, "prompt_input", lambda p: "n")
    _run("/prefs", s, "reset")
    assert resets == [] and "Reset cancelled" in _out(capsys)
    monkeypatch.setattr(cli, "prompt_input", lambda p: "y")
    _run("/prefs", s, "reset")
    assert "file" in resets and s.reset_calls == 1


# ── json and verbose keep say in sync ─────────────────────────────────

def test_json_toggle_routes_notices_to_stderr(capsys):
    s = FakeSession()
    s.json_output = False
    _run("/json", s)
    captured = capsys.readouterr()
    assert captured.out == "" and "JSON output mode: ENABLED" in captured.err


def test_verbose_toggle_updates_say_visibility(capsys):
    s = FakeSession()
    _run("/verbose", s, "on")
    assert say.is_verbose()
    say.detail("extra detail")
    assert "extra detail" in _out(capsys)
    _run("/verbose", s, "off")
    assert not say.is_verbose()
