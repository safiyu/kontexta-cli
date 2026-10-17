import pytest

from myagy.terminal import say
from myagy.terminal.ui import UI


@pytest.fixture(autouse=True)
def reset_say():
    say.configure(False, False)
    yield
    say.configure(False, False)


def test_json_mode_routes_everything_to_stderr_without_ansi(capsys):
    say.configure(json_output=True)
    say.info("a")
    say.ok("b")
    say.warn("c")
    say.err("d")
    say.dim("e")
    say.plain("f")
    say.step("g")
    cap = capsys.readouterr()
    assert cap.out == ""
    for word in "abcdefg":
        assert word in cap.err
    assert "\033[" not in cap.err
    assert say.is_json() and not say.is_verbose()


def test_normal_mode_prints_to_stdout(capsys):
    say.plain("hello")
    cap = capsys.readouterr()
    assert "hello" in cap.out and cap.err == ""


def test_step_has_middle_dot_and_detail_needs_verbose(capsys):
    say.step("routed to ROCm")
    say.detail("hidden detail")
    out = capsys.readouterr().out
    assert "·" in out and "routed to ROCm" in out and "hidden detail" not in out
    say.configure(verbose=True)
    say.detail("shown detail")
    assert "shown detail" in capsys.readouterr().out


def test_detail_in_json_mode_only_when_verbose(capsys):
    say.configure(json_output=True)
    say.detail("x1")
    assert capsys.readouterr().err == ""
    say.configure(json_output=True, verbose=True)
    say.detail("x2")
    assert "x2" in capsys.readouterr().err


def test_truncate_lines():
    text = "\n".join(str(i) for i in range(20))
    shown, hidden = say.truncate_lines(text, 5)
    assert shown.splitlines() == ["0", "1", "2", "3", "4"] and hidden == 15
    assert say.truncate_lines("a\nb", 12) == ("a\nb", 0)
    assert say.truncate_lines("a\nb", 0) == ("", 2)


def test_card_plain_fallback_is_ascii_box(capsys):
    say.card("Run command", [("Command", "rm -rf build"), ("Risk", "0.05")], "[y]es / [n]o / [a]lways", "danger")
    out = capsys.readouterr().out
    assert out.isascii()
    lines = out.strip().splitlines()
    assert lines[0].startswith("+-- Run command") and lines[-1].startswith("+---")
    assert any("Command  rm -rf build" in ln for ln in lines)
    assert any("[y]es / [n]o / [a]lways" in ln for ln in lines)
    assert len({len(ln) for ln in lines}) == 1


def test_card_unknown_style_and_multiline_value(capsys):
    say.card("t", [("k", "line1\nline2")], style="bogus")
    out = capsys.readouterr().out
    assert "line1" in out and "line2" in out


def test_card_rich_path_renders_when_color_on(monkeypatch, capsys):
    pytest.importorskip("rich")
    monkeypatch.setattr(UI, "_use_color", True)
    say.card("Heads up", [("Label", "Value")], "[y]es", "warn")
    out = capsys.readouterr().out
    assert "Heads up" in out and "Label" in out and "Value" in out and "[y]es" in out


def test_tool_row_format_and_truncation(monkeypatch, capsys):
    monkeypatch.setattr(say.shutil, "get_terminal_size", lambda fallback=None: type("S", (), {"columns": 40})())
    say.tool_row("grep", '"foo"', True, 0.3)
    assert capsys.readouterr().out.strip() == '✓ grep "foo" · 0.3s'
    say.tool_row("bash", "x" * 200, False, 12.0, "exit 1")
    line = capsys.readouterr().out.rstrip("\n")
    assert line.startswith("  ✗ bash") and "…" in line and line.endswith("· 12.0s · exit 1")
    assert len(line) <= 40


def test_turn_header_and_group_and_table(capsys):
    say.turn_header("you", "9000")
    say.group("Models")
    say.table("Endpoints", ["port", "state"], [["9000", "up"], ["9001", "down"]])
    out = capsys.readouterr().out
    assert "── you " in out and "Models" in out and "port" in out and "9001" in out and "down" in out
    assert "Endpoints" in out


def test_rule_line_and_fmt_elapsed():
    assert "receipt" in say.rule_line("receipt", "", 30)
    assert say.fmt_elapsed(5) == "5.0s" and say.fmt_elapsed(125) == "2m05s"


def test_say_markdown(capsys):
    say.markdown("**bold text**")
    out = capsys.readouterr().out
    assert "bold text" in out
