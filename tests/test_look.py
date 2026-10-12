import io

import pytest

rich_console = pytest.importorskip("rich.console")

from myagy.terminal import receipt, ui


def _render(monkeypatch, text, width=60):
    buf = io.StringIO()
    monkeypatch.setattr(ui, "console", rich_console.Console(file=buf, force_terminal=False, width=width))
    r = ui.StreamRenderer(markdown=True)
    r.feed(text)
    r.stop()
    return buf.getvalue()


def test_code_block_in_bordered_panel_with_language(monkeypatch):
    out = _render(monkeypatch, "```python\nx = 1\n```\n")
    assert "python" in out and "x = 1" in out
    assert "╭" in out and "╰" in out
    assert "```" not in out


def test_code_block_without_language_has_no_title(monkeypatch):
    out = _render(monkeypatch, "```\nplain\n```\n")
    assert "plain" in out and "text" not in out


def test_h1_h2_get_rules_and_h3_does_not(monkeypatch):
    out = _render(monkeypatch, "# One\n\n## Two\n\n### Three\n")
    assert "━" in out and "─" in out
    assert out.count("Three") == 1
    assert "#" not in out


def test_table_renders_as_table_not_raw_pipes(monkeypatch):
    out = _render(monkeypatch, "| a | b |\n| - | - |\n| 1 | 2 |\n")
    assert "|" not in out and "a" in out and "2" in out


def test_blockquote_has_left_bar(monkeypatch):
    out = _render(monkeypatch, "> quoted text\n")
    assert "▌" in out and "quoted text" in out


def test_long_paragraph_wraps_to_width(monkeypatch):
    out = _render(monkeypatch, "word " * 40 + "\n", width=40)
    assert all(len(line) <= 40 for line in out.splitlines())


def test_no_ansi_escapes_without_color(monkeypatch):
    out = _render(monkeypatch, "# T\n\n`code` and **bold**\n")
    assert "\x1b" not in out


def test_numstat_parsing_and_binary():
    files = receipt.files_from_numstat("3\t1\ta.py\n-\t-\timg.png\n\nbad line\n0\t2\tdir/b.py\n")
    assert files == [
        {"path": "a.py", "added": 3, "removed": 1},
        {"path": "img.png", "added": 0, "removed": 0},
        {"path": "dir/b.py", "added": 0, "removed": 2},
    ]


def test_receipt_none_when_nothing_happened():
    assert receipt.build_receipt([], [], None, 1.0, 10.0, "9000") is None


def test_receipt_contents():
    files = [{"path": "a.py", "added": 3, "removed": 1}, {"path": "b.py", "added": 2, "removed": 0}]
    text = receipt.build_receipt(files, ["pytest -q"], "pass", 75.0, 12.34, "ROCm:9000")
    assert "changed 2 files" in text and "+5" in text and "-1" in text
    assert "a.py" in text and "b.py" in text
    assert "$ pytest -q" in text
    assert "tests" in text and "pass" in text
    assert "ROCm:9000" in text and "1m15s" in text and "12.3 tok/s" in text


def test_receipt_collapses_commands_beyond_five():
    cmds = [f"cmd{i}" for i in range(8)]
    text = receipt.build_receipt([], cmds, None, 2.0, None, "auto")
    assert "cmd4" in text and "cmd5" not in text
    assert "+3 more" in text and "tok/s" not in text


def test_receipt_failed_tests_shown_and_commands_only_ok():
    text = receipt.build_receipt([], ["make"], "3 failed", 1.0, None, "9000")
    assert "3 failed" in text and "changed" not in text


def test_render_receipt_prints_or_skips():
    buf = io.StringIO()
    assert receipt.render_receipt([], [], None, 1.0, None, "9000", file=buf) is False
    assert buf.getvalue() == ""
    assert receipt.render_receipt([], ["ls"], None, 1.0, None, "9000", file=buf) is True
    assert "$ ls" in buf.getvalue()
