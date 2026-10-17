import io

import pytest

rich_console = pytest.importorskip("rich.console")

from kontexta_cli.terminal import ui


def _capture(monkeypatch, width=60):
    buf = io.StringIO()
    monkeypatch.setattr(ui, "console", rich_console.Console(file=buf, force_terminal=False, width=width))
    return buf


def test_markdown_rendered_when_enabled(monkeypatch):
    buf = _capture(monkeypatch)
    r = ui.StreamRenderer(markdown=True)
    for tok in ["# Title\n\n", "- item one\n", "- item two\n"]:
        r.feed(tok)
    r.stop()
    out = buf.getvalue()
    assert "Title" in out and "item one" in out
    assert "# Title" not in out  # heading marker consumed by the renderer


def test_raw_mode_streams_tokens(monkeypatch, capsys):
    _capture(monkeypatch)
    r = ui.StreamRenderer(markdown=False)
    r.feed("hello ")
    r.feed("world")
    r.stop()
    assert capsys.readouterr().out == "hello world\n"


def test_disabled_renderer_prints_nothing(monkeypatch, capsys):
    buf = _capture(monkeypatch)
    r = ui.StreamRenderer(markdown=True, enabled=False)
    r.feed("secret")
    r.stop()
    assert capsys.readouterr().out == "" and buf.getvalue() == ""


def test_stop_is_idempotent(monkeypatch):
    _capture(monkeypatch)
    r = ui.StreamRenderer(markdown=True)
    r.feed("x")
    r.stop()
    r.stop()


def test_markdown_no_duplicate_lines_on_multiblock(monkeypatch):
    buf = _capture(monkeypatch)
    r = ui.StreamRenderer(markdown=True)
    tokens = [
        "# Architectural Plan\n\n",
        "UniqueParagraphA: First paragraph content.\n\n",
        "---\n\n",
        "UniqueParagraphB: Second paragraph content.\n\n",
        "```python\ndef unique_func():\n    return 42\n```\n",
        "- UniqueItem1\n",
        "- UniqueItem2\n\n",
        "UniqueConclusion: Final sentence.",
    ]
    for tok in tokens:
        r.feed(tok)
    r.stop()
    out = buf.getvalue()
    assert out.count("UniqueParagraphA") == 1
    assert out.count("UniqueParagraphB") == 1
    assert out.count("unique_func") == 1
    assert out.count("UniqueItem1") == 1
    assert out.count("UniqueItem2") == 1
    assert out.count("UniqueConclusion") == 1


def test_stream_renderer_reset_between_segments(monkeypatch):
    """Verify that calling stop() finalizes a segment and allows a fresh segment without repeating previous text."""
    buf = _capture(monkeypatch)
    r = ui.StreamRenderer(markdown=True)
    r.feed("SegmentOne: Introductory text.\n\n")
    r.stop()
    assert "SegmentOne" in buf.getvalue()

    r.feed("SegmentTwo: Followup text after tool.\n\n")
    r.stop()
    out = buf.getvalue()
    assert out.count("SegmentOne") == 1
    assert out.count("SegmentTwo") == 1

