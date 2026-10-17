import pytest

from kontexta_cli.terminal import ui
from kontexta_cli.terminal.ui import UI

ATTRS = ["ROCM", "CUDA", "CLOUD", "LAYA", "AMBER", "GREEN", "RED", "CYAN",
         "ROCM_BOLD", "CUDA_BOLD", "CLOUD_BOLD", "LAYA_BOLD", "AMBER_BOLD", "GREEN_BOLD", "RED_BOLD", "CYAN_BOLD",
         "WHITE", "GRAY", "DARK_GRAY"]


@pytest.fixture(autouse=True)
def restore_theme(monkeypatch):
    monkeypatch.setattr(UI, "_use_color", True)
    UI.set_theme("dark")
    yield
    monkeypatch.undo()
    UI.set_theme(UI.detect_theme())


def snapshot():
    return {a: getattr(UI, a) for a in ATTRS}


def test_switching_changes_every_attribute_and_reverts():
    dark = snapshot()
    assert UI.THEMES == ("dark", "light", "mono")
    assert UI.set_theme("light") and UI.theme_name == "light"
    light = snapshot()
    assert all(light[a] != dark[a] for a in ATTRS if a not in ("GRAY",)) or light != dark
    assert light["ROCM"] == "\033[38;5;25m" and light["WHITE"] == "\033[1;30m"
    assert UI.set_theme("dark")
    assert snapshot() == dark


def test_existing_references_pick_up_theme():
    before = UI.ok("x")
    UI.set_theme("light")
    assert UI.ok("x") != before


def test_mono_has_no_hue():
    assert UI.set_theme("mono")
    for a in ATTRS:
        value = getattr(UI, a)
        assert "38;5" not in value and "97" not in value
        assert value.startswith("\033[") and value.endswith("m")
    assert UI.RED == "\033[7m" and UI.GRAY == "\033[2m"


def test_unknown_theme_rejected_and_state_kept():
    assert UI.set_theme("nope") is False
    assert UI.theme_name == "dark"
    assert UI.set_theme("LIGHT") and UI.theme_name == "light"


def test_no_color_records_name_but_clears_attributes(monkeypatch):
    monkeypatch.setattr(UI, "_use_color", False)
    assert UI.set_theme("light")
    assert UI.theme_name == "light"
    assert all(getattr(UI, a) == "" for a in ATTRS)


def test_detect_theme_env(monkeypatch):
    monkeypatch.setenv("KXTC_THEME", "Light")
    assert UI.detect_theme() == "light"
    monkeypatch.setenv("KXTC_THEME", "mono")
    assert UI.detect_theme() == "mono"
    monkeypatch.setenv("KXTC_THEME", "neon")
    assert UI.detect_theme() == "dark"
    monkeypatch.delenv("KXTC_THEME")
    assert UI.detect_theme() == "dark"


def test_rich_theme_and_code_theme_follow_theme():
    pytest.importorskip("rich")
    UI.set_theme("light")
    assert ui._code_theme == "default"
    assert str(ui.MD_THEME.styles["markdown.h1"]) == "bold color(25)"
    UI.set_theme("mono")
    assert ui._code_theme == "bw"
    assert "color" not in str(ui.MD_THEME.styles["markdown.h1"])
    UI.set_theme("dark")
    assert ui._code_theme == "monokai"
    assert str(ui.MD_THEME.styles["markdown.h1"]) == "bold color(39)"
    assert "ui.border" in ui.MD_THEME.styles
