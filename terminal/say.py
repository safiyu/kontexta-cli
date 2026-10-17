"""One output layer: semantic, theme-aware, json-mode-aware printing for the terminal."""

import re
import shutil
import sys
from typing import List, Optional, Tuple

from . import ui as _ui
from .ui import UI

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")
_state = {"json": False, "verbose": False}


def configure(json_output: bool = False, verbose: bool = False):
    _state["json"] = bool(json_output)
    _state["verbose"] = bool(verbose)


def is_json() -> bool:
    return _state["json"]


def is_verbose() -> bool:
    return _state["verbose"]


def _strip(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _emit(text: str):
    """Everything goes to stderr, uncolored, in json mode so stdout stays machine-readable."""
    if _state["json"]:
        print(_strip(text), file=sys.stderr, flush=True)
    else:
        cur = None
        printed = False
        try:
            from .esc_listener import EscListener
            had_box = bool(EscListener._current and getattr(EscListener._current, "_box_drawn", False))
            if had_box:
                EscListener.clear_box()
            print(text, flush=True)
            printed = True
            cur = EscListener._current
            if cur and (cur.buffered_text or cur.submit_pending):
                cur._render_live_input()
        except Exception:
            if not printed:
                print(text, flush=True)


def _width(cap: int = 0) -> int:
    cols = max(20, shutil.get_terminal_size((80, 24)).columns)
    return min(cols, cap) if cap else cols


def _rich_ok() -> bool:
    return bool(_ui.RICH_AVAILABLE and _ui.console and UI._use_color and not _state["json"])


def fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def rule_line(label: str = "", color: str = "", width: int = 0) -> str:
    """A thin horizontal rule with an optional inline label, sized to the terminal."""
    total = width or _width(80)
    color = color or UI.DARK_GRAY
    if not label:
        return f"{color}{'─' * total}{UI.RST}"
    head = f"── {label} "
    return f"{color}{head}{'─' * max(2, total - len(head))}{UI.RST}"


def info(msg):
    _emit(f"{UI.GRAY}[i]{UI.RST} {msg}")


def ok(msg):
    _emit(UI.ok(str(msg)))


def warn(msg):
    _emit(UI.warn(str(msg)))


def err(msg):
    _emit(UI.err(str(msg)))


def dim(msg):
    _emit(f"{UI.GRAY}{msg}{UI.RST}")


def plain(msg):
    _emit(str(msg))


def markdown(text: str):
    if _state["json"]:
        _emit(text)
        return
    if _rich_ok():
        try:
            from rich.markdown import Markdown
            _ui.console.push_theme(_ui.MD_THEME)
            try:
                _ui.console.print(Markdown(str(text)))
            finally:
                _ui.console.pop_theme()
            return
        except Exception:
            pass
    plain(text)


def step(msg):
    """Quiet-tier progress line: routing, compaction, checkpoints."""
    _emit(f"{UI.DARK_GRAY}·{UI.RST} {UI.GRAY}{msg}{UI.RST}")


def detail(msg):
    if _state["verbose"]:
        _emit(f"{UI.GRAY}{msg}{UI.RST}")


def truncate_lines(text: str, max_lines: int = 12) -> Tuple[str, int]:
    """Returns (shown_text, hidden_line_count)."""
    lines = str(text).splitlines()
    keep = max(0, max_lines)
    if len(lines) <= keep:
        return str(text), 0
    return "\n".join(lines[:keep]), len(lines) - keep


def turn_header(label: str, target: str):
    _emit(rule_line(label, UI.target_color(target)))


def group(title: str):
    _emit(f"\n{UI.WHITE}{title}{UI.RST}\n{UI.DARK_GRAY}{'─' * len(_strip(title))}{UI.RST}")


def tool_row(name: str, summary: str, ok: bool = True, elapsed: Optional[float] = None, extra: str = ""):
    """One compact indented line per tool call, truncated to the terminal width."""
    mark = f"{UI.GREEN}✓{UI.RST}" if ok else f"{UI.RED}✗{UI.RST}"
    tail_parts = ([fmt_elapsed(elapsed)] if elapsed is not None else []) + ([extra] if extra else [])
    tail = f" · {' · '.join(tail_parts)}" if tail_parts else ""
    name = " ".join(str(name).split())
    summary = " ".join(str(summary).split())
    body = f"{name} {summary}".strip()
    room = max(8, _width() - len("  ✓ ") - len(tail) - 1)
    if len(body) > room:
        body = body[: room - 1] + "…"
    shown_name = body[: len(name)]
    shown_summary = body[len(name):]
    line = f"  {mark} {UI.WHITE}{shown_name}{UI.RST}{UI.GRAY}{shown_summary}{tail}{UI.RST}"
    _emit(line)


def _plain_lines(rows: List[Tuple[str, str]]) -> List[str]:
    label_w = max((len(_strip(str(k))) for k, _ in rows), default=0)
    out = []
    for key, value in rows:
        parts = _strip(str(value)).splitlines() or [""]
        label = _strip(str(key)).ljust(label_w)
        out.append(f"{label}  {parts[0]}" if label_w else parts[0])
        out.extend(f"{' ' * label_w}  {p}" for p in parts[1:])
    return out


def card(title: str, rows: List[Tuple[str, str]], actions: str = "", style: str = "info"):
    """A bordered panel of label/value rows with an optional action footer."""
    try:
        _ui.stop_status()
    except Exception:
        pass
    style = style if style in ("info", "warn", "danger") else "info"
    if _rich_ok():
        console = _ui.console
        grid = _ui.Table.grid(padding=(0, 2))
        grid.add_column(style="ui.dim", no_wrap=True)
        grid.add_column(overflow="fold")
        for key, value in rows:
            grid.add_row(_ui.Text.from_ansi(str(key)), _ui.Text.from_ansi(str(value)))
        panel = _ui.Panel(grid, title=_ui.Text(_strip(title), style="bold"), title_align="left", box=_ui.box.ROUNDED,
                          border_style=f"ui.{style}", subtitle=_ui.Text(_strip(actions)) if actions else None, subtitle_align="left", padding=(0, 1))
        console.push_theme(_ui.MD_THEME)
        try:
            console.print(panel)
        finally:
            console.pop_theme()
        return

    lines = _plain_lines(rows)
    cap = _width() - 4
    lines = [ln if len(ln) <= cap else ln[: cap - 3] + "..." for ln in lines]
    foot = _strip(actions)[:cap] if actions else ""
    head = f"-- {_strip(title)} "
    inner = max([len(head) - 2, *(len(ln) for ln in lines), len(foot)] or [0])
    out = [f"+{head}{'-' * max(0, inner + 2 - len(head))}+"]
    out += [f"| {ln.ljust(inner)} |" for ln in lines]
    if foot:
        out.append(f"+{'-' * (inner + 2)}+")
        out.append(f"| {foot.ljust(inner)} |")
    out.append(f"+{'-' * (inner + 2)}+")
    _emit("\n".join(out))


def table(title: str, headers: List[str], rows: List[List[str]]):
    """A Rich table with a light box, or aligned plain text without color."""
    if _rich_ok():
        console = _ui.console
        tbl = _ui.Table(title=_ui.Text(_strip(title)) if title else None, title_justify="left", box=_ui.box.SIMPLE_HEAD, pad_edge=False,
                        show_edge=False, header_style="ui.head", border_style="ui.border")
        for h in headers:
            tbl.add_column(str(h))
        for row in rows:
            tbl.add_row(*[_ui.Text.from_ansi(str(c)) for c in row])
        console.push_theme(_ui.MD_THEME)
        try:
            console.print(tbl)
        finally:
            console.pop_theme()
        return

    heads = [_strip(str(h)) for h in headers]
    body = [[_strip(str(c)) for c in row] for row in rows]
    widths = [max([len(h)] + [len(r[i]) if i < len(r) else 0 for r in body]) for i, h in enumerate(heads)]
    fmt = lambda cells: "  ".join((cells[i] if i < len(cells) else "").ljust(widths[i]) for i in range(len(heads))).rstrip()
    out = []
    if title:
        out.append(_strip(title))
    out.append(fmt(heads))
    out.append("  ".join("-" * w for w in widths))
    out.extend(fmt(r) for r in body)
    _emit("\n".join(out))
