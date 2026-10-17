"""Terminal styling, curated 256-color palette, and box decorations."""

import os
import re
import sys
from typing import Any

try:
    from rich.console import Console, Group
    from rich.markdown import Markdown
    from rich.table import Table
    from rich.live import Live
    from rich import box
    from rich.text import Text
    from rich.theme import Theme
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.rule import Rule
    from rich import markdown as _rich_md
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None
    Markdown = None
    Table = None
    Live = None
    box = None
    Text = None
    Theme = None
    Panel = None
    Group = None


# 256-color codes per theme; targets identify models only, amber=warn, red=error, gray=chrome
_THEME_COLORS = {
    "dark": {"ROCM": 39, "CUDA": 48, "CLOUD": 75, "LAYA": 141, "AMBER": 214, "GREEN": 40,
             "RED": 203, "CYAN": 51, "GRAY": 244, "DARK_GRAY": 239},
    "light": {"ROCM": 25, "CUDA": 29, "CLOUD": 27, "LAYA": 91, "AMBER": 130, "GREEN": 28,
              "RED": 160, "CYAN": 30, "GRAY": 240, "DARK_GRAY": 245},
}
_THEME_WHITE = {"dark": "1;97", "light": "1;30"}
_BOLD_ATTRS = ("ROCM", "CUDA", "CLOUD", "LAYA", "AMBER", "GREEN", "RED", "CYAN")
_MONO = {
    "ROCM": "1", "CUDA": "1", "CLOUD": "1", "LAYA": "1", "AMBER": "1", "GREEN": "1", "CYAN": "1",
    "RED": "7", "WHITE": "1", "GRAY": "2", "DARK_GRAY": "2",
    "ROCM_BOLD": "1", "CUDA_BOLD": "1", "CLOUD_BOLD": "1", "LAYA_BOLD": "1", "AMBER_BOLD": "1",
    "GREEN_BOLD": "1", "RED_BOLD": "1;7", "CYAN_BOLD": "1",
}
_COLOR_ATTRS = tuple(_BOLD_ATTRS) + tuple(f"{a}_BOLD" for a in _BOLD_ATTRS) + ("WHITE", "GRAY", "DARK_GRAY")


class UI:
    """Terminal styling, themeable 256-color palette, and box decorations."""
    _use_color = sys.stdout.isatty() and not os.getenv("NO_COLOR")

    RST = "\033[0m" if _use_color else ""
    BOLD = "\033[1m" if _use_color else ""
    DIM = "\033[2m" if _use_color else ""
    ITALIC = "\033[3m" if _use_color else ""

    THEMES = ("dark", "light", "mono")
    theme_name = "dark"

    @classmethod
    def detect_theme(cls) -> str:
        """Theme from env MYAGY_THEME (dark|light|mono), default dark."""
        name = os.environ.get("MYAGY_THEME", "").strip().lower()
        return name if name in cls.THEMES else "dark"

    @classmethod
    def set_theme(cls, name: str) -> bool:
        """Switches every color attribute (and the Rich markdown theme); False for unknown names."""
        name = str(name).strip().lower()
        if name not in cls.THEMES:
            return False
        if cls._use_color:
            if name == "mono":
                codes = dict(_MONO)
            else:
                colors = _THEME_COLORS[name]
                codes = {a: f"38;5;{colors[a]}" for a in _BOLD_ATTRS}
                codes.update({f"{a}_BOLD": f"1;38;5;{colors[a]}" for a in _BOLD_ATTRS})
                codes.update({"WHITE": _THEME_WHITE[name], "GRAY": f"38;5;{colors['GRAY']}",
                              "DARK_GRAY": f"38;5;{colors['DARK_GRAY']}"})
            for attr in _COLOR_ATTRS:
                setattr(cls, attr, f"\033[{codes[attr]}m")
        else:
            for attr in _COLOR_ATTRS:
                setattr(cls, attr, "")
        cls.theme_name = name
        _apply_rich_theme(name)
        return True

    @classmethod
    def target_color(cls, target: str) -> str:
        t = str(target).lower().strip()
        if t in ("9000", "rocm", "amd", "gpu0"):
            return cls.ROCM
        if t in ("9001", "9002", "cuda", "nvidia", "gpu1"):
            return cls.CUDA
        if t in ("cloud", "gemini", "oauth"):
            return cls.CLOUD
        if t in ("auto", "laya", "smart"):
            return cls.LAYA
        return cls.WHITE

    @classmethod
    def badge(cls, text: str, color: str = "") -> str:
        color = color or cls.GRAY
        return f"{color}{cls.BOLD}[{text}]{cls.RST}"

    @classmethod
    def ok(cls, text: str) -> str:
        return f"{cls.GREEN_BOLD}[✓]{cls.RST} {cls.WHITE}{text}{cls.RST}"

    @classmethod
    def warn(cls, text: str) -> str:
        return f"{cls.AMBER_BOLD}[⚠]{cls.RST} {cls.AMBER}{text}{cls.RST}"

    @classmethod
    def err(cls, text: str) -> str:
        return f"{cls.RED_BOLD}[✗]{cls.RST} {cls.RED}{text}{cls.RST}"


# Rich style strings per theme; dark values are the original palette
_MD_STYLES = {
    "dark": {
        "h1": "bold color(39)", "h2": "bold color(75)", "h3": "bold color(141)", "h4": "bold color(48)",
        "code": "color(214) on color(236)", "quote": "color(244)", "rule": "color(239)",
        "accent": "color(39)", "link": "color(75)", "head": "bold color(75)",
        "info": "color(75)", "warn": "color(214)", "danger": "color(203)", "dim": "color(244)",
        "rule1": "color(39)",
    },
    "light": {
        "h1": "bold color(25)", "h2": "bold color(27)", "h3": "bold color(91)", "h4": "bold color(29)",
        "code": "color(130) on color(254)", "quote": "color(240)", "rule": "color(245)",
        "accent": "color(25)", "link": "color(27)", "head": "bold color(27)",
        "info": "color(27)", "warn": "color(130)", "danger": "color(160)", "dim": "color(240)",
        "rule1": "color(25)",
    },
    "mono": {
        "h1": "bold", "h2": "bold", "h3": "bold", "h4": "bold",
        "code": "bold", "quote": "dim", "rule": "dim",
        "accent": "bold", "link": "underline", "head": "bold",
        "info": "bold", "warn": "bold", "danger": "reverse", "dim": "dim",
        "rule1": "bold",
    },
}
_CODE_THEMES = {"dark": "monokai", "light": "default", "mono": "bw"}
_code_theme = "monokai"
MD_THEME = None


def _apply_rich_theme(name: str):
    """Rebuilds the Rich theme and code syntax theme for the active UI theme."""
    global MD_THEME, _code_theme
    _code_theme = _CODE_THEMES.get(name, "monokai")
    if not RICH_AVAILABLE:
        return
    st = _MD_STYLES.get(name, _MD_STYLES["dark"])
    MD_THEME = Theme({
        "markdown.h1": st["h1"],
        "markdown.h2": st["h2"],
        "markdown.h3": st["h3"],
        "markdown.h4": st["h4"],
        "markdown.h5": "bold",
        "markdown.h6": "bold dim",
        "markdown.code": st["code"],
        "markdown.block_quote": st["quote"],
        "markdown.hr": st["rule"],
        "markdown.item.bullet": f"bold {st['accent']}" if st["accent"] != "bold" else "bold",
        "markdown.item.number": f"bold {st['accent']}" if st["accent"] != "bold" else "bold",
        "markdown.link": f"underline {st['link']}" if st["link"] != "underline" else "underline",
        "markdown.link_url": f"dim underline {st['link']}" if st["link"] != "underline" else "dim underline",
        "markdown.table.header": st["head"],
        "markdown.table.border": st["rule"],
        "ui.rule1": st["rule1"],
        "ui.rule2": st["rule"],
        "ui.border": st["rule"],
        "ui.info": st["info"],
        "ui.warn": st["warn"],
        "ui.danger": st["danger"],
        "ui.dim": st["dim"],
        "ui.head": st["head"],
    })


if RICH_AVAILABLE:
    class _StyledHeading(_rich_md.Heading):
        def __rich_console__(self, console, options):
            text = self.text.copy()
            text.justify = "left"
            yield Text("")
            yield text
            if self.tag == "h1":
                yield Rule(characters="\u2501", style="ui.rule1")
            elif self.tag == "h2":
                yield Rule(characters="\u2500", style="ui.rule2")

    class _StyledCodeBlock(_rich_md.CodeBlock):
        def __rich_console__(self, console, options):
            code = str(self.text).rstrip()
            syntax = Syntax(code, self.lexer_name, theme=self.theme, word_wrap=True,
                            background_color="default", padding=(0, 1))
            title = self.lexer_name if self.lexer_name != "text" else None
            yield Panel(syntax, title=title, title_align="left", box=box.ROUNDED,
                        border_style="ui.border", padding=0)

    class _StyledTable(_rich_md.TableElement):
        def __rich_console__(self, console, options):
            table = Table(box=box.SIMPLE_HEAD, pad_edge=False, style="markdown.table.border",
                          show_edge=False, collapse_padding=True)
            if self.header is not None and self.header.row is not None:
                for column in self.header.row.cells:
                    heading = column.content.copy()
                    heading.stylize("markdown.table.header")
                    table.add_column(heading)
            if self.body is not None:
                for row in self.body.rows:
                    table.add_row(*[element.content for element in row.cells])
            yield table

    class StyledMarkdown(Markdown):
        elements = {
            **Markdown.elements,
            "heading_open": _StyledHeading,
            "fence": _StyledCodeBlock,
            "code_block": _StyledCodeBlock,
            "table_open": _StyledTable,
        }

        def __init__(self, markup: str):
            super().__init__(markup, code_theme=_code_theme)


UI.set_theme(UI.detect_theme())

_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}(\s|$)")
_HR_RE = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
_SETEXT_RE = re.compile(r"^\s{0,3}(=+|-+)\s*$")
_LIST_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+")
_TABLE_RE = re.compile(r"^\s*\|")
_QUOTE_RE = re.compile(r"^\s{0,3}>")


class StreamRenderer:
    """Streams model output: block-rendered Markdown with rich, raw tokens otherwise."""

    def __init__(self, markdown: bool = True, enabled: bool = True):
        self.enabled = enabled
        self._md = bool(markdown and RICH_AVAILABLE and enabled)
        self._buf = ""
        self._in_code_block = False
        self._code_lines: list = []
        self._pending: list = []
        self._kind = ""
        self.started = False

    def feed(self, text: str):
        if not self.enabled or not text:
            return
        self.started = True
        if self._md:
            self._buf += text
            self._process_buffer()
        else:
            sys.stdout.write(text)
            sys.stdout.flush()

    def _render_block(self, block: str):
        b = block.strip()
        if not b or not console:
            return
        had_box = False
        try:
            from .esc_listener import EscListener
            had_box = bool(EscListener._current and getattr(EscListener._current, "_box_drawn", False))
            if had_box:
                EscListener.clear_box()
        except Exception:
            pass
        console.push_theme(MD_THEME)
        try:
            console.print(StyledMarkdown(b))
        finally:
            console.pop_theme()
        if had_box:
            try:
                from .esc_listener import EscListener
                cur = EscListener._current
                if cur and (cur.buffered_text or cur.submit_pending):
                    cur._render_live_input()
            except Exception:
                pass

    def _flush_pending(self):
        if self._pending:
            self._render_block("\n".join(self._pending))
        self._pending = []
        self._kind = ""

    def _start(self, kind: str, line: str):
        self._flush_pending()
        self._pending, self._kind = [line], kind

    def _process_buffer(self):
        while True:
            nl = self._buf.find("\n")
            if nl == -1:
                return
            line = self._buf[:nl].rstrip("\r")
            self._buf = self._buf[nl + 1:]
            self._process_line(line)

    def _process_line(self, line: str):
        if self._in_code_block:
            self._code_lines.append(line)
            if re.match(r"^\s{0,3}(`{3,}|~{3,})\s*$", line):
                self._in_code_block = False
                self._render_block("\n".join(self._code_lines))
                self._code_lines = []
            return

        if not line.strip():
            self._flush_pending()
            return

        if _FENCE_RE.match(line):
            self._flush_pending()
            self._in_code_block = True
            self._code_lines = [line]
            return

        if _SETEXT_RE.match(line) and self._kind == "para" and len(self._pending) == 1:
            self._pending.append(line)
            self._flush_pending()
            return

        if _HEADING_RE.match(line) or _HR_RE.match(line):
            self._flush_pending()
            self._render_block(line)
            return

        if _TABLE_RE.match(line):
            if self._kind == "table":
                self._pending.append(line)
            else:
                self._start("table", line)
        elif _QUOTE_RE.match(line):
            if self._kind == "quote":
                self._pending.append(line)
            else:
                self._start("quote", line)
        elif _LIST_RE.match(line):
            if self._kind == "list":
                self._pending.append(line)
            else:
                self._start("list", line)
        elif self._kind in ("para", "list", "quote"):
            self._pending.append(line)
        else:
            self._start("para", line)

    def stop(self):
        """Finalizes output; safe to call multiple times."""
        if self._md:
            if self._buf:
                self._process_line(self._buf.rstrip("\r"))
            self._buf = ""
            if self._in_code_block:
                self._render_block("\n".join(self._code_lines))
            self._code_lines = []
            self._in_code_block = False
            self._flush_pending()
        elif self.started and not self._md and self.enabled:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self.started = False


_active_status = None


def start_status(msg: str = "Thinking & generating response..."):
    """Starts or updates a Rich animated dots spinner, thread-safely."""
    global _active_status
    if not RICH_AVAILABLE or not console:
        return None
    if _active_status is not None:
        try:
            _active_status.update(f"[bold cyan]{msg}[/bold cyan]")
            return _active_status
        except Exception:
            stop_status()
    try:
        _active_status = console.status(f"[bold cyan]{msg}[/bold cyan]", spinner="dots")
        _active_status.start()
    except Exception:
        _active_status = None
    return _active_status


def stop_status():
    """Stops and clears any active Rich status spinner from the terminal."""
    global _active_status
    if _active_status is not None:
        try:
            _active_status.stop()
        except Exception:
            pass
        _active_status = None


async def render_turn_live(session, prompt: str, cancel_event=None) -> Any:
    """Connects native triple-stream callbacks into a single rich.live.Live layout."""
    if not RICH_AVAILABLE or not console or getattr(session, "json_output", False):
        if hasattr(session, "run_turn"):
            return await session.run_turn(prompt, cancel_event=cancel_event)
        return await session.chat(prompt)

    thought_buf: list[str] = []
    answer_buf: list[str] = []
    active_tools: list[str] = []

    def _build_renderable():
        blocks = []
        if thought_buf and not answer_buf:
            snippet = "".join(thought_buf)[-400:]
            blocks.append(
                Panel(Text(snippet, style="dim italic"), title="🧠 Reasoning", border_style="dim")
            )
        if active_tools:
            blocks.append(Text(f"⚙ Running: {active_tools[-1]}", style="bold cyan"))
        if answer_buf:
            blocks.append(Markdown("".join(answer_buf)))
        return Group(*blocks) if blocks else Text("")

    with Live(_build_renderable(), console=console, refresh_per_second=20) as live:
        receipt = await session.run_turn(
            prompt=prompt,
            on_thought=lambda t: (thought_buf.append(t), live.update(_build_renderable())),
            on_tool_start=lambda name, _: (active_tools.append(name), live.update(_build_renderable())),
            on_token=lambda tok: (answer_buf.append(tok), live.update(_build_renderable())),
            cancel_event=cancel_event,
        )

    # Render compact Turn Receipt footer
    console.print(
        f"[dim]── {receipt.target.upper()} ({receipt.model_id}) │ "
        f"Δfiles: {receipt.files_changed} │ cmds: {receipt.commands_run} │ "
        f"TTFT: {receipt.ttft_ms:.0f}ms │ {receipt.tok_per_sec:.1f} tok/s "
        f"({receipt.elapsed_s:.1f}s)[/dim]"
    )
    return receipt


# ── 3-Zone Prefix-Preserving Memory Telemetry ──────────────────────────

try:
    from ..core.session import ContextZoneTelemetry
except Exception:
    try:
        from myagy.core.session import ContextZoneTelemetry
    except Exception:
        ContextZoneTelemetry = Any


def _build_stacked_context_bar(ctx: ContextZoneTelemetry, width: int = 56) -> Any:
    """
    Renders a single-line multi-zone proportional bar with a watermark tick (`│`):
      [cyan]█[/cyan] Zone 1 (System/AST)
      [magenta]█[/magenta] Zone 2 (Frozen Epochs)
      [yellow]█[/yellow] Zone 3 (Active Window)
      [dim]░[/dim] Headroom
    """
    if not RICH_AVAILABLE or Text is None:
        return ""
    total_max = max(1, ctx.n_ctx_max)
    z1_cells = round((ctx.zone1_tokens / total_max) * width)
    z2_cells = round((ctx.zone2_tokens / total_max) * width)
    z3_cells = round((ctx.zone3_tokens / total_max) * width)

    # Guarantee non-zero zones show at least 1 cell for visibility
    if ctx.zone1_tokens > 0 and z1_cells == 0:
        z1_cells = 1
    if ctx.zone2_tokens > 0 and z2_cells == 0:
        z2_cells = 1
    if ctx.zone3_tokens > 0 and z3_cells == 0:
        z3_cells = 1

    used_cells = min(width, z1_cells + z2_cells + z3_cells)
    watermark_pos = min(width - 1, round(ctx.compaction_watermark_ratio * width))

    bar = Text()
    for i in range(width):
        if i == watermark_pos:
            bar.append("┃", style="bold red")
            continue
        if i < z1_cells:
            bar.append("█", style="bold cyan")
        elif i < z1_cells + z2_cells:
            bar.append("█", style="bold magenta")
        elif i < used_cells:
            bar.append("█", style="bold yellow")
        else:
            bar.append("░", style="dim bright_black")

    return bar


def render_context_telemetry(ctx: ContextZoneTelemetry) -> None:
    """Renders the `/context` (`/ctx`) panel with Zone 1-3 breakdown & KV cache state."""
    if not RICH_AVAILABLE or not console or Text is None or Table is None or Panel is None or Group is None:
        pct = ctx.utilization_pct
        print(f"--- Context Window Telemetry [{ctx.target.upper()} ({ctx.model_id})] ---")
        print(f"Window: {ctx.total_used_tokens:,} / {ctx.n_ctx_max:,} tok ({pct:.1f}%)")
        print(f"  Zone 1: {ctx.zone1_tokens:,} tok")
        print(f"  Zone 2: {ctx.zone2_tokens:,} tok ({ctx.frozen_epoch_count} blocks)")
        print(f"  Zone 3: {ctx.zone3_tokens:,} tok")
        print(f"  Headroom: {ctx.headroom_tokens:,} tok")
        print(f"  KV Reuse: {ctx.kv_prefix_hit_pct:.0f}% ({ctx.kv_cached_tokens:,} tok)")
        return

    bar = _build_stacked_context_bar(ctx, width=58)

    # Color-code overall utilization severity
    pct = ctx.utilization_pct
    pct_style = "bold green" if pct < 60 else ("bold yellow" if pct < 80 else "bold red")

    header_line = Text.assemble(
        ("  Window: ", "dim"),
        bar,
        (f"  {pct:5.1f}% ", pct_style),
        (f"({ctx.total_used_tokens:,} / {ctx.n_ctx_max:,} tok)", "dim"),
    )

    legend_line = Text.assemble(
        ("  Legend: ", "dim"),
        ("█ Zone 1: System/AST ", "bold cyan"),
        ("█ Zone 2: Frozen Epochs ", "bold magenta"),
        ("█ Zone 3: Active Turns ", "bold yellow"),
        ("┃ Compaction Trigger (75%)", "bold red"),
    )

    # Zone 1-3 Breakdown Table
    table = Table(
        box=box.SIMPLE_HEAVY if box else None,
        expand=True,
        show_edge=False,
        pad_edge=False,
        header_style="bold white",
    )
    table.add_column("Memory Zone", style="white", ratio=3)
    table.add_column("Sub-Component", style="dim", ratio=4)
    table.add_column("Tokens", justify="right", ratio=2)
    table.add_column("% of Max", justify="right", ratio=1)
    table.add_column("KV Cache Policy", justify="left", ratio=3)

    def _pct_str(tokens: int) -> str:
        return f"{(tokens / ctx.n_ctx_max * 100.0):.1f}%" if ctx.n_ctx_max else "0.0%"

    # Zone 1 Rows
    table.add_row(
        "[bold cyan]Zone 1: Static Prefix[/bold cyan]",
        "GEMINI.md / instructions.py",
        f"{ctx.instructions_tokens:,}",
        _pct_str(ctx.instructions_tokens),
        "[green]Pinned (100% KV Hit)[/green]",
    )
    table.add_row(
        "",
        "RepoMap AST Outline (repomap.py)",
        f"{ctx.repomap_tokens:,}",
        _pct_str(ctx.repomap_tokens),
        "[green]mtime-cached (<1.5ms)[/green]",
    )
    table.add_row(
        "",
        "MCP Tool Schemas (kxta + native)",
        f"{ctx.mcp_schema_tokens:,}",
        _pct_str(ctx.mcp_schema_tokens),
        "[green]Pinned Prefix[/green]",
        end_section=True,
    )

    # Zone 2 Row
    table.add_row(
        "[bold magenta]Zone 2: Working Memory[/bold magenta]",
        f"Frozen Epochs ({ctx.frozen_epoch_count} blocks)",
        f"{ctx.frozen_epoch_tokens:,}",
        _pct_str(ctx.frozen_epoch_tokens),
        "[magenta]Append-Only (KV Safe)[/magenta]",
        end_section=True,
    )

    # Zone 3 Rows
    table.add_row(
        "[bold yellow]Zone 3: Active Window[/bold yellow]",
        f"Tier-0 Pruned Turns (-{ctx.tier0_saved_tokens:,} tok saved)",
        f"{ctx.pruned_turn_tokens:,}",
        _pct_str(ctx.pruned_turn_tokens),
        "[yellow]Tool stdout elided[/yellow]",
    )
    table.add_row(
        "",
        "Verbatim Recent Turns",
        f"{ctx.verbatim_turn_tokens:,}",
        _pct_str(ctx.verbatim_turn_tokens),
        "[yellow]Dynamic Tail[/yellow]",
        end_section=True,
    )

    # Summary / Headroom Row
    watermark_tok = int(ctx.n_ctx_max * ctx.compaction_watermark_ratio)
    tokens_until_compact = max(0, watermark_tok - ctx.total_used_tokens)
    table.add_row(
        "[bold green]Available Headroom[/bold green]",
        f"{tokens_until_compact:,} tok until next CUDA:9002 epoch compaction",
        f"[bold green]{ctx.headroom_tokens:,}[/bold green]",
        f"[bold green]{100.0 - pct:.1f}%[/bold green]",
        f"[cyan]KV Reuse: {ctx.kv_prefix_hit_pct:.0f}% ({ctx.kv_cached_tokens:,} tok)[/cyan]",
    )

    panel = Panel(
        Group(header_line, legend_line, Text(""), table),
        title=(
            f"[bold cyan]🧠 Context Window Telemetry[/bold cyan] "
            f"[dim]── Target: {ctx.target.upper()} ({ctx.model_id})[/dim]"
        ),
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(panel)


def format_toolbar_context_segment(ctx: ContextZoneTelemetry) -> str:
    """
    Returns an ANSI/prompt_toolkit compact status string for the bottom bar:
    e.g., 'CTX: 6.4k/32k (20%) [Epochs:2 | KV:94%]'
    """
    used_k = ctx.total_used_tokens / 1000.0
    max_k = ctx.n_ctx_max / 1000.0
    return (
        f"CTX: {used_k:.1f}k/{max_k:.0f}k ({ctx.utilization_pct:.0f}%) "
        f"[Epochs:{ctx.frozen_epoch_count} | KV:{ctx.kv_prefix_hit_pct:.0f}%]"
    )



