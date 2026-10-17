"""Interactive command-line interface, multi-line prompt reader, and main runner."""

import os
import sys
import json
import signal
import select
import asyncio
import logging
import threading
import argparse
import subprocess
import shutil
import time
import traceback
from typing import Any, Optional, Dict, List

# Suppress raw WARNING:root: noise from the google.antigravity library (e.g. "System step error");
# myagy surfaces those errors via its own say.warn() / say.err() paths.
logging.getLogger().setLevel(logging.ERROR)

from ..config import (
    MCP_CONFIG_PATH,
    DEFAULT_CLOUD_MODEL,
    HISTORY_PATH,
    PORT_ROCM,
    setup_antigravity_sdk,
)
from . import say
from .ui import UI, StreamRenderer, RICH_AVAILABLE, console
from ..core.prefs import PREF_DEFAULTS, load_prefs, reset_prefs
from ..core.llamashift import wake_compactor
from ..context.mcp_loader import load_mcp_servers, McpStdioServer
from ..core.session import MultiGpuHybridSession
from ..core.sudo_manager import SudoManager
from ..core.reloader import (
    ReloadRequested,
    perform_execv_reload,
    write_reload_snapshot,
    consume_reload_snapshot,
)
from .esc_listener import EscListener, prompt_input


COMMANDS: dict = {}
COMMAND_META: list = []
SLASH_COMMANDS: list = []
GROUP_ORDER = ("Session", "Models", "Agents", "Git", "Debug", "Settings")
_COMPLETION_ONLY = ["/paste"]
_EXTRA_HELP = [(("/paste",), "Session", "Paste a multi-line block (end with EOF, . or END)")]
_EXIT = "exit"
_NO_SAVE = "no_save"
_TARGET_LABELS = {"auto": "AUTO:Laya", "9000": "ROCm:9000", "9002": "CUDA:9002", "cloud": "CLOUD:OAuth"}


class PromptInterrupted(Exception):
    """Ctrl+C pressed at the first prompt line."""


class _ExitSession(Exception):
    """Raised into a pending prompt read to leave the session."""


def command(*names, group=None, help=None):
    """Registers an async slash-command handler under each name (first registration wins)."""
    def deco(fn):
        COMMAND_META.append({"names": names, "group": group or "Other", "help": help or "", "fn": fn})
        for n in names:
            COMMANDS.setdefault(n, fn)
        return fn
    return deco


def help_groups() -> dict:
    """Maps group name to (aliases, description) rows, built from the command registry."""
    groups = {g: [] for g in GROUP_ORDER}
    entries = [(m["names"], m["group"], m["help"]) for m in COMMAND_META] + _EXTRA_HELP
    for names, group, text in entries:
        groups.setdefault(group, []).append((", ".join(names), text))
    return {g: sorted(rows) for g, rows in groups.items() if rows}


def _sync_say(session) -> None:
    say.configure(json_output=session.json_output, verbose=session.verbose)


AT_TARGETS = ["@rocm", "@cuda", "@cloud", "@auto", "@local", "@9000", "@9001", "@9002"]
_dynamic_targets: list = []
_pt_session = None
_toolbar_provider = None


def set_completion_targets(ids) -> None:
    """Registers model ids for @<id> completion."""
    global _dynamic_targets
    _dynamic_targets = [f"@{i}" for i in ids if i]


def set_toolbar(provider) -> None:
    """Registers a callable returning prompt_toolkit (style, text) fragments for the bottom toolbar."""
    global _toolbar_provider
    _toolbar_provider = provider


def _toolbar_fragments():
    # A broken toolbar must never take the prompt down with it
    try:
        return _toolbar_provider() if _toolbar_provider else []
    except Exception:
        return []


COMMAND_USAGES = {
    "/auto": "/auto",
    "/9000": "/9000",
    "/rocm": "/rocm",
    "/amd": "/amd",
    "/9001": "/9001",
    "/9002": "/9002",
    "/cuda": "/cuda",
    "/nvidia": "/nvidia",
    "/cloud": "/cloud [<model-id>]",
    "/gemini": "/gemini [<model-id>]",
    "/oauth": "/oauth [<model-id>]",
    "/model": "/model [<id|port>]",
    "/switch": "/switch [<id|port>]",
    "/shift": "/shift [<id|port>]",
    "/use": "/use [<id|port>]",
    "/models": "/models",
    "/mode": "/mode <num|id>",
    "/laya": "/laya",
    "/autoshift": "/autoshift [on|off]",
    "/auto-shift": "/auto-shift [on|off]",
    "/model-shift": "/model-shift [on|off]",
    "/git-shift": "/git-shift [on|off]",
    "/gitshift": "/gitshift [on|off]",
    "/skip-permissions": "/skip-permissions [on|off]",
    "/dangerously-skip-permissions": "/dangerously-skip-permissions [on|off]",
    "/permissions": "/permissions [auto|skip|safe]",
    "/sudo": "/sudo [<cmd>|clear|status]",
    "/theme": "/theme [dark|light|mono]",
    "/mcp": "/mcp [status|on|off|reload]",
    "/kontexta": "/kontexta [status|on|off|reload|register]",
    "/kxta": "/kxta [status|on|off|reload|register]",
    "/memory": "/memory [on|off]",
    "/journal": "/journal [on|off]",
    "/autocompact": "/autocompact [on|off|<turns>]",
    "/auto-compact": "/auto-compact [on|off|<turns>]",
    "/curation": "/curation [on|off]",
    "/continuous": "/continuous [on|off]",
    "/repomap": "/repomap [view|on|off|refresh|stats]",
    "/ast": "/ast [view|on|off|refresh|stats]",
    "/instructions": "/instructions [view|on|off|refresh]",
    "/rules": "/rules [view|on|off|refresh]",
    "/guidelines": "/guidelines [view|on|off|refresh]",
    "/hooks": "/hooks [view|on|off|refresh]",
    "/hook": "/hook [view|on|off|refresh]",
    "/markdown": "/markdown [on|off]",
    "/md": "/md [on|off]",
    "/multiline": "/multiline",
    "/multi": "/multi",
    "/singleline": "/singleline",
    "/single": "/single",
    "/format": "/format [text|json]",
    "/prefs": "/prefs [reset]",
    "/preferences": "/preferences [reset]",
    "/metrics": "/metrics [on|off]",
    "/steps": "/steps [<n>|off]",
    "/maxsteps": "/maxsteps [<n>|off]",
    "/max-steps": "/max-steps [<n>|off]",
    "/verbose": "/verbose [on|off]",
    "/v": "/v [on|off]",
    "/json": "/json [on|off]",
    "/stats": "/stats [reset]",
    "/coordinator": "/coordinator",
    "/status": "/status",
    "/history": "/history",
    "/clear": "/clear",
    "/save": "/save [<name>]",
    "/load": "/load <name|file>",
    "/sessions": "/sessions",
    "/export": "/export [<file>]",
    "/summarize": "/summarize",
    "/context": "/context",
    "/ctx": "/ctx",
    "/compact": "/compact [<turns>]",
    "/last": "/last",
    "/reload": "/reload",
    "/restart": "/restart",
    "/help": "/help [<group>]",
    "/exit": "/exit",
    "/quit": "/quit",
    "/compactor": "/compactor [status|start]",
    "/curator": "/curator [status|start]",
    "/spawn": "/spawn <prompt>",
    "/subagent": "/subagent <prompt>",
    "/task": "/task <prompt>",
    "/build": "/build [<cmd>]",
    "/test": "/test [<cmd>]",
    "/tasks": "/tasks",
    "/subagents": "/subagents",
    "/brainstorm": "/brainstorm <goal>",
    "/spec": "/spec <goal>",
    "/plan": "/plan <goal>",
    "/search": "/search <query>",
    "/web": "/web <query>",
    "/google": "/google <query>",
    "/ddg": "/ddg <query>",
    "/fetch": "/fetch <url>",
    "/browse": "/browse <url>",
    "/readurl": "/readurl <url>",
    "/hermes": "/hermes [<prompt>|tui|status]",
    "/diff": "/diff [<n>]",
    "/undo": "/undo",
    "/paste": "/paste",
}

AT_TARGET_DOCS = {
    "@rocm": "@rocm — AMD ROCm model on port 9000",
    "@9000": "@9000 — AMD ROCm model on port 9000",
    "@amd": "@amd — AMD ROCm model on port 9000",
    "@cuda": "@cuda — NVIDIA CUDA model on port 9001/9002",
    "@9001": "@9001 — NVIDIA CUDA model on port 9001",
    "@9002": "@9002 — NVIDIA CUDA model on port 9002",
    "@nvidia": "@nvidia — NVIDIA CUDA model on port 9001/9002",
    "@cloud": "@cloud — Google Cloud model via OAuth",
    "@gemini": "@gemini — Google Cloud Gemini via OAuth",
    "@oauth": "@oauth — Google Cloud OAuth target",
    "@auto": "@auto — Automatic routing with Laya",
    "@laya": "@laya — Automatic routing with Laya",
    "@local": "@local — Active local model",
}


def command_doc(name: str) -> str:
    """Returns formatted 'usage — description' meta for slash command completion."""
    desc = ""
    for m in COMMAND_META:
        if name in m.get("names", ()):
            desc = m.get("help", "")
            break
    if not desc:
        for names, _, text in _EXTRA_HELP:
            if name in names:
                desc = text
                break
    usage = COMMAND_USAGES.get(name, name)
    if usage and desc:
        return f"{usage} — {desc}"
    return desc or usage


def _get_pt_session():
    """Lazily builds a prompt_toolkit session (history + completion); None if unavailable."""
    global _pt_session
    if _pt_session is not None:
        return _pt_session or None
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.styles import Style

        from prompt_toolkit.shortcuts import CompleteStyle

        class _InputCompleter(Completer):
            def get_completions(self, document, complete_event):
                text = document.text_before_cursor
                if " " in text or "\n" in text:
                    return
                if text.startswith("/"):
                    for cand in SLASH_COMMANDS:
                        if cand.lower().startswith(text.lower()):
                            meta = command_doc(cand)
                            yield Completion(cand, start_position=-len(text), display_meta=meta)
                elif text.startswith("@"):
                    for cand in AT_TARGETS + _dynamic_targets:
                        if cand.lower().startswith(text.lower()):
                            meta = AT_TARGET_DOCS.get(cand, f"{cand} — Local model")
                            yield Completion(cand, start_position=-len(text), display_meta=meta)

        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        style = Style.from_dict({
            "bottom-toolbar": "noreverse",
            "bottom-toolbar.text": "noreverse",
            "bottom-toolbar.rule": "#444444",
            "completion-menu.completion": "bg:#2b2b2b #ffffff",
            "completion-menu.completion.current": "bg:#005f87 #ffffff bold",
            "completion-menu.meta": "bg:#222222 #9e9e9e",
            "completion-menu.meta.current": "bg:#005f87 #e0e0e0",
        })
        from prompt_toolkit.key_binding import KeyBindings
        kb = KeyBindings()

        @kb.add("escape")
        def _handle_escape(event):
            buf = event.current_buffer
            if buf.text:
                buf.text = ""
                buf.cursor_position = 0
            else:
                event.app.exit(result="")

        _pt_session = PromptSession(
            history=FileHistory(HISTORY_PATH),
            completer=_InputCompleter(),
            key_bindings=kb,
            style=style,
            complete_while_typing=True,
            complete_style=CompleteStyle.COLUMN,
            reserve_space_for_menu=8,
        )
    except Exception:
        _pt_session = False
    return _pt_session or None


def _toolbar_active() -> bool:
    """True when the live bottom toolbar will be drawn (a real TTY with prompt_toolkit)."""
    return bool(_toolbar_provider and sys.stdin.isatty() and sys.stdout.isatty() and _get_pt_session() is not None)


def _read_line(prompt: str, default: str = "") -> str:
    """One line of input with history/completion on a TTY, plain input() otherwise.

    `default` prefills the editable line (prompt_toolkit only); plain input() ignores it.
    """
    if sys.stdin.isatty() and sys.stdout.isatty():
        pt = _get_pt_session()
        if pt is not None:
            from prompt_toolkit.formatted_text import ANSI, FormattedText
            kw = {"default": default} if default else {}
            if _toolbar_provider:
                return pt.prompt(ANSI(prompt), bottom_toolbar=lambda: FormattedText(_toolbar_fragments()), refresh_interval=1.0, **kw)
            return pt.prompt(ANSI(prompt), **kw)
    return input(prompt)


def read_input_prompt(first_prompt: str, multiline: bool = True, prefill: str = "") -> str:
    """
    Reads user input with multi-line support enabled by default.
    - Ctrl+C on the first line raises PromptInterrupted; Ctrl+D returns "".
    - Slash commands (/help, /status, etc.) and exit words submit immediately on first line.
    - Multi-line mode: subsequent lines are prompted with '... '.
    - Pressing Enter on an empty line or pressing Ctrl+D submits the multi-line input.
    - Pasted text with internal blank lines is preserved without prematurely submitting.
    """
    if sys.stdout.isatty() and not say.is_json():
        print(say.rule_line("", UI.DARK_GRAY), flush=True)

    try:
        first_line = _read_line(first_prompt, default=prefill) if prefill else _read_line(first_prompt)
    except EOFError:
        return ""
    except KeyboardInterrupt:
        raise PromptInterrupted()

    stripped = first_line.strip()
    if not multiline:
        if sys.stdout.isatty() and not say.is_json() and stripped:
            print(say.rule_line("", UI.DARK_GRAY), flush=True)
        return stripped

    # A multi-line paste arrives as one buffer when prompt_toolkit is active
    if "\n" in first_line:
        if sys.stdout.isatty() and not say.is_json() and stripped:
            print(say.rule_line("", UI.DARK_GRAY), flush=True)
        return stripped

    # Slash commands and session exit keywords submit immediately on first Enter
    if stripped.startswith("/") or stripped.lower() in ("exit", "quit", "q", "/exit", "/quit"):
        if sys.stdout.isatty() and not say.is_json() and stripped:
            print(say.rule_line("", UI.DARK_GRAY), flush=True)
        return stripped

    # Empty first line returns immediately
    if not stripped:
        return ""

    lines = [first_line]
    continuation_prompt = f"{UI.DARK_GRAY}  │{UI.RST} " if UI._use_color else "... "
    while True:
        try:
            line = input(continuation_prompt)
            if line == "":
                # Check if data is actively buffered in stdin (e.g. rapid terminal paste burst)
                try:
                    r, _, _ = select.select([sys.stdin], [], [], 0.02)
                    if r:
                        # Internal empty line within pasted text -> keep it
                        lines.append("")
                        continue
                except Exception:
                    pass
                # Human pressed Enter on an empty line -> submit
                break
            lines.append(line)
        except EOFError:
            # Ctrl+D submits what has been entered so far
            break
        except KeyboardInterrupt:
            # Ctrl+C cancels the current multi-line entry and returns to main prompt
            say.dim("[^C] Input cancelled.")
            return ""

    result = "\n".join(lines).strip()
    if sys.stdout.isatty() and not say.is_json() and result:
        print(say.rule_line("", UI.DARK_GRAY), flush=True)
    return result


def _settle(fut, result, exc):
    if not fut.done():
        if exc is not None:
            fut.set_exception(exc)
        else:
            fut.set_result(result)


def _spawn_reader(loop, fn, *args):
    """Runs a blocking prompt read on a daemon thread so exiting never waits on stdin."""
    fut = loop.create_future()

    def work():
        try:
            res, exc = fn(*args), None
        except BaseException as e:
            res, exc = None, e
        try:
            loop.call_soon_threadsafe(_settle, fut, res, exc)
        except RuntimeError:
            pass

    threading.Thread(target=work, daemon=True, name="PromptReader").start()
    return fut


class _LoopCtx:
    def __init__(self, loop):
        self.loop = loop
        self.task = None
        self.read_fut = None
        self.sigint = 0
        # Queue-ahead typing: text captured during a turn, awaiting next iteration
        self.queued_input: str = ""
        self.queued_submit: bool = False


def _prompt_string(session) -> str:
    """The input prompt itself; status lives in the bottom toolbar."""
    if UI._use_color:
        return f"{UI.target_color(session.active_target)}❯{UI.RST} "
    return "> "


def _plain_pill(session) -> str:
    """One-line status for terminals without the live toolbar."""
    label = _TARGET_LABELS.get(str(session.active_target), str(session.active_target))
    tokens = session.get_context_metrics(session.active_target)["tokens"]
    shown = f"{tokens / 1000:.1f}k" if tokens >= 1000 else str(tokens)
    return f"[{label} | {shown} ctx]"


async def _confirm(title, rows, prompt="Proceed?", style="warn", assume_yes=False) -> bool:
    """Shows a confirmation card, then reads y/N unless the caller already said yes."""
    say.card(title, rows, actions="[y]es / [n]o", style=style)
    if assume_yes:
        return True
    answer = await asyncio.to_thread(prompt_input, f"{UI.AMBER_BOLD}{prompt} [y/N]: {UI.RST}")
    return answer.strip().lower() in ("y", "yes")


# ── Models ────────────────────────────────────────────────────────────

@command("/auto", group="Models", help="Route every prompt automatically with Laya")
async def _cmd_auto(session, arg, ctx):
    session.auto_route = not getattr(session, "auto_route", False)
    session.active_target = "auto" if session.auto_route else "rocm"
    state = "enabled" if session.auto_route else "disabled"
    avg_lat = getattr(session.laya.telemetry, "avg_latency_ms", 32.8) if hasattr(session, "laya") and hasattr(session.laya, "telemetry") else 32.8
    if RICH_AVAILABLE and console:
        console.print(
            f"[cyan]⚡ Laya System 1 auto-routing (:8003) {state} "
            f"(avg {avg_lat:.1f}ms)[/cyan]"
        )
    else:
        say.ok(f"Laya System 1 auto-routing (:8003) {state} (avg {avg_lat:.1f}ms)")


@command("/9000", "/rocm", "/amd", group="Models", help="Use the AMD ROCm model on port 9000")
async def _cmd_9000(session, arg, ctx):
    session.active_target = "9000"
    await asyncio.to_thread(session.refresh_endpoints)
    fn = session.endpoints['9000']['filename']
    say.ok(f"Default target set to AMD ROCm Port 9000 ({fn})")


@command("/9002", "/9001", "/cuda", "/nvidia", group="Models", help="Use the NVIDIA CUDA model on port 9001/9002")
async def _cmd_9002(session, arg, ctx):
    target = session.resolve_target_alias("9001" if (ctx and "/9001" in str(ctx)) else "9002")
    session.active_target = target
    await asyncio.to_thread(session.refresh_endpoints)
    ep = session.get_endpoint(target) if hasattr(session, "get_endpoint") else session.endpoints.get(target, {})
    fn = ep.get('filename') or session.endpoints.get('9002', {}).get('filename') or session.endpoints.get('9001', {}).get('filename') or "model"
    say.ok(f"Default target set to NVIDIA CUDA Port {ep.get('port', target)} ({fn})")


@command("/cloud", "/gemini", "/oauth", group="Models", help="Use Google cloud via OAuth, or /cloud <model-id>")
async def _cmd_cloud(session, arg, ctx):
    clean = arg.strip().strip("[]<>") if arg else ""
    if clean and clean.lower() not in ("id", "model", "model-id", "model_id"):
        session.cloud_model = clean
        session.active_target = "cloud"
        session.save_preferences()
        say.ok(f"Cloud model set to {clean} and target switched to CLOUD (saved to prefs).")
    else:
        if arg and arg.strip():
            say.warn(f"Ignoring placeholder '{arg.strip()}'. Using cloud model '{session.cloud_model}'.")
        # Ensure session.cloud_model itself isn't a placeholder
        if not session.cloud_model or session.cloud_model.strip().strip("[]<>").lower() in ("", "id", "model", "model-id", "model_id"):
            session.cloud_model = DEFAULT_CLOUD_MODEL
            session.save_preferences()
        session.active_target = "cloud"
        say.ok(f"Default target set to CLOUD via OAuth ({session.cloud_model})")


@command("/model", "/switch", "/shift", "/use", group="Models", help="List local models or switch to one")
async def _cmd_model(session, arg, ctx):
    if not arg:
        session.print_models()
    else:
        await session.switch_model(arg)


@command("/models", group="Models", help="List available models and their status")
async def _cmd_models(session, arg, ctx):
    session.print_models()


@command("/mode", group="Models", help="Switch model by number or id")
async def _cmd_mode(session, arg, ctx):
    if arg:
        await session.switch_model(arg)
    else:
        say.warn("Usage: /mode <# or id>. Type /models to view list.")


@command("/laya", group="Models", help="Show the Laya routing engine details")
async def _cmd_laya(session, arg, ctx):
    perm_tag = f"{UI.GREEN_BOLD}ACTIVE{UI.RST}" if session.use_laya_adaptive_permissions else f"{UI.GRAY}INACTIVE{UI.RST}"
    say.card("Laya System 1 decision engine", [
        ("Backbone", "ModernBERT-large (~421M params, non-autoregressive)"),
        ("Latency", "~33ms"),
        ("Protocol", f"/v1/systemone ({session.laya.endpoint})"),
        ("Adaptive perm", perm_tag),
    ])


@command("/autoshift", "/auto-shift", "/model-shift", "/git-shift", "/gitshift", group="Models", help="Toggle auto-shifting models (fast for git/tools, dense for deep code)")
async def _cmd_autoshift(session, arg, ctx):
    arg_low = arg.lower().strip()
    if arg_low in ("on", "true", "1", "enable"):
        session.auto_shift = True
        session.save_preferences()
        say.ok(f"Auto-shift: ENABLED ({session.fast_git_model} for git/tools, {session.dense_code_model} for deep code)")
    elif arg_low in ("off", "false", "0", "disable"):
        session.auto_shift = False
        session.save_preferences()
        say.ok("Auto-shift: DISABLED (manual model control)")
    elif not arg_low:
        session.auto_shift = not session.auto_shift
        session.save_preferences()
        st = f"ENABLED ({session.fast_git_model} for git, {session.dense_code_model} for deep code)" if session.auto_shift else "DISABLED"
        say.ok(f"Auto-shift: {st}")
    else:
        say.warn("Usage: /autoshift [on|off]")


_cmd_git_shift = _cmd_autoshift


# ── Settings ──────────────────────────────────────────────────────────

@command("/skip-permissions", "/dangerously-skip-permissions", group="Settings", help="Toggle auto-approving every tool call")
async def _cmd_skip_permissions(session, arg, ctx):
    new_val = not session.dangerously_skip_permissions
    session.use_laya_adaptive_permissions = False
    session.set_dangerously_skip_permissions(new_val)
    say.ok(f"Dangerously skip permissions: {'ENABLED' if new_val else 'DISABLED'}")


@command("/permissions", group="Settings", help="Permission mode: auto (Laya), strict, or yolo")
async def _cmd_permissions(session, arg, ctx):
    parts = arg.split() if arg else []
    mode = parts[0].lower() if parts else "auto"
    if mode in ("auto", "laya", "dynamic", "smart"):
        session.permission_mode = "auto"
        session.set_dangerously_skip_permissions(False)
        session.use_laya_adaptive_permissions = True
        session.save_preferences()
        await session.reset_local_agent()
        thresh = getattr(session.laya, "safety_threshold", 0.90) if hasattr(session, "laya") else 0.90
        if RICH_AVAILABLE and console:
            console.print(
                f"[green]🛡️ Permission mode set to 'auto' "
                f"(Laya /noul threshold: P(Safe) >= {thresh:.2f})[/green]"
            )
        else:
            say.ok(f"🛡️ Permission mode set to 'auto' (Laya /noul threshold: P(Safe) >= {thresh:.2f})")
    elif mode in ("on", "skip", "true", "1", "yolo"):
        session.permission_mode = "yolo"
        session.use_laya_adaptive_permissions = False
        session.set_dangerously_skip_permissions(True)
        session.save_preferences()
        thresh = getattr(session.laya, "safety_threshold", 0.90) if hasattr(session, "laya") else 0.90
        if RICH_AVAILABLE and console:
            console.print(
                f"[yellow]🛡️ Permission mode set to 'yolo' "
                f"(Laya /noul threshold: P(Safe) >= {thresh:.2f})[/yellow]"
            )
        else:
            say.ok("Dangerously skip permissions: ENABLED")
    elif mode in ("off", "safe", "prompt", "false", "0", "strict"):
        session.permission_mode = "strict"
        session.use_laya_adaptive_permissions = False
        session.set_dangerously_skip_permissions(False)
        session.save_preferences()
        thresh = getattr(session.laya, "safety_threshold", 0.90) if hasattr(session, "laya") else 0.90
        if RICH_AVAILABLE and console:
            console.print(
                f"[green]🛡️ Permission mode set to 'strict' "
                f"(Laya /noul threshold: P(Safe) >= {thresh:.2f})[/green]"
            )
        else:
            say.ok("Permissions: SAFE MODE")
    else:
        say.warn("Usage: /permissions auto | strict | yolo")


@command("/sudo", group="Settings", help="Authenticate for root, check status, clear, or run a sudo command")
async def _cmd_sudo(session, arg, ctx):
    arg_strip = arg.strip()
    if not arg_strip:
        if SudoManager.is_authenticated():
            rem = int(SudoManager.time_remaining() // 60)
            say.ok(f"Sudo access is ACTIVE (~{rem}m remaining in session cache).")
            say.dim("Tip: '/sudo clear' revokes it, '/sudo <command>' runs a command with root privileges.")
        else:
            SudoManager.prompt_password_entry()
    elif arg_strip in ("status", "check"):
        if SudoManager.is_authenticated():
            rem = int(SudoManager.time_remaining() // 60)
            say.card("Sudo status", [
                ("State", f"{UI.GREEN_BOLD}ACTIVE{UI.RST} (~{rem} minutes remaining)"),
                ("User", os.environ.get("USER", "root")),
                ("Timeout", "15 min"),
                ("Askpass", "Active"),
            ])
        else:
            say.warn("Sudo status: INACTIVE (no cached credentials). Type '/sudo' to authenticate.")
    elif arg_strip in ("clear", "lock", "reset", "revoke"):
        SudoManager.clear_credentials()
        say.ok("Sudo credentials cleared. Root access revoked.")
    else:
        subcmd = arg_strip
        if not subcmd.startswith("sudo "):
            subcmd = f"sudo {subcmd}"
        say.dim(f"⚡ Executing: {subcmd}")
        if not SudoManager.is_authenticated():
            if not SudoManager.prompt_password_entry(reason=subcmd):
                say.warn("Execution aborted: sudo authentication required.")
                return _NO_SAVE
        try:
            proc = await asyncio.to_thread(subprocess.run, subcmd, shell=True, env=os.environ)
            if proc.returncode == 0:
                say.ok("Command completed cleanly (exit code 0).")
            else:
                say.err(f"Command exited with code {proc.returncode}.")
        except Exception as ex:
            say.err(f"Execution failed: {ex}")


@command("/theme", group="Settings", help="Show or switch the color theme (dark, light, mono)")
async def _cmd_theme(session, arg, ctx):
    name = arg.strip().lower()
    if not name:
        say.table("Themes", ["Theme", "Active"], [[t, "yes" if t == UI.theme_name else ""] for t in UI.THEMES])
        say.dim("Use /theme <name> to switch.")
    elif session.set_theme(name):
        say.ok(f"Theme set to {name}.")
    else:
        say.warn(f"Unknown theme '{name}'. Choose: {', '.join(UI.THEMES)}.")


@command("/mcp", group="Settings", help="MCP servers: status, on, off, reload")
async def _cmd_mcp(session, arg, ctx):
    arg_low = arg.lower()
    if arg_low in ("off", "disable"):
        session._mcp_servers = []
        session.enable_mcp = False
        await session.reset_local_agent()
        say.ok("MCP disabled.")
    elif arg_low in ("on", "enable", "reload"):
        session._mcp_servers = load_mcp_servers()
        session.enable_mcp = True
        await session.reset_local_agent()
        names = ", ".join(s.name for s in session._mcp_servers) or "none"
        say.ok(f"MCP reloaded: {len(session._mcp_servers)} server(s): {names}")
    else:
        rows = [("Enabled", f"{UI.GREEN_BOLD}YES{UI.RST}" if session.enable_mcp else f"{UI.RED_BOLD}NO{UI.RST}"), ("Config", MCP_CONFIG_PATH)]
        for s in session._mcp_servers:
            kind = "stdio" if isinstance(s, McpStdioServer) else "http"
            target = s.command if isinstance(s, McpStdioServer) else s.url
            rows.append((f"[{kind}] {s.name}", target))
        if not session._mcp_servers:
            rows.append(("Servers", "(none loaded)"))
        say.card("MCP server status", rows)
        say.dim("Commands: /mcp on | /mcp off | /mcp reload")


@command("/kontexta", "/kxta", group="Settings", help="Kontexta integration: status, on, off, reload, register")
async def _cmd_kontexta(session, arg, ctx):
    from ..context.kontexta import KontextaManager
    arg_low = (arg or "").strip().lower()
    if arg_low in ("off", "disable"):
        session.enable_kontexta = False
        session._mcp_servers = [s for s in session._mcp_servers if s.name not in ("kontexta", "kxta")]
        await session.reset_local_agent()
        session.save_preferences()
        say.ok("Inbuilt Kontexta MCP server DISABLED.")
    elif arg_low in ("on", "enable"):
        session.enable_kontexta = True
        session._mcp_servers = load_mcp_servers(enable_kontexta=True)
        await session.reset_local_agent()
        session.save_preferences()
        say.ok("Inbuilt Kontexta MCP server ENABLED.")
    elif arg_low in ("sync", "reload", "refresh"):
        session._mcp_servers = load_mcp_servers(enable_kontexta=session.enable_kontexta)
        await session.reset_local_agent()
        say.ok("Kontexta MCP integration reloaded.")
    elif arg_low.startswith("register") or arg_low.startswith("add"):
        parts = arg.split(maxsplit=1)
        p_name = parts[1].strip() if len(parts) > 1 else None
        if KontextaManager.register_project(".", name=p_name):
            say.ok(f"Project '{os.path.basename(os.path.abspath('.'))}' registered in Kontexta database!")
        else:
            say.err("Failed to register project in Kontexta (database locked or unreadable).")
    else:
        stat = KontextaManager.get_status(".")
        st_col = UI.GREEN_BOLD if stat["available"] else UI.RED_BOLD
        st_text = "READY / ACTIVE" if stat["available"] else "NOT FOUND / OFFLINE"
        active_proj = stat.get("active_project")
        proj_str = f"{active_proj['name']} [{active_proj['slug']}]" if active_proj else f"{UI.GRAY}(unmapped; type /kontexta register){UI.RST}"
        say.card("Kontexta MCP integration", [
            ("Status", f"{st_col}{st_text}{UI.RST} (enabled: {'YES' if session.enable_kontexta else 'NO'})"),
            ("Entrypoint", str(stat['mcp_entrypoint'])),
            ("Data dir", str(stat['data_dir'])),
            ("Database", f"{stat['db_path']} ({stat['db_size_kb']} KB)"),
            ("Knowledge", f"{stat['files_count']} indexed file(s), {stat['tags_count']} tag(s)"),
            ("Active repo", proj_str),
            ("Projects", f"{stat['projects_count']} registered"),
        ])
        say.dim("Commands: /kontexta on | off | reload | register [name]")


@command("/memory", "/journal", group="Settings", help="Journal brainstorm results and commits to Kontexta (on|off)")
async def _cmd_memory(session, arg, ctx):
    parts = arg.split()
    if len(parts) > 0 and parts[0].lower() in ("on", "off"):
        session.memory_sync_enabled = (parts[0].lower() == "on")
        session.kontexta_sync = session.memory_sync_enabled
    state = "enabled" if getattr(session, "memory_sync_enabled", getattr(session, "kontexta_sync", True)) else "disabled"
    if console and RICH_AVAILABLE:
        console.print(f"[cyan]📓 Kontexta brainstorm/commit journal sync: {state}[/cyan]")
    else:
        say.plain(f"Kontexta brainstorm/commit journal sync: {state}")
    return True


@command("/autocompact", "/auto-compact", group="Settings", help="Auto-compact threshold in turns (on, off, <turns>)")
async def _cmd_autocompact(session, arg, ctx):
    arg_low = arg.lower()
    if arg_low in ("off", "disable", "false", "0"):
        session.autocompact_threshold = 0
        say.ok("Auto-compact DISABLED.")
    elif arg_low in ("on", "enable", "true"):
        session.autocompact_threshold = 20
        say.ok("Auto-compact ENABLED (threshold: 20 turns).")
    elif arg.isdigit():
        session.autocompact_threshold = int(arg)
        say.ok(f"Auto-compact threshold set to {session.autocompact_threshold} turns.")
    else:
        st = f"ENABLED ({session.autocompact_threshold} turns)" if session.autocompact_threshold > 0 else "DISABLED"
        say.card("Auto-compact configuration", [
            ("Status", st),
            ("History", f"{len(session.history)} entries recorded"),
            ("Preserved", f"Keeps {session.max_recent_turns} recent turns verbatim"),
        ])
        say.dim("Commands: /autocompact on | off | <turns>")


@command("/curation", "/continuous", group="Settings", help="Background context curation on the CUDA model (on|off)")
async def _cmd_curation(session, arg, ctx):
    arg_low = arg.lower()
    if arg_low in ("off", "disable", "false", "0"):
        session.continuous_curation = False
        say.ok("Continuous background context curation DISABLED.")
    elif arg_low in ("on", "enable", "true", "1"):
        session.continuous_curation = True
        say.ok("Continuous background context curation ENABLED (auto-distills older turns on RTX 4060).")
    else:
        st = f"{UI.GREEN}ENABLED{UI.RST}" if session.continuous_curation else f"{UI.GRAY}DISABLED{UI.RST}"
        say.card("Continuous curation", [
            ("Status", st),
            ("Engine", "CUDA:9002 (NVIDIA RTX 4060, Granite 8B)"),
            ("Window", f"Preserves last {session.max_recent_turns} turns verbatim"),
            ("Trigger", "Runs in the background after every turn"),
        ])
        say.dim("Commands: /curation on | off")


@command("/repomap", "/ast", group="Settings", help="Repo symbol map: view, on, off, refresh, stats")
async def _cmd_repomap(session, arg, ctx):
    parts = arg.split(maxsplit=1)
    sub = parts[0].lower().strip() if parts else ""

    if sub in ("on", "enable", "true"):
        if hasattr(session, "repomap"):
            session.repomap.enabled = True
        session.auto_repomap = True
        await session.reset_local_agent()
        say.ok("✓ Automatic RepoMap injection enabled.")
    elif sub in ("off", "disable", "false"):
        if hasattr(session, "repomap"):
            session.repomap.enabled = False
        session.auto_repomap = False
        await session.reset_local_agent()
        say.warn("○ Automatic RepoMap injection disabled.")
    elif sub in ("refresh", "reload", "sync"):
        if hasattr(session, "repomap"):
            session.repomap.invalidate_and_refresh()
            scan_ms = session.repomap._last_scan_ms
        else:
            session.refresh_repomap(".")
            scan_ms = 0.0
        await session.reset_local_agent()
        say.ok(f"✓ RepoMap cache rebuilt in {scan_ms:.2f}ms.")
    elif sub == "stats":
        from ..context.repomap import RepoMap
        st = session.repomap.cache_stats() if hasattr(session, "repomap") and hasattr(session.repomap, "cache_stats") else RepoMap.cache_stats()
        repomap_enabled = getattr(session.repomap, "enabled", True) if hasattr(session, "repomap") else session.auto_repomap
        say.card("RepoMap Cache (<1.5ms Two-Tier)", [
            ("Status", f"{UI.GREEN}ENABLED{UI.RST}" if repomap_enabled else f"{UI.GRAY}DISABLED{UI.RST}"),
            ("Root", str(st.get("root", session._repo_root or os.getcwd()))),
            ("Cached Files", f"{st.get('file_count', 0)} files ({st.get('char_count', 0)} chars)"),
            ("Scan Time", f"{st.get('last_scan_ms', 0.0):.2f}ms"),
            ("Engine", "Tree-Sitter + SQLite WAL + RAM"),
        ])
    else:
        # Render interactive AST symbol tree for repo or subpath
        subpath = arg.strip() if arg.strip() else None
        if hasattr(session, "repomap") and hasattr(session.repomap, "build_rich_tree"):
            tree = session.repomap.build_rich_tree(subpath=subpath)
            if RICH_AVAILABLE and console and tree:
                console.print(tree)
            else:
                say.dim(session.repomap.get_cached_outline(subpath=subpath))
        else:
            session.print_repomap(arg or ".")
        repomap_enabled = getattr(session.repomap, "enabled", True) if hasattr(session, "repomap") else session.auto_repomap
        st_tag = "Active (auto-injected)" if repomap_enabled else "Disabled"
        say.dim(f"Auto-injection: {st_tag} | Commands: /repomap [path|on|off|refresh|stats]")


@command("/instructions", "/rules", "/guidelines", group="Settings", help="Project rule files: view, on, off, refresh")
async def _cmd_instructions(session, arg, ctx):
    from ..context.instructions import ProjectInstructions
    arg_low = arg.lower().strip()
    if arg_low in ("on", "enable", "true"):
        session.auto_instructions = True
        await session.reset_local_agent()
        say.ok("Automatic project instructions injection (GEMINI.md / antigravity.md) ENABLED.")
    elif arg_low in ("off", "disable", "false"):
        session.auto_instructions = False
        await session.reset_local_agent()
        say.ok("Automatic project instructions injection DISABLED.")
    elif arg_low in ("refresh", "reload"):
        content, loaded = ProjectInstructions.load_instructions(".", force_refresh=True)
        await session.reset_local_agent()
        say.ok(f"Project instructions refreshed. Loaded: {', '.join(loaded) if loaded else 'None'}")
    elif arg_low in ("view", "show", "cat") or not arg:
        content, loaded = ProjectInstructions.load_instructions(".", force_refresh=False)
        if loaded:
            say.group(f"Project instructions ({', '.join(loaded)})")
            say.plain(content)
        else:
            say.warn("No instruction files found (looking for GEMINI.md, antigravity.md, AGENTS.md, etc.)")
    else:
        say.dim("Usage: /instructions [on|off|refresh|view]")


@command("/hooks", "/hook", group="Settings", help="Lifecycle hooks: view, on, off, refresh")
async def _cmd_hooks(session, arg, ctx):
    from ..context.hooks_loader import ExternalHooksManager
    arg_low = arg.lower().strip()
    if arg_low in ("on", "enable", "true"):
        session.enable_hooks = True
        await session.reset_local_agent()
        say.ok("External lifecycle hooks (hooks.json) ENABLED.")
    elif arg_low in ("off", "disable", "false"):
        session.enable_hooks = False
        await session.reset_local_agent()
        say.ok("External lifecycle hooks (hooks.json) DISABLED.")
    elif arg_low in ("refresh", "reload"):
        entries, sources = ExternalHooksManager.load_hook_configs(".", force_refresh=True)
        await session.reset_local_agent()
        say.ok(f"Lifecycle hooks reloaded. Found {len(entries)} active hook(s) from {len(sources)} file(s).")
    elif arg_low in ("view", "list", "status") or not arg:
        entries, sources = ExternalHooksManager.load_hook_configs(".", force_refresh=False)
        state = "ENABLED" if session.enable_hooks else "DISABLED"
        if sources:
            rows = [("Source", sf) for sf in sources]
            for entry in entries:
                events = [ev for ev in ["PreToolUse", "PostToolUse", "PreTurn", "PostTurn", "PreInvocation", "PostInvocation", "Stop"] if entry["spec"].get(ev)]
                rows.append((entry["name"], ", ".join(events)))
        else:
            rows = [("Hooks", "No hooks.json found in .agents/hooks.json or ~/.gemini/config/hooks.json")]
        say.card(f"Lifecycle hooks ({state})", rows)
        say.dim("Commands: /hooks on | off | refresh | view")
    else:
        say.dim("Usage: /hooks [on|off|refresh|view]")


@command("/markdown", "/md", group="Settings", help="Toggle Markdown rendering of answers")
async def _cmd_markdown(session, arg, ctx):
    arg_low = arg.lower()
    if arg_low in ("on", "enable", "true", "1"):
        session.render_markdown = True
    elif arg_low in ("off", "disable", "false", "0"):
        session.render_markdown = False
    else:
        session.render_markdown = not session.render_markdown
    say.ok(f"Markdown rendering of responses: {'ENABLED' if session.render_markdown else 'DISABLED (raw streaming)'}")


@command("/multiline", "/multi", group="Settings", help="Toggle multi-line input (Enter on an empty line submits)")
async def _cmd_multiline(session, arg, ctx):
    arg_low = arg.lower()
    if arg_low in ("off", "disable", "false", "0", "single"):
        session.multiline_input = False
        say.ok("Multi-line input DISABLED (single Enter submits).")
    elif arg_low in ("on", "enable", "true", "1"):
        session.multiline_input = True
        say.ok("Multi-line input ENABLED (Enter on empty line to submit).")
    else:
        session.multiline_input = not session.multiline_input
        st = "ENABLED (Enter on empty line to submit)" if session.multiline_input else "DISABLED (single Enter submits)"
        say.ok(f"Multi-line input mode: {st}")


@command("/singleline", "/single", group="Settings", help="Single Enter submits")
async def _cmd_singleline(session, arg, ctx):
    session.multiline_input = False
    say.ok("Single-line input mode ENABLED (single Enter submits).")


@command("/format", group="Settings", help="Output format: json or text")
async def _cmd_format(session, arg, ctx):
    arg_low = arg.lower()
    if arg_low in ("json", "js"):
        session.json_output = True
        _sync_say(session)
        say.ok("Structured JSON output mode: ENABLED")
    elif arg_low in ("text", "stream", "plain"):
        session.json_output = False
        _sync_say(session)
        say.ok("Standard text stream output mode: ENABLED")
    else:
        say.warn("Usage: /format json | text")


@command("/prefs", "/preferences", group="Settings", help="Show saved preferences, or '/prefs reset'")
async def _cmd_prefs(session, arg, ctx):
    words = arg.lower().split()
    if words and words[0] == "reset":
        if not await _confirm("Reset preferences", [("Resets", "all saved preferences to defaults")],
                              prompt="Reset to defaults?", assume_yes=("-y" in words or "yes" in words)):
            say.warn("Reset cancelled.")
            return
        reset_prefs()
        session.apply_preferences(PREF_DEFAULTS)
        await session.reset_local_agent()
        say.ok("Preferences reset to defaults.")
    else:
        say.card("Saved preferences (~/.myagy/config.json)", [(k, f"{UI.CYAN}{v}{UI.RST}") for k, v in session.preferences().items()])
        say.dim("Settings save automatically after each command. /prefs reset restores defaults.")


# ── Debug ─────────────────────────────────────────────────────────────

@command("/metrics", group="Debug", help="Toggle the per-turn generation metrics line")
async def _cmd_metrics(session, arg, ctx):
    arg_low = arg.lower()
    if arg_low in ("off", "disable", "false", "0"):
        session.show_metrics = False
        say.ok("Generation metrics display DISABLED.")
    elif arg_low in ("on", "enable", "true", "1"):
        session.show_metrics = True
        say.ok("Generation metrics display ENABLED.")
    else:
        session.show_metrics = not session.show_metrics
        say.ok(f"Generation metrics: {'ENABLED' if session.show_metrics else 'DISABLED'}")


@command("/steps", "/maxsteps", "/max-steps", group="Debug", help="Cap tool steps per turn (<n>, or off)")
async def _cmd_steps(session, arg, ctx):
    arg_low = arg.lower()
    if arg_low in ("off", "disable", "unlimited", "none", "0"):
        session.max_tool_steps_per_turn = 0
        say.ok("Tool step limit DISABLED (unlimited steps per turn; consecutive loop guard remains active).")
    elif arg.isdigit() and int(arg) > 0:
        session.max_tool_steps_per_turn = int(arg)
        say.ok(f"Tool execution step limit set to {session.max_tool_steps_per_turn} steps per turn.")
    else:
        limit_str = f"{session.max_tool_steps_per_turn} steps" if session.max_tool_steps_per_turn > 0 else "UNLIMITED (disabled)"
        say.card("Tool step limit and anti-loop guard", [
            ("Max steps / turn", limit_str),
            ("Loop detection", f"{UI.GREEN_BOLD}ACTIVE{UI.RST} (blocks identical consecutive tool calls)"),
        ])
        say.dim("Commands: /steps off (unlimited) | /steps <number>")


@command("/verbose", "/v", group="Debug", help="Toggle full tool arguments and outputs")
async def _cmd_verbose(session, arg, ctx):
    arg_low = arg.lower()
    if arg_low in ("off", "disable", "false", "0"):
        session.verbose = False
    elif arg_low in ("on", "enable", "true", "1"):
        session.verbose = True
    else:
        session.verbose = not session.verbose
    _sync_say(session)
    say.ok("Verbose mode ENABLED (full tool inputs, outputs and step traces)." if session.verbose else "Verbose mode DISABLED (concise tool status).")


@command("/json", group="Debug", help="Toggle structured JSON output")
async def _cmd_json(session, arg, ctx):
    session.json_output = not session.json_output
    _sync_say(session)
    say.ok(f"Structured JSON output mode: {'ENABLED' if session.json_output else 'DISABLED'}")


@command("/stats", group="Debug", help="Usage ledger per target (/stats reset clears it)")
async def _cmd_stats(session, arg, ctx):
    if arg.lower() == "reset":
        session.stats.clear()
        say.ok("Usage ledger reset.")
    else:
        session.print_stats()


@command("/coordinator", group="Debug", help="CUDA slot arbitration between compactor and subagents")
async def _cmd_coordinator(session, arg, ctx):
    from ..core.coordinator import CudaCoordinator
    c_status = CudaCoordinator.status_summary()
    st_col = UI.GREEN if c_status['state'] == 'idle' else (UI.CUDA_BOLD if c_status['state'] == 'subagent' else UI.AMBER)
    permit = f"{UI.GREEN}ALLOWED (idle){UI.RST}" if c_status['can_compact'] else f"{UI.AMBER}PAUSED (subagent has priority){UI.RST}"
    say.card("NVIDIA CUDA:9002 resource coordinator", [
        ("State", f"{st_col}{c_status['state'].upper()}{UI.RST}"),
        ("Subagents", f"{c_status['active_subagents']} running"),
        ("Compacting", "YES" if c_status['compactor_running'] else "NO"),
        ("Idle permit", permit),
    ])


# ── Session ───────────────────────────────────────────────────────────

@command("/status", group="Session", help="Full status: endpoints, permissions, context, integrations")
async def _cmd_status(session, arg, ctx):
    session.print_status()


@command("/history", group="Session", help="Show the conversation history")
async def _cmd_history(session, arg, ctx):
    if not session.history:
        say.dim("(no conversation history)")
        return
    rows = []
    for idx, t in enumerate(session.history, 1):
        target = t.get("target", "").lower()
        label = {"rocm": "ROCm:9000", "9000": "ROCm:9000", "cuda": "CUDA:9002", "9002": "CUDA:9002", "cloud": "Cloud"}.get(target, target.upper())
        role = f"{UI.WHITE}USER{UI.RST}" if t["role"] == "user" else f"{UI.GREEN}ASSISTANT{UI.RST}"
        preview = t["content"].replace("\n", " ").strip()
        if len(preview) > 120:
            preview = preview[:120] + "..."
        rows.append([str(idx), f"{UI.target_color(target)}{label}{UI.RST}", role, preview])
    say.table(f"Conversation history ({len(session.history)} entries)", ["#", "Target", "Role", "Preview"], rows)


@command("/clear", group="Session", help="Clear the conversation history (asks first)")
async def _cmd_clear(session, arg, ctx):
    if session.history:
        rows = [("Removes", f"{len(session.history)} history entries from this session")]
        if not await _confirm("Clear conversation", rows, prompt="Clear history?", assume_yes=arg.strip().lower() in ("-y", "--yes", "yes")):
            say.warn("Clear cancelled.")
            return
    session.history.clear()
    session._cached_curated_memory = None
    await session.reset_local_agent()
    say.ok("Conversation history cleared.")


@command("/save", group="Session", help="Save the session to a file")
async def _cmd_save(session, arg, ctx):
    say.ok(f"Session saved to {session.save_session(arg or None)}")


@command("/load", "/resume", group="Session", help="Resume a saved session by name/file, or '/resume' to load the latest")
async def _cmd_load(session, arg, ctx):
    target = arg.strip()
    if not target:
        sessions = session.list_sessions()
        if sessions:
            target = sessions[0]["file"]
        else:
            say.warn("Usage: /load <session_name_or_file> (no saved sessions found to resume).")
            return
    if session.load_session(target):
        say.ok(f"Session '{target}' loaded! Restored {len(session.history)//2} turns.")
    else:
        say.err(f"Could not find session '{target}'. Type /sessions to view available sessions.")


@command("/sessions", group="Session", help="List saved sessions")
async def _cmd_sessions(session, arg, ctx):
    sess_list = session.list_sessions()
    if not sess_list:
        say.dim("(No saved sessions found in ~/.gemini/antigravity-cli/sessions/)")
        return
    rows = [[s['file'], f"{s['turns']}", s['saved_at'][:19].replace('T', ' '), f"{s['size_kb']} KB"] for s in sess_list]
    say.table(f"Saved sessions ({len(sess_list)})", ["File", "Turns", "Saved", "Size"], rows)
    say.dim("Use '/load <filename>' to resume a session.")


@command("/export", group="Session", help="Export the transcript as Markdown")
async def _cmd_export(session, arg, ctx):
    say.ok(f"Session transcript exported to {session.export_markdown(arg or None)}")


@command("/summarize", group="Session", help="Summarize the conversation so far")
async def _cmd_summarize(session, arg, ctx):
    if not session.history:
        say.warn("No conversation history to summarize.")
    else:
        say.group("Conversation context summary")
        say.plain(session.summarize_history_sync(session.history).strip())


@command("/context", "/ctx", group="Session", help="Show context window usage per target")
async def _cmd_context(session, arg, ctx):
    if hasattr(session, "get_context_telemetry"):
        from .ui import render_context_telemetry
        telemetry = await session.get_context_telemetry()
        render_context_telemetry(telemetry)
        return True
    session.print_context_summary()
    return True


@command("/compact", group="Session", help="Compact older history now (/compact <turns> keeps that many)")
async def _cmd_compact(session, arg, ctx):
    keep = int(arg) if arg.isdigit() else None
    removed = session.compact_history(keep_recent=keep)
    if removed > 0:
        await session.reset_local_agent()


@command("/last", group="Session", help="Re-show the previous answer in full")
async def _cmd_last(session, arg, ctx):
    text = session.last_answer
    if not text:
        say.warn("No previous answer yet.")
        return
    renderer = StreamRenderer(markdown=session.render_markdown)
    renderer.feed(text)
    renderer.stop()


def _reload_flags(prefs: dict, json_output: bool = False) -> list:
    """CLI flags that reproduce the saved preferences in the restarted process."""
    flags = []
    if prefs.get("cloud_model"):
        flags += ["--cloud-model", prefs["cloud_model"]]
    if prefs.get("active_target") in ("auto", "9000", "9002", "cloud"):
        flags += ["--target", prefs["active_target"]]
    if "dangerously_skip_permissions" in prefs:
        flags.append("-y" if prefs["dangerously_skip_permissions"] else "--safe")
    if prefs.get("verbose"):
        flags.append("--verbose")
    if json_output:
        flags.append("--json")
    if prefs.get("enable_mcp") is False:
        flags.append("--no-mcp")
    if prefs.get("enable_kontexta") is False:
        flags.append("--no-kontexta")
    if prefs.get("multiline_input") is False:
        flags.append("--single-line")
    return flags


@command("/reload", "/restart", group="Session", help="Restart the agent in place, keeping this session")
async def _cmd_reload(session, arg, ctx):
    say.dim("↻ Snapshotting session to /dev/shm and hot-reloading...")
    session.save_preferences()
    snap = session.export_reload_snapshot()
    shm_path = write_reload_snapshot(snap)
    await session.graceful_shutdown_for_reload()
    raise ReloadRequested(shm_path)


@command("/help", group="Session", help="Show this list, or '/help <group>' for one group")
async def _cmd_help(session, arg, ctx):
    wanted = arg.strip().lower()
    groups = help_groups()
    shown = {g: rows for g, rows in groups.items() if not wanted or g.lower() == wanted}
    if not shown:
        say.warn(f"Unknown group '{arg}'. Groups: {', '.join(groups)}.")
        return
    for group_name, rows in shown.items():
        say.group(group_name)
        say.table("", ["Command", "What it does"], [[name, text] for name, text in rows])
    say.dim("!<cmd> runs a shell command | @model <prompt> targets one turn | Esc stops generation | Ctrl+C twice exits")


@command("/exit", "/quit", group="Session", help="Leave the session")
async def _cmd_exit(session, arg, ctx):
    say.dim("Exiting session.")
    return _EXIT


# ── Agents ────────────────────────────────────────────────────────────

@command("/compactor", "/curator", group="Agents", help="CUDA compactor: status or start")
async def _cmd_compactor(session, arg, ctx):
    arg_low = arg.lower()
    if arg_low in ("start", "on", "launch", "wake"):
        say.step("starting the compactor model on port 9002 via LlamaShift")
        res = await asyncio.to_thread(wake_compactor, 9002)
        for warning in res["warnings"]:
            say.warn(warning)
        if res["ok"]:
            say.ok(f"NVIDIA Compactor ready in {res['elapsed']:.1f}s.")
            await asyncio.to_thread(session.refresh_endpoints)
        else:
            say.err(f"Compactor not started: {res['reason']}")
            if res["log_tail"]:
                say.detail(res["log_tail"])
    elif arg_low in ("status", ""):
        await asyncio.to_thread(session.refresh_endpoints)
        ep = session.endpoints.get("9002", {})
        st = f"{UI.GREEN}ONLINE{UI.RST}" if ep.get("status") == "online" else f"{UI.RED}OFFLINE{UI.RST}"
        fn = ep.get("filename") or "granite-4.2-8b-Q4_K_M.gguf"
        say.plain(f"{UI.WHITE}NVIDIA Compactor (RTX 4060 :9002){UI.RST}: {st} ({fn})")
        say.dim("Controls: /compactor start | /compactor status")


def handle_slash_command(cmd: str, session: Any) -> bool:
    """Dispatches background subagent and build tasks on CUDA:9002."""
    mgr = getattr(session, "subagent_mgr", None) or SubagentManager.get_instance()
    if cmd.startswith("/spawn"):
        task_prompt = cmd.removeprefix("/spawn").strip()
        if not task_prompt:
            console.print("[yellow]Usage: /spawn <background task description>[/yellow]")
            return True
        if hasattr(session, "spawn_subagent") and not hasattr(session, "subagent_mgr"):
            session.spawn_subagent(task_prompt)
            return True
        task = mgr.spawn(task_prompt, kind="spawn")
        console.print(
            f"[bold cyan]🛰️ Enqueued Subagent #{task.task_id} on CUDA:9002[/bold cyan] "
            f"[dim](use `/tasks` or `/subagent view {task.task_id}` to inspect)[/dim]"
        )
        return True

    if cmd.startswith("/build"):
        custom_cmd = cmd.removeprefix("/build").strip() or "python3 -m pytest -q"
        task = mgr.spawn_build(custom_cmd)
        console.print(
            f"[bold green]🔨 Enqueued Build & Test Subagent #{task.task_id} on CUDA:9002[/bold green] "
            f"[dim](`{custom_cmd}`)[/dim]"
        )
        return True

    if cmd.strip() == "/tasks":
        mgr.render_tasks_table()
        return True

    if cmd.startswith("/subagent"):
        parts = cmd.split(maxsplit=2)
        if len(parts) < 3:
            console.print("[yellow]Usage: /subagent <view|inject|cancel> <#|all>[/yellow]")
            return True

        action, target_arg = parts[1].lower(), parts[2].strip()
        if action == "view" and target_arg.isdigit():
            mgr.render_task_detail(int(target_arg))
        elif action == "cancel" and target_arg.isdigit():
            ok, msg = mgr.cancel_task(int(target_arg))
            style = "green" if ok else "yellow"
            console.print(f"[{style}]{msg}[/{style}]")
        elif action == "inject":
            count, ids = mgr.inject_into_session(session, target_arg)
            if count:
                console.print(
                    f"[bold magenta]✓ Injected findings from Subagent(s) {ids} into active context.[/bold magenta]"
                )
            else:
                console.print("[yellow]○ No completed un-injected subagent findings matched.[/yellow]")
        return True
    return False


@command("/spawn", "/subagent", "/task", group="Agents", help="Run a background subagent (view, inject, cancel, list)")
async def _cmd_spawn(session, arg, ctx):
    sub_parts = arg.split(maxsplit=1)
    sub_cmd = sub_parts[0].lower() if sub_parts else ""
    sub_arg = sub_parts[1].strip() if len(sub_parts) > 1 else ""

    if sub_cmd == "view":
        if sub_arg.isdigit():
            if hasattr(session, "subagent_mgr"):
                session.subagent_mgr.render_task_detail(int(sub_arg))
            else:
                session.print_subagent_detail(int(sub_arg))
        else:
            say.dim("Usage: /subagent view <#>")
    elif sub_cmd == "inject":
        if sub_arg.lower() in ("all", "*"):
            count = session.inject_all_subagents() if hasattr(session, "inject_all_subagents") else (session.subagent_mgr.inject_into_session(session, "all")[0] if hasattr(session, "subagent_mgr") else 0)
            if count > 0:
                say.ok(f"Successfully injected findings from {count} completed subagent(s) into conversation context.")
            else:
                say.warn("No uninjected completed subagents found to inject.")
        elif sub_arg.isdigit():
            if hasattr(session, "inject_subagent"):
                if session.inject_subagent(int(sub_arg)):
                    say.ok(f"Subagent #{sub_arg} result successfully injected into conversation memory.")
                else:
                    say.warn(f"Could not inject Subagent #{sub_arg}. Ensure it has finished and produced output.")
            elif hasattr(session, "subagent_mgr"):
                c, _ = session.subagent_mgr.inject_into_session(session, sub_arg)
                if c:
                    say.ok(f"Subagent #{sub_arg} result successfully injected into conversation memory.")
                else:
                    say.warn(f"Could not inject Subagent #{sub_arg}. Ensure it has finished and produced output.")
        else:
            say.dim("Usage: /subagent inject <#|all>")
    elif sub_cmd in ("cancel", "kill", "stop"):
        if not sub_arg.isdigit():
            say.dim("Usage: /subagent cancel <#>")
        elif hasattr(session, "cancel_subagent") and session.cancel_subagent(int(sub_arg)):
            say.ok(f"Subagent #{sub_arg} cancelled.")
        elif hasattr(session, "subagent_mgr") and session.subagent_mgr.cancel_task(int(sub_arg))[0]:
            say.ok(f"Subagent #{sub_arg} cancelled.")
        else:
            say.warn(f"Subagent #{sub_arg} is not running or not found in queue.")
    elif sub_cmd == "list" or not arg:
        if hasattr(session, "subagent_mgr"):
            session.subagent_mgr.render_tasks_table()
        else:
            session.print_subagents()
    elif sub_cmd and sub_cmd.isdigit() and not sub_arg:
        if hasattr(session, "subagent_mgr"):
            session.subagent_mgr.render_task_detail(int(sub_cmd))
        else:
            session.print_subagent_detail(int(sub_cmd))
    else:
        if hasattr(session, "subagent_mgr"):
            handle_slash_command(f"/spawn {arg}".strip(), session)
        else:
            session.spawn_subagent(arg)


@command("/build", "/test", group="Agents", help="Run build and tests on the CUDA subagent")
async def _cmd_build(session, arg, ctx):
    handle_slash_command(f"/build {arg}".strip() if arg else "/build", session)


@command("/tasks", "/subagents", group="Agents", help="List subagent tasks")
async def _cmd_tasks(session, arg, ctx):
    handle_slash_command("/tasks", session)


@command("/brainstorm", "/spec", "/plan", group="Agents", help="Spec-driven plan, build, review and commit workflow")
async def _cmd_brainstorm(session, arg, ctx):
    idea = arg.strip()
    if not idea:
        if console and RICH_AVAILABLE:
            console.print("[yellow]Usage: /brainstorm <architectural idea or feature spec>[/yellow]")
        else:
            say.warn("Usage: /brainstorm <architectural idea or feature spec>")
        return True
    from ..agents.brainstorm import AutonomousBrainstormWorkflow
    workflow = AutonomousBrainstormWorkflow(session)
    ctx.task = asyncio.create_task(
        workflow.run(
            idea=idea,
            memory_sync_enabled=getattr(session, "memory_sync_enabled", getattr(session, "kontexta_sync", True)),
        )
    )
    try:
        await ctx.task
    except asyncio.CancelledError:
        if console and RICH_AVAILABLE:
            console.print("[yellow][^C] Brainstorm workflow cancelled.[/yellow]")
        else:
            say.warn("[^C] Brainstorm workflow cancelled.")
    finally:
        ctx.task = None
    return True


@command("/search", "/web", "/google", "/ddg", group="Agents", help="Search the web")
async def _cmd_search(session, arg, ctx):
    from ..tools.web_search import WebSearchEngine
    if not arg:
        say.dim("Usage: /search <query to search online>")
    else:
        say.step(f"Searching the web for '{arg}'...")
        results = await asyncio.to_thread(WebSearchEngine.search, arg, 5)
        WebSearchEngine.print_search_results(arg, results)


@command("/fetch", "/browse", "/readurl", group="Agents", help="Fetch a web page as clean Markdown")
async def _cmd_fetch(session, arg, ctx):
    from ..tools.web_search import WebSearchEngine
    if not arg:
        say.dim("Usage: /fetch <url to inspect>")
    else:
        say.step(f"Fetching and distilling {arg}...")
        content = await asyncio.to_thread(WebSearchEngine.fetch_url, arg, 4000)
        say.group(f"Webpage content ({arg})")
        say.plain(content)


@command("/hermes", group="Agents", help="Run Hermes Agent strictly targeting AMD ROCm (:9000)")
async def _cmd_hermes(session, arg, ctx):
    hermes_bin = os.environ.get("HERMES_BIN") or shutil.which("hermes") or os.path.expanduser("~/.local/bin/hermes")
    if not hermes_bin or not os.path.exists(hermes_bin):
        say.err("Hermes CLI not found. Install or ensure ~/.local/bin/hermes exists.")
        return

    await asyncio.to_thread(session.refresh_endpoints)
    ep = session.endpoints.get("9000", {})
    port = ep.get("port", PORT_ROCM)
    rocm_url = f"http://localhost:{port}/v1"
    is_online = (ep.get("status") == "online")
    model_name = ep.get("name_friendly") or ep.get("filename") or "ROCm Model"

    if not is_online:
        say.warn(f"AMD ROCm (: {port}) is currently OFFLINE.")
        say.dim(f"Use '/switch <model_id>' or start your model server on port {port} before invoking Hermes.")
        return

    arg_clean = arg.strip()
    if not arg_clean or arg_clean in ("status", "info", "-h", "--help"):
        say.card(f"Hermes Agent · AMD ROCm ({port})", [
            ("Target Port", f":{port} (AMD ROCm GPU)"),
            ("Status", f"{UI.GREEN}ONLINE{UI.RST}"),
            ("Loaded Model", f"{UI.ROCM}{model_name}{UI.RST}"),
            ("Endpoint", rocm_url),
            ("Hermes Binary", hermes_bin),
            ("Usage", "/hermes <prompt>"),
            ("Launch TUI", "/hermes tui"),
        ])
        return

    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = rocm_url
    env["OPENAI_API_KEY"] = "local"
    env["HERMES_INFERENCE_MODEL"] = "local"
    env["PYTHONUNBUFFERED"] = "1"

    if arg_clean in ("tui", "--tui", "repl", "interactive"):
        say.step(f"Launching Hermes TUI targeting AMD ROCm ({rocm_url})...")
        try:
            subprocess.run([hermes_bin, "--tui", "--provider", "custom", "-m", "local"], env=env)
        except Exception as e:
            say.err(f"Failed to launch Hermes TUI: {e}")
        return

    say.turn_header(f"Hermes Agent · AMD ROCm ({model_name})", "9000")
    say.step(f"Hermes active on AMD ROCm (: {port})...")

    t0 = time.perf_counter()
    try:
        proc = await asyncio.create_subprocess_exec(
            hermes_bin,
            "chat",
            "--provider", "custom",
            "-m", "local",
            "--yolo",
            "--accept-hooks",
            "-q", arg_clean,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        ctx.task = proc

        response_lines = []
        accumulated_lines = []
        in_response_box = False

        while True:
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
            accumulated_lines.append(line)

            # Filter noisy preambles
            if "Warning: Unknown toolsets" in line:
                continue
            if line.startswith("Query: ") or line == "Initializing agent..." or line.startswith("────"):
                continue

            # Detect and print live tool steps
            if "┊" in line or line.strip().startswith(("$", "💻", "📁", "🔍", "⚙", "preparing", "running")):
                clean_step = line.strip().lstrip("┊").strip()
                say.step(f"hermes: {clean_step}")
                continue

            # Detect Hermes response block
            if "╭─ ⚕ Hermes" in line:
                in_response_box = True
                say.group(f"Hermes ({model_name})")
                continue
            elif "╰─" in line and in_response_box:
                in_response_box = False
                continue

            if in_response_box:
                # Strip leading 4-space box margin
                content = line[4:] if line.startswith("    ") else line
                response_lines.append(content)
                say.plain(content)
                continue

            # Dim post-session summary lines
            if line.startswith(("Resume this session with:", "Session:", "Duration:", "Messages:", "hermes --resume")):
                say.dim(f"  {line.strip()}")
                continue

            # Any other tool or diagnostic line
            if line.strip():
                say.detail(f"  {line.strip()}")

        stderr_bytes = await proc.stderr.read()
        err_text = stderr_bytes.decode("utf-8", errors="replace").strip()
        await proc.wait()
        dur = time.perf_counter() - t0

        out_text = "\n".join(response_lines).strip() or "\n".join(accumulated_lines).strip()

        if proc.returncode != 0 and not out_text:
            say.err(f"Hermes failed (exit code {proc.returncode}): {err_text or 'unknown error'}")
            return

        if session.show_metrics and not session.json_output:
            words = len(out_text.split())
            ctx_badge = session.format_context_badge("9000", reset_col=UI.GRAY)
            say.dim(f"  hermes [ROCm:9000] · {dur:.2f}s · ~{words} words · {ctx_badge}")

        session.history.append({"role": "user", "target": "9000", "content": f"[Hermes Prompt]: {arg_clean}"})
        session.history.append({"role": "assistant", "target": "9000", "content": out_text})
        session.last_answer = out_text
        session._record_stats("9000", "9000", len(out_text.split()), None, None, dur)
    except asyncio.CancelledError:
        try:
            proc.terminate()
        except Exception:
            pass
        say.warn("[^C] Hermes execution cancelled.")
    except Exception as e:
        say.err(f"Hermes execution error: {e}")
    finally:
        ctx.task = None


# ── Git ───────────────────────────────────────────────────────────────

@command("/diff", group="Git", help="Show the file changes from a recent turn (/diff <n>)")
async def _cmd_diff(session, arg, ctx):
    parts = arg.split()
    n = int(parts[0]) if parts and parts[0].isdigit() else 1
    if hasattr(session, "checkpointer") and session.checkpointer.checkpoints:
        diff_output = session.checkpointer.get_diff(nth_latest=n)
        if RICH_AVAILABLE and console:
            console.print(diff_output)
        else:
            print(diff_output)
        return

    text = await asyncio.to_thread(session.turn_diff, n)
    if not text:
        say.warn("No recorded file changes for that turn (needs a git repo and a turn that edited files).")
        return
    stat = await asyncio.to_thread(session.turn_diff, n, True)
    say.dim(stat)
    for line in text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            say.plain(f"{UI.GREEN}{line}{UI.RST}")
        elif line.startswith("-") and not line.startswith("---"):
            say.plain(f"{UI.RED}{line}{UI.RST}")
        elif line.startswith("@@") or line.startswith("diff "):
            say.plain(f"{UI.CYAN}{line}{UI.RST}")
        else:
            say.plain(line)


@command("/undo", group="Git", help="Revert the files changed by the last turn (asks first)")
async def _cmd_undo(session, arg, ctx):
    if hasattr(session, "checkpointer") and session.checkpointer.checkpoints:
        success, message, files = session.checkpointer.undo_latest()
        if success:
            if RICH_AVAILABLE and console:
                console.print(f"[green]✓ {message}[/green]")
                for f in files:
                    console.print(f"  [dim]• Restored: {f}[/dim]")
            else:
                say.ok(message)
                for f in files:
                    say.dim(f"  • Restored: {f}")
        else:
            if RICH_AVAILABLE and console:
                console.print(f"[yellow]○ {message}[/yellow]")
            else:
                say.warn(message)
        return

    plan = await asyncio.to_thread(session.undo_plan)
    if not plan or not plan["files"]:
        say.warn("Nothing to undo: no recorded turn with file changes.")
        return
    cp = plan["checkpoint"]
    verbs = {"A": "delete (created by turn)", "M": "restore", "D": "restore (deleted by turn)"}
    rows = [(st, f"{path}  ->  {verbs.get(st, 'restore')}") for path, st in sorted(plan["files"].items())]
    if await _confirm(f"Revert turn: {cp['prompt']}", rows, prompt="Revert these files?"):
        touched = await asyncio.to_thread(session.apply_undo, plan)
        say.ok(f"Reverted {len(touched)} file(s). The model has been told.")
    else:
        say.warn("Undo cancelled.")


SLASH_COMMANDS[:] = list(COMMANDS) + [c for c in _COMPLETION_ONLY if c not in COMMANDS]


async def interactive_loop(session: MultiGpuHybridSession):
    _sync_say(session)
    set_toolbar(lambda: session.toolbar_fragments())
    if not session.json_output:
        session.print_header()

    set_completion_targets([m['id'] for m in session.get_catalog() if m.get('type') == 'local'])
    loop = asyncio.get_running_loop()
    ctx = _LoopCtx(loop)

    def _sigint_handler():
        ctx.sigint += 1
        if ctx.task and not ctx.task.done():
            say.warn("[^C] Cancelling request...")
            ctx.task.cancel()
        elif ctx.read_fut and not ctx.read_fut.done():
            if ctx.sigint >= 2:
                say.dim("Exiting session.")
                ctx.read_fut.set_exception(_ExitSession())
            else:
                say.dim("[^C] Press Ctrl+C again to exit.")

    loop.add_signal_handler(signal.SIGINT, _sigint_handler)

    try:
        while True:
            # Queue-ahead typing: if the user hit Enter during the previous turn, auto-send it
            if ctx.queued_submit and ctx.queued_input:
                user_input = ctx.queued_input
                ctx.queued_input = ""
                ctx.queued_submit = False
                if not session.json_output:
                    say.dim(f"» (sending queued input: {user_input!r})")
            else:
                try:
                    if not session.json_output and not _toolbar_active():
                        say.dim(_plain_pill(session))
                    prefill = ctx.queued_input
                    ctx.queued_input = ""
                    ctx.queued_submit = False
                    if prefill:
                        ctx.read_fut = _spawn_reader(loop, read_input_prompt, _prompt_string(session), session.multiline_input, prefill)
                    else:
                        ctx.read_fut = _spawn_reader(loop, read_input_prompt, _prompt_string(session), session.multiline_input)
                    user_input = await ctx.read_fut
                except PromptInterrupted:
                    ctx.sigint += 1
                    if ctx.sigint >= 2:
                        say.dim("Exiting session.")
                        break
                    say.dim("[^C] Press Ctrl+C again to exit.")
                    continue
                except (EOFError, SystemExit, _ExitSession):
                    break
                finally:
                    ctx.read_fut = None
            ctx.sigint = 0
            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q", "/exit", "/quit"):
                say.dim("Exiting session.")
                break

            # Multi-line block detection (inline triple-quotes)
            if user_input.startswith(('"""', "'''")):
                quote_type = user_input[:3]
                remainder = user_input[3:]
                if quote_type in remainder:
                    user_input = remainder.split(quote_type, 1)[0].strip()
                else:
                    collected = [remainder]
                    say.dim(f"[Multi-line block mode: enter text, type {quote_type} on an empty line to finish]")
                    q_prompt = f"{UI.DARK_GRAY}  │{UI.RST} " if UI._use_color else "... "
                    while True:
                        try:
                            line = await asyncio.to_thread(input, q_prompt)
                            if quote_type in line:
                                collected.append(line.split(quote_type, 1)[0])
                                break
                            collected.append(line)
                        except (EOFError, KeyboardInterrupt):
                            break
                    user_input = "\n".join(collected).strip()
                    if not user_input:
                        continue

            # Multi-line paste mode
            if user_input.lower().startswith("/paste"):
                say.card("Paste mode", [
                    ("Paste", "Paste code or text below."),
                    ("Submit", "Type EOF, . or END on an empty line (or Ctrl+D)."),
                ])
                paste_lines = []
                while True:
                    try:
                        line = await asyncio.to_thread(input)
                        if line.strip() in ("EOF", ".", "END", "```"):
                            break
                        paste_lines.append(line)
                    except (EOFError, KeyboardInterrupt):
                        break
                pasted_text = "\n".join(paste_lines).strip()
                if not pasted_text:
                    say.warn("No input provided in paste mode.")
                    continue
                user_input = pasted_text

            # Direct shell command execution with '!' prefix
            if user_input.startswith("!"):
                raw_cmd = user_input[1:].strip()
                if not raw_cmd:
                    say.warn("Usage: !<shell command>")
                    continue

                is_sudo = SudoManager.is_sudo_command(raw_cmd)
                if is_sudo and not SudoManager.is_authenticated():
                    if not SudoManager.prompt_password_entry(reason=raw_cmd):
                        say.warn("Execution aborted: sudo authentication required.")
                        continue

                try:
                    say.dim(f"$ {raw_cmd}")
                    proc = await asyncio.to_thread(subprocess.run, raw_cmd, shell=True, env=os.environ)
                    if proc.returncode != 0:
                        say.err(f"Exit {proc.returncode}")
                except Exception as ex:
                    say.err(f"Shell execution failed: {ex}")
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""
                handler = COMMANDS.get(cmd)
                if handler is None:
                    say.warn(f"Unknown command '{cmd}'. Type /help for available commands.")
                else:
                    try:
                        result = await handler(session, arg, ctx)
                    except ReloadRequested:
                        raise
                    except Exception as exc:
                        say.err(f"{cmd} failed: {type(exc).__name__}: {exc}")
                        say.dim(traceback.format_exc())
                        continue
                    if result is _EXIT:
                        break
                    if result is _NO_SAVE:
                        continue
                session.save_preferences()
                continue

            # Handle one-shot @ prefixes
            force_target = None
            prompt = user_input
            lower_input = user_input.lower()
            if lower_input.startswith(("@9000 ", "@rocm ", "@amd ")):
                force_target, prompt = "9000", user_input.split(maxsplit=1)[1].strip()
            elif lower_input.startswith(("@9001 ", "@9002 ", "@cuda ", "@nvidia ")):
                t_key = "9001" if lower_input.startswith("@9001 ") else "9002"
                force_target, prompt = session.resolve_target_alias(t_key), user_input.split(maxsplit=1)[1].strip()
            elif lower_input.startswith(("@cloud ", "@gemini ", "@oauth ")):
                force_target, prompt = "cloud", user_input.split(maxsplit=1)[1].strip()
            elif lower_input.startswith(("@auto ", "@laya ")):
                force_target, prompt = "auto", user_input.split(maxsplit=1)[1].strip()
            elif lower_input.startswith("@local "):
                force_target = session.resolve_target_alias("local")
                prompt = user_input.split(maxsplit=1)[1].strip()
            elif lower_input.startswith("@") and " " in user_input:
                first_tok, rest_prompt = user_input.split(maxsplit=1)
                prefix_query = first_tok[1:].strip()
                cat = session.get_catalog()
                matched = None
                if prefix_query.isdigit():
                    pidx = int(prefix_query)
                    if 1 <= pidx <= len(cat):
                        matched = cat[pidx - 1]
                if not matched:
                    for m in cat:
                        if m["id"].lower() == prefix_query.lower():
                            matched = m
                            break
                if not matched:
                    partials = [m for m in cat if prefix_query.lower() in m["id"].lower()]
                    if len(partials) == 1:
                        matched = partials[0]

                if matched:
                    if matched["type"] == "local" and matched["status"] != "running":
                        ok = await session.switch_model(matched["id"])
                        if not ok:
                            continue
                        force_target = str(matched.get("port", "9000"))
                    elif matched["type"] == "local":
                        force_target = str(matched.get("port", "9000"))
                    else:
                        force_target = matched["id"]
                    prompt = rest_prompt.strip()
                else:
                    say.warn(f"Unknown model target '@{prefix_query}'. Type /models to list targets; nothing was sent.")
                    continue

            # Run chat as a cancellable task with ESC-key interrupt support
            ctx.task = asyncio.create_task(session.chat(prompt, force_target=force_target))
            esc_listener = EscListener(
                on_escape=lambda: loop.call_soon_threadsafe(
                    lambda: ctx.task.cancel() if ctx.task and not ctx.task.done() else None
                ),
                session=session,
            )
            turn_ok = False
            try:
                with esc_listener:
                    await ctx.task
                turn_ok = True
            except asyncio.CancelledError:
                if session.json_output:
                    print(json.dumps({"status": "cancelled", "interrupted_by": "esc" if esc_listener.interrupted else "sigint", "prompt": prompt}, indent=2))
                elif esc_listener.interrupted:
                    say.warn("[Esc] Interrupted by user. Resuming prompt.")
                else:
                    say.warn("[Cancelled] Request stopped. Resuming prompt.")
                await session.reset_local_agent()
            except Exception as e:
                err = str(e)
                if "CancelledError" in err or "cancelled" in err.lower() or esc_listener.interrupted:
                    if session.json_output:
                        print(json.dumps({"status": "cancelled", "interrupted_by": "esc" if esc_listener.interrupted else "exception", "prompt": prompt}, indent=2))
                    elif esc_listener.interrupted:
                        say.warn("[Esc] Interrupted by user. Resuming prompt.")
                    else:
                        say.warn("[Cancelled] Request stopped.")
                    await session.reset_local_agent()
                elif session.json_output:
                    print(json.dumps({"status": "error", "error": err, "prompt": prompt}, indent=2))
                else:
                    say.err(f"Error during turn: {e}")
                await session.reset_local_agent()
            finally:
                ctx.task = None
                # Pick up anything typed during the turn; discard on abort/error
                queued_text, queued_submit = esc_listener.drain()
                if turn_ok and queued_text:
                    ctx.queued_input = queued_text
                    ctx.queued_submit = queued_submit

    finally:
        try:
            loop.remove_signal_handler(signal.SIGINT)
        except Exception:
            pass


async def main():
    parser = argparse.ArgumentParser(description="Antigravity Multi-GPU Hybrid Runner with Laya Decision Engine")
    parser.add_argument(
        "--target",
        "--mode",
        choices=["auto", "9000", "9002", "rocm", "cuda", "cloud"],
        default=None,
        help="Initial active model target (default: saved preference, else 'rocm'/port 9000)",
    )
    parser.add_argument(
        "--auto",
        action="store_const",
        dest="target",
        const="auto",
        help="Enable Laya System 1 auto-routing (default)",
    )
    parser.add_argument(
        "--rocm",
        action="store_const",
        dest="target",
        const="9000",
        help="Alias to start with AMD ROCm GPU (port 9000)",
    )
    parser.add_argument(
        "--cuda",
        action="store_const",
        dest="target",
        const="9002",
        help="Alias to start with NVIDIA CUDA GPU (port 9002)",
    )
    parser.add_argument(
        "--dangerously-skip-permissions",
        "--skip-permissions",
        "-y",
        "--yes",
        dest="dangerously_skip_permissions",
        action="store_true",
        default=None,
        help="Auto-approve all tool permission requests without prompting",
    )
    parser.add_argument(
        "--safe",
        dest="dangerously_skip_permissions",
        action="store_false",
        default=None,
        help="Start in safe permission mode (prompts for bash commands)",
    )
    parser.add_argument(
        "--cloud-model",
        type=str,
        default=DEFAULT_CLOUD_MODEL,
        help="Cloud Gemini model identifier",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to execute directly",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Force interactive mode",
    )
    parser.add_argument(
        "--no-mcp",
        dest="enable_mcp",
        action="store_false",
        default=True,
        help="Disable MCP server integration at startup (re-enable with /mcp on)",
    )
    parser.add_argument(
        "--no-kontexta",
        dest="enable_kontexta",
        action="store_false",
        default=None,
        help="Disable Inbuilt Kontexta MCP server integration at startup",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        default=False,
        help="Output response in structured JSON format (ideal for scripts and pipelines)",
    )
    parser.add_argument(
        "--single-line",
        "--singleline",
        dest="multiline_input",
        action="store_false",
        default=None,
        help="Disable multi-line input by default (require single Enter to submit)",
    )
    parser.add_argument(
        "--load",
        dest="load_session",
        type=str,
        default=None,
        help="Restore a saved session file by name or path upon startup",
    )
    parser.add_argument(
        "--resume",
        dest="resume_latest",
        action="store_true",
        default=False,
        help="Automatically resume the most recent saved session",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose mode (detailed tool inputs, output previews, step numbers)",
    )
    parser.add_argument(
        "--restore-shm",
        type=str,
        default=None,
        help=argparse.SUPPRESS,  # Internal flag used by /reload os.execv handoff
    )
    parser.add_argument(
        "--sdk-path",
        type=str,
        default=None,
        help="Explicit directory path to google-antigravity Python SDK",
    )

    args = parser.parse_args()

    if getattr(args, "sdk_path", None):
        setup_antigravity_sdk(args.sdk_path)

    # Precedence: explicit CLI flag > saved preference > built-in default
    prefs = load_prefs()
    session = MultiGpuHybridSession(
        initial_target=args.target or prefs.get("active_target", "rocm"),
        cloud_model=args.cloud_model if args.cloud_model != DEFAULT_CLOUD_MODEL else prefs.get("cloud_model", DEFAULT_CLOUD_MODEL),
        dangerously_skip_permissions=args.dangerously_skip_permissions if args.dangerously_skip_permissions is not None else prefs.get("dangerously_skip_permissions", True),
        use_laya_adaptive_permissions=prefs.get("use_laya_adaptive_permissions", True),
        enable_mcp=args.enable_mcp,
        enable_kontexta=args.enable_kontexta if args.enable_kontexta is not None else prefs.get("enable_kontexta", True),
        json_output=args.json_output,
        multiline_input=args.multiline_input if args.multiline_input is not None else prefs.get("multiline_input", True),
        verbose=args.verbose or prefs.get("verbose", False),
    )
    handled = {"active_target", "cloud_model", "dangerously_skip_permissions", "use_laya_adaptive_permissions", "enable_kontexta", "multiline_input", "verbose"}
    session.apply_preferences({k: v for k, v in prefs.items() if k not in handled})
    _sync_say(session)

    if getattr(args, "restore_shm", None):
        snap = consume_reload_snapshot(args.restore_shm)
        if snap is not None:
            session.restore_from_snapshot(snap)
            elapsed_ms = (time.time() - snap.timestamp) * 1000.0
            turns_cnt = len(snap.chat_history) // 2
            say.card("Hot reload", [
                ("Status", f"Restored in {elapsed_ms:.0f}ms"),
                ("Target", session.active_target.upper()),
                ("History", f"{turns_cnt} turn(s)"),
                ("Checkpoints", f"{len(snap.checkpoint_trees)}"),
            ])
    elif args.load_session:
        session.load_session(args.load_session)
    elif args.resume_latest:
        sessions = session.list_sessions()
        if sessions:
            session.load_session(sessions[0]["file"])

    try:
        if args.prompt:
            await session.chat(args.prompt)
        elif not sys.stdin.isatty() and not args.interactive:
            prompt = sys.stdin.read().strip()
            if prompt:
                await session.chat(prompt)
            else:
                await interactive_loop(session)
        else:
            await interactive_loop(session)
    except ReloadRequested as req:
        return req.shm_path
    finally:
        await session.close()
    return None


if __name__ == "__main__":
    _shm = asyncio.run(main())
    if _shm is not None:
        perform_execv_reload(_shm)
