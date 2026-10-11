"""Interactive command-line interface, multi-line prompt reader, and main runner."""

import os
import sys
import json
import signal
import select
import asyncio
import argparse
import subprocess
from typing import Optional

try:
    from rich.table import Table
    from rich.text import Text
    from rich import box
except ImportError:
    pass

from ..config import (
    MCP_CONFIG_PATH,
    DEFAULT_CLOUD_MODEL,
    HISTORY_PATH,
)
from .ui import UI, RICH_AVAILABLE, console
from ..core.prefs import PREF_DEFAULTS, load_prefs, reset_prefs
from ..core.llamashift import trigger_llamashift_switch
from ..context.mcp_loader import load_mcp_servers, McpStdioServer
from ..core.session import MultiGpuHybridSession
from ..core.sudo_manager import SudoManager
from .esc_listener import EscListener, prompt_input


SLASH_COMMANDS = [
    "/auto", "/9000", "/rocm", "/9001", "/cuda", "/cloud", "/gemini", "/models", "/model", "/mode",
    "/spawn", "/tasks", "/subagent", "/repomap", "/instructions", "/hooks", "/kontexta", "/kxta", "/mcp", "/compact",
    "/autocompact", "/compactor", "/curation", "/context", "/ctx", "/save", "/load", "/sessions",
    "/export", "/summarize", "/history", "/clear", "/status", "/metrics", "/verbose", "/steps",
    "/permissions", "/sudo", "/laya", "/json", "/format", "/multiline", "/singleline", "/paste", "/reload",
    "/brainstorm", "/search", "/fetch", "/diff", "/undo", "/stats", "/prefs", "/markdown", "/help",
    "/exit", "/quit",
]
AT_TARGETS = ["@rocm", "@cuda", "@cloud", "@auto", "@local", "@9000", "@9001"]
_dynamic_targets: list = []
_pt_session = None


def set_completion_targets(ids) -> None:
    """Registers model ids for @<id> completion."""
    global _dynamic_targets
    _dynamic_targets = [f"@{i}" for i in ids if i]


def _get_pt_session():
    """Lazily builds a prompt_toolkit session (history + completion); None if unavailable."""
    global _pt_session
    if _pt_session is not None:
        return _pt_session or None
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.completion import Completer, Completion

        class _InputCompleter(Completer):
            def get_completions(self, document, complete_event):
                text = document.text_before_cursor
                if " " in text or "\n" in text:
                    return
                if text.startswith("/"):
                    pool = SLASH_COMMANDS
                elif text.startswith("@"):
                    pool = AT_TARGETS + _dynamic_targets
                else:
                    return
                for cand in pool:
                    if cand.lower().startswith(text.lower()):
                        yield Completion(cand, start_position=-len(text))

        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        _pt_session = PromptSession(history=FileHistory(HISTORY_PATH), completer=_InputCompleter())
    except Exception:
        _pt_session = False
    return _pt_session or None


def _read_line(prompt: str) -> str:
    """One line of input with history/completion on a TTY, plain input() otherwise."""
    if sys.stdin.isatty() and sys.stdout.isatty():
        pt = _get_pt_session()
        if pt is not None:
            from prompt_toolkit.formatted_text import ANSI
            return pt.prompt(ANSI(prompt))
    return input(prompt)


def read_input_prompt(first_prompt: str, multiline: bool = True) -> str:
    """
    Reads user input with multi-line support enabled by default.
    - Slash commands (/help, /status, etc.) and exit words submit immediately on first line.
    - Multi-line mode: subsequent lines are prompted with '... '.
    - Pressing Enter on an empty line or pressing Ctrl+D submits the multi-line input.
    - Pasted text with internal blank lines is preserved without prematurely submitting.
    """
    try:
        first_line = _read_line(first_prompt)
    except (EOFError, KeyboardInterrupt):
        return ""

    stripped = first_line.strip()
    if not multiline:
        return stripped

    # A multi-line paste arrives as one buffer when prompt_toolkit is active
    if "\n" in first_line:
        return stripped

    # Slash commands and session exit keywords submit immediately on first Enter
    if stripped.startswith("/") or stripped.lower() in ("exit", "quit", "q", "/exit", "/quit"):
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
            print(f"\n{UI.GRAY}[^C] Input cancelled.{UI.RST}", flush=True)
            return ""

    return "\n".join(lines).strip()


async def interactive_loop(session: MultiGpuHybridSession):
    if not session.json_output:
        if UI._use_color:
            print(f"\n{UI.DARK_GRAY}╭──────────────────────────────────────────────────────────────────────────╮{UI.RST}")
            print(f"{UI.DARK_GRAY}│{UI.RST}  {UI.WHITE}Antigravity Multi-GPU Hybrid Session{UI.RST}  {UI.ROCM}[ROCm]{UI.RST} {UI.DARK_GRAY}+{UI.RST} {UI.CUDA}[CUDA]{UI.RST} {UI.DARK_GRAY}+{UI.RST} {UI.CLOUD}[Cloud]{UI.RST}         {UI.DARK_GRAY}│{UI.RST}")
            print(f"{UI.DARK_GRAY}│{UI.RST}  {UI.GRAY}Powered by Laya System 1 (ModernBERT) & Google Antigravity OAuth{UI.RST}       {UI.DARK_GRAY}│{UI.RST}")
            print(f"{UI.DARK_GRAY}│{UI.RST}  {UI.DIM}Multi-line input enabled (Enter on empty line to submit │ Esc stops agent){UI.RST} {UI.DARK_GRAY}│{UI.RST}")
            print(f"{UI.DARK_GRAY}╰──────────────────────────────────────────────────────────────────────────╯{UI.RST}")
        else:
            print("=" * 65)
            print(" Antigravity Multi-GPU Hybrid Session (ROCm + CUDA + Cloud)")
            print(" Powered by Laya System 1 Decision Model & Google OAuth")
            print(" Multi-line input active (Enter on empty line to submit)")
            print(" Press Esc to immediately stop agent generation. Type '/help' for info.")
            print("=" * 65)
        session.print_status()

    set_completion_targets([m['id'] for m in session.get_catalog() if m.get('type') == 'local'])
    loop = asyncio.get_event_loop()
    _current_task: Optional[asyncio.Task] = None
    _sigint_count = 0

    def _sigint_handler():
        nonlocal _sigint_count, _current_task
        _sigint_count += 1
        if _current_task and not _current_task.done():
            # First Ctrl+C: cancel the running request, stay in session
            print(f"\n{UI.AMBER_BOLD}[^C] Cancelling request... (Ctrl+C again to exit){UI.RST}", flush=True)
            _current_task.cancel()
        else:
            # Ctrl+C at the prompt or second press: exit
            print(f"\n{UI.GRAY}Exiting session.{UI.RST}", flush=True)
            raise SystemExit(0)

    loop.add_signal_handler(signal.SIGINT, _sigint_handler)

    try:
        while True:
            _sigint_count = 0

            # Build prompt label and badge with live context size pill
            ctx_pill = session.format_context_pill(session.active_target)
            if session.active_target == "auto":
                target_label = "AUTO:Laya"
                target_badge = f"{UI.LAYA_BOLD}[⚡ AUTO:Laya │ {ctx_pill}{UI.LAYA_BOLD}]{UI.RST}"
                arrow_col = UI.LAYA
            elif session.active_target == "9000":
                target_label = "ROCm:9000"
                target_badge = f"{UI.ROCM_BOLD}[⚡ ROCm:9000 │ {ctx_pill}{UI.ROCM_BOLD}]{UI.RST}"
                arrow_col = UI.ROCM
            elif session.active_target == "9001":
                target_label = "CUDA:9001"
                target_badge = f"{UI.CUDA_BOLD}[⚡ CUDA:9001 │ {ctx_pill}{UI.CUDA_BOLD}]{UI.RST}"
                arrow_col = UI.CUDA
            elif session.active_target == "cloud":
                target_label = "CLOUD:OAuth"
                target_badge = f"{UI.CLOUD_BOLD}[☁ CLOUD:OAuth │ {ctx_pill}{UI.CLOUD_BOLD}]{UI.RST}"
                arrow_col = UI.CLOUD
            else:
                target_label = str(session.active_target)
                target_badge = f"{UI.GRAY}[{session.active_target} │ {ctx_pill}{UI.GRAY}]{UI.RST}"
                arrow_col = UI.GRAY

            try:
                if UI._use_color:
                    prompt_str = f"{target_badge} {UI.WHITE}You{UI.RST} {arrow_col}❯{UI.RST} "
                else:
                    raw_ctx = session.get_context_metrics(session.active_target)
                    t_val = raw_ctx["tokens"]
                    t_str = f"{t_val/1000:.1f}k" if t_val >= 1000 else str(t_val)
                    prompt_str = f"[{target_label} | {t_str} ctx] You > "
                user_input = await asyncio.to_thread(read_input_prompt, prompt_str, session.multiline_input)
            except (EOFError, SystemExit):
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q", "/exit", "/quit"):
                print(f"{UI.GRAY}Exiting session.{UI.RST}")
                break

            # Multi-line block detection (inline triple-quotes)
            if user_input.startswith(('"""', "'''")):
                quote_type = user_input[:3]
                remainder = user_input[3:]
                if quote_type in remainder:
                    user_input = remainder.split(quote_type, 1)[0].strip()
                else:
                    collected = [remainder]
                    print(f"{UI.GRAY}[Multi-line block mode — enter text, type {quote_type} on an empty line to finish]:{UI.RST}")
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
                print(f"\n{UI.DARK_GRAY}╭─── {UI.WHITE}PASTE MODE{UI.RST}{UI.DARK_GRAY} ──────────────────────────────────────────────╮{UI.RST}")
                print(f"{UI.DARK_GRAY}│{UI.RST}  Paste code or text below.                                   {UI.DARK_GRAY}│{UI.RST}")
                print(f"{UI.DARK_GRAY}│{UI.RST}  Type {UI.WHITE}EOF{UI.RST}, {UI.WHITE}.{UI.RST}, or {UI.WHITE}END{UI.RST} on an empty line (or Ctrl+D) to submit.   {UI.DARK_GRAY}│{UI.RST}")
                print(f"{UI.DARK_GRAY}╰─────────────────────────────────────────────────────────────╯{UI.RST}\n")
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
                    print(UI.warn("No input provided in paste mode."))
                    continue
                user_input = pasted_text

            # Direct shell command execution with '!' prefix
            if user_input.startswith("!"):
                raw_cmd = user_input[1:].strip()
                if not raw_cmd:
                    print(UI.warn("Usage: !<shell command>"))
                    continue

                is_sudo = SudoManager.is_sudo_command(raw_cmd)
                if is_sudo and not SudoManager.is_authenticated():
                    if not SudoManager.prompt_password_entry(reason=raw_cmd):
                        print(UI.warn("Execution aborted: sudo authentication required."))
                        continue

                try:
                    print(f"{UI.GRAY}$ {raw_cmd}{UI.RST}\n")
                    proc = await asyncio.to_thread(subprocess.run, raw_cmd, shell=True, env=os.environ)
                    if proc.returncode != 0:
                        print(f"\n{UI.RED_BOLD}[Exit {proc.returncode}]{UI.RST}\n")
                    else:
                        print()
                except Exception as ex:
                    print(UI.err(f"Shell execution failed: {ex}"))
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""

                if cmd == "/auto":
                    session.active_target = "auto"
                    print(UI.ok("Default target set to AUTO (Laya System 1 Non-Autoregressive Routing in ~33ms)"))
                elif cmd in ("/9000", "/rocm", "/amd"):
                    session.active_target = "9000"
                    await asyncio.to_thread(session.refresh_endpoints)
                    fn = session.endpoints['9000']['filename']
                    print(UI.ok(f"Default target set to AMD ROCm Port 9000 ({fn})"))
                elif cmd in ("/9001", "/cuda", "/nvidia"):
                    session.active_target = "9001"
                    await asyncio.to_thread(session.refresh_endpoints)
                    fn = session.endpoints['9001']['filename']
                    print(UI.ok(f"Default target set to NVIDIA CUDA Port 9001 ({fn})"))
                elif cmd in ("/cloud", "/gemini", "/oauth"):
                    if arg:
                        # /cloud <model-id> — change cloud model mid-session and persist
                        new_model = arg.strip()
                        session.cloud_model = new_model
                        session.active_target = "cloud"
                        session.save_preferences()
                        print(UI.ok(f"Cloud model set to {UI.CLOUD}{new_model}{UI.RST} and target switched to CLOUD (saved to prefs)."))
                    else:
                        session.active_target = "cloud"
                        print(UI.ok(f"Default target set to CLOUD via OAuth ({session.cloud_model})"))
                elif cmd in ("/model", "/switch", "/shift", "/use"):
                    if not arg:
                        session.print_models()
                    else:
                        await session.switch_model(arg)
                elif cmd == "/mode":
                    if arg:
                        await session.switch_model(arg)
                    else:
                        print(UI.warn("Usage: /mode <# or id>. Type /models to view list."))
                elif cmd in ("/skip-permissions", "/dangerously-skip-permissions"):
                    new_val = not session.dangerously_skip_permissions
                    session.use_laya_adaptive_permissions = False
                    session.set_dangerously_skip_permissions(new_val)
                    status_text = "ENABLED" if new_val else "DISABLED"
                    print(UI.ok(f"Dangerously skip permissions: {status_text}"))
                elif cmd == "/permissions":
                    arg_low = arg.lower()
                    if arg_low in ("auto", "laya", "dynamic", "smart"):
                        session.use_laya_adaptive_permissions = True
                        session._active_local_agent = None
                        print(UI.ok("Permissions: LAYA ADAPTIVE"))
                    elif arg_low in ("on", "skip", "true", "1"):
                        session.use_laya_adaptive_permissions = False
                        session.set_dangerously_skip_permissions(True)
                        print(UI.ok("Dangerously skip permissions: ENABLED"))
                    elif arg_low in ("off", "safe", "prompt", "false", "0"):
                        session.use_laya_adaptive_permissions = False
                        session.set_dangerously_skip_permissions(False)
                        print(UI.ok("Permissions: SAFE MODE"))
                    else:
                        print(UI.warn("Usage: /permissions auto | skip | safe"))
                elif cmd == "/sudo":
                    arg_strip = arg.strip()
                    if not arg_strip:
                        if SudoManager.is_authenticated():
                            rem = int(SudoManager.time_remaining() // 60)
                            print(f"\n{UI.GREEN_BOLD}[✓] Sudo access is currently ACTIVE (~{rem}m remaining in session cache).{UI.RST}")
                            print(f"{UI.DARK_GRAY}Tip: Use '/sudo clear' to revoke or '/sudo <command>' to execute with root privileges.{UI.RST}\n")
                        else:
                            SudoManager.prompt_password_entry()
                    elif arg_strip in ("status", "check"):
                        if SudoManager.is_authenticated():
                            rem = int(SudoManager.time_remaining() // 60)
                            print(f"\n{UI.GREEN_BOLD}[✓] Sudo Status: ACTIVE (~{rem} minutes remaining in session){UI.RST}")
                            print(f"{UI.DARK_GRAY}User: {os.environ.get('USER', 'root')} │ Timeout: 15 min │ Askpass: Active{UI.RST}\n")
                        else:
                            print(f"\n{UI.AMBER_BOLD}[○] Sudo Status: INACTIVE (No cached credentials){UI.RST}")
                            print(f"{UI.DARK_GRAY}Type '/sudo' to authenticate and unlock root access.{UI.RST}\n")
                    elif arg_strip in ("clear", "lock", "reset", "revoke"):
                        SudoManager.clear_credentials()
                        print(f"\n{UI.ok('Sudo credentials cleared. Root access revoked.')}\n")
                    else:
                        subcmd = arg_strip
                        if not subcmd.startswith("sudo "):
                            subcmd = f"sudo {subcmd}"
                        print(f"{UI.GRAY}⚡ Executing: {UI.WHITE}{subcmd}{UI.RST}\n")
                        if not SudoManager.is_authenticated():
                            if not SudoManager.prompt_password_entry(reason=subcmd):
                                print(UI.warn("Execution aborted: sudo authentication required."))
                                continue
                        try:
                            proc = await asyncio.to_thread(subprocess.run, subcmd, shell=True, env=os.environ)
                            if proc.returncode == 0:
                                print(f"\n{UI.GREEN_BOLD}[✓] Command completed cleanly (exit code 0).{UI.RST}\n")
                            else:
                                print(f"\n{UI.RED_BOLD}[x] Command exited with code {proc.returncode}.{UI.RST}\n")
                        except Exception as ex:
                            print(UI.err(f"Execution failed: {ex}"))
                elif cmd in ("/git-shift", "/gitshift", "/autoshift"):
                    arg_low = arg.lower().strip()
                    if arg_low in ("on", "true", "1", "enable"):
                        session.auto_git_shift = True
                        print(UI.ok(f"Git Auto-Shift: ENABLED ({session.fast_git_model} for git/tools ↔ {session.dense_code_model} for deep code)"))
                    elif arg_low in ("off", "false", "0", "disable"):
                        session.auto_git_shift = False
                        print(UI.ok("Git Auto-Shift: DISABLED (manual model control)"))
                    elif not arg_low:
                        session.auto_git_shift = not session.auto_git_shift
                        st = f"ENABLED ({session.fast_git_model} for git ↔ {session.dense_code_model} for deep code)" if session.auto_git_shift else "DISABLED"
                        print(UI.ok(f"Git Auto-Shift: {st}"))
                    else:
                        print(UI.warn("Usage: /git-shift [on|off]"))
                elif cmd == "/laya":
                    print(f"\n{UI.DARK_GRAY}╭─── {UI.LAYA_BOLD}LAYA SYSTEM 1 DECISION ENGINE (Convai Innovations){UI.RST}{UI.DARK_GRAY} ─────╮{UI.RST}")
                    print(f"{UI.DARK_GRAY}│{UI.RST}  {UI.WHITE}Backbone{UI.RST}     : {UI.LAYA}ModernBERT-large (~421M params, Non-Autoregressive){UI.RST}")
                    print(f"{UI.DARK_GRAY}│{UI.RST}  {UI.WHITE}Latency{UI.RST}      : {UI.GREEN}~33ms{UI.RST}  |  {UI.WHITE}Protocol{UI.RST}: /v1/systemone ({session.laya.endpoint})")
                    perm_tag = f"{UI.GREEN_BOLD}ACTIVE{UI.RST}" if session.use_laya_adaptive_permissions else f"{UI.GRAY}INACTIVE{UI.RST}"
                    print(f"{UI.DARK_GRAY}│{UI.RST}  {UI.WHITE}Adaptive Perm{UI.RST}: {perm_tag}")
                    print(f"{UI.DARK_GRAY}╰─────────────────────────────────────────────────────────────────╯{UI.RST}\n")
                elif cmd == "/status":
                    session.print_status()
                elif cmd == "/models":
                    session.print_models()
                elif cmd == "/history":
                    if RICH_AVAILABLE and not session.json_output:
                        if not session.history:
                            print(f"\n{UI.GRAY}(no conversation history){UI.RST}\n")
                        else:
                            table = Table(
                                title=Text.from_ansi(f"{UI.WHITE}CROSS-GPU CONVERSATION HISTORY ({len(session.history)} entries){UI.RST}"),
                                box=box.ROUNDED,
                                header_style="bold cyan",
                                border_style="bright_black"
                            )
                            table.add_column("Turn", justify="center", style="bold dim", width=6)
                            table.add_column("Target", style="bold", width=12)
                            table.add_column("Role", style="bold", width=11)
                            table.add_column("Content Preview", style="none")

                            for idx, t in enumerate(session.history, 1):
                                target = t.get("target", "").lower()
                                if target in ("rocm", "9000"):
                                    t_col = f"{UI.ROCM}ROCm:9000{UI.RST}"
                                elif target in ("cuda", "9001"):
                                    t_col = f"{UI.CUDA}CUDA:9001{UI.RST}"
                                elif target == "cloud":
                                    t_col = f"{UI.CLOUD}Cloud{UI.RST}"
                                else:
                                    t_col = f"{UI.LAYA}{target.upper()}{UI.RST}"

                                role_str = f"{UI.WHITE}USER{UI.RST}" if t["role"] == "user" else f"{UI.GREEN}ASSISTANT{UI.RST}"
                                content_preview = t["content"].replace("\n", " ").strip()
                                if len(content_preview) > 120:
                                    content_preview = content_preview[:120] + "..."

                                table.add_row(
                                    str(idx),
                                    Text.from_ansi(t_col),
                                    Text.from_ansi(role_str),
                                    Text.from_ansi(content_preview)
                                )

                            console.print()
                            console.print(table)
                            console.print()
                    else:
                        print(f"\n{UI.DARK_GRAY}╭─── {UI.WHITE}Cross-GPU Session History ({len(session.history)} entries){UI.RST}{UI.DARK_GRAY} ──────────────╮{UI.RST}")
                        if session.history:
                            for t in session.history:
                                c = UI.target_color(t.get('target', ''))
                                role_color = UI.WHITE if t['role'] == 'user' else c
                                print(f"{UI.DARK_GRAY}│{UI.RST} {c}[{t['target'].upper()}]{UI.RST} {role_color}{t['role'].upper()}{UI.RST}: {t['content']}")
                        else:
                            print(f"{UI.DARK_GRAY}│{UI.RST}   {UI.GRAY}(no conversation history){UI.RST}")
                        print(f"{UI.DARK_GRAY}╰─────────────────────────────────────────────────────────────────╯{UI.RST}\n")
                elif cmd == "/clear":
                    session.history.clear()
                    session._cached_curated_memory = None
                    await session.reset_local_agent()
                    print(UI.ok("Conversation history cleared."))
                elif cmd == "/mcp":
                    arg_low = arg.lower()
                    if arg_low in ("off", "disable"):
                        session._mcp_servers = []
                        session.enable_mcp = False
                        session._active_local_agent = None
                        print(UI.ok("MCP disabled."))
                    elif arg_low in ("on", "enable", "reload"):
                        session._mcp_servers = load_mcp_servers()
                        session.enable_mcp = True
                        session._active_local_agent = None
                        names = ", ".join(s.name for s in session._mcp_servers) or "none"
                        print(UI.ok(f"MCP reloaded: {len(session._mcp_servers)} server(s): {names}"))
                    else:
                        print(f"\n{UI.DARK_GRAY}╭─── {UI.WHITE}MCP SERVER STATUS{UI.RST}{UI.DARK_GRAY} ──────────────────────────────────────────╮{UI.RST}")
                        mcp_st = f"{UI.GREEN_BOLD}YES{UI.RST}" if session.enable_mcp else f"{UI.RED_BOLD}NO{UI.RST}"
                        print(f"{UI.DARK_GRAY}│{UI.RST}  {UI.WHITE}Enabled{UI.RST} : {mcp_st}")
                        print(f"{UI.DARK_GRAY}│{UI.RST}  {UI.WHITE}Config{UI.RST}  : {UI.GRAY}{MCP_CONFIG_PATH}{UI.RST}")
                        if session._mcp_servers:
                            for s in session._mcp_servers:
                                kind = "stdio" if isinstance(s, McpStdioServer) else "http"
                                detail = s.command if isinstance(s, McpStdioServer) else s.url
                                print(f"{UI.DARK_GRAY}│{UI.RST}   {UI.AMBER}[{kind}]{UI.RST} {UI.WHITE}{s.name}{UI.RST}: {UI.GRAY}{detail}{UI.RST}")
                        else:
                            print(f"{UI.DARK_GRAY}│{UI.RST}   {UI.GRAY}(no servers loaded){UI.RST}")
                        print(f"{UI.DARK_GRAY}╰─────────────────────────────────────────────────────────────╯{UI.RST}")
                        print(f"{UI.GRAY}Commands: /mcp on | /mcp off | /mcp reload{UI.RST}\n")
                elif cmd in ("/kontexta", "/kxta"):
                    from ..context.kontexta import KontextaManager
                    arg_low = (arg or "").strip().lower()
                    if arg_low in ("off", "disable"):
                        session.enable_kontexta = False
                        session._mcp_servers = [s for s in session._mcp_servers if s.name not in ("kontexta", "kxta")]
                        session._active_local_agent = None
                        session.save_preferences()
                        print(UI.ok("Inbuilt Kontexta MCP server DISABLED."))
                    elif arg_low in ("on", "enable"):
                        session.enable_kontexta = True
                        session._mcp_servers = load_mcp_servers(enable_kontexta=True)
                        session._active_local_agent = None
                        session.save_preferences()
                        print(UI.ok("Inbuilt Kontexta MCP server ENABLED."))
                    elif arg_low in ("sync", "reload", "refresh"):
                        session._mcp_servers = load_mcp_servers(enable_kontexta=session.enable_kontexta)
                        session._active_local_agent = None
                        print(UI.ok("Kontexta MCP integration reloaded."))
                    elif arg_low.startswith("register") or arg_low.startswith("add"):
                        parts = arg.split(maxsplit=1)
                        p_name = parts[1].strip() if len(parts) > 1 else None
                        ok = KontextaManager.register_project(".", name=p_name)
                        if ok:
                            print(UI.ok(f"Project '{os.path.basename(os.path.abspath('.'))}' registered in Kontexta database!"))
                        else:
                            print(UI.err("Failed to register project in Kontexta (database locked or unreadable)."))
                    else:
                        stat = KontextaManager.get_status(".")
                        st_col = UI.GREEN_BOLD if stat["available"] else UI.RED_BOLD
                        st_text = "READY / ACTIVE" if stat["available"] else "NOT FOUND / OFFLINE"
                        active_proj = stat.get("active_project")
                        proj_str = f"{active_proj['name']} [{active_proj['slug']}]" if active_proj else f"{UI.GRAY}(unmapped; type /kontexta register){UI.RST}"

                        print(f"\n{UI.DARK_GRAY}╭─── {UI.WHITE}INBUILT KONTEXTA MCP INTEGRATION{UI.RST}{UI.DARK_GRAY} ─────────────────────────────╮{UI.RST}")
                        print(f"{UI.DARK_GRAY}│{UI.RST}  Status       : {st_col}{st_text}{UI.RST} (Enabled: {'YES' if session.enable_kontexta else 'NO'})")
                        print(f"{UI.DARK_GRAY}│{UI.RST}  Entrypoint   : {UI.CYAN}{stat['mcp_entrypoint']}{UI.RST}")
                        print(f"{UI.DARK_GRAY}│{UI.RST}  Data Dir     : {UI.GRAY}{stat['data_dir']}{UI.RST}")
                        print(f"{UI.DARK_GRAY}│{UI.RST}  Database     : {UI.GRAY}{stat['db_path']}{UI.RST} ({stat['db_size_kb']} KB)")
                        print(f"{UI.DARK_GRAY}│{UI.RST}  Knowledge    : {stat['files_count']} indexed file(s), {stat['tags_count']} tag(s)")
                        print(f"{UI.DARK_GRAY}│{UI.RST}  Active Repo  : {UI.WHITE}{proj_str}{UI.RST}")
                        print(f"{UI.DARK_GRAY}│{UI.RST}  Total Projects: {stat['projects_count']} registered")
                        print(f"{UI.DARK_GRAY}╰──────────────────────────────────────────────────────────────────╯{UI.RST}")
                        print(f"{UI.GRAY}Commands: /kontexta on | /kontexta off | /kontexta reload | /kontexta register [name]{UI.RST}\n")
                elif cmd == "/metrics":
                    arg_low = arg.lower()
                    if arg_low in ("off", "disable", "false", "0"):
                        session.show_metrics = False
                        print(UI.ok("Generation metrics display DISABLED."))
                    elif arg_low in ("on", "enable", "true", "1"):
                        session.show_metrics = True
                        print(UI.ok("Generation metrics display ENABLED."))
                    else:
                        session.show_metrics = not session.show_metrics
                        st = "ENABLED" if session.show_metrics else "DISABLED"
                        print(UI.ok(f"Generation metrics: {st}"))
                elif cmd == "/save":
                    saved_path = session.save_session(arg or None)
                    print(UI.ok(f"Session saved to {saved_path}"))
                elif cmd == "/load":
                    if not arg:
                        print(UI.warn("Usage: /load <session_name_or_file>. Type /sessions to list saved sessions."))
                    else:
                        ok = session.load_session(arg)
                        if ok:
                            print(UI.ok(f"Session '{arg}' loaded! Restored {len(session.history)//2} turns."))
                        else:
                            print(UI.err(f"Could not find session '{arg}'. Type /sessions to view available sessions."))
                elif cmd == "/sessions":
                    sess_list = session.list_sessions()
                    print(f"\n{UI.DARK_GRAY}╭─── {UI.WHITE}SAVED SESSIONS ({len(sess_list)}){UI.RST}{UI.DARK_GRAY} ────────────────────────────────────────╮{UI.RST}")
                    if sess_list:
                        for s in sess_list:
                            saved_dt = s['saved_at'][:19].replace('T', ' ')
                            print(f"{UI.DARK_GRAY}│{UI.RST}  • {UI.CYAN}{s['file']:30s}{UI.RST} │ {s['turns']:2d} turns │ {UI.GRAY}{saved_dt}{UI.RST} │ {s['size_kb']} KB")
                        print(f"{UI.DARK_GRAY}│{UI.RST}\n{UI.DARK_GRAY}│{UI.RST}  {UI.GRAY}Use '/load <filename>' to resume a session.{UI.RST}")
                    else:
                        print(f"{UI.DARK_GRAY}│{UI.RST}  {UI.GRAY}(No saved sessions found in ~/.gemini/antigravity-cli/sessions/){UI.RST}")
                    print(f"{UI.DARK_GRAY}╰─────────────────────────────────────────────────────────────╯{UI.RST}\n")
                elif cmd == "/export":
                    export_path = session.export_markdown(arg or None)
                    print(UI.ok(f"Session transcript exported to {export_path}"))
                elif cmd == "/summarize":
                    if not session.history:
                        print(UI.warn("No conversation history to summarize."))
                    else:
                        summary = session.summarize_history_sync(session.history)
                        print(f"\n{UI.DARK_GRAY}╭─── {UI.WHITE}CONVERSATION CONTEXT SUMMARY{UI.RST}{UI.DARK_GRAY} ─────────────────────────────╮{UI.RST}")
                        for line in summary.strip().splitlines():
                            print(f"{UI.DARK_GRAY}│{UI.RST}  {line}")
                        print(f"{UI.DARK_GRAY}╰─────────────────────────────────────────────────────────────╯{UI.RST}\n")
                elif cmd in ("/context", "/ctx"):
                    session.print_context_summary()
                elif cmd == "/compact":
                    keep = int(arg) if arg.isdigit() else None
                    removed = session.compact_history(keep_recent=keep)
                    if removed > 0:
                        session._active_local_agent = None
                elif cmd in ("/autocompact", "/auto-compact"):
                    arg_low = arg.lower()
                    if arg_low in ("off", "disable", "false", "0"):
                        session.autocompact_threshold = 0
                        print(UI.ok("Auto-compact DISABLED."))
                    elif arg_low in ("on", "enable", "true"):
                        session.autocompact_threshold = 20
                        print(UI.ok("Auto-compact ENABLED (threshold: 20 turns)."))
                    elif arg.isdigit():
                        session.autocompact_threshold = int(arg)
                        print(UI.ok(f"Auto-compact threshold set to {session.autocompact_threshold} turns."))
                    else:
                        st = f"ENABLED ({session.autocompact_threshold} turns)" if session.autocompact_threshold > 0 else "DISABLED"
                        print(f"\n{UI.DARK_GRAY}╭─── {UI.WHITE}AUTO-COMPACT CONFIGURATION{UI.RST}{UI.DARK_GRAY} ─────────────────────────────╮{UI.RST}")
                        print(f"{UI.DARK_GRAY}│{UI.RST}  Status     : {st}")
                        print(f"{UI.DARK_GRAY}│{UI.RST}  History    : {len(session.history)} turn(s) recorded")
                        print(f"{UI.DARK_GRAY}│{UI.RST}  Preserved  : Keeps {session.max_recent_turns} recent turns verbatim")
                        print(f"{UI.DARK_GRAY}╰─────────────────────────────────────────────────────────────╯{UI.RST}")
                        print(f"{UI.GRAY}Commands: /autocompact on | /autocompact off | /autocompact <turns>{UI.RST}\n")
                elif cmd in ("/compactor", "/curator"):
                    arg_low = arg.lower()
                    if arg_low in ("start", "on", "launch", "wake"):
                        print(f"{UI.AMBER}[⚡ Llama-Launcher]{UI.RST} Starting IBM Granite 4.2 8B on NVIDIA RTX 4060 (Port 9001)...")
                        res = await asyncio.to_thread(trigger_llamashift_switch, "granite42_8b")
                        if res.get("success"):
                            print(UI.ok("NVIDIA Compactor started successfully via Llama-Launcher."))
                            session.refresh_endpoints()
                        else:
                            print(UI.err(f"Failed to start compactor: {res.get('error', res)}"))
                    elif arg_low in ("status", ""):
                        session.refresh_endpoints()
                        ep = session.endpoints.get("9001", {})
                        st = f"{UI.GREEN}ONLINE{UI.RST}" if ep.get("status") == "online" else f"{UI.RED}OFFLINE{UI.RST}"
                        fn = ep.get("filename") or "granite-4.2-8b-Q4_K_M.gguf"
                        print(f"{UI.WHITE}NVIDIA Compactor (RTX 4060 :9001){UI.RST}: {st} ({fn})")
                        print(f"{UI.GRAY}Controls: /compactor start | /compactor status{UI.RST}")
                elif cmd in ("/curation", "/continuous"):
                    arg_low = arg.lower()
                    if arg_low in ("off", "disable", "false", "0"):
                        session.continuous_curation = False
                        print(UI.ok("Continuous background context curation DISABLED."))
                    elif arg_low in ("on", "enable", "true", "1"):
                        session.continuous_curation = True
                        print(UI.ok("Continuous background context curation ENABLED (auto-distills older turns on RTX 4060)."))
                    else:
                        st = f"{UI.GREEN}ENABLED{UI.RST}" if session.continuous_curation else f"{UI.GRAY}DISABLED{UI.RST}"
                        print(f"\n{UI.DARK_GRAY}╭─── {UI.WHITE}CONTINUOUS CURATION CONFIGURATION{UI.RST}{UI.DARK_GRAY} ──────────────╮{UI.RST}")
                        print(f"{UI.DARK_GRAY}│{UI.RST}  Status    : {st}")
                        print(f"{UI.DARK_GRAY}│{UI.RST}  Engine    : CUDA:9001 (NVIDIA RTX 4060 - Granite 8B)")
                        print(f"{UI.DARK_GRAY}│{UI.RST}  Window    : Preserves last {session.max_recent_turns} turns verbatim")
                        print(f"{UI.DARK_GRAY}│{UI.RST}  Trigger   : Asynchronously runs after EVERY turn in background")
                        print(f"{UI.DARK_GRAY}╰─────────────────────────────────────────────────────────────╯{UI.RST}")
                        print(f"{UI.GRAY}Commands: /curation on | /curation off{UI.RST}\n")
                elif cmd in ("/spawn", "/subagent", "/task"):
                    sub_parts = arg.split(maxsplit=1)
                    sub_cmd = sub_parts[0].lower() if sub_parts else ""
                    sub_arg = sub_parts[1].strip() if len(sub_parts) > 1 else ""

                    if sub_cmd == "view" and sub_arg.isdigit():
                        session.print_subagent_detail(int(sub_arg))
                    elif sub_cmd == "inject":
                        if sub_arg.lower() in ("all", "*"):
                            count = session.inject_all_subagents()
                            if count > 0:
                                print(UI.ok(f"Successfully injected findings from {count} completed subagent(s) into conversation context."))
                            else:
                                print(UI.warn("No uninjected completed subagents found to inject."))
                        elif sub_arg.isdigit():
                            ok = session.inject_subagent(int(sub_arg))
                            if ok:
                                print(UI.ok(f"Subagent #{sub_arg} result successfully injected into conversation memory."))
                            else:
                                print(UI.warn(f"Could not inject Subagent #{sub_arg}. Ensure it has finished and produced output."))
                        else:
                            print(f"{UI.GRAY}Usage: /subagent inject <#|all>{UI.RST}")
                    elif sub_cmd in ("cancel", "kill", "stop") and sub_arg.isdigit():
                        ok = session.cancel_subagent(int(sub_arg))
                        if ok:
                            print(UI.ok(f"Subagent #{sub_arg} cancelled."))
                        else:
                            print(UI.warn(f"Subagent #{sub_arg} is not running or not found in queue."))
                    elif sub_cmd == "list" or not arg:
                        session.print_subagents()
                    elif sub_cmd and sub_cmd.isdigit() and not sub_arg:
                        session.print_subagent_detail(int(sub_cmd))
                    else:
                        # Full argument string is treated as a new task description
                        session.spawn_subagent(arg)
                elif cmd in ("/build", "/test"):
                    custom_cmd = arg.strip() if arg else None
                    task = session.run_build_subagent(custom_cmd=custom_cmd, repo_root=".")
                    if not session.json_output:
                        print(f"{UI.CUDA}⚡ Pushed build & test verification to NVIDIA Subagent #{task.id} (Port 9001). Type {UI.CYAN}/subagent view {task.id}{UI.CUDA} to inspect.{UI.RST}\n")
                elif cmd in ("/tasks", "/subagents"):
                    session.print_subagents()
                elif cmd in ("/cuda", "/coordinator"):
                    from ..core.coordinator import CudaCoordinator
                    c_status = CudaCoordinator.status_summary()
                    st_col = UI.GREEN if c_status['state'] == 'idle' else (UI.CUDA_BOLD if c_status['state'] == 'subagent' else UI.AMBER)
                    print(f"\n{UI.DARK_GRAY}╭─── {UI.WHITE}NVIDIA CUDA:9001 RESOURCE COORDINATOR{UI.RST}{UI.DARK_GRAY} ─────────────╮{UI.RST}")
                    print(f"{UI.DARK_GRAY}│{UI.RST}  State        : {st_col}{c_status['state'].upper()}{UI.RST}")
                    print(f"{UI.DARK_GRAY}│{UI.RST}  Subagents    : {c_status['active_subagents']} running")
                    print(f"{UI.DARK_GRAY}│{UI.RST}  Compacting   : {'YES' if c_status['compactor_running'] else 'NO'}")
                    permit_str = f"{UI.GREEN}ALLOWED (IDLE){UI.RST}" if c_status['can_compact'] else f"{UI.AMBER}PAUSED (Subagent has priority){UI.RST}"
                    print(f"{UI.DARK_GRAY}│{UI.RST}  Idle Permit  : {permit_str}")
                    print(f"{UI.DARK_GRAY}╰──────────────────────────────────────────────────────────────────╯{UI.RST}\n")
                elif cmd in ("/repomap", "/ast"):
                    arg_low = arg.lower().strip()
                    if arg_low in ("on", "enable", "true"):
                        session.auto_repomap = True
                        session._active_local_agent = None
                        print(UI.ok("Automatic repository AST symbol map injection ENABLED (cached, zero-latency)."))
                    elif arg_low in ("off", "disable", "false"):
                        session.auto_repomap = False
                        session._active_local_agent = None
                        print(UI.ok("Automatic repository AST symbol map injection DISABLED."))
                    elif arg_low in ("refresh", "reload", "sync"):
                        session.refresh_repomap(".")
                        print(UI.ok("Repository AST symbol map cache invalidated & refreshed."))
                    elif arg_low == "stats":
                        from ..context.repomap import RepoMap
                        st = RepoMap.cache_stats()
                        print(f"\n{UI.DARK_GRAY}╭─── {UI.WHITE}REPO MAP CACHE TELEMETRY{UI.RST}{UI.DARK_GRAY} ─────────────────────────────╮{UI.RST}")
                        print(f"{UI.DARK_GRAY}│{UI.RST}  Auto-Inject : {UI.GREEN}ENABLED{UI.RST}" if session.auto_repomap else f"{UI.DARK_GRAY}│{UI.RST}  Auto-Inject : {UI.GRAY}DISABLED{UI.RST}")
                        print(f"{UI.DARK_GRAY}│{UI.RST}  Root Path   : {UI.CYAN}{st['root']}{UI.RST}")
                        print(f"{UI.DARK_GRAY}│{UI.RST}  Indexed     : {st['file_count']} Python files ({st['char_count']} chars)")
                        print(f"{UI.DARK_GRAY}╰─────────────────────────────────────────────────────────────╯{UI.RST}\n")
                    else:
                        scan_path = arg or "."
                        session.print_repomap(scan_path)
                        st_tag = f"{UI.GREEN}Active (Auto-injected into model memory){UI.RST}" if session.auto_repomap else f"{UI.GRAY}Disabled{UI.RST}"
                        print(f"{UI.GRAY}Auto-injection status: {st_tag} │ Toggle: /repomap [on|off|refresh]{UI.RST}\n")

                elif cmd in ("/instructions", "/rules", "/guidelines"):
                    from ..context.instructions import ProjectInstructions
                    arg_low = arg.lower().strip()
                    if arg_low in ("on", "enable", "true"):
                        session.auto_instructions = True
                        session._active_local_agent = None
                        print(UI.ok("Automatic project instructions injection (GEMINI.md / antigravity.md) ENABLED."))
                    elif arg_low in ("off", "disable", "false"):
                        session.auto_instructions = False
                        session._active_local_agent = None
                        print(UI.ok("Automatic project instructions injection DISABLED."))
                    elif arg_low in ("refresh", "reload"):
                        content, loaded = ProjectInstructions.load_instructions(".", force_refresh=True)
                        session._active_local_agent = None
                        print(UI.ok(f"Project instructions refreshed. Loaded: {', '.join(loaded) if loaded else 'None'}"))
                    elif arg_low in ("view", "show", "cat") or not arg:
                        content, loaded = ProjectInstructions.load_instructions(".", force_refresh=False)
                        if loaded:
                            print(f"\n{UI.DARK_GRAY}╭─── {UI.WHITE}PROJECT INSTRUCTIONS ({', '.join(loaded)}){UI.RST}{UI.DARK_GRAY} ─────────────╮{UI.RST}")
                            for line in content.splitlines():
                                print(f"{UI.DARK_GRAY}│{UI.RST}  {line}")
                            print(f"{UI.DARK_GRAY}╰─────────────────────────────────────────────────────────────╯{UI.RST}\n")
                        else:
                            print(UI.warn("No instruction files found (looking for GEMINI.md, antigravity.md, AGENTS.md, etc.)"))
                    else:
                        print(f"{UI.GRAY}Usage: /instructions [on|off|refresh|view]{UI.RST}")

                elif cmd in ("/hooks", "/hook"):
                    from ..context.hooks_loader import ExternalHooksManager
                    arg_low = arg.lower().strip()
                    if arg_low in ("on", "enable", "true"):
                        session.enable_hooks = True
                        session._active_local_agent = None
                        print(UI.ok("External lifecycle hooks (hooks.json) ENABLED."))
                    elif arg_low in ("off", "disable", "false"):
                        session.enable_hooks = False
                        session._active_local_agent = None
                        print(UI.ok("External lifecycle hooks (hooks.json) DISABLED."))
                    elif arg_low in ("refresh", "reload"):
                        entries, sources = ExternalHooksManager.load_hook_configs(".", force_refresh=True)
                        session._active_local_agent = None
                        print(UI.ok(f"Lifecycle hooks reloaded. Found {len(entries)} active hook(s) from {len(sources)} file(s)."))
                    elif arg_low in ("view", "list", "status") or not arg:
                        entries, sources = ExternalHooksManager.load_hook_configs(".", force_refresh=False)
                        st = f"{UI.GREEN}ENABLED{UI.RST}" if session.enable_hooks else f"{UI.GRAY}DISABLED{UI.RST}"
                        print(f"\n{UI.DARK_GRAY}╭─── {UI.WHITE}EXTERNAL LIFECYCLE HOOKS ({st}){UI.RST}{UI.DARK_GRAY} ────────────────────────╮{UI.RST}")
                        if sources:
                            print(f"{UI.DARK_GRAY}│{UI.RST}  Source Files:")
                            for sf in sources:
                                print(f"{UI.DARK_GRAY}│{UI.RST}    • {UI.CYAN}{sf}{UI.RST}")
                            print(f"{UI.DARK_GRAY}│{UI.RST}")
                            print(f"{UI.DARK_GRAY}│{UI.RST}  Active Hooks ({len(entries)}):")
                            for entry in entries:
                                events = [ev for ev in ["PreToolUse", "PostToolUse", "PreTurn", "PostTurn", "PreInvocation", "PostInvocation", "Stop"] if entry["spec"].get(ev)]
                                print(f"{UI.DARK_GRAY}│{UI.RST}    • {UI.WHITE}{entry['name']:20s}{UI.RST} ➔ Events: {UI.AMBER}{', '.join(events)}{UI.RST}")
                        else:
                            print(f"{UI.DARK_GRAY}│{UI.RST}  {UI.GRAY}No hooks.json discovered in .agents/hooks.json or ~/.gemini/config/hooks.json{UI.RST}")
                        print(f"{UI.DARK_GRAY}╰─────────────────────────────────────────────────────────────╯{UI.RST}")
                        print(f"{UI.GRAY}Commands: /hooks on | /hooks off | /hooks refresh | /hooks view{UI.RST}\n")
                    else:
                        print(f"{UI.GRAY}Usage: /hooks [on|off|refresh|view]{UI.RST}")

                elif cmd in ("/search", "/web", "/google", "/ddg"):
                    from ..tools.web_search import WebSearchEngine
                    if not arg:
                        print(f"{UI.GRAY}Usage: /search <query to search online>{UI.RST}")
                    else:
                        print(f"{UI.CYAN}Searching the web for '{arg}'...{UI.RST}")
                        results = await asyncio.to_thread(WebSearchEngine.search, arg, 5)
                        WebSearchEngine.print_search_results(arg, results)

                elif cmd in ("/fetch", "/browse", "/readurl"):
                    from ..tools.web_search import WebSearchEngine
                    if not arg:
                        print(f"{UI.GRAY}Usage: /fetch <url to inspect>{UI.RST}")
                    else:
                        print(f"{UI.CYAN}Fetching and distilling {arg}...{UI.RST}")
                        content = await asyncio.to_thread(WebSearchEngine.fetch_url, arg, 4000)
                        print(f"\n{UI.DARK_GRAY}╭─── {UI.WHITE}WEBPAGE CONTENT ({arg}){UI.RST}{UI.DARK_GRAY} ─────────────────────────────╮{UI.RST}")
                        for line in content.splitlines():
                            print(f"{UI.DARK_GRAY}│{UI.RST}  {line}")
                        print(f"{UI.DARK_GRAY}╰────────────────────────────────────────────────────────────────────────╯\n")

                elif cmd in ("/steps", "/maxsteps", "/max-steps"):
                    arg_low = arg.lower()
                    if arg_low in ("off", "disable", "unlimited", "none", "0"):
                        session.max_tool_steps_per_turn = 0
                        print(UI.ok("Tool step limit DISABLED (unlimited steps per turn; consecutive loop guard remains active)."))
                    elif arg.isdigit() and int(arg) > 0:
                        session.max_tool_steps_per_turn = int(arg)
                        print(UI.ok(f"Tool execution step limit set to {session.max_tool_steps_per_turn} steps per turn."))
                    else:
                        limit_str = f"{session.max_tool_steps_per_turn} steps" if session.max_tool_steps_per_turn > 0 else "UNLIMITED (disabled)"
                        print(f"\n{UI.DARK_GRAY}╭─── {UI.WHITE}TOOL STEP LIMIT & ANTI-LOOP GUARD{UI.RST}{UI.DARK_GRAY} ────────────────────────╮{UI.RST}")
                        print(f"{UI.DARK_GRAY}│{UI.RST}  Max steps / turn : {UI.CYAN_BOLD}{limit_str}{UI.RST}")
                        print(f"{UI.DARK_GRAY}│{UI.RST}  Loop detection   : {UI.GREEN_BOLD}ACTIVE{UI.RST} (blocks identical consecutive tool calls)")
                        print(f"{UI.DARK_GRAY}╰─────────────────────────────────────────────────────────────╯{UI.RST}")
                        print(f"{UI.GRAY}Commands: /steps off (unlimited) | /steps <number>{UI.RST}\n")
                elif cmd in ("/verbose", "/v"):
                    arg_low = arg.lower()
                    if arg_low in ("off", "disable", "false", "0"):
                        session.verbose = False
                        print(UI.ok("Verbose mode DISABLED (showing concise tool status)."))
                    elif arg_low in ("on", "enable", "true", "1"):
                        session.verbose = True
                        print(UI.ok("Verbose mode ENABLED (showing full tool arguments, outputs, and detailed step traces)."))
                    else:
                        session.verbose = not session.verbose
                        st = "ENABLED (showing full tool inputs and outputs)" if session.verbose else "DISABLED (showing concise tool status)"
                        print(UI.ok(f"Verbose mode: {st}"))
                elif cmd == "/json":
                    session.json_output = not session.json_output
                    st = "ENABLED" if session.json_output else "DISABLED"
                    print(UI.ok(f"Structured JSON output mode: {st}"), file=sys.stderr if session.json_output else sys.stdout)
                elif cmd == "/format":
                    arg_low = arg.lower()
                    if arg_low in ("json", "js"):
                        session.json_output = True
                        print(UI.ok("Structured JSON output mode: ENABLED"), file=sys.stderr)
                    elif arg_low in ("text", "stream", "plain"):
                        session.json_output = False
                        print(UI.ok("Standard text stream output mode: ENABLED"))
                    else:
                        print(UI.warn("Usage: /format json | text"))
                elif cmd in ("/multiline", "/multi"):
                    arg_low = arg.lower()
                    if arg_low in ("off", "disable", "false", "0", "single"):
                        session.multiline_input = False
                        print(UI.ok("Multi-line input DISABLED (single Enter submits)."))
                    elif arg_low in ("on", "enable", "true", "1"):
                        session.multiline_input = True
                        print(UI.ok("Multi-line input ENABLED (Enter on empty line to submit)."))
                    else:
                        session.multiline_input = not session.multiline_input
                        st = "ENABLED (Enter on empty line to submit)" if session.multiline_input else "DISABLED (single Enter submits)"
                        print(UI.ok(f"Multi-line input mode: {st}"))
                elif cmd in ("/singleline", "/single"):
                    session.multiline_input = False
                    print(UI.ok("Single-line input mode ENABLED (single Enter submits)."))
                elif cmd in ("/reload", "/restart"):
                    if session.history:
                        session.save_session("latest_session")
                    session.save_preferences()
                    print(f"\n{UI.CYAN}⚡ Hot-reloading antigravity_agent.py (preserving session context)...{UI.RST}\n")
                    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    runner_script = os.path.join(pkg_root, "antigravity_agent.py")
                    new_args = [sys.executable, runner_script, "--load", "latest_session", "--cloud-model", session.cloud_model]
                    if session.active_target in ("auto", "9000", "9001", "cloud"):
                        new_args += ["--target", session.active_target]
                    new_args.append("-y" if session.dangerously_skip_permissions else "--safe")
                    if session.verbose:
                        new_args.append("--verbose")
                    if session.json_output:
                        new_args.append("--json")
                    if not session.enable_mcp:
                        new_args.append("--no-mcp")
                    if not session.multiline_input:
                        new_args.append("--single-line")
                    os.execv(sys.executable, new_args)
                elif cmd in ("/brainstorm", "/spec", "/plan"):
                    from ..agents.brainstorm import BrainstormWorkflow
                    _current_task = asyncio.create_task(BrainstormWorkflow.execute_workflow(arg, session, repo_root="."))
                    esc_listener = EscListener(
                        on_escape=lambda: loop.call_soon_threadsafe(
                            lambda: _current_task.cancel() if _current_task and not _current_task.done() else None
                        )
                    )
                    try:
                        with esc_listener:
                            await _current_task
                    except asyncio.CancelledError:
                        print(f"\n{UI.AMBER_BOLD}[Esc]{UI.RST} {UI.AMBER}Brainstorm workflow stopped by user.{UI.RST}\n")
                    finally:
                        _current_task = None
                elif cmd == "/diff":
                    nth = int(arg) if arg.isdigit() else 1
                    text = await asyncio.to_thread(session.turn_diff, nth)
                    if not text:
                        print(UI.warn("No recorded file changes for that turn (needs a git repo and a turn that edited files)."))
                    else:
                        stat = await asyncio.to_thread(session.turn_diff, nth, True)
                        print(f"\n{UI.GRAY}{stat}{UI.RST}")
                        for line in text.splitlines():
                            if line.startswith("+") and not line.startswith("+++"):
                                print(f"{UI.GREEN}{line}{UI.RST}")
                            elif line.startswith("-") and not line.startswith("---"):
                                print(f"{UI.RED}{line}{UI.RST}")
                            elif line.startswith("@@") or line.startswith("diff "):
                                print(f"{UI.CYAN}{line}{UI.RST}")
                            else:
                                print(line)
                        print()
                elif cmd == "/undo":
                    plan = await asyncio.to_thread(session.undo_plan)
                    if not plan or not plan["files"]:
                        print(UI.warn("Nothing to undo: no recorded turn with file changes."))
                    else:
                        cp = plan["checkpoint"]
                        print(f"\n{UI.AMBER_BOLD}Revert turn:{UI.RST} {cp['prompt']}")
                        for path, st in sorted(plan["files"].items()):
                            verb = {"A": "delete (created by turn)", "M": "restore", "D": "restore (deleted by turn)"}.get(st, "restore")
                            print(f"  {UI.GRAY}{st}{UI.RST} {path} {UI.DARK_GRAY}-> {verb}{UI.RST}")
                        ans = (await asyncio.to_thread(prompt_input, f"{UI.AMBER_BOLD}Revert these files? [y/N]: {UI.RST}")).strip().lower()
                        if ans in ("y", "yes"):
                            touched = await asyncio.to_thread(session.apply_undo, plan)
                            print(UI.ok(f"Reverted {len(touched)} file(s). The model has been told."))
                        else:
                            print(UI.warn("Undo cancelled."))
                elif cmd == "/stats":
                    if arg.lower() == "reset":
                        session.stats.clear()
                        print(UI.ok("Usage ledger reset."))
                    else:
                        session.print_stats()
                elif cmd in ("/markdown", "/md"):
                    arg_low = arg.lower()
                    if arg_low in ("on", "enable", "true", "1"):
                        session.render_markdown = True
                    elif arg_low in ("off", "disable", "false", "0"):
                        session.render_markdown = False
                    else:
                        session.render_markdown = not session.render_markdown
                    print(UI.ok(f"Markdown rendering of responses: {'ENABLED' if session.render_markdown else 'DISABLED (raw streaming)'}"))
                elif cmd in ("/prefs", "/preferences"):
                    if arg.lower() == "reset":
                        reset_prefs()
                        session.apply_preferences(PREF_DEFAULTS)
                        session._active_local_agent = None
                        print(UI.ok("Preferences reset to defaults."))
                    else:
                        print(f"\n{UI.DARK_GRAY}╭─── {UI.WHITE}SAVED PREFERENCES (~/.myagy/config.json){UI.RST}{UI.DARK_GRAY} ─────────╮{UI.RST}")
                        for k, v in session.preferences().items():
                            print(f"{UI.DARK_GRAY}│{UI.RST}  {k:32s} {UI.CYAN}{v}{UI.RST}")
                        print(f"{UI.DARK_GRAY}╰─────────────────────────────────────────────────────────────╯{UI.RST}")
                        print(f"{UI.GRAY}Settings save automatically after each command. /prefs reset restores defaults.{UI.RST}\n")
                elif cmd == "/help":
                    session.print_help()
                elif cmd in ("/exit", "/quit"):
                    print(f"{UI.GRAY}Exiting session.{UI.RST}")
                    break
                else:
                    print(UI.warn(f"Unknown command '{cmd}'. Type /help for available commands."))
                session.save_preferences()
                continue

            # Handle one-shot @ prefixes
            force_target = None
            prompt = user_input
            lower_input = user_input.lower()
            if lower_input.startswith(("@9000 ", "@rocm ", "@amd ")):
                force_target, prompt = "9000", user_input.split(maxsplit=1)[1].strip()
            elif lower_input.startswith(("@9001 ", "@cuda ", "@nvidia ")):
                force_target, prompt = "9001", user_input.split(maxsplit=1)[1].strip()
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

            # Run chat as a cancellable task with ESC-key interrupt support
            _current_task = asyncio.create_task(session.chat(prompt, force_target=force_target))
            esc_listener = EscListener(
                on_escape=lambda: loop.call_soon_threadsafe(
                    lambda: _current_task.cancel() if _current_task and not _current_task.done() else None
                )
            )
            try:
                with esc_listener:
                    await _current_task
            except asyncio.CancelledError:
                if session.json_output:
                    print(json.dumps({"status": "cancelled", "interrupted_by": "esc" if esc_listener.interrupted else "sigint", "prompt": prompt}, indent=2))
                else:
                    if esc_listener.interrupted:
                        print(f"\n{UI.AMBER_BOLD}[Esc]{UI.RST} {UI.AMBER}Interrupted by user. Resuming prompt.{UI.RST}\n", flush=True)
                    else:
                        print(f"\n{UI.AMBER}[Cancelled] Request stopped. Resuming prompt.{UI.RST}\n", flush=True)
                session._active_local_agent = None
            except Exception as e:
                err = str(e)
                if "CancelledError" in err or "cancelled" in err.lower() or esc_listener.interrupted:
                    if session.json_output:
                        print(json.dumps({"status": "cancelled", "interrupted_by": "esc" if esc_listener.interrupted else "exception", "prompt": prompt}, indent=2))
                    else:
                        if esc_listener.interrupted:
                            print(f"\n{UI.AMBER_BOLD}[Esc]{UI.RST} {UI.AMBER}Interrupted by user. Resuming prompt.{UI.RST}\n", flush=True)
                        else:
                            print(f"\n{UI.AMBER}[Cancelled] Request stopped.{UI.RST}\n", flush=True)
                    session._active_local_agent = None
                else:
                    if session.json_output:
                        print(json.dumps({"status": "error", "error": err, "prompt": prompt}, indent=2))
                    else:
                        print(f"\n{UI.RED_BOLD}[Error during turn]:{UI.RST} {UI.RED}{e}{UI.RST}\n")
            finally:
                _current_task = None

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
        choices=["auto", "9000", "9001", "rocm", "cuda", "cloud"],
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
        const="9001",
        help="Alias to start with NVIDIA CUDA GPU (port 9001)",
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

    args = parser.parse_args()

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

    if args.load_session:
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
    finally:
        await session.close()
