"""Multi-GPU Hybrid Agent Session with unified memory, Laya System 1 decision engine, and tool execution."""

import os
import re
import sys
import json
import subprocess
import time
import shutil
import asyncio
import urllib.request
import copyreg
from datetime import datetime
from typing import Optional, List, Dict, Any

# Prevent deepcopy/pickle crashes when asyncio Tasks or sessions are referenced during config copying
try:
    import _asyncio
    copyreg.pickle(_asyncio.Task, lambda t: (lambda: None, ()))
except Exception:
    pass

from google.antigravity import (
    Agent,
    LocalAgentConfig,
    LocalOpenAIAgentConfig,
    CapabilitiesConfig,
    types,
    hooks,
)
from google.antigravity.hooks import policy
from google.antigravity.types import McpStdioServer, McpStreamableHttpServer

from ..config import (
    SESSIONS_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_ENDPOINTS,
    DEFAULT_CLOUD_MODEL,
    DEFAULT_LAYA_ENDPOINT,
    FAST_GIT_MODEL,
    DENSE_CODE_MODEL,
)
from ..terminal.ui import UI, RICH_AVAILABLE, console, StreamRenderer
from ..terminal.esc_listener import prompt_input
from .llamashift import (
    detect_port_model,
    trigger_llamashift_switch,
    get_llamashift_active,
    get_llamashift_models,
    get_logs_tail,
    model_is_running,
    shift_state,
)
from ..context.mcp_loader import load_mcp_servers, check_oauth_available
from ..context.kontexta import KontextaManager
from ..context import kontexta_sync
from .laya import LayaDecisionEngine
from ..agents.compactor import ContextCurator
from .coordinator import CudaCoordinator
from .checkpoints import Checkpoints
from ..terminal import receipt, say
from .prefs import PREF_DEFAULTS, load_prefs, save_prefs
from ..agents.subagent import SubagentManager, SubagentTask
from ..context.repomap import RepoMap
from ..context.instructions import ProjectInstructions
from ..context.hooks_loader import ExternalHooksManager
from .sudo_manager import SudoManager


_SGR_256 = re.compile(r"38;5;(\d+)")
_XTERM16 = ["#000000", "#cd0000", "#00cd00", "#cdcd00", "#0000ee", "#cd00cd", "#00cdcd", "#e5e5e5",
            "#7f7f7f", "#ff0000", "#00ff00", "#ffff00", "#5c5cff", "#ff00ff", "#00ffff", "#ffffff"]


def _xterm_hex(idx: int) -> str:
    if idx < 16:
        return _XTERM16[idx]
    if idx >= 232:
        v = 8 + 10 * (idx - 232)
        return f"#{v:02x}{v:02x}{v:02x}"
    levels = (0, 95, 135, 175, 215, 255)
    idx -= 16
    return f"#{levels[idx // 36]:02x}{levels[(idx // 6) % 6]:02x}{levels[idx % 6]:02x}"


def _pt_style(ansi: str, bold: bool = False) -> str:
    """Turns a UI color escape into a prompt_toolkit style string; empty when colors are off."""
    parts = []
    m = _SGR_256.search(ansi or "")
    if m:
        parts.append(f"fg:{_xterm_hex(int(m.group(1)))}")
    if bold or (ansi and re.search(r"\[(?:\d+;)*1(?:;\d+)*m", ansi)):
        parts.append("bold")
    return " ".join(parts)


def _short_count(n: float) -> str:
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M".replace(".0M", "M")
    if n >= 100_000:
        return f"{n / 1000:.0f}k"
    if n >= 1000:
        return f"{n / 1000:.1f}k".replace(".0k", "k")
    return str(n)


class MultiGpuHybridSession:
    """Manages conversations and routing across dual local GPUs (ROCm & CUDA) and Cloud."""

    def __deepcopy__(self, memo):
        # Preserve session singleton identity and prevent deepcopy recursion into asyncio Tasks/queues
        return self

    def __init__(
        self,
        initial_target: str = "9000",
        cloud_model: str = DEFAULT_CLOUD_MODEL,
        api_key: Optional[str] = None,
        dangerously_skip_permissions: bool = True,
        use_laya_adaptive_permissions: bool = True,
        laya_endpoint: str = DEFAULT_LAYA_ENDPOINT,
        enable_mcp: bool = True,
        enable_kontexta: bool = True,
        json_output: bool = False,
        multiline_input: bool = True,
        verbose: bool = False,
    ):
        self._shift_noted: str = "up"
        self.cloud_model = cloud_model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.has_oauth = check_oauth_available()
        self._dangerously_skip_permissions = dangerously_skip_permissions
        self.use_laya_adaptive_permissions = use_laya_adaptive_permissions
        self.enable_mcp = enable_mcp
        self.enable_kontexta = enable_kontexta
        self.json_output = json_output
        self.multiline_input = multiline_input
        self.verbose = verbose
        self.auto_repomap: bool = True
        self.auto_instructions: bool = True
        self.kontexta_sync: bool = True
        self.enable_hooks: bool = True
        self.render_markdown: bool = True
        self.last_answer: str = ""
        self._tool_pending: List[Dict[str, Any]] = []
        self._approved_actions: set = set()
        self._ctx_cache: tuple = (0.0, None)

        # Load MCP servers once at startup (including Inbuilt Kontexta)
        self._mcp_servers: List[McpStdioServer | McpStreamableHttpServer] = []
        if enable_mcp:
            self._mcp_servers = load_mcp_servers(enable_kontexta=self.enable_kontexta)
            if not self.json_output:
                if self._mcp_servers:
                    names = ", ".join(s.name for s in self._mcp_servers)
                    say.step(f"MCP: {len(self._mcp_servers)} server(s) ({names})")
                else:
                    say.step("MCP: no servers found (add to ~/.gemini/config/mcp_config.json)")

        # Laya Decision Engine (System 1)
        self.laya = LayaDecisionEngine(endpoint=laya_endpoint)

        # Endpoints map - dynamically populated without hardcoding
        self.endpoints: Dict[str, Dict[str, Any]] = {
            k: dict(v) for k, v in DEFAULT_ENDPOINTS.items()
        }

        # Track active local agent and its instantiated model
        self._active_local_agent: Optional[Agent] = None
        self._current_agent_target: Optional[str] = None
        self._active_local_model: Optional[str] = None
        # Agents detached from sync code; closed on the next async call so MCP children are not leaked
        self._stale_agents: List[Agent] = []
        # Post-routing target of the turn in flight, used to attribute errors in /stats
        self._turn_target: Optional[str] = None

        # Dynamically discover live models on ports 9000 and 9001
        self.refresh_endpoints()

        # Active default target: 'auto', '9000', or 'cloud' (9001 is reserved for compaction)
        self.active_target = self.resolve_target_alias(initial_target)

        # Conversation history: list of {"role": "user"|"assistant", "target": str, "content": str}
        self.history: List[Dict[str, str]] = []
        self.show_metrics: bool = True
        # Both settings count user/assistant turns; history stores two entries per turn
        self.max_recent_turns: int = 6
        self.autocompact_threshold: int = 20
        # Anti-loop guard: 0 = unlimited tool steps per turn; consecutive duplicate loop detection remains active
        self.max_tool_steps_per_turn: int = 0
        self._current_turn_tool_count: int = 0
        self._recent_tool_calls: List[tuple] = []
        self._turn_executed_tools: List[Dict[str, Any]] = []
        self._last_tool_call: Dict[str, Any] = {}
        # Per-target usage ledger for /stats
        self.stats: Dict[str, Dict[str, float]] = {}
        # Working-tree checkpoints per turn for /diff and /undo
        self.checkpoints: List[Dict[str, Any]] = []
        self._repo_root: Optional[str] = None

        # Dynamic Auto-Shift (Git Actions ➔ Gemma 4 MoE, Deep Reasoning ➔ Qwen 27B Dense)
        self.auto_git_shift: bool = True
        self.fast_git_model: str = FAST_GIT_MODEL
        self.dense_code_model: str = DENSE_CODE_MODEL

        # NVIDIA RTX 4060 (:9001) Background Context Curator Cache
        self._cached_curated_memory: Optional[Dict[str, Any]] = None
        self._synthesis_task: Optional[asyncio.Task] = None
        self.continuous_curation: bool = True

    @property
    def _recent_entries(self) -> int:
        return max(self.max_recent_turns, 0) * 2

    @property
    def json_output(self) -> bool:
        return self._json_output

    @json_output.setter
    def json_output(self, value: bool):
        self._json_output = bool(value)
        say.configure(self._json_output, getattr(self, "_verbose", False))

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool):
        self._verbose = bool(value)
        say.configure(getattr(self, "_json_output", False), self._verbose)

    @property
    def theme(self) -> str:
        return UI.theme_name

    def set_theme(self, name: str) -> bool:
        """Switches the color theme, remembers it, and returns False for unknown names."""
        if not UI.set_theme(str(name).strip().lower()):
            return False
        self.save_preferences()
        return True

    def _notice(self, msg: str):
        say.plain(msg)

    def _invalidate_local_agent(self):
        """Sync-safe: detaches the cached agent; it is closed by the next async call."""
        if self._active_local_agent is not None:
            self._stale_agents.append(self._active_local_agent)
            self._active_local_agent = None

    async def _close_stale_agents(self):
        stale, self._stale_agents = self._stale_agents, []
        for agent in stale:
            try:
                await agent.__aexit__(None, None, None)
            except Exception:
                pass

    def refresh_endpoints(self) -> Dict[str, Any]:
        """Refreshes port 9000/9001 model state from one cached LlamaShift /api/status, then probes ports directly."""
        model_dir = DEFAULT_MODEL_DIR

        # Resolve running models from the cached LlamaShift status in one shot
        active_by_port: Dict[int, Dict] = {}
        try:
            active_list = get_llamashift_active()
            for m in active_list:
                ep_url = m.get("endpoint", "")
                m_port = m.get("port")
                if not m_port and ":" in ep_url:
                    try:
                        m_port = int(ep_url.rstrip("/").rsplit(":", 1)[-1].split("/")[0])
                    except (ValueError, IndexError):
                        pass
                if m_port:
                    active_by_port[int(m_port)] = m
        except Exception:
            pass

        state = shift_state()
        if state != self._shift_noted:
            note = {"down": "LlamaShift is not reachable, probing the model ports directly",
                    "starting": "LlamaShift is still starting",
                    "unauthorized": "LlamaShift rejected the request, set LLAMASHIFT_API_KEY to its management key",
                    "up": "LlamaShift is reachable again"}[state]
            say.step(note)
            self._shift_noted = state

        for port_str, ep in self.endpoints.items():
            port_int = ep["port"]
            if port_int in active_by_port:
                m = active_by_port[port_int]
                fname = m.get("filename") or ""
                full_p = os.path.join(model_dir, fname) if fname and os.path.exists(os.path.join(model_dir, fname)) else fname
                ep["model"] = full_p or fname
                ep["filename"] = fname or os.path.basename(full_p)
                ep["name_friendly"] = m.get("name") or fname
                ep["status"] = "online"
                ep["source"] = "llamashift-active"
                if m.get("gpu"):
                    ep["desc"] = m["gpu"]
            else:
                # Fallback: per-port detection (tries /api/status then /v1/models)
                det = detect_port_model(port_int, ep["url"])
                ep["model"] = det["model_path"]
                ep["filename"] = det["filename"]
                ep["name_friendly"] = det["name"]
                ep["status"] = det["status"]
                ep["source"] = det["source"]
                if det.get("gpu"):
                    ep["desc"] = det["gpu"]

        return self.endpoints

    def resolve_target_alias(self, alias_or_name: str) -> str:
        s = alias_or_name.lower().strip()
        if s in ("auto", "laya", "smart", "dynamic"):
            return "auto"
        if s in ("rocm", "amd", "9000", "gpu0"):
            return "9000"
        if s in ("cuda", "nvidia", "9001", "gpu1"):
            return "9001"
        if s in ("cloud", "gemini", "remote", "oauth"):
            return "cloud"
        if s in ("local", "default"):
            return "9000" if self.endpoints["9000"]["status"] == "online" else "9001"
        return s

    @property
    def dangerously_skip_permissions(self) -> bool:
        return self._dangerously_skip_permissions

    def set_dangerously_skip_permissions(self, enable: bool):
        if self._dangerously_skip_permissions != enable:
            self._dangerously_skip_permissions = enable
            self._invalidate_local_agent()

    @staticmethod
    def _ask(prompt: str) -> str:
        return prompt_input(f"{UI.AMBER_BOLD}{prompt}{UI.RST} ").strip().lower()

    @staticmethod
    def _action_signature(tool_name: str, args: Any) -> tuple:
        try:
            return (tool_name, json.dumps(args, sort_keys=True, default=str))
        except Exception:
            return (tool_name, str(args))

    @staticmethod
    def _arg_rows(args: Any) -> List[tuple]:
        if not isinstance(args, dict):
            return [("Args", str(args))] if args else []
        rows = []
        for key, val in args.items():
            text = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False, default=str)
            shown, hidden = say.truncate_lines(text, 6)
            rows.append(("Command" if key == "CommandLine" else str(key), shown + (f"\n+{hidden} lines" if hidden else "")))
        return rows

    def _sudo_gate(self, tool_name: str, cmd_line: str) -> bool:
        user = os.environ.get("USER", "root")
        authed = SudoManager.is_authenticated()
        rows = [("Tool", f"{tool_name} (root elevation)"), ("Command", cmd_line), ("Target", f"host system ({user})")]
        if authed:
            rows.append(("Credentials", f"cached, ~{int(SudoManager.time_remaining() // 60)}m remaining"))
        say.card(
            "Sudo elevation required", rows,
            actions="[Y]es / [n]o / [c]hange password" if authed else "[y]es / [N]o", style="danger",
        )
        try:
            choice = self._ask("Authorize root execution?" if authed else "Authorize execution with sudo?")
        except (KeyboardInterrupt, EOFError):
            say.warn("Execution denied")
            return False

        if authed:
            if choice in ("c", "change", "change-password"):
                if not SudoManager.prompt_password_entry(reason=cmd_line):
                    return False
                say.step("root execution approved")
                return True
            if choice in ("", "y", "yes"):
                say.step("root execution approved")
                return True
            say.warn("Execution denied by user")
            return False

        if choice in ("y", "yes"):
            if SudoManager.prompt_password_entry(reason=cmd_line):
                say.step("root execution approved")
                return True
            say.err("Sudo authorization failed. Execution denied.")
            return False
        say.warn("Execution denied by user")
        return False

    def permission_prompt_handler(self, tool: Any, args: Dict[str, Any]) -> bool:
        tool_name = getattr(tool, "name", str(tool))
        cmd_line = args.get("CommandLine", "") if isinstance(args, dict) else ""
        if tool_name == "run_command" and SudoManager.is_sudo_command(cmd_line):
            return self._sudo_gate(tool_name, cmd_line)

        if self._dangerously_skip_permissions and not self.use_laya_adaptive_permissions:
            say.step(f"auto-approved {tool_name} (skip-permissions mode)")
            return True

        reason = "Safe mode: confirmation required before running tools"
        risk = ""
        if self.use_laya_adaptive_permissions:
            action_summary = f"Tool: {tool_name} Args: {json.dumps(args)}"
            is_safe, p_safe = self.laya.evaluate_action_safety(action_summary)
            if is_safe:
                say.step(f"auto-approved {tool_name} (laya P(safe)={p_safe:.2f})")
                say.detail(f"Laya System 1 'noul': P(Safe)={p_safe:.2f} >= 0.90")
                return True
            risk = f"P(safe) {p_safe:.2f} is below the 0.90 threshold"
            reason = "Laya flagged elevated risk"

        sig = self._action_signature(tool_name, args)
        if sig in self._approved_actions:
            say.step(f"approved {tool_name} (same call allowed earlier this session)")
            return True

        rows = [("Tool", tool_name), *self._arg_rows(args)]
        if risk:
            rows.append(("Risk", risk))
        rows.append(("Reason", reason))
        say.card("Permission required", rows, actions="[y]es / [n]o / [a]lways", style="warn")
        try:
            choice = self._ask("Authorize?")
        except (KeyboardInterrupt, EOFError):
            say.warn("Execution denied")
            return False
        if choice in ("a", "always"):
            self._approved_actions.add(sig)
            say.step(f"approved {tool_name} (this exact call, rest of session)")
            return True
        if choice in ("y", "yes"):
            say.step(f"approved {tool_name}")
            return True
        say.warn("Execution denied by user")
        return False

    def get_policies(self) -> List[Any]:
        return policy.confirm_run_command(handler=self.permission_prompt_handler)

    # ── Context & Synthesis Delegation ─────────────────────────────────

    def synthesize_context_cuda_sync(self, turns: List[Dict[str, str]]) -> Optional[str]:
        return ContextCurator.synthesize_context_cuda_sync(self, turns)

    async def _async_synthesize_cuda(self):
        return await ContextCurator.async_synthesize_cuda(self)

    def trigger_background_synthesis(self):
        return ContextCurator.trigger_background_synthesis(self)

    def summarize_history_sync(self, turns: List[Dict[str, str]]) -> str:
        return ContextCurator.summarize_history_sync(self, turns)

    def compact_history(self, keep_recent: Optional[int] = None, silent: bool = False) -> int:
        """keep_recent is in turns; the curator works on history entries (2 per turn)."""
        keep_entries = self._recent_entries if keep_recent is None else max(keep_recent, 0) * 2
        return ContextCurator.compact_history(self, keep_recent=keep_entries, silent=silent)

    def target_name(self, target: str) -> str:
        tgt = self.resolve_target_alias(target)
        if tgt in self.endpoints:
            return self.endpoints[tgt]["name"]
        return "Google Cloud"

    def get_context_metrics(self, target: Optional[str] = None) -> Dict[str, Any]:
        return ContextCurator.get_context_metrics(self, target)

    def format_context_badge(self, target: Optional[str] = None, reset_col: str = UI.RST) -> str:
        return ContextCurator.format_context_badge(self, target, reset_col)

    def format_context_pill(self, target: Optional[str] = None) -> str:
        return ContextCurator.format_context_pill(self, target)

    def print_context_summary(self):
        return ContextCurator.print_context_summary(self)

    # ── Subagent Delegation & AST Repo Map ─────────────────────────────

    def spawn_subagent(self, task_description: str, command: Optional[str] = None) -> SubagentTask:
        return SubagentManager.spawn(task_description, self, command=command)

    def run_build_subagent(self, custom_cmd: Optional[str] = None, repo_root: str = ".") -> SubagentTask:
        """Pushes project build & test verification to the dedicated NVIDIA RTX 4060 Subagent (:9001)."""
        from ..agents.brainstorm import BrainstormWorkflow
        build_cmd = custom_cmd or BrainstormWorkflow._detect_test_command(repo_root) or "make test"
        desc = f"Build & Test Verification: execute '{build_cmd}' in {repo_root}"
        task = self.spawn_subagent(desc, command=build_cmd)
        return task

    def list_subagents(self) -> List[SubagentTask]:
        return SubagentManager.list_tasks()

    def get_subagent(self, task_id: int) -> Optional[SubagentTask]:
        return SubagentManager.get(task_id)

    def cancel_subagent(self, task_id: int) -> bool:
        return SubagentManager.cancel(task_id)

    def inject_subagent(self, task_id: int) -> bool:
        return SubagentManager.inject(task_id, self)

    def inject_all_subagents(self) -> int:
        return SubagentManager.inject_all(self)

    def print_subagents(self):
        return SubagentManager.print_tasks(self)

    def print_subagent_detail(self, task_id: int):
        return SubagentManager.print_detail(task_id)

    def get_repomap(self, root_dir: str = ".") -> str:
        return RepoMap.generate_compact_markdown(root_dir)

    def print_repomap(self, root_dir: str = "."):
        return RepoMap.print_tree(root_dir)

    def refresh_repomap(self, root_dir: str = ".") -> str:
        RepoMap.invalidate_cache()
        self._invalidate_local_agent()
        return RepoMap.get_cached_map(root_dir)

    # ── Tool Hooks & System Context ───────────────────────────────────

    def _pop_pending_tool(self, res: Any) -> Dict[str, Any]:
        """Matches a finished tool result to the call recorded when it started."""
        rid = getattr(res, "id", None)
        raw_name = getattr(res, "name", "")
        rname = str(getattr(raw_name, "value", raw_name) or "")
        if rid is not None:
            for i, p in enumerate(self._tool_pending):
                if p["id"] == rid:
                    return self._tool_pending.pop(i)
        if rname:
            for i, p in enumerate(self._tool_pending):
                if p["name"] == rname:
                    return self._tool_pending.pop(i)
        if self._tool_pending:
            return self._tool_pending.pop(0)
        return {"name": rname or "tool", "summary": "", "t0": None, "step": 0}

    def build_tool_hooks(self) -> List[Any]:
        """Creates async PreToolCallDecideHook and PostToolCallHook for live step tracking,
        loop detection, and step budget protection."""
        @hooks.pre_tool_call_decide
        def loop_guard(call: types.ToolCall) -> types.HookResult:
            self._current_turn_tool_count += 1
            c_name = getattr(call, "name", str(call))
            c_args = getattr(call, "args", {}) or {}

            summary = (
                c_args.get("CommandLine")
                or c_args.get("AbsolutePath")
                or c_args.get("TargetFile")
                or c_args.get("file_path")
                or c_args.get("Query")
                or c_args.get("prompt")
                or c_args.get("Instruction")
                or c_args.get("task")
                or ""
            )
            summary_clean = " ".join(str(summary).split())

            if not self.json_output and self.verbose and c_args:
                try:
                    for line in json.dumps(c_args, indent=2, ensure_ascii=False).splitlines():
                        say.detail(f"    {line}")
                except Exception:
                    pass

            # Consecutive duplicate loop detection
            try:
                sig = (c_name, json.dumps(c_args, sort_keys=True))
            except Exception:
                sig = (c_name, str(c_args))

            dup_count = sum(1 for s in self._recent_tool_calls[-2:] if s == sig)
            if dup_count >= 2:
                msg = (
                    f"Execution blocked: Loop detected. Tool '{c_name}' was called with identical arguments "
                    f"3 times consecutively. You must stop calling tools and provide your final response to the user now."
                )
                if not self.json_output:
                    say.warn(f"Loop guard: blocked repeated {c_name} call (3x identical), forcing the model to respond")
                return types.HookResult(allow=False, message=msg)

            self._recent_tool_calls.append(sig)
            self._turn_executed_tools.append({"name": c_name, "args": c_args})
            self._last_tool_call = {"name": c_name, "args": c_args}

            # Max steps budget check
            if self.max_tool_steps_per_turn > 0 and self._current_turn_tool_count > self.max_tool_steps_per_turn:
                msg = (
                    f"Maximum tool call limit of {self.max_tool_steps_per_turn} steps reached for this turn. "
                    f"Do not call any more tools. Conclude your turn and provide your final response to the user now."
                )
                if not self.json_output:
                    say.warn(f"Step limit reached ({self.max_tool_steps_per_turn} steps), forcing a final response")
                return types.HookResult(allow=False, message=msg)

            self._tool_pending.append({
                "id": getattr(call, "id", None), "name": str(getattr(c_name, "value", c_name)), "summary": summary_clean,
                "t0": time.perf_counter(), "step": self._current_turn_tool_count,
            })
            return types.HookResult(allow=True)

        @hooks.post_tool_call
        def on_tool_result(res: types.ToolResult):
            pending = self._pop_pending_tool(res)
            if self.json_output:
                return
            elapsed = (time.perf_counter() - pending["t0"]) if pending.get("t0") else None
            step_tag = f"step {pending['step']}/{self.max_tool_steps_per_turn}" if (self.max_tool_steps_per_turn > 0 and pending.get("step")) else ""

            if res.error:
                err_msg = " ".join(str(res.error).split())
                say.tool_row(pending["name"], pending["summary"], ok=False, elapsed=elapsed, extra=err_msg[:40])
                say.detail(f"    error: {err_msg[:400]}")
                return

            out_str = ""
            if isinstance(res.result, dict):
                out_str = str(res.result.get("output") or res.result.get("content") or res.result)
            elif isinstance(res.result, str):
                out_str = res.result
            elif res.result is not None:
                out_str = str(res.result)

            lines = [l for l in out_str.strip().splitlines() if l.strip()]
            extra = " · ".join(x for x in (f"{len(lines)} lines" if lines else "", step_tag) if x)
            say.tool_row(pending["name"], pending["summary"], ok=True, elapsed=elapsed, extra=extra)

            if self.verbose and lines:
                max_line = max(60, shutil.get_terminal_size((120, 24)).columns - 8)
                shown, hidden = say.truncate_lines("\n".join(lines), 10)
                for l in shown.splitlines():
                    say.detail("    " + (l if len(l) <= max_line else l[:max_line - 3] + "..."))
                if hidden:
                    say.detail(f"    +{hidden} lines")

        all_hooks = [loop_guard, on_tool_result]

        # External lifecycle hooks (hooks.json)
        if self.enable_hooks:
            ext_hooks = ExternalHooksManager.build_lifecycle_hooks(self, ".")
            all_hooks.extend(ext_hooks)

        return all_hooks

    def build_system_context(self, target: str, include_history: bool = True) -> str:
        base_inst = (
            "You are Antigravity, an expert software engineering autonomous agent.\n"
            "You have full access to codebase inspection, editing, and execution tools.\n\n"
            "CRITICAL OPERATIONAL RULES:\n"
            "1. NO MONOLOGUE / NO PREAMBLE: When you need to read or search files, execute commands, or inspect code, "
            "DO NOT write conversational sentences such as 'Let me check...', 'Now I will inspect...', 'Let me confirm...', "
            "or 'I will now run...'. Emit the tool calls IMMEDIATELY. Explanations before tool calls are strictly forbidden.\n"
            "2. DIRECT ACTION: Analyze findings silently and proceed directly to the necessary action. Every turn must make tangible "
            "progress with real tool calls or a concise final resolution.\n"
            "3. NO REDUNDANT CALLS: Do not re-run the same command or re-read the same file if you already have the result.\n"
            "4. WEB RESEARCH: When dealing with unfamiliar external libraries, modern API deprecations, or unexpected errors, "
            "search the web (e.g. search_web, read_url_content) or inspect online documentation before guessing.\n"
            "5. ERROR HANDLING: If a tool fails or reports an error, inspect the output and correct your approach in the next tool call.\n"
            "6. COMPLETION: Once your task is finished, output a crisp, formatted markdown summary of your findings or changes, and STOP calling tools."
        )

        # Automatic project instructions injection (GEMINI.md, antigravity.md, etc.)
        if self.auto_instructions:
            proj_instr, loaded_files = ProjectInstructions.load_instructions(".")
            if proj_instr:
                base_inst += (
                    f"\n\n[Project Guidelines & Context Files ({', '.join(loaded_files)})]\n"
                    f"{proj_instr}\n"
                    f"[End of Project Guidelines]\n"
                )

        # Automatic repository symbol map injection
        if self.auto_repomap:
            repomap_text = RepoMap.get_cached_map(".")
            if repomap_text:
                base_inst += (
                    f"\n\n[Codebase Architecture & AST Symbol Map (Auto-Discovered)]\n"
                    f"{repomap_text}\n"
                    f"[End of Codebase Symbol Map]\n"
                )

        # Automatic Inbuilt Kontexta Context & Knowledge Engine injection
        if self.enable_kontexta:
            kxta_prompt = KontextaManager.generate_context_prompt(".")
            if kxta_prompt:
                base_inst += f"\n\n{kxta_prompt}\n"

        if not self.history or not include_history:
            return base_inst

        # Sliding window: keep the last max_recent_turns verbatim, summarize earlier turns
        history_lines = ["\n[Conversation History]"]
        if len(self.history) > self._recent_entries:
            older = self.history[:-self._recent_entries]
            recent = self.history[-self._recent_entries:]
            condensed = self.summarize_history_sync(older)
            history_lines.append("[Prior Context Summary (Condensed)]")
            history_lines.append(condensed)
            history_lines.append("\n[Recent Conversation Turns]")
            for turn in recent:
                tag = f"[{turn['target'].upper()} {turn['role'].upper()}]"
                history_lines.append(f"{tag}: {turn['content']}")
        else:
            for turn in self.history:
                tag = f"[{turn['target'].upper()} {turn['role'].upper()}]"
                history_lines.append(f"{tag}: {turn['content']}")

        history_lines.append("[End of Conversation History]\n")
        return base_inst + "\n" + "\n".join(history_lines)

    # ── Session Persistence ───────────────────────────────────────────

    def save_session(self, name: Optional[str] = None) -> str:
        """Saves current conversation history, active target, and state to disk."""
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        if not name:
            name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not name.endswith(".json"):
            name += ".json"
        filepath = os.path.join(SESSIONS_DIR, name)
        data = {
            "version": 1,
            "saved_at": datetime.now().isoformat(),
            "active_target": self.active_target,
            "cloud_model": self.cloud_model,
            "turns_count": len(self.history) // 2,
            "history": self.history,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return filepath

    def load_session(self, name: str) -> bool:
        """Loads a saved session file, restoring conversation history and routing."""
        target_path = name
        if not os.path.exists(target_path):
            if not target_path.endswith(".json"):
                target_path += ".json"
            target_path = os.path.join(SESSIONS_DIR, target_path)

        if not os.path.exists(target_path):
            return False

        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.history = data.get("history", [])
        if "active_target" in data:
            self.active_target = self.resolve_target_alias(data["active_target"])
        self._invalidate_local_agent()
        return True

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Lists all saved sessions in the sessions directory."""
        if not os.path.exists(SESSIONS_DIR):
            return []
        sessions = []
        for fn in sorted(os.listdir(SESSIONS_DIR), reverse=True):
            if fn.endswith(".json"):
                fp = os.path.join(SESSIONS_DIR, fn)
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    sessions.append({
                        "file": fn,
                        "saved_at": data.get("saved_at", "Unknown"),
                        "turns": data.get("turns_count", len(data.get("history", [])) // 2),
                        "target": data.get("active_target", "auto"),
                        "size_kb": round(os.path.getsize(fp) / 1024, 1),
                    })
                except Exception:
                    pass
        sessions.sort(key=lambda x: x["saved_at"] if x["saved_at"] != "Unknown" else "", reverse=True)
        return sessions

    def export_markdown(self, filepath: Optional[str] = None) -> str:
        """Exports the active conversation transcript to a readable markdown file."""
        if not filepath:
            filepath = os.path.expanduser(f"~/session_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        lines = [
            "# Antigravity Hybrid Session Transcript",
            f"*Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            f"*Total Turns: {len(self.history) // 2}*",
            "",
            "---",
            "",
        ]
        for turn in self.history:
            role = turn["role"].capitalize()
            target = turn["target"].upper()
            lines.append(f"### {role} ({target})\n")
            lines.append(turn["content"].strip() + "\n")
            lines.append("---\n")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return filepath

    # ── Agent Management & Execution ───────────────────────────────────

    async def get_or_switch_local_agent(self, target: str) -> Agent:
        target = self.resolve_target_alias(target)
        await self._close_stale_agents()
        await asyncio.to_thread(self.refresh_endpoints)
        ep = self.endpoints[target]
        live_model = ep["model"]

        if not live_model:
            det = await asyncio.to_thread(detect_port_model, ep["port"], ep["url"])
            if det["model_path"]:
                live_model = det["model_path"]
                ep["model"] = live_model
                ep["filename"] = det["filename"]
                ep["name_friendly"] = det["name"]
                ep["status"] = "online"

        if not live_model:
            raise RuntimeError(
                f"No active model detected on Port {target} ({ep['name']}). "
                f"Use '/switch <model_id>' or check llama-shift on Port 8002."
            )

        if (
            self._active_local_agent is not None
            and self._current_agent_target == target
            and self._active_local_model == live_model
        ):
            return self._active_local_agent

        if self._active_local_agent is not None:
            if self._active_local_model != live_model:
                say.step(f"model switched on {ep['name']}: {ep['filename']}")
            await self.reset_local_agent()

        system_instructions = await asyncio.to_thread(self.build_system_context, target)
        config = LocalOpenAIAgentConfig(
            model=ep["model"],
            base_url=ep["url"],
            system_instructions=system_instructions,
            capabilities=CapabilitiesConfig(),
            policies=self.get_policies(),
            hooks=self.build_tool_hooks(),
            mcp_servers=self._mcp_servers,
        )

        new_agent = Agent(config)
        await new_agent.__aenter__()

        self._active_local_agent = new_agent
        self._current_agent_target = target
        self._active_local_model = live_model
        return self._active_local_agent

    async def _consume_token_stream(self, token_stream, status_msg: str = "Thinking & generating response...") -> tuple[str, int, Optional[float]]:
        """Consumes an async token stream, rendering Markdown live when rich is available."""
        full_response: List[str] = []
        token_count = 0
        t_first = None
        status = console.status(f"[bold cyan]{status_msg}[/bold cyan]", spinner="dots") if (RICH_AVAILABLE and not self.json_output) else None
        if status:
            status.start()
        renderer = StreamRenderer(markdown=self.render_markdown, enabled=not self.json_output)

        try:
            async for token in token_stream:
                if t_first is None:
                    t_first = time.perf_counter()
                    if status:
                        status.stop()
                        status = None
                full_response.append(token)
                token_count += 1
                renderer.feed(token)
        finally:
            if status:
                status.stop()
            renderer.stop()

        return "".join(full_response).strip(), token_count, t_first

    async def chat_cloud_oauth(self, prompt: str, with_project_context: bool = True) -> str:
        """Executes a cloud turn using the existing Google OAuth session via agy CLI."""
        context_prefix = ""
        if self.history:
            lines = []
            recent = self.history[-self._recent_entries:] if self._recent_entries else []
            # Keep the distilled working memory if it fell outside the recent window
            first = self.history[0]
            if first not in recent and "[SYSTEM: Context compacted" in first.get("content", ""):
                lines.append(f"[WORKING MEMORY]: {first['content'].strip()[:3000]}")
            for t in recent:
                role = t.get("role", "").upper()
                content = t.get("content", "").strip()
                lines.append(f"[{role}]: {content[:1500]}")
            context_prefix = "\n".join(lines) + "\n\n"

        project_ctx = self._cloud_project_context() if with_project_context else ""
        full_prompt = f"{project_ctx}{context_prefix}[USER]: {prompt}" if (context_prefix or project_ctx) else prompt

        IDLE_TIMEOUT_SECS = 300
        MAX_TOTAL_SECS = 7200

        cmd = [
            "agy",
            "--model", self.cloud_model,
            "--print-timeout", "0",
            "--output-format", "stream-json",
        ]
        if self._dangerously_skip_permissions:
            cmd.append("--dangerously-skip-permissions")
        cmd.extend(["--print", full_prompt])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        status = console.status(f"[bold #60a5fa]Contacting Google Cloud ({self.cloud_model})...[/bold #60a5fa]", spinner="dots") if (RICH_AVAILABLE and not self.json_output) else None
        if status:
            status.start()
        elif not self.json_output:
            say.step(f"contacting Google Cloud ({self.cloud_model})")

        stderr_lines = []
        async def drain_stderr():
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                stderr_lines.append(line.decode("utf-8", errors="replace"))

        stderr_task = asyncio.create_task(drain_stderr())
        collected_text = []
        cleared_spinner = False
        renderer = StreamRenderer(markdown=self.render_markdown, enabled=not self.json_output)
        t_turn_start = time.perf_counter()

        async def read_stream():
            nonlocal cleared_spinner, status
            while True:
                elapsed = time.perf_counter() - t_turn_start
                if elapsed > MAX_TOTAL_SECS:
                    raise asyncio.TimeoutError(f"Cloud turn exceeded maximum limit ({MAX_TOTAL_SECS}s)")

                try:
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=IDLE_TIMEOUT_SECS)
                except asyncio.TimeoutError:
                    raise asyncio.TimeoutError(f"Cloud model stalled (no activity for {IDLE_TIMEOUT_SECS}s)")

                if not line:
                    break
                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                except Exception:
                    continue

                event = data.get("event")
                if event == "step_update":
                    su = data.get("step_update", {})
                    stype = su.get("step_type")
                    state = su.get("state")

                    if stype == "tool" and state == "ACTIVE" and not self.json_output:
                        renderer.stop()
                        tool_name = su.get("tool_name", "tool")
                        params = su.get("tool_info", {}).get("parameters", {})
                        summary = (
                            params.get("CommandLine")
                            or params.get("AbsolutePath")
                            or params.get("TargetFile")
                            or params.get("file_path")
                            or params.get("Query")
                            or params.get("prompt")
                            or params.get("Instruction")
                            or params.get("task")
                            or ""
                        )
                        summary_clean = " ".join(str(summary).split())
                        if status:
                            clip = summary_clean if len(summary_clean) <= 90 else summary_clean[:87] + "..."
                            status.update(f"[bold yellow]⚙ {tool_name}[/bold yellow] [dim]{clip.replace('[', chr(92) + '[')}[/dim]")
                        else:
                            sys.stdout.write("\r\033[K")
                            sys.stdout.flush()
                            say.tool_row(f"cloud:{tool_name}", summary_clean)

                    elif stype == "agent_response":
                        delta = su.get("text_delta")
                        if delta:
                            if status:
                                status.stop()
                                status = None
                            if not cleared_spinner and not self.json_output:
                                sys.stdout.write("\r\033[K")
                                sys.stdout.flush()
                                cleared_spinner = True
                            collected_text.append(delta)
                            renderer.feed(delta)

                elif event == "result":
                    pass

        try:
            await read_stream()
            await stderr_task
            await proc.wait()
        except asyncio.CancelledError:
            # ESC/Ctrl+C: don't leave the agy subprocess running in the background
            stderr_task.cancel()
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except BaseException:
                pass
            if status:
                status.stop()
            renderer.stop()
            raise
        except asyncio.TimeoutError as te:
            stderr_task.cancel()
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            if status:
                status.stop()
            renderer.stop()
            if not cleared_spinner and not self.json_output:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
            raise RuntimeError(
                f"Cloud model timed out ({te}) — Laya will escalate to local GPU"
            )

        if status:
            status.stop()
        renderer.stop()
        if not cleared_spinner and not self.json_output:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

        combined_out = "".join(collected_text).strip()
        err_decoded = "".join(stderr_lines).strip()

        if not combined_out:
            if err_decoded:
                raise RuntimeError(f"Cloud model unreachable: {err_decoded}")
            elif proc.returncode != 0:
                raise RuntimeError(f"Cloud provider returned error code {proc.returncode}")

        return combined_out

    def _validate_target(self, target: str) -> str:
        if target != "cloud" and target not in self.endpoints:
            raise ValueError(f"Unknown target '{target}'. Use auto, 9000 (rocm), 9001 (cuda) or cloud.")
        return target

    async def _route_auto(self, prompt: str) -> str:
        """Laya System 1 decision pass for target 'auto'; returns a concrete target id."""
        if self.auto_git_shift:
            is_git, git_conf = self.laya.is_git_action(prompt)
            if is_git and self.endpoints["9000"]["status"] == "online":
                say.step(f"routed to ROCm (git action, laya {git_conf * 100:.0f}%)")
                say.detail(f"Laya System 1 (~33ms): Git Action -> AMD ROCm (Gemma 4 A4B Fast MoE), confidence {git_conf * 100:.1f}%")
                return "9000"
        is_build, build_conf = self.laya.is_build_action(prompt)
        if is_build:
            say.step(f"routed to CUDA (build and test, laya {build_conf * 100:.0f}%)")
            say.detail(f"Laya System 1 (~33ms): Build & Test Verification -> NVIDIA CUDA Subagent (:9001), confidence {build_conf * 100:.1f}%")
            return "9001"
        routed_id, conf = self.laya.route_prompt(prompt)
        if routed_id == "cloud":
            short, routed_name = "Cloud", f"Google Cloud ({self.cloud_model})"
        elif routed_id in self.endpoints:
            ep = self.endpoints[routed_id]
            fn = ep.get("filename") or ep.get("name_friendly") or "Unknown Model"
            short, routed_name = ep["name"].split(" ")[0], f"{ep['desc']} [{fn}]"
        else:
            short = routed_name = routed_id
        say.step(f"routed to {short} (laya {conf * 100:.0f}%)")
        say.detail(f"Laya System 1 (~33ms): 'choice' -> {routed_name}, confidence {conf * 100:.1f}%")
        return routed_id

    async def _autoshift_rocm_model(self, prompt: str):
        """Hot-swaps the :9000 model (Gemma 4 MoE for git, Qwen Dense for heavy reasoning) when needed."""
        ep_9000 = self.endpoints.get("9000", {})
        current = " ".join((ep_9000.get(k) or "").lower() for k in ("id", "filename", "name_friendly"))
        is_git, git_conf = self.laya.is_git_action(prompt)
        if is_git:
            if "gemma" in current:
                return
            say.step("git action: swapping :9000 to Gemma 4 A4B")
            say.detail(f"Laya Auto-Shift: git action detected (P={git_conf*100:.1f}%), hot-swapping for rapid execution")
            wanted = self.fast_git_model
        else:
            is_heavy, heavy_conf = self.laya.is_heavy_reasoning(prompt)
            if not is_heavy or "qwen" in current:
                return
            catalog = await asyncio.to_thread(self.get_catalog)
            if not any(self.dense_code_model.lower() in m.get("id", "").lower() for m in catalog):
                return
            say.step("deep reasoning: swapping :9000 to Qwen 27B Dense")
            say.detail(f"Laya Auto-Shift: deep reasoning detected (P={heavy_conf*100:.1f}%)")
            wanted = self.dense_code_model
        if await self.switch_model(wanted):
            await asyncio.to_thread(self.refresh_endpoints)

    def _escalation_label(self, target: str) -> str:
        if target == "cloud":
            return f"Cloud ({self.cloud_model})"
        ep = self.endpoints[target]
        return f"{ep['name']} [{ep['filename']}]"

    def _print_local_metrics(self, target: str, tokens: int, tps: float, ttft_ms: float, total_dur: float):
        if not self.show_metrics or self.json_output or tokens <= 0:
            return
        ctx_badge = self.format_context_badge(target, reset_col=UI.GRAY)
        say.dim(f"  {tps:.1f} tok/s · {tokens} tokens · TTFT {ttft_ms:.0f}ms · {total_dur:.2f}s · {ctx_badge}")

    async def _run_escalation(self, escalate_to: str, prompt: str):
        """Runs the prompt on the escalation target; returns (text, tokens, tps, ttft_ms, total_s, final_target)."""
        if escalate_to == "cloud":
            t_start = time.perf_counter()
            text = await self.chat_cloud_oauth(prompt)
            total = time.perf_counter() - t_start
            if self.show_metrics and not self.json_output:
                ctx_badge = self.format_context_badge("cloud", reset_col=UI.GRAY)
                say.dim(f"  cloud · {total:.2f}s · {ctx_badge}")
            return text, 0, None, None, total, "cloud"

        agent = await self.get_or_switch_local_agent(escalate_to)
        t_start = time.perf_counter()
        stream = await agent.chat(prompt)
        text, tokens, t_first = await self._consume_token_stream(stream)
        t_end = time.perf_counter()
        ttft_ms = ((t_first - t_start) * 1000) if t_first else 0
        gen_dur = t_end - (t_first or t_start)
        tps = (tokens / gen_dur) if gen_dur > 0 else 0
        total = t_end - t_start
        self._print_local_metrics(escalate_to, tokens, tps, ttft_ms, total)
        return text, tokens, tps, ttft_ms, total, escalate_to

    async def _chat_impl(self, prompt: str, force_target: Optional[str] = None) -> str:
        requested = self.resolve_target_alias(force_target or self.active_target)
        target = requested if requested == "auto" else self._validate_target(requested)

        # Dynamic check of ports before routing
        await asyncio.to_thread(self.refresh_endpoints)

        # Telemetry & metric trackers
        self._current_turn_tool_count = 0
        self._recent_tool_calls.clear()
        self._tool_pending.clear()
        self._ctx_cache = (0.0, None)
        executed_tools: List[Dict[str, Any]] = []
        self._turn_executed_tools = executed_tools
        token_count: int = 0
        tps: Optional[float] = None
        ttft_ms: Optional[float] = None
        total_dur: Optional[float] = None

        if target == "auto":
            target = self._validate_target(await self._route_auto(prompt))

        if self.auto_git_shift and target == "9000":
            await self._autoshift_rocm_model(prompt)

        initial_target = target
        self._turn_target = target

        if target == "cloud":
            auth_info = "OAuth token" if self.has_oauth else "API key"
            if not self.json_output:
                say.turn_header(f"Gemini {self.cloud_model} · {auth_info}", "cloud")
            else:
                say.step(f"Google Cloud (Gemini: {self.cloud_model}) thinking and responding")
            t_cloud_start = time.perf_counter()
            try:
                response_text = await self.chat_cloud_oauth(prompt)
                dur = time.perf_counter() - t_cloud_start
                total_dur = dur
                if self.show_metrics and not self.json_output:
                    approx_words = len(response_text.split())
                    ctx_badge = self.format_context_badge(target, reset_col=UI.GRAY)
                    say.dim(f"  cloud · {dur:.2f}s · ~{approx_words} words · {ctx_badge}")
            except Exception as cloud_err:
                say.warn(f"Cloud model unreachable: {cloud_err}")
                candidates = []
                if self.endpoints["9000"]["status"] == "online":
                    candidates.append("rocm")
                if self.endpoints["9001"]["status"] == "online":
                    candidates.append("cuda")
                if not candidates:
                    raise
                winner, conf = self.laya.choose_escalation_target("cloud", prompt, candidates)
                escalate_to = self._validate_target(self.resolve_target_alias(winner))
                say.step(f"escalating to {self._escalation_label(escalate_to)} (laya {conf*100:.0f}%)")
                response_text, token_count, tps, ttft_ms, total_dur, target = await self._run_escalation(escalate_to, prompt)
                self._turn_target = target
        else:
            ep = self.endpoints[target]
            display_fn = ep["filename"] or ep["model"].split("/")[-1]
            header = f"{ep['name']} · {display_fn}"
            if not self.json_output:
                say.turn_header(header, target)
            else:
                say.step(f"{header} thinking and responding")

            agent = await self.get_or_switch_local_agent(target)

            def _is_empty_turn_400(e: Exception) -> bool:
                s = str(e)
                return "400" in s and ("content" in s or "tool_calls" in s)

            def _is_exceed_context_error(e: Exception) -> bool:
                s = str(e).lower()
                return "exceed_context_size_error" in s or "exceeds the available context size" in s

            def _is_unreachable_error(e: Exception) -> bool:
                s = str(e).lower()
                triggers = [
                    "connection refused", "connecterror", "unreachable", "cannot connect",
                    "failed to connect", "timed out", "timeout", "remote disconnected",
                    "service unavailable", "502", "503", "504"
                ]
                return any(t in s for t in triggers)

            def _prune_empty_history():
                self.history = [
                    h for h in self.history
                    if h.get("content") and h["content"].strip()
                ]

            MAX_ATTEMPTS = 3
            response_text = ""

            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    t_start = time.perf_counter()
                    t_first = None
                    token_count = 0
                    current_response = await agent.chat(prompt)

                    response_text, token_count, t_first = await self._consume_token_stream(current_response)
                    t_end = time.perf_counter()

                    ttft_ms = ((t_first - t_start) * 1000) if t_first else 0
                    gen_dur = t_end - (t_first or t_start)
                    tps = (token_count / gen_dur) if gen_dur > 0 else 0
                    total_dur = t_end - t_start

                    self._print_local_metrics(target, token_count, tps, ttft_ms, total_dur)
                    break

                except Exception as turn_err:
                    can_retry = attempt < MAX_ATTEMPTS
                    if _is_exceed_context_error(turn_err) and can_retry:
                        say.warn("Context size exceeded, auto-compacting conversation history")
                        removed = await asyncio.to_thread(self.compact_history, keep_recent=2, silent=True)
                        if removed == 0 and len(self.history) > 2:
                            removed = await asyncio.to_thread(self.compact_history, keep_recent=1, silent=True)
                        await self.reset_local_agent()
                        say.step(f"compacted {removed} turn(s), retrying with a fresh context window")
                        agent = await self.get_or_switch_local_agent(target)
                    elif _is_empty_turn_400(turn_err) and can_retry:
                        say.warn(
                            f"Model produced an empty turn after tool execution "
                            f"(HTTP 400, attempt {attempt}/{MAX_ATTEMPTS}), pruning history and retrying"
                        )
                        _prune_empty_history()
                        await self.reset_local_agent()
                        agent = await self.get_or_switch_local_agent(target)
                    elif _is_empty_turn_400(turn_err) or _is_unreachable_error(turn_err) or _is_exceed_context_error(turn_err):
                        _prune_empty_history()
                        await self.reset_local_agent()
                        await asyncio.to_thread(self.refresh_endpoints)

                        candidates = []
                        if target != "9000" and self.endpoints["9000"]["status"] == "online":
                            candidates.append("rocm")
                        if target != "9001" and self.endpoints["9001"]["status"] == "online":
                            candidates.append("cuda")
                        candidates.append("cloud")

                        winner, conf = self.laya.choose_escalation_target(target, prompt, candidates)
                        escalate_to = self._validate_target(self.resolve_target_alias(winner))

                        if _is_unreachable_error(turn_err):
                            cause = "Endpoint unreachable"
                        elif _is_exceed_context_error(turn_err):
                            cause = "Context window still exceeded after compaction"
                        else:
                            cause = f"Failed after {MAX_ATTEMPTS} attempts"
                        say.warn(f"{target.upper()}: {cause}")
                        say.step(f"escalating to {self._escalation_label(escalate_to)} (laya {conf*100:.0f}%)")
                        response_text, token_count, tps, ttft_ms, total_dur, target = await self._run_escalation(escalate_to, prompt)
                        self._turn_target = target
                        break
                    else:
                        raise

        self.history.append({"role": "user", "target": target, "content": prompt})
        if response_text:
            self.history.append({"role": "assistant", "target": target, "content": response_text})
            self.last_answer = response_text
        else:
            if not self.json_output:
                say.warn("Model returned an empty response, skipping history entry to prevent API errors")

        # Proactive autocompact
        if self.autocompact_threshold > 0 and len(self.history) // 2 >= self.autocompact_threshold:
            removed = await asyncio.to_thread(self.compact_history, silent=True)
            if removed > 0:
                say.step(f"auto-compacted {removed} old turns into a summary (threshold {self.autocompact_threshold})")
        else:
            self.trigger_background_synthesis()

        approx_tokens = token_count if token_count else len(response_text.split())
        if response_text:
            self._record_stats(initial_target, target, approx_tokens, tps, ttft_ms, total_dur)
        else:
            self._record_error(target)

        if self.json_output:
            json_payload = {
                "status": "success" if response_text else "empty_response",
                "target": target,
                "target_name": self.endpoints[target]["name"] if target in self.endpoints else "Google Cloud",
                "model": self.endpoints[target]["filename"] if target in self.endpoints else self.cloud_model,
                "prompt": prompt,
                "response": response_text,
                "tool_calls": executed_tools,
                "metrics": {
                    "tokens": token_count,
                    "tok_per_sec": round(tps, 2) if tps is not None else None,
                    "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
                    "duration_s": round(total_dur, 2) if total_dur is not None else None,
                },
            }
            print(json.dumps(json_payload, indent=2, ensure_ascii=False))
        return response_text

    MAX_STOP_RETRIES = 2

    async def chat(self, prompt: str, force_target: Optional[str] = None) -> str:
        """One user turn: PreTurn hooks, model run, checkpoint, PostTurn/Stop hooks."""
        self._turn_target = None
        if self.enable_hooks:
            verdict = await asyncio.to_thread(ExternalHooksManager.run_turn_event, self, "PreTurn", {"prompt": prompt})
            if verdict.get("deny"):
                msg = f"Turn blocked by PreTurn hook: {verdict.get('reason') or 'denied'}"
                self._notice(UI.warn(msg) if not self.json_output else msg)
                return ""
            prompt = verdict.get("prompt") or prompt

        before = await self._checkpoint_before()
        t_turn = time.perf_counter()
        response = ""
        try:
            response = await self._chat_impl(prompt, force_target) or ""
            if self.enable_hooks:
                payload = {"prompt": prompt, "response": response}
                await asyncio.to_thread(ExternalHooksManager.run_turn_event, self, "PostTurn", payload)
                for _ in range(self.MAX_STOP_RETRIES):
                    stop = await asyncio.to_thread(
                        ExternalHooksManager.run_turn_event, self, "Stop", {"prompt": prompt, "response": response},
                    )
                    if not stop.get("block"):
                        break
                    note = f"Stop hook requires more work: {stop['reason']}"
                    self._notice(UI.warn(note) if not self.json_output else note)
                    response = await self._chat_impl(stop["reason"], force_target) or response
        except Exception:
            self._record_error(self._error_target(force_target))
            raise
        finally:
            cp = await self._checkpoint_after(before, prompt)
            await self._print_receipt(cp, time.perf_counter() - t_turn)
            await self._sync_commit_to_kontexta()
        return response

    async def _print_receipt(self, cp: Optional[Dict[str, Any]], elapsed: float):
        """Shows what the turn actually did (files, commands) under the answer; never fails the turn."""
        if self.json_output:
            return
        try:
            files = []
            if cp and self._repo_root:
                raw = await asyncio.to_thread(Checkpoints.diff, self._repo_root, cp["before"], cp["after"], False, True)
                files = receipt.files_from_numstat(raw)
            commands = []
            for call in self._turn_executed_tools:
                args = call.get("args") or {}
                cmd = next((args[k] for k in ("command", "CommandLine", "cmd") if isinstance(args.get(k), str)), None)
                if cmd:
                    commands.append(cmd)
            receipt.render_receipt(files, commands, None, elapsed, getattr(self, "_last_tps", None), self._turn_target or self.active_target)
        except Exception:
            pass

    async def _sync_commit_to_kontexta(self):
        """Journals a commit the agent made this turn; never affects the turn itself."""
        try:
            if not self._repo_root or not any(kontexta_sync.is_commit_call(c) for c in self._turn_executed_tools):
                return
            info = await asyncio.to_thread(kontexta_sync.commit_info, self._repo_root)
            if info:
                kontexta_sync.sync_event(self, "commit", info["subject"], self._repo_root, commit=info, quiet=True)
        except Exception:
            pass

    def _error_target(self, force_target: Optional[str]) -> str:
        target = self._turn_target or self.resolve_target_alias(force_target or self.active_target)
        return target if (target == "cloud" or target in self.endpoints) else "unrouted"

    def _record_error(self, target: str):
        st = self.stats.setdefault(target, {})
        st["errors"] = st.get("errors", 0) + 1

    # ── Checkpoints (/diff, /undo) ────────────────────────────────────

    async def _checkpoint_before(self) -> Optional[str]:
        if self._repo_root is None:
            self._repo_root = await asyncio.to_thread(Checkpoints.repo_root) or ""
        if not self._repo_root:
            return None
        return await asyncio.to_thread(Checkpoints.snapshot, self._repo_root)

    async def _checkpoint_after(self, before: Optional[str], prompt: str) -> Optional[Dict[str, Any]]:
        if not before or not self._repo_root:
            return None
        after = await asyncio.to_thread(Checkpoints.snapshot, self._repo_root)
        if after and after != before:
            cp = {"prompt": prompt.strip().splitlines()[0][:80] if prompt.strip() else "", "before": before, "after": after, "time": time.time()}
            self.checkpoints.append(cp)
            del self.checkpoints[:-Checkpoints.MAX_TURNS]
            return cp
        return None

    def turn_diff(self, nth: int = 1, stat: bool = False) -> Optional[str]:
        """Diff of the nth most recent turn that changed files (1 = latest); None if unavailable."""
        if not self.checkpoints or nth < 1 or nth > len(self.checkpoints):
            return None
        cp = self.checkpoints[-nth]
        return Checkpoints.diff(self._repo_root, cp["before"], cp["after"], stat=stat)

    def undo_plan(self) -> Optional[Dict[str, Any]]:
        """Files that /undo would revert for the latest changed turn, or None."""
        if not self.checkpoints or not self._repo_root:
            return None
        cp = self.checkpoints[-1]
        current = Checkpoints.snapshot(self._repo_root)
        if not current:
            return None
        return {"checkpoint": cp, "files": Checkpoints.changed(self._repo_root, cp["before"], current), "current": current}

    def apply_undo(self, plan: Dict[str, Any]) -> List[str]:
        """Reverts the working tree to just before the planned turn and tells the model."""
        cp = plan["checkpoint"]
        touched = Checkpoints.restore(self._repo_root, cp["before"], plan["current"])
        self.checkpoints.remove(cp)
        self.history.extend([
            {"role": "user", "target": self.active_target, "content": f"[SYSTEM: The user reverted all file changes from the turn \"{cp['prompt']}\" with /undo. Files restored: {', '.join(touched) or 'none'}.]"},
            {"role": "assistant", "target": self.active_target, "content": "[SYSTEM: Acknowledged. Those file changes no longer exist.]"},
        ])
        self._invalidate_local_agent()
        return touched

    # ── Usage ledger (/stats) ─────────────────────────────────────────

    def _record_stats(self, routed: str, final: str, tokens: int, tps: Optional[float], ttft_ms: Optional[float], dur: Optional[float]):
        self._last_tps = tps
        st = self.stats.setdefault(final, {})
        st["turns"] = st.get("turns", 0) + 1
        st["tokens"] = st.get("tokens", 0) + tokens
        st["seconds"] = st.get("seconds", 0.0) + (dur or 0.0)
        if tps:
            st["tps_sum"] = st.get("tps_sum", 0.0) + tps
            st["tps_n"] = st.get("tps_n", 0) + 1
        if ttft_ms:
            st["ttft_sum"] = st.get("ttft_sum", 0.0) + ttft_ms
            st["ttft_n"] = st.get("ttft_n", 0) + 1
        if routed != final:
            origin = self.stats.setdefault(routed, {})
            origin["escalations"] = origin.get("escalations", 0) + 1

    def print_stats(self):
        if not self.stats:
            say.dim("(no turns recorded yet this session)")
            return
        rows = []
        for tgt, st in sorted(self.stats.items()):
            tps = f"{st['tps_sum'] / st['tps_n']:.1f}" if st.get("tps_n") else "-"
            ttft = f"{st['ttft_sum'] / st['ttft_n']:.0f}ms" if st.get("ttft_n") else "-"
            label = self.endpoints[tgt]["name"] if tgt in self.endpoints else ("Cloud" if tgt == "cloud" else tgt)
            rows.append([label, str(int(st.get("turns", 0))), str(int(st.get("tokens", 0))), tps, ttft, f"{st.get('seconds', 0.0):.1f}s", str(int(st.get("escalations", 0))), str(int(st.get("errors", 0)))])
        say.group("Session usage")
        say.table("", ["Target", "Turns", "Tokens", "Avg tok/s", "Avg TTFT", "Time", "Escalated away", "Errors"], rows)
        say.dim("Cloud token counts are word-count estimates.")

    # ── Preferences ───────────────────────────────────────────────────

    def preferences(self) -> Dict[str, Any]:
        out = {k: getattr(self, k, None) for k in PREF_DEFAULTS}
        return {k: v for k, v in out.items() if v is not None}

    def apply_preferences(self, prefs: Dict[str, Any]):
        for key, val in prefs.items():
            if key not in PREF_DEFAULTS:
                continue
            if key == "active_target":
                self.active_target = self.resolve_target_alias(val)
            elif key == "dangerously_skip_permissions":
                self._dangerously_skip_permissions = val
            elif key == "cloud_model" and isinstance(val, str) and val.strip():
                self.cloud_model = val.strip()
            elif key == "theme":
                UI.set_theme(str(val).strip().lower())
            else:
                setattr(self, key, val)

    def save_preferences(self) -> bool:
        return save_prefs(self.preferences())

    def _cloud_project_context(self) -> str:
        """Project rules and symbol map for cloud turns (agy only sees the prompt text we pass)."""
        parts = []
        if self.auto_instructions:
            text, _ = ProjectInstructions.load_instructions(".")
            if text:
                parts.append(f"[Project Guidelines]\n{text[:6000]}\n[End of Project Guidelines]")
        if self.auto_repomap:
            rmap = RepoMap.get_cached_map(".")
            if rmap:
                parts.append(f"[Codebase Symbol Map]\n{rmap[:6000]}\n[End of Symbol Map]")
        return "\n\n".join(parts) + "\n\n" if parts else ""

    async def reset_local_agent(self):
        """Closes and drops the cached local agent so the next turn starts fresh."""
        self._invalidate_local_agent()
        await self._close_stale_agents()

    async def chat_turn(self, prompt: str, target: Optional[str] = None):
        """Runs one chat turn on an explicit target (used by workflows)."""
        return await self.chat(prompt, force_target=target)

    async def close(self):
        if self.history:
            try:
                self.save_session("latest_session")
            except Exception:
                pass
        task = self._synthesis_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await kontexta_sync.drain()
        await self.reset_local_agent()
        SudoManager.cleanup()

    # ── Status, Catalog & Model Switch UI ─────────────────────────────

    def _status_rows(self) -> List[tuple]:
        if self.use_laya_adaptive_permissions:
            perm_desc = f"{UI.LAYA}LAYA ADAPTIVE{UI.RST} (Bayesian 'noul' safe-action gating)"
        elif self._dangerously_skip_permissions:
            perm_desc = f"{UI.AMBER}AUTO-APPROVED{UI.RST} (Dangerously Skip Permissions: ON)"
        else:
            perm_desc = f"{UI.GREEN}SAFE MODE{UI.RST} (Prompt before running commands)"

        if self.active_target == "auto":
            active_label = f"{UI.LAYA_BOLD}⚡ AUTO (Laya System 1: ROCm ✦ Cloud){UI.RST}"
        elif self.active_target == "9000":
            active_label = f"{UI.ROCM_BOLD}ROCm:9000 (AMD ROCm GPU - Primary Executive){UI.RST}"
        elif self.active_target == "9001":
            active_label = f"{UI.CUDA_BOLD}CUDA:9001 (NVIDIA RTX 4060 - Context Compactor){UI.RST}"
        else:
            active_label = f"{UI.CLOUD_BOLD}Google Cloud ({self.cloud_model}){UI.RST}"

        auth_status = f"{UI.GREEN_BOLD}✓ Google OAuth Active{UI.RST} (~/.gemini/antigravity-cli)" if self.has_oauth else f"{UI.AMBER}API Key{UI.RST}"

        def _fmt_model(ep, col):
            st_color = UI.GREEN if ep['status'] == 'online' else UI.RED
            status_badge = f"{st_color}[{ep['status'].upper()}]{UI.RST}"
            model_name = f"{col}{ep['name_friendly'] or ep['filename']}{UI.RST}"
            src = f" {UI.GRAY}[{ep['source']}]{UI.RST}" if ep['source'] != 'none' else ""
            return f"{status_badge} {model_name}{src}"

        def _on_off(enabled, on_text):
            return f"{UI.GREEN}{on_text}{UI.RST}" if enabled else f"{UI.GRAY}Disabled{UI.RST}"

        ml_status = f"{UI.GREEN}ON{UI.RST} (Enter on empty line submits)" if self.multiline_input else f"{UI.GRAY}OFF{UI.RST} (single Enter)"
        _, instr_files = ProjectInstructions.load_instructions(".")
        instr_desc = f"{UI.GREEN}Loaded ({', '.join(instr_files)}){UI.RST}" if instr_files else _on_off(self.auto_instructions, "Active (Scanning root)")
        hook_entries, hook_files = ExternalHooksManager.load_hook_configs(".")
        hook_tag = f"{UI.GREEN}Active ({len(hook_entries)} hooks loaded from {', '.join(hook_files)}){UI.RST}" if hook_entries else _on_off(self.enable_hooks, "Active (Scanning hooks.json)")

        if self.enable_kontexta:
            k_stat = KontextaManager.get_status(".")
            if k_stat["available"]:
                p_info = k_stat["active_project"]["name"] if k_stat["active_project"] else f"{k_stat['projects_count']} projects"
                k_str = f"{UI.GREEN}Active{UI.RST} (Inbuilt MCP 'kxta' ➔ {p_info})"
            else:
                k_str = f"{UI.AMBER}Offline (Node script missing){UI.RST}"
        else:
            k_str = f"{UI.GRAY}Disabled{UI.RST}"

        sudo_badge = f"{UI.GREEN}Active (~{int(SudoManager.time_remaining() // 60)}m cached){UI.RST}" if SudoManager.is_authenticated() else f"{UI.GRAY}Inactive (Prompt on demand){UI.RST}"
        return [
            ("Active Target", active_label),
            ("Permissions", perm_desc),
            ("Laya Decision Engine", f"{UI.LAYA}ModernBERT-large (~421M, ~33ms){UI.RST} {UI.GRAY}[/v1/systemone]{UI.RST}"),
            ("Primary Agent (ROCm :9000)", _fmt_model(self.endpoints['9000'], UI.ROCM)),
            ("Compactor (NVIDIA :9001)", _fmt_model(self.endpoints['9001'], UI.CUDA)),
            ("Continuous Curation", _on_off(self.continuous_curation, "Active (Auto-distills in background)")),
            ("Codebase AST Map", _on_off(self.auto_repomap, "Active (Auto-injected into context)")),
            ("Project Guidelines", instr_desc),
            ("Lifecycle Hooks", hook_tag),
            ("Kontexta Engine", k_str),
            ("Cloud Model", f"{UI.CLOUD}{self.cloud_model}{UI.RST}"),
            ("Cloud Auth", auth_status),
            ("Multi-line Input", ml_status),
            ("Conversation Turns", f"{UI.CYAN}{len(self.history) // 2}{UI.RST}"),
            ("Auto Model Shift", _on_off(self.auto_git_shift, "Active (Git ➔ Gemma 4 A4B │ Deep ➔ Qwen 27B)")),
            ("Sudo Privileges", sudo_badge),
            ("Kontexta Journal Sync", _on_off(self.kontexta_sync, "On (brainstorm results and commits)")),
            ("Theme", f"{UI.CYAN}{UI.theme_name}{UI.RST}"),
        ]

    def print_status(self):
        self.refresh_endpoints()
        rows = self._status_rows()
        say.group("Status")
        say.table("", ["Setting", "Value"], [[label, value] for label, value in rows])

    def get_catalog(self) -> List[Dict[str, Any]]:
        self.refresh_endpoints()
        ls_models = get_llamashift_models()
        catalog: List[Dict[str, Any]] = []

        for m in ls_models:
            port = str(m.get("port", "9000"))
            ep = self.endpoints.get(port)
            is_running = (m.get("status") == "running") or (
                ep is not None and ep.get("status") == "online" and ep.get("filename") == m.get("filename")
            )
            is_active = (self.active_target == port and is_running)

            devs = m.get("devices", [])
            dev_str = ", ".join(devs) if devs else ("ROCm0" if port == "9000" or "AMD" in m.get("gpu", "") else "CUDA0")
            gpu_desc = f"{m.get('gpu', 'Local GPU')} [{dev_str}] :{port}"

            catalog.append({
                "id": m.get("id", ""),
                "name": m.get("name", m.get("id", "")),
                "filename": m.get("filename", ""),
                "port": port,
                "status": "running" if is_running else "stopped",
                "gpu": m.get("gpu", "Local GPU"),
                "gpu_desc": gpu_desc,
                "devices": devs,
                "size": m.get("size", ""),
                "desc": m.get("desc", ""),
                "ctxSize": m.get("ctxSize", 0),
                "type": "local",
                "is_active": is_active,
                "dedicated": m.get("dedicated", False),
            })

        # Cloud Gemini entry
        is_cloud_active = (self.active_target == "cloud")
        catalog.append({
            "id": "cloud",
            "name": f"Gemini ({self.cloud_model})",
            "filename": self.cloud_model,
            "port": "cloud",
            "status": "online",
            "gpu": "Google Cloud",
            "gpu_desc": "Google Cloud [OAuth Session]",
            "devices": ["OAuth"],
            "size": "Cloud scale",
            "desc": "Gemini Multimodal Reasoning via Google Cloud OAuth",
            "type": "cloud",
            "is_active": is_cloud_active,
            "dedicated": False,
        })

        # Laya System 1 Auto-Routing entry
        is_auto_active = (self.active_target == "auto")
        catalog.append({
            "id": "auto",
            "name": "Laya System 1 Decision Engine",
            "filename": "ModernBERT-large (~421M)",
            "port": "auto",
            "status": "online",
            "gpu": "Dual-GPU Meta-Router",
            "gpu_desc": "Laya System 1 [/v1/systemone]",
            "devices": ["HTTP:8003"],
            "size": "~421M",
            "desc": "Dynamic Bayesian choice auto-routing in ~33ms",
            "type": "system",
            "is_active": is_auto_active,
            "dedicated": False,
        })

        return catalog

    def print_models(self):
        catalog = self.get_catalog()
        rows = []
        for idx, item in enumerate(catalog, 1):
            st = item["status"].upper()
            st_str = f"{UI.GREEN}● {st}{UI.RST}" if st in ("RUNNING", "ONLINE") else f"{UI.GRAY}○ {st}{UI.RST}"
            if item["is_active"]:
                st_str += f" {UI.AMBER_BOLD}★ ACTIVE{UI.RST}"
            gpu_text = item["gpu_desc"]
            if "9000" in gpu_text or "ROCm" in gpu_text:
                dev_styled = f"{UI.ROCM}{gpu_text}{UI.RST}"
            elif "9001" in gpu_text or "CUDA" in gpu_text:
                dev_styled = f"{UI.CUDA}{gpu_text}{UI.RST}"
            elif "Cloud" in gpu_text:
                dev_styled = f"{UI.CLOUD}{gpu_text}{UI.RST}"
            else:
                dev_styled = f"{UI.LAYA}{gpu_text}{UI.RST}"
            size_tag = f" ({item['size']})" if item.get("size") else ""
            rows.append([str(idx), item["id"], st_str, dev_styled, f"{UI.WHITE}{item['name']}{UI.RST}{UI.GRAY}{size_tag}{UI.RST}", item.get("desc") or ""])
        say.group("Models and endpoints")
        say.table("", ["#", "ID", "Status", "Device and port", "Model", "Specialty"], rows)
        say.dim("Select with /model <# or id>, or send a one-shot with @<id> <prompt>. Inactive models hot-swap automatically.")

    async def _swap_ready(self, port: int, model_id: str) -> bool:
        """Port answers /health and LlamaShift agrees the new model is the one running (the old one may still answer)."""
        if not await asyncio.to_thread(self._port_healthy, port):
            return False
        return await asyncio.to_thread(model_is_running, model_id) is not False

    @staticmethod
    def _port_healthy(port: int) -> bool:
        try:
            req = urllib.request.Request(f"http://localhost:{port}/health", headers={"User-Agent": "Antigravity/1.0"})
            with urllib.request.urlopen(req, timeout=0.8) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def switch_model(self, query: str) -> bool:
        """
        Switches the active model by index (1..N), model ID, or alias.
        If the selected model is currently stopped, triggers llama-shift hot-swap.
        """
        query_str = str(query).strip()
        if not query_str:
            self.print_models()
            return False

        catalog = await asyncio.to_thread(self.get_catalog)
        selected = None

        if query_str.isdigit():
            idx = int(query_str)
            if 1 <= idx <= len(catalog):
                selected = catalog[idx - 1]
            else:
                say.warn(f"Model number {idx} out of range (1-{len(catalog)}). Type /models to view list.")
                return False

        if not selected:
            q_lower = query_str.lower()
            for m in catalog:
                if m["id"].lower() == q_lower:
                    selected = m
                    break

        if not selected:
            q_lower = query_str.lower()
            alias_map = {
                "rocm": "9000",
                "amd": "9000",
                "9000": "9000",
                "cuda": "9001",
                "nvidia": "9001",
                "9001": "9001",
                "gemini": "cloud",
                "oauth": "cloud",
                "laya": "auto",
                "smart": "auto",
            }
            if q_lower in alias_map:
                target_val = alias_map[q_lower]
                if target_val in ("cloud", "auto"):
                    for m in catalog:
                        if m["id"] == target_val:
                            selected = m
                            break
                else:
                    for m in catalog:
                        if m.get("port") == target_val and m.get("status") == "running":
                            selected = m
                            break
                    if not selected:
                        for m in catalog:
                            if m.get("port") == target_val:
                                selected = m
                                break

        if not selected:
            q_lower = query_str.lower()
            matches = [
                m for m in catalog
                if q_lower in m["id"].lower() or q_lower in m["name"].lower()
            ]
            if len(matches) == 1:
                selected = matches[0]
            elif len(matches) > 1:
                id_prefix = [m for m in matches if m["id"].lower().startswith(q_lower)]
                if len(id_prefix) == 1:
                    selected = id_prefix[0]
                else:
                    names = ", ".join(f"'{m['id']}'" for m in matches)
                    say.warn(f"Multiple models matched '{query_str}': {names}. Please use /model <#> to specify.")
                    return False

        if not selected:
            say.warn(f"Model '{query_str}' not found. Type /models to view available models.")
            return False

        mid = selected["id"]
        mtype = selected["type"]

        if mtype == "cloud":
            self.active_target = "cloud"
            say.ok(f"Active target set to Google Cloud Gemini ({self.cloud_model}) via OAuth")
            return True

        if mtype == "system" or mid == "auto":
            self.active_target = "auto"
            say.ok("Active target set to Laya System 1 Auto-Routing (~33ms)")
            return True

        # Local model
        port = selected.get("port", "9000")
        is_running = (selected.get("status") == "running")
        gpu_name = selected.get("gpu", "GPU")
        dev_str = ", ".join(selected.get("devices", [])) or ("ROCm0" if port == "9000" or "AMD" in gpu_name else "CUDA0")
        model_name = selected.get("name", mid)
        model_size = selected.get("size", "")

        if is_running:
            self.active_target = str(port)
            await asyncio.to_thread(self.refresh_endpoints)
            say.ok(f"Target set to {gpu_name} [{dev_str}] (Port {port}): {model_name} (already running)")
            return True

        # Inactive model: trigger hot-swap
        say.step(f"hot-swapping {model_name} ({model_size}) onto {gpu_name} [{dev_str}] (port {port})")
        res = await asyncio.to_thread(trigger_llamashift_switch, mid)
        for warning in res.get("warnings", []):
            say.warn(str(warning))
        if "error" in res and not res.get("timeout"):
            say.err(f"Llama-Shift switch request failed: {res['error']}")
            return False

        start_time = time.time()
        timeout = 45.0 + (60.0 if res.get("timeout") else 0.0)
        success = False
        target_port_int = int(port) if str(port).isdigit() else 9000

        if RICH_AVAILABLE and not self.json_output:
            with console.status(f"[bold cyan]Loading {model_name} into VRAM ({gpu_name})...", spinner="dots") as spin:
                while time.time() - start_time < timeout:
                    elapsed = time.time() - start_time
                    spin.update(f"[bold cyan]Loading {model_name} into {gpu_name} [{dev_str}] VRAM... [bold yellow]{elapsed:.1f}s[/bold yellow]")
                    await asyncio.sleep(0.5)
                    if await self._swap_ready(target_port_int, mid):
                        success = True
                        break
        else:
            say.step("loading weights into VRAM")
            while time.time() - start_time < timeout:
                await asyncio.sleep(1.0)
                if await self._swap_ready(target_port_int, mid):
                    success = True
                    break

        if not success:
            say.err(f"Timed out after {timeout:.0f}s waiting for {model_name} on port {target_port_int}.")
            tail = await asyncio.to_thread(get_logs_tail, mid)
            if tail:
                say.detail(tail)
            return False

        elapsed = time.time() - start_time
        await self.reset_local_agent()

        await asyncio.to_thread(self.refresh_endpoints)
        self.active_target = str(port)
        say.ok(f"Hot-swap complete in {elapsed:.1f}s. Active target set to {gpu_name} (port {port}): {model_name}")
        return True

    # ── Prompt chrome: toolbar and startup header ─────────────────────

    def _toolbar_ctx(self) -> Optional[Dict[str, Any]]:
        """Context metrics cached briefly; computing them can touch the network."""
        stamp, cached = self._ctx_cache
        if cached is not None and time.time() - stamp < 10:
            return cached
        try:
            cached = self.get_context_metrics(self.active_target)
        except Exception:
            cached = None
        self._ctx_cache = (time.time(), cached)
        return cached

    def _target_label_and_model(self, target: str) -> tuple:
        if target == "auto":
            return "AUTO", "laya"
        if target == "cloud":
            return "CLOUD", self.cloud_model
        ep = self.endpoints.get(target)
        if not ep:
            return str(target), "unknown"
        label = {"9000": "ROCm:9000", "9001": "CUDA:9001"}.get(target, ep["name"])
        if ep.get("status") != "online":
            return label, "offline"
        return label, ep.get("name_friendly") or ep.get("filename") or "no model"

    def _permission_mode(self) -> tuple:
        if self.use_laya_adaptive_permissions:
            return "laya", UI.LAYA
        if self._dangerously_skip_permissions:
            return "auto-approve", UI.AMBER
        return "safe", UI.GREEN

    def toolbar_fragments(self) -> List[tuple]:
        """Bottom-toolbar segments as (prompt_toolkit style, text) pairs, themed from the active palette."""
        label, model = self._target_label_and_model(self.active_target)
        perm, perm_color = self._permission_mode()
        segments: List[tuple] = [
            (_pt_style(UI.target_color(self.active_target), bold=True), label),
            (_pt_style(UI.WHITE), model),
        ]
        ctx = self._toolbar_ctx()
        if ctx:
            col = UI.RED_BOLD if ctx["percentage"] >= 85 else UI.AMBER if ctx["percentage"] >= 65 else UI.CYAN
            segments.append((_pt_style(col), f"{_short_count(ctx['tokens'])}/{_short_count(ctx['max_ctx'])} ctx"))
        segments.append((_pt_style(perm_color), perm))
        try:
            qs = SubagentManager.get_queue_status()
            active = int(qs.get("running", 0)) + int(qs.get("queued", 0))
        except Exception:
            active = 0
        if active:
            segments.append((_pt_style(UI.AMBER), f"{active} subagent{'s' if active != 1 else ''}"))
        if self.kontexta_sync:
            segments.append((_pt_style(UI.GREEN), "memory on"))

        sep_style = _pt_style(UI.DARK_GRAY)
        frags: List[tuple] = [("", " ")]
        for i, seg in enumerate(segments):
            if i:
                frags.append((sep_style, " · "))
            frags.append(seg)
        frags.append(("", " "))
        return frags

    def print_header(self):
        """Compact startup header: endpoints, target, permissions, repo and memory state."""
        if self.json_output:
            return
        dots = []
        for key, name in (("9000", "ROCm:9000"), ("9001", "CUDA:9001")):
            ep = self.endpoints[key]
            online = ep["status"] == "online"
            dot = f"{UI.GREEN}●{UI.RST}" if online else f"{UI.RED}○{UI.RST}"
            model = (ep.get("name_friendly") or ep.get("filename") or "no model") if online else "offline"
            dots.append(f"{dot} {UI.target_color(key)}{name}{UI.RST} {UI.GRAY}{model}{UI.RST}")
        cloud_ok = self.has_oauth or bool(self.api_key)
        dots.append(f"{UI.GREEN + '●' if cloud_ok else UI.AMBER + '○'}{UI.RST} {UI.CLOUD}Cloud{UI.RST} {UI.GRAY}{self.cloud_model if cloud_ok else 'no credentials'}{UI.RST}")

        label, model = self._target_label_and_model(self.active_target)
        perm, perm_color = self._permission_mode()
        try:
            res = subprocess.run(["git", "rev-parse", "--show-toplevel", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=2)
            top, _, branch = res.stdout.strip().partition("\n")
            repo = f"{os.path.basename(top)} ({branch})" if res.returncode == 0 and top else "no git repo"
        except Exception:
            repo = "no git repo"
        project = None
        if self.enable_kontexta:
            try:
                project = KontextaManager.find_project_for_dir(".")
            except Exception:
                project = None
        if not self.kontexta_sync:
            memory = f"{UI.GRAY}off{UI.RST}"
        elif kontexta_sync.enabled(self, "."):
            memory = f"{UI.GREEN}on{UI.RST} {UI.GRAY}({project['name'] if project and project.get('name') else 'project'}){UI.RST}"
        else:
            memory = f"{UI.GRAY}on, inactive here{UI.RST}"

        g, r = UI.GRAY, UI.RST
        say.plain(say.rule_line("myagy", UI.DARK_GRAY))
        say.plain("  " + "   ".join(dots))
        say.plain(f"  {g}target{r} {UI.target_color(self.active_target)}{label}{r} {g}{model}{r}  {g}·{r}  {g}permissions{r} {perm_color}{perm}{r}  {g}·{r}  {g}theme{r} {UI.theme_name}")
        say.plain(f"  {g}repo{r} {repo}  {g}·{r}  {g}memory{r} {memory}")
        say.plain(f"  {g}/help for commands · Esc stops generation · /theme dark|light|mono{r}")
        say.plain(say.rule_line("", UI.DARK_GRAY))
