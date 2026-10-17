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
import httpx
import copyreg
import inspect
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Literal

TargetBackend = Literal["9000", "9002", "cloud", "rocm", "cuda", "auto"]


@dataclass(slots=True)
class ContextZoneTelemetry:
    target: str
    model_id: str
    n_ctx_max: int                    # Hardware slot max context (e.g., 32768 on :9000, 16384 on :9002)
    kv_cached_tokens: int             # Actual n_past tokens held in llama-server KV slot 0

    # Zone 1: Static System Prefix (100% KV-cached)
    instructions_tokens: int          # GEMINI.md / antigravity.md / AGENTS.md
    repomap_tokens: int               # <1.5ms AST symbol outline
    mcp_schema_tokens: int            # Kontexta (kxta) + tool declarations

    # Zone 2: Append-Only Frozen Epochs (KV-cached prefix)
    frozen_epoch_count: int
    frozen_epoch_tokens: int

    # Zone 3: Sliding Active Window
    pruned_turn_tokens: int           # Tier-0 pruned tool outputs (Turns N-5 .. N-3)
    verbatim_turn_tokens: int         # Unpruned recent turns (Turns N-2 .. N)
    tier0_saved_tokens: int           # Tokens saved by Tier-0 deterministic pruning

    # Compactor trigger watermark (e.g., 75% of n_ctx_max)
    compaction_watermark_ratio: float = 0.75

    @property
    def zone1_tokens(self) -> int:
        return self.instructions_tokens + self.repomap_tokens + self.mcp_schema_tokens

    @property
    def zone2_tokens(self) -> int:
        return self.frozen_epoch_tokens

    @property
    def zone3_tokens(self) -> int:
        return self.pruned_turn_tokens + self.verbatim_turn_tokens

    @property
    def total_used_tokens(self) -> int:
        return self.zone1_tokens + self.zone2_tokens + self.zone3_tokens

    @property
    def headroom_tokens(self) -> int:
        return max(0, self.n_ctx_max - self.total_used_tokens)

    @property
    def utilization_pct(self) -> float:
        return (self.total_used_tokens / self.n_ctx_max * 100.0) if self.n_ctx_max else 0.0

    @property
    def kv_prefix_hit_pct(self) -> float:
        if self.total_used_tokens == 0:
            return 100.0
        cached = min(self.kv_cached_tokens, self.total_used_tokens)
        return (cached / self.total_used_tokens) * 100.0


@dataclass
class TurnReceipt:
    target: str
    model_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    ttft_ms: float = 0.0
    elapsed_s: float = 0.0
    files_changed: int = 0
    commands_run: int = 0

    @property
    def tok_per_sec(self) -> float:
        return self.completion_tokens / self.elapsed_s if self.elapsed_s > 0 else 0.0

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
    CUDA_BASE_URL,
    ROCM_BASE_URL,
    SESSIONS_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_ENDPOINTS,
    DEFAULT_CLOUD_MODEL,
    DEFAULT_CLOUD_IDLE_TIMEOUT,
    DEFAULT_LAYA_ENDPOINT,
    FAST_GIT_MODEL,
    DENSE_CODE_MODEL,
    FAST_GIT_FAMILY,
    DENSE_CODE_FAMILY,
    COMPACTOR_MODEL_ID,
    service_url,
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
    vram_snapshot,
    invalidate_cache as invalidate_llamashift_cache,
)
from ..context.mcp_loader import load_mcp_servers, check_oauth_available
from ..context.kontexta import KontextaManager
from ..context import kontexta_sync
from .laya import LayaDecisionEngine
from ..agents.compactor import ContextCurator
from .coordinator import CudaCoordinator
from .checkpoints import Checkpoints, GitCheckpointer, CheckpointRecord
from .reloader import ReloadSnapshot
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


class OAuthCloudResponse:
    """Async stream wrapper for Google OAuth Cloud responses."""
    def __init__(self, text: str = "", queue: Optional[asyncio.Queue] = None, task: Optional[asyncio.Task] = None):
        self._text = text
        self._queue = queue
        self._task = task

    async def _empty_stream(self):
        if False:
            yield ""

    @property
    def thoughts(self):
        return self._empty_stream()

    @property
    def tool_calls(self):
        return self._empty_stream()

    async def cancel(self):
        if self._task and not self._task.done():
            self._task.cancel()

    def __aiter__(self):
        async def _gen():
            if self._queue is not None:
                while True:
                    tok = await self._queue.get()
                    if tok is None:
                        if self._task and not self._task.cancelled() and self._task.exception() is not None:
                            raise self._task.exception()
                        break
                    yield tok
            else:
                words = self._text.split(" ")
                for i, w in enumerate(words):
                    chunk = w + (" " if i < len(words) - 1 else "")
                    yield chunk
                    await asyncio.sleep(0.002)
        return _gen()


class OAuthCloudAgent:
    """Agent implementation routing turns through the existing Google OAuth session via agy CLI."""
    def __init__(self, session):
        self.session = session
        self._conversation = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def chat(self, prompt: str):
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        async def _run():
            try:
                def _cb(tok: str):
                    queue.put_nowait(tok)
                await self.session.chat_cloud_oauth(prompt, with_project_context=False, on_token=_cb, quiet=False)
            finally:
                queue.put_nowait(None)

        task = asyncio.create_task(_run())
        return OAuthCloudResponse(queue=queue, task=task)

    async def send_message_stream(self, prompt: str):
        text = await self.session.chat_cloud_oauth(prompt, with_project_context=False, quiet=True)
        async def _stream():
            yield text
        return _stream()


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
        clean_cm = (cloud_model or "").strip().strip("[]<>")
        self.cloud_model = DEFAULT_CLOUD_MODEL if clean_cm.lower() in ("", "id", "model", "model-id", "model_id") else cloud_model.strip()
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
        try:
            self.cloud_idle_timeout: int = int(os.environ.get("KXTC_CLOUD_IDLE_TIMEOUT", DEFAULT_CLOUD_IDLE_TIMEOUT))
        except (ValueError, TypeError):
            self.cloud_idle_timeout = DEFAULT_CLOUD_IDLE_TIMEOUT
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

        # CUDA Slot Arbitrator & Preemption Engine
        self.cuda_coord = CudaCoordinator()

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

        # Dynamically discover live models on ports 9000 and 9002
        self.refresh_endpoints()

        # Active default target: 'auto', '9000', or 'cloud' (9002 is reserved for compaction)
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
        self._repo_root: Optional[str] = None
        self.checkpointer = GitCheckpointer(repo_root=self._repo_root or os.getcwd())
        self.checkpoints: List[Dict[str, Any]] = []

        # Dynamic Auto-Shift (Git Actions ➔ Fast MoE, Deep Reasoning ➔ Dense Code)
        self.auto_shift: bool = True
        self.fast_git_model: str = FAST_GIT_MODEL
        self.dense_code_model: str = DENSE_CODE_MODEL

        # NVIDIA RTX 4060 (:9002) Background Context Curator Cache
        self._cached_curated_memory: Optional[Dict[str, Any]] = None
        self._synthesis_task: Optional[asyncio.Task] = None
        self.continuous_curation: bool = True

        # Reload & Subagent State
        self.subagent_mgr = SubagentManager(
            cuda_coord=self.cuda_coord,
            laya=self.laya,
            hooks_provider=lambda: ExternalHooksManager.build_lifecycle_hooks(self, ".") if self.enable_hooks else [],
            mcp_tools=getattr(self, "_mcp_servers", []),
            system_prompt_provider=lambda: self.build_system_context("9002"),
            model_id=self.endpoints.get("9002", {}).get("model") or self.endpoints.get("9001", {}).get("model") or COMPACTOR_MODEL_ID,
        )
        self.active_models: Dict[str, str] = {
            "rocm": self.endpoints.get("9000", {}).get("model", DENSE_CODE_MODEL),
            "cuda": self.endpoints.get("9002", {}).get("model") or self.endpoints.get("9001", {}).get("model", COMPACTOR_MODEL_ID),
            "cloud": self.cloud_model,
        }
        self.auto_route: bool = (self.active_target == "auto")
        self.permission_mode: str = "skip" if self._dangerously_skip_permissions else ("auto" if self.use_laya_adaptive_permissions else "safe")
        self.markdown_enabled: bool = self.render_markdown
        self.curation_enabled: bool = self.continuous_curation
        self.kontexta_enabled: bool = self.enable_kontexta
        self.working_memory: str = ""
        self.repomap = RepoMap(repo_root=self._repo_root or os.getcwd())
        from ..agents.compactor import HippocampusCompactor
        self.compactor = HippocampusCompactor(
            coordinator=self.cuda_coord,
            keep_verbatim_turns=self.max_recent_turns,
        )
        self.kontexta = KontextaManager()
        self.memory_sync_enabled: bool = True

    @property
    def auto_git_shift(self) -> bool:
        return self.auto_shift

    @auto_git_shift.setter
    def auto_git_shift(self, val: bool):
        self.auto_shift = bool(val)

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

    def refresh_endpoints(self, force: bool = True) -> Dict[str, Any]:
        """Refreshes port 9000/9002 model state from one cached LlamaShift /api/status, then probes ports directly."""
        if force:
            invalidate_llamashift_cache()
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
        if s in ("cuda", "nvidia", "9002", "gpu1"):
            return "9002"
        if s == "9001":
            return "9001" if "9001" in self.endpoints and "9002" not in self.endpoints else "9002"
        if s in ("cloud", "gemini", "remote", "oauth"):
            return "cloud"
        if s in ("local", "default"):
            return "9000" if self.endpoints.get("9000", {}).get("status") == "online" else "9002"
        return s

    def get_endpoint(self, target: str) -> Dict[str, Any]:
        """Resolves endpoint config for target ('9000', '9001', '9002', 'cuda', etc.)."""
        canonical = self.resolve_target_alias(target)
        if canonical in self.endpoints:
            return self.endpoints[canonical]
        if canonical == "9001" and "9002" in self.endpoints:
            return self.endpoints["9002"]
        if canonical == "9002" and "9001" in self.endpoints:
            return self.endpoints["9001"]
        return self.endpoints.get(canonical, {})

    @property
    def dangerously_skip_permissions(self) -> bool:
        return self._dangerously_skip_permissions

    def set_dangerously_skip_permissions(self, enable: bool):
        if self._dangerously_skip_permissions != enable:
            self._dangerously_skip_permissions = enable
            self._invalidate_local_agent()
        self.save_preferences()

    @staticmethod
    def _ask(prompt: str) -> str:
        try:
            from ..terminal.ui import stop_status
            stop_status()
        except Exception:
            pass
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
        self._stop_status()
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
            choice = self._ask(
                "Authorize root execution? [Y/n/c]:" if authed else "Authorize execution with sudo? [y/N]:"
            )
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

        if self._dangerously_skip_permissions:
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

        self._stop_status()
        rows = [("Tool", tool_name), *self._arg_rows(args)]
        if risk:
            rows.append(("Risk", risk))
        rows.append(("Reason", reason))
        say.card("Permission required", rows, actions="[y]es / [n]o / [a]lways", style="warn")
        try:
            choice = self._ask("Authorize? [y/n/a]:")
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

    def _schedule_epoch_compaction(self) -> None:
        """Fires Tier 1/2 epoch compaction in the background once enough turns have accumulated."""
        compactor = getattr(self, "compactor", None)
        if compactor is None or not self.continuous_curation:
            return
        if not CudaCoordinator.can_compact():
            return
        task = getattr(self, "_epoch_task", None)
        if task is not None and not task.done():
            return
        if len(self.history) < (compactor.epoch_chunk_turns + compactor.keep_verbatim_turns) * 2:
            return

        if self.endpoints.get("9002", {}).get("status") != "online":
            return

        async def _run() -> None:
            snapshot = list(self.history)
            try:
                async with httpx.AsyncClient(base_url=CUDA_BASE_URL.removesuffix("/v1"), timeout=60.0) as client:
                    compactor.http = client
                    try:
                        trimmed = await compactor.maybe_compact_epoch(snapshot)
                    finally:
                        compactor.http = None
            except Exception:
                return
            # Apply only if an epoch was frozen and history was not edited meanwhile
            if len(trimmed) < len(snapshot) and self.history[:len(snapshot)] == snapshot:
                self.history[:] = trimmed + self.history[len(snapshot):]

        self._epoch_task = asyncio.create_task(_run())

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

    @staticmethod
    def _est_tokens(text: str) -> int:
        """Fast deterministic token estimator (~3.8 chars/token for code + JSON)."""
        return int(len(text) / 3.8) if text else 0

    async def get_context_telemetry(self) -> ContextZoneTelemetry:
        """Reconciles the 3-Zone Prefix-Preserving logical token budgets with live hardware /slots KV cache metrics."""
        rules_txt = ProjectInstructions.load_instructions(getattr(self, "_repo_root", ".") or ".")[0] if hasattr(ProjectInstructions, "load_instructions") else ""
        repomap_txt = self.repomap.get_cached_outline() if hasattr(self, "repomap") and getattr(self.repomap, "enabled", True) else ""
        epoch_txt = self.compactor.render_working_memory_prefix() if hasattr(self, "compactor") and hasattr(self.compactor, "render_working_memory_prefix") else ""
        mcp_tools = getattr(self, "mcp_tools", []) or getattr(self, "_mcp_servers", [])
        mcp_txt = str([getattr(t, "schema", str(t)) for t in mcp_tools])

        history: list[dict[str, Any]] = []
        if hasattr(self, "agent") and hasattr(self.agent, "get_history"):
            try:
                history = self.agent.get_history() or []
            except Exception:
                history = []
        if not history and hasattr(self, "history"):
            history = list(self.history)

        keep_turns = getattr(self.compactor, "keep_verbatim_turns", 4) if hasattr(self, "compactor") else 4
        cutoff = max(0, len(history) - (keep_turns * 2))
        older_slice = history[:cutoff]
        recent_slice = history[cutoff:]

        if hasattr(self, "compactor") and hasattr(self.compactor, "prune_stale_tool_outputs"):
            pruned_older = self.compactor.prune_stale_tool_outputs(older_slice)
        else:
            pruned_older = older_slice

        raw_older_tokens = sum(self._est_tokens(str(m.get("content", ""))) for m in older_slice)
        pruned_older_tokens = sum(self._est_tokens(str(m.get("content", ""))) for m in pruned_older)
        verbatim_tokens = sum(self._est_tokens(str(m.get("content", ""))) for m in recent_slice)

        target = getattr(self, "active_target", "rocm")
        model_id = (self.active_models.get(target) if hasattr(self, "active_models") else None) or (
            self.endpoints.get(target, {}).get("model") if hasattr(self, "endpoints") and target in self.endpoints else "unknown"
        )
        n_ctx_max = 32768 if target in ("rocm", "9000") else (16384 if target in ("cuda", "9001", "9002") else 1_000_000)
        kv_cached = 0

        if target in ("rocm", "cuda", "9000", "9001", "9002"):
            ep_url = ""
            if hasattr(self, "endpoints") and target in self.endpoints:
                ep_url = self.endpoints[target].get("url", "")
            if not ep_url:
                ep_url = ROCM_BASE_URL if target in ("rocm", "9000") else CUDA_BASE_URL
            base = ep_url.removesuffix("/v1")
            try:
                async with httpx.AsyncClient(timeout=0.4) as client:
                    resp = await client.get(f"{base}/slots")
                    if resp.status_code == 200 and (slots := resp.json()):
                        if isinstance(slots, list) and len(slots) > 0:
                            s0 = slots[0]
                            n_ctx_max = int(s0.get("n_ctx", n_ctx_max))
                            kv_cached = int(
                                s0.get("n_past", 0)
                                or (s0.get("next_token", [{}])[0].get("n_decoded", 0) if s0.get("next_token") else 0)
                            )
            except Exception:
                pass

        frozen_epochs = getattr(self.compactor, "frozen_epochs", []) if hasattr(self, "compactor") else []

        return ContextZoneTelemetry(
            target=target,
            model_id=str(model_id or "unknown"),
            n_ctx_max=n_ctx_max,
            kv_cached_tokens=kv_cached,
            instructions_tokens=self._est_tokens(rules_txt),
            repomap_tokens=self._est_tokens(repomap_txt),
            mcp_schema_tokens=self._est_tokens(mcp_txt),
            frozen_epoch_count=len(frozen_epochs),
            frozen_epoch_tokens=self._est_tokens(epoch_txt),
            pruned_turn_tokens=pruned_older_tokens,
            verbatim_turn_tokens=verbatim_tokens,
            tier0_saved_tokens=max(0, raw_older_tokens - pruned_older_tokens),
        )

    # ── Subagent Delegation & AST Repo Map ─────────────────────────────

    def spawn_subagent(self, task_description: str, command: Optional[str] = None) -> SubagentTask:
        if hasattr(self, "subagent_mgr") and self.subagent_mgr:
            kind = "build" if command else "spawn"
            task = self.subagent_mgr.spawn(task_description, kind=kind, command=command)
            return task
        return SubagentManager.spawn(task_description, self, command=command)

    def run_build_subagent(self, custom_cmd: Optional[str] = None, repo_root: str = ".") -> SubagentTask:
        """Pushes project build & test verification to the dedicated NVIDIA RTX 4060 Subagent (:9002)."""
        from ..agents.brainstorm import BrainstormWorkflow
        build_cmd = custom_cmd or BrainstormWorkflow._detect_test_command(repo_root) or "make test"
        if hasattr(self, "subagent_mgr") and self.subagent_mgr:
            return self.subagent_mgr.spawn_build(build_cmd)
        desc = f"Build & Test Verification: execute '{build_cmd}' in {repo_root}"
        task = self.spawn_subagent(desc, command=build_cmd)
        return task

    def list_subagents(self) -> List[SubagentTask]:
        if hasattr(self, "subagent_mgr") and self.subagent_mgr:
            return list(self.subagent_mgr.tasks.values())
        return SubagentManager.list_tasks()

    def get_subagent(self, task_id: int) -> Optional[SubagentTask]:
        if hasattr(self, "subagent_mgr") and self.subagent_mgr:
            return self.subagent_mgr.tasks.get(task_id) or SubagentManager.get(task_id)
        return SubagentManager.get(task_id)

    def cancel_subagent(self, task_id: int) -> bool:
        if hasattr(self, "subagent_mgr") and self.subagent_mgr:
            ok, _ = self.subagent_mgr.cancel_task(task_id)
            return ok
        return SubagentManager.cancel(task_id)

    def inject_subagent(self, task_id: int) -> bool:
        if hasattr(self, "subagent_mgr") and self.subagent_mgr:
            count, _ = self.subagent_mgr.inject_into_session(self, str(task_id))
            return count > 0
        return SubagentManager.inject(task_id, self)

    def inject_all_subagents(self) -> int:
        if hasattr(self, "subagent_mgr") and self.subagent_mgr:
            count, _ = self.subagent_mgr.inject_into_session(self, "all")
            return count
        return SubagentManager.inject_all(self)

    def print_subagents(self):
        if hasattr(self, "subagent_mgr") and self.subagent_mgr:
            self.subagent_mgr.render_tasks_table()
        else:
            SubagentManager.print_tasks(self)

    def print_subagent_detail(self, task_id: int):
        if hasattr(self, "subagent_mgr") and self.subagent_mgr:
            self.subagent_mgr.render_task_detail(task_id)
        else:
            SubagentManager.print_detail(task_id)

    def get_repomap(self, root_dir: str = ".") -> str:
        return self.repomap.get_cached_outline() if hasattr(self, "repomap") else RepoMap.generate_compact_markdown(root_dir)

    def print_repomap(self, root_dir: str = "."):
        tree = self.repomap.build_rich_tree(subpath=root_dir if root_dir != "." else None) if hasattr(self, "repomap") else None
        if tree and RICH_AVAILABLE and console:
            console.print(tree)
        else:
            RepoMap.print_tree(root_dir)

    def refresh_repomap(self, root_dir: str = ".") -> str:
        if hasattr(self, "repomap"):
            self.repomap.invalidate_and_refresh()
        RepoMap.invalidate_cache(root_dir)
        self._invalidate_local_agent()
        return self.repomap.get_cached_outline() if hasattr(self, "repomap") else RepoMap.get_cached_map(root_dir)

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
            self._start_status(f"[bold yellow]⚙ {c_name}[/bold yellow] [dim]{summary_clean[:60]}[/dim]")
            return types.HookResult(allow=True)

        @hooks.post_tool_call
        def on_tool_result(res: types.ToolResult):
            self._stop_status()
            pending = self._pop_pending_tool(res)
            if self.json_output:
                return
            elapsed = (time.perf_counter() - pending["t0"]) if pending.get("t0") else None
            step_tag = f"step {pending['step']}/{self.max_tool_steps_per_turn}" if (self.max_tool_steps_per_turn > 0 and pending.get("step")) else ""

            if res.error:
                err_msg = " ".join(str(res.error).split())
                say.tool_row(pending["name"], pending["summary"], ok=False, elapsed=elapsed, extra=err_msg[:40])
                say.detail(f"    error: {err_msg[:400]}")
                self._start_status("Thinking & generating response...")
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

            self._start_status("Thinking & generating response...")

        all_hooks = [loop_guard, on_tool_result]

        # External and extended lifecycle hooks (hooks.json, compaction, Bayesian gate)
        if self.enable_hooks:
            @hooks.on_compaction
            def on_sdk_compaction(step: Any):
                say.step("SDK Context Compaction hook triggered")
                self.trigger_background_synthesis()

            all_hooks.append(on_sdk_compaction)

            if self.use_laya_adaptive_permissions:
                @hooks.pre_tool_call_decide
                def laya_bayesian_decide(call: types.ToolCall) -> types.HookResult:
                    if not self.use_laya_adaptive_permissions or self._dangerously_skip_permissions:
                        return types.HookResult(allow=True)
                    c_name = getattr(call, "name", str(call))
                    c_args = getattr(call, "args", {}) or {}
                    action_summary = f"Tool: {c_name} Args: {json.dumps(c_args)}"
                    is_safe, p_safe = self.laya.evaluate_action_safety(action_summary)
                    if is_safe:
                        say.step(f"auto-approved {c_name} (laya P(safe)={p_safe:.2f})")
                        return types.HookResult(allow=True)
                    sig = self._action_signature(c_name, c_args)
                    if sig in self._approved_actions:
                        return types.HookResult(allow=True)
                    if self.json_output:
                        return types.HookResult(allow=False, message=f"Blocked by Laya Bayesian gate: P(Safe)={p_safe:.2f} < 0.90")
                    return types.HookResult(allow=True)

                all_hooks.append(laya_bayesian_decide)

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
        if self.auto_repomap and getattr(self.repomap, "enabled", True):
            repomap_text = self.repomap.get_cached_outline() if hasattr(self, "repomap") else RepoMap.get_cached_map(".")
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

        # Zone 2: Append-Only Frozen Epoch Blocks (KV-Cached Prefix)
        if hasattr(self, "compactor") and hasattr(self.compactor, "render_working_memory_prefix"):
            mem_prefix = self.compactor.render_working_memory_prefix()
            if mem_prefix:
                base_inst += f"\n\n{mem_prefix}\n"

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

    # ── Zero-Leak In-Place /reload Lifecycle ─────────────────────────

    async def graceful_shutdown_for_reload(self) -> None:
        """
        Cleanly closes child MCP stdio pipes, HTTP pools, and CUDA:9002 background
        tasks before os.execv so no subprocesses or sockets leak across reloads.
        """
        # 1. Stop CUDA:9002 arbitrator & preempt any active background compaction
        if getattr(self, "cuda_coord", None):
            try:
                await self.cuda_coord.stop()
            except Exception:
                pass

        # Stop subagent manager background worker and HTTP pool
        if getattr(self, "subagent_mgr", None) and hasattr(self.subagent_mgr, "stop"):
            try:
                await self.subagent_mgr.stop()
            except Exception:
                pass

        # 2. Close MCP server stdio subprocesses
        mcp_servers = getattr(self, "_mcp_servers", [])
        for srv in mcp_servers:
            close_fn = getattr(srv, "aclose", None) or getattr(srv, "close", None)
            if callable(close_fn):
                try:
                    res = close_fn()
                    if hasattr(res, "__await__"):
                        await res
                except Exception:
                    pass

        # 3. Clean up MCP tools
        mcp_tools = getattr(self, "mcp_tools", [])
        for tool in mcp_tools:
            close_fn = getattr(tool, "aclose", None) or getattr(tool, "close", None)
            if callable(close_fn):
                try:
                    res = close_fn()
                    if hasattr(res, "__await__"):
                        await res
                except Exception:
                    pass

        # 4. Close session agent / client pools
        if hasattr(self, "checkpointer") and hasattr(self.checkpointer, "close"):
            try:
                self.checkpointer.close()
            except Exception:
                pass

        if hasattr(self, "laya") and hasattr(self.laya, "aclose"):
            try:
                await self.laya.aclose()
            except Exception:
                pass

        if hasattr(self, "close") and callable(self.close):
            try:
                res = self.close()
                if hasattr(res, "__await__"):
                    await res
            except Exception:
                pass

    async def quiesce_session(self) -> None:
        """Alias for graceful_shutdown_for_reload() to cleanly terminate session background resources."""
        await self.graceful_shutdown_for_reload()

    def export_reload_snapshot(self) -> ReloadSnapshot:
        """Captures full in-memory session state prior to os.execv."""
        chat_hist = []
        if hasattr(self, "agent") and hasattr(self.agent, "get_history"):
            try:
                chat_hist = self.agent.get_history() or []
            except Exception:
                chat_hist = list(self.history)
        if not chat_hist:
            chat_hist = list(self.history)

        working_mem = getattr(self, "working_memory", "")
        if not working_mem and self._cached_curated_memory:
            working_mem = self._cached_curated_memory.get("summary", "")

        pending_tasks = []
        if hasattr(self, "subagent_mgr") and hasattr(self.subagent_mgr, "export_pending"):
            pending_tasks = self.subagent_mgr.export_pending()
        elif hasattr(SubagentManager, "export_pending"):
            pending_tasks = SubagentManager.export_pending()

        frozen_epochs_data = []
        if hasattr(self, "compactor") and hasattr(self.compactor, "frozen_epochs"):
            for ep in self.compactor.frozen_epochs:
                frozen_epochs_data.append({
                    "epoch_id": ep.epoch_id,
                    "turn_range": list(ep.turn_range),
                    "kontexta_ref": ep.kontexta_ref,
                    "git_tree_sha": ep.git_tree_sha,
                    "ledger": ep.ledger,
                })

        return ReloadSnapshot(
            active_target=self.active_target,
            active_models=dict(getattr(self, "active_models", {})),
            auto_route=getattr(self, "auto_route", True),
            permission_mode=getattr(self, "permission_mode", "auto"),
            markdown_enabled=getattr(self, "markdown_enabled", getattr(self, "render_markdown", True)),
            repomap_enabled=getattr(self, "auto_repomap", True),
            curation_enabled=getattr(self, "curation_enabled", getattr(self, "continuous_curation", True)),
            kontexta_enabled=getattr(self, "kontexta_enabled", getattr(self, "enable_kontexta", True)),
            chat_history=chat_hist,
            compacted_working_memory=working_mem,
            frozen_epochs=frozen_epochs_data,
            checkpoint_trees=self.checkpointer.export_stack() if hasattr(self, "checkpointer") and self.checkpointer.checkpoints else list(self.checkpoints),
            stats_ledger=dict(self.stats),
            pending_subagent_tasks=pending_tasks,
        )

    def restore_from_snapshot(self, snap: ReloadSnapshot) -> None:
        """Rehydrates session state after os.execv finishes booting."""
        if hasattr(self, "active_models") and snap.active_models:
            self.active_models.update(snap.active_models)
        self.auto_route = snap.auto_route
        self.permission_mode = snap.permission_mode
        self.markdown_enabled = snap.markdown_enabled
        self.render_markdown = snap.markdown_enabled
        self.auto_repomap = snap.repomap_enabled
        self.curation_enabled = snap.curation_enabled
        self.continuous_curation = snap.curation_enabled
        self.kontexta_enabled = snap.kontexta_enabled
        self.enable_kontexta = snap.kontexta_enabled
        self.working_memory = snap.compacted_working_memory
        if snap.compacted_working_memory:
            self._cached_curated_memory = {
                "summary": snap.compacted_working_memory,
                "timestamp": snap.timestamp,
            }

        if hasattr(self, "compactor") and getattr(snap, "frozen_epochs", None):
            from ..agents.compactor import FrozenEpoch
            self.compactor.frozen_epochs = [
                FrozenEpoch(
                    epoch_id=d["epoch_id"],
                    turn_range=tuple(d["turn_range"]),
                    kontexta_ref=d["kontexta_ref"],
                    git_tree_sha=d.get("git_tree_sha"),
                    ledger=d.get("ledger", {}),
                )
                for d in snap.frozen_epochs
            ]

        # Restore target and conversation history
        self.active_target = self.resolve_target_alias(snap.active_target) if hasattr(self, "resolve_target_alias") else snap.active_target
        if snap.chat_history:
            self.history = list(snap.chat_history)
        if hasattr(self, "agent") and hasattr(self.agent, "set_history"):
            try:
                self.agent.set_history(snap.chat_history)
            except Exception:
                pass
        self._invalidate_local_agent()

        # Restore checkpoints and stats
        if hasattr(self, "checkpointer") and getattr(snap, "checkpoint_trees", None):
            self.checkpointer.restore_stack(snap.checkpoint_trees)
        self.checkpoints = list(snap.checkpoint_trees)
        if snap.stats_ledger:
            self.stats = dict(snap.stats_ledger)
        if hasattr(self, "subagent_mgr") and hasattr(self.subagent_mgr, "restore_pending"):
            self.subagent_mgr.restore_pending(snap.pending_subagent_tasks)
        elif hasattr(SubagentManager, "restore_pending"):
            SubagentManager.restore_pending(snap.pending_subagent_tasks)

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

        if target == "cloud":
            live_model = self.cloud_model
            if (
                self._active_local_agent is not None
                and self._current_agent_target == "cloud"
                and self._active_local_model == live_model
            ):
                if isinstance(self._active_local_agent, OAuthCloudAgent):
                    return self._active_local_agent
                conv = getattr(self._active_local_agent, "_conversation", None)
                conn = getattr(conv, "_connection", None)
                if conv is None or (conn and getattr(conn, "is_idle", True) is False):
                    await self.reset_local_agent()
                else:
                    return self._active_local_agent

            if self._active_local_agent is not None:
                await self.reset_local_agent()

            # If no API key is provided, prefer Google OAuth session if available
            if not self.api_key and self.has_oauth:
                self._active_local_agent = OAuthCloudAgent(self)
                self._current_agent_target = "cloud"
                self._active_local_model = live_model
                return self._active_local_agent

            if not self.api_key and not self.has_oauth:
                raise RuntimeError(
                    "A Gemini API key is required. Set it via GEMINI_API_KEY environment variable "
                    "or login via Google Antigravity OAuth ('agy auth login')."
                )

            system_instructions = await asyncio.to_thread(self.build_system_context, "cloud")
            config = LocalAgentConfig(
                model=self.cloud_model,
                api_key=self.api_key,
                system_instructions=system_instructions,
                capabilities=CapabilitiesConfig(),
                policies=self.get_policies(),
                hooks=self.build_tool_hooks(),
                mcp_servers=self._mcp_servers,
            )
            try:
                new_agent = Agent(config)
                await new_agent.__aenter__()
                self._active_local_agent = new_agent
                self._current_agent_target = "cloud"
                self._active_local_model = live_model
                return self._active_local_agent
            except Exception as e:
                if self.has_oauth:
                    say.warn(f"Cloud SDK agent failed ({e}), falling back to agy OAuth session.")
                    self._active_local_agent = OAuthCloudAgent(self)
                    self._current_agent_target = "cloud"
                    self._active_local_model = live_model
                    return self._active_local_agent
                raise

        await asyncio.to_thread(self.refresh_endpoints)
        if target not in self.endpoints:
            if target == "9001" and "9002" in self.endpoints:
                target = "9002"
            else:
                target = "9000"
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
            conv = getattr(self._active_local_agent, "_conversation", None)
            conn = getattr(conv, "_connection", None)
            if conv is None or (conn and getattr(conn, "is_idle", True) is False):
                await self.reset_local_agent()
            else:
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

    def switch_target(self, target: str, model_id: Optional[str] = None) -> None:
        """Hot-swaps compute backend while preserving conversation state."""
        target = self.resolve_target_alias(target)
        if model_id:
            if target in self.endpoints:
                self.endpoints[target]["model"] = model_id
            elif target == "cloud":
                self.cloud_model = model_id
        if target != self.active_target or model_id:
            self.active_target = target
            self._invalidate_local_agent()

    def _make_backend_config(self, target: str):
        target = self.resolve_target_alias(target)
        if target == "cloud":
            return LocalAgentConfig(
                model=self.cloud_model,
                api_key=self.api_key,
                capabilities=CapabilitiesConfig(),
                policies=self.get_policies(),
            )
        ep = self.get_endpoint(target)
        default_url = CUDA_BASE_URL if target in ("9001", "9002", "cuda") else ROCM_BASE_URL
        return LocalOpenAIAgentConfig(
            model=ep.get("model", "default"),
            base_url=ep.get("url", default_url),
            capabilities=CapabilitiesConfig(),
            policies=self.get_policies(),
        )

    def _build_ephemeral_review_agent(self, cfg):
        session_ref = self

        class EphemeralReviewAgentWrapper:
            def __init__(self, config):
                self.config = config

            async def send_message_stream(self, prompt: str):
                is_cloud = isinstance(self.config, LocalAgentConfig) or getattr(self.config, "base_url", None) is None
                api_key = getattr(self.config, "api_key", None)

                if is_cloud and not api_key:
                    if session_ref.has_oauth:
                        try:
                            text = await session_ref.chat_cloud_oauth(prompt, with_project_context=False, quiet=True)
                            async def _stream_oauth():
                                yield text
                            return _stream_oauth()
                        except Exception as e:
                            say.warn(f"Cloud review via OAuth failed ({e}); local review authoritative.")
                    async def _cloud_fallback():
                        yield "VERDICT: APPROVE\n- Cloud Gemini reviewer skipped (no GEMINI_API_KEY or active OAuth; ROCm review authoritative)."
                    return _cloud_fallback()

                try:
                    async with Agent(self.config) as agent:
                        resp = await agent.chat(prompt)
                        tokens = []
                        async for tok in resp:
                            tokens.append(tok)
                        full_text = "".join(tokens)
                        async def _stream():
                            yield full_text
                        return _stream()
                except Exception as e:
                    if is_cloud and session_ref.has_oauth:
                        try:
                            text = await session_ref.chat_cloud_oauth(prompt, with_project_context=False, quiet=True)
                            async def _stream_oauth2():
                                yield text
                            return _stream_oauth2()
                        except Exception:
                            pass
                    async def _default():
                        yield f"VERDICT: APPROVE\n- Review completed with fallback ({e})."
                    return _default()

        return EphemeralReviewAgentWrapper(cfg)

    @property
    def agent(self):
        if self._active_local_agent is not None:
            if not hasattr(self._active_local_agent, "send_message_stream"):
                agent_ref = self._active_local_agent
                async def _send_stream(prompt: str):
                    if hasattr(agent_ref, "chat"):
                        resp = await agent_ref.chat(prompt)
                        async def _stream():
                            async for tok in resp:
                                yield tok
                        return _stream()
                    async def _empty():
                        yield ""
                    return _empty()
                agent_ref.send_message_stream = _send_stream
            return self._active_local_agent
        cfg = self._make_backend_config(self.active_target)
        return self._build_ephemeral_review_agent(cfg)

    async def run_turn(
        self,
        prompt: str,
        on_thought: Optional[Callable[[str], None]] = None,
        on_tool_start: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> TurnReceipt:
        """Routes prompt via Laya and multiplexes the 3 native SDK streams."""
        t0 = time.perf_counter()
        ttft_ms: float = 0.0

        target = self.active_target
        if target == "auto":
            route_res = self.laya.route_prompt(prompt)
            if inspect.isawaitable(route_res):
                route_res = await route_res
            winner = route_res.recommended_target if hasattr(route_res, "recommended_target") else route_res[0]
            target = self.resolve_target_alias(winner)

        agent = await self.get_or_switch_local_agent(target)
        model_id = self.endpoints.get(target, {}).get("filename") or (self.cloud_model if target == "cloud" else target)

        turn_receipt = TurnReceipt(
            target=target,
            model_id=model_id,
        )

        before_cp = await self._checkpoint_before()
        response = await agent.chat(prompt)

        def _mark_ttft() -> None:
            nonlocal ttft_ms
            if ttft_ms == 0.0:
                ttft_ms = (time.perf_counter() - t0) * 1000.0

        async def _consume_thoughts():
            if hasattr(response, "thoughts"):
                try:
                    async for chunk in response.thoughts:
                        if cancel_event and cancel_event.is_set():
                            break
                        _mark_ttft()
                        if on_thought and chunk:
                            on_thought(chunk)
                except Exception:
                    pass

        async def _consume_tools():
            if hasattr(response, "tool_calls"):
                try:
                    async for tc in response.tool_calls:
                        if cancel_event and cancel_event.is_set():
                            break
                        _mark_ttft()
                        c_name = getattr(tc, "name", str(tc))
                        c_args = getattr(tc, "args", {}) or {}
                        if on_tool_start:
                            on_tool_start(c_name, c_args)
                except Exception:
                    pass

        stream_err: Optional[Exception] = None
        async def _consume_tokens():
            nonlocal stream_err
            try:
                async for tok in response:
                    if cancel_event and cancel_event.is_set():
                        break
                    _mark_ttft()
                    if on_token and tok:
                        on_token(tok)
            except Exception as e:
                stream_err = e

        stream_tasks = [
            asyncio.create_task(_consume_thoughts()),
            asyncio.create_task(_consume_tools()),
            asyncio.create_task(_consume_tokens()),
        ]

        if cancel_event:
            cancel_waiter = asyncio.create_task(cancel_event.wait())
            done, pending = await asyncio.wait(
                [*stream_tasks, cancel_waiter],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_waiter in done:
                for t in stream_tasks:
                    t.cancel()
                if hasattr(response, "cancel"):
                    try:
                        await response.cancel()
                    except Exception:
                        pass
            else:
                cancel_waiter.cancel()
                await asyncio.gather(*stream_tasks, return_exceptions=True)
        else:
            await asyncio.gather(*stream_tasks, return_exceptions=True)

        if stream_err and not (cancel_event and cancel_event.is_set()):
            raise stream_err

        usage = getattr(response, "usage_metadata", None)
        if usage:
            turn_receipt.prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            turn_receipt.completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
            turn_receipt.thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0

        turn_receipt.ttft_ms = ttft_ms
        turn_receipt.elapsed_s = time.perf_counter() - t0
        turn_receipt.commands_run = self._current_turn_tool_count

        after_cp = await self._checkpoint_after(before_cp, prompt)
        if after_cp and self._repo_root:
            raw = await asyncio.to_thread(Checkpoints.diff, self._repo_root, after_cp["before"], after_cp["after"], False, True)
            files = receipt.files_from_numstat(raw)
            turn_receipt.files_changed = len(files)

        # Record in usage ledger
        self._record_stats(
            self.active_target,
            target,
            turn_receipt.completion_tokens,
            turn_receipt.tok_per_sec,
            turn_receipt.ttft_ms,
            turn_receipt.elapsed_s,
            usage_metadata=usage,
        )

        return turn_receipt

    def _start_status(self, msg: str = "Thinking & generating response..."):
        if self.json_output:
            return None
        try:
            from ..terminal import ui
            c = console if console and console is not ui.console else None
            if c and hasattr(c, "status"):
                st = c.status(msg)
                if hasattr(st, "start"):
                    st.start()
                self._active_status_mock = st
                return st
            return ui.start_status(msg)
        except Exception:
            return None

    def _stop_status(self):
        if getattr(self, "_active_status_mock", None):
            try:
                self._active_status_mock.stop()
            except Exception:
                pass
            self._active_status_mock = None
        try:
            from ..terminal.ui import stop_status
            stop_status()
        except Exception:
            pass

    async def _consume_token_stream(self, token_stream, status_msg: str = "Thinking & generating response...") -> tuple[str, int, Optional[float]]:
        """Consumes an async token stream, rendering Markdown live when rich is available.
        Multiplexes native SDK triple-streams (.thoughts, .tool_calls, and text deltas) concurrently."""
        full_response: List[str] = []
        token_count = 0
        t_first = None
        status = self._start_status(status_msg)
        renderer = StreamRenderer(markdown=self.render_markdown, enabled=not self.json_output)

        if hasattr(token_stream, "thoughts") and hasattr(token_stream, "tool_calls"):
            thought_tail = ""
            width = max(20, (console.width if console else 80) - 16)

            async def _drain_thoughts():
                nonlocal thought_tail
                try:
                    async for chunk in token_stream.thoughts:
                        if status and chunk:
                            from rich.markup import escape
                            thought_tail = (thought_tail + chunk)[-400:]
                            line = " ".join(thought_tail.split())[-width:]
                            if line:
                                status.update(f"[dim]thinking: {escape(line)}[/dim]")
                except Exception:
                    pass

            async def _drain_tool_calls():
                try:
                    async for tc in token_stream.tool_calls:
                        c_name = getattr(tc, "name", str(tc))
                        c_args = getattr(tc, "args", {}) or {}
                        summary = (
                            c_args.get("CommandLine")
                            or c_args.get("AbsolutePath")
                            or c_args.get("TargetFile")
                            or c_args.get("file_path")
                            or c_args.get("Query")
                            or ""
                        )
                        sum_str = f" [dim]{str(summary)[:40]}[/dim]" if summary else ""
                        if status:
                            from rich.markup import escape
                            status.update(f"[bold cyan]⚡ {c_name}[/bold cyan]{escape(sum_str)}")
                except Exception:
                    pass

            async def _drain_text():
                nonlocal t_first, token_count
                async for token in token_stream:
                    if t_first is None:
                        t_first = time.perf_counter()
                        self._stop_status()
                    full_response.append(token)
                    token_count += 1
                    renderer.feed(token)

            try:
                await asyncio.gather(_drain_thoughts(), _drain_tool_calls(), _drain_text())
            finally:
                self._stop_status()
                renderer.stop()

            # Capture SDK native usage_metadata
            self._last_usage_metadata = getattr(token_stream, "usage_metadata", None)
            if self._last_usage_metadata:
                reported_cand = getattr(self._last_usage_metadata, "candidates_token_count", None)
                if reported_cand is not None and reported_cand > 0:
                    token_count = reported_cand

            return "".join(full_response).strip(), token_count, t_first
        else:
            chunks = getattr(token_stream, "chunks", None)
            thought_tail = ""
            width = max(20, (console.width if console else 80) - 16)
            try:
                async for item in (chunks if chunks is not None else token_stream):
                    kind = type(item).__name__ if chunks is not None else "Text"
                    token = getattr(item, "text", "") if chunks is not None else item
                    if kind == "Thought":
                        if status and token:
                            from rich.markup import escape
                            thought_tail = (thought_tail + token)[-400:]
                            line = " ".join(thought_tail.split())[-width:]
                            if line:
                                status.update(f"[dim]thinking: {escape(line)}[/dim]")
                        continue
                    if kind != "Text":
                        continue
                    if t_first is None:
                        t_first = time.perf_counter()
                        self._stop_status()
                    full_response.append(token)
                    token_count += 1
                    renderer.feed(token)
            finally:
                self._stop_status()
                renderer.stop()

            self._last_usage_metadata = getattr(token_stream, "usage_metadata", None)
            return "".join(full_response).strip(), token_count, t_first

    async def chat_cloud_oauth(
        self,
        prompt: str,
        with_project_context: bool = True,
        on_token: Optional[Callable[[str], None]] = None,
        quiet: bool = False,
    ) -> str:
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

        idle_timeout_secs = getattr(self, "cloud_idle_timeout", DEFAULT_CLOUD_IDLE_TIMEOUT)
        try:
            env_val = os.environ.get("KXTC_CLOUD_IDLE_TIMEOUT")
            if env_val is not None:
                idle_timeout_secs = int(env_val)
        except (ValueError, TypeError):
            pass

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

        status = None
        if not quiet and (RICH_AVAILABLE and not self.json_output):
            status = console.status(f"[bold #60a5fa]Contacting Google Cloud ({self.cloud_model})...[/bold #60a5fa]", spinner="dots")
            status.start()
        elif not quiet and not self.json_output:
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
        renderer = StreamRenderer(markdown=self.render_markdown, enabled=(not quiet and not self.json_output))
        t_turn_start = time.perf_counter()
        in_tool_execution = False

        async def read_stream():
            nonlocal cleared_spinner, status, in_tool_execution
            while True:
                elapsed = time.perf_counter() - t_turn_start
                if elapsed > MAX_TOTAL_SECS:
                    raise asyncio.TimeoutError(f"Cloud turn exceeded maximum limit ({MAX_TOTAL_SECS}s)")

                # When a tool is actively running (e.g. running test suites, builds),
                # do not abort on idle timeout; only enforce idle timeout when waiting for model tokens.
                wait_timeout = None if in_tool_execution else (idle_timeout_secs if idle_timeout_secs > 0 else None)
                if wait_timeout is not None:
                    wait_timeout = min(wait_timeout, max(1.0, MAX_TOTAL_SECS - elapsed))
                    try:
                        line = await asyncio.wait_for(proc.stdout.readline(), timeout=wait_timeout)
                    except asyncio.TimeoutError:
                        raise asyncio.TimeoutError(f"Cloud model stalled (no activity for {int(wait_timeout)}s)")
                else:
                    line = await proc.stdout.readline()

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

                    if stype == "tool":
                        in_tool_execution = (state == "ACTIVE")
                        if state == "ACTIVE" and not quiet and not self.json_output:
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
                                status.stop()
                                status = None
                            sys.stdout.write("\r\033[K")
                            sys.stdout.flush()
                            clip = summary_clean if len(summary_clean) <= 100 else summary_clean[:97] + "..."
                            if console:
                                console.print(f" [bold cyan]⚙ {tool_name}[/bold cyan] [dim]{clip}[/dim]")
                            else:
                                say.tool_row(f"cloud:{tool_name}", clip)

                    elif stype == "agent_response":
                        in_tool_execution = False
                        delta = su.get("text_delta")
                        if delta:
                            if status:
                                status.stop()
                                status = None
                            if not quiet and not cleared_spinner and not self.json_output:
                                sys.stdout.write("\r\033[K")
                                sys.stdout.flush()
                                cleared_spinner = True
                            collected_text.append(delta)
                            if on_token:
                                on_token(delta)
                            if not quiet:
                                renderer.feed(delta)

                elif event == "result":
                    in_tool_execution = False

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
            raise ValueError(f"Unknown target '{target}'. Use auto, 9000 (rocm), 9002 (cuda) or cloud.")
        return target

    async def _route_auto(self, prompt: str) -> str:
        """Laya System 1 decision pass for target 'auto'; returns a concrete target id."""
        if self.auto_shift:
            is_git, git_conf = self.laya.is_git_action(prompt)
            if is_git and self.endpoints["9000"]["status"] == "online":
                say.step(f"routed to ROCm (git action, laya {git_conf * 100:.0f}%)")
                say.detail(f"Laya System 1 (~33ms): Git Action -> AMD ROCm (Gemma 4 A4B Fast MoE), confidence {git_conf * 100:.1f}%")
                return "9000"
        is_build, build_conf = self.laya.is_build_action(prompt)
        if is_build:
            say.step(f"routed to CUDA (build and test, laya {build_conf * 100:.0f}%)")
            say.detail(f"Laya System 1 (~33ms): Build & Test Verification -> NVIDIA CUDA Subagent (:9002), confidence {build_conf * 100:.1f}%")
            return "9002"
        route_res = self.laya.route_prompt(prompt)
        if inspect.isawaitable(route_res):
            route_res = await route_res
        if hasattr(route_res, "recommended_target"):
            routed_id = route_res.recommended_target
            conf = round(1.0 - route_res.complexity_score, 3)
        else:
            routed_id, conf = route_res
        routed_id = self.resolve_target_alias(routed_id)
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
        """Hot-swaps the :9000 model (Gemma 4 MoE for git, Qwen Dense for dev activities/reasoning) when needed."""
        ep_9000 = self.endpoints.get("9000", {})
        current = " ".join((ep_9000.get(k) or "").lower() for k in ("id", "filename", "name_friendly"))
        is_git, git_conf = self.laya.is_git_action(prompt)
        if is_git:
            if FAST_GIT_FAMILY in current:
                return
            say.step("git action: swapping :9000 to Gemma 4 A4B")
            say.detail(f"Laya Auto-Shift: git action detected (P={git_conf*100:.1f}%), hot-swapping for rapid execution")
            wanted = self.fast_git_model
        else:
            if DENSE_CODE_FAMILY in current:
                return
            catalog = await asyncio.to_thread(self.get_catalog)
            if not any(self.dense_code_model.lower() in m.get("id", "").lower() for m in catalog):
                return
            say.step("dev activity: swapping :9000 to Qwen 27B Dense")
            say.detail("Laya Auto-Shift: dev activity detected, switching to Qwen 27B Dense for code execution")
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
        self._last_usage_metadata = None  # never reuse a previous turn's token usage
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

        if self.auto_shift and target == "9000":
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
                if self.endpoints["9002"]["status"] == "online":
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

            # Proactively check context headroom before executing turn on local endpoint
            ctx_metrics = self.get_context_metrics(target)
            if (
                ctx_metrics.get("max_ctx")
                and ctx_metrics.get("tokens", 0) >= (ctx_metrics["max_ctx"] - 4000)
                and len(self.history) > 2
            ):
                say.warn("Context approaching window limit, proactively auto-compacting history")
                removed = await asyncio.to_thread(self.compact_history, keep_recent=2, silent=True)
                await self.reset_local_agent()
                agent = await self.get_or_switch_local_agent(target)
                if removed > 0:
                    say.step(f"proactively compacted {removed} turn(s)")

            def _is_empty_turn_400(e: Exception) -> bool:
                s = str(e)
                return "400" in s and ("content" in s or "tool_calls" in s)

            def _is_exceed_context_error(e: Exception) -> bool:
                s = str(e).lower()
                triggers = [
                    "exceed_context_size_error",
                    "exceeds the available context size",
                    "context_length_exceeded",
                    "maximum context length",
                    "too many tokens",
                    "context window is full",
                    "request exceeds",
                ]
                return any(t in s for t in triggers)

            def _is_unreachable_error(e: Exception) -> bool:
                if isinstance(e, (ConnectionError, TimeoutError)):
                    return True
                s = str(e).lower()
                triggers = [
                    "connection refused", "connecterror", "unreachable", "cannot connect",
                    "failed to connect", "timed out", "timeout", "remote disconnected",
                    "remotedisconnected", "service unavailable", "502", "503", "504",
                    "unexpected eof", "eof occurred", "premature eof", "connection reset",
                    "broken pipe", "connection closed", "clientpayloaderror",
                    "server disconnected", "remoteprotocolerror",
                ]
                return any(t in s for t in triggers) or bool(re.search(r"\beof\b", s))

            def _prune_empty_history():
                self.history = [
                    h for h in self.history
                    if h.get("content") and h["content"].strip()
                ]

            MAX_ATTEMPTS = 3
            response_text = ""
            last_turn_err = None

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
                    last_turn_err = turn_err
                    can_retry = attempt < MAX_ATTEMPTS
                    if _is_exceed_context_error(turn_err) and can_retry:
                        say.warn("Context size exceeded, auto-compacting conversation history")
                        removed = await asyncio.to_thread(self.compact_history, keep_recent=2, silent=True)
                        if removed == 0 and len(self.history) > 2:
                            removed = await asyncio.to_thread(self.compact_history, keep_recent=1, silent=True)
                        await self.reset_local_agent()
                        if removed == 0:
                            # Normal compaction cannot shrink further; proceed to total distillation
                            break
                        else:
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
                    elif _is_empty_turn_400(turn_err) or _is_unreachable_error(turn_err):
                        _prune_empty_history()
                        await self.reset_local_agent()
                        await asyncio.to_thread(self.refresh_endpoints)

                        candidates = []
                        if target != "9000" and self.endpoints["9000"]["status"] == "online":
                            candidates.append("rocm")
                        if target != "9002" and self.endpoints["9002"]["status"] == "online":
                            candidates.append("cuda")
                        candidates.append("cloud")

                        winner, conf = self.laya.choose_escalation_target(target, prompt, candidates)
                        escalate_to = self._validate_target(self.resolve_target_alias(winner))

                        if _is_unreachable_error(turn_err):
                            cause = "Endpoint unreachable or connection dropped"
                        else:
                            cause = f"Failed after {MAX_ATTEMPTS} attempts"
                        say.warn(f"{target.upper()}: {cause}")
                        say.step(f"escalating to {self._escalation_label(escalate_to)} (laya {conf*100:.0f}%)")
                        response_text, token_count, tps, ttft_ms, total_dur, target = await self._run_escalation(escalate_to, prompt)
                        self._turn_target = target
                        break
                    elif _is_exceed_context_error(turn_err):
                        await self.reset_local_agent()
                        break
                    else:
                        raise

            # If all attempts failed due to context limit, force total distillation before cloud fallback
            if not response_text and last_turn_err and _is_exceed_context_error(last_turn_err):
                say.warn(f"{target.upper()}: Context window exceeded after all attempts; forcing total distillation")
                distilled = await asyncio.to_thread(self.compact_history, keep_recent=0, silent=True)
                await self.reset_local_agent()

                total_distill_succeeded = False
                if distilled > 0:
                    say.step(f"totally distilled {distilled} prior entry(s) into working memory; retrying on {target.upper()}")
                    try:
                        agent = await self.get_or_switch_local_agent(target)
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
                        total_distill_succeeded = True
                    except Exception as distill_err:
                        say.warn(f"{target.upper()}: Context window still exceeded after total distillation")
                        await self.reset_local_agent()

                if not total_distill_succeeded:
                    say.step(f"falling back to Cloud ({self._escalation_label('cloud')})")
                    response_text, token_count, tps, ttft_ms, total_dur, target = await self._run_escalation("cloud", prompt)
                    self._turn_target = target

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
            self._schedule_epoch_compaction()

        approx_tokens = token_count if token_count else len(response_text.split())
        if response_text:
            self._record_stats(
                initial_target,
                target,
                approx_tokens,
                tps,
                ttft_ms,
                total_dur,
                usage_metadata=getattr(self, "_last_usage_metadata", None),
            )
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
            await self.reset_local_agent()
            raise
        finally:
            cp = await self._checkpoint_after(before, prompt)
            await self._print_receipt(cp, time.perf_counter() - t_turn)
            await self._sync_commit_to_kontexta()
            if self.history:
                try:
                    await asyncio.to_thread(self.save_session, "latest_session")
                except Exception:
                    pass
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
        if hasattr(self, "checkpointer"):
            await asyncio.to_thread(self.checkpointer.snapshot_pre_turn)
        if self._repo_root is None:
            self._repo_root = await asyncio.to_thread(Checkpoints.repo_root) or ""
        if not self._repo_root:
            return None
        return await asyncio.to_thread(Checkpoints.snapshot, self._repo_root)

    async def _checkpoint_after(self, before: Optional[str], prompt: str) -> Optional[Dict[str, Any]]:
        if hasattr(self, "checkpointer"):
            await asyncio.to_thread(self.checkpointer.commit_post_turn_snapshot)
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

    def _record_stats(
        self,
        routed: str,
        final: str,
        tokens: int,
        tps: Optional[float],
        ttft_ms: Optional[float],
        dur: Optional[float],
        usage_metadata: Optional[Any] = None,
    ):
        self._last_tps = tps
        st = self.stats.setdefault(final, {})
        st["turns"] = st.get("turns", 0) + 1
        st["tokens"] = st.get("tokens", 0) + tokens
        st["seconds"] = st.get("seconds", 0.0) + (dur or 0.0)
        if usage_metadata:
            st["prompt_tokens"] = st.get("prompt_tokens", 0) + (getattr(usage_metadata, "prompt_token_count", 0) or 0)
            st["candidates_tokens"] = st.get("candidates_tokens", 0) + (getattr(usage_metadata, "candidates_token_count", 0) or 0)
            st["cached_tokens"] = st.get("cached_tokens", 0) + (getattr(usage_metadata, "cached_content_token_count", 0) or 0)
            st["thoughts_tokens"] = st.get("thoughts_tokens", 0) + (getattr(usage_metadata, "thoughts_token_count", 0) or 0)
            st["total_tokens"] = st.get("total_tokens", 0) + (getattr(usage_metadata, "total_token_count", 0) or 0)
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

        has_telemetry = any(st.get("prompt_tokens") or st.get("thoughts_tokens") for st in self.stats.values())
        if has_telemetry:
            meta_rows = []
            for tgt, st in sorted(self.stats.items()):
                if st.get("prompt_tokens") or st.get("thoughts_tokens") or st.get("candidates_tokens"):
                    label = self.endpoints[tgt]["name"] if tgt in self.endpoints else ("Cloud" if tgt == "cloud" else tgt)
                    meta_rows.append([
                        label,
                        str(int(st.get("prompt_tokens", 0))),
                        str(int(st.get("cached_tokens", 0))),
                        str(int(st.get("thoughts_tokens", 0))),
                        str(int(st.get("candidates_tokens", 0))),
                        str(int(st.get("total_tokens", 0))),
                    ])
            if meta_rows:
                say.table("SDK Usage Metadata Telemetry", ["Target", "Prompt Tok", "Cached Tok", "Thinking Tok", "Candidate Tok", "Total Tok"], meta_rows)
        else:
            say.dim("Cloud token counts are word-count estimates.")

    # ── Preferences ───────────────────────────────────────────────────

    def preferences(self) -> Dict[str, Any]:
        out = {k: getattr(self, k, None) for k in PREF_DEFAULTS}
        return {k: v for k, v in out.items() if v is not None}

    def apply_preferences(self, prefs: Dict[str, Any]):
        for key, val in prefs.items():
            if key not in PREF_DEFAULTS:
                if key == "auto_git_shift":
                    self.auto_shift = bool(val)
                continue
            if key == "active_target":
                self.active_target = self.resolve_target_alias(val)
            elif key == "dangerously_skip_permissions":
                self._dangerously_skip_permissions = val
            elif key == "cloud_model" and isinstance(val, str) and val.strip():
                clean_cm = val.strip().strip("[]<>")
                self.cloud_model = DEFAULT_CLOUD_MODEL if clean_cm.lower() in ("", "id", "model", "model-id", "model_id") else val.strip()
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
        elif self.active_target == "9002":
            active_label = f"{UI.CUDA_BOLD}CUDA:9002 (NVIDIA RTX 4060 - Context Compactor){UI.RST}"
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
            ("Compactor (NVIDIA :9002)", _fmt_model(self.endpoints['9002'], UI.CUDA)),
            ("Continuous Curation", _on_off(self.continuous_curation, "Active (Auto-distills in background)")),
            ("Codebase AST Map", _on_off(self.auto_repomap, "Active (Auto-injected into context)")),
            ("Project Guidelines", instr_desc),
            ("Lifecycle Hooks", hook_tag),
            ("Kontexta Engine", k_str),
            ("Cloud Model", f"{UI.CLOUD}{self.cloud_model}{UI.RST}"),
            ("Cloud Auth", auth_status),
            ("Multi-line Input", ml_status),
            ("Conversation Turns", f"{UI.CYAN}{len(self.history) // 2}{UI.RST}"),
            ("Auto-Shift", _on_off(self.auto_shift, f"Active (Git ➔ {self.fast_git_model} │ Code ➔ {self.dense_code_model})")),
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
            elif "9002" in gpu_text or "CUDA" in gpu_text:
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
            req = urllib.request.Request(service_url(port, "/health"), headers={"User-Agent": "Antigravity/1.0"})
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
                "cuda": "9002",
                "nvidia": "9002",
                "9002": "9002",
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
        label = {"9000": "ROCm:9000", "9002": "CUDA:9002"}.get(target, ep["name"])
        if ep.get("status") != "online":
            return label, "offline"
        return label, ep.get("name_friendly") or ep.get("filename") or "no model"

    def _permission_mode(self) -> tuple:
        if self.use_laya_adaptive_permissions:
            return "laya", UI.LAYA
        if self._dangerously_skip_permissions:
            return "auto-approve", UI.AMBER
        return "safe", UI.GREEN

    def _vram_segments(self) -> List[tuple]:
        """VRAM used/total for the GPU behind the active target (both under auto); empty when unknown."""
        snap = vram_snapshot()
        wanted = {"auto": ["amd", "nvidia"], "9000": ["amd"], "9002": ["nvidia"]}.get(self.active_target)
        if not snap or not wanted:
            return []
        labels = {"amd": "rocm", "nvidia": "cuda"}
        out = []
        for gpu in sorted((g for g in snap if g["vendor"] in wanted), key=lambda g: wanted.index(g["vendor"])):
            pct = gpu["used_mib"] / gpu["total_mib"] * 100
            col = UI.RED_BOLD if pct >= 90 else UI.AMBER if pct >= 75 else UI.CYAN
            total = gpu["total_mib"] / 1024
            label = "vram" if len(wanted) == 1 else labels[gpu["vendor"]]
            out.append((_pt_style(col), f"{label} {gpu['used_mib'] / 1024:.1f}/{total:.0f}G" if total >= 10 else f"{label} {gpu['used_mib'] / 1024:.1f}/{total:.1f}G"))
        return out

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
        segments.extend(self._vram_segments())
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
        for key, name in (("9000", "ROCm:9000"), ("9002", "CUDA:9002")):
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
        say.plain(say.rule_line("kontexta-cli", UI.DARK_GRAY))
        say.plain("  " + "   ".join(dots))
        say.plain(f"  {g}target{r} {UI.target_color(self.active_target)}{label}{r} {g}{model}{r}  {g}·{r}  {g}permissions{r} {perm_color}{perm}{r}  {g}·{r}  {g}theme{r} {UI.theme_name}")
        say.plain(f"  {g}repo{r} {repo}  {g}·{r}  {g}memory{r} {memory}")
        say.plain(f"  {g}/help for commands · Esc stops generation · /theme dark|light|mono{r}")
        say.plain(say.rule_line("", UI.DARK_GRAY))
