"""Multi-GPU Hybrid Agent Session with unified memory, Laya System 1 decision engine, and tool execution."""

import os
import sys
import json
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

try:
    from rich.live import Live
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.text import Text
    from rich import box
except ImportError:
    pass

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
)
from ..context.mcp_loader import load_mcp_servers, check_oauth_available
from ..context.kontexta import KontextaManager
from .laya import LayaDecisionEngine
from ..agents.compactor import ContextCurator
from .coordinator import CudaCoordinator
from .checkpoints import Checkpoints
from .prefs import PREF_DEFAULTS, load_prefs, save_prefs
from ..agents.subagent import SubagentManager, SubagentTask
from ..context.repomap import RepoMap
from ..context.instructions import ProjectInstructions
from ..context.hooks_loader import ExternalHooksManager
from .sudo_manager import SudoManager


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
        self.enable_hooks: bool = True
        self.render_markdown: bool = True

        # Load MCP servers once at startup (including Inbuilt Kontexta)
        self._mcp_servers: List[McpStdioServer | McpStreamableHttpServer] = []
        if enable_mcp:
            self._mcp_servers = load_mcp_servers(enable_kontexta=self.enable_kontexta)
            if not self.json_output:
                if self._mcp_servers:
                    names = ", ".join(s.name for s in self._mcp_servers)
                    print(f"{UI.GRAY}[MCP] Loaded {len(self._mcp_servers)} server(s): {names}{UI.RST}")
                else:
                    print(f"{UI.GRAY}[MCP] No MCP servers found (add to ~/.gemini/config/mcp_config.json){UI.RST}")

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

        # Dynamically discover live models on ports 9000 and 9001
        self.refresh_endpoints()

        # Active default target: 'auto', '9000', or 'cloud' (9001 is reserved for compaction)
        self.active_target = self.resolve_target_alias(initial_target)

        # Conversation history: list of {"role": "user"|"assistant", "target": str, "content": str}
        self.history: List[Dict[str, str]] = []
        self.show_metrics: bool = True
        self.max_recent_turns: int = 6
        # Autocompact: compress history when turn count exceeds threshold (0 = disabled)
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

    def refresh_endpoints(self) -> Dict[str, Any]:
        """
        Dynamically refreshes Port 9000 and Port 9001 model state. Zero hardcoding.
        Strategy:
          1. Single GET /api/active call — resolves all running models in one shot.
          2. Per-port fallback via detect_port_model() for any port not covered by /api/active.
        """
        model_dir = DEFAULT_MODEL_DIR

        # Try to resolve all ports in one shot via /api/active
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
            self._active_local_agent = None

    def permission_prompt_handler(self, tool: Any, args: Dict[str, Any]) -> bool:
        tool_name = getattr(tool, "name", str(tool))
        cmd_line = args.get("CommandLine", "") if isinstance(args, dict) else ""
        is_sudo = (tool_name == "run_command" and SudoManager.is_sudo_command(cmd_line))

        # ── SUDO COMMAND HANDLING & PASSWORD ENTRY SECTION ──
        if is_sudo:
            user = os.environ.get("USER", "root")
            print(f"\n{UI.AMBER_BOLD}╭──────────────── 🛡️  SUDO PRIVILEGE ELEVATION REQUIRED ────────────────╮{UI.RST}")
            print(f"{UI.AMBER_BOLD}│{UI.RST}  Tool     : {UI.WHITE}{tool_name}{UI.RST} (Administrative / Root Elevation)")
            print(f"{UI.AMBER_BOLD}│{UI.RST}  Command  : {UI.CYAN}{cmd_line}{UI.RST}")
            print(f"{UI.AMBER_BOLD}│{UI.RST}  Target   : {UI.WHITE}Host System ({user}){UI.RST}")
            print(f"{UI.AMBER_BOLD}╰───────────────────────────────────────────────────────────────────────╯{UI.RST}")

            if SudoManager.is_authenticated():
                rem_min = int(SudoManager.time_remaining() // 60)
                print(f"{UI.GREEN_BOLD}[🛡️  Sudo credentials active: ~{rem_min}m remaining in session cache]{UI.RST}")
                try:
                    choice = prompt_input(f"{UI.AMBER_BOLD}Authorize root execution? [Y/n/change-password]: {UI.RST}").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    print(f"\n{UI.RED_BOLD}[x] Execution denied.{UI.RST}")
                    return False

                if choice in ("c", "change", "change-password"):
                    if not SudoManager.prompt_password_entry(reason=cmd_line):
                        return False
                    print(f"{UI.GREEN_BOLD}[✓] Root execution approved.{UI.RST}")
                    return True
                elif choice in ("", "y", "yes"):
                    print(f"{UI.GREEN_BOLD}[✓] Root execution approved.{UI.RST}")
                    return True
                else:
                    print(f"{UI.RED_BOLD}[x] Execution denied by user.{UI.RST}")
                    return False
            else:
                try:
                    choice = prompt_input(f"{UI.AMBER_BOLD}Authorize execution with sudo? [y/N]: {UI.RST}").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    print(f"\n{UI.RED_BOLD}[x] Execution denied.{UI.RST}")
                    return False

                if choice in ("y", "yes"):
                    ok = SudoManager.prompt_password_entry(reason=cmd_line)
                    if ok:
                        print(f"{UI.GREEN_BOLD}[✓] Root execution approved.{UI.RST}")
                        return True
                    else:
                        print(f"{UI.RED_BOLD}[x] Sudo authorization failed. Execution denied.{UI.RST}")
                        return False
                else:
                    print(f"{UI.RED_BOLD}[x] Execution denied by user.{UI.RST}")
                    return False

        if self._dangerously_skip_permissions and not self.use_laya_adaptive_permissions:
            print(f"{UI.AMBER_BOLD}[⚡ Auto-approved '{tool_name}' (skip-permissions mode)]{UI.RST}")
            return True

        action_summary = f"Tool: {tool_name} Args: {json.dumps(args)}"

        if self.use_laya_adaptive_permissions:
            is_safe, p_safe = self.laya.evaluate_action_safety(action_summary)
            if is_safe:
                print(f"{UI.LAYA_BOLD}[⚡ Laya System 1 (33ms) 'noul': P(Safe)={p_safe:.2f} ≥ 0.90 ➔ {UI.GREEN_BOLD}Auto-approved '{tool_name}']{UI.RST}")
                return True
            else:
                print(f"\n{UI.AMBER_BOLD}[⚡ Laya System 1 (33ms) 'noul': P(Safe)={p_safe:.2f} < 0.90 ➔ Elevated Risk Detected]{UI.RST}")

        print(f"\n{UI.AMBER_BOLD}╭────────────────────── ⚠ PERMISSION REQUIRED ──────────────────────╮{UI.RST}")
        print(f"{UI.AMBER_BOLD}│{UI.RST} Tool: {UI.WHITE}{tool_name}{UI.RST}")
        if args:
            args_str = json.dumps(args, indent=2)
            for line in args_str.splitlines():
                print(f"{UI.AMBER_BOLD}│{UI.RST}   {line}")
        print(f"{UI.AMBER_BOLD}╰───────────────────────────────────────────────────────────────────╯{UI.RST}")
        try:
            choice = prompt_input(f"{UI.AMBER_BOLD}Authorize execution? [y/N]: {UI.RST}").strip().lower()
            allowed = choice in ("y", "yes")
            if allowed:
                print(f"{UI.GREEN_BOLD}[✓] Execution approved.{UI.RST}")
            else:
                print(f"{UI.RED_BOLD}[x] Execution denied by user.{UI.RST}")
            return allowed
        except (KeyboardInterrupt, EOFError):
            print(f"\n{UI.RED_BOLD}[x] Execution denied.{UI.RST}")
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
        return ContextCurator.compact_history(self, keep_recent=keep_recent, silent=silent)

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
        self._active_local_agent = None
        return RepoMap.get_cached_map(root_dir)

    # ── Tool Hooks & System Context ───────────────────────────────────

    def build_tool_hooks(self) -> List[Any]:
        """Creates async PreToolCallDecideHook and PostToolCallHook for live step tracking,
        loop detection, and step budget protection."""
        @hooks.pre_tool_call_decide
        def loop_guard(call: types.ToolCall) -> types.HookResult:
            self._current_turn_tool_count += 1
            c_name = getattr(call, "name", str(call))
            c_args = getattr(call, "args", {}) or {}

            # Human-readable summary for live display
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
            term_cols = shutil.get_terminal_size((120, 24)).columns
            max_summary = max(100, term_cols - 35)
            summary_clean = str(summary).replace("\r\n", " ").replace("\n", " ").strip()
            summary_str = f" ➔ {summary_clean}" if len(summary_clean) <= max_summary else f" ➔ {summary_clean[:max_summary - 3]}..."
            ctx_pill = self.format_context_pill()
            if self.max_tool_steps_per_turn > 0:
                step_badge = f"{UI.DARK_GRAY}[Step {self._current_turn_tool_count}/{self.max_tool_steps_per_turn} │ {ctx_pill}{UI.DARK_GRAY}]{UI.RST} "
            else:
                step_badge = f"{UI.DARK_GRAY}[Step {self._current_turn_tool_count} │ {ctx_pill}{UI.DARK_GRAY}]{UI.RST} "

            if not self.json_output:
                print(f"\n{step_badge}{UI.AMBER_BOLD}[⚙ Local Tool: {c_name}]{UI.RST}{UI.WHITE}{summary_str}{UI.RST}", flush=True)
                if self.verbose and c_args:
                    try:
                        args_json = json.dumps(c_args, indent=2, ensure_ascii=False)
                        for line in args_json.splitlines():
                            print(f"{UI.DARK_GRAY}  │   {line}{UI.RST}")
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
                    print(f"\n{UI.RED_BOLD}[⚠ Loop Guard Triggered]{UI.RST} {UI.AMBER}Blocked repeated {c_name} call (3x identical). Forcing model to respond.{UI.RST}", flush=True)
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
                    print(f"\n{UI.RED_BOLD}[⚠ Step Limit Reached ({self.max_tool_steps_per_turn} steps)]{UI.RST} {UI.AMBER}Halting tool execution to prevent infinite loop. Forcing final response.{UI.RST}", flush=True)
                return types.HookResult(allow=False, message=msg)

            return types.HookResult(allow=True)

        @hooks.post_tool_call
        def on_tool_result(res: types.ToolResult):
            if self.json_output:
                return

            if res.error:
                err_msg = str(res.error)
                print(f"{UI.DARK_GRAY}  └── {UI.RED_BOLD}[✗ Error]:{UI.RST} {UI.RED}{err_msg[:140]}{UI.RST}", flush=True)
                return

            out_str = ""
            if isinstance(res.result, dict):
                out_str = str(res.result.get("output") or res.result.get("content") or res.result)
            elif isinstance(res.result, str):
                out_str = res.result
            elif res.result is not None:
                out_str = str(res.result)

            lines = [l for l in out_str.strip().splitlines() if l.strip()]
            line_count = len(lines)
            char_count = len(out_str)

            if self.verbose and lines:
                term_cols = shutil.get_terminal_size((120, 24)).columns
                max_line = max(100, term_cols - 20)
                print(f"{UI.DARK_GRAY}  │ [Result preview ({line_count} lines, {char_count} chars)]:{UI.RST}")
                for l in lines[:10]:
                    l_str = l if len(l) <= max_line else (l[:max_line - 3] + "...")
                    print(f"{UI.DARK_GRAY}  │   {UI.GRAY}{l_str}{UI.RST}")
                if line_count > 10:
                    print(f"{UI.DARK_GRAY}  │   {UI.DARK_GRAY}... (+{line_count - 10} more lines){UI.RST}")

            if line_count > 0:
                print(f"{UI.DARK_GRAY}  └── {UI.GREEN}✓ Completed{UI.RST} {UI.GRAY}({line_count} lines, {char_count} chars){UI.RST}", flush=True)
            else:
                print(f"{UI.DARK_GRAY}  └── {UI.GREEN}✓ Completed{UI.RST}", flush=True)

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
        if len(self.history) > self.max_recent_turns:
            older = self.history[:-self.max_recent_turns]
            recent = self.history[-self.max_recent_turns:]
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
        self._active_local_agent = None
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
                print(f"{UI.AMBER_BOLD}[⚡ llama-shift]{UI.RST} {UI.WHITE}Seamless switch detected on {ep['name']} ➔ {ep['filename']}{UI.RST}")
            try:
                await self._active_local_agent.__aexit__(None, None, None)
            except Exception:
                pass
            self._active_local_agent = None

        config = LocalOpenAIAgentConfig(
            model=ep["model"],
            base_url=ep["url"],
            system_instructions=self.build_system_context(target),
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
            recent = self.history[-self.max_recent_turns:]
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
            print(f"{UI.GRAY}⚡ Contacting Google Cloud ({self.cloud_model})...{UI.RST}", flush=True)

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
                        term_cols = shutil.get_terminal_size((120, 24)).columns
                        max_summary = max(100, term_cols - 35)
                        summary_clean = str(summary).replace("\r\n", " ").replace("\n", " ").strip()
                        summary_str = f" ➔ {summary_clean}" if len(summary_clean) <= max_summary else f" ➔ {summary_clean[:max_summary - 3]}..."
                        tool_msg = f"{UI.AMBER_BOLD}[⚡ Cloud Tool: {tool_name}]{UI.RST}{UI.WHITE}{summary_str}{UI.RST}"
                        if status:
                            status.update(f"[bold yellow]⚙ Cloud Tool: {tool_name}[/bold yellow] [dim]{summary_str}[/dim]")
                        else:
                            sys.stdout.write(f"\r\033[K{tool_msg}\n")
                            sys.stdout.flush()

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
            try:
                proc.kill()
                stderr_task.cancel()
                await asyncio.wait_for(proc.wait(), timeout=2)
            except BaseException:
                pass
            if status:
                status.stop()
            renderer.stop()
            raise
        except asyncio.TimeoutError as te:
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

    async def _chat_impl(self, prompt: str, force_target: Optional[str] = None) -> str:
        target = self.resolve_target_alias(force_target or self.active_target)

        # Dynamic check of ports before routing
        await asyncio.to_thread(self.refresh_endpoints)

        # Telemetry & metric trackers
        self._current_turn_tool_count = 0
        self._recent_tool_calls.clear()
        executed_tools: List[Dict[str, Any]] = []
        self._turn_executed_tools = executed_tools
        token_count: int = 0
        tps: Optional[float] = None
        ttft_ms: Optional[float] = None
        total_dur: Optional[float] = None

        # Laya System 1 Auto-Routing (~33ms decision pass)
        if target == "auto":
            # Priority 1: Git and Version Control actions fast-track to local Gemma 4 MoE
            if self.auto_git_shift:
                is_git, git_conf = self.laya.is_git_action(prompt)
                if is_git:
                    msg = f"\n{UI.LAYA_BOLD}[⚡ Laya System 1 (~33ms): Git Action ➔ AMD ROCm (Gemma 4 A4B Fast MoE) (Confidence: {git_conf * 100:.1f}%)]\033[0m"
                    if not self.json_output:
                        print(msg)
                    else:
                        print(msg, file=sys.stderr)
            # Priority 2: Project Build, Compilation & Automated Tests push to NVIDIA Subagent (:9001)
            if target == "auto":
                is_build, build_conf = self.laya.is_build_action(prompt)
                if is_build:
                    msg = f"\n{UI.LAYA_BOLD}[⚡ Laya System 1 (~33ms): Build & Test Verification ➔ NVIDIA CUDA Subagent (:9001) (Confidence: {build_conf * 100:.1f}%)]\033[0m"
                    if not self.json_output:
                        print(msg)
                    else:
                        print(msg, file=sys.stderr)
                    target = "9001"

            if target == "auto":
                routed_id, conf = self.laya.route_prompt(prompt)
                if routed_id == "cloud":
                    routed_name = f"Google Cloud ({self.cloud_model})"
                elif routed_id in self.endpoints:
                    ep = self.endpoints[routed_id]
                    fn = ep.get("filename") or ep.get("name_friendly") or "Unknown Model"
                    routed_name = f"{ep['desc']} [{fn}]"
                else:
                    routed_name = routed_id
                msg = f"\n{UI.LAYA_BOLD}[⚡ Laya System 1 (~33ms): 'choice' ➔ {routed_name} (Confidence: {conf * 100:.1f}%)]\033[0m"
                if not self.json_output:
                    print(msg)
                else:
                    print(msg, file=sys.stderr)
                target = routed_id

        # Dynamic in-GPU Model Hot-Swap on Port 9000 (Gemma 4 A4B ↔ Qwen 27B Dense)
        if self.auto_git_shift and target in ("9000", "rocm", "local"):
            is_git, git_conf = self.laya.is_git_action(prompt)
            ep_9000 = self.endpoints.get("9000", {})
            curr_id = (ep_9000.get("id") or "").lower()
            curr_fn = (ep_9000.get("filename") or "").lower()
            curr_name = (ep_9000.get("name_friendly") or "").lower()

            if is_git:
                is_gemma_active = ("gemma" in curr_id or "gemma" in curr_fn or "gemma" in curr_name)
                if not is_gemma_active:
                    shift_msg = f"\n{UI.LAYA_BOLD}[⚡ Laya Auto-Shift (~33ms)]{UI.RST} {UI.WHITE}Git action detected (P={git_conf*100:.1f}%) ➔ Hot-swapping to Gemma 4 A4B for rapid execution...{UI.RST}"
                    if not self.json_output:
                        print(shift_msg)
                    else:
                        print(shift_msg, file=sys.stderr)
                    swapped = await self.switch_model(self.fast_git_model)
                    if swapped:
                        await asyncio.to_thread(self.refresh_endpoints)
            else:
                is_heavy, heavy_conf = self.laya.is_heavy_reasoning(prompt)
                if is_heavy:
                    is_qwen_active = ("qwen" in curr_id or "qwen" in curr_fn or "qwen" in curr_name)
                    if not is_qwen_active:
                        catalog = self.get_catalog()
                        has_dense = any(self.dense_code_model.lower() in m.get("id", "").lower() for m in catalog)
                        if has_dense:
                            shift_msg = f"\n{UI.LAYA_BOLD}[⚡ Laya Auto-Shift (~33ms)]{UI.RST} {UI.WHITE}Deep reasoning task detected (P={heavy_conf*100:.1f}%) ➔ Hot-swapping to Qwen 27B Dense...{UI.RST}"
                            if not self.json_output:
                                print(shift_msg)
                            else:
                                print(shift_msg, file=sys.stderr)
                            swapped = await self.switch_model(self.dense_code_model)
                            if swapped:
                                await asyncio.to_thread(self.refresh_endpoints)

        initial_target = target

        if target == "cloud":
            auth_info = "OAuth token" if self.has_oauth else "API key"
            if not self.json_output:
                print(f"\n{UI.CLOUD_BOLD}┌── [Google Cloud ✦ Gemini: {self.cloud_model} ✦ {auth_info}]{UI.RST}\n")
            else:
                print(f"[Google Cloud (Gemini: {self.cloud_model})] Thinking & responding...", file=sys.stderr)
            t_cloud_start = time.perf_counter()
            try:
                response_text = await self.chat_cloud_oauth(prompt)
                t_cloud_end = time.perf_counter()
                dur = t_cloud_end - t_cloud_start
                total_dur = dur
                if self.show_metrics and not self.json_output:
                    approx_words = len(response_text.split())
                    ctx_badge = self.format_context_badge(target, reset_col=UI.GRAY)
                    print(f"\n{UI.DARK_GRAY}└── {UI.badge(f'Metrics: Cloud Gemini │ Duration {dur:.2f}s │ ~{approx_words} words │ {ctx_badge}', UI.GRAY)}{UI.RST}\n")
            except Exception as cloud_err:
                msg = f"\n{UI.AMBER_BOLD}[⚠ Cloud model unreachable: {cloud_err}]{UI.RST}\n"
                if not self.json_output:
                    print(msg)
                else:
                    print(msg, file=sys.stderr)
                candidates = []
                if self.endpoints["9000"]["status"] == "online":
                    candidates.append("rocm")
                if self.endpoints["9001"]["status"] == "online":
                    candidates.append("cuda")
                if candidates:
                    winner, conf = self.laya.choose_escalation_target("cloud", prompt, candidates)
                    escalate_to = self.resolve_target_alias(winner)
                    ep_esc = self.endpoints[escalate_to]
                    esc_label = f"{ep_esc['name']} [{ep_esc['filename']}]"
                    esc_msg = f"{UI.LAYA_BOLD}[⚡ Laya Escalation (~33ms): Cloud unreachable ➔ Escalating to {esc_label} (Confidence: {conf*100:.1f}%)]\033[0m\n"
                    if not self.json_output:
                        print(esc_msg)
                    else:
                        print(esc_msg, file=sys.stderr)
                    agent = await self.get_or_switch_local_agent(escalate_to)
                    t_esc_start = time.perf_counter()
                    escalate_resp = await agent.chat(prompt)
                    response_text, esc_tok_count, t_esc_first = await self._consume_token_stream(escalate_resp)
                    target = escalate_to
                    t_esc_end = time.perf_counter()
                    total_dur = t_esc_end - t_esc_start
                    dur = t_esc_end - (t_esc_first or t_esc_start)
                    tps = (esc_tok_count / dur) if dur > 0 else 0
                    ttft_ms = ((t_esc_first - t_esc_start) * 1000) if t_esc_first else 0
                    if self.show_metrics and not self.json_output and esc_tok_count > 0:
                        ctx_badge = self.format_context_badge(target, reset_col=UI.GRAY)
                        print(f"{UI.DARK_GRAY}└── {UI.badge(f'Metrics: {tps:.1f} tok/s │ {esc_tok_count} tokens │ TTFT {ttft_ms:.0f}ms │ Total {total_dur:.2f}s │ {ctx_badge}', UI.GRAY)}{UI.RST}\n")
                else:
                    raise
        else:
            ep = self.endpoints[target]
            display_fn = ep["filename"] or ep["model"].split("/")[-1]
            t_color = UI.target_color(target)
            header = f"{ep['name']} ✦ {display_fn} ✦ {ep['desc']}"
            if not self.json_output:
                print(f"\n{t_color}┌── [{header}]{UI.RST}\n")
            else:
                print(f"[{header}] Thinking & responding...", file=sys.stderr)

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
                current_response = None
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

                    if self.show_metrics and not self.json_output and token_count > 0:
                        ctx_badge = self.format_context_badge(target, reset_col=UI.GRAY)
                        print(f"{UI.DARK_GRAY}└── {UI.badge(f'Metrics: {tps:.1f} tok/s │ {token_count} tokens │ TTFT {ttft_ms:.0f}ms │ Total {total_dur:.2f}s │ {ctx_badge}', UI.GRAY)}{UI.RST}\n")
                    break

                except Exception as turn_err:
                    if _is_exceed_context_error(turn_err):
                        compact_msg = (
                            f"\n{UI.AMBER_BOLD}[⚠ Context size exceeded!]{UI.RST} {UI.WHITE}Auto-compacting conversation history...{UI.RST}"
                        )
                        if not self.json_output:
                            print(compact_msg)
                        else:
                            print(compact_msg, file=sys.stderr)
                        removed = self.compact_history(keep_recent=4, silent=True)
                        if removed == 0 and len(self.history) > 2:
                            removed = self.compact_history(keep_recent=2, silent=True)
                        self._active_local_agent = None
                        if not self.json_output:
                            ctx_msg = f"{UI.GREEN}✓ Compacted {removed} turn(s). Retrying with fresh context window...{UI.RST}"
                            print(ctx_msg)
                        else:
                            print(f"[AutoCompact] Compacted {removed} turn(s). Retrying.", file=sys.stderr)
                        agent = await self.get_or_switch_local_agent(target)
                    elif _is_empty_turn_400(turn_err) and attempt < MAX_ATTEMPTS:
                        msg = (
                            f"\n{UI.AMBER_BOLD}[⚠ Model produced empty turn after tool execution "
                            f"(HTTP 400, attempt {attempt}/{MAX_ATTEMPTS}). Pruning history and retrying...]{UI.RST}\n"
                        )
                        if not self.json_output:
                            print(msg)
                        else:
                            print(msg, file=sys.stderr)
                        _prune_empty_history()
                        self._active_local_agent = None
                        agent = await self.get_or_switch_local_agent(target)
                    elif _is_empty_turn_400(turn_err) or _is_unreachable_error(turn_err):
                        _prune_empty_history()
                        self._active_local_agent = None
                        await asyncio.to_thread(self.refresh_endpoints)

                        candidates = []
                        if target != "9000" and self.endpoints["9000"]["status"] == "online":
                            candidates.append("rocm")
                        if target != "9001" and self.endpoints["9001"]["status"] == "online":
                            candidates.append("cuda")
                        candidates.append("cloud")

                        winner, conf = self.laya.choose_escalation_target(target, prompt, candidates)
                        escalate_to = self.resolve_target_alias(winner)

                        if escalate_to == "cloud":
                            escalate_label = f"Cloud ({self.cloud_model})"
                        else:
                            ep_esc = self.endpoints[escalate_to]
                            escalate_label = f"{ep_esc['name']} [{ep_esc['filename']}]"

                        cause = "Endpoint unreachable" if _is_unreachable_error(turn_err) else f"Failed after {MAX_ATTEMPTS} attempts"
                        esc_msg = (
                            f"\n{UI.AMBER_BOLD}⚠ [{target.upper()} {cause}]{UI.RST}\n"
                            f"{UI.LAYA_BOLD}[⚡ Laya Escalation (~33ms): Escalating to {escalate_label} (Confidence: {conf*100:.1f}%)]\033[0m\n"
                        )
                        if not self.json_output:
                            print(esc_msg)
                        else:
                            print(esc_msg, file=sys.stderr)

                        if escalate_to == "cloud":
                            t_esc_start = time.perf_counter()
                            response_text = await self.chat_cloud_oauth(prompt)
                            t_esc_end = time.perf_counter()
                            total_dur = t_esc_end - t_esc_start
                            target = "cloud"
                            if self.show_metrics and not self.json_output:
                                ctx_badge = self.format_context_badge(target, reset_col=UI.GRAY)
                                print(f"\n{UI.DARK_GRAY}└── {UI.badge(f'Metrics: Cloud Gemini │ Duration {total_dur:.2f}s │ {ctx_badge}', UI.GRAY)}{UI.RST}\n")
                        else:
                            agent = await self.get_or_switch_local_agent(escalate_to)
                            t_esc_start = time.perf_counter()
                            escalate_resp = await agent.chat(prompt)
                            response_text, esc_tok_count, t_esc_first = await self._consume_token_stream(escalate_resp)
                            t_esc_end = time.perf_counter()
                            token_count = esc_tok_count
                            ttft_ms = ((t_esc_first - t_esc_start) * 1000) if t_esc_first else 0
                            dur = t_esc_end - (t_esc_first or t_esc_start)
                            tps = (esc_tok_count / dur) if dur > 0 else 0
                            total_dur = t_esc_end - t_esc_start
                            target = escalate_to
                            if self.show_metrics and not self.json_output and esc_tok_count > 0:
                                ctx_badge = self.format_context_badge(target, reset_col=UI.GRAY)
                                print(f"{UI.DARK_GRAY}└── {UI.badge(f'Metrics: {tps:.1f} tok/s │ {esc_tok_count} tokens │ TTFT {ttft_ms:.0f}ms │ Total {total_dur:.2f}s │ {ctx_badge}', UI.GRAY)}{UI.RST}\n")
                        break
                    else:
                        raise

        self.history.append({"role": "user", "target": target, "content": prompt})
        if response_text:
            self.history.append({"role": "assistant", "target": target, "content": response_text})
        else:
            if not self.json_output:
                print(UI.warn("Model returned an empty response — skipping history entry to prevent API errors."))

        # Proactive autocompact
        if self.autocompact_threshold > 0 and len(self.history) >= self.autocompact_threshold:
            removed = self.compact_history(silent=True)
            if removed > 0:
                if not self.json_output:
                    print(f"\n{UI.DARK_GRAY}╭── {UI.CYAN_BOLD}[Auto-compact]{UI.RST} {UI.GRAY}Context reached {self.autocompact_threshold} turns → compacted {removed} old turns into summary.{UI.RST}\n")
                else:
                    print(f"[AutoCompact] Proactively compacted {removed} turns (threshold={self.autocompact_threshold}).", file=sys.stderr)
        else:
            self.trigger_background_synthesis()

        approx_tokens = token_count if token_count else len(response_text.split())
        self._record_stats(initial_target, target, approx_tokens, tps, ttft_ms, total_dur)

        if self.json_output:
            json_payload = {
                "status": "success",
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
        if self.enable_hooks:
            verdict = await asyncio.to_thread(ExternalHooksManager.run_turn_event, self, "PreTurn", {"prompt": prompt})
            if verdict.get("deny"):
                msg = f"Turn blocked by PreTurn hook: {verdict.get('reason') or 'denied'}"
                print(UI.warn(msg) if not self.json_output else msg, file=sys.stderr if self.json_output else sys.stdout)
                return ""
            prompt = verdict.get("prompt") or prompt

        before = await self._checkpoint_before()
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
                    print(UI.warn(note) if not self.json_output else note, file=sys.stderr if self.json_output else sys.stdout)
                    response = await self._chat_impl(stop["reason"], force_target) or response
        except Exception:
            st = self.stats.setdefault(self.resolve_target_alias(force_target or self.active_target), {})
            st["errors"] = st.get("errors", 0) + 1
            raise
        finally:
            self._checkpoint_after(before, prompt)
        return response

    # ── Checkpoints (/diff, /undo) ────────────────────────────────────

    async def _checkpoint_before(self) -> Optional[str]:
        if self._repo_root is None:
            self._repo_root = await asyncio.to_thread(Checkpoints.repo_root) or ""
        if not self._repo_root:
            return None
        return await asyncio.to_thread(Checkpoints.snapshot, self._repo_root)

    def _checkpoint_after(self, before: Optional[str], prompt: str):
        if not before or not self._repo_root:
            return
        after = Checkpoints.snapshot(self._repo_root)
        if after and after != before:
            self.checkpoints.append({"prompt": prompt.strip().splitlines()[0][:80] if prompt.strip() else "", "before": before, "after": after, "time": time.time()})
            del self.checkpoints[:-Checkpoints.MAX_TURNS]

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
        self._active_local_agent = None
        return touched

    # ── Usage ledger (/stats) ─────────────────────────────────────────

    def _record_stats(self, routed: str, final: str, tokens: int, tps: Optional[float], ttft_ms: Optional[float], dur: Optional[float]):
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
            print(f"\n{UI.GRAY}(no turns recorded yet this session){UI.RST}\n")
            return
        rows = []
        for tgt, st in sorted(self.stats.items()):
            tps = f"{st['tps_sum'] / st['tps_n']:.1f}" if st.get("tps_n") else "-"
            ttft = f"{st['ttft_sum'] / st['ttft_n']:.0f}ms" if st.get("ttft_n") else "-"
            label = self.endpoints[tgt]["name"] if tgt in self.endpoints else ("Cloud" if tgt == "cloud" else tgt)
            rows.append((label, int(st.get("turns", 0)), int(st.get("tokens", 0)), tps, ttft, f"{st.get('seconds', 0.0):.1f}s", int(st.get("escalations", 0)), int(st.get("errors", 0))))
        headers = ("Target", "Turns", "Tokens", "Avg tok/s", "Avg TTFT", "Time", "Escalated away", "Errors")
        if RICH_AVAILABLE and not self.json_output:
            table = Table(title=Text.from_ansi(f"{UI.WHITE}SESSION USAGE LEDGER{UI.RST}"), box=box.ROUNDED, header_style="bold cyan", border_style="bright_black")
            for i, h in enumerate(headers):
                table.add_column(h, justify="left" if i == 0 else "right")
            for r in rows:
                table.add_row(*[str(c) for c in r])
            console.print()
            console.print(table)
            console.print(f"[dim]Cloud token counts are word-count estimates.[/dim]\n")
        else:
            print()
            for r in rows:
                print("  " + " | ".join(f"{h}: {c}" for h, c in zip(headers, r)))
            print()

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

    async def chat_turn(self, prompt: str, target: Optional[str] = None):
        """Runs one chat turn on an explicit target (used by workflows)."""
        await self.chat(prompt, force_target=target)

    async def reset_local_agent(self):
        """Closes and drops the cached local agent so the next turn starts fresh."""
        agent, self._active_local_agent = self._active_local_agent, None
        if agent is not None:
            try:
                await agent.__aexit__(None, None, None)
            except Exception:
                pass

    async def chat_turn(self, prompt: str, target: Optional[str] = None):
        """Convenience alias for chat() with explicit target routing."""
        return await self.chat(prompt, force_target=target)

    async def close(self):
        if self.history:
            try:
                self.save_session("latest_session")
            except Exception:
                pass
        if self._active_local_agent is not None:
            try:
                await self._active_local_agent.__aexit__(None, None, None)
            except Exception:
                pass
            self._active_local_agent = None
        SudoManager.cleanup()

    # ── Status, Catalog & Model Switch UI ─────────────────────────────

    def print_status(self):
        self.refresh_endpoints()
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

        ep9000 = self.endpoints['9000']
        ep9001 = self.endpoints['9001']

        def _fmt_model(ep, col):
            fn = ep['filename']
            st_color = UI.GREEN if ep['status'] == 'online' else UI.RED
            status_badge = f"{st_color}[{ep['status'].upper()}]{UI.RST}"
            model_name = f"{col}{ep['name_friendly'] or fn}{UI.RST}"
            src = f" {UI.GRAY}[{ep['source']}]{UI.RST}" if ep['source'] != 'none' else ""
            return f"{status_badge} {model_name}{src}"

        m0_str = _fmt_model(ep9000, UI.ROCM)
        m1_str = _fmt_model(ep9001, UI.CUDA)
        ml_status = f"{UI.GREEN}ON{UI.RST} (Enter on empty line submits)" if self.multiline_input else f"{UI.GRAY}OFF{UI.RST} (single Enter)"
        curation_badge = f"{UI.GREEN}Active (Auto-distills in background){UI.RST}" if self.continuous_curation else f"{UI.GRAY}Disabled{UI.RST}"

        if RICH_AVAILABLE and not self.json_output:
            table = Table(
                title=Text.from_ansi(f"{UI.WHITE}MULTI-GPU HYBRID AGENT STATUS{UI.RST}"),
                box=box.ROUNDED,
                header_style="bold cyan",
                border_style="bright_black",
                show_header=False,
            )
            table.add_column("Property", style="bold white", width=25)
            table.add_column("Value")
            table.add_row("Active Target", Text.from_ansi(active_label))
            table.add_row("Permissions", Text.from_ansi(perm_desc))
            table.add_row("Laya Decision Engine", Text.from_ansi(f"{UI.LAYA}ModernBERT-large (~421M, ~33ms){UI.RST} {UI.GRAY}[/v1/systemone]{UI.RST}"))
            table.add_row("Primary Agent (ROCm :9000)", Text.from_ansi(m0_str))
            table.add_row("Compactor (NVIDIA :9001)", Text.from_ansi(m1_str))
            table.add_row("Continuous Curation", Text.from_ansi(curation_badge))
            repomap_tag = f"{UI.GREEN}Active (Auto-injected into context){UI.RST}" if self.auto_repomap else f"{UI.GRAY}Disabled{UI.RST}"
            table.add_row("Codebase AST Map", Text.from_ansi(repomap_tag))
            _, instr_files = ProjectInstructions.load_instructions(".")
            instr_desc = f"{UI.GREEN}Loaded ({', '.join(instr_files)}){UI.RST}" if instr_files else (f"{UI.GREEN}Active (Scanning root){UI.RST}" if self.auto_instructions else f"{UI.GRAY}Disabled{UI.RST}")
            table.add_row("Project Guidelines", Text.from_ansi(instr_desc))
            hook_entries, hook_files = ExternalHooksManager.load_hook_configs(".")
            hook_tag = f"{UI.GREEN}Active ({len(hook_entries)} hooks loaded from {', '.join(hook_files)}){UI.RST}" if hook_entries else (f"{UI.GREEN}Active (Scanning hooks.json){UI.RST}" if self.enable_hooks else f"{UI.GRAY}Disabled{UI.RST}")
            table.add_row("Lifecycle Hooks", Text.from_ansi(hook_tag))

            if self.enable_kontexta:
                k_stat = KontextaManager.get_status(".")
                if k_stat["available"]:
                    p_info = k_stat["active_project"]["name"] if k_stat["active_project"] else f"{k_stat['projects_count']} projects"
                    k_str = f"{UI.GREEN}Active{UI.RST} (Inbuilt MCP 'kxta' ➔ {p_info})"
                else:
                    k_str = f"{UI.AMBER}Offline (Node script missing){UI.RST}"
            else:
                k_str = f"{UI.GRAY}Disabled{UI.RST}"
            table.add_row("Kontexta Engine", Text.from_ansi(k_str))

            table.add_row("Cloud Model", Text.from_ansi(f"{UI.CLOUD}{self.cloud_model}{UI.RST}"))
            table.add_row("Cloud Auth", Text.from_ansi(auth_status))
            table.add_row("Multi-line Input", Text.from_ansi(ml_status))
            table.add_row("Conversation Turns", Text.from_ansi(f"{UI.CYAN}{len(self.history) // 2}{UI.RST}"))
            shift_tag = f"{UI.GREEN}Active (Git ➔ Gemma 4 A4B │ Deep ➔ Qwen 27B){UI.RST}" if self.auto_git_shift else f"{UI.GRAY}Disabled{UI.RST}"
            table.add_row("Auto Model Shift", Text.from_ansi(shift_tag))
            sudo_badge = f"{UI.GREEN}Active (~{int(SudoManager.time_remaining() // 60)}m cached){UI.RST}" if SudoManager.is_authenticated() else f"{UI.GRAY}Inactive (Prompt on demand){UI.RST}"
            table.add_row("Sudo Privileges", Text.from_ansi(sudo_badge))
            console.print()
            console.print(table)
            console.print()
            return

        print()
        print(f"{UI.GRAY}╭─── {UI.WHITE}MULTI-GPU HYBRID AGENT STATUS{UI.RST}{UI.GRAY} ──────────────────────────────────────╮{UI.RST}")
        print(f"{UI.GRAY}│{UI.RST}  {UI.WHITE}Active Target{UI.RST}       : {active_label}")
        print(f"{UI.GRAY}│{UI.RST}  {UI.WHITE}Permissions{UI.RST}         : {perm_desc}")
        sudo_badge_text = f"{UI.GREEN}Active (~{int(SudoManager.time_remaining() // 60)}m cached){UI.RST}" if SudoManager.is_authenticated() else f"{UI.GRAY}Inactive (Prompt on demand){UI.RST}"
        print(f"{UI.GRAY}│{UI.RST}  {UI.WHITE}Sudo Privileges{UI.RST}     : {sudo_badge_text}")
        print(f"{UI.GRAY}│{UI.RST}  {UI.WHITE}Laya Decision Engine{UI.RST}: {UI.LAYA}ModernBERT-large (~421M, ~33ms){UI.RST} [/v1/systemone]")
        print(f"{UI.GRAY}│{UI.RST}  {UI.WHITE}Primary (ROCm :9000){UI.RST} : {m0_str}")
        print(f"{UI.GRAY}│{UI.RST}  {UI.WHITE}Compactor (CUDA :9001){UI.RST}: {m1_str}")
        print(f"{UI.GRAY}│{UI.RST}  {UI.WHITE}Continuous Curation{UI.RST} : {curation_badge}")
        print(f"{UI.GRAY}│{UI.RST}  {UI.WHITE}Codebase AST Map{UI.RST}    : {repomap_tag}")
        _, instr_files = ProjectInstructions.load_instructions(".")
        instr_desc = f"{UI.GREEN}Loaded ({', '.join(instr_files)}){UI.RST}" if instr_files else (f"{UI.GREEN}Active (Scanning root){UI.RST}" if self.auto_instructions else f"{UI.GRAY}Disabled{UI.RST}")
        print(f"{UI.GRAY}│{UI.RST}  {UI.WHITE}Project Guidelines{UI.RST}  : {instr_desc}")
        hook_entries, hook_files = ExternalHooksManager.load_hook_configs(".")
        hook_tag = f"{UI.GREEN}Active ({len(hook_entries)} hooks loaded from {', '.join(hook_files)}){UI.RST}" if hook_entries else (f"{UI.GREEN}Active (Scanning hooks.json){UI.RST}" if self.enable_hooks else f"{UI.GRAY}Disabled{UI.RST}")
        print(f"{UI.GRAY}│{UI.RST}  {UI.WHITE}Lifecycle Hooks{UI.RST}     : {hook_tag}")
        if self.enable_kontexta:
            k_stat = KontextaManager.get_status(".")
            if k_stat["available"]:
                p_info = k_stat["active_project"]["name"] if k_stat["active_project"] else f"{k_stat['projects_count']} projects"
                k_str = f"{UI.GREEN}Active{UI.RST} (Inbuilt MCP 'kxta' ➔ {p_info})"
            else:
                k_str = f"{UI.AMBER}Offline (Node script missing){UI.RST}"
        else:
            k_str = f"{UI.GRAY}Disabled{UI.RST}"
        print(f"{UI.GRAY}│{UI.RST}  {UI.WHITE}Kontexta Engine{UI.RST}     : {k_str}")
        shift_tag_text = f"{UI.GREEN}Active (Git ➔ Gemma 4 A4B │ Deep ➔ Qwen 27B){UI.RST}" if self.auto_git_shift else f"{UI.GRAY}Disabled{UI.RST}"
        print(f"{UI.GRAY}│{UI.RST}  {UI.WHITE}Auto Model Shift{UI.RST}    : {shift_tag_text}")
        print(f"{UI.GRAY}│{UI.RST}  {UI.WHITE}Cloud Model{UI.RST}         : {UI.CLOUD}{self.cloud_model}{UI.RST}")
        print(f"{UI.GRAY}│{UI.RST}  {UI.WHITE}Cloud Auth{UI.RST}          : {auth_status}")
        print(f"{UI.GRAY}│{UI.RST}  {UI.WHITE}Multi-line Input{UI.RST}    : {ml_status}")
        print(f"{UI.GRAY}│{UI.RST}  {UI.WHITE}Conversation Turns{UI.RST}  : {UI.CYAN}{len(self.history) // 2}{UI.RST}")
        print(f"{UI.GRAY}╰───────────────────────────────────────────────────────────────────╯{UI.RST}")
        print()

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
        if RICH_AVAILABLE and not self.json_output:
            table = Table(
                title=Text.from_ansi(f"{UI.WHITE}MODEL CATALOG & HARDWARE ENDPOINTS{UI.RST} {UI.GRAY}(Select with '/model <# or id>'){UI.RST}"),
                box=box.ROUNDED,
                header_style="bold cyan",
                border_style="bright_black"
            )
            table.add_column("#", justify="center", style="bold dim", no_wrap=True)
            table.add_column("ID / Alias", style="bold cyan", no_wrap=True)
            table.add_column("Status", justify="left", no_wrap=True)
            table.add_column("Target Device & Port", style="white", no_wrap=True)
            table.add_column("Model (Size)", style="none")
            table.add_column("Specialty / Capability", style="dim")

            for idx, item in enumerate(catalog, 1):
                st = item["status"].upper()
                if st in ("RUNNING", "ONLINE"):
                    st_str = f"{UI.GREEN}● {st}{UI.RST}"
                else:
                    st_str = f"{UI.GRAY}○ {st}{UI.RST}"

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
                model_str = f"{UI.WHITE}{item['name']}{UI.RST}{UI.GRAY}{size_tag}{UI.RST}"

                table.add_row(
                    str(idx),
                    item["id"],
                    Text.from_ansi(st_str),
                    Text.from_ansi(dev_styled),
                    Text.from_ansi(model_str),
                    item.get("desc") or ""
                )

            console.print()
            console.print(table)
            tip_msg = (
                f"{UI.GRAY}💡 Select: {UI.CYAN}/model <# or id>{UI.RST}{UI.GRAY} (e.g. {UI.CYAN}/model 4{UI.RST}{UI.GRAY} or {UI.CYAN}/model gemma4{UI.RST}{UI.GRAY})\n"
                f"   One-shot query: {UI.CYAN}@<id> <prompt>{UI.RST}{UI.GRAY} (auto hot-swaps if inactive){UI.RST}\n"
            )
            print(tip_msg)
            return

        print()
        print(f"{UI.GRAY}╭─── {UI.WHITE}MODEL CATALOG & HARDWARE ENDPOINTS{UI.RST}{UI.GRAY} ─────────────────────────────╮{UI.RST}")
        for idx, item in enumerate(catalog, 1):
            st = item["status"].upper()
            st_color = UI.GREEN if st in ("RUNNING", "ONLINE") else UI.GRAY
            active_mark = f" {UI.AMBER_BOLD}★ ACTIVE{UI.RST}" if item["is_active"] else ""
            size_tag = f" ({item['size']})" if item.get("size") else ""
            print(f"{UI.GRAY}│{UI.RST} [{idx}] {UI.CYAN}{item['id']:13s}{UI.RST} {st_color}[{st:7s}]{UI.RST}{active_mark} ➔ {UI.WHITE}{item['name']}{size_tag}{UI.RST}")
            print(f"{UI.GRAY}│{UI.RST}     Target: {item['gpu_desc']} | {item.get('desc') or ''}")
        print(f"{UI.GRAY}╰──────────────────────────────────────────────────────────────────╯{UI.RST}")
        print(f"{UI.GRAY}Type /model <# or id> to select. Inactive models hot-swap automatically.{UI.RST}\n")

    async def switch_model(self, query: str) -> bool:
        """
        Switches the active model by index (1..N), model ID, or alias.
        If the selected model is currently stopped, triggers llama-shift hot-swap.
        """
        query_str = str(query).strip()
        if not query_str:
            self.print_models()
            return False

        catalog = self.get_catalog()
        selected = None

        if query_str.isdigit():
            idx = int(query_str)
            if 1 <= idx <= len(catalog):
                selected = catalog[idx - 1]
            else:
                print(UI.warn(f"Model number {idx} out of range (1-{len(catalog)}). Type /models to view list."))
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
                    print(UI.warn(f"Multiple models matched '{query_str}': {names}. Please use /model <#> to specify."))
                    return False

        if not selected:
            print(UI.warn(f"Model '{query_str}' not found. Type /models to view available models."))
            return False

        mid = selected["id"]
        mtype = selected["type"]

        if mtype == "cloud":
            self.active_target = "cloud"
            print(UI.ok(f"Active target set to Google Cloud Gemini ({self.cloud_model}) via OAuth"))
            return True

        if mtype == "system" or mid == "auto":
            self.active_target = "auto"
            print(UI.ok("Active target set to Laya System 1 Auto-Routing (~33ms)"))
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
            self.refresh_endpoints()
            print(UI.ok(f"Target set to {gpu_name} [{dev_str}] (Port {port}): {model_name} (already running)"))
            return True

        # Inactive model: trigger hot-swap
        print(f"\n{UI.AMBER}[⚡ Llama-Shift]{UI.RST} Hot-swapping {UI.WHITE}{model_name}{UI.RST} ({model_size}) onto {UI.CYAN}{gpu_name} [{dev_str}]{UI.RST} (Port {port})...")
        res = await asyncio.to_thread(trigger_llamashift_switch, mid)
        if "error" in res:
            print(UI.err(f"Llama-Shift switch request failed: {res['error']}"))
            return False

        start_time = time.time()
        timeout = 45.0
        success = False
        target_port_int = int(port) if str(port).isdigit() else 9000

        if RICH_AVAILABLE and not self.json_output:
            with console.status(f"[bold cyan]Loading {model_name} into VRAM ({gpu_name})...", spinner="dots") as spin:
                while time.time() - start_time < timeout:
                    elapsed = time.time() - start_time
                    spin.update(f"[bold cyan]Loading {model_name} into {gpu_name} [{dev_str}] VRAM... [bold yellow]{elapsed:.1f}s[/bold yellow]")
                    await asyncio.sleep(0.5)
                    try:
                        req = urllib.request.Request(f"http://localhost:{target_port_int}/health", headers={"User-Agent": "Antigravity/1.0"})
                        with urllib.request.urlopen(req, timeout=0.8) as resp:
                            if resp.status == 200:
                                success = True
                                break
                    except Exception:
                        pass
        else:
            print(f"{UI.GRAY}[*] Loading weights into VRAM...{UI.RST}", end="", flush=True)
            while time.time() - start_time < timeout:
                await asyncio.sleep(1.0)
                print(".", end="", flush=True)
                try:
                    req = urllib.request.Request(f"http://localhost:{target_port_int}/health", headers={"User-Agent": "Antigravity/1.0"})
                    with urllib.request.urlopen(req, timeout=0.8) as resp:
                        if resp.status == 200:
                            success = True
                            print()
                            break
                except Exception:
                    pass

        if not success:
            print(UI.err(f"\nTimed out after {timeout:.0f}s waiting for {model_name} on port {target_port_int}."))
            return False

        elapsed = time.time() - start_time
        if self._active_local_agent:
            try:
                await self._active_local_agent.__aexit__(None, None, None)
            except Exception:
                pass
            self._active_local_agent = None

        self.refresh_endpoints()
        self.active_target = str(port)
        print(UI.ok(f"Hot-swap complete in {elapsed:.1f}s! Active target set to {gpu_name} (Port {port}) ➔ {model_name}\n"))
        return True

    def print_help(self):
        w = UI.WHITE
        g = UI.GRAY
        c = UI.CYAN
        r = UI.RST
        print(f"\n{UI.GRAY}╭─── {UI.WHITE}COMMAND PALETTE & KEYBOARD CONTROLS{UI.RST}{UI.GRAY} ────────────────────────╮{UI.RST}")
        print(f"{g}│{r} {UI.BOLD}Target & Model Switching:{r}")
        print(f"{g}│{r}   {c}/models{r}              : List all models (running & stopped) in catalog")
        print(f"{g}│{r}   {c}/model <# or id>{r}     : Switch model (auto hot-swaps inactive via llama-shift)")
        print(f"{g}│{r}   {c}/auto{r}               : Laya System 1 auto-routing (routes in ~33ms)")
        print(f"{g}│{r}   {c}/9000{r} or {c}/rocm{r}       : AMD ROCm GPU (Port 9000)")
        print(f"{g}│{r}   {c}/9001{r} or {c}/cuda{r}       : NVIDIA CUDA RTX 4060 (Port 9001)")
        print(f"{g}│{r}   {c}/cloud{r}              : Google Gemini (uses your OAuth token)")
        print(f"{g}│{r}   {c}/mode <target>{r}      : Switch target by alias, model ID, or number")
        print(f"{g}│{r}   {c}/git-shift [on|off]{r} : Auto-swap to Gemma 4 A4B for git actions & Qwen 27B for deep code")
        print(f"{g}│{r}")
        print(f"{g}│{r} {UI.BOLD}One-Shot Prefixes:{r}")
        print(f"{g}│{r}   {c}@<model_id> <prompt>{r} : Route prompt to any model (auto hot-swaps if inactive)")
        print(f"{g}│{r}   {c}@rocm <prompt>{r}      : Route single prompt to ROCm (Port 9000)")
        print(f"{g}│{r}   {c}@cuda <prompt>{r}      : Route single prompt to CUDA (Port 9001)")
        print(f"{g}│{r}   {c}@cloud <prompt>{r}     : Route single prompt to Cloud Gemini")
        print(f"{g}│{r}")
        print(f"{g}│{r} {UI.BOLD}Multi-line Input & Controls:{r}")
        print(f"{g}│{r}   {c}<Esc>{r}                  : Instantly stop model generation and regain prompt control")
        print(f"{g}│{r}   {c}<Ctrl+C>{r}               : Cancel current generation (press twice at prompt to exit)")
        print(f"{g}│{r}   {c}<Enter on empty line>{r}: Submit multi-line prompt (or press Ctrl+D)")
        print(f"{g}│{r}   {c}/multiline [on|off]{r}  : Toggle multi-line input mode (default: ON)")
        print(f"{g}│{r}   {c}/singleline{r}          : Switch to single-line prompt mode")
        print(f"{g}│{r}   {c}/paste{r}               : Enter dedicated paste mode")
        print(f"{g}│{r}")
        print(f"{g}│{r} {UI.BOLD}Session Management & Persistence:{r}")
        print(f"{g}│{r}   {c}/save [name]{r}         : Save conversation session to disk")
        print(f"{g}│{r}   {c}/load <name>{r}         : Restore a saved session")
        print(f"{g}│{r}   {c}/sessions{r}            : List all saved sessions")
        print(f"{g}│{r}   {c}/export [path.md]{r}    : Export transcript as Markdown")
        print(f"{g}│{r}   {c}/summarize{r}           : View condensed summary of context")
        print(f"{g}│{r}   {c}/context{r} or {c}/ctx{r}       : Show live context size, headroom & usage bar")
        print(f"{g}│{r}   {c}/compact [keep]{r}        : Compress history into summary (preserves context)")
        print(f"{g}│{r}   {c}/autocompact [on|off]{r}  : Auto-compress context before hitting window limit")
        print(f"{g}│{r}   {c}/curation [on|off]{r}    : Toggle continuous RTX 4060 background hippocampus")
        print(f"{g}│{r}   {c}/compactor [start]{r}    : Start/check dedicated RTX 4060 context engine")
        print(f"{g}│{r}   {c}/reload{r}               : Hot-reload agent code in-place (preserves context)")
        print(f"{g}│{r}   {c}/metrics [on|off]{r}    : Toggle generation speed (tok/s, TTFT) badge")
        print(f"{g}│{r}")
        print(f"{g}│{r} {UI.BOLD}Subagent Delegation & Codebase Map:{r}")
        print(f"{g}│{r}   {c}/spawn <task>{r}         : Delegate background task to NVIDIA RTX 4060 (compactor yields)")
        print(f"{g}│{r}   {c}/build [cmd]{r}          : Push project build & test verification to NVIDIA RTX 4060 Subagent (:9001)")
        print(f"{g}│{r}   {c}/tasks{r}                : List active and completed background subagents")
        print(f"{g}│{r}   {c}/subagent view <#>{r}    : View full output and tool steps of subagent #")
        print(f"{g}│{r}   {c}/subagent inject <#>{r}  : Inject subagent findings into current conversation context")
        print(f"{g}│{r}   {c}/subagent cancel <#>{r}  : Cancel a running subagent task")
        print(f"{g}│{r}   {c}/cuda{r}                 : Show NVIDIA CUDA:9001 resource arbitrator state")
        print(f"{g}│{r}   {c}/repomap [path]{r}       : Generate AST symbol map of codebase classes & functions")
        print(f"{g}│{r}   {c}/instructions [view]{r}   : Inspect or toggle auto-injected GEMINI.md / antigravity.md")
        print(f"{g}│{r}   {c}/hooks [view|refresh]{r}  : Inspect, reload or toggle lifecycle hooks (hooks.json)")
        print(f"{g}│{r}   {c}/kontexta [on|off|reg]{r} : Inbuilt Kontexta MCP knowledge engine & project linking")
        print(f"{g}│{r} {UI.BOLD}Autonomous Spec-Driven Workflow:{r}")
        print(f"{g}│{r}   {c}/brainstorm <idea>{r}    : Full Cloud Spec ➔ Approval ➔ ROCm Code ➔ NVIDIA Build Subagent ➔ Dual Review ➔ Commit")
        print(f"{g}│{r}")
        print(f"{g}│{r} {UI.BOLD}Web Search & Live Documentation:{r}")
        print(f"{g}│{r}   {c}/search <query>{r}       : Live web search (DuckDuckGo + GitHub) with clean snippets")
        print(f"{g}│{r}   {c}/fetch <url>{r}          : Fetch webpage & convert HTML to clean markdown text")
        print(f"{g}│{r}")
        print(f"{g}│{r} {UI.BOLD}Review, Undo & Usage:{r}")
        print(f"{g}│{r}   {c}/diff [n]{r}            : Show file changes from the nth latest turn (default 1)")
        print(f"{g}│{r}   {c}/undo{r}                : Revert the latest turn's file changes (asks first)")
        print(f"{g}│{r}   {c}/stats [reset]{r}       : Per-target tokens, tok/s, TTFT, escalations, errors")
        print(f"{g}│{r}   {c}/markdown [on|off]{r}   : Toggle rendered Markdown for responses")
        print(f"{g}│{r}   {c}/prefs [reset]{r}       : Show or reset saved preferences (~/.myagy/config.json)")
        print(f"{g}│{r}")
        print(f"{g}│{r} {UI.BOLD}Scripting & Output Formatting:{r}")
        print(f"{g}│{r}   {c}/json{r}                : Toggle structured JSON output mode")
        print(f"{g}│{r}   {c}/format json|text{r}    : Switch format between JSON and streaming text")
        print(f"{g}│{r}")
        print(f"{g}│{r} {UI.BOLD}Permissions & Decision Control:{r}")
        print(f"{g}│{r}   {c}/permissions auto{r}    : Enable Laya Bayesian dynamic gating ('noul')")
        print(f"{g}│{r}   {c}/permissions skip{r}    : Auto-approve all tool execution requests")
        print(f"{g}│{r}   {c}/permissions safe{r}    : Prompt before running any command")
        print(f"{g}│{r}   {c}/sudo [status|clear|<cmd>]{r}: Sudo credentials management & elevated execution")
        print(f"{g}│{r}   {c}!<command>{r}                 : Directly execute a shell command in host terminal")
        print(f"{g}│{r}   {c}/steps off|<#>{r}        : Disable or set max tool steps per turn (default: off)")
        print(f"{g}│{r}   {c}/verbose [on|off]{r}     : Toggle full tool arguments & output traces")
        print(f"{g}│{r}   {c}/laya{r}                 : Display Laya System 1 Decision Model details")
        print(f"{g}│{r}   {c}/status{r}               : Show active GPU/endpoint status & OAuth")
        print(f"{g}│{r}   {c}/models{r}               : List endpoints and active GGUF models")
        print(f"{g}│{r}   {c}/history{r}              : Show cross-GPU conversation history")
        print(f"{g}│{r}   {c}/clear{r}                : Clear conversation memory")
        print(f"{g}│{r}   {c}/exit{r}, {c}exit{r}, or {c}quit{r}   : Exit session")
        print(f"{UI.GRAY}╰──────────────────────────────────────────────────────────────────╯{UI.RST}\n")
