"""
Antigravity Lifecycle Hooks Loader (hooks.json).
Discovers and executes external shell commands at Agent lifecycle points:
  - PreToolUse: Gate, block, or rewrite tool parameters
  - PostToolUse: Run post-execution linters, analysis, or cleanup
  - PreTurn / PostTurn: Lifecycle context hooks
  - Stop: Enforce completion conditions

Conforms to Antigravity Customization Specification (.agents/hooks.json, ~/.gemini/config/hooks.json).
"""

import json
import os
import re
import signal
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from google.antigravity import hooks, types
from ..terminal.ui import UI


HOOKS_CANDIDATE_PATHS = [
    # Workspace specific
    os.path.join(".agents", "hooks.json"),
    os.path.join(".agent", "hooks.json"),
    "hooks.json",
    # Global machine config
    os.path.expanduser("~/.gemini/config/hooks.json"),
]


EVENT_KEYS = ("PreToolUse", "PostToolUse", "PreTurn", "PostTurn", "PreInvocation", "PostInvocation", "Stop")


def _normalize_hook(h: Any) -> Optional[Dict[str, Any]]:
    if isinstance(h, str):
        h = {"command": h}
    if not isinstance(h, dict) or not isinstance(h.get("command"), str) or not h["command"].strip():
        return None
    timeout = h.get("timeout", 30)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        timeout = 30
    return {"command": h["command"], "timeout": timeout}


def _normalize_event(value: Any) -> List[Dict[str, Any]]:
    """Coerces one event entry into [{"matcher": str, "hooks": [{"command", "timeout"}]}]."""
    if value is None:
        return []
    if isinstance(value, (dict, str)):
        value = [value]
    if not isinstance(value, list):
        return []
    groups = []
    for group in value:
        if isinstance(group, str):
            group = {"hooks": [group]}
        if not isinstance(group, dict):
            continue
        raw_hooks = group.get("hooks", [])
        if isinstance(raw_hooks, (dict, str)):
            raw_hooks = [raw_hooks]
        if not isinstance(raw_hooks, list):
            continue
        hooks_list = [n for n in (_normalize_hook(h) for h in raw_hooks) if n]
        if not hooks_list:
            continue
        matcher = group.get("matcher", "*")
        groups.append({"matcher": matcher if isinstance(matcher, str) else "*", "hooks": hooks_list})
    return groups


def normalize_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(spec)
    for key in EVENT_KEYS:
        if key in out:
            out[key] = _normalize_event(out[key])
    return out


class ExternalHooksManager:
    """Discovers, parses, and wires external hooks.json scripts into Antigravity."""

    _cached_configs: Optional[List[Dict[str, Any]]] = None
    _cached_sources: List[str] = []
    _cached_mtime: float = 0.0
    _in_process_hooks: List[Any] = []

    @classmethod
    def register_in_process_hook(cls, hook_obj: Any):
        """Registers an in-process native SDK hook (Inspect, Decide, or Transform)."""
        cls._in_process_hooks.append(hook_obj)

    @classmethod
    def clear_in_process_hooks(cls):
        """Clears registered in-process hooks."""
        cls._in_process_hooks.clear()

    @classmethod
    def discover_hook_files(cls, repo_root: str = ".") -> List[str]:
        """Finds all existing hooks.json files across workspace and global configurations."""
        found = []
        for p in HOOKS_CANDIDATE_PATHS:
            full_path = p if os.path.isabs(p) else os.path.join(repo_root, p)
            if os.path.isfile(full_path):
                real = os.path.realpath(full_path)
                if real not in [os.path.realpath(x) for x in found]:
                    found.append(full_path)
        return found

    @classmethod
    def load_hook_configs(cls, repo_root: str = ".", force_refresh: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Loads and parses discovered hooks.json files.
        Returns a tuple of (merged_hook_entries, source_files).
        """
        files = cls.discover_hook_files(repo_root)
        if not files:
            cls._cached_configs = []
            cls._cached_sources = []
            return [], []

        try:
            latest_mtime = max(os.path.getmtime(f) for f in files)
        except Exception:
            latest_mtime = 0.0

        if (
            not force_refresh
            and cls._cached_configs is not None
            and latest_mtime <= cls._cached_mtime
            and cls._cached_sources == files
        ):
            return cls._cached_configs, cls._cached_sources

        parsed_hooks = []
        for fp in files:
            base_dir = os.path.dirname(os.path.abspath(fp))
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    for hook_name, hook_spec in raw.items():
                        if not isinstance(hook_spec, dict):
                            continue
                        if not hook_spec.get("enabled", True):
                            continue
                        parsed_hooks.append({
                            "name": hook_name,
                            "source_file": fp,
                            "base_dir": base_dir,
                            "spec": normalize_spec(hook_spec),
                        })
            except Exception as e:
                print(f"{UI.RED_BOLD}[✗ Hook Error]{UI.RST} Failed to parse {fp}: {e}")

        cls._cached_configs = parsed_hooks
        cls._cached_sources = files
        cls._cached_mtime = latest_mtime
        return parsed_hooks, files

    @classmethod
    def execute_command_hook(
        cls,
        cmd: str,
        base_dir: str,
        payload: Dict[str, Any],
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """
        Executes a shell command hook with context on stdin, expecting JSON on stdout.
        """
        try:
            expanded_cmd = os.path.expanduser(cmd)
            proc = subprocess.Popen(
                expanded_cmd,
                shell=True,
                cwd=base_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                out, _ = proc.communicate(json.dumps(payload), timeout=timeout)
            except subprocess.TimeoutExpired:
                # Kill the whole group so grandchildren of the shell don't linger
                if hasattr(os, "killpg"):
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    proc.kill()
                proc.communicate()
                return {"error": f"Hook timed out after {timeout}s"}
            stdout = (out or "").strip()
            if stdout:
                try:
                    parsed = json.loads(stdout)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
                return {"raw_output": stdout, "exit_code": proc.returncode}
            return {"exit_code": proc.returncode}
        except Exception as e:
            return {"error": str(e)}

    TURN_EVENT_ALIASES = {
        "PreTurn": ("PreTurn", "PreInvocation"),
        "PostTurn": ("PostTurn", "PostInvocation"),
        "Stop": ("Stop",),
    }

    @classmethod
    def run_turn_event(cls, session_obj: Any, event: str, payload: Dict[str, Any], repo_root: str = ".") -> Dict[str, Any]:
        """
        Runs PreTurn / PostTurn / Stop hooks and returns a verdict dict:
          PreTurn: {"deny": True, "reason"} or {"prompt": <possibly rewritten/prefixed prompt>}
          Stop:    {"block": True, "reason"} to make the agent keep working
          PostTurn: informational only
        """
        entries, _ = cls.load_hook_configs(repo_root)
        prompt = payload.get("prompt")
        extra_context = []

        for entry in entries:
            for spec_key in cls.TURN_EVENT_ALIASES.get(event, (event,)):
                for group in entry["spec"].get(spec_key, []):
                    for h in group["hooks"]:
                        cmd = h["command"]
                        body = dict(payload)
                        body.update({
                            "event": event,
                            "conversationId": getattr(session_obj, "conversation_id", "kontexta-cli-session"),
                            "modelName": session_obj.active_target,
                        })
                        result = cls.execute_command_hook(cmd, entry["base_dir"], body, timeout=h["timeout"])
                        decision = str(result.get("decision", "")).lower()
                        reason = str(result.get("reason", ""))

                        if event == "PreTurn":
                            if decision == "deny":
                                return {"deny": True, "reason": reason}
                            overwrite = result.get("overwrite")
                            if isinstance(overwrite, dict) and isinstance(overwrite.get("prompt"), str):
                                prompt = overwrite["prompt"]
                            if isinstance(result.get("context"), str) and result["context"].strip():
                                extra_context.append(result["context"].strip())
                        elif event == "Stop" and decision in ("block", "deny"):
                            return {"block": True, "reason": reason or "Completion condition not met."}

        if prompt is not None and extra_context:
            prompt = "\n".join(extra_context) + "\n\n" + prompt
        return {"prompt": prompt}

    @classmethod
    def build_lifecycle_hooks(
        cls,
        session_obj: Any,
        repo_root: str = ".",
    ) -> List[Any]:
        """
        Builds and returns Antigravity SDK Hook instances wired to external hooks.json commands.
        """
        hook_entries, source_files = cls.load_hook_configs(repo_root)
        if not hook_entries:
            return list(cls._in_process_hooks)

        active_hooks = []

        def collect(event: str) -> List[Dict[str, Any]]:
            handlers = []
            for entry in hook_entries:
                for group in entry["spec"].get(event, []):
                    for h in group["hooks"]:
                        handlers.append({
                            "name": entry["name"],
                            "matcher": group["matcher"],
                            "command": h["command"],
                            "base_dir": entry["base_dir"],
                            "timeout": h["timeout"],
                        })
            return handlers

        # ── PreToolUse Handlers ───────────────────────────────────────
        pre_tool_handlers = collect("PreToolUse")

        if pre_tool_handlers:
            @hooks.pre_tool_call_decide
            def external_pre_tool_decide(call: types.ToolCall) -> types.HookResult:
                c_name = getattr(call, "name", str(call))
                c_args = getattr(call, "args", {}) or {}

                for handler in pre_tool_handlers:
                    pattern = handler["matcher"]
                    matched = False
                    if pattern in ("*", "", ".*"):
                        matched = True
                    else:
                        try:
                            if re.search(pattern, c_name):
                                matched = True
                        except Exception:
                            if pattern == c_name:
                                matched = True

                    if not matched:
                        continue

                    payload = {
                        "toolCall": {"name": c_name, "args": c_args},
                        "conversationId": getattr(session_obj, "conversation_id", "kontexta-cli-session"),
                        "modelName": session_obj.active_target,
                    }

                    if not session_obj.json_output and session_obj.verbose:
                        print(f"{UI.DARK_GRAY}  │ [Executing Hook: {handler['name']} (PreToolUse)]{UI.RST}")

                    result = cls.execute_command_hook(
                        handler["command"],
                        handler["base_dir"],
                        payload,
                        timeout=handler["timeout"],
                    )

                    decision = str(result.get("decision", "")).lower()
                    reason = str(result.get("reason", ""))
                    overwrite = result.get("overwrite")

                    if decision == "deny":
                        msg = f"Execution blocked by hook '{handler['name']}': {reason or 'Permission denied'}"
                        if not session_obj.json_output:
                            print(f"{UI.RED_BOLD}[x Hook Blocked]{UI.RST} {UI.RED}{msg}{UI.RST}")
                        return types.HookResult(allow=False, message=msg)

                    if overwrite and isinstance(overwrite, dict):
                        # Shallow top-level argument merge according to Antigravity hook spec
                        c_args.update(overwrite)
                        if not session_obj.json_output and session_obj.verbose:
                            print(f"{UI.DARK_GRAY}  │ [Hook Overwrote Tool Args]: {overwrite}{UI.RST}")

                return types.HookResult(allow=True)

            active_hooks.append(external_pre_tool_decide)

        # ── PostToolUse Handlers ──────────────────────────────────────
        post_tool_handlers = collect("PostToolUse")

        if post_tool_handlers:
            @hooks.post_tool_call
            def external_post_tool(res: types.ToolResult):
                call = getattr(res, "call", None)
                last = getattr(session_obj, "_last_tool_call", None) or {}
                c_name = (getattr(call, "name", None) if call else None) or last.get("name", "unknown")
                c_args = (getattr(call, "args", None) if call else None) or last.get("args", {})
                output = getattr(res, "result", None)
                output_text = (output if isinstance(output, str) else json.dumps(output, default=str)) if output is not None else ""

                for handler in post_tool_handlers:
                    pattern = handler["matcher"]
                    matched = False
                    if pattern in ("*", "", ".*"):
                        matched = True
                    else:
                        try:
                            if re.search(pattern, c_name):
                                matched = True
                        except Exception:
                            if pattern == c_name:
                                matched = True

                    if not matched:
                        continue

                    payload = {
                        "toolCall": {"name": c_name, "args": c_args},
                        "output": output_text[:4000],
                        "error": str(res.error) if res.error else "",
                        "conversationId": getattr(session_obj, "conversation_id", "kontexta-cli-session"),
                        "modelName": session_obj.active_target,
                    }

                    result = cls.execute_command_hook(
                        handler["command"],
                        handler["base_dir"],
                        payload,
                        timeout=handler["timeout"],
                    )
                    if session_obj.verbose and not session_obj.json_output:
                        print(f"{UI.DARK_GRAY}  │ [Hook Completed: {handler['name']} (PostToolUse)]{UI.RST}")

            active_hooks.append(external_post_tool)

        if cls._in_process_hooks:
            active_hooks.extend(cls._in_process_hooks)

        return active_hooks
