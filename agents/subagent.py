"""
Subagent delegation engine: Asynchronously queues and executes on-demand background engineering tasks
strictly 1-at-a-time on the NVIDIA RTX 4060 accelerator to honor VRAM and compute constraints.
"""

import os
import time
import json
import re
import shutil
import signal
import asyncio
import urllib.request
from typing import Optional, List, Dict, Any

from google.antigravity import Agent, LocalOpenAIAgentConfig, CapabilitiesConfig, hooks, types
from google.antigravity.hooks import policy

from ..terminal.ui import UI
from ..terminal import say
from ..core.coordinator import CudaCoordinator
from ..core.llamashift import trigger_llamashift_switch
from ..core import shellparse


class SubagentTask:
    """Represents an asynchronous background engineering task delegated to the NVIDIA accelerator."""

    def __deepcopy__(self, memo):
        return self

    def __init__(self, task_id: int, description: str, command: Optional[str] = None):
        self.id = task_id
        self.description = description
        self.command = command
        self.status = "queued"  # queued | running | completed | failed | cancelled
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.duration: Optional[float] = None
        self.result: Optional[str] = None
        self.tool_calls: List[Dict[str, Any]] = []
        self.error: Optional[str] = None
        self.exit_code: Optional[int] = None
        self.injected: bool = False
        self.mode: str = "pending"  # tools | no-tools (fallback)
        self._async_task: Optional[asyncio.Task] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "duration": round(self.duration, 2) if self.duration is not None else None,
            "injected": self.injected,
            "result_preview": (self.result[:120] + "...") if self.result and len(self.result) > 120 else (self.result or ""),
        }


class SubagentManager:
    """Manages the FIFO execution queue, sequential execution, and context injection of background subagents."""

    _tasks: List[SubagentTask] = []
    _next_id: int = 1
    _queue: asyncio.Queue = None
    _worker_loop_task: Optional[asyncio.Task] = None
    _active_task: Optional[SubagentTask] = None
    _session: Any = None

    @classmethod
    def _ensure_queue_init(cls, session: Any):
        """Initializes the asyncio FIFO queue and background dispatcher if not already running."""
        cls._session = session
        if cls._queue is None:
            cls._queue = asyncio.Queue()
        if cls._worker_loop_task is None or cls._worker_loop_task.done():
            cls._worker_loop_task = asyncio.create_task(cls._queue_dispatcher())

    @classmethod
    def list_tasks(cls) -> List[SubagentTask]:
        return cls._tasks

    @classmethod
    def get(cls, task_id: int) -> Optional[SubagentTask]:
        for t in cls._tasks:
            if t.id == task_id:
                return t
        return None

    @classmethod
    def get_queue_status(cls) -> Dict[str, Any]:
        """Returns statistics on active, queued, and completed subagents."""
        queued_count = sum(1 for t in cls._tasks if t.status == "queued")
        running_count = sum(1 for t in cls._tasks if t.status == "running")
        completed_count = sum(1 for t in cls._tasks if t.status == "completed")
        failed_count = sum(1 for t in cls._tasks if t.status == "failed")
        return {
            "queued": queued_count,
            "running": running_count,
            "completed": completed_count,
            "failed": failed_count,
            "active_task": cls._active_task.to_dict() if cls._active_task else None,
        }

    @classmethod
    def spawn(cls, description: str, session: Any, command: Optional[str] = None) -> SubagentTask:
        """
        Enqueues a subagent task. If the NVIDIA GPU is currently idle, it begins executing immediately.
        If a task is already running, this task queues in FIFO order, guaranteeing strict 1-at-a-time execution.
        """
        cls._ensure_queue_init(session)

        task = SubagentTask(task_id=cls._next_id, description=description, command=command)
        cls._next_id += 1
        cls._tasks.append(task)

        # Enqueue for serialized execution
        cls._queue.put_nowait(task)

        queued_ahead = sum(1 for t in cls._tasks if t.status == "queued") - 1
        if not session.json_output:
            if queued_ahead > 0:
                say.step(f"subagent #{task.id} queued (position {queued_ahead + 1}): {description}")
                say.detail("  one task at a time on :9001; /tasks shows the queue")
            else:
                say.step(f"subagent #{task.id} launched: {description}")
                say.detail("  :9001 compactor paused while it runs; /tasks shows status")

        return task

    @classmethod
    def cancel(cls, task_id: int) -> bool:
        """Cancels a subagent task whether it is running or waiting in the FIFO queue."""
        task = cls.get(task_id)
        if not task:
            return False

        if task.status == "queued":
            task.status = "cancelled"
            task.finished_at = time.time()
            return True

        if task.status == "running" and task._async_task and not task._async_task.done():
            task._async_task.cancel()
            task.status = "cancelled"
            task.finished_at = time.time()
            if task.started_at:
                task.duration = task.finished_at - task.started_at
            return True

        return False

    @classmethod
    def inject(cls, task_id: int, session: Any) -> bool:
        """Injects completed subagent findings directly into the main conversation history."""
        task = cls.get(task_id)
        if not task or not task.result or task.status != "completed":
            return False

        summary_entry = {
            "role": "user",
            "target": session.active_target,
            "content": f"[SYSTEM: Subagent #{task.id} Autonomous Findings on \"{task.description}\"]\n{task.result}",
        }
        companion_entry = {
            "role": "assistant",
            "target": session.active_target,
            "content": f"[SYSTEM: Acknowledged Subagent #{task.id} findings. Ready to apply insights.]",
        }

        session.history.extend([summary_entry, companion_entry])
        task.injected = True
        session._active_local_agent = None  # refresh agent context
        return True

    @classmethod
    def inject_all(cls, session: Any) -> int:
        """Injects all uninjected completed subagent results into the session context in one shot."""
        completed_uninjected = [t for t in cls._tasks if t.status == "completed" and not t.injected]
        if not completed_uninjected:
            return 0

        entries = []
        for t in completed_uninjected:
            entries.append({
                "role": "user",
                "target": session.active_target,
                "content": f"[SYSTEM: Subagent #{t.id} Autonomous Findings on \"{t.description}\"]\n{t.result}",
            })
            t.injected = True

        entries.append({
            "role": "assistant",
            "target": session.active_target,
            "content": f"[SYSTEM: Acknowledged findings from {len(completed_uninjected)} background subagent(s). Ready to apply collective insights.]",
        })

        session.history.extend(entries)
        session._active_local_agent = None
        return len(completed_uninjected)

    @classmethod
    async def _queue_dispatcher(cls):
        """
        Background loop that pulls subagents from the FIFO queue and executes them strictly one-at-a-time.
        Prevents dual-subagent VRAM contention on NVIDIA RTX 4060.
        """
        while True:
            try:
                task: SubagentTask = await cls._queue.get()
                if task.status == "cancelled":
                    cls._queue.task_done()
                    continue

                cls._active_task = task
                # Synchronously acquire lock so compactor yields
                CudaCoordinator.reserve_for_subagent()
                if cls._session and not getattr(cls._session, "json_output", False):
                    say.detail(f"subagent #{task.id} executing on :9001")

                task._async_task = asyncio.create_task(cls._run_worker(task, cls._session))
                try:
                    await task._async_task
                except asyncio.CancelledError:
                    task.status = "cancelled"
                finally:
                    cls._active_task = None
                    cls._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                await asyncio.sleep(1.0)

    TOOL_RUN_TIMEOUT = 300
    SHELL_TIMEOUT = 600

    @staticmethod
    def _kill_process_group(proc: "asyncio.subprocess.Process"):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, AttributeError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    @classmethod
    async def _run_shell(cls, shell_cmd: str) -> "tuple[int, str]":
        """Runs a shell command in its own process group, killing it on timeout or cancellation."""
        proc = await asyncio.create_subprocess_shell(
            shell_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=cls.SHELL_TIMEOUT)
        except BaseException as exc:
            cls._kill_process_group(proc)
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except Exception:
                pass
            if isinstance(exc, asyncio.TimeoutError):
                raise TimeoutError(f"Command timed out after {cls.SHELL_TIMEOUT}s and was killed")
            raise
        output = (stdout.decode("utf-8", errors="replace") + stderr.decode("utf-8", errors="replace")).strip()
        return proc.returncode, output

    @classmethod
    async def _run_with_tools(cls, task: SubagentTask, session: Any, ep: Dict[str, Any]) -> str:
        """Runs the task as a real tool-using agent on :9001 (same SDK path as the main agent)."""

        @hooks.pre_tool_call_decide
        def log_tool(call: types.ToolCall) -> types.HookResult:
            task.tool_calls.append({"name": getattr(call, "name", str(call)), "args": getattr(call, "args", {}) or {}, "timestamp": time.time()})
            return types.HookResult(allow=True)

        def background_permission(tool: Any, args: Dict[str, Any]) -> bool:
            # Background agents can't prompt the user: only Laya-approved safe actions run
            name = getattr(tool, "name", str(tool))
            cmd_line = args.get("CommandLine", "") if isinstance(args, dict) else ""
            quiet = getattr(session, "json_output", False)
            if shellparse.elevates(cmd_line):
                task.tool_calls.append({"name": f"{name} (denied: sudo is not allowed in background tasks)", "args": args, "timestamp": time.time()})
                if not quiet:
                    say.warn(f"Subagent #{task.id}: denied {name} (sudo is not allowed in background tasks)")
                return False
            safe, _ = session.laya.evaluate_action_safety(f"Tool: {name} Args: {json.dumps(args, default=str)}")
            if not safe:
                task.tool_calls.append({"name": f"{name} (denied: not auto-approved as safe)", "args": args, "timestamp": time.time()})
                if not quiet:
                    say.warn(f"Subagent #{task.id}: denied {name} (not auto-approved as safe)")
            return safe

        instructions = (
            "You are an autonomous engineering sub-agent running in the background.\n"
            f"TASK: {task.description}\n\n"
            "GUIDELINES:\n"
            "1. Use your tools to inspect files, search, and run read-only checks as needed.\n"
            "2. Do not repeat identical tool calls. Prefer small, targeted reads.\n"
            "3. If a tool call is denied, continue without it and say what you could not verify.\n"
            "4. Finish with a dense, factual summary preserving file paths, function names and exact numbers."
        )
        config = LocalOpenAIAgentConfig(
            model=ep.get("model") or ep.get("filename"),
            base_url=ep.get("url", "http://localhost:9001/v1"),
            system_instructions=instructions,
            capabilities=CapabilitiesConfig(),
            policies=policy.confirm_run_command(handler=background_permission),
            hooks=[log_tool],
            mcp_servers=session._mcp_servers,
        )
        agent = Agent(config)
        await agent.__aenter__()
        try:
            response = await agent.chat(task.description)
            parts = []
            async for token in response:
                parts.append(token)
            return "".join(parts).strip()
        finally:
            try:
                await agent.__aexit__(None, None, None)
            except Exception:
                pass

    @classmethod
    async def _run_plain(cls, task: SubagentTask, ep: Dict[str, Any]) -> str:
        """Fallback: single chat completion with no tools."""
        instructions = (
            "You are a reasoning sub-agent running on the NVIDIA RTX 4060 accelerator.\n"
            "You have NO tools: you cannot read files, run commands, or browse. Work only from the task text.\n\n"
            f"TASK: {task.description}\n\n"
            "GUIDELINES:\n"
            "1. Answer from the information given; state clearly what you cannot verify without file or shell access.\n"
            "2. Never invent file contents, command output, or numbers.\n"
            "3. Conclude with a dense, structured summary and preserve any names and numbers from the task."
        )
        payload = {
            "model": ep.get("filename") or "granite-4.2-8b",
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": task.description},
            ],
            "max_tokens": 1500,
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            "http://127.0.0.1:9001/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        def _call_http():
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))

        data = await asyncio.to_thread(_call_http)
        choice = data["choices"][0]["message"]
        return (choice.get("content") or choice.get("reasoning_content") or "").strip()

    @classmethod
    async def _run_worker(cls, task: SubagentTask, session: Any):
        """Worker executing autonomously on NVIDIA RTX 4060 with tool access."""
        task.started_at = time.time()
        task.status = "running"

        try:
            # 1. Ensure Port 9001 is online (auto-wake via llama-launcher if needed)
            ep9001 = session.endpoints.get("9001", {})
            if ep9001.get("status") != "online":
                try:
                    start_res = trigger_llamashift_switch("granite42_8b")
                    if start_res.get("success"):
                        for _ in range(8):
                            await asyncio.sleep(0.5)
                            session.refresh_endpoints()
                            if session.endpoints.get("9001", {}).get("status") == "online":
                                ep9001 = session.endpoints["9001"]
                                break
                except Exception:
                    pass

            if ep9001.get("status") != "online":
                raise RuntimeError("NVIDIA RTX 4060 (Port 9001) is offline. Start it via llama-launcher or /compactor start.")

            res_text = None
            # Check if this task involves running a build or shell command
            shell_cmd = task.command
            if not shell_cmd:
                cmd_match = re.search(r"execute ['\"]([^'\"]+)['\"]", task.description, re.IGNORECASE)
                if cmd_match:
                    shell_cmd = cmd_match.group(1)

            if shell_cmd:
                task.tool_calls.append({
                    "name": "run_command",
                    "args": {"CommandLine": shell_cmd},
                    "timestamp": time.time(),
                })
                if shellparse.elevates(shell_cmd):
                    raise RuntimeError("sudo is not allowed in background subagent tasks")
                try:
                    return_code, cmd_output = await cls._run_shell(shell_cmd)
                except Exception as ex:
                    return_code = 124 if isinstance(ex, TimeoutError) else 1
                    cmd_output = f"Execution error: {ex}"
                task.exit_code = return_code

                subagent_instructions = (
                    "You are an expert autonomous sub-agent running on the NVIDIA RTX 4060 accelerator.\n"
                    "Your objective is to analyze the build and test execution results:\n\n"
                    f"TASK: {task.description}\n"
                    f"COMMAND: `{shell_cmd}`\n"
                    f"EXIT CODE: {return_code}\n"
                    f"OUTPUT SNIPPET:\n{cmd_output[-3000:]}\n\n"
                    "OPERATIONAL GUIDELINES:\n"
                    "1. Direct execution analysis: State clearly whether the build/test PASSED or FAILED.\n"
                    "2. If FAILED, provide exact root cause analysis, failing test names, compiler errors, and recommended code fixes.\n"
                    "3. Conclude with a dense, structured, factual summary."
                )
                user_msg = f"Analyze build/test execution for `{shell_cmd}` (exit code {return_code}). Output summary and diagnostics."

                payload = {
                    "model": ep9001.get("filename") or "granite-4.2-8b",
                    "messages": [
                        {"role": "system", "content": subagent_instructions},
                        {"role": "user", "content": user_msg},
                    ],
                    "max_tokens": 1500,
                    "temperature": 0.2,
                }
                req_data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:9001/v1/chat/completions",
                    data=req_data,
                    headers={"Content-Type": "application/json"},
                )

                def _call_http():
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        return json.loads(resp.read().decode("utf-8"))

                data = await asyncio.to_thread(_call_http)
                choice = data["choices"][0]["message"]
                res_text = (choice.get("content") or choice.get("reasoning_content") or "").strip()
                if not res_text and choice.get("reasoning_content"):
                    res_text = choice.get("reasoning_content").strip()
                task.mode = "tools"
            else:
                try:
                    res_text = await asyncio.wait_for(cls._run_with_tools(task, session, ep9001), timeout=cls.TOOL_RUN_TIMEOUT)
                    task.mode = "tools"
                except asyncio.CancelledError:
                    raise
                except Exception as tool_err:
                    task.tool_calls.append({"name": "(tool agent unavailable, using plain completion)", "args": {"error": str(tool_err)[:200]}, "timestamp": time.time()})

                if not res_text:
                    res_text = await cls._run_plain(task, ep9001)
                    task.mode = "no-tools"

            if res_text:
                task.result = res_text
            else:
                raise RuntimeError("Port 9001 returned empty response")

        except asyncio.CancelledError:
            task.status = "cancelled"
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
        else:
            task.status = "completed"
        finally:
            task.finished_at = time.time()
            task.duration = round((task.finished_at - task.started_at), 2) if task.started_at else 0.0
            # Release lock back to idle coordinator
            await CudaCoordinator.release_from_subagent()

            # Terminal notification
            if not session.json_output:
                queued_left = sum(1 for t in cls._tasks if t.status == "queued")
                queue_suffix = f" ({queued_left} next in queue)" if queued_left > 0 else ""
                elapsed = say.fmt_elapsed(task.duration or 0.0)
                term_cols = shutil.get_terminal_size((120, 24)).columns
                max_desc = max(80, term_cols - 40)
                desc_str = task.description if len(task.description) <= max_desc else (task.description[:max_desc - 3] + "...")
                if task.status == "completed":
                    say.ok(f"Subagent #{task.id} done in {elapsed}: {desc_str}{queue_suffix}")
                    say.dim(f"  /subagent view {task.id} to inspect, /subagent inject {task.id} to add to history")
                elif task.status == "failed":
                    say.err(f"Subagent #{task.id} failed after {elapsed}: {desc_str}: {task.error}{queue_suffix}")
                elif task.status == "cancelled":
                    say.step(f"subagent #{task.id} cancelled after {elapsed}: {desc_str}")

    @classmethod
    def print_tasks(cls, session: Any):
        """Renders an interactive status table of all background subagents with FIFO queue ordering."""
        if not cls._tasks:
            say.dim("(no background subagents spawned. Use /spawn <task> to delegate)")
            return

        coord_status = CudaCoordinator.status_summary()
        c_state = coord_status["state"].upper()
        c_label = {"IDLE": "idle", "SUBAGENT": "subagent running"}.get(c_state, "compacting")
        queued_tasks = sum(1 for t in cls._tasks if t.status == "queued")
        queue_tag = f"{queued_tasks} waiting" if queued_tasks else "empty"

        marks = {"completed": f"{UI.GREEN}done{UI.RST}", "running": f"{UI.CUDA_BOLD}running{UI.RST}",
                 "cancelled": f"{UI.GRAY}cancelled{UI.RST}", "failed": f"{UI.RED}failed{UI.RST}"}
        rows = []
        for t in cls._tasks:
            dur_str = f"{t.duration:.1f}s" if t.duration is not None else (f"{time.time() - t.started_at:.1f}s" if t.started_at else "-")
            preview = (t.result or t.error or "").replace("\n", " ").strip()
            if len(preview) > 100:
                preview = preview[:97] + "..."
            rows.append([str(t.id), marks.get(t.status, f"{UI.AMBER}queued{UI.RST}"), dur_str, t.description, preview, "yes" if t.injected else "no"])

        say.table(f"Background subagents on :9001 ({c_label}, queue {queue_tag})", ["#", "status", "time", "task", "result", "injected"], rows)
        say.dim("/subagent view <#> | inject <#|all> | cancel <#>")

    @classmethod
    def print_detail(cls, task_id: int):
        task = cls.get(task_id)
        if not task:
            say.warn(f"Subagent #{task_id} not found. Type /tasks to view active tasks.")
            return

        dur_str = f"{task.duration:.1f}s" if task.duration is not None else (f"{time.time() - task.started_at:.1f}s" if task.started_at else "-")
        mode_tag = {"tools": "with tools", "no-tools": "plain completion, no tools"}.get(task.mode, task.mode)
        rows = [
            ("task", task.description),
            ("status", f"{task.status} ({dur_str})"),
            ("engine", f"port 9001, {mode_tag}"),
            ("injected", "yes" if task.injected else "no"),
        ]
        if task.tool_calls:
            calls = [f"{c['name']}({json.dumps(c.get('args', {}))[:80]})" for c in task.tool_calls]
            rows.append(("tool calls", "\n".join(calls)))
        say.card(f"Subagent #{task.id}", rows, style="info")
        res_text = task.result or task.error or ("(waiting in queue...)" if task.status == "queued" else "(running...)")
        say.plain(res_text)
        if task.status == "completed" and not task.injected:
            say.dim(f"/subagent inject {task.id} adds this result to the conversation context.")
