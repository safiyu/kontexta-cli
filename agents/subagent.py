from __future__ import annotations

import asyncio
import itertools
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, List, Dict

import signal
import urllib.request
import httpx

try:
    from google.antigravity import (
        Agent,
        LocalOpenAIAgentConfig,
        hooks,
        types,
        CapabilitiesConfig,
    )
    try:
        from google.antigravity.hooks import policy
    except ImportError:
        policy = getattr(hooks, "policy", None)
except ImportError:
    Agent = None
    LocalOpenAIAgentConfig = None
    hooks = None
    types = None
    policy = None
    CapabilitiesConfig = None

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from ..config import CUDA_BASE_URL, COMPACTOR_MODEL_ID, COMPACTOR_MODEL_FILE
from ..core.coordinator import CudaCoordinator
from ..core.laya import LayaDecisionEngine
from ..core import shellparse
from ..core.llamashift import wake_compactor
from ..terminal import say
from ..terminal.ui import UI

console = Console()

SubagentState = Literal["queued", "running", "completed", "failed", "cancelled"]


class ModeStr(str):
    """String sub-type allowing both 'agent+tools'/'tools' and 'fallback-no-tools'/'no-tools' equality."""
    def __eq__(self, other: Any) -> bool:
        s = str(self)
        o = str(other)
        s_tool = (s == "agent+tools" or s == "tools")
        o_tool = (o == "agent+tools" or o == "tools")
        if s_tool and o_tool:
            return True
        s_notool = (s == "fallback-no-tools" or s == "no-tools")
        o_notool = (o == "fallback-no-tools" or o == "no-tools")
        if s_notool and o_notool:
            return True
        return super().__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()


@dataclass(slots=True)
class SubagentToolTrace:
    tool_name: str
    args: dict[str, Any]
    status: Literal["executed", "denied", "error"]
    detail: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class SubagentTask:
    task_id: int
    prompt: str = ""
    kind: Literal["spawn", "build"] = "spawn"
    state: SubagentState = "queued"
    mode: str = field(default_factory=lambda: ModeStr("agent+tools"))

    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    output_chunks: list[str] = field(default_factory=list)
    tool_traces: list[SubagentToolTrace] = field(default_factory=list)
    _legacy_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    injected: bool = False

    command: Optional[str] = None
    exit_code: Optional[int] = None
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _runner_task: asyncio.Task[None] | None = field(default=None, repr=False)

    def __init__(
        self,
        task_id: int,
        prompt: str = "",
        command: Optional[str] = None,
        kind: Literal["spawn", "build"] = "spawn",
        description: Optional[str] = None,
        **kwargs,
    ):
        self.task_id = task_id
        self.prompt = prompt or description or command or ""
        self.command = command or kwargs.get("command", None)
        self.exit_code = kwargs.get("exit_code", None)
        self.kind = kind
        self.state = kwargs.get("state", kwargs.get("status", "queued"))
        raw_mode = kwargs.get("mode", "agent+tools")
        self.mode = ModeStr(raw_mode) if isinstance(raw_mode, str) else raw_mode
        self.created_at = kwargs.get("created_at", time.time())
        self.started_at = kwargs.get("started_at", None)
        self.finished_at = kwargs.get("finished_at", None)
        self.output_chunks = kwargs.get("output_chunks", [])
        self.tool_traces = kwargs.get("tool_traces", [])
        self._legacy_tool_calls = kwargs.get("tool_calls", [])
        self.error = kwargs.get("error", None)
        self.injected = kwargs.get("injected", False)
        self._cancel_event = asyncio.Event()
        self._runner_task = None

    def __deepcopy__(self, memo):
        return self

    @property
    def result_text(self) -> str:
        return "".join(self.output_chunks).strip()

    @property
    def elapsed_s(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or time.time()
        return max(0.0, end - self.started_at)

    @property
    def progress_preview(self) -> str:
        text = self.result_text.replace("\n", " ")
        if len(text) > 72:
            return "..." + text[-69:]
        return text or (self.tool_traces[-1].tool_name if self.tool_traces else "waiting...")

    # Backward compatibility properties
    @property
    def id(self) -> int:
        return self.task_id

    @id.setter
    def id(self, val: int):
        self.task_id = val

    @property
    def description(self) -> str:
        return self.prompt

    @description.setter
    def description(self, val: str):
        self.prompt = val

    @property
    def status(self) -> str:
        return self.state

    @status.setter
    def status(self, val: str):
        self.state = val

    @property
    def duration(self) -> float:
        return self.elapsed_s

    @duration.setter
    def duration(self, val: float):
        if self.started_at is not None:
            self.finished_at = self.started_at + val
        else:
            self.started_at = 0.0
            self.finished_at = val

    @property
    def result(self) -> str:
        return self.result_text

    @result.setter
    def result(self, val: str):
        self.output_chunks = [str(val)] if val else []

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        if not self._legacy_tool_calls and self.tool_traces:
            for tr in self.tool_traces:
                self._legacy_tool_calls.append({
                    "name": tr.tool_name,
                    "args": tr.args,
                    "status": tr.status,
                    "detail": tr.detail,
                    "timestamp": tr.timestamp,
                })
        return self._legacy_tool_calls

    @tool_calls.setter
    def tool_calls(self, val: list[dict[str, Any]]):
        self._legacy_tool_calls = list(val)
        self.tool_traces = [
            SubagentToolTrace(
                tool_name=c.get("name", ""),
                args=c.get("args", {}),
                status=c.get("status", "executed"),
                detail=c.get("detail", ""),
                timestamp=c.get("timestamp", time.time()),
            )
            for c in val
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.task_id,
            "description": self.prompt,
            "status": self.state,
            "duration": round(self.elapsed_s, 2),
            "injected": self.injected,
            "result_preview": (self.result_text[:120] + "...") if len(self.result_text) > 120 else self.result_text,
        }


class SubagentManager:
    """
    FIFO Background Task & Subagent Engine on NVIDIA RTX 4060 (CUDA:9002).
    Supports /spawn, /build, /tasks, and /subagent view|inject|cancel.
    """
    _instance: Optional["SubagentManager"] = None
    _tasks: list[SubagentTask] = []

    def __init__(
        self,
        cuda_coord: CudaCoordinator = None,
        laya: LayaDecisionEngine = None,
        hooks_provider: Optional[Callable[[], list]] = None,
        mcp_tools: list[Any] = None,
        system_prompt_provider: Any = None,
        model_id: str = COMPACTOR_MODEL_ID,
        safety_threshold: float = 0.90,
    ) -> None:
        self.coord = cuda_coord or CudaCoordinator()
        self.laya = laya
        self.hooks_provider = hooks_provider
        self.mcp_tools = mcp_tools or []
        self.get_system_prompt = system_prompt_provider or (lambda: "")
        self.model_id = model_id
        self.safety_threshold = safety_threshold

        self._id_gen = itertools.count(1)
        self.tasks: dict[int, SubagentTask] = {}
        self._fifo_queue: asyncio.Queue[SubagentTask] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._http = httpx.AsyncClient(base_url=CUDA_BASE_URL, timeout=60.0)
        SubagentManager._instance = self

    @classmethod
    def get_instance(cls) -> "SubagentManager":
        if cls._instance is None:
            cls._instance = SubagentManager()
        return cls._instance

    def _ensure_worker_running(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = loop.create_task(
                self._fifo_worker_loop(), name="cuda-subagent-fifo-worker"
            )

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except (asyncio.CancelledError, Exception):
                pass
        for t in self.tasks.values():
            if t.state == "running":
                t.state = "cancelled"
                t._cancel_event.set()
                if t._runner_task and not t._runner_task.done():
                    t._runner_task.cancel()
        await self._http.aclose()

    # -------------------------------------------------------------------------
    # Enqueueing (/spawn <task> and /build [cmd])
    # -------------------------------------------------------------------------

    @classmethod
    def _spawn_from_class(
        cls,
        prompt: str,
        session: Any = None,
        command: Optional[str] = None,
        kind: Literal["spawn", "build"] = "spawn",
        **kwargs,
    ) -> SubagentTask:
        inst = cls.get_instance()
        actual_kind = "build" if (command or kind == "build") else "spawn"
        task = inst.spawn(prompt, kind=actual_kind, command=command)
        return task

    def spawn(
        self,
        prompt: str,
        *args,
        kind: Literal["spawn", "build"] = "spawn",
        command: Optional[str] = None,
        **kwargs,
    ) -> SubagentTask:
        """Enqueues a new background subagent task on CUDA:9002 (FIFO serialized)."""
        if not isinstance(self, SubagentManager):
            return SubagentManager._spawn_from_class(self, *args, command=command, kind=kind, **kwargs)

        self._ensure_worker_running()
        tid = next(self._id_gen)
        actual_kind = "build" if (command or kind == "build") else "spawn"
        task = SubagentTask(task_id=tid, prompt=prompt, kind=actual_kind, command=command)
        if command:
            task.command = command
        self.tasks[tid] = task
        SubagentManager._tasks.append(task)
        self._fifo_queue.put_nowait(task)
        return task

    def spawn_build(self, build_cmd: str = "python3 -m pytest -q") -> SubagentTask:
        """Implements `/build [cmd]` — delegates build & test verification to :9002."""
        prompt = (
            f"Run the verification command `{build_cmd}` and analyze the output. "
            "If all tests pass, summarize the test count and runtime in 2 lines. "
            "If any test or build step fails, inspect the relevant source files (read-only) "
            "and provide the exact file path, line number, root cause, and minimal code patch."
        )
        return self.spawn(prompt=prompt, kind="build", command=build_cmd)

    # -------------------------------------------------------------------------
    # FIFO Worker Loop & Tool-Gated Execution on :9002
    # -------------------------------------------------------------------------

    async def _fifo_worker_loop(self) -> None:
        while True:
            task = await self._fifo_queue.get()
            if task.state == "cancelled" or task._cancel_event.is_set():
                self._fifo_queue.task_done()
                continue

            # Acquire Priority 10 slot on CUDA:9002 (immediately preempts compactor.py)
            try:
                async with self.coord.acquire_subagent_slot(f"#{task.task_id}:{task.kind}") as abort_ev:
                    task.state = "running"
                    task.started_at = time.time()
                    task._runner_task = asyncio.current_task()

                    await self._execute_subagent_task(task, abort_ev)

                    if task._cancel_event.is_set() or abort_ev.is_set():
                        task.state = "cancelled"
                    elif task.state != "failed":
                        task.state = "completed"

            except asyncio.CancelledError:
                task.state = "cancelled"
                if not task._cancel_event.is_set():
                    # Worker itself is being shut down (stop/reload), not a per-task cancel.
                    raise
                current = asyncio.current_task()
                if current is not None and hasattr(current, "uncancel"):
                    current.uncancel()
            except Exception as exc:
                task.state = "failed"
                task.error = str(exc)
            finally:
                task.finished_at = time.time()
                task._runner_task = None
                self._fifo_queue.task_done()

    async def _execute_subagent_task(
        self, task: SubagentTask, coord_abort: asyncio.Event
    ) -> None:
        """
        Attempts to run a real tool-using Antigravity Agent on :9002 with strict
        non-interactive safety gating; falls back to plain completion on init failure.
        """
        agent = None
        try:
            agent = self._build_headless_subagent(task)
            await agent.__aenter__()
            response = await agent.chat(task.prompt)

            closer = getattr(response, "aclose", None)
            if closer:
                self.coord.bind_stream_closer(closer)

            async for token in response:
                if task._cancel_event.is_set() or coord_abort.is_set():
                    if closer:
                        await closer()
                    return
                task.output_chunks.append(token)

        except Exception as agent_exc:
            # If the agent failed before producing tokens, fall back to plain no-tools completion
            if not task.output_chunks and not task._cancel_event.is_set():
                task.mode = ModeStr("fallback-no-tools")
                task.tool_traces.append(
                    SubagentToolTrace(
                        tool_name="agent_init",
                        args={},
                        status="error",
                        detail=f"Agent fallback triggered ({agent_exc}); running plain completion",
                    )
                )
                await self._run_plain_completion_fallback(task, coord_abort)
            else:
                task.state = "failed"
                task.error = str(agent_exc)
        finally:
            if agent is not None and hasattr(agent, "__aexit__"):
                try:
                    await agent.__aexit__(None, None, None)
                except Exception:
                    pass

    def _gate_tool_call(self, task: SubagentTask, name: str, args: Dict[str, Any]) -> tuple[bool, str]:
        """Non-interactive safety gate: Laya P(Safe) must reach the threshold, never prompts."""
        args = args if isinstance(args, dict) else {}
        cmd_line = str(args.get("CommandLine", "") or "")
        if "rm -rf" in cmd_line or (hasattr(shellparse, "elevates") and shellparse.elevates(cmd_line)):
            reason = "sudo/destructive command is not allowed in background tasks"
            task.tool_traces.append(SubagentToolTrace(name, dict(args), "denied", reason))
            return False, reason

        p_safe = 1.0
        if self.laya is not None and hasattr(self.laya, "evaluate_action_safety"):
            is_safe, p_safe = self.laya.evaluate_action_safety(
                f"Tool: {name} Args: {json.dumps(args, default=str)}"
            )
            if not is_safe:
                p_safe = min(p_safe, self.safety_threshold - 1e-6)

        if p_safe < self.safety_threshold:
            reason = (
                f"Background subagent denied unsafe action `{name}` "
                f"(P(Safe)={p_safe:.2f} < {self.safety_threshold:.2f}). "
                "Use read-only inspection tools instead."
            )
            task.tool_traces.append(SubagentToolTrace(name, dict(args), "denied", reason))
            return False, reason

        task.tool_traces.append(SubagentToolTrace(name, dict(args), "executed", f"P(Safe)={p_safe:.2f}"))
        return True, ""

    def _build_headless_subagent(self, task: SubagentTask) -> Any:
        """
        Builds an unentered :9002 Agent (same SDK path as the main agent) with hooks.json
        hooks and a non-interactive Laya gate. Callers must enter/exit its context.
        """
        if Agent is None or LocalOpenAIAgentConfig is None:
            raise RuntimeError("google.antigravity Agent unavailable")

        sys_prompt = (
            f"{self.get_system_prompt()}\n\n"
            "# Background Subagent Mode (CUDA:9002)\n"
            "You are running asynchronously in the background. You may inspect files, search code, "
            "and run safe read-only test/diagnostic commands. Destructive or mutating actions "
            "will be automatically denied by the security gate."
        )

        def background_permission(tool: Any, args: Dict[str, Any]) -> bool:
            allowed, _ = self._gate_tool_call(task, getattr(tool, "name", str(tool)), args)
            return allowed

        @hooks.pre_tool_call_decide
        def subagent_gate(call: types.ToolCall) -> types.HookResult:
            name = str(getattr(getattr(call, "name", ""), "value", getattr(call, "name", "")))
            allowed, reason = self._gate_tool_call(task, name, getattr(call, "args", {}) or {})
            if allowed:
                return types.HookResult(allow=True)
            return types.HookResult(allow=False, message=reason)

        agent_hooks = [subagent_gate]
        if self.hooks_provider is not None:
            agent_hooks.extend(self.hooks_provider() or [])

        config = LocalOpenAIAgentConfig(
            model=self.model_id,
            base_url=CUDA_BASE_URL,
            system_instructions=sys_prompt,
            capabilities=CapabilitiesConfig(),
            policies=policy.confirm_run_command(handler=background_permission),
            hooks=agent_hooks,
            mcp_servers=self.mcp_tools,
        )
        return Agent(config)

    async def _run_plain_completion_fallback(
        self, task: SubagentTask, coord_abort: asyncio.Event
    ) -> None:
        """Plain no-tools completion fallback on :9002 if the tool Agent cannot start."""
        try:
            async with self._http.stream(
                "POST",
                "/chat/completions",
                json={
                    "model": self.model_id,
                    "messages": [
                        {"role": "system", "content": self.get_system_prompt()},
                        {"role": "user", "content": task.prompt},
                    ],
                    "stream": True,
                    "temperature": 0.2,
                },
            ) as resp:
                closer = getattr(resp, "aclose", None)
                if closer:
                    self.coord.bind_stream_closer(closer)
                async for line in resp.aiter_lines():
                    if task._cancel_event.is_set() or coord_abort.is_set():
                        if closer:
                            await closer()
                        return
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            task.output_chunks.append(delta)
                    except Exception:
                        pass
        except Exception as e:
            if not task.output_chunks:
                task.state = "failed"
                task.error = f"Fallback error: {e}"

    # -------------------------------------------------------------------------
    # Slash Command Actions: /subagent cancel, /subagent inject, /tasks, /subagent view
    # -------------------------------------------------------------------------

    def cancel_task(self, task_id: int) -> tuple[bool, str]:
        """Implements `/subagent cancel <#>`."""
        task = self.tasks.get(task_id)
        if task is None:
            return False, f"Subagent task #{task_id} not found."
        if task.state in ("completed", "failed", "cancelled"):
            return False, f"Subagent task #{task_id} is already {task.state}."

        task._cancel_event.set()
        task.state = "cancelled"
        if task._runner_task and not task._runner_task.done():
            task._runner_task.cancel()
        return True, f"Cancelled subagent task #{task_id}."

    def inject_into_session(
        self, session: Any, target_spec: str
    ) -> tuple[int, list[int]]:
        """
        Implements `/subagent inject <#|all>`.
        Appends completed subagent findings into the active chat context.
        Returns: (count_injected, list_of_task_ids)
        """
        if target_spec.lower() in ("all", "*"):
            candidates = [
                t for t in self.tasks.values()
                if t.state == "completed" and not t.injected and t.result_text
            ]
        elif target_spec.isdigit():
            t = self.tasks.get(int(target_spec))
            candidates = [t] if (t and t.result_text and not t.injected) else []
        else:
            return 0, []

        if not candidates:
            return 0, []

        injected_ids: list[int] = []

        for task in candidates:
            block = (
                f"[Background Subagent #{task.task_id} ({task.kind.upper()} on CUDA:9002) Findings]\n"
                f"Task: {task.prompt}\n\n"
                f"{task.result_text}"
            )
            target = getattr(session, "active_target", "9002")
            user_entry = {"role": "user", "target": target, "content": block}
            ack_entry = {
                "role": "assistant",
                "target": target,
                "content": f"[SYSTEM: Acknowledged Subagent #{task.task_id} findings. Ready to apply insights.]",
            }
            if isinstance(getattr(session, "history", None), list):
                session.history.extend([user_entry, ack_entry])
                if hasattr(session, "_invalidate_local_agent"):
                    session._invalidate_local_agent()
            elif hasattr(session, "agent") and hasattr(session.agent, "get_history") and hasattr(session.agent, "set_history"):
                history = session.agent.get_history() or []
                history.extend([user_entry, ack_entry])
                session.agent.set_history(history)

            task.injected = True
            injected_ids.append(task.task_id)

        return len(injected_ids), injected_ids

    # -------------------------------------------------------------------------
    # Rich UI Renderers for `/tasks` and `/subagent view <#>`
    # -------------------------------------------------------------------------

    def render_tasks_table(self) -> None:
        """Implements `/tasks` — lists subagent queue, active task, and progress."""
        if not self.tasks:
            console.print("[dim]No subagent tasks queued or completed in this session.[/dim]")
            return

        table = Table(
            title="[bold cyan]🛰️ NVIDIA CUDA:9002 Subagent Execution Queue (/tasks)[/bold cyan]",
            box=box.SIMPLE_HEAVY,
            expand=True,
        )
        table.add_column("#", style="bold white", width=4)
        table.add_column("Kind", width=7)
        table.add_column("State", width=12)
        table.add_column("Mode", style="dim", width=16)
        table.add_column("Tools (Ok/Deny)", justify="right", width=15)
        table.add_column("Time", justify="right", width=8)
        table.add_column("Task / Live Progress", ratio=1)

        state_styles = {
            "queued": "[yellow]⏳ queued[/yellow]",
            "running": "[bold cyan]⚙ running[/bold cyan]",
            "completed": "[bold green]✓ done[/bold green]",
            "failed": "[bold red]✗ failed[/bold red]",
            "cancelled": "[dim]⊘ cancelled[/dim]",
        }

        for t in self.tasks.values():
            ok_tools = sum(1 for tr in t.tool_traces if tr.status == "executed")
            denied_tools = sum(1 for tr in t.tool_traces if tr.status == "denied")
            injected_tag = " [magenta](injected)[/magenta]" if t.injected else ""
            summary = f"[bold]{t.prompt[:48]}[/bold]{injected_tag}\n[dim]{t.progress_preview}[/dim]"

            table.add_row(
                str(t.task_id),
                t.kind,
                state_styles.get(t.state, t.state),
                str(t.mode),
                f"{ok_tools} / {denied_tools}",
                f"{t.elapsed_s:.1f}s",
                summary,
            )

        console.print(table)

    def render_task_detail(self, task_id: int) -> None:
        """Implements `/subagent view <#>` — shows full tool trace & Markdown output."""
        task = self.tasks.get(task_id)
        if task is None:
            console.print(f"[yellow]Subagent task #{task_id} not found.[/yellow]")
            return

        trace_lines: list[str] = []
        for tr in task.tool_traces:
            badge = "✓" if tr.status == "executed" else ("🛡️ DENIED" if tr.status == "denied" else "✗")
            args_preview = json.dumps(tr.args)[:80]
            trace_lines.append(f"- `{badge}` **{tr.tool_name}** `{args_preview}` — _{tr.detail}_")

        trace_md = "\n".join(trace_lines) if trace_lines else "_No tool calls recorded._"
        body_md = (
            f"**Task Prompt:** {task.prompt}\n\n"
            f"### Tool Call Audit Trail ({task.mode})\n{trace_md}\n\n"
            f"### Subagent Result\n{task.result_text or '_No output yet._'}"
        )
        if task.error:
            body_md += f"\n\n### Error\n```\n{task.error}\n```"

        console.print(
            Panel(
                Markdown(body_md),
                title=(
                    f"[bold cyan]🛰️ Subagent #{task.task_id} ({task.state.upper()})[/bold cyan] "
                    f"[dim]── {task.elapsed_s:.1f}s on CUDA:9002[/dim]"
                ),
                border_style="green" if task.state == "completed" else "cyan",
            )
        )

    # -------------------------------------------------------------------------
    # /reload Serialization Helpers
    # -------------------------------------------------------------------------

    def export_pending(self) -> list[dict[str, Any]]:
        """Exports queued/unstarted tasks so /reload can re-enqueue them after execv."""
        return [
            {"prompt": t.prompt, "kind": t.kind}
            for t in self.tasks.values()
            if t.state in ("queued", "running")
        ]

    def restore_pending(self, pending: list[dict[str, Any]]) -> None:
        for item in pending:
            self.spawn(prompt=item["prompt"], kind=item.get("kind", "spawn"))

    # -------------------------------------------------------------------------
    # Backward compatibility classmethods
    # -------------------------------------------------------------------------

    @classmethod
    def list_tasks(cls) -> list[SubagentTask]:
        inst = cls._instance
        return list(inst.tasks.values()) if inst else []

    @classmethod
    def get(cls, task_id: int) -> Optional[SubagentTask]:
        for t in cls._tasks:
            if t.id == task_id or t.task_id == task_id:
                return t
        inst = cls._instance
        return inst.tasks.get(task_id) if inst else None

    @classmethod
    def get_queue_status(cls) -> dict[str, Any]:
        inst = cls._instance
        tasks = list(inst.tasks.values()) if inst else []
        queued_count = sum(1 for t in tasks if t.state == "queued")
        running_count = sum(1 for t in tasks if t.state == "running")
        completed_count = sum(1 for t in tasks if t.state == "completed")
        failed_count = sum(1 for t in tasks if t.state == "failed")
        return {
            "queued": queued_count,
            "running": running_count,
            "completed": completed_count,
            "failed": failed_count,
            "total": len(tasks),
        }

    SHELL_TIMEOUT = 120
    TOOL_RUN_TIMEOUT = 300

    @classmethod
    def _kill_process_group(cls, proc: asyncio.subprocess.Process):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, AttributeError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    @classmethod
    async def _run_shell(cls, shell_cmd: str) -> tuple[int, str]:
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
        """Runs the task as a real tool-using agent on :9002 (same SDK path as the main agent)."""
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
        url = ep.get("url") or CUDA_BASE_URL
        pol_mod = getattr(hooks, "policy", None) or policy
        pol_obj = pol_mod.confirm_run_command(handler=background_permission) if (pol_mod and hasattr(pol_mod, "confirm_run_command")) else None
        config = LocalOpenAIAgentConfig(
            model=ep.get("model") or ep.get("filename"),
            base_url=url,
            system_instructions=instructions,
            capabilities=CapabilitiesConfig() if CapabilitiesConfig else None,
            policies=pol_obj,
            hooks=[log_tool] if hooks else [],
            mcp_servers=getattr(session, "_mcp_servers", []),
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
            "model": ep.get("filename") or COMPACTOR_MODEL_FILE,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": task.description},
            ],
            "max_tokens": 1500,
            "temperature": 0.2,
        }
        url = (ep.get("url") or CUDA_BASE_URL).rstrip("/") + "/chat/completions"
        import urllib.request
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        def _call_http():
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))

        data = await asyncio.to_thread(_call_http)
        choice = data["choices"][0]["message"]
        res_text = (choice.get("content") or choice.get("reasoning_content") or "").strip()
        return res_text

    @classmethod
    async def _run_worker(cls, task: SubagentTask, session: Any):
        """Worker executing autonomously on NVIDIA RTX 4060 with tool access."""
        task.started_at = time.time()
        task.status = "running"
        try:
            ep9002 = getattr(session, "endpoints", {}).get("9002") or getattr(session, "endpoints", {}).get("9001", {})
            if ep9002.get("status") != "online":
                woke = await asyncio.to_thread(wake_compactor, 9002)
                if not woke.get("ok"):
                    raise RuntimeError(f"NVIDIA RTX 4060 (Port 9002) is offline ({woke.get('reason')}).")

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
                    "model": ep9002.get("filename") or COMPACTOR_MODEL_FILE,
                    "messages": [
                        {"role": "system", "content": subagent_instructions},
                        {"role": "user", "content": user_msg},
                    ],
                    "max_tokens": 1500,
                    "temperature": 0.2,
                }
                req_data = json.dumps(payload).encode("utf-8")
                url = (ep9002.get("url") or CUDA_BASE_URL).rstrip("/") + "/chat/completions"
                req = urllib.request.Request(
                    url,
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
                task.mode = ModeStr("tools")
            else:
                try:
                    res_text = await asyncio.wait_for(cls._run_with_tools(task, session, ep9002), timeout=cls.TOOL_RUN_TIMEOUT)
                    task.mode = ModeStr("tools")
                except asyncio.CancelledError:
                    raise
                except Exception as tool_err:
                    task.tool_calls.append({
                        "name": "(tool agent unavailable, using plain completion)",
                        "args": {"error": str(tool_err)[:200]},
                        "timestamp": time.time(),
                    })

                if not res_text:
                    res_text = await cls._run_plain(task, ep9002)
                    task.mode = ModeStr("no-tools")

            if res_text:
                task.result = res_text
            else:
                raise RuntimeError("Port 9002 returned empty response")
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
            if hasattr(CudaCoordinator, "release_from_subagent"):
                await CudaCoordinator.release_from_subagent()

            # Terminal notification
            if not getattr(session, "json_output", False):
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
    def cancel(cls, task_id: int) -> bool:
        inst = cls._instance
        if inst:
            ok, _ = inst.cancel_task(task_id)
            return ok
        return False

    @classmethod
    def inject(cls, task_id: int, session: Any) -> bool:
        inst = cls._instance
        if inst:
            count, _ = inst.inject_into_session(session, str(task_id))
            return count > 0
        return False

    @classmethod
    def inject_all(cls, session: Any) -> int:
        inst = cls._instance
        if inst:
            count, _ = inst.inject_into_session(session, "all")
            return count
        return 0

    @classmethod
    def print_tasks(cls, session: Any = None):
        if cls._tasks:
            coord_status = CudaCoordinator.status_summary() if hasattr(CudaCoordinator, "status_summary") else {"state": "idle"}
            c_state = coord_status.get("state", "idle").upper()
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

            say.table(f"Background subagents on :9002 ({c_label}, queue {queue_tag})", ["#", "status", "time", "task", "result", "injected"], rows)
            say.dim("/subagent view <#> | inject <#|all> | cancel <#>")
            return

        inst = cls._instance
        if inst:
            inst.render_tasks_table()
        else:
            say.dim("(no background subagents spawned. Use /spawn <task> to delegate)")

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
            ("engine", f"port 9002, {mode_tag}"),
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
