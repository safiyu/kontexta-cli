import asyncio
import types
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from kontexta_cli.agents.subagent import (
    SubagentManager,
    SubagentTask,
    SubagentToolTrace,
    ModeStr,
)
from kontexta_cli.core.coordinator import CudaCoordinator
from kontexta_cli.terminal.cli import handle_slash_command


class MockLaya:
    def __init__(self, p_safe: float = 0.95, reason: str = "safe"):
        self.p_safe = p_safe
        self.reason = reason

    def evaluate_action_safety(self, action_desc: str):
        return self.p_safe >= 0.90, self.p_safe


@pytest.fixture
def cuda_coordinator():
    return CudaCoordinator()


@pytest.fixture
async def subagent_mgr(cuda_coordinator, monkeypatch):
    laya = MockLaya(p_safe=0.95)
    mgr = SubagentManager(
        cuda_coord=cuda_coordinator,
        laya=laya,
        mcp_tools=[],
        system_prompt_provider=lambda: "System instructions",
        model_id="granite42_8b",
        safety_threshold=0.90,
    )
    # Prevent real network calls during tests
    async def fast_exec(task, abort_ev):
        task.output_chunks.append("done")
    monkeypatch.setattr(mgr, "_execute_subagent_task", fast_exec)
    yield mgr
    await mgr.stop()


@pytest.mark.anyio
async def test_spawn_and_fifo_serialization(subagent_mgr):
    """Enqueued tasks run 1-at-a-time through acquire_subagent_slot."""
    task1 = subagent_mgr.spawn("Task 1")
    task2 = subagent_mgr.spawn("Task 2")
    assert task1.task_id == 1
    assert task2.task_id == 2
    assert task1.kind == "spawn"
    assert 1 in subagent_mgr.tasks and 2 in subagent_mgr.tasks

    # Allow worker to run
    await asyncio.sleep(0.1)
    assert task1.state in ("running", "completed")


@pytest.mark.anyio
async def test_spawn_build_command(subagent_mgr):
    """spawn_build creates a build kind task with structured verification prompt."""
    task = subagent_mgr.spawn_build("pytest -q")
    assert task.kind == "build"
    assert "pytest -q" in task.prompt
    assert "root cause" in task.prompt


@pytest.mark.anyio
async def test_non_interactive_safety_denies_low_score(subagent_mgr):
    """Tool calls with P(Safe) < 0.90 are denied in-place without terminal prompts."""
    subagent_mgr.laya = MockLaya(p_safe=0.75, reason="potential destructive write")
    task = SubagentTask(task_id=1, prompt="delete files")

    allowed, reason = subagent_mgr._gate_tool_call(task, "write_to_file", {"path": "/important"})
    assert allowed is False
    assert "denied unsafe action" in reason
    assert len(task.tool_traces) == 1
    assert task.tool_traces[0].status == "denied"
    assert "P(Safe)=0.75" in task.tool_traces[0].detail


@pytest.mark.anyio
async def test_safe_tool_call_allowed(subagent_mgr):
    task = SubagentTask(task_id=2, prompt="read files")
    allowed, _ = subagent_mgr._gate_tool_call(task, "view_file", {"path": "a.py"})
    assert allowed is True
    assert task.tool_traces[0].status == "executed"


@pytest.mark.anyio
async def test_destructive_command_denied(subagent_mgr):
    task = SubagentTask(task_id=3, prompt="clean")
    allowed, _ = subagent_mgr._gate_tool_call(task, "run_command", {"CommandLine": "rm -rf /"})
    assert allowed is False


@pytest.mark.anyio
async def test_graceful_plain_completion_fallback(subagent_mgr, monkeypatch):
    """Falls back to plain no-tools completion when agent init or execution fails."""
    task = SubagentTask(task_id=3, prompt="Explain quicksort")
    abort_ev = asyncio.Event()

    subagent_mgr._execute_subagent_task = SubagentManager._execute_subagent_task.__get__(subagent_mgr, SubagentManager)
    # Mock _build_headless_subagent to fail
    monkeypatch.setattr(subagent_mgr, "_build_headless_subagent", MagicMock(side_effect=RuntimeError("MCP failed")))

    # Mock _run_plain_completion_fallback
    async def mock_fallback(t, abort):
        t.output_chunks.append("Quicksort explanation")

    monkeypatch.setattr(subagent_mgr, "_run_plain_completion_fallback", mock_fallback)

    await subagent_mgr._execute_subagent_task(task, abort_ev)

    assert task.mode == "fallback-no-tools"
    assert "Quicksort explanation" in task.result_text
    assert any(tr.tool_name == "agent_init" and tr.status == "error" for tr in task.tool_traces)


@pytest.mark.anyio
async def test_selective_context_injection(subagent_mgr):
    """Selective context injection isolates findings until /subagent inject is invoked."""
    task = SubagentTask(task_id=1, prompt="Audit security", state="completed")
    task.output_chunks.append("Found zero vulnerabilities.")
    subagent_mgr.tasks[1] = task

    class MockSessionAgent:
        def __init__(self):
            self.history = []

        def get_history(self):
            return list(self.history)

        def set_history(self, hist):
            self.history = list(hist)

    class MockSession:
        def __init__(self):
            self.agent = MockSessionAgent()

    session = MockSession()

    # Inject specific task
    count, ids = subagent_mgr.inject_into_session(session, "1")
    assert count == 1
    assert ids == [1]
    assert task.injected is True
    assert len(session.agent.history) == 2
    assert session.agent.history[1]["role"] == "assistant"
    assert "[Background Subagent #1" in session.agent.history[0]["content"]
    assert "Found zero vulnerabilities." in session.agent.history[0]["content"]

    # Re-injecting returns 0 (already injected)
    count, ids = subagent_mgr.inject_into_session(session, "1")
    assert count == 0


@pytest.mark.anyio
async def test_cli_slash_commands(subagent_mgr):
    """Tests handle_slash_command wiring for /spawn, /build, /tasks, /subagent."""
    class MockSession:
        def __init__(self, mgr):
            self.subagent_mgr = mgr
            self.agent = types.SimpleNamespace(
                get_history=lambda: [],
                set_history=lambda h: None,
            )

    session = MockSession(subagent_mgr)

    # Test /spawn
    assert handle_slash_command("/spawn scan memory leaks", session) is True
    assert 1 in subagent_mgr.tasks
    assert subagent_mgr.tasks[1].prompt == "scan memory leaks"

    # Test /build
    assert handle_slash_command("/build cargo test", session) is True
    assert 2 in subagent_mgr.tasks
    assert subagent_mgr.tasks[2].kind == "build"

    # Test /tasks
    assert handle_slash_command("/tasks", session) is True

    # Test /subagent view
    assert handle_slash_command("/subagent view 1", session) is True

    # Test /subagent cancel
    assert handle_slash_command("/subagent cancel 1", session) is True
    assert subagent_mgr.tasks[1].state == "cancelled"

    # Test /subagent inject
    subagent_mgr.tasks[2].state = "completed"
    subagent_mgr.tasks[2].output_chunks = ["Build passed!"]
    assert handle_slash_command("/subagent inject 2", session) is True
    assert subagent_mgr.tasks[2].injected is True


@pytest.mark.anyio
async def test_stop_terminates_worker_while_task_is_running(subagent_mgr, monkeypatch):
    """Cancelling the worker (e.g. /reload) must not be swallowed by the per-task cancel handler."""
    started = asyncio.Event()

    async def long_exec(task, abort_ev):
        started.set()
        await asyncio.sleep(60)

    monkeypatch.setattr(subagent_mgr, "_execute_subagent_task", long_exec)
    subagent_mgr.spawn("slow")
    await asyncio.wait_for(started.wait(), 2)
    await asyncio.wait_for(subagent_mgr.stop(), 3)
    assert subagent_mgr._worker_task.done()


@pytest.mark.anyio
async def test_cancel_task_keeps_worker_alive_for_next_task(subagent_mgr, monkeypatch):
    started = asyncio.Event()
    calls = []

    async def exec_fn(task, abort_ev):
        calls.append(task.task_id)
        if task.task_id == 1:
            started.set()
            await asyncio.sleep(60)
        task.output_chunks.append("ok")

    monkeypatch.setattr(subagent_mgr, "_execute_subagent_task", exec_fn)
    t1 = subagent_mgr.spawn("first")
    t2 = subagent_mgr.spawn("second")
    await asyncio.wait_for(started.wait(), 2)
    ok, _ = subagent_mgr.cancel_task(t1.task_id)
    assert ok
    for _ in range(50):
        if t2.state == "completed":
            break
        await asyncio.sleep(0.05)
    assert t1.state == "cancelled"
    assert t2.state == "completed"
    assert calls == [1, 2]
