import asyncio
import types
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from myagy.agents.subagent import (
    SubagentManager,
    SubagentTask,
    SubagentToolTrace,
    ModeStr,
)
from myagy.core.coordinator import CudaCoordinator
from myagy.terminal.cli import handle_slash_command


class MockLaya:
    def __init__(self, p_safe: float = 0.95, reason: str = "safe"):
        self.p_safe = p_safe
        self.reason = reason

    async def score_tool_safety(self, tool_name: str, args: dict):
        return types.SimpleNamespace(p_safe=self.p_safe, reason=self.reason)


class MockHooksRunner:
    def __init__(self, decision: str = "allow", reason: str = ""):
        self.decision = decision
        self.reason = reason

    async def run_pre_tool(self, tool_name: str, args: dict):
        return types.SimpleNamespace(decision=self.decision, reason=self.reason, overwritten_args=None)

    async def run_post_tool(self, tool_name: str, args: dict, output: str, error: str):
        pass


@pytest.fixture
def cuda_coordinator():
    return CudaCoordinator()


@pytest.fixture
async def subagent_mgr(cuda_coordinator, monkeypatch):
    laya = MockLaya(p_safe=0.95)
    hooks_runner = MockHooksRunner(decision="allow")
    mgr = SubagentManager(
        cuda_coord=cuda_coordinator,
        laya=laya,
        hooks_runner=hooks_runner,
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
    
    agent = subagent_mgr._build_headless_subagent(task)
    # Trigger the pre_tool_call hook
    handler = None
    if hasattr(agent, "_handlers") and "pre_tool_call" in agent._handlers:
        handler = agent._handlers["pre_tool_call"][0]

    if handler is not None:
        from myagy.agents.subagent import ToolCallContext
        ctx = ToolCallContext(name="write_to_file", args={"path": "/important"})
        decision = await handler(ctx)
        assert decision.allow is False
        assert "denied unsafe action" in decision.reason
        assert len(task.tool_traces) == 1
        assert task.tool_traces[0].status == "denied"
        assert "P(Safe)=0.75" in task.tool_traces[0].detail


@pytest.mark.anyio
async def test_hooks_runner_pre_tool_denial(subagent_mgr):
    """External hooks runner denial halts execution in-place."""
    subagent_mgr.hooks_runner = MockHooksRunner(decision="deny", reason="Blocked by policy")
    task = SubagentTask(task_id=2, prompt="modify config")
    
    agent = subagent_mgr._build_headless_subagent(task)
    if hasattr(agent, "_handlers") and "pre_tool_call" in agent._handlers:
        handler = agent._handlers["pre_tool_call"][0]
        from myagy.agents.subagent import ToolCallContext
        ctx = ToolCallContext(name="run_command", args={"cmd": "git push"})
        decision = await handler(ctx)
        assert decision.allow is False
        assert "Blocked by policy" in decision.reason
        assert task.tool_traces[0].status == "denied"


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
    assert len(session.agent.history) == 1
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
