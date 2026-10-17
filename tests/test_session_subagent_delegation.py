import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

from kontexta_cli.agents.subagent import SubagentManager, SubagentTask
from kontexta_cli.core.session import MultiGpuHybridSession
from kontexta_cli.terminal.cli import _cmd_9002


def test_session_target_resolution_and_get_endpoint(session):
    # Tests target resolution for 9002 (primary) and 9001 (aliased since portainer occupies 9001)
    assert session.resolve_target_alias("9000") == "9000"
    assert session.resolve_target_alias("rocm") == "9000"
    # When 9002 is registered in endpoints:
    assert session.resolve_target_alias("9001") == "9002"
    assert session.resolve_target_alias("9002") == "9002"
    assert session.resolve_target_alias("cuda") == "9002"

    ep = session.get_endpoint("9002")
    assert ep["port"] == 9002

    ep9001 = session.get_endpoint("9001")
    assert ep9001["port"] == 9002


def test_session_subagent_delegation(session, monkeypatch):
    assert hasattr(session, "subagent_mgr")
    mgr = session.subagent_mgr

    # Fast mock for execute
    async def fast_exec(task, abort_ev):
        task.output_chunks.append("build passed")
        task.state = "completed"
    monkeypatch.setattr(mgr, "_execute_subagent_task", fast_exec)

    # 1. spawn_subagent
    t1 = session.spawn_subagent("Run unit tests")
    assert isinstance(t1, SubagentTask)
    assert t1.kind == "spawn"
    assert t1.prompt == "Run unit tests"
    assert t1.task_id in mgr.tasks

    # 2. run_build_subagent
    t2 = session.run_build_subagent("make test")
    assert isinstance(t2, SubagentTask)
    assert t2.kind == "build"
    assert t2.command == "make test"
    assert t2.task_id in mgr.tasks

    # 3. list_subagents & get_subagent
    tasks = session.list_subagents()
    assert len(tasks) >= 2
    assert session.get_subagent(t1.task_id) == t1

    # 4. cancel_subagent
    t3 = session.spawn_subagent("Long task")
    assert session.cancel_subagent(t3.task_id) is True
    assert t3.state == "cancelled"

    # 5. inject_subagent & inject_all_subagents
    t1.state = "completed"
    t1.output_chunks = ["Result of task 1"]
    assert session.inject_subagent(t1.task_id) is True
    assert t1.injected is True
    # Duplicate injection should return False
    assert session.inject_subagent(t1.task_id) is False

    t2.state = "completed"
    t2.output_chunks = ["Result of task 2"]
    count = session.inject_all_subagents()
    assert count == 1
    assert t2.injected is True

    # 6. print_subagents & print_subagent_detail don't crash
    session.print_subagents()
    session.print_subagent_detail(t1.task_id)


def test_subagent_manager_spawn_classmethod():
    # Calling SubagentManager.spawn with legacy signatures
    t = SubagentManager.spawn("Do work", None, command="pytest -v")
    assert isinstance(t, SubagentTask)
    assert t.command == "pytest -v"
    assert t.kind == "build"
    assert t.prompt == "Do work"


@pytest.mark.anyio
async def test_session_quiesce_and_graceful_shutdown(session):
    mgr = session.subagent_mgr
    mgr.stop = AsyncMock()

    await session.quiesce_session()
    mgr.stop.assert_awaited_once()


@pytest.mark.anyio
async def test_cli_cmd_9002_and_9001(session):
    ctx_9001 = "/9001"
    await _cmd_9002(session, "", ctx_9001)
    assert session.active_target in ("9001", "9002")
