import asyncio
import types

import pytest

from kontexta_cli.agents import subagent as sa
from kontexta_cli.agents.subagent import SubagentManager, SubagentTask
from kontexta_cli.core.coordinator import CudaCoordinator


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def __aiter__(self):
        async def gen():
            for tok in self.text.split(" "):
                yield tok + " "
        return gen()


def _fake_agent(text=None, fail=False):
    class FakeAgent:
        closed = False

        def __init__(self, config):
            self.config = config
            FakeAgent.last = self

        async def __aenter__(self):
            if fail:
                raise RuntimeError("sdk exploded")
            return self

        async def __aexit__(self, *exc):
            FakeAgent.closed = True

        async def chat(self, prompt):
            return FakeResponse(text)
    return FakeAgent


@pytest.fixture
def online_session(session):
    session.endpoints["9002"].update(status="online", filename="granite.gguf", model="/m/granite.gguf", url="http://localhost:9002/v1")
    return session


def _run(task, session):
    async def go():
        CudaCoordinator.reserve_for_subagent()
        await SubagentManager._run_worker(task, session)
    asyncio.run(go())


def test_tool_agent_success(online_session, monkeypatch):
    Fake = _fake_agent("all done here")
    monkeypatch.setattr(sa, "Agent", Fake)
    task = SubagentTask(1, "inspect things")
    _run(task, online_session)
    assert task.status == "completed" and task.mode == "tools"
    assert task.result == "all done here"
    assert Fake.closed
    assert Fake.last.config.kwargs["base_url"] == "http://localhost:9002/v1"


def test_falls_back_to_plain_when_agent_fails(online_session, monkeypatch):
    monkeypatch.setattr(sa, "Agent", _fake_agent(fail=True))

    async def plain(task, ep):
        return "plain answer"

    monkeypatch.setattr(SubagentManager, "_run_plain", classmethod(lambda cls, t, e: plain(t, e)))
    task = SubagentTask(2, "think")
    _run(task, online_session)
    assert task.status == "completed" and task.mode == "no-tools"
    assert task.result == "plain answer"
    assert any("tool agent unavailable" in c["name"] for c in task.tool_calls)


def test_offline_port_fails_cleanly(session, monkeypatch):
    monkeypatch.setattr(sa, "wake_compactor", lambda port=9002, **k: {"ok": False, "reason": "down", "log_tail": "", "elapsed": 0.0, "warnings": []})
    task = SubagentTask(3, "x")
    _run(task, session)
    assert task.status == "failed" and "offline" in task.error
    assert CudaCoordinator.state() == "idle"


def test_background_permission_denies_unsafe(online_session, monkeypatch):
    captured = {}

    class Fake(_fake_agent("ok")):
        def __init__(self, config):
            captured["policies"] = config.kwargs["policies"]
            super().__init__(config)

    monkeypatch.setattr(sa, "Agent", Fake)
    task = SubagentTask(4, "danger")
    _run(task, online_session)
    handler = captured["policies"][1]
    assert handler(types.SimpleNamespace(name="run_command"), {"CommandLine": "rm -rf /"}) is False
    assert handler(types.SimpleNamespace(name="run_command"), {"CommandLine": "ls"}) is True
    assert any("denied" in c["name"] for c in task.tool_calls)
