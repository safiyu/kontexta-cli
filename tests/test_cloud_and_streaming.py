import asyncio
import io
import json
import os
import time

import pytest

FAKE_AGY = r'''#!/usr/bin/env python3
import json, os, sys, time
open(os.environ["AGY_ARGS_FILE"], "w").write(json.dumps(sys.argv[1:]))
open(os.environ["AGY_PID_FILE"], "w").write(str(os.getpid()))
if os.environ.get("AGY_MODE") == "hang":
    print(json.dumps({"event": "step_update", "step_update": {"step_type": "agent_response", "text_delta": "partial "}}), flush=True)
    time.sleep(60)
if os.environ.get("AGY_MODE") == "tool_long":
    print(json.dumps({"event": "step_update", "step_update": {"step_type": "tool", "state": "ACTIVE", "tool_name": "run_command", "tool_info": {"parameters": {"CommandLine": "vitest"}}}}), flush=True)
    time.sleep(float(os.environ.get("AGY_TOOL_SECS", "0.3")))
    print(json.dumps({"event": "step_update", "step_update": {"step_type": "agent_response", "text_delta": "Tests passed"}}), flush=True)
    print(json.dumps({"event": "result"}), flush=True)
    sys.exit(0)
for part in ("Hello ", "**world**"):
    print(json.dumps({"event": "step_update", "step_update": {"step_type": "agent_response", "text_delta": part}}), flush=True)
print(json.dumps({"event": "result"}), flush=True)
'''


@pytest.fixture
def fake_agy(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = bindir / "agy"
    script.write_text(FAKE_AGY)
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("AGY_ARGS_FILE", str(tmp_path / "args.json"))
    monkeypatch.setenv("AGY_PID_FILE", str(tmp_path / "pid"))
    return tmp_path


def _sent_prompt(tmp):
    return json.loads((tmp / "args.json").read_text())[-1]


def test_cloud_streams_text_and_sends_context(session, fake_agy, git_repo, monkeypatch):
    (git_repo / "AGENTS.md").write_text("Prefer tabs.")
    session.auto_instructions = True
    session.history = [
        {"role": "user", "target": "cloud", "content": "earlier question"},
        {"role": "assistant", "target": "cloud", "content": "earlier answer"},
    ]
    text = asyncio.run(session.chat_cloud_oauth("new question"))
    assert text == "Hello **world**"
    sent = _sent_prompt(fake_agy)
    assert "Prefer tabs." in sent and "earlier question" in sent and sent.endswith("[USER]: new question")


def test_cloud_context_can_be_disabled(session, fake_agy, git_repo):
    (git_repo / "AGENTS.md").write_text("Prefer tabs.")
    session.auto_instructions = True
    asyncio.run(session.chat_cloud_oauth("plain", with_project_context=False))
    assert "Prefer tabs." not in _sent_prompt(fake_agy)


def test_cloud_keeps_working_memory_outside_recent_window(session, fake_agy):
    session.max_recent_turns = 2
    session.history = (
        [{"role": "user", "target": "cloud", "content": "[SYSTEM: Context compacted. 9 prior turn(s)]\nGOAL: ship it"}]
        + [{"role": "assistant", "target": "cloud", "content": "ack"}]
        + [{"role": "user", "target": "cloud", "content": f"t{i}"} for i in range(4)]
    )
    asyncio.run(session.chat_cloud_oauth("q", with_project_context=False))
    sent = _sent_prompt(fake_agy)
    assert "[WORKING MEMORY]" in sent and "GOAL: ship it" in sent


def test_cancelling_a_cloud_turn_kills_the_subprocess(session, fake_agy, monkeypatch):
    monkeypatch.setenv("AGY_MODE", "hang")

    async def go():
        task = asyncio.create_task(session.chat_cloud_oauth("long", with_project_context=False))
        for _ in range(50):
            await asyncio.sleep(0.1)
            if (fake_agy / "pid").exists() and (fake_agy / "pid").read_text():
                break
        pid = int((fake_agy / "pid").read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return pid

    pid = asyncio.run(go())
    deadline = time.time() + 3
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.1)
    pytest.fail("agy subprocess survived cancellation")


async def _tokens(*parts):
    for p in parts:
        yield p


@pytest.mark.parametrize("markdown", [True, False])
def test_consume_token_stream(session, monkeypatch, markdown, capsys):
    rich_console = pytest.importorskip("rich.console")
    from kontexta_cli.terminal import ui
    from kontexta_cli.core import session as session_mod
    buf = io.StringIO()
    cap = rich_console.Console(file=buf, force_terminal=False, width=60)
    monkeypatch.setattr(ui, "console", cap)
    monkeypatch.setattr(session_mod, "console", cap)
    session.json_output = False
    session.render_markdown = markdown
    text, count, t_first = asyncio.run(session._consume_token_stream(_tokens("# Hi\n\n", "- a\n", "- b\n")))
    assert text == "# Hi\n\n- a\n- b" and count == 3 and t_first is not None
    out = buf.getvalue() + capsys.readouterr().out
    assert "Hi" in out and "a" in out


def test_consume_token_stream_json_mode_prints_nothing(session, capsys):
    session.json_output = True
    text, count, _ = asyncio.run(session._consume_token_stream(_tokens("x", "y")))
    assert (text, count) == ("xy", 2)
    assert capsys.readouterr().out == ""


def test_consume_token_stream_finalizes_on_cancel(session, monkeypatch):
    rich_console = pytest.importorskip("rich.console")
    from kontexta_cli.terminal import ui
    from kontexta_cli.core import session as session_mod
    cap = rich_console.Console(file=io.StringIO(), force_terminal=False, width=60)
    monkeypatch.setattr(ui, "console", cap)
    monkeypatch.setattr(session_mod, "console", cap)
    session.json_output = False

    async def slow():
        yield "partial "
        await asyncio.sleep(30)

    async def go():
        task = asyncio.create_task(session._consume_token_stream(slow()))
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())  # must not leave a live display running or raise anything else


def test_cloud_tool_execution_does_not_trigger_idle_timeout(session, fake_agy, monkeypatch):
    monkeypatch.setenv("AGY_MODE", "tool_long")
    monkeypatch.setenv("AGY_TOOL_SECS", "1.6")
    # Idle timeout leaves headroom for interpreter startup of the fake agent; the tool
    # then runs longer than it, and being ACTIVE must not trigger the idle timeout.
    session.cloud_idle_timeout = 1.0
    text = asyncio.run(session.chat_cloud_oauth("run vitest", with_project_context=False))
    assert text == "Tests passed"


def test_cloud_idle_timeout_triggers_when_not_in_tool(session, fake_agy, monkeypatch):
    monkeypatch.setenv("AGY_MODE", "hang")
    session.cloud_idle_timeout = 0.1
    with pytest.raises(RuntimeError, match="Cloud model timed out"):
        asyncio.run(session.chat_cloud_oauth("hang", with_project_context=False))

