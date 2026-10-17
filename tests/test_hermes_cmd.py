import asyncio
import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from myagy.terminal import cli


def test_hermes_slash_command_registered():
    assert "/hermes" in cli.SLASH_COMMANDS
    assert "/hermes" in cli.COMMANDS


def test_hermes_status_when_no_arg(monkeypatch, capsys):
    class FakeSession:
        endpoints = {
            "9000": {
                "port": 9000,
                "status": "online",
                "name_friendly": "Qwen 27B Dense",
                "filename": "qwen27.gguf",
            }
        }
        def refresh_endpoints(self):
            pass

    session = FakeSession()
    ctx = MagicMock()

    asyncio.run(cli._cmd_hermes(session, "", ctx))
    # Command should execute without error and display Hermes ROCm info


def test_hermes_warns_when_offline(monkeypatch):
    class FakeSession:
        endpoints = {
            "9000": {
                "port": 9000,
                "status": "offline",
                "name_friendly": "Offline",
                "filename": "",
            }
        }
        def refresh_endpoints(self):
            pass

    session = FakeSession()
    ctx = MagicMock()

    asyncio.run(cli._cmd_hermes(session, "test prompt", ctx))


def test_hermes_prompt_execution(monkeypatch):
    class FakeSession:
        render_markdown = False
        show_metrics = False
        json_output = False
        history = []
        last_answer = ""
        endpoints = {
            "9000": {
                "port": 9000,
                "status": "online",
                "name_friendly": "Qwen 27B Dense",
                "filename": "qwen27.gguf",
            }
        }
        def refresh_endpoints(self):
            pass
        def format_context_badge(self, target, reset_col=""):
            return ""
        def _record_stats(self, *a, **k):
            pass

    test_lines = [
        b"Query: analyze code\n",
        b"Initializing agent...\n",
        b"  \xe2\x95\x8d \xf0\x9f\x92\xbb preparing terminal\xe2\x80\xa6\n",
        b"  \xe2\x95\x8d \xf0\x9f\x92\xbb $ ls -1 config.py 0.1s\n",
        b"\xe2\x95\xad\xe2\x94\x80 \xe2\x9a\x95 Hermes \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x95\xae\n",
        b"    Hermes generated answer\n",
        b"\xe2\x95\xb0\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x95\xaf\n",
        b"Session: 123\n",
        b"",
    ]
    lines_iter = iter(test_lines)

    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.stdout.readline = AsyncMock(side_effect=lambda: next(lines_iter))
    fake_proc.stderr.read = AsyncMock(return_value=b"")
    fake_proc.wait = AsyncMock(return_value=0)

    async def mock_subprocess_exec(*args, **kwargs):
        assert kwargs["env"]["OPENAI_BASE_URL"] == "http://localhost:9000/v1"
        assert kwargs["env"]["OPENAI_API_KEY"] == "local"
        assert "chat" in args
        assert "-q" in args
        return fake_proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_subprocess_exec)

    session = FakeSession()
    ctx = MagicMock()
    asyncio.run(cli._cmd_hermes(session, "analyze code", ctx))

    assert session.last_answer == "Hermes generated answer"
    assert session.history[-1] == {"role": "assistant", "target": "9000", "content": "Hermes generated answer"}
