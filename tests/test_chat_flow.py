import asyncio
import json
import os
import stat

import pytest

from kontexta_cli.context.hooks_loader import ExternalHooksManager


def _write_hook(repo, event, script_body):
    script = repo / "hook.sh"
    script.write_text("#!/bin/sh\n" + script_body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    (repo / "hooks.json").write_text(json.dumps({"h": {event: [{"matcher": "*", "hooks": [{"command": str(script)}]}]}}))


def _stub_impl(session, calls, edit=None, reply="done"):
    async def impl(prompt, force_target=None):
        calls.append(prompt)
        if edit:
            edit()
        return reply
    session._chat_impl = impl


def test_chat_turn_delegates_with_target(session):
    seen = {}

    async def fake(prompt, force_target=None):
        seen.update(prompt=prompt, target=force_target)
        return "ok"

    session.chat = fake
    asyncio.run(session.chat_turn("hi", target="9000"))
    assert seen == {"prompt": "hi", "target": "9000"}


def test_checkpoint_diff_and_undo(session, git_repo):
    session.enable_hooks = False
    calls = []

    def edit():
        (git_repo / "a.txt").write_text("edited\n")
        (git_repo / "made.py").write_text("x = 1\n")

    _stub_impl(session, calls, edit=edit)
    asyncio.run(session.chat("change things"))

    assert len(session.checkpoints) == 1
    assert "+edited" in session.turn_diff(1)
    plan = session.undo_plan()
    assert plan["files"] == {"a.txt": "M", "made.py": "A"}

    session.apply_undo(plan)
    assert (git_repo / "a.txt").read_text() == "a\n"
    assert not (git_repo / "made.py").exists()
    assert session.checkpoints == []
    assert "reverted all file changes" in session.history[-2]["content"]
    assert session.turn_diff(1) is None


def test_no_checkpoint_when_nothing_changed(session, git_repo):
    session.enable_hooks = False
    _stub_impl(session, [])
    asyncio.run(session.chat("just talk"))
    assert session.checkpoints == []


def test_outside_git_repo_is_harmless(session, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session.enable_hooks = False
    _stub_impl(session, [])
    assert asyncio.run(session.chat("hello")) == "done"
    assert session.checkpoints == [] and session.undo_plan() is None


def test_preturn_hook_denies(session, git_repo):
    _write_hook(git_repo, "PreTurn", 'echo \'{"decision":"deny","reason":"nope"}\'\n')
    calls = []
    _stub_impl(session, calls)
    assert asyncio.run(session.chat("anything")) == ""
    assert calls == []


def test_preturn_hook_rewrites_prompt_and_adds_context(session, git_repo):
    _write_hook(git_repo, "PreTurn", 'cat >/dev/null; echo \'{"context":"CTX","overwrite":{"prompt":"rewritten"}}\'\n')
    calls = []
    _stub_impl(session, calls)
    asyncio.run(session.chat("original"))
    assert calls == ["CTX\n\nrewritten"]


def test_stop_hook_blocks_then_allows(session, git_repo):
    counter = git_repo / "count"
    counter.write_text("0")
    _write_hook(git_repo, "Stop", f'''n=$(cat {counter}); echo $((n+1)) > {counter}
if [ "$n" -lt 1 ]; then echo '{{"decision":"block","reason":"run the tests"}}'; else echo '{{}}'; fi
''')
    calls = []
    _stub_impl(session, calls)
    asyncio.run(session.chat("do work"))
    assert calls == ["do work", "run the tests"]


def test_stop_hook_retries_are_capped(session, git_repo):
    _write_hook(git_repo, "Stop", 'echo \'{"decision":"block","reason":"again"}\'\n')
    calls = []
    _stub_impl(session, calls)
    asyncio.run(session.chat("x"))
    assert len(calls) == 1 + session.MAX_STOP_RETRIES


def test_posttool_payload_has_call_details_and_output(session, git_repo):
    from google.antigravity import types
    out = git_repo / "payload.json"
    script = git_repo / "post.sh"
    script.write_text(f"#!/bin/sh\ncat > {out}\n")
    script.chmod(0o755)
    (git_repo / "hooks.json").write_text(json.dumps({"h": {"PostToolUse": [{"matcher": "*", "hooks": [{"command": str(script)}]}]}}))

    hooks = ExternalHooksManager.build_lifecycle_hooks(session, ".")
    post = hooks[-1]
    session._last_tool_call = {"name": "run_command", "args": {"CommandLine": "ls"}}
    post(types.ToolResult(result="file1\nfile2", error=None))

    payload = json.loads(out.read_text())
    assert payload["toolCall"] == {"name": "run_command", "args": {"CommandLine": "ls"}}
    assert payload["output"] == "file1\nfile2"


def test_stats_ledger_counts_turns_and_escalations(session):
    session._record_stats("9000", "9000", 100, 50.0, 200.0, 2.0)
    session._record_stats("9000", "cloud", 40, None, None, 3.0)
    assert session.stats["9000"]["turns"] == 1 and session.stats["9000"]["escalations"] == 1
    assert session.stats["cloud"]["turns"] == 1
    session.print_stats()  # must not raise


def test_cloud_project_context_respects_toggles(session, git_repo):
    (git_repo / "AGENTS.md").write_text("Always use tabs.")
    session.auto_instructions = True
    assert "Always use tabs." in session._cloud_project_context()
    session.auto_instructions = False
    assert session._cloud_project_context() == ""


def test_list_sessions_newest_first(session, tmp_path, monkeypatch):
    import kontexta_cli.core.session as ss
    monkeypatch.setattr(ss, "SESSIONS_DIR", str(tmp_path))
    for name, ts in (("session_zzz.json", "2020-01-01T00:00:00"), ("latest_session.json", "2026-10-05T10:00:00")):
        (tmp_path / name).write_text(json.dumps({"saved_at": ts, "history": []}))
    assert [s["file"] for s in session.list_sessions()] == ["latest_session.json", "session_zzz.json"]


def test_system_context_history_toggle(session):
    session.summarize_history_sync = lambda turns: "SUMMARY"
    session.history = [{"role": "user", "target": "9000", "content": "hello"}] * 10
    assert "[Conversation History]" in session.build_system_context("9000")
    assert "[Conversation History]" not in session.build_system_context("9000", include_history=False)
