import asyncio
import subprocess

import pytest

from kontexta_cli.context import kontexta_sync as ks


class FakeSession:
    kontexta_sync = True
    json_output = True


@pytest.fixture(autouse=True)
def clean_state():
    ks._synced_commits.clear()
    ks._pending.clear()


def test_redact_masks_credentials():
    text = "password=hunter2 Authorization: Bearer abcdefghijklmnop1234 key sk-abcdefghijklmnopqrstuv ghp_" + "a" * 24
    out = ks.redact(text)
    for leaked in ("hunter2", "abcdefghijklmnop1234", "sk-abcdefghijklmnopqrstuv", "ghp_" + "a" * 24):
        assert leaked not in out
    assert "[REDACTED]" in out


def test_is_commit_call_detects_variants():
    assert ks.is_commit_call({"name": "run_command", "args": {"command": "git commit -m 'x'"}})
    assert ks.is_commit_call({"name": "run_command", "args": {"command": "git -C . commit -am x"}})
    assert not ks.is_commit_call({"name": "run_command", "args": {"command": "git status"}})
    assert not ks.is_commit_call({"name": "grep", "args": {"q": "commit"}})


def test_commit_info_and_build_entry(git_repo):
    (git_repo / "b.txt").write_text("1\n2\n")
    subprocess.run(["git", "add", "."], cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-qm", "add b"], cwd=git_repo, check=True)
    info = ks.commit_info(str(git_repo))
    assert info["subject"] == "add b"
    assert info["files"] == [{"path": "b.txt", "added": "2", "removed": "0"}]
    text, tags = ks.build_entry("commit", "add b", str(git_repo), commit=info)
    assert "b.txt (+2 -0)" in text and "[kontexta-cli commit]" in text
    assert tags[:2] == ["kontexta-cli", "commit"]


def test_build_entry_truncates_and_redacts():
    text, _ = ks.build_entry("brainstorm", "t", ".", notes=["x" * 9000, "token: abc123secret"])
    assert len(text) <= ks.MAX_ENTRY_CHARS + 20
    assert "abc123secret" not in text


def test_disabled_when_pref_off_or_not_registered(monkeypatch):
    s = FakeSession()
    monkeypatch.setattr(ks.KontextaManager, "is_available", classmethod(lambda cls: True))
    monkeypatch.setattr(ks.KontextaManager, "find_project_for_dir", classmethod(lambda cls, d=".": None))
    assert not ks.enabled(s)
    monkeypatch.setattr(ks.KontextaManager, "find_project_for_dir", classmethod(lambda cls, d=".": {"id": 1}))
    assert ks.enabled(s)
    s.kontexta_sync = False
    assert not ks.enabled(s)


def test_sync_event_pushes_once_per_commit_and_drains(monkeypatch):
    s = FakeSession()
    monkeypatch.setattr(ks, "enabled", lambda session, root=".": True)
    sent = []

    async def fake_push(text, tags, repo_root=".", timeout=0):
        sent.append((text, tags))
        return True

    monkeypatch.setattr(ks, "push", fake_push)
    commit = {"hash": "abc1234", "subject": "s", "files": []}

    async def run():
        assert ks.sync_event(s, "commit", "s", ".", commit=commit)
        assert not ks.sync_event(s, "commit", "s", ".", commit=commit)
        await ks.drain()

    asyncio.run(run())
    assert len(sent) == 1


def test_push_never_raises_without_server(monkeypatch):
    monkeypatch.setattr(ks.KontextaManager, "find_mcp_entrypoint", classmethod(lambda cls: None))
    assert asyncio.run(ks.push("x", ["kontexta-cli"])) is False
    monkeypatch.setattr(ks.KontextaManager, "find_mcp_entrypoint", classmethod(lambda cls: "/nonexistent/index.js"))
    assert asyncio.run(ks.push("x", ["kontexta-cli"], timeout=5)) is False


def test_prefs_include_kontexta_sync():
    from kontexta_cli.core.prefs import PREF_DEFAULTS
    assert PREF_DEFAULTS["kontexta_sync"] is True


def _turn(session, monkeypatch, body):
    async def fake_impl(prompt, force_target=None):
        body(session)
        return "done"
    monkeypatch.setattr(session, "_chat_impl", fake_impl)
    return asyncio.run(session.chat("do it"))


def test_agent_commit_is_journaled(session, git_repo, monkeypatch):
    calls = []
    monkeypatch.setattr(ks, "sync_event", lambda *a, **kw: calls.append((a, kw)) or True)

    def body(s):
        (git_repo / "c.txt").write_text("c\n")
        subprocess.run(["git", "add", "."], cwd=git_repo, check=True)
        subprocess.run(["git", "commit", "-qm", "add c"], cwd=git_repo, check=True)
        s._turn_executed_tools.append({"name": "run_command", "args": {"command": "git commit -m 'add c'"}})

    assert _turn(session, monkeypatch, body) == "done"
    assert len(calls) == 1
    assert calls[0][0][1] == "commit" and calls[0][0][2] == "add c"
    assert calls[0][1]["commit"]["files"][0]["path"] == "c.txt"


def test_turn_without_commit_does_not_sync(session, git_repo, monkeypatch):
    calls = []
    monkeypatch.setattr(ks, "sync_event", lambda *a, **kw: calls.append(a) or True)
    _turn(session, monkeypatch, lambda s: s._turn_executed_tools.append({"name": "grep", "args": {"q": "x"}}))
    assert calls == []


def test_receipt_printed_after_file_changing_turn(session, git_repo, monkeypatch, capsys):
    session.json_output = False
    monkeypatch.setattr(session, "_notice", lambda *a, **k: None)

    def body(s):
        (git_repo / "new.txt").write_text("1\n2\n3\n")
        s._turn_executed_tools.append({"name": "run_command", "args": {"command": "echo hi"}})

    _turn(session, monkeypatch, body)
    out = capsys.readouterr().out
    assert "new.txt" in out and "echo hi" in out


def test_no_receipt_when_nothing_happened(session, git_repo, monkeypatch, capsys):
    session.json_output = False
    _turn(session, monkeypatch, lambda s: None)
    assert "new.txt" not in capsys.readouterr().out
