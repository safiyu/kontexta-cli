import os
import stat
import time
import pytest
from pathlib import Path

from kontexta_cli.core.reloader import (
    ReloadSnapshot,
    ReloadRequested,
    write_reload_snapshot,
    consume_reload_snapshot,
)
from kontexta_cli.core.session import MultiGpuHybridSession


def test_write_and_consume_reload_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr("kontexta_cli.core.reloader._get_shm_dir", lambda: tmp_path)

    snap = ReloadSnapshot(
        active_target="9000",
        active_models={"rocm": "test-qwen", "cuda": "test-granite", "cloud": "test-gemini"},
        auto_route=False,
        permission_mode="strict",
        chat_history=[{"role": "user", "target": "9000", "content": "hello reload"}],
        compacted_working_memory="working memory test",
        checkpoint_trees=[{"tree": "abc1234", "timestamp": time.time()}],
        stats_ledger={"9000": {"prompt_tokens": 123.0}},
        pending_subagent_tasks=[{"id": 1, "description": "pending task", "status": "queued"}],
    )

    shm_path = write_reload_snapshot(snap)
    assert shm_path.exists()

    # Permissions must be 0o600 (owner read/write only)
    file_stat = os.stat(shm_path)
    file_mode = stat.S_IMODE(file_stat.st_mode)
    assert file_mode == 0o600

    # Consume snapshot
    restored = consume_reload_snapshot(str(shm_path))
    assert restored is not None
    assert restored.active_target == "9000"
    assert restored.active_models["rocm"] == "test-qwen"
    assert restored.chat_history[0]["content"] == "hello reload"
    assert restored.compacted_working_memory == "working memory test"
    assert len(restored.checkpoint_trees) == 1
    assert restored.stats_ledger["9000"]["prompt_tokens"] == 123.0
    assert len(restored.pending_subagent_tasks) == 1

    # Snapshot must be unlinked immediately after consume
    assert not shm_path.exists()


def test_session_export_and_restore_cycle(session):
    session.active_target = "9000"
    session.history = [{"role": "user", "target": "9000", "content": "prior turn"}]
    session.working_memory = "synthesized prior knowledge"
    session.stats = {"9000": {"prompt_tokens": 42.0}}
    session.checkpoints = [{"id": 1, "tree": "tree_xyz"}]

    snap = session.export_reload_snapshot()
    assert isinstance(snap, ReloadSnapshot)
    assert snap.active_target == "9000"
    assert snap.chat_history[0]["content"] == "prior turn"
    assert snap.compacted_working_memory == "synthesized prior knowledge"
    assert snap.checkpoint_trees == [{"id": 1, "tree": "tree_xyz"}]

    # Create fresh session and restore
    new_session = MultiGpuHybridSession(initial_target="cloud")
    new_session.restore_from_snapshot(snap)

    assert new_session.active_target == "9000"
    assert len(new_session.history) == 1
    assert new_session.history[0]["content"] == "prior turn"
    assert new_session.working_memory == "synthesized prior knowledge"
    assert new_session.checkpoints == [{"id": 1, "tree": "tree_xyz"}]
    assert new_session.stats["9000"]["prompt_tokens"] == 42.0


def test_graceful_shutdown_for_reload(session):
    import asyncio
    async def _test():
        await session.graceful_shutdown_for_reload()
    asyncio.run(_test())
