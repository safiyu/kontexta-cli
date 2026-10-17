import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from kontexta_cli.core.coordinator import CudaCoordinator, PreemptedByHigherPriority
from kontexta_cli.agents.compactor import (
    HippocampusCompactor,
    FrozenEpoch,
    EPOCH_LEDGER_SCHEMA,
)


def test_tier0_deterministic_tool_output_pruning():
    coord = CudaCoordinator(server_root_url="http://127.0.0.1:9002")
    compactor = HippocampusCompactor(
        coordinator=coord,
        keep_verbatim_turns=2,  # Last 2 turns (4 messages) kept verbatim
    )

    long_file = "def foo():\n" * 100  # > 600 chars
    long_stdout = "\n".join(f"line {i}: " + ("x" * 20) for i in range(50))  # > 1200 chars
    short_content = "ok, tests passed"

    messages = [
        # Old turn 1 (to be pruned)
        {"role": "user", "content": "read the file"},
        {"role": "tool", "name": "read_file", "content": long_file},
        # Old turn 2 (to be pruned)
        {"role": "user", "content": "run pytest"},
        {"role": "tool", "name": "bash", "content": long_stdout},
        # Old turn 3 with short content (not pruned because <= 600 chars)
        {"role": "user", "content": "check status"},
        {"role": "tool", "name": "bash", "content": short_content},
        # Recent turn 4 (kept verbatim because within keep_verbatim_turns)
        {"role": "user", "content": "read again"},
        {"role": "tool", "name": "read_file", "content": long_file},
        # Recent turn 5 (kept verbatim)
        {"role": "user", "content": "final answer"},
        {"role": "assistant", "content": "done"},
    ]

    pruned = compactor.prune_stale_tool_outputs(messages)

    # 1. Old read_file should be pruned with RepoMap note
    assert "[Tier-0 Pruned `read_file` output" in pruned[1]["content"]
    assert "Current symbol signatures are live in RepoMap" in pruned[1]["content"]

    # 2. Old bash stdout should have head and tail preserved
    assert "[Tier-0 Pruned `bash` stdout" in pruned[3]["content"]
    assert "line 0" in pruned[3]["content"]
    assert "line 49" in pruned[3]["content"]

    # 3. Old short content kept unchanged
    assert pruned[5]["content"] == short_content

    # 4. Recent turns kept 100% verbatim
    assert pruned[7]["content"] == long_file
    assert pruned[9]["content"] == "done"


@pytest.mark.anyio
async def test_tier1_and_tier2_epoch_compaction_and_prefix_rendering():
    coord = CudaCoordinator(server_root_url="http://127.0.0.1:9002")

    # Mock HTTP client streaming response matching EPOCH_LEDGER_SCHEMA
    mock_ledger = {
        "epoch_goal": "Implement distributed SQLite cache",
        "files_touched": [
            {"path": "core/cache.py", "changes": "Added WAL mode and LRU eviction"},
            {"path": "config.py", "changes": "Added CACHE_DIR setting"},
        ],
        "decisions_and_invariants": ["Use SQLite WAL mode", "1.5ms latency SLA"],
        "failed_hypotheses_do_not_repeat": ["In-memory only cache lost state on restart"],
        "unresolved_blockers": ["Benchmark multi-process concurrency"],
    }

    mock_resp = MagicMock()
    async def fake_aiter_lines():
        import json
        yield f'data: {{"choices": [{{"delta": {{"content": {json.dumps(json.dumps(mock_ledger))}}}}}]}}'
        yield "data: [DONE]"
    mock_resp.aiter_lines = fake_aiter_lines
    mock_resp.aclose = AsyncMock()

    mock_http = MagicMock()
    mock_http.stream.return_value.__aenter__.return_value = mock_resp
    mock_http.stream.return_value.__aexit__.return_value = None

    # Mock Kontexta bridge
    mock_kontexta = AsyncMock()
    mock_kontexta.archive_raw_turns.return_value = "kxta://session/test/epoch-1"

    compactor = HippocampusCompactor(
        coordinator=coord,
        cuda_http_client=mock_http,
        kontexta_bridge=mock_kontexta,
        keep_verbatim_turns=2,
        epoch_chunk_turns=3,  # 3 turns = 6 messages per epoch
    )

    # 5 turns = 10 messages (>= (3 + 2) * 2 = 10 messages -> triggers compaction)
    messages = []
    for i in range(1, 6):
        messages.append({"role": "user", "content": f"User turn {i}"})
        messages.append({"role": "assistant", "content": f"Assistant response {i}"})

    remaining = await compactor.maybe_compact_epoch(messages, latest_git_tree_sha="a1b2c3d")

    # Sliced off first 3 turns (6 messages); 2 turns (4 messages) remain
    assert len(remaining) == 4
    assert remaining[0]["content"] == "User turn 4"

    # 1 epoch was frozen
    assert len(compactor.frozen_epochs) == 1
    epoch = compactor.frozen_epochs[0]
    assert epoch.epoch_id == 1
    assert epoch.turn_range == (1, 3)
    assert epoch.git_tree_sha == "a1b2c3d"
    assert epoch.kontexta_ref == "kxta://session/test/epoch-1"
    assert epoch.ledger["epoch_goal"] == "Implement distributed SQLite cache"

    # Check to_prompt_block formatting
    block = epoch.to_prompt_block()
    assert "Frozen Epoch 1 | Turns 1-3 | git:a1b2c3d | raw:kxta://session/test/epoch-1" in block
    assert "`core/cache.py` (Added WAL mode and LRU eviction)" in block
    assert "In-memory only cache lost state on restart" in block
    assert "Benchmark multi-process concurrency" in block

    # Check render_working_memory_prefix for Zone 2 append-only injection
    prefix = compactor.render_working_memory_prefix()
    assert prefix.startswith("## Working Memory (Frozen Epochs)\n")
    assert block in prefix

    await coord.stop()


@pytest.mark.anyio
async def test_compactor_preemption_leaves_messages_intact():
    coord = CudaCoordinator(server_root_url="http://127.0.0.1:9002")

    # Mock agent that triggers preemption
    class PreemptingAgent:
        async def send_message_stream(self, prompt):
            # Simulate being aborted
            raise PreemptedByHigherPriority()

    compactor = HippocampusCompactor(
        coordinator=coord,
        compactor_agent=PreemptingAgent(),
        keep_verbatim_turns=1,
        epoch_chunk_turns=2,
    )

    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]

    # When preempted, messages are preserved intact for the next turn
    res = await compactor.maybe_compact_epoch(messages)
    assert len(res) == 6
    assert len(compactor.frozen_epochs) == 0

    await coord.stop()
