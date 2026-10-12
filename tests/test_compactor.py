import asyncio

import pytest

from myagy.agents import compactor
from myagy.agents.compactor import ContextCurator
from myagy.core.coordinator import CudaCoordinator


@pytest.fixture(autouse=True)
def no_real_llamashift(monkeypatch):
    monkeypatch.setattr(compactor, "wake_compactor", lambda port=9001, **k: {"ok": False, "reason": "disabled in tests", "log_tail": "", "elapsed": 0.0, "warnings": []})


def _history(n):
    return [{"role": "user", "target": "9000", "content": f"turn {n_}"} for n_ in range(n)]


def test_background_synthesis_runs_and_releases_slot(session, monkeypatch):
    calls = []
    monkeypatch.setattr(ContextCurator, "synthesize_context_cuda_sync", staticmethod(lambda s, t: calls.append(len(t)) or "SUMMARY"))
    session.history = _history(24)  # 12 turns; the default window keeps 6 turns (12 entries)

    async def go():
        ContextCurator.trigger_background_synthesis(session)
        await asyncio.sleep(0.2)

    asyncio.run(go())
    assert calls == [12]
    assert CudaCoordinator.state() == "idle"
    assert len(session.history) == 14  # 2 summary entries + 12 recent entries


def test_subagent_preempts_compaction_and_slot_recovers(session, monkeypatch):
    monkeypatch.setattr(ContextCurator, "synthesize_context_cuda_sync", staticmethod(lambda s, t: "SUMMARY"))
    session.history = _history(24)

    async def go():
        ContextCurator.trigger_background_synthesis(session)
        CudaCoordinator.reserve_for_subagent()
        await asyncio.sleep(0.2)
        assert CudaCoordinator.state() == "subagent"
        await CudaCoordinator.release_from_subagent()

    asyncio.run(go())
    assert CudaCoordinator.can_compact()


def test_no_synthesis_when_history_short(session, monkeypatch):
    calls = []
    monkeypatch.setattr(ContextCurator, "synthesize_context_cuda_sync", staticmethod(lambda s, t: calls.append(1)))
    session.history = _history(4)

    async def go():
        ContextCurator.trigger_background_synthesis(session)
        await asyncio.sleep(0.1)

    asyncio.run(go())
    assert calls == [] and CudaCoordinator.state() == "idle"


def test_recent_window_is_measured_in_turns_not_entries(session, monkeypatch):
    calls = []
    monkeypatch.setattr(ContextCurator, "synthesize_context_cuda_sync", staticmethod(lambda s, t: calls.append(len(t)) or "S"))
    session.max_recent_turns = 3
    session.history = _history(6)  # exactly 3 turns: nothing is older than the window

    async def go():
        ContextCurator.trigger_background_synthesis(session)
        await asyncio.sleep(0.1)

    asyncio.run(go())
    assert calls == [] and len(session.history) == 6

    session.history = _history(10)  # 5 turns: 2 turns (4 entries) are older than the window
    asyncio.run(go())
    assert calls == [4]
    assert len(session.history) == 2 + 6


def test_compact_history_default_keeps_window_in_entries(session, monkeypatch):
    monkeypatch.setattr(ContextCurator, "synthesize_context_cuda_sync", staticmethod(lambda s, t: "S"))
    session.max_recent_turns = 2
    session.history = _history(10)
    removed = ContextCurator.compact_history(session, silent=True)
    assert removed == 6
    assert len(session.history) == 2 + 4


def test_context_estimate_does_not_summarize(session, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("estimate must not run inference")
    monkeypatch.setattr(session, "summarize_history_sync", boom)
    session.history = _history(30)
    metrics = ContextCurator.get_context_metrics(session, "cloud")
    assert metrics["tokens"] > 0


def test_completion_parser_handles_reasoning_and_thinking_tags():
    from myagy.agents.compactor import _parse_completion_response

    # Standard content
    assert _parse_completion_response({"choices": [{"message": {"content": "Clean summary"}}]}) == "Clean summary"

    # Empty content with reasoning_content fallback
    assert _parse_completion_response({"choices": [{"message": {"content": "", "reasoning_content": "Reasoning summary"}}]}) == "Reasoning summary"

    # None content with reasoning_content fallback
    assert _parse_completion_response({"choices": [{"message": {"content": None, "reasoning_content": "Deep reasoning"}}]}) == "Deep reasoning"

    # Content with <think>...</think> and post-think output
    assert _parse_completion_response({"choices": [{"message": {"content": "<think>pondering</think>Final working memory"}}]}) == "Final working memory"

    # Content where thinking consumed everything (no post-think)
    assert _parse_completion_response({"choices": [{"message": {"content": "<think>Thinking summary extracted</think>"}}]}) == "Thinking summary extracted"

    # Fully empty response returns None
    assert _parse_completion_response({"choices": [{"message": {"content": "   ", "reasoning_content": ""}}]}) is None
    assert _parse_completion_response({}) is None


def test_fallback_to_rocm_when_cuda_returns_empty(session, monkeypatch):
    monkeypatch.setattr(ContextCurator, "synthesize_context_cuda_sync", staticmethod(lambda s, t: None))
    monkeypatch.setattr(ContextCurator, "synthesize_context_rocm_sync", staticmethod(lambda s, t: "ROCM SUMMARY"))
    session.endpoints["9000"]["status"] = "online"
    session.history = _history(10)

    removed = ContextCurator.compact_history(session, keep_recent=4, silent=True)
    assert removed == 6
    assert "ROCM SUMMARY" in session.history[0]["content"]
    assert "AMD ROCm (:9000)" in session.history[0]["content"]


def test_fallback_to_cloud_when_local_gpus_return_empty(session, monkeypatch):
    monkeypatch.setattr(ContextCurator, "synthesize_context_cuda_sync", staticmethod(lambda s, t: None))
    monkeypatch.setattr(ContextCurator, "synthesize_context_rocm_sync", staticmethod(lambda s, t: None))
    monkeypatch.setattr(ContextCurator, "synthesize_context_cloud_sync", staticmethod(lambda s, t: "CLOUD SUMMARY"))
    session.has_oauth = True
    session.history = _history(10)

    removed = ContextCurator.compact_history(session, keep_recent=4, silent=True)
    assert removed == 6
    assert "CLOUD SUMMARY" in session.history[0]["content"]
    assert "Google Cloud" in session.history[0]["content"]


def test_fallback_to_deterministic_extractor_when_all_fail(session, monkeypatch):
    monkeypatch.setattr(ContextCurator, "synthesize_context_cuda_sync", staticmethod(lambda s, t: None))
    monkeypatch.setattr(ContextCurator, "synthesize_context_rocm_sync", staticmethod(lambda s, t: None))
    monkeypatch.setattr(ContextCurator, "synthesize_context_cloud_sync", staticmethod(lambda s, t: None))
    session.has_oauth = False

    session.history = [
        {"role": "user", "target": "9000", "content": "Fix bug in terminal/cli.py and run pytest"},
        {"role": "assistant", "target": "9000", "content": "Ran pytest and modified terminal/cli.py. Error: Traceback occurred."},
        {"role": "user", "target": "9000", "content": "Now update core/session.py"},
        {"role": "assistant", "target": "9000", "content": "Updated core/session.py successfully."},
        {"role": "user", "target": "9000", "content": "recent 1"},
        {"role": "assistant", "target": "9000", "content": "recent 2"},
    ]

    removed = ContextCurator.compact_history(session, keep_recent=2, silent=True)
    assert removed == 4
    content = session.history[0]["content"]
    assert "Deterministic Extractive Synthesizer" in content
    assert "Goals & Requirements" in content
    assert "Fix bug in terminal/cli.py" in content
    assert "`terminal/cli.py`" in content or "`core/session.py`" in content
    assert "Traceback" in content or "pytest" in content


def test_compaction_sanitizes_empty_turns(session, monkeypatch):
    monkeypatch.setattr(ContextCurator, "synthesize_context_cuda_sync", staticmethod(lambda s, t: "OK"))
    session.history = [
        {"role": "user", "target": "9000", "content": "old turn 1"},
        {"role": "assistant", "target": "9000", "content": "old turn 2"},
        {"role": "user", "target": "9000", "content": "recent turn 1"},
        {"role": "assistant", "target": "9000", "content": "   "},  # Corrupted empty turn
        {"role": "user", "target": "9000", "content": "recent turn 2"},
    ]

    removed = ContextCurator.compact_history(session, keep_recent=3, silent=True)
    assert removed == 2
    # Ensure no empty content message remains in history
    for entry in session.history:
        assert entry["content"].strip() != ""

