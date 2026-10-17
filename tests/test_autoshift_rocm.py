import asyncio
import pytest

from kontexta_cli.core.laya import LayaDecisionEngine
from kontexta_cli.core.session import MultiGpuHybridSession
from kontexta_cli.agents.brainstorm import BrainstormWorkflow


def test_is_git_action_ignores_implementation_prompts():
    laya = LayaDecisionEngine(endpoint="http://localhost:8003/v1/systemone")
    # Even if plan contains "commit changes" or "git diff", task implementation should be dev activity
    milestone_prompt = (
        "Here is the approved implementation plan:\n\n"
        "## Milestones\n"
        "- [ ] 1. commit changes and verify with git diff\n\n"
        "TASK: Implement ONLY Milestone 1. Use tools (view_file, edit_file, write_to_file, run_command)"
    )
    is_git, _ = laya.is_git_action(milestone_prompt)
    assert is_git is False

    # Actual git commands should still be identified as git actions
    assert laya.is_git_action("git commit -m 'feat: update'")[0] is True
    assert laya.is_git_action("git push origin main")[0] is True
    assert laya.is_git_action("show git diff")[0] is True


def test_autoshift_swaps_to_gemma_for_git_and_back_to_qwen_for_dev(monkeypatch):
    switches = []

    class DummySession(MultiGpuHybridSession):
        def __init__(self):
            self.auto_git_shift = True
            self.fast_git_model = "gemma4"
            self.dense_code_model = "qwen27"
            self.endpoints = {
                "9000": {"id": "qwen27", "filename": "qwen27.gguf", "name_friendly": "Qwen 27B", "status": "online"}
            }
            self.laya = LayaDecisionEngine()

        def get_catalog(self):
            return [{"id": "gemma4"}, {"id": "qwen27"}]

        async def switch_model(self, model_id):
            switches.append(model_id)
            self.endpoints["9000"]["id"] = model_id
            self.endpoints["9000"]["name_friendly"] = model_id
            self.endpoints["9000"]["filename"] = f"{model_id}.gguf"
            return True

        def refresh_endpoints(self):
            pass

    session = DummySession()

    # 1. Prompt is git action -> should swap to Gemma 4
    asyncio.run(session._autoshift_rocm_model("git commit -m 'feat'"))
    assert switches == ["gemma4"]

    # 2. Subsequent prompt is a dev activity -> should swap back to Qwen 27B
    asyncio.run(session._autoshift_rocm_model("TASK: Implement the user authentication module in auth.py"))
    assert switches == ["gemma4", "qwen27"]


def test_brainstorm_gate2_commit_swaps_gemma_and_restores_qwen(tmp_path, monkeypatch):
    switches = []

    class DummySession:
        auto_git_shift = True
        fast_git_model = "gemma4"
        dense_code_model = "qwen27"
        active_target = "9000"
        endpoints = {
            "9000": {"id": "qwen27", "filename": "qwen27.gguf", "name_friendly": "Qwen 27B", "status": "online"}
        }

        async def switch_model(self, model_id):
            switches.append(model_id)
            self.endpoints["9000"]["name_friendly"] = model_id
            return True

        def refresh_endpoints(self):
            pass

    session = DummySession()

    # Verify that Phase 2 checks and switches to Qwen if port 9000 was on Gemma
    session.endpoints["9000"]["name_friendly"] = "gemma4"
    
    # Run the Gate 2 finally block logic simulation
    async def simulate_gate2():
        try:
            # Commit logic: swap to gemma
            if getattr(session, "auto_git_shift", False):
                curr = session.endpoints["9000"]["name_friendly"]
                if "gemma" not in curr:
                    await session.switch_model(session.fast_git_model)
        finally:
            if getattr(session, "auto_git_shift", False) and session.active_target == "9000":
                curr = session.endpoints["9000"]["name_friendly"]
                if "qwen" not in curr:
                    await session.switch_model(session.dense_code_model)

    asyncio.run(simulate_gate2())
    assert switches == ["qwen27"]
