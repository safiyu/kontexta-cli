import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from kontexta_cli.agents.brainstorm import (
    AutonomousBrainstormWorkflow,
    BrainstormSpec,
    BuildVerificationResult,
    ReviewVerdict,
)
from kontexta_cli.terminal import cli


def test_brainstorm_spec_dataclass():
    spec = BrainstormSpec(
        title="Add Telemetry",
        commit_message="feat(telemetry): add /context metrics",
        summary="A summary of the telemetry feature",
        target_files=["core/session.py", "terminal/ui.py"],
        implementation_steps=["Step 1", "Step 2"],
        test_command="pytest tests/test_context_telemetry.py",
        raw_markdown="# Title\n## Commit Message\nfeat: foo",
    )
    assert spec.title == "Add Telemetry"
    assert spec.target_files == ["core/session.py", "terminal/ui.py"]
    assert spec.test_command == "pytest tests/test_context_telemetry.py"


def test_build_verification_result():
    res = BuildVerificationResult(
        passed=True,
        exit_code=0,
        command="pytest",
        stdout_tail="8 passed in 0.19s",
    )
    assert res.passed
    assert res.exit_code == 0
    assert not res.failure_analysis

    res_fail = BuildVerificationResult(
        passed=False,
        exit_code=1,
        command="pytest",
        stdout_tail="FAILED test_foo",
        failure_analysis="Line 10 assertion failure",
    )
    assert not res_fail.passed
    assert "Line 10" in res_fail.failure_analysis


def test_review_verdict():
    verdict = ReviewVerdict(
        reviewer="ROCm :9000 (Local Executive)",
        approved=True,
        severity="clean",
        findings_markdown="VERDICT: APPROVE\n- Looks good",
        elapsed_s=1.2,
    )
    assert verdict.approved
    assert verdict.severity == "clean"


def test_prompt_builders_and_spec_parser():
    session = MagicMock()
    workflow = AutonomousBrainstormWorkflow(session)

    prompt = workflow._build_spec_prompt("Add Caching", "Use Redis")
    assert "Add Caching" in prompt
    assert "**User Refinement Feedback:** Use Redis" in prompt

    raw_md = """# Caching System
## Commit Message
feat(cache): add redis caching

## Target Files
- `core/cache.py`
- `config.py`

## Test Command
```bash
pytest tests/test_cache.py
```

## Implementation Plan
1. Add redis dependency
"""
    spec = workflow._parse_spec_output(raw_md, "Add Caching")
    assert spec.commit_message == "feat(cache): add redis caching"
    assert "core/cache.py" in spec.target_files
    assert "config.py" in spec.target_files
    assert "pytest tests/test_cache.py" in spec.test_command

    impl_prompt = workflow._build_implementation_prompt(spec, "Fix import error on line 5")
    assert "Fix import error on line 5" in impl_prompt
    assert "Implement the following approved specification" in impl_prompt


def test_phase1_generate_and_approve_spec():
    async def _test():
        session = MagicMock()
        workflow = AutonomousBrainstormWorkflow(session)
        workflow._stream_from_target = AsyncMock(return_value="""# Title
## Commit Message
feat: cool feature
## Target Files
- file.py
## Test Command
pytest
""")

        with patch("rich.prompt.Prompt.ask", return_value="a"):
            spec = await workflow._phase1_generate_and_approve_spec("cool feature")
            assert spec is not None
            assert spec.commit_message == "feat: cool feature"

        with patch("rich.prompt.Prompt.ask", return_value="c"):
            spec = await workflow._phase1_generate_and_approve_spec("cool feature")
            assert spec is None

    asyncio.run(_test())


def test_phase2_and_3_self_healing_build_loop():
    async def _test():
        session = MagicMock()
        session.active_models = {"rocm": "qwen27"}
        session.checkpointer = MagicMock()
        workflow = AutonomousBrainstormWorkflow(session, max_build_retries=1)

        spec = BrainstormSpec(
            title="Feature",
            commit_message="feat: x",
            summary="sum",
            target_files=["a.py"],
            implementation_steps=[],
            test_command="pytest",
            raw_markdown="md",
        )

        workflow._stream_from_target = AsyncMock(return_value="code written")

        # Attempt 1 fails, attempt 2 passes
        res_fail = BuildVerificationResult(
            passed=False, exit_code=1, command="pytest", stdout_tail="SyntaxError", failure_analysis="Missing colon"
        )
        res_pass = BuildVerificationResult(
            passed=True, exit_code=0, command="pytest", stdout_tail="1 passed"
        )
        workflow._run_cuda_build_subagent = AsyncMock(side_effect=[res_fail, res_pass])

        ok = await workflow._phase2_and_3_implement_and_verify(spec)
        assert ok is True
        assert workflow._run_cuda_build_subagent.call_count == 2
        assert session.checkpointer.commit_post_turn_snapshot.call_count == 2

    asyncio.run(_test())


def test_phase4_dual_review_parallel_taskgroup():
    async def _test():
        session = MagicMock()
        session.checkpointer = MagicMock()
        session.checkpointer.get_diff.return_value = "+ new line"

        workflow = AutonomousBrainstormWorkflow(session)

        async def mock_reviewer(target, label, spec, git_diff):
            return ReviewVerdict(
                reviewer=label,
                approved=True,
                severity="clean",
                findings_markdown="VERDICT: APPROVE\n- No bugs found.",
                elapsed_s=0.5,
            )

        workflow._run_single_reviewer = AsyncMock(side_effect=mock_reviewer)

        spec = BrainstormSpec(
            title="Feature",
            commit_message="feat: x",
            summary="sum",
            target_files=["a.py"],
            implementation_steps=[],
            test_command="pytest",
            raw_markdown="md",
        )

        with patch("rich.prompt.Prompt.ask", return_value="c"):
            approved = await workflow._phase4_dual_review_and_gate(spec)
            assert approved is True
            assert workflow._run_single_reviewer.call_count == 2

    asyncio.run(_test())


def test_phase5_commit_and_journal(tmp_path):
    async def _test():
        session = MagicMock()
        session._repo_root = str(tmp_path)
        session.repomap = MagicMock()
        session.repomap.repo_root = str(tmp_path)
        session.kontexta = MagicMock()
        session.kontexta.journal_brainstorm_commit = AsyncMock(return_value=True)

        # Initialize mock git repository
        import subprocess
        subprocess.run(["git", "init"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(tmp_path), check=True)

        (tmp_path / "hello.py").write_text("print('hello')\n")

        workflow = AutonomousBrainstormWorkflow(session)
        spec = BrainstormSpec(
            title="Feature",
            commit_message="feat(core): initial commit",
            summary="sum",
            target_files=["hello.py"],
            implementation_steps=[],
            test_command="pytest",
            raw_markdown="md",
        )

        sha = await workflow._phase5_commit_and_journal(spec, "Feature", memory_sync_enabled=True)
        assert len(sha) == 40
        # Wait a tick for background task
        await asyncio.sleep(0.05)
        session.kontexta.journal_brainstorm_commit.assert_called_once()

    asyncio.run(_test())


def test_full_workflow_run():
    async def _test():
        session = MagicMock()
        session.active_target = "rocm"
        session.checkpointer = MagicMock()

        workflow = AutonomousBrainstormWorkflow(session)

        mock_spec = BrainstormSpec(
            title="Workflow",
            commit_message="feat: complete",
            summary="sum",
            target_files=["a.py"],
            implementation_steps=[],
            test_command="pytest",
            raw_markdown="md",
        )

        workflow._phase1_generate_and_approve_spec = AsyncMock(return_value=mock_spec)
        workflow._phase2_and_3_implement_and_verify = AsyncMock(return_value=True)
        workflow._phase4_dual_review_and_gate = AsyncMock(return_value=True)
        workflow._phase5_commit_and_journal = AsyncMock(return_value="abcdef1234567890abcdef1234567890abcdef12")

        res = await workflow.run("new idea", memory_sync_enabled=True)
        assert res is True
        session.switch_target.assert_called_with("rocm")
        session.checkpointer.snapshot_pre_turn.assert_called_once()

    asyncio.run(_test())


def test_cli_memory_and_brainstorm_commands():
    async def _test():
        session = MagicMock()
        ctx = MagicMock()

        # Test /memory on and off
        await cli.COMMANDS["/memory"](session, "on", ctx)
        assert session.memory_sync_enabled is True
        assert session.kontexta_sync is True

        await cli.COMMANDS["/memory"](session, "off", ctx)
        assert session.memory_sync_enabled is False
        assert session.kontexta_sync is False

        # Test /brainstorm with empty argument
        await cli.COMMANDS["/brainstorm"](session, "", ctx)

        # Test /brainstorm with idea
        with patch.object(AutonomousBrainstormWorkflow, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = True
            await cli.COMMANDS["/brainstorm"](session, "Build caching engine", ctx)
            mock_run.assert_called_once_with(
                idea="Build caching engine",
                memory_sync_enabled=session.memory_sync_enabled,
            )

    asyncio.run(_test())


def test_stream_from_target_cloud_failure_falls_back_to_rocm():
    async def _test():
        session = MagicMock()
        session.active_target = "cloud"

        # First call (cloud) raises AntigravityValidationError, second call (rocm) succeeds
        async def fake_run_turn(prompt, on_thought=None, on_tool_start=None, on_token=None, cancel_event=None):
            if session.active_target == "cloud":
                raise RuntimeError("A Gemini API key is required. Set it via GEMINI_API_KEY environment variable")
            # ROCm turn yields tokens
            if on_token:
                on_token("# ROCm Spec\n## Commit Message\nfeat: fallback")

        def fake_switch(tgt):
            session.active_target = tgt

        session.run_turn = AsyncMock(side_effect=fake_run_turn)
        session.switch_target = MagicMock(side_effect=fake_switch)

        workflow = AutonomousBrainstormWorkflow(session)
        res = await workflow._stream_from_target(
            target="cloud",
            fallback_target="rocm",
            prompt="Draft spec",
            panel_title="Spec Title",
        )
        assert res is not None
        assert "# ROCm Spec" in res
        assert session.switch_target.call_count >= 2

    asyncio.run(_test())


def test_oauth_cloud_agent_instantiated_when_no_api_key_and_has_oauth():
    async def _test():
        from kontexta_cli.core.session import MultiGpuHybridSession, OAuthCloudAgent
        session = MultiGpuHybridSession(api_key=None)
        session.has_oauth = True
        session.cloud_model = "gemini-2.5-pro"

        agent = await session.get_or_switch_local_agent("cloud")
        assert isinstance(agent, OAuthCloudAgent)

        # Cleanup
        await session.reset_local_agent()

    asyncio.run(_test())


def test_cloud_review_fallback_when_no_api_key():
    async def _test():
        from kontexta_cli.core.session import MultiGpuHybridSession, LocalAgentConfig
        session = MultiGpuHybridSession(api_key=None)
        session.has_oauth = False  # no OAuth and no api_key

        cfg = session._make_backend_config("cloud")
        assert isinstance(cfg, LocalAgentConfig)
        assert cfg.api_key is None

        reviewer = session._build_ephemeral_review_agent(cfg)
        stream = await reviewer.send_message_stream("Review this diff")
        text = "".join([tok async for tok in stream])
        assert "VERDICT: APPROVE" in text

        # Cleanup
        await session.reset_local_agent()

    asyncio.run(_test())

