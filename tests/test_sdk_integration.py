import asyncio
import pytest
from types import SimpleNamespace
from kontexta_cli.core.session import MultiGpuHybridSession
from kontexta_cli.context.hooks_loader import ExternalHooksManager
from google.antigravity import hooks, types

class FakeToolCall:
    def __init__(self, name: str, args: dict):
        self.name = name
        self.args = args

class FakeChatResponse:
    """Simulates SDK ChatResponse with thoughts, tool_calls, text streams, and usage_metadata."""
    def __init__(self, thoughts_list, tool_calls_list, text_tokens, usage_metadata=None):
        self._thoughts = thoughts_list
        self._tool_calls = tool_calls_list
        self._text = text_tokens
        self.usage_metadata = usage_metadata

    @property
    def thoughts(self):
        async def _gen():
            for t in self._thoughts:
                await asyncio.sleep(0.001)
                yield t
        return _gen()

    @property
    def tool_calls(self):
        async def _gen():
            for tc in self._tool_calls:
                await asyncio.sleep(0.001)
                yield tc
        return _gen()

    def __aiter__(self):
        async def _gen():
            for tok in self._text:
                await asyncio.sleep(0.001)
                yield tok
        return _gen()


def test_triple_stream_multiplexing_and_telemetry(session):
    async def _run():
        usage_meta = SimpleNamespace(
            prompt_token_count=120,
            candidates_token_count=45,
            cached_content_token_count=30,
            thoughts_token_count=60,
            total_token_count=225,
        )
        fake_resp = FakeChatResponse(
            thoughts_list=["Let me check...", " Now compiling the result..."],
            tool_calls_list=[FakeToolCall("view_file", {"AbsolutePath": "/test.py"})],
            text_tokens=["Hello", " ", "world!"],
            usage_metadata=usage_meta,
        )

        text, tokens, t_first = await session._consume_token_stream(fake_resp)
        assert text == "Hello world!"
        assert tokens == 45  # Reported candidates token count from usage_metadata
        assert t_first is not None
        assert session._last_usage_metadata is usage_meta

        # Record stats
        session._record_stats("9000", "9000", tokens, 45.0, 150.0, 1.0, usage_metadata=usage_meta)
        st = session.stats["9000"]
        assert st["prompt_tokens"] == 120
        assert st["cached_tokens"] == 30
        assert st["thoughts_tokens"] == 60
        assert st["candidates_tokens"] == 45
        assert st["total_tokens"] == 225

        # print_stats should render without error
        session.print_stats()

    asyncio.run(_run())


def test_unified_multi_backend_routing_cloud(session):
    async def _run():
        session.api_key = "test-key"
        cloud_agent = await session.get_or_switch_local_agent("cloud")
        assert cloud_agent is not None
        assert session._current_agent_target == "cloud"
        assert session._active_local_model == session.cloud_model

    asyncio.run(_run())


def test_in_process_lifecycle_hook_registration():
    ExternalHooksManager.clear_in_process_hooks()
    called = []

    @hooks.post_tool_call
    def my_in_proc_hook(res):
        called.append(res)

    ExternalHooksManager.register_in_process_hook(my_in_proc_hook)
    built = ExternalHooksManager.build_lifecycle_hooks(SimpleNamespace(json_output=False, verbose=False), ".")
    assert my_in_proc_hook in built
    ExternalHooksManager.clear_in_process_hooks()


def test_native_compaction_and_bayesian_hooks_wired(session):
    session.use_laya_adaptive_permissions = True
    session._dangerously_skip_permissions = False
    all_hooks = session.build_tool_hooks()
    # Ensure on_compaction and laya decide gate are among hooks
    hook_names = [getattr(h, "__name__", "") for h in all_hooks]
    assert any("compaction" in n for n in hook_names)
    assert any("laya" in n for n in hook_names)
