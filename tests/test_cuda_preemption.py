import asyncio
import pytest
from kontexta_cli.core.coordinator import (
    CudaCoordinator,
    CudaPriority,
    PreemptedByHigherPriority,
    CudaTelemetry,
)
from kontexta_cli.agents.compactor import HippocampusCompactor


def test_cuda_preemption_lifecycle():
    async def _test():
        coord = CudaCoordinator(server_root_url="http://127.0.0.1:9002")
        # Mock llama-server HTTP calls to avoid needing real server
        called_erase = []
        async def mock_erase():
            called_erase.append(True)
            return True
        coord.erase_llama_slot_kv = mock_erase
        coord._wait_for_llama_slot_idle = lambda timeout_s=1.5: asyncio.sleep(0.01)

        events_order = []
        stream_closed = []

        async def fake_compactor():
            try:
                async with coord.acquire_compaction_slot() as abort_event:
                    events_order.append("compactor_started")
                    coord.bind_stream_closer(lambda: stream_closed.append(True) or asyncio.sleep(0))
                    # Simulate waiting or streaming tokens
                    for _ in range(50):
                        if abort_event.is_set():
                            raise PreemptedByHigherPriority("Compactor preempted")
                        await asyncio.sleep(0.02)
                    events_order.append("compactor_finished")
            except PreemptedByHigherPriority:
                events_order.append("compactor_caught_preempted")

        async def fake_subagent():
            await asyncio.sleep(0.05)  # Let compactor start first
            events_order.append("subagent_requesting")
            async with coord.acquire_subagent_slot("unit-test-task") as abort_event:
                events_order.append("subagent_running")
                await asyncio.sleep(0.05)
                events_order.append("subagent_finished")

        await asyncio.gather(fake_compactor(), fake_subagent())
        await coord.stop()

        assert "compactor_started" in events_order
        assert "subagent_requesting" in events_order
        assert "compactor_caught_preempted" in events_order
        assert "subagent_running" in events_order
        assert "subagent_finished" in events_order
        assert len(stream_closed) == 1
        assert len(called_erase) == 1

    asyncio.run(_test())


def test_hippocampus_compactor_retry_on_preemption():
    async def _test():
        coord = CudaCoordinator(server_root_url="http://127.0.0.1:9002")
        coord.erase_llama_slot_kv = lambda: asyncio.sleep(0) or True
        coord._wait_for_llama_slot_idle = lambda timeout_s=1.5: asyncio.sleep(0)

        attempt_count = 0

        class MockStream:
            def __init__(self, should_fail_first):
                self.should_fail_first = should_fail_first
                self.tokens = ["Fact 1", "Fact 2", "Fact 3"]
                self.closed = False

            async def aclose(self):
                self.closed = True

            def __aiter__(self):
                self._idx = 0
                return self

            async def __anext__(self):
                await asyncio.sleep(0.01)
                if self._idx < len(self.tokens):
                    tok = self.tokens[self._idx]
                    self._idx += 1
                    return tok
                raise StopAsyncIteration

        class MockAgent:
            async def send_message_stream(self, prompt):
                nonlocal attempt_count
                attempt_count += 1
                return MockStream(should_fail_first=(attempt_count == 1))

        hc = HippocampusCompactor(coordinator=coord, compactor_agent=MockAgent())

        async def run_compaction():
            return await hc.compact_with_preemption_retry([{"role": "user", "content": "hello"}])

        async def trigger_subagent():
            await asyncio.sleep(0.01)
            # Interrupt the first attempt
            async with coord.acquire_subagent_slot("interrupt-test"):
                await asyncio.sleep(0.02)

        result, _ = await asyncio.gather(run_compaction(), trigger_subagent())
        await coord.stop()

        assert attempt_count == 2
        assert result == "Fact 1Fact 2Fact 3"

    asyncio.run(_test())


def test_cuda_telemetry():
    async def _test():
        coord = CudaCoordinator(server_root_url="http://127.0.0.1:9002")
        status = await coord.get_status()
        assert isinstance(status, CudaTelemetry)
        assert status.state == "idle"
        assert status.preemptions_total == 0
        await coord.stop()

    asyncio.run(_test())
