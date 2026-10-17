import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from myagy.core.session import ContextZoneTelemetry, MultiGpuHybridSession
from myagy.terminal.ui import (
    _build_stacked_context_bar,
    render_context_telemetry,
    format_toolbar_context_segment,
    console,
    RICH_AVAILABLE,
)
from myagy.terminal import cli


def test_context_zone_telemetry_properties():
    ctx = ContextZoneTelemetry(
        target="rocm",
        model_id="gemma4",
        n_ctx_max=32768,
        kv_cached_tokens=8192,
        instructions_tokens=1500,
        repomap_tokens=500,
        mcp_schema_tokens=1000,
        frozen_epoch_count=2,
        frozen_epoch_tokens=4000,
        pruned_turn_tokens=1200,
        verbatim_turn_tokens=1800,
        tier0_saved_tokens=600,
        compaction_watermark_ratio=0.75,
    )

    # Zone 1 = instructions + repomap + mcp_schema
    assert ctx.zone1_tokens == 3000
    # Zone 2 = frozen_epoch_tokens
    assert ctx.zone2_tokens == 4000
    # Zone 3 = pruned_turn_tokens + verbatim_turn_tokens
    assert ctx.zone3_tokens == 3000
    # Total = 3000 + 4000 + 3000 = 10000
    assert ctx.total_used_tokens == 10000
    # Headroom = 32768 - 10000 = 22768
    assert ctx.headroom_tokens == 22768
    # Utilization % = 10000 / 32768 * 100 = 30.5175...%
    assert round(ctx.utilization_pct, 1) == 30.5
    # KV prefix hit % = 8192 / 10000 * 100 = 81.92%
    assert round(ctx.kv_prefix_hit_pct, 1) == 81.9


def test_context_zone_telemetry_edge_cases():
    empty_ctx = ContextZoneTelemetry(
        target="cuda",
        model_id="granite42_8b",
        n_ctx_max=16384,
        kv_cached_tokens=0,
        instructions_tokens=0,
        repomap_tokens=0,
        mcp_schema_tokens=0,
        frozen_epoch_count=0,
        frozen_epoch_tokens=0,
        pruned_turn_tokens=0,
        verbatim_turn_tokens=0,
        tier0_saved_tokens=0,
    )
    assert empty_ctx.total_used_tokens == 0
    assert empty_ctx.utilization_pct == 0.0
    assert empty_ctx.kv_prefix_hit_pct == 100.0
    assert empty_ctx.headroom_tokens == 16384

    # KV cached tokens exceeding total used tokens is capped
    overflow_cache_ctx = ContextZoneTelemetry(
        target="rocm",
        model_id="gemma4",
        n_ctx_max=1000,
        kv_cached_tokens=2000,
        instructions_tokens=500,
        repomap_tokens=0,
        mcp_schema_tokens=0,
        frozen_epoch_count=0,
        frozen_epoch_tokens=0,
        pruned_turn_tokens=0,
        verbatim_turn_tokens=0,
        tier0_saved_tokens=0,
    )
    assert overflow_cache_ctx.total_used_tokens == 500
    assert overflow_cache_ctx.kv_prefix_hit_pct == 100.0


def test_est_tokens():
    assert MultiGpuHybridSession._est_tokens("") == 0
    # 38 chars / 3.8 = 10 tokens
    text = "a" * 38
    assert MultiGpuHybridSession._est_tokens(text) == 10


def test_stacked_context_bar():
    ctx = ContextZoneTelemetry(
        target="rocm",
        model_id="gemma4",
        n_ctx_max=32768,
        kv_cached_tokens=4000,
        instructions_tokens=2000,
        repomap_tokens=1000,
        mcp_schema_tokens=1000,
        frozen_epoch_count=1,
        frozen_epoch_tokens=4000,
        pruned_turn_tokens=2000,
        verbatim_turn_tokens=2000,
        tier0_saved_tokens=500,
        compaction_watermark_ratio=0.75,
    )
    bar = _build_stacked_context_bar(ctx, width=56)
    if RICH_AVAILABLE:
        plain_text = bar.plain
        assert len(plain_text) == 56
        # Contains watermark tick "┃"
        assert "┃" in plain_text
        assert "█" in plain_text
        assert "░" in plain_text


def test_format_toolbar_context_segment():
    ctx = ContextZoneTelemetry(
        target="rocm",
        model_id="gemma4",
        n_ctx_max=32768,
        kv_cached_tokens=8000,
        instructions_tokens=1000,
        repomap_tokens=1000,
        mcp_schema_tokens=1000,
        frozen_epoch_count=2,
        frozen_epoch_tokens=3000,
        pruned_turn_tokens=1000,
        verbatim_turn_tokens=1000,
        tier0_saved_tokens=200,
    )
    seg = format_toolbar_context_segment(ctx)
    assert "CTX:" in seg
    assert "Epochs:2" in seg
    assert "KV:" in seg


def test_render_context_telemetry(capsys):
    ctx = ContextZoneTelemetry(
        target="rocm",
        model_id="gemma4",
        n_ctx_max=32768,
        kv_cached_tokens=5000,
        instructions_tokens=1500,
        repomap_tokens=500,
        mcp_schema_tokens=1000,
        frozen_epoch_count=1,
        frozen_epoch_tokens=2000,
        pruned_turn_tokens=1000,
        verbatim_turn_tokens=1000,
        tier0_saved_tokens=300,
    )
    render_context_telemetry(ctx)
    # Shouldn't raise any exceptions


def test_get_context_telemetry_live_slots():
    async def _test():
        session = MultiGpuHybridSession()
        session.active_target = "rocm"
        session.active_models["rocm"] = "gemma4"
        session.repomap.enabled = True
        session.history = [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi there"},
        ]

        mock_slots_response = [
            {
                "id": 0,
                "n_ctx": 32768,
                "n_past": 4096,
            }
        ]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_slots_response

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_resp
            telemetry = await session.get_context_telemetry()

            assert telemetry.target == "rocm"
            assert telemetry.n_ctx_max == 32768
            assert telemetry.kv_cached_tokens == 4096
            assert telemetry.instructions_tokens >= 0
            assert telemetry.total_used_tokens > 0
            assert telemetry.headroom_tokens == 32768 - telemetry.total_used_tokens

    asyncio.run(_test())


def test_cli_context_command():
    async def _test():
        session = MagicMock()
        mock_telemetry = ContextZoneTelemetry(
            target="rocm",
            model_id="gemma4",
            n_ctx_max=32768,
            kv_cached_tokens=1024,
            instructions_tokens=1000,
            repomap_tokens=500,
            mcp_schema_tokens=500,
            frozen_epoch_count=0,
            frozen_epoch_tokens=0,
            pruned_turn_tokens=0,
            verbatim_turn_tokens=1000,
            tier0_saved_tokens=0,
        )
        session.get_context_telemetry = AsyncMock(return_value=mock_telemetry)

        with patch("myagy.terminal.ui.render_context_telemetry") as mock_render:
            ctx = MagicMock()
            await cli.COMMANDS["/context"](session, "", ctx)
            mock_render.assert_called_once_with(mock_telemetry)

            mock_render.reset_mock()
            await cli.COMMANDS["/ctx"](session, "", ctx)
            mock_render.assert_called_once_with(mock_telemetry)

    asyncio.run(_test())
