# Projects/myagy/core/coordinator.py
from __future__ import annotations

import asyncio
import itertools
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, Dict, List

import httpx

from myagy.config import CUDA_BASE_URL  # e.g. "http://127.0.0.1:9002/v1"


class CudaPriority(IntEnum):
    DIRECT_REPL = 0   # /9002 interactive user turn
    SUBAGENT = 10     # /spawn and /build tasks (FIFO serialized)
    COMPACTOR = 50    # Background hippocampus curation


class PreemptedByHigherPriority(asyncio.CancelledError):
    """Raised inside a background compaction task when preempted by a subagent."""


@dataclass(order=True)
class _SlotWaiter:
    priority: int
    seq: int
    label: str = field(compare=False)
    future: asyncio.Future[None] = field(compare=False)
    abort_event: asyncio.Event = field(compare=False, default_factory=asyncio.Event)
    enqueued_at: float = field(compare=False, default_factory=time.monotonic)


@dataclass
class CudaTelemetry:
    state: str                  # "idle" | "subagent" | "compacting" | "preempting"
    active_label: str | None
    active_priority: int | None
    queued_subagents: int
    queued_compactors: int
    preemptions_total: int
    slot_kv_used: int = 0
    slot_kv_max: int = 0


class CudaCoordinator:
    """
    Hardware Slot Arbitrator for NVIDIA RTX 4060 (CUDA:9002).
    Enforces strict priority scheduling and llama-server slot preemption.
    Also provides backward-compatible classmethods for existing callers/tests.
    """

    # Class-level state for backward compatibility
    _state: str = "idle"  # "idle" | "compacting" | "subagent" | "preempting"
    _active_subagents: int = 0
    _compactor_task: Optional[asyncio.Task] = None
    _state_lock = asyncio.Lock()
    _last_state_change: float = time.time()
    _active_coordinator: Optional["CudaCoordinator"] = None

    def __init__(
        self,
        server_root_url: str = CUDA_BASE_URL.removesuffix("/v1"),
        slot_id: int = 0,
    ) -> None:
        self.server_root = server_root_url.rstrip("/")
        self.slot_id = slot_id

        self._queue: asyncio.PriorityQueue[_SlotWaiter] = asyncio.PriorityQueue()
        self._seq = itertools.count()

        self._active_waiter: _SlotWaiter | None = None
        self._active_task: asyncio.Task[Any] | None = None
        self._active_http_cancel: Callable[[], Awaitable[None]] | None = None

        self._slot_released = asyncio.Event()
        self._slot_released.set()

        self._preemptions_total = 0
        self._state: str = "idle"
        self._http = httpx.AsyncClient(base_url=self.server_root, timeout=5.0)
        self._scheduler_task: asyncio.Task[None] | None = None
        CudaCoordinator._active_coordinator = self

    def start(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._scheduler_task is None or self._scheduler_task.done():
            self._scheduler_task = loop.create_task(
                self._scheduler_loop(), name="cuda-slot-arbitrator"
            )

    async def stop(self) -> None:
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None
        await self._http.aclose()

    async def _scheduler_loop(self) -> None:
        """Continuously pops the highest-priority waiter whenever the GPU slot is free."""
        while True:
            waiter = await self._queue.get()
            if waiter.future.cancelled():
                self._queue.task_done()
                continue

            self._slot_released.clear()
            self._active_waiter = waiter
            self._state = (
                "compacting"
                if waiter.priority >= CudaPriority.COMPACTOR
                else "subagent"
            )
            CudaCoordinator._state = self._state
            CudaCoordinator._last_state_change = time.time()

            # Grant the GPU lock to the waiter
            if not waiter.future.done():
                waiter.future.set_result(None)

            # Wait until the holder exits its context manager
            await self._slot_released.wait()
            self._queue.task_done()

    # -------------------------------------------------------------------------
    # Preemption Engine + llama-server /slots Control
    # -------------------------------------------------------------------------

    async def _preempt_active_compactor(self, incoming_label: str) -> None:
        """
        Aborts an in-flight compaction stream, waits for llama-server slot 0
        to exit its forward pass, and flushes dirty KV cache tokens.
        """
        if (
            self._active_waiter is None
            or self._active_waiter.priority < CudaPriority.COMPACTOR
        ):
            return

        self._state = "preempting"
        CudaCoordinator._state = "preempting"
        CudaCoordinator._last_state_change = time.time()
        self._preemptions_total += 1

        # 1. Signal cooperative abort + close in-flight httpx stream immediately
        self._active_waiter.abort_event.set()
        if self._active_http_cancel is not None:
            try:
                await self._active_http_cancel()
            except Exception:
                pass

        # 2. Cancel the running Python task with our typed Preempted exception
        if self._active_task and not self._active_task.done():
            self._active_task.cancel(
                f"Preempted on :9002 by higher-priority task '{incoming_label}'"
            )

        # 3. Poll llama-server GET /slots until slot 0 stops processing the aborted batch
        await self._wait_for_llama_slot_idle(timeout_s=1.5)

        # 4. Erase dirty partial compaction KV cache so the subagent starts clean
        await self.erase_llama_slot_kv()

    async def _wait_for_llama_slot_idle(self, timeout_s: float = 1.5) -> None:
        """Polls llama-server /slots until is_processing is False (~5-25ms after TCP close)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                resp = await self._http.get("/slots")
                if resp.status_code == 200:
                    slots = resp.json()
                    slot = next(
                        (s for s in slots if s.get("id") == self.slot_id),
                        slots[0] if slots else None,
                    )
                    if not slot or not slot.get("is_processing", False):
                        return
            except httpx.HTTPError:
                return
            await asyncio.sleep(0.015)  # 15ms poll interval

    async def erase_llama_slot_kv(self) -> bool:
        """Calls POST /slots/{id}?action=erase on llama-server (:9002)."""
        try:
            resp = await self._http.post(
                f"/slots/{self.slot_id}", params={"action": "erase"}
            )
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def save_llama_slot_kv(self, filename: str = "compactor_prefix.bin") -> bool:
        """Saves KV cache slot snapshot to RAM disk for fast restore."""
        try:
            resp = await self._http.post(
                f"/slots/{self.slot_id}",
                params={"action": "save"},
                json={"filename": filename},
            )
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def restore_llama_slot_kv(self, filename: str = "compactor_prefix.bin") -> bool:
        """Restores KV cache slot snapshot from RAM disk."""
        try:
            resp = await self._http.post(
                f"/slots/{self.slot_id}",
                params={"action": "restore"},
                json={"filename": filename},
            )
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    # -------------------------------------------------------------------------
    # Context Managers for Subagents (/spawn, /build) & Compactor
    # -------------------------------------------------------------------------

    @asynccontextmanager
    async def acquire_slot(
        self,
        priority: CudaPriority,
        label: str,
        register_http_closer: Callable[[Callable[[], Awaitable[None]]], None] | None = None,
    ) -> AsyncIterator[asyncio.Event]:
        """
        Acquires exclusive access to CUDA:9002.
        If `priority < COMPACTOR` and the compactor is currently running,
        immediately preempts the compactor and reclaims the llama-server slot.
        """
        self.start()
        loop = asyncio.get_running_loop()
        waiter = _SlotWaiter(
            priority=int(priority),
            seq=next(self._seq),
            label=label,
            future=loop.create_future(),
        )
        self._queue.put_nowait(waiter)

        # Preempt running background compaction if an on-demand task arrives
        if (
            priority < CudaPriority.COMPACTOR
            and self._active_waiter is not None
            and self._active_waiter.priority >= CudaPriority.COMPACTOR
        ):
            await self._preempt_active_compactor(incoming_label=label)

        try:
            await waiter.future
        except asyncio.CancelledError:
            waiter.future.cancel()
            raise

        self._active_task = asyncio.current_task()

        try:
            yield waiter.abort_event
        except asyncio.CancelledError as exc:
            if waiter.abort_event.is_set():
                raise PreemptedByHigherPriority(
                    f"Task '{label}' was preempted on CUDA:9002"
                ) from exc
            raise
        finally:
            self._active_waiter = None
            self._active_task = None
            self._active_http_cancel = None
            self._state = "idle"
            CudaCoordinator._state = "idle"
            CudaCoordinator._last_state_change = time.time()
            self._slot_released.set()

    def bind_stream_closer(self, closer: Callable[[], Awaitable[None]]) -> None:
        """Allows the active holder to register its httpx/SDK stream `.aclose` callback."""
        self._active_http_cancel = closer

    # -------------------------------------------------------------------------
    # Convenience Wrappers for /spawn & compactor.py
    # -------------------------------------------------------------------------

    @asynccontextmanager
    async def acquire_subagent_slot(self, task_name: str) -> AsyncIterator[asyncio.Event]:
        """Priority 10 FIFO slot for `/spawn` and `/build`."""
        async with self.acquire_slot(CudaPriority.SUBAGENT, label=f"subagent:{task_name}") as ev:
            yield ev

    @asynccontextmanager
    async def acquire_compaction_slot(self) -> AsyncIterator[asyncio.Event]:
        """Priority 50 preemptible slot for background hippocampus compaction."""
        async with self.acquire_slot(CudaPriority.COMPACTOR, label="hippocampus-compact") as ev:
            yield ev

    async def synthesize_working_memory(self, messages: list[dict[str, Any]]) -> str:
        """Synthesizes working memory on CUDA:9002 for compaction hook."""
        from myagy.agents.compactor import ContextCurator
        return ContextCurator.summarize_history_sync(None, messages)

    # -------------------------------------------------------------------------
    # Live Telemetry for `/cuda` command and Bottom Status Toolbar
    # -------------------------------------------------------------------------

    async def get_status(self) -> CudaTelemetry:
        items = list(self._queue._queue)  # Snapshot internal heap without mutating
        q_sub = sum(1 for w in items if w.priority < CudaPriority.COMPACTOR and not w.future.cancelled())
        q_comp = sum(1 for w in items if w.priority >= CudaPriority.COMPACTOR and not w.future.cancelled())

        kv_used, kv_max = 0, 0
        try:
            resp = await self._http.get("/slots")
            if resp.status_code == 200 and (slots := resp.json()):
                s0 = slots[0]
                kv_max = s0.get("n_ctx", 0)
                kv_used = s0.get("n_past", 0) or s0.get("next_token", [{}])[0].get("n_decoded", 0)
        except httpx.HTTPError:
            pass

        return CudaTelemetry(
            state=self._state,
            active_label=self._active_waiter.label if self._active_waiter else None,
            active_priority=self._active_waiter.priority if self._active_waiter else None,
            queued_subagents=q_sub,
            queued_compactors=q_comp,
            preemptions_total=self._preemptions_total,
            slot_kv_used=kv_used,
            slot_kv_max=kv_max,
        )

    # -------------------------------------------------------------------------
    # Backward Compatibility Classmethods
    # -------------------------------------------------------------------------

    @classmethod
    def state(cls) -> str:
        return cls._state

    @classmethod
    def is_idle(cls) -> bool:
        return cls._state == "idle" and cls._active_subagents == 0

    @classmethod
    def can_compact(cls) -> bool:
        """Compactor is only allowed to run when CUDA 9001/9002 is completely idle."""
        return cls._state == "idle" and cls._active_subagents == 0

    @classmethod
    def start_compaction(cls, task: Optional[asyncio.Task] = None) -> bool:
        """Attempts to lock Port 9001/9002 for background compaction."""
        if not cls.can_compact():
            return False
        cls._state = "compacting"
        cls._compactor_task = task
        cls._last_state_change = time.time()
        return True

    @classmethod
    def finish_compaction(cls) -> None:
        """Releases Port 9001/9002 back to idle after compaction completes."""
        if cls._state == "compacting":
            cls._state = "idle"
            cls._compactor_task = None
            cls._last_state_change = time.time()

    @classmethod
    def reserve_for_subagent(cls) -> None:
        """
        Synchronously reserves CUDA immediately upon subagent spawn,
        preempting active background compaction task.
        """
        if cls._compactor_task and not cls._compactor_task.done():
            cls._compactor_task.cancel()
            cls._compactor_task = None
        cls._active_subagents += 1
        cls._state = "subagent"
        cls._last_state_change = time.time()

    @classmethod
    async def acquire_for_subagent(cls) -> bool:
        """Async reserve for subagent."""
        async with cls._state_lock:
            if cls._state == "compacting" and cls._compactor_task and not cls._compactor_task.done():
                cls._compactor_task.cancel()
                try:
                    await asyncio.sleep(0.05)
                except Exception:
                    pass
                cls._compactor_task = None

            cls._active_subagents += 1
            cls._state = "subagent"
            cls._last_state_change = time.time()
            return True

    @classmethod
    async def release_from_subagent(cls) -> None:
        """Releases Port 9001/9002 from a subagent task."""
        async with cls._state_lock:
            cls._active_subagents = max(0, cls._active_subagents - 1)
            if cls._active_subagents == 0:
                cls._state = "idle"
            cls._last_state_change = time.time()

    @classmethod
    def status_summary(cls) -> dict:
        return {
            "state": cls._state,
            "active_subagents": cls._active_subagents,
            "compactor_running": (cls._state == "compacting"),
            "can_compact": cls.can_compact(),
            "uptime_in_state": round(time.time() - cls._last_state_change, 1),
        }
