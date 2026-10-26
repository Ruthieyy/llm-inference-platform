"""Bounded inflight + waiting-room admission for proxied backends."""

from __future__ import annotations

import asyncio
import time

from llm_platform.errors import QueueFullError


class AsyncAdmission:
    """FIFO gate in front of an upstream server.

    ``max_inflight`` requests may be inside the upstream call. Additional
    requests wait, up to ``max_waiting``. Past that the caller gets
    ``QueueFullError`` immediately (backpressure) instead of growing without bound.
    """

    def __init__(self, max_inflight: int, max_waiting: int):
        if max_inflight < 1:
            raise ValueError("max_inflight must be >= 1")
        if max_waiting < 0:
            raise ValueError("max_waiting must be >= 0")
        self.max_inflight = max_inflight
        self.max_waiting = max_waiting
        self.inflight = 0
        self.waiting = 0
        self._cond = asyncio.Condition()

    async def acquire(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        async with self._cond:
            if self.inflight >= self.max_inflight and self.waiting >= self.max_waiting:
                raise QueueFullError()
            self.waiting += 1
            try:
                while self.inflight >= self.max_inflight:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError()
                    try:
                        await asyncio.wait_for(self._cond.wait(), timeout=remaining)
                    except TimeoutError as exc:
                        raise TimeoutError() from exc
                self.inflight += 1
            finally:
                self.waiting -= 1

    async def release(self) -> None:
        async with self._cond:
            self.inflight = max(0, self.inflight - 1)
            self._cond.notify(1)
