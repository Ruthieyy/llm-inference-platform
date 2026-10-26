"""Cohort continuous batching for the in-process CPU engine.

Idle worker: wait for one request, then up to ``max_wait_ms`` so concurrent
arrivals share a single left-padded prefill.

Busy worker: after each decode step, if a slot is free and the queue head's
prompt fits in the current KV length, prefill that prompt left-padded to the
current length and append the row. Sequences that hit a stop token or their
token cap are removed with an index select so the rest of the cohort keeps
decoding.

This is continuous batching at cohort granularity. A prompt longer than the
live KV length cannot join mid-flight (it would need a different cache length).
The head of the queue is not skipped, so admission stays FIFO. Paged-attention
joining of arbitrary prompts is what the vLLM and SGLang backends are for.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from llm_platform.engines.base import GenerationEngine
from llm_platform.errors import QueueFullError

log = logging.getLogger(__name__)


@dataclass
class WorkItem:
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    temperature: float
    top_p: float
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue
    enqueued_at: float
    cancel: threading.Event = field(default_factory=threading.Event)
    generated: list[int] = field(default_factory=list)
    done: bool = False
    first_token_at: float | None = None
    joined_prefill: bool = False


def _emit(item: WorkItem, event: dict) -> None:
    if item.done and event.get("type") != "done":
        return
    try:
        item.loop.call_soon_threadsafe(item.queue.put_nowait, event)
    except RuntimeError:
        item.cancel.set()


class CohortBatcher:
    def __init__(
        self,
        engine: GenerationEngine,
        max_batch_size: int,
        max_wait_ms: int,
        max_queue_size: int,
        metrics: Any | None = None,
    ):
        self.engine = engine
        self.max_batch_size = max(1, max_batch_size)
        self.max_wait_ms = max(0, max_wait_ms)
        self.max_queue_size = max_queue_size
        self.metrics = metrics
        self._queue: list[WorkItem] = []
        self._cv = threading.Condition()
        self._stop = False
        self._active = 0
        self._ids = itertools.count(1)
        self._thread = threading.Thread(target=self._loop, name="cohort-batcher", daemon=True)
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def stop(self) -> None:
        self._stop = True
        with self._cv:
            self._cv.notify_all()
        thread = self._thread
        if thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=5)

    def update_limits(self, max_batch_size: int | None = None, max_wait_ms: int | None = None) -> None:
        with self._cv:
            if max_batch_size is not None:
                self.max_batch_size = max(1, int(max_batch_size))
            if max_wait_ms is not None:
                self.max_wait_ms = max(0, int(max_wait_ms))
            self._cv.notify_all()

    def pending(self) -> int:
        with self._cv:
            return len(self._queue)

    def active(self) -> int:
        return self._active

    def submit(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> WorkItem:
        loop = asyncio.get_running_loop()
        item = WorkItem(
            request_id=f"req-{next(self._ids)}",
            prompt_token_ids=list(prompt_token_ids),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            loop=loop,
            queue=asyncio.Queue(),
            enqueued_at=time.perf_counter(),
        )
        with self._cv:
            if len(self._queue) >= self.max_queue_size:
                raise QueueFullError()
            self._queue.append(item)
            depth = len(self._queue)
            self._cv.notify()
        self._set_queue_depth(depth)
        return item

    def _set_queue_depth(self, depth: int) -> None:
        if self.metrics is not None:
            self.metrics.queue_depth.set(depth)

    def _set_batch_rows(self, rows: int) -> None:
        if self.metrics is not None:
            self.metrics.batch_rows.set(rows)

    def _take_cancelled_locked(self) -> list[WorkItem]:
        cancelled = [item for item in self._queue if item.cancel.is_set()]
        if cancelled:
            self._queue = [item for item in self._queue if not item.cancel.is_set()]
        return cancelled

    def _loop(self) -> None:
        while not self._stop:
            first = self._wait_first()
            if first is None:
                continue
            cohort: list[WorkItem] = []
            try:
                cohort = self._gather_cohort(first)
                if cohort:
                    self._run_cohort(cohort)
            except Exception:
                log.exception("cohort failed")
                for item in cohort:
                    self._fail(item, "internal engine error")
            finally:
                self._active = 0
                self._set_batch_rows(0)

    def _wait_first(self) -> WorkItem | None:
        while not self._stop:
            with self._cv:
                cancelled = self._take_cancelled_locked()
                item = None
                if self._queue:
                    item = self._queue.pop(0)
                self._set_queue_depth(len(self._queue))
                if item is None and not self._stop:
                    self._cv.wait(timeout=0.5)
            for cancelled_item in cancelled:
                self._finish(cancelled_item, "cancelled")
            if item is not None:
                return item
        return None

    def _gather_cohort(self, first: WorkItem) -> list[WorkItem]:
        cohort = [] if first.cancel.is_set() else [first]
        if first.cancel.is_set():
            self._finish(first, "cancelled")
        deadline = time.perf_counter() + (self.max_wait_ms / 1000.0)
        cancelled: list[WorkItem] = []
        with self._cv:
            while len(cohort) < self.max_batch_size:
                cancelled.extend(self._take_cancelled_locked())
                while self._queue and len(cohort) < self.max_batch_size:
                    cohort.append(self._queue.pop(0))
                if len(cohort) >= self.max_batch_size:
                    break
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._cv.wait(timeout=remaining)
            depth = len(self._queue)
        for cancelled_item in cancelled:
            self._finish(cancelled_item, "cancelled")
        self._set_queue_depth(depth)
        alive = []
        for item in cohort:
            if item.cancel.is_set():
                self._finish(item, "cancelled")
            else:
                alive.append(item)
        if alive and self.metrics is not None:
            self.metrics.batch_size.observe(len(alive))
        return alive

    def _note_queue_wait(self, item: WorkItem) -> None:
        if item.joined_prefill:
            return
        item.joined_prefill = True
        if self.metrics is not None:
            self.metrics.queue_wait.observe(time.perf_counter() - item.enqueued_at)

    def _run_cohort(self, cohort: list[WorkItem]) -> None:
        for item in cohort:
            self._note_queue_wait(item)
        self._active = len(cohort)
        self._set_batch_rows(len(cohort))
        try:
            state = self.engine.prefill([item.prompt_token_ids for item in cohort])
        except Exception as exc:
            log.exception("prefill failed")
            for item in cohort:
                self._fail(item, str(exc))
            return

        active = cohort
        while active and not self._stop:
            active, state = self._admit(active, state)
            self._active = len(active)
            self._set_batch_rows(len(active))
            try:
                self._sample_emit(active, state)
            except Exception as exc:
                log.exception("sample failed")
                for item in active:
                    self._fail(item, str(exc))
                return
            active, state = self._drop_finished(active, state)
            if not active:
                break
            started = time.perf_counter()
            try:
                state = self.engine.decode(state, [item.generated[-1] for item in active])
            except Exception as exc:
                log.exception("decode failed")
                for item in active:
                    self._fail(item, str(exc))
                return
            if self.metrics is not None:
                self.metrics.decode_step.observe(time.perf_counter() - started)

    def _admit(self, active: list[WorkItem], state: Any) -> tuple[list[WorkItem], Any]:
        while len(active) < self.max_batch_size:
            with self._cv:
                cancelled = self._take_cancelled_locked()
                if not self._queue:
                    depth = 0
                    head = None
                    current_len = 0
                else:
                    head = self._queue[0]
                    current_len = self.engine.seq_len(state)
                    if len(head.prompt_token_ids) > current_len:
                        depth = len(self._queue)
                        head = None
                    else:
                        self._queue.pop(0)
                        depth = len(self._queue)
            for cancelled_item in cancelled:
                self._finish(cancelled_item, "cancelled")
            self._set_queue_depth(depth)
            if head is None:
                return active, state
            self._note_queue_wait(head)
            try:
                extra = self.engine.prefill([head.prompt_token_ids], target_len=current_len)
                if self.engine.seq_len(extra) != current_len:
                    self._fail(head, "join prefill produced a different cache length")
                    continue
                state = self.engine.concat_batch(state, extra)
                active.append(head)
                if self.metrics is not None:
                    self.metrics.joins.inc()
            except Exception as exc:
                log.exception("join failed")
                self._fail(head, str(exc))
        return active, state

    def _sample_emit(self, active: list[WorkItem], state: Any) -> None:
        tokens = self.engine.sample(
            state,
            [item.temperature for item in active],
            [item.top_p for item in active],
        )
        if len(tokens) != len(active):
            raise RuntimeError("sampler returned the wrong batch width")
        now = time.perf_counter()
        for item, token in zip(active, tokens, strict=True):
            item.generated.append(int(token))
            if item.cancel.is_set():
                continue
            if item.first_token_at is None:
                item.first_token_at = now
            _emit(
                item,
                {
                    "type": "token",
                    "id": int(token),
                    "ttft_s": item.first_token_at - item.enqueued_at,
                },
            )

    def _drop_finished(
        self, active: list[WorkItem], state: Any
    ) -> tuple[list[WorkItem], Any]:
        keep: list[int] = []
        for index, item in enumerate(active):
            if item.cancel.is_set():
                self._finish(item, "cancelled")
                continue
            if item.generated and self.engine.is_stop(item.generated[-1]):
                self._finish(item, "stop")
                continue
            if len(item.generated) >= item.max_new_tokens:
                self._finish(item, "length")
                continue
            keep.append(index)
        if len(keep) == len(active):
            return active, state
        if not keep:
            return [], state
        return [active[index] for index in keep], self.engine.filter_batch(state, keep)

    def _finish(self, item: WorkItem, reason: str) -> None:
        if item.done:
            return
        ttft = None
        if item.first_token_at is not None:
            ttft = item.first_token_at - item.enqueued_at
        _emit(
            item,
            {
                "type": "done",
                "reason": reason,
                "token_ids": list(item.generated),
                "ttft_s": ttft,
            },
        )
        item.done = True

    def _fail(self, item: WorkItem, message: str) -> None:
        if item.done:
            return
        _emit(item, {"type": "error", "message": message})
        item.done = True
