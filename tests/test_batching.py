import asyncio

from tests.fake_engine import FakeEngine

from llm_platform.batching import CohortBatcher


def _run(coro):
    return asyncio.run(coro)


async def _collect(item):
    tokens = []
    reason = None
    while True:
        event = await item.queue.get()
        if event["type"] == "token":
            tokens.append(event["id"])
        elif event["type"] == "done":
            reason = event["reason"]
            break
        elif event["type"] == "error":
            raise RuntimeError(event["message"])
    return tokens, reason


def test_concurrent_requests_share_a_prefill():
    async def scenario():
        engine = FakeEngine()
        batcher = CohortBatcher(engine, max_batch_size=4, max_wait_ms=200, max_queue_size=8)
        batcher.start()
        try:
            items = [
                batcher.submit([1, 2, 3, 4], max_new_tokens=3, temperature=0, top_p=1)
                for _ in range(4)
            ]
            results = await asyncio.gather(*[_collect(item) for item in items])
        finally:
            batcher.stop()
        assert engine.prefill_widths[0] == 4
        assert all(len(tokens) == 3 and reason == "length" for tokens, reason in results)
        return engine

    _run(scenario())


def test_finished_row_leaves_and_waiting_request_joins():
    async def scenario():
        engine = FakeEngine()
        batcher = CohortBatcher(engine, max_batch_size=2, max_wait_ms=200, max_queue_size=8)
        batcher.start()
        try:
            long_req = batcher.submit([1, 2, 3, 4], max_new_tokens=4, temperature=0, top_p=1)
            short_req = batcher.submit([1, 2, 3, 4], max_new_tokens=1, temperature=0, top_p=1)
            joiner = batcher.submit([5, 6, 7], max_new_tokens=2, temperature=0, top_p=1)
            long_tokens, long_reason = await _collect(long_req)
            short_tokens, short_reason = await _collect(short_req)
            join_tokens, join_reason = await _collect(joiner)
        finally:
            batcher.stop()
        assert short_reason == "length" and len(short_tokens) == 1
        assert long_reason == "length" and len(long_tokens) == 4
        assert join_reason == "length" and len(join_tokens) == 2
        assert engine.prefill_widths[0] == 2
        assert 1 in engine.prefill_widths[1:]
        assert engine.concat_calls >= 1

    _run(scenario())


def test_long_prompt_does_not_skip_the_queue_head():
    async def scenario():
        engine = FakeEngine()
        batcher = CohortBatcher(engine, max_batch_size=1, max_wait_ms=20, max_queue_size=8)
        batcher.start()
        try:
            first = batcher.submit([1, 2, 3, 4], max_new_tokens=2, temperature=0, top_p=1)
            # Longer than the live cache (prompt 4 + 2 generated = 6) so it cannot join.
            blocked = batcher.submit(list(range(1, 40)), max_new_tokens=1, temperature=0, top_p=1)
            behind = batcher.submit([8, 9], max_new_tokens=1, temperature=0, top_p=1)
            await _collect(first)
            await _collect(blocked)
            await _collect(behind)
        finally:
            batcher.stop()
        assert engine.concat_calls == 0
        # Three separate prefills, FIFO: short, then the long head, then the one behind it.
        assert engine.prefill_widths == [1, 1, 1]

    _run(scenario())


def test_stop_token_ends_generation_early():
    async def scenario():
        engine = FakeEngine(eos_after=2)
        batcher = CohortBatcher(engine, max_batch_size=1, max_wait_ms=0, max_queue_size=4)
        batcher.start()
        try:
            item = batcher.submit([1, 2, 3], max_new_tokens=10, temperature=0, top_p=1)
            tokens, reason = await _collect(item)
        finally:
            batcher.stop()
        assert reason == "stop"
        assert tokens[-1] == engine.eos_token_id
        assert len(tokens) == 2

    _run(scenario())
