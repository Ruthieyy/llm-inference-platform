import asyncio
import json
import threading

import httpx
from fastapi.testclient import TestClient

from llm_platform.app import create_app
from tests.conftest import chat_body, make_settings
from tests.fake_engine import FakeEngine


def test_chat_completion_schema_and_metrics():
    engine = FakeEngine()
    app = create_app(make_settings(), engine=engine)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=chat_body(max_tokens=3))
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "test-model"
        assert body["choices"][0]["finish_reason"] == "length"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["usage"]["completion_tokens"] == 3
        assert body["usage"]["prompt_tokens"] > 0
        assert len(body["choices"][0]["message"]["content"]) == 3

        text = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "abc",
                "max_tokens": 2,
                "temperature": 0,
            },
        )
        assert text.status_code == 200
        assert text.json()["object"] == "text_completion"
        assert text.json()["usage"]["completion_tokens"] == 2

        models = client.get("/v1/models")
        assert models.json()["data"][0]["id"] == "test-model"
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "llm_requests_total" in metrics.text
        assert "llm_ttft_seconds" in metrics.text
        assert "llm_queue_depth" in metrics.text
        assert "llm_e2e_latency_seconds" in metrics.text


def test_streaming_matches_unary_text():
    app = create_app(make_settings(), engine=FakeEngine())
    with TestClient(app) as client:
        unary = client.post("/v1/chat/completions", json=chat_body(max_tokens=5, content="stream-me"))
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json=chat_body(max_tokens=5, stream=True, content="stream-me"),
        ) as response:
            status = response.status_code
            raw = response.read().decode()
    assert status == 200
    parts = []
    saw_done = False
    saw_role = False
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: ") :]
        if data == "[DONE]":
            saw_done = True
            continue
        payload = json.loads(data)
        delta = payload["choices"][0]["delta"]
        if delta.get("role") == "assistant":
            saw_role = True
        if delta.get("content"):
            parts.append(delta["content"])
        if payload["choices"][0]["finish_reason"] == "length":
            assert payload["usage"]["completion_tokens"] == 5
    assert saw_done and saw_role
    assert "".join(parts) == unary.json()["choices"][0]["message"]["content"]


def test_validation_errors():
    app = create_app(make_settings(max_new_tokens_cap=8), engine=FakeEngine())
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json=chat_body(max_tokens=0)).status_code == 400
        assert client.post("/v1/chat/completions", json=chat_body(max_tokens=50)).status_code == 400
        assert client.post("/v1/chat/completions", json=chat_body(n=2)).status_code == 400
        empty = chat_body()
        empty["messages"] = []
        assert client.post("/v1/chat/completions", json=empty).status_code == 400
        assert client.post("/v1/chat/completions", json=chat_body(model="nope")).status_code == 404


def test_admin_updates_batch_limits():
    app = create_app(make_settings(), engine=FakeEngine())
    with TestClient(app) as client:
        updated = client.post("/admin/batching", json={"max_batch_size": 1, "max_wait_ms": 0})
        assert updated.status_code == 200
        assert updated.json() == {"max_batch_size": 1, "max_wait_ms": 0}
    hidden = create_app(make_settings(enable_admin=False), engine=FakeEngine())
    with TestClient(hidden) as client:
        assert client.post("/admin/batching", json={"max_batch_size": 2}).status_code == 404


def test_queue_full_returns_503_and_timeout_returns_504():
    async def run():
        entered = threading.Event()
        release = threading.Event()
        engine = FakeEngine(entered_prefill=entered, release_prefill=release)
        app = create_app(
            make_settings(max_batch_size=1, max_wait_ms=0, max_queue_size=1, request_timeout_s=2),
            engine=engine,
        )
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                first = asyncio.create_task(client.post("/v1/chat/completions", json=chat_body()))
                assert await asyncio.to_thread(entered.wait, 2)
                second = asyncio.create_task(client.post("/v1/chat/completions", json=chat_body()))
                for _ in range(50):
                    if app.state.batcher.pending() >= 1:
                        break
                    await asyncio.sleep(0.01)
                else:
                    raise AssertionError("second request was not queued")
                third = await client.post("/v1/chat/completions", json=chat_body())
                assert third.status_code == 503
                assert third.json()["error"]["code"] == "queue_full"
                release.set()
                assert (await first).status_code == 200
                assert (await second).status_code == 200

        slow = FakeEngine(sleep_s=1.5)
        timeout_app = create_app(
            make_settings(max_batch_size=1, max_wait_ms=0, request_timeout_s=0.2),
            engine=slow,
        )
        transport = httpx.ASGITransport(app=timeout_app)
        async with timeout_app.router.lifespan_context(timeout_app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                timed = await client.post("/v1/chat/completions", json=chat_body(max_tokens=2))
        assert timed.status_code == 504
        assert timed.json()["error"]["code"] == "timeout"

    asyncio.run(run())
