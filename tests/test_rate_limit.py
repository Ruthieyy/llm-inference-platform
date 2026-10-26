import asyncio

import httpx
from fastapi.testclient import TestClient
from tests.conftest import chat_body, make_settings
from tests.fake_engine import FakeEngine

from llm_platform.app import create_app
from llm_platform.rate_limit import RateLimiter


def test_token_bucket_enforces_burst_then_refills():
    limiter = RateLimiter(rate=0.0, burst=2)
    assert limiter.allow("a")
    assert limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.retry_after("a") >= 0


def test_global_bucket_caps_distinct_keys():
    limiter = RateLimiter(rate=0.0, burst=1)
    assert limiter.allow("one")
    assert not limiter.allow("two")


def test_http_rate_limit_returns_429():
    engine = FakeEngine()
    app = create_app(
        make_settings(rate_limit_rps=0.0, rate_limit_burst=1),
        engine=engine,
    )
    with TestClient(app) as client:
        first = client.post("/v1/chat/completions", json=chat_body())
        second = client.post("/v1/chat/completions", json=chat_body())
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["retry-after"]
    assert second.json()["error"]["code"] == "rate_limit_exceeded"
    body = client_metrics(app)
    assert "llm_rate_limited_total" in body


def client_metrics(app) -> str:
    from prometheus_client import generate_latest

    return generate_latest(app.state.registry).decode()


def test_unknown_model_is_not_rate_limited_as_success():
    async def run():
        app = create_app(make_settings(rate_limit_rps=0.0, rate_limit_burst=1), engine=FakeEngine())
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                missing = await client.post("/v1/chat/completions", json=chat_body(model="other"))
                allowed = await client.post("/v1/chat/completions", json=chat_body())
        assert missing.status_code == 404
        assert allowed.status_code == 200

    asyncio.run(run())
