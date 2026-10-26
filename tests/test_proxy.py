import asyncio
import json

import httpx
from tests.conftest import chat_body, make_settings

from llm_platform.app import create_app


def _upstream_payload(stream: bool) -> httpx.Response:
    if not stream:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-up",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        )
    first = {
        "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
    }
    done = {
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
    text = f"data: {json.dumps(first)}\n\ndata: {json.dumps(done)}\n\ndata: [DONE]\n\n"
    return httpx.Response(200, text=text)


def test_proxy_chat_and_stream():
    async def run():
        started = asyncio.Event()
        release = asyncio.Event()
        calls = {"n": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            body = json.loads(request.content.decode())
            assert request.url.path.endswith("/chat/completions")
            if calls["n"] == 1:
                started.set()
                await release.wait()
            if body.get("stream"):
                assert body["stream_options"]["include_usage"] is True
            return _upstream_payload(bool(body.get("stream")))

        transport = httpx.MockTransport(handler)
        app = create_app(
            make_settings(
                backend="vllm",
                upstream_max_inflight=1,
                max_queue_size=0,
                request_timeout_s=2,
            )
        )
        app.state.proxy.client = httpx.AsyncClient(
            base_url="http://upstream/v1",
            transport=transport,
        )
        app.state.proxy._owns_client = True
        gateway = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=gateway, base_url="http://test") as client:
                first = asyncio.create_task(client.post("/v1/chat/completions", json=chat_body()))
                await asyncio.wait_for(started.wait(), timeout=2)
                blocked = await client.post("/v1/chat/completions", json=chat_body())
                assert blocked.status_code == 503
                release.set()
                ok = await first
                assert ok.status_code == 200
                assert ok.json()["choices"][0]["message"]["content"] == "Hi"
                streamed = await client.post(
                    "/v1/chat/completions", json=chat_body(stream=True)
                )
                assert streamed.status_code == 200
                assert "Hi" in streamed.text
                assert "data: [DONE]" in streamed.text
                metrics = (await client.get("/metrics")).text
                assert "llm_upstream_inflight" in metrics
                assert 'backend="vllm"' in metrics

    asyncio.run(run())
