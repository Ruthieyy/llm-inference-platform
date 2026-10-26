"""OpenAI-compatible HTTP proxy for vLLM, SGLang, and llama.cpp server.

Those servers already batch. The gateway keeps rate limiting, admission,
timeouts, and its own latency metrics in front of them.
"""

from __future__ import annotations

from typing import AsyncIterator

import httpx

from llm_platform.admission import AsyncAdmission
from llm_platform.config import Settings
from llm_platform.errors import UpstreamError
from llm_platform.metrics import GatewayMetrics


class OpenAIProxy:
    def __init__(
        self,
        settings: Settings,
        metrics: GatewayMetrics,
        client: httpx.AsyncClient | None = None,
    ):
        if not settings.upstream_base_url:
            raise ValueError("upstream_base_url is required for a proxied backend")
        self.settings = settings
        self.metrics = metrics
        self._owns_client = client is None
        if client is None:
            timeout = httpx.Timeout(settings.request_timeout_s, connect=10.0)
            client = httpx.AsyncClient(
                base_url=settings.upstream_base_url.rstrip("/"),
                timeout=timeout,
            )
        self.client = client
        self.admission = AsyncAdmission(settings.upstream_max_inflight, settings.max_queue_size)

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.upstream_api_key}"}

    def _sync_gauges(self) -> None:
        self.metrics.queue_depth.set(self.admission.waiting)
        self.metrics.upstream_inflight.set(self.admission.inflight)

    async def post_json(self, path: str, payload: dict) -> tuple[int, dict]:
        await self.admission.acquire(self.settings.request_timeout_s)
        self._sync_gauges()
        try:
            response = await self.client.post(path, json=payload, headers=self._headers())
            try:
                body = response.json()
            except Exception:
                body = {"error": {"message": response.text[:500], "type": "upstream_error"}}
            return response.status_code, body
        finally:
            await self.admission.release()
            self._sync_gauges()

    async def iter_sse(self, path: str, payload: dict) -> AsyncIterator[str]:
        await self.admission.acquire(self.settings.request_timeout_s)
        self._sync_gauges()
        try:
            async with self.client.stream(
                "POST", path, json=payload, headers=self._headers()
            ) as response:
                if response.status_code >= 400:
                    raw = await response.aread()
                    raise UpstreamError(response.status_code, raw.decode("utf-8", "replace")[:500])
                async for line in response.aiter_lines():
                    yield line
        finally:
            await self.admission.release()
            self._sync_gauges()
