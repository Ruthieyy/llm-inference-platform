"""FastAPI application factory.

The transformers backend loads weights on startup without blocking import of
this module. Tests inject an engine and never touch Hugging Face.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import CollectorRegistry

from llm_platform import __version__
from llm_platform.api import register_routes
from llm_platform.batching import CohortBatcher
from llm_platform.config import Settings
from llm_platform.engines.base import GenerationEngine
from llm_platform.engines.proxy import OpenAIProxy
from llm_platform.metrics import GatewayMetrics
from llm_platform.rate_limit import RateLimiter

log = logging.getLogger(__name__)

_PROXIED = {"vllm", "sglang", "llamacpp"}


def _attach_batcher(app: FastAPI, engine: GenerationEngine) -> None:
    settings: Settings = app.state.settings
    batcher = CohortBatcher(
        engine,
        max_batch_size=settings.max_batch_size,
        max_wait_ms=settings.max_wait_ms,
        max_queue_size=settings.max_queue_size,
        metrics=app.state.metrics,
    )
    batcher.start()
    app.state.batcher = batcher


async def _load_engine(app: FastAPI) -> None:
    try:
        engine = await asyncio.to_thread(app.state.engine_loader)
        _attach_batcher(app, engine)
        log.info("in-process engine is ready")
    except Exception as exc:
        log.exception("failed to load in-process engine")
        app.state.load_error = str(exc)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    task = None
    if app.state.engine_loader is not None and app.state.batcher is None:
        task = asyncio.create_task(_load_engine(app))
    try:
        yield
    finally:
        if task is not None and not task.done():
            task.cancel()
        batcher = app.state.batcher
        if batcher is not None:
            await asyncio.to_thread(batcher.stop)
        proxy = app.state.proxy
        if proxy is not None:
            await proxy.aclose()


def create_app(
    settings: Settings | None = None,
    engine: GenerationEngine | None = None,
) -> FastAPI:
    settings = settings or Settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    registry = CollectorRegistry()
    metrics = GatewayMetrics(registry)
    metrics.mark_build(settings.backend, settings.model_id, settings.quantization)

    app = FastAPI(title="LLM Inference Platform", version=__version__, lifespan=_lifespan)
    app.state.settings = settings
    app.state.metrics = metrics
    app.state.registry = registry
    app.state.limiter = RateLimiter(settings.rate_limit_rps, settings.rate_limit_burst)
    app.state.batcher = None
    app.state.proxy = None
    app.state.load_error = None
    app.state.engine_loader = None

    backend = settings.backend.lower()
    if engine is not None:
        _attach_batcher(app, engine)
    elif backend == "transformers":
        from llm_platform.engines.transformers_engine import load_transformers_engine

        app.state.engine_loader = lambda: load_transformers_engine(settings)
    elif backend in _PROXIED:
        app.state.proxy = OpenAIProxy(settings, metrics)
    else:
        raise RuntimeError(
            f"unknown backend {settings.backend!r}; expected transformers, vllm, sglang, or llamacpp"
        )

    register_routes(app)
    return app
