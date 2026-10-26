"""OpenAI-compatible HTTP surface."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from llm_platform.errors import QueueFullError, UpstreamError

log = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = 64
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    n: int = 1


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int | None = 64
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    n: int = 1


class BatchingUpdate(BaseModel):
    max_batch_size: int | None = Field(default=None, ge=1)
    max_wait_ms: int | None = Field(default=None, ge=0)


def register_routes(app: FastAPI) -> None:
    @app.get("/")
    def index() -> dict[str, str]:
        return {
            "service": "llm-inference-platform",
            "health": "/health",
            "metrics": "/metrics",
            "models": "/v1/models",
            "chat": "/v1/chat/completions",
            "completions": "/v1/completions",
        }

    @app.get("/health")
    @app.get("/ready")
    def health(request: Request) -> JSONResponse:
        return _health_response(request.app)

    @app.get("/metrics")
    def metrics(request: Request) -> Response:
        payload = generate_latest(request.app.state.registry)
        return Response(content=payload, media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models")
    def models(request: Request) -> dict[str, Any]:
        settings = request.app.state.settings
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": settings.backend,
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        return await _handle(
            request,
            model=body.model,
            messages=[message.model_dump() for message in body.messages],
            prompt_text=None,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            stream=body.stream,
            n=body.n,
            kind="chat",
            proxy_payload=body.model_dump(),
        )

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, request: Request):
        return await _handle(
            request,
            model=body.model,
            messages=None,
            prompt_text=body.prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            stream=body.stream,
            n=body.n,
            kind="text",
            proxy_payload=body.model_dump(),
        )

    @app.post("/admin/batching")
    def update_batching(body: BatchingUpdate, request: Request):
        settings = request.app.state.settings
        if not settings.enable_admin:
            return _error("not found", "not_found", "not_found", 404)
        batcher = request.app.state.batcher
        if batcher is None:
            return _error("in-process batcher is not active", "conflict", "batcher_unavailable", 409)
        batcher.update_limits(body.max_batch_size, body.max_wait_ms)
        return {
            "max_batch_size": batcher.max_batch_size,
            "max_wait_ms": batcher.max_wait_ms,
        }


def _health_response(app: FastAPI) -> JSONResponse:
    settings = app.state.settings
    if app.state.load_error:
        status = "error"
        code = 500
    elif app.state.batcher is None and app.state.proxy is None:
        status = "loading"
        code = 503
    else:
        status = "ok"
        code = 200
    depth = 0
    if app.state.batcher is not None:
        depth = app.state.batcher.pending()
    elif app.state.proxy is not None:
        depth = app.state.proxy.admission.waiting
    body = {
        "status": status,
        "backend": settings.backend,
        "model": settings.model_id,
        "quantization": settings.quantization,
        "queue_depth": depth,
        "batching": app.state.batcher is not None,
        "load_error": app.state.load_error,
    }
    return JSONResponse(status_code=code, content=body)


def _error(message: str, err_type: str, code: str, status: int, headers: dict | None = None):
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "param": None, "code": code}},
        headers=headers,
    )


def _api_key(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        token = header.split(None, 1)[1].strip()
        return token or "anonymous"
    return "anonymous"


def _validate(
    request: Request,
    model: str,
    messages: list[dict[str, str]] | None,
    prompt_text: str | None,
    max_tokens: int | None,
    temperature: float,
    top_p: float,
    n: int,
) -> JSONResponse | int:
    settings = request.app.state.settings
    if model != settings.model_id:
        return _error(f"model {model!r} is not served", "invalid_request_error", "model_not_found", 404)
    if n != 1:
        return _error("only n=1 is supported", "invalid_request_error", "unsupported_n", 400)
    if messages is not None and not messages:
        return _error("messages must not be empty", "invalid_request_error", "empty_messages", 400)
    if prompt_text is not None and prompt_text == "":
        return _error("prompt must not be empty", "invalid_request_error", "empty_prompt", 400)
    if temperature < 0 or temperature > 2:
        return _error("temperature must be between 0 and 2", "invalid_request_error", "bad_temperature", 400)
    if top_p <= 0 or top_p > 1:
        return _error("top_p must be in (0, 1]", "invalid_request_error", "bad_top_p", 400)
    limit = 64 if max_tokens is None else max_tokens
    if limit < 1:
        return _error("max_tokens must be >= 1", "invalid_request_error", "bad_max_tokens", 400)
    if limit > settings.max_new_tokens_cap:
        return _error(
            f"max_tokens exceeds cap of {settings.max_new_tokens_cap}",
            "invalid_request_error",
            "max_tokens_cap",
            400,
        )
    return limit


def _public_reason(reason: str) -> str:
    if reason == "stop":
        return "stop"
    return "length"


async def _handle(
    request: Request,
    model: str,
    messages: list[dict[str, str]] | None,
    prompt_text: str | None,
    max_tokens: int | None,
    temperature: float,
    top_p: float,
    stream: bool,
    n: int,
    kind: str,
    proxy_payload: dict,
):
    app = request.app
    metrics = app.state.metrics
    validated = _validate(
        request, model, messages, prompt_text, max_tokens, temperature, top_p, n
    )
    if not isinstance(validated, int):
        return validated
    max_tokens = validated

    if not app.state.limiter.allow(_api_key(request)):
        metrics.rate_limited.inc()
        metrics.requests.labels(status="rate_limited", stream=str(stream).lower()).inc()
        retry = f"{app.state.limiter.retry_after(_api_key(request)):.2f}"
        return _error(
            "rate limit exceeded",
            "rate_limit_error",
            "rate_limit_exceeded",
            429,
            headers={"Retry-After": retry},
        )

    if app.state.load_error:
        return _error(app.state.load_error, "server_error", "model_load_failed", 500)
    if app.state.batcher is None and app.state.proxy is None:
        metrics.rejected.labels(reason="loading").inc()
        return _error("model is still loading", "server_error", "loading", 503)

    metrics.active.inc()
    handed = {"on": False}
    try:
        if app.state.proxy is not None:
            return await _proxy(
                request, proxy_payload, stream, kind, max_tokens, temperature, top_p, handed
            )
        return await _inprocess(
            request,
            messages,
            prompt_text,
            max_tokens,
            temperature,
            top_p,
            stream,
            kind,
            handed,
        )
    finally:
        if not handed["on"]:
            metrics.active.dec()


async def _inprocess(
    request: Request,
    messages: list[dict[str, str]] | None,
    prompt_text: str | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stream: bool,
    kind: str,
    handed: dict,
):
    app = request.app
    batcher = app.state.batcher
    engine = batcher.engine
    settings = app.state.settings
    metrics = app.state.metrics
    try:
        if messages is not None:
            prompt_ids = engine.encode_chat(messages)
        else:
            prompt_ids = engine.encode_text(prompt_text or "")
    except Exception as exc:
        metrics.requests.labels(status="error", stream=str(stream).lower()).inc()
        return _error(f"failed to tokenize prompt: {exc}", "invalid_request_error", "tokenize_error", 400)
    if len(prompt_ids) > settings.max_prompt_tokens:
        metrics.rejected.labels(reason="prompt_too_long").inc()
        return _error(
            f"prompt has {len(prompt_ids)} tokens; cap is {settings.max_prompt_tokens}",
            "invalid_request_error",
            "prompt_too_long",
            400,
        )
    try:
        item = batcher.submit(prompt_ids, max_tokens, temperature, top_p)
    except QueueFullError:
        metrics.rejected.labels(reason="queue_full").inc()
        metrics.requests.labels(status="rejected", stream=str(stream).lower()).inc()
        return _error("server is at capacity", "server_error", "queue_full", 503)

    started = time.perf_counter()
    deadline = started + settings.request_timeout_s
    if stream:
        generator = _stream_local(
            request, item, engine, prompt_ids, kind, started, deadline
        )
        handed["on"] = True
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    try:
        visible, reason, ttft = await _collect_local(item, engine, deadline)
    except TimeoutError:
        item.cancel.set()
        metrics.rejected.labels(reason="timeout").inc()
        metrics.requests.labels(status="timeout", stream="false").inc()
        _log_request("timeout", False, len(prompt_ids), 0, started)
        return _error("request timed out", "timeout", "timeout", 504)
    except RuntimeError as exc:
        metrics.requests.labels(status="error", stream="false").inc()
        return _error(str(exc), "server_error", "engine_error", 500)
    text = engine.decode_tokens(visible)
    _record_ok(metrics, False, started, ttft, len(prompt_ids), len(visible))
    _log_request("ok", False, len(prompt_ids), len(visible), started)
    if kind == "chat":
        return _chat_json(settings.model_id, item.request_id, text, reason, len(prompt_ids), len(visible))
    return _text_json(settings.model_id, item.request_id, text, reason, len(prompt_ids), len(visible))


async def _collect_local(item, engine, deadline: float) -> tuple[list[int], str, float | None]:
    visible: list[int] = []
    ttft = None
    reason = "length"
    while True:
        event = await _next_event(item, deadline)
        if event["type"] == "token":
            ttft = ttft if ttft is not None else event.get("ttft_s")
            if engine.is_stop(event["id"]):
                continue
            visible.append(event["id"])
        elif event["type"] == "done":
            reason = _public_reason(event["reason"])
            ttft = ttft if ttft is not None else event.get("ttft_s")
            break
        elif event["type"] == "error":
            raise RuntimeError(event["message"])
        else:
            raise RuntimeError("unknown batcher event")
    return visible, reason, ttft


async def _stream_local(request, item, engine, prompt_ids, kind, started, deadline) -> AsyncIterator[str]:
    metrics = request.app.state.metrics
    model = request.app.state.settings.model_id
    created = int(time.time())
    completion_id = _id_for(kind, item.request_id)
    visible: list[int] = []
    previous = ""
    ttft = None
    reason = "length"
    try:
        if kind == "chat":
            yield _sse(
                _chunk(completion_id, created, model, {"role": "assistant"}, None, chat=True)
            )
        while True:
            try:
                event = await _next_event(item, deadline)
            except TimeoutError:
                item.cancel.set()
                metrics.rejected.labels(reason="timeout").inc()
                metrics.requests.labels(status="timeout", stream="true").inc()
                _log_request("timeout", True, len(prompt_ids), len(visible), started)
                yield _sse(
                    {
                        "error": {
                            "message": "request timed out",
                            "type": "timeout",
                            "code": "timeout",
                        }
                    }
                )
                return
            if event["type"] == "token":
                ttft = ttft if ttft is not None else event.get("ttft_s")
                if engine.is_stop(event["id"]):
                    continue
                visible.append(event["id"])
                text = engine.decode_tokens(visible)
                delta = text[len(previous) :]
                previous = text
                if not delta:
                    continue
                if kind == "chat":
                    yield _sse(_chunk(completion_id, created, model, {"content": delta}, None, chat=True))
                else:
                    yield _sse(_text_chunk(completion_id, created, model, delta, None))
            elif event["type"] == "done":
                reason = _public_reason(event["reason"])
                ttft = ttft if ttft is not None else event.get("ttft_s")
                break
            elif event["type"] == "error":
                metrics.requests.labels(status="error", stream="true").inc()
                yield _sse(
                    {"error": {"message": event["message"], "type": "server_error", "code": "engine_error"}}
                )
                return
        usage = _usage(len(prompt_ids), len(visible))
        if kind == "chat":
            final = _chunk(completion_id, created, model, {}, reason, chat=True)
            final["usage"] = usage
            yield _sse(final)
        else:
            final = _text_chunk(completion_id, created, model, "", reason)
            final["usage"] = usage
            yield _sse(final)
        yield "data: [DONE]\n\n"
        _record_ok(metrics, True, started, ttft, len(prompt_ids), len(visible))
        _log_request("ok", True, len(prompt_ids), len(visible), started)
    finally:
        if not item.done:
            item.cancel.set()
        metrics.active.dec()


async def _proxy(
    request,
    payload: dict,
    stream: bool,
    kind: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    handed: dict,
):
    proxy = request.app.state.proxy
    metrics = request.app.state.metrics
    path = "/chat/completions" if kind == "chat" else "/completions"
    body = dict(payload)
    body["max_tokens"] = max_tokens
    body["temperature"] = temperature
    body["top_p"] = top_p
    body["stream"] = stream
    if stream:
        body["stream_options"] = {"include_usage": True}
        generator = _stream_proxy(request, path, body)
        handed["on"] = True
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    started = time.perf_counter()
    try:
        status, upstream = await proxy.post_json(path, body)
    except QueueFullError:
        metrics.rejected.labels(reason="queue_full").inc()
        metrics.requests.labels(status="rejected", stream="false").inc()
        return _error("server is at capacity", "server_error", "queue_full", 503)
    except TimeoutError:
        metrics.rejected.labels(reason="timeout").inc()
        metrics.requests.labels(status="timeout", stream="false").inc()
        return _error("timed out waiting for upstream capacity", "timeout", "timeout", 504)
    except Exception as exc:
        metrics.requests.labels(status="error", stream="false").inc()
        return _error(f"upstream request failed: {exc}", "server_error", "upstream_error", 502)
    if status >= 400:
        metrics.requests.labels(status="error", stream="false").inc()
        return JSONResponse(status_code=status, content=upstream)
    usage = upstream.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    _record_ok(metrics, False, started, None, prompt_tokens, completion_tokens)
    _log_request("ok", False, prompt_tokens, completion_tokens, started)
    return JSONResponse(status_code=200, content=upstream)


async def _stream_proxy(request, path: str, body: dict) -> AsyncIterator[str]:
    proxy = request.app.state.proxy
    metrics = request.app.state.metrics
    started = time.perf_counter()
    ttft = None
    prompt_tokens = 0
    completion_tokens = 0
    content_chunks = 0
    try:
        try:
            async for line in proxy.iter_sse(path, body):
                if line.startswith("data:"):
                    raw = line[5:].strip()
                    if raw and raw != "[DONE]":
                        parsed = _parse_json(raw)
                        if parsed is not None:
                            ttft, content_chunks, prompt_tokens, completion_tokens = _note_upstream_chunk(
                                parsed, started, ttft, content_chunks, prompt_tokens, completion_tokens
                            )
                yield line + "\n"
        except QueueFullError:
            metrics.rejected.labels(reason="queue_full").inc()
            metrics.requests.labels(status="rejected", stream="true").inc()
            yield _sse({"error": {"message": "server is at capacity", "type": "server_error", "code": "queue_full"}})
            return
        except TimeoutError:
            metrics.rejected.labels(reason="timeout").inc()
            metrics.requests.labels(status="timeout", stream="true").inc()
            yield _sse({"error": {"message": "timed out waiting for capacity", "type": "timeout", "code": "timeout"}})
            return
        except UpstreamError as exc:
            metrics.requests.labels(status="error", stream="true").inc()
            yield _sse({"error": {"message": exc.message, "type": "upstream_error", "code": "upstream_error"}})
            return
        if completion_tokens == 0:
            completion_tokens = content_chunks
        _record_ok(metrics, True, started, ttft, prompt_tokens, completion_tokens)
        _log_request("ok", True, prompt_tokens, completion_tokens, started)
    finally:
        metrics.active.dec()


def _note_upstream_chunk(parsed, started, ttft, content_chunks, prompt_tokens, completion_tokens):
    usage = parsed.get("usage") or {}
    if usage.get("prompt_tokens"):
        prompt_tokens = int(usage["prompt_tokens"])
    if usage.get("completion_tokens"):
        completion_tokens = int(usage["completion_tokens"])
    for choice in parsed.get("choices") or []:
        delta = choice.get("delta") or {}
        text = delta.get("content")
        if text is None:
            text = choice.get("text")
        if text:
            content_chunks += 1
            if ttft is None:
                ttft = time.perf_counter() - started
    return ttft, content_chunks, prompt_tokens, completion_tokens


def _parse_json(raw: str) -> dict | None:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


async def _next_event(item, deadline: float) -> dict:
    timeout = deadline - time.perf_counter()
    if timeout <= 0:
        raise TimeoutError()
    import asyncio

    return await asyncio.wait_for(item.queue.get(), timeout=timeout)


def _record_ok(metrics, stream: bool, started: float, ttft: float | None, prompt_tokens: int, completion_tokens: int):
    label = "true" if stream else "false"
    elapsed = max(time.perf_counter() - started, 1e-6)
    metrics.requests.labels(status="ok", stream=label).inc()
    metrics.e2e.labels(stream=label).observe(elapsed)
    if ttft is not None:
        metrics.ttft.labels(stream=label).observe(ttft)
    if prompt_tokens:
        metrics.prompt_tokens.inc(prompt_tokens)
    if completion_tokens:
        metrics.tokens.inc(completion_tokens)
        metrics.tokens_per_second.observe(completion_tokens / elapsed)


def _log_request(status: str, stream: bool, prompt_tokens: int, completion_tokens: int, started: float) -> None:
    log.info(
        "status=%s stream=%s prompt_tokens=%s completion_tokens=%s latency_s=%.3f",
        status,
        stream,
        prompt_tokens,
        completion_tokens,
        time.perf_counter() - started,
    )


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _id_for(kind: str, request_id: str) -> str:
    prefix = "chatcmpl" if kind == "chat" else "cmpl"
    return f"{prefix}-{request_id}"


def _chat_json(model, request_id, text, reason, prompt_tokens, completion_tokens):
    return {
        "id": _id_for("chat", request_id),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": reason,
            }
        ],
        "usage": _usage(prompt_tokens, completion_tokens),
    }


def _text_json(model, request_id, text, reason, prompt_tokens, completion_tokens):
    return {
        "id": _id_for("text", request_id),
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": reason}],
        "usage": _usage(prompt_tokens, completion_tokens),
    }


def _chunk(completion_id, created, model, delta, finish_reason, chat: bool) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _text_chunk(completion_id, created, model, text, finish_reason) -> dict:
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
    }


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
