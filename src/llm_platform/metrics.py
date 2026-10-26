"""Prometheus collectors for one gateway process.

Each app builds its own registry so tests can construct more than one app.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
)

BATCH_BUCKETS = (1, 2, 4, 8, 16, 32)


class GatewayMetrics:
    def __init__(self, registry: CollectorRegistry):
        self.registry = registry
        labels = ["backend", "model", "quantization"]
        self.build_info = Gauge(
            "llm_build_info",
            "Static gateway configuration. Value is always 1.",
            labels,
            registry=registry,
        )
        self.requests = Counter(
            "llm_requests_total",
            "Chat and completion requests by terminal status.",
            ["status", "stream"],
            registry=registry,
        )
        self.e2e = Histogram(
            "llm_e2e_latency_seconds",
            "End-to-end request latency measured at the gateway.",
            ["stream"],
            registry=registry,
            buckets=LATENCY_BUCKETS,
        )
        self.ttft = Histogram(
            "llm_ttft_seconds",
            "Time to first generated token, including queue wait.",
            ["stream"],
            registry=registry,
            buckets=LATENCY_BUCKETS,
        )
        self.queue_wait = Histogram(
            "llm_queue_wait_seconds",
            "Time from enqueue until the request joins a prefill.",
            registry=registry,
            buckets=LATENCY_BUCKETS,
        )
        self.tokens = Counter(
            "llm_tokens_generated_total",
            "Completion tokens returned to clients.",
            registry=registry,
        )
        self.prompt_tokens = Counter(
            "llm_prompt_tokens_total",
            "Prompt tokens accepted by the gateway.",
            registry=registry,
        )
        self.tokens_per_second = Histogram(
            "llm_request_tokens_per_second",
            "Completion tokens divided by end-to-end latency, per request.",
            registry=registry,
            buckets=(0.5, 1, 2, 5, 10, 20, 40, 80, 160),
        )
        self.batch_size = Histogram(
            "llm_batch_size",
            "Number of requests in an initial prefill cohort.",
            registry=registry,
            buckets=BATCH_BUCKETS,
        )
        self.batch_rows = Gauge(
            "llm_batch_rows",
            "Sequences currently inside the decode batch.",
            registry=registry,
        )
        self.queue_depth = Gauge(
            "llm_queue_depth",
            "Requests waiting to be admitted to the batcher or upstream.",
            registry=registry,
        )
        self.active = Gauge(
            "llm_active_requests",
            "Requests currently inside the gateway handler.",
            registry=registry,
        )
        self.rate_limited = Counter(
            "llm_rate_limited_total",
            "Requests rejected by the token bucket.",
            registry=registry,
        )
        self.rejected = Counter(
            "llm_rejected_total",
            "Requests rejected for admission reasons other than rate limit.",
            ["reason"],
            registry=registry,
        )
        self.decode_step = Histogram(
            "llm_decode_step_seconds",
            "Wall time of one batched decode forward.",
            registry=registry,
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
        )
        self.joins = Counter(
            "llm_batch_joins_total",
            "Requests admitted into an already-decoding cohort.",
            registry=registry,
        )
        self.upstream_inflight = Gauge(
            "llm_upstream_inflight",
            "In-flight proxied requests to a GPU or llama.cpp backend.",
            registry=registry,
        )

    def mark_build(self, backend: str, model: str, quantization: str) -> None:
        self.build_info.labels(backend=backend, model=model, quantization=quantization).set(1)
