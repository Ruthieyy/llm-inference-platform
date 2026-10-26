"""Async load generator for the OpenAI-compatible gateway.

Records client-side end-to-end latency, time to first token, and token throughput.
Numbers printed by this script are measurements, not estimates.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from typing import Any

import httpx


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


async def _one(client: httpx.AsyncClient, url: str, payload: dict) -> dict[str, Any]:
    started = time.perf_counter()
    ttft = None
    text_parts: list[str] = []
    completion_tokens = 0
    prompt_tokens = 0
    status = 0
    error = None
    try:
        async with client.stream("POST", url, json=payload) as response:
            status = response.status_code
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    parsed = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "error" in parsed and "choices" not in parsed:
                    error = parsed["error"].get("message", "error")
                    continue
                usage = parsed.get("usage") or {}
                if usage.get("completion_tokens"):
                    completion_tokens = int(usage["completion_tokens"])
                if usage.get("prompt_tokens"):
                    prompt_tokens = int(usage["prompt_tokens"])
                for choice in parsed.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if not delta:
                        delta = choice.get("text")
                    if delta:
                        text_parts.append(delta)
                        if ttft is None:
                            ttft = time.perf_counter() - started
    except Exception as exc:
        error = str(exc)
    elapsed = time.perf_counter() - started
    if completion_tokens == 0:
        completion_tokens = len(text_parts)
    return {
        "status": status,
        "error": error,
        "latency_s": elapsed,
        "ttft_s": ttft,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "text": "".join(text_parts),
    }


async def run_load(
    base_url: str,
    requests: int,
    concurrency: int,
    max_tokens: int,
    model: str,
    prompt: str,
    timeout_s: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    semaphore = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(timeout_s, connect=10.0)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:

        async def wrapped() -> dict[str, Any]:
            async with semaphore:
                return await _one(client, "/v1/chat/completions", payload)

        wall_start = time.perf_counter()
        results = await asyncio.gather(*[wrapped() for _ in range(requests)])
        wall = time.perf_counter() - wall_start

    ok = [row for row in results if row["status"] == 200 and not row["error"]]
    latencies = [row["latency_s"] for row in ok]
    ttfts = [row["ttft_s"] for row in ok if row["ttft_s"] is not None]
    completion_tokens = sum(row["completion_tokens"] for row in ok)
    summary = {
        "requests": requests,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "ok": len(ok),
        "errors": len(results) - len(ok),
        "wall_s": wall,
        "completion_tokens": completion_tokens,
        "tokens_per_s": (completion_tokens / wall) if wall > 0 else 0.0,
        "requests_per_s": (len(ok) / wall) if wall > 0 else 0.0,
        "latency_s": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "mean": statistics.fmean(latencies) if latencies else None,
        },
        "ttft_s": {
            "p50": _percentile(ttfts, 0.50),
            "p95": _percentile(ttfts, 0.95),
            "p99": _percentile(ttfts, 0.99),
            "mean": statistics.fmean(ttfts) if ttfts else None,
        },
        "sample_text": ok[0]["text"] if ok else None,
        "error_samples": [row["error"] for row in results if row["error"]][:3],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="Reply with one short sentence about batching.")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    summary = asyncio.run(
        run_load(
            args.base_url,
            args.requests,
            args.concurrency,
            args.max_tokens,
            args.model,
            args.prompt,
            args.timeout,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
