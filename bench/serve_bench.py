"""Start the CPU gateway once and compare max_batch_size=1 against a batched cohort.

Requires ENABLE_ADMIN so the batch size can change without reloading weights.
Writes bench/results/batching.json. Every figure comes from bench/load_test.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from bench.load_test import run_load  # noqa: E402


def _wait_ready(base: str, timeout_s: float = 600) -> None:
    deadline = time.time() + timeout_s
    last = "not started"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as response:
                body = json.loads(response.read().decode())
            if body.get("status") == "ok":
                return
            last = json.dumps(body)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = str(exc)
        time.sleep(2)
    raise RuntimeError(f"gateway did not become ready: {last}")


def _set_batch(base: str, max_batch_size: int, max_wait_ms: int) -> dict:
    response = httpx.post(
        base + "/admin/batching",
        json={"max_batch_size": max_batch_size, "max_wait_ms": max_wait_ms},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    import asyncio

    model = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    quant = os.environ.get("QUANTIZATION", "fp32")
    port = os.environ.get("BENCH_PORT", "8000")
    base = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "MODEL_ID": model,
            "BACKEND": "transformers",
            "QUANTIZATION": quant,
            "MAX_BATCH_SIZE": "1",
            "MAX_WAIT_MS": "0",
            "MAX_QUEUE_SIZE": "64",
            "REQUEST_TIMEOUT_S": "300",
            "RATE_LIMIT_RPS": "1000",
            "RATE_LIMIT_BURST": "1000",
            "ENABLE_ADMIN": "true",
            "TORCH_THREADS": os.environ.get("TORCH_THREADS", "4"),
            "LOG_LEVEL": "info",
            "HF_HOME": os.environ.get("HF_HOME", str(ROOT / ".cache" / "huggingface")),
        }
    )
    python = os.environ.get("BENCH_PYTHON", sys.executable)
    proc = subprocess.Popen(
        [
            python,
            "-m",
            "uvicorn",
            "llm_platform.asgi:app",
            "--host",
            "127.0.0.1",
            "--port",
            port,
        ],
        cwd=str(ROOT),
        env=env,
    )
    try:
        _wait_ready(base)
        prompt = "In one short sentence, say what a token is."
        max_tokens = int(os.environ.get("BENCH_MAX_TOKENS", "16"))
        # Warm the kernels and the allocator. This request is not part of the report.
        asyncio.run(
            run_load(base, requests=1, concurrency=1, max_tokens=8, model=model, prompt=prompt, timeout_s=300)
        )
        scenarios = []
        for name, batch, wait_ms, requests, concurrency in (
            ("unbatched", 1, 0, 4, 1),
            ("batched", 4, 30, 8, 4),
        ):
            applied = _set_batch(base, batch, wait_ms)
            summary = asyncio.run(
                run_load(
                    base,
                    requests=requests,
                    concurrency=concurrency,
                    max_tokens=max_tokens,
                    model=model,
                    prompt=prompt,
                    timeout_s=300,
                )
            )
            summary["name"] = name
            summary["batch_config"] = applied
            scenarios.append(summary)
            print(json.dumps({"name": name, "tokens_per_s": summary["tokens_per_s"], "ok": summary["ok"]}))
        report = {
            "model": model,
            "quantization": quant,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "scenarios": scenarios,
        }
        out = ROOT / "bench" / "results" / "batching.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {out}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
