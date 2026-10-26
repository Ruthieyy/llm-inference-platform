"""Compare FP32, FP16, dynamic INT8, and weight-only INT4 on CPU.

One mode is resident at a time. The FP32 run also checks that the cohort
engine's greedy tokens match ``model.generate``. Results are written to
bench/results/quant.json with the raw completion text.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROMPTS = [
    {
        "name": "addition",
        "messages": [{"role": "user", "content": "What is 17 + 25? Reply with digits only."}],
        "expect_substr": "42",
    },
    {
        "name": "planet",
        "messages": [{"role": "user", "content": "Name the planet humans live on. Reply with one word."}],
        "expect_substr": "earth",
    },
]


def _host() -> dict:
    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    mem_gb = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_gb = round(int(line.split()[1]) / 1024 / 1024, 2)
                break
    except OSError:
        pass
    return {
        "cpu": cpu,
        "cores": os.cpu_count(),
        "mem_total_gb": mem_gb,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _jaccard(left: str, right: str) -> float:
    a = set(left.lower().split())
    b = set(right.lower().split())
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _run_greedy(engine, messages: list[dict], max_new_tokens: int) -> dict:
    prompt_ids = engine.encode_chat(messages)
    t0 = time.perf_counter()
    state = engine.prefill([prompt_ids])
    prefill_s = time.perf_counter() - t0
    tokens: list[int] = []
    first_token_s = None
    decode_s = 0.0
    for _ in range(max_new_tokens):
        token = engine.sample(state, [0.0], [1.0])[0]
        now = time.perf_counter()
        if first_token_s is None:
            first_token_s = now - t0
        tokens.append(token)
        if engine.is_stop(token):
            break
        step = time.perf_counter()
        state = engine.decode(state, [token])
        decode_s += time.perf_counter() - step
    elapsed = time.perf_counter() - t0
    visible = []
    for token in tokens:
        if engine.is_stop(token):
            break
        visible.append(token)
    text = engine.decode_tokens(visible)
    produced = len(visible)
    return {
        "text": text,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": produced,
        "latency_s": elapsed,
        "ttft_s": first_token_s,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "tokens_per_s": (produced / elapsed) if elapsed > 0 else 0.0,
    }


def _verify_against_generate(engine, max_new_tokens: int = 8) -> dict:
    import torch

    prompt = "The capital of France is"
    ids = engine.encode_text(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long)
    generated = engine.model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=engine.pad_token_id,
        eos_token_id=list(engine.stop_token_ids) or engine.eos_token_id,
    )
    expected = [int(token) for token in generated[0, len(ids) :].tolist()]
    state = engine.prefill([ids])
    got: list[int] = []
    for _ in range(max_new_tokens):
        token = engine.sample(state, [0.0], [1.0])[0]
        got.append(token)
        if engine.is_stop(token):
            break
        state = engine.decode(state, [token])
    match_len = min(len(got), len(expected))
    matches = got[:match_len] == expected[:match_len]
    return {
        "prompt": prompt,
        "match": matches,
        "engine_tokens": got,
        "generate_tokens": expected,
        "engine_text": engine.decode_tokens([t for t in got if not engine.is_stop(t)]),
        "generate_text": engine.decode_tokens([t for t in expected if not engine.is_stop(t)]),
    }


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _bench_mode(settings, mode: str, max_new_tokens: int, repeats: int) -> dict:
    from llm_platform.engines.transformers_engine import load_transformers_engine
    from llm_platform.quantization import resident_nbytes

    tuned = settings.model_copy(update={"quantization": mode})
    started = time.perf_counter()
    try:
        engine = load_transformers_engine(tuned)
    except Exception as exc:
        return {"mode": mode, "status": "load_failed", "error": f"{type(exc).__name__}: {exc}"}
    load_s = time.perf_counter() - started
    try:
        nbytes = resident_nbytes(engine.model)
        record: dict = {
            "mode": mode,
            "status": "ok",
            "resident_bytes": nbytes,
            "resident_mb": round(nbytes / (1024 * 1024), 2),
            "load_s": round(load_s, 3),
        }
        if mode == "fp32":
            record["generate_check"] = _verify_against_generate(engine)
        latency_prompt = PROMPTS[0]["messages"]
        try:
            _run_greedy(engine, latency_prompt, max_new_tokens=4)
            runs = [_run_greedy(engine, latency_prompt, max_new_tokens) for _ in range(repeats)]
        except Exception as exc:
            record["status"] = "forward_failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            return record
        record["latency_runs"] = runs
        record["latency_s_median"] = _median([row["latency_s"] for row in runs])
        record["ttft_s_median"] = _median([row["ttft_s"] for row in runs if row["ttft_s"] is not None])
        record["tokens_per_s_median"] = _median([row["tokens_per_s"] for row in runs])
        qualities = []
        for prompt in PROMPTS:
            try:
                generated = _run_greedy(engine, prompt["messages"], max_new_tokens)
            except Exception as exc:
                qualities.append({"name": prompt["name"], "error": str(exc)})
                continue
            text = generated["text"]
            qualities.append(
                {
                    "name": prompt["name"],
                    "text": text,
                    "expect_substr": prompt["expect_substr"],
                    "substr_hit": prompt["expect_substr"].lower() in text.lower(),
                    "completion_tokens": generated["completion_tokens"],
                }
            )
        record["quality"] = qualities
        return record
    finally:
        del engine
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def main() -> None:
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from llm_platform.config import Settings

    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    threads = int(os.environ.get("TORCH_THREADS", "4"))
    max_new_tokens = int(os.environ.get("BENCH_MAX_TOKENS", "16"))
    repeats = int(os.environ.get("BENCH_REPEATS", "3"))
    settings = Settings(model_id=model_id, quantization="fp32", torch_threads=threads, log_level="warning")
    modes = ["fp32", "fp16", "int8", "int4"]
    results = []
    for mode in modes:
        print(f"=== {mode} ===", flush=True)
        record = _bench_mode(settings, mode, max_new_tokens, repeats)
        results.append(record)
        print(json.dumps({k: record[k] for k in record if k != "latency_runs"}), flush=True)
    import torch

    report = {
        "model": model_id,
        "device": "cpu",
        "max_new_tokens": max_new_tokens,
        "repeats": repeats,
        "torch": torch.__version__,
        "host": _host(),
        "modes": results,
        "notes": [
            "Latency is a single greedy request, median of the timed repeats after one warmup of 4 tokens.",
            "INT8 is torch dynamic quantization of Linear layers.",
            "INT4 is group-wise weight-only packing with dequantize-on-forward and lm_head left in fp32.",
            "FP16 is a full cast. On CPU it may be unsupported; that failure is recorded rather than imputed.",
            "Quality substrings are a capability check. Agreement with FP32 text is the quantization check.",
        ],
    }
    fp32 = next((row for row in results if row["mode"] == "fp32" and row.get("status") == "ok"), None)
    if fp32 is not None:
        reference = {row["name"]: row.get("text", "") for row in fp32.get("quality", [])}
        for row in results:
            if row.get("status") != "ok":
                continue
            for quality in row.get("quality", []):
                ref = reference.get(quality.get("name", ""), "")
                text = quality.get("text", "")
                quality["exact_match_fp32"] = text == ref
                quality["word_jaccard_fp32"] = _jaccard(text, ref) if ref else None
    out = ROOT / "bench" / "results" / "quant.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
