from llm_platform.config import Settings


def make_settings(**overrides) -> Settings:
    values = dict(
        model_id="test-model",
        backend="transformers",
        quantization="fp32",
        max_batch_size=4,
        max_wait_ms=40,
        max_queue_size=8,
        request_timeout_s=2.0,
        rate_limit_rps=1000,
        rate_limit_burst=1000,
        max_new_tokens_cap=64,
        max_prompt_tokens=128,
        enable_admin=True,
        torch_threads=1,
        log_level="warning",
        upstream_base_url="http://upstream/v1",
        upstream_max_inflight=2,
    )
    values.update(overrides)
    return Settings(**values)


def chat_body(**overrides) -> dict:
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 4,
        "temperature": 0,
        "stream": False,
    }
    body.update(overrides)
    return body
